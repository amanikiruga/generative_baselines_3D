"""
Evaluate DFoT (Diffusion Forcing Transformer) NVS on RE10K eval outputs.

Reads gt_rgb.mp4 + gt_cameras.npz from each sample dir, conditions the model
on the first and last frames + GT camera poses (interpolation task), generates
intermediate frames, and computes PSNR / LPIPS / SSIM on frames [1:-1].

Input dir should contain sample_XXXXX/ subdirs with:
  - gt_rgb.mp4        (T frames of ground-truth RGB)
  - gt_cameras.npz    (keys: extrinsics (T,4,4) c2w, intrinsics (T,3,3))

Recommended input dir:
  ignore/outputs/ray_mot_eval_histg_1_frames_50_apr08   (has cameras + gt_rgb)
  ignore/eval_outputs/nvs_aria_on_aria                  (Aria, has cameras)

Camera conditioning format for DFoT RE10K model (16-vector per frame):
  [fx/W, fy/H, cx/W, cy/H, R_w2c.flatten(), T_w2c]
  i.e. the first 4 are normalised intrinsics, the next 12 are flattened w2c [R|T].

Run from: /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

DFOT_DIR = "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/diffusion-forcing-transformer"
sys.path.insert(0, DFOT_DIR)

DFOT_N_FRAMES = 8       # DFoT RE10K model works with 8 frames
DFOT_RESOLUTION = 256   # DFoT RE10K model resolution

# ---------------------------------------------------------------------------
# Aria coordinate-frame correction (same derivation as pose eval scripts)
# c2w_opencv = c2w_aria @ T_fix  where T_fix = R_hardware @ R_image_roll
# ---------------------------------------------------------------------------
_R_ARIA_HARDWARE = np.array([
    [ 0.99606003, -0.04388682,  0.07706079],
    [ 0.08210934,  0.78468796, -0.61442889],
    [-0.03350334,  0.61833547,  0.78519983],
], dtype=np.float64)
_R_ARIA_IMAGE_ROLL = np.array([
    [ 0.0,  1.0,  0.0],
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
], dtype=np.float64)

def _aria_fix() -> np.ndarray:
    R = _R_ARIA_HARDWARE @ _R_ARIA_IMAGE_ROLL
    T = np.eye(4, dtype=np.float64); T[:3, :3] = R
    return T

def aria_c2w_to_opencv(c2w: np.ndarray) -> np.ndarray:
    return c2w @ _aria_fix()


# ---------------------------------------------------------------------------
# Metric helpers (same as recompute_metrics_offline.py)
# ---------------------------------------------------------------------------

def load_video_as_tensor(mp4_path: str) -> torch.Tensor:
    """Returns (T, 3, H, W) float32 in [-1, 1]."""
    frames = imageio.mimread(mp4_path, memtest=False)
    arr = np.stack(frames, axis=0).astype(np.float32)
    arr = arr / 255.0 * 2.0 - 1.0
    arr = arr.transpose(0, 3, 1, 2)
    return torch.from_numpy(arr)


@torch.no_grad()
def per_frame_psnr(pred: torch.Tensor, gt: torch.Tensor) -> list:
    pred_u = (pred * 0.5 + 0.5).clamp(0, 1)
    gt_u   = (gt   * 0.5 + 0.5).clamp(0, 1)
    psnrs = []
    for t in range(pred_u.shape[0]):
        mse = float(F.mse_loss(pred_u[t], gt_u[t]).item())
        psnrs.append(float(-10.0 * math.log10(mse)) if mse > 0 else float("inf"))
    return psnrs


@torch.no_grad()
def per_frame_lpips(pred: torch.Tensor, gt: torch.Tensor, lpips_fn) -> list:
    return [float(lpips_fn(pred[t:t+1].float(), gt[t:t+1].float()).item())
            for t in range(pred.shape[0])]


def per_frame_ssim(pred: torch.Tensor, gt: torch.Tensor) -> list:
    from skimage.metrics import structural_similarity
    pred_np = (pred * 0.5 + 0.5).clamp(0, 1).float().cpu().numpy()
    gt_np   = (gt   * 0.5 + 0.5).clamp(0, 1).float().cpu().numpy()
    return [
        float(structural_similarity(
            pred_np[t].transpose(1, 2, 0),
            gt_np[t].transpose(1, 2, 0),
            data_range=1.0, channel_axis=2,
        ))
        for t in range(pred_np.shape[0])
    ]


# ---------------------------------------------------------------------------
# DFoT model loading
# ---------------------------------------------------------------------------

def load_dfot_model(ckpt_path: str, device: torch.device):
    """
    Load DFoT RE10K model from a checkpoint.

    Uses Hydra to compose the config, then instantiates DFoTVideoPose and
    loads the checkpoint weights.

    Checkpoint: download from HuggingFace with the DFoT repo's ckpt_utils or
    manually from https://huggingface.co/kiwhansong/DFoT/tree/main/pretrained_models
    Expected file: DFoT_RE10K.ckpt
    """
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    config_dir = os.path.join(DFOT_DIR, "configurations")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "+name=eval_re10k_nvs",
                "dataset=realestate10k",
                "algorithm=dfot_video_pose",
                "experiment=video_generation",
                # continuous diffusion overrides (equivalent to @diffusion/continuous)
                "algorithm.diffusion.is_continuous=true",
                "+algorithm.diffusion.precond_scale=0.125",
                "+algorithm.backbone.use_fourier_noise_embedding=true",
                # disable torch.compile so checkpoint keys match (no _orig_mod prefix)
                "algorithm.compile=false",
            ],
        )

    from algorithms.dfot.dfot_video_pose import DFoTVideoPose

    # Disable WandB / Lightning logging during eval; strip heavy metrics that require
    # torch-fidelity (fid, fvd, is) since we compute our own metrics externally.
    with open_dict(cfg):
        cfg.wandb = OmegaConf.create({"mode": "disabled", "entity": "", "project": ""})
        cfg.algorithm.logging.metrics = ["lpips", "psnr", "ssim"]

    # Instantiate the model
    model = DFoTVideoPose(cfg.algorithm)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    # Strip 'module.' prefix if present (DDP checkpoints)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing keys when loading checkpoint")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys when loading checkpoint")

    model.eval().to(device)
    return model, cfg


# ---------------------------------------------------------------------------
# Camera conditioning helpers
# ---------------------------------------------------------------------------

def build_dfot_conditions(c2w: np.ndarray, intrinsics_33: np.ndarray,
                           n_frames: int, orig_hw: tuple[int, int],
                           model_res: int = DFOT_RESOLUTION) -> torch.Tensor:
    """
    Build DFoT camera conditions tensor of shape (1, n_frames, 16).

    Format matches CameraPose.from_vectors in the DFoT codebase:
      [fx/W, fy/H, cx/W, cy/H,  R00, R01, R02, T0,  R10, R11, R12, T1,  R20, R21, R22, T2]
    i.e. normalised intrinsics followed by the 3×4 [R|t] w2c matrix, flattened row-major.

    Args:
        c2w:          (T, 4, 4) camera-to-world matrices
        intrinsics_33:(T, 3, 3) pixel-unit intrinsics at orig resolution
        n_frames:     number of frames to use
        orig_hw:      (H, W) original video resolution
        model_res:    DFoT model spatial resolution (square)
    """
    T = len(c2w)
    if T >= n_frames:
        indices = np.linspace(0, T - 1, n_frames, dtype=int)
    else:
        indices = np.concatenate([np.arange(T),
                                   np.full(n_frames - T, T - 1)])
    c2w_sub = c2w[indices]           # (n_frames, 4, 4)
    K_sub   = intrinsics_33[indices] # (n_frames, 3, 3)

    H, W = orig_hw
    # Normalised intrinsics: divide pixel-space K by the *corresponding* image dimension.
    # The model was trained with K normalised so that image coords are in [0,1]×[0,1].
    # After an anisotropic resize to model_res×model_res, scaling is (model_res/W) for x
    # and (model_res/H) for y; dividing by model_res gives just 1/W and 1/H respectively.
    fx_norm = K_sub[:, 0, 0] / W
    fy_norm = K_sub[:, 1, 1] / H
    cx_norm = K_sub[:, 0, 2] / W
    cy_norm = K_sub[:, 1, 2] / H
    intrinsics_4 = np.stack([fx_norm, fy_norm, cx_norm, cy_norm], axis=-1)  # (n_frames, 4)

    # w2c extrinsics as 3×4 [R|t] matrix, stored row-major:
    #   [R00, R01, R02, T0,  R10, R11, R12, T1,  R20, R21, R22, T2]
    w2c   = np.linalg.inv(c2w_sub)          # (n_frames, 4, 4)
    R_w2c = w2c[:, :3, :3]                  # (n_frames, 3, 3)
    T_w2c = w2c[:, :3, 3:4]                 # (n_frames, 3, 1)
    RT_34 = np.concatenate([R_w2c, T_w2c], axis=-1)  # (n_frames, 3, 4)
    extrinsics_12 = RT_34.reshape(n_frames, 12)       # row-major → correct format

    conds_np = np.concatenate([intrinsics_4, extrinsics_12], axis=-1)  # (n_frames, 16)
    return torch.from_numpy(conds_np).float().unsqueeze(0)             # (1, n_frames, 16)


# ---------------------------------------------------------------------------
# DFoT interpolation inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_dfot_interpolation(model, gt_video_chw: torch.Tensor,
                            conditions: torch.Tensor,
                            device: torch.device) -> torch.Tensor:
    """
    Run DFoT frame interpolation given first+last frames.

    Args:
        model:          DFoTVideoPose instance
        gt_video_chw:   (1, T, C, H, W) video in [-1, 1] at model resolution
        conditions:     (1, T, 16) camera pose conditions
        device:         torch device

    Returns:
        pred_video:     (1, T, C, H, W) in [-1, 1]
    """
    B, T = gt_video_chw.shape[:2]
    context = gt_video_chw.to(device)
    conditions = conditions.to(device)

    # context_mask: True for first and last frames only
    context_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    context_mask[:, 0]  = True
    context_mask[:, -1] = True

    # Zero out all frames except first and last before passing
    context_filled = context.clone()
    for t in range(1, T - 1):
        context_filled[:, t] = 0.0

    pred = model._interpolate_videos(
        context=context_filled,
        context_mask=context_mask,
        conditions=conditions,
    )
    return pred.cpu()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Dir with sample_XXXXX/ containing gt_rgb.mp4 + gt_cameras.npz")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt_path", required=True,
                        help="Path to DFoT_RE10K.ckpt checkpoint file")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--n_frames", type=int, default=DFOT_N_FRAMES,
                        help=f"Frames to interpolate (default {DFOT_N_FRAMES} for RE10K model)")
    parser.add_argument("--resolution", type=int, default=DFOT_RESOLUTION,
                        help=f"Model spatial resolution (default {DFOT_RESOLUTION})")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--custom_run_name", default="dfot_nvs_re10k_eval")
    parser.add_argument("--dataset", default="re10k", choices=["re10k", "aria"],
                        help="'aria' converts GT cameras from Aria device frame to OpenCV "
                             "convention before passing to DFoT.")
    args = parser.parse_args()

    wandb.init(project="video_world_model", name=args.custom_run_name)
    wandb.config.update(vars(args))

    device = torch.device(args.device)
    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()

    print(f"Loading DFoT model from {args.ckpt_path} …")
    model, cfg = load_dfot_model(args.ckpt_path, device)
    print("DFoT model loaded successfully.")

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(p for p in input_dir.iterdir()
                         if p.is_dir() and p.name.startswith("sample_"))

    per_sample_csv = output_dir / "per_sample_metrics.csv"
    fieldnames = ["sample", "n_gen_frames", "avg_psnr", "avg_lpips", "avg_ssim"]

    running_psnr = running_lpips = running_ssim = 0.0
    n_processed = 0

    # RE10K normalisation constants from DFoT dataset config
    data_mean = torch.tensor([[[0.577]], [[0.517]], [[0.461]]])  # (C, 1, 1)
    data_std  = torch.tensor([[[0.249]], [[0.249]], [[0.268]]])  # (C, 1, 1)

    with open(per_sample_csv, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        for sample_dir in tqdm(sample_dirs[:args.max_samples], desc="DFoT NVS eval"):
            gt_rgb_path     = sample_dir / "gt_rgb.mp4"
            gt_cameras_path = sample_dir / "gt_cameras.npz"

            if not gt_rgb_path.exists() or not gt_cameras_path.exists():
                print(f"Skipping {sample_dir.name}: missing gt_rgb.mp4 or gt_cameras.npz")
                continue

            cam_data   = np.load(str(gt_cameras_path))
            c2w        = cam_data["extrinsics"].astype(np.float64)  # (T, 4, 4) c2w
            intrinsics = cam_data["intrinsics"].astype(np.float64)  # (T, 3, 3)

            gt_frames_np = np.stack(imageio.mimread(str(gt_rgb_path), memtest=False))  # (T,H,W,3)

            # Load our method's NVS prediction if available in the input dir
            ours_rgb_path = sample_dir / "pred_rgb.mp4"
            ours_frames_np = None
            if ours_rgb_path.exists():
                ours_frames_np = np.stack(imageio.mimread(str(ours_rgb_path), memtest=False))
            T_full = min(len(gt_frames_np), len(c2w))
            gt_frames_np = gt_frames_np[:T_full]
            c2w          = c2w[:T_full]
            intrinsics   = intrinsics[:T_full]

            # Convert GT cameras from Aria device frame → OpenCV for model conditioning
            if args.dataset == "aria":
                c2w = aria_c2w_to_opencv(c2w)
            orig_h, orig_w = gt_frames_np.shape[1:3]

            if T_full < 3:
                print(f"Skipping {sample_dir.name}: fewer than 3 frames")
                continue

            # Resize frames to DFoT model resolution
            res = args.resolution
            frames_t = torch.from_numpy(gt_frames_np).permute(0, 3, 1, 2).float() / 255.0
            frames_t = F.interpolate(frames_t, size=(res, res), mode="bilinear",
                                     align_corners=False)  # (T, 3, res, res) in [0, 1]

            # Normalise to DFoT's training distribution
            frames_norm = (frames_t - data_mean) / data_std   # (T, 3, res, res)

            # Subsample/pad to n_frames for the model
            n_frames = args.n_frames
            if T_full >= n_frames:
                sub_idx = np.linspace(0, T_full - 1, n_frames, dtype=int)
            else:
                sub_idx = np.concatenate([np.arange(T_full),
                                           np.full(n_frames - T_full, T_full - 1)])
            frames_sub = frames_norm[torch.from_numpy(sub_idx.copy())]  # (n_frames, 3, res, res)
            frames_bthwc = frames_sub.unsqueeze(0)                       # (1, n_frames, 3, res, res)

            # Build camera conditions
            conditions = build_dfot_conditions(
                c2w, intrinsics, n_frames, (orig_h, orig_w), args.resolution,
            )  # (1, n_frames, 16)

            try:
                pred_bthwc = run_dfot_interpolation(model, frames_bthwc, conditions, device)
            except Exception as e:
                print(f"Error running DFoT on {sample_dir.name}: {e}")
                import traceback; traceback.print_exc()
                continue

            # Denormalise DFoT output back to [0, 1] then to [-1, 1] for metrics
            pred_01 = pred_bthwc[0] * data_std + data_mean   # (n_frames, 3, res, res)
            pred_01 = pred_01.clamp(0, 1)
            pred_m1_1 = pred_01 * 2.0 - 1.0                  # → [-1, 1]

            # Get GT at sub-sampled indices in [-1, 1]
            gt_sub_np = gt_frames_np[sub_idx]               # (n_frames, H, W, 3)
            gt_t = torch.from_numpy(gt_sub_np).permute(0, 3, 1, 2).float()
            gt_t = F.interpolate(gt_t / 255.0, size=(res, res), mode="bilinear",
                                  align_corners=False) * 2.0 - 1.0   # [-1, 1]

            # Resize to original resolution for metrics
            pred_m1_1_orig = F.interpolate(pred_m1_1, size=(orig_h, orig_w),
                                            mode="bilinear", align_corners=False)
            gt_orig = F.interpolate(gt_t, size=(orig_h, orig_w),
                                     mode="bilinear", align_corners=False)

            pred_m1_1_orig = pred_m1_1_orig.to(device)
            gt_orig        = gt_orig.to(device)

            # Exclude first and last frame (conditioning frames)
            if n_frames < 3:
                print(f"Skipping {sample_dir.name}: n_frames < 3")
                continue
            pred_gen = pred_m1_1_orig[1:-1]
            gt_gen   = gt_orig[1:-1]

            psnrs  = per_frame_psnr(pred_gen, gt_gen)
            lpipss = per_frame_lpips(pred_gen, gt_gen, lpips_fn)
            ssims  = per_frame_ssim(pred_gen, gt_gen)

            avg_p = sum(psnrs)  / len(psnrs)
            avg_l = sum(lpipss) / len(lpipss)
            avg_s = sum(ssims)  / len(ssims)

            writer.writerow({"sample": sample_dir.name, "n_gen_frames": len(psnrs),
                             "avg_psnr": avg_p, "avg_lpips": avg_l, "avg_ssim": avg_s})
            csvf.flush()

            n_processed += 1
            running_psnr  = (running_psnr  * (n_processed - 1) + avg_p) / n_processed
            running_lpips = (running_lpips * (n_processed - 1) + avg_l) / n_processed
            running_ssim  = (running_ssim  * (n_processed - 1) + avg_s) / n_processed

            # Save generated video, GT video, and side-by-side comparison
            sample_out = output_dir / sample_dir.name
            sample_out.mkdir(parents=True, exist_ok=True)
            pred_hwc = ((pred_m1_1_orig * 0.5 + 0.5).clamp(0, 1) * 255).byte()
            pred_hwc = pred_hwc.cpu().permute(0, 2, 3, 1).numpy()
            gt_hwc = ((gt_orig * 0.5 + 0.5).clamp(0, 1) * 255).byte()
            gt_hwc = gt_hwc.cpu().permute(0, 2, 3, 1).numpy()
            imageio.mimwrite(str(sample_out / "pred_rgb.mp4"), pred_hwc, fps=10)
            imageio.mimwrite(str(sample_out / "gt_rgb.mp4"), gt_hwc, fps=10)
            # comparison: GT | ours | DFoT  (insert ours column if available)
            cols = [gt_hwc, pred_hwc]
            if ours_frames_np is not None:
                ours_n = ours_frames_np[:len(gt_hwc)]
                if ours_n.shape[1:3] != gt_hwc.shape[1:3]:
                    ours_t = torch.from_numpy(ours_n).permute(0, 3, 1, 2).float()
                    ours_t = F.interpolate(ours_t, size=gt_hwc.shape[1:3], mode="bilinear",
                                           align_corners=False)
                    ours_n = ours_t.permute(0, 2, 3, 1).numpy().astype(np.uint8)
                cols.insert(1, ours_n)
            comparison = np.concatenate(cols, axis=2)
            imageio.mimwrite(str(sample_out / "comparison.mp4"), comparison, fps=10)

            wandb.log({"sample/psnr": avg_p, "sample/lpips": avg_l, "sample/ssim": avg_s})
            tqdm.write(f"  [{n_processed}] {sample_dir.name} | "
                       f"PSNR={avg_p:.3f} LPIPS={avg_l:.4f} SSIM={avg_s:.4f} || "
                       f"Running: PSNR={running_psnr:.3f} LPIPS={running_lpips:.4f} SSIM={running_ssim:.4f}")

    summary = {"n_samples": n_processed, "avg_psnr": running_psnr,
               "avg_lpips": running_lpips, "avg_ssim": running_ssim,
               "excluded_frames": "frame 0 (first) and frame T-1 (last) per sample"}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== DFoT NVS RE10K final ({n_processed} samples) ===")
    print(f"  PSNR:  {running_psnr:.4f}")
    print(f"  LPIPS: {running_lpips:.4f}")
    print(f"  SSIM:  {running_ssim:.4f}")

    wandb.log({"final/psnr": running_psnr, "final/lpips": running_lpips,
               "final/ssim": running_ssim})
    wandb.finish()


if __name__ == "__main__":
    main()
