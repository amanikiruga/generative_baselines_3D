"""
Evaluate GEN3C novel view synthesis on eval outputs.

Reads gt_rgb.mp4 + gt_cameras.npz from each sample dir, uses the first frame
as a seed image + the GT camera trajectory, runs GEN3C to generate the video,
then computes PSNR / LPIPS / SSIM on generated frames [1:-1].

Works for both Aria and RE10K datasets.  Provide --input_dir pointing to a
directory whose sample_XXXXX/ subdirs contain:
  - gt_rgb.mp4    (ground-truth video, T frames)
  - gt_cameras.npz  (keys: extrinsics (T,4,4) c2w, intrinsics (T,3,3))

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

GEN3C_DIR = "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/GEN3C"
sys.path.insert(0, GEN3C_DIR)

DEFAULT_CKPT_DIR = os.path.join(GEN3C_DIR, "checkpoints")

# ---------------------------------------------------------------------------
# Aria coordinate-frame correction
# GT cameras in gt_cameras.npz are in the Aria device frame; camera-conditioned
# models (GEN3C, DFoT, SEVA) were trained on OpenCV-convention cameras.
# T_fix (pose eval) converts opencv→aria: T_fix @ w2c_opencv = w2c_aria
# Inverse: w2c_opencv = T_fix^{-1} @ w2c_aria = T_fix.T @ w2c_aria
# Equivalently for c2w: c2w_opencv = c2w_aria @ T_fix
# (T_fix is a pure rotation so T_fix^{-1} = T_fix.T)
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
    """Convert c2w (T,4,4) from Aria device frame to OpenCV convention."""
    T_fix = _aria_fix()
    # c2w_opencv = c2w_aria @ T_fix
    return c2w @ T_fix


# ---------------------------------------------------------------------------
# Video / metric helpers
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
        mse = float(torch.nn.functional.mse_loss(pred_u[t], gt_u[t]).item())
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
# GEN3C model loading & inference
# ---------------------------------------------------------------------------

def create_gen3c_args(ckpt_dir: str, height: int, width: int,
                      num_steps: int, seed: int) -> "argparse.Namespace":
    """Build a minimal args namespace for Gen3cPersistentModel."""
    from cosmos_predict1.diffusion.inference.gen3c_single_image import create_parser

    parser = create_parser()
    args = parser.parse_args([
        "--checkpoint_dir", ckpt_dir,
        "--checkpoint_name", "Gen3C_7B",
        "--disable_prompt_upsampler",
        "--disable_guardrail",
        "--offload_network",
        "--height", str(height),
        "--width", str(width),
        "--num_steps", str(num_steps),
        "--seed", str(seed),
        "--trajectory", "none",
        "--prompt", "",
    ])
    return args


def load_gen3c_model(ckpt_dir: str, height: int = 704, width: int = 1280,
                     num_steps: int = 35, seed: int = 42):
    from cosmos_predict1.diffusion.inference.gen3c_persistent import Gen3cPersistentModel
    args = create_gen3c_args(ckpt_dir, height, width, num_steps, seed)
    model = Gen3cPersistentModel(args)
    return model


def run_gen3c_on_sample(model, gt_frames_hwc: np.ndarray,
                        c2w: np.ndarray, intrinsics_33: np.ndarray,
                        prompt: str = "") -> np.ndarray | None:
    """
    Run GEN3C on a single video sample.

    Args:
        gt_frames_hwc: (T, H, W, 3) uint8 RGB frames
        c2w:           (T, 4, 4) camera-to-world matrices
        intrinsics_33: (T, 3, 3) camera intrinsic matrices (pixel units)
        prompt:        text prompt (can be empty)

    Returns:
        gen_frames_hwc: (T, H, W, 3) uint8 generated frames, or None on failure
    """
    from cosmos_predict1.diffusion.inference.gen3c_persistent import Gen3cPersistentModel

    T = len(gt_frames_hwc)
    # Convert c2w → w2c: GEN3C expects world-to-camera matrices (N, 4, 4)
    w2c = np.linalg.inv(c2w)                    # (T, 4, 4)
    w2c_b = w2c[np.newaxis]                      # (1, T, 4, 4)
    K_b   = intrinsics_33[np.newaxis].astype(np.float32)  # (1, T, 3, 3)

    # GEN3C uses the first frame as seed; pass it as the initial image
    seed_image = gt_frames_hwc[0]               # (H, W, 3) uint8
    seed_image_tensor = torch.from_numpy(seed_image).permute(2, 0, 1).float() / 255.0
    seed_image_tensor = seed_image_tensor.unsqueeze(0)  # (1, 3, H, W)

    # Call the persistent model
    result = model.inference_on_cameras(
        view_cameras_w2cs=w2c_b,
        view_camera_intrinsics=K_b,
        prompt=prompt,
        image=seed_image_tensor,
    )

    if result is None:
        return None

    # result is (1, T, H, W, 3) uint8 or similar — adapt to what the API returns
    if isinstance(result, torch.Tensor):
        result = result.cpu().numpy()
    if result.ndim == 5:
        result = result[0]   # remove batch dim → (T, H, W, 3)
    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Sample dirs with gt_rgb.mp4 + gt_cameras.npz")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR,
                        help="GEN3C checkpoint directory (contains Gen3C_7B/)")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--height", type=int, default=480,
                        help="Resize height for GEN3C (must be divisible by 32)")
    parser.add_argument("--width", type=int, default=704,
                        help="Resize width for GEN3C (must be divisible by 32)")
    parser.add_argument("--num_steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--custom_run_name", default="gen3c_nvs_eval")
    parser.add_argument("--dataset", default="re10k", choices=["re10k", "aria"],
                        help="'aria' converts GT cameras from Aria device frame to OpenCV "
                             "convention before passing to GEN3C.")
    args = parser.parse_args()

    wandb.init(project="video_world_model", name=args.custom_run_name)
    wandb.config.update(vars(args))

    device = torch.device(args.device)
    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()

    print(f"Loading GEN3C from {args.ckpt_dir} …")
    gen3c_model = load_gen3c_model(args.ckpt_dir, args.height, args.width,
                                   args.num_steps, args.seed)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(p for p in input_dir.iterdir()
                         if p.is_dir() and p.name.startswith("sample_"))

    per_sample_csv = output_dir / "per_sample_metrics.csv"
    fieldnames = ["sample", "n_gen_frames", "avg_psnr", "avg_lpips", "avg_ssim"]

    running_psnr = running_lpips = running_ssim = 0.0
    n_processed = 0

    with open(per_sample_csv, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        for sample_dir in tqdm(sample_dirs[:args.max_samples], desc="GEN3C NVS eval"):
            gt_rgb_path     = sample_dir / "gt_rgb.mp4"
            gt_cameras_path = sample_dir / "gt_cameras.npz"

            if not gt_rgb_path.exists() or not gt_cameras_path.exists():
                print(f"Skipping {sample_dir.name}: missing gt_rgb.mp4 or gt_cameras.npz")
                continue

            gt_frames = np.stack(imageio.mimread(str(gt_rgb_path), memtest=False))  # (T, H, W, 3)

            # Load our method's prediction if available in the input dir
            ours_rgb_path = sample_dir / "pred_rgb.mp4"
            ours_frames = None
            if ours_rgb_path.exists():
                ours_frames = np.stack(imageio.mimread(str(ours_rgb_path), memtest=False))

            cam_data  = np.load(str(gt_cameras_path))
            c2w        = cam_data["extrinsics"].astype(np.float64)   # (T, 4, 4)
            intrinsics = cam_data["intrinsics"].astype(np.float32)   # (T, 3, 3)

            T = min(len(gt_frames), len(c2w))
            gt_frames  = gt_frames[:T]
            c2w        = c2w[:T]
            intrinsics = intrinsics[:T]

            # Convert GT cameras from Aria device frame → OpenCV for model conditioning
            if args.dataset == "aria":
                c2w = aria_c2w_to_opencv(c2w)

            if T < 3:
                print(f"Skipping {sample_dir.name}: fewer than 3 frames")
                continue

            # Resize frames to GEN3C resolution
            orig_h, orig_w = gt_frames.shape[1:3]
            gen_h, gen_w = args.height, args.width
            if orig_h != gen_h or orig_w != gen_w:
                frames_t = torch.from_numpy(gt_frames).permute(0, 3, 1, 2).float()  # (T, 3, H, W)
                frames_t = F.interpolate(frames_t, size=(gen_h, gen_w), mode="bilinear",
                                         align_corners=False)
                gt_frames_resized = frames_t.permute(0, 2, 3, 1).numpy().astype(np.uint8)
                # Adjust intrinsics for resize
                intrinsics_resized = intrinsics.copy()
                intrinsics_resized[:, 0, :] *= gen_w / orig_w
                intrinsics_resized[:, 1, :] *= gen_h / orig_h
            else:
                gt_frames_resized = gt_frames
                intrinsics_resized = intrinsics

            try:
                pred_frames_hwc = run_gen3c_on_sample(
                    gen3c_model, gt_frames_resized, c2w,
                    intrinsics_resized.astype(np.float64),
                )
            except Exception as e:
                print(f"Error running GEN3C on {sample_dir.name}: {e}")
                import traceback; traceback.print_exc()
                continue

            if pred_frames_hwc is None:
                print(f"GEN3C returned None for {sample_dir.name}, skipping")
                continue

            # Resize predictions back to original resolution for metric computation
            if pred_frames_hwc.shape[1:3] != (orig_h, orig_w):
                pred_t = torch.from_numpy(pred_frames_hwc).permute(0, 3, 1, 2).float()
                pred_t = F.interpolate(pred_t, size=(orig_h, orig_w), mode="bilinear",
                                       align_corners=False)
                pred_frames_hwc = pred_t.permute(0, 2, 3, 1).numpy().astype(np.uint8)

            # Convert to tensors in [-1, 1]
            gt_tensor = torch.from_numpy(gt_frames.astype(np.float32)).permute(0, 3, 1, 2)
            gt_tensor = (gt_tensor / 255.0) * 2.0 - 1.0                   # (T, 3, H, W)
            pred_tensor = torch.from_numpy(pred_frames_hwc.astype(np.float32)).permute(0, 3, 1, 2)
            pred_tensor = (pred_tensor / 255.0) * 2.0 - 1.0

            # Ensure matching frame count
            n = min(len(pred_tensor), len(gt_tensor))
            pred_tensor = pred_tensor[:n].to(device)
            gt_tensor   = gt_tensor[:n].to(device)

            # Exclude first and last frame (reference frames in NVS eval protocol)
            if n < 3:
                print(f"Skipping {sample_dir.name}: fewer than 3 frames after GEN3C")
                continue
            pred_gen = pred_tensor[1:-1]
            gt_gen   = gt_tensor[1:-1]

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

            # Save generated video, GT, and side-by-side comparison
            sample_out = output_dir / sample_dir.name
            sample_out.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(str(sample_out / "pred_rgb.mp4"), pred_frames_hwc, fps=10)
            imageio.mimwrite(str(sample_out / "gt_rgb.mp4"), gt_frames[:n], fps=10)
            # comparison: GT | ours | GEN3C  (insert ours column if available)
            cols = [gt_frames[:n], pred_frames_hwc[:n]]
            if ours_frames is not None:
                ours_n = ours_frames[:n]
                if ours_n.shape[1:3] != gt_frames.shape[1:3]:
                    ours_t = torch.from_numpy(ours_n).permute(0, 3, 1, 2).float()
                    ours_t = F.interpolate(ours_t, size=gt_frames.shape[1:3], mode="bilinear",
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

    print(f"\n=== GEN3C NVS final ({n_processed} samples) ===")
    print(f"  PSNR:  {running_psnr:.4f}")
    print(f"  LPIPS: {running_lpips:.4f}")
    print(f"  SSIM:  {running_ssim:.4f}")

    wandb.log({"final/psnr": running_psnr, "final/lpips": running_lpips,
               "final/ssim": running_ssim})
    wandb.finish()


if __name__ == "__main__":
    main()
