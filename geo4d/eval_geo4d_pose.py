"""
Evaluate Geo4D camera pose estimation on rgb_to_ray_depth_aria eval outputs.

Reads gt_rgb.mp4 from each sample dir (produced by
inference_single_gpu_rgb_to_ray_depth_aria_eval.py), runs Geo4D to get
predicted camera extrinsics, then compares against GT cameras saved in
gt_cameras.npz (c2w, shape T x 4 x 4).


Metrics: AUC3 and AUC30.
For each pair (i, j) in a sequence, compute max(rotation_error_deg, translation_direction_error_deg).
AUC@k = average fraction of pairs with error < t, for t in [0, k degrees].
"""

import argparse
import json
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import wandb

GEO4D_DIR = "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D"

# ---------------------------------------------------------------------------
# Aria coordinate-frame correction
# (see eval_da3_pose_debug_alt.py with --procrustes_align_pred for derivation)
# Geo4D outputs OpenCV-convention cameras; Aria GT is in the Aria device frame.
# R_hardware: Aria device → camera (from Aria API hardware calibration)
# R_image_roll: undo the 90° CW rotation baked into Aria gt_rgb.mp4
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

def _aria_fix_matrix() -> np.ndarray:
    R = _R_ARIA_HARDWARE @ _R_ARIA_IMAGE_ROLL
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T

def apply_aria_correction(pred_w2c: np.ndarray) -> np.ndarray:
    T_fix = _aria_fix_matrix()
    return np.array([T_fix @ w2c for w2c in pred_w2c])


def setup_geo4d_path():
    if GEO4D_DIR not in sys.path:
        sys.path.insert(0, GEO4D_DIR)


# ── Geo4D model loading ───────────────────────────────────────────────────────

def load_geo4d_model(ckpt_path: str, config_path: str, gpu_no: int = 0):
    from omegaconf import OmegaConf
    from utils.utils import instantiate_from_config
    from scripts.evaluation.test_geo4d import load_model_checkpoint

    config = OmegaConf.load(config_path)
    model_config = config.pop("model", OmegaConf.create())
    model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    model.perframe_ae = True
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    model = load_model_checkpoint(model, ckpt_path)
    model.eval()

    pointmap_vae = None
    if "vae_path" in config:
        pointmap_vae_config = config.pop("pointmap_vae_config", OmegaConf.create())
        pointmap_vae = instantiate_from_config(pointmap_vae_config).eval().cuda(gpu_no)
        from lvdm.basics import disabled_train
        pointmap_vae.train = disabled_train
        for param in pointmap_vae.parameters():
            param.requires_grad = False
        vae_weights = torch.load(config["vae_path"])["state_dict"]
        new_sd = OrderedDict(
            (k[6:], v) for k, v in vae_weights.items() if k.startswith("model.")
        )
        pointmap_vae.load_state_dict(new_sd, strict=True)

    return model, config, pointmap_vae


# ── Per-video Geo4D inference ─────────────────────────────────────────────────

def infer_geo4d_video(
    model, config, pointmap_vae,
    video_path: str,
    sample_savedir: str,
    ddim_steps: int = 5,
    height: int = 320,
    width: int = 512,
    gpu_no: int = 0,
) -> str:
    """Run Geo4D on a single video. Returns path to saved pred_traj.txt."""
    from einops import rearrange
    from pytorch_lightning import seed_everything
    from utils.funcs import load_video_batch
    from scripts.evaluation.test_geo4d import (
        image_guided_synthesis, post_optimization,
        get_sky_mask, get_far_away_mask, denormalize_pc_bbox2,
        raymap_to_camera_matrix, decode_pm_confhead,
    )

    seed_everything(123)

    seq_name = Path(video_path).stem
    seq_savedir = os.path.join(sample_savedir, seq_name)
    os.makedirs(seq_savedir, exist_ok=True)
    traj_path = os.path.join(seq_savedir, "pred_traj.txt")

    video_frames, fps_list = load_video_batch(
        [video_path],
        frame_stride=1,
        video_size=(height, width),
        video_frames=-1,
    )
    B, C, T, H, W = video_frames.shape

    if T < 16:
        pad = video_frames[:, :, -1:, :, :].expand(-1, -1, 16 - T, -1, -1)
        video_frames = torch.cat([video_frames, pad], dim=2)
        B, C, T, H, W = video_frames.shape

    views = [{"img": video_frames[0, :, i, :, :], "idx": (i,)} for i in range(T)]

    channels = model.model.diffusion_model.out_channels
    noise_shape = [1, channels, 16, H // 8, W // 8]

    videos_all = video_frames.cuda(gpu_no)
    prompts = ["Output a video that assigns each 3D location in the world a consistent color."]

    stride = 4
    slice_list = list(range(0, T - 16 + 1, stride))
    slice_list = [slice(s, s + 16, 1) for s in slice_list]
    if not slice_list or slice_list[-1] != slice(T - 16, T, 1):
        slice_list.append(slice(T - 16, T, 1))

    pred_list = []
    view_list = []
    pnt_valid_mask = torch.ones((T, H, W, 1), device=f"cuda:{gpu_no}") > 0

    for sl in tqdm(slice_list, desc="  Geo4D windows", leave=False):
        videos = videos_all[:, :, sl, :, :].clone()
        view_list.append(views[sl])

        fps = fps_list[0]
        batch_samples = image_guided_synthesis(
            model, prompts, videos, noise_shape,
            n_samples=1,
            ddim_steps=ddim_steps,
            ddim_eta=0.0,
            unconditional_guidance_scale=1.0,
            cfg_img=None,
            fs=fps,
            text_input=True,
            multiple_cond_cfg=False,
            loop=False,
            interp=False,
            timestep_spacing="uniform_trailing",
            guidance_rescale=0.7,
            pointmap_vae=pointmap_vae,
        )
        assert batch_samples.shape[1] == 1
        batch_samples = batch_samples[:, 0]

        raymap = crossmap = inverse_depthmap = traj = None
        if model.modality == "pc_ray_cross_depth":
            raymap = batch_samples[:, 4:7]
            crossmap = batch_samples[:, 7:10]
            traj = raymap_to_camera_matrix(raymap, crossmap)
            raymap = rearrange(raymap, "b c t h w -> (b t) c h w")
            raymap = rearrange(raymap, "t c h w -> t h w c")
            crossmap = rearrange(crossmap, "b c t h w -> (b t) c h w")
            crossmap = rearrange(crossmap, "t c h w -> t h w c")
            inverse_depthmap = batch_samples[:, 10:11]
            inverse_depthmap = rearrange(inverse_depthmap, "b c t h w -> (b t) c h w")
            inverse_depthmap = rearrange(inverse_depthmap, "t c h w -> t h w c")
            inverse_depthmap = (inverse_depthmap + 1.0) / 2.0

        batch_samples = batch_samples[:, :4]
        x_recon = rearrange(batch_samples, "b c t h w -> (b t) c h w")
        confidence = torch.nn.Softplus()(x_recon[:, [-1], :, :])
        confidence = rearrange(confidence, "t c h w -> t h w c")
        if pointmap_vae is None:
            confidence = torch.ones_like(confidence)

        x_recon = x_recon[:, :-1, :, :]
        x_recon_reshape = rearrange(x_recon, "t c h w -> t h w c")
        invalid_pts = get_sky_mask(x_recon_reshape, sky_value=1.05, eps=0.35)
        far_away_mask = get_far_away_mask(x_recon_reshape, far_away_value=1.99)
        invalid_pts = invalid_pts | far_away_mask
        confidence[invalid_pts] = 999.0
        pnt_valid_mask[sl] = pnt_valid_mask[sl] * (~invalid_pts)
        inverse_confidence = 1 / confidence
        inverse_confidence[invalid_pts] = 0.0

        x_recon = rearrange(x_recon, "t c h w -> t h w c")
        x_recon = denormalize_pc_bbox2(x_recon, alpha=2.0, beta=2.0)

        pred_pts = {"pts3d": x_recon, "conf": inverse_confidence}
        if inverse_depthmap is not None:
            pred_pts["inverse_depthmap"] = inverse_depthmap
        if traj is not None:
            pred_pts["traj"] = traj
        pred_list.append(pred_pts)

    scene = post_optimization(
        view_list, pred_list, config.postprocess,
        conf_optimize=True, init_method="group", lr=0.03,
        opt_raydir=False,
    )
    scene.save_tum_poses(traj_path)
    return traj_path


# ── Pose parsing / metrics ────────────────────────────────────────────────────

def read_tum_trajectory(traj_path: str) -> np.ndarray:
    """Parse TUM format trajectory → c2w matrices (N, 4, 4)."""
    poses = []
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            _, tx, ty, tz, qx, qy, qz, qw = [float(v) for v in parts]
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = [tx, ty, tz]
            poses.append(c2w)
    return np.stack(poses, axis=0)


def _procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    mu_s, mu_t = source.mean(0), target.mean(0)
    s_c, t_c = source - mu_s, target - mu_t
    U, _, Vt = np.linalg.svd(t_c.T @ s_c)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    ss = (s_c ** 2).sum()
    scale = (t_c * (s_c @ R.T)).sum() / ss if ss > 0 else 1.0
    return scale * (source - mu_s) @ R.T + mu_t


def plot_camera_trajectory(pred_w2c: np.ndarray, gt_w2c: np.ndarray, output_path: str) -> None:
    pred_c = np.linalg.inv(pred_w2c)[:, :3, 3]
    gt_c   = np.linalg.inv(gt_w2c)[:, :3, 3]
    pred_aligned = _procrustes_align(pred_c, gt_c)

    T = len(pred_c)
    colors = [plt.get_cmap("hsv")(i / max(T - 1, 1)) for i in range(T)]
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    for i in range(T - 1):
        ax.plot(pred_aligned[i:i+2, 0], pred_aligned[i:i+2, 1], pred_aligned[i:i+2, 2],
                color=colors[i], linewidth=2)
    ax.scatter(pred_aligned[:, 0], pred_aligned[:, 1], pred_aligned[:, 2],
               c=colors, s=30, zorder=5, label="Geo4D pred (aligned)")
    ax.plot(gt_c[:, 0], gt_c[:, 1], gt_c[:, 2], color="black", linewidth=1.5, linestyle="--")
    ax.scatter(gt_c[:, 0], gt_c[:, 1], gt_c[:, 2], color="black", s=20, zorder=4, label="GT")
    ax.legend()
    ax.set_title("Camera Trajectories — Geo4D pred aligned to GT (Procrustes)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_relative_pose_errors(w2c_pred: np.ndarray, w2c_gt: np.ndarray) -> np.ndarray:
    N = len(w2c_pred)
    errors = []
    for i in range(N):
        for j in range(i + 1, N):
            rel_pred = w2c_pred[j] @ np.linalg.inv(w2c_pred[i])
            rel_gt   = w2c_gt[j]   @ np.linalg.inv(w2c_gt[i])

            R_err = rel_pred[:3, :3] @ rel_gt[:3, :3].T
            cos_a = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
            rot_err = float(np.degrees(np.arccos(cos_a)))

            t_pred, t_gt = rel_pred[:3, 3], rel_gt[:3, 3]
            n_pred, n_gt = np.linalg.norm(t_pred), np.linalg.norm(t_gt)
            if n_pred < 1e-8 or n_gt < 1e-8:
                trans_err = 0.0
            else:
                cos_t = np.clip(np.dot(t_pred / n_pred, t_gt / n_gt), -1.0, 1.0)
                trans_err = float(np.degrees(np.arccos(cos_t)))

            errors.append(max(rot_err, trans_err))
    return np.array(errors, dtype=np.float64)


def compute_auc(errors: np.ndarray, max_threshold: float, n_points: int = 1000) -> float:
    thresholds = np.linspace(0.0, max_threshold, n_points)
    fractions = np.array([(errors <= t).mean() for t in thresholds])
    return float(fractions.mean()) * 100.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Output dir from inference_single_gpu_rgb_to_ray_depth_aria_eval.py")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str,
                        default=os.path.join(GEO4D_DIR, "checkpoints/geo4d/model.ckpt"))
    parser.add_argument("--config", type=str,
                        default=os.path.join(GEO4D_DIR, "configs/inference_geo4d.yaml"))
    parser.add_argument("--ddim_steps", type=int, default=5)
    parser.add_argument("--height", type=int, default=320,
                        help="Geo4D input height (supported: 384/320/256/192)")
    parser.add_argument("--width", type=int, default=512,
                        help="Geo4D input width (supported: 512/576/640)")
    parser.add_argument("--rotate_frames_k", type=int, default=0,
                        help="Apply np.rot90(frame, k=K) before Geo4D inference. "
                             "Use 1 for Aria (undo the 90° CW correction in the dataset).")
    parser.add_argument("--rerun", action="store_true",
                        help="Rerun Geo4D even if pred_traj.txt already exists")
    parser.add_argument("--dataset", default="re10k", choices=["re10k", "aria"],
                        help="Dataset being evaluated. 'aria' applies the Aria "
                             "device-to-OpenCV coordinate-frame correction to predicted poses.")
    parser.add_argument("--custom_run_name", type=str, default="geo4d_pose_eval")
    args = parser.parse_args()

    setup_geo4d_path()

    wandb.init(project="video_world_model", name=args.custom_run_name)
    wandb.config.update(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available — will be very slow or broken")

    print(f"Loading Geo4D model from {args.ckpt_path} ...")
    model, config, pointmap_vae = load_geo4d_model(args.ckpt_path, args.config, gpu_no=0)

    input_dir     = Path(args.input_dir)
    output_dir    = Path(args.output_dir)
    geo4d_raw_dir = output_dir / "geo4d_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    geo4d_raw_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted([
        p for p in input_dir.iterdir()
        if p.is_dir() and p.name.startswith("sample_")
    ])

    all_errors: list[np.ndarray] = []
    count = 0

    for sample_dir in tqdm(sample_dirs, desc="Geo4D pose eval"):
        if args.max_samples is not None and count >= args.max_samples:
            break

        gt_rgb_path     = sample_dir / "gt_rgb.mp4"
        gt_cameras_path = sample_dir / "gt_cameras.npz"

        if not gt_rgb_path.exists():
            print(f"Skipping {sample_dir.name}: missing gt_rgb.mp4")
            continue
        if not gt_cameras_path.exists():
            print(f"Skipping {sample_dir.name}: missing gt_cameras.npz")
            continue

        gt_data = np.load(gt_cameras_path)
        gt_c2w  = gt_data["extrinsics"].astype(np.float64)  # (T, 4, 4)
        gt_w2c  = np.linalg.inv(gt_c2w)
        n_gt_frames = len(gt_c2w)

        sample_geo4d_dir = str(geo4d_raw_dir / sample_dir.name)
        traj_path = os.path.join(sample_geo4d_dir, "input_video", "pred_traj.txt")

        if not os.path.exists(traj_path) or args.rerun:
            # If rotation is needed, write a rotated copy of the video
            if args.rotate_frames_k != 0:
                frames = imageio.mimread(str(gt_rgb_path))
                frames_rot = [np.rot90(f, k=args.rotate_frames_k) for f in frames]
                rot_video_dir = Path(sample_geo4d_dir)
                rot_video_dir.mkdir(parents=True, exist_ok=True)
                rot_video_path = str(rot_video_dir / "input_video.mp4")
                imageio.mimwrite(rot_video_path, frames_rot, fps=10)
                video_to_process = rot_video_path
            else:
                video_to_process = str(gt_rgb_path)

            try:
                traj_path = infer_geo4d_video(
                    model, config, pointmap_vae,
                    video_path=video_to_process,
                    sample_savedir=sample_geo4d_dir,
                    ddim_steps=args.ddim_steps,
                    height=args.height,
                    width=args.width,
                    gpu_no=0,
                )
            except Exception as e:
                print(f"Error running Geo4D on {sample_dir.name}: {e}")
                import traceback; traceback.print_exc()
                continue

        if not os.path.exists(traj_path):
            print(f"WARNING: Geo4D produced no trajectory for {sample_dir.name}, skipping")
            continue

        try:
            pred_c2w = read_tum_trajectory(traj_path)  # (N, 4, 4), c2w
        except Exception as e:
            print(f"Error reading trajectory {traj_path}: {e}")
            continue

        if len(pred_c2w) < n_gt_frames:
            print(f"WARNING: Geo4D only produced {len(pred_c2w)} poses but GT has {n_gt_frames} "
                  f"for {sample_dir.name}, skipping")
            continue
        pred_c2w = pred_c2w[:n_gt_frames]
        pred_w2c = np.linalg.inv(pred_c2w)

        # Aria GT is in Aria device frame; Geo4D predicts OpenCV cameras.
        if args.dataset == "aria":
            pred_w2c = apply_aria_correction(pred_w2c)

        errors = compute_relative_pose_errors(pred_w2c, gt_w2c)
        all_errors.append(errors)

        auc3  = compute_auc(errors, max_threshold=3.0)
        auc30 = compute_auc(errors, max_threshold=30.0)

        sample_out = output_dir / sample_dir.name
        sample_out.mkdir(parents=True, exist_ok=True)
        with open(sample_out / "pose_metrics.json", "w") as f:
            json.dump({"auc3": auc3, "auc30": auc30, "n_pairs": len(errors)}, f, indent=2)
        np.savez_compressed(str(sample_out / "pred_cameras.npz"),
                            extrinsics=np.linalg.inv(pred_w2c))
        plot_camera_trajectory(pred_w2c, gt_w2c, str(sample_out / "camera_trajectory.png"))

        count += 1

    if all_errors:
        all_errors_cat = np.concatenate(all_errors)
        final_auc3  = compute_auc(all_errors_cat, max_threshold=3.0)
        final_auc30 = compute_auc(all_errors_cat, max_threshold=30.0)
    else:
        final_auc3 = final_auc30 = 0.0

    final_stats = {"auc3": final_auc3, "auc30": final_auc30, "n_samples": count}
    print(f"\nFinal  AUC3={final_auc3:.2f}  AUC30={final_auc30:.2f}  (n={count} sequences)")

    wandb.log({"final/auc3": final_auc3, "final/auc30": final_auc30})
    with open(output_dir / "final_stats.json", "w") as f:
        json.dump(final_stats, f, indent=2)

    wandb.finish()


if __name__ == "__main__":
    main()
