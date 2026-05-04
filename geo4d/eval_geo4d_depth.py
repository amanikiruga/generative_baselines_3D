"""
Compute depth metrics for Geo4D from existing eval outputs.

Runs Geo4D on gt_rgb.mp4 from each sample dir (produced by
inference_single_gpu_rgb_to_ray_depth_aria_eval.py), extracts post-optimised
depth maps via scene.get_depthmaps(), aligns to GT depth (gt_depth_raw.npz)
via RANSAC + least-squares scale+shift, and computes depth metrics in [0, 1]
space — directly comparable to eval_rgb_to_ray_depth_aria_depth.py and
eval_depth_anything3_outputs.py.

GT depth is loaded from gt_depth_raw.npz (saved by inference scripts in
[-1, 1] space) and converted to [0, 1].  Geo4D depth is in arbitrary metric
units; RANSAC alignment brings it into the same [0, 1] space as GT before
metrics are computed.
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

GEO4D_DIR = "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D"

METRIC_KEYS = ("d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog")


# ---------------------------------------------------------------------------
# GT depth loader  (verbatim from eval_depth_anything3_outputs.py)
# ---------------------------------------------------------------------------

def read_gt_depth_npz(npz_path: Path) -> np.ndarray:
    """
    Reads GT depth from gt_depth_raw.npz (saved by inference scripts in [-1, 1] space).
    Converts [-1, 1] → [0, 1] so downstream metrics treat 0 as invalid (same convention
    as raw_npy and mp4 sources).  Returns (T, H, W) float32.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing gt_depth_raw.npz: {npz_path}")
    depth = np.load(npz_path)["depth"].astype(np.float32)  # (T, H, W) in [-1, 1]
    if depth.ndim != 3:
        raise ValueError(f"Expected shape (T, H, W), got {depth.shape}")
    # [-1, 1] → [0, 1]; values <= -1 (e.g. padding) map to 0 and are treated as invalid
    depth_01 = np.clip(depth * 0.5 + 0.5, 0.0, 1.0)
    return depth_01


# ---------------------------------------------------------------------------
# Depth metrics  (verbatim from eval_depth_anything3_outputs.py)
# ---------------------------------------------------------------------------

def compute_depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray) -> dict | None:
    """Returns None if fewer than 10 valid pixels (sample is skipped entirely)."""
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(f"Shape mismatch: pred {pred_depth.shape} vs gt {gt_depth.shape}")

    valid = (gt_depth > 0) & (pred_depth > 0) & np.isfinite(pred_depth) & np.isfinite(gt_depth)
    if valid.sum() < 10:
        return None

    pred = pred_depth[valid].astype(np.float64)
    gt   = gt_depth[valid].astype(np.float64)

    diff     = pred - gt
    diff_log = np.log(pred) - np.log(gt)
    ratio    = np.maximum(pred / gt, gt / pred)

    return {
        "d1":      float((ratio < 1.25     ).mean()),
        "d2":      float((ratio < 1.25 ** 2).mean()),
        "d3":      float((ratio < 1.25 ** 3).mean()),
        "abs_rel": float(np.mean(np.abs(diff) / gt)),
        "sq_rel":  float(np.mean(diff ** 2 / gt)),
        "rmse":    float(np.sqrt(np.mean(diff ** 2))),
        "rmse_log":float(np.sqrt(np.mean(diff_log ** 2))),
        "log10":   float(np.mean(np.abs(np.log10(pred) - np.log10(gt)))),
        "silog":   float(np.sqrt(np.mean(diff_log ** 2) - 0.5 * np.mean(diff_log) ** 2)),
    }


def fit_scale_shift_ransac(pred: np.ndarray, gt: np.ndarray, max_iters: int = 200, min_inliers: int = 1000) -> tuple[float, float]:
    """Fits scale (s) and shift (t) such that: s * pred + t ~ gt"""
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)

    valid = (gt_flat > 0) & np.isfinite(pred_flat) & np.isfinite(gt_flat)
    pred_valid = pred_flat[valid]
    gt_valid = gt_flat[valid]

    if pred_valid.size < 2:
        return 1.0, 0.0

    rng = np.random.default_rng(0)
    best_inliers = -1
    best_s, best_t = 1.0, 0.0
    best_mse = np.inf
    n_points = pred_valid.size

    for _ in range(max_iters):
        idx = rng.choice(n_points, size=2, replace=False)
        p1, p2 = pred_valid[idx[0]], pred_valid[idx[1]]
        g1, g2 = gt_valid[idx[0]], gt_valid[idx[1]]

        denom = (p1 - p2)
        if denom == 0: continue

        s = (g1 - g2) / denom
        if s <= 0: continue
        t = g1 - s * p1

        estimate = s * pred_valid + t
        residuals = np.abs(estimate - gt_valid)

        median_res = np.median(residuals)
        mad = np.median(np.abs(residuals - median_res))
        threshold = mad if mad > 0 else 1e-3

        inliers = residuals <= (3 * threshold)
        num_inliers = np.count_nonzero(inliers)

        if num_inliers < min_inliers:
            continue

        if num_inliers > best_inliers:
            p_in = pred_valid[inliers]
            g_in = gt_valid[inliers]
            A = np.vstack([p_in, np.ones_like(p_in)]).T
            solution, _, _, _ = np.linalg.lstsq(A, g_in, rcond=None)
            s_ls, t_ls = solution[0], solution[1]

            if s_ls > 0:
                mse = np.mean((s_ls * p_in + t_ls - g_in) ** 2)
                if num_inliers > best_inliers or (num_inliers == best_inliers and mse < best_mse):
                    best_inliers = num_inliers
                    best_mse = mse
                    best_s, best_t = float(s_ls), float(t_ls)

    return best_s, best_t


# ---------------------------------------------------------------------------
# Visualisation  (verbatim from eval_depth_anything3_outputs.py)
# ---------------------------------------------------------------------------

def plot_depth_comparison(
    gt_depth: np.ndarray,
    pred_depth_raw: np.ndarray,
    pred_depth_aligned: np.ndarray,
    metrics_raw: dict,
    metrics_aligned: dict,
    output_path: str,
) -> None:
    """Save GT / raw-pred / aligned-pred side-by-side (first frame only)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    vmin = gt_depth[0].min()
    vmax = gt_depth[0].max()

    im0 = axes[0].imshow(gt_depth[0], cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0].set_title("GT depth")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_depth_raw[0], cmap="turbo")
    axes[1].set_title(
        f"Pred (raw)\nAbsRel={metrics_raw['abs_rel']:.3f}  d1={metrics_raw['d1']:.3f}"
    )
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(pred_depth_aligned[0], cmap="turbo", vmin=vmin, vmax=vmax)
    axes[2].set_title(
        f"Pred (aligned)\nAbsRel={metrics_aligned['abs_rel']:.3f}  d1={metrics_aligned['d1']:.3f}"
    )
    plt.colorbar(im2, ax=axes[2])

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def accumulate_metrics(stats: dict, metrics: dict) -> dict:
    """Sample-averaged accumulation."""
    n = stats["count"]
    for k in METRIC_KEYS:
        stats[k] = (stats[k] * n + metrics[k]) / (n + 1)
    stats["count"] += 1
    return stats


# ---------------------------------------------------------------------------
# Geo4D model loading  (verbatim from eval_geo4d_pose.py)
# ---------------------------------------------------------------------------

def setup_geo4d_path():
    if GEO4D_DIR not in sys.path:
        sys.path.insert(0, GEO4D_DIR)


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


# ---------------------------------------------------------------------------
# Geo4D depth inference
# ---------------------------------------------------------------------------

def infer_geo4d_depth(
    model, config, pointmap_vae,
    video_path: str,
    ddim_steps: int = 5,
    height: int = 320,
    width: int = 512,
    gpu_no: int = 0,
) -> np.ndarray:
    """
    Run Geo4D on a single video and return post-optimised depth maps.
    Returns (T, H, W) float32 numpy array in arbitrary metric units.
    """
    from einops import rearrange
    from pytorch_lightning import seed_everything
    from utils.funcs import load_video_batch
    from scripts.evaluation.test_geo4d import (
        image_guided_synthesis, post_optimization,
        get_sky_mask, get_far_away_mask, denormalize_pc_bbox2,
        raymap_to_camera_matrix, decode_pm_confhead,
    )

    seed_everything(123)

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

    # Extract post-optimised depth: list of T tensors (H, W) in metric units
    depthmaps = scene.get_depthmaps()
    depth = torch.stack(depthmaps, dim=0).detach().cpu().float().numpy()  # (T, H, W)
    return depth


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--rerun", action="store_true",
                        help="Rerun Geo4D even if cached depth exists")
    parser.add_argument("--custom_run_name", type=str, default="geo4d_depth_eval")
    args = parser.parse_args()

    setup_geo4d_path()

    wandb.init(project="video_world_model", name=args.custom_run_name)
    wandb.config.update(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available — will be very slow or broken")

    print(f"Loading Geo4D model from {args.ckpt_path} ...")
    model, config, pointmap_vae = load_geo4d_model(args.ckpt_path, args.config, gpu_no=0)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted([
        p for p in input_dir.iterdir()
        if p.is_dir() and p.name.startswith("sample_")
    ])

    stats_raw     = {k: 0.0 for k in METRIC_KEYS}
    stats_aligned = {k: 0.0 for k in METRIC_KEYS}
    stats_raw["count"] = 0
    stats_aligned["count"] = 0

    for sample_dir in tqdm(sample_dirs, desc="Geo4D depth eval"):
        if args.max_samples is not None and stats_raw["count"] >= args.max_samples:
            break

        gt_rgb_path       = sample_dir / "gt_rgb.mp4"
        gt_depth_npz_path = sample_dir / "gt_depth_raw.npz"

        if not gt_rgb_path.exists():
            raise FileNotFoundError(f"{gt_rgb_path} not found — inference output dir is incomplete")

        gt_depth = read_gt_depth_npz(gt_depth_npz_path)  # (T, H, W) in [0, 1]

        # Try to load cached depth to avoid re-running Geo4D
        sample_out   = output_dir / sample_dir.name
        cached_depth = sample_out / "pred_depth_geo4d.npz"

        if cached_depth.exists() and not args.rerun:
            pred_depth = np.load(cached_depth)["depth"]
        else:
            try:
                pred_depth = infer_geo4d_depth(
                    model, config, pointmap_vae,
                    video_path=str(gt_rgb_path),
                    ddim_steps=args.ddim_steps,
                    height=args.height,
                    width=args.width,
                    gpu_no=0,
                )
            except Exception as e:
                print(f"Error running Geo4D on {sample_dir.name}: {e}")
                import traceback; traceback.print_exc()
                continue

            sample_out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cached_depth, depth=pred_depth)

        # Resize pred to GT spatial resolution if needed
        if pred_depth.shape != gt_depth.shape:
            t_pred = torch.from_numpy(pred_depth).unsqueeze(1)  # (T, 1, H, W)
            t_pred = F.interpolate(t_pred, size=gt_depth.shape[-2:], mode="bilinear", align_corners=False)
            pred_depth = t_pred.squeeze(1).numpy()

        # --- Raw (unaligned) metrics ---
        # Note: pred is in arbitrary metric units, gt is in [0,1]; raw metrics are not
        # meaningful in absolute terms but are included for completeness.
        metrics_raw = compute_depth_metrics(pred_depth, gt_depth)
        if metrics_raw is None:
            print(f"Skipping {sample_dir.name}: fewer than 10 valid pixels (raw)")
            continue

        # --- Aligned metrics: RANSAC + LS, clamp to [0, 1], compute in [0, 1] space ---
        scale, shift = fit_scale_shift_ransac(pred_depth, gt_depth)
        pred_depth_aligned = np.clip(scale * pred_depth + shift, 0.0, 1.0)

        metrics_aligned = compute_depth_metrics(pred_depth_aligned, gt_depth)
        if metrics_aligned is None:
            print(f"Skipping {sample_dir.name}: fewer than 10 valid pixels after alignment")
            continue

        stats_raw     = accumulate_metrics(stats_raw,     metrics_raw)
        stats_aligned = accumulate_metrics(stats_aligned, metrics_aligned)

        sample_out.mkdir(parents=True, exist_ok=True)
        with open(sample_out / "depth_metrics.json", "w") as f:
            json.dump({
                "raw":         metrics_raw,
                "aligned":     metrics_aligned,
                "align_scale": scale,
                "align_shift": shift,
            }, f, indent=2)

        plot_depth_comparison(
            gt_depth, pred_depth, pred_depth_aligned, metrics_raw, metrics_aligned,
            str(sample_out / "depth_comparison.png"),
        )

        use_wandb = (stats_raw["count"] % 10 == 0)
        if use_wandb:
            wandb.log({
                **{f"sample/raw/{k}":     metrics_raw[k]     for k in METRIC_KEYS},
                **{f"sample/aligned/{k}": metrics_aligned[k] for k in METRIC_KEYS},
                "sample/align_scale": scale,
                "sample/align_shift": shift,
            })

    n = stats_raw["count"]
    print(f"\nFinal stats (n={n} samples):")
    print("  Raw (no alignment):")
    for k in METRIC_KEYS:
        print(f"    {k}: {stats_raw[k]:.4f}")
    print("  Aligned (RANSAC + least-squares):")
    for k in METRIC_KEYS:
        print(f"    {k}: {stats_aligned[k]:.4f}")

    wandb.log({
        **{f"final/raw/{k}":     stats_raw[k]     for k in METRIC_KEYS},
        **{f"final/aligned/{k}": stats_aligned[k] for k in METRIC_KEYS},
    })
    with open(output_dir / "geo4d_final_stats.json", "w") as f:
        json.dump({"raw": stats_raw, "aligned": stats_aligned}, f, indent=2)

    wandb.finish()


if __name__ == "__main__":
    main()
