"""
Evaluate ChronoDepth video depth estimation on Aria eval outputs.

Reads gt_rgb.mp4 from each sample dir (produced by
inference_single_gpu_rgb_to_ray_depth_aria_eval.py), runs ChronoDepth to
predict per-frame depth maps, aligns to GT depth (gt_depth_raw.npz) via
RANSAC + least-squares scale+shift, then computes depth metrics in [0, 1]
space — directly comparable to eval_rgb_to_ray_depth_aria_depth.py.

GT depth is in [-1, 1] normalised space (from gt_depth_raw.npz) and converted
to [0, 1] before alignment/metrics.  ChronoDepth outputs [0, 1] relative depth
which is aligned to GT before metrics are computed.

Run from: /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

CHRONODEPTH_DIR = "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/ChronoDepth"
sys.path.insert(0, CHRONODEPTH_DIR)

METRIC_KEYS = ("d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog")


# ---------------------------------------------------------------------------
# Depth metric helpers (verbatim from eval_rgb_to_ray_depth_aria_depth.py)
# ---------------------------------------------------------------------------

def compute_depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict | None:
    valid = (gt > 0) & (pred > 0) & np.isfinite(pred) & np.isfinite(gt)
    if valid.sum() < 10:
        return None
    p = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)
    diff = p - g
    diff_log = np.log(p) - np.log(g)
    ratio = np.maximum(p / g, g / p)
    return {
        "d1":       float((ratio < 1.25     ).mean()),
        "d2":       float((ratio < 1.25 ** 2).mean()),
        "d3":       float((ratio < 1.25 ** 3).mean()),
        "abs_rel":  float(np.mean(np.abs(diff) / g)),
        "sq_rel":   float(np.mean(diff ** 2 / g)),
        "rmse":     float(np.sqrt(np.mean(diff ** 2))),
        "rmse_log": float(np.sqrt(np.mean(diff_log ** 2))),
        "log10":    float(np.mean(np.abs(np.log10(p) - np.log10(g)))),
        "silog":    float(np.sqrt(np.mean(diff_log ** 2) - 0.5 * np.mean(diff_log) ** 2)),
    }


def fit_scale_shift_ransac(pred: np.ndarray, gt: np.ndarray,
                            max_iters: int = 200, min_inliers: int = 1000,
                            ) -> tuple[float, float]:
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)
    valid = (gt_flat > 0) & np.isfinite(pred_flat) & np.isfinite(gt_flat)
    pv, gv = pred_flat[valid], gt_flat[valid]
    if pv.size < 2:
        return 1.0, 0.0
    rng = np.random.default_rng(0)
    best_inliers, best_s, best_t, best_mse = -1, 1.0, 0.0, np.inf
    n = pv.size
    for _ in range(max_iters):
        idx = rng.choice(n, size=2, replace=False)
        p1, p2 = pv[idx[0]], pv[idx[1]]
        g1, g2 = gv[idx[0]], gv[idx[1]]
        denom = p1 - p2
        if denom == 0:
            continue
        s = (g1 - g2) / denom
        if s <= 0:
            continue
        t = g1 - s * p1
        res = np.abs(s * pv + t - gv)
        med = np.median(res)
        mad = np.median(np.abs(res - med)) or 1e-3
        inlier_mask = res <= 3 * mad
        ni = int(inlier_mask.sum())
        if ni < min_inliers:
            continue
        A = np.vstack([pv[inlier_mask], np.ones(ni)]).T
        sol, _, _, _ = np.linalg.lstsq(A, gv[inlier_mask], rcond=None)
        s_ls, t_ls = float(sol[0]), float(sol[1])
        if s_ls <= 0:
            continue
        mse = float(np.mean((s_ls * pv[inlier_mask] + t_ls - gv[inlier_mask]) ** 2))
        if ni > best_inliers or (ni == best_inliers and mse < best_mse):
            best_inliers, best_s, best_t, best_mse = ni, s_ls, t_ls, mse
    return best_s, best_t


def plot_depth_comparison(pred, gt, pred_aligned, metrics_raw, metrics_aligned, path,
                           ours_01: np.ndarray | None = None, ours_metrics: dict | None = None):
    """
    Side-by-side depth comparison.
    Columns: GT | ours (our method) | ChronoDepth raw | ChronoDepth aligned
    If ours_01 is None, falls back to 3-column layout.
    """
    if ours_01 is not None:
        ncols = 4
        imgs   = [gt[0], ours_01[0], pred[0], pred_aligned[0]]
        titles = [
            "GT depth",
            f"Ours (d1={ours_metrics['d1']:.3f})" if ours_metrics else "Ours",
            f"ChronoDepth raw (d1={metrics_raw['d1']:.3f})",
            f"ChronoDepth aligned (d1={metrics_aligned['d1']:.3f})",
        ]
    else:
        ncols = 3
        imgs   = [gt[0], pred[0], pred_aligned[0]]
        titles = ["GT depth",
                  f"ChronoDepth raw (d1={metrics_raw['d1']:.3f})",
                  f"ChronoDepth aligned (d1={metrics_aligned['d1']:.3f})"]

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    vmin, vmax = gt[0].min(), gt[0].max()
    for ax, img, title in zip(axes, imgs, titles):
        im = ax.imshow(img, cmap="turbo", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# ChronoDepth inference
# ---------------------------------------------------------------------------

def load_chronodepth(unet_id: str = "jhshao/ChronoDepth-v1",
                     model_base: str = "stabilityai/stable-video-diffusion-img2vid-xt",
                     device: str = "cuda") -> object:
    from chronodepth.unet_chronodepth import (
        DiffusersUNetSpatioTemporalConditionModelChronodepth,
    )
    from chronodepth.chronodepth_pipeline import ChronoDepthPipeline

    unet = DiffusersUNetSpatioTemporalConditionModelChronodepth.from_pretrained(
        unet_id, low_cpu_mem_usage=True, torch_dtype=torch.float16,
    )
    pipeline = ChronoDepthPipeline.from_pretrained(
        model_base, unet=unet, torch_dtype=torch.float16, variant="fp16",
    )
    pipeline.n_tokens = 10
    pipeline.chunk_size = 5
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass
    pipeline.to(device)
    return pipeline


@torch.no_grad()
def predict_depth(pipeline, frames_hwc_uint8: np.ndarray,
                  denoise_steps: int = 5, max_res: int = 1024,
                  device: str = "cuda") -> np.ndarray:
    """
    Args:
        frames_hwc_uint8: (T, H, W, 3) uint8 RGB frames
    Returns:
        depth: (T, H, W) float32 in [0, 1]
    """
    from src.utils.video_utils import resize_max_res

    T, H, W = frames_hwc_uint8.shape[:3]
    video_rgb = resize_max_res(frames_hwc_uint8, max_res).astype(np.float32) / 255.0

    generator = torch.Generator(device=device).manual_seed(42)
    pipe_out = pipeline(
        video_rgb,
        num_inference_steps=denoise_steps,
        decode_chunk_size=8,
        motion_bucket_id=127,
        fps=7,
        noise_aug_strength=0.0,
        generator=generator,
        infer_mode="ours",
        sigma_epsilon=-4.0,
    )
    depth = torch.from_numpy(pipe_out.frames).to(device)   # (T, 1, H', W') or (T, H', W')
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)
    depth = depth.clamp(0, 1).squeeze(1)                   # (T, H, W)
    return depth.cpu().float().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Output dir from inference_single_gpu_rgb_to_ray_depth_aria_eval.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--unet", default="jhshao/ChronoDepth-v1")
    parser.add_argument("--model_base",
                        default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--denoise_steps", type=int, default=5)
    parser.add_argument("--max_res", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--custom_run_name", default="chronodepth_depth_aria")
    args = parser.parse_args()

    wandb.init(project="video_world_model", name=args.custom_run_name)
    wandb.config.update(vars(args))

    print("Loading ChronoDepth …")
    pipeline = load_chronodepth(args.unet, args.model_base, args.device)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(p for p in input_dir.iterdir()
                         if p.is_dir() and p.name.startswith("sample_"))

    agg_raw     = {k: 0.0 for k in METRIC_KEYS}
    agg_aligned = {k: 0.0 for k in METRIC_KEYS}
    count = 0

    for sample_dir in tqdm(sample_dirs, desc="ChronoDepth depth eval"):
        if args.max_samples is not None and count >= args.max_samples:
            break

        gt_rgb_path   = sample_dir / "gt_rgb.mp4"
        gt_depth_path = sample_dir / "gt_depth_raw.npz"

        if not gt_rgb_path.exists():
            print(f"Skipping {sample_dir.name}: missing gt_rgb.mp4")
            continue
        if not gt_depth_path.exists():
            print(f"Skipping {sample_dir.name}: missing gt_depth_raw.npz")
            continue

        # Load GT depth: [-1, 1] → [0, 1]
        gt_norm = np.load(gt_depth_path)["depth"].astype(np.float32)  # (T, H, W) in [-1, 1]
        gt_01   = np.clip(gt_norm * 0.5 + 0.5, 0.0, 1.0)

        # Load our method's predicted depth if present (pred_depth_raw.npz saved by inference script)
        ours_depth_path = sample_dir / "pred_depth_raw.npz"
        ours_01 = None
        ours_metrics = None
        if ours_depth_path.exists():
            ours_norm = np.load(ours_depth_path)["depth"].astype(np.float32)
            ours_01   = np.clip(ours_norm * 0.5 + 0.5, 0.0, 1.0)
            if ours_01.shape != gt_01.shape:
                ot = torch.from_numpy(ours_01).unsqueeze(1)
                ot = F.interpolate(ot, size=gt_01.shape[-2:], mode="bilinear", align_corners=False)
                ours_01 = ot.squeeze(1).numpy()
            ours_metrics = compute_depth_metrics(ours_01, gt_01)

        # Load RGB video
        frames = np.stack(imageio.mimread(str(gt_rgb_path), memtest=False))  # (T, H, W, 3) uint8

        try:
            pred_01 = predict_depth(pipeline, frames, args.denoise_steps, args.max_res, args.device)
        except Exception as e:
            print(f"Error running ChronoDepth on {sample_dir.name}: {e}")
            import traceback; traceback.print_exc()
            continue

        # Resize pred to GT spatial resolution if needed
        if pred_01.shape != gt_01.shape:
            pt = torch.from_numpy(pred_01).unsqueeze(1)
            pt = F.interpolate(pt, size=gt_01.shape[-2:], mode="bilinear", align_corners=False)
            pred_01 = pt.squeeze(1).numpy()

        # Raw metrics
        metrics_raw = compute_depth_metrics(pred_01, gt_01)
        if metrics_raw is None:
            print(f"Skipping {sample_dir.name}: fewer than 10 valid pixels")
            continue

        # Aligned metrics
        scale, shift = fit_scale_shift_ransac(pred_01, gt_01)
        pred_01_aligned = np.clip(scale * pred_01 + shift, 0.0, 1.0)
        metrics_aligned = compute_depth_metrics(pred_01_aligned, gt_01)
        if metrics_aligned is None:
            print(f"Skipping {sample_dir.name}: fewer than 10 valid pixels after alignment")
            continue

        for k in METRIC_KEYS:
            agg_raw[k]     = (agg_raw[k]     * count + metrics_raw[k])     / (count + 1)
            agg_aligned[k] = (agg_aligned[k] * count + metrics_aligned[k]) / (count + 1)
        count += 1

        sample_out = output_dir / sample_dir.name
        sample_out.mkdir(parents=True, exist_ok=True)
        with open(sample_out / "depth_metrics.json", "w") as f:
            json.dump({"raw": metrics_raw, "aligned": metrics_aligned,
                       "align_scale": scale, "align_shift": shift}, f, indent=2)

        np.savez_compressed(str(sample_out / "pred_depth_raw.npz"), depth=pred_01)
        plot_depth_comparison(pred_01, gt_01, pred_01_aligned,
                              metrics_raw, metrics_aligned,
                              str(sample_out / "depth_comparison.png"),
                              ours_01=ours_01, ours_metrics=ours_metrics)

        # Colorised depth videos using turbo colormap
        import matplotlib.cm as cm
        _cmap = cm.get_cmap("turbo")

        def _colorize(depth_thw: np.ndarray) -> np.ndarray:
            """(T, H, W) float [0,1] → (T, H, W, 3) uint8 via turbo colormap."""
            return (_cmap(depth_thw)[..., :3] * 255).astype(np.uint8)

        imageio.mimwrite(str(sample_out / "pred_depth.mp4"),
                         _colorize(pred_01), fps=10)
        imageio.mimwrite(str(sample_out / "pred_depth_aligned.mp4"),
                         _colorize(pred_01_aligned), fps=10)
        imageio.mimwrite(str(sample_out / "gt_depth.mp4"),
                         _colorize(gt_01), fps=10)

        # Side-by-side comparison video: GT | ours | ChronoDepth (aligned)
        cols = [_colorize(gt_01), _colorize(pred_01_aligned)]
        labels = ["GT", "ChronoDepth (aligned)"]
        if ours_01 is not None:
            cols.insert(1, _colorize(ours_01))
            labels.insert(1, "Ours")
        comparison_depth = np.concatenate(cols, axis=2)  # (T, H, W*N, 3)
        imageio.mimwrite(str(sample_out / "comparison_depth.mp4"),
                         comparison_depth, fps=10)

        wandb.log({**{f"sample/raw/{k}": metrics_raw[k] for k in METRIC_KEYS},
                   **{f"sample/aligned/{k}": metrics_aligned[k] for k in METRIC_KEYS}})

    final_stats = {"raw": agg_raw, "aligned": agg_aligned, "n_samples": count}
    print(f"\nFinal (n={count})  raw d1={agg_raw['d1']:.4f}  aligned d1={agg_aligned['d1']:.4f}")
    wandb.log({**{f"final/raw/{k}": agg_raw[k] for k in METRIC_KEYS},
               **{f"final/aligned/{k}": agg_aligned[k] for k in METRIC_KEYS}})
    with open(output_dir / "final_stats.json", "w") as f:
        json.dump(final_stats, f, indent=2)
    wandb.finish()


if __name__ == "__main__":
    main()
