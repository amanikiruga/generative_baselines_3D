"""
V3 — ChronoDepth depth eval against OURS_FINAL eval outputs.

Same ChronoDepth inference plumbing as V2 (eval_chronodepth_depth_aria.py),
but metrics are computed via Geo4D's vendored depth_evaluation
(per-video global LAD2 scale-shift in metric depth space).

GT depth is loaded as metric meters (`gt_depth_raw["depth"] * scale`).
ChronoDepth pred is in arbitrary-units relative depth ([0,1]); the per-video
LAD2 fit handles the unit conversion automatically.
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

# V3: depth metrics via vendored Geo4D code in generative_baselines/eval_common_v3.py
_GB_DIR = "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines"
if _GB_DIR not in sys.path:
    sys.path.insert(0, _GB_DIR)
from eval_common_v3 import (  # noqa: E402
    DEPTH_KEYS, aggregate, eval_depth_sequence, load_depth_metric_from_npz,
)

METRIC_KEYS = ("d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog")  # legacy V2


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
    parser.add_argument("--dataset", required=True,
                        help="dataset name (controls Geo4D max_depth clip)")
    parser.add_argument("--custom_run_name", default="chronodepth_depth_v3")
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

    per_seq: list[dict] = []
    count = 0

    for sample_dir in tqdm(sample_dirs, desc=f"ChronoDepth depth v3 ({args.dataset})"):
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

        # Load GT depth in metric meters via the shared helper.
        gt_metric = load_depth_metric_from_npz(gt_depth_path)

        # Load RGB video
        frames = np.stack(imageio.mimread(str(gt_rgb_path), memtest=False))  # (T, H, W, 3) uint8

        try:
            pred_01 = predict_depth(pipeline, frames, args.denoise_steps, args.max_res, args.device)
        except Exception as e:
            print(f"Error running ChronoDepth on {sample_dir.name}: {e}")
            import traceback; traceback.print_exc()
            continue

        # Resize pred to GT spatial resolution if needed
        if pred_01.shape != gt_metric.shape:
            pt = torch.from_numpy(pred_01).unsqueeze(1)
            pt = F.interpolate(pt, size=gt_metric.shape[-2:],
                               mode="bilinear", align_corners=False)
            pred_01 = pt.squeeze(1).numpy()

        # ChronoDepth pred_01 is *relative depth* in [0,1] (closer = darker,
        # farther = brighter — empirically Pearson +0.83 with gt metric depth).
        # No inversion needed: LAD2 fits s*pred + t = gt across both depth-space
        # signals.
        try:
            metrics = eval_depth_sequence(
                pred_depth_THW=pred_01,
                gt_depth_THW=gt_metric,
                dataset=args.dataset,
                normalize_unit_per_video=True,
            )
        except Exception as e:
            print(f"Error evaluating {sample_dir.name}: {e}")
            import traceback; traceback.print_exc()
            continue

        sample_out = output_dir / sample_dir.name
        sample_out.mkdir(parents=True, exist_ok=True)
        with open(sample_out / "depth_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        np.savez_compressed(str(sample_out / "pred_depth_raw.npz"), depth=pred_01)
        wandb.log({f"sample/{k}": metrics[k] for k in DEPTH_KEYS})
        per_seq.append(metrics)
        count += 1

    final = aggregate(per_seq, DEPTH_KEYS)
    print(f"\nFinal (n={final['n_samples']})  "
          + "  ".join(f"{k}={final[k]:.4f}" for k in DEPTH_KEYS))
    wandb.log({f"final/{k}": final[k] for k in DEPTH_KEYS})
    with open(output_dir / "final_stats.json", "w") as f:
        json.dump(final, f, indent=2)
    wandb.finish()


if __name__ == "__main__":
    main()
