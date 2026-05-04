"""
Baseline evaluation: Wan2.1-FLF2V-14B on the same RE10K test samples used by our model.

Conditions the model on the exact same first and last RGB frames from each test video,
generates the same number of frames (13), then evaluates PSNR/LPIPS/SSIM only on the
generated frames [1:-1] (excluding the conditioned first/last frames) — identical to
the evaluation protocol in inference_single_gpu_ray2rgb_firstlast_eval.py.

Wan2.1-FLF2V generates at a higher resolution (~544×736 for 3:4 landscape input at
max_area=832*480) and the output is resized back to the native dataset resolution
(240×320) before metric computation for a fair apple-to-apple comparison.
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra import compose, initialize
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

# ── project root ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── Wan2.1 repo ─────────────────────────────────────────────────────────────
WAN21_ROOT = Path(__file__).resolve().parent.parent.parent / "Wan2.1"
sys.path.insert(0, str(WAN21_ROOT))

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS

from datasets.re10k import RE10KDataset

DEVICE    = "cuda"
TASK      = "flf2v-14B"
HF_REPO   = "Wan-AI/Wan2.1-FLF2V-14B-720P"


def ensure_ckpt(ckpt_dir: str) -> str:
    """Download the FLF2V checkpoint from HuggingFace if it isn't already present.

    Uses huggingface_hub.snapshot_download(), which is equivalent to:
        huggingface-cli download Wan-AI/Wan2.1-FLF2V-14B-720P --local-dir <ckpt_dir>

    Returns the resolved local directory path.
    """
    ckpt_path = Path(ckpt_dir)
    # Consider the checkpoint present if the directory exists and has at least
    # one .safetensors or .bin file (model weights).
    weight_files = list(ckpt_path.glob("*.safetensors")) + list(ckpt_path.glob("*.bin"))
    if ckpt_path.exists() and weight_files:
        print(f"INFO: Checkpoint found at {ckpt_path} ({len(weight_files)} weight file(s)).")
        return str(ckpt_path)

    print(f"INFO: Checkpoint not found at {ckpt_path}. Downloading {HF_REPO} from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for auto-download. "
            "Install it with:  pip install 'huggingface_hub[cli]'"
        )
    local_dir = snapshot_download(
        repo_id=HF_REPO,
        local_dir=str(ckpt_path),
        repo_type="model",
    )
    print(f"INFO: Downloaded to {local_dir}")
    return local_dir

# frame_num must satisfy 4n+1.  13 = 4*3+1 ✓  (matches our model's T=13)
# Wan2.1-FLF2V has the mask hardcoded to 81 frames in first_last_frame2video.py
# (line 231: `msk = torch.ones(1, 81, ...)`).  Other frame counts crash.
FLFF2V_FRAME_NUM = 81
# Smallest supported max_area for FLF2V.  Model internally resizes to preserve
# the input aspect ratio up to this pixel budget.
MAX_AREA    = MAX_AREA_CONFIGS["832*480"]   # 399 360 px


# ── helpers shared with the ray2rgb eval script ──────────────────────────────

def unnormalize_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(T,3,H,W) in [-1,1] → (T,3,H,W) uint8"""
    return ((t * 0.5 + 0.5).clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8)


def resize_video(video_tchw: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(T,C,H',W') → (T,C,h,w) bilinear."""
    if video_tchw.shape[-2:] == (h, w):
        return video_tchw
    return F.interpolate(video_tchw.float(), size=(h, w), mode="bilinear", align_corners=False)


@torch.no_grad()
def compute_rgb_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict:
    """(T,3,H,W) in [-1,1] → {mse, psnr}"""
    p = (pred * 0.5 + 0.5).clamp(0, 1)
    g = (gt   * 0.5 + 0.5).clamp(0, 1)
    mse  = float(torch.nn.functional.mse_loss(p, g).item())
    psnr = float(-10.0 * math.log10(mse)) if mse > 0 else float("inf")
    return {"mse": mse, "psnr": psnr}


@torch.no_grad()
def compute_per_frame_psnr(pred: torch.Tensor, gt: torch.Tensor) -> list:
    p = (pred * 0.5 + 0.5).clamp(0, 1)
    g = (gt   * 0.5 + 0.5).clamp(0, 1)
    out = []
    for t in range(p.shape[0]):
        mse = float(torch.nn.functional.mse_loss(p[t], g[t]).item())
        out.append(float(-10.0 * math.log10(mse)) if mse > 0 else float("inf"))
    return out


@torch.no_grad()
def compute_per_frame_lpips(pred: torch.Tensor, gt: torch.Tensor, lpips_fn) -> list:
    return [float(lpips_fn(pred[t:t+1].float(), gt[t:t+1].float()).item())
            for t in range(pred.shape[0])]


@torch.no_grad()
def compute_per_frame_ssim(pred: torch.Tensor, gt: torch.Tensor) -> list:
    from skimage.metrics import structural_similarity
    p = (pred * 0.5 + 0.5).clamp(0, 1).float().cpu().numpy()
    g = (gt   * 0.5 + 0.5).clamp(0, 1).float().cpu().numpy()
    return [float(structural_similarity(
                p[t].transpose(1, 2, 0), g[t].transpose(1, 2, 0),
                data_range=1.0, channel_axis=2))
            for t in range(p.shape[0])]


def subsample_to(video_tchw: torch.Tensor, target_t: int) -> torch.Tensor:
    """Uniformly subsample (T,C,H,W) to target_t frames, keeping first and last exactly."""
    T = video_tchw.shape[0]
    if T == target_t:
        return video_tchw
    indices = torch.linspace(0, T - 1, target_t).round().long()
    return video_tchw[indices]


def _accumulate(stats: Dict, metrics: Dict) -> Dict:
    n = stats["count"]
    stats["mse"]   = (stats["mse"]  * n + metrics["mse"])  / (n + 1)
    stats["psnr"]  = (stats["psnr"] * n + metrics["psnr"]) / (n + 1)
    stats["count"] += 1
    return stats


def _save_video(path: str, video_tchw: np.ndarray, fps: int) -> None:
    imageio.mimwrite(path, rearrange(video_tchw, "t c h w -> t h w c"), fps=fps)


def tensor_to_pil(frame_chw: torch.Tensor) -> Image.Image:
    """(3,H,W) in [-1,1] → PIL RGB image."""
    arr = ((frame_chw * 0.5 + 0.5).clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr.transpose(1, 2, 0))  # HWC


# ── main eval loop ────────────────────────────────────────────────────────────

def load_re10k_cfg(split: str):
    with initialize(version_base=None, config_path="../configurations"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=exp_video",
                "algorithm=wan_t2v_ray_mot",
                "dataset=re10k",
                "experiment.tasks=[test]",
            ],
        )
        OmegaConf.resolve(cfg)
    return cfg


@torch.no_grad()
def infer_and_eval_flf2v_baseline(
    wan_flf2v,
    dataset: RE10KDataset,
    max_samples: int,
    output_dir: str,
    fps: int,
    save_stats: bool = False,
    sample_steps: int = 50,
    guide_scale: float = 5.0,
    seed: int = 42,
) -> Dict:
    # Native dataset resolution
    H, W = dataset[0]["height"], dataset[0]["width"]

    lpips_fn = None
    if save_stats:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(DEVICE).eval()

    stats = {"mse": 0.0, "psnr": 0.0, "count": 0}
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in tqdm(loader, desc="eval flf2v baseline"):
        if stats["count"] >= max_samples:
            break

        # ── extract first / last frames as PIL ──────────────────────────────
        videos = batch["videos"].squeeze(0)   # (T, 3, H, W) in [-1, 1]
        T = videos.shape[0]
        first_pil = tensor_to_pil(videos[0])
        last_pil  = tensor_to_pil(videos[T - 1])

        prompt = batch["prompts"][0] if "prompts" in batch else ""

        # ── run Wan2.1-FLF2V ────────────────────────────────────────────────
        # frame_num must be 81 — the mask is hardcoded in first_last_frame2video.py
        video_cthw = wan_flf2v.generate(
            prompt,
            first_pil,
            last_pil,
            max_area=MAX_AREA,
            frame_num=FLFF2V_FRAME_NUM,
            shift=16,
            sample_solver="unipc",
            sampling_steps=sample_steps,
            guide_scale=guide_scale,
            seed=seed,
            offload_model=False,
        )
        # video_cthw: (C, 81, H', W') in [-1, 1] on CPU (Wan returns CPU tensor)
        video_cthw = video_cthw.to(DEVICE)

        # rearrange to (81, C, H', W')
        pred_tchw = rearrange(video_cthw, "c t h w -> t c h w")

        # resize to native dataset resolution for fair metric comparison
        pred_tchw = resize_video(pred_tchw, H, W)

        # Temporally subsample 81 → T frames so first/last map exactly to the
        # GT conditioning frames and intermediate positions are proportionally aligned.
        pred_tchw = subsample_to(pred_tchw, T)

        # gt video at native res, on DEVICE
        gt_tchw = videos.to(DEVICE)   # (T, 3, H, W) in [-1, 1]

        # ── metrics on generated frames only (exclude conditioned first/last) ──
        rgb_metrics    = compute_rgb_metrics(pred_tchw[1:-1], gt_tchw[1:-1])
        per_frame_psnr = compute_per_frame_psnr(pred_tchw, gt_tchw)  # full list
        stats = _accumulate(stats, rgb_metrics)

        # ── save sample ──────────────────────────────────────────────────────
        sample_idx = stats["count"] - 1
        sample_dir = os.path.join(output_dir, f"sample_{str(sample_idx).zfill(5)}")
        os.makedirs(sample_dir, exist_ok=True)

        pred_uint8 = unnormalize_to_uint8(pred_tchw)
        gt_uint8   = unnormalize_to_uint8(gt_tchw)

        _save_video(os.path.join(sample_dir, "pred_rgb.mp4"), pred_uint8, fps=fps)
        _save_video(os.path.join(sample_dir, "gt_rgb.mp4"),   gt_uint8,   fps=fps)

        side_by_side = np.concatenate([pred_uint8, gt_uint8], axis=-1)
        _save_video(os.path.join(sample_dir, "viz_grid.mp4"), side_by_side, fps=fps)

        with open(os.path.join(sample_dir, "rgb_metrics.json"), "w") as f:
            json.dump({**rgb_metrics, "per_frame_psnr": per_frame_psnr}, f)

        if save_stats and lpips_fn is not None:
            gen_psnr = per_frame_psnr[1:-1]
            per_frame_lpips = compute_per_frame_lpips(
                pred_tchw[1:-1], gt_tchw[1:-1], lpips_fn)
            per_frame_ssim = compute_per_frame_ssim(
                pred_tchw[1:-1], gt_tchw[1:-1])
            with open(os.path.join(sample_dir, "metrics.csv"), "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["avg_per_frame_psnr", "avg_per_frame_lpips",
                                   "avg_per_frame_ssim"])
                writer.writeheader()
                writer.writerow({
                    "avg_per_frame_psnr":  sum(gen_psnr)       / len(gen_psnr),
                    "avg_per_frame_lpips": sum(per_frame_lpips) / len(per_frame_lpips),
                    "avg_per_frame_ssim":  sum(per_frame_ssim)  / len(per_frame_ssim),
                })

        # ── running stats written after every sample ─────────────────────────
        with open(os.path.join(output_dir, "final_stats.json"), "w") as f:
            json.dump(stats, f)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wan2.1-FLF2V baseline eval — same RE10K test split as ray2rgb eval.")
    parser.add_argument("--ckpt_dir",   type=str,
                        default=str(WAN21_ROOT.parent / "Wan2.1-FLF2V-14B-720P"),
                        help="Path to Wan2.1-FLF2V-14B-720P checkpoint directory. "
                             "Downloaded automatically from HuggingFace if absent.")
    parser.add_argument("--output_dir", type=str,
                        default="ignore/outputs/flf2v_baseline_eval")
    parser.add_argument("--max_samples",   type=int,   default=50)
    parser.add_argument("--sample_steps",  type=int,   default=50,
                        help="Diffusion sampling steps (default 50 for FLF2V).")
    parser.add_argument("--guide_scale",   type=float, default=5.0)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--split",         type=str,   default="test")
    parser.add_argument("--save_stats",    action="store_true",
                        help="Save per-sample metrics.csv (PSNR/LPIPS/SSIM).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── download checkpoint if needed ────────────────────────────────────────
    args.ckpt_dir = ensure_ckpt(args.ckpt_dir)

    # ── load RE10K dataset (same config path as the ray2rgb eval) ────────────
    cfg = load_re10k_cfg(args.split)
    dataset = RE10KDataset(cfg.dataset, split=args.split)
    fps = cfg.dataset.fps
    print(f"INFO: Loaded {len(dataset)} videos from split={args.split}")
    print(f"INFO: Native resolution: {dataset[0]['height']}×{dataset[0]['width']}, "
          f"frames={FRAME_NUM}, fps={fps}")

    # ── load Wan2.1-FLF2V pipeline ───────────────────────────────────────────
    print(f"INFO: Loading WanFLF2V from {args.ckpt_dir}")
    cfg_wan = WAN_CONFIGS[TASK]
    wan_flf2v = wan.WanFLF2V(
        config=cfg_wan,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )

    print(f"INFO: Starting evaluation — max_samples={args.max_samples}, "
          f"steps={args.sample_steps}, guide_scale={args.guide_scale}, "
          f"max_area={MAX_AREA} ({MAX_AREA//480}×480)")

    final_stats = infer_and_eval_flf2v_baseline(
        wan_flf2v=wan_flf2v,
        dataset=dataset,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        fps=fps,
        save_stats=args.save_stats,
        sample_steps=args.sample_steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
    )

    print(f"\nINFO: Final stats over {final_stats['count']} samples:")
    print(f"  MSE  : {final_stats['mse']:.6f}")
    print(f"  PSNR : {final_stats['psnr']:.2f} dB")
    print(f"INFO: Outputs saved to {args.output_dir}")
