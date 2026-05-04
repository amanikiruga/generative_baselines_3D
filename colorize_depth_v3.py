"""
Post-hoc colourised-depth video producer for V3 baseline depth dirs.

The V3 depth evaluators dump only `*_depth_raw.npz` (a single (T,H,W) tensor)
plus `depth_metrics.json`. The index/HTML expects `pred_depth.mp4` and
`gt_depth.mp4` at sample dirs. This script renders missing mp4s by
turbo-colormapping each depth tensor and saving alongside the npz. Idempotent —
skips files that already exist.

Usage:
    python colorize_depth_v3.py [--root eval_outputs_v3] [--root eval_outputs_v3_n50 ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
import matplotlib.cm as cm

_TURBO = cm.get_cmap("turbo")


def _normalize(d_thw: np.ndarray) -> np.ndarray:
    """Per-video 2/98-percentile → [0,1] for colormap input."""
    d = d_thw.astype(np.float32)
    finite = np.isfinite(d) & (d > 0)
    if not finite.any():
        return np.zeros_like(d)
    lo = float(np.percentile(d[finite],  2.0))
    hi = float(np.percentile(d[finite], 98.0))
    if hi <= lo:
        return np.zeros_like(d)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0)


def _depth_to_disparity(d_thw: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Convert depth → disparity (1/d) for visualisation. NaNs/non-positive → 0."""
    d = d_thw.astype(np.float32)
    out = np.zeros_like(d)
    valid = np.isfinite(d) & (d > 0)
    out[valid] = 1.0 / np.clip(d[valid], eps, None)
    return out


def _colorize(d_thw: np.ndarray) -> np.ndarray:
    """(T,H,W) depth → (T,H,W,3) uint8 turbo, rendered in *disparity space* so
    the visualisation matches Ours/GT (which the inference script also writes
    out as disparity-colormapped). Closer = brighter, farther = darker."""
    disp = _depth_to_disparity(d_thw)
    n = _normalize(disp)
    return (_TURBO(n)[..., :3] * 255).astype(np.uint8)


def _write_mp4(path: Path, frames_uint8: np.ndarray, fps: int = 10,
               force: bool = False) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames_uint8, fps=fps)
    print(f"wrote {path}")


def colorize_dir(eval_root: Path, force: bool = False) -> None:
    if not eval_root.exists():
        print(f"skip {eval_root} — does not exist")
        return
    n_done = 0
    for ds_dir in sorted(eval_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        # Geo4D depth: pred_depth_geo4d.npz → pred_depth.mp4
        for sd in sorted((ds_dir / "geo4d_depth").glob("sample_*")):
            npz = sd / "pred_depth_geo4d.npz"
            out = sd / "pred_depth.mp4"
            if not npz.exists() or (out.exists() and not force):
                continue
            try:
                d = np.load(npz)["depth"]
                _write_mp4(out, _colorize(d), force=force)
                n_done += 1
            except Exception as e:
                print(f"err {npz}: {e}")
        # ChronoDepth: pred_depth_raw.npz → pred_depth.mp4
        for sd in sorted((ds_dir / "chronodepth_depth").glob("sample_*")):
            npz = sd / "pred_depth_raw.npz"
            out = sd / "pred_depth.mp4"
            if not npz.exists() or (out.exists() and not force):
                continue
            try:
                d = np.load(npz)["depth"]
                _write_mp4(out, _colorize(d), force=force)
                n_done += 1
            except Exception as e:
                print(f"err {npz}: {e}")
    print(f"[{eval_root.name}] colorized {n_done} {'updated' if force else 'new'} mp4s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=None,
                    help="eval-output dir (repeatable); defaults to "
                         "eval_outputs_v3 + eval_outputs_v3_n50")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing pred_depth.mp4 (e.g. after a "
                         "viz-convention change)")
    args = ap.parse_args()
    base = Path("/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines")
    if args.root:
        roots = [Path(r) if Path(r).is_absolute() else base / r for r in args.root]
    else:
        roots = [base / "eval_outputs_v3", base / "eval_outputs_v3_n50"]
    for r in roots:
        colorize_dir(r, force=args.force)


if __name__ == "__main__":
    main()
