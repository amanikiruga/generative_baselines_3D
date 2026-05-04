"""
V3 depth-space (re-)evaluation — offline post-hoc pass over cached predictions.

Mirror of eval_depth_disp_offline_v3.py but for the *depth-space* track. Reads
the same cached `*_depth_raw.npz` / `pred_depth_geo4d.npz` files but applies
`eval_depth_sequence(..., normalize_unit_per_video=True)` so LAD2 fits on a
GT-anchored, bounded depth range. Overwrites the existing `*_depth/` and
`*_depth_eval/` final stats so the next index/RESULTS rebuild reflects the
corrected numbers.

No GPU, no model inference — only metric computation. Idempotent.

Usage:
  python eval_depth_offline_v3.py
  python eval_depth_offline_v3.py --root eval_outputs_v3
  python eval_depth_offline_v3.py --root eval_outputs_v3 --dataset dl3dv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_GB = Path("/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines")
sys.path.insert(0, str(_GB))
from eval_common_v3 import (  # noqa: E402
    DEPTH_KEYS,
    aggregate,
    eval_depth_sequence,
    load_depth_metric_from_npz,
)

DEPTH_DATASETS = ("dl3dv", "scannetpp", "vkitti2", "aria", "scenenet_depth")


def _pred_metric(method: str, sample_dir: Path) -> np.ndarray | None:
    """Predicted depth in metric meters (or method-native depth-flavored space).
    Inversion / unit alignment is left to LAD2 inside eval_depth_sequence."""
    if method == "ours":
        p = sample_dir / "pred_depth_raw.npz"
        return load_depth_metric_from_npz(p) if p.exists() else None
    if method == "geo4d":
        p = sample_dir / "pred_depth_geo4d.npz"
        return np.load(p)["depth"].astype(np.float32) if p.exists() else None
    if method == "chronodepth":
        p = sample_dir / "pred_depth_raw.npz"
        return np.load(p)["depth"].astype(np.float32) if p.exists() else None
    raise ValueError(method)


def _resize_pred_to_gt(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape == gt.shape:
        return pred
    t = torch.from_numpy(pred).unsqueeze(1).float()
    t = F.interpolate(t, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    return t.squeeze(1).numpy()


def _eval_dataset(eval_root: Path, dataset: str, method: str, src_subdir: str,
                  out_subdir: str) -> int:
    src = eval_root / dataset / src_subdir
    if not src.exists():
        print(f"[{eval_root.name}/{dataset}/{method}] no {src_subdir}, skip")
        return 0
    out = eval_root / dataset / out_subdir
    out.mkdir(parents=True, exist_ok=True)

    per_seq: list[dict] = []
    sample_dirs = sorted(p for p in src.iterdir()
                         if p.is_dir() and p.name.startswith("sample_"))

    for sd in tqdm(sample_dirs, desc=f"  {method:11s} {dataset}", leave=False):
        gt_path = eval_root / dataset / "ours_pose_depth" / sd.name / "gt_depth_raw.npz"
        if not gt_path.exists():
            continue
        try:
            pred = _pred_metric(method, sd)
            if pred is None:
                continue
            gt = load_depth_metric_from_npz(gt_path)
            pred = _resize_pred_to_gt(pred, gt)
            metrics = eval_depth_sequence(
                pred_depth_THW=pred,
                gt_depth_THW=gt,
                dataset=dataset,
                normalize_unit_per_video=True,
            )
        except Exception as e:
            print(f"  err {sd.name}: {type(e).__name__}: {e}")
            continue

        sample_out = out / sd.name
        sample_out.mkdir(parents=True, exist_ok=True)
        with open(sample_out / "depth_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        per_seq.append(metrics)

    final = aggregate(per_seq, DEPTH_KEYS)
    with open(out / "final_stats.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"[{eval_root.name}/{dataset}/{method}] depth eval: n={final['n_samples']}  "
          + " ".join(f"{k}={final[k]:.3f}" for k in DEPTH_KEYS))
    return final["n_samples"]


# Map method → (src cache subdir, output dir for depth-space metrics).
METHOD_DIR = {
    "ours":        ("ours_pose_depth",   "ours_depth_eval"),
    "geo4d":       ("geo4d_depth",       "geo4d_depth"),       # in-place overwrite
    "chronodepth": ("chronodepth_depth", "chronodepth_depth"), # in-place overwrite
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=None)
    ap.add_argument("--dataset", action="append", default=None)
    ap.add_argument("--method", action="append", default=None,
                    choices=list(METHOD_DIR.keys()))
    args = ap.parse_args()

    roots = ([Path(r) if Path(r).is_absolute() else _GB / r for r in args.root]
             if args.root else [_GB / "eval_outputs_v3", _GB / "eval_outputs_v3_n50"])
    datasets = args.dataset or list(DEPTH_DATASETS)
    methods = args.method or list(METHOD_DIR.keys())

    for root in roots:
        if not root.exists():
            print(f"skip {root} — does not exist")
            continue
        for ds in datasets:
            if not (root / ds).exists():
                continue
            for m in methods:
                src, out = METHOD_DIR[m]
                _eval_dataset(root, ds, m, src, out)


if __name__ == "__main__":
    main()
