"""
V3 disparity-space depth eval — offline post-hoc pass over existing predictions.

For each (dataset, method) combo with cached depth predictions, recompute the
Geo4D depth metric set with LAD2 alignment in *disparity* (1/depth) space
instead of metric-depth space. Writes outputs to a parallel `*_depth_disp/`
sister dir so the index can show "Method (disp)" rows alongside the depth ones.

No model inference — only reads cached `*_depth_raw.npz` / `pred_depth_geo4d.npz`,
loads GT from `ours_pose_depth/sample_*/gt_depth_raw.npz`, runs
`eval_depth_sequence(..., eval_in_disparity=True)`.

Methods handled:
  ours        — pred at ours_pose_depth/sample_*/pred_depth_raw.npz   (norm-disparity → depth via load_depth_metric_from_npz, then inverted to disparity inside eval_depth_sequence).
  geo4d       — pred at geo4d_depth/sample_*/pred_depth_geo4d.npz     (metric depth, inverted to disparity inside).
  chronodepth — pred at chronodepth_depth/sample_*/pred_depth_raw.npz ([0,1] relative depth, inverted to disparity inside).

Output dirs (mirrors `*_depth_eval` schema):
  ours_depth_disp_eval/
  geo4d_depth_disp/
  chronodepth_depth_disp/

Usage:
  python eval_depth_disp_offline_v3.py --root eval_outputs_v3_n50
  python eval_depth_disp_offline_v3.py --root eval_outputs_v3_n50 --dataset dl3dv
  python eval_depth_disp_offline_v3.py            # both v3 and v3_n50
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
    DEPTH_MAX_BY_DATASET,
    aggregate,
    eval_depth_sequence,
    load_depth_metric_from_npz,
)

DEPTH_DATASETS = ("dl3dv", "scannetpp", "vkitti2", "aria", "scenenet_depth")


def _native_disparity(method: str, sample_dir: Path) -> np.ndarray | None:
    """
    Return predicted depth-or-disparity ALREADY IN DISPARITY SPACE — no
    disparity→depth→disparity round-trip. Inputs by method:
      ours        — pred_depth_raw.npz["depth"] is [-1,1] normalised disparity.
                    Map to [0,1] disparity directly: (d + 1) / 2.
      geo4d       — pred_depth_geo4d.npz["depth"] is positive metric depth.
                    Convert depth → disparity: 1 / max(d, eps).
      chronodepth — pred_depth_raw.npz["depth"] is [0,1] relative depth (Pearson
                    +0.83 with gt depth — depth-flavored). Convert to disparity-like
                    via 1 / max(d, eps).
    Caller passes the result + a *native disparity* GT to eval_depth_sequence
    with eval_in_disparity=False (already disparity).
    """
    eps = 1e-3
    if method in OURS_LIKE:
        p = sample_dir / "pred_depth_raw.npz"
        if not p.exists(): return None
        d_norm = np.load(p)["depth"].astype(np.float32)
        return np.clip((d_norm + 1.0) * 0.5, eps, 1.0 - eps)
    if method == "geo4d":
        p = sample_dir / "pred_depth_geo4d.npz"
        if not p.exists(): return None
        d = np.load(p)["depth"].astype(np.float32)
        return 1.0 / np.clip(d, eps, None)
    if method == "chronodepth":
        p = sample_dir / "pred_depth_raw.npz"
        if not p.exists(): return None
        d = np.load(p)["depth"].astype(np.float32)
        return 1.0 / np.clip(d, eps, None)
    raise ValueError(method)


def _native_gt_disparity(gt_npz_path: Path) -> np.ndarray:
    """GT npz stores [-1,1] normalised disparity in 'depth' field. Map straight
    to [0,1] disparity (no roundtrip through metric depth)."""
    eps = 1e-3
    d_norm = np.load(gt_npz_path)["depth"].astype(np.float32)
    return np.clip((d_norm + 1.0) * 0.5, eps, 1.0 - eps)


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
        # GT lives in any of the ours pose_depth dirs (V3 or V4 layout).
        gt_path = None
        for cand in ("ours_pose_depth", "ours_final_pose_depth", "ours_nvs_pose_depth"):
            p = eval_root / dataset / cand / sd.name / "gt_depth_raw.npz"
            if p.exists():
                gt_path = p
                break
        if gt_path is None:
            continue
        try:
            pred_disp = _native_disparity(method, sd)
            if pred_disp is None:
                continue
            gt_disp = _native_gt_disparity(gt_path)
            pred_disp = _resize_pred_to_gt(pred_disp, gt_disp)
            # Both inputs ARE disparity (no round-trip). Pass through with
            # eval_in_disparity=False; LAD2 fits s*pred_disp + t = gt_disp,
            # metrics computed in disparity units.
            metrics = eval_depth_sequence(
                pred_depth_THW=pred_disp,
                gt_depth_THW=gt_disp,
                dataset=dataset,
                eval_in_disparity=False,
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
    print(f"[{eval_root.name}/{dataset}/{method}] disp eval: n={final['n_samples']}  "
          + " ".join(f"{k}={final[k]:.3f}" for k in DEPTH_KEYS))
    return final["n_samples"]


# V3 used "ours_pose_depth"; V4 splits into "ours_final_pose_depth" and
# "ours_nvs_pose_depth". Both layouts are listed; the script auto-skips ones
# whose source dir doesn't exist for a given (root, dataset).
METHOD_DIR = {
    "ours":          ("ours_pose_depth",          "ours_depth_disp_eval"),
    "ours_final":    ("ours_final_pose_depth",    "ours_final_depth_disp_eval"),
    "ours_nvs":      ("ours_nvs_pose_depth",      "ours_nvs_depth_disp_eval"),
    "geo4d":         ("geo4d_depth",              "geo4d_depth_disp"),
    "chronodepth":   ("chronodepth_depth",        "chronodepth_depth_disp"),
}
# Methods whose pred uses the same disparity decoding as the V3 "ours" branch.
OURS_LIKE = {"ours", "ours_final", "ours_nvs"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=None,
                    help="eval-output root (repeatable); defaults to "
                         "eval_outputs_v3 + eval_outputs_v3_n50")
    ap.add_argument("--dataset", action="append", default=None,
                    help="restrict to dataset (repeatable); defaults to all "
                         "depth datasets present")
    ap.add_argument("--method", action="append", default=None,
                    choices=list(METHOD_DIR.keys()),
                    help="restrict to method (repeatable); defaults to all 3")
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
