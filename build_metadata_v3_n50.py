"""
V3-N50 reproducibility index.

Walks eval_outputs_v3_n50/{dataset}/{ours_pose_depth, ours_nvs}/sample_XXXXX/
and emits a single metadata JSON listing every per-sample artifact path the
baselines (Geo4D, RayDiffusion, ChronoDepth, SEVA, Wan-FLF) consume:

  - gt_rgb.mp4         (ground-truth video)
  - gt_cameras.npz     ({extrinsics: T x 4 x 4 c2w,  intrinsics: T x 3 x 3})
  - gt_depth_raw.npz   ({depth: T x H x W in [-1,1] norm-disparity, scale: float})
  - prompt.txt         (text caption — re10k / dl3dv / agibot_world only)
  - pred_*             (our predictions; useful for test-time-search anchors)

Output:
  generative_baselines/metadata_v3_n50.json     (full index)
  generative_baselines/metadata_v3_n50/<dataset>.json   (per-dataset slice)

Schema:
  {
    "version": "v3_n50",
    "ckpt": "outputs/2026-05-03/20-03-01/last.ckpt",
    "datasets": {
      "<dataset>": {
        "n_samples_pose_depth": <int>,
        "n_samples_nvs": <int>,
        "samples": [
          {
            "sample_id": "sample_00000",
            "n_frames": <int or null>,
            "prompt": <str or null>,
            "pose_depth": {
              "gt_rgb_mp4":       "...",
              "gt_cameras_npz":   "...",   # contains 'extrinsics' (c2w) + 'intrinsics'
              "gt_depth_raw_npz": "...",   # contains 'depth' + 'scale'
              "gt_ray_d_mp4":     "...",
              "gt_ray_m_mp4":     "...",
              "gt_depth_mp4":     "...",
              "prompt_txt":       "..." or null,
              "pred_cameras_npz": "...",
              "pred_depth_raw_npz": "...",
              "pred_rgb_mp4":     "...",   (if rendered)
            },
            "nvs": {
              "gt_rgb_mp4":       "...",
              "gt_cameras_npz":   "...",
              "gt_ray_d_mp4":     "...",
              "gt_ray_m_mp4":     "...",
              "prompt_txt":       "..." or null,
              "pred_rgb_mp4":     "...",
              "pred_depth_mp4":   "...",
            }
          },
          ...
        ]
      },
      ...
    }
  }
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path("/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines")
EVAL = ROOT / "eval_outputs_v3_n50"

# Files we expect inside each sample dir, with their schema-stable keys.
POSE_DEPTH_FILES = {
    "gt_rgb_mp4":         "gt_rgb.mp4",
    "gt_cameras_npz":     "gt_cameras.npz",
    "gt_depth_raw_npz":   "gt_depth_raw.npz",
    "gt_ray_d_mp4":       "gt_ray_d.mp4",
    "gt_ray_m_mp4":       "gt_ray_m.mp4",
    "gt_depth_mp4":       "gt_depth.mp4",
    "prompt_txt":         "prompt.txt",
    "pred_cameras_npz":   "pred_cameras.npz",
    "pred_depth_raw_npz": "pred_depth_raw.npz",
    "pred_rgb_mp4":       "pred_rgb.mp4",
    "pred_ray_d_mp4":     "pred_ray_d.mp4",
    "pred_ray_m_mp4":     "pred_ray_m.mp4",
    "pred_depth_mp4":     "pred_depth.mp4",
    "viz_grid_mp4":       "viz_grid.mp4",
}
NVS_FILES = {
    "gt_rgb_mp4":     "gt_rgb.mp4",
    "gt_cameras_npz": "gt_cameras.npz",
    "gt_ray_d_mp4":   "gt_ray_d.mp4",
    "gt_ray_m_mp4":   "gt_ray_m.mp4",
    "gt_depth_mp4":   "gt_depth.mp4",
    "prompt_txt":     "prompt.txt",
    "pred_rgb_mp4":   "pred_rgb.mp4",
    "pred_depth_mp4": "pred_depth.mp4",
    "rgb_metrics_json": "rgb_metrics.json",
}


def _scan_branch(branch_dir: Path, expected: dict[str, str]) -> dict | None:
    """For one branch (ours_pose_depth or ours_nvs) return per-sample dict."""
    if not branch_dir.exists():
        return None
    out = {}
    for sd in sorted(branch_dir.iterdir()):
        if not (sd.is_dir() and sd.name.startswith("sample_")):
            continue
        files = {}
        for key, fname in expected.items():
            p = sd / fname
            files[key] = str(p.resolve()) if p.exists() else None
        out[sd.name] = files
    return out


def _read_prompt(branch_files: dict, sample_id: str) -> str | None:
    p = (branch_files or {}).get(sample_id, {}).get("prompt_txt")
    if not p:
        return None
    try:
        return Path(p).read_text().strip() or None
    except Exception:
        return None


def _n_frames_from_npz(branch_files: dict, sample_id: str) -> int | None:
    """Cheap n_frames lookup from gt_cameras.npz extrinsics shape."""
    if not branch_files:
        return None
    p = branch_files.get(sample_id, {}).get("gt_cameras_npz")
    if not p or not os.path.exists(p):
        return None
    try:
        import numpy as np
        return int(np.load(p)["extrinsics"].shape[0])
    except Exception:
        return None


def build_dataset(dataset: str) -> dict | None:
    base = EVAL / dataset
    if not base.exists():
        return None
    pose_depth = _scan_branch(base / "ours_pose_depth", POSE_DEPTH_FILES) or {}
    nvs        = _scan_branch(base / "ours_nvs",        NVS_FILES)        or {}
    sample_ids = sorted(set(pose_depth.keys()) | set(nvs.keys()))

    samples = []
    for sid in sample_ids:
        prompt = _read_prompt(pose_depth, sid) or _read_prompt(nvs, sid)
        nf = _n_frames_from_npz(pose_depth, sid) or _n_frames_from_npz(nvs, sid)
        samples.append({
            "sample_id":  sid,
            "n_frames":   nf,
            "prompt":     prompt,
            "pose_depth": pose_depth.get(sid),
            "nvs":        nvs.get(sid),
        })

    return {
        "n_samples_pose_depth": len(pose_depth),
        "n_samples_nvs":        len(nvs),
        "samples":              samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "metadata_v3_n50.json"))
    ap.add_argument("--per_dataset_dir",
                    default=str(ROOT / "metadata_v3_n50"),
                    help="Also write a per-dataset slice for easier loading.")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Restrict to these dataset names. Default: all.")
    args = ap.parse_args()

    all_ds = sorted([p.name for p in EVAL.iterdir() if p.is_dir()])
    if args.datasets:
        all_ds = [d for d in all_ds if d in set(args.datasets)]

    index = {
        "version":  "v3_n50",
        "eval_root": str(EVAL.resolve()),
        "ckpt":     "outputs/2026-05-03/20-03-01/last.ckpt",
        "ours_inference_seed": 42,
        "note": ("Reproducibility index — each sample dir contains the GT "
                 "artifacts every baseline consumes (gt_rgb.mp4, "
                 "gt_cameras.npz [c2w extrinsics + intrinsics], "
                 "gt_depth_raw.npz [depth+scale], prompt.txt). "
                 "Source-dataset record paths are NOT yet captured at "
                 "inference time; the *_eval files in this dir are the "
                 "canonical inputs for any sparse-eval or test-time-search."),
        "datasets": {},
    }

    per_ds_dir = Path(args.per_dataset_dir)
    per_ds_dir.mkdir(parents=True, exist_ok=True)

    for ds in all_ds:
        ds_idx = build_dataset(ds)
        if ds_idx is None:
            continue
        index["datasets"][ds] = ds_idx
        # also write per-dataset slice
        with open(per_ds_dir / f"{ds}.json", "w") as f:
            json.dump({**{k: v for k, v in index.items() if k != "datasets"},
                       "dataset": ds, **ds_idx}, f, indent=2)

    with open(args.out, "w") as f:
        json.dump(index, f, indent=2)

    n_total = sum(len(d["samples"]) for d in index["datasets"].values())
    print(f"wrote {args.out}  ({len(index['datasets'])} datasets, {n_total} samples)")
    print(f"per-dataset slices in {per_ds_dir}/")


if __name__ == "__main__":
    main()
