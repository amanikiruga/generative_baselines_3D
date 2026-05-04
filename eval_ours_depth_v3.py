"""
V3 depth eval for OURS — reads ours_pose_depth/sample_*/{pred,gt}_depth_raw.npz
and reports Geo4D-style depth metrics via eval_common_v3 (per-video LAD2).

Both pred_depth_raw.npz and gt_depth_raw.npz contain:
    depth: (T, H, W) float32  — model-native units (often disparity-like)
    scale: () float32          — scalar to multiply by → metric depth (m)

We call depth_evaluation on the *metric depth* tensor (depth * scale) to match
Geo4D's convention. max_depth (clip) is taken from eval_common_v3.DEPTH_MAX_BY_DATASET.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import wandb
from tqdm import tqdm

from eval_common_v3 import (
    DEPTH_KEYS,
    aggregate,
    eval_depth_sequence,
    load_depth_metric_from_npz,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",  required=True, help="OURS pose_depth output dir")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dataset",    required=True,
                    help="dataset name, controls Geo4D max_depth clip")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--custom_run_name", default="ours_depth_eval_v3")
    args = ap.parse_args()

    wandb.init(project="video_world_model", name=args.custom_run_name)
    wandb.config.update(vars(args))

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(p for p in input_dir.iterdir()
                         if p.is_dir() and p.name.startswith("sample_"))

    per_seq: list[dict] = []
    count = 0
    for sample_dir in tqdm(sample_dirs, desc=f"OURS depth v3 ({args.dataset})"):
        if args.max_samples is not None and count >= args.max_samples:
            break
        pred_path = sample_dir / "pred_depth_raw.npz"
        gt_path   = sample_dir / "gt_depth_raw.npz"
        if not pred_path.exists() or not gt_path.exists():
            print(f"Skipping {sample_dir.name}: missing depth npz")
            continue

        # Recover metric depth from the [-1,1] normalised disparity format
        # written by our inference scripts (see eval_common_v3.disparity_norm_to_metric_depth).
        pred_metric = load_depth_metric_from_npz(pred_path)
        gt_metric   = load_depth_metric_from_npz(gt_path)

        if pred_metric.shape != gt_metric.shape:
            t = min(pred_metric.shape[0], gt_metric.shape[0])
            print(f"WARNING {sample_dir.name}: depth shape mismatch "
                  f"pred={pred_metric.shape} gt={gt_metric.shape}; truncating to T={t}")
            pred_metric = pred_metric[:t]; gt_metric = gt_metric[:t]

        try:
            metrics = eval_depth_sequence(
                pred_depth_THW=pred_metric,
                gt_depth_THW=gt_metric,
                dataset=args.dataset,
                normalize_unit_per_video=True,
            )
        except Exception as e:
            print(f"Error on {sample_dir.name}: {e}")
            import traceback; traceback.print_exc()
            continue

        sample_out = output_dir / sample_dir.name
        sample_out.mkdir(parents=True, exist_ok=True)
        with open(sample_out / "depth_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
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
