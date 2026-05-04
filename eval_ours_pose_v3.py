"""
V3 pose eval for OURS — reads ours_pose_depth/sample_*/{pred,gt}_cameras.npz
and reports Geo4D-style ATE/RPE_trans/RPE_rot via eval_common_v3.

Drops the V2 AUC machinery; drops aria correction (now baked into aria.py).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb
from tqdm import tqdm

from eval_common_v3 import POSE_KEYS, aggregate, eval_pose_sequence


def plot_camera_trajectory(pred_c2w: np.ndarray, gt_c2w: np.ndarray, out_path: str) -> None:
    pred_c = pred_c2w[:, :3, 3]
    gt_c   = gt_c2w[:, :3, 3]
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot(pred_c[:, 0], pred_c[:, 1], pred_c[:, 2], color="tab:blue", label="pred (raw)")
    ax.plot(gt_c[:, 0],   gt_c[:, 1],   gt_c[:, 2],   color="black", linestyle="--", label="GT")
    ax.legend()
    ax.set_title("Camera trajectories — OURS (raw, before Sim(3) align)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",  required=True, help="OURS pose_depth output dir")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--custom_run_name", default="ours_pose_eval_v3")
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
    for sample_dir in tqdm(sample_dirs, desc="OURS pose v3"):
        if args.max_samples is not None and count >= args.max_samples:
            break

        gt_path   = sample_dir / "gt_cameras.npz"
        pred_path = sample_dir / "pred_cameras.npz"
        if not gt_path.exists() or not pred_path.exists():
            print(f"Skipping {sample_dir.name}: missing cameras npz")
            continue

        gt_c2w   = np.load(gt_path)["extrinsics"].astype(np.float64)
        pred_c2w = np.load(pred_path)["extrinsics"].astype(np.float64)
        if len(pred_c2w) != len(gt_c2w):
            n = min(len(pred_c2w), len(gt_c2w))
            print(f"WARNING {sample_dir.name}: frame count mismatch "
                  f"pred={len(pred_c2w)} gt={len(gt_c2w)}; truncating to {n}")
            pred_c2w = pred_c2w[:n]; gt_c2w = gt_c2w[:n]

        sample_out = output_dir / sample_dir.name
        sample_out.mkdir(parents=True, exist_ok=True)

        try:
            metrics = eval_pose_sequence(pred_c2w, gt_c2w, seq=sample_dir.name,
                                         save_dir=sample_out)
        except Exception as e:
            print(f"Error on {sample_dir.name}: {e}")
            import traceback; traceback.print_exc()
            continue

        with open(sample_out / "pose_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        plot_camera_trajectory(pred_c2w, gt_c2w, str(sample_out / "camera_trajectory.png"))
        wandb.log({f"sample/{k}": metrics[k] for k in POSE_KEYS})
        per_seq.append(metrics)
        count += 1

    final = aggregate(per_seq, POSE_KEYS)
    print(f"\nFinal (n={final['n_samples']})  "
          + "  ".join(f"{k}={final[k]:.4f}" for k in POSE_KEYS))
    wandb.log({f"final/{k}": final[k] for k in POSE_KEYS})
    with open(output_dir / "final_stats.json", "w") as f:
        json.dump(final, f, indent=2)
    wandb.finish()


if __name__ == "__main__":
    main()
