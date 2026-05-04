"""
V3 metric wrappers, shared across all baselines + ours.

Pose:  Geo4D `dust3r/utils/vo_eval.py:eval_metrics` (ATE/RPE_trans/RPE_rot via evo).
Depth: Geo4D `dust3r/depth_eval.py:depth_evaluation` with align_with_lad2=True
       (per-video global LAD2 scale-shift fit).

Both metric implementations are vendored verbatim under `geo4d_eval/`.
This module exposes a tiny convenience layer so each per-method evaluator
just has to provide (pred_c2w, gt_c2w) for pose and (pred_T_H_W, gt_T_H_W)
for depth.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Vendored eval code (do not modify).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from geo4d_eval import vo_eval as _vo_eval  # noqa: E402
from geo4d_eval import depth_eval as _depth_eval  # noqa: E402

from evo.core.trajectory import PosePath3D, PoseTrajectory3D  # noqa: E402


# ── Pose ─────────────────────────────────────────────────────────────────────

def _c2w_to_traj(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a stack of c2w 4×4 matrices to TUM-format `(traj_tum, timestamps)`,
    matching `vo_eval.load_replica_traj` output convention:
      traj_tum: (N, 7) with columns (x, y, z, qw, qx, qy, qz)
      timestamps: (N,) frame indices as floats
    """
    c2w = np.asarray(c2w, dtype=np.float64)
    N = len(c2w)
    poses_se3 = [c2w[i] for i in range(N)]
    pose_path = PosePath3D(poses_se3=poses_se3)
    timestamps = np.arange(N, dtype=np.float64)
    traj = PoseTrajectory3D(poses_se3=pose_path.poses_se3, timestamps=timestamps)
    xyz = traj.positions_xyz
    quat_wxyz = traj.orientations_quat_wxyz
    traj_tum = np.column_stack((xyz, quat_wxyz))
    return traj_tum, timestamps


def eval_pose_sequence(
    pred_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    seq: str,
    save_dir: str | os.PathLike,
) -> dict:
    """
    Compute Geo4D-style pose metrics for one sequence and write the
    `<seq>_eval_metric.txt` file expected by `vo_eval.process_directory`.

    Returns a dict {"ate", "rpe_trans", "rpe_rot", "n_frames"}.
    """
    pred_c2w = np.asarray(pred_c2w, dtype=np.float64)
    gt_c2w   = np.asarray(gt_c2w,   dtype=np.float64)
    assert pred_c2w.shape == gt_c2w.shape, (pred_c2w.shape, gt_c2w.shape)

    pred_traj = _c2w_to_traj(pred_c2w)
    gt_traj   = _c2w_to_traj(gt_c2w)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metric_file = save_dir / f"{seq}_eval_metric.txt"

    ate, rpe_trans, rpe_rot = _vo_eval.eval_metrics(
        pred_traj=pred_traj,
        gt_traj=gt_traj,
        seq=seq,
        filename=str(metric_file),
    )

    return {
        "ate":       float(ate),
        "rpe_trans": float(rpe_trans),
        "rpe_rot":   float(rpe_rot),
        "n_frames":  int(len(pred_c2w)),
    }


# ── Depth ────────────────────────────────────────────────────────────────────

# Geo4D's per-dataset max_depth defaults (from infer_geo4d.py:538-540 + the
# Sintel/Bonn/KITTI/ScanNet++ presets in their evaluation_script.md).
DEPTH_MAX_BY_DATASET: dict[str, float | None] = {
    "vkitti2":        70.0,
    "scannetpp":      70.0,
    "aria":           70.0,
    "scenenet_depth": None,
    "dl3dv":          None,
}


def eval_depth_sequence(
    pred_depth_THW: np.ndarray | torch.Tensor,
    gt_depth_THW:   np.ndarray | torch.Tensor,
    dataset:        str | None = None,
    custom_mask:    np.ndarray | torch.Tensor | None = None,
    use_gpu:        bool = True,
) -> dict:
    """
    Per-video global LAD2 scale-shift alignment, Geo4D-style.

    Inputs are (T, H, W) depth tensors in metric-or-disparity space (whatever
    the model produced / the GT specifies). `depth_evaluation` flattens across
    all frames and fits a single (s, t) — i.e. one scale-shift per video, not
    per frame.
    """
    pd = pred_depth_THW
    gd = gt_depth_THW
    if isinstance(pd, np.ndarray): pd = torch.from_numpy(pd)
    if isinstance(gd, np.ndarray): gd = torch.from_numpy(gd)
    pd = pd.float()
    gd = gd.float()

    max_depth = DEPTH_MAX_BY_DATASET.get(dataset, None)
    # Geo4D uses lr=1e-2/iters=5000 when masking with `custom_mask` (their
    # outdoor branch); lr=1e-4/iters=1000 otherwise. Mirror that.
    lr        = 1e-2 if custom_mask is not None else 1e-4
    max_iters = 5000 if custom_mask is not None else 1000
    post_clip_max = max_depth

    results, _err_map, _pred_full, _gt_full = _depth_eval.depth_evaluation(
        predicted_depth_original=pd,
        ground_truth_depth_original=gd,
        max_depth=max_depth,
        custom_mask=custom_mask,
        align_with_lad2=True,
        lr=lr,
        max_iters=max_iters,
        use_gpu=use_gpu,
        post_clip_max=post_clip_max,
    )
    # Stringify the seven Geo4D metric keys to JSON-friendly snake_case.
    return {
        "abs_rel":   float(results["Abs Rel"]),
        "sq_rel":    float(results["Sq Rel"]),
        "rmse":      float(results["RMSE"]),
        "log_rmse":  float(results["Log RMSE"]),
        "delta_1":   float(results["δ < 1.25"]),
        "delta_2":   float(results["δ < 1.25^2"]),
        "delta_3":   float(results["δ < 1.25^3"]),
        "valid_pixels": int(results["valid_pixels"]),
    }


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(per_seq: list[dict], keys: list[str]) -> dict:
    """Arithmetic mean over sequences (matches Geo4D `calculate_averages`)."""
    if not per_seq:
        return {k: 0.0 for k in keys} | {"n_samples": 0}
    out = {}
    for k in keys:
        vals = [float(d[k]) for d in per_seq if k in d]
        out[k] = sum(vals) / max(len(vals), 1)
    out["n_samples"] = len(per_seq)
    return out


POSE_KEYS  = ["ate", "rpe_trans", "rpe_rot"]
DEPTH_KEYS = ["abs_rel", "sq_rel", "rmse", "log_rmse", "delta_1", "delta_2", "delta_3"]


# ── Depth-format conversion ──────────────────────────────────────────────────

def disparity_norm_to_metric_depth(
    depth_norm: np.ndarray,
    scale: float,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Convert the ``[-1, 1]`` normalised disparity stored by our inference
    pipeline (`*_depth_raw.npz["depth"]` / `["scale"]`) into metric depth
    in meters.

    Pipeline (matches scripts/eval_rgb_to_ray_depth_depth.py:53-71):
        disp_01  = (depth_norm + 1) / 2
        d_scaled = 1 / clip(disp_01, eps, 1 - eps) - 1
        d_metric = d_scaled * scale

    Pixels at the far-clip end of the disparity range (disp_01 ≤ eps) become
    the maximum representable scaled depth and are clamped — they remain
    valid for Geo4D's max_depth-clip mask.
    """
    disp_01 = (np.asarray(depth_norm, dtype=np.float32) + 1.0) * 0.5
    disp_01 = np.clip(disp_01, eps, 1.0 - eps)
    d_scaled = 1.0 / disp_01 - 1.0
    return (d_scaled * float(scale)).astype(np.float32)


def load_depth_metric_from_npz(npz_path) -> np.ndarray:
    """
    Read `*_depth_raw.npz` (with keys 'depth' [-1,1] and 'scale' float) and
    return a (T, H, W) float32 metric-depth tensor.
    """
    npz = np.load(npz_path)
    return disparity_norm_to_metric_depth(
        depth_norm=npz["depth"].astype(np.float32),
        scale=float(npz["scale"]),
    )
