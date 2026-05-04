# Generative Baselines — V3 Eval Pipeline

## Spec

- [README_CLEAN_V3.md](README_CLEAN_V3.md)

## Vendored Geo4D Metric Code

Verbatim copies from upstream Geo4D, kept under [geo4d_eval/](geo4d_eval/):

| File | Origin | Notes |
| --- | --- | --- |
| [geo4d_eval/__init__.py](geo4d_eval/__init__.py) | — | package marker |
| [geo4d_eval/vo_eval.py](geo4d_eval/vo_eval.py) | `Geo4D/dust3r/utils/vo_eval.py` | ATE / RPE via `evo` |
| [geo4d_eval/depth_eval.py](geo4d_eval/depth_eval.py) | `Geo4D/dust3r/depth_eval.py` | per-video LAD2; dust3r-coupled wrappers stripped |

## Shared Eval Layer

- [eval_common_v3.py](eval_common_v3.py) — wraps `vo_eval.eval_metrics` and `depth_eval.depth_evaluation`. Exposes:
  - `eval_pose_sequence`, `eval_depth_sequence`, `aggregate`
  - `load_depth_metric_from_npz` — handles the `[-1, 1]` normalized-disparity → metric-meters conversion
  - `POSE_KEYS`, `DEPTH_KEYS`, `DEPTH_MAX_BY_DATASET`

## Per-Method Evaluators

All new in V3; V2 is untouched.

| Evaluator | Inputs | Outputs / Notes |
| --- | --- | --- |
| [eval_ours_pose_v3.py](eval_ours_pose_v3.py) | `ours_pose_depth/sample_*/{pred,gt}_cameras.npz` | `pose_metrics.json`, `final_stats.json` |
| [eval_ours_depth_v3.py](eval_ours_depth_v3.py) | `ours_pose_depth/sample_*/{pred,gt}_depth_raw.npz` | depth metrics |
| [geo4d/eval_geo4d_pose_v3.py](geo4d/eval_geo4d_pose_v3.py) | Geo4D inference plumbing kept | metric block swapped to `eval_pose_sequence` |
| [geo4d/eval_geo4d_depth_v3.py](geo4d/eval_geo4d_depth_v3.py) | — | same swap; Aria-style coordinate fix dropped |
| [RayDiffusion/eval_raydiffusion_pose_v3.py](RayDiffusion/eval_raydiffusion_pose_v3.py) | pytorch3d → c2w plumbing kept | AUC swapped to ATE / RPE; Aria correction dropped |
| [ChronoDepth/eval_chronodepth_depth_aria_v3.py](ChronoDepth/eval_chronodepth_depth_aria_v3.py) | ChronoDepth inference kept | metric block swapped to `eval_depth_sequence` |

## Runners

- [run_dataset_v3.sh](run_dataset_v3.sh) — per-dataset pipeline. Reads `$V2_OUT` to symlink V2 prediction caches:
  - Geo4D pose `geo4d_raw/`
  - per-sample `pred_depth_geo4d.npz`
  - full NVS dirs
- [run_all_v3.sh](run_all_v3.sh) — 3 waves over GPUs 0 + 1, then aggregators.

## Aggregators

- [collect_results_v3.py](collect_results_v3.py) → [RESULTS_V3.md](RESULTS_V3.md)
- [build_index_v3.py](build_index_v3.py) → [index_v3.html](index_v3.html)
