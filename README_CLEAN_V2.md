# Generative-baselines evaluation — V2

Single-checkpoint evaluation of our model (`OURS_FINAL`) vs published baselines, across
10 datasets covering camera-pose, depth, and novel-view-synthesis (NVS).

This is the V2 successor to `README_CLEAN.md`. **Key differences:**

| | V1 | V2 |
|---|---|---|
| Our checkpoints | `OURS_CKPT` (1.3B mixture) + `NVS_ONLY_CKPT` | **single `OURS_FINAL`** (drop nvs_only branch) |
| Text conditioning | not propagated | `--load_prompt_embed` for re10k / dl3dv / agibot_world |
| Baselines run | 7 (ours×2 + 5) | **5** (gen3c removed; broken — no usable ckpt) |
| Baseline order | wan_flf in middle | **wan_flf last** (slowest, finishes after the rest land) |
| Hardware | 4× H200 (143 GB ea) | 4× H100 (80 GB ea) |
| Output dir | `eval_outputs → /n/netscratch/.../generative_baselines_eval` | `eval_outputs_v2 → /n/netscratch/.../generative_baselines_eval_v2` |
| index.html | per-task table + carousel | **+ per-scene prompt + per-scene metrics in each carousel slide** |

V1 outputs are preserved at `eval_outputs/` — V2 writes to a fresh tree at `eval_outputs_v2/`.

---

## §0 Quick start (autonomous, fire-and-forget)

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
nohup ./run_all_v2.sh > eval_logs_v2/_run_all.log 2>&1 &
```

That kicks off 8 datasets in wave 1 (2 per H100), then 2 in wave 2, then aggregates
into `RESULTS_V2.md` + `index_v2.html`.

To watch live:

```bash
tail -f eval_logs_v2/_run_all.log
ls eval_logs_v2/         # one log per dataset
nvidia-smi               # 4 H100s, all four pinned
```

---

## §1 Checkpoint paths

```bash
export OURS_FINAL=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-30/23-29-19/checkpoints/last_archive.ckpt
export GEO4D_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
export DFOT_RE10K_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/diffusion-forcing-transformer/pretrained/DFoT_RE10K.ckpt
export RAYDIFF_DIR=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion
export OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs_v2
```

`run_all_v2.sh` exports all of the above before launching pipelines.

`OURS_FINAL` supports text conditioning end-to-end on three datasets:
`re10k`, `dl3dv`, `agibot_world`. Their dataset YAMLs all have `load_prompt_embed: true`
and the loader emits `prompts` + `prompt_embeds` for every batch item.

---

## §2 Datasets and task coverage

| dataset | pose | depth | nvs | text-conditioned |
|---|---|---|---|---|
| re10k           | ✓ | ·  | ✓ | **yes** |
| dl3dv           | ✓ | ✓  | ✓ | **yes** |
| dl3dv_test      | ✓ | ·  | ✓ | · |
| tanksandtemples | ✓ | ·  | ✓ | · |
| scannetpp       | ✓ | ✓  | ✓ | · |
| aria            | ✓ | ✓  | ✓ | · |
| vkitti2         | ✓ | ✓  | ✓ | · |
| scenenet_depth  | ·  | ✓  | ·  | · |
| spatialvid_nvs  | ✓ | ·  | ✓ | · |
| agibot_world    | ✓ | ·  | ✓ | **yes** |

Test-percentage overrides applied automatically by the inference scripts:
- `aria`, `tanksandtemples` → `dataset.test_percentage=1.0` (small or heavily filtered)
- `vkitti2`, `scenenet_depth` → `dataset.test_percentage=0.1`
- otherwise the dataset's default applies.

---

## §3 Per-dataset pipeline (`run_dataset_v2.sh`)

For one dataset on one GPU:

```bash
./run_dataset_v2.sh <dataset> <gpu_id>
```

Order of operations (serial within one dataset):
1. **Our inference, RGB→ray+depth** (camera pose & depth) — single ckpt.
2. **Our inference, ray→RGB+depth** (NVS) — single ckpt, only on datasets with `has_nvs=1`.
3. **Our evals**: `eval_ray_mot_pose.py`, `eval_rgb_to_ray_depth_depth.py` (on depth datasets), `recompute_metrics_offline.py` (NVS).
4. **Pose baselines**: GEO4D, RayDiffusion.
5. **Depth baselines** (where applicable): GEO4D-depth, ChronoDepth.
6. **NVS baselines**: DFoT (re10k+aria only), SEVA, **Wan 2.1 FLF (last)**.

`--load_prompt_embed` is added automatically for `re10k / dl3dv / agibot_world`.

`gen3c` is **not** invoked anywhere.

---

## §4 Parallel launch (`run_all_v2.sh`)

```
GPU 0: re10k  + aria
GPU 1: dl3dv  + vkitti2
GPU 2: dl3dv_test + scannetpp
GPU 3: tanksandtemples + spatialvid_nvs
```
…wait for wave 1 → then wave 2 (2 datasets × 2 GPUs):
```
GPU 0: scenenet_depth
GPU 1: agibot_world
```

Each pipeline runs serially within itself, so each H100 sees ≤ 2 concurrent
inference jobs at a time → comfortable on 80 GB.

---

## §5 Aggregation

```bash
mamba run -n test2 python collect_results_v2.py   # writes RESULTS_V2.md
mamba run -n test2 python build_index_v2.py       # writes index_v2.html
```

Both are also invoked at the end of `run_all_v2.sh`.

`build_index_v2.py` reads:
- per-task aggregate from `<eval_dir>/final_stats.json` or `per_sample_metrics.csv`
- **per-sample** metric: from `<eval_dir>/sample_<i>/pose_metrics.json` (pose),
  `<eval_dir>/sample_<i>/depth_metrics.json` (depth),
  or per-row in `<eval_dir>/per_sample_metrics.csv` (nvs)
- **per-sample prompt**: from `ours_pose_depth/sample_<i>/prompt.txt`
  (saved automatically by both inference scripts when the dataset emits `prompts`).

Each carousel slide now shows:
- The text prompt (only on the 3 text-supporting datasets)
- Per-method media (video/png)
- Below each method's media: that method's per-scene metric

---

## §6 Outputs

```
eval_outputs_v2/                      → /n/netscratch/.../generative_baselines_eval_v2
├── re10k/
│   ├── ours_pose_depth/sample_*/     # videos, npz, prompt.txt, camera_trajectory.png
│   ├── ours_nvs/sample_*/
│   ├── ours_pose_eval/sample_*/      # pose_metrics.json + final_stats.json
│   ├── ours_nvs_eval/per_sample_metrics.csv
│   ├── geo4d_pose/, raydiffusion_pose/
│   ├── dfot_nvs/, seva_nvs/, wan_flf_nvs/
│   └── ...
├── dl3dv/ ... (same pattern + ours_depth_eval, geo4d_depth, chronodepth_depth)
└── ...

eval_logs_v2/<dataset>.log            # full stdout/stderr per dataset
RESULTS_V2.md                         # per-task tables (markdown)
index_v2.html                         # browsable per-scene comparison
```

---

## §7 Known gotchas (carried over + new)

- **gen3c — broken:** persistent `ImportError: apply_rotary_pos_emb`. Removed from V2.
- **aria** post-rotation filter drops 14 k → 20 sequences; `test_percentage=1.0` applied to keep n=10.
- **Tanks & Temples** has only 5 valid scenes; 1.0 test-percentage applied so n is ≥ 1.
- **agibot_world** uses static head-cam (extrinsics identity, ray moments zero); blocks task 446.
- **scenenet_depth** loader stub returns `scale=1.0` although eval marks `metric_depth_m` — depth numbers are valid as scaled-disparity.
- **vkitti2** outdoor depth is genuinely hard for our model — disparity saturates near zero. d1 reads OK but abs_rel is poor; not an eval bug.
- **depth eval visualization** uses `1/(d+1)` on disparity space; eval-script divides metric depth by `scale` first to recover disparity before colormapping.

---

## §8 Adding a new dataset (9 steps)

1. Create `world_model_4d/video_world_model_new/datasets/<name>.py` (load_prompt_embed support optional).
2. Create `world_model_4d/video_world_model_new/configurations/dataset/<name>.yaml`. If the dataset has cached prompt embeds, set `load_prompt_embed: true`.
3. Register `<name>` in the `_DATASETS` dispatch dict of both inference scripts:
   - `scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py`
   - `scripts/inference_single_gpu_ray_to_rgb_depth_eval.py`
4. Add any `test_percentage` override in both inference scripts if the dataset is small.
5. In `generative_baselines/run_dataset_v2.sh`: add to `has_depth/has_nvs` gates and to the relevant baseline-gate `case` arm.
6. If text-conditioned: add to the `prompt_flag` `case` in `run_dataset_v2.sh` and to `PROMPT_DATASETS` in `build_index_v2.py`.
7. Add the dataset to `run_all_v2.sh`'s wave 1 / 2 launcher.
8. Add the dataset to `DATASET_ORDER` in `build_index_v2.py` and to `datasets_*` lists in `collect_results_v2.py`.
9. Run `./run_dataset_v2.sh <name> <gpu>` once standalone before adding it to the parallel launcher.
