# Eval Commands

All inference + eval commands below run from `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new` with `mamba run -n test2`. The orchestrator scripts (`run_*.sh`, `status.sh`, `collect_results.py`, `build_index.py`, `serve_index.py`) live in `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/` and run from there.

## 0. Quick start (one-shot reproduction)

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines

# Wipe netscratch eval dir contents (the symlink target) — keep the symlink.
TARGET=$(readlink -f $OUT); mv "$TARGET" "${TARGET}.bak.$(date +%s)"; mkdir -p "$TARGET"

# (a) Recommended: 8 datasets fanned across 4 GPUs (2 datasets per GPU)
./run_all.sh

# (b) Single dataset, sequential pipeline on one GPU (steps 1→3 below):
./run_dataset.sh <dataset> <gpu_id>          # e.g. ./run_dataset.sh dl3dv 1

# (c) Single dataset, fan 4 inference jobs across 4 GPUs (faster), then evals+baselines on one GPU:
./run_dataset_parallel.sh <dataset> <gpu_for_evals_and_baselines>

# Watch progress + auto-refresh index.html every 60 s:
./status.sh --watch       # foreground; or:
nohup ./status.sh --watch > eval_logs/_status.log 2>&1 & disown

# Serve the live index:
python serve_index.py
```

`$OUT` is a symlink `eval_outputs → /n/netscratch/ydu_lab/Everyone/akiruga/generative_baselines_eval/`. Per-dataset logs land in `eval_logs/<dataset>.log`.

## Env

```bash
export OURS_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-27/22-44-50/checkpoints/last.ckpt
export NVS_ONLY_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-29/14-31-15/checkpoints/last.ckpt
export GEO4D_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
export DFOT_RE10K_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/diffusion-forcing-transformer/pretrained/DFoT_RE10K.ckpt
export GEN3C_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/GEN3C/checkpoints
export RAYDIFF_DIR=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion
export OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs
```

## Task / dataset matrix

| Dataset           | pose | depth | NVS |
|-------------------|------|-------|-----|
| re10k             | ✓    | ✗     | ✓   |
| dl3dv             | ✓    | ✓     | ✓   |
| dl3dv_test        | ✓    | ✗     | ✓   |
| tanksandtemples   | ✓    | ✗     | ✓   |
| scannetpp         | ✓    | ✓     | ✓   |
| aria              | ✓    | ✓     | ✓   |
| vkitti2           | ✓    | ✓     | ✓   |
| scenenet_depth    | ✗    | ✓     | ✗   |
| spatialvid_nvs    | ✓    | ✗     | ✓   |
| agibot_world      | ✓    | ✗     | ✓   |

For every dataset run inference twice: `CKPT=$OURS_CKPT METHOD=ours` and `CKPT=$NVS_ONLY_CKPT METHOD=ours_nvs_only`.

## 1. Our method — inference

### pose+depth (RGB → ray+depth)
```bash
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
  --ckpt_path $CKPT --dataset $DATASET --output_dir $OUT/$DATASET/${METHOD}_pose_depth \
  --max_samples 10 --seed 42 --no_augmentations
```

### NVS (ray+firstlast → rgb+depth)
```bash
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/inference_single_gpu_ray_to_rgb_depth_eval.py \
  --ckpt_path $CKPT --dataset $DATASET --output_dir $OUT/$DATASET/${METHOD}_nvs \
  --max_samples 10 --seed 42 --no_augmentations
```

## 2. Our method — eval

```bash
# pose
mamba run -n test2 python scripts/eval_ray_mot_pose.py \
  --input_dir $OUT/$DATASET/${METHOD}_pose_depth --output_dir $OUT/$DATASET/${METHOD}_pose_eval

# depth
mamba run -n test2 python scripts/eval_rgb_to_ray_depth_depth.py \
  --input_dir $OUT/$DATASET/${METHOD}_pose_depth --output_dir $OUT/$DATASET/${METHOD}_depth_eval \
  --dataset $DATASET

# NVS
mamba run -n test2 python scripts/recompute_metrics_offline.py \
  --input_dir $OUT/$DATASET/${METHOD}_nvs --output_dir $OUT/$DATASET/${METHOD}_nvs_eval
```

## 3. Baselines

Inputs: `$IN_PD = $OUT/$DATASET/ours_pose_depth`, `$IN_NVS = $OUT/$DATASET/ours_nvs`.

**`$DATASET_FLAG` rule (passed via `--dataset` to baselines):**
- `aria` → `$DATASET_FLAG=aria` (enables Aria device→OpenCV camera transform)
- everything else → `$DATASET_FLAG=re10k` (uses raw OpenCV cameras; works for all OpenCV-frame datasets including dl3dv, dl3dv_test, tanksandtemples, scannetpp, vkitti2, spatialvid_nvs, agibot_world).

### GEO4D — pose
```bash
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_geo4d_pose_2.py \
  --input_dir $IN_PD --output_dir $OUT/$DATASET/geo4d_pose \
  --ckpt_path $GEO4D_CKPT --max_samples 10 --rerun
```

### GEO4D — depth
```bash
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_geo4d_depth.py \
  --input_dir $IN_PD --output_dir $OUT/$DATASET/geo4d_depth \
  --ckpt_path $GEO4D_CKPT --dataset $DATASET --max_samples 10 --rerun
```

### ChronoDepth — depth
```bash
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/eval_chronodepth_depth_aria.py \
  --input_dir $IN_PD --output_dir $OUT/$DATASET/chronodepth_depth \
  --dataset $DATASET --max_samples 10
```

### RayDiffusion — pose
```bash
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/eval_raydiffusion_pose.py \
  --input_dir $IN_PD --output_dir $OUT/$DATASET/raydiffusion_pose \
  --model_dir $RAYDIFF_DIR --max_samples 10 --dataset $DATASET_FLAG
```

### GEN3C — NVS
```bash
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_gen3c_nvs.py \
  --input_dir $IN_NVS --output_dir $OUT/$DATASET/gen3c_nvs \
  --ckpt_dir $GEN3C_CKPT --max_samples 10 --dataset $DATASET_FLAG
```

### DFoT — NVS (re10k, aria only)
```bash
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/eval_dfot_nvs_re10k.py \
  --input_dir $IN_NVS --output_dir $OUT/$DATASET/dfot_nvs \
  --ckpt_path $DFOT_RE10K_CKPT --max_samples 10 --dataset $DATASET_FLAG
```

### SEVA — NVS
```bash
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_svc_nvs.py \
  --input_dir $IN_NVS --output_dir $OUT/$DATASET/seva_nvs --max_samples 10 --dataset $DATASET_FLAG
```

### Wan FLF — NVS
```bash
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/inference_single_gpu_flf2v_from_mp4_eval.py \
  --input_dir $IN_NVS --output_dir $OUT/$DATASET/wan_flf_nvs \
  --max_samples 10 --show_metrics --save_stats
```

## 4. Per-dataset baseline coverage

| Dataset          | geo4d_pose | raydiffusion_pose | geo4d_depth | chronodepth_depth | gen3c_nvs | dfot_nvs | seva_nvs | wan_flf_nvs |
|------------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| re10k            | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| dl3dv            | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| dl3dv_test       | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ |
| tanksandtemples  | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ |
| scannetpp        | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| aria             | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| vkitti2          | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| scenenet_depth   | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| spatialvid_nvs   | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ |
| agibot_world     | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ |

## 5. Aggregate / serve

Run from `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/`:

```bash
mamba run -n test2 python collect_results.py    # writes RESULTS.md
mamba run -n test2 python build_index.py        # writes index.html
python serve_index.py                           # local web server (Ctrl-C to stop)
```

`status.sh --watch` re-runs the two aggregate scripts every 60 s automatically — point at this if the watch loop is up.

## 6. Outputs layout

```
$OUT/<dataset>/
├── ours_pose_depth/                ours_nvs_only_pose_depth/
├── ours_nvs/                       ours_nvs_only_nvs/
├── ours_{pose,depth,nvs}_eval/     ours_nvs_only_{pose,depth,nvs}_eval/
├── geo4d_pose/   geo4d_depth/
├── chronodepth_depth/
├── raydiffusion_pose/
├── gen3c_nvs/  dfot_nvs/  seva_nvs/  wan_flf_nvs/
```

`$OUT` (`generative_baselines/eval_outputs`) is a symlink to `/n/netscratch/ydu_lab/Everyone/akiruga/generative_baselines_eval/`. Logs: `generative_baselines/eval_logs/<job>.log`.

## 7. Known gotchas

- **gen3c eval fails on every dataset** with `ImportError: cannot import name 'apply_rotary_pos_emb' from 'transformer_engine.pytorch.attention'`. Treat gen3c columns as N/A in `RESULTS.md` until the GEN3C transformer_engine version is reconciled.
- **Aria** has 14,967 raw clips → only ~20 retained after the 3.5°/frame rotation filter. Inference scripts auto-set `dataset.test_percentage=1.0` for aria so the `[:n_test]` validation slice keeps all 20. With `--max_samples 10` you get 10.
- **Tanks & Temples** has only 5 valid scenes on disk (`/n/holylfs05/.../tanksandtemples`). Inference auto-sets `test_percentage=1.0`. Max samples = 5 regardless of `--max_samples`.
- **agibot_world** uses a static head-camera (zero motion). Pose AUC is comparing against trivial all-identity GT — interpret with caution. NVS is meaningful (dynamic objects, static viewpoint = video frame interpolation).
- **scenenet_depth** loader does not emit a real `raymap_scale` (no GT cameras), so `scale=1.0` ends up in NPZ. Eval still selects the metric branch but `pred_d *= 1; gt_d *= 1` is a no-op → results are in scaled-depth space, not metres. Relative metrics (d1/d2/d3, abs_rel) are trustworthy; `rmse` is in scaled-depth units, not metres.
- **vkitti2** gates 46% of pixels (sky > 80 m). Metrics are computed on the foreground 36%.
- **Depth viz** for metric datasets (aria, vkitti2, scenenet_depth): the per-clip viz mp4 was previously broken (missing `/scale` in inverse formula). Fixed in `eval_rgb_to_ray_depth_depth.py`. Re-run the depth eval if you have stale viz.

## 8. Adding a new dataset

When wiring up a new dataset, you need to touch these files:

1. `datasets/<name>.py` — the loader class.
2. `configurations/dataset/<name>.yaml` — config; must set `height`, `width`, and `raymap_height = height/8`, `raymap_width = width/8`. Set `load_prompt_embed: true` if the model's `prepare_embeds()` requires `batch["prompt_embeds"]`.
3. `scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py` — add to imports + `_DATASETS` dispatch + `test_percentage` bump if the default yields too few records.
4. `scripts/inference_single_gpu_ray_to_rgb_depth_eval.py` — same as above.
5. `generative_baselines/run_dataset.sh` — add to `has_depth` / `has_nvs` task gates and to the `b_*` baseline gate `case` block.
6. `generative_baselines/status.sh` — add to `DATASETS`, `has_nvs`, and the `b_has` gate.
7. `generative_baselines/collect_results.py` — add to `datasets_nvs` / `datasets_pose` / `datasets_depth`.
8. `generative_baselines/build_index.py` — add to `DATASET_ORDER` (with the relevant `["nvs", "pose", "depth"]` subset).
9. `README_CLEAN.md` — both matrices in §"Task / dataset matrix" and §4.

If the dataset is slow-motion or low-FPS, also add it to `SLOW_DATASETS` in `build_index.py:302` so the index plays it at 0.5×.
