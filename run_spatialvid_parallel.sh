#!/usr/bin/env bash
# spatialvid_nvs: fan all 4 inference jobs out concurrently on one GPU,
# then run evals + baselines serially. Single-shot, not generalised.
set -uo pipefail

DATASET=spatialvid_nvs

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new
M="mamba run -n test2 python"
DFLAG=re10k

run() { echo "[$DATASET] $*"; "$@"; }

# ── 1. fire all 4 inference jobs in parallel — one per GPU ───────────────────
echo "[$DATASET] starting 4 parallel inference jobs (1 per GPU) at $(date)"
CUDA_VISIBLE_DEVICES=0 $M scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
  --ckpt_path $OURS_CKPT --dataset $DATASET \
  --output_dir $OUT/$DATASET/ours_pose_depth \
  --max_samples 10 --seed 42 --no_augmentations &
P1=$!

CUDA_VISIBLE_DEVICES=1 $M scripts/inference_single_gpu_ray_to_rgb_depth_eval.py \
  --ckpt_path $OURS_CKPT --dataset $DATASET \
  --output_dir $OUT/$DATASET/ours_nvs \
  --max_samples 10 --seed 42 --no_augmentations &
P2=$!

CUDA_VISIBLE_DEVICES=2 $M scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
  --ckpt_path $NVS_ONLY_CKPT --dataset $DATASET \
  --output_dir $OUT/$DATASET/ours_nvs_only_pose_depth \
  --max_samples 10 --seed 42 --no_augmentations &
P3=$!

CUDA_VISIBLE_DEVICES=3 $M scripts/inference_single_gpu_ray_to_rgb_depth_eval.py \
  --ckpt_path $NVS_ONLY_CKPT --dataset $DATASET \
  --output_dir $OUT/$DATASET/ours_nvs_only_nvs \
  --max_samples 10 --seed 42 --no_augmentations &
P4=$!

wait $P1 $P2 $P3 $P4
echo "[$DATASET] all inference done at $(date)"
# pin GPU 2 for evals + baselines below
export CUDA_VISIBLE_DEVICES=2

# ── 2. our-method evals (CPU-light, run serially) ────────────────────────────
for METHOD in ours ours_nvs_only; do
  run $M scripts/eval_ray_mot_pose.py \
    --input_dir $OUT/$DATASET/${METHOD}_pose_depth \
    --output_dir $OUT/$DATASET/${METHOD}_pose_eval

  run $M scripts/recompute_metrics_offline.py \
    --input_dir $OUT/$DATASET/${METHOD}_nvs \
    --output_dir $OUT/$DATASET/${METHOD}_nvs_eval
done

# ── 3. baselines (heavy — keep serial on one GPU) ────────────────────────────
IN_PD=$OUT/$DATASET/ours_pose_depth
IN_NVS=$OUT/$DATASET/ours_nvs

run $M scripts/eval_geo4d_pose_2.py \
    --input_dir $IN_PD --output_dir $OUT/$DATASET/geo4d_pose \
    --ckpt_path $GEO4D_CKPT --max_samples 10 --rerun

run $M scripts/eval_raydiffusion_pose.py \
    --input_dir $IN_PD --output_dir $OUT/$DATASET/raydiffusion_pose \
    --model_dir $RAYDIFF_DIR --max_samples 10 --dataset $DFLAG

run $M scripts/eval_gen3c_nvs.py \
    --input_dir $IN_NVS --output_dir $OUT/$DATASET/gen3c_nvs \
    --ckpt_dir $GEN3C_CKPT --max_samples 10 --dataset $DFLAG

run $M scripts/eval_svc_nvs.py \
    --input_dir $IN_NVS --output_dir $OUT/$DATASET/seva_nvs \
    --max_samples 10 --dataset $DFLAG

run $M scripts/inference_single_gpu_flf2v_from_mp4_eval.py \
    --input_dir $IN_NVS --output_dir $OUT/$DATASET/wan_flf_nvs \
    --max_samples 10 --show_metrics --save_stats

echo "[$DATASET gpu$GPU] DONE at $(date)"
