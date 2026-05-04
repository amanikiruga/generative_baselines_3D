#!/usr/bin/env bash
# usage: ./run_dataset_v2.sh <dataset> <gpu_id>
#
# v2 differences vs run_dataset.sh:
#   - single OURS_FINAL checkpoint (no ours_nvs_only)
#   - --load_prompt_embed for re10k / dl3dv / agibot_world (text-conditioned)
#   - gen3c dropped (broken / no checkpoint available)
#   - wan_flf moved to the LAST baseline (it's the slowest)
#   - writes into $OUT (= eval_outputs_v2 → /n/netscratch/.../generative_baselines_eval_v2)
#
# Required env exports (set by run_all_v2.sh):
#   OURS_FINAL  GEO4D_CKPT  DFOT_RE10K_CKPT  RAYDIFF_DIR  OUT

set -uo pipefail
DATASET=$1
GPU=$2
export CUDA_VISIBLE_DEVICES=$GPU

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new

run() { echo "[$DATASET gpu$GPU] $*"; "$@"; }
M="mamba run -n test2 python"

# task gates
has_depth=0; has_nvs=0
case $DATASET in
  re10k|dl3dv_test|tanksandtemples|spatialvid_nvs|agibot_world) has_nvs=1 ;;
  dl3dv|aria|vkitti2|scannetpp)                                 has_depth=1; has_nvs=1 ;;
  scenenet_depth)                                               has_depth=1 ;;
esac

# Text conditioning: the inference scripts unconditionally pass
# `algorithm.load_prompt_embed=True` as a hydra override, so prompt-embed loading
# is gated by the dataset YAML's `load_prompt_embed:` flag. No CLI flag needed.

# baseline --dataset flag (Aria→OpenCV branch only on aria)
if [ "$DATASET" = "aria" ]; then DFLAG=aria; else DFLAG=re10k; fi

# baseline gates (gen3c dropped from all)
b_geo4d_pose=0; b_raydiff=0; b_geo4d_depth=0; b_chrono=0
b_dfot=0; b_seva=0; b_wan=0
case $DATASET in
  re10k)
    b_geo4d_pose=1; b_raydiff=1; b_dfot=1; b_seva=1; b_wan=1 ;;
  dl3dv)
    b_geo4d_pose=1; b_raydiff=1; b_geo4d_depth=1; b_chrono=1; b_seva=1; b_wan=1 ;;
  dl3dv_test|tanksandtemples|spatialvid_nvs|agibot_world)
    b_geo4d_pose=1; b_raydiff=1; b_seva=1; b_wan=1 ;;
  scannetpp)
    b_geo4d_pose=1; b_raydiff=1; b_geo4d_depth=1; b_chrono=1; b_seva=1; b_wan=1 ;;
  aria)
    b_geo4d_pose=1; b_raydiff=1; b_geo4d_depth=1; b_chrono=1; b_dfot=1; b_seva=1; b_wan=1 ;;
  vkitti2)
    b_geo4d_pose=1; b_raydiff=1; b_geo4d_depth=1; b_chrono=1 ;;
  scenenet_depth)
    b_geo4d_depth=1; b_chrono=1 ;;
esac

# ── 1. our-method inference (single ckpt = OURS_FINAL) ─────────────────────────
METHOD=ours
CKPT=$OURS_FINAL

run $M scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
    --ckpt_path $CKPT --dataset $DATASET \
--output_dir $OUT/$DATASET/${METHOD}_pose_depth \
    --max_samples 10 --seed 42 --no_augmentations

if [ $has_nvs -eq 1 ]; then
    run $M scripts/inference_single_gpu_ray_to_rgb_depth_eval.py \
        --ckpt_path $CKPT --dataset $DATASET \
        --output_dir $OUT/$DATASET/${METHOD}_nvs \
        --max_samples 10 --seed 42 --no_augmentations
fi

# ── 2. our-method evals ────────────────────────────────────────────────────────
run $M scripts/eval_ray_mot_pose.py \
    --input_dir $OUT/$DATASET/${METHOD}_pose_depth \
    --output_dir $OUT/$DATASET/${METHOD}_pose_eval

if [ $has_depth -eq 1 ]; then
    run $M scripts/eval_rgb_to_ray_depth_depth.py \
        --input_dir $OUT/$DATASET/${METHOD}_pose_depth \
        --output_dir $OUT/$DATASET/${METHOD}_depth_eval \
        --dataset $DATASET
fi

if [ $has_nvs -eq 1 ]; then
    run $M scripts/recompute_metrics_offline.py \
        --input_dir $OUT/$DATASET/${METHOD}_nvs \
        --output_dir $OUT/$DATASET/${METHOD}_nvs_eval
fi

# ── 3. baselines (consume ours_pose_depth / ours_nvs); wan_flf last ────────────
IN_PD=$OUT/$DATASET/ours_pose_depth
IN_NVS=$OUT/$DATASET/ours_nvs

[ $b_geo4d_pose -eq 1 ] && run $M scripts/eval_geo4d_pose_2.py \
    --input_dir $IN_PD --output_dir $OUT/$DATASET/geo4d_pose \
    --ckpt_path $GEO4D_CKPT --max_samples 10 --rerun

[ $b_raydiff -eq 1 ] && run $M scripts/eval_raydiffusion_pose.py \
    --input_dir $IN_PD --output_dir $OUT/$DATASET/raydiffusion_pose \
    --model_dir $RAYDIFF_DIR --max_samples 10 --dataset $DFLAG

[ $b_geo4d_depth -eq 1 ] && run $M scripts/eval_geo4d_depth.py \
    --input_dir $IN_PD --output_dir $OUT/$DATASET/geo4d_depth \
    --ckpt_path $GEO4D_CKPT --dataset $DATASET --max_samples 10 --rerun

[ $b_chrono -eq 1 ] && run $M scripts/eval_chronodepth_depth_aria.py \
    --input_dir $IN_PD --output_dir $OUT/$DATASET/chronodepth_depth \
    --dataset $DATASET --max_samples 10

[ $b_dfot -eq 1 ] && run $M scripts/eval_dfot_nvs_re10k.py \
    --input_dir $IN_NVS --output_dir $OUT/$DATASET/dfot_nvs \
    --ckpt_path $DFOT_RE10K_CKPT --max_samples 10 --dataset $DFLAG

[ $b_seva -eq 1 ] && run $M scripts/eval_svc_nvs.py \
    --input_dir $IN_NVS --output_dir $OUT/$DATASET/seva_nvs \
    --max_samples 10 --dataset $DFLAG

# Wan 2.1 FLF — slowest baseline, last so faster ones land first.
[ $b_wan -eq 1 ] && run $M scripts/inference_single_gpu_flf2v_from_mp4_eval.py \
    --input_dir $IN_NVS --output_dir $OUT/$DATASET/wan_flf_nvs \
    --max_samples 10 --show_metrics --save_stats

echo "[$DATASET gpu$GPU] DONE"
