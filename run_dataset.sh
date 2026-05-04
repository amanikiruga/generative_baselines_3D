#!/usr/bin/env bash
# usage: ./run_dataset.sh <dataset> <gpu_id>
# all envs (OURS_CKPT, NVS_ONLY_CKPT, GEO4D_CKPT, DFOT_RE10K_CKPT, GEN3C_CKPT, RAYDIFF_DIR, OUT)
# must be exported by the caller (run_all.sh).

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

# baseline --dataset flag (Aria→OpenCV branch only on aria)
if [ "$DATASET" = "aria" ]; then DFLAG=aria; else DFLAG=re10k; fi

# baseline gates (from README_CLEAN §4)
b_geo4d_pose=0; b_raydiff=0; b_geo4d_depth=0; b_chrono=0
b_gen3c=0; b_dfot=0; b_seva=0; b_wan=0
case $DATASET in
  re10k)
    b_geo4d_pose=1; b_raydiff=1; b_gen3c=1; b_dfot=1; b_seva=1; b_wan=1 ;;
  dl3dv)
    b_geo4d_pose=1; b_raydiff=1; b_geo4d_depth=1; b_chrono=1; b_gen3c=1; b_seva=1; b_wan=1 ;;
  dl3dv_test|tanksandtemples|spatialvid_nvs|agibot_world)
    b_geo4d_pose=1; b_raydiff=1; b_gen3c=1; b_seva=1; b_wan=1 ;;
  scannetpp)
    b_geo4d_pose=1; b_raydiff=1; b_geo4d_depth=1; b_chrono=1; b_gen3c=1; b_seva=1; b_wan=1 ;;
  aria)
    b_geo4d_pose=1; b_raydiff=1; b_geo4d_depth=1; b_chrono=1; b_gen3c=1; b_dfot=1; b_seva=1; b_wan=1 ;;
  vkitti2)
    b_geo4d_pose=1; b_raydiff=1; b_geo4d_depth=1; b_chrono=1 ;;
  scenenet_depth)
    b_geo4d_depth=1; b_chrono=1 ;;
esac

# ── 1. our-method inference (both checkpoints, identical task coverage) ────────
for pair in "ours:$OURS_CKPT" "ours_nvs_only:$NVS_ONLY_CKPT"; do
  METHOD=${pair%%:*}; CKPT=${pair#*:}

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
done

# ── 2. our-method evals (both checkpoints) ─────────────────────────────────────
for METHOD in ours ours_nvs_only; do
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
done

# ── 3. baselines (consume ours_pose_depth / ours_nvs) ──────────────────────────
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

[ $b_gen3c -eq 1 ] && run $M scripts/eval_gen3c_nvs.py \
    --input_dir $IN_NVS --output_dir $OUT/$DATASET/gen3c_nvs \
    --ckpt_dir $GEN3C_CKPT --max_samples 10 --dataset $DFLAG

[ $b_dfot -eq 1 ] && run $M scripts/eval_dfot_nvs_re10k.py \
    --input_dir $IN_NVS --output_dir $OUT/$DATASET/dfot_nvs \
    --ckpt_path $DFOT_RE10K_CKPT --max_samples 10 --dataset $DFLAG

[ $b_seva -eq 1 ] && run $M scripts/eval_svc_nvs.py \
    --input_dir $IN_NVS --output_dir $OUT/$DATASET/seva_nvs \
    --max_samples 10 --dataset $DFLAG

[ $b_wan -eq 1 ] && run $M scripts/inference_single_gpu_flf2v_from_mp4_eval.py \
    --input_dir $IN_NVS --output_dir $OUT/$DATASET/wan_flf_nvs \
    --max_samples 10 --show_metrics --save_stats

echo "[$DATASET gpu$GPU] DONE"
