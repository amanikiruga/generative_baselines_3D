#!/bin/bash
# Retry re10k inference (post-bugfix). Waits for both GPUs to be idle of inference jobs.
set -uo pipefail
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new

CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-27/22-44-50/checkpoints/epoch=0-step=2850.ckpt
OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs
LOGS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs

# Wait for GPU 0 to be free (no inference_single python process)
echo "[re10k retry] waiting for GPU 0 to free up"
while pgrep -f "CUDA_VISIBLE_DEVICES=0.*inference_single" >/dev/null 2>&1; do sleep 30; done
echo "[re10k retry] GPU 0 free; launching re10k pose_depth"
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL mamba run -n test2 python -u scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
  --ckpt_path "$CKPT" --dataset re10k \
  --output_dir "$OUT/re10k/ours_pose_depth" \
  --max_samples 10 --seed 42 --no_augmentations \
  > "$LOGS/ours_re10k_pose_depth.log" 2>&1 &
PID0=$!

echo "[re10k retry] waiting for GPU 1 to free up"
while pgrep -f "CUDA_VISIBLE_DEVICES=1.*inference_single" >/dev/null 2>&1; do sleep 30; done
echo "[re10k retry] GPU 1 free; launching re10k nvs"
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL mamba run -n test2 python -u scripts/inference_single_gpu_ray_to_rgb_depth_eval.py \
  --ckpt_path "$CKPT" --dataset re10k \
  --output_dir "$OUT/re10k/ours_nvs" \
  --max_samples 10 --seed 42 --no_augmentations \
  > "$LOGS/ours_re10k_nvs.log" 2>&1 &
PID1=$!

wait $PID0; echo "[re10k retry] pose_depth exit=$?"
wait $PID1; echo "[re10k retry] nvs exit=$?"
