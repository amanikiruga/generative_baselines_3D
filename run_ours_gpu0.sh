#!/bin/bash
# GPU 0 chain: re10k pose_depth (already running) -> dl3dv pose_depth -> scannetpp pose_depth
# Started after the currently-running RE10K pose_depth job exits.
set -uo pipefail
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new

CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-27/22-44-50/checkpoints/epoch=0-step=2850.ckpt
OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs
LOGS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs

# Wait for the currently-running re10k pose_depth job to exit (PID lookup once)
PID=$(ps -ef | grep "scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval" | grep "ours_pose_depth" | grep "re10k" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "${PID:-}" ]; then
  echo "[chain] waiting for re10k pose_depth PID=$PID"
  while kill -0 "$PID" 2>/dev/null; do sleep 30; done
  echo "[chain] re10k pose_depth done"
fi

for DS in dl3dv scannetpp re10k; do
  echo "[chain] launching ours pose_depth on $DS"
  CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL mamba run -n test2 python -u scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
    --ckpt_path "$CKPT" --dataset "$DS" \
    --output_dir "$OUT/$DS/ours_pose_depth" \
    --max_samples 10 --seed 42 --no_augmentations \
    > "$LOGS/ours_${DS}_pose_depth.log" 2>&1
  echo "[chain] $DS pose_depth exit=$?"
done
