#!/bin/bash
# GPU 1 chain: re10k nvs (running) -> dl3dv nvs
set -uo pipefail
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new

CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-27/22-44-50/checkpoints/epoch=0-step=2850.ckpt
OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs
LOGS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs

PID=$(ps -ef | grep "scripts/inference_single_gpu_ray_to_rgb_depth_eval" | grep "ours_nvs" | grep "re10k" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "${PID:-}" ]; then
  echo "[chain] waiting for re10k nvs PID=$PID"
  while kill -0 "$PID" 2>/dev/null; do sleep 30; done
  echo "[chain] re10k nvs done"
fi

for DS in dl3dv re10k; do
  echo "[chain] launching ours nvs on $DS"
  CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL mamba run -n test2 python -u scripts/inference_single_gpu_ray_to_rgb_depth_eval.py \
    --ckpt_path "$CKPT" --dataset "$DS" \
    --output_dir "$OUT/$DS/ours_nvs" \
    --max_samples 10 --seed 42 --no_augmentations \
    > "$LOGS/ours_${DS}_nvs.log" 2>&1
  echo "[chain] $DS nvs exit=$?"
done
