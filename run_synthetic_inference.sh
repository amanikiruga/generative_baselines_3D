#!/bin/bash
# Run ours + ours_nvs_only on aria + vkitti2.
# Per dataset: launch both ckpts in parallel on the same H100 (peak ~60GB).
set -uo pipefail
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new

CKPT_OURS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-27/22-44-50/checkpoints/last.ckpt
CKPT_NVS_ONLY=/net/holy-isilon/ifs/rc_labs/ydu_lab/zhiyi24/workspace/video_world_model/outputs/2026-04-26/07-31-50/checkpoints/last.ckpt
OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs
LOGS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs
N=10
SHARED=( --max_samples $N --seed 42 --no_augmentations )

for DS in aria vkitti2 scenenet_depth; do
  echo "[syn] === pose_depth on $DS (ours || ours_nvs_only) ==="
  CUDA_VISIBLE_DEVICES=0 stdbuf -oL mamba run -n test2 python -u scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
    --ckpt_path "$CKPT_OURS" --dataset "$DS" --output_dir "$OUT/$DS/ours_pose_depth" "${SHARED[@]}" \
    > "$LOGS/ours_${DS}_pose_depth.log" 2>&1 &
  PID_O=$!
  CUDA_VISIBLE_DEVICES=0 stdbuf -oL mamba run -n test2 python -u scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
    --ckpt_path "$CKPT_NVS_ONLY" --dataset "$DS" --output_dir "$OUT/$DS/ours_nvs_only_pose_depth" "${SHARED[@]}" \
    > "$LOGS/ours_nvs_only_${DS}_pose_depth.log" 2>&1 &
  PID_N=$!
  wait $PID_O; echo "[syn] ours_${DS}_pose_depth exit=$?"
  wait $PID_N; echo "[syn] ours_nvs_only_${DS}_pose_depth exit=$?"

  if [ "$DS" != "scenenet_depth" ]; then
    echo "[syn] === nvs on $DS (ours || ours_nvs_only) ==="
    CUDA_VISIBLE_DEVICES=0 stdbuf -oL mamba run -n test2 python -u scripts/inference_single_gpu_ray_to_rgb_depth_eval.py \
      --ckpt_path "$CKPT_OURS" --dataset "$DS" --output_dir "$OUT/$DS/ours_nvs" "${SHARED[@]}" \
      > "$LOGS/ours_${DS}_nvs.log" 2>&1 &
    PID_O=$!
    CUDA_VISIBLE_DEVICES=0 stdbuf -oL mamba run -n test2 python -u scripts/inference_single_gpu_ray_to_rgb_depth_eval.py \
      --ckpt_path "$CKPT_NVS_ONLY" --dataset "$DS" --output_dir "$OUT/$DS/ours_nvs_only_nvs" "${SHARED[@]}" \
      > "$LOGS/ours_nvs_only_${DS}_nvs.log" 2>&1 &
    PID_N=$!
    wait $PID_O; echo "[syn] ours_${DS}_nvs exit=$?"
    wait $PID_N; echo "[syn] ours_nvs_only_${DS}_nvs exit=$?"
  fi
done

# Eval after inference (CPU-only, parallel)
EVAL_PIDS=()
for DS in aria vkitti2 scenenet_depth; do
  for METHOD in ours ours_nvs_only; do
    SUFFIX=$([ "$METHOD" = "ours" ] && echo "" || echo "_nvs_only")
    PD="$OUT/$DS/${METHOD}_pose_depth"
    NVS="$OUT/$DS/${METHOD}_nvs"
    if [ -d "$PD" ]; then
      if [ "$DS" != "scenenet_depth" ]; then
        stdbuf -oL mamba run -n test2 python -u scripts/eval_ray_mot_pose.py \
          --input_dir "$PD" --output_dir "$OUT/$DS/${METHOD}_pose_eval" \
          > "$LOGS/${DS}_${METHOD}_pose_eval.log" 2>&1 &
        EVAL_PIDS+=( $! )
      fi
      stdbuf -oL mamba run -n test2 python -u scripts/eval_rgb_to_ray_depth_depth.py \
        --input_dir "$PD" --output_dir "$OUT/$DS/${METHOD}_depth_eval" --dataset "$DS" \
        > "$LOGS/${DS}_${METHOD}_depth_eval.log" 2>&1 &
      EVAL_PIDS+=( $! )
    fi
    if [ -d "$NVS" ]; then
      stdbuf -oL mamba run -n test2 python -u scripts/recompute_metrics_offline.py \
        --input_dir "$NVS" --output_dir "$OUT/$DS/${METHOD}_nvs_eval" \
        > "$LOGS/${DS}_${METHOD}_nvs_eval.log" 2>&1 &
      EVAL_PIDS+=( $! )
    fi
  done
done
for p in "${EVAL_PIDS[@]}"; do wait "$p"; done
echo "[syn] all evals done"
python3 /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/collect_results.py >/dev/null
python3 /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/build_index.py >/dev/null
echo "[syn] DONE"
