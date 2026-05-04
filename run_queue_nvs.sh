#!/bin/bash
# NVS queue: run after depth queue. Same max-2-parallel pattern. SceneNet has no
# real cameras so NVS is skipped; Aria + VKitti2 only.
set -uo pipefail
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new

PY=/net/holy-isilon/ifs/rc_labs/ydu_lab/akiruga/.conda/envs/test2/bin/python
CKPT_OURS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-27/22-44-50/checkpoints/last.ckpt
CKPT_NVS_ONLY=/net/holy-isilon/ifs/rc_labs/ydu_lab/zhiyi24/workspace/video_world_model/outputs/2026-04-26/07-31-50/checkpoints/last.ckpt
OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs
LOGS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs
N=10

declare -a JOBS=(
  "ours_aria_nvs|$CKPT_OURS|aria|aria/ours_nvs"
  "ours_nvs_only_aria_nvs|$CKPT_NVS_ONLY|aria|aria/ours_nvs_only_nvs"
  "ours_vkitti2_nvs|$CKPT_OURS|vkitti2|vkitti2/ours_nvs"
  "ours_nvs_only_vkitti2_nvs|$CKPT_NVS_ONLY|vkitti2|vkitti2/ours_nvs_only_nvs"
)

run_one() {
  local label="$1" ckpt="$2" ds="$3" out="$4"
  echo "[Qn] start $label"
  CUDA_VISIBLE_DEVICES=0 stdbuf -oL "$PY" -u scripts/inference_single_gpu_ray_to_rgb_depth_eval.py \
    --ckpt_path "$ckpt" --dataset "$ds" --output_dir "$OUT/$out" \
    --max_samples $N --seed 42 --no_augmentations \
    > "$LOGS/queue_${label}.log" 2>&1
  echo "[Qn] done $label exit=$?"
}

i=0
while [ $i -lt ${#JOBS[@]} ]; do
  IFS='|' read -r L1 C1 D1 O1 <<< "${JOBS[$i]}"
  PID1=""; PID2=""
  run_one "$L1" "$C1" "$D1" "$O1" & PID1=$!
  if [ $((i+1)) -lt ${#JOBS[@]} ]; then
    IFS='|' read -r L2 C2 D2 O2 <<< "${JOBS[$((i+1))]}"
    run_one "$L2" "$C2" "$D2" "$O2" & PID2=$!
  fi
  [ -n "$PID1" ] && wait "$PID1"
  [ -n "$PID2" ] && wait "$PID2"
  i=$((i+2))
done
echo "[Qn] === all NVS inference done ==="

EPIDS=()
for DS in aria vkitti2; do
  for METHOD in ours ours_nvs_only; do
    NVS="$OUT/$DS/${METHOD}_nvs"
    if [ -d "$NVS" ] && [ -n "$(ls -A "$NVS" 2>/dev/null)" ]; then
      stdbuf -oL "$PY" -u scripts/recompute_metrics_offline.py \
        --input_dir "$NVS" --output_dir "$OUT/$DS/${METHOD}_nvs_eval" \
        > "$LOGS/queue_${DS}_${METHOD}_nvs_eval.log" 2>&1 &
      EPIDS+=( $! )
    fi
  done
done
for p in "${EPIDS[@]}"; do wait "$p"; done
echo "[Qn] === all NVS evals done ==="

"$PY" /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/collect_results.py >/dev/null
"$PY" /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/build_index.py    >/dev/null
echo "[Qn] DONE"
