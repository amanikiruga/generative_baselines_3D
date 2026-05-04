#!/bin/bash
# Depth-priority queue: pose_depth inference (rgb→ray+depth) for the 3 new datasets,
# then depth eval. Order: scenenet_depth → vkitti2 → aria. Max 2 parallel.
set -uo pipefail
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new

PY=/net/holy-isilon/ifs/rc_labs/ydu_lab/akiruga/.conda/envs/test2/bin/python
CKPT_OURS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-27/22-44-50/checkpoints/last.ckpt
CKPT_NVS_ONLY=/net/holy-isilon/ifs/rc_labs/ydu_lab/zhiyi24/workspace/video_world_model/outputs/2026-04-26/07-31-50/checkpoints/last.ckpt
OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs
LOGS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs
N=10

# pose_depth jobs in priority order: (label|ckpt|dataset|out)
# vkitti2 ours_nvs_only_pose_depth already DONE 10/10 — skip.
declare -a JOBS=(
  "ours_scenenet_pose_depth|$CKPT_OURS|scenenet_depth|scenenet_depth/ours_pose_depth"
  "ours_nvs_only_scenenet_pose_depth|$CKPT_NVS_ONLY|scenenet_depth|scenenet_depth/ours_nvs_only_pose_depth"
  "ours_vkitti2_pose_depth|$CKPT_OURS|vkitti2|vkitti2/ours_pose_depth"
  "ours_aria_pose_depth|$CKPT_OURS|aria|aria/ours_pose_depth"
  "ours_nvs_only_aria_pose_depth|$CKPT_NVS_ONLY|aria|aria/ours_nvs_only_pose_depth"
)

run_one() {
  local label="$1" ckpt="$2" ds="$3" out="$4"
  echo "[Qd] start $label"
  CUDA_VISIBLE_DEVICES=0 stdbuf -oL "$PY" -u scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
    --ckpt_path "$ckpt" --dataset "$ds" --output_dir "$OUT/$out" \
    --max_samples $N --seed 42 --no_augmentations \
    > "$LOGS/queue_${label}.log" 2>&1
  echo "[Qd] done $label exit=$?"
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
echo "[Qd] === all pose_depth inference done ==="

# Depth + pose evals (CPU, parallel)
EPIDS=()
for DS in scenenet_depth vkitti2 aria; do
  for METHOD in ours ours_nvs_only; do
    PD="$OUT/$DS/${METHOD}_pose_depth"
    if [ -d "$PD" ] && [ -n "$(ls -A "$PD" 2>/dev/null)" ]; then
      stdbuf -oL "$PY" -u scripts/eval_rgb_to_ray_depth_depth.py \
        --input_dir "$PD" --output_dir "$OUT/$DS/${METHOD}_depth_eval" --dataset "$DS" \
        > "$LOGS/queue_${DS}_${METHOD}_depth_eval.log" 2>&1 &
      EPIDS+=( $! )
      if [ "$DS" != "scenenet_depth" ]; then
        stdbuf -oL "$PY" -u scripts/eval_ray_mot_pose.py \
          --input_dir "$PD" --output_dir "$OUT/$DS/${METHOD}_pose_eval" \
          > "$LOGS/queue_${DS}_${METHOD}_pose_eval.log" 2>&1 &
        EPIDS+=( $! )
      fi
    fi
  done
done
for p in "${EPIDS[@]}"; do wait "$p"; done
echo "[Qd] === all depth+pose evals done ==="

"$PY" /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/collect_results.py >/dev/null
"$PY" /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/build_index.py    >/dev/null
echo "[Qd] DONE"
