#!/bin/bash
# Run the OURS_NVS_ONLY checkpoint through the same pipeline as Ours.
# Inference runs serially on the H100 (peak ~30 GB per job, two-in-parallel OOMs).
# Eval phase is CPU-only and runs in parallel.
set -uo pipefail

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new

CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/zhiyi24/workspace/video_world_model/outputs/2026-04-26/07-31-50/checkpoints/last.ckpt
OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs
LOGS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs
N=10
SHARED_ARGS=(--ckpt_path "$CKPT" --max_samples $N --seed 42 --no_augmentations)

infer_one() {  # name, infer_script, dataset, out_subdir
  local name=$1; local script=$2; local DS=$3; local subdir=$4
  echo "[bg] start $name"
  CUDA_VISIBLE_DEVICES=0 stdbuf -oL mamba run -n test2 python -u "scripts/$script" \
    "${SHARED_ARGS[@]}" --dataset "$DS" --output_dir "$OUT/$DS/$subdir" \
    > "$LOGS/${name}.log" 2>&1
  echo "[bg] $name exit=$?"
}

# ============ Inference (serial) ============
for DS in dl3dv re10k; do
  infer_one "ours_nvs_only_${DS}_pose_depth" inference_single_gpu_rgb_to_ray_depth_aria_eval.py "$DS" ours_nvs_only_pose_depth
  infer_one "ours_nvs_only_${DS}_nvs"        inference_single_gpu_ray_to_rgb_depth_eval.py     "$DS" ours_nvs_only_nvs
done
infer_one "ours_nvs_only_scannetpp_pose_depth" inference_single_gpu_rgb_to_ray_depth_aria_eval.py scannetpp ours_nvs_only_pose_depth

# ============ Eval (CPU, parallel) ============
EVAL_PIDS=()
for DS in dl3dv scannetpp re10k; do
  PD="$OUT/$DS/ours_nvs_only_pose_depth"
  if [ -d "$PD" ] && [ -n "$(ls -A "$PD" 2>/dev/null)" ]; then
    echo "[bg] start ${DS}_ours_nvs_only_pose_eval"
    stdbuf -oL mamba run -n test2 python -u scripts/eval_ray_mot_pose.py \
      --input_dir "$PD" --output_dir "$OUT/$DS/ours_nvs_only_pose_eval" \
      > "$LOGS/${DS}_ours_nvs_only_pose_eval.log" 2>&1 &
    EVAL_PIDS+=( $! )
    if [ "$DS" != "re10k" ]; then
      echo "[bg] start ${DS}_ours_nvs_only_depth_eval"
      stdbuf -oL mamba run -n test2 python -u scripts/eval_rgb_to_ray_depth_depth.py \
        --input_dir "$PD" --output_dir "$OUT/$DS/ours_nvs_only_depth_eval" --dataset "$DS" \
        > "$LOGS/${DS}_ours_nvs_only_depth_eval.log" 2>&1 &
      EVAL_PIDS+=( $! )
    fi
  fi
done
for DS in dl3dv re10k; do
  NVS="$OUT/$DS/ours_nvs_only_nvs"
  if [ -d "$NVS" ] && [ -n "$(ls -A "$NVS" 2>/dev/null)" ]; then
    echo "[bg] start ${DS}_ours_nvs_only_nvs_eval"
    stdbuf -oL mamba run -n test2 python -u scripts/recompute_metrics_offline.py \
      --input_dir "$NVS" --output_dir "$OUT/$DS/ours_nvs_only_nvs_eval" \
      > "$LOGS/${DS}_ours_nvs_only_nvs_eval.log" 2>&1 &
    EVAL_PIDS+=( $! )
  fi
done
for p in "${EVAL_PIDS[@]}"; do wait "$p"; done
echo "[bg] eval done"

python3 /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/collect_results.py >/dev/null
python3 /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/build_index.py >/dev/null
echo "[bg] ours_nvs_only chain done"
