#!/bin/bash
# Phased orchestrator. Picks up after the in-flight scenenet pair.
# Phase 1: wait for scenenet pair → run vkitti2 ours pose_depth (alone)
# Phase 2: scenenet+vkitti2 evals (CPU, parallel)
# Phase 3: scenenet+vkitti2 baselines (geo4d_depth, chronodepth)
# Phase 4: aria pair pose_depth (max 2 parallel)
# Phase 5: aria evals + aria baselines
# Final: collect_results.py + build_index.py
set -uo pipefail
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new

PY=/net/holy-isilon/ifs/rc_labs/ydu_lab/akiruga/.conda/envs/test2/bin/python
CKPT_OURS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-27/22-44-50/checkpoints/last.ckpt
CKPT_NVS_ONLY=/net/holy-isilon/ifs/rc_labs/ydu_lab/zhiyi24/workspace/video_world_model/outputs/2026-04-26/07-31-50/checkpoints/last.ckpt
GEO4D=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs
LOGS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs
N=10

wait_for_pids() {
  for pid in "$@"; do
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
    echo "[ph] PID $pid finished"
  done
}

# ----- Phase 1: wait for in-flight scenenet pair, then vkitti2 ours pose_depth -----
echo "[ph] === Phase 1: wait for scenenet pair (3810432/3810433), then vkitti2 ours ==="
wait_for_pids 3810432 3810433

echo "[ph] vkitti2 ours pose_depth (alone)"
CUDA_VISIBLE_DEVICES=0 stdbuf -oL "$PY" -u scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
  --ckpt_path "$CKPT_OURS" --dataset vkitti2 --output_dir "$OUT/vkitti2/ours_pose_depth" \
  --max_samples $N --seed 42 --no_augmentations \
  > "$LOGS/queue_ours_vkitti2_pose_depth.log" 2>&1
echo "[ph] vkitti2 ours pose_depth exit=$?"

# ----- Phase 2: scenenet + vkitti2 evals (CPU, parallel) -----
echo "[ph] === Phase 2: scenenet + vkitti2 evals ==="
EPIDS=()
for DS in scenenet_depth vkitti2; do
  for METHOD in ours ours_nvs_only; do
    PD="$OUT/$DS/${METHOD}_pose_depth"
    [ -d "$PD" ] && [ -n "$(ls -A "$PD" 2>/dev/null)" ] || continue
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
  done
done
for p in "${EPIDS[@]}"; do wait "$p"; done
echo "[ph] Phase 2 done"

# ----- Phase 3: scenenet + vkitti2 baselines (geo4d_depth, chronodepth) -----
# These need GPU but small footprint. Run serially on GPU 0.
echo "[ph] === Phase 3: scenenet + vkitti2 baselines ==="
for DS in scenenet_depth vkitti2; do
  PD="$OUT/$DS/ours_pose_depth"
  [ -d "$PD" ] && [ -n "$(ls -A "$PD" 2>/dev/null)" ] || continue
  CUDA_VISIBLE_DEVICES=0 stdbuf -oL "$PY" -u scripts/eval_geo4d_depth.py \
    --input_dir "$PD" --output_dir "$OUT/$DS/geo4d_depth" \
    --ckpt_path "$GEO4D" --dataset "$DS" --max_samples $N --rerun \
    > "$LOGS/queue_${DS}_geo4d_depth.log" 2>&1
  echo "[ph] $DS geo4d_depth exit=$?"
  CUDA_VISIBLE_DEVICES=0 stdbuf -oL "$PY" -u scripts/eval_chronodepth_depth_aria.py \
    --input_dir "$PD" --output_dir "$OUT/$DS/chronodepth_depth" --dataset "$DS" --max_samples $N --rerun \
    > "$LOGS/queue_${DS}_chronodepth_depth.log" 2>&1
  echo "[ph] $DS chronodepth exit=$?"
done

# ----- Phase 4: aria pair (max 2 parallel) -----
echo "[ph] === Phase 4: aria ours + ours_nvs_only pose_depth ==="
CUDA_VISIBLE_DEVICES=0 stdbuf -oL "$PY" -u scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
  --ckpt_path "$CKPT_OURS" --dataset aria --output_dir "$OUT/aria/ours_pose_depth" \
  --max_samples $N --seed 42 --no_augmentations \
  > "$LOGS/queue_ours_aria_pose_depth.log" 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=0 stdbuf -oL "$PY" -u scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
  --ckpt_path "$CKPT_NVS_ONLY" --dataset aria --output_dir "$OUT/aria/ours_nvs_only_pose_depth" \
  --max_samples $N --seed 42 --no_augmentations \
  > "$LOGS/queue_ours_nvs_only_aria_pose_depth.log" 2>&1 &
P2=$!
wait "$P1"; echo "[ph] aria ours exit=$?"
wait "$P2"; echo "[ph] aria ours_nvs_only exit=$?"

# ----- Phase 5: aria evals + aria baselines -----
echo "[ph] === Phase 5: aria evals + baselines ==="
EPIDS=()
for METHOD in ours ours_nvs_only; do
  PD="$OUT/aria/${METHOD}_pose_depth"
  [ -d "$PD" ] && [ -n "$(ls -A "$PD" 2>/dev/null)" ] || continue
  stdbuf -oL "$PY" -u scripts/eval_rgb_to_ray_depth_depth.py \
    --input_dir "$PD" --output_dir "$OUT/aria/${METHOD}_depth_eval" --dataset aria \
    > "$LOGS/queue_aria_${METHOD}_depth_eval.log" 2>&1 &
  EPIDS+=( $! )
  stdbuf -oL "$PY" -u scripts/eval_ray_mot_pose.py \
    --input_dir "$PD" --output_dir "$OUT/aria/${METHOD}_pose_eval" \
    > "$LOGS/queue_aria_${METHOD}_pose_eval.log" 2>&1 &
  EPIDS+=( $! )
done
for p in "${EPIDS[@]}"; do wait "$p"; done

PD="$OUT/aria/ours_pose_depth"
if [ -d "$PD" ] && [ -n "$(ls -A "$PD" 2>/dev/null)" ]; then
  CUDA_VISIBLE_DEVICES=0 stdbuf -oL "$PY" -u scripts/eval_geo4d_depth.py \
    --input_dir "$PD" --output_dir "$OUT/aria/geo4d_depth" \
    --ckpt_path "$GEO4D" --dataset aria --max_samples $N --rerun \
    > "$LOGS/queue_aria_geo4d_depth.log" 2>&1
  echo "[ph] aria geo4d_depth exit=$?"
  CUDA_VISIBLE_DEVICES=0 stdbuf -oL "$PY" -u scripts/eval_chronodepth_depth_aria.py \
    --input_dir "$PD" --output_dir "$OUT/aria/chronodepth_depth" --dataset aria --max_samples $N --rerun \
    > "$LOGS/queue_aria_chronodepth_depth.log" 2>&1
  echo "[ph] aria chronodepth exit=$?"
fi

# ----- Final: collect + index -----
"$PY" /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/collect_results.py >/dev/null
"$PY" /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/build_index.py    >/dev/null
echo "[ph] DONE"
