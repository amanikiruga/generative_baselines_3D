#!/bin/bash
# Re-run depth eval on existing DL3DV + ScanNet++ NPZs with the new gates.
# CPU-only methods (ours, ours_nvs_only, chronodepth-cached) run in parallel.
# Geo4D follows serially (its model loads even when using cached preds).
set -uo pipefail
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new

OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs
LOGS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs
GEO4D=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt

PIDS=()
for DS in dl3dv scannetpp; do
  for METHOD in ours ours_nvs_only; do
    PD="$OUT/$DS/${METHOD}_pose_depth"
    EVAL_OUT="$OUT/$DS/${METHOD}_depth_eval"
    [ -d "$PD" ] || continue
    echo "[redo] ${DS} ${METHOD}_depth_eval"
    stdbuf -oL mamba run -n test2 python -u scripts/eval_rgb_to_ray_depth_depth.py \
      --input_dir "$PD" --output_dir "$EVAL_OUT" --dataset "$DS" --max_samples 10 \
      > "$LOGS/redo_${DS}_${METHOD}_depth_eval.log" 2>&1 &
    PIDS+=( $! )
  done
  CD_OUT="$OUT/$DS/chronodepth_depth"
  if [ -d "$CD_OUT" ]; then
    echo "[redo] ${DS} chronodepth_depth (cached)"
    stdbuf -oL mamba run -n test2 python -u scripts/eval_chronodepth_depth_aria.py \
      --input_dir "$OUT/$DS/ours_pose_depth" --output_dir "$CD_OUT" --dataset "$DS" --max_samples 10 \
      > "$LOGS/redo_${DS}_chronodepth_depth.log" 2>&1 &
    PIDS+=( $! )
  fi
done
for p in "${PIDS[@]}"; do wait "$p"; done
echo "[redo] CPU evals done — starting geo4d (GPU)"

for DS in dl3dv scannetpp; do
  G_OUT="$OUT/$DS/geo4d_depth"
  if [ -d "$G_OUT" ]; then
    echo "[redo] ${DS} geo4d_depth (cached preds, model loads)"
    CUDA_VISIBLE_DEVICES=0 stdbuf -oL mamba run -n test2 python -u scripts/eval_geo4d_depth.py \
      --input_dir "$OUT/$DS/ours_pose_depth" --output_dir "$G_OUT" --dataset "$DS" \
      --ckpt_path "$GEO4D" --max_samples 10 \
      > "$LOGS/redo_${DS}_geo4d_depth.log" 2>&1
  fi
done

python3 /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/collect_results.py >/dev/null
python3 /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/build_index.py    >/dev/null
echo "[redo] DONE"
