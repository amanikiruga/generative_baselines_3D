#!/bin/bash
# Run all baselines on the existing ours_pose_depth/ and ours_nvs/ output dirs.
# Run after our method's outputs are available for a given (dataset, mode).
# Usage: bash run_baselines.sh <dataset>   # dataset ∈ {re10k, dl3dv, scannetpp}
set -uo pipefail
DS="${1:?usage: $0 <dataset>}"

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new

OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs
LOGS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs
GEO4D_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
DFOT_RE10K_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/diffusion-forcing-transformer/pretrained_models/DFoT_RE10K.ckpt
RAYDIFFUSION_MODEL=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion
GEN3C_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/GEN3C/checkpoints
N=10

PD_IN="$OUT/$DS/ours_pose_depth"
NVS_IN="$OUT/$DS/ours_nvs"

# Pose baselines
[ -d "$PD_IN" ] && {
  CUDA_VISIBLE_DEVICES=0 stdbuf -oL mamba run -n test2 python -u scripts/eval_geo4d_pose_2.py \
    --input_dir "$PD_IN" --output_dir "$OUT/$DS/geo4d_pose" \
    --ckpt_path "$GEO4D_CKPT" --max_samples $N --rerun \
    > "$LOGS/${DS}_geo4d_pose.log" 2>&1 &
  PID_GEO_POSE=$!

  CUDA_VISIBLE_DEVICES=1 stdbuf -oL mamba run -n test2 python -u scripts/eval_raydiffusion_pose.py \
    --input_dir "$PD_IN" --output_dir "$OUT/$DS/raydiffusion_pose" \
    --model_dir "$RAYDIFFUSION_MODEL" --max_samples $N --dataset re10k \
    > "$LOGS/${DS}_raydiffusion_pose.log" 2>&1 &
  PID_RAYDIFF=$!
  wait $PID_GEO_POSE; echo "geo4d_pose exit=$?"
  wait $PID_RAYDIFF;  echo "raydiff_pose exit=$?"
}

# Depth baselines (skip on re10k — no GT depth)
if [ "$DS" != "re10k" ] && [ -d "$PD_IN" ]; then
  CUDA_VISIBLE_DEVICES=0 stdbuf -oL mamba run -n test2 python -u scripts/eval_geo4d_depth.py \
    --input_dir "$PD_IN" --output_dir "$OUT/$DS/geo4d_depth" --dataset "$DS" \
    --ckpt_path "$GEO4D_CKPT" --max_samples $N --rerun \
    > "$LOGS/${DS}_geo4d_depth.log" 2>&1 &
  PID_GEO_DEPTH=$!

  CUDA_VISIBLE_DEVICES=1 stdbuf -oL mamba run -n test2 python -u scripts/eval_chronodepth_depth_aria.py \
    --input_dir "$PD_IN" --output_dir "$OUT/$DS/chronodepth_depth" --dataset "$DS" --max_samples $N \
    > "$LOGS/${DS}_chronodepth_depth.log" 2>&1 &
  PID_CHRONO=$!
  wait $PID_GEO_DEPTH; echo "geo4d_depth exit=$?"
  wait $PID_CHRONO;    echo "chronodepth exit=$?"
fi

# NVS baselines (skip scannetpp per spec)
if [ "$DS" != "scannetpp" ] && [ -d "$NVS_IN" ]; then
  CUDA_VISIBLE_DEVICES=0 stdbuf -oL mamba run -n test2 python -u scripts/eval_gen3c_nvs.py \
    --input_dir "$NVS_IN" --output_dir "$OUT/$DS/gen3c_nvs" \
    --ckpt_dir "$GEN3C_CKPT" --max_samples $N --dataset re10k \
    > "$LOGS/${DS}_gen3c_nvs.log" 2>&1 &
  PID_GEN3C=$!

  CUDA_VISIBLE_DEVICES=1 stdbuf -oL mamba run -n test2 python -u scripts/eval_dfot_nvs_re10k.py \
    --input_dir "$NVS_IN" --output_dir "$OUT/$DS/dfot_nvs" \
    --ckpt_path "$DFOT_RE10K_CKPT" --max_samples $N \
    > "$LOGS/${DS}_dfot_nvs.log" 2>&1 &
  PID_DFOT=$!
  wait $PID_GEN3C; echo "gen3c_nvs exit=$?"
  wait $PID_DFOT;  echo "dfot_nvs exit=$?"

  CUDA_VISIBLE_DEVICES=0 stdbuf -oL mamba run -n test2 python -u scripts/eval_svc_nvs.py \
    --input_dir "$NVS_IN" --output_dir "$OUT/$DS/seva_nvs" --max_samples $N \
    > "$LOGS/${DS}_seva_nvs.log" 2>&1 &
  PID_SEVA=$!

  CUDA_VISIBLE_DEVICES=1 stdbuf -oL mamba run -n test2 python -u scripts/inference_single_gpu_flf2v_from_mp4_eval.py \
    --input_dir "$NVS_IN" --output_dir "$OUT/$DS/wan_flf_nvs" \
    --max_samples $N --show_metrics --save_stats \
    > "$LOGS/${DS}_wan_flf.log" 2>&1 &
  PID_WAN=$!
  wait $PID_SEVA; echo "seva exit=$?"
  wait $PID_WAN;  echo "wan_flf exit=$?"
fi

# Always compute our method's own metrics on its outputs
if [ -d "$PD_IN" ]; then
  stdbuf -oL mamba run -n test2 python -u scripts/eval_ray_mot_pose.py \
    --input_dir "$PD_IN" --output_dir "$OUT/$DS/ours_pose_eval" \
    > "$LOGS/${DS}_ours_pose_eval.log" 2>&1
  if [ "$DS" != "re10k" ]; then
    stdbuf -oL mamba run -n test2 python -u scripts/eval_rgb_to_ray_depth_depth.py \
      --input_dir "$PD_IN" --output_dir "$OUT/$DS/ours_depth_eval" --dataset "$DS" \
      > "$LOGS/${DS}_ours_depth_eval.log" 2>&1
  fi
fi
if [ -d "$NVS_IN" ] && [ "$DS" != "scannetpp" ]; then
  stdbuf -oL mamba run -n test2 python -u scripts/recompute_metrics_offline.py \
    --input_dir "$NVS_IN" --output_dir "$OUT/$DS/ours_nvs_eval" \
    > "$LOGS/${DS}_ours_nvs_eval.log" 2>&1
fi

echo "[baselines $DS] all done"
