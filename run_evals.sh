#!/bin/bash
# Master eval launcher. Runs from method dir; writes outputs into generative_baselines/eval_outputs/<dataset>/<method>/.
set -uo pipefail

METHOD_DIR=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new
BASELINE_DIR=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
OUT=${BASELINE_DIR}/eval_outputs
LOGS=${BASELINE_DIR}/eval_logs
mkdir -p "$LOGS"

export OURS_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-27/22-44-50/checkpoints/last.ckpt
export GEO4D_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
export DFOT_RE10K_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/diffusion-forcing-transformer/pretrained_models/DFoT_RE10K.ckpt
export RAYDIFFUSION_MODEL=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion
export GEN3C_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/GEN3C/checkpoints

MAX_SAMPLES=${MAX_SAMPLES:-10}
N_FRAMES=${N_FRAMES:-50}

cd "$METHOD_DIR"

run_log () {
  local name="$1"; shift
  local logf="$LOGS/${name}.log"
  echo "[launch] $name -> $logf"
  ( "$@" ) >"$logf" 2>&1
  echo "[done $?]  $name"
}
export -f run_log
export LOGS

# Datasets to run our method on. Format: <dataset> <task>
# task ∈ {pose_depth, nvs}
