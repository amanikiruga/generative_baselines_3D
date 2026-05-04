#!/bin/bash
# Retry failed baselines with corrected paths.
set -uo pipefail
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new

OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs
LOGS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs
DFOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/diffusion-forcing-transformer/pretrained/DFoT_RE10K.ckpt

# Wait until both chains are quiet (no eval scripts running)
echo "[retry] waiting for chains to end"
while pgrep -f "eval_dfot_nvs_re10k\|eval_gen3c_nvs\|eval_svc_nvs\|eval_geo4d\|eval_chronodepth\|eval_raydiffusion\|inference_single_gpu_flf2v_from_mp4_eval" >/dev/null 2>&1; do sleep 30; done
echo "[retry] chains idle, starting retries"

run() { local gpu=$1; shift; local name=$1; shift; echo "[retry] start $name on GPU $gpu"; CUDA_VISIBLE_DEVICES=$gpu stdbuf -oL mamba run -n test2 python -u "$@" > "$LOGS/${name}_retry.log" 2>&1; echo "[retry] $name exit=$?"; }

# DFoT NVS on dl3dv & re10k with correct ckpt path
run 0 dl3dv_dfot_nvs scripts/eval_dfot_nvs_re10k.py --input_dir "$OUT/dl3dv/ours_nvs" --output_dir "$OUT/dl3dv/dfot_nvs" --ckpt_path "$DFOT" --max_samples 10 &
PID0=$!
run 1 re10k_dfot_nvs scripts/eval_dfot_nvs_re10k.py --input_dir "$OUT/re10k/ours_nvs" --output_dir "$OUT/re10k/dfot_nvs" --ckpt_path "$DFOT" --max_samples 10 &
PID1=$!
wait $PID0; echo "[retry] dl3dv_dfot done"
wait $PID1; echo "[retry] re10k_dfot done"
echo "[retry] all retries done"
