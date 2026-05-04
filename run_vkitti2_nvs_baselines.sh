#!/usr/bin/env bash
# Run NVS baselines (SEVA + Wan-FLF) for vkitti2 n=50, plus the dropped
# wave-2 baselines for scenenet_depth + agibot_world. Free GPUs 0 + 1.
set -uo pipefail
ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
WMD=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new
OUT=$ROOT/eval_outputs_v3_n50
LOGS=$ROOT/eval_logs_v3_n50
mkdir -p $LOGS
M="mamba run -n test2 python"
cd $WMD

run_seva()  { local DS=$1; local GPU=$2; CUDA_VISIBLE_DEVICES=$GPU $M scripts/eval_svc_nvs.py --input_dir $OUT/$DS/ours_nvs --output_dir $OUT/$DS/seva_nvs --max_samples 50 --dataset re10k > $LOGS/${DS}_seva_nvs.log 2>&1; }
run_wan()   { local DS=$1; local GPU=$2; CUDA_VISIBLE_DEVICES=$GPU $M scripts/inference_single_gpu_flf2v_from_mp4_eval.py --input_dir $OUT/$DS/ours_nvs --output_dir $OUT/$DS/wan_flf_nvs --max_samples 50 --show_metrics --save_stats > $LOGS/${DS}_wan_flf_nvs.log 2>&1; }

# vkitti2 — was gated off in run_dataset_v3_n50.sh; running fresh here.
echo "[$(date)] vkitti2 SEVA on GPU 0 / Wan-FLF on GPU 1"
run_seva vkitti2 0 &
run_wan  vkitti2 1 &
wait

# Aggregate
$M $ROOT/eval_depth_disp_offline_v3.py --root eval_outputs_v3_n50
$M $ROOT/colorize_depth_v3.py
$M $ROOT/build_index_v3.py
$M $ROOT/build_index_v3_n50.py
$M $ROOT/collect_results_v3.py
$M $ROOT/collect_results_v3_n50.py
echo "[$(date)] DONE"
