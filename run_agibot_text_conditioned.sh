#!/usr/bin/env bash
# Re-run agibot_world n=50 NVS branches with full text conditioning:
#   - Ours: NEW first-frame-only inference (no last-frame anchor) + text
#           via inference_single_gpu_ray_to_rgb_depth_first_eval.py
#   - Wan-FLF: NEW text-conditioned variant that reads sample/prompt.txt
#           via inference_single_gpu_flf2v_from_mp4_text_eval.py
#
# Outputs:
#   eval_outputs_v3_n50/agibot_world/ours_nvs_first/    (Ours, first-frame-only)
#   eval_outputs_v3_n50/agibot_world/wan_flf_nvs_text/  (Wan-FLF with prompt)
#
# Existing dirs (ours_nvs from the firstlast script, wan_flf_nvs no-text) are
# left untouched.
#
# Waits for the running vkitti2 SEVA/Wan jobs to release GPUs 0 + 1.
set -uo pipefail

ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
WMD=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new
OUT=$ROOT/eval_outputs_v3_n50
LOGS=$ROOT/eval_logs_v3_n50
mkdir -p $LOGS
M="mamba run -n test2 python"

CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-05-03/20-03-01/last.ckpt

echo "[$(date)] waiting for GPUs 0 + 1 to free up (vkitti2 SEVA/Wan)…"
while pgrep -f "run_vkitti2_nvs_baselines" >/dev/null 2>&1; do sleep 30; done
echo "[$(date)] GPUs free; launching agibot_world text-conditioned runs"

cd $WMD

# Ours — first-frame + text only, GPU 0
CUDA_VISIBLE_DEVICES=0 $M scripts/inference_single_gpu_ray_to_rgb_depth_first_eval.py \
    --ckpt_path "$CKPT" --dataset agibot_world \
    --output_dir $OUT/agibot_world/ours_nvs_first \
    --max_samples 50 --seed 42 --no_augmentations \
    > $LOGS/agibot_ours_nvs_first.log 2>&1 &
PID_OURS=$!

# Wan-FLF — text-conditioned, GPU 1, reads prompt.txt from ours_pose_depth
CUDA_VISIBLE_DEVICES=1 $M scripts/inference_single_gpu_flf2v_from_mp4_text_eval.py \
    --input_dir $OUT/agibot_world/ours_pose_depth \
    --output_dir $OUT/agibot_world/wan_flf_nvs_text \
    --max_samples 50 --show_metrics --save_stats \
    > $LOGS/agibot_wan_flf_nvs_text.log 2>&1 &
PID_WAN=$!

wait $PID_OURS; echo "[$(date)] ours_nvs_first exit=$?"
wait $PID_WAN;  echo "[$(date)] wan_flf_nvs_text exit=$?"

# Refresh aggregator/index so the new dirs appear (they need to be wired in
# build_index/collect for the table to show them — see follow-up edits).
$M $ROOT/eval_depth_disp_offline_v3.py --root eval_outputs_v3_n50
$M $ROOT/colorize_depth_v3.py
$M $ROOT/build_index_v3.py
$M $ROOT/build_index_v3_n50.py
$M $ROOT/collect_results_v3.py
$M $ROOT/collect_results_v3_n50.py
echo "[$(date)] DONE"
