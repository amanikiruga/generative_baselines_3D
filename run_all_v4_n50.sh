#!/usr/bin/env bash
# v4 launcher — 3 ckpts × baselines, n=10 (default) or n=50 scenes per dataset.
#
# Hardware: GPUs 0,1,2,3 (H200/H100). 2 datasets per GPU concurrent → 8 datasets
# in parallel per wave. Within each dataset, the 3 ours ckpts run serially on
# the same GPU, then baselines.
#
# Usage:
#   ./run_all_v4_n50.sh                # n=10 default
#   ./run_all_v4_n50.sh --max_samples 50

set -uo pipefail

MAXN=50
while [ $# -gt 0 ]; do
    case $1 in
        --max_samples) MAXN=$2; shift 2 ;;
        *) echo "unknown arg $1"; exit 1 ;;
    esac
done

# Checkpoints
export OURS_FINAL=/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model/outputs/2026-05-04/12-34-53/checkpoints/last.ckpt
export OURS_NVS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-29/14-31-15/checkpoints/last.ckpt
export OURS_FINAL_OLD=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-05-03/20-03-01/checkpoints/last.ckpt

export GEO4D_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
export DFOT_RE10K_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/diffusion-forcing-transformer/pretrained/DFoT_RE10K.ckpt
export RAYDIFF_DIR=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion

export METADATA_V4=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/metadata_v3_n50.json
export OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs_v4_n50
export V2_OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs_v3_n50

ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
LOGDIR=$ROOT/eval_logs_v4_n50
mkdir -p $OUT $LOGDIR

cd $ROOT
chmod +x run_dataset_v4_n50.sh

echo "[run_all_v4_n50] OURS_FINAL=$OURS_FINAL"
echo "[run_all_v4_n50] OURS_NVS=$OURS_NVS"
echo "[run_all_v4_n50] OURS_FINAL_OLD=$OURS_FINAL_OLD"
echo "[run_all_v4_n50] MAXN=$MAXN"

# 2 H200 GPUs. 1 dataset per GPU, but the 3 ours ckpts run concurrently on the
# same GPU per dataset (parallelism is inside run_dataset_v4_n50.sh). 5 waves
# cover all 10 datasets.
launch_wave() {
    local A=$1; local B=$2
    echo "[run_all_v4_n50] wave starting ($A on gpu0, $B on gpu1) at $(date)"
    ./run_dataset_v4_n50.sh $A 0 $MAXN > $LOGDIR/$A.log 2>&1 &
    ./run_dataset_v4_n50.sh $B 1 $MAXN > $LOGDIR/$B.log 2>&1 &
    wait
    echo "[run_all_v4_n50] wave done ($A, $B) at $(date)"
}
launch_wave_solo() {
    local A=$1
    echo "[run_all_v4_n50] wave starting ($A on gpu0) at $(date)"
    ./run_dataset_v4_n50.sh $A 0 $MAXN > $LOGDIR/$A.log 2>&1 &
    wait
    echo "[run_all_v4_n50] wave done ($A) at $(date)"
}
# Skipped initially due to bad metadata check; queued as follow-on:
# vkitti2, tanksandtemples, spatialvid_nvs — see run_all_v4_n50_followon.sh.
launch_wave re10k           dl3dv
launch_wave aria            dl3dv_test
launch_wave scannetpp       scenenet_depth
launch_wave_solo agibot_world

mamba run -n test2 python eval_depth_disp_offline_v3.py --root eval_outputs_v4_n50
mamba run -n test2 python colorize_depth_v3.py
mamba run -n test2 python collect_results_v4_n50.py
mamba run -n test2 python build_index_v4_n50.py
echo "[run_all_v4_n50] ALL DONE at $(date)"
