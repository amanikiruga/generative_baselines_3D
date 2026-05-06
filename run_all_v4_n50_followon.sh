#!/usr/bin/env bash
# Follow-on for run_all_v4_n50.sh: covers vkitti2, tanksandtemples,
# spatialvid_nvs which were missed in the original launch (metadata jsons were
# present after all). Launch this AFTER run_all_v4_n50.sh exits.
set -uo pipefail

MAXN=50
while [ $# -gt 0 ]; do
    case $1 in
        --max_samples) MAXN=$2; shift 2 ;;
        *) echo "unknown arg $1"; exit 1 ;;
    esac
done

export OURS_FINAL=/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model/outputs/2026-05-04/12-34-53/checkpoints/last.ckpt
export OURS_NVS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-29/14-31-15/checkpoints/last.ckpt
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

echo "[followon] OURS_FINAL=$OURS_FINAL"
echo "[followon] OURS_NVS=$OURS_NVS"
echo "[followon] MAXN=$MAXN"

launch_wave() {
    local A=$1; local B=$2
    echo "[followon] wave starting ($A on gpu0, $B on gpu1) at $(date)"
    ./run_dataset_v4_n50.sh $A 0 $MAXN > $LOGDIR/$A.log 2>&1 &
    ./run_dataset_v4_n50.sh $B 1 $MAXN > $LOGDIR/$B.log 2>&1 &
    wait
    echo "[followon] wave done ($A, $B) at $(date)"
}
launch_wave_solo() {
    local A=$1
    echo "[followon] wave starting ($A on gpu0) at $(date)"
    ./run_dataset_v4_n50.sh $A 0 $MAXN > $LOGDIR/$A.log 2>&1 &
    wait
    echo "[followon] wave done ($A) at $(date)"
}

launch_wave vkitti2 tanksandtemples
launch_wave_solo spatialvid_nvs

mamba run -n test2 python eval_depth_disp_offline_v3.py --root eval_outputs_v4_n50
mamba run -n test2 python colorize_depth_v3.py
mamba run -n test2 python collect_results_v4_n50.py
mamba run -n test2 python build_index_v4_n50.py
echo "[followon] ALL DONE at $(date)"
