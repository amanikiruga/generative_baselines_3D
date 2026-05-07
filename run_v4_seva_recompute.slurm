#!/bin/bash -l
#SBATCH --job-name=v4-seva-recompute
#SBATCH --output=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs_v4_n50_baselines/slurm-%A_%a.out
#SBATCH --error=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs_v4_n50_baselines/slurm-%A_%a.err
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-3

# 4 GPUs total. 11 NVS-eligible datasets are split into 4 buckets and run
# sequentially within each task. Bucket sizes balance long-tail Wan-not-here
# (we only run SEVA for the wiped cell — other baselines skip via
# final_stats.json triggers). Bucket assignment chosen so partial-frame
# datasets (7scenes, tum, agibot, vkitti2) are spread across tasks.

set -uo pipefail
ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
cd $ROOT

case $SLURM_ARRAY_TASK_ID in
    0) DATASETS=(re10k dl3dv dl3dv_test) ;;
    1) DATASETS=(tanksandtemples scannetpp aria) ;;
    2) DATASETS=(vkitti2 spatialvid_nvs agibot_world) ;;
    3) DATASETS=(7scenes tum) ;;
    *) echo "bad array idx"; exit 2 ;;
esac

export OUT=$ROOT/eval_outputs_v4_n50
export V2_OUT=$ROOT/eval_outputs_v3_n50
export METADATA_V4=$ROOT/metadata_v3_n50.json
export OURS_FINAL=/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model/outputs/2026-05-04/12-34-53/checkpoints/last.ckpt
export OURS_NVS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-29/14-31-15/checkpoints/last.ckpt
export GEO4D_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
export DFOT_RE10K_CKPT=$ROOT/diffusion-forcing-transformer/pretrained/DFoT_RE10K.ckpt
export RAYDIFF_DIR=$ROOT/RayDiffusion/models/co3d_diffusion

LOGDIR=$ROOT/eval_logs_v4_n50_baselines
mkdir -p $OUT $LOGDIR
export LOGDIR_V4=$LOGDIR
export SERIAL_OURS=1

echo "[seva-recompute task=$SLURM_ARRAY_TASK_ID datasets=${DATASETS[*]}] start at $(date)"
echo "[seva-recompute task=$SLURM_ARRAY_TASK_ID] node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

chmod +x run_dataset_v4_n50.sh
RC=0
for DATASET in "${DATASETS[@]}"; do
    echo "[seva-recompute task=$SLURM_ARRAY_TASK_ID] >>> $DATASET start at $(date)"
    ./run_dataset_v4_n50.sh $DATASET 0 50 \
        > $LOGDIR/${DATASET}_seva_recompute.log 2>&1
    SUB_RC=$?
    echo "[seva-recompute task=$SLURM_ARRAY_TASK_ID] <<< $DATASET rc=$SUB_RC at $(date)"
    [ $SUB_RC -ne 0 ] && RC=$SUB_RC
done

echo "[seva-recompute task=$SLURM_ARRAY_TASK_ID] all done rc=$RC at $(date)"
exit $RC
