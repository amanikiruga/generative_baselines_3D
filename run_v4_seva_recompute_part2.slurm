#!/bin/bash -l
#SBATCH --job-name=v4-seva-pt2
#SBATCH --output=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs_v4_n50_baselines/slurm-%A_%a.out
#SBATCH --error=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs_v4_n50_baselines/slurm-%A_%a.err
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-1

# Resubmit of the two seva-recompute buckets that were wasting time on
# baseline reruns. Per-dataset MAXN matches actual sample count to avoid
# the pose_depth skip-trigger re-running ours_final/ours_nvs inference.
# Missing dfot final_stats.json have been stubbed.

set -uo pipefail
ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
cd $ROOT

case $SLURM_ARRAY_TASK_ID in
    # task 1: tanksandtemples(5) → scannetpp(7) → aria(20)
    0) DATASETS=(tanksandtemples scannetpp aria); MAXN_LIST=(5 7 20) ;;
    # task 3: 7scenes(50) → tum(20)
    1) DATASETS=(7scenes tum);                    MAXN_LIST=(50 20) ;;
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

echo "[seva-pt2 task=$SLURM_ARRAY_TASK_ID datasets=${DATASETS[*]} maxn=${MAXN_LIST[*]}] start at $(date)"
echo "[seva-pt2 task=$SLURM_ARRAY_TASK_ID] node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

chmod +x run_dataset_v4_n50.sh
RC=0
for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[$i]}
    MAXN=${MAXN_LIST[$i]}
    echo "[seva-pt2 task=$SLURM_ARRAY_TASK_ID] >>> $DATASET (MAXN=$MAXN) start at $(date)"
    ./run_dataset_v4_n50.sh $DATASET 0 $MAXN \
        > $LOGDIR/${DATASET}_seva_pt2.log 2>&1
    SUB_RC=$?
    echo "[seva-pt2 task=$SLURM_ARRAY_TASK_ID] <<< $DATASET rc=$SUB_RC at $(date)"
    [ $SUB_RC -ne 0 ] && RC=$SUB_RC
done

echo "[seva-pt2 task=$SLURM_ARRAY_TASK_ID] all done rc=$RC at $(date)"
exit $RC
