#!/bin/bash -l
#SBATCH --job-name=v4-n50-array
#SBATCH --output=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs_v4_n50_array/slurm-%A_%a.out
#SBATCH --error=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs_v4_n50_array/slurm-%A_%a.err
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_h100,kempner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-9%10

set -uo pipefail

ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
cd $ROOT

# 10 datasets, one per array task. Index → dataset.
DATASETS=(re10k dl3dv dl3dv_test tanksandtemples scannetpp vkitti2 aria scenenet_depth spatialvid_nvs agibot_world)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# Separate output tree so we don't clobber the in-flight foreground run.
export OUT=$ROOT/eval_outputs_v4_n50_array
# Reuse V3 baseline caches (Geo4D / RayDiffusion / DFoT / SEVA / Wan-FLF deltas
# are unchanged across V3↔V4 because metadata is unchanged).
export V2_OUT=$ROOT/eval_outputs_v3_n50

export METADATA_V4=$ROOT/metadata_v3_n50.json
export OURS_FINAL=/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model/outputs/2026-05-04/12-34-53/checkpoints/last.ckpt
export OURS_NVS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-29/14-31-15/checkpoints/last.ckpt
export GEO4D_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
export DFOT_RE10K_CKPT=$ROOT/diffusion-forcing-transformer/pretrained/DFoT_RE10K.ckpt
export RAYDIFF_DIR=$ROOT/RayDiffusion/models/co3d_diffusion

LOGDIR=$ROOT/eval_logs_v4_n50_array
mkdir -p $OUT $LOGDIR
# Tell run_dataset_v4_n50.sh to write per-method logs into the array tree, not
# the foreground tree (otherwise re10k/dl3dv array tasks would clobber the
# foreground run's logs in eval_logs_v4_n50/).
export LOGDIR_V4=$LOGDIR

# Inside the slurm job the allocated GPU is exposed as device 0 (slurm sets
# CUDA_VISIBLE_DEVICES already). run_dataset_v4_n50.sh re-exports it from arg
# $2; passing 0 is what we want here.
echo "[array task=$SLURM_ARRAY_TASK_ID dataset=$DATASET] start at $(date)"
echo "[array task=$SLURM_ARRAY_TASK_ID] node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

chmod +x run_dataset_v4_n50.sh
./run_dataset_v4_n50.sh $DATASET 0 50 \
    > $LOGDIR/${DATASET}.log 2>&1
RC=$?
echo "[array task=$SLURM_ARRAY_TASK_ID dataset=$DATASET] done rc=$RC at $(date)"
exit $RC
