#!/bin/bash -l
#SBATCH --job-name=v4-geo4d-zeroshot
#SBATCH --output=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs_v4_n50_baselines/slurm-%A_%a.out
#SBATCH --error=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs_v4_n50_baselines/slurm-%A_%a.err
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3

# Run Geo4D pose+depth on the 3 zeroshot datasets. The earlier retry array's
# Geo4D runs failed with FileNotFoundError on relative path 'checkpoints/geo4d/vae.ckpt';
# fixed by adding a symlink at $ROOT/checkpoints → Geo4D/checkpoints. This
# slurm only runs the Geo4D evals (skips ours/raydiff/chrono/nvs baselines via
# the script's standard skip-trigger logic).

set -uo pipefail
ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
cd $ROOT

DATASETS=(7scenes tum bonn)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

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

echo "[geo4d-zs task=$SLURM_ARRAY_TASK_ID dataset=$DATASET] start at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

chmod +x run_dataset_v4_n50.sh
./run_dataset_v4_n50.sh $DATASET 0 50 \
    > $LOGDIR/${DATASET}_geo4d.log 2>&1
RC=$?
echo "[geo4d-zs task=$SLURM_ARRAY_TASK_ID dataset=$DATASET] done rc=$RC at $(date)"
exit $RC
