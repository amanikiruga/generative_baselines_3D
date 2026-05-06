#!/bin/bash -l
#SBATCH --job-name=v4-n50-baselines
#SBATCH --output=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs_v4_n50_baselines/slurm-%A_%a.out
#SBATCH --error=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs_v4_n50_baselines/slurm-%A_%a.err
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-5%6

# Fill-in pass: NVS-eligible datasets that didn't have all 3 NVS baselines
# (DFoT, SEVA, Wan-FLF) and depth datasets that didn't have all depth
# baselines run earlier. Pre-populate inherits every existing cell from FG
# tree, so ours/pose/depth/eval all skip and only the missing baselines run.

set -uo pipefail
ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
cd $ROOT

# Datasets with NVS baselines that need DFoT/SEVA/Wan added:
#   - vkitti2 (DFoT, SEVA, Wan-FLF — all missing in V3+V4)
#   - 7scenes, tum (DFoT, SEVA, Wan-FLF — never enabled)
#   - scannetpp, spatialvid_nvs, agibot_world (DFoT missing)
# dl3dv, dl3dv_test, tanksandtemples — DFoT done in retry array.
# Plus aria already had DFoT before; SEVA, Wan-FLF were V3-cached.
DATASETS=(vkitti2 7scenes tum scannetpp spatialvid_nvs agibot_world)
# Missing baseline cells per dataset:
#   vkitti2:        dfot_nvs, seva_nvs, wan_flf_nvs (3)
#   7scenes:        dfot_nvs, seva_nvs, wan_flf_nvs (3)
#   tum:            dfot_nvs, seva_nvs, wan_flf_nvs (3)
#   scannetpp:      dfot_nvs (1)
#   spatialvid_nvs: dfot_nvs (1)
#   agibot_world:   dfot_nvs (1)
# Existing cells (ours, pose, depth, evals, the other baselines) skip via
# LAST_SAMPLE / final_stats.json triggers. Only the missing baselines run.
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

echo "[fillin task=$SLURM_ARRAY_TASK_ID dataset=$DATASET] start at $(date)"
echo "[fillin task=$SLURM_ARRAY_TASK_ID] node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Existing FG cells stay as-is (script's skip-trigger uses LAST_SAMPLE for
# pose_depth and final_stats.json for nvs/baselines, all of which are present
# for the existing-completed cells). Only missing baseline cells will run.

chmod +x run_dataset_v4_n50.sh
./run_dataset_v4_n50.sh $DATASET 0 50 \
    > $LOGDIR/${DATASET}.log 2>&1
RC=$?
echo "[fillin task=$SLURM_ARRAY_TASK_ID dataset=$DATASET] done rc=$RC at $(date)"
exit $RC
