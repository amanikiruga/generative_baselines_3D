#!/bin/bash -l
#SBATCH --job-name=v4-seva-finish
#SBATCH --output=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs_v4_n50_baselines/slurm-%j.out
#SBATCH --error=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs_v4_n50_baselines/slurm-%j.err
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00

# Finish leftover seva cells: dl3dv_test (50) and agibot_world (2).
# Both walltime'd or didn't start in 10786311. dfot stubs already in place
# from earlier; pose_depth skip-trigger fine for these (dl3dv_test has 50,
# agibot has 2 → use MAXN=2 for agibot). Single GPU, sequential.

set -uo pipefail
ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
cd $ROOT

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

echo "[seva-finish] start at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

chmod +x run_dataset_v4_n50.sh
RC=0
for spec in "dl3dv_test 50" "agibot_world 2"; do
    DATASET=${spec% *}
    MAXN=${spec##* }
    echo "[seva-finish] >>> $DATASET (MAXN=$MAXN) start at $(date)"
    ./run_dataset_v4_n50.sh $DATASET 0 $MAXN > $LOGDIR/${DATASET}_seva_finish.log 2>&1
    SUB_RC=$?
    echo "[seva-finish] <<< $DATASET rc=$SUB_RC at $(date)"
    [ $SUB_RC -ne 0 ] && RC=$SUB_RC
done

echo "[seva-finish] all done rc=$RC at $(date)"
exit $RC
