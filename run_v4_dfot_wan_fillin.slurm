#!/bin/bash -l
#SBATCH --job-name=v4-dfot-wan-fillin
#SBATCH --output=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs_v4_n50_baselines/slurm-%A_%a.out
#SBATCH --error=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs_v4_n50_baselines/slurm-%A_%a.err
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-8
#
# 9 tasks, one (model, dataset) per GPU. No co-location of models on a GPU.
#
# DFOT (2 tasks) — datasets where v4 only has v2's stale 10-sample cache:
#   0: dfot   re10k        (target 50)
#   1: dfot   aria         (target 20)
#
# WAN-FLF (7 tasks) — datasets where v4 cell is partial or v2-stale:
#   2: wan    re10k        (10 → 50)
#   3: wan    dl3dv        (10 → 50)
#   4: wan    dl3dv_test   (10 → 50)
#   5: wan    aria         (10 → 20)
#   6: wan    vkitti2      (22 → 50)
#   7: wan    spatialvid_nvs (10 → 50)
#   8: wan    7scenes      (39 → 50)
#
# Each task wipes its target cell (FG + V3) so nvs_link doesn't restore a
# stale symlink, then runs only its own model.

set -uo pipefail
ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
WMD=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new
cd $ROOT

# (model, dataset, target_n, dflag) per array index
case $SLURM_ARRAY_TASK_ID in
    0) MODEL=dfot; DATASET=re10k;          MAXN=50; DFLAG=re10k ;;
    1) MODEL=dfot; DATASET=aria;           MAXN=20; DFLAG=aria  ;;
    2) MODEL=wan;  DATASET=re10k;          MAXN=50; DFLAG=re10k ;;
    3) MODEL=wan;  DATASET=dl3dv;          MAXN=50; DFLAG=re10k ;;
    4) MODEL=wan;  DATASET=dl3dv_test;     MAXN=50; DFLAG=re10k ;;
    5) MODEL=wan;  DATASET=aria;           MAXN=20; DFLAG=aria  ;;
    6) MODEL=wan;  DATASET=vkitti2;        MAXN=50; DFLAG=re10k ;;
    7) MODEL=wan;  DATASET=spatialvid_nvs; MAXN=50; DFLAG=re10k ;;
    8) MODEL=wan;  DATASET=7scenes;        MAXN=50; DFLAG=re10k ;;
    *) echo "bad array idx $SLURM_ARRAY_TASK_ID"; exit 2 ;;
esac

OUT=$ROOT/eval_outputs_v4_n50
V2_OUT=$ROOT/eval_outputs_v3_n50
DFOT_RE10K_CKPT=$ROOT/diffusion-forcing-transformer/pretrained/DFoT_RE10K.ckpt

if [ "$MODEL" = "dfot" ]; then
    CELL=dfot_nvs
elif [ "$MODEL" = "wan" ]; then
    CELL=wan_flf_nvs
fi

LOGDIR=$ROOT/eval_logs_v4_n50_baselines
mkdir -p $LOGDIR
LOG=$LOGDIR/${DATASET}_${MODEL}_fillin.log

echo "[$SLURM_ARRAY_TASK_ID $MODEL $DATASET MAXN=$MAXN] start at $(date)" | tee $LOG
echo "node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"          | tee -a $LOG
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | tee -a $LOG

# Wipe stale FG + V3 cells so a v2 fallback can't sneak in
rm -rf $OUT/$DATASET/$CELL $V2_OUT/$DATASET/$CELL

IN_NVS=$OUT/$DATASET/ours_final_nvs
if [ ! -d "$IN_NVS" ]; then
    echo "ERROR: input dir missing: $IN_NVS" | tee -a $LOG
    exit 3
fi

if [ "$MODEL" = "dfot" ]; then
    mamba run -n test2 python $WMD/scripts/eval_dfot_nvs_re10k.py \
        --input_dir  $IN_NVS \
        --output_dir $OUT/$DATASET/dfot_nvs \
        --ckpt_path  $DFOT_RE10K_CKPT \
        --max_samples $MAXN \
        --dataset $DFLAG \
        2>&1 | tee -a $LOG
elif [ "$MODEL" = "wan" ]; then
    mamba run -n test2 python $WMD/scripts/inference_single_gpu_flf2v_from_mp4_eval.py \
        --input_dir  $IN_NVS \
        --output_dir $OUT/$DATASET/wan_flf_nvs \
        --max_samples $MAXN \
        --show_metrics --save_stats \
        2>&1 | tee -a $LOG
fi
RC=${PIPESTATUS[0]}

echo "[$SLURM_ARRAY_TASK_ID $MODEL $DATASET] done rc=$RC at $(date)" | tee -a $LOG
exit $RC
