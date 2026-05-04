#!/usr/bin/env bash
set -uo pipefail

export OURS_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-27/22-44-50/checkpoints/last.ckpt
export NVS_ONLY_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-29/14-31-15/checkpoints/last.ckpt
export GEO4D_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
export DFOT_RE10K_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/diffusion-forcing-transformer/pretrained/DFoT_RE10K.ckpt
export GEN3C_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/GEN3C/checkpoints
export RAYDIFF_DIR=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion
export OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs

ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
LOGDIR=$ROOT/eval_logs
mkdir -p $OUT $LOGDIR

cd $ROOT
chmod +x run_dataset.sh

# 8 datasets, 4 GPUs → 2 datasets per GPU (4 process slots / GPU = plenty of VRAM)
./run_dataset.sh re10k           0 > $LOGDIR/re10k.log           2>&1 &
./run_dataset.sh aria            0 > $LOGDIR/aria.log            2>&1 &
./run_dataset.sh dl3dv           1 > $LOGDIR/dl3dv.log           2>&1 &
./run_dataset.sh vkitti2         1 > $LOGDIR/vkitti2.log         2>&1 &
./run_dataset.sh dl3dv_test      2 > $LOGDIR/dl3dv_test.log      2>&1 &
./run_dataset.sh scannetpp       2 > $LOGDIR/scannetpp.log       2>&1 &
./run_dataset.sh tanksandtemples 3 > $LOGDIR/tanksandtemples.log 2>&1 &
./run_dataset.sh scenenet_depth  3 > $LOGDIR/scenenet_depth.log  2>&1 &
wait

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
mamba run -n test2 python collect_results.py
mamba run -n test2 python build_index.py

echo "ALL DONE"
