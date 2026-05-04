#!/usr/bin/env bash
# v2 launcher — single OURS_FINAL ckpt, gen3c removed, 4× H100 80GB.
#
# Layout: 8 datasets in wave 1 (2 per GPU concurrent), then 2 in wave 2.
# Within each dataset run_dataset_v2.sh runs serially (1 inference job at a time),
# so each GPU sees ≤ 2 concurrent inferences → comfortably fits 80GB.

set -uo pipefail

export OURS_FINAL=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-30/23-29-19/checkpoints/last_archive.ckpt
export GEO4D_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
export DFOT_RE10K_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/diffusion-forcing-transformer/pretrained/DFoT_RE10K.ckpt
export RAYDIFF_DIR=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion
export OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs_v2

ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
LOGDIR=$ROOT/eval_logs_v2
mkdir -p $OUT $LOGDIR

cd $ROOT
chmod +x run_dataset_v2.sh

echo "[run_all_v2] wave 1 starting at $(date)"
# Wave 1 — 8 datasets, 2 per GPU
./run_dataset_v2.sh re10k           0 > $LOGDIR/re10k.log           2>&1 &
./run_dataset_v2.sh aria            0 > $LOGDIR/aria.log            2>&1 &
./run_dataset_v2.sh dl3dv           1 > $LOGDIR/dl3dv.log           2>&1 &
./run_dataset_v2.sh vkitti2         1 > $LOGDIR/vkitti2.log         2>&1 &
./run_dataset_v2.sh dl3dv_test      2 > $LOGDIR/dl3dv_test.log      2>&1 &
./run_dataset_v2.sh scannetpp       2 > $LOGDIR/scannetpp.log       2>&1 &
./run_dataset_v2.sh tanksandtemples 3 > $LOGDIR/tanksandtemples.log 2>&1 &
./run_dataset_v2.sh spatialvid_nvs  3 > $LOGDIR/spatialvid_nvs.log  2>&1 &
wait
echo "[run_all_v2] wave 1 DONE at $(date)"

echo "[run_all_v2] wave 2 starting at $(date)"
# Wave 2 — remaining 2 datasets
./run_dataset_v2.sh scenenet_depth  0 > $LOGDIR/scenenet_depth.log  2>&1 &
./run_dataset_v2.sh agibot_world    1 > $LOGDIR/agibot_world.log    2>&1 &
wait
echo "[run_all_v2] wave 2 DONE at $(date)"

# Aggregate
mamba run -n test2 python collect_results_v2.py
mamba run -n test2 python build_index_v2.py
echo "[run_all_v2] ALL DONE at $(date)"
