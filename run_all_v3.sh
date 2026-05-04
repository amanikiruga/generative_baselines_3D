#!/usr/bin/env bash
# v3 launcher — Geo4D-style metrics; new OURS_FINAL ckpt 2026-05-03.
#
# Hardware: GPUs 0 and 1 only (H200s, 143 GB each). 2 datasets per GPU concurrent
# → 4 datasets in parallel per wave. Within each dataset run_dataset_v3.sh runs
# serially, so each H200 sees ≤ 2 concurrent inference jobs at a time.
#
# Wave 1: 4 datasets (re10k+aria on GPU0, dl3dv+vkitti2 on GPU1)
# Wave 2: 4 datasets (dl3dv_test+scannetpp on GPU0, tanksandtemples+spatialvid_nvs on GPU1)
# Wave 3: 2 datasets (scenenet_depth on GPU0, agibot_world on GPU1)

set -uo pipefail

export OURS_FINAL=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-05-03/20-03-01/last.ckpt
export GEO4D_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
export DFOT_RE10K_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/diffusion-forcing-transformer/pretrained/DFoT_RE10K.ckpt
export RAYDIFF_DIR=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion
export OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs_v3
# Reuse V2 baseline prediction caches (Geo4D pose/depth, NVS) — same seed.
export V2_OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs_v2

ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
LOGDIR=$ROOT/eval_logs_v3
mkdir -p $OUT $LOGDIR

cd $ROOT
chmod +x run_dataset_v3.sh

echo "[run_all_v3] wave 1 starting at $(date)"
./run_dataset_v3.sh re10k           0 > $LOGDIR/re10k.log           2>&1 &
./run_dataset_v3.sh aria            0 > $LOGDIR/aria.log            2>&1 &
./run_dataset_v3.sh dl3dv           1 > $LOGDIR/dl3dv.log           2>&1 &
./run_dataset_v3.sh vkitti2         1 > $LOGDIR/vkitti2.log         2>&1 &
wait
echo "[run_all_v3] wave 1 DONE at $(date)"

echo "[run_all_v3] wave 2 starting at $(date)"
./run_dataset_v3.sh dl3dv_test      0 > $LOGDIR/dl3dv_test.log      2>&1 &
./run_dataset_v3.sh scannetpp       0 > $LOGDIR/scannetpp.log       2>&1 &
./run_dataset_v3.sh tanksandtemples 1 > $LOGDIR/tanksandtemples.log 2>&1 &
./run_dataset_v3.sh spatialvid_nvs  1 > $LOGDIR/spatialvid_nvs.log  2>&1 &
wait
echo "[run_all_v3] wave 2 DONE at $(date)"

echo "[run_all_v3] wave 3 starting at $(date)"
./run_dataset_v3.sh scenenet_depth  0 > $LOGDIR/scenenet_depth.log  2>&1 &
./run_dataset_v3.sh agibot_world    1 > $LOGDIR/agibot_world.log    2>&1 &
wait
echo "[run_all_v3] wave 3 DONE at $(date)"

mamba run -n test2 python collect_results_v3.py
mamba run -n test2 python build_index_v3.py
echo "[run_all_v3] ALL DONE at $(date)"
