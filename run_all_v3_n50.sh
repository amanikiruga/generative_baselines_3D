#!/usr/bin/env bash
# v3-n50 launcher — Geo4D-style metrics, n=50 scenes per dataset.
# Additive on top of run_all_v3.sh; does NOT modify eval_outputs_v3/ or RESULTS_V3.md.
#
# Hardware: GPUs 0,1,2,3 (H200/H100). 2 datasets per GPU concurrent → 8 datasets
# in parallel per wave. Within each dataset run_dataset_v3_n50.sh runs serially,
# so each GPU sees ≤ 2 concurrent inference jobs.
#
# Wave 1 (8 datasets, 4 GPUs × 2):
#   GPU 0: re10k, aria
#   GPU 1: dl3dv, vkitti2
#   GPU 2: dl3dv_test, scannetpp
#   GPU 3: tanksandtemples, spatialvid_nvs
# Wave 2 (2 datasets):
#   GPU 0: scenenet_depth
#   GPU 1: agibot_world

set -uo pipefail

export OURS_FINAL=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-05-03/20-03-01/last.ckpt
export GEO4D_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
export DFOT_RE10K_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/diffusion-forcing-transformer/pretrained/DFoT_RE10K.ckpt
export RAYDIFF_DIR=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion
export OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs_v3_n50
# Reuse V3 (n=10) caches for the first 10 samples — same seed, same ckpt.
# Geo4D pose/depth caches and full NVS dirs survive the symlink shim, so we only
# rerun model inference for samples 10..49.
export V2_OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs_v3

ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
LOGDIR=$ROOT/eval_logs_v3_n50
mkdir -p $OUT $LOGDIR

cd $ROOT
chmod +x run_dataset_v3_n50.sh

echo "[run_all_v3_n50] wave 1 starting at $(date)"
./run_dataset_v3_n50.sh re10k           0 > $LOGDIR/re10k.log           2>&1 &
./run_dataset_v3_n50.sh aria            0 > $LOGDIR/aria.log            2>&1 &
./run_dataset_v3_n50.sh dl3dv           1 > $LOGDIR/dl3dv.log           2>&1 &
./run_dataset_v3_n50.sh vkitti2         1 > $LOGDIR/vkitti2.log         2>&1 &
./run_dataset_v3_n50.sh dl3dv_test      2 > $LOGDIR/dl3dv_test.log      2>&1 &
./run_dataset_v3_n50.sh scannetpp       2 > $LOGDIR/scannetpp.log       2>&1 &
./run_dataset_v3_n50.sh tanksandtemples 3 > $LOGDIR/tanksandtemples.log 2>&1 &
./run_dataset_v3_n50.sh spatialvid_nvs  3 > $LOGDIR/spatialvid_nvs.log  2>&1 &
wait
echo "[run_all_v3_n50] wave 1 DONE at $(date)"

echo "[run_all_v3_n50] wave 2 starting at $(date)"
./run_dataset_v3_n50.sh scenenet_depth  0 > $LOGDIR/scenenet_depth.log  2>&1 &
./run_dataset_v3_n50.sh agibot_world    1 > $LOGDIR/agibot_world.log    2>&1 &
wait
echo "[run_all_v3_n50] wave 2 DONE at $(date)"

mamba run -n test2 python eval_depth_disp_offline_v3.py --root eval_outputs_v3_n50
mamba run -n test2 python colorize_depth_v3.py
mamba run -n test2 python collect_results_v3_n50.py
mamba run -n test2 python build_index_v3_n50.py
echo "[run_all_v3_n50] ALL DONE at $(date)"
