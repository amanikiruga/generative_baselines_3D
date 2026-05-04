#!/usr/bin/env bash
# Rerun the V3 (n=10) baselines on dl3dv only — fixes the stale-symlink
# contamination where V2's gt_rgb (different scenes, due to dataset list drift
# between May 1 and May 3) was used as input.
#
# Drops these stale links and re-infers fresh against V3's gt_rgb:
#   geo4d_pose/geo4d_raw
#   geo4d_depth/sample_*/pred_depth_geo4d.npz
#   seva_nvs       (whole dir)
#   wan_flf_nvs    (whole dir)
# RayDiffusion + ChronoDepth ran fresh in V3 — left alone.

set -uo pipefail
DATASET=dl3dv
GPU=${1:-3}            # default to GPU 3 (least loaded right now)
export CUDA_VISIBLE_DEVICES=$GPU

ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
WMD=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new
OUT=$ROOT/eval_outputs_v3
LOG=$ROOT/eval_logs_v3/dl3dv_baselines_rerun.log

GEO4D_CKPT=$ROOT/../Geo4D/checkpoints/geo4d/model.ckpt
DFOT_RE10K_CKPT=$ROOT/diffusion-forcing-transformer/pretrained/DFoT_RE10K.ckpt
RAYDIFF_DIR=$ROOT/RayDiffusion/models/co3d_diffusion

cd $WMD
M="mamba run -n test2 python"
log() { echo "[dl3dv_rerun gpu$GPU] $*" | tee -a $LOG; }
run() { log "$*"; "$@" >> $LOG 2>&1; }

mkdir -p $(dirname $LOG); : > $LOG

log "begin $(date)"

# 1. Drop stale symlinks (per-sample geo4d_depth + whole-dir seva/wan + geo4d_pose/geo4d_raw)
for f in $OUT/$DATASET/geo4d_depth/sample_*/pred_depth_geo4d.npz; do
    [ -L "$f" ] && { log "rm symlink $f"; rm -f "$f"; }
done
[ -L $OUT/$DATASET/geo4d_pose/geo4d_raw ] && { log "rm symlink geo4d_pose/geo4d_raw"; rm -f $OUT/$DATASET/geo4d_pose/geo4d_raw; }
# seva_nvs and wan_flf_nvs are full-dir symlinks → unlink (do NOT use rm -rf which would walk into V2)
[ -L $OUT/$DATASET/seva_nvs    ] && { log "unlink seva_nvs (was V2 symlink)";    unlink $OUT/$DATASET/seva_nvs; }
[ -L $OUT/$DATASET/wan_flf_nvs ] && { log "unlink wan_flf_nvs (was V2 symlink)"; unlink $OUT/$DATASET/wan_flf_nvs; }
# Stale per-sample final_stats from the symlinked runs are misleading too — clear them so the new runs overwrite cleanly.
rm -f $OUT/$DATASET/geo4d_pose/final_stats.json    $OUT/$DATASET/geo4d_pose/sample_*/pose_metrics.json
rm -f $OUT/$DATASET/geo4d_depth/final_stats.json   $OUT/$DATASET/geo4d_depth/sample_*/depth_metrics.json $OUT/$DATASET/geo4d_depth/sample_*/pred_depth.mp4

IN_PD=$OUT/$DATASET/ours_pose_depth
IN_NVS=$OUT/$DATASET/ours_nvs

# 2. Re-run the 4 contaminated baselines fresh
log "--- geo4d_pose ---"
run $M $ROOT/geo4d/eval_geo4d_pose_v3.py \
    --input_dir $IN_PD --output_dir $OUT/$DATASET/geo4d_pose \
    --ckpt_path $GEO4D_CKPT --max_samples 10

log "--- geo4d_depth ---"
run $M $ROOT/geo4d/eval_geo4d_depth_v3.py \
    --input_dir $IN_PD --output_dir $OUT/$DATASET/geo4d_depth \
    --ckpt_path $GEO4D_CKPT --dataset $DATASET --max_samples 10

log "--- seva_nvs ---"
run $M scripts/eval_svc_nvs.py \
    --input_dir $IN_NVS --output_dir $OUT/$DATASET/seva_nvs \
    --max_samples 10 --dataset re10k

log "--- wan_flf_nvs ---"
run $M scripts/inference_single_gpu_flf2v_from_mp4_eval.py \
    --input_dir $IN_NVS --output_dir $OUT/$DATASET/wan_flf_nvs \
    --max_samples 10 --show_metrics --save_stats

# 3. Colorize new geo4d depth outputs + refresh aggregator/index
log "--- post: colorize + rebuild ---"
cd $ROOT
$M colorize_depth_v3.py >> $LOG 2>&1
$M build_index_v3.py    >> $LOG 2>&1
$M collect_results_v3.py >> $LOG 2>&1
log "done $(date)"
