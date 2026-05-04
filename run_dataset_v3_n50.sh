#!/usr/bin/env bash
# usage: ./run_dataset_v3_n50.sh <dataset> <gpu_id>
#
# V3-N50 — same metrics as V3 but n=50 scenes per dataset (vs n=10 in V3).
# Additive only: writes to $OUT (= eval_outputs_v3_n50). Does NOT touch the
# n=10 V3 tree.  V3 prediction caches under $V2_OUT (= eval_outputs_v3) are
# reused for the first 10 sample dirs — Geo4D pose/depth and the NVS dirs
# rerun fresh for samples 10..49.
#
# V3 base behaviour (unchanged):
#   - new OURS_FINAL ckpt (2026-05-03/20-03-01/last.ckpt) — re-runs ours inference
#   - eval scripts swapped to *_v3.py (Geo4D vendored metrics)
#   - aria correction NOT applied at eval (now baked into aria.py at infer time)
#   - if $V2_OUT is set, the per-baseline prediction dirs are symlinked from
#     V2 → V3 to skip re-running baseline inference (same seed, deterministic).
#     Only the metric step is rerun on top of those symlinks.
#   - NVS evaluators (recompute_metrics_offline / dfot / seva / wan_flf) are
#     reused from V2 unchanged — V3 only changes pose+depth metrics.
#
# Required env exports (set by run_all_v3.sh):
#   OURS_FINAL  GEO4D_CKPT  DFOT_RE10K_CKPT  RAYDIFF_DIR  OUT
# Optional:
#   V2_OUT  (if set, symlink baseline predictions from this V2 tree)

set -uo pipefail
DATASET=$1
GPU=$2
export CUDA_VISIBLE_DEVICES=$GPU

ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
WMD=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new
cd $WMD

run() { echo "[$DATASET gpu$GPU] $*"; "$@"; }
M="mamba run -n test2 python"

# task gates
has_depth=0; has_nvs=0
case $DATASET in
  re10k|dl3dv_test|tanksandtemples|spatialvid_nvs|agibot_world) has_nvs=1 ;;
  dl3dv|aria|vkitti2|scannetpp)                                 has_depth=1; has_nvs=1 ;;
  scenenet_depth)                                               has_depth=1 ;;
esac

if [ "$DATASET" = "aria" ]; then DFLAG=aria; else DFLAG=re10k; fi

b_geo4d_pose=0; b_raydiff=0; b_geo4d_depth=0; b_chrono=0
b_dfot=0; b_seva=0; b_wan=0
case $DATASET in
  re10k)
    b_geo4d_pose=1; b_raydiff=1; b_dfot=1; b_seva=1; b_wan=1 ;;
  dl3dv)
    b_geo4d_pose=1; b_raydiff=1; b_geo4d_depth=1; b_chrono=1; b_seva=1; b_wan=1 ;;
  dl3dv_test|tanksandtemples|spatialvid_nvs|agibot_world)
    b_geo4d_pose=1; b_raydiff=1; b_seva=1; b_wan=1 ;;
  scannetpp)
    b_geo4d_pose=1; b_raydiff=1; b_geo4d_depth=1; b_chrono=1; b_seva=1; b_wan=1 ;;
  aria)
    b_geo4d_pose=1; b_raydiff=1; b_geo4d_depth=1; b_chrono=1; b_dfot=1; b_seva=1; b_wan=1 ;;
  vkitti2)
    b_geo4d_pose=1; b_raydiff=1; b_geo4d_depth=1; b_chrono=1 ;;
  scenenet_depth)
    b_geo4d_depth=1; b_chrono=1 ;;
esac

# ── 1. our-method inference (OURS_FINAL = 2026-05-03 ckpt) ────────────────────
METHOD=ours
CKPT=$OURS_FINAL

run $M scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
    --ckpt_path $CKPT --dataset $DATASET \
    --output_dir $OUT/$DATASET/${METHOD}_pose_depth \
    --max_samples 50 --seed 42 --no_augmentations

if [ $has_nvs -eq 1 ]; then
    run $M scripts/inference_single_gpu_ray_to_rgb_depth_eval.py \
        --ckpt_path $CKPT --dataset $DATASET \
        --output_dir $OUT/$DATASET/${METHOD}_nvs \
        --max_samples 50 --seed 42 --no_augmentations
fi

# ── 2. our-method evals (V3) ──────────────────────────────────────────────────
run $M $ROOT/eval_ours_pose_v3.py \
    --input_dir $OUT/$DATASET/${METHOD}_pose_depth \
    --output_dir $OUT/$DATASET/${METHOD}_pose_eval

if [ $has_depth -eq 1 ]; then
    run $M $ROOT/eval_ours_depth_v3.py \
        --input_dir $OUT/$DATASET/${METHOD}_pose_depth \
        --output_dir $OUT/$DATASET/${METHOD}_depth_eval \
        --dataset $DATASET
fi

# NVS metrics unchanged from V2 — reuse existing recompute_metrics_offline.py.
if [ $has_nvs -eq 1 ]; then
    run $M scripts/recompute_metrics_offline.py \
        --input_dir $OUT/$DATASET/${METHOD}_nvs \
        --output_dir $OUT/$DATASET/${METHOD}_nvs_eval
fi

# ── 3. baselines ─────────────────────────────────────────────────────────────
IN_PD=$OUT/$DATASET/ours_pose_depth
IN_NVS=$OUT/$DATASET/ours_nvs

# Symlink only the *prediction-cache subdirs* from V2, never the top method dir
# (otherwise V3 final_stats.json would overwrite V2 through the link). The V3
# evaluators reuse those caches and only the metric step is re-run.
#
#   geo4d_pose  → cache subdir: <dir>/geo4d_raw/sample_*/<seq>/pred_traj.txt
#   geo4d_depth → cache file:   <dir>/sample_*/pred_depth_geo4d.npz
#   raydiffusion / chronodepth → no inference cache (re-run; fast)
maybe_link_v2_cache() {  # $1 = baseline subdir, $2 = cache subdir under it
    local sub=$1; local cache=$2
    local src=$V2_OUT/$DATASET/$sub/$cache
    local dst_parent=$OUT/$DATASET/$sub
    local dst=$dst_parent/$cache
    if [ -n "${V2_OUT:-}" ] && [ -d "$src" ] && [ ! -e "$dst" ]; then
        mkdir -p "$dst_parent"
        echo "[$DATASET gpu$GPU] symlink V2 cache → V3: $sub/$cache"
        ln -s "$src" "$dst"
    fi
}
# For per-sample geo4d depth caches, link individual sample subdirs (the
# pred_depth_geo4d.npz lives one level deep in V2 too).
maybe_link_v2_geo4d_depth_caches() {
    if [ -z "${V2_OUT:-}" ]; then return; fi
    local src_root=$V2_OUT/$DATASET/geo4d_depth
    local dst_root=$OUT/$DATASET/geo4d_depth
    [ -d "$src_root" ] || return
    mkdir -p "$dst_root"
    for s in "$src_root"/sample_*; do
        [ -d "$s" ] || continue
        local name=$(basename "$s")
        local cache=$s/pred_depth_geo4d.npz
        if [ -f "$cache" ] && [ ! -e "$dst_root/$name/pred_depth_geo4d.npz" ]; then
            mkdir -p "$dst_root/$name"
            ln -s "$cache" "$dst_root/$name/pred_depth_geo4d.npz"
        fi
    done
}

# Pose: Geo4D, RayDiffusion. Both V3 scripts honour cached pred_traj.txt /
# pred_cameras.npz inside the (linked) V2 dirs and only recompute metrics.
[ $b_geo4d_pose -eq 1 ] && {
    maybe_link_v2_cache geo4d_pose geo4d_raw
    run $M $ROOT/geo4d/eval_geo4d_pose_v3.py \
        --input_dir $IN_PD --output_dir $OUT/$DATASET/geo4d_pose \
        --ckpt_path $GEO4D_CKPT --max_samples 50
}

[ $b_raydiff -eq 1 ] && \
    run $M $ROOT/RayDiffusion/eval_raydiffusion_pose_v3.py \
        --input_dir $IN_PD --output_dir $OUT/$DATASET/raydiffusion_pose \
        --model_dir $RAYDIFF_DIR --max_samples 50 --dataset $DFLAG

# Depth: Geo4D, ChronoDepth. Geo4D depth has a per-sample pred_depth_geo4d.npz
# cache; symlink the V2 versions per-sample to skip Geo4D inference.
[ $b_geo4d_depth -eq 1 ] && {
    maybe_link_v2_geo4d_depth_caches
    run $M $ROOT/geo4d/eval_geo4d_depth_v3.py \
        --input_dir $IN_PD --output_dir $OUT/$DATASET/geo4d_depth \
        --ckpt_path $GEO4D_CKPT --dataset $DATASET --max_samples 50
}

[ $b_chrono -eq 1 ] && \
    run $M $ROOT/ChronoDepth/eval_chronodepth_depth_aria_v3.py \
        --input_dir $IN_PD --output_dir $OUT/$DATASET/chronodepth_depth \
        --dataset $DATASET --max_samples 50

# NVS metrics are unchanged in V3 — symlink V2 NVS-eval dirs whole; build_index_v3
# reads from them directly. (We don't recompute NVS metrics here.)
nvs_link() {  # $1 = subdir
    local sub=$1
    local src=$V2_OUT/$DATASET/$sub
    local dst=$OUT/$DATASET/$sub
    if [ -n "${V2_OUT:-}" ] && [ -d "$src" ] && [ ! -e "$dst" ]; then
        echo "[$DATASET gpu$GPU] symlink V2 NVS → V3: $sub (read-only reuse)"
        ln -s "$src" "$dst"
    fi
}
[ $b_dfot -eq 1 ] && nvs_link dfot_nvs
[ $b_seva -eq 1 ] && nvs_link seva_nvs
[ $b_wan  -eq 1 ] && nvs_link wan_flf_nvs

# Fall back to running NVS baselines if V2 symlink isn't available.
if [ $b_dfot -eq 1 ] && [ ! -e "$OUT/$DATASET/dfot_nvs/final_stats.json" ] && [ ! -L "$OUT/$DATASET/dfot_nvs" ]; then
    run $M scripts/eval_dfot_nvs_re10k.py \
        --input_dir $IN_NVS --output_dir $OUT/$DATASET/dfot_nvs \
        --ckpt_path $DFOT_RE10K_CKPT --max_samples 50 --dataset $DFLAG
fi
if [ $b_seva -eq 1 ] && [ ! -e "$OUT/$DATASET/seva_nvs/final_stats.json" ] && [ ! -L "$OUT/$DATASET/seva_nvs" ]; then
    run $M scripts/eval_svc_nvs.py \
        --input_dir $IN_NVS --output_dir $OUT/$DATASET/seva_nvs \
        --max_samples 50 --dataset $DFLAG
fi
if [ $b_wan -eq 1 ] && [ ! -e "$OUT/$DATASET/wan_flf_nvs/final_stats.json" ] && [ ! -L "$OUT/$DATASET/wan_flf_nvs" ]; then
    run $M scripts/inference_single_gpu_flf2v_from_mp4_eval.py \
        --input_dir $IN_NVS --output_dir $OUT/$DATASET/wan_flf_nvs \
        --max_samples 50 --show_metrics --save_stats
fi

echo "[$DATASET gpu$GPU] DONE"
