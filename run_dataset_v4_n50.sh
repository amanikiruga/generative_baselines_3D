#!/usr/bin/env bash
# usage: ./run_dataset_v4_n50.sh <dataset> <gpu_id> [<max_samples>]
#
# V4 — 2 ckpts (OURS_FINAL, OURS_NVS) evaluated against the V3 baselines on
# the metadata-pinned video set. The 2 ckpts run *concurrently* on the same
# GPU (~25 GB each, fits on H200 144 GB even with another job).
#
# Required env exports:
#   OURS_FINAL  OURS_NVS
#   GEO4D_CKPT  DFOT_RE10K_CKPT  RAYDIFF_DIR
#   METADATA_V4  OUT
# Optional:
#   V2_OUT  (eval_outputs_v3_n50; for symlinking baseline caches)

set -uo pipefail
DATASET=$1
GPU=$2
MAXN=${3:-50}
export CUDA_VISIBLE_DEVICES=$GPU

ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
WMD=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new

run() { echo "[$DATASET gpu$GPU] $*"; "$@"; }
M="mamba run -n test2 python"

# task gates (mirrors V3)
has_depth=0; has_nvs=0
case $DATASET in
  re10k|dl3dv_test|tanksandtemples|spatialvid_nvs|agibot_world) has_nvs=1 ;;
  dl3dv|aria|vkitti2|scannetpp)                                 has_depth=1; has_nvs=1 ;;
  scenenet_depth)                                               has_depth=1 ;;
esac
if [ "$DATASET" = "aria" ]; then DFLAG=aria; else DFLAG=re10k; fi

# Zeroshot datasets: 7scenes/tum have GT cameras (pose+depth+nvs eligible).
# bonn has no GT cameras and no ray mp4s — depth-only, like scenenet_depth.
# All three reuse dl3dv's Hydra config (image dims / normalization defaults).
HYDRA_DATASET_OVERRIDE=
case $DATASET in
  7scenes|tum)
    has_depth=1; has_nvs=1
    METADATA_V4=$ROOT/metadata_zeroshot_v4.json
    HYDRA_DATASET_OVERRIDE=dl3dv
    ;;
  bonn)
    has_depth=1; has_nvs=0
    METADATA_V4=$ROOT/metadata_zeroshot_v4.json
    HYDRA_DATASET_OVERRIDE=dl3dv
    ;;
esac

# Depth-only datasets (no GT cameras): skip pose eval and pose baselines.
DEPTH_ONLY=0
case $DATASET in
  scenenet_depth|bonn) DEPTH_ONLY=1 ;;
esac

b_geo4d_pose=0; b_raydiff=0; b_geo4d_depth=0; b_chrono=0
b_dfot=0; b_seva=0; b_wan=0
case $DATASET in
  re10k)                  b_geo4d_pose=1; b_raydiff=1;                            b_dfot=1; b_seva=1; b_wan=1 ;;
  dl3dv)                  b_geo4d_pose=1; b_raydiff=1; b_geo4d_depth=1; b_chrono=1; b_dfot=1; b_seva=1; b_wan=1 ;;
  dl3dv_test|tanksandtemples)
                          b_geo4d_pose=1; b_raydiff=1;                            b_dfot=1; b_seva=1; b_wan=1 ;;
  spatialvid_nvs|agibot_world)
                          b_geo4d_pose=1; b_raydiff=1;                            b_dfot=1; b_seva=1; b_wan=1 ;;
  scannetpp)              b_geo4d_pose=1; b_raydiff=1; b_geo4d_depth=1; b_chrono=1; b_dfot=1; b_seva=1; b_wan=1 ;;
  aria)                   b_geo4d_pose=1; b_raydiff=1; b_geo4d_depth=1; b_chrono=1; b_dfot=1; b_seva=1; b_wan=1 ;;
  vkitti2)                b_geo4d_pose=1; b_raydiff=1; b_geo4d_depth=1; b_chrono=1; b_dfot=1; b_seva=1; b_wan=1 ;;
  scenenet_depth)                                      b_geo4d_depth=1; b_chrono=1 ;;
  7scenes|tum)            b_geo4d_pose=1; b_raydiff=1; b_geo4d_depth=1; b_chrono=1; b_dfot=1; b_seva=1; b_wan=1 ;;
  bonn)                                                b_geo4d_depth=1; b_chrono=1 ;;
esac

cd $ROOT
# Allow caller to override the per-method log dir (slurm array uses a separate
# tree so it doesn't clobber the foreground run's logs).
LOGDIR_V4=${LOGDIR_V4:-$ROOT/eval_logs_v4_n50}
mkdir -p $LOGDIR_V4

# SERIAL_OURS=1 → run ours_final and ours_nvs sequentially in pose_depth/NVS
# phases (required on H100 80GB; both ckpts in parallel exceed memory). On
# H200 144GB the parallel default is fine.
SERIAL_OURS=${SERIAL_OURS:-0}

declare -A CKPTS=(
  [ours_final]="$OURS_FINAL"
  [ours_nvs]="$OURS_NVS"
)

# ── 1. ours inference: 3 ckpts × pose_depth in parallel on the same GPU ───────
echo "[$DATASET gpu$GPU] === pose_depth × 3 ckpts (parallel) at $(date) ==="
for METHOD in ours_final ours_nvs; do
    CKPT=${CKPTS[$METHOD]}
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "[$DATASET gpu$GPU] WARN: ckpt missing for $METHOD ($CKPT) — skipping"
        continue
    fi
    OUT_PD=$OUT/$DATASET/${METHOD}_pose_depth
    LAST_SAMPLE=$OUT_PD/sample_$(printf '%05d' $((MAXN-1)))/pred_cameras.npz
    if [ -f "$LAST_SAMPLE" ]; then
        echo "[$DATASET gpu$GPU] skip $METHOD pose_depth — already complete"
        continue
    fi
    if [ "$SERIAL_OURS" = "1" ]; then
        $M $ROOT/ours_v4_infer/inference_rgb_to_ray_depth_v4.py \
            --ckpt_path $CKPT --dataset $DATASET ${HYDRA_DATASET_OVERRIDE:+--hydra_dataset $HYDRA_DATASET_OVERRIDE} \
            --metadata_json $METADATA_V4 --task pose_depth \
            --output_dir $OUT_PD \
            --max_samples $MAXN --seed 42 --no_augmentations \
            > $LOGDIR_V4/${DATASET}_${METHOD}_pose_depth.log 2>&1
    else
        $M $ROOT/ours_v4_infer/inference_rgb_to_ray_depth_v4.py \
            --ckpt_path $CKPT --dataset $DATASET ${HYDRA_DATASET_OVERRIDE:+--hydra_dataset $HYDRA_DATASET_OVERRIDE} \
            --metadata_json $METADATA_V4 --task pose_depth \
            --output_dir $OUT_PD \
            --max_samples $MAXN --seed 42 --no_augmentations \
            > $LOGDIR_V4/${DATASET}_${METHOD}_pose_depth.log 2>&1 &
    fi
done
wait
echo "[$DATASET gpu$GPU] pose_depth wave done at $(date)"

# ── 2. ours inference: 3 ckpts × nvs in parallel (if applicable) ──────────────
if [ $has_nvs -eq 1 ]; then
    echo "[$DATASET gpu$GPU] === nvs × 3 ckpts (parallel) at $(date) ==="
    for METHOD in ours_final ours_nvs; do
        CKPT=${CKPTS[$METHOD]}
        [ -z "$CKPT" ] || [ ! -f "$CKPT" ] && continue
        OUT_NV=$OUT/$DATASET/${METHOD}_nvs
        if [ -f "$OUT_NV/final_stats.json" ]; then
            echo "[$DATASET gpu$GPU] skip $METHOD nvs — already complete"
            continue
        fi
        if [ "$SERIAL_OURS" = "1" ]; then
            $M $ROOT/ours_v4_infer/inference_ray_to_rgb_depth_v4.py \
                --ckpt_path $CKPT --dataset $DATASET ${HYDRA_DATASET_OVERRIDE:+--hydra_dataset $HYDRA_DATASET_OVERRIDE} \
                --metadata_json $METADATA_V4 --task nvs \
                --output_dir $OUT_NV \
                --max_samples $MAXN --seed 42 --no_augmentations \
                --show_metrics \
                > $LOGDIR_V4/${DATASET}_${METHOD}_nvs.log 2>&1
        else
            $M $ROOT/ours_v4_infer/inference_ray_to_rgb_depth_v4.py \
                --ckpt_path $CKPT --dataset $DATASET ${HYDRA_DATASET_OVERRIDE:+--hydra_dataset $HYDRA_DATASET_OVERRIDE} \
                --metadata_json $METADATA_V4 --task nvs \
                --output_dir $OUT_NV \
                --max_samples $MAXN --seed 42 --no_augmentations \
                --show_metrics \
                > $LOGDIR_V4/${DATASET}_${METHOD}_nvs.log 2>&1 &
        fi
    done
    wait
    echo "[$DATASET gpu$GPU] nvs wave done at $(date)"
fi

# ── 3. ours evals × 3 ckpts (cheap; serial) ──────────────────────────────────
for METHOD in ours_final ours_nvs; do
    OUT_PD=$OUT/$DATASET/${METHOD}_pose_depth
    [ -d "$OUT_PD" ] || continue

    if [ $DEPTH_ONLY -eq 0 ]; then
        run $M $ROOT/eval_ours_pose_v3.py \
            --input_dir $OUT_PD \
            --output_dir $OUT/$DATASET/${METHOD}_pose_eval
    fi

    if [ $has_depth -eq 1 ]; then
        run $M $ROOT/eval_ours_depth_v3.py \
            --input_dir $OUT_PD \
            --output_dir $OUT/$DATASET/${METHOD}_depth_eval \
            --dataset $DATASET
    fi

    if [ $has_nvs -eq 1 ] && [ -d "$OUT/$DATASET/${METHOD}_nvs" ]; then
        run $M $WMD/scripts/recompute_metrics_offline.py \
            --input_dir $OUT/$DATASET/${METHOD}_nvs \
            --output_dir $OUT/$DATASET/${METHOD}_nvs_eval || \
            echo "[$DATASET gpu$GPU] recompute_metrics_offline failed for $METHOD (non-fatal)"
    fi
done

# ── 4. baselines (V3 caches symlinked when compatible) ───────────────────────
IN_PD=$OUT/$DATASET/ours_final_pose_depth
IN_NVS=$OUT/$DATASET/ours_final_nvs

maybe_link_v2_cache() {
    local sub=$1; local cache=$2
    local src=$V2_OUT/$DATASET/$sub/$cache
    local dst_parent=$OUT/$DATASET/$sub
    local dst=$dst_parent/$cache
    if [ -n "${V2_OUT:-}" ] && [ -d "$src" ] && [ ! -e "$dst" ]; then
        mkdir -p "$dst_parent"
        echo "[$DATASET gpu$GPU] symlink V3→V4 cache: $sub/$cache"
        ln -s "$src" "$dst"
    fi
}
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

# Whole-dir baseline reuse: if V3 produced this baseline's full output (final_stats.json
# present), symlink the entire dir into the V4 tree so we don't recompute.
# GT inputs are identical V3↔V4 (metadata-pinned), so V3 baseline outputs are valid.
maybe_link_v2_full() {
    local sub=$1
    local src=$V2_OUT/$DATASET/$sub
    local dst=$OUT/$DATASET/$sub
    if [ -n "${V2_OUT:-}" ] && [ -d "$src" ] && [ -f "$src/final_stats.json" ] && [ ! -e "$dst" ]; then
        echo "[$DATASET gpu$GPU] symlink V3 baseline → V4: $sub (full reuse)"
        ln -s "$src" "$dst"
        return 0
    fi
    [ -L "$OUT/$DATASET/$sub" ] && return 0
    return 1
}

[ $b_geo4d_pose -eq 1 ] && {
    if ! maybe_link_v2_full geo4d_pose; then
        maybe_link_v2_cache geo4d_pose geo4d_raw
        run $M $ROOT/geo4d/eval_geo4d_pose_v3.py \
            --input_dir $IN_PD --output_dir $OUT/$DATASET/geo4d_pose \
            --ckpt_path $GEO4D_CKPT --max_samples $MAXN
    fi
}
[ $b_raydiff -eq 1 ] && {
    if ! maybe_link_v2_full raydiffusion_pose; then
        run $M $ROOT/RayDiffusion/eval_raydiffusion_pose_v3.py \
            --input_dir $IN_PD --output_dir $OUT/$DATASET/raydiffusion_pose \
            --model_dir $RAYDIFF_DIR --max_samples $MAXN --dataset $DFLAG
    fi
}

[ $b_geo4d_depth -eq 1 ] && {
    if ! maybe_link_v2_full geo4d_depth; then
        maybe_link_v2_geo4d_depth_caches
        run $M $ROOT/geo4d/eval_geo4d_depth_v3.py \
            --input_dir $IN_PD --output_dir $OUT/$DATASET/geo4d_depth \
            --ckpt_path $GEO4D_CKPT --dataset $DATASET --max_samples $MAXN
    fi
}
[ $b_chrono -eq 1 ] && {
    if ! maybe_link_v2_full chronodepth_depth; then
        run $M $ROOT/ChronoDepth/eval_chronodepth_depth_aria_v3.py \
            --input_dir $IN_PD --output_dir $OUT/$DATASET/chronodepth_depth \
            --dataset $DATASET --max_samples $MAXN
    fi
}

nvs_link() {
    local sub=$1
    local src=$V2_OUT/$DATASET/$sub
    local dst=$OUT/$DATASET/$sub
    if [ -n "${V2_OUT:-}" ] && [ -d "$src" ] && [ ! -e "$dst" ]; then
        echo "[$DATASET gpu$GPU] symlink V3 NVS → V4: $sub"
        ln -s "$src" "$dst"
    fi
}
[ $b_dfot -eq 1 ] && nvs_link dfot_nvs
[ $b_seva -eq 1 ] && nvs_link seva_nvs
[ $b_wan  -eq 1 ] && nvs_link wan_flf_nvs

if [ $b_dfot -eq 1 ] && [ ! -e "$OUT/$DATASET/dfot_nvs/final_stats.json" ] && [ ! -L "$OUT/$DATASET/dfot_nvs" ]; then
    run $M $WMD/scripts/eval_dfot_nvs_re10k.py \
        --input_dir $IN_NVS --output_dir $OUT/$DATASET/dfot_nvs \
        --ckpt_path $DFOT_RE10K_CKPT --max_samples $MAXN --dataset $DFLAG
fi
if [ $b_seva -eq 1 ] && [ ! -e "$OUT/$DATASET/seva_nvs/final_stats.json" ] && [ ! -L "$OUT/$DATASET/seva_nvs" ]; then
    run $M $WMD/scripts/eval_svc_nvs.py \
        --input_dir $IN_NVS --output_dir $OUT/$DATASET/seva_nvs \
        --max_samples $MAXN --dataset $DFLAG
fi
if [ $b_wan -eq 1 ] && [ ! -e "$OUT/$DATASET/wan_flf_nvs/final_stats.json" ] && [ ! -L "$OUT/$DATASET/wan_flf_nvs" ]; then
    run $M $WMD/scripts/inference_single_gpu_flf2v_from_mp4_eval.py \
        --input_dir $IN_NVS --output_dir $OUT/$DATASET/wan_flf_nvs \
        --max_samples $MAXN --show_metrics --save_stats
fi

echo "[$DATASET gpu$GPU] DONE at $(date)"
