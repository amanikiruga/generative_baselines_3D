#!/bin/bash -l
#SBATCH --job-name=v4-n50-retry
#SBATCH --output=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs_v4_n50_retry/slurm-%A_%a.out
#SBATCH --error=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs_v4_n50_retry/slurm-%A_%a.err
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-7%8

# Retry / fold-into-V4 array. Covers:
#   - 3 datasets where the previous H100 array got only half the matrix due to
#     ours_final + ours_nvs racing on 80GB (tanksandtemples, vkitti2,
#     spatialvid_nvs). SERIAL_OURS=1 fixes the OOM. Baselines from those array
#     tasks are inherited as symlinks (skip-trip via maybe_link_v2_full); ours
#     inference re-runs cleanly into eval_outputs_v4_n50/.
#   - 2 zeroshot datasets with full geom (7scenes, tum) — fresh runs against
#     metadata_zeroshot_v4.json.
#   - 2 depth-only datasets (scenenet_depth, bonn) — no GT cameras; loader
#     synthesizes zero raymaps, model runs RGB → ray+depth, ray output is
#     ignored, depth output is used. DEPTH_ONLY=1 in run_dataset_v4_n50.sh
#     skips pose eval and pose baselines.
#
# Output writes into the main eval_outputs_v4_n50/ tree (folded into V4).

set -uo pipefail

ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
cd $ROOT

DATASETS=(tanksandtemples vkitti2 spatialvid_nvs 7scenes tum bonn dl3dv dl3dv_test)
# scenenet_depth dropped from this retry — has a downstream tensor shape bug
# (Expected 240 got 30) that needs separate investigation.
# dl3dv and dl3dv_test are added so we can: (a) run DFoT NVS on them (V3 cache
# didn't have DFoT for these); (b) regenerate dl3dv geo4d_depth final_stats.json
# (V4 fg has the per-sample preds but missed the aggregate). Existing ours
# inference cells get skipped via the LAST_SAMPLE/final_stats.json triggers.
# bonn depth-only: synthesizes randn raymaps in the loader; DEPTH_ONLY=1 in
# run_dataset_v4_n50.sh skips pose eval/baselines.
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

export OUT=$ROOT/eval_outputs_v4_n50
export V2_OUT=$ROOT/eval_outputs_v3_n50
export METADATA_V4=$ROOT/metadata_v3_n50.json   # overridden inside run_dataset for zeroshot
export OURS_FINAL=/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model/outputs/2026-05-04/12-34-53/checkpoints/last.ckpt
export OURS_NVS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-29/14-31-15/checkpoints/last.ckpt
export GEO4D_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
export DFOT_RE10K_CKPT=$ROOT/diffusion-forcing-transformer/pretrained/DFoT_RE10K.ckpt
export RAYDIFF_DIR=$ROOT/RayDiffusion/models/co3d_diffusion

LOGDIR=$ROOT/eval_logs_v4_n50_retry
mkdir -p $OUT $LOGDIR
export LOGDIR_V4=$LOGDIR

# H100 80GB cannot fit ours_final + ours_nvs in parallel during inference;
# serialize them.
export SERIAL_OURS=1

echo "[retry task=$SLURM_ARRAY_TASK_ID dataset=$DATASET] start at $(date)"
echo "[retry task=$SLURM_ARRAY_TASK_ID] node=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Pre-populate eval_outputs_v4_n50/$DATASET/ with the array's already-done
# cells, so run_dataset_v4_n50.sh's skip-if-complete checks trip and only the
# missing halves re-run. Only applies to the 3 array-half-matrix datasets;
# zeroshot datasets fall through (no array tree to inherit from).
ARRAY_OUT=$ROOT/eval_outputs_v4_n50_array
link_if() {
    local src=$1 dst=$2 label=$3
    if [ -e "$dst" ] || [ -L "$dst" ]; then return; fi
    ln -s "$(realpath "$src")" "$dst"
    echo "[retry/$DATASET] inherit $label: $(basename "$dst")"
}

if [ -d "$ARRAY_OUT/$DATASET" ]; then
    mkdir -p "$OUT/$DATASET"
    MAXN=50

    # (1) Inference dirs (ours_*_pose_depth / ours_*_nvs): inherit only when
    # the script's exact skip-trigger file is present. That guarantees the
    # script's skip-if-complete check fires BEFORE any write — no mutation
    # of the array tree is possible. Datasets where the array got only a
    # partial run (e.g. tanksandtemples 5/50) won't have the trigger and
    # will re-run cleanly into the FG tree.
    for ckpt in ours_final ours_nvs; do
        # pose_depth trigger: last-sample pred_cameras.npz
        sub="$ARRAY_OUT/$DATASET/${ckpt}_pose_depth"
        trigger="$sub/sample_$(printf '%05d' $((MAXN - 1)))/pred_cameras.npz"
        if [ -f "$trigger" ]; then
            link_if "$sub" "$OUT/$DATASET/${ckpt}_pose_depth" "ours pose_depth"
        fi
        # nvs trigger: final_stats.json
        sub="$ARRAY_OUT/$DATASET/${ckpt}_nvs"
        if [ -f "$sub/final_stats.json" ]; then
            link_if "$sub" "$OUT/$DATASET/${ckpt}_nvs" "ours nvs"
        fi
    done

    # (2) Baseline dirs: inherit if final_stats.json present (they've finished).
    # The script's maybe_link_v2_full / nvs_link guards skip them when the
    # symlink already exists, so no writes through.
    for b in geo4d_pose raydiffusion_pose geo4d_depth chronodepth_depth dfot_nvs seva_nvs wan_flf_nvs; do
        sub="$ARRAY_OUT/$DATASET/$b"
        if [ -L "$sub" ] || [ -f "$sub/final_stats.json" ] || [ -f "$sub/per_sample_metrics.csv" ]; then
            link_if "$sub" "$OUT/$DATASET/$b" "baseline"
        fi
    done

    # (3) Eval dirs (*_eval): NEVER inherit — they're rewritten unconditionally.
fi

chmod +x run_dataset_v4_n50.sh
./run_dataset_v4_n50.sh $DATASET 0 50 \
    > $LOGDIR/${DATASET}.log 2>&1
RC=$?
echo "[retry task=$SLURM_ARRAY_TASK_ID dataset=$DATASET] done rc=$RC at $(date)"
exit $RC
