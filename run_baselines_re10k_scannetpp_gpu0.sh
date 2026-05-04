#!/bin/bash
# RE10K + Scannet++ baselines on GPU 0. Waits for our re10k pose_depth job to finish first.
set -uo pipefail
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new

OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs
LOGS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs
GEO4D=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
DFOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/diffusion-forcing-transformer/pretrained_models/DFoT_RE10K.ckpt
RAYDM=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion
GEN3C=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/GEN3C/checkpoints

run() { local name=$1; shift; echo "[bl] start $name"; CUDA_VISIBLE_DEVICES=0 stdbuf -oL mamba run -n test2 python -u "$@" > "$LOGS/${name}.log" 2>&1; echo "[bl] $name exit=$?"; }

# Wait for re10k pose_depth inference to finish
echo "[bl] waiting for re10k pose_depth to free GPU 0"
while pgrep -f "CUDA_VISIBLE_DEVICES=0.*inference_single_gpu_rgb_to_ray_depth_aria_eval" >/dev/null 2>&1; do sleep 30; done
echo "[bl] GPU 0 free"

# Scannet++ pose + depth (no NVS per spec)
SPD=$OUT/scannetpp/ours_pose_depth
run scannetpp_geo4d_pose      scripts/eval_geo4d_pose_2.py     --input_dir "$SPD" --output_dir "$OUT/scannetpp/geo4d_pose"   --ckpt_path "$GEO4D" --max_samples 10 --rerun
run scannetpp_raydiffusion_pose scripts/eval_raydiffusion_pose.py --input_dir "$SPD" --output_dir "$OUT/scannetpp/raydiffusion_pose" --model_dir "$RAYDM" --max_samples 10 --dataset re10k
run scannetpp_geo4d_depth     scripts/eval_geo4d_depth.py     --input_dir "$SPD" --output_dir "$OUT/scannetpp/geo4d_depth"  --ckpt_path "$GEO4D" --dataset scannetpp --max_samples 10 --rerun
run scannetpp_chronodepth_depth scripts/eval_chronodepth_depth_aria.py --input_dir "$SPD" --output_dir "$OUT/scannetpp/chronodepth_depth" --dataset scannetpp --max_samples 10

# RE10K pose + NVS (skip depth — re10k has no GT depth)
RPD=$OUT/re10k/ours_pose_depth
RNVS=$OUT/re10k/ours_nvs
run re10k_geo4d_pose          scripts/eval_geo4d_pose_2.py     --input_dir "$RPD" --output_dir "$OUT/re10k/geo4d_pose"   --ckpt_path "$GEO4D" --max_samples 10 --rerun
run re10k_raydiffusion_pose   scripts/eval_raydiffusion_pose.py --input_dir "$RPD" --output_dir "$OUT/re10k/raydiffusion_pose" --model_dir "$RAYDM" --max_samples 10 --dataset re10k
run re10k_gen3c_nvs           scripts/eval_gen3c_nvs.py        --input_dir "$RNVS" --output_dir "$OUT/re10k/gen3c_nvs"  --ckpt_dir "$GEN3C" --max_samples 10 --dataset re10k
run re10k_dfot_nvs            scripts/eval_dfot_nvs_re10k.py  --input_dir "$RNVS" --output_dir "$OUT/re10k/dfot_nvs"   --ckpt_path "$DFOT" --max_samples 10
run re10k_seva_nvs            scripts/eval_svc_nvs.py         --input_dir "$RNVS" --output_dir "$OUT/re10k/seva_nvs"   --max_samples 10
run re10k_wan_flf_nvs         scripts/inference_single_gpu_flf2v_from_mp4_eval.py --input_dir "$RNVS" --output_dir "$OUT/re10k/wan_flf_nvs" --max_samples 10 --show_metrics --save_stats
echo "[bl] all gpu0 baselines done"
