#!/bin/bash
# DL3DV baselines on GPU 1 (sequential to be safe).
set -uo pipefail
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new

OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs
LOGS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_logs
GEO4D=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
DFOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/diffusion-forcing-transformer/pretrained_models/DFoT_RE10K.ckpt
RAYDM=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion
GEN3C=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/GEN3C/checkpoints

PD=$OUT/dl3dv/ours_pose_depth
NVS=$OUT/dl3dv/ours_nvs

run() { local name=$1; shift; echo "[bl] start $name"; CUDA_VISIBLE_DEVICES=1 stdbuf -oL mamba run -n test2 python -u "$@" > "$LOGS/dl3dv_${name}.log" 2>&1; echo "[bl] $name exit=$?"; }

run geo4d_pose       scripts/eval_geo4d_pose_2.py     --input_dir "$PD"  --output_dir "$OUT/dl3dv/geo4d_pose"   --ckpt_path "$GEO4D" --max_samples 10 --rerun
run raydiffusion_pose scripts/eval_raydiffusion_pose.py --input_dir "$PD" --output_dir "$OUT/dl3dv/raydiffusion_pose" --model_dir "$RAYDM" --max_samples 10 --dataset re10k
run geo4d_depth      scripts/eval_geo4d_depth.py     --input_dir "$PD"  --output_dir "$OUT/dl3dv/geo4d_depth"  --ckpt_path "$GEO4D" --dataset dl3dv --max_samples 10 --rerun
run chronodepth      scripts/eval_chronodepth_depth_aria.py --input_dir "$PD" --output_dir "$OUT/dl3dv/chronodepth_depth" --dataset dl3dv --max_samples 10
run gen3c_nvs        scripts/eval_gen3c_nvs.py        --input_dir "$NVS" --output_dir "$OUT/dl3dv/gen3c_nvs"  --ckpt_dir "$GEN3C" --max_samples 10 --dataset re10k
run dfot_nvs         scripts/eval_dfot_nvs_re10k.py  --input_dir "$NVS" --output_dir "$OUT/dl3dv/dfot_nvs"   --ckpt_path "$DFOT" --max_samples 10
run seva_nvs         scripts/eval_svc_nvs.py         --input_dir "$NVS" --output_dir "$OUT/dl3dv/seva_nvs"   --max_samples 10
run wan_flf_nvs      scripts/inference_single_gpu_flf2v_from_mp4_eval.py --input_dir "$NVS" --output_dir "$OUT/dl3dv/wan_flf_nvs" --max_samples 10 --show_metrics --save_stats
echo "[bl] all dl3dv baselines done"
