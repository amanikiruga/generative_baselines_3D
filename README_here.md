# generative baseline methods 
0 - downloaded 
1 - installed and written eval script

[1] Chronodepth - depth 
[1] History guidance diffusion (DFoT) - pose NVS 
[1] Wan 2.1 14B - pose-free NVS 
[1] GEO4D + post optim - depth, camera pose 
[1] SEVA (SVC) - NVS
[1] GEN3C - NVS 
[1] raydiffusion - camera pose 


# Goal 
I want to evaluate the performance of the above methods so as to compare with our method. 
my method's directory is `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new`
In order to ensure apples-to-apples comparison, I have already run the following commands
inside my method's directory. All the paths are relative to my method's directory.
From the eval scripts you can see the metrics i am using and we need to use the same metrics to evaluate the other methods for apples-to-apples comparison.
## Aria 
### Depth + camera pose prediction (from RGB)
*command:* `python scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py --ckpt_path $ARIA_CKPT --dataset aria --output_dir ignore/eval_outputs/depth_aria_on_aria --max_samples 50 --seed 42 --no_augmentations`
*output:* `ignore/eval_outputs/depth_aria_on_aria`
#### eval commands 
*depth eval:* 
```
python scripts/eval_rgb_to_ray_depth_depth.py \
    --input_dir ignore/eval_outputs/depth_aria_on_aria \
    --output_dir ignore/eval_outputs/aria_on_aria_depth_eval \
    --custom_run_name depth_aria_on_aria_apr13
```
*camera pose eval:* 
```
python scripts/eval_ray_mot_pose.py \
    --input_dir ignore/eval_outputs/depth_aria_on_aria \
    --output_dir ignore/eval_outputs/aria_on_aria_pose_eval \
    --custom_run_name depth_aria_on_aria_mar24

```
### NVS (conditioned on camera pose and first and last frame of RGB)
*command:* `CUDA_VISIBLE_DEVICES=2 python scripts/inference_single_gpu_ray_to_rgb_depth_eval.py --ckpt_path $ARIA_CKPT --dataset aria --output_dir ignore/eval_outputs/nvs_aria_on_aria --max_samples 50 --seed 42 --no_augmentations`
*output:* `ignore/eval_outputs/nvs_aria_on_aria`
#### eval commands 
*NVS eval:* 
```
python scripts/recompute_metrics_offline.py \
  --input_dir ignore/eval_outputs/nvs_aria_on_aria \
  --output_dir ignore/eval_outputs/nvs_aria_on_aria_nvs_eval
```

## RE10K 
### Camera pose prediction (from RGB)
*command:*
```
CUDA_VISIBLE_DEVICES=0 python scripts/inference_single_gpu_ray_mot_eval.py \
  --ckpt_path /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-04/12-48-12/checkpoints/last.ckpt \
  --output_dir ignore/outputs/ray_mot_eval_histg_1_frames_50_apr08 \
  --hist_guidance 1.0 \
  --max_samples 50 \
  --n_frames 50 \
  --no_augmentations \
  --seed 42
```
*output:* `ignore/outputs/ray_mot_eval_histg_1_frames_50_apr08`
#### eval commands 
*camera pose eval:* 
```

python scripts/eval_ray_mot_pose.py \
  --input_dir ignore/outputs/ray_mot_eval_histg_1_frames_50_apr08 \
  --output_dir ignore/outputs/ray_mot_eval_histg_1_frames_50_apr08_pose_eval

```
### NVS (conditioned on camera pose and first and last frame of RGB)
*command:* `CUDA_VISIBLE_DEVICES=2 python scripts/inference_single_gpu_ray2rgb_firstlast_eval.py --ckpt_path $RE10K_CKPT --dataset re10k --output_dir ignore/eval_outputs/nvs_re10k_on_re10k --max_samples 50 --seed 42 --no_augmentations`

*output:* `ignore/eval_outputs/nvs_re10k_on_re10k`
#### eval commands 
*NVS eval:* 
```
python scripts/recompute_metrics_offline.py \
  --input_dir ignore/eval_outputs/nvs_re10k_on_re10k \
  --output_dir ignore/eval_outputs/nvs_re10k_on_re10k_nvs_eval
```

# Metrics 
## NVS 
*PSNR:* 
*LPIPS:* 
*SSIM:* 

## Camera pose
*AUC3*
*AUC30*

## Depth
*Delta1*
*Delta2*
*Delta3*



# Special notes
## Aria mismatch with opencv camera format
There is a device_to_camera transformation matrix to align the Aria camera format with the opencv camera format. please refer to `scripts/eval_da3_pose_debug_alt.py` in my repository when ran with `--procrustes_align_pred` to understand further. 


# Run commands for each baseline 
All commands written hereare relative to my method's directory `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new`

new scripts should expect to be run from my method's directory. even if the script is in this directory.
## Wan 2.1 14B 
Note: pose-free NVS, conditions only on first and last RGB frame — no camera pose needed.
### RE10K
#### NVS (conditioned on first and last frame of RGB, w/o camera pose)
*command:*
```
CUDA_VISIBLE_DEVICES=1 python scripts/inference_single_gpu_flf2v_from_mp4_eval.py \
  --input_dir ignore/eval_outputs/nvs_re10k_on_re10k \
  --output_dir ignore/eval_outputs/nvs_re10k_on_re10k_flf2v_eval \
  --max_samples 50 \
  --show_metrics \
  --save_stats
```
*output:* `ignore/eval_outputs/nvs_re10k_on_re10k_flf2v_eval`

### Aria
#### NVS (conditioned on first and last frame of RGB, w/o camera pose)
*command:*
```
CUDA_VISIBLE_DEVICES=1 python scripts/inference_single_gpu_flf2v_from_mp4_eval.py \
  --input_dir ignore/eval_outputs/nvs_aria_on_aria \
  --output_dir ignore/eval_outputs/nvs_aria_on_aria_flf2v_eval \
  --max_samples 50 \
  --show_metrics \
  --save_stats
```
*output:* `ignore/eval_outputs/nvs_aria_on_aria_flf2v_eval`
## GEO4D
### Aria 
#### Depth
*command:* `python scripts/eval_geo4d_depth.py --input_dir ignore/eval_outputs/depth_aria_on_aria --output_dir ignore/eval_outputs/geo4d_depth_on_aria --ckpt_path $GEO4D_CKPT --max_samples 50`
*output:* `ignore/eval_outputs/geo4d_depth_on_aria`

#### Camera pose
*command:* `CUDA_VISIBLE_DEVICES=0 python scripts/eval_geo4d_pose.py --input_dir ignore/eval_outputs/depth_aria_on_aria --output_dir ignore/eval_outputs/geo4d_on_aria --ckpt_path $GEO4D_CKPT --max_samples 50 --rerun --dataset aria`
*output:* `ignore/eval_outputs/geo4d_on_aria`

### RE10K
#### Camera pose
*command:* `python scripts/eval_geo4d_pose_2.py --input_dir ignore/outputs/ray_mot_eval_histg_1_frames_50_apr08 --output_dir ignore/eval_outputs/geo4d_on_re10k --ckpt_path $GEO4D_CKPT --max_samples 50 --rerun`
*output:* `ignore/eval_outputs/geo4d_on_re10k`

## ChronoDepth
Note: outputs relative (affine-invariant) depth, aligned to GT via RANSAC+LS before metrics.
### Aria
#### Depth
*command:*
```
CUDA_VISIBLE_DEVICES=0 python scripts/eval_chronodepth_depth_aria.py \
  --input_dir ignore/eval_outputs/depth_aria_on_aria \
  --output_dir ignore/eval_outputs/chronodepth_depth_on_aria \
  --max_samples 50 \
  --custom_run_name chronodepth_depth_aria_apr17
```
*output:* `ignore/eval_outputs/chronodepth_depth_on_aria`

## RayDiffusion
Note: operates on unordered image sets (not video); evaluates at most 8 frames per sequence sampled evenly.
### Aria
#### Camera pose
*command:*
```
CUDA_VISIBLE_DEVICES=2 python scripts/eval_raydiffusion_pose.py \
  --input_dir ignore/eval_outputs/depth_aria_on_aria \
  --output_dir ignore/eval_outputs/raydiffusion_pose_on_aria \
  --model_dir /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion \
  --max_samples 50 \
  --dataset aria \
  --custom_run_name raydiffusion_pose_aria_apr17
```
*output:* `ignore/eval_outputs/raydiffusion_pose_on_aria`

### RE10K
#### Camera pose
*command:*
```
CUDA_VISIBLE_DEVICES=0 python scripts/eval_raydiffusion_pose.py \
  --input_dir ignore/outputs/ray_mot_eval_histg_1_frames_50_apr08 \
  --output_dir ignore/eval_outputs/raydiffusion_pose_on_re10k \
  --model_dir /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion \
  --max_samples 50 \
  --custom_run_name raydiffusion_pose_re10k_apr17
```
*output:* `ignore/eval_outputs/raydiffusion_pose_on_re10k`

## GEN3C
Note: camera-controlled NVS; conditions on first frame + GT camera trajectory. Metrics on frames [1:-1].
### Aria
#### NVS
*command:*
```
CUDA_VISIBLE_DEVICES=0 python scripts/eval_gen3c_nvs.py \
  --input_dir ignore/eval_outputs/nvs_aria_on_aria \
  --output_dir ignore/eval_outputs/gen3c_nvs_on_aria \
  --ckpt_dir /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/GEN3C/checkpoints \
  --max_samples 50 \
  --dataset aria \
  --custom_run_name gen3c_nvs_aria_apr17
```
*output:* `ignore/eval_outputs/gen3c_nvs_on_aria`

### RE10K
#### NVS
Note: uses ray_mot eval dir (has gt_cameras.npz + gt_rgb.mp4). Cameras needed for GEN3C conditioning.
*command:*
```
CUDA_VISIBLE_DEVICES=0 python scripts/eval_gen3c_nvs.py \
  --input_dir ignore/outputs/ray_mot_eval_histg_1_frames_50_apr08 \
  --output_dir ignore/eval_outputs/gen3c_nvs_on_re10k \
  --ckpt_dir /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/GEN3C/checkpoints \
  --max_samples 50 \
  --custom_run_name gen3c_nvs_re10k_apr17
```
*output:* `ignore/eval_outputs/gen3c_nvs_on_re10k`

## DFoT (History Guidance Diffusion)
Note: camera-conditioned NVS via frame interpolation (first+last frame known). Download pretrained checkpoint from HuggingFace: `kiwhansong/DFoT` → `pretrained_models/DFoT_RE10K.ckpt`. Uses ray_mot eval dir for RE10K since it has gt_cameras.npz.
### RE10K
#### NVS
*command:*
```
CUDA_VISIBLE_DEVICES=2 python scripts/eval_dfot_nvs_re10k.py \
  --input_dir ignore/eval_outputs/nvs_re10k_on_re10k \
  --output_dir ignore/eval_outputs/dfot_nvs_on_re10k \
  --ckpt_path $DFOT_RE10K_CKPT \
  --max_samples 50 \
  --custom_run_name dfot_nvs_re10k_apr17
```
*output:* `ignore/eval_outputs/dfot_nvs_on_re10k`

### Aria
#### NVS
*command:*
```
CUDA_VISIBLE_DEVICES=0 python scripts/eval_dfot_nvs_re10k.py \
  --input_dir ignore/eval_outputs/nvs_aria_on_aria \
  --output_dir ignore/eval_outputs/dfot_nvs_on_aria \
  --ckpt_path $DFOT_RE10K_CKPT \
  --max_samples 50 \
  --dataset aria \
  --custom_run_name dfot_nvs_aria_apr17
```
*output:* `ignore/eval_outputs/dfot_nvs_on_aria`

## SEVA (SVC)
### Aria
#### NVS
*command:*
```
CUDA_VISIBLE_DEVICES=0 python scripts/eval_svc_nvs.py \
  --input_dir ignore/eval_outputs/nvs_aria_on_aria \
  --output_dir ignore/eval_outputs/seva_nvs_on_aria \
  --max_samples 50 \
  --dataset aria
```
*output:* `ignore/eval_outputs/seva_nvs_on_aria`

### RE10K
#### NVS
Note: uses ray_mot eval dir for cameras.
*command:*
```
CUDA_VISIBLE_DEVICES=0 python scripts/eval_svc_nvs.py \
  --input_dir ignore/outputs/ray_mot_eval_histg_1_frames_50_apr08 \
  --output_dir ignore/eval_outputs/seva_nvs_on_re10k \
  --max_samples 50
```
*output:* `ignore/eval_outputs/seva_nvs_on_re10k`


# checklist of what to run 
x means done 
`-` means running 
` ` means not yet started

## GEO4D
- [x] run geo4d depth on aria 
- [x] run geo4d pose on aria 
- [x] run geo4d pose on re10k 

## Wan 2.1 FLF
- [x] run wan flf nvs on re10k 
- [x] run wan flf nvs on aria 

## ChronoDepth
- [x] run chronodepth depth on aria 

## RayDiffusion
- [x] run raydiffusion pose on aria 
- [x] run raydiffusion pose on re10k 

## GEN3C
- [ ] run gen3c nvs on aria 
- [ ] run gen3c nvs on re10k 

## DFoT (History Guidance)
- [x] run dfot nvs on re10k 
- [-] run dfot nvs on aria 

## SEVA (SVC)
- [x] run seva nvs on aria 
- [x] run seva nvs on re10k 




# EDIT: datasets and what to test 

RE10K test split 
NVS, pose-pred

DL3DV-validation split 
NVS, pose-pred
Use colmap pose

Tanks and Temples
NVS

KITTI
depth-pred

Sintel
Pose-pred
depth-pred

Scannet++
pose-pred
depth-pred

ETH3D
Pose-pred
Depth-pred
 
DTU
Pose-pred
Depth-pred 


dataset directories table
dataset | directory | 
RE10K | /net/holy-isilon/ifs/rc_labs/ydu_lab/zhiyi24/dataset/re10k
Scannet++ | /net/holy-isilon/ifs/rc_labs/ydu_lab/zhiyi24/workspace/video_world_model/data/scannetpp
DL3DV-eval | /n/holylfs05/LABS/rcai_lab/Lab/dataset/data_downloads
Tanks and Temples | /net/holy-isilon/ifs/rc_labs/ydu_lab/zhiyi24/workspace/video_world_model/data/

---

# Phase-2 (NeurIPS): full eval grid on new datasets

All commands assume cwd = `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new` (our method's directory). Run with `mamba run -n test2 python ...`.

## Outputs are saved here

All inference and evaluation outputs go into:

```
/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs/<dataset>/<method>/
```

Note: `eval_outputs` is a symlink to `/n/netscratch/ydu_lab/Everyone/akiruga/generative_baselines_eval/` because the Holyoke ydu_lab volume was at 100% capacity when this ran. The symlink keeps paths under the codebase while writing to free disk.

Layout per dataset:

```
eval_outputs/<dataset>/
├── ours_pose_depth/         # output of our method's RGB→ray+depth inference
│   └── sample_XXXXX/{gt,pred}_{rgb,ray_d,ray_m,depth}.mp4 + {gt,pred}_cameras.npz + *_depth_raw.npz
├── ours_nvs/                # output of our method's ray+firstlast→rgb+depth inference
├── geo4d_pose/              # geo4d eval that consumes ours_pose_depth/
├── geo4d_depth/
├── chronodepth_depth/
├── raydiffusion_pose/
├── gen3c_nvs/
├── dfot_nvs/
├── seva_nvs/
└── wan_flf_nvs/
```

Logs go to `generative_baselines/eval_logs/<job_name>.log`.

## Checkpoint used

`OURS_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-27/22-44-50/checkpoints/epoch=0-step=2850.ckpt`

The 1.3B mixture model (re10k+dl3dv+omniworld+scannetpp+scenenet_depth+vkitti2+point_odyssey, *not* aria). We use the latest stable epoch checkpoint instead of `last.ckpt` because `last.ckpt` is being actively rewritten by the training run that's still in progress; switch to `last.ckpt` once training stops. Use 

## Dataset support / availability

| Dataset    | Loader present | Data on disk | Status |
|------------|---------------|--------------|--------|
| RE10K      | datasets/re10k.py | yes | runnable |
| DL3DV      | datasets/dl3dv.py (ViPE) | yes (processed_dl3dv_ours) | runnable |
| Scannet++  | datasets/scannetpp.py (ViPE) | yes | runnable |
| Aria       | datasets/aria.py | yes (project_aria_full/train_undistorted) | runnable (synthetic-grade GT depth + cameras) |
| VKitti2    | datasets/vkitti2.py | yes (zhiyi24 data root) | runnable (synthetic outdoor, metric depth) |
| SceneNet   | datasets/scenenet_depth.py | yes (netscratch scenenetrgbd) | runnable for **depth only** (loader exposes RGB+depth; no cameras — inference script uses zero-raymap stub so the rgb→ray+depth pipeline runs end-to-end. Pose eval is not meaningful; NVS task is skipped) |
| ETH3D      | datasets/eth3d.py | data root empty | TODO: download |
| T&T        | none | image_sets zips at /n/holylfs05/.../tanksandtemples | TODO: write image-folder loader + COLMAP poses |
| Sintel     | none | /net/holy-isilon/ifs/rc_labs/ydu_lab/zhiyi24/dataset/sintel | TODO: write loader (uses bundler poses + flow_code GT) |
| KITTI      | none | not located | TODO |
| DTU        | none | not located | TODO |

Two inference scripts (`inference_single_gpu_rgb_to_ray_depth_aria_eval.py` and `inference_single_gpu_ray_to_rgb_depth_eval.py`) had their `_DATASETS` dispatch extended to recognize `dl3dv`, `scannetpp`, and `vkitti2` (aria + scenenet_depth were already there). Both also auto-bump `dataset.test_percentage=0.1` for {aria, vkitti2, scenenet_depth} since their config defaults (0.01 / 0.0 / 0.0) yield 0 records in the validation/test slice.

### Two checkpoints compared

We compare two checkpoints from the same architecture family and refer to them as `ours` and `ours_nvs_only` throughout. **Different model, same task suite** — not different tasks.

```bash
export OURS_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-27/22-44-50/checkpoints/last.ckpt
export NVS_ONLY_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/zhiyi24/workspace/video_world_model/outputs/2026-04-26/07-31-50/checkpoints/last.ckpt
```

For each dataset, every command below should be run **twice** — once with `--ckpt_path $OURS_CKPT --output_dir .../ours_pose_depth` and once with `--ckpt_path $NVS_ONLY_CKPT --output_dir .../ours_nvs_only_pose_depth` (and analogously for `ours_nvs` / `ours_nvs_only_nvs`). The eval scripts then consume those two output dirs and produce two table rows.

### Depth eval — DA-V2-aligned, per-dataset gates

The depth eval scripts (`eval_rgb_to_ray_depth_depth.py`, `eval_geo4d_depth.py`, `eval_chronodepth_depth_aria.py`) now require `--dataset <name>` and apply DA-V2-style metrics:

1. Convert pred + GT disparity to depth via `d = 1/clip(disp, 1e-3, 1-1e-3) − 1`.
2. Build a GT-only valid mask in depth space, per dataset:
   - **dl3dv, scannetpp** (ViPE, no metric): `gt_disp > 1e-3 & gt_d > 1e-3`, no max.
   - **aria, scenenet_depth** (synthetic, indoor): `gt_m ∈ [1e-3, 20]` m (loader scale recovered from NPZ).
   - **vkitti2** (synthetic, outdoor): `gt_m ∈ [1e-3, 80]` m.
3. RANSAC + LS scale+shift fit `s>0` on `(pred_d[valid], gt_d[valid])`.
4. DA-V2 formula on aligned pred over the same `valid` mask: d1/d2/d3, abs_rel, sq_rel, rmse, rmse_log, log10, silog.

The inference scripts also save the per-clip `raymap_scale` into `pred_depth_raw.npz` and `gt_depth_raw.npz` so eval can recover metres.

## ENV (export once)

```bash
export OURS_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-27/22-44-50/checkpoints/last.ckpt
export NVS_ONLY_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/zhiyi24/workspace/video_world_model/outputs/2026-04-26/07-31-50/checkpoints/last.ckpt
export GEO4D_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
export DFOT_RE10K_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/diffusion-forcing-transformer/pretrained_models/DFoT_RE10K.ckpt
export GEN3C_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/GEN3C/checkpoints
export OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs
```

## Our method on each dataset (10 samples per BFS pass)

> Each row reads as one command. Use `CUDA_VISIBLE_DEVICES=0` for GPU 0, `=1` for GPU 1. 

### RE10K
```bash
# pose+depth (RGB→ray+depth)
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
  --ckpt_path $OURS_CKPT --dataset re10k --output_dir $OUT/re10k/ours_pose_depth \
  --max_samples 10 --seed 42 --no_augmentations

# NVS (ray+firstlast→rgb+depth)
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/inference_single_gpu_ray_to_rgb_depth_eval.py \
  --ckpt_path $OURS_CKPT --dataset re10k --output_dir $OUT/re10k/ours_nvs \
  --max_samples 10 --seed 42 --no_augmentations
```

### DL3DV
```bash
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
  --ckpt_path $OURS_CKPT --dataset dl3dv --output_dir $OUT/dl3dv/ours_pose_depth \
  --max_samples 10 --seed 42 --no_augmentations

CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/inference_single_gpu_ray_to_rgb_depth_eval.py \
  --ckpt_path $OURS_CKPT --dataset dl3dv --output_dir $OUT/dl3dv/ours_nvs \
  --max_samples 10 --seed 42 --no_augmentations
```

### Scannet++
```bash
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
  --ckpt_path $OURS_CKPT --dataset scannetpp --output_dir $OUT/scannetpp/ours_pose_depth \
  --max_samples 10 --seed 42 --no_augmentations
```

### Aria
Synthetic-grade indoor (Aria capture + MPS densified depth, metric m). NVS task supported (real cameras).
```bash
# pose+depth (RGB→ray+depth)
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
  --ckpt_path $OURS_CKPT --dataset aria --output_dir $OUT/aria/ours_pose_depth \
  --max_samples 10 --seed 42 --no_augmentations

# NVS (ray+firstlast→rgb+depth)
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/inference_single_gpu_ray_to_rgb_depth_eval.py \
  --ckpt_path $OURS_CKPT --dataset aria --output_dir $OUT/aria/ours_nvs \
  --max_samples 10 --seed 42 --no_augmentations

# Repeat both with $NVS_ONLY_CKPT and outputs $OUT/aria/ours_nvs_only_pose_depth and $OUT/aria/ours_nvs_only_nvs.
```

### Virtual KITTI 2 (vkitti2)
Synthetic outdoor, metric depth m. NVS supported.
```bash
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
  --ckpt_path $OURS_CKPT --dataset vkitti2 --output_dir $OUT/vkitti2/ours_pose_depth \
  --max_samples 10 --seed 42 --no_augmentations

CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/inference_single_gpu_ray_to_rgb_depth_eval.py \
  --ckpt_path $OURS_CKPT --dataset vkitti2 --output_dir $OUT/vkitti2/ours_nvs \
  --max_samples 10 --seed 42 --no_augmentations

# Repeat with $NVS_ONLY_CKPT for ours_nvs_only_*.
```

### SceneNet (scenenet_depth)
Synthetic indoor, metric depth m. **Depth only — loader exposes no cameras.** The inference script provides zero raymaps so the rgb→ray+depth pipeline runs end-to-end; pose eval is meaningless and NVS is skipped (no GT cameras to condition on).
```bash
# pose_depth path runs but only depth metrics are valid
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py \
  --ckpt_path $OURS_CKPT --dataset scenenet_depth --output_dir $OUT/scenenet_depth/ours_pose_depth \
  --max_samples 10 --seed 42 --no_augmentations

# Repeat with $NVS_ONLY_CKPT.
```

### Tanks & Temples / KITTI / Sintel / ETH3D / DTU

Pending: see Dataset support table above. Each requires a dataset loader and a yaml config. Once added, the same `inference_single_gpu_rgb_to_ray_depth_aria_eval.py` (with `--dataset <name>`) and `inference_single_gpu_ray_to_rgb_depth_eval.py` will work end-to-end provided the loader returns the schema that DL3DV/RE10K does (rgb, raymap_d, raymap_m, optional depth, intrinsics, c2w).

## Baselines on each dataset (consume ours_*/ output dirs)

For non-aria datasets pass `--dataset re10k` to baselines that have a hardcoded choices list — that disables the Aria→OpenCV camera conversion, which is what we want for re10k/dl3dv/scannetpp (already OpenCV).

### RE10K
```bash
# Pose
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_geo4d_pose_2.py \
  --input_dir $OUT/re10k/ours_pose_depth --output_dir $OUT/re10k/geo4d_pose \
  --ckpt_path $GEO4D_CKPT --max_samples 10 --rerun
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/eval_raydiffusion_pose.py \
  --input_dir $OUT/re10k/ours_pose_depth --output_dir $OUT/re10k/raydiffusion_pose \
  --model_dir /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion \
  --max_samples 10

# Depth
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_geo4d_depth.py \
  --input_dir $OUT/re10k/ours_pose_depth --output_dir $OUT/re10k/geo4d_depth \
  --ckpt_path $GEO4D_CKPT --max_samples 10 --rerun
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/eval_chronodepth_depth_aria.py \
  --input_dir $OUT/re10k/ours_pose_depth --output_dir $OUT/re10k/chronodepth_depth --max_samples 10

# NVS  (input_dir is *_nvs since it has gt+gen3c-friendly camera npz)
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_gen3c_nvs.py \
  --input_dir $OUT/re10k/ours_nvs --output_dir $OUT/re10k/gen3c_nvs \
  --ckpt_dir $GEN3C_CKPT --max_samples 10
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/eval_dfot_nvs_re10k.py \
  --input_dir $OUT/re10k/ours_nvs --output_dir $OUT/re10k/dfot_nvs \
  --ckpt_path $DFOT_RE10K_CKPT --max_samples 10
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_svc_nvs.py \
  --input_dir $OUT/re10k/ours_nvs --output_dir $OUT/re10k/seva_nvs --max_samples 10
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/inference_single_gpu_flf2v_from_mp4_eval.py \
  --input_dir $OUT/re10k/ours_nvs --output_dir $OUT/re10k/wan_flf_nvs \
  --max_samples 10 --show_metrics --save_stats
```

### DL3DV
```bash
# Pose
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_geo4d_pose_2.py \
  --input_dir $OUT/dl3dv/ours_pose_depth --output_dir $OUT/dl3dv/geo4d_pose \
  --ckpt_path $GEO4D_CKPT --max_samples 10 --rerun
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/eval_raydiffusion_pose.py \
  --input_dir $OUT/dl3dv/ours_pose_depth --output_dir $OUT/dl3dv/raydiffusion_pose \
  --model_dir /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion \
  --max_samples 10 --dataset re10k

# NVS
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_gen3c_nvs.py \
  --input_dir $OUT/dl3dv/ours_nvs --output_dir $OUT/dl3dv/gen3c_nvs \
  --ckpt_dir $GEN3C_CKPT --max_samples 10 --dataset re10k
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/eval_svc_nvs.py \
  --input_dir $OUT/dl3dv/ours_nvs --output_dir $OUT/dl3dv/seva_nvs --max_samples 10
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/inference_single_gpu_flf2v_from_mp4_eval.py \
  --input_dir $OUT/dl3dv/ours_nvs --output_dir $OUT/dl3dv/wan_flf_nvs \
  --max_samples 10 --show_metrics --save_stats
```

### Scannet++
```bash
# Pose
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_geo4d_pose_2.py \
  --input_dir $OUT/scannetpp/ours_pose_depth --output_dir $OUT/scannetpp/geo4d_pose \
  --ckpt_path $GEO4D_CKPT --max_samples 10 --rerun
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/eval_raydiffusion_pose.py \
  --input_dir $OUT/scannetpp/ours_pose_depth --output_dir $OUT/scannetpp/raydiffusion_pose \
  --model_dir /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion \
  --max_samples 10 --dataset re10k

# Depth
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_geo4d_depth.py \
  --input_dir $OUT/scannetpp/ours_pose_depth --output_dir $OUT/scannetpp/geo4d_depth \
  --ckpt_path $GEO4D_CKPT --dataset scannetpp --max_samples 10 --rerun
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/eval_chronodepth_depth_aria.py \
  --input_dir $OUT/scannetpp/ours_pose_depth --output_dir $OUT/scannetpp/chronodepth_depth \
  --dataset scannetpp --max_samples 10
```

### Aria
```bash
# Pose
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_geo4d_pose_2.py \
  --input_dir $OUT/aria/ours_pose_depth --output_dir $OUT/aria/geo4d_pose \
  --ckpt_path $GEO4D_CKPT --max_samples 10 --rerun
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/eval_raydiffusion_pose.py \
  --input_dir $OUT/aria/ours_pose_depth --output_dir $OUT/aria/raydiffusion_pose \
  --model_dir /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion \
  --max_samples 10 --dataset aria

# Depth
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_geo4d_depth.py \
  --input_dir $OUT/aria/ours_pose_depth --output_dir $OUT/aria/geo4d_depth \
  --ckpt_path $GEO4D_CKPT --dataset aria --max_samples 10 --rerun
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/eval_chronodepth_depth_aria.py \
  --input_dir $OUT/aria/ours_pose_depth --output_dir $OUT/aria/chronodepth_depth \
  --dataset aria --max_samples 10

# NVS
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_gen3c_nvs.py \
  --input_dir $OUT/aria/ours_nvs --output_dir $OUT/aria/gen3c_nvs \
  --ckpt_dir $GEN3C_CKPT --max_samples 10 --dataset aria
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/eval_dfot_nvs_re10k.py \
  --input_dir $OUT/aria/ours_nvs --output_dir $OUT/aria/dfot_nvs \
  --ckpt_path $DFOT_RE10K_CKPT --max_samples 10 --dataset aria
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_svc_nvs.py \
  --input_dir $OUT/aria/ours_nvs --output_dir $OUT/aria/seva_nvs --max_samples 10 --dataset aria
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/inference_single_gpu_flf2v_from_mp4_eval.py \
  --input_dir $OUT/aria/ours_nvs --output_dir $OUT/aria/wan_flf_nvs \
  --max_samples 10 --show_metrics --save_stats
```

### Virtual KITTI 2 (vkitti2)
```bash
# Pose
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_geo4d_pose_2.py \
  --input_dir $OUT/vkitti2/ours_pose_depth --output_dir $OUT/vkitti2/geo4d_pose \
  --ckpt_path $GEO4D_CKPT --max_samples 10 --rerun
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/eval_raydiffusion_pose.py \
  --input_dir $OUT/vkitti2/ours_pose_depth --output_dir $OUT/vkitti2/raydiffusion_pose \
  --model_dir /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion \
  --max_samples 10 --dataset re10k

# Depth
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_geo4d_depth.py \
  --input_dir $OUT/vkitti2/ours_pose_depth --output_dir $OUT/vkitti2/geo4d_depth \
  --ckpt_path $GEO4D_CKPT --dataset vkitti2 --max_samples 10 --rerun
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/eval_chronodepth_depth_aria.py \
  --input_dir $OUT/vkitti2/ours_pose_depth --output_dir $OUT/vkitti2/chronodepth_depth \
  --dataset vkitti2 --max_samples 10
```

### SceneNet (scenenet_depth)
**Depth-only baselines** — pose and NVS skipped (no GT cameras).
```bash
CUDA_VISIBLE_DEVICES=0 mamba run -n test2 python scripts/eval_geo4d_depth.py \
  --input_dir $OUT/scenenet_depth/ours_pose_depth --output_dir $OUT/scenenet_depth/geo4d_depth \
  --ckpt_path $GEO4D_CKPT --dataset scenenet_depth --max_samples 10 --rerun
CUDA_VISIBLE_DEVICES=1 mamba run -n test2 python scripts/eval_chronodepth_depth_aria.py \
  --input_dir $OUT/scenenet_depth/ours_pose_depth --output_dir $OUT/scenenet_depth/chronodepth_depth \
  --dataset scenenet_depth --max_samples 10
```

## Computing our method's own metrics

For each dataset we run the same eval scripts that consume the same per-sample `gt_*.mp4` / `pred_*.mp4` saved by inference. Run **twice per dataset** — once for `ours` (METHOD=ours, input=`ours_pose_depth`/`ours_nvs`) and once for `ours_nvs_only` (METHOD=ours_nvs_only, input=`ours_nvs_only_pose_depth`/`ours_nvs_only_nvs`).

```bash
# Pose (uses gt_cameras.npz + pred_cameras.npz from <METHOD>_pose_depth/)
mamba run -n test2 python scripts/eval_ray_mot_pose.py \
  --input_dir $OUT/<dataset>/<METHOD>_pose_depth --output_dir $OUT/<dataset>/<METHOD>_pose_eval

# Depth (uses pred_depth_raw.npz + gt_depth_raw.npz; --dataset is REQUIRED for the gates)
mamba run -n test2 python scripts/eval_rgb_to_ray_depth_depth.py \
  --input_dir $OUT/<dataset>/<METHOD>_pose_depth --output_dir $OUT/<dataset>/<METHOD>_depth_eval \
  --dataset <dataset>

# NVS (uses gt_rgb.mp4 + pred_rgb.mp4 from <METHOD>_nvs/)
mamba run -n test2 python scripts/recompute_metrics_offline.py \
  --input_dir $OUT/<dataset>/<METHOD>_nvs --output_dir $OUT/<dataset>/<METHOD>_nvs_eval
```

## Orchestrators

End-to-end runs that I used for Phase 2/3:
- `run_phases.sh` — phased pipeline: scenenet pair (pose_depth) → vkitti2 ours → scenenet+vkitti2 evals → scenenet+vkitti2 baselines → aria pair → aria evals + baselines → final `collect_results.py` + `build_index.py`. Uses env-python directly (so `wait $!` works) and runs max-2 inference jobs in parallel on a single H100.
- `run_queue_depth.sh` — depth-priority queue (pose_depth + depth eval only).
- `run_queue_nvs.sh` — NVS-only queue, deferred sibling of the above for aria + vkitti2.
- `rerun_depth_evals.sh` — rerun the new depth-space DA-V2-aligned eval on existing DL3DV + ScanNet++ NPZs (no inference rerun).
- `/tmp/refresh_loop.sh` — every 90 s re-evals pose_depth dirs + rebuilds RESULTS.md and index.html so the page updates as samples land.

## Status — first BFS pass (10 samples each)

| Dataset    | Ours pose+depth | Ours NVS | Baselines pose | Baselines depth | Baselines NVS |
|------------|----------------|----------|----------------|-----------------|---------------|
| RE10K      | done           | done     | done           | n/a (no GT depth) | done        |
| DL3DV      | done           | done     | done           | done            | done          |
| Scannet++  | done           | n/a      | done           | done            | n/a           |
| Aria       | running (Phase 4) | pending NVS queue | pending     | pending         | pending       |
| VKitti2    | done           | partial (ours_nvs_only 6/10, others pending NVS queue) | pending | done | pending |
| SceneNet   | done           | n/a (no cameras) | n/a    | done            | n/a           |
| T&T        | blocked (loader) | -      | -              | -               | -             |
| Sintel     | blocked (loader) | -      | -              | -               | -             |
| KITTI      | blocked (loader) | -      | -              | -               | -             |
| ETH3D      | blocked (data) | -        | -              | -               | -             |
| DTU        | blocked (loader) | -      | -              | -               | -             |

Numbers go in `RESULTS.md`.


# to view index.html on live website you can run 

`python serve_index.py`