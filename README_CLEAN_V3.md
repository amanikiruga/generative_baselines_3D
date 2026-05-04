# Generative-baselines evaluation — V3

Single-checkpoint evaluation of our model (`OURS_FINAL`) vs published baselines, across
10 datasets covering camera-pose, depth, and novel-view-synthesis (NVS).

V3 successor to `README_CLEAN_V2.md`. **Key differences (V3 vs V2):**

| | V2 | V3 |
|---|---|---|
| Our checkpoint | 2026-04-30 / 23-29-19 | **2026-05-03 / 20-03-01 (`last.ckpt`)** |
| Pose metrics | AUC3 / AUC30 (pairwise) | **ATE / RPE_trans / RPE_rot** (Geo4D / `evo` Sim(3)-Umeyama) |
| Depth alignment | per-frame median (and per-sample LAD2 in some scripts) | **per-video global LAD2 scale-shift** (Geo4D verbatim) |
| Depth metrics keys | `abs_rel, rmse, delta_1, …` (ours) | **Abs Rel / Sq Rel / RMSE / Log RMSE / δ<1.25 / δ<1.25² / δ<1.25³** (Geo4D verbatim) |
| Eval code source | hand-rolled per method | **`geo4d_eval/` (verbatim copy of `Geo4D/dust3r/utils/vo_eval.py` + `dust3r/depth_eval.py`)** |
| Aria correction at eval | applied in pose evals | **dropped** — fixed at inference time inside `aria.py` |
| Output dir | `eval_outputs_v2 → /n/netscratch/.../generative_baselines_eval_v2` | `eval_outputs_v3 → /n/netscratch/.../generative_baselines_eval_v3` |
| index.html | `index_v2.html` | `index_v3.html` |
| Aggregator | `collect_results_v2.py` / `build_index_v2.py` | `collect_results_v3.py` / `build_index_v3.py` |

V2 outputs at `eval_outputs_v2/` are preserved. V3 writes to a fresh tree.

**No V2 file is modified.** Every V3 artefact is a new copy: scripts, dataset runner,
launch script, aggregators, and HTML index.

---

## §0 Quick start

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
nohup ./run_all_v3.sh > eval_logs_v3/_run_all.log 2>&1 &
```

To watch:

```bash
tail -f eval_logs_v3/_run_all.log
ls eval_logs_v3/
nvidia-smi
```

---

## §1 Checkpoint paths

```bash
export OURS_FINAL=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-05-03/20-03-01/last.ckpt
export GEO4D_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
export DFOT_RE10K_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/diffusion-forcing-transformer/pretrained/DFoT_RE10K.ckpt
export RAYDIFF_DIR=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion
export OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs_v3
```

Inference re-runs against `OURS_FINAL` from scratch. We do **not** reuse V2 prediction
dirs — pose/depth metrics changed, and we want a single end-to-end ledger.

---

## §2 Datasets and task coverage

(unchanged from V2)

| dataset | pose | depth | nvs | text-conditioned |
|---|---|---|---|---|
| re10k           | ✓ | ·  | ✓ | **yes** |
| dl3dv           | ✓ | ✓  | ✓ | **yes** |
| dl3dv_test      | ✓ | ·  | ✓ | · |
| tanksandtemples | ✓ | ·  | ✓ | · |
| scannetpp       | ✓ | ✓  | ✓ | · |
| aria            | ✓ | ✓  | ✓ | · |
| vkitti2         | ✓ | ✓  | ✓ | · |
| scenenet_depth  | ·  | ✓  | ·  | · |
| spatialvid_nvs  | ✓ | ·  | ✓ | · |
| agibot_world    | ✓ | ·  | ✓ | **yes** |

---

## §3 New eval code layout

```
generative_baselines/
├── geo4d_eval/
│   ├── __init__.py
│   ├── vo_eval.py                ← verbatim from Geo4D/dust3r/utils/vo_eval.py
│   └── depth_eval.py             ← verbatim from Geo4D/dust3r/depth_eval.py
│                                    (the eval_mono_depth* dust3r-coupled wrapper
│                                     stripped; only depth_evaluation + helpers kept)
│
├── eval_common_v3.py             ← thin wrapper:
│   ├── eval_pose_sequence(pred_c2w, gt_c2w, seq, save_dir)
│   │     builds TUM PoseTrajectory3D from c2w stacks
│   │     calls vo_eval.eval_metrics → (ATE, RPE_trans, RPE_rot)
│   ├── eval_depth_sequence(pred_depth_THW, gt_depth_THW,
│   │                       pred_in_disparity=False, gt_in_disparity=False, …)
│   │     If a method natively predicts disparity (1/depth) it sets the flag and
│   │     the input is inverted before LAD2 so all alignment is in *depth space*.
│   │     calls depth_eval.depth_evaluation(align_with_lad2=True, use_gpu=True)
│   │     → dict with Abs Rel / Sq Rel / RMSE / Log RMSE / δ<1.25 / δ<1.25² / δ<1.25³
│   ├── load_depth_metric_from_npz(path)
│   │     turns `*_depth_raw.npz` ([-1,1] norm-disparity + scale) into metric meters
│   └── aggregate(per_seq, keys)  arithmetic mean (Geo4D-style averaging over seqs)
│
├── colorize_depth_v3.py          ← post-eval visualisation step:
│     turns `pred_depth_geo4d.npz` / `pred_depth_raw.npz` into `pred_depth.mp4`
│     (turbo colormap). Run BEFORE `build_index_v3.py` so the depth carousel slides
│     have media to display. Idempotent.
│   └── aggregate(results)        arithmetic mean over sequences (matches Geo4D)
│
├── geo4d/
│   ├── eval_geo4d_pose_v3.py     ← cp from eval_geo4d_pose_2.py, then:
│   │       drop apply_aria_correction, drop AUC*, replace metric block
│   │       with eval_common_v3.eval_pose_sequence
│   ├── eval_geo4d_depth_v3.py    ← cp from eval_geo4d_depth.py, then:
│   │       per-video stack → eval_common_v3.eval_depth_sequence
│
├── ChronoDepth/
│   └── eval_chronodepth_depth_aria_v3.py     ← cp + swap depth metric block
│
├── RayDiffusion/
│   └── eval_raydiffusion_pose_v3.py          ← cp + swap pose metric block;
│                                                drop apply_aria_correction
│
├── (dfot / seva / wan_flf NVS scripts unchanged in V3 — they don't touch the
│  pose/depth metric pipeline. They keep V2 behaviour. Only the *_v3.sh runner
│  routes their inputs through the v3 output tree.)
│
├── (our scripts under world_model_4d/.../scripts/ stay V2)
│   we add **two new wrappers** that live in generative_baselines/, not in
│   the model repo, so the model repo is untouched:
│   ├── eval_ours_pose_v3.py      ← reads ours_pose_depth/sample_*/{pred_cameras.npz,
│   │                                gt_cameras.npz} produced by the existing
│   │                                inference_single_gpu_rgb_to_ray_depth_aria_eval.py
│   │                                and reports ATE/RPE_trans/RPE_rot
│   └── eval_ours_depth_v3.py     ← reads ours_pose_depth/sample_*/{pred_depth.npz,
│                                    gt_depth.npz} (or per-frame *.npy) and runs
│                                    per-video LAD2 alignment
│
├── run_dataset_v3.sh             ← cp from run_dataset_v2.sh, swap eval invocations
├── run_all_v3.sh                 ← cp from run_all_v2.sh, point at v3 dirs/scripts
├── collect_results_v3.py         ← cp from v2, read new metric keys
├── build_index_v3.py             ← cp from v2, render new metric keys
├── RESULTS_V3.md                 ← generated
└── index_v3.html                 ← generated
```

---

## §4 Metrics — exact definitions

### Pose (Geo4D `dust3r/utils/vo_eval.py:eval_metrics`, verbatim via vendored copy)

For each sequence:
1. Load pred+GT trajectories as `evo.core.trajectory.PoseTrajectory3D`.
2. `evo.core.sync.associate_trajectories` (timestamp = frame index).
3. **ATE** = `evo.main_ape.ape(traj_ref, traj_est, pose_relation=translation_part,
   align=True, correct_scale=True)` `.stats["rmse"]`.
   Internally aligns via Sim(3) **Umeyama** — algorithmically identical to the old
   per-method Procrustes, but reported as RMSE of aligned positions instead of
   AUC over a pairwise CDF.
4. **RPE_trans** = `evo.main_rpe.rpe(translation_part, delta=1, delta_unit=frames,
   all_pairs=True)` `.stats["rmse"]`.
5. **RPE_rot** = `evo.main_rpe.rpe(rotation_angle_deg, delta=1, delta_unit=frames,
   all_pairs=True)` `.stats["rmse"]`.

Aggregate = arithmetic mean over sequences (matches `vo_eval.calculate_averages`).

### Depth (Geo4D `dust3r/depth_eval.py:depth_evaluation`, verbatim via vendored copy)

For each sequence:
1. Stack predicted depth (T,H,W) and GT depth (T,H,W); flatten internally.
2. Mask `gt > 0` (and `gt < max_depth` for outdoor benchmarks).
3. Fit **one global (s, t) per video** by Adam-optimised L1 on `|s·d + t − gt|`
   (`align_with_lad2=True`, `lr=1e-2/1e-4`, `max_iters=5000/1000`), init from
   median ratio. **Not per-frame.**
4. Report Abs Rel, Sq Rel, RMSE, Log RMSE, δ<1.25, δ<1.25², δ<1.25³.

Aggregate = arithmetic mean over sequences.

`max_depth` per dataset (taken from Geo4D's defaults):
- vkitti2 / scannetpp / aria: `70`
- scenenet_depth / dl3dv: `None` (no clip)

### NVS

NVS metrics (PSNR / SSIM / LPIPS) are **unchanged** from V2 — V3 only changes the
geometry-task metrics. The NVS evaluators (`recompute_metrics_offline.py`,
`eval_dfot_nvs_re10k.py`, `eval_svc_nvs.py`, `inference_single_gpu_flf2v_from_mp4_eval.py`)
are reused as-is.

---

## §5 Per-dataset pipeline (`run_dataset_v3.sh`)

```bash
./run_dataset_v3.sh <dataset> <gpu_id>
```

Order (serial within one dataset):

1. Our inference RGB→ray+depth (existing V2 script, single ckpt = `OURS_FINAL`).
2. Our inference ray→RGB+depth (existing V2 script).
3. Our pose eval: **`eval_ours_pose_v3.py`** (NEW) — reads ours_pose_depth output,
   writes ATE / RPE_trans / RPE_rot.
4. Our depth eval (depth datasets): **`eval_ours_depth_v3.py`** (NEW) — per-video
   LAD2.
5. Our NVS eval: `recompute_metrics_offline.py` (V2, unchanged).
6. Pose baselines: `eval_geo4d_pose_v3.py`, `eval_raydiffusion_pose_v3.py`.
7. Depth baselines: `eval_geo4d_depth_v3.py`, `eval_chronodepth_depth_aria_v3.py`.
8. NVS baselines: DFoT, SEVA, **Wan 2.1 FLF (last)** — all V2 scripts.

---

## §6 Aggregation

```bash
mamba run -n test2 python colorize_depth_v3.py    # *required before build_index*
mamba run -n test2 python collect_results_v3.py   # writes RESULTS_V3.md
mamba run -n test2 python build_index_v3.py       # writes index_v3.html
```

`colorize_depth_v3.py` is part of the flow: V3 baseline depth evaluators
(`geo4d/eval_geo4d_depth_v3.py`, `ChronoDepth/eval_chronodepth_depth_aria_v3.py`)
dump only `*_depth_raw.npz` + metric JSON; the colorizer turbo-colormaps each
into `pred_depth.mp4` next to the npz so `index_v3.html` can render the depth
carousel slides. Idempotent — safe to call repeatedly. Scans both
`eval_outputs_v3/` and `eval_outputs_v3_n50/`.

`build_index_v3.py` highlights the best-row per task using **direction-aware**
primaries (`build_index_v3.py:PRIMARY_SPEC`):
- nvs   — PSNR ↑ (max)
- pose  — ATE ↓ (min)        ← V3 metric is loss-style, lower is better
- depth — δ<1.25 ↑ (max)     ← canonical depth headline

The same logic applies to `build_index_v3_n50.py`.

All three aggregators (`collect_results_v3.py`, `build_index_v3.py`,
`build_index_v3_n50.py`) and `colorize_depth_v3.py` are invoked automatically
at the end of `run_all_v3.sh` / `run_all_v3_n50.sh`.

---

## §7 Outputs

```
eval_outputs_v3/                      → /n/netscratch/.../generative_baselines_eval_v3
├── re10k/
│   ├── ours_pose_depth/sample_*/        # videos, npz, prompt.txt, camera_trajectory.png
│   ├── ours_nvs/sample_*/
│   ├── ours_pose_eval/sample_*/         # pose_metrics.json {ate, rpe_trans, rpe_rot}
│   ├── ours_nvs_eval/per_sample_metrics.csv
│   ├── geo4d_pose/, raydiffusion_pose/
│   ├── dfot_nvs/, seva_nvs/, wan_flf_nvs/
│   └── ...
├── dl3dv/ ... (+ ours_depth_eval, geo4d_depth, chronodepth_depth)
└── ...

eval_logs_v3/<dataset>.log            # per-dataset stdout/stderr
RESULTS_V3.md                         # per-task tables (markdown)
index_v3.html                         # browsable per-scene comparison
```

---

## §8 Implementation order

1. `geo4d_eval/vo_eval.py`, `geo4d_eval/depth_eval.py` (verbatim copies).
2. `eval_common_v3.py` (the two thin wrappers + aggregate + load_depth helper).
3. `eval_ours_pose_v3.py`, `eval_ours_depth_v3.py` (new wrappers for our predictions).
4. `geo4d/eval_geo4d_pose_v3.py`, `geo4d/eval_geo4d_depth_v3.py` (`cp` + swap metric block).
5. `RayDiffusion/eval_raydiffusion_pose_v3.py` (`cp` + swap metric block).
6. `ChronoDepth/eval_chronodepth_depth_aria_v3.py` (`cp` + swap metric block).
7. `run_dataset_v3.sh` (`cp` from v2, swap eval calls + OUT path).
8. `run_all_v3.sh` (`cp` from v2, point at v3 runner).
9. `collect_results_v3.py`, `build_index_v3.py` (`cp` + new metric keys + direction-aware highlight).
10. `colorize_depth_v3.py` (turns baseline depth npz → mp4 for the index).
11. Smoke-test on `re10k` GPU 0; if numbers look sane, fire `run_all_v3.sh`.

For an n=50 follow-up sweep, mirror these into `*_v3_n50` variants — see the
`run_all_v3_n50.sh` / `run_dataset_v3_n50.sh` / `collect_results_v3_n50.py`
/ `build_index_v3_n50.py` already in the repo.

---

## §9 Open questions before implementing

(carried for reference; nothing blocks step 1.)

- **Per-sample depth output format from our inference scripts:** confirm whether
  ours dumps `pred_depth.npz` (single T×H×W tensor) or per-frame `*.npy`.
  `eval_ours_depth_v3.py` will adapt — but stack-then-LAD2-once either way.
- **GT depth mask for ours** — same mask the V2 `eval_rgb_to_ray_depth_depth.py`
  uses. Re-use whatever it loaded.
- **`evo` install:** new dep. Add `pip install evo --upgrade` to env bootstrap.
