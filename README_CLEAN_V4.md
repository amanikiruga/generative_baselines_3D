# Generative-baselines evaluation — V4

Three-checkpoint evaluation of our model vs published baselines, across the same
10 datasets as V3 (camera-pose, depth, NVS), driven from a frozen
**metadata-pinned video set** (`metadata_v3_n50.json`) so every run consumes the
exact same GT inputs.

V4 successor to `README_CLEAN_V3.md`. **Key differences (V4 vs V3):**

| | V3 | V4 |
|---|---|---|
| Our checkpoints | 1 (`OURS_FINAL` = 2026-05-03/20-03-01) | **3:** `OURS_FINAL` (NEW), `OURS_NVS`, `OURS_FINAL_OLD` |
| Inference input source | dataset loaders (re-decoded each run) | **frozen GT mp4/npz from `metadata_v3_n50.json`** (same videos every run) |
| Ours inference scripts | `scripts/inference_single_gpu_rgb_to_ray_depth_aria_eval.py` etc. (in model repo) | **NEW thin wrappers** under `generative_baselines/ours_v4_infer/` that read the metadata index instead of the dataloader. Model code untouched. |
| Aria correction | baked in at infer time inside `aria.py` | **same** (unchanged from V3) |
| Pose / depth / NVS metric definitions | Geo4D `vo_eval` + `depth_eval` + offline NVS | **same** (unchanged from V3) |
| Eval code | `geo4d_eval/`, `eval_common_v3.py`, etc. | **reused verbatim** — V4 only adds a 3-method aggregator pass |
| Output dir | `eval_outputs_v3_n50` | **`eval_outputs_v4_n50`** (`/n/netscratch/.../generative_baselines_eval_v4_n50`) |
| index.html | `index_v3_n50.html` | `index_v4_n50.html` |
| Aggregator | `collect_results_v3_n50.py` / `build_index_v3_n50.py` | `collect_results_v4_n50.py` / `build_index_v4_n50.py` |
| Sample count | n=50 | **first 10 to start; n=50 once parity verified** |

V3 outputs at `eval_outputs_v3_n50/` are preserved. V4 writes a fresh tree.

**No V3 file is modified.** Every V4 artefact is a new copy: ours-inference
wrappers, dataset runner, launch script, aggregators, and HTML index. The
`geo4d_eval/`, `eval_common_v3.py`, `eval_ours_pose_v3.py`, `eval_ours_depth_v3.py`,
`geo4d/eval_geo4d_*_v3.py`, `RayDiffusion/eval_raydiffusion_pose_v3.py`,
`ChronoDepth/eval_chronodepth_depth_aria_v3.py`, `colorize_depth_v3.py`,
`eval_depth_disp_offline_v3.py` are reused as-is — they only consume directory
trees, so pointing them at `eval_outputs_v4_n50/<dataset>/<method>/...` works
without modification.

---

## §0 Quick start

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines
# Step 0 — parity check (run BEFORE any new-ckpt sweep)
nohup ./run_parity_v4.sh > eval_logs_v4_n50/_parity.log 2>&1
# Step 1 — first-10-scene sweep across all 3 ckpts + baselines
nohup ./run_all_v4_n50.sh --max_samples 10 > eval_logs_v4_n50/_run_all_n10.log 2>&1 &
# Step 2 — full n=50 sweep (only after step 1 looks sane)
nohup ./run_all_v4_n50.sh --max_samples 50 > eval_logs_v4_n50/_run_all_n50.log 2>&1 &
```

Watch:

```bash
tail -f eval_logs_v4_n50/_run_all_n10.log
ls eval_logs_v4_n50/
nvidia-smi
```

---

## §1 Checkpoint paths

```bash
# NEW primary
export OURS_FINAL=/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model/outputs/2026-05-04/12-34-53/checkpoints/last.ckpt
# NVS-tuned
export OURS_NVS=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-04-29/14-31-15/checkpoints/last.ckpt
# Previous final (V3) — kept as a regression anchor
export OURS_FINAL_OLD=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/outputs/2026-05-03/20-03-01/checkpoints/last.ckpt

export GEO4D_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D/checkpoints/geo4d/model.ckpt
export DFOT_RE10K_CKPT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/diffusion-forcing-transformer/pretrained/DFoT_RE10K.ckpt
export RAYDIFF_DIR=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion

export METADATA_V4=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/metadata_v3_n50.json
export OUT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs_v4_n50
```

The V4 sweep reuses `metadata_v3_n50.json` and `metadata_v3_n50/` verbatim
(the GT artifacts they index are dataset-derived, not checkpoint-derived). No
`metadata_v4*.json` is generated — V4 *consumes* the V3 index, it does not
rebuild it.

---

## §2 Datasets and task coverage

(unchanged from V3 — same 10 datasets, same task gates.)

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

## §3 The metadata-driven inference layer (the part most likely to go wrong)

### Goal

Pin every method's input to the same set of mp4/npz files so:
1. No re-decoding of source datasets between runs.
2. The 3 ours checkpoints + baselines all consume bit-identical GT.
3. We can swap checkpoints without touching the model repo or the dataloaders.

### Hard constraint: in-distribution behaviour must be preserved

The existing model expects tensors with a specific layout, normalization, frame
count, and conditioning prompt. The V3 dataloader produces them. If the V4
metadata-loader produces anything different — even subtly — the model goes
out-of-distribution and numbers become meaningless.

### Design — minimal-delta wrappers

Two new files in `generative_baselines/ours_v4_infer/`:

```
ours_v4_infer/
├── __init__.py
├── metadata_loader.py            ← single source of truth for "from-mp4" input
└── (no model code — wrappers below import from world_model_4d.scripts)
```

**`metadata_loader.py`** exposes one class:

```python
class MetadataIterableDataset(torch.utils.data.IterableDataset):
    """
    Drop-in replacement for the dataset object the existing inference scripts
    iterate over. Yields the SAME dict keys, dtypes, shapes, and value ranges
    that the V3 dataloader yields — but reads from disk paths captured in
    metadata_v3_n50.json instead of decoding from the source dataset.

    Per sample, returns:
      - rgb        : (T, 3, H, W) float32 in [-1, 1]   ← from gt_rgb.mp4
      - ray_d      : (T, 3, H, W) float32              ← from gt_ray_d.mp4
      - ray_m      : (T, 3, H, W) float32              ← from gt_ray_m.mp4
      - depth      : (T, 1, H, W) float32 (metric m)   ← from gt_depth_raw.npz
                       (decoded via eval_common_v3.load_depth_metric_from_npz —
                        same helper the V3 depth eval already uses, which means
                        the conversion is verified by V3's regression numbers)
      - intrinsics : (T, 3, 3) float32                 ← from gt_cameras.npz["K"]
      - c2w        : (T, 4, 4) float32                 ← from gt_cameras.npz["c2w"]
      - prompt     : str                               ← from prompt.txt
      - sample_id  : str                               ← from metadata
      - dataset    : str                               ← from metadata key
    """
```

**Wrappers.** Each wrapper is a `cp` of the corresponding model-repo script
with **exactly two** edits — nothing else:

```
ours_v4_infer/
├── inference_rgb_to_ray_depth_v4.py     ← cp of world_model_4d/.../scripts/
│                                          inference_single_gpu_rgb_to_ray_depth_aria_eval.py
├── inference_ray_to_rgb_depth_v4.py     ← cp of world_model_4d/.../scripts/
│                                          inference_single_gpu_ray_to_rgb_depth_eval.py
```

Edits inside each wrapper:

1. **Replace dataset construction.** The original script builds a dataset via
   the project's `build_dataset(dataset_name, split=…, max_samples=…, seed=…,
   no_augmentations=True)` call. We replace just that one call with:
   ```python
   from generative_baselines.ours_v4_infer.metadata_loader import \
       MetadataIterableDataset
   dataset = MetadataIterableDataset(
       metadata_json=os.environ["METADATA_V4"],
       dataset=args.dataset,
       task=args.task,                   # "pose_depth" or "nvs"
       max_samples=args.max_samples,
   )
   ```
   The DataLoader (`batch_size=1`, `num_workers=0`, `shuffle=False`) wrapping
   it stays identical.

2. **Replace `--no_augmentations` plumbing with a no-op.** The metadata loader
   never augments — augmentation is irrelevant here — so the flag is accepted
   for CLI compatibility but ignored.

Everything else — preprocessing, normalization, model construction, ckpt load,
sampler, autocast policy, seed handling, output mp4 encoding, npz layout — is
**byte-identical to the V3 script**. The diff `git diff
inference_single_gpu_rgb_to_ray_depth_aria_eval.py inference_rgb_to_ray_depth_v4.py`
should be ~30 lines, all in the `dataset = …` region plus the import block.

### Verification protocol — `run_parity_v4.sh`

A script that establishes the V4 wrappers behave like the V3 dataloader path:

1. Run `inference_rgb_to_ray_depth_v4.py` with `OURS_FINAL_OLD` (the V3 ckpt)
   on `re10k`, `n=10`, seed 42, `--no_augmentations`. Output goes to
   `eval_outputs_v4_n50/_parity/re10k/ours_pose_depth/`.
2. Diff against V3's cached predictions at
   `eval_outputs_v3_n50/re10k/ours_pose_depth/sample_*/pred_cameras.npz` and
   `pred_depth_raw.npz`:
   - `pred_cameras.npz["c2w"]` must match within `atol=1e-4` (sampler is
     deterministic given same seed + same input + same ckpt; any drift means
     the input tensor diverged).
   - `pred_depth_raw.npz["disparity"]` must match within `atol=1e-3`.
3. If parity fails, **stop and inspect** — do not proceed to the n=50 sweep.
   The most likely failure modes (in priority order) to diagnose:
   - mp4 decode RGB range/order: the decoder must yield `[-1, 1]` float, not
     `[0, 255]` uint8 or `[0, 1]` float. Check the V3 dataloader's exact
     normalization step.
   - depth npz unpacking: scale + normalized-disparity round-trip via
     `load_depth_metric_from_npz` (V3's verified path). Do NOT reimplement.
   - frame count / temporal stride: re10k uses 16 frames at a specific stride;
     metadata mp4s are already pre-sliced at this stride (V3 wrote them this
     way). Confirm `T == 16` for re10k samples.
   - prompt text: byte-identical to what the V3 dataloader passed to the text
     encoder. Read prompt.txt with `open(p).read().strip()` — same as V3.

Only when the parity diff passes does V4 become a meaningful sweep.

---

## §4 New eval code layout

```
generative_baselines/
├── ours_v4_infer/                          ← NEW
│   ├── __init__.py
│   ├── metadata_loader.py
│   ├── inference_rgb_to_ray_depth_v4.py    ← cp + dataset swap
│   └── inference_ray_to_rgb_depth_v4.py    ← cp + dataset swap
│
├── geo4d_eval/                             ← reused from V3
├── eval_common_v3.py                       ← reused from V3
├── eval_ours_pose_v3.py                    ← reused from V3 (consumes pose_depth dir)
├── eval_ours_depth_v3.py                   ← reused from V3
├── geo4d/eval_geo4d_pose_v3.py             ← reused from V3
├── geo4d/eval_geo4d_depth_v3.py            ← reused from V3
├── RayDiffusion/eval_raydiffusion_pose_v3.py    ← reused from V3
├── ChronoDepth/eval_chronodepth_depth_aria_v3.py  ← reused from V3
├── colorize_depth_v3.py                    ← reused from V3 (scans v4 tree too — see §5)
├── eval_depth_disp_offline_v3.py           ← reused from V3 (--root flag)
│
├── run_parity_v4.sh                        ← NEW (§3 verification)
├── run_dataset_v4_n50.sh                   ← cp of run_dataset_v3_n50.sh, swap to v4
├── run_all_v4_n50.sh                       ← cp of run_all_v3_n50.sh, swap to v4
├── collect_results_v4_n50.py               ← cp of v3_n50, add 3 ours methods
├── build_index_v4_n50.py                   ← cp of v3_n50, add 3 ours methods + new title
├── RESULTS_V4_N50.md                       ← generated
└── index_v4_n50.html                       ← generated
```

`colorize_depth_v3.py` already scans both `eval_outputs_v3/` and
`eval_outputs_v3_n50/`. Add `eval_outputs_v4_n50/` to its scan list (a 1-line
change is acceptable here since the script is shared infrastructure; it does
not alter V3 behaviour).

---

## §5 Per-dataset pipeline (`run_dataset_v4_n50.sh`)

```bash
./run_dataset_v4_n50.sh <dataset> <gpu_id> [<max_samples>]
```

Order (serial within one dataset):

1. **Ours inference × 3 ckpts.** For each of `(ours_final, ours_nvs, ours_final_old)`:
   1. RGB→ray+depth: `ours_v4_infer/inference_rgb_to_ray_depth_v4.py
      --ckpt_path $CKPT --dataset $DATASET
      --metadata_json $METADATA_V4 --task pose_depth
      --output_dir $OUT/$DATASET/${METHOD}_pose_depth
      --max_samples $MAXN --seed 42 --no_augmentations`.
   2. Ray→RGB+depth (NVS-eligible datasets only):
      `ours_v4_infer/inference_ray_to_rgb_depth_v4.py …
      --task nvs --output_dir $OUT/$DATASET/${METHOD}_nvs …`.
   `${METHOD}` ∈ `{ours_final, ours_nvs, ours_final_old}`.
2. **Ours evals × 3 ckpts.** For each method, run the V3 ours-eval scripts
   pointed at that method's directory:
   - `eval_ours_pose_v3.py --input_dir $OUT/$DATASET/${METHOD}_pose_depth
     --output_dir $OUT/$DATASET/${METHOD}_pose_eval`
   - `eval_ours_depth_v3.py …` (depth datasets only)
   - `recompute_metrics_offline.py` for NVS (NVS-eligible datasets only)
3. **Pose baselines:** `eval_geo4d_pose_v3.py`, `eval_raydiffusion_pose_v3.py`
   — pointed at `$OUT/$DATASET/ours_final_pose_depth` as the `--input_dir`
   (the GT side is identical across our 3 methods, so picking any one method
   dir suffices; using `ours_final` keeps the dependency clear).
4. **Depth baselines:** `eval_geo4d_depth_v3.py`,
   `eval_chronodepth_depth_aria_v3.py` — same convention.
5. **NVS baselines:** DFoT, SEVA, Wan 2.1 FLF — pointed at
   `$OUT/$DATASET/ours_final_nvs` as `--input_dir`.

Re-use of V3 baseline caches: `run_dataset_v4_n50.sh` honours `V2_OUT` (set
to `eval_outputs_v3_n50`) and symlinks the same baseline cache subdirs the V3
runner does (`geo4d_pose/geo4d_raw`, per-sample `geo4d_depth/sample_*/pred_depth_geo4d.npz`,
NVS dirs whole). Baseline GT is unchanged across V3↔V4 because the metadata
files are unchanged, so cached predictions remain valid.

---

## §6 Aggregation

```bash
mamba run -n test2 python eval_depth_disp_offline_v3.py --root eval_outputs_v4_n50
mamba run -n test2 python colorize_depth_v3.py
mamba run -n test2 python collect_results_v4_n50.py    # writes RESULTS_V4_N50.md
mamba run -n test2 python build_index_v4_n50.py        # writes index_v4_n50.html
```

`collect_results_v4_n50.py` adds three header rows for ours:

| Method label | Source dir |
|---|---|
| `Ours (final)`     | `<dataset>/ours_final_*_eval` |
| `Ours (NVS)`       | `<dataset>/ours_nvs_*_eval`   |
| `Ours (final old)` | `<dataset>/ours_final_old_*_eval` |

`build_index_v4_n50.py` keeps the V3 direction-aware best-row highlighter
(`PRIMARY_SPEC` unchanged: NVS=PSNR↑, pose=ATE↓, depth=δ<1.25↑) but renders
the three ours rows side-by-side. All four aggregators
(`collect_results_v4_n50.py`, `build_index_v4_n50.py`, `colorize_depth_v3.py`,
`eval_depth_disp_offline_v3.py`) are invoked automatically at the end of
`run_all_v4_n50.sh`.

---

## §7 Outputs

```
eval_outputs_v4_n50/                    → /n/netscratch/.../generative_baselines_eval_v4_n50
├── _parity/                              # parity check artifacts (§3)
├── re10k/
│   ├── ours_final_pose_depth/sample_*/        # OURS_FINAL (NEW ckpt)
│   ├── ours_final_nvs/sample_*/
│   ├── ours_final_pose_eval/sample_*/
│   ├── ours_final_nvs_eval/per_sample_metrics.csv
│   ├── ours_nvs_pose_depth/sample_*/          # OURS_NVS ckpt
│   ├── ours_nvs_nvs/sample_*/
│   ├── ours_nvs_pose_eval/sample_*/
│   ├── ours_nvs_nvs_eval/per_sample_metrics.csv
│   ├── ours_final_old_pose_depth/sample_*/    # OURS_FINAL_OLD (V3 ckpt)
│   ├── ours_final_old_nvs/sample_*/
│   ├── ours_final_old_pose_eval/sample_*/
│   ├── ours_final_old_nvs_eval/per_sample_metrics.csv
│   ├── geo4d_pose/, raydiffusion_pose/        # V3 caches symlinked here
│   ├── dfot_nvs/, seva_nvs/, wan_flf_nvs/
│   └── ...
├── dl3dv/ ...
└── ...

eval_logs_v4_n50/<dataset>.log            # per-dataset stdout/stderr
RESULTS_V4_N50.md                         # per-task tables (markdown)
index_v4_n50.html                         # browsable per-scene comparison
```

---

## §8 Implementation order

1. `ours_v4_infer/metadata_loader.py` — implement `MetadataIterableDataset`. Verify
   in isolation (unit script) that for one re10k sample the yielded dict matches
   the V3 dataloader's dict key-by-key, dtype-by-dtype, shape-by-shape, and
   `torch.allclose` on each tensor.
2. `ours_v4_infer/inference_rgb_to_ray_depth_v4.py` — `cp` + dataset swap (§3).
3. `ours_v4_infer/inference_ray_to_rgb_depth_v4.py` — `cp` + dataset swap.
4. `run_parity_v4.sh` — runs §3.2 parity diff. **Block on this passing.**
5. `run_dataset_v4_n50.sh` — `cp` from v3_n50, loop over 3 ckpts, swap inference
   command to the V4 wrappers, leave baseline section unchanged.
6. `run_all_v4_n50.sh` — `cp` from v3_n50, repoint env exports, accept
   `--max_samples` flag (default 10 first, 50 once parity confirmed).
7. `collect_results_v4_n50.py`, `build_index_v4_n50.py` — `cp` + add 3 ours rows.
8. **First execution: `run_all_v4_n50.sh --max_samples 10`** on the same wave
   layout as V3 (4 GPUs × 2 datasets concurrent). Inspect `RESULTS_V4_N50.md`.
9. If sane: rerun with `--max_samples 50` (same script, samples 0..9 cache so
   only 10..49 do model inference for ours; baselines symlink from V3 n=50).

For the 3-ckpt iteration in step 1 of `run_dataset_v4_n50.sh`, run **serially
within a single GPU** (not parallel) so we don't OOM. Wave-level parallelism
across 8 datasets × 4 GPUs is unchanged from V3.

---

## §9 Open questions / risks before implementing

(carried for reference; only items with a "BLOCKER" tag block step 1.)

- **(BLOCKER) RGB mp4 decode parity.** The V3 dataloader's RGB normalization
  pipeline must be replicated exactly. Identify it by reading the V3 dataset
  class for `re10k` and copy the normalization helper into
  `metadata_loader.py` verbatim — do not reimplement from imagined defaults.
- **(BLOCKER) ray_d / ray_m mp4 decode.** These are not natural images;
  they're encoded direction/mask tensors. Confirm the encoding scheme used
  when V3 wrote `gt_ray_d.mp4` / `gt_ray_m.mp4`, and decode with the inverse.
  If lossy mp4 round-trip is unacceptable for rays, fall back to deriving
  rays from `gt_cameras.npz` intrinsics+extrinsics on the fly (the V3
  dataloader did this; it's also bit-exact and removes the codec dependency).
  **Default plan: derive rays from cameras, do not decode the ray mp4s.**
- **GT depth decode.** Use `eval_common_v3.load_depth_metric_from_npz` —
  already verified by the V3 depth-eval numbers.
- **`OURS_NVS` checkpoint compatibility.** Confirm the 2026-04-29 ckpt has
  the same head dimensions as `OURS_FINAL` so the same inference script
  loads it without arch surgery. If not, fall back to running it through
  the `inference_ray_to_rgb_depth_v4.py` path only and skip its
  pose/depth row.
- **`OURS_FINAL_OLD` path note.** The V3 README listed
  `outputs/2026-05-03/20-03-01/last.ckpt`; the V4 spec uses
  `outputs/2026-05-03/20-03-01/checkpoints/last.ckpt` (note `checkpoints/`).
  Verify which file actually exists and adjust the export accordingly — the
  parity test's "regenerate the V3 numbers" requirement makes this a hard
  pin: the wrapper must load the same weights V3 loaded.
