#!/usr/bin/env python3
"""Generate index.html: clean results page with per-task tables + sample carousels.

Layout:
  - Section header "Results"
  - For each dataset (DL3DV, RE10K, Scannet++): tasks rendered in nvs/pose/depth order
  - Each task: per-dataset metric table + a carousel iterating over all samples,
    each slide showing the method-by-method comparison (videos / camera-trajectory PNGs).

Reads:
  - eval_outputs/<dataset>/<method_dir>/sample_*/  (media + per-sample dirs)
  - eval_outputs/<dataset>/<eval_dir>/             (aggregate metrics)

Writes:
  - index.html
"""
import json, csv, os, re
from pathlib import Path

ROOT = Path("/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines")
EVAL = ROOT / "eval_outputs"


# ---------------- numbers ----------------
def read_json(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None

def avg_csv(path, fields):
    if not Path(path).exists(): return None
    rows = list(csv.DictReader(open(path)))
    if not rows: return None
    return {f: sum(float(r[f]) for r in rows if r.get(f) not in (None,""))/max(1,len([r for r in rows if r.get(f) not in (None,"")])) for f in fields}

def nvs(dirp):
    a = avg_csv(Path(dirp)/"per_sample_metrics.csv", ["avg_psnr","avg_lpips","avg_ssim"])
    if a: return a["avg_psnr"], a["avg_lpips"], a["avg_ssim"]
    for c in ("metrics.json","summary.json"):
        d = read_json(Path(dirp)/c)
        if d and "psnr" in d and "lpips" in d and "ssim" in d:
            return d["psnr"], d["lpips"], d["ssim"]
    # Wan-FLF-style: aggregate per-sample metrics.csv (avg_per_frame_*).
    p = Path(dirp)
    if p.exists():
        psnrs, lpipss, ssims = [], [], []
        for sd in sorted(p.glob("sample_*")):
            csv_path = sd / "metrics.csv"
            if not csv_path.exists():
                continue
            try:
                row = next(csv.DictReader(csv_path.open()))
                psnrs.append(float(row["avg_per_frame_psnr"]))
                lpipss.append(float(row["avg_per_frame_lpips"]))
                ssims.append(float(row["avg_per_frame_ssim"]))
            except Exception:
                continue
        if psnrs:
            n = len(psnrs)
            return sum(psnrs)/n, sum(lpipss)/n, sum(ssims)/n
    return None

def pose(dirp):
    for c in ("final_stats.json","aggregate_metrics.json"):
        d = read_json(Path(dirp)/c)
        if not d: continue
        a3 = d.get("auc03", d.get("auc3"))
        a30 = d.get("auc30")
        if a3 is not None and a30 is not None:
            return a3, a30
    return None

def depth(dirp):
    for c in ("final_stats.json","geo4d_final_stats.json","chronodepth_final_stats.json"):
        d = read_json(Path(dirp)/c)
        if not d: continue
        if "aligned" in d:
            a = d["aligned"]; return a["d1"], a["d2"], a["d3"]
        if "d1" in d:
            return d["d1"], d["d2"], d["d3"]
    return None


def fmt(t, fmts):
    if t is None: return '<span class="na">&mdash;</span>'
    return " / ".join(f.format(v) for f, v in zip(fmts, t))


# ---------------- task config ----------------
# Per-task: list of (display_label, method_dir_for_media, eval_dir_for_metrics, media_filename, kind)
NVS_METHODS = [
    ("Ground truth",    "ours_nvs",              None,                       "gt_rgb.mp4",      "video"),
    ("Ours",            "ours_nvs",              "ours_nvs_eval",            "pred_rgb.mp4",    "video"),
    ("Ours (NVS-only)", "ours_nvs_only_nvs",     "ours_nvs_only_nvs_eval",   "pred_rgb.mp4",    "video"),
    ("Wan 2.1 FLF",     "wan_flf_nvs",           "wan_flf_nvs",              "pred_rgb.mp4",    "video"),
    ("DFoT (only 8 frames)", "dfot_nvs",         "dfot_nvs",                 "pred_rgb.mp4",    "video"),
    ("SEVA",            "seva_nvs",              "seva_nvs",                 "samples-rgb.mp4", "video"),
    ("GEN3C",           "gen3c_nvs",             "gen3c_nvs",                "pred_rgb.mp4",    "video"),
]
POSE_METHODS = [
    ("Ground truth (RGB)", "ours_pose_depth",         None,                        "gt_rgb.mp4",            "video"),
    ("Ours",            "ours_pose_depth",            "ours_pose_eval",            "camera_trajectory.png", "img"),
    ("Ours (NVS-only)", "ours_nvs_only_pose_depth",   "ours_nvs_only_pose_eval",   "camera_trajectory.png", "img"),
    ("GEO4D",           "geo4d_pose",                 "geo4d_pose",                "camera_trajectory.png", "img"),
    ("RayDiffusion",    "raydiffusion_pose",          "raydiffusion_pose",         "camera_trajectory.png", "img"),
]
DEPTH_METHODS = [
    ("Ground truth (RGB)", "ours_pose_depth",         None,                        "gt_rgb.mp4",            "video"),
    ("GT depth",        "ours_depth_eval",            None,                        "gt_depth.mp4",          "video"),
    # Use the eval-script-generated aligned video (same colorization as baselines),
    # not the inference-script raw video.
    ("Ours",            "ours_depth_eval",            "ours_depth_eval",           "pred_depth.mp4",        "video"),
    ("Ours (NVS-only)", "ours_nvs_only_depth_eval",   "ours_nvs_only_depth_eval",  "pred_depth.mp4",        "video"),
    ("ChronoDepth",     "chronodepth_depth",          "chronodepth_depth",         "pred_depth_aligned.mp4","video"),
    # GEO4D depth: prefer aligned video if present, fall back to comparison PNG.
    ("GEO4D",           "geo4d_depth",                "geo4d_depth",               ["pred_depth.mp4", "depth_comparison.png"], "auto"),
]

TASK_METHODS = {"nvs": NVS_METHODS, "pose": POSE_METHODS, "depth": DEPTH_METHODS}
TASK_TITLES  = {"nvs": "Novel-view synthesis", "pose": "Camera pose", "depth": "Depth"}
TASK_HEADERS = {
    "nvs":   ("PSNR &uarr; / LPIPS &darr; / SSIM &uarr;", "PSNR / LPIPS / SSIM",
              nvs,   ("{:.2f}", "{:.3f}", "{:.3f}")),
    "pose":  ("AUC@3&deg; &uarr; / AUC@30&deg; &uarr;",   "AUC@3&deg; / AUC@30&deg;",
              pose,  ("{:.2f}", "{:.2f}")),
    "depth": ("&delta;1 &uarr; / &delta;2 &uarr; / &delta;3 &uarr;", "&delta;1 / &delta;2 / &delta;3",
              depth, ("{:.3f}", "{:.3f}", "{:.3f}")),
}

# Order requested: DL3DV, RE10K, Scannet++; within each, only tasks that exist.
DATASET_ORDER = [
    ("scenenet_depth", "SceneNet (synthetic, indoor)", ["depth"]),
    ("vkitti2",        "Virtual KITTI 2 (synthetic, outdoor)", ["nvs", "pose", "depth"]),
    ("aria",           "Aria (synthetic-grade indoor)",        ["nvs", "pose", "depth"]),
    ("dl3dv",          "DL3DV",                                ["nvs", "pose", "depth"]),
    ("dl3dv_test",     "DL3DV (test split)",                   ["nvs", "pose"]),
    ("tanksandtemples","Tanks & Temples",                      ["nvs", "pose"]),
    ("re10k",          "RealEstate10K",                        ["nvs", "pose"]),
    ("scannetpp",      "Scannet++",                            ["nvs", "pose", "depth"]),
    ("spatialvid_nvs", "SpatialVid",                           ["nvs", "pose"]),
    ("agibot_world",   "AgiBotWorld (static head-camera)",     ["nvs", "pose"]),
]


# ---------------- discovery ----------------
def count_samples(dataset, task):
    """Number of sample_* dirs under the primary method dir for this (dataset, task)."""
    primary = {"nvs": "ours_nvs", "pose": "ours_pose_depth", "depth": "ours_pose_depth"}[task]
    base = EVAL / dataset / primary
    if not base.exists(): return 0
    return sum(1 for p in base.iterdir() if p.is_dir() and p.name.startswith("sample_"))


def js_recipe_for(dataset, task):
    """Build a JS arrow function `(i) => [[label, kind, path], ...]` for one (dataset,task)."""
    methods = TASK_METHODS[task]
    entries = []
    for label, mdir, _edir, fname, kind in methods:
        # Skip methods whose root dir doesn't exist for this dataset.
        if not (EVAL / dataset / mdir).exists():
            continue
        if kind == "auto":
            # Pick first existing filename across samples — assume sample_00000 representative.
            chosen = None
            for cand in fname:
                if (EVAL / dataset / mdir / "sample_00000" / cand).exists():
                    chosen = cand
                    break
            if chosen is None:
                continue
            actual_kind = "video" if chosen.endswith(".mp4") else "img"
            path_tpl = f"eval_outputs/{dataset}/{mdir}/sample_${{i}}/{chosen}"
            entries.append(f'    [{json.dumps(label)}, {json.dumps(actual_kind)}, `{path_tpl}`],')
        else:
            path_tpl = f"eval_outputs/{dataset}/{mdir}/sample_${{i}}/{fname}"
            entries.append(f'    [{json.dumps(label)}, {json.dumps(kind)}, `{path_tpl}`],')
    body = "\n".join(entries)
    return f"(i) => [\n{body}\n  ]"


# ---------------- HTML ----------------
def build_metric_table(dataset, task):
    sub_units, col_header, getter, fmts = TASK_HEADERS[task]
    methods = TASK_METHODS[task]
    # Primary-metric index per task (higher is always better for these).
    PRIMARY_IDX = {"nvs": 0, "pose": 1, "depth": 0}  # PSNR / AUC@30° / δ1
    primary_idx = PRIMARY_IDX[task]

    # Collect (label, metric tuple) for ranking.
    table_rows = []
    for label, _mdir, edir, _fname, _kind in methods:
        if label in ("Ground truth", "GT depth"):
            continue
        if edir is None:
            continue
        m = getter(EVAL / dataset / edir) if (EVAL / dataset / edir).exists() else None
        table_rows.append((label, m))

    # Find the row with the best primary-metric value.
    best_label = None
    best_val = float("-inf")
    for label, m in table_rows:
        if m is None:
            continue
        v = m[primary_idx]
        if v is None:
            continue
        if v > best_val:
            best_val = v
            best_label = label

    rows = []
    for label, m in table_rows:
        cls = ' class="ours"' if label == best_label else ""
        rows.append(
            f'<tr{cls}><td class="method">{label}</td>'
            f'<td class="num">{fmt(m, fmts)}</td></tr>'
        )
    return (
        f'<div class="task-meta">{sub_units}</div>'
        f'<table class="metric-table">'
        f'<tr><th>Method</th><th>{col_header}</th></tr>'
        f'{"".join(rows)}'
        f'</table>'
    )


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>generative baselines comparison</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
  html, body { background: #fff !important; color: #111 !important; }
  body { font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         font-size: 15px; line-height: 1.6; }
  .results-section { padding: 48px 24px 80px; max-width: 1280px; margin: 0 auto; }
  .results-section h2.section-title {
    font-family: "Source Serif 4", Georgia, serif;
    font-size: 36px; font-weight: 700; color: #111;
    text-align: center; margin: 0 0 36px; letter-spacing: -0.01em;
  }
  .subsection-title { font-family: "Source Serif 4", Georgia, serif;
    font-size: 24px; font-weight: 700; color: #111;
    margin: 40px 0 6px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }
  .dataset-title { font-size: 18px; font-weight: 600; color: #111; margin: 24px 0 6px; }
  .task-meta { color: #6b7280; font-size: 13px; margin-bottom: 8px; }

  table.metric-table { border-collapse: collapse; margin: 6px 0 16px; font-size: 13px;
                       background: #fff; border: 1px solid #e5e7eb; }
  table.metric-table th, table.metric-table td {
    padding: 6px 14px; border-bottom: 1px solid #e5e7eb; text-align: left; color: #111;
  }
  table.metric-table th { background: #f7f7f8; font-weight: 600;
    font-size: 11px; letter-spacing: 0.04em; text-transform: uppercase; color: #4b5563; }
  table.metric-table td.method { font-weight: 600; }
  table.metric-table tr.ours td { background: #fff7d6; }
  table.metric-table td.num { font-variant-numeric: tabular-nums; font-family: "SF Mono", Menlo, monospace; }
  .na { color: #cbd5e1; }

  .carousel { position: relative; background: #fafafa; border: 1px solid #e5e7eb;
              border-radius: 8px; padding: 18px 18px 12px; margin: 8px 0 12px; }
  .carousel-inner { overflow: hidden; }
  .carousel-track { display: flex; transition: transform 0.35s ease; }
  .carousel-slide { flex: 0 0 100%; min-width: 0; }
  .method-grid { display: grid; gap: 10px;
                 grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
  .method-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; }
  .method-card .label { font-size: 12px; font-weight: 600; color: #111;
                        padding: 6px 10px; background: #f7f7f8; border-bottom: 1px solid #e5e7eb;
                        letter-spacing: 0.02em; }
  .method-card video, .method-card img { display: block; width: 100%; height: auto;
                                         background: #f3f4f6; }

  .carousel-controls { display: flex; align-items: center; justify-content: center;
                       gap: 14px; margin-top: 12px; }
  .carousel-arrow { background: #fff; border: 1px solid #e5e7eb; border-radius: 999px;
                    width: 34px; height: 34px; cursor: pointer; color: #111;
                    display: flex; align-items: center; justify-content: center; }
  .carousel-arrow:hover { background: #f3f4f6; }
  .carousel-arrow:disabled { opacity: 0.3; cursor: default; }
  .carousel-dots { display: flex; gap: 6px; }
  .carousel-dot { width: 8px; height: 8px; border-radius: 50%; background: #d1d5db;
                  border: 0; padding: 0; cursor: pointer; }
  .carousel-dot.active { background: #111; }
  .carousel-counter { font-size: 12px; color: #6b7280; font-variant-numeric: tabular-nums; min-width: 48px; text-align: center; }
</style>
</head>
<body>

<section class="results-section">
  <h2 class="section-title">Results</h2>
__BODY__
</section>

<script>
const RECIPES = {
__RECIPES__
};

const SLOW_DATASETS = new Set(["dl3dv", "dl3dv_test", "tanksandtemples"]);
const SLOW_RATE = 0.5;

function pad5(n) { return String(n).padStart(5, "0"); }

function makeMediaCard(label, kind, src, slow) {
  const card = document.createElement("div");
  card.className = "method-card";
  const lab = document.createElement("div");
  lab.className = "label"; lab.textContent = label;
  card.appendChild(lab);
  let media;
  if (kind === "video") {
    media = document.createElement("video");
    media.muted = true; media.loop = true; media.playsInline = true;
    media.autoplay = true; media.preload = "metadata";
    media.src = src;
    if (slow) {
      media.addEventListener("loadedmetadata", () => { media.playbackRate = SLOW_RATE; });
    }
  } else {
    media = document.createElement("img");
    media.loading = "lazy"; media.src = src;
  }
  card.appendChild(media);
  return card;
}

function buildCarousel(el) {
  const dataset = el.dataset.dataset;
  const task = el.dataset.task;
  const total = parseInt(el.dataset.samples, 10);
  const recipe = RECIPES[`${dataset}::${task}`];
  if (!recipe || !total) return;
  const slow = SLOW_DATASETS.has(dataset);

  const inner = document.createElement("div"); inner.className = "carousel-inner";
  const track = document.createElement("div"); track.className = "carousel-track";
  inner.appendChild(track);

  for (let i = 0; i < total; i++) {
    const slide = document.createElement("div"); slide.className = "carousel-slide";
    const grid = document.createElement("div"); grid.className = "method-grid";
    for (const [label, kind, path] of recipe(pad5(i))) {
      grid.appendChild(makeMediaCard(label, kind, path, slow));
    }
    slide.appendChild(grid);
    track.appendChild(slide);
  }

  const controls = document.createElement("div"); controls.className = "carousel-controls";
  const prev = document.createElement("button");
  prev.className = "carousel-arrow"; prev.innerHTML = '<i class="fas fa-chevron-left"></i>';
  const next = document.createElement("button");
  next.className = "carousel-arrow"; next.innerHTML = '<i class="fas fa-chevron-right"></i>';
  const counter = document.createElement("div"); counter.className = "carousel-counter";
  const dots = document.createElement("div"); dots.className = "carousel-dots";
  controls.append(prev, dots, counter, next);

  el.append(inner, controls);

  let idx = 0;
  const dotEls = [];
  for (let i = 0; i < total; i++) {
    const d = document.createElement("button"); d.className = "carousel-dot";
    d.addEventListener("click", () => go(i));
    dots.appendChild(d); dotEls.push(d);
  }

  function go(i) {
    idx = Math.max(0, Math.min(total - 1, i));
    track.style.transform = `translateX(-${idx * 100}%)`;
    dotEls.forEach((d, k) => d.classList.toggle("active", k === idx));
    counter.textContent = `${idx + 1} / ${total}`;
    prev.disabled = idx === 0;
    next.disabled = idx === total - 1;
  }
  prev.addEventListener("click", () => go(idx - 1));
  next.addEventListener("click", () => go(idx + 1));
  go(0);
}

function initAll() {
  document.querySelectorAll(".carousel:not([data-built])").forEach(el => {
    el.setAttribute("data-built", "1");
    buildCarousel(el);
  });
}
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initAll);
} else {
  initAll();
}
new MutationObserver(initAll).observe(document.documentElement, { childList: true, subtree: true });
</script>

</body>
</html>
"""


def main():
    body_parts = []
    recipes = []

    for ds_key, ds_label, tasks in DATASET_ORDER:
        body_parts.append(f'  <h3 class="subsection-title">{ds_label}</h3>')
        for task in tasks:
            n = count_samples(ds_key, task)
            if n == 0:
                continue
            body_parts.append(f'  <h4 class="dataset-title">{TASK_TITLES[task]}</h4>')
            body_parts.append("  " + build_metric_table(ds_key, task))
            body_parts.append(
                f'  <div class="carousel" data-samples="{n}" '
                f'data-dataset="{ds_key}" data-task="{task}"></div>'
            )
            recipes.append(f'  "{ds_key}::{task}": {js_recipe_for(ds_key, task)},')

    html = (HTML_TEMPLATE
            .replace("__BODY__", "\n".join(body_parts))
            .replace("__RECIPES__", "\n".join(recipes)))

    out = ROOT / "index.html"
    out.write_text(html)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
