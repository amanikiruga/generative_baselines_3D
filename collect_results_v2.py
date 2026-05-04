#!/usr/bin/env python3
"""Walk eval_outputs/<dataset>/<method>_eval/ for metric files and emit RESULTS.md."""
import json, csv, os
from pathlib import Path

ROOT = Path("/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs_v2")

# (dataset, method, eval_dir_relative_to_dataset, kind)
# kind: nvs (per_sample_metrics.csv → PSNR/LPIPS/SSIM)
#       pose (aggregate_metrics.json or "Final" log → AUC03/AUC30)
#       depth (json with raw/aligned d1 d2 d3)


def avg_csv(path: Path, fields: list[str]) -> dict[str, float] | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return None
    out = {}
    for f in fields:
        vals = [float(r[f]) for r in rows if r.get(f) not in (None, "")]
        out[f] = sum(vals) / len(vals) if vals else float("nan")
    return out


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def collect_nvs(eval_dir: Path) -> str:
    a = avg_csv(eval_dir / "per_sample_metrics.csv", ["avg_psnr", "avg_lpips", "avg_ssim"])
    if a is not None:
        return f"{a['avg_psnr']:.2f} / {a['avg_lpips']:.3f} / {a['avg_ssim']:.3f}"
    # SEVA-style aggregate
    for cand in ("metrics.json", "summary.json"):
        d = read_json(eval_dir / cand)
        if d and "psnr" in d and "lpips" in d and "ssim" in d:
            return f"{d['psnr']:.2f} / {d['lpips']:.3f} / {d['ssim']:.3f}"
    # Wan-FLF-style: aggregate per-sample metrics.csv (schema: avg_per_frame_*)
    if eval_dir.exists():
        psnrs, lpipss, ssims = [], [], []
        for sd in sorted(eval_dir.glob("sample_*")):
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
            suffix = f" (n={n})" if n < 10 else ""
            return f"{sum(psnrs)/n:.2f} / {sum(lpipss)/n:.3f} / {sum(ssims)/n:.3f}{suffix}"
    return ""


def collect_pose(eval_dir: Path) -> str:
    for cand in ("final_stats.json", "aggregate_metrics.json", "summary.json", "metrics.json"):
        d = read_json(eval_dir / cand)
        if not d:
            continue
        # ours/raydiffusion use auc03; geo4d uses auc3
        a3 = d.get("auc03", d.get("auc3"))
        a30 = d.get("auc30")
        if a3 is not None and a30 is not None:
            return f"{a3:.2f} / {a30:.2f}"
    return ""


def collect_depth(eval_dir: Path) -> str:
    for cand in ("final_stats.json", "geo4d_final_stats.json", "chronodepth_final_stats.json",
                 "aggregate_metrics.json", "depth_metrics.json", "summary.json"):
        d = read_json(eval_dir / cand)
        if not d:
            continue
        if "aligned" in d:
            a = d["aligned"]
            return f"{a['d1']:.3f} / {a['d2']:.3f} / {a['d3']:.3f}"
        if "d1" in d:
            return f"{d['d1']:.3f} / {d['d2']:.3f} / {d['d3']:.3f}"
    return ""


def main():
    methods_nvs   = ["ours_nvs_eval", "wan_flf_nvs", "dfot_nvs", "seva_nvs"]
    methods_pose  = ["ours_pose_eval", "geo4d_pose", "raydiffusion_pose"]
    methods_depth = ["ours_depth_eval", "geo4d_depth", "chronodepth_depth"]
    datasets_nvs   = ["re10k", "dl3dv", "dl3dv_test", "tanksandtemples", "scannetpp", "vkitti2", "aria", "spatialvid_nvs", "agibot_world"]
    datasets_pose  = ["re10k", "dl3dv", "dl3dv_test", "tanksandtemples", "scannetpp", "vkitti2", "aria", "spatialvid_nvs", "agibot_world", "sintel", "eth3d", "dtu"]
    datasets_depth = ["scenenet_depth", "vkitti2", "aria", "dl3dv", "scannetpp", "kitti", "sintel", "eth3d", "dtu"]

    name_map = {
        "ours_nvs_eval": "Ours",
        "wan_flf_nvs": "Wan 2.1 FLF",
        "dfot_nvs": "DFoT", "seva_nvs": "SEVA",
        "ours_pose_eval": "Ours",
        "geo4d_pose": "GEO4D", "raydiffusion_pose": "RayDiffusion",
        "ours_depth_eval": "Ours",
        "geo4d_depth": "GEO4D", "chronodepth_depth": "ChronoDepth",
    }

    lines = []
    lines.append("# Results — generative baselines vs ours (1.3B mixture ckpt)")
    lines.append("")
    lines.append("Checkpoint: `outputs/2026-04-30/23-29-19/checkpoints/last_archive.ckpt` (OURS_FINAL — single ckpt, supersedes 1.3B mixture + nvs_only).")
    lines.append("Sample budget: 10 scenes per dataset (BFS pass).")
    lines.append("Output base: `eval_outputs/<dataset>/<method>/`.")
    lines.append("")
    lines.append("## NVS — interior-frame metrics (PSNR ↑ / LPIPS ↓ / SSIM ↑)")
    lines.append("")
    header = "| Method | " + " | ".join(d.upper() for d in datasets_nvs) + " |"
    sep    = "|--------|" + "|".join("-------" for _ in datasets_nvs) + "|"
    lines += [header, sep]
    for m in methods_nvs:
        row = [name_map[m]]
        for d in datasets_nvs:
            ev = ROOT / d / m
            row.append(collect_nvs(ev) or "")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Camera pose — relative pose AUC (AUC@3° ↑ / AUC@30° ↑)")
    lines.append("")
    header = "| Method | " + " | ".join(d for d in datasets_pose) + " |"
    sep    = "|--------|" + "|".join("---" for _ in datasets_pose) + "|"
    lines += [header, sep]
    for m in methods_pose:
        row = [name_map[m]]
        for d in datasets_pose:
            ev = ROOT / d / m
            row.append(collect_pose(ev) or "")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Depth — δ-thresholds, aligned scale+shift (δ1 ↑ / δ2 ↑ / δ3 ↑)")
    lines.append("")
    header = "| Method | " + " | ".join(d for d in datasets_depth) + " |"
    sep    = "|--------|" + "|".join("---" for _ in datasets_depth) + "|"
    lines += [header, sep]
    for m in methods_depth:
        row = [name_map[m]]
        for d in datasets_depth:
            ev = ROOT / d / m
            row.append(collect_depth(ev) or "")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Empty cell = not yet run / not applicable / data unavailable. T&T, KITTI, Sintel, ETH3D, DTU pending dataset loaders.")

    out_path = ROOT.parent / "RESULTS_V2.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
