#!/usr/bin/env python3
"""Walk eval_outputs/<dataset>/<method>_eval/ for metric files and emit RESULTS.md."""
import json, csv, os
from pathlib import Path

ROOT = Path("/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs_v3")

# V3 metrics:
#   nvs   — unchanged: PSNR / LPIPS / SSIM from per_sample_metrics.csv
#   pose  — Geo4D-style: ATE (m) / RPE_trans (m) / RPE_rot (deg)
#   depth — Geo4D-style: Abs Rel / RMSE / δ<1.25  (selected from full Geo4D dict)


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
    """V3: Geo4D ATE / RPE_trans / RPE_rot from final_stats.json."""
    d = read_json(eval_dir / "final_stats.json")
    if not d:
        return ""
    if all(k in d for k in ("ate", "rpe_trans", "rpe_rot")):
        return f"{d['ate']:.3f} / {d['rpe_trans']:.3f} / {d['rpe_rot']:.2f}"
    return ""


def collect_depth(eval_dir: Path) -> str:
    """V3: Geo4D Abs Rel / RMSE / δ<1.25."""
    d = read_json(eval_dir / "final_stats.json")
    if not d:
        return ""
    if all(k in d for k in ("abs_rel", "rmse", "delta_1")):
        return f"{d['abs_rel']:.3f} / {d['rmse']:.3f} / {d['delta_1']:.3f}"
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
    lines.append("# Results V3 — Geo4D-style metrics, OURS_FINAL = 2026-05-03/20-03-01/last.ckpt")
    lines.append("")
    lines.append("Pose: ATE (m) / RPE_trans (m) / RPE_rot (deg) — Sim(3)-Umeyama via evo (Geo4D verbatim).")
    lines.append("Depth: Abs Rel / RMSE / δ<1.25 — per-video LAD2 alignment (Geo4D verbatim).")
    lines.append("Output base: `eval_outputs_v3/<dataset>/<method>/`.")
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
    lines.append("## Camera pose — ATE ↓ / RPE_trans ↓ / RPE_rot ↓")
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
    lines.append("## Depth — Geo4D, per-video LAD2 (Abs Rel ↓ / RMSE ↓ / δ<1.25 ↑)")
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

    out_path = ROOT.parent / "RESULTS_V3.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
