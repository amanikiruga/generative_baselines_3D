#!/usr/bin/env python3
"""Rebuild eval_outputs_v4_n50_merged/ as a symlink-only union.

Priority order per (dataset, method) cell:
    1. eval_outputs_v4_n50/                 (FG primary)
    2. eval_outputs_v3_n50/                 (V3 cache)
    3. eval_outputs_v4_n50_array/           (array overflow)
    4. /n/netscratch/.../generative_baselines_eval_v2/  (V2 legacy)

Special: seva_nvs is never inherited from V2 (stale, pre-fix). It must come
from FG (the recompute tree) or be omitted.
"""
from pathlib import Path
import os, shutil

ROOT = Path("/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines")
MERGED = ROOT / "eval_outputs_v4_n50_merged"
SOURCES = [
    ROOT / "eval_outputs_v4_n50",
    ROOT / "eval_outputs_v3_n50",
    ROOT / "eval_outputs_v4_n50_array",
    Path("/n/netscratch/ydu_lab/Everyone/akiruga/generative_baselines_eval_v2"),
]
SEVA_NO_LEGACY = {"seva_nvs"}  # never inherit from V2

if MERGED.exists():
    shutil.rmtree(MERGED)
MERGED.mkdir(parents=True)

# Discover all (dataset, method) cells across sources.
cells: dict[tuple[str, str], Path] = {}
for src in SOURCES:
    if not src.exists():
        continue
    is_legacy = "generative_baselines_eval_v2" in str(src)
    for ds_dir in src.iterdir():
        if not ds_dir.is_dir():
            continue
        ds = ds_dir.name
        for m_dir in ds_dir.iterdir():
            if not (m_dir.is_dir() or m_dir.is_symlink()):
                continue
            method = m_dir.name
            if is_legacy and method in SEVA_NO_LEGACY:
                continue
            key = (ds, method)
            if key not in cells:
                # Resolve through one symlink so the merged link points at
                # real content even if a source is itself a symlink chain.
                target = m_dir.resolve() if m_dir.is_symlink() else m_dir
                cells[key] = target

n_total = 0
ds_set = set()
for (ds, method), target in cells.items():
    out_ds = MERGED / ds
    out_ds.mkdir(parents=True, exist_ok=True)
    link = out_ds / method
    os.symlink(str(target), str(link))
    n_total += 1
    ds_set.add(ds)

print(f"merged {n_total} cells across {len(ds_set)} datasets → {MERGED}")
