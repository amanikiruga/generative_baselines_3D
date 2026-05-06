"""One-shot converter: zeroshot dataset metadata → V4 nested-by-task layout.

Input:  /n/netscratch/kempner_rcai_lab/Everyone/datasets/processed_zeroshot/metadata_zeroshot.json
Output: generative_baselines/metadata_zeroshot_v4.json
        generative_baselines/metadata_zeroshot_v4/<dataset>.json

Differences resolved:
  - Flat schema → nested-by-task ({pose_depth: {...}, nvs: {...}}). Both blocks point
    at the same GT artifacts since these datasets supply RGB+rays+depth+cameras and
    our model can be evaluated on either task from the same inputs.
  - Key rename: gt_rays_d_mp4 → gt_ray_d_mp4 (drop the 's').
  - Inline n_frames inferred via mp4 frame count (imageio).
  - prompt_txt was null in source. Converter writes an empty `prompt.txt` next to each
    sample's gt_rgb.mp4 (so the V4 metadata_loader's path-exists check still passes)
    and points prompt_txt at it.
  - valid_depth_mask path passed through under pose_depth (depth eval may use it
    later; current loader ignores extra keys).
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import imageio.v3 as iio


SRC = Path("/n/netscratch/kempner_rcai_lab/Everyone/datasets/processed_zeroshot/metadata_zeroshot.json")
ROOT = Path("/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines")
OUT_TOP = ROOT / "metadata_zeroshot_v4.json"
OUT_DIR = ROOT / "metadata_zeroshot_v4"
# Source dataset dir is read-only; stash sidecar prompt.txt files here.
PROMPT_SIDECAR = ROOT / "metadata_zeroshot_v4_prompts"


def count_frames(mp4: str) -> int | None:
    try:
        # iio.improps avoids decoding the whole video.
        props = iio.improps(mp4, plugin="pyav")
        return int(props.shape[0])
    except Exception:
        # Fallback: stream-count.
        try:
            n = 0
            for _ in iio.imiter(mp4):
                n += 1
            return n
        except Exception:
            return None


def ensure_prompt_txt(dataset: str, sample_id: str) -> str:
    """Write an empty prompt.txt in the sidecar dir (the source dataset path is
    read-only). The V4 loader only checks this path exists. Idempotent."""
    d = PROMPT_SIDECAR / dataset / sample_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / "prompt.txt"
    if not p.exists():
        p.write_text("")
    return str(p)


def convert_sample(s: dict, dataset: str) -> dict:
    rgb = s["gt_rgb_mp4"]
    prompt_txt = ensure_prompt_txt(dataset, s["sample_id"])
    paths = {
        "gt_rgb_mp4":       rgb,
        "gt_cameras_npz":   s["gt_cameras_npz"],
        "gt_depth_raw_npz": s.get("gt_depth_raw_npz"),
        "gt_ray_d_mp4":     s.get("gt_rays_d_mp4"),
        "gt_ray_m_mp4":     s.get("gt_rays_m_mp4"),
        "gt_depth_mp4":     s.get("gt_depth_mp4"),
        "prompt_txt":       prompt_txt,
        "valid_depth_mask": s.get("valid_depth_mask"),
        "viz_grid_mp4":     s.get("vis_grid_mp4"),
    }
    return {
        "sample_id": s["sample_id"],
        "n_frames":  count_frames(rgb),
        "prompt":    "",
        "pose_depth": dict(paths),
        "nvs":        dict(paths),
    }


def main():
    src = json.loads(SRC.read_text())
    OUT_DIR.mkdir(exist_ok=True)

    out_top = {"datasets": {}}
    for ds_name, ds in src["datasets"].items():
        new_samples = [convert_sample(s, ds_name) for s in ds["samples"]]
        n = len(new_samples)
        ds_out = {
            "version": "v4_zeroshot",
            "dataset": ds_name,
            "n_samples_pose_depth": n,
            "n_samples_nvs":        n,
            "samples": new_samples,
        }
        (OUT_DIR / f"{ds_name}.json").write_text(json.dumps(ds_out, indent=2))
        out_top["datasets"][ds_name] = ds_out
        print(f"  {ds_name}: {n} samples (n_frames range "
              f"{min((s['n_frames'] or 0) for s in new_samples)}–"
              f"{max((s['n_frames'] or 0) for s in new_samples)})")

    OUT_TOP.write_text(json.dumps(out_top, indent=2))
    print(f"\nwrote {OUT_TOP}")
    print(f"wrote {OUT_DIR}/<dataset>.json × {len(src['datasets'])}")


if __name__ == "__main__":
    main()
