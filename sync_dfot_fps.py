#!/usr/bin/env python3
"""Re-encode DFoT pred_rgb.mp4 (8 frames) at the fps that makes its duration
match Ours' pred_rgb.mp4 (50 frames @ dataset fps). One-shot, idempotent."""
import imageio, numpy as np
from pathlib import Path

ROOT = Path("/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs")
OURS_FPS = {"re10k": 8, "dl3dv": 30}
OURS_T   = 50  # frames in ours pred_rgb.mp4

for ds, ours_fps in OURS_FPS.items():
    duration = OURS_T / ours_fps           # seconds
    base = ROOT / ds / "dfot_nvs"
    if not base.exists():
        continue
    for sample in sorted(base.glob("sample_*")):
        mp4 = sample / "pred_rgb.mp4"
        if not mp4.exists():
            continue
        frames = np.stack(imageio.mimread(str(mp4), memtest=False))  # (T, H, W, 3)
        T = len(frames)
        new_fps = T / duration
        imageio.mimwrite(str(mp4), frames, fps=new_fps)
        print(f"{mp4}: T={T}, new fps={new_fps:.3f} (matches ours {OURS_T}@{ours_fps}={duration:.2f}s)")
