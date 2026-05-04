# Results — generative baselines vs ours (1.3B mixture ckpt)

Checkpoint: `outputs/2026-04-30/23-29-19/checkpoints/last_archive.ckpt` (OURS_FINAL — single ckpt, supersedes 1.3B mixture + nvs_only).
Sample budget: 10 scenes per dataset (BFS pass).
Output base: `eval_outputs/<dataset>/<method>/`.

## NVS — interior-frame metrics (PSNR ↑ / LPIPS ↓ / SSIM ↑)

| Method | RE10K | DL3DV | DL3DV_TEST | TANKSANDTEMPLES | SCANNETPP | VKITTI2 | ARIA | SPATIALVID_NVS | AGIBOT_WORLD |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Ours | 17.97 / 0.214 / 0.580 | 16.51 / 0.241 / 0.393 | 14.17 / 0.327 / 0.321 | 10.44 / 0.592 / 0.187 | 13.79 / 0.401 / 0.422 | 18.53 / 0.286 / 0.608 | 21.51 / 0.299 / 0.682 | 13.93 / 0.409 / 0.357 | 15.65 / 0.242 / 0.648 |
| Wan 2.1 FLF | 15.87 / 0.327 / 0.478 | 13.56 / 0.391 / 0.253 (n=8) | 12.17 / 0.493 / 0.232 | 9.48 / 0.629 / 0.164 (n=5) | 11.21 / 0.611 / 0.342 (n=7) |  | 18.17 / 0.465 / 0.628 | 12.43 / 0.529 / 0.295 | 14.76 / 0.292 / 0.628 |
| DFoT | 19.96 / 0.220 / 0.650 |  |  |  |  |  | 17.64 / 0.548 / 0.644 |  |  |
| SEVA | 15.88 / 0.330 / 0.468 | 14.83 / 0.384 / 0.323 | 12.94 / 0.466 / 0.279 | 11.86 / 0.543 / 0.265 | 12.25 / 0.567 / 0.385 |  | 15.40 / 0.637 / 0.515 | 13.01 / 0.526 / 0.328 | 12.38 / 0.537 / 0.358 |

## Camera pose — relative pose AUC (AUC@3° ↑ / AUC@30° ↑)

| Method | re10k | dl3dv | dl3dv_test | tanksandtemples | scannetpp | vkitti2 | aria | spatialvid_nvs | agibot_world | sintel | eth3d | dtu |
|--------|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours | 2.76 / 59.97 | 0.49 / 47.15 | 0.33 / 45.73 | 0.01 / 16.18 | 0.11 / 32.49 | 0.26 / 31.80 | 0.64 / 46.34 | 0.18 / 47.18 | 0.00 / 0.00 |  |  |  |
| GEO4D | 0.11 / 6.97 | 0.01 / 3.40 | 0.00 / 1.18 | 0.02 / 1.39 | 0.00 / 1.46 | 0.05 / 4.59 | 3.02 / 3.13 | 0.59 / 0.92 | 86.79 / 98.63 |  |  |  |
| RayDiffusion | 0.12 / 6.94 | 0.15 / 9.72 | 0.00 / 4.01 | 0.00 / 1.38 | 0.00 / 0.88 | 0.00 / 7.57 | 0.00 / 1.80 | 0.00 / 1.61 | 0.00 / 0.00 |  |  |  |

## Depth — δ-thresholds, aligned scale+shift (δ1 ↑ / δ2 ↑ / δ3 ↑)

| Method | scenenet_depth | vkitti2 | aria | dl3dv | scannetpp | kitti | sintel | eth3d | dtu |
|--------|---|---|---|---|---|---|---|---|---|
| Ours | 0.427 / 0.708 / 0.842 | 0.391 / 0.435 / 0.519 | 0.905 / 0.914 / 0.925 | 0.430 / 0.716 / 0.863 | 0.512 / 0.816 / 0.915 |  |  |  |  |
| GEO4D | 0.749 / 0.904 / 0.958 | 0.535 / 0.552 / 0.566 | 0.311 / 0.562 / 0.773 | 0.751 / 0.870 / 0.908 | 0.930 / 0.993 / 1.000 |  |  |  |  |
| ChronoDepth | 0.610 / 0.839 / 0.945 |  |  | 0.687 / 0.861 / 0.909 |  |  |  |  |  |

Empty cell = not yet run / not applicable / data unavailable. T&T, KITTI, Sintel, ETH3D, DTU pending dataset loaders.
