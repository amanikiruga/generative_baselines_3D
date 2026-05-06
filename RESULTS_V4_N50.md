# Results V4 (n=10/50) — 3 ckpts × baselines, metadata-pinned video set (metadata_v3_n50.json)
# OURS_FINAL = 2026-05-04/12-34-53 ; OURS_NVS = 2026-04-29/14-31-15

Pose: ATE (m) / RPE_trans (m) / RPE_rot (deg) — Sim(3)-Umeyama via evo (Geo4D verbatim).
Depth: Abs Rel / RMSE / δ<1.25 — per-video LAD2 alignment (Geo4D verbatim).
Output base: `eval_outputs_v3/<dataset>/<method>/`.

## NVS — interior-frame metrics (PSNR ↑ / LPIPS ↓ / SSIM ↑)

| Method | RE10K | DL3DV | DL3DV_TEST | TANKSANDTEMPLES | SCANNETPP | VKITTI2 | ARIA | SPATIALVID_NVS | AGIBOT_WORLD |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Ours (final) | 19.98 / 0.154 / 0.658 | 15.09 / 0.316 / 0.408 | 13.69 / 0.325 / 0.308 |  | 14.34 / 0.346 / 0.451 |  | 21.28 / 0.241 / 0.697 |  | 16.22 / 0.208 / 0.695 |
| Ours (NVS) | 17.17 / 0.244 / 0.526 | 14.25 / 0.398 / 0.361 | 12.82 / 0.404 / 0.252 |  | 12.63 / 0.486 / 0.395 |  | 19.16 / 0.378 / 0.654 |  | 16.29 / 0.233 / 0.626 |
| Wan 2.1 FLF | 15.87 / 0.327 / 0.478 | 12.41 / 0.493 / 0.241 | 12.17 / 0.493 / 0.232 |  | 11.21 / 0.611 / 0.342 (n=7) |  | 18.17 / 0.465 / 0.628 |  | 15.26 / 0.250 / 0.665 (n=2) |
| Wan 2.1 FLF (text) |  |  |  |  |  |  |  |  |  |
| DFoT | 19.96 / 0.220 / 0.650 |  |  |  |  |  | 17.64 / 0.548 / 0.644 |  |  |
| SEVA | 15.88 / 0.330 / 0.468 | 13.92 / 0.424 / 0.308 | 12.94 / 0.466 / 0.279 |  | 12.25 / 0.567 / 0.385 |  | 15.40 / 0.637 / 0.515 |  | 12.33 / 0.526 / 0.353 |

## Camera pose — ATE ↓ / RPE_trans ↓ / RPE_rot ↓

| Method | re10k | dl3dv | dl3dv_test | tanksandtemples | scannetpp | vkitti2 | aria | spatialvid_nvs | agibot_world | sintel | eth3d | dtu |
|--------|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours (final) | 0.769 / 0.304 / 0.85 | 1.692 / 0.516 / 2.16 | 1.994 / 0.759 / 1.61 |  | 3.136 / 0.915 / 2.54 |  | 1.303 / 0.937 / 1.77 |  | 0.000 / 0.000 / 0.00 |  |  |  |
| Ours (NVS) | 1.187 / 0.396 / 6.10 | 2.513 / 0.669 / 7.10 | 2.385 / 0.829 / 6.55 |  | 4.077 / 1.113 / 7.50 |  | 1.989 / 1.037 / 7.52 |  | 0.000 / 0.000 / 0.00 |  |  |  |
| GEO4D |  |  | 1.524 / 1.437 / 1.45 |  | 1.716 / 0.764 / 2.41 |  | 0.934 / 2.107 / 1.10 |  |  |  |  |  |
| RayDiffusion | 5.712 / 7.216 / 11.56 | 5.847 / 9.582 / 36.64 | 5.840 / 10.708 / 25.67 |  | 2.824 / 4.082 / 53.92 |  | 7.776 / 7.574 / 35.13 |  | 0.000 / 0.000 / 0.00 |  |  |  |

## Depth — depth-space LAD2 (Abs Rel ↓ / RMSE ↓ / δ<1.25 ↑)

| Method | scenenet_depth | vkitti2 | aria | dl3dv | scannetpp | kitti | sintel | eth3d | dtu |
|--------|---|---|---|---|---|---|---|---|---|
| Ours (final) | 0.000 / 0.000 / 0.000 |  | 2.423 / 0.932 / 0.346 | 0.352 / 0.185 / 0.524 | 0.234 / 0.198 / 0.795 |  |  |  |  |
| Ours (NVS) | 0.000 / 0.000 / 0.000 |  | 0.790 / 0.255 / 0.377 | 0.399 / 0.201 / 0.499 | 0.176 / 0.224 / 0.795 |  |  |  |  |
| GEO4D |  |  | 7.044 / 2.071 / 0.373 |  | 0.468 / 0.268 / 0.673 |  |  |  |  |
| ChronoDepth | 0.000 / 0.000 / 0.000 |  | 9.440 / 3.225 / 0.324 | 0.783 / 0.274 / 0.278 | 0.424 / 0.279 / 0.662 |  |  |  |  |

## Depth (disparity space) — disparity-space LAD2 (Abs Rel ↓ / RMSE ↓ / δ<1.25 ↑)

_Different units from the depth table above — do not compare cells across the two tables._

| Method | scenenet_depth | vkitti2 | aria | dl3dv | scannetpp | kitti | sintel | eth3d | dtu |
|--------|---|---|---|---|---|---|---|---|---|
| Ours (final) |  |  |  |  |  |  |  |  |  |
| Ours (NVS) |  |  |  |  |  |  |  |  |  |
| GEO4D |  |  | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |  |  |  |  |
| ChronoDepth | 0.000 / 0.000 / 0.000 |  | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |  |  |  |  |

Empty cell = not yet run / not applicable / data unavailable. T&T, KITTI, Sintel, ETH3D, DTU pending dataset loaders.
