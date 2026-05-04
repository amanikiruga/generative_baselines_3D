# Results V3 (n=50) — Geo4D-style metrics, OURS_FINAL = 2026-05-03/20-03-01/last.ckpt

Pose: ATE (m) / RPE_trans (m) / RPE_rot (deg) — Sim(3)-Umeyama via evo (Geo4D verbatim).
Depth: Abs Rel / RMSE / δ<1.25 — per-video LAD2 alignment (Geo4D verbatim).
Output base: `eval_outputs_v3/<dataset>/<method>/`.

## NVS — interior-frame metrics (PSNR ↑ / LPIPS ↓ / SSIM ↑)

| Method | RE10K | DL3DV | DL3DV_TEST | TANKSANDTEMPLES | SCANNETPP | VKITTI2 | ARIA | SPATIALVID_NVS | AGIBOT_WORLD |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Ours | 19.41 / 0.173 / 0.625 | 15.03 / 0.327 / 0.392 | 13.26 / 0.344 / 0.279 | 10.46 / 0.579 / 0.201 | 13.89 / 0.387 / 0.430 | 19.14 / 0.239 / 0.630 | 21.64 / 0.262 / 0.700 | 14.34 / 0.348 / 0.377 |  |
| Ours (first only) |  |  |  |  |  |  |  |  |  |
| Wan 2.1 FLF | 15.87 / 0.327 / 0.478 | 12.41 / 0.493 / 0.241 | 12.17 / 0.493 / 0.232 | 9.48 / 0.629 / 0.164 (n=5) | 11.21 / 0.611 / 0.342 (n=7) | 16.09 / 0.389 / 0.550 | 18.17 / 0.465 / 0.628 | 12.43 / 0.529 / 0.295 |  |
| Wan 2.1 FLF (text) |  |  |  |  |  |  |  |  |  |
| DFoT | 19.96 / 0.220 / 0.650 |  |  |  |  |  | 17.64 / 0.548 / 0.644 |  |  |
| SEVA | 15.88 / 0.330 / 0.468 | 13.92 / 0.424 / 0.308 | 12.94 / 0.466 / 0.279 | 11.86 / 0.543 / 0.265 | 12.25 / 0.567 / 0.385 | 18.09 / 0.308 / 0.591 | 15.40 / 0.637 / 0.515 | 13.01 / 0.526 / 0.328 |  |

## Camera pose — ATE ↓ / RPE_trans ↓ / RPE_rot ↓

| Method | re10k | dl3dv | dl3dv_test | tanksandtemples | scannetpp | vkitti2 | aria | spatialvid_nvs | agibot_world | sintel | eth3d | dtu |
|--------|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours | 0.599 / 0.219 / 1.18 | 1.186 / 0.353 / 2.52 | 2.095 / 0.699 / 1.81 | 4.400 / 1.165 / 5.65 | 1.454 / 0.351 / 2.38 | 2.325 / 0.569 / 3.47 | 1.722 / 1.000 / 1.80 | 2.348 / 0.731 / 1.79 |  |  |  |  |
| GEO4D | 0.683 / 0.982 / 0.26 | 1.196 / 0.835 / 2.00 | 1.524 / 1.437 / 1.45 | 4.232 / 2.260 / 5.03 | 1.716 / 0.764 / 2.41 | 0.730 / 1.874 / 0.20 | 0.934 / 2.107 / 1.10 | 2.533 / 1.674 / 0.84 |  |  |  |  |
| RayDiffusion | 4.335 / 5.426 / 11.87 | 4.007 / 6.397 / 34.19 | 5.840 / 10.708 / 25.67 | 4.875 / 14.318 / 83.99 | 2.824 / 4.082 / 53.92 | 11.331 / 11.661 / 13.09 | 7.776 / 7.574 / 35.13 | 6.262 / 8.610 / 31.84 |  |  |  |  |

## Depth — depth-space LAD2 (Abs Rel ↓ / RMSE ↓ / δ<1.25 ↑)

| Method | scenenet_depth | vkitti2 | aria | dl3dv | scannetpp | kitti | sintel | eth3d | dtu |
|--------|---|---|---|---|---|---|---|---|---|
| Ours | 1.571 / 0.702 / 0.257 | 0.682 / 0.522 / 0.334 | 3.225 / 1.544 / 0.316 | 0.731 / 0.531 / 0.394 | 0.256 / 0.271 / 0.667 |  |  |  |  |
| GEO4D |  | 1.290 / 0.400 / 0.123 | 7.044 / 2.071 / 0.373 | 0.819 / 0.276 / 0.269 | 0.468 / 0.268 / 0.673 |  |  |  |  |
| ChronoDepth |  | 1.290 / 0.400 / 0.123 | 9.440 / 3.225 / 0.324 | 0.801 / 0.268 / 0.282 | 0.424 / 0.279 / 0.662 |  |  |  |  |

## Depth (disparity space) — disparity-space LAD2 (Abs Rel ↓ / RMSE ↓ / δ<1.25 ↑)

_Different units from the depth table above — do not compare cells across the two tables._

| Method | scenenet_depth | vkitti2 | aria | dl3dv | scannetpp | kitti | sintel | eth3d | dtu |
|--------|---|---|---|---|---|---|---|---|---|
| Ours | 4.160 / 0.112 / 0.502 | 11.981 / 0.349 / 0.479 | 4.224 / 4.469 / 0.616 | 0.278 / 0.165 / 0.692 | 22.437 / 0.137 / 0.748 |  |  |  |  |
| GEO4D |  | 11.921 / 0.077 / 0.585 | 0.536 / 0.174 / 0.624 | 0.427 / 0.146 / 0.619 | 32.483 / 0.120 / 0.904 |  |  |  |  |
| ChronoDepth |  | 15.842 / 0.266 / 0.222 | 1.266 / 2.311 / 0.280 | 0.569 / 0.850 / 0.335 | 27.396 / 1.764 / 0.430 |  |  |  |  |

Empty cell = not yet run / not applicable / data unavailable. T&T, KITTI, Sintel, ETH3D, DTU pending dataset loaders.
