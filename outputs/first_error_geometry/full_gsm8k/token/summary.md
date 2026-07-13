# First-Error Geometry Event Audit: token

This report treats the geometry measures as hypotheses to be tested, not as validated error detectors.

- Trajectories: `395` (`205` error, `190` correct)
- Matched error/control events: `205`
- Layers: `[10, 14, 18, 22]`
- Event definition: first token of the first-error step; angle/curvature require one future token
- Effective compute device: `cuda`
- Primary-table minimum pair coverage: `80%`

## Geometry Definitions

For state $z_t^{(\ell)}$, the incoming update is $\Delta z_t^{(\ell)}=z_t^{(\ell)}-z_{t-1}^{(\ell)}$.
The turning angle is $\theta_t^{(\ell)}=\arccos\!\left(\langle \Delta z_t,\Delta z_{t+1}\rangle/(\|\Delta z_t\|\|\Delta z_{t+1}\|)\right)$.
Menger curvature is $\kappa_t^{(\ell)}=2\sin\theta_t^{(\ell)}/\|z_{t+1}^{(\ell)}-z_{t-1}^{(\ell)}\|$.

Offset `0` is the first-error step/token boundary. Turning angle and curvature use one future state and are therefore diagnostic rather than strictly pre-error signals.

## Matched Event Effects at Offset 0

Rows below use cross-fitted nuisance residuals learned from correct chains only. Matching uses chain length, relative event position, and event-step length, never geometry.

| metric | layer | pairs | coverage | error-control | 95% CI | paired dz | AUROC | q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `scale_free_curvature` | 22 | 170 | 0.829 | -0.012 | [-0.028, 0.005] | -0.107 | 0.454 | 0.713 |
| `scale_free_curvature` | 18 | 170 | 0.829 | -0.014 | [-0.033, 0.006] | -0.107 | 0.455 | 0.713 |
| `relative_delta_norm` | 22 | 170 | 0.829 | -0.007 | [-0.022, 0.007] | -0.075 | 0.458 | 0.878 |
| `turn_angle_rad` | 18 | 170 | 0.829 | -0.025 | [-0.064, 0.011] | -0.101 | 0.459 | 0.713 |
| `relative_delta_norm` | 14 | 170 | 0.829 | 0.010 | [-0.005, 0.025] | 0.102 | 0.540 | 0.713 |
| `menger_curvature` | 10 | 170 | 0.829 | 0.002 | [-0.001, 0.005] | 0.120 | 0.533 | 0.713 |
| `turn_angle_rad` | 22 | 170 | 0.829 | -0.017 | [-0.049, 0.015] | -0.079 | 0.469 | 0.872 |
| `delta_norm` | 10 | 170 | 0.829 | -0.054 | [-0.152, 0.044] | -0.083 | 0.474 | 0.832 |
| `delta_norm` | 18 | 170 | 0.829 | -0.037 | [-0.263, 0.179] | -0.024 | 0.477 | 0.917 |
| `relative_delta_norm` | 10 | 170 | 0.829 | 0.003 | [-0.010, 0.015] | 0.035 | 0.519 | 0.917 |
| `menger_curvature` | 14 | 170 | 0.829 | 0.001 | [-0.002, 0.003] | 0.043 | 0.516 | 0.917 |
| `delta_norm` | 14 | 170 | 0.829 | -0.029 | [-0.169, 0.113] | -0.032 | 0.487 | 0.917 |
| `delta_norm` | 22 | 170 | 0.829 | -0.003 | [-0.315, 0.305] | -0.002 | 0.488 | 0.986 |
| `relative_delta_norm` | 18 | 170 | 0.829 | 0.000 | [-0.014, 0.015] | 0.005 | 0.490 | 0.986 |
| `scale_free_curvature` | 14 | 170 | 0.829 | -0.004 | [-0.022, 0.015] | -0.028 | 0.492 | 0.917 |
| `turn_angle_rad` | 14 | 170 | 0.829 | -0.006 | [-0.039, 0.030] | -0.026 | 0.493 | 0.917 |
| `menger_curvature` | 22 | 170 | 0.829 | 0.000 | [-0.001, 0.001] | 0.002 | 0.496 | 0.986 |
| `menger_curvature` | 18 | 170 | 0.829 | 0.000 | [-0.002, 0.003] | 0.024 | 0.496 | 0.917 |
| `scale_free_curvature` | 10 | 170 | 0.829 | -0.002 | [-0.018, 0.013] | -0.023 | 0.498 | 0.917 |
| `turn_angle_rad` | 10 | 170 | 0.829 | -0.004 | [-0.036, 0.026] | -0.020 | 0.499 | 0.917 |

## First-Error Localization

This is a secondary diagnostic. It compares the gold first-error point against earlier points in the same erroneous trajectory and remains vulnerable to causal-position structure.

| variant | metric | layer | rows | positives | eligible chains | AUROC | expected top1 | mean rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `nuisance_residual` | `turn_angle_rad` | 22 | 20377 | 170 | 170 | 0.779 | 0.100 | 30.388 |
| `nuisance_residual` | `scale_free_curvature` | 22 | 20377 | 170 | 170 | 0.759 | 0.071 | 33.094 |
| `nuisance_residual` | `relative_delta_norm` | 10 | 20377 | 170 | 170 | 0.755 | 0.000 | 28.700 |
| `nuisance_residual` | `turn_angle_rad` | 18 | 20377 | 170 | 170 | 0.746 | 0.094 | 33.912 |
| `nuisance_residual` | `scale_free_curvature` | 18 | 20377 | 170 | 170 | 0.738 | 0.076 | 34.741 |
| `nuisance_residual` | `relative_delta_norm` | 18 | 20377 | 170 | 170 | 0.729 | 0.006 | 33.588 |
| `nuisance_residual` | `relative_delta_norm` | 22 | 20377 | 170 | 170 | 0.711 | 0.006 | 36.835 |
| `nuisance_residual` | `delta_norm` | 18 | 20377 | 170 | 170 | 0.711 | 0.006 | 34.235 |
| `nuisance_residual` | `delta_norm` | 10 | 20377 | 170 | 170 | 0.692 | 0.006 | 37.518 |
| `nuisance_residual` | `delta_norm` | 22 | 20377 | 170 | 170 | 0.679 | 0.000 | 37.300 |
| `nuisance_residual` | `relative_delta_norm` | 14 | 20377 | 170 | 170 | 0.668 | 0.000 | 40.676 |
| `nuisance_residual` | `scale_free_curvature` | 14 | 20377 | 170 | 170 | 0.667 | 0.012 | 39.882 |
| `nuisance_residual` | `turn_angle_rad` | 14 | 20377 | 170 | 170 | 0.658 | 0.006 | 40.424 |
| `nuisance_residual` | `delta_norm` | 14 | 20377 | 170 | 170 | 0.640 | 0.000 | 44.494 |
| `nuisance_residual` | `scale_free_curvature` | 10 | 20377 | 170 | 170 | 0.629 | 0.006 | 43.259 |
| `nuisance_residual` | `turn_angle_rad` | 10 | 20377 | 170 | 170 | 0.622 | 0.000 | 43.900 |
| `nuisance_residual` | `menger_curvature` | 22 | 20377 | 170 | 170 | 0.544 | 0.012 | 58.988 |
| `nuisance_residual` | `menger_curvature` | 14 | 20377 | 170 | 170 | 0.487 | 0.000 | 58.112 |
| `nuisance_residual` | `menger_curvature` | 18 | 20377 | 170 | 170 | 0.468 | 0.000 | 65.224 |
| `nuisance_residual` | `menger_curvature` | 10 | 20377 | 170 | 170 | 0.431 | 0.000 | 67.312 |

## Interpretation Guardrails

- A high raw score is insufficient; the nuisance-residualized matched effect is the primary result.
- Metrics with pair coverage below the stated threshold are excluded from the headline table.
- Menger curvature here is an inverse-distance geometric curvature, not directional concentration with the same symbol.
- A positive event association does not establish causal awareness or online detectability.
