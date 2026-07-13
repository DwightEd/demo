# First-Error Geometry Event Audit: step

This report treats the geometry measures as hypotheses to be tested, not as validated error detectors.

- Trajectories: `395` (`205` error, `190` correct)
- Matched error/control events: `205`
- Layers: `[8, 10, 12, 14, 16, 18, 20, 22]`
- Event definition: step index; delta_norm at offset 0 is the edge entering the first-error step
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
| `relative_delta_norm` | 14 | 170 | 0.829 | 0.025 | [0.007, 0.044] | 0.208 | 0.595 | 0.312 |
| `relative_delta_norm` | 16 | 170 | 0.829 | 0.019 | [0.002, 0.036] | 0.157 | 0.575 | 0.472 |
| `relative_delta_norm` | 12 | 170 | 0.829 | 0.022 | [0.001, 0.041] | 0.166 | 0.574 | 0.472 |
| `relative_delta_norm` | 10 | 170 | 0.829 | 0.021 | [-0.000, 0.041] | 0.156 | 0.566 | 0.472 |
| `relative_delta_norm` | 8 | 170 | 0.829 | 0.022 | [0.000, 0.042] | 0.156 | 0.564 | 0.472 |
| `relative_delta_norm` | 18 | 170 | 0.829 | 0.015 | [-0.002, 0.034] | 0.125 | 0.562 | 0.502 |
| `delta_norm` | 8 | 170 | 0.829 | 0.056 | [-0.007, 0.119] | 0.133 | 0.550 | 0.502 |
| `relative_delta_norm` | 20 | 170 | 0.829 | 0.010 | [-0.009, 0.028] | 0.082 | 0.539 | 0.606 |
| `delta_norm` | 10 | 170 | 0.829 | 0.040 | [-0.032, 0.111] | 0.084 | 0.533 | 0.606 |
| `delta_norm` | 12 | 170 | 0.829 | 0.033 | [-0.043, 0.112] | 0.063 | 0.527 | 0.706 |
| `relative_delta_norm` | 22 | 170 | 0.829 | 0.007 | [-0.010, 0.025] | 0.059 | 0.527 | 0.706 |
| `delta_norm` | 18 | 170 | 0.829 | 0.056 | [-0.099, 0.211] | 0.055 | 0.525 | 0.712 |
| `delta_norm` | 20 | 170 | 0.829 | 0.067 | [-0.123, 0.253] | 0.054 | 0.523 | 0.712 |
| `delta_norm` | 14 | 170 | 0.829 | 0.022 | [-0.071, 0.110] | 0.036 | 0.521 | 0.813 |
| `delta_norm` | 16 | 170 | 0.829 | 0.026 | [-0.088, 0.140] | 0.036 | 0.519 | 0.813 |
| `delta_norm` | 22 | 170 | 0.829 | 0.063 | [-0.167, 0.308] | 0.041 | 0.514 | 0.784 |

## First-Error Localization

This is a secondary diagnostic. It compares the gold first-error point against earlier points in the same erroneous trajectory and remains vulnerable to causal-position structure.

| variant | metric | layer | rows | positives | eligible chains | AUROC | expected top1 | mean rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `nuisance_residual` | `relative_delta_norm` | 14 | 385 | 170 | 170 | 0.622 | 0.735 | 1.418 |
| `nuisance_residual` | `relative_delta_norm` | 16 | 385 | 170 | 170 | 0.609 | 0.712 | 1.447 |
| `nuisance_residual` | `delta_norm` | 8 | 385 | 170 | 170 | 0.608 | 0.682 | 1.435 |
| `nuisance_residual` | `relative_delta_norm` | 18 | 385 | 170 | 170 | 0.608 | 0.688 | 1.441 |
| `nuisance_residual` | `relative_delta_norm` | 8 | 385 | 170 | 170 | 0.607 | 0.676 | 1.453 |
| `nuisance_residual` | `relative_delta_norm` | 12 | 385 | 170 | 170 | 0.604 | 0.694 | 1.424 |
| `nuisance_residual` | `relative_delta_norm` | 10 | 385 | 170 | 170 | 0.600 | 0.718 | 1.418 |
| `nuisance_residual` | `relative_delta_norm` | 20 | 385 | 170 | 170 | 0.597 | 0.671 | 1.447 |
| `nuisance_residual` | `delta_norm` | 14 | 385 | 170 | 170 | 0.596 | 0.694 | 1.465 |
| `nuisance_residual` | `delta_norm` | 12 | 385 | 170 | 170 | 0.593 | 0.712 | 1.412 |
| `nuisance_residual` | `delta_norm` | 10 | 385 | 170 | 170 | 0.593 | 0.712 | 1.400 |
| `nuisance_residual` | `relative_delta_norm` | 22 | 385 | 170 | 170 | 0.592 | 0.688 | 1.435 |
| `nuisance_residual` | `delta_norm` | 18 | 385 | 170 | 170 | 0.591 | 0.688 | 1.447 |
| `nuisance_residual` | `delta_norm` | 16 | 385 | 170 | 170 | 0.588 | 0.671 | 1.488 |
| `nuisance_residual` | `delta_norm` | 20 | 385 | 170 | 170 | 0.586 | 0.682 | 1.435 |
| `nuisance_residual` | `delta_norm` | 22 | 385 | 170 | 170 | 0.583 | 0.688 | 1.435 |
| `nuisance_residual` | `scale_free_curvature` | 8 | 369 | 154 | 154 | 0.533 | 0.669 | 1.494 |
| `nuisance_residual` | `turn_angle_rad` | 8 | 369 | 154 | 154 | 0.526 | 0.675 | 1.506 |
| `nuisance_residual` | `scale_free_curvature` | 12 | 369 | 154 | 154 | 0.522 | 0.669 | 1.481 |
| `nuisance_residual` | `scale_free_curvature` | 10 | 369 | 154 | 154 | 0.519 | 0.675 | 1.500 |

## Interpretation Guardrails

- A high raw score is insufficient; the nuisance-residualized matched effect is the primary result.
- Metrics with pair coverage below the stated threshold are excluded from the headline table.
- Menger curvature here is an inverse-distance geometric curvature, not directional concentration with the same symbol.
- A positive event association does not establish causal awareness or online detectability.
