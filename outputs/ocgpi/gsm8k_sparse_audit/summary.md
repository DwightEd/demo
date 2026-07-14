# Output-Conditioned Geometric Predictive Information Audit

## Frozen Research Question

Does internal reasoning geometry carry usable information about future output drift and response failure after conditioning on the causal output-distribution history?

The reported conditional usable information is a cross-fitted model-relative quantity, not an unrestricted mutual-information estimate.

## Preflight

- Joined chains: `395`
- Error responses: `205`
- Output features: `288`
- Geometry features: `21`
- Geometry tier: `exploratory_legacy_or_partial`
- State source: `legacy_sparse_stepvec`
- Layers: `[8, 10, 12, 14, 16, 18, 20, 22]`

## Bidirectional Coupling

Cross-fitted output-to-geometry explained variance: `-21.2361`.

This is the fraction of standardized geometry recoverable from causal output history under the fixed ridge family. The remaining chart is the geometry tested for incremental prediction.

## Response Detection

| prefix | output AUROC | controls+geometry AUROC | output+residual geometry AUROC | usable bits | delta AUROC | null delta AUROC |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.5032 | 0.6026 | 0.4926 | -0.0547 [-0.1698, +0.0113] | -0.0106 [-0.0815, +0.0638] | +0.0018 [-0.0796, +0.0821] |
| 0.50 | 0.4701 | 0.6721 | 0.5207 | -0.0655 [-0.1983, +0.0085] | +0.0506 [+0.0013, +0.1025] | +0.0265 [-0.0365, +0.0892] |
| 0.75 | 0.4326 | 0.6787 | 0.5159 | -0.0366 [-0.1013, +0.0083] | +0.0833 [+0.0317, +0.1355] | -0.0022 [-0.0645, +0.0646] |
| 1.00 | 0.5971 | 0.6714 | 0.5780 | -0.0150 [-0.0440, +0.0023] | -0.0191 [-0.0594, +0.0204] | -0.0182 [-0.0627, +0.0269] |

## Shared Online Prefix Detector

- Prefix rows: `2072`
- Output-only AUROC: `0.5112`
- Controls + geometry AUROC: `0.6311`
- Output + residual geometry AUROC: `0.5142`
- Conditional usable information (bits): `-0.0223 [-0.0471, -0.0021]`
- AUROC increment: `+0.0030 [-0.0502, +0.0549]`

The shared detector is evaluated on every observed prefix and never uses the eventual response length to select its input. Relative checkpoints above are retrospective diagnostic slices.

## Future Output Forecast

- MSE space: `outer_train_fold_standardized_target`
- Output-only MSE: `25.575384`
- Controls + geometry MSE: `39.087787`
- Output + residual geometry MSE: `26.416289`
- Partial R2: `-0.0329 [-0.0646, -0.0135]`
- Partial R2 versus length-matched null: `+0.2222 [+0.0645, +0.6067]`
- Gaussian conditional information (bits): `-0.0233 [-0.0444, -0.0096]`

## Output Baseline Saturation Ladder

| output tier | output AUROC | +geometry AUROC | usable bits | future partial R2 |
|---|---:|---:|---:|---:|
| scalar | 0.6356 | 0.6488 | +0.0110 [-0.0015, +0.0237] | -0.0005 [-0.0019, -0.0001] |
| distribution | 0.5204 | 0.5199 | -0.0188 [-0.0409, -0.0002] | -0.0247 [-0.0499, -0.0121] |
| full_compact | 0.5112 | 0.5142 | -0.0223 [-0.0471, -0.0021] | -0.0329 [-0.0646, -0.0135] |

## Geometry Group Ablation

| group | response usable bits | response delta AUROC | future partial R2 |
|---|---:|---:|---:|
| depth | -0.0100 [-0.0254, +0.0037] | +0.0076 [-0.0449, +0.0584] | -0.0051 [-0.0178, -0.0031] |
| temporal | -0.0098 [-0.0206, -0.0017] | -0.0155 [-0.0451, +0.0150] | -0.0163 [-0.0362, -0.0041] |
| coupling | -0.0040 [-0.0078, -0.0008] | +0.0048 [-0.0201, +0.0289] | -0.0008 [-0.0119, +0.0028] |
| final_control | -0.0019 [-0.0044, +0.0003] | -0.0072 [-0.0281, +0.0136] | -0.0050 [-0.0147, -0.0003] |
| legacy_geometry | -0.0024 [-0.0082, +0.0027] | +0.0347 [+0.0024, +0.0682] | -0.0154 [-0.0313, -0.0061] |

## Decision Gate

- Mechanism supported: `0`
- Detector increment supported: `0`
- Confirmatory ready: `0`

- `response_usable_information_ci_above_zero`: `0`
- `response_beats_length_matched_null`: `0`
- `future_output_partial_r2_ci_above_zero`: `0`
- `future_output_beats_length_matched_null`: `1`
- `problem_groups_at_least_100`: `1`
- `whole_layer_geometry`: `0`
- `observer_model_identity_verified`: `1`

## Interpretation Guardrail

A positive response increment alone is not a mechanism result. The mechanism claim requires residual geometry to predict future output change and to beat the matched null. A negative result falsifies the proposed geometry family, not the existence of every possible internal signal.
