# Same-Problem Feasible-Tangent Gate

This audit is geometry-only. It neither reads logits nor trains a label classifier.

## Data

- Samples: `3452`
- Errors / correct: `532` / `2920`
- Problems: `300`
- Layers: `[16]`
- The unavailable prompt-to-first-step transition is excluded.

## Gate 1: Does A Low-Rank Feasible Tangent Exist?

**Pass: `False`**

Rank-supported coverage (problem-equal): `0.3459` CI `[0.310527817469563, 0.3797859212850169]`.

| correct-target contrast | mean null - primary | CI95 | problems |
|---|---:|---|---:|
| phase_minus_primary | 0.0000 | [0.0, 0.0] | 250 |
| shuffle_minus_primary | 0.3517 | [0.333016110894197, 0.3692851302593513] | 250 |
| wrong_problem_minus_primary | 0.3518 | [0.3326405353400063, 0.3693856536774226] | 250 |
| random_minus_primary | 0.6054 | [0.5881838550291278, 0.6218901659930269] | 250 |

## Gate 2: Do Errors Show Persistent Normal Escape?

**Pass: `False`**

Primary preregistered score: `primary_coherent_escape.length_residual`.

| score | coverage | pooled AUROC | within-problem AUROC | CI95 | problems |
|---|---:|---:|---:|---|---:|
| control.log1p_n_steps | 1.0000 | 0.6694 | 0.5594 | [0.51001662780011, 0.6051466201740437] | 147 |
| control.log1p_response_chars | 1.0000 | 0.7186 | 0.5781 | [0.534000450937951, 0.6209262557957286] | 147 |
| primary_escape_mean | 0.6304 | 0.6941 | 0.5976 | [0.5376708018195088, 0.6601671122486497] | 87 |
| primary_escape_mean.length_residual | 0.6304 | 0.6413 | 0.5834 | [0.5243630613745557, 0.6455771591116418] | 87 |
| primary_escape_late | 0.5070 | 0.7457 | 0.6611 | [0.5915968513208774, 0.7258654787672643] | 77 |
| primary_escape_late.length_residual | 0.5070 | 0.6950 | 0.6455 | [0.5751931525364968, 0.7192958902569292] | 77 |
| primary_coherent_escape | 0.6304 | 0.5489 | 0.6157 | [0.5521513325302837, 0.6773868871390423] | 87 |
| primary_coherent_escape.length_residual | 0.6304 | 0.5734 | 0.6591 | [0.5969556565587455, 0.7186470807555575] | 87 |
| primary_late_coherent_escape | 0.5070 | 0.5830 | 0.6073 | [0.5326536351455183, 0.6802205258365972] | 77 |
| primary_late_coherent_escape.length_residual | 0.5070 | 0.6534 | 0.6899 | [0.6153163563979311, 0.7612638851798754] | 77 |
| primary_normal_persistence | 0.6304 | 0.4349 | 0.5049 | [0.4438787544471812, 0.5727455862504354] | 87 |
| primary_normal_persistence.length_residual | 0.6304 | 0.5212 | 0.6159 | [0.5521677295671549, 0.682267863708237] | 87 |
| phase_coherent_escape | 0.6304 | 0.5489 | 0.6157 | [0.5555596373214908, 0.6764678669563727] | 87 |
| phase_coherent_escape.length_residual | 0.6304 | 0.5734 | 0.6591 | [0.5950407364559663, 0.7195715571146606] | 87 |
| shuffle_coherent_escape | 0.6304 | 0.4207 | 0.5009 | [0.43411122894240717, 0.5666146567619268] | 87 |
| shuffle_coherent_escape.length_residual | 0.6304 | 0.5034 | 0.6228 | [0.5564988857858829, 0.684938072051937] | 87 |
| wrong_problem_coherent_escape | 0.6304 | 0.4543 | 0.5185 | [0.4551092893908987, 0.5858715758087164] | 87 |
| wrong_problem_coherent_escape.length_residual | 0.6304 | 0.5307 | 0.6168 | [0.5526194242645392, 0.6764149785001077] | 87 |

## Decision

- Status: **`blocked_until_feasible_tangent_passes`**
- Advance to exact output cotangent extraction: `False`
- No output sensitivity claim is permitted unless both gates pass.
