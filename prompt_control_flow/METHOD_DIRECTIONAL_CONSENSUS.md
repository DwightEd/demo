# Debiased Directional Consensus

## 1. Question and Fixed Hypothesis

The previous same-problem audit found that trajectory geometry retained only a
weak signal after length control. The strongest unresolved possibility is more
specific than "errors are geometrically diffuse": token directions may lose
consensus, while the old resultant-based spread is partly a finite-token proxy.

The fixed hypothesis is:

> For the same problem, incorrect responses have lower hidden-state token
> direction consensus than correct responses, and this difference survives
> exact removal of the finite-token self-pair term, cross-fitted length control,
> and explicit token-length matching.

This is a response-level hypothesis. It does not assert first-error
localization, temporal causality, or that the model knows it is wrong.

## 2. Existing Data

No teacher-forcing extraction is required. Use the same-problem multisample
artifacts already recorded in `md/guides/DATA.md`:

```text
/gz-data/research/demo/data/gsm8k_v2_custom.npz
/gz-data/research/demo/data/gsm8k_v2_5shot.npz
```

Required keys are:

```text
sv_vec_step_exp   per-response step trajectory, used for sample alignment
sv_clouds         per-response token states, shape (N, L, D)
cloud_sizes       token counts of the semantic steps, sum equals N
cloud_layers      actual model layer ids for the L cloud slices
problem_ids       same-question group id
sample_idx        rollout id within a problem
is_correct        final-answer label
format_ok         formatting filter used by answer_format_ok
responses         response text, used only for the character-length control
```

The canonical artifacts currently store token clouds at layer 16. The code
does not infer that layer from the step-vector axis: it checks `cloud_layers`
against the actual cloud depth and fails on a mismatch.

## 3. Geometry on the Unit Sphere

For one response, layer `l`, and a set of `n` token states, normalize each
state:

\[
u_i^{(l)} = \frac{h_i^{(l)}}{\lVert h_i^{(l)}\rVert_2}.
\]

The old resultant length is

\[
R^{(l)} = \left\lVert \frac{1}{n}\sum_{i=1}^{n}u_i^{(l)}\right\rVert_2.
\]

Its square contains all `n` self-pairs:

\[
\left(R^{(l)}\right)^2
= \frac{1}{n} + \frac{n-1}{n}\bar c^{(l)},
\]

where the exact off-diagonal mean cosine is

\[
\bar c^{(l)}
= \frac{1}{n(n-1)}\sum_{i\ne j}\left(u_i^{(l)}\right)^\top u_j^{(l)}
= \frac{n\left(R^{(l)}\right)^2-1}{n-1}.
\]

The proposed risk score is the debiased directional dispersion

\[
\mathcal D^{(l)} = 1-\bar c^{(l)}.
\]

This is an exact identity, not a fitted correction. It removes the deterministic
self-pair contribution before labels are inspected. It can be computed in
linear time from the vector sum; constructing an `N x N` cosine matrix is
unnecessary.

## 4. Response Summaries

The implementation reports both the old and proposed quantities:

- `raw_spread.global`: `1 - R` over all response tokens.
- `raw_spread.step_mean`: equal-weight mean over semantic steps.
- `debiased_dispersion.global`: `1 - c_bar` over all response tokens.
- `debiased_dispersion.step_mean`: equal-weight mean over semantic steps.
- `debiased_dispersion.late_mean`: mean over the final one-third of steps.
- `debiased_dispersion.step_max`: maximum step event.
- `fixed_window_dispersion.mean`: mean over fixed non-overlapping token windows.

The fixed-window score is a segmentation control. It asks whether the signal
survives when every local estimate uses the same number of tokens and no
semantic-step boundary is needed.

All token normalization, grouped sums, step aggregation, and fixed-window
statistics are batched on the requested PyTorch device. Ragged samples are
concatenated and reduced with `index_add_`; raw full pairwise matrices are not
materialized. The batch budget counts token-layer states, so selecting more
layers automatically reduces the number of responses placed in one GPU batch.

## 5. Evaluation Protocol

The primary estimand is same-problem paired AUROC: every incorrect response is
compared only with correct responses to the same question. Problems, not
individual responses, are the bootstrap unit.

Every geometry score is evaluated in four forms:

1. raw same-problem paired AUROC;
2. cross-fitted residual after controlling for
   `log1p(n_steps)`, `log1p(response_chars)`, and
   `log1p(response_cloud_tokens)`;
3. explicit correct/error pairs whose token counts differ by at most a fixed
   ratio, default `1.25`;
4. correlation with response token and character counts.

The six predeclared confirmatory tests are the global, equal-step, and
fixed-window debiased scores before and after length residualization. Their
same-problem permutation p-values are corrected together with Benjamini-Hochberg
q-values. Other summaries are exploratory. Problem bootstrap and confirmatory
label permutations are tensorized on the requested compute device.

The continue gate passes only when the length-residualized equal-step score has:

\[
\operatorname{CI}_{95\%}(\operatorname{AUC}_{same})_{low} > 0.5,
\quad q < 0.05,
\quad
\operatorname{CI}_{95\%}(\operatorname{AUC}_{token-match})_{low} > 0.5.
\]

In addition, the report bootstraps whether debiasing improves same-problem
AUROC over raw spread. Passing the main gate without improving over raw spread
would mean the signal is real but the new estimator is not demonstrably better.

## 6. Commands

Preflight the primary custom-prompt artifact:

```bash
cd /gz-data/research/demo
python audit_directional_consensus.py \
  --input data/gsm8k_v2_custom.npz \
  --vector_key sv_vec_step_exp \
  --cloud_layers all \
  --label_policy answer_format_ok \
  --preflight
```

Run the confirmatory audit on GPU:

```bash
python audit_directional_consensus.py \
  --input data/gsm8k_v2_custom.npz \
  --output outputs/directional_consensus/gsm8k_custom_scores.npz \
  --output_dir outputs/directional_consensus/gsm8k_custom_audit \
  --vector_key sv_vec_step_exp \
  --cloud_layers all \
  --label_policy answer_format_ok \
  --fixed_window_tokens 16 \
  --compute_device cuda \
  --bootstrap 2000 \
  --permutations 2000
```

Replicate without changing the frozen settings:

```bash
python audit_directional_consensus.py \
  --input data/gsm8k_v2_5shot.npz \
  --output outputs/directional_consensus/gsm8k_5shot_scores.npz \
  --output_dir outputs/directional_consensus/gsm8k_5shot_audit \
  --vector_key sv_vec_step_exp \
  --cloud_layers all \
  --label_policy answer_format_ok \
  --fixed_window_tokens 16 \
  --compute_device cuda \
  --bootstrap 2000 \
  --permutations 2000
```

## 7. Decision and Next Step

- **Gate fails:** retire generic token-direction dispersion as a detection
  mechanism. Do not rescue it with a learned nonlinear ensemble.
- **Gate passes, debiasing does not beat raw spread:** retain the phenomenon as
  a controlled descriptive result, not a method contribution.
- **Gate and estimator comparison pass on both prompts:** next test whether the
  escaping directions overlap output-sensitive directions from the unembedding
  Jacobian or direct logit attribution. That is the missing bridge from
  representational geometry to generation behavior.
