# Conditional Ordered Reasoning-Flow Signatures

## Status

This is the current mainline hypothesis test. The previous layer-time
holonomy construction is retained only as a compatibility/negative baseline.
It must not be described as a validated curvature model of reasoning.

The implementation is deliberately narrow: it tests whether correct and
incorrect hidden-state trajectories become separable after conditioning on
the problem and representing each trajectory by its ordered flow geometry.
It does not train a generic correctness classifier and it does not combine a
large menu of hand-picked scalar signals.

## 1. Why the layer-time plaquette did not work

The old method formed a loop from one reasoning-step edge and one layer-depth
edge. These edges are different operations: adding a generated prefix token is
not the same kind of perturbation as applying another Transformer block. Their
commutator therefore has no justified interpretation as the curvature of a
single connection. In addition, phase-matched samples are not shared local
input neighborhoods, so Procrustes transport mostly measured neighborhood
mismatch. The observed near-chance step AUROCs and very low reliable-Wilson
coverage are consistent with this structural problem.

The useful lesson from representation holonomy is methodological rather than
literal: compare paths only after fixing the coordinate gauge, use shared
identities when estimating transport, and require invariance/null tests. The
new method does not construct a layer-time Wilson loop.

## 2. Falsifiable hypothesis

The original statement, "errors are more spread," is too weak. A wrong chain
can be internally coherent and converge to a wrong attractor. The revised
hypothesis is:

> After removing static semantic offset, coordinate scale, path speed, and
> question difficulty, correct solutions to the same problem occupy a
> concentrated class of ordered hidden-state flows. Incorrect solutions leave
> that conditional flow class or enter a different ordered mode.

This hypothesis predicts more than a scalar mean difference:

1. A conditional correct-flow score separates wrong and correct responses for
   the same problem.
2. Second-order ordered information adds beyond net displacement.
3. Permuting the same hidden-state increments degrades the signal.
4. The result survives response-length and step-count controls.
5. Error residuals have larger radius and, if the "more directions" claim is
   true, higher effective rank than leave-one-out correct residuals.

Any failure is reported as a failed claim gate, not hidden by selecting another
metric.

## 3. Hidden-state flow object

For response (i), stored layer \(\ell\), and reasoning step \(t\), let

\[
z_{i,t,\ell}\in\mathbb{R}^{D}
\]

be the pooled hidden state already stored in `stepvec` or
`sv_vec_step_exp`. The method works on increments

\[
\Delta z_{i,t,\ell}=z_{i,t+1,\ell}-z_{i,t,\ell},
\]

so a constant semantic offset cancels exactly. A seeded, data-independent
orthogonal sketch \(R\in\mathbb{R}^{D\times d}\) gives

\[
x_{i,t,\ell}=R^{\top}z_{i,t,\ell}.
\]

The sketch is not a learned correctness representation. It is a leakage-free
computational approximation whose dimension and seed must be ablated.

Each prefix is normalized by hidden-path total variation

\[
V_{i,s,\ell}=\sum_{t<s}\lVert\Delta x_{i,t,\ell}\rVert_2.
\]

This removes raw trajectory scale. The progress channel is the normalized
cumulative arc length shared across stored layers. If

\[
a_{i,t}=\frac{1}{L}\sum_{\ell=1}^{L}
\lVert\Delta x_{i,t,\ell}\rVert_2,
\]

then

\[
\Delta\tau_{i,t}=\frac{a_{i,t}}{\sum_s a_{i,s}+\varepsilon}.
\]

This arc-length parameter keeps direction order while removing raw step count,
speed, and piecewise-linear re-segmentation as predictors.

## 4. Ordered log-signature

The first level is normalized net displacement:

\[
S^{(1)}_{i,s,\ell}
=
\frac{\sum_{t<s}\Delta x_{i,t,\ell}}
{V_{i,s,\ell}+\varepsilon}.
\]

The second level is the antisymmetric Levy area of the augmented increments
\(\Delta\widetilde{x}_t=[\Delta x_t,\Delta\tau_t]\):

\[
A^{(2)}_{i,s,\ell}
=
\frac{1}{2}\sum_{a<b<s}
\left(
\Delta\widetilde{x}_{a}\otimes\Delta\widetilde{x}_{b}
-
\Delta\widetilde{x}_{b}\otimes\Delta\widetilde{x}_{a}
\right).
\]

Hidden-hidden entries are divided by \(V^2\), while hidden-progress entries
are divided by \(V\). The feature is

\[
\phi_{i,s}
=
\operatorname{concat}_{\ell}
\left[S^{(1)}_{i,s,\ell},\operatorname{vech}_{<}A^{(2)}_{i,s,\ell}\right].
\]

Unlike endpoint distance, \(A^{(2)}\) changes when the same increments occur in
a different order. Unlike a mean of per-step scalars, it is an iterated path
functional. The implementation updates it online in
\(O(TLd^2)\) time and does not materialize all step pairs.

The common arc-length phase grid used for saved prefix profiles is exact: a
grid point inside an original segment splits that segment analytically. It does
not linearly reconnect resampled states across a corner.

## 5. Conditional correct-flow support

Outer folds are split by `problem_id`. Feature scales are fitted only from
correct trajectories in training problems. For a held-out problem \(p\), the
same-problem support diagnostic uses its correct responses as a conditional
reference. Correct candidates are scored leave-one-out:

\[
\mu_{p,-i,s}
=
\frac{1}{|\mathcal{C}_{p,-i}|}
\sum_{j\in\mathcal{C}_{p,-i}}\phi_{j,s}.
\]

With robust train-fold scale \(\sigma_s\), conditional escape energy is

\[
E_{i,s}
=
\frac{1}{m}
\left\lVert
\frac{\phi_{i,s}-\mu_{p,-i,s}}{\sigma_s}
\right\rVert_2^2.
\]

The preregistered response score is endpoint energy \(E_{i,1}\). Prefix
energy profiles and their integral are saved for diagnosis but are not used to
replace a failed primary score.

The same-problem score is an oracle diagnostic of the scientific hypothesis;
it is not claimed as zero-support deployment. A separate global score uses
only correct training-problem trajectories and is the deployable baseline.

## 6. Required controls and claim gates

The report always includes:

- order-1 endpoint versus order-2 endpoint;
- chronological order versus deterministically shuffled increments;
- `n_steps`, response characters, and total variation;
- a cross-fitted length-residualized primary score;
- same-problem paired AUROC, clustered bootstrap confidence interval, and
  within-problem label-permutation p-value;
- cross-problem AUROC/AUPRC as secondary context;
- leave-one-out correct/error residual radius and effective rank.

The hypothesis is supported only if all five gates pass:

1. the conditional AUROC confidence interval is above chance;
2. order two improves over order one;
3. chronological order improves over shuffled order;
4. the primary score beats all length/scale controls;
5. length residualization remains above chance.

Synthetic self-test success proves only that the implementation can recover a
pure order effect. It is not empirical evidence about ProcessBench.

## 7. Data and direct commands

Primary same-problem inputs documented in `md/guides/DATA.md` are:

```text
data/gsm8k_v2_5shot.npz
data/gsm8k_v2_custom.npz
```

They must contain `sv_vec_step_exp`, which is written only when extraction used
`--store_vectors`. Canonical ProcessBench files such as
`data/features/full_gsm8k.npz` contain `stepvec` and can run the global
cross-problem baseline, but they cannot produce a same-problem support test.

Preflight:

```bash
python audit_reasoning_flow_signatures.py \
  --input data/gsm8k_v2_custom.npz \
  --preflight
```

Primary GPU audit:

```bash
python audit_reasoning_flow_signatures.py \
  --input data/gsm8k_v2_custom.npz \
  --output outputs/reasoning_flow_signatures/gsm8k_custom_scores.npz \
  --output_dir outputs/reasoning_flow_signatures/gsm8k_custom_audit \
  --vector_key sv_vec_step_exp \
  --layers all \
  --label_policy answer_format_ok \
  --projection_dim 8 \
  --phase_points 16 \
  --compute_device cuda \
  --batch_size 64 \
  --bootstrap 2000 \
  --permutations 1000
```

Implementation-only self-test:

```bash
python audit_reasoning_flow_signatures.py \
  --selftest \
  --compute_device cuda \
  --assert_gates
```

## 8. What is and is not new

Path signatures, Levy area, random projection, and one-class distances are
established tools. The proposed research contribution is their claim-driven
combination for LLM reasoning: a question-conditional correct-flow support
test over multi-layer hidden trajectories, with order-one, chronology,
length, and same-problem falsification controls. Novelty must ultimately be
established by literature review and real-data gains; the implementation does
not assert it by naming the method.
