# Same-Problem Feasible-Tangent Gate

## Status

This is the geometry-first replacement for extending OC-GPI with more legacy
handcrafted metrics. It is a falsification gate, not yet a final detector.
Logits and output cotangents are intentionally excluded until the geometry
itself passes.

## Core Hypothesis

For a fixed problem (q), independently sampled correct responses locally
reuse a low-rank set of feasible state updates. At transition (t) and layer
(ell), let

\[
u_{i,t}^{(\ell)}=
\frac{h_{i,t+1}^{(\ell)}-h_{i,t}^{(\ell)}}
{\left\|h_{i,t+1}^{(\ell)}-h_{i,t}^{(\ell)}\right\|_2}
\]

be the unit update of response (i). Other correct responses to the same
problem define a local feasible tangent

\[
\mathcal T_{q,t}^{(\ell)}=
\operatorname{span}_r
\left\{u_{j,\tau_j(t)}^{(\ell)}:
j\neq i,\; y_j=\text{correct}\right\}.
\]

The target response is always excluded from its donor set. The transverse
residual and escape ratio are

\[
e_{i,t}^{(\ell)}=
\left(I-P_{\mathcal T_{q,t}^{(\ell)}}\right)u_{i,t}^{(\ell)},
\qquad
E_{i,t}^{(\ell)}=\left\|e_{i,t}^{(\ell)}\right\|_2^2.
\]

The claim has two logically ordered parts:

1. Correct held-out updates must be reconstructed better by their aligned,
   same-problem tangent than by matched structural nulls.
2. Only if that tangent exists may persistent normal escape be tested as an
   error signal.

If either part fails, adding logits cannot rescue the geometric mechanism
claim.

## Why Same-Problem Multisampling Is Necessary

The canonical `full_*.npz` files contain one response per problem. A
problem-conditioned feasible tangent cannot be identified from one response,
and cross-problem nearest neighbors silently replace “same constraints” with
semantic similarity. This audit therefore uses:

```text
data/gsm8k_v2_custom.npz
data/gsm8k_v2_5shot.npz
```

with `sv_vec_step_exp`, `problem_ids`, `sample_idx`, and final correctness.
No new model forward pass is needed.

The artifacts do not contain a prompt anchor, so the unobserved
prompt-to-first-step transition is excluded. It is not reconstructed using a
zero vector or another synthetic origin.

## Alignment Without Future-Length Leakage

For a target predecessor state, each correct donor contributes exactly one
update. Its donor transition maximizes

\[
\cos\!\left(h_{i,t}^{(\ell)},h_{j,\tau}^{(\ell)}\right)
-\frac{1}{2}
\left(
\frac{(\tau-t)/s}{\sigma}
\right)^2.
\]

Here (t) and (	au) are causal step indices, (s) is
`causal_time_scale`, and (sigma) is `phase_sigma`. Normalized response phase
is not used because it reveals final response length.

## Adaptive Rank Is A Gate, Not A Hyperparameter Excuse

Let (lambda_1geqcdots) be eigenvalues of the donor-direction Gram matrix.
The selected rank is the smallest rank satisfying

\[
\frac{\sum_{k=1}^{r}\lambda_k}{\sum_k\lambda_k}
\geq \rho,
\]

subject to (r\leq r_{\max}). If (r_{\max}) cannot capture (ho), the
transition is marked rank-unsupported. Its raw escape remains available for
debugging, but it is excluded from confirmatory scores. Thus the code cannot
force an arbitrary high-dimensional donor cloud to support the low-rank
hypothesis.

## Structural Nulls

Every null uses the same donor count and the rank selected by the primary
tangent.

1. **Phase-only:** same problem and nearest causal step, but no state matching.
2. **Time-shuffle:** same donor responses, with each selected transition moved
   to a different time whenever possible.
3. **Wrong-problem:** equally sized correct donor sets from control-matched
   other problems, aligned by the same state-time rule.
4. **Random subspace:** seeded Haar-like orthonormal subspace with matched rank.

The wrong-problem matcher uses median step count and response character count
only to make the null equally difficult. These controls never define the
primary tangent.

## Persistence

For each layer, response-level coherent escape is

\[
C_i^{(\ell)}=
\left\|
\frac{1}{T_i}\sum_t e_{i,t}^{(\ell)}
\right\|_2^2,
\]

and directional persistence is

\[
R_i^{(\ell)}=
\frac{\left\|\sum_t e_{i,t}^{(\ell)}\right\|_2^2}
{T_i\sum_t\left\|e_{i,t}^{(\ell)}\right\|_2^2}.
\]

Layers are summarized separately and then averaged; residual vectors from
different layers are never added as if they occupied one coordinate system.

## Two Decision Gates

### Gate 1: Geometric Existence

On correct held-out targets, compute one mean contrast per problem, then give
every problem equal weight in the bootstrap. The primary tangent must have
lower escape than time-shuffled and wrong-problem tangents, while rank support
and problem coverage exceed their thresholds. Phase-only is reported as a
state-conditioning subgate.

### Gate 2: Error Escape

The preregistered response score is coherent normal escape after cross-fitted
residualization against

\[
\log(1+T),\quad \log(1+C),\quad
\log^2(1+T),\quad \log^2(1+C),\quad
\log(1+T)\log(1+C),
\]

where (T) is step count and (C) is response character count. AUROC is first
computed independently within every contrastive problem and then averaged
equally across problems. Gate 2 additionally requires the primary tangent to
beat the time-shuffled tangent.

Only if both gates pass does the report recommend extracting an exact output
cotangent, for example

\[
M_t=J_t^\top
\left(\operatorname{diag}(p_t)-p_t p_t^\top\right)J_t,
\]

to test whether the normal residual is output-sensitive. That calculation is
not approximated by entropy, NLL, or an MLP score in this audit.

## Direct Run

```bash
python audit_feasible_tangent_gate.py \
  --input data/gsm8k_v2_custom.npz \
  --output_dir outputs/feasible_tangent/gsm8k_custom_l16 \
  --vector_key sv_vec_step_exp \
  --layers 16 \
  --label_policy answer_format_ok \
  --rank_energy 0.90 \
  --max_rank 4 \
  --min_donors 6 \
  --max_donors 12 \
  --wrong_problem_draws 3 \
  --device cuda \
  --bootstrap 2000 \
  --permutations 1000
```

Run `--preflight` first. Outputs are compact:

```text
feasible_tangent_scores.npz
chain_scores.csv
summary.json
summary.md
```

No tangent bases or full-dimensional residual vectors are persisted.
Layer alignment is chunked by `--layer_batch_size`, while donor-state cosine
matching and all Gram eigendecompositions run as batched PyTorch operations on
the selected device.

## Claim Boundary

A pass supports a local, supervised healthy-reference mechanism on the tested
observer model, dataset, layer, and sampling regime. It does not establish a
global smooth manifold, an online unsupervised detector, causal output impact,
or cross-model transfer. Those are separate experiments after this gate.
