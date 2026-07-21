# Causal Constraint Transport Refactor

## Research Contract

The new mainline tests one falsifiable hypothesis: the first reasoning error
coincides with loss of output-effective transport from prompt constraints and
an output-relevant escape from their local write subspace.

For source token $j$, receiver $t$, layer $l$, and head $h$,

\[
c_{j\to t}^{l,h}
=a_{tj}^{l,h}
\left\langle
W_O^{l,h}W_V^{l,h}\operatorname{LN}(h_j^l),g_t^l
\right\rangle.
\]

This separates QK routing, OV content, and output sensitivity. Prompt-origin
writes span a local constraint bundle $T_t^l$. For residual update $u_t^l$,
the primary escape measure is

\[
e_t^l=
\frac{
\left|\left\langle(I-\Pi_{T_t^l})u_t^l,g_t^l\right\rangle\right|
}{
\lVert u_t^l\rVert_2\lVert g_t^l\rVert_2+\epsilon
}.
\]

This is not ambient curvature: both the reference subspace and the measured
escape are conditioned on the problem and on output-sensitive directions. The
norm-product denominator keeps the directional escape bounded and avoids a
singularity when parallel and transverse logit effects cancel.

## Hypergraph Criterion

A source set is promoted to a hyperedge only when an intervention gives a
non-additive effect:

\[
\operatorname{Syn}(S\to t)
=\Delta\ell(S\to t)-\sum_{j\in S}\Delta\ell(j\to t).
\]

Below the synergy threshold, the builder emits directed pair edges. Higher-order
structure is therefore an empirical claim rather than an architectural prior.

## Objective

For step hazards $r_s$, response risk is

\[
P(\mathrm{error})=1-\prod_s(1-r_s).
\]

Correct traces contribute survival terms at every step. A trace whose first
error is $k$ contributes survival terms only for $s<k$ and one event term at
$k$; post-error steps are ignored.

## Clean Architecture

```text
hypergraph/attention/
  cct/
    contracts.py       fixed numerical contracts
    processbench.py    typed data parsing and token alignment
    hf_backend.py      compact OV/gradient/intervention extraction
    contribution.py    output-effective transport
    geometry.py        constraint bundle and transverse escape
    hypergraph.py      intervention-calibrated graph construction
    pipeline.py        single mechanism assembly path
    data.py            versioned pickle-free trace repository
    model.py           directed receiver-only encoder
    hazard.py          first-error survival objective
    training.py        normalization, optimization, and evaluation
    cli.py             extract / inspect / train / benchmark
  evaluation.py        discrimination, calibration, and localization
  splitting.py         problem-disjoint fixed holdout
```

This is a clean break. The threshold-attention baseline, generic training CLI,
compatibility aliases, and reproduction wrappers are retired.

## Compact Contract

Directly storing every full hidden-dimensional OV write costs $O(HSD)$. The
backend spans a small output-relevant basis with all step output gradients and
residual updates, then stores

```text
content_effect  [heads, steps, sources]
source_writes   [steps, sources, projected_rank]
```

This preserves all inner products used by CCT-HG while reducing storage to
$O(HQS+QSR)$. Core dataclasses use explicit fields; no free-form metadata is
passed or serialized.

The observer forward uses SDPA and captures only the selected transformer
block. Required attention rows are reconstructed from that block's Q/K and
RoPE states, so extraction never materializes all-layer attention tensors.
Training forms disjoint variable-size graph batches and performs one GPU
forward per batch while retaining equal response-level survival weighting.

## Validation Gates

1. Contribution scores must predict held-out logit changes under ablation.
2. Hypergraph models must beat hidden-only, no-edge, pairwise, and
   receiver/cardinality-preserving rewired controls on exactly the same split.
3. Escape effects must survive same-problem comparison and length/position
   controls.
4. AUROC/AUPRC must be accompanied by calibration, balanced accuracy, MCC, and
   problem-level uncertainty intervals.
5. Model selection uses validation only; the fixed test partition is evaluated
   once after early stopping.
