# Signal Validity And Mechanism Plan

Date: 2026-07-08

This plan answers a specific paper-design question:

```text
Which current signals are real evidence, what hypotheses do they support,
and what mechanism experiments are still required before the paper can make
a top-venue claim?
```

The main conclusion is deliberately conservative.  We have already run
EDIS-like response-level tests, but the previous reports are not yet a
publishable diagnostic package.  They show that hidden geometry and entropy
dynamics operate at different granularities:

- hidden geometry is strongest for step-level first-error localization;
- entropy dynamics are strongest for response-level correctness selection;
- a paper-level method should use a step hazard model to bridge the two,
  rather than average unrelated scalars.

## Current EDIS-Like Coverage

We have already run several EDIS-adjacent analyses.

| Script | Granularity | What It Tested | Current Readout |
|---|---|---|---|
| `resp_detect.py` | response | EDIS-style dynamics on the per-step \(\kappa\) trajectory, plus min/mean \(\kappa\), plus entropy EDIS | \(\kappa\)-GDIS is weak, about \(0.555\)-\(0.619\).  Entropy EDIS is strong at response level: \(0.736/0.731/0.812/0.739\). |
| `resp_cusum.py` | response / online | CUSUM-style aggregation of geometry deviations | helps on GSM8K, weak after length control on harder tasks. |
| `step_edis_vs_geom.py` | step | within-step entropy EDIS versus step resultant for first-error localization | step geometry is the right object for first-error localization; entropy dynamics are not the whole story. |
| `within_edis.py` | step | EDIS-form cumulative instability inside a semantic step, applied to ordered hidden-direction fan-out | tests whether within-step order adds signal beyond static concentration; current evidence suggests static concentration dominates. |
| `trajectory_signal_report.py` | descriptive | per-chain trajectory shape and distribution summaries | useful for exploration, but not enough for mechanism claims. |

So the answer is:

```text
Yes, we have response-level EDIS-like analyses.
No, they are not yet comprehensive enough for paper evidence.
```

The missing part is not another scalar.  The missing part is a diagnostic
evidence chain: distribution, monotonicity, calibration, controls, mechanism
decomposition, response aggregation, and intervention.

## Paper-Level Hypothesis

The current paper hypothesis should be sharpened to:

\[
\text{reasoning failure}
=
\text{local computation fragmentation}
\;\lor\;
\text{source anchoring failure}
\;\lor\;
\text{decoder uncertainty instability}.
\]

This separates three mechanisms that previous reports sometimes mixed.

### Mechanism 1: Local Computation Fragmentation

Within a reasoning step, hidden token directions should form a coherent local
computational state.  Let \(h_{j,i}^{\ell}\) be the layer-\(\ell\) hidden state
for token \(i\) in step \(j\), and define

\[
u_{j,i}^{\ell}
=
\frac{h_{j,i}^{\ell}}{\|h_{j,i}^{\ell}\|_2}.
\]

The step consensus score is

\[
\kappa_j^{\ell}
=
\left\|
\frac{\sum_i w_i u_{j,i}^{\ell}}{\sum_i w_i}
\right\|_2,
\qquad
\mathrm{spread}_j^{\ell}=1-\kappa_j^{\ell}.
\]

Low \(\kappa_j^{\ell}\) supports only this claim:

```text
the step token cloud lost local directional consensus.
```

It does not prove that the model is aware of the error, detached from the
question, or following an invalid source.  Those require separate tests.

### Mechanism 2: Source Anchoring Failure

A step can be internally coherent and still wrong.  This is the
coherent-but-wrong case.  It requires a source model:

\[
\mathrm{source}_j
\in
\{\text{question},\text{verified prior step},\text{tainted prior step},
\text{recent self text},\text{other}\}.
\]

The hypothesis is:

\[
\text{coherent wrong}
\approx
\text{high local consensus}
\;\land\;
\text{attention or hidden alignment to the wrong source}.
\]

Our current `constraint_anchor_flow_audit.py` does not yet establish this.
Its strong within-problem numbers are likely position-contaminated, and its
cross-problem increment is small.  Anchor/source attribution remains a design
target, not a validated contribution.

### Mechanism 3: Decoder Uncertainty Instability

EDIS works on token entropy

\[
H_t
=
-\sum_{v\in V} p_t(v)\log p_t(v).
\]

It detects burst and rebound events:

\[
S_{\mathrm{burst}}
=
\sum_t
\mathbf{1}[H_{t+w}-H_t>\tau_b],
\]

\[
S_{\mathrm{rebound}}
=
\sum_t
\mathbf{1}[H_t-\min_{s<t}H_s>\tau_r],
\]

and combines them as

\[
\mathrm{EDIS}(H)
=
\frac{S_{\mathrm{burst}}+S_{\mathrm{rebound}}}{2}
\left(1+\mathrm{Var}(H)\right).
\]

This signal supports:

```text
incorrect responses often have unstable confidence trajectories.
```

It does not directly explain hidden-state fragmentation.  In our data,
geometry and entropy are complementary rather than equivalent.

## Effective Signals, Claims, And Missing Proof

| Signal family | Current evidence | What it can claim | What it cannot claim yet | Required next proof |
|---|---|---|---|---|
| \(\kappa\) / `spread` | step AUROC around \(0.70\)-\(0.77\); strict geometry increment over entropy/EDIS/length is positive in ProcessBench; same-problem signal exists but weaker | first-error steps show lower local directional consensus | source detachment, causal awareness, response correctness by itself | monotone risk bins, matched length/position/problem controls, calibration, low-\(\kappa\) morphology |
| multi-layer \(\kappa\) family + entropy battery | strongest current step detector; fair increment over EDIS exists in same classifier | geometry carries information not reducible to entropy | a single scalar explains everything | feature-group ablation and residualized increment table |
| residual/effective rank | high-rank dispersion subtype has elevated error rate; rank alone does not beat spread globally | some low-\(\kappa\) errors are broad residual dispersion | rank is independent detector across all settings | condition on \(\kappa\) and length; show morphology taxonomy |
| prefix innovation | code implemented; remote results pending | direct test of "step-flow break" as leaving prior-step subspace | established mechanism | same-problem and transition-matched increment over spread/length/position |
| entropy EDIS | strong response-level AUROC in existing scripts | response-level reliability and selection | step-local hidden mechanism | response-level comparison against noisy-or step hazard |
| anchor/source flow | conceptually important; current implementation not reliable | possible route for coherent-but-wrong failures | validated detector | typed spans, attention-backed source mass, hidden-source alignment, position-matched null |
| whole-chain Gram / effective rank | response-level geometry object exists; risk of length/difficulty entanglement | final response geometry may summarize global reasoning state | online first-error localization | exact paper metric replication, length controls, perturbation-normalized response score |
| hypergraph reader | strong localization readout | hidden token-step-layer relations contain signal | lightweight mechanistic scalar | use as heavier relational upper bound, not core hand metric |

## Necessary Experiments Before Top-Venue Claims

### Experiment A: Distribution Law

Question:

```text
Do wrong steps occupy a different signal distribution, or do we only see a
thin AUROC number?
```

For each dataset and signal, report:

- correct/error histograms and ECDFs;
- medians, quartiles, Cohen's \(d\), and odds ratios;
- decile risk curves;
- length-stratified and position-stratified versions;
- same-problem paired distributions when multiple samples exist.

Required pass condition:

\[
P(y=1\mid S\in \text{risk bin})
\]

must change monotonically or near-monotonically across bins after controlling
for length and position.  If the curve only works cross-problem, the signal is
a difficulty proxy.

### Experiment B: Structural Validity

Question:

```text
Does the scalar encode structured information aligned with correctness, not
random fluctuation?
```

Run four nulls:

1. shuffle labels within length buckets;
2. shuffle step positions within each chain;
3. permute signals within each problem;
4. residualize signal against \(\log n\), step index, number of steps, and
   entropy.

Report:

\[
\Delta_{\mathrm{resid}}
=
\mathrm{AUROC}(S_{\mathrm{resid}})
-\mathrm{AUROC}(\text{controls}).
\]

Required pass condition: the residualized signal remains above null with
bootstrap confidence interval excluding zero.

### Experiment C: Calibration And Selective Risk

Question:

```text
Can the scalar map to a usable error probability, not just a ranking?
```

Fit calibration only on training folds:

\[
\hat p_j
=
\mathrm{IsoCal}(S_j)
\quad\text{or}\quad
\hat p_j=\sigma(aS_j+b).
\]

Report ECE, Brier score, NLL, and risk-coverage curves.  The paper should not
claim a "risk monitor" unless calibration is usable.

Required pass condition:

- calibrated bins have increasing observed error;
- risk coverage improves over entropy-only and length-only baselines;
- calibration transfers across held-out problems.

### Experiment D: Mechanism Morphology Of Low \(\kappa\)

Question:

```text
When \(\kappa\) is low, what geometry produced it?
```

Use the identity

\[
A_j=\sum_i w_i u_{j,i}u_{j,i}^{\top},
\qquad
C_j=A_j-\mu_j\mu_j^{\top},
\qquad
\mathrm{tr}(C_j)=1-\kappa_j^2.
\]

The trace is not new information.  Only the eigenvalue shape of \(C_j\)
can add mechanism:

- high residual effective rank: broad dispersion;
- high \(\lambda_1(C_j)\) with sign balance: bipolar cancellation;
- multi-cluster Gram structure: mixed sub-computations;
- early/late coherent halves with different means: legitimate multi-substep
  complexity.

Required pass condition:

\[
\mathrm{morphology}(y=1\mid \kappa\text{ bin},\text{length bin})
\neq
\mathrm{morphology}(y=0\mid \kappa\text{ bin},\text{length bin}).
\]

If morphology cannot distinguish errors after matching \(\kappa\) and length,
then rank/spectrum are explanations for \(\kappa\), not independent signals.

### Experiment E: Step-To-Response Hazard

Question:

```text
How do step-local signals detect final response correctness without losing
localization?
```

Do not average step scores as the primary response detector.  Use a hazard:

\[
p_j
=
P(\text{first error at step }j\mid x,y_{<j}),
\]

\[
P(\text{response wrong})
=
1-\prod_j(1-p_j).
\]

Compare against:

- entropy EDIS;
- mean entropy;
- min/max/mean \(\kappa\);
- whole-chain Gram metrics;
- CUSUM variants.

Before fitting a response hazard, diagnose whether the step signal actually
appears as a phase transition.  For a risk-high step signal \(s_j\), compute
prefix-relative scores:

\[
z^{\mathrm{level}}_j
=
\frac{s_j-\mathrm{median}(s_{<j})}{\mathrm{scale}(s_{<j})},
\]

\[
z^{\mathrm{jump}}_j
=
\frac{s_j-s_{j-1}}{\mathrm{scale}(s_{<j})},
\]

\[
z^{\mathrm{break}}_j
=
\max(0,z^{\mathrm{level}}_j)
+
\max(0,z^{\mathrm{jump}}_j).
\]

This separates three cases that min/mean aggregation collapses:

- stable prefix, sharp first-error break;
- gradual drift before the first error;
- persistently unstable but not locally diagnostic trajectories.

The concrete audit script is `trajectory_phase_transition_audit.py`.  It
reports aligned profiles around the gold first-error step, event ranks within
each chain, and mode counts for the three cases above.

Required pass condition:

```text
step-hazard noisy-or should match or improve response-level detection while
preserving first-error localization.
```

If it fails, the honest story is granularity separation: geometry is a
step-level localizer, while EDIS is the response-level selector.

### Experiment F: Coherent-But-Wrong Boundary

Question:

```text
Which errors are invisible to \(\kappa\)?
```

Construct regions:

- fragmented wrong: high spread, high entropy or high rank;
- coherent wrong: low spread, low entropy, wrong answer;
- uncertain correct: high entropy, correct;
- long legitimate step: high spread, correct, long step.

For coherent wrong, \(\kappa\) should fail by design.  The needed signal is
source attribution:

\[
\mathrm{lostAnchor}_j
=
1-\mathrm{mass}_j(\text{question or verified prior steps}),
\]

\[
\mathrm{coherentWrong}_j
=
\mathbf{1}[\kappa_j\text{ high}]
\cdot
\mathbf{1}[\mathrm{lostAnchor}_j\text{ high}].
\]

Required pass condition: source/anchor features improve specifically inside
the coherent-wrong region, not only through position or length.

### Experiment G: Intervention And Causal Evidence

REDEEP is persuasive because it does not stop at detection.  It runs
mechanism-aligned intervention: add attention to external context, reduce FFN
parametric injection.

Our analog should be:

1. low-risk intervention: triggered verification or re-anchoring prompt;
2. medium-risk intervention: regenerate from the last healthy step;
3. high-risk mechanistic intervention: patch or steer hidden states only after
   a clear component is identified.

For the current paper, the minimum intervention is:

\[
\text{trigger}(j)=\mathbf{1}[\hat p_j>\tau],
\]

then ask the model to re-evaluate the current step against the problem
conditions and prior verified steps.  Report correction rate, false-trigger
cost, and answer accuracy.

Required pass condition: triggered correction improves final accuracy at a
fixed compute budget over random-trigger and entropy-trigger baselines.

## REDEEP-Style Paper Organization To Borrow

REDEEP's strength is not that it has many metrics.  It has a clean chain:

```text
problem reframing -> causal mechanism study -> two decoupled scores ->
regression detector -> ablation -> intervention -> efficiency -> case study
```

Our analogous chain should be:

```text
reasoning failures are not only output uncertainty
-> first-error steps lose hidden directional consensus
-> low consensus has identifiable morphologies
-> source anchoring catches coherent-wrong failures
-> step hazards aggregate to response risk
-> triggered re-anchoring improves outcomes
```

## Figure And Table Plan

| Artifact | Purpose | Required Content |
|---|---|---|
| Figure 1 | Motivation | correct vs wrong step hidden-token cloud sketch plus entropy trace contrast |
| Figure 2 | Signal distributions | \(\kappa\), spread, entropy EDIS, rank distributions with matched controls |
| Figure 3 | Structural validity | risk decile curves and residualized curves |
| Figure 4 | Mechanism taxonomy | low-\(\kappa\) split into high-rank, bipolar, multicluster, ordered-shift |
| Figure 5 | Response bridge | step hazard noisy-or versus EDIS and whole-chain Gram |
| Figure 6 | Coherent-wrong | region plot: entropy versus spread, with anchor/source overlay |
| Table 1 | Main detection | step-level AUROC/AUPR/length-bucket/same-problem across datasets |
| Table 2 | Strict increments | geometry over entropy/EDIS/length; rank over spread; anchor over geometry |
| Table 3 | Calibration | ECE, Brier, NLL, risk coverage |
| Table 4 | Intervention | correction rate, final accuracy, trigger cost |

## Implementation Plan

### Script 1: `signal_validity_mechanism_audit.py`

Purpose: one-stop signal validity report.

Inputs:

- `full_gsm8k.npz`, `full_math.npz`, `full_omnimath.npz`, `full_olympiad.npz`;
- optional same-problem sampled files;
- optional stored hidden clouds.

Outputs:

- `signal_distributions.csv`;
- `risk_bins.csv`;
- `residualized_increment.json`;
- `calibration.json`;
- `mechanism_taxonomy.csv`;
- `signal_validity_report.md`.

Required computations:

- all scalar distributions;
- length/position/problem controls;
- bootstrap confidence intervals by chain;
- residualized AUROC;
- calibration;
- low-\(\kappa\) morphology.

GPU policy: if hidden clouds or Gram/eigendecomposition are required, use
`--backend auto|cpu|torch|cuda`, `--device`, and move each chain to GPU once.
Pure scalar reports can stay CPU.

### Script 2: `trajectory_phase_transition_audit.py`

Purpose: explain why step-level signals may vanish under response-level
aggregation.

Inputs:

- `stepcloud` and `cloud_feature_names`;
- `gold_error_step`;
- optional `tok_U_D` and `step_token_ranges` for entropy summaries.

Outputs:

- event-level first-error AUROC for `signal_value`, `level_z`, `jump_z`,
  `break_z`, and `shock_z`;
- response-level AUROC for max/mean aggregates;
- gold first-error rank within each chain;
- aligned transition profiles around the first error;
- first-error mode taxonomy.

Main diagnostic:

\[
z^{\mathrm{break}}_j
=
\max(0,z^{\mathrm{level}}_j)
+
\max(0,z^{\mathrm{jump}}_j).
\]

This script should be treated as the bridge between step-local geometry and
response-level hazard modeling, not as another final detector.

### Script 3: `response_hazard_audit.py`

Purpose: turn step signals into response-level risk without destroying
localization.

Inputs:

- step rows with \(p_j\) or raw signals;
- token entropy for EDIS;
- response correctness.

Outputs:

- response AUROC/AUPR;
- risk-coverage;
- response calibration;
- comparison against EDIS, mean entropy, min \(\kappa\), CUSUM, whole-chain
  Gram.

Main score:

\[
\hat R
=
1-\prod_j(1-\hat p_j).
\]

### Script 4: `coherent_wrong_source_audit.py`

Purpose: test source anchoring only where geometry should fail.

Inputs:

- typed span boundaries for question, step prefix, recent self text;
- attention if available;
- hidden span vectors if attention is unavailable.

Outputs:

- coherent-wrong subgroup AUROC;
- anchor-source mass table;
- position-matched nulls;
- case studies.

This script should not reuse the current anchor entropy as a headline metric.
It must test whether the current step leaves the valid prefix/source subspace
or routes attention to tainted/recent self text.

## Pass/Fail Interpretation

The paper claim should be upgraded only if the following gates pass:

1. \(\kappa\)/spread show monotone and calibrated step-level risk after length
   and position controls.
2. geometry adds over entropy/EDIS/length in a strict same-classifier setup.
3. low-\(\kappa\) morphology gives a credible mechanism taxonomy, even if rank
   does not improve the main detector.
4. response hazard competes with response EDIS or clearly explains why the
   two operate at different granularities.
5. coherent-wrong failures receive a separate source/anchor treatment.
6. at least one intervention shows that the diagnostic signal can improve
   generation or verification, not only post-hoc ranking.

If only gates 1 and 2 pass, the paper should claim:

```text
hidden geometry provides a step-level diagnostic signal complementary to
entropy dynamics.
```

If gates 3 and 5 pass, the paper can claim:

```text
reasoning failures split into fragmented wrong and coherent-but-wrong modes.
```

If gate 6 passes, the paper can claim:

```text
the diagnostic signal supports corrective inference-time control.
```

## Immediate Next Step

Run `trajectory_phase_transition_audit.py` first to inspect whether the
geometry signal is a local phase transition, gradual drift, or persistent
instability.  Then build `signal_validity_mechanism_audit.py`; it should not
introduce a new method.  It should make existing signals accountable by
reporting:

```text
distribution -> monotonicity -> residual increment -> calibration ->
mechanism subgroup -> response aggregation
```

Only after this report exists should we decide whether to invest in attention
source attribution or hidden-only prefix/source geometry.
