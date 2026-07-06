# Hypothesis Evidence Matrix

Date: 2026-07-06

This note collects the current status of the project's major hypotheses.  It
separates validated anchors, retired branches, and untested next steps.  The
goal is to avoid re-running old walls under new names.

## Online Segmentation Assumption

For a deployable monitor, assuming pre-segmented reasoning steps is too strong.
Step labels should be used for offline validation, not as a required runtime
input.

The deployable version should be boundary-free:

```text
for each generated token t:
  h_t^l -> normalize to u_t^l
  multi-scale recent windows W in {8,16,32,64}
  R_t(W,l) = || sum_i exp(-beta*(t-i)) u_i^l || / sum_i exp(-beta*(t-i))
  spread_t(W,l) = 1 - R_t(W,l)
```

This gives token-level kappa-like concentration and breadth without a parsed
step.  Step ranges, when available, are used only to score whether alarms align
with human or dataset boundaries.

Required controls:

- multi-scale windows to trade latency and noise;
- causal z-scoring from previous tokens only;
- conformal thresholds from correct prefixes;
- report FPR, recall, delay, and endpoint fraction;
- compare against entropy and length/position baselines.

## Evidence Matrix

| Hypothesis / direction | Status | Evidence | Decision |
|---|---|---|---|
| H1: wrong reasoning step has lower directional concentration / higher spread | validated anchor | ProcessBench L14 `spread/resultant` around AUROC 0.772; same-problem `cloud_spread` around 0.634 on 5shot and 0.659 on custom | Keep as core physiological signal, but not as universal detector |
| H2: signal is directional, not only norm | mostly supported | unit-sphere features improve over norm-only by roughly +0.049 OOF; however some raw norm/energy proxies remain confounded | Keep kappa/resultant, keep norm as control |
| H3: confidence and geometry are complementary | supported | `anchor_uncertainty` / geometry+uncertainty group around OOF AUROC 0.811 vs static 0.779 on full GSM8K | Keep fusion baseline; do not claim entropy is obsolete |
| H4: dynamic precursor / rupture before the error | retired in current channels | temporal rupture, event study, HSMM, and path-kernel variants do not beat static spread; alarms are often late or endpoint-like | Do not sell current signals as early precursor |
| HSMM / latent EM over scalar emissions | negative | HSMM full AUROC 0.538, censor80 0.506 vs static `mean:cloud_spread` 0.682 | Retire scalar-state HMM |
| Path kernel / functional shape | negative | best static `mean:cloud_spread` 0.683 vs best witness 0.668; shape witness about 0.615 | Retire simple path-kernel shape models |
| Step-token Gram / second moment / spectral-tail hidden geometry | negative | `gsm8k_v2_custom`: baseline 0.685; best Gram group `token_matrix_level` 0.660; `token_spectral_tail` 0.634, increment -0.051 CI [-0.080,-0.020] | Retire direct hidden Gram/spectral-tail as mainline claim |
| Long-tail / spectral-tail inflation | negative in same-problem OOF | spectral-tail group is below baseline and remains below in high-spread, low-entropy, and hard-problem subsets | Treat as tested failure for current data |
| Regex arithmetic premise ledger | negative | custom: constraint 0.562 vs baseline 0.659, inc -0.097; 5shot: 0.548 vs 0.634, inc -0.085 | Keep as negative baseline; do not tune regex |
| Fallback AnchorFlow with qvec partitions | negative as semantic claim | transport+baseline adds only +0.001 to +0.003 ns; shuffled-kind behaves like transport-only | Retire q_partition fallback; real prompt-span anchors still untested |
| Manifold health v1: q_align + cloud volume + spread | negative or too weak | manifold alone below anchor_uncertainty; cloud_V almost equals logN/difficulty | Retire v1 scalar formula; volume must be anchor/length-conditioned |
| Hypergraph / token relational reader | promising but heavy | HGN step AUROC about 0.760 and first-error top1 about 0.854 | Keep as evidence that relations exist, not as lightweight hook |
| Attention graph spectral metrics | untested locally | `Geometry of Reason` uses attention graph Laplacian; our Gram tests are hidden-cloud, not attention-graph | Needs attention maps and same-problem increment tests |
| Token-level boundary-free kappa / alpha | untested as main online monitor | Current saved step-level data can validate boundary alignment; online extraction must save per-token hidden | Next priority if deployment is the goal |
| Perturbation-normalized geometry | untested | Inspired by `What do Geometric Hallucination Detection Metrics Actually Measure?`; could normalize original response against counterfactual perturbations | Next priority for coherent-but-wrong / domain-shift control |
| Operation / premise-choice consistency | untested | Regex arithmetic failed because valid equations can still use wrong operation or binding | Next semantic branch after perturbation design |
| Causal evidence flow / attention lookback | untested locally | Step-saliency hook exists in `step-saliency`; attention intervention infrastructure is available | Next white-box branch after signal-only audits |

## Why Spectral Geometry of Thought Reports High AUC

The paper reports high in-distribution AUC for spectral alpha, including a
perfect AUC on Qwen2.5-7B.  This does not contradict the project's negative
same-problem Gram results because the evaluation target is different:

- their correctness predictor is chain/problem-level, not same-question
  multi-sample paired ranking;
- it uses 5-fold stratified CV over generated problems, not same-problem
  contrastive folds;
- it selects layer/phase spectral alpha and predicts final correctness after
  much of the reasoning trace is already available, though before final answer;
- OOD validation is much weaker than the in-distribution value;
- the paper itself notes token dynamics are tested mainly on one model and a
  limited task set.

Therefore the useful idea is not "alpha will solve our detector".  The useful
idea is boundary-free token monitoring:

```text
token-level alpha / kappa traces
phase-change peaks
alignment to step boundaries only for validation
OOD and same-problem controls
```

## Next Concrete Validation

The next deployable audit should not require pre-split steps.

Proposed script:

```text
token_stream_geometry_audit.py
```

Inputs:

- full hidden shards or per-token hidden arrays;
- optional step ranges for validation only;
- labels: `gold_error_step` for ProcessBench or final correctness for
  same-problem samples.

Signals:

- multi-scale streaming kappa/spread over token windows;
- token-level spectral alpha on sliding windows;
- derivatives: `d_kappa`, `d_alpha`, local slope, local amplitude;
- entropy/logprob traces when available;
- boundary-free phase score from peaks in derivative magnitude.

Gates:

- same-problem paired AUROC for final correctness;
- ProcessBench first-error boundary alignment;
- FPR/recall/delay at fixed conformal FPR;
- increment over static spread/resultant + entropy + length;
- endpoint fraction.

Pass condition:

```text
The token-stream signal improves online alarm recall/delay or rescues
baseline-missed same-problem pairs without relying on parsed step boundaries.
```

If it fails, then the project should accept that the current hidden-geometry
family is a weak physiological marker rather than a standalone online detector,
and move to perturbation-normalized geometry or attention/evidence-flow.
