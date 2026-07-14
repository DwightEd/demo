# Output-Conditioned Geometric Predictive Information

## One-paragraph story

Two reasoning traces can expose nearly identical current confidence while occupying different internal computational states. OC-GPI tests whether this difference is real and useful: first predict internal geometry from a rich causal sketch of the output-distribution history, remove the predictable component, learn a low-dimensional chart of the remaining geometry without labels, and ask whether that residual chart forecasts future output drift or response failure. The method therefore does not claim that curvature, rank, or dispersion is intrinsically an error score. It asks a sharper question: **does geometry reveal a pending change in model belief that the model's current observable output distribution has not revealed yet?**

## Frozen research target

For causal length controls $C_{\le t}$, prompt-induced output-distribution
history $Z_{\le t}$, internal geometry $G_{\le t}$, future output change
$\Delta Z_{t\rightarrow t+k}$, and response error $Y$, the primary target is
model-relative conditional usable information:

\[
I_{\mathcal V}(Y;G_{\le t}\mid Z_{\le t},C_{\le t})
=
\mathcal L_{\mathcal V}(Y\mid Z_{\le t},C_{\le t})
-
\mathcal L_{\mathcal V}(Y\mid Z_{\le t},G_{\le t},C_{\le t}).
\]

The mechanism target is the corresponding future-output forecast increment:

\[
R^2_{\mathrm{partial}}
=
1-
\frac{\sum_i\left\|\Delta Z_i-
\widehat{\Delta Z}_{i,Z+G}\right\|_2^2}
{\sum_i\left\|\Delta Z_i-
\widehat{\Delta Z}_{i,Z}\right\|_2^2}.
\]

This is not an unrestricted mutual-information estimator. It is explicitly
relative to a frozen predictor family and must be reported as such.

The default ProcessBench label is `process_error`, namely

\[
Y=\mathbb 1[\texttt{gold\_error\_step}\ge 0].
\]

Final-answer correctness is available only as an explicit secondary
`--label_policy final_answer` analysis. The audit refuses this run when the
source did not explicitly provide a final-answer label; process correctness is
never substituted for a missing answer label.

## Hypotheses

### H1: output-to-geometry coupling

A nonzero fraction of geometry should be recoverable from causal logits and
observed prefix-length controls:

\[
\widehat G_{\le t}=f_Z(Z_{\le t},C_{\le t}).
\]

This fraction is not novel information. It includes confidence, lexical and
length effects already visible at the output.

### H2: predictive residual geometry

The conditional residual

\[
R^G_{\le t}=G_{\le t}-\widehat G_{\le t}
\]

contains low-dimensional modes that predict future output change:

\[
\Delta Z_{t\rightarrow t+k}
=f_{Z}(Z_{\le t},C_{\le t})+a(U^\top R^G_{\le t})+\epsilon.
\]

This is the main mechanism claim. It must beat a dimension- and
prefix-length-matched permutation null. Null donors are drawn across problem
groups within cumulative-prefix-length strata, so repeated rows from the same
problem cannot masquerade as a destroyed pairing. Forecast controls use only
causally observed prefix lengths; final response length and relative position
are never inputs.

### H3: detector increment

If H2 holds, the same residual modes may improve online response-error
detection beyond a rich logits-only baseline. The deployable task applies one
shared detector to every observed prefix; it never uses the response's eventual
length to select the prefix. Relative 25/50/75/100 percent checkpoints are
retained only as retrospective diagnostic slices. H3 is secondary to H2: a small
AUROC gain without future-output forecasting evidence is a detector result,
not a mechanism discovery.

## Compact output trace

For every response token at absolute position $p$, features are computed
from the logits at $p-1$, which predict that token. The extractor stores:

- normalized entropy;
- chosen-token log probability;
- top-1/top-2 logit margin;
- top-5 and top-20 probability mass;
- chosen-token log rank;
- Jensen-Shannon and Hellinger velocity between adjacent distributions;
- a fixed signed count-sketch of the complete vocabulary distribution;
- a second count-sketch of top-$k$ token probabilities.

The two count-sketches retain both probability-tail structure and high-mass
token identity without saving a vocabulary-sized tensor. All heavy
softmax/top-$k$ work is performed on GPU in small token chunks. Full logits
are never written to disk.

Step summaries retain mean, maximum, last value, and slope. The forecast target
uses only `.last` coordinates so that

\[
\Delta Z_{t\rightarrow t+k}=Z^{\mathrm{last}}_{t+k}-Z^{\mathrm{last}}_t
\]

has a clear current-state interpretation.
Forecast errors are evaluated after scaling every target coordinate with the
corresponding outer-training-fold mean and standard deviation; no test-fold
scale is used.

## Geometry map

Given step-pooled layer states $h_{t,\ell}$, first normalize coordinates:

\[
u_{t,\ell}=\frac{h_{t,\ell}}{\|h_{t,\ell}\|_2}.
\]

The label-free map includes:

- depth path length, endpoint displacement, and tortuosity;
- depth update turning angle and directional dispersion;
- spectral entropy, effective rank, and anisotropy of layer updates;
- temporal velocity, velocity variation, and cross-layer directional dispersion;
- adjacent-layer transport misalignment;
- temporal turning and temporal spectral entropy;
- depth-time coupling misalignment;
- final-layer velocity and turning as explicit negative controls;
- optional ICR/residual-mismatch and legacy spread features with provenance tags.

These features are invariant to one global orthogonal basis change and one
global positive scaling of the hidden states. The implementation does not call
these scalars a manifold by themselves.

For extraction efficiency, chains with the same `[step, layer, hidden]` shape
are bucketed and the complete label-free geometry map is evaluated as a GPU
batch. Variable-length chains remain separate buckets, so no padding can enter
the geometric statistics.

## Conditional geometry chart

Every outer training fold performs the following operations using training
problems only:

1. Median-impute and standardize output and geometry histories.
2. Fit ridge regression $G=f_Z(Z)$.
3. Form residual geometry $R^G=G-f_Z(Z)$.
4. Fit an unsupervised PCA chart on $R^G$, retaining 95% variance with a
   fixed maximum dimension.
5. Fit a regularized residual adapter in chart coordinates.

The chart is not a VAE and does not use correctness labels. Its purpose is
specific: represent geometry that is linearly unavailable from the chosen
output baseline, rather than reconstruct all hidden states.

## Strict cross-fitting

- Outer folds are grouped by `problem_id`.
- The response text hash, problem ID, step count, and observer-model metadata
  are checked before trace/geometry joining. Exact-trace tokenizer identity is
  checked during compact-trace extraction when the source declares it.
- Within each outer training fold, the logits-only model generates inner
  out-of-fold predictions.
- Binary residual adapters use these inner predictions as fixed offsets.
- Continuous adapters learn only the inner out-of-fold forecast residual.
- Geometry is residualized and charted using outer-training rows only.
- Null geometry is permuted within approximate length bins.
- Training losses and reported row-level errors use inverse problem-frequency
  weights, so long chains or heavily sampled problems do not dominate.

This prevents a geometry adapter from exploiting an overfit baseline's
training residual or from seeing a held-out problem during chart learning.
The prompt determines the replayed trajectory and defines the grouping unit,
but no unimplemented prompt embedding is claimed as a detector input.

## Reports

The audit writes:

- `summary.md` and `summary.json`;
- `oof_predictions.npz`;
- `geometry_explained_by_output.csv`;
- `conditional_geometry_importance.csv`.

The primary quantities are:

\[
\Delta\mathrm{NLL},\quad
\Delta\mathrm{Brier},\quad
\Delta\mathrm{AUROC},\quad
I_{\mathcal V}/\log 2,\quad
R^2_{\mathrm{partial}}.
\]

All increments receive problem-cluster bootstrap confidence intervals.
Every task also reports the frozen three-model comparison

\[
M_Z,\qquad M_{C+G},\qquad M_{Z+G},
\]

where $C$ contains only length/position nuisance controls. The paper claim is
based on $M_{Z+G}-M_Z$, not on the usually easier $M_{C+G}$ score.

## Claim gate

The mechanism claim requires both:

1. future-output partial $R^2$ has a confidence interval above zero;
2. the ordered residual geometry beats the length-matched null.

The detector claim additionally requires positive response usable information
and a positive AUROC increment over the null. Whole-layer geometry and at least
100 problem groups are required for a confirmatory run. Sparse legacy
`stepvec` runs are exploratory.

## What would be a new finding

The framework itself is a new experimental paradigm only if it changes what
can be learned from the data. A publishable mechanism finding would be one of:

- intermediate-layer transport modes forecast future confidence collapse while
  final-layer geometry does not;
- ICR residual mismatch is weakly visible in logits but strongly predicts their
  next-step movement;
- different ProcessBench regimes show different residual modes, such as
  constraint-loss in GSM8K versus capability-boundary expansion in OmniMath;
- the conditional modes support an intervention that changes future logits and
  error probability in the predicted direction.

Without one of these results, OC-GPI remains a rigorous negative-result audit,
not a completed paper claim.
