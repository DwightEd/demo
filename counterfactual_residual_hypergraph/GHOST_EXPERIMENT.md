# GHOST-style mid-layer Mahalanobis experiment

This is the real ProcessBench path for testing the idea:

> fit a Mahalanobis geometry from normal examples at middle LLM layers, then
> detect held-out errors by distance from that geometry.

It is independent of the supervised CCT-HG experiment and does not use a
hypergraph, BCE loss, optimizer, or training epochs.

## Reproduction boundary

The supplied abstract states that GHOST uses six probe layers, a Mahalanobis
estimator, normal examples, and one forward pass. It does not state the six
exact layers, response pooling, covariance regularization, or layer fusion.
No public paper body or official code was found under the supplied title when
this implementation was prepared.

The implementation is therefore named:

`GHOST-inspired abstract-level mid-layer Mahalanobis`

It must not be described as an exact reproduction until those unspecified
choices are checked against the paper body.

The registered default choices for a 32-block Llama are:

- middle block-output depths: `8, 10, 11, 13, 14, 16`;
- corresponding zero-based block indices: `7, 9, 10, 12, 13, 15`;
- corresponding Hugging Face `hidden_states` indices: `8, 10, 11, 13, 14, 16`;
- control: normalized final `AutoModel.last_hidden_state`, depth `32`;
- primary response representation: mean over content tokens owned by the
  ProcessBench reasoning steps;
- preregistered sensitivity representation: final content token of the
  response;
- six middle layers are calibrated separately and then fused with a fixed
  equal-weight mean. The final layer never enters the middle-layer fusion.

## What is fitted

For layer \(l\), normal response vectors \(h_l(x)\) define

\[
\mu_l = \frac{1}{n}\sum_i h_l(x_i),
\]

\[
\Sigma_l =
  (1-\alpha)\widehat{\Sigma}_l
  + \alpha\frac{\operatorname{tr}(\widehat{\Sigma}_l)}{d}I
  + \epsilon I,
\]

and the anomaly distance is

\[
D_l(x) =
  (h_l(x)-\mu_l)^\top
  \Sigma_l^{-1}
  (h_l(x)-\mu_l).
\]

The default is `alpha=0.1`, `epsilon=1e-8`. The code uses an exact Woodbury
form of this shrinkage covariance. It does not introduce a random projection,
and it does not materialize a `4096 x 4096` inverse.

Each layer's distance is converted to an empirical anomaly percentile using
only validation-normal responses. The primary score is the equal mean of the
six middle-layer percentiles, calibrated once more against the
validation-normal fused distribution. AUROC and AUPR use the pre-final-ECDF
six-layer fusion score; final-layer and individual-layer AUROC/AUPR use raw
Mahalanobis distance. This avoids quantizing ranking metrics to the small
calibration set's ECDF grid.

A fixed `0.95` normal-tail percentile is used for thresholded diagnostics.
This is not a finite-sample guarantee of 5% FPR. With `n` calibration-normal
responses, the minimum rank-grid tail resolution is `1/(n+1)`. Every metric
record therefore includes calibration size, grid step, grid-implied tail rate, and
the actually achieved test FPR. In a small smoke run, sensitivity,
specificity, balanced accuracy, and MCC are pipeline diagnostics rather than
scientific 5%-FPR operating points.

## Leakage controls

The selected cohort is split by `problem_id`, 60/20/20 by default:

- train-normal: fits means and covariance;
- validation-normal: calibrates layer scales and the fixed normal-tail score;
- held-out test: receives frozen scores and is then used for metrics;
- train/validation errors: recorded as unused and never enter fitting or
  calibration.

The evaluator fails if fit/calibration contain an error, if their problem
groups overlap test, or if held-out test lacks either class. ProcessBench
`label=-1` identifies a normal answer; `label>=0` identifies a first error.
Consequently this is **normal-reference/one-class supervision**, not fully
label-free discovery of which examples are normal.

The unlabeled test score table is written before labels are attached for the
metric table. Both are retained for audit.

## One-forward-pass extraction

The observer is loaded with `AutoModel`, not a causal-LM head. Six block hooks
retain only a pooled `4096`-dimensional response vector. The normalized final
state is taken from the returned base-model state. Extraction uses:

- BF16;
- batch size 1;
- `torch.inference_mode()`;
- no KV cache;
- no attention tensors;
- no generation;
- no truncation;
- one teacher-forced forward pass per fixed response.

The prompt is rendered with the observer tokenizer's chat template. Step spans
are token-aligned, and only tokens owned by reasoning steps enter pooling. The
extractor verifies that separate offset tokenization of the response exactly
matches its token segment in the canonical full user/assistant chat; any BPE
boundary mismatch fails closed. BF16 model states are converted to FP32 before
response-mean accumulation.

Middle probes are decoder-block residual outputs, while the final control is
the normalized `AutoModel.last_hidden_state` after final RMSNorm. Consequently,
the comparison may include a final-normalization effect. The supported
estimand is “fixed six-middle-layer ensemble versus one normalized final-layer
control,” not an isolated causal effect of layer depth.

The ProcessBench responses in the current GSM8K file were produced by Qwen
generator models, while Meta-Llama-3.1-8B-Instruct is the observer. Thus this
tests Llama geometry on externally generated, fixed reasoning traces. It is not
a test of Llama detecting only its own generations. Generator-stratified
metrics are written because generator identity can be a major confounder.
The shared splitter is not generator-stratified; the audit reports fit,
calibration, and both test-class coverage for every generator. Pooled AUROC is
not sufficient when coverage is incomplete or within-generator AUROC is
undefined.

## Remote commands

From the declared server repository:

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/demo
git pull --ff-only origin main
cd counterfactual_residual_hypergraph

PYTHON_BIN=python3 \
INSTALL_DEPS=0 \
PROFILE=smoke \
bash scripts/run_ghost_real_4090.sh
```

If the dependency preflight reports a missing package, rerun once with
`INSTALL_DEPS=1`. The requirements file deliberately does not install or
replace the server's CUDA PyTorch.

After the 48-trace smoke run succeeds:

```bash
PYTHON_BIN=python3 \
INSTALL_DEPS=0 \
PROFILE=pilot \
bash scripts/run_ghost_real_4090.sh
```

The 160-trace pilot is the first useful directional result. For the final
natural-prevalence run over every eligible GSM8K trace:

```bash
PYTHON_BIN=python3 \
INSTALL_DEPS=0 \
PROFILE=full \
bash scripts/run_ghost_real_4090.sh
```

`smoke` and `pilot` select balanced, problem-unique cohorts so both test
classes are reliably present. `full` includes every eligible trace, retains
natural prevalence, and keeps repeated questions in one split group.

## Result locations

Runs are created under:

```text
counterfactual_residual_hypergraph/outputs/ghost_remote_4090/
```

Find the newest run and inspect its main result:

```bash
RUN_DIR="$(find outputs/ghost_remote_4090 -mindepth 1 -maxdepth 1 \
  -type d -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"

cat "${RUN_DIR}/summary.json"
cat "${RUN_DIR}/evaluation/metrics.json"
head -n 5 "${RUN_DIR}/evaluation/scores-test.csv"
```

Important artifacts:

- `cohort-manifest.json`: source hash, selection, exclusions, class/generator
  counts;
- `extraction-manifest.json`: observer, exact layers, pooling, input hash, peak
  CUDA allocation;
- `ghost-embeddings.npz`: response-mean and response-last vectors;
- `evaluation/split.json`: problem-disjoint assignments and unused errors;
- `evaluation/fit-audit.json`: proof that fit/calibration positive counts are
  zero;
- `evaluation/anomaly-scores-test.csv`: frozen scores without labels;
- `evaluation/scores-test.csv`: the same traces with labels for audit;
- `evaluation/metrics.json`: fused, per-layer, final-layer,
  generator-stratified, bootstrap, and nuisance-only metrics;
- `evaluation/reference-model-*.npz`: reloadable Mahalanobis models;
- `_SUCCESS` or `_FAILED`: fail-closed terminal marker.

The primary test is:

```text
response_mean.mid_fused.auroc
response_mean.mid_fused.aupr
response_mean.mid_minus_final_bootstrap
```

This setup tests the registered estimand “fixed six-middle-layer ensemble
versus one normalized final-layer control.” Evidence favors that estimand only
if the fusion wins on the same held-out traces and its paired
problem-bootstrap interval is favorable. It does not by itself establish the
paper's broader “middle beats final” depth claim: that would additionally
require a normalization-matched control and a prespecified multiplicity rule
for the individual-layer comparisons.
Bootstrap intervals condition on the one fixed split, fitted normal geometry,
and calibration set; they only resample held-out problem groups and therefore
do not include reference-estimation or split uncertainty. A high pooled AUROC
with poor within-generator results instead suggests generator/style
confounding.
