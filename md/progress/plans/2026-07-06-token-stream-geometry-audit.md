# Token-Stream Geometry Audit Plan

Date: 2026-07-06
Status update: 2026-07-07

## Result Summary

Run:

```text
token-stream geometry | gsm8k_v2_custom.npz | sv_clouds L16 | backend torch:cuda:0
```

Headline:

```text
samples 3452 | err 532 | problems 147
baseline length_entropy_static within 0.668 cross 0.808
best stream token_stream_alpha within 0.670 cross 0.805
increment +0.002 CI [-0.023, +0.027]
decision: no robust token-stream increment over length/entropy/static controls
```

Group results:

```text
token_stream_alpha     within 0.670 inc +0.002 [-0.023,+0.027]
token_stream_dynamics  within 0.647 inc -0.021 [-0.048,+0.006]
token_stream_all       within 0.633 inc -0.035 [-0.069,-0.001]
token_stream_kappa     within 0.631 inc -0.037 [-0.063,-0.010]
token_stream_spectrum  within 0.623 inc -0.045 [-0.081,-0.010]
```

Online alarm:

```text
best alarm spread_w64
FPR 0.049
recall 0.269
gold-time recall 0.000
median delay nan
endpoint fraction 0.573
```

Interpretation:

- The boundary-free stream protocol is implemented, but it does not produce a
  robust deployable detector on this data.
- Kappa/spread stream features are weaker than the static/length/entropy
  baseline.
- Alpha/effective-rank stream summaries nearly match the baseline, but their
  increment is statistically null.
- The alarm is late/end-point-like and does not localize gold first-error time.
- Effective-rank trajectories do show a broad rise-then-fall morphology, but
  this morphology is common to correct and incorrect traces and is not yet a
  correctness mechanism.

Decision:

```text
Retire token-stream kappa/alpha/effective-rank as a standalone online monitor.
Keep the profile exporter for descriptive physiology and phase annotation.
Move the main research effort to source-aware anchoring:
  prompt-span anchors, anchor-source attribution, coherent-but-wrong subsets,
  and counterfactual sibling traces.
```

## Motivation

The deployable monitor cannot assume pre-segmented reasoning steps.  Step
boundaries may be available in ProcessBench-style offline data, but a real LLM
hook receives one token at a time.  The new audit therefore computes all primary
signals from causal token windows and uses step labels only for evaluation.

## Core Hypothesis

The already validated anchor is:

```text
reasoning error <-> loss of directional consistency in middle-layer hidden states
```

The new question is stricter:

```text
Can this signal remain useful online after removing step-boundary assumptions
and controlling for length, entropy, and available static spread baselines?
```

## Signals

For each token and layer:

```text
u_t = h_t / ||h_t||
R_t(W) = ||sum_i exp(-decay * (t-i)) u_i|| / sum_i exp(-decay * (t-i))
spread_t(W) = 1 - R_t(W)
```

Windows default to `8,16,32,64`.  The script also computes sliding spectral
alpha on token windows:

```text
H_{t-W:t} -> eigen spectrum of H H^T -> log-log spectral slope alpha
```

Alpha is summarized by level, slope, amplitude, and phase-change magnitude.

The spectral branch now also records sliding effective rank:

```text
eff_rank_t(W) = exp( entropy( spectrum(H_{t-W:t} H_{t-W:t}^T) ) )
```

For each saved trajectory the audit can export:

- `resultant_w*`: direction concentration;
- `spread_w*`: direction breadth;
- `eff_rank_raw_w*`: raw hidden token-window effective rank;
- `eff_rank_unit_w*`: direction-only effective rank;
- `alpha_raw/unit_w*`: spectral slope.

The report includes a descriptive hump test for "rise then fall" shapes:

```text
hump_present = interior peak AND early->peak rise AND peak->late fall
```

This does not classify correctness by itself.  It tests whether the expected
Reasoning-Fails-Where-Step-Flow-Breaks style flow morphology actually exists in
our traces before building a heavier breakpoint model.

## Speed Notes

The first version of `--no_alpha` was still slow because it only disabled the
sliding spectrum.  The kappa/resultant trace was still computed by repeatedly
scanning the same 4096-dimensional token stream once per window on CPU.

Current optimization:

- multi-window resultant is computed in one token pass on CPU;
- hidden rows are processed as `float32`, matching the stored fp16/fp32
  precision better than unnecessary `float64`;
- `--stream_backend auto|cuda|torch` can compute the multi-window resultant
  with grouped `conv1d` on GPU;
- alpha remains the expensive branch and should be enabled only after the fast
  kappa audit is interpreted.

## Anti-Confound Gates

The script reports three baseline families:

- `length`: response token count and available step-count controls.
- `length_entropy`: length plus entropy/committal traces when present.
- `length_entropy_static`: length, entropy, and static step-spread summaries
  when step boundaries are available.

The headline increment is the out-of-fold score of:

```text
baseline + token_stream_group
```

over the strongest available baseline.  Same-problem paired AUROC is the primary
metric when contrastive problems are present.

## Online Alarm Metrics

The script also reports fixed-FPR online alarms:

- observed correct-chain false positive rate;
- error-chain recall;
- first-error gold-time recall when `gold_error_step` and token ranges exist;
- pre-error/onset recall;
- median token delay;
- median endpoint fraction.

This is where the method becomes deployment-facing rather than another static
post-hoc scalar.

## Novelty Boundary

`R_t` is not claimed as a new mathematical object.  The research contribution,
if the audit succeeds, is the protocol:

```text
causal token-stream physiology + strict length/difficulty controls +
same-problem ranking + fixed-FPR alarm delay
```

If this fails, the branch should be retired as a weak physiological marker, not
renamed as a new geometry metric.
