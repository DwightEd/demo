# Token-Stream Geometry Audit Plan

Date: 2026-07-06

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

