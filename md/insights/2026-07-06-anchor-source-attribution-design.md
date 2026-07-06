# Anchor-Source Attribution Design

Date: 2026-07-06

This note defines the proposed anchor-source branch.  The key distinction is:

```text
kappa asks: do local response-token hidden states agree?
anchor-source asks: what are they agreeing with?
```

This is meant to target coherent-but-wrong reasoning, where the local hidden
states may remain concentrated but become anchored to an invalid source.

## Core Hypothesis

Reasoning failures are not all direction-dispersion failures.

There are at least two geometrically different failure modes:

1. **Fragmented failure**: token directions lose local consensus.
   - low resultant / high spread;
   - high anchor entropy;
   - often visible to current kappa-like metrics.
2. **Misanchored coherent failure**: token directions remain locally coherent,
   but their consensus is supported by the wrong source.
   - high resultant;
   - low entropy/confident text;
   - low prompt/constraint mass;
   - high mass on recent self-generated text or a wrong prior step.

Therefore, the missing variable is:

```text
source of consensus
```

not another geometry scalar over the same step-token cloud.

## Residual Stream Or Attention?

### Short Answer

Yes, anchor-source can be estimated from the residual stream.  But it should
not be called "attention".  It is an **effective source-affinity / transport
readout** from hidden states.

Use three evidence levels:

| Level | Signal | What It Means | Claim Strength |
| --- | --- | --- | --- |
| L1 | residual-source affinity | current state resembles / aligns with an anchor source | diagnostic readout |
| L2 | attention or value-flow to anchors | model routed information from anchor tokens through attention | mechanistic routing evidence |
| L3 | attention-gradient / attribution graph / patching | source pathway has causal influence on output | causal mechanism evidence |

The paper should not overclaim L1 as causal.  L1 is valuable because it is
cheap, works with saved hidden states, integrates attention+MLP+residual
effects, and aligns with the existing kappa pipeline.  L2/L3 should be used as
validation or intervention evidence.

### Why Not Directly Use Attention First?

Direct attention has real advantages:

- it is an actual model routing object;
- it gives token-to-token edges;
- it can be intervened on;
- it connects naturally to Step-Saliency / StepFlow-style work.

But raw attention is a risky first-line source signal:

- raw attention weights are not always faithful explanations;
- heads/layers disagree and are high-dimensional;
- sink tokens and formatting tokens can dominate;
- attention ignores MLP contributions and later residual accumulation;
- storing all attention maps is expensive;
- a single layer's attention is not equal to the final reasoning state.

So the recommended design is:

```text
residual anchor transport = cheap diagnostic branch
attention/value-flow       = mechanistic validation branch
attention-gradient/patching = causal branch
```

If residual transport fails under real prompt anchors, attention should still be
tested as a separate branch.  But if residual transport already works, attention
becomes a mechanism check rather than the first detector.

## Anchor Types

The anchor bank must be semantic and span-based, not a qvec fallback.

### Prompt Anchors

Extract prompt spans:

- `quantity`: numbers, variables, units, rates;
- `entity`: people, objects, named variables;
- `constraint`: conditions, comparisons, equations, inequalities;
- `goal`: final target quantity / question;
- `format`: answer format requirements.

For each anchor span `k` at layer `l`:

```text
a_k^l = normalize(pool_{tokens in span k}(h_i^l))
```

Pooling can be mean, exponential late pooling, or last-token pooling.  Mean is
the safest first version.

### Reasoning Anchors

For response-side anchors, use causal sources only:

- previous step summary vector;
- previous valid prefix vector when gold labels exist offline;
- recent window vector;
- first wrong step vector offline for analysis only;
- generated intermediate claim anchors from parsed equations or noun phrases.

Offline source groups:

```text
prompt_goal
prompt_quantity
prompt_constraint
previous_correct_step
first_wrong_step
post_wrong_step
recent_self
```

Online source groups:

```text
prompt_goal
prompt_quantity
prompt_constraint
previous_steps
last_safe_step_estimate
recent_self
```

## Residual Anchor Transport

For a response token window `W_t` and layer `l`:

```text
u_i^l = normalize(center(h_i^l))
mu_t^l = normalize(sum_{i in W_t} w_i u_i^l)
R_t^l = ||sum_{i in W_t} w_i u_i^l|| / sum_i w_i
```

Let anchor vectors be:

```text
A^l = {a_1^l, ..., a_K^l}
```

Define anchor affinity:

```text
s_{t,k}^l = cos(mu_t^l, a_k^l)
```

Prefer a null-adjusted score:

```text
z_{t,k}^l = (s_{t,k}^l - median_null(s_{t,*}^l)) / MAD_null
```

where the null anchors are random prompt spans, shuffled anchors, or
problem-mismatched anchors.

Then define source distribution:

```text
pi_{t,k}^l = softmax(z_{t,k}^l / tau)
```

Aggregate by anchor kind:

```text
M_t(kind) = sum_{k: kind(k)=kind} pi_{t,k}
```

Important summary features:

```text
prompt_mass_t      = M_t(goal) + M_t(quantity) + M_t(constraint)
goal_mass_t        = M_t(goal)
constraint_mass_t  = M_t(constraint)
quantity_mass_t    = M_t(quantity)
previous_mass_t    = M_t(previous_correct_step or previous_steps)
wrong_mass_t       = M_t(first_wrong_step)       # offline only
recent_self_mass_t = M_t(recent_self)
anchor_entropy_t   = -sum_k pi_{t,k} log pi_{t,k}
anchor_margin_t    = top1(pi_t) - top2(pi_t)
anchor_jump_t      = TV(pi_t, pi_{t-1})
```

This is a residual-derived "where does the current local state point?"
measurement.

## Attention Baselines

If attention maps are available, compare residual transport to:

### Raw Attention Mass

For each layer/head:

```text
AttnMass_{t -> kind}^{l,h}
  = sum_{j in anchors of kind} A_{t,j}^{l,h}
```

Aggregate across response tokens in a window and across heads/layers.

### Value-Flow Mass

Raw attention ignores the value vector magnitude and output projection.
Approximate contribution:

```text
VFlow_{t -> kind}^{l,h}
  = || sum_{j in kind} A_{t,j}^{l,h} W_O^{l,h} W_V^{l,h} h_j^l ||
    / || sum_j A_{t,j}^{l,h} W_O^{l,h} W_V^{l,h} h_j^l ||
```

This is closer to actual information contribution than raw attention mass.

### Attention-Gradient / Step-Saliency

For a smaller subset, use attention-gradient or attribution-graph methods as a
causal validation layer.  This is expensive but makes stronger mechanism
claims.

## Defining Detachment

A response state is detached when local consensus is no longer supported by
required sources.

### Source Detachment

Prompt/constraint source drops:

```text
D_prompt(t) = 1 - prompt_mass_t
D_goal(t) = 1 - goal_mass_t
D_constraint(t) = 1 - constraint_mass_t
```

Use causal z-scores rather than raw thresholds:

```text
Z_D(t) = (D_prompt(t) - median_correct_prefix(t_phase)) / MAD_correct_prefix
```

### Diffuse Detachment

The model is not anchored to any clear source:

```text
D_diffuse(t) = anchor_entropy_t / log(K)
```

This should correlate with fragmented failures.

### Misanchoring

The model is strongly anchored, but to recent self-generated or wrong-prefix
content instead of prompt/validated constraints:

```text
D_misanchor(t) =
  recent_self_mass_t + wrong_mass_t - prompt_mass_t
```

Online version:

```text
D_misanchor_online(t) =
  recent_self_mass_t - prompt_mass_t
```

plus a last-safe-step estimate.

### Anchor Rupture

The source distribution changes abruptly:

```text
D_jump(t) = TV(pi_t, pi_{t-1})
```

or a multi-scale CUSUM over `pi_t`.

### Combined Anchor-Support Index

Do not use this as the only score, but it is useful as a dashboard:

```text
ASI_t = R_t * prompt_mass_t * (1 - normalized_anchor_entropy_t)
```

Failure modes:

```text
fragmented:     low R_t, high entropy, low max anchor
misanchored:    high R_t, low uncertainty, low prompt_mass, high recent/wrong mass
uncertain:      medium/low R_t, high uncertainty, mixed anchors
valid anchored: high R_t, high prompt/previous-valid mass, low/mid entropy
```

## Tracking Model

Start with a descriptive state tracker before fitting another HMM.

Per token/window:

```text
x_t = [
  R_t,
  spread_t,
  prompt_mass_t,
  goal_mass_t,
  constraint_mass_t,
  previous_mass_t,
  recent_self_mass_t,
  anchor_entropy_t,
  anchor_jump_t,
  token_entropy_t,
  committal_t
]
```

Then define states with interpretable rules:

```text
valid_anchored:
  R high, prompt/previous mass high, uncertainty resolving

fragmented:
  R low or spread high, anchor entropy high

misanchored:
  R high, uncertainty low, prompt mass low, recent/wrong mass high

persistent_uncertain:
  uncertainty high or oscillating, anchor mass mixed, no stable source
```

After rule-based validation, a semi-Markov/state-space model can be fitted over
`x_t`, but only if it beats the rules under same-problem controls.  The earlier
scalar HSMM failure is a warning: state modeling is only meaningful if the
emission variables represent genuinely new information.

## Evaluation Gates

The branch should be killed unless it passes hard controls.

### Primary Subset

Coherent-but-wrong:

```text
y_err = 1
R_t high or chain mean kappa high
entropy low
baseline spread/entropy does not alarm
```

Pass condition:

```text
anchor-source features rescue baseline misses in coherent-but-wrong pairs
```

### Controls

Required controls:

- single qvec anchor;
- random prompt spans;
- shuffled anchor kinds;
- problem-mismatched anchors;
- text-only anchor count features;
- raw attention mass if available;
- value-flow mass if available;
- length and position residualization;
- same-problem paired AUROC;
- endpoint fraction and delay for online alarms.

### Key Decision Table

| Outcome | Interpretation | Next Action |
| --- | --- | --- |
| residual transport works, attention also works | strong evidence for source-flow story | test intervention |
| residual works, attention fails | source similarity is diagnostic but not routing | avoid causal claims |
| attention works, residual fails | actual routing matters more than final state | build attention branch |
| both fail | coherent-wrong needs semantic/symbolic verifier | retire anchor-source geometry |

## Why This Is Necessary

The project already knows that:

- step-token directional concentration is real;
- dynamic scalar shape and spectral variants mostly fail;
- qvec fallback AnchorFlow does not prove semantic anchoring;
- coherent-but-wrong is the likely blind spot.

Anchor-source attribution introduces a different axis:

```text
source identity
```

not a more complex statistic over the same source-free cloud.

This is why it is worth trying.

## Source Notes

- `Reasoning Fails Where Step Flow Breaks` motivates measuring reasoning
  information flow and shows that attention-gradient saliency can support
  interventions.
- `Attention is not Explanation` is a warning against treating raw attention
  weights as faithful causal explanations.
- `Verifying Chain-of-Thought Reasoning via Its Computational Graph` motivates
  moving beyond gray-box activation probes toward execution-trace and
  attribution-graph evidence.
- `When Chain-of-Thought Fails, the Solution Hides in the Hidden States`
  motivates token-level hidden-state causal patching as a validation test.
