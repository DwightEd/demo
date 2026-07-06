# Top-Venue Review and Revision Audit

Date: 2026-07-06

This document is a senior-reviewer audit of the current project state.  It is
written to answer one question:

```text
What must change for this work to plausibly reach NeurIPS/ICLR/ICML/AAAI
spotlight or oral level?
```

The current answer is blunt: the existing evidence is promising, but the paper
is not yet a top-venue paper if framed as "a hidden-state geometry metric for
reasoning error detection."  The path to a stronger paper is to turn the weak
but real geometry signal into a controlled study of reasoning telemetry:

```text
reasoning errors = failures of constraint-supported anchoring,
not simply low kappa or high hidden-state spread.
```

## Paper Map

### Current Main Claim

Incorrect reasoning steps show lower within-step token directional consensus in
middle-layer hidden states.  The signal is complementary to entropy but weak
under same-problem, length, and difficulty controls.

### Current Contribution Type

Empirical analysis plus diagnostic metric.  The draft gestures toward an online
monitor and anchoring mechanism, but these are not yet proven.

### Current Core Insight

The real signal in the step-token direction geometry subspace is mostly the
vMF-like concentration coordinate:

```text
R = ||sum_i w_i h_i / ||h_i|||| / sum_i w_i
spread = 1 - R
```

Many more complex variants collapse under strict controls:

- scalar HSMM / EM;
- path kernels;
- hidden Gram and spectral-tail features;
- static second-moment variants;
- simple premise/equation regex ledgers;
- dynamic precursor claims.

### Current Boundary

The project has a real physiological readout, not a complete verifier.  It is
good at fragmented or unstable errors, but weak for coherent-but-wrong traces
and long/difficult same-problem cases.

### Closest Prior Work Pressure

1. Geometry/manifold work says reasoning has low-dimensional structure.
2. Step-saliency/flow work says reasoning failures can be repaired by changing
   information flow.
3. Hidden-state error-awareness work says hidden signals may be diagnostic but
   not causal.
4. Process supervision and PRM work can localize errors with explicit step
   labels or generated counterfactuals.

The paper must therefore explain why it is not merely:

```text
another weak hidden-state probe with careful statistics.
```

## P0 Reviewer Risks

| Priority | Issue | Why It Can Reject The Paper | Required Fix |
| --- | --- | --- | --- |
| P0 | Main claim is still too metric-centered | Reviewers have seen many hidden-state probes and geometry metrics.  A weak AUROC signal is not enough. | Reframe around a new variable: constraint-supported anchoring. |
| P0 | No causal mechanism yet | Hidden Error Awareness-style work explicitly warns that hidden error signals can be diagnostic but not causal. | Add causal tests: patching, counterfactual siblings, and matched interventions. |
| P0 | Coherent-but-wrong remains unsolved | The hardest failures are high-confidence, internally coherent wrong traces.  Kappa cannot detect them by construction. | Add an anchoring-source channel: where the consensus points, not only whether consensus exists. |
| P0 | Massive-activation story may be false or secondary | Local notes already show `resultant_bulk` remains strong after removing massive dimensions and `massive_frac` is near random. | Do not make massive activations the central mechanism unless new evidence overturns this. |
| P0 | Step segmentation is too strong for deployment | A method that assumes pre-segmented steps is not a real-time hook. | Keep step labels only for evaluation; primary signals must be token-causal. |
| P0 | Negative results currently read as lack of progress | The paper has many retired branches but no decisive new positive mechanism. | Convert negative results into a saturation theorem/audit, then introduce a genuinely different axis. |

## The Stronger Research Thesis

The paper should pivot from:

```text
Can hidden geometry classify wrong steps?
```

to:

```text
Can internal reasoning telemetry distinguish whether a model is
(1) confidently anchored to valid constraints,
(2) unanchored and fragmented,
(3) anchored to the wrong source, or
(4) uncertain but recoverable?
```

This gives the paper a new variable:

```text
anchor support = strength of local consensus + source of that consensus
```

Kappa measures only the first term.  The missing term is the source:

```text
What is the local consensus anchored to?
```

That source can be:

- the question/givens;
- a previous valid step;
- a wrong intermediate conclusion;
- the model's recent self-generated text;
- no stable source.

This is the cleanest way to make the work nontrivial.  It explains why kappa is
real but insufficient, and why coherent-but-wrong failures exist.

## Proposed Method: Constraint-Supported Reasoning Telemetry

### State Variables

For each token window or step `t`, estimate a low-dimensional telemetry state:

```text
S_t = anchoring strength
A_t = anchoring source distribution
U_t = uncertainty / commitment
C_t = constraint consistency
```

Concrete definitions:

- `S_t`: directional concentration/resultant and spread.
- `A_t`: transport/saliency/attention attribution to prompt anchors and prior
  reasoning anchors.
- `U_t`: entropy, committal uncertainty, margin, EDIS-style burst/rebound.
- `C_t`: whether the step is supported by the prefix constraints, estimated by
  symbolic checks where possible, sibling-consensus, or counterfactual
  derivability tests.

### Four Failure States

| State | `S_t` | `A_t` | `U_t` | Meaning | Intervention |
| --- | --- | --- | --- | --- | --- |
| valid anchored | high | prompt/valid-prefix | low or resolving | healthy local computation | continue |
| fragmented error | low | diffuse | medium/high | local computation not stabilized | rollback/resample/check |
| coherent wrong | high | wrong-prefix/self-loop | low | stable but invalid basin | re-ground to prompt/constraints |
| persistent uncertainty | low/medium | mixed | high or oscillating | model has not resolved state | allocate compute/verifier |

This taxonomy is a major narrative upgrade because it admits that high
concentration is not always good.  A coherent wrong answer can be highly
concentrated.

## Why This Is Different From A Probe

A supervised hidden-state probe answers:

```text
Is this hidden state statistically associated with an error?
```

The proposed telemetry framework answers:

```text
Which constraint source supports the current reasoning state, and what
intervention is appropriate if that source is wrong, missing, or unstable?
```

The distinction must be enforced experimentally:

- no full-hidden black-box classifier as the main method;
- low-dimensional named variables;
- same-problem and counterfactual controls;
- random-anchor and shuffled-anchor kill tests;
- intervention policies matched by trigger rate and compute.

## Strongest New Experiment: Counterfactual Sibling Traces

The project needs data where difficulty, length, prompt, and prefix are not just
controlled statistically but held fixed by construction.

### Construction

For each correct trajectory:

1. choose a step `j`;
2. inject a controlled wrong transition:
   - wrong operation;
   - wrong number binding;
   - dropped premise;
   - invalid algebraic transformation;
   - premature conclusion;
3. verify that the prefix before `j` is valid and the injected transition is
   unsupported;
4. let the model continue from the corrupted prefix;
5. store the matched correct sibling and corrupted sibling hidden traces.

### Why This Is Top-Venue Grade

It directly attacks the hardest confound:

```text
wrong traces are longer and harder
```

In a sibling pair, the prompt and prefix are identical until the injected
transition.  Any signal at or after the first wrong step is much harder to
explain as problem difficulty.

### What It Tests

| Prediction | If True | If False |
| --- | --- | --- |
| kappa drops at unsupported transition | fragmented-error physiology is causal/local | kappa is mainly dataset/difficulty artifact |
| anchor source shifts to wrong prefix | coherent wrong has wrong-source anchoring | need stronger semantic/attention channel |
| entropy and kappa split failure modes | multi-channel telemetry is justified | a simpler entropy baseline may dominate |
| intervention restores correct continuation more than random trigger | monitor is actionable | signal is diagnostic only |

This connects directly to verifiable counterfactual process supervision, where
negative trajectories share a valid prefix and deviate at a controlled
unsupported step.

## Second Strong Experiment: Anchor-Source Attribution

Kappa answers whether tokens agree.  Anchor-source attribution asks what they
agree with.

### Prompt Anchors

Extract anchor spans from the problem:

- quantities;
- entities;
- units;
- constraints;
- target question;
- answer format.

For each response token window, compute:

```text
mass_to_prompt_anchors
mass_to_previous_valid_steps
mass_to_recent_generated_text
anchor_entropy
anchor_switch_rate
self_loop_mass
wrong_prefix_mass
```

Implementation can start simple:

- cosine/transport to prompt-span hidden banks;
- attention lookback if attention maps are available;
- random and shuffled span controls;
- no claim until real prompt spans beat fallback qvec partitions.

### Key Claim

Coherent-but-wrong failures are not unanchored.  They are anchored to the wrong
source.

This is the most important missing insight.

## Third Strong Experiment: Intervention Replay

Top-venue reviewers will ask:

```text
Does the monitor improve reasoning, or only classify failures after the fact?
```

Use offline replay before live activation steering.

### Matched Trigger Policies

Compare:

- random trigger at same rate;
- entropy-only trigger;
- kappa/spread-only trigger;
- anchor-source trigger;
- fused telemetry trigger;
- oracle first-error trigger.

### Matched Actions

Actions should be low-risk first:

- regenerate from last safe step;
- restate prompt constraints;
- ask a local consistency check;
- allocate more samples only for persistent uncertainty;
- stop overthinking if late-chain drift is detected.

### Metrics

Report:

- final answer accuracy;
- repair success conditional on trigger;
- false-intervention harm on originally correct traces;
- token cost;
- trigger rate;
- delay from true first-error step;
- matched compute baselines.

No activation steering claim should be made until replay succeeds.

## Role Of Massive Activations

Massive activations are useful background, but currently risky as the main
mechanism.

Local evidence already says:

- `resultant_bulk`, after dropping massive dimensions, remains strong;
- `massive_frac` is near random;
- the true signal is directional concentration, not merely energy in a few huge
  dimensions.

Therefore the paper should not claim:

```text
errors happen because massive activations weaken
```

unless a new controlled decomposition proves it.

Safer framing:

```text
Massive dimensions are one candidate anchor channel.  We test them as a
negative/positive mechanism probe, but the broader claim is constraint-supported
anchoring, not massive activation failure.
```

## Relation To Recent Work

| Work | What It Adds | How We Should Position Against It |
| --- | --- | --- |
| Reasoning Fails Where Step Flow Breaks, ACL 2026 | Step-Saliency finds Shallow Lock-in and Deep Decay; StepFlow repairs information flow. | We need comparable intervention evidence or stay diagnostic.  Use their "flow break" language only if we measure source/flow. |
| Hidden Error Awareness in CoT Reasoning, arXiv 2026 | Hidden states encode early error-predictive signals but interventions fail; signal may be diagnostic, not causal. | This is the key objection.  Our paper must separate diagnostic telemetry from causal repair and test matched interventions. |
| Verifying CoT via Computational Graph, arXiv 2025 | Argues gray-box hidden-state probes detect correlation but not underlying computation. | Our answer: do not use full probes; use named constraint-source variables and causal sibling controls. |
| Know More, Know Clearer, arXiv/ICML 2026 spotlight | Partitions model state into mastered/confused/missing regions and uses differentiated intervention/calibration. | Borrow the meta-cognitive region idea: valid/fragmented/misanchored/uncertain states need different interventions. |
| Verifiable Counterfactual Supervision for PRMs, arXiv 2026 | Builds negative trajectories with valid prefix and verifiably unsupported transition. | Borrow this for controlled sibling traces; it directly solves the length/difficulty confound. |
| Stepwise Confidence Attribution, ICML 2026 | Uses same-question trajectory consensus to assign step confidence and improve self-correction. | Our hidden telemetry should be tested against same-problem consensus, not only final labels. |
| What do Geometric Hallucination Detection Metrics Actually Measure?, 2026 | Geometry metrics respond to different hallucination properties and domain shifts; normalization helps. | We must state what kappa measures: fragmentation, not truth.  Add domain/length normalization. |
| Massive Activations in LLMs, COLM 2024 | Rare huge activations act as bias/attention sinks. | Use as a mechanistic control, not the paper's core unless evidence is strong. |
| CREST, arXiv 2025/2026 | Finds cognitive heads for verification/backtracking and steers reasoning behavior at test time. | A strong intervention baseline/parallel; our monitor must be more failure-type-specific or cheaper. |

## Claim-Experiment Matrix

| Claim | Required Experiment | Alternative Explanation | Control |
| --- | --- | --- | --- |
| Error steps lose directional consensus | ProcessBench step labels plus same-problem paired ranking | error steps are longer/harder | length buckets, same-problem pairs, prefix-matched counterfactual siblings |
| Kappa measures fragmentation, not truth | coherent-wrong subgroup where kappa is high but answer wrong | subgroup is mislabeled or rare | low-entropy/high-kappa wrong cases, human/sample inspection |
| Coherent wrong means wrong-source anchoring | anchor-source attribution on sibling traces | anchor score is prompt length/domain artifact | random anchors, shuffled anchor kinds, same prompt siblings |
| Geometry and uncertainty separate failure modes | 2D taxonomy with entropy/kappa quadrants | entropy alone explains all | fusion increment, rescue of entropy misses, matched subsets |
| Telemetry is actionable | intervention replay improves final accuracy at matched trigger rate | more compute/retry explains gain | random trigger, always retry, entropy trigger, same token budget |
| Hidden signal is not just diagnostic | causal patch/replay changes continuation quality | patching breaks distribution or copies answer | sibling patch, no-answer leakage, negative controls |

## Revised Paper Architecture

### Title Direction

Best current title:

```text
Reasoning Errors Are Failures of Constraint-Supported Anchoring
```

Safer title if intervention is weak:

```text
What Hidden Geometry Can and Cannot Tell Us About Reasoning Errors
```

### Figure 1

The page-one figure should not be a pipeline diagram.  It should show the new
variable.

Panel A: Same problem, two sibling traces with identical valid prefix, one
controlled unsupported transition.

Panel B: Kappa/spread detects fragmented errors but misses coherent wrong
cases.

Panel C: Anchor-source distribution reveals whether a high-kappa trace is
anchored to prompt constraints or to a wrong prefix.

Panel D: Failure-type-specific intervention: fragmented -> rollback/resample,
misanchored -> re-ground constraints, uncertain -> allocate verification.

### Abstract Skeleton

Sentence 1: Long reasoning fails locally, while final-answer labels and output
confidence are delayed and coarse.

Sentence 2: Existing hidden-geometry metrics are promising but confounded by
length, difficulty, and domain, and they cannot distinguish coherent wrong
reasoning from valid reasoning.

Sentence 3: We introduce constraint-supported anchoring, a telemetry view that
separates the strength of local hidden-state consensus from the source that
supports that consensus.

Sentence 4: Using step-labeled and counterfactual sibling traces, we show that
directional concentration detects fragmented failures, while anchor-source
attribution identifies coherent-but-wrong failures missed by concentration.

Sentence 5: Matched intervention replay shows when this telemetry improves
reasoning quality and when hidden signals remain diagnostic only.

### Contributions

1. A confound-controlled audit showing that step-token directional
   concentration is a real but saturated physiological signal.
2. A failure taxonomy separating fragmented, misanchored, persistent-uncertain,
   and valid anchored reasoning states.
3. A counterfactual sibling-trace evaluation that fixes prompt and valid prefix,
   directly testing first-error effects.
4. An anchor-source attribution channel that targets coherent-but-wrong cases.
5. A matched intervention replay protocol that distinguishes useful monitors
   from merely diagnostic hidden-state signals.

## Experiments To Run Next

### E1: Sibling Counterfactual Pilot

Start with GSM8K where symbolic arithmetic is easiest.

Minimum viable version:

- 100 correct traces;
- inject one wrong arithmetic operation or number binding;
- continue generation from corrupted prefix;
- store hidden traces, entropy, step ranges, final answer.

Primary table:

```text
matched correct sibling vs corrupted sibling
pre-error prefix | injected step | post-error continuation
kappa | entropy | anchor-source | final correctness
```

### E2: Coherent-Wrong Subset Audit

Define:

```text
wrong chain AND low entropy AND high kappa/spread not alarming
```

Question:

```text
Do anchor-source features rescue this subset?
```

This is the most important subgroup.  If it fails, the paper should admit that
internal geometry mainly detects fragmented errors.

### E3: Real Prompt Anchor Extraction

Do not reuse fallback qvec partitions.

Extract:

- numeric quantities;
- entities;
- units;
- target phrase;
- constraints;
- answer format.

Run:

- real anchors;
- random prompt spans;
- shuffled anchor kinds;
- whole-question qvec;
- no-anchor baseline.

### E4: Matched Intervention Replay

Use generated samples where prefixes and continuations can be replayed.

Policies:

- random trigger;
- entropy trigger;
- spread trigger;
- anchor trigger;
- fused state trigger.

Actions:

- re-ground constraints;
- retry from last safe prefix;
- local verifier prompt;
- extra sample branch.

Acceptance:

```text
fused state trigger improves final answer accuracy or repair success at the
same trigger rate and token budget.
```

### E5: Diagnostic-vs-Causal Patch Test

Because recent work warns that hidden error signals may not be causal, run a
small patching test:

- patch hidden states from correct sibling to corrupted sibling at the injected
  step;
- patch only anchor-source subspace if defined;
- patch random same-layer state as control;
- continue generation and evaluate final answer.

Interpretation:

- if patch helps, mechanism evidence is strong;
- if patch fails but replay helps, present the method as monitor-driven
  behavioral intervention, not activation repair;
- if both fail, downgrade to diagnostic paper.

## Reviewer Objections And Answers

| Objection | Likely Reviewer Wording | Needed Answer |
| --- | --- | --- |
| "This is just another hidden-state probe." | The method uses hidden states and predicts correctness; novelty unclear. | Emphasize named telemetry variables, no full-state classifier headline, and causal sibling controls. |
| "Kappa is not novel and weak." | Mean resultant length is standard and effect size is modest. | Agree.  Kappa is not the contribution; it is the saturated readout motivating anchor-source analysis. |
| "Same-problem controls kill the result." | The useful signal disappears when difficulty is controlled. | Show sibling counterfactuals and coherent-wrong subgroup; report boundaries honestly. |
| "Interventions may not work." | Hidden signals can be diagnostic but not causal. | Run matched intervention replay and patch tests; avoid activation-steering claims unless passed. |
| "Massive activations are a stretch." | Removing massive dims preserves signal. | Treat massive dims as a tested control, not core thesis. |
| "The paper overfits to math/GSM8K." | Claims about reasoning are broad. | Use ProcessBench subsets, MATH/OmniMath, and at least one non-arithmetic reasoning dataset if possible. |

## Readiness Checklist

| Category | Status | Reason |
| --- | --- | --- |
| Problem importance | Risk | Step-level reasoning monitoring is important, but the paper must connect to process supervision and online deployment. |
| Core insight | Risk | Kappa is real but insufficient; anchor-source variable is not yet implemented. |
| Novelty | Blocker | Metric novelty is weak.  Need counterfactual sibling + anchor-source taxonomy. |
| Mechanism | Blocker | No causal evidence yet.  Massive-activation story is currently risky. |
| Experiments | Blocker | Need sibling counterfactuals, coherent-wrong subgroup, anchor controls, intervention replay. |
| Statistics | Pass/Risk | Current project has strong discipline: same-problem, CI, length controls.  Keep it. |
| Figures | Risk | Need a page-one figure showing the new variable, not a pipeline. |
| Writing | Risk | Current draft is honest but still defensive.  It needs a positive mechanism spine. |
| Venue fit | Unknown | Could be top-tier if mechanism + intervention pass; otherwise a strong empirical audit/workshop paper. |

## Bottom Line

The strongest path is:

```text
Kappa is not the method.  Kappa is the physiological symptom.
The method is constraint-supported anchoring telemetry.
The dataset/evaluation innovation is counterfactual sibling traces.
The key new signal is anchor source, designed to catch coherent-but-wrong.
The decisive evidence is matched intervention replay.
```

If these pieces land, the paper can credibly claim a new evaluation and
monitoring paradigm.  If they do not, the honest paper is still useful, but it
should be framed as a rigorous negative/positive audit of hidden geometry
rather than a top-venue method paper.

## Verified Source Links

- Reasoning Fails Where Step Flow Breaks, ACL 2026:
  https://arxiv.org/abs/2604.06695
- Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is
  Diagnostic, Not Causal, arXiv 2026:
  https://arxiv.org/abs/2605.09502
- Verifying Chain-of-Thought Reasoning via Its Computational Graph, arXiv 2025:
  https://arxiv.org/abs/2510.09312
- Know More, Know Clearer: A Meta-Cognitive Framework for Knowledge
  Augmentation in Large Language Models, arXiv / ICML 2026 spotlight:
  https://arxiv.org/abs/2602.12996
- Verifiable Counterfactual Supervision for Process Reward Models, arXiv 2026:
  https://arxiv.org/abs/2605.02395
- Diagnosing Multi-step Reasoning Failures via Stepwise Confidence Attribution,
  arXiv / ICML 2026:
  https://arxiv.org/abs/2605.19228
- What do Geometric Hallucination Detection Metrics Actually Measure?, arXiv
  2026:
  https://arxiv.org/abs/2602.09158
- Massive Activations in Large Language Models, COLM 2024:
  https://arxiv.org/abs/2402.17762
- Understanding and Steering the Cognitive Behaviors of Reasoning Models at
  Test-Time, arXiv 2025/2026:
  https://arxiv.org/abs/2512.24574
