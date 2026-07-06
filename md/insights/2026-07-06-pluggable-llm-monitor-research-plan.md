# Pluggable LLM Reasoning Monitor: Literature Notes and Research Plan

Date: 2026-07-06

Local paper folder reviewed: `C:\Users\613\Desktop\papers\推理`

Project status: repo has been pulled to `origin/main`; current local tracked change outside this note is the pre-existing edit in `md/insights/TRAJECTORY_README.md`.

## 0. Executive Decision

The next research line should not be "add another static geometry score".  The current project already shows that the strongest reliable geometry signal is a mostly one-dimensional concentration axis:

```text
spread = 1 - resultant
resultant = || mean_i normalize(h_{step, token_i}) ||
```

This signal is real, but it is not mature enough as a standalone online reasoning monitor.  It must be upgraded into a **stateful, pluggable monitor-intervention module** that observes how uncertainty, hidden-state geometry, and prompt-constraint anchoring evolve during generation.

The central research hypothesis should be:

> Correct reasoning is a constrained flow: uncertainty tends to resolve, hidden token clouds stay coherently anchored to task constraints, and transitions remain inside a calibrated healthy tube.  Reasoning errors occur when this flow loses constraint: either the model commits early to a wrong basin, or uncertainty persists/oscillates until the reasoning state detaches from the healthy manifold.  Online detection should therefore use dynamic shape and constraint detachment, not just final confidence or raw geometric level.

This hypothesis connects the existing project evidence with the most useful literature themes:

- entropy dynamics: EDIS, EDRM, entropy monotonicity, uncertainty trace profiles;
- hidden trajectory geometry: transport, layer-wise displacement, basins, Lyapunov-style stability;
- intervention: strategy routing, retry/rollback, particle filtering, prompt repair, and only later attention/activation steering.

## 1. What The Current Project Already Establishes

### 1.1 Valid signals

- `spread/resultant` is the cleanest current geometry readout.
  - ProcessBench GSM8K first-error step: `resultant` AUROC around 0.772.
  - Same-problem `answer_format_ok` setting is weaker but credible: `cloud_spread` paired AUROC around 0.634 on 5-shot and 0.659 on custom.
- Geometry and uncertainty are complementary.
  - `resultant` / concentration is close to orthogonal to `U_D`.
  - Fusion reaches stronger OOF performance than either alone.
- Chain-localization exists but is modest.
  - First-error step is more diffuse than matched non-error steps after length/position controls, but the effect is not a strong standalone localizer.
- Hypergraph/token relation work is informative but heavy.
  - HGN step AUROC around 0.760 and high first-error top1, but this is not the lightweight hookable path.

### 1.2 Retired or dangerous interpretations

- Static trajectory H1-H3 and simple phase-instability story are not supported.
- `cloud_D`, effective rank, volume-like metrics often read length/difficulty.
- Cross-layer coordination/spectral axes mostly failed as independent mechanisms.
- Same-problem abrupt rupture is not robust with current saved signals.
- Scalar HMM/latent EM did not beat `cloud_spread`.
- Fallback AnchorFlow based on `qvec` partitions has no semantic increment.  It is a scaffold, not evidence for semantic anchor transport.
- Raw cross-problem AUROC is not enough.  Same-problem paired AUROC, length residualization, GroupKFold, FPR/recall/delay, and intervention gain must be primary gates.

### 1.3 Consequence

The research should move from:

```text
Does one geometry scalar classify errors?
```

to:

```text
Can a streaming monitor infer the reasoning state, identify failure mode, and trigger a useful low-risk intervention before the answer is fixed?
```

## 2. Literature Notes From Local Papers

I treated duplicate files as duplicate evidence:

- `WhereDoesReasoningBreak.pdf` and `Alvarez和Baheri - 2026 - Where Does Reasoning Break ...pdf` appear to be the same paper.
- `Entropy Phase Transitions.pdf` and `ADynamicalSystemsViewvia.pdf` appear to be the same EDRM paper.

### 2.1 Highest-priority papers for this project

| Paper | Method / technique | Borrow for our project | Caution |
| --- | --- | --- | --- |
| `EDIS-Diagnosing LLM Reasoning via Entropy Dynamics.pdf` | Defines token entropy trajectory `H_t`; detects burst spikes `H_{t+w}-H_t > tau_b` and rebound spikes `H_t - min_{s<t} H_s > tau_r`; combines spike count with variance as EDIS. | Implement online entropy-instability events alongside `spread/resultant`.  Use burst/rebound as event features, not just mean entropy. | Paper mainly uses selection/curation; our online intervention still needs FPR/delay validation. |
| `Entropy Phase Transitions.pdf` / `ADynamicalSystemsViewvia.pdf` | EDRM router computes early `SH=sum H_i`, `Vsp=Spearman(time,H)`, and `avnr=MSD/(Var+eps)`; routes among Direct, Standard, CoT. | Use an early probing window to choose strategy and budget.  This is a natural low-cost hook before expensive hidden-state monitoring. | This routes strategy, not error correction.  Needs task/domain calibration. |
| `ENTROPY TRAJECTORY SHAPE PREDICTS LLM REASONING.pdf` | Per-step answer-distribution entropy; monotone decrease predicts correctness; violation count is graded signal; total entropy reduction is less predictive. | Test "shape over magnitude" in our same-problem samples.  Add violation count and monotonicity features. | It samples answer completions at steps, so cost is higher than pure logits.  Strongest for discrete-answer tasks. |
| `Tracing Uncertainty in Language Model Reasoning.pdf` | Builds uncertainty trace profile: early/mid/late means, slope, `r2` over uncertainty channels; reports early correctness prediction. | Add compact temporal summaries for entropy, committal uncertainty, and spread.  Use interpretable logistic/GBDT baselines. | Some features require backward pass for epistemic uncertainty; start with forward-only versions. |
| `15744_TokUR_Token_Level_Uncert.pdf` | Low-rank random weight perturbation approximates posterior; token-level total/aleatoric/epistemic uncertainty; response uncertainty is sum over tokens; online particle filtering uses uncertainty as intrinsic reward. | Optional high-cost uncertainty plugin: separate epistemic vs aleatoric, and test uncertainty-guided particle filtering/retry. | More expensive than logits.  Online gains reported as modest. |
| `WhereDoesReasoningBreak.pdf` | Hidden-state transport geometry.  Teacher uses label-conditioned trace normalization, contrastive PCA, position/velocity/acceleration features, and transport cost to correct-state cloud; student is BiLSTM distilled from teacher. | The best conceptual match: errors are localized excursions from a healthy transition manifold.  Borrow transition vector `[z_t, delta z_t, delta2 z_t]` and transport-margin framing. | Their deployable student is post-hoc BiLSTM over full trace, not truly online.  Label-conditioned cPCA cannot be used at inference. |
| `Reasoning Fails Where Step Flow Breaks.pdf` | Step-Saliency pools attention-gradient into step-to-step maps; identifies Shallow Lock-in and Deep Decay; StepFlow uses Odds-Equal Bridge on shallow attention logits and Step Momentum Injection in deep residual stream. | Use as the "high-permission intervention" reference.  Also borrow the failure language: constraint/information flow can break. | Attention-gradient is heavy; activation editing must pass causal tests because hidden error signals can be diagnostic but non-causal. |
| `How Language Models Fail-Token-Level Signatures of Committed and Persistent Reasoning Failures.pdf` | Uses token-level uncertainty over cumulative prefix windows; distinguishes committed failure vs persistent uncertainty; defines commitment point where prefix signal is maximally predictive. | Add failure-mode classifier.  Committed failures should trigger early rollback/retry; persistent uncertainty should trigger extra compute or verification. | Requires enough failure rate and logprobs.  Closed APIs may limit top-k/logprob access. |
| `Hidden Error Awareness in Chain-of-Thought Reasoning-The Signal Is Diagnostic, Not Causal.pdf` | Linear hidden probes detect wrong CoT early; but activation steering, best-of-N probe selection, retry, and patching mostly fail. | Critical warning: detection score is not automatically a causal lever.  Use interventions that alter decoding strategy first. | Do not claim hidden-state error direction can be edited unless separately proven. |
| `HowLLMsDetectandCorrect Their Own Errors-The Role of Internal Confidence Signals.pdf` | Post-answer newline (PANL) residual stream encodes second-order confidence; predicts verification and correction better than logprobs. | For finished answers, add post-answer evaluator state as a separate monitor phase.  Useful for final abstain/retry. | PANL is post-hoc; it does not solve mid-reasoning online detection by itself. |

### 2.2 Geometry, trajectory, and stability papers

| Paper | Method / technique | Borrow | Caution |
| --- | --- | --- | --- |
| `Truth as a Trajectory.pdf` | Uses layer-wise displacement rather than static activations to reduce lexical confounds. | Represent hidden states as deltas across layers/steps; use displacement features where static probes overfit. | Still a detector/probe style; must test same-problem and length controls. |
| `TheGeometryofTruth.pdf` | Layer-wise Semantic Dynamics: align hidden states to factual encoder embeddings with contrastive loss; uses semantic velocity, acceleration, convergence. | Good template for "alignment trajectory" metrics if we build real prompt semantic anchors. | Needs ground-truth/factual encoder; less suitable for math reasoning without semantic anchor design. |
| `LLMReasoning as Trajectories.pdf` | Step-specific representation subspaces; late divergence predicts correctness; trajectory steering and length control from ideal trajectories. | Use explicit step-boundary states and ideal-trajectory residuals. | Late-stage divergence may be too late for intervention; verify with delay metrics. |
| `Reasoning Models Don’t Just Think Longer,.pdf` | Shows raw trajectory geometry is strongly shaped by generation length; residualizes trajectory statistics on length before comparing difficulty. | Make length correction mandatory for every trajectory/geometry claim. | Reinforces current project concerns; raw path statistics are dangerous. |
| `Hallucination Basins.pdf` | Hallucinations as task-dependent basins/attractors; radial distance and contraction toward reference regions; adaptive geometry-aware steering. | Use "basin / committed failure" framing.  Detect when trajectory enters a wrong attractor or loses input sensitivity. | Broad hallucination setting; task-dependent separability means no universal threshold. |
| `Lyapunov Probes for Hallucination Detection in Large Foundation Models.pdf` | Stability view: stable known, stable unknown, unstable boundary; probe takes multi-layer hidden + perturbation strength and enforces monotonic confidence decay under perturbation. | Add perturbation-stability audit: correct reasoning should be locally stable; boundary cases show instability. | Probe training and perturbation passes are not cheap; start offline. |
| `Reasoning emerges from constrained inference.pdf` | Reasoning self-organizes into low-dimensional manifolds, but reliable reasoning also preserves non-degenerate information volume. | Strong theoretical support for "constrained flow" hypothesis. | Need avoid repeating failed raw volume metrics; must length/control-anchor condition volume. |
| `Limited Reasoning Space.pdf` | Over-planning crosses reasoning boundary; entropy-driven dual controller / MPC uses Measure-then-Plan, inverse constraints, semantic compression. | Useful for intervention policy: stop extending reasoning when entropy/spread indicates boundary crossing; compress/recap constraints. | Work-in-progress style; use as inspiration, not proof. |
| `The Shape of Reasoning - Topological Analysis of.pdf` | TDA / persistent homology features assess reasoning trace quality. | Potential later diagnostic for trace shape; maybe useful for offline analysis. | Too heavy for first online hook. |
| `Effective Reasoning Chains Reduce Intrinsic Dimensionality.pdf` | Reasoning strategies that generalize better reduce intrinsic dimensionality. | Relates to constrained-manifold frame. | It is training/generalization oriented, not online error intervention. |
| `The Origins of Representation Manifolds in Large.pdf` | Feature manifolds, cosine paths, intrinsic geometry of features. | Supports real semantic anchor design: features may be manifolds, not single vectors. | Conceptual; not a detector. |
| `Atheory of multineuronal dimensionality, dynamics and measurement.pdf` | Neural dimensionality/trajectory recovery theory. | Useful for rigor around sampling hidden dimensions and low-dimensional portraits. | Neuroscience theory, not LLM-specific. |

### 2.3 Attention, graph, and heavy white-box methods

| Paper | Method / technique | Borrow | Caution |
| --- | --- | --- | --- |
| `Geometry of Reason Spectral Signatures of Valid Mathematical Reasoning.pdf` | Treats attention as token graph; computes Laplacian diagnostics: Dirichlet energy, HFER, spectral entropy, Fiedler value, smoothness; uses HFER reranking. | Add attention-graph spectral features as optional heavy observer. | Our project already found many spectral/length confounds.  Needs strong controls and architecture-specific interpretation. |
| `The Spectral Geometry of Thought ...pdf` | Activation SVD power-law alpha; prompt-response spectral delta; token-level spectral cascade; claims correctness prediction. | Use only as hypothesis source for spectral compression/cascade. | Treat strong claims cautiously; our local spectral/trajectory audits have falsified several simple versions. |
| `VERIFYING CHAIN-OF-THOUGHT REASONING VIA ITS.pdf` | Circuit-based Reasoning Verification: attribution graphs as execution traces; structural graph features classify step correctness; targeted transcoder-feature interventions. | Long-term mechanistic direction: graph fingerprints for failure diagnosis and causal repair. | Too computationally intensive for a drop-in module. |
| `TowardsLong-HorizonInterpretability.pdf` | FlashTrace: efficient multi-token attribution with recursive tracing through reasoning chains. | If we later need attribution channels, use span-wise aggregation and recursive propagation. | Attribution is mostly explanation/diagnosis, not immediate intervention. |
| `Large Vision-Language Models Get Lost in Attention.pdf` | Attention failure modes in VLMs. | Optional if extending to multimodal reasoning; attention drift can be another constraint-loss channel. | Not central to current text/math setting. |

### 2.4 Uncertainty, RL, and diffusion-side references

| Paper | Method / technique | Borrow | Caution |
| --- | --- | --- | --- |
| `An Isotropic Approach to Efficient Uncertainty.pdf` | Single forward-backward uncertainty: epistemic as squared gradient norm, aleatoric as Bernoulli variance; benchmark-dependent utility. | Offline compare gradient-norm epistemic vs entropy/spread. | Backward pass is expensive; may fail on factual recall where uncertainty means something different. |
| `REVOLUTIONIZING REINFORCEMENT LEARNING.pdf` | TraceRL for diffusion language models: trajectory-aware RL and value model using inference trajectory. | Training-time extension: use monitor signals as process rewards later. | Diffusion LM setting; not first-line for autoregressive hook. |
| `Muon Dynamics as a Spectral Wasserstein Flow.pdf` | Spectral/Wasserstein flow view of optimizer dynamics. | Conceptual support for transport-flow language. | Mostly optimizer/RL dynamics; not directly an online reasoning detector. |
| `Latent Spatio-Temporal Chain-of-Thought for Robotic.pdf` | Latent spatio-temporal CoT for robotics. | Future multimodal/agent extension. | Not directly relevant to current math hidden-state project. |
| `DYNAMICS WITHIN LATENT CHAIN-OF-THOUGHT.pdf` | Latent CoT as structural causal model; step-wise do-interventions; mode-conditional and stability-aware analysis. | Use causal-intervention mindset and mode-conditional metrics. | Latent CoT setting differs from explicit CoT. |

## 3. Proposed Pluggable Module

### 3.1 Hook points

The module should be model-adapter based.  It should work at several access levels:

1. **Black/gray-box logits mode**
   - Inputs: token, top-k logprobs, entropy, optional final answer parser.
   - Works for APIs or local models with logprobs.
2. **Hidden-state mode**
   - Inputs: selected layer hidden states, token/window/step boundaries.
   - Enables `spread/resultant`, transport residuals, anchor detachment.
3. **Attention/activation intervention mode**
   - Inputs: attention logits or attention probs, value states, residual stream.
   - Enables StepFlow-like OEB/SMI or activation steering.
   - This is experimental and should not be the default path.

### 3.2 Streaming state

Maintain an online state object:

```text
ReasoningState
  token_index
  step_index
  phase: prefill | early_probe | reasoning | answer | post_answer_eval
  entropy_trace
  uncertainty_trace
  spread_trace
  anchor_transport_trace
  alarm_trace
  failure_mode: healthy | cautious | committed | persistent | detached | unknown
  last_safe_step
```

### 3.3 Signal bank

Start with features that have the best cost/evidence tradeoff:

**Level 0: forward logits only**

- token entropy `H_t`;
- mean / early-middle-late means;
- slope and `r2`;
- EDIS burst count;
- EDIS rebound count;
- monotonicity violation count;
- cumulative entropy `SH`;
- Spearman trend `Vsp`;
- volatility ratio `avnr`;
- top-token margin / near-tie / NLL if top-k logprobs are available.

**Level 1: hidden-state geometry**

- `resultant`, `spread = 1 - resultant`;
- step/window changes: `delta spread`, `delta resultant`;
- length/position residualized versions;
- transition vector `[z_t, delta z_t, delta2 z_t]`;
- healthy-tube residual / transport cost;
- CUSUM/conformal p-values over causal residuals.

**Level 2: real AnchorFlow**

- prompt semantic anchors from prompt spans, not `qvec` fallback;
- token/window transport to anchors;
- target mass, constraint detachment, transport entropy;
- anchor coverage;
- length-residualized anchored volume/effective-rank;
- random-anchor and shuffled-kind controls.

**Level 3: heavy white-box**

- attention-graph diagnostics: Fiedler, HFER, spectral entropy, smoothness;
- step saliency / lookback;
- gradient or perturbation stability probes;
- attribution graph fingerprints.

### 3.4 Detector

Use a staged detector rather than one threshold:

1. **Early strategy router**
   - EDRM-style `SH`, `Vsp`, `avnr`.
   - Chooses Direct / Standard / CoT / larger budget.
2. **Online health monitor**
   - Tracks EDIS events, spread residuals, anchor detachment.
   - Produces calibrated alarm with FPR control.
3. **Failure-mode classifier**
   - Committed failure: early prefix window maximally predictive; spread/entropy alarm appears and then stabilizes into wrong path.
   - Persistent uncertainty: uncertainty/spread keeps accumulating; full trace remains informative.
   - Detached constraint: anchor transport mass shifts away from critical prompt constraints.
4. **Post-answer evaluator**
   - Uses final hidden/PANL-like evaluation state when available.
   - Decides answer acceptance, abstention, or retry.

### 3.5 Intervention ladder

Interventions should be ordered from low-risk to high-risk:

1. **Route / budget control**
   - Direct vs Standard vs CoT.
   - Stop overthinking if entropy/spread indicates boundary crossing.
2. **Prompt-level local repair**
   - Ask model to re-check the last safe step.
   - Restate constraints from anchors.
   - Compress current reasoning and regenerate from last safe prefix.
3. **Selective resampling**
   - Best-of-N only when failure mode predicts benefit.
   - Particle-filter style branching with intrinsic reward from lower EDIS / lower detachment / healthier spread.
4. **Verifier/self-correction phase**
   - Use post-answer evaluator and answer-format policy.
   - Trigger abstain or retry when confidence and hidden state disagree.
5. **Activation/attention repair**
   - StepFlow-like OEB/SMI.
   - Only after offline causal validation shows intervention gain over random/oracle baselines.

## 4. Concrete Experiments

### Experiment A: Dynamic entropy on existing data

Goal: establish whether entropy shape adds over current uncertainty and spread.

Signals:

- EDIS burst/rebound;
- entropy slope, `r2`, early/mid/late means;
- monotonicity violations;
- `SH`, `Vsp`, `avnr`;
- existing `U_D`, `resultant`, `cloud_spread`.

Gates:

- ProcessBench step/gold-error AUROC;
- same-problem paired AUROC under `answer_format_ok`;
- GroupKFold by problem;
- length/position residualization;
- bootstrap increment over `cloud_spread + U_D`.

Pass condition:

```text
dynamic entropy adds stable OOF increment over static entropy/spread,
or improves online FPR/recall/delay at fixed FPR.
```

### Experiment B: Failure-mode taxonomy

Goal: test committed vs persistent failure modes in our saved samples.

Method:

- Compute prefix-window AUROC/PR-AUC as a function of prefix position.
- Commitment point = earliest or best prefix position where predictive power peaks.
- Persistent mode = performance improves toward full trace.
- Compare with self-consistency/retry benefit where multi-sample data exists.

Pass condition:

```text
failure-mode label predicts which intervention helps:
committed -> rollback/retry/strategy switch;
persistent -> extra compute/verification.
```

### Experiment C: Real AnchorFlow

Goal: replace `qvec` fallback with prompt semantic anchors.

Needed extraction:

- prompt spans: question, givens, constraints, target, units/options;
- anchor hidden states from actual prompt span tokens;
- response token/window hidden states.

Features:

- transport mass to each anchor;
- constraint detachment;
- transport entropy;
- anchor coverage;
- phase break in transport simplex;
- length-residualized anchored logdet/effective rank.

Controls:

- random anchors;
- shuffled anchor kinds;
- single qvec;
- no-transport independent anchor cosines.

Pass condition:

```text
real AnchorFlow > anchor_uncertainty baseline,
real anchors > random/shuffled anchors,
and gains survive same-problem/length controls.
```

### Experiment D: Online alarm

Goal: turn signals into a calibrated alarm.

Method:

- causal z-scoring from past prefix only;
- conformal threshold from correct-chain max alarm;
- report FPR, recall, delay, early-warning fraction.

Pass condition:

```text
at FPR 5% and 20%, alarm recall/delay improves over existing conformal CUSUM
without becoming a late endpoint detector.
```

### Experiment E: Intervention replay

Goal: prove that detection improves reasoning quality, not only AUROC.

Offline replay policies:

- random trigger;
- entropy-only trigger;
- spread-only trigger;
- fusion trigger;
- failure-mode trigger;
- oracle trigger.

Actions:

- local retry from last safe step;
- constraint restatement;
- shorter/longer budget;
- best-of-N only on triggered cases;
- uncertainty-guided particle selection if multiple continuations are available.

Metrics:

- final answer accuracy;
- token cost;
- trigger rate;
- repair success conditional on trigger;
- false intervention harm on correct traces;
- delay from true first-error step.

Pass condition:

```text
triggered intervention beats random trigger at matched trigger rate,
and beats always-retry / always-CoT / majority-vote where applicable.
```

## 5. Data Requirements

### 5.1 Minimal new extraction

For each generated token:

- token id/text;
- top-k logprobs or at least selected-token logprob;
- vocabulary entropy if available;
- top-1/top-2 margin;
- selected layer hidden states for response tokens;
- step boundary metadata;
- answer marker / final answer span.

For each problem:

- strict correctness;
- `answer_format_ok`;
- final parsed answer;
- prompt text and prompt span annotations.

### 5.2 Nice-to-have extraction

- attention maps for selected layers/heads;
- value states for StepFlow-style residual experiment;
- gradient-based uncertainty or saliency on a small audit subset;
- multiple continuations from alarm prefixes.

## 6. Evaluation Rules

Do not accept a result unless it reports:

- same-problem paired AUROC where possible;
- GroupKFold / cluster bootstrap by problem;
- length and position baselines;
- format-controlled policy (`answer_format_ok`);
- high-spread subset and confident-wrong subset;
- online FPR, recall, delay;
- intervention gain at matched trigger rate;
- token-cost tradeoff.

Do not promote:

- raw cross-problem AUROC alone;
- endpoint-only alarms;
- oracle same-problem tubes as deployable detectors;
- hidden-state steering without causal validation;
- fallback AnchorFlow as semantic AnchorFlow.

## 7. Suggested Code Structure

Do this as a separate pluggable package rather than mixing more scripts into the root:

```text
monitor/
  __init__.py
  hooks.py              # model adapter hooks: logits, hidden, attention
  state.py              # ReasoningState dataclass
  signals_entropy.py    # EDIS, EDRM, monotonicity, trace profile
  signals_geometry.py   # resultant/spread, transition vectors, residuals
  signals_anchorflow.py # real prompt-anchor transport
  detectors.py          # calibrated online alarms and failure modes
  policies.py           # intervention policy ladder
  interventions.py      # retry, route, prompt repair, optional activation hooks
  eval.py               # same-problem, GroupKFold, FPR/recall/delay

scripts/
  extract_monitor_signals.py
  run_monitor_offline_audit.py
  run_intervention_replay.py
```

Implementation order:

1. `signals_entropy.py` + offline audit.
2. `signals_geometry.py` wrap existing spread/resultant extraction.
3. `detectors.py` with causal z + conformal alarm.
4. `policies.py` for replay-only interventions.
5. Real `signals_anchorflow.py`.
6. Live generation hooks.
7. Attention/activation repair only after replay success.

## 8. Paper Story If The Plan Works

Possible framing:

> We present a plug-in monitor for LLM reasoning that detects online loss of constraint in the generation flow.  Unlike confidence-only monitors, it combines entropy dynamics, hidden token-cloud concentration, and prompt-anchor transport.  The monitor distinguishes committed failures from persistent uncertainty and triggers targeted interventions.  Across step-labeled and same-problem settings, it improves calibrated early warning and intervention outcomes over static confidence, static geometry, and naive retry.

Claims to avoid:

- "We found the reasoning manifold" unless real anchors and controls pass.
- "Hidden probe causally fixes reasoning" unless activation intervention passes matched random/oracle controls.
- "Spectral/volume metric proves reasoning quality" without length/domain residualization.

## 9. Immediate Next Actions

1. Add entropy dynamic features to the same-problem and ProcessBench audit:
   - EDIS burst/rebound,
   - monotonicity violation count,
   - `SH`, `Vsp`, `avnr`,
   - early/mid/late means, slope, `r2`.
2. Build a prefix-window failure-mode audit:
   - committed vs persistent,
   - commitment point,
   - relation to self-consistency benefit.
3. Define prompt-anchor schema for GSM8K/MATH:
   - givens,
   - quantities,
   - constraints,
   - target,
   - answer format.
4. Re-extract or augment samples with token logprobs/top-k and prompt spans.
5. Run an offline intervention replay before touching live activation hooks.
