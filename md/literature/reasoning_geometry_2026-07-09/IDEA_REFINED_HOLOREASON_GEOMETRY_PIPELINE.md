# HoloReason: Gauge-Invariant Holonomy Fields for Detecting Unfaithful Reasoning

Date: 2026-07-09

Revision note: this v1 idea has a major theoretical risk because it frames faithful reasoning as approximately zero-holonomy / locally commutative. The stronger version is **HCR-Holo: Healthy-Connection Residual Holonomy**, which treats correct reasoning as following a phase- and operation-conditioned healthy connection and detects errors as residual holonomy relative to that connection. See `HOLOREASON_REVIEW_AND_V2_REVISION.md` in the same folder before implementing.

## One-Sentence Paper Idea

Reasoning failure is not merely hidden-state dispersion or entropy fluctuation; it is a local non-integrability of the transformer's computation surface: in faithful chain-of-thought, moving forward in reasoning steps and moving upward through layers should commute on the solution-relevant latent bundle, while erroneous or post-hoc reasoning creates a measurable holonomy defect.

## Why the Previous Direction Was Not Enough

The previous `spread` / spectral / phase-transition direction has three structural problems.

First, many scalar statistics are length and difficulty proxies. On GSM8K, the computationally central step is often longer, harder, and closer to the first arithmetic error. A scalar such as spread can therefore rank the erroneous step without really measuring the model's internal awareness or reasoning state.

Second, response-level aggregation destroys the only useful information. Max and mean either dilute the event or collapse into step count and chain length. This explains why step-level AUC can be acceptable while response-level AUC remains weak.

Third, a broad "spectral geometry field" is too close to existing work. *The Spectral Geometry of Thought* already studies layer-wise and token-wise spectral decay, spectral compression, token-level cascades, step punctuation, and correctness prediction. Simply adding more spectral summaries risks becoming an incremental feature bank.

Therefore the new method should not be another scalar family. It needs one sharp geometric claim, one invariant object, and a detection pipeline whose online score is local, normalized, and not reducible to length.

## Refined Hypothesis

Let hidden states form a two-dimensional computation surface:

- depth axis: transformer layer computation;
- time axis: reasoning-step progression.

For a faithful reasoning trajectory, these two flows should be locally compatible. That is, "process this reasoning state through another layer, then advance one step" should approximately match "advance one reasoning step, then process through another layer", at least inside the latent subspace that carries the current solution state.

Formally, let \(h_{\ell,t}\) denote the hidden state at layer \(\ell\) and reasoning step \(t\). Let \(D_{\ell,t}\) be the local depth transport and \(S_{\ell,t}\) be the local step transport:

$$
D_{\ell,t}: h_{\ell,t} \mapsto h_{\ell+1,t},
\qquad
S_{\ell,t}: h_{\ell,t} \mapsto h_{\ell,t+1}.
$$

Faithful reasoning should approximately satisfy local path-independence:

$$
S_{\ell+1,t}D_{\ell,t}h_{\ell,t}
\approx
D_{\ell,t+1}S_{\ell,t}h_{\ell,t}.
$$

The central diagnostic is the non-commutativity:

$$
\mathcal{C}_{\ell,t}
=
S_{\ell+1,t}D_{\ell,t}
-
D_{\ell,t+1}S_{\ell,t}.
$$

If this commutator is small on the solution-relevant subbundle, the model's internal computation is geometrically consistent. If it spikes, the hidden state has entered a locally inconsistent reasoning regime. This is a **holonomy defect**: after moving around a small layer-step loop, the state does not return to a compatible representation.

This gives a cleaner story than "wrong reasoning is more scattered":

> The model can be smooth and confident while wrong. What breaks is not necessarily dispersion, but the compatibility between depth-wise computation and step-wise reasoning evolution.

## Why This Is Different From GeoFaith

GeoFaith uses a spatio-temporal view: latent manifold geometry plus entropy dynamics, with bootstrapped faithfulness labels and a detector/RL pipeline. That is useful, but it treats the trajectory mostly as a path on a latent manifold.

HoloReason instead treats CoT as a **surface with a connection**. The diagnostic is not the distance of a point from a manifold, nor the entropy of a step, but the curvature/holonomy of the local computation loop:

$$
h_{\ell,t}
\rightarrow h_{\ell+1,t}
\rightarrow h_{\ell+1,t+1}
\quad\text{versus}\quad
h_{\ell,t}
\rightarrow h_{\ell,t+1}
\rightarrow h_{\ell+1,t+1}.
$$

This creates a mechanistic gap:

- GeoFaith: Is the trajectory geometrically faithful and temporally stable?
- HoloReason: Are the model's two internal reasoning flows locally integrable?

That gap is important because a wrong chain may still be low entropy, smooth, and geometrically close to a learned manifold. The holonomy defect can catch cases where the chain is fluent but the computational update and the textual reasoning update disagree.

## Why This Is Different From Spectral Geometry

The spectral geometry paper studies spectra of hidden-state matrices:

$$
H^{(\ell)} \in \mathbb{R}^{T \times d},
\qquad
\sigma_k(H^{(\ell)}) \propto k^{-\alpha}.
$$

HoloReason studies spectra of **transition inconsistency operators**, not activation matrices:

$$
\mathcal{C}_{\ell,t}^{\top}\mathcal{C}_{\ell,t}.
$$

This is a different object. It is operator-level and local-loop based. The spectral signature is attached to a failure of path-independence, not to the global rank or decay of hidden activations.

## Method

### 1. Build the Layer-Step Hidden Surface

For each response, segment the reasoning into steps. For each step representative token, collect hidden states across layers:

$$
\{h_{\ell,t}\}_{\ell=1}^{L}, \qquad t=1,\dots,T.
$$

The method should support three granularities:

- step-level: one pooled vector per reasoning step;
- token-level: all generated tokens;
- hybrid: step anchor token plus local token window.

The core method should be evaluated first at token-level and step-level separately. Step segmentation must not be allowed to become a hidden confound.

### 2. Learn Local Transport Operators Offline

For a local neighborhood \(\mathcal{N}_{\ell,t}\), learn regularized affine or linear transports:

$$
D_{\ell,t}
=
\arg\min_D
\sum_{j\in\mathcal{N}_{\ell,t}}
\lVert h^{(j)}_{\ell+1,t}-Dh^{(j)}_{\ell,t}\rVert_2^2
+\lambda\lVert D\rVert_F^2,
$$

$$
S_{\ell,t}
=
\arg\min_S
\sum_{j\in\mathcal{N}_{\ell,t}}
\lVert h^{(j)}_{\ell,t+1}-Sh^{(j)}_{\ell,t}\rVert_2^2
+\lambda\lVert S\rVert_F^2.
$$

The neighborhood should be controlled by phase and length:

- same dataset split only;
- same normalized step phase bin;
- similar step length bin;
- similar problem length/difficulty proxy;
- no gold error label in neighborhood construction.

This avoids learning an operator that simply encodes "later steps are longer".

A scalable version learns conditional low-rank transports:

$$
D_{\ell,t}(h)
=
U^D_{\ell,p}
\operatorname{diag}(a^D_{\theta}(h,\ell,p))
(V^D_{\ell,p})^\top,
$$

$$
S_{\ell,t}(h)
=
U^S_{\ell,p}
\operatorname{diag}(a^S_{\theta}(h,\ell,p))
(V^S_{\ell,p})^\top,
$$

where \(p=t/T\) is normalized reasoning phase. This gives offline training and online inference without recomputing neighborhoods.

### 3. Project Onto a Solution-Relevant Bundle

A full hidden vector contains token identity, style, length, and formatting. The holonomy score should be computed on a solution-relevant latent bundle \(\mathcal{B}_{\ell,t}\), not the entire residual stream.

Three possible bundle definitions should be compared:

1. **Slow tangent bundle**: local PCA directions with stable variance across neighboring correct chains.
2. **Predictive bundle**: directions that predict next-step semantic state or final answer class on training data.
3. **Causal-residual bundle**: directions whose removal changes next-token answer logits or verification logits.

The main paper can start with the slow tangent bundle because it is geometry-first and label-light. The predictive and causal bundles become stronger variants.

Let \(P_{\ell,t}\) be the projection onto \(\mathcal{B}_{\ell,t}\). The projected commutator is:

$$
\mathcal{C}^{\mathcal{B}}_{\ell,t}
=
P_{\ell+1,t+1}
\left(
S_{\ell+1,t}D_{\ell,t}
-
D_{\ell,t+1}S_{\ell,t}
\right)
P_{\ell,t}.
$$

### 4. Define the Gauge-Invariant Holonomy Score

Avoid raw frame coordinates. Use the action of the commutator on the local covariance:

$$
\operatorname{Hol}_{\ell,t}
=
\frac{
\lVert
\mathcal{C}^{\mathcal{B}}_{\ell,t}
\Sigma_{\ell,t}^{1/2}
\rVert_F^2
}{
\operatorname{tr}(\Sigma_{\ell,t})+\epsilon
}.
$$

This asks: how much does the local reasoning cloud deform when transported around the small depth-step loop?

Also compute a spectral signature of the holonomy operator:

$$
\lambda_i^{\ell,t}
=
\lambda_i\left(
(\mathcal{C}^{\mathcal{B}}_{\ell,t})^\top
\mathcal{C}^{\mathcal{B}}_{\ell,t}
\right).
$$

Useful summaries:

$$
\operatorname{HolRank}_{\ell,t}
=
\frac{
\left(\sum_i \lambda_i^{\ell,t}\right)^2
}{
\sum_i (\lambda_i^{\ell,t})^2+\epsilon
},
$$

$$
\operatorname{HolAniso}_{\ell,t}
=
\frac{\lambda_1^{\ell,t}}
{\sum_i \lambda_i^{\ell,t}+\epsilon}.
$$

Interpretation:

- high holonomy, high anisotropy: one dominant inconsistent update direction;
- high holonomy, high rank: distributed confusion across many directions;
- low holonomy, low entropy answer: coherent computation;
- low holonomy, wrong answer: possible learned shortcut or faithfully wrong premise.

### 5. Online Detector

At inference, the method should not need multiple rollouts. It uses the offline learned transport field.

For each generated step:

$$
\hat{h}^{d\rightarrow s}_{\ell+1,t+1}
=
S_{\ell+1,t}D_{\ell,t}h_{\ell,t},
$$

$$
\hat{h}^{s\rightarrow d}_{\ell+1,t+1}
=
D_{\ell,t+1}S_{\ell,t}h_{\ell,t}.
$$

The online event score is:

$$
z_{\ell,t}
=
\frac{
\lVert
P_{\ell+1,t+1}
(\hat{h}^{d\rightarrow s}_{\ell+1,t+1}
-
\hat{h}^{s\rightarrow d}_{\ell+1,t+1})
\rVert_2^2
}{
\widehat{\operatorname{Var}}_{\ell,p,\text{len}}+\epsilon
}.
$$

Normalize within layer, phase bin, and length bin. This is essential; otherwise the score will reproduce the earlier length problem.

The response-level score should not be a crude mean or max. Use survival modeling:

$$
\Pr(\text{first error at }t\mid z_{\leq t})
=
1-\exp
\left(
-\exp(g_\theta(z_{t-w:t},p_t,\ell))
\right).
$$

The windowed hazard head is allowed to be small. The novelty is not the head; it is the holonomy field.

## Expected Failure Modes and Why They Are Informative

### Case A: First arithmetic error

Prediction: holonomy spikes before or at the first arithmetic error because the step update encodes a textual continuation inconsistent with the depth-wise computation of the previous state.

### Case B: Smooth but wrong reasoning

Prediction: spread and entropy may stay low. Holonomy may still rise if the model's latent computation and generated textual step diverge. If holonomy also stays low, the model is not "internally confused"; it is coherently following a wrong algorithm. This distinction itself is a paper-worthy insight.

### Case C: Long but correct chain

Prediction: raw spread increases with length, but normalized holonomy remains stable if the computation is locally integrable.

### Case D: Self-correction

Prediction: holonomy spikes at the suspected wrong step, then decays after correction. A correction step should look like a deliberate re-alignment of the two flows.

## Experiments

### Main Detection Tasks

1. **Step-level first-error localization**
   - Label: first incorrect reasoning step.
   - Metric: within-chain AUROC, top-1 accuracy, mean gold percentile.

2. **Pre-error hazard**
   - Label: whether a currently correct prefix will later fail.
   - Metric: prefix-level AUROC, time-to-error calibration.

3. **Response-level correctness**
   - Label: final answer correct/incorrect.
   - Metric: AUROC, AUPRC, calibration.

4. **Faithfulness / post-hoc rationalization**
   - Label: perturbation-based faithfulness or generated wrong rationale.
   - Metric: separation between faithful-correct, faithful-wrong, unfaithful-correct, unfaithful-wrong.

### Baselines

Mandatory baselines:

- length, step index, normalized phase;
- token count and equation-count controls;
- entropy dynamics;
- hidden spread / spectral decay;
- GeoFaith-style spatial-temporal features if reproducible;
- transport-cost excursion from "Where Does Reasoning Break";
- previous CTG / hazard features;
- random subspace and permuted-step controls.

### Ablations

1. Full hidden space vs solution bundle.
2. Raw commutator vs covariance-normalized holonomy.
3. Euclidean distance vs Grassmannian/principal-angle score.
4. Step-level vs token-level loops.
5. Offline local regression vs learned conditional operator.
6. With vs without length/phase normalization.
7. Cross-domain transfer: GSM8K to MATH, StrategyQA, symbolic tasks.
8. Cross-model transfer: same detector on different model families.

### Negative Controls

These are essential because earlier results were contaminated by length.

1. Step permutation within a response.
2. Layer permutation.
3. Random orthogonal gauge rotation.
4. Same-length correct/incorrect matching.
5. Same-phase matching.
6. Wrong-answer anchor matching.
7. Gold-error label shuffling.
8. Synthetic long-correct chains.
9. Synthetic short-wrong chains.

The method only survives if holonomy remains predictive after these controls.

## Mechanistic Validation

Detection alone is not enough. The paper should include at least three mechanism experiments.

### 1. Holonomy-Guided Activation Patching

Patch the subbundle component at high-holonomy loops from a correct trajectory into an incorrect one:

$$
h'_{\ell,t}
=
h_{\ell,t}
+
P_{\ell,t}
\left(
h^{\text{correct}}_{\ell,t}
-
h^{\text{wrong}}_{\ell,t}
\right).
$$

Claim is supported if high-holonomy positions give larger recovery than low-holonomy positions at matched phase/length.

### 2. Counter-Holonomy Steering

Use the estimated commutator direction to reduce local inconsistency:

$$
h'_{\ell,t}
=
h_{\ell,t}
-
\eta
P_{\ell,t}
(\mathcal{C}^{\mathcal{B}}_{\ell,t})^\top
\mathcal{C}^{\mathcal{B}}_{\ell,t}
h_{\ell,t}.
$$

If this improves next-step correctness or lowers future error probability, the holonomy direction is not just diagnostic.

### 3. Self-Correction Prediction

When a model corrects itself, test whether holonomy decreases before the explicit correction text. If yes, the model has an internal re-alignment signal before verbalizing the correction.

This directly addresses the user's question: does the model know it is going wrong?

## Paper Narrative

The article should not claim "incorrect reasoning is more dispersed." That story is too weak and too easily confounded by length.

The stronger narrative is:

> Chain-of-thought exposes a second computation axis. A transformer reasoner is not only moving through layers; it is also moving through emitted reasoning steps. Faithful reasoning requires these two flows to be locally compatible. We show that first errors and unfaithful rationalizations appear as holonomy defects on the layer-step hidden-state surface. This geometric defect is local, online-computable after offline calibration, robust to length controls, and causally actionable through activation patching and counter-holonomy steering.

## Why This Could Beat GeoFaith

GeoFaith's strength is broad spatio-temporal supervision and faithfulness-aware training. HoloReason can beat or complement it if it demonstrates:

1. better first-error localization;
2. online single-trajectory scoring without multiple rollouts;
3. robustness after length/phase matching;
4. stronger cross-domain generalization because the score is a local operator inconsistency, not a domain-specific label;
5. mechanistic intervention: reducing holonomy improves reasoning or reveals whether the model is coherently wrong.

The target contribution is not just higher AUC. It is a new object: **the holonomy of reasoning computation**.

## Implementation Roadmap

### Stage 0: Audit Current Data

- Verify which arrays store all-layer hidden states.
- If only final-layer states exist, regenerate a smaller all-layer dataset first.
- Ensure correct data path is always documented: `data/features/full_gsm8k.npz`.

### Stage 1: Minimal HoloReason Prototype

- Use existing step segmentation.
- Use all-layer hidden states.
- Estimate local depth and step transports with ridge regression in phase/length bins.
- Compute covariance-normalized holonomy.
- Evaluate first-error localization.

### Stage 2: Token-Level Version

- Remove reliance on step segmentation.
- Compute token-level loops.
- Aggregate into step-level only after detecting token-level defects.

### Stage 3: Bundle Learning

- Compare full space, slow tangent bundle, predictive bundle, and causal-residual bundle.
- Keep the main method geometry-first; use predictive/causal variants as ablations.

### Stage 4: Online Hazard Model

- Train a small temporal hazard head on normalized holonomy windows.
- Compare against mean/max response aggregation.

### Stage 5: Mechanistic Tests

- Holonomy-guided patching.
- Counter-holonomy steering.
- Self-correction prediction.

## Go / No-Go Criteria

Continue only if Stage 1 satisfies at least two:

- step-level first-error top-1 improves over spread by at least 10 percentage points;
- within-chain AUROC improves over controls;
- response-level AUC improves over length/phase controls;
- same-length matched evaluation remains positive;
- high-holonomy patching positions recover more than random matched positions.

If Stage 1 only reproduces length or phase, abandon the method.

## Relation to Existing Literature

- GeoFaith: spatio-temporal faithful CoT detector using latent manifold geometry and entropy dynamics.
- Spectral Geometry of Thought: activation spectral decay, token-level cascades, correctness prediction.
- Where Does Reasoning Break: hidden-state transport excursion from stable manifold.
- Reasoning Emerges from Constrained Inference Manifolds: compression alone is not enough; reasoning needs non-degenerate information volume.
- Reasoning Models Don't Just Think Longer: raw trajectory geometry is strongly confounded by generation length, so normalization is non-negotiable.
- Lines of Thought: reasoning trajectories lie on low-dimensional non-Euclidean manifolds, motivating operator transport rather than raw Euclidean features.

## Short Title Options

1. HoloReason: Holonomy Defects Reveal Unfaithful Chain-of-Thought
2. When Reasoning Flows Fail to Commute
3. Curvature of Thought: Detecting Reasoning Errors by Layer-Step Holonomy
4. Gauge-Invariant Geometry of Chain-of-Thought Faithfulness
