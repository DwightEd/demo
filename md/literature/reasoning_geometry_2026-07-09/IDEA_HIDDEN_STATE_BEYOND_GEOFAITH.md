# Hidden-State Methods Beyond GeoFaith

Date: 2026-07-09

## One-Sentence Thesis

Correct chain-of-thought is not merely a smoother or lower-curvature hidden-state trajectory; it is a trajectory whose hidden state remains causally responsive to the original problem constraints while maintaining the ability to revise self-generated prefix commitments. Reasoning errors arise when this control structure breaks: the state becomes self-locked to its generated prefix, weakly responsive to constraint edits, and unable to convert internal conflict into revision.

## Why This Can Go Beyond GeoFaith

GeoFaith frames faithfulness through spatio-temporal hidden-state geometry: faithful reasoning has structured trajectory geometry, lower uncertainty, and coherent manifold behavior. That is already stronger than static scalar probes.

The gap is that geometry alone does not identify the causal source of the trajectory. A smooth trajectory can still be confidently wrong. A curved trajectory can be healthy exploration. A high-spread step can simply be a longer or harder local computation. The missing question is:

> Is the current reasoning state still controlled by the problem, or has it become controlled by its own generated prefix?

This suggests moving from geometric description to hidden-state control diagnosis.

## Main Proposal: CRESCENT

**CRESCENT: Causal Responsiveness and Self-Locking for Chain-of-Thought Error Tracking**

CRESCENT treats a reasoning chain as a controlled hidden-state dynamical system. The model receives two sources of control:

1. The original problem constraints.
2. The self-generated reasoning prefix.

A faithful chain should remain responsive to constraint-relevant counterfactual edits and should revise when the prefix conflicts with those constraints. An unfaithful chain becomes prefix-locked: it stays internally coherent but stops being causally governed by the question.

## Core Hidden-State Objects

For a reasoning step \(t\), let \(h_t\) be the hidden state used to generate the next step or token.

Construct four matched counterfactual siblings:

\[
x^{\mathrm{rel}}: \text{question with a relevant constraint edited}
\]

\[
x^{\mathrm{irr}}: \text{question with irrelevant wording edited}
\]

\[
y_{\le t}^{\mathrm{prefix}}: \text{prefix with a local self-generated claim edited}
\]

\[
y_{\le t}^{\mathrm{repair}}: \text{prefix minimally corrected toward the gold next state}
\]

Then measure hidden-state responses:

\[
\delta_t^{q}=h_t(x^{\mathrm{rel}},y_{\le t})-h_t(x,y_{\le t})
\]

\[
\delta_t^{r}=h_t(x^{\mathrm{irr}},y_{\le t})-h_t(x,y_{\le t})
\]

\[
\delta_t^{p}=h_t(x,y_{\le t}^{\mathrm{prefix}})-h_t(x,y_{\le t})
\]

\[
\delta_t^{c}=h_t(x,y_{\le t}^{\mathrm{repair}})-h_t(x,y_{\le t})
\]

The method is not based on the magnitude of one scalar. It compares which source actually moves the hidden state.

## Three Main Signals

### 1. Constraint Responsiveness

A state is healthy if relevant problem edits move it more than irrelevant edits:

\[
\mathrm{CR}(t)=
\frac{\|\delta_t^{q}\|_2}
{\|\delta_t^{q}\|_2+\|\delta_t^{r}\|_2+\epsilon}
\]

Low \(\mathrm{CR}(t)\) means the state is no longer sensitive to the original constraint. This directly attacks the "smooth but wrong" case that geometry-only methods can miss.

### 2. Prefix Locking

A state is risky if prefix edits dominate problem edits:

\[
\mathrm{PL}(t)=
\frac{\|\delta_t^{p}\|_2}
{\|\delta_t^{q}\|_2+\|\delta_t^{p}\|_2+\epsilon}
\]

High \(\mathrm{PL}(t)\) means the model is mostly following its own generated reasoning, not the problem.

### 3. Revision Readiness

Errors are especially dangerous when the model contains conflict but does not revise. Define a conflict direction from correct-vs-wrong paired states:

\[
v_{\mathrm{err}}=
\mathbb{E}[h_t^{\mathrm{wrong}}-h_t^{\mathrm{correct}}]
\]

The conflict score is:

\[
\mathrm{Conflict}(t)=
\sigma(\langle h_t, v_{\mathrm{err}}\rangle)
\]

The repair response is:

\[
\mathrm{RR}(t)=
\frac{\|\delta_t^{c}\|_2}
{\|\delta_t^{c}\|_2+\|\delta_t^{p}\|_2+\epsilon}
\]

The failure mode is:

\[
\mathrm{AwareButLocked}(t)=
\mathrm{Conflict}(t)\cdot (1-\mathrm{RR}(t))\cdot \mathrm{PL}(t)
\]

This gives an attractive story: the model may internally sense something is wrong, but the generated prefix has become a stronger controller than the original problem.

## Online Version

Counterfactual siblings are expensive. Use them offline to train a hidden-state student:

\[
g_{\theta}(h_t, h_{t-1}, \Delta h_t, t, \ell_t)
\rightarrow
\left[
\widehat{\mathrm{CR}}(t),
\widehat{\mathrm{PL}}(t),
\widehat{\mathrm{RR}}(t),
\widehat{p}_{\mathrm{err}}(t)
\right]
\]

At inference, the online detector only needs the normal forward pass and cached hidden states. This gives the desired setting:

> expensive offline causal training, cheap online real-time response.

## Response-Level Aggregation Without Dilution

Avoid mean/max over raw step signals. Use a survival process:

\[
P(\mathrm{error\ by\ }T)=
1-\prod_{t=1}^{T}(1-\lambda_t)
\]

where:

\[
\lambda_t=
\sigma(
w_1(1-\widehat{\mathrm{CR}}(t))
+w_2\widehat{\mathrm{PL}}(t)
+w_3\widehat{\mathrm{AwareButLocked}}(t)
+w_4\Delta\widehat{\mathrm{PL}}(t)
)
\]

This preserves local spikes and models first-error timing directly, instead of averaging away step-level evidence.

## Why This Is Not Another Toy Geometry Score

The unit of analysis is no longer "does hidden geometry spread?" but "which causal source controls the hidden state?"

The method distinguishes:

1. Hard but faithful computation: high movement, high constraint responsiveness.
2. Smooth correct reasoning: low conflict, stable constraint responsiveness.
3. Smooth wrong reasoning: high prefix locking, low constraint responsiveness.
4. Chaotic exploration: high movement but no stable prefix locking.
5. Aware-but-unrevised error: conflict signal appears, but repair response is weak.

These cases collapse under scalar spread, curvature, entropy, or \(\kappa\).

## Stronger Variant: Causal Subspace Decomposition

Instead of using raw norms, learn three orthogonal or oblique subspaces:

\[
\mathcal{S}_q: \text{question-control subspace}
\]

\[
\mathcal{S}_p: \text{prefix-control subspace}
\]

\[
\mathcal{S}_e: \text{error/conflict subspace}
\]

Using paired counterfactual responses, estimate:

\[
U_q=\mathrm{PCA}(\{\delta_t^q-\delta_t^r\})
\]

\[
U_p=\mathrm{PCA}(\{\delta_t^p\})
\]

\[
U_e=\mathrm{LDA}(\{h_t^{\mathrm{wrong}},h_t^{\mathrm{correct}}\})
\]

Then replace raw norms with projected energies:

\[
E_q(t)=\|\Pi_{U_q}h_t\|_2^2
\]

\[
E_p(t)=\|\Pi_{U_p}h_t\|_2^2
\]

\[
E_e(t)=\|\Pi_{U_e}h_t\|_2^2
\]

The key diagnostic becomes a control-ratio trajectory:

\[
\rho(t)=
\frac{E_q(t)}
{E_q(t)+E_p(t)+E_e(t)+\epsilon}
\]

The expected failure signature is a phase shift:

\[
\rho_q(t)\downarrow,\quad
\rho_p(t)\uparrow,\quad
\rho_e(t)\uparrow\ \text{or remains suppressed}
\]

This is a hidden-state phase transition, but defined by causal control rather than by intrinsic geometry alone.

## Alternative Ideas Worth Keeping

### A. Hidden Counterfactual Equivariance

Faithful reasoning should transform predictably under semantics-preserving or relation-preserving edits. If numbers/entities are permuted, the hidden trajectory should follow an equivariant transformation. Wrong reasoning breaks this property before the answer is visibly wrong.

Core score:

\[
\mathrm{EqErr}(t)=
\|h_t(Tx)-T_h h_t(x)\|_2
\]

This is stronger than spread because it defines the expected direction of movement.

### B. Latent Revision Field

Learn a vector field that maps wrong states toward correct states:

\[
F_{\mathrm{rev}}(h_t)\approx h_t^{\mathrm{correct\ continuation}}-h_t^{\mathrm{wrong\ continuation}}
\]

Then ask whether the model's natural next hidden update aligns with this revision field:

\[
\mathrm{RevAlign}(t)=
\cos(\Delta h_t,F_{\mathrm{rev}}(h_t))
\]

Wrong reasoning can be described as moving orthogonally to the revision field even when conflict is internally present.

### C. Hidden-State Mediation Test

Instead of asking whether a hidden state predicts correctness, test whether it mediates the causal effect of the question on the answer. Use activation patching or learned interventions:

\[
x \rightarrow h_t \rightarrow y
\]

The paper claim becomes: faithful CoT is not a trajectory with certain geometry; it is a trajectory whose hidden states mediate problem constraints into answer logits.

## Experiments Needed

### Detection

Compare against:

1. GeoFaith.
2. EDIS or entropy-dynamics methods.
3. Where Does Reasoning Break style token/step localization.
4. Spectral geometry / trajectory phase transition baselines.
5. Static hidden probes.
6. Semantic entropy and self-consistency.
7. Length, position, and step-index controls.

Metrics:

1. First-error AUROC/AUPRC.
2. Response-level AUROC/AUPRC.
3. Risk-coverage.
4. Calibration.
5. Length-matched AUC.
6. Hard-step matched AUC.
7. Coherent-wrong subset recall.

### Mechanism

1. Patch high-\(\mathrm{CR}\) states from correct runs into wrong runs.
2. Patch high-\(\mathrm{PL}\) states from wrong runs into correct runs.
3. Edit question constraints and verify hidden response changes before output changes.
4. Edit irrelevant tokens and show low response.
5. Replace erroneous prefix sentence and test whether repair response predicts recovery.

### Online Real-Time Claim

Train the counterfactual teacher offline, then distill to a student that uses only hidden states from the normal forward pass. Report latency, memory, and token-level streaming behavior.

## Main Paper Narrative

Geometry-based faithfulness detection has shown that wrong reasoning often follows different hidden-state trajectories, but trajectory shape alone is not enough: many errors are smooth, confident, and locally coherent. We argue that faithful reasoning is better understood as a hidden-state control problem. During correct reasoning, the state remains causally responsive to the original problem constraints and can revise self-generated commitments when they conflict with those constraints. During failure, the state becomes prefix-locked: it is driven more by its own generated chain than by the problem. CRESCENT operationalizes this view with counterfactual hidden-state response fields, learns question-control, prefix-control, and revision subspaces offline, and distills them into an online detector. This gives a causal, step-level, real-time account of reasoning failure that explains why scalar geometry sometimes works, when it fails, and how to detect coherent-but-wrong chains that geometry-only methods miss.

## Recommendation

The strongest direction is not to abandon geometry, but to make geometry conditional on causal responsiveness:

\[
\text{faithfulness} \neq \text{low curvature}
\]

\[
\text{faithfulness} = \text{problem-controlled hidden dynamics with revision capacity}
\]

This is the cleanest route to a method that can plausibly beat GeoFaith in novelty, mechanism, and practical online detection.
