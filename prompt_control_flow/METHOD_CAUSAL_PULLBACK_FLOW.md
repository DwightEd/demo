# Causal Pullback Flow Field

## 1. Why the Geometry-Only View Is Insufficient

The same-problem conditional spherical field passed its existence gate, but its
error-excursion hypothesis failed:

```text
geometry existence: True
error excursion: False
length-residual within-problem AUROC: 0.545
```

This rules out the simple story that an erroneous response must leave a
low-density region of correct hidden-state directions. A wrong response may be
geometrically plausible. The unresolved question is whether that plausible
state is coupled to the output map in an abnormal way.

This method therefore does not classify a response from curvature, distance,
or density alone. It estimates a **causal geometry-to-output operator**.

## 2. Hypothesis

For problem $q$, observed transition $t$, and hidden layer \(\ell\), let
\(u_{q,t,\ell}\) be the normalized hidden-state update. Correct same-problem
responses define a causal-phase-conditioned reference distribution
\(\mathcal F_{q,t,\ell}\).

The revised hypothesis is:

> Errors need not leave \(\mathcal F_{q,t,\ell}\). Instead, directions normal
> to the local feasible field may be amplified abnormally by the downstream
> map from residual state to future token distributions.

The hypothesis has two distinct parts:

1. **field geometry**: estimate a direction derived from the correct reference
   field;
2. **output consequence**: causally measure how that direction changes future
   output distributions.

Neither part is collapsed into one handcrafted scalar during extraction.

## 3. Geometry-Derived Witness

The correct references use the same problem and matched causal transition
index. For chordal energy

\[
\mathcal E(u;\mathcal F)
=
\frac{1}{K}\sum_{j=1}^{K}\lVert u-v_j\rVert_2
-
\frac{1}{2K(K-1)}\sum_{j\ne k}\lVert v_j-v_k\rVert_2,
\]

the reference-spread term is constant with respect to $u$. The witness is
the spherical tangent gradient

\[
w_t
=
\frac{(I-u_tu_t^\top)\nabla_u\mathcal E(u_t;\mathcal F_{q,t})}
{\left\lVert(I-u_tu_t^\top)\nabla_u\mathcal E(u_t;\mathcal F_{q,t})\right\rVert_2}.
\]

The gradient norm is retained separately. It measures local field pressure;
the normalized direction defines the intervention axis.

There is one explicit modeling assumption here. The stored transition
direction and the residual state at the intervention site live in the same
ambient hidden coordinates, but identifying a tangent displacement direction
with an admissible residual-state perturbation is **not** guaranteed by a
manifold theorem. It is an empirical tangent-space identification. Replay
alignment, finite-difference convergence, the time-shuffled witness, and the
random tangent are included precisely to falsify this identification when it
does not carry meaningful downstream information.

Two matched controls are mandatory:

- `shuffle`: references from the same correct donors but time-shuffled;
- `random`: a deterministic random tangent orthogonal to the current update
  and the primary witness.

Donor count is fixed per problem and does not depend on the target label.

## 4. Strict Causal Timing

Transition $t$ is the displacement from step $t$ to step $t+1$. It is not
observable until step $t+1$ has finished. The intervention is therefore
inserted at the **endpoint of step $t+1$** after decoder layer \(\ell-1\),
which corresponds to stored hidden-state index \(\ell\).

Only output steps strictly after $t+1$ are scored. All past/current cells are
stored as missing, and any measured acausal KL leakage is a numerical-failure
diagnostic. This prevents future geometry from being injected into the past.

## 5. Categorical-Fisher Pullback

Let $p_s$ be the baseline token distribution at future output step $s$, and
let $J_{s\leftarrow t}$ denote the downstream Jacobian from the intervention
state to output logits. The directional observability is

\[
\mathcal O_{s,t}
=
w_t^\top J_{s\leftarrow t}^\top
\left(\operatorname{diag}(p_s)-p_sp_s^\top\right)
J_{s\leftarrow t}w_t.
\]

The implementation never materializes $J$ or a vocabulary-sized Fisher
matrix. It uses central KL curvature:

\[
\widehat{\mathcal O}_{s,t}
=
\frac{
D_{\mathrm{KL}}(p_s\Vert p_s^{+\epsilon})
+D_{\mathrm{KL}}(p_s\Vert p_s^{-\epsilon})
}{\epsilon^2}.
\]

The perturbation magnitude is a fixed fraction of the source hidden-state
norm. A second estimate at \(\epsilon/2\) tests finite-difference convergence.
The extraction also records signed derivatives of chosen-token log probability
and entropy.

The primary consequential operator multiplies Fisher sensitivity by squared
field pressure:

\[
\mathcal C_{s,t}=\lVert\nabla\mathcal E_t\rVert_2^2
\widehat{\mathcal O}_{s,t}.
\]

With source transition on rows and future output step on columns, the full
strictly upper-triangular operator is saved for every response. Phase-grid
pooling, spectral concentration, propagation horizon, and null contrasts are
derived only in the audit.

## 6. Detection and Increment Protocol

The output-only model contains:

- entropy;
- chosen-token NLL;
- top-1/top-2 margin;
- top-1 probability;
- step count and response length;
- replay and finite-difference diagnostics.

The causal branch contains the phase-resolved field, shuffled, and random
operators plus signed output derivatives. Evaluation is grouped by problem and
cross-fitted:

\[
P(Y\mid Z_{1:T})
\quad\text{versus}\quad
P(Y\mid Z_{1:T},\mathcal C),
\]

where $Z$ denotes the compact output history. The report includes AUROC,
AUPRC, conditional usable information in bits, a length-matched permutation
null, and the amount of operator geometry recoverable from output history.

The direct mechanism test is equal-weight within-problem AUROC after
cross-fitted length residualization. No per-dataset sign flipping is allowed.

## 7. Ordered Decision Gates

1. **Numerical validity**
   - replay/stored-state cosine passes the threshold;
   - finite-difference relative error passes;
   - acausal KL leakage is negligible;
   - valid coverage and contrastive-problem support pass.
2. **Mechanism support**
   - field consequentiality is above chance after length control;
   - it beats both the time-shuffled and random tangent controls.
3. **Detector increment**
   - conditional usable-information confidence interval is above zero;
   - AUROC increment is above zero;
   - the joint model beats the length-matched operator null.
4. **Confirmatory status**
   - all prior gates pass;
   - exact generation traces are used;
   - observer checkpoint identity is verified.

The legacy multisample artifact can be replayed, but it remains exploratory
even when stored-vector cosine is high. Exact-trace re-extraction is justified
only after the exploratory operator passes.

## 8. Code Structure

```text
prompt_control_flow/causal_pullback/
  schema.py       variable-size operator artifact and configs
  data.py         legacy/exact replay protocol
  field.py        correct field and matched witness controls
  replay.py       layer interventions and central-KL operator
  extraction.py   checkpointed extraction orchestration
  features.py     phase-grid operator representation
  audit.py        same-problem and conditional-increment tests
```

The implementation batches perturbation variants on GPU and chunks vocabulary
projection across response-token positions. It never saves full attention or
full logits.

## 9. Remote Commands

Preflight does not load the model:

```bash
python extract_causal_pullback.py \
  --input data/gsm8k_v2_custom.npz \
  --output outputs/causal_pullback/gsm8k_custom_l16/pullback_trace.npz \
  --layer 16 \
  --prompt_style custom_zeroshot \
  --preflight
```

Run a separate replay pilot first. Do not resume this small artifact into the
full run because donor supports differ after sample truncation:

```bash
python extract_causal_pullback.py \
  --input data/gsm8k_v2_custom.npz \
  --output outputs/causal_pullback/gsm8k_custom_l16_pilot/pullback_trace.npz \
  --model /gz-data/models/Meta-Llama-3.1-8B-Instruct \
  --layer 16 \
  --prompt_style custom_zeroshot \
  --max_samples 120 \
  --variant_batch_size 8 \
  --checkpoint_every 5
```

If replay cosine and finite-difference diagnostics pass, run the complete
artifact:

```bash
python extract_causal_pullback.py \
  --input data/gsm8k_v2_custom.npz \
  --output outputs/causal_pullback/gsm8k_custom_l16/pullback_trace.npz \
  --model /gz-data/models/Meta-Llama-3.1-8B-Instruct \
  --layer 16 \
  --prompt_style custom_zeroshot \
  --variant_batch_size 8 \
  --checkpoint_every 10 \
  --resume
```

Audit the operator:

```bash
python audit_causal_pullback.py \
  --input outputs/causal_pullback/gsm8k_custom_l16/pullback_trace.npz \
  --output_dir outputs/causal_pullback/gsm8k_custom_l16/audit \
  --bootstrap 2000
```

## 10. Claim Boundary

This experiment is an offline same-problem ensemble/reference-teacher test. A
positive result would justify distilling the causal operator into a
single-trajectory online student. It would not by itself establish a deployable
real-time detector. A negative result retires this particular field witness and
pullback construction; it does not prove that hidden states contain no output-
relevant information.
