# Reboot Idea: Learned Latent Separability for Reasoning Failure

Date: 2026-07-09

## Why We Must Reset

The previous line of work should not be patched further. The results already
show that hand-built step geometry scores, residualized transition scores, and
healthy-connection residuals do not add reliable value beyond length, position,
and step difficulty controls. The falsified object is:

$$
\text{raw or lightly processed hidden geometry scalar}
\Rightarrow
\text{independent reasoning-error signal}.
$$

This does not falsify the stronger hypothesis we actually care about:

$$
\exists f_\theta:\ h_t \mapsto z_t
\quad\text{such that}\quad
z_t\ \text{separates faithful/correct reasoning states from failing states}.
$$

The reset is therefore to stop designing scalar diagnostics in raw hidden space
and instead learn a latent representation whose training objective is explicitly
about separability, invariance, and first-error dynamics.

## One-Paragraph Paper Story

Correct and incorrect reasoning are not reliably separable in raw hidden-state
space because raw activations entangle reasoning quality with step length,
position, surface form, operation type, and problem difficulty. We argue that
faithful reasoning becomes separable only after learning a problem-conditioned
latent chart that removes these nuisance factors while preserving the dynamical
boundary between a constraint-satisfying solution basin and self-consistent
wrong basins. Our method, **Latent Separatrix Reasoning Monitor** (LSRM), learns
this chart from hidden states using within-problem contrastive learning,
first-error survival supervision, nuisance-adversarial deconfounding, and a
learned energy landscape. In this chart, correct trajectories contract toward a
low-energy solution basin, while wrong trajectories cross a learned separatrix
before or at the first erroneous step. Unlike GeoFaith-style global latent
geometry or hand-crafted trajectory metrics, LSRM directly optimizes the latent
space for controlled separability and tests whether the separation survives
length-matched, phase-matched, and difficulty-matched controls.

## Core Hypothesis

The hypothesis should be upgraded from "correct is low-dimensional" to:

$$
\textbf{H1:}\quad
\text{faithful reasoning occupies a problem-conditioned solution basin in a
learned latent chart.}
$$

$$
\textbf{H2:}\quad
\text{reasoning errors correspond to crossing a latent separatrix into a
self-consistent wrong basin.}
$$

$$
\textbf{H3:}\quad
\text{this separatrix is not visible in raw hidden space because nuisance
variables dominate Euclidean geometry.}
$$

This is a different object from spread, curvature, or transition residual. The
claim is not that wrong steps are always farther, longer, noisier, or more
curved. The claim is that there exists a learned coordinate system in which
reasoning-state correctness becomes a basin-membership problem.

## What We Borrow From Prior Work

### GeoFaith

GeoFaith motivates learning a latent manifold from hidden states and using
geometry plus entropy dynamics to supervise faithfulness. The part worth keeping
is the move from raw hidden space to a learned latent chart. The part we should
not copy is relying on global trajectory-level geometric mining as the main
source of evidence, because that can still be confounded by length, domain, and
difficulty.

### GeoSteer

GeoSteer explicitly trains a VAE on high-quality reasoning trajectories and
steers hidden states in the learned latent manifold. This supports the premise
that learned latent manifolds are actionable, not just visualizations.

### Latent-GRPO / Self-Verifier Geometry

Latent-GRPO observes a terminal hidden-state core-periphery pattern: correct
terminal states cluster, incorrect ones scatter. This is close to the user's
original intuition, but it is mostly terminal-response geometry. We need to
extend the idea to step-level first-error dynamics and explicitly test nuisance
controls.

### Hidden Error Awareness

Hidden-state probes can predict correctness strongly, but the signal may be
diagnostic rather than causal. This is crucial: LSRM should not claim that the
latent direction repairs reasoning unless intervention tests show it. Detection
and mechanism must be separated.

### Chain-of-Embedding

Chain-of-Embedding treats progressive hidden states as a latent thinking path
and shows response correctness can be estimated without reading output text.
This supports the online, hidden-state-only monitor goal.

## Method: LSRM

### Data Unit

For each reasoning chain \(i\), step \(t\), and selected layer or layer pool:

$$
h_{i,t}\in\mathbb{R}^{d}.
$$

Labels:

$$
y_{i,t}=1
\quad\text{iff step }t\text{ is the first erroneous step or a post-error state},
$$

or for first-error localization:

$$
g_i=\text{gold first-error step},\quad g_i=-1\text{ for fully correct chains}.
$$

Important nuisance variables:

$$
n_{i,t}=(\text{step length},\ \text{absolute position},\ \text{relative phase},
\ \text{operation type},\ \text{dataset/domain},\ \text{model/source}).
$$

### Encoder

Learn a latent chart:

$$
z_{i,t}=f_\theta(h_{i,t}, \ell, \phi_t)\in\mathbb{R}^{m},
$$

where \(\ell\) is layer identity and \(\phi_t\) is a light phase embedding. The
encoder should be an MLP or small temporal transformer over step hidden states,
not a scalar feature block.

Recommended first implementation:

$$
f_\theta:
\mathbb{R}^{d}\rightarrow
\mathbb{R}^{256}\rightarrow
\mathbb{R}^{128}\rightarrow
\mathbb{R}^{m},
\quad m\in\{16,32,64\}.
$$

Use LayerNorm or whitening before the encoder. Do not feed old spread,
entropy, or HCR features into the main model.

### Loss 1: Step Hazard Supervision

Predict a first-error hazard:

$$
\lambda_{i,t}=\sigma(w^\top z_{i,t}+b).
$$

The response-level probability is survival-style:

$$
P_i(\text{error by }T)
=
1-\prod_{t=1}^{T}(1-\lambda_{i,t}).
$$

For chains with gold error \(g_i\), use a discrete-time survival loss:

$$
\mathcal{L}_{\mathrm{surv}}
=
-\log \lambda_{i,g_i}
-\sum_{t<g_i}\log(1-\lambda_{i,t}).
$$

For fully correct chains:

$$
\mathcal{L}_{\mathrm{surv}}
=
-\sum_{t=1}^{T_i}\log(1-\lambda_{i,t}).
$$

This directly avoids mean/max dilution.

### Loss 2: Within-Problem Contrastive Separability

If multiple rollouts exist for the same problem, define positives and negatives
within the same problem:

$$
\mathcal{P}(i,t)=
\{(j,s): q_j=q_i,\ y_{j,s}=0,\ \text{matched phase/operation}\},
$$

$$
\mathcal{N}(i,t)=
\{(j,s): q_j=q_i,\ y_{j,s}=1,\ \text{matched phase/operation}\}.
$$

Use supervised contrastive learning:

$$
\mathcal{L}_{\mathrm{supcon}}
=
-\sum_{a}
\frac{1}{|\mathcal{P}(a)|}
\sum_{p\in\mathcal{P}(a)}
\log
\frac{\exp(\operatorname{sim}(z_a,z_p)/\tau)}
{\sum_{u\in\mathcal{P}(a)\cup\mathcal{N}(a)}
\exp(\operatorname{sim}(z_a,z_u)/\tau)}.
$$

This is the core reason the method is not just a global probe: the contrast is
within problem and phase, so problem difficulty and length are controlled by
construction.

If the existing ProcessBench file lacks multiple rollouts per problem, use
two-stage validation:

1. Train with available labels and group by problem if IDs repeat.
2. Generate a multi-rollout subset for 100-300 problems to validate the real
within-problem version.

### Loss 3: Nuisance Deconfounding

Attach adversarial heads to \(z_{i,t}\) to predict nuisance variables:

$$
\hat n_{i,t}=a_\psi(z_{i,t}).
$$

The encoder minimizes detection loss while maximizing nuisance prediction loss:

$$
\min_{\theta}
\mathcal{L}_{\mathrm{surv}}
\alpha\mathcal{L}_{\mathrm{supcon}}
-\beta\mathcal{L}_{\mathrm{nuis}},
$$

$$
\min_{\psi}\mathcal{L}_{\mathrm{nuis}}.
$$

This is necessary because our earlier failures show that length and phase can
dominate apparent geometry.

### Loss 4: Energy Basin Separatrix

Learn an energy function:

$$
E_\eta(z,q)\in\mathbb{R}.
$$

Correct states should have low energy, erroneous states high energy:

$$
\mathcal{L}_{\mathrm{energy}}
=
\sum_{y=0} E_\eta(z,q)
+
\sum_{y=1}
\max(0,\gamma-E_\eta(z,q)).
$$

The learned separatrix is:

$$
\mathcal{S}_q=\{z:E_\eta(z,q)=\gamma/2\}.
$$

The geometric story becomes testable:

$$
\Delta E_t=E_\eta(z_{t+1},q)-E_\eta(z_t,q).
$$

Faithful reasoning should mostly satisfy:

$$
\Delta E_t \le 0
\quad\text{or remain low-energy},
$$

while first-error transitions should show:

$$
E_\eta(z_{g_i},q)>\gamma/2
\quad\text{or}\quad
\Delta E_{g_i-1}>0.
$$

This is a Lyapunov-like diagnostic, but not a fake theorem. It is a learned
energy certificate whose validity is empirical and must be tested.

### Optional VAE / VIB Variant

If we want a generative latent chart closer to GeoFaith/GeoSteer:

$$
q_\theta(z\mid h)=
\mathcal{N}(\mu_\theta(h),\operatorname{diag}(\sigma^2_\theta(h))).
$$

Train with a variational information bottleneck:

$$
\mathcal{L}_{\mathrm{VIB}}
=
\mathcal{L}_{\mathrm{surv}}
\alpha\mathcal{L}_{\mathrm{supcon}}
\beta D_{\mathrm{KL}}(q_\theta(z\mid h)\Vert\mathcal{N}(0,I)).
$$

The uncertainty term then becomes native:

$$
U_t=\frac{1}{m}\sum_{k=1}^m \log \sigma_{\theta,k}^2(h_t).
$$

But this should be an ablation, not the first implementation. A discriminative
latent separatrix is simpler and better aligned with ProcessBench labels.

## Why This Is Different From Our Failed Methods

Failed methods:

$$
h_t
\xrightarrow{\text{hand feature}}
s_t
\xrightarrow{\text{aggregation}}
\text{risk}.
$$

LSRM:

$$
h_t
\xrightarrow{\text{learned nuisance-invariant chart}}
z_t
\xrightarrow{\text{survival + contrastive + energy}}
\text{step hazard and basin membership}.
$$

The main difference is not the neural network itself. It is the training
constraint:

1. compare correct and wrong states under matched problem/phase;
2. punish length/position leakage;
3. model response risk as a survival process;
4. evaluate separability directly in the learned latent space.

## Minimum Experiments

### Experiment 1: Existing ProcessBench StepVec

Use canonical files:

```text
data/features/full_gsm8k.npz
data/features/full_math.npz
data/features/full_omnimath.npz
```

Train LSRM on `stepvec` only. No old geometry signals.

Report:

- step-level first-error AUROC/AUPRC;
- within-chain first-error rank;
- response-level AUROC/AUPRC via survival aggregation;
- calibration;
- latent separability under matched length/position bins.

Baselines:

- controls only: step length, position, relative phase, n_steps;
- logistic probe on raw hidden state;
- MLP on raw hidden state;
- CoE-style hidden trajectory features;
- GeoFaith-style VAE features if implemented;
- old spread/entropy/HCR only as negative historical controls.

### Experiment 2: Nuisance Stress Test

Evaluate whether latent separability survives after:

1. length-matched sampling;
2. position-matched sampling;
3. same-problem or same-template grouping;
4. operation-type matching;
5. dataset transfer: train GSM8K, test MATH/OmniMATH.

The main claim is valid only if:

$$
\mathrm{AUC}(\mathrm{LSRM})
>
\mathrm{AUC}(\mathrm{controls})
$$

and:

$$
\mathrm{AUC}(\mathrm{controls}+\mathrm{LSRM})
>
\mathrm{AUC}(\mathrm{controls})
$$

under these matched settings.

### Experiment 3: Multi-Rollout Conditional Chart

Generate multiple rollouts for the same problem. This is the experiment that
most directly tests the user's original hypothesis:

$$
\text{correct rollouts for the same problem cluster in }z,
\quad
\text{wrong rollouts split into wrong basins}.
$$

Metrics:

$$
D_{\mathrm{same}}^{+}
=
\mathbb{E}\|z^{\mathrm{correct}}_{i,r,t}
-z^{\mathrm{correct}}_{i,r',t'}\|,
$$

$$
D_{\mathrm{wrong}}^{-}
=
\mathbb{E}\|z^{\mathrm{correct}}_{i,r,t}
-z^{\mathrm{wrong}}_{i,r',t'}\|.
$$

Require:

$$
D_{\mathrm{same}}^{+}<D_{\mathrm{wrong}}^{-}
$$

after phase/operation matching.

### Experiment 4: Intervention Boundary Test

Do not claim causality unless this passes.

Patch or steer a wrong state toward the nearest correct basin in \(z\)-space:

$$
z'_t=z_t+\epsilon(c_q-z_t),
$$

decode approximately via a learned linear inverse or intervention direction in
hidden space:

$$
h'_t=h_t+B(z'_t-z_t).
$$

Test whether future step correctness improves or whether only the detector
score changes. If correctness does not improve, the method is a diagnostic
latent monitor, not a causal repair method.

## Decision Rule

This line is worth continuing only if the first implementation shows:

1. learned \(z\) beats raw hidden-state MLP under length/position controls;
2. survival response aggregation beats mean/max;
3. latent separability is visible within problem or matched bins;
4. adding nuisance adversarial training reduces length predictability without
   destroying correctness AUC.

If not, the premise "correct/error are separable in a learned latent
representation" is weak for this dataset/model, and we should pivot to verifier
training or explicit process supervision instead of inventing more geometry.

## Implementation Plan

File to create:

```text
latent_separatrix_audit.py
```

Modules:

1. `load_step_records`: load `stepvec`, labels, step text, lengths, ids.
2. `StepDataset`: flatten chain-step records while preserving chain groups.
3. `LatentEncoder`: MLP or small temporal encoder.
4. `SurvivalHead`: per-step hazard and response probability.
5. `NuisanceHeads`: length/position/domain adversaries via gradient reversal.
6. `EnergyHead`: optional basin energy.
7. `train_fold`: group split by problem/chain.
8. `evaluate_fold`: AUROC/AUPRC/rank/calibration/matched-bin AUC.
9. `visualize_latent`: UMAP/PCA, energy timeline, per-chain case cards.

Run command:

```bash
python latent_separatrix_audit.py \
  data/features/full_gsm8k.npz \
  --output_dir outputs/latent_separatrix_full_gsm8k \
  --tag full_gsm8k \
  --latent_dim 32 \
  --hidden_dim 256 \
  --epochs 50 \
  --batch_size 32 \
  --amp
```

## Implemented Code Path

Implemented file:

```text
latent_separatrix_audit.py
```

Canonical GPU-box inputs:

```text
data/features/full_gsm8k.npz
data/features/full_math.npz
data/features/full_omnimath.npz
```

The implementation is intentionally strict about the method:

1. **Load raw step hidden vectors only.**
   Necessity: previous failures show that old spread/HCR/entropy features can
   become length and phase proxies. The new test must answer whether raw hidden
   states can be mapped into a better latent chart.
   Code: `load_records`, `record_step_features`.

2. **Fit PCA/whitening inside each training fold only.**
   Necessity: raw vectors are high-dimensional and expensive; fold-local PCA
   gives a stable input coordinate without leaking test statistics.
   Code: `fit_pca_projector`, `transform_records`.

3. **Train a learned latent chart.**
   Necessity: the hypothesis is about learned separability, not handcrafted
   Euclidean geometry. The encoder is therefore the object under test.
   Code: `LatentSeparatrixNet.encoder`.

4. **Use discrete survival supervision instead of mean/max aggregation.**
   Necessity: response-level mean/max diluted earlier step signals. First-error
   timing is naturally a hazard process:

   $$
   P(\mathrm{error\ by\ }T)=1-\prod_t(1-\lambda_t).
   $$

   Code: `survival_loss`, `score_dataset`.

5. **Add contrastive pressure in latent space.**
   Necessity: if the latent chart is meaningful, correct and failed states
   should be separated by representation geometry, not merely by a classifier
   head.
   Code: `supervised_contrastive_loss`.

6. **Add an energy/basin diagnostic.**
   Necessity: the narrative is a basin/separatrix story. The model must expose
   a scalar energy that can be inspected along a chain, while still not being
   the only score.
   Code: `energy_loss`, `energy` head.

7. **Adversarial nuisance heads for length, position, operation, and n_steps.**
   Necessity: earlier empirical failures were dominated by nuisance geometry.
   This step tests whether the latent chart can retain error information while
   suppressing easy nuisance leakage.
   Code: `GradReverse`, `nuisance_loss`.

8. **GroupKFold by problem id.**
   Necessity: no chain from the same problem should be both train and test when
   repeated problem ids exist.
   Code: `make_folds`.

9. **Report both detection and latent evidence.**
   Necessity: high AUROC alone is not enough for the paper story. We also need
   first-error rank, response survival score, matched-bin AUC, and latent
   silhouette/separation diagnostics.
   Code: `evaluate_rows`, `rank_first_errors`, `weighted_bin_auc`,
   `latent_separability`, `write_markdown`.

Minimal remote self-test:

```bash
cd /gz-data/research/demo
python latent_separatrix_audit.py \
  --selftest \
  --output_dir outputs/latent_separatrix_selftest
```

Main ProcessBench run:

```bash
cd /gz-data/research/demo
python latent_separatrix_audit.py \
  data/features/full_gsm8k.npz \
  --output_dir outputs/latent_separatrix_full_gsm8k \
  --tag full_gsm8k \
  --latent_dim 32 \
  --hidden_dim 256 \
  --pca_dim 256 \
  --epochs 40 \
  --batch_size 32 \
  --eval_batch_size 64 \
  --amp
```

Expected outputs:

```text
outputs/latent_separatrix_full_gsm8k/full_gsm8k_latent_separatrix_rows.csv
outputs/latent_separatrix_full_gsm8k/full_gsm8k_latent_separatrix_chains.csv
outputs/latent_separatrix_full_gsm8k/full_gsm8k_latent_separatrix_latents.npz
outputs/latent_separatrix_full_gsm8k/full_gsm8k_latent_separatrix_summary.json
outputs/latent_separatrix_full_gsm8k/full_gsm8k_latent_separatrix_summary.md
```

## Bottom Line

The real target is not:

$$
\text{wrong}=\text{more spread}.
$$

The target is:

$$
\text{wrong}=\text{crossing a learned latent separatrix after nuisance factors
are removed}.
$$

That is the first formulation in this project that directly matches the user's
original intuition while avoiding the old trap of length-sensitive toy geometry.
