# HCR-Holo ProcessBench Validation Plan

Date: 2026-07-09

## One-Paragraph Draft

Existing hidden-state geometry methods usually ask whether a reasoning trajectory is smooth, dispersed, curved, or entropic, but these quantities often collapse into length, phase, and step difficulty. We instead treat chain-of-thought as a hidden-state surface with two coupled flows: layer-wise computation and step-wise prefix evolution. Faithful reasoning need not be flat; it only needs to follow the healthy connection learned from correct chains under matched phase, length, and operation conditions. A first error is detected as **healthy-connection residual holonomy**: the local layer-step loop either fails to close where the healthy connection predicts, or its two traversal paths disagree more than healthy loops of the same type. A finite-time contraction score then separates unstable confusion from stable wrong-basin capture. This gives a geometry-first detector with a clearer mechanism than spread or entropy: reasoning fails when its internal computation leaves the healthy connection, not merely when its hidden states move more.

## Revised Core Hypothesis

The old hypothesis was too weak:

$$
\text{wrong reasoning} \Rightarrow \text{more hidden-state dispersion}.
$$

The revised hypothesis is:

$$
\text{faithful reasoning}
\Rightarrow
\text{phase-conditioned compatibility between depth flow and step flow}.
$$

For a prefix state \(c_t=(x,y_{\le t})\), let \(h_\ell(c_t)\) be the step-level residual-stream state at layer \(\ell\). Correct chains induce a healthy connection:

$$
\mathcal{A}^{+}_{\ell,p,b,o}
=
\{A^{d,+}_{\ell,p,b,o}, A^{s,+}_{\ell,p,b,o}\},
$$

where \(p\) is normalized step phase, \(b\) is a step-length bin, and \(o\) is a coarse operation type. The diagnostic is not raw curvature. It is the residual relative to this healthy connection:

$$
\operatorname{HCR}_{\ell,t}
=
\frac{
q_{\ell,t}-\mu^{+}_{\ell,p,b,o}
}{
\sigma^{+}_{\ell,p,b,o}+\epsilon
}.
$$

## Geometry Object

For each step and layer, project hidden states into a low-rank healthy bundle:

$$
z_{\ell,t}=P_\ell^\top(h_{\ell,t}-\mu_\ell).
$$

The first validation uses a fold-specific PCA/whitening bundle learned only from training-fold correct chains. This is deliberately weaker than a supervised probe and avoids turning the method into a hidden label classifier.

The local depth and step transports are ridge maps in this bundle:

$$
A^{d,+}_{\ell,p,b,o}
=
\arg\min_A
\sum_i
\|z^{(i)}_{\ell+1,t}-Az^{(i)}_{\ell,t}\|_2^2
+\lambda\|A\|_F^2,
$$

$$
A^{s,+}_{\ell,p,b,o}
=
\arg\min_A
\sum_i
\|z^{(i)}_{\ell,t+1}-Az^{(i)}_{\ell,t}\|_2^2
+\lambda\|A\|_F^2.
$$

The local plaquette has two paths:

$$
\hat{z}^{ds}_{\ell+1,t+1}
=
A^{s,+}_{\ell+1,p,b,o}
A^{d,+}_{\ell,p,b,o}
z_{\ell,t},
$$

$$
\hat{z}^{sd}_{\ell+1,t+1}
=
A^{d,+}_{\ell,p^+,b^+,o^+}
A^{s,+}_{\ell,p,b,o}
z_{\ell,t}.
$$

Two complementary scores are used:

$$
\operatorname{Hol}^{comm}_{\ell,t}
=
\|\hat{z}^{ds}_{\ell+1,t+1}-\hat{z}^{sd}_{\ell+1,t+1}\|_2^2,
$$

$$
\operatorname{Hol}^{close}_{\ell,t}
=
\left\|
z_{\ell+1,t+1}
-
\frac{1}{2}
(\hat{z}^{ds}_{\ell+1,t+1}+\hat{z}^{sd}_{\ell+1,t+1})
\right\|_2^2.
$$

The first score asks whether the two induced flows are mutually compatible. The second asks whether the actual next state lands where the healthy connection says it should.

## Basin Interpretation

Raw anomalies do not tell us whether the model is confused or confidently wrong. We therefore add a finite-time contraction diagnostic in the same learned step transport:

$$
\Lambda_{\ell,t}
=
\log \sigma_{\max}(A^{s,+}_{\ell,p,b,o}).
$$

This is not a full asymptotic Lyapunov theorem. It is an online, fold-estimated local contraction score:

- high HCR and high \(\Lambda\): unstable transition or unresolved conflict;
- high HCR and low \(\Lambda\): capture into a stable wrong basin;
- low HCR and high \(\Lambda\): healthy exploration or naturally difficult transition;
- low HCR and low \(\Lambda\): healthy stable reasoning.

This is the part that can produce a stronger story than AUC alone: the model may internally leave the healthy connection before it verbally errors, and after the error it can become stable again, but in the wrong basin.

## Claims To Validate First

1. **Localization**: HCR identifies the first wrong step better than raw geometry, length, phase, and random-subspace controls.
2. **Residualization**: HCR remains useful after phase/length/op conditioning, while raw spread-style signals lose much of their apparent gain.
3. **Basin split**: the contraction score separates unstable-error chains from committed-wrong chains, giving a qualitative mechanism beyond ranking.
4. **Online feasibility**: all maps are trained offline; online scoring requires only step vectors and small \(k\times k\) maps.

## Required ProcessBench Data

The validation script expects a ProcessBench extraction with raw step vectors:

```bash
python 01_extract_spectral_field.py \
  --model /path/to/model \
  --dataset Qwen/ProcessBench \
  --subset gsm8k \
  --n_correct 500 \
  --n_error 500 \
  --layers all \
  --step_vectors \
  --sv_modes step_exp \
  --store_vectors \
  --output data/processbench_gsm8k_stepvec.npz
```

The key fields are:

- `labels`: `-1` means fully correct; non-negative means first wrong step.
- `layers_used`: extracted layer indices.
- `sv_vec_step_exp`: object array of hidden-state step vectors shaped `(T, L, d)`.
- optional `ids`, `kept_steps`, `n_steps`, and `sv_pr_step_exp`.

The script also supports older files with `stepvec` and `gold_error_step`.

## First Implementation

The first implementation is `hcr_holo_audit.py`.

It intentionally implements the simplest faithful version of the theory:

- fold-specific healthy PCA bundle;
- correct-chain-only transport learning;
- phase/length/operation-conditioned maps with global fallback;
- commutator and closure residuals;
- local contraction score;
- first-error and pre-error-future evaluations;
- random-subspace and permuted-step controls can be added after the first pass if the signal is nontrivial.

This is the right first validation because it tests the new object directly before building a larger pipeline around it.
