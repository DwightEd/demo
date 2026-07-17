# Causal Belief Routing Under Predictive Aliasing

## One-sentence thesis

A pretrained Transformer can preserve future-relevant constraint beliefs that
are invisible to the current output distribution, and the preservation is
implemented by evidence-token attention routes whose OV writes move the
residual stream along analytically known belief-update directions.

This is a mechanism claim, not a claim that every language-model computation is
Bayesian.

## Problem anchor

The earlier geometry audits failed because they started from generic observables
such as curvature, spread, local dimension, or trajectory length. Those
quantities do not specify *what computation* the geometry should encode and are
easy to confound with step length or response position.

The new experiment starts from a latent variable with an exact posterior and an
exact update operator. Geometry is evaluated only after these objects are fixed.

## Controlled world

Let the hidden assignment be

\[
x\in\mathbb F_p^n,
\]

with a uniform prior. Evidence consists of affine constraints

\[
A_t x=c_t \pmod p.
\]

The exact posterior is uniform on the affine feasible set

\[
\mathcal S_t=\{x:A_t x=c_t\},
\qquad
b_t(x)=\frac{\mathbf 1[x\in\mathcal S_t]}{|\mathcal S_t|}.
\]

Each world contains two matched branches. They share the same common evidence
and differ in one final constraint with the same coefficient vector and a
different right-hand side. Therefore the two branches have equal posterior
entropy and equal evidence information gain.

## Predictive alias

For each branch pair, choose a current query vector \(q\) linearly independent
of all observed constraints. Then

\[
q^\top x\mid A_t x=c_t
\sim \operatorname{Uniform}(\mathbb F_p)
\]

in both branches. The exact current output distributions are identical.

Choose a future query \(f\) in the row span of the observed constraints. Its
answer is deterministic, and the two branch-specific right-hand sides make the
future answers different. Thus

\[
P(Y_{\mathrm{now}}\mid C_A)
=P(Y_{\mathrm{now}}\mid C_B),
\qquad
P(Y_{\mathrm{future}}\mid C_A)
\ne P(Y_{\mathrm{future}}\mid C_B).
\]

This removes the principal ambiguity in a generic hidden-versus-logits test:
the benchmark itself guarantees that current output information is
insufficient.

## Fourier belief geometry

The primary chart is not PCA, VAE, Isomap, or an arbitrary learned manifold.
For a frequency \(k\in\mathbb F_p^n\), define the character coordinate

\[
\phi_t(k)
=\mathbb E_{x\sim b_t}
\left[\exp\left(\frac{2\pi i}{p}k^\top x\right)\right].
\]

For a uniform posterior on an affine subspace, the coordinates have an exact
form:

\[
\phi_t(k)=
\begin{cases}
\exp\left(\frac{2\pi i}{p}\lambda^\top c_t\right),
& k=A_t^\top\lambda,\\
0,&k\notin\operatorname{rowspan}(A_t).
\end{cases}
\]

Consequently, a new independent constraint grows the non-zero Fourier support
from \(p^{r}\) to \(p^{r+1}\), while its right-hand side controls the phases of
the newly active modes. This is the relative direction structure tested by the
project.

The answer distribution for any linear query is recovered without an auxiliary
classifier:

\[
P(q^\top x=r)
=\frac{1}{p}\sum_{\lambda\in\mathbb F_p}
\phi_t(\lambda q)
\exp\left(-\frac{2\pi i}{p}\lambda r\right).
\]

## Representation test

At each selected residual depth \(\ell\), fit a group-cross-fitted affine chart
from boundary residual state \(h_{\ell,t}\) to the real and imaginary Fourier
coordinates:

\[
\widehat\Phi_{\ell,t}=W_\ell h_{\ell,t}+a_\ell.
\]

The held-out unit is an alias pair, never an individual row. The chart is judged
by:

1. held-out Fourier reconstruction error;
2. exact future-query NLL recovered from \(\widehat\Phi\);
3. branch identification within an alias pair;
4. improvement over current residue logits and a fixed full-vocabulary logit
   sketch;
5. shuffled-belief and random-subspace controls;
6. individual-layer versus concatenated-layer performance.

The central representation claim passes only if current outputs remain aliased,
future behavior is above chance, and hidden states recover future information
beyond the actual current logits.

## Attention-mediated update test

Raw attention weights are routing evidence, not causal contribution. For block
\(\ell\), head \(h\), boundary token \(t\), and evidence-token set \(E\), the
source-specific pre-output contribution is

\[
u_{\ell h t}^{E}
=\sum_{s\in E}\alpha_{\ell hts}v_{\ell hs}.
\]

The residual write is

\[
w_{\ell h t}^{E}=W_{O,\ell}^{(h)}u_{\ell h t}^{E}.
\]

Applying the held-out Fourier chart Jacobian gives the head's belief-coordinate
write:

\[
\delta\widehat\Phi_{\ell h t}^{E}
=J_\ell w_{\ell h t}^{E}.
\]

Its primary alignment score is the cosine with the analytically known update

\[
\Delta\Phi_t^\star=\Phi(b_t)-\Phi(b_{t-1}).
\]

Required controls are same-length non-evidence tokens, the opposite branch's
update, and same-layer randomly selected heads. Fourier-label permutation is
performed only within each training fold, so held-out alias pairs never shape
their own null model.

## Causal mediation test

For a donor and recipient in one predictive-alias pair, replace only the
recipient's evidence-source contribution for a selected head:

\[
u'_{h}=u_{h}
-u_{h}^{E,\mathrm{recipient}}
+u_{h}^{E,\mathrm{donor}}.
\]

The intervention succeeds if the future-query log-odds move from the recipient
answer toward the donor answer, while matched random-head and matched-token
patches do not. The implemented intervention patches the source-specific
pre-\(W_O\) component. QK-only and V-only interventions remain follow-up
ablations and are not required for the first causal gate.

## Natural-reasoning transfer

ProcessBench is not used to discover the geometry. It is a transfer test after
all controlled gates pass. The transferable mechanism score is the mismatch
between output commitment and evidence-supported residual writes, conditioned
on logits history, length, and position. Generic curvature/spread features are
reported only as baselines.

## Claim gates

### Gate A: exact data

- current branch distributions equal to numerical tolerance;
- future branch distributions differ and are deterministic;
- equal branch support size and information gain;
- no pair leakage across folds.

### Gate B: representation

- current model-output aliasing is empirically adequate;
- hidden chart predicts future query better than current-output features;
- shuffled correspondence and random subspace fail;
- result replicates across templates and at least two model families or sizes.

### Gate C: routing

- evidence-source OV writes align with the true Fourier update more than all
  matched controls;
- effect is localized to a sparse, reproducible set of layer-head paths.

### Gate D: causality

- donor evidence patches move future logits toward the donor answer;
- mediation effect is larger for aligned heads than random heads/tokens;
- patching does not simply increase output entropy or norm.

### Gate E: transfer

- geometry-mediated evidence mismatch adds held-out usable information over
  logits, length, position, and lexical controls on ProcessBench;
- the increment replicates on GSM8K, MATH, and at least one harder subset.

Failure at one gate blocks stronger downstream claims. In particular, a good
linear probe alone does not establish a belief-update mechanism.
