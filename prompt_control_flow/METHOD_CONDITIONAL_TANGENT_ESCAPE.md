# Question-Conditioned Feasible-Tangent Escape

## One-Sentence Claim

Correct reasoning updates should mostly lie in a low-rank transition space that
is feasible for the current problem and reasoning phase. A consequential error
is not merely a long, curved, or diffuse step: it is a persistent escape into
the normal bundle whose normal component also overlaps a downstream
logit-sensitive cotangent.

This is a falsifiable hypothesis, not a conclusion from the current data.

## Why This Object

Earlier token-cloud concentration, direction-consistency, and spectral scores
describe a trajectory intrinsically. They can report that a step is diffuse or
high-rank, but not whether its update violates the constraints of the problem.
They are also exposed to finite-sample and step-length effects.

The new object is relational. It asks whether an observed transition is
compatible with transitions made by held-out correct traces for similar
questions at a similar reasoning phase. PCA/SVD is only the local estimator;
it is not the contribution.

## State and Transition

For chain (i), step (t), and layer \(\ell\), let the stored step state be
\(z_{i,t}^{(\ell)}\) and let \(q_i^{(\ell)}\) be the stored question state.
The first and subsequent updates are

\[
\Delta z_{i,0}^{(\ell)} = z_{i,0}^{(\ell)}-q_i^{(\ell)},
\qquad
\Delta z_{i,t}^{(\ell)} = z_{i,t}^{(\ell)}-z_{i,t-1}^{(\ell)}.
\]

The tangent audit uses the unit update

\[
u_{i,t}^{(\ell)} =
\frac{\Delta z_{i,t}^{(\ell)}}
{\|\Delta z_{i,t}^{(\ell)}\|_2+\varepsilon},
\]

while retaining update speed as a nuisance/control variable. This separation
prevents a long step from becoming a large escape merely because its state
increment has a larger norm.

## Cross-Fitted Conditional Transition Space

All reference transitions come from training-fold chains only. The default
mainline uses every step from fully correct chains and excludes all error-chain
prefixes. The latter can be enabled only as an explicit ablation.

For a held-out target \((i,t)\), reference update \((j,s)\) receives weight

\[
w_{j,s\mid i,t}\propto
\exp\left(
\frac{\cos(q_i^{(\ell)},q_j^{(\ell)})}{T_q}
-\frac{(\tau_{i,t}-\tau_{j,s})^2}{2\sigma_\tau^2}
\right),
\]

The default online-safe clock is

\[
\tau_t=\frac{t}{t+s_0},
\]

which depends only on the current step index. Whole-chain normalized phase is
available only as an offline ablation because it leaks the future number of
steps. The weighted second moment of the nearest reference directions is
decomposed on GPU. Its leading \(r\)
eigenvectors form the feasible transition space
\(T_{q_i,\tau_{i,t}}^{(\ell)}\).

The primary instantaneous score is

\[
E_{\perp,i,t}^{(\ell)} =
\left\|
\left(I-P_{T_{q_i,\tau_{i,t}}^{(\ell)}}\right)
u_{i,t}^{(\ell)}
\right\|_2^2.
\]

Matched-rank random, global, phase-only, and shuffled-question spaces are
computed from the same training fold. These are structural nulls, not optional
visualizations.

## Persistent Normal Escape

Let

\[
e_{i,t}^{(\ell)}=
\left(I-P_{T_{q_i,\tau_{i,t}}^{(\ell)}}\right)u_{i,t}^{(\ell)}.
\]

For a causal window \(W_t\), the implementation records normal energy and
coherent drift:

\[
N_t^{(\ell)} = \frac{1}{|W_t|}\sum_{s\in W_t}\|e_s^{(\ell)}\|_2^2,
\qquad
C_t^{(\ell)} =
\left\|\frac{1}{|W_t|}\sum_{s\in W_t}e_s^{(\ell)}\right\|_2^2.
\]

Large \(N_t\) alone can be an isolated shock. Large \(C_t\) means that normal
motion keeps a common direction instead of cancelling. The persistence claim
passes only if \(C_t\) improves problem-grouped out-of-fold response diagnosis
over a baseline that already contains position, length, and instantaneous
\(E_{\perp,t}\). Comparing persistence only with length controls is not
sufficient.

## Output-Sensitive Coupling

The strong mechanism claim additionally needs an exact downstream cotangent
\(g_{i,t}^{(\ell)}\), such as the gradient of a teacher-forced logit margin or a
pullback Fisher direction with respect to the same step-layer state. The
transverse coupling is

\[
O_{i,t}^{(\ell)} =
\left\langle
\frac{g_{i,t}^{(\ell)}}{\|g_{i,t}^{(\ell)}\|_2},
e_{i,t}^{(\ell)}
\right\rangle^2.
\]

Because (O_t) mixes escape magnitude and directional alignment, the audit
also reports

\[
A_{\perp\to y,t}^{(\ell)} =
\frac{\langle \hat g_t^{(\ell)},e_t^{(\ell)}\rangle^2}
{\|e_t^{(\ell)}\|_2^2+\varepsilon},
\qquad
A_{\parallel\to y,t}^{(\ell)} =
\frac{\langle \hat g_t^{(\ell)},P_Tu_t^{(\ell)}\rangle^2}
{\|P_Tu_t^{(\ell)}\|_2^2+\varepsilon}.
\]

Thus (E_\perp) asks whether the update escaped, while
(A_{\perp\to y}) asks whether the escape points along a direction that can
change the downstream output. Gate 4 tests (A_{\perp\to y}) after already
including (E_\perp) in the baseline; a large escape alone cannot pass it.

Entropy, NLL, gradient norm, or raw unembedding overlap cannot substitute for
this quantity. Without a stored exact cotangent, the report marks this gate
`not_tested_missing_cotangent`.

Accepted arrays are record-aligned object arrays with one
`[step, layer, hidden]` tensor per chain:

```text
step_output_cotangent
step_output_cotangent_layers
step_output_cotangent_kind = exact_downstream_cotangent
```

`step_hidden_grad` and `output_cotangent` are supported aliases. The current
canonical `full_*.npz` artifacts do not contain these vectors, so this gate
requires a model replay or a separately merged cotangent artifact.

## Four Claim Gates

1. **Conditioning:** on held-out correct chains, the question-and-phase space
   reconstructs transitions better than both shuffled-question and phase-only
   spaces. Each chain contributes one equal-weight mean before bootstrap.
2. **Escape:** nuisance-residualized \(E_\perp\) rises at the first error against
   matched correct pseudo-events. Post-error steps are excluded from the
   first-error classifier. Layer selection is corrected across tested layers.
3. **Persistence:** coherent normal drift adds response-level OOF AUROC beyond
   controls plus instantaneous escape, with a problem-cluster bootstrap lower
   bound above zero. Step-level nuisance controls use only the current clock,
   current/previous step length, and cumulative already-seen tokens; future
   step length and final chain length are excluded.
4. **Consequence:** nuisance-residualized normal alignment rises at the matched
   first error, and exact (A_{\perp\to y}) adds response-level OOF signal over
   controls plus instantaneous (E_\perp). If the exact cotangent is absent,
   the full mechanism claim remains untested.

Failure of Gate 1 kills the proposed conditional tangent estimator. Failure of
Gates 2 and 3 means the geometry does not diagnose reasoning errors. Failure or
absence of Gate 4 permits only a correlational geometry claim.

## Length Audit for Earlier Signals

The same run re-evaluates the stored direction/spread family using four views:

1. pre-registered signed AUROC;
2. AUROC within step-length quantile buckets;
3. Spearman correlation with log step length and relative position;
4. problem-grouped OOF improvement over a causal step clock, current/previous
   step length, and cumulative seen-token controls, with a problem-cluster
   bootstrap interval.

Historical project results already show substantial, but not total, length
confounding. On GSM8K, raw spread/resultant AUROC near 0.772 fell to about 0.708
inside length buckets, while the correlation with length was large in
magnitude. Thus the raw number cannot be treated as length-clean, but the
bucket result is not random either. Across harder subsets the length-controlled
AUROC was weaker (roughly 0.57--0.61). Spectral shape was usually redundant;
the notable exception was Omnimath effective rank, which added about 0.013 over
concentration plus length in the recorded audit. These are prior observations,
not results of this new method, and the new report recomputes the controls on
the exact input artifact.

## Novelty Boundary and Closest Collision

The broad claim that a first error is an excursion from a stable hidden-state
transition manifold is **not** novel. The closest known collision is
[Where Does Reasoning Break?](https://arxiv.org/abs/2605.13772), which uses
step labels to learn a global contrastive-PCA lens separating first-error from
correct states, then feeds seven position/velocity/acceleration features to an
MLP teacher and distils a BiLSTM student. Its geometric teacher is explicitly
label-conditioned and its deployable student is post-hoc and bidirectional.

Accordingly, this project must not claim novelty for PCA, a correct-state
manifold, transition distance, or first-error excursion by themselves. The
candidate increment is the conjunction of:

1. a **question- and causal-time-conditioned transition tangent**, estimated
   without using first-error examples to choose its directions;
2. a **vector-valued persistent normal drift**, tested beyond instantaneous
   escape rather than another scalar smoothness feature; and
3. an **exact output-cotangent alignment test** that separates harmless
   off-manifold motion from motion capable of changing logits.

This boundary is intentionally strict. Recent trajectory work supports
step-dependent reasoning geometry
([LLM Reasoning as Trajectories](https://arxiv.org/abs/2604.05655)) and
low-rank shared directions
([Invariant Reasoning Directions](https://arxiv.org/abs/2606.29164)), while
pullback-Fisher work motivates measuring output-induced rather than purely
Euclidean geometry
([FishBack](https://arxiv.org/abs/2605.17231)). None of those observations
guarantees that the three gates above will pass on ProcessBench.

Layer selection remains an exploratory multiple-comparison problem. The report
applies Benjamini-Hochberg correction across tested layers for matched events;
the final paper claim still requires a frozen layer rule or replication on a
held-out dataset/model. Response controls use complete-chain length summaries
and therefore describe post-hoc response diagnosis, whereas the step-level
mainline uses only causally available controls.

## Existing-Data Run

The first three gates and the legacy length audit use the existing canonical
artifact; no hidden-state extraction is required:

```bash
python audit_conditional_tangent_escape.py \
  --input data/features/full_gsm8k.npz \
  --output_dir outputs/conditional_tangent/full_gsm8k \
  --layers 8,10,12,14,16,18,20,22 \
  --phase_mode causal_step \
  --device cuda \
  --bootstrap 2000
```

Run the preflight first when changing datasets:

```bash
python audit_conditional_tangent_escape.py \
  --input data/features/full_gsm8k.npz \
  --layers 8,10,12,14,16,18,20,22 \
  --preflight
```

Main outputs:

```text
conditional_tangent_summary.md
conditional_tangent_summary.json
step_length_audit.csv
first_error_event_curves.csv
response_diagnosis.csv
first_error_ranks.csv
event_*.png
conditional_tangent_scores.npz
```

All local tangent decompositions, fixed-basis projections, and legacy token
spectra are batched on CUDA when available. Cross-validation and bootstrap
bookkeeping remain on CPU because their matrices are small.
