# Predictive State Geometry Pilot

## 1. Motivation and Fixed Claim

The debiased directional-consensus result established a weak response-level
signal, but its estimator comparisons did not improve over raw spread:

\[
\Delta\operatorname{AUC}_{\text{debiased-raw}}=0,
\qquad
\Delta\operatorname{AUC}_{\text{fixed-raw}}=0.0279
\]

with both confidence intervals covering zero. Changing another static
dispersion estimator is therefore not a justified next step.

This pilot changes the estimand. Its fixed hypothesis is:

> Correct reasoning admits a compact predictive state: after lexical nuisance
> control, the future hidden-state window is more predictable from its current
> hidden-state window than it is for an incorrect response. The useful signal
> is predictive innovation, not static distance or curvature alone.

The pilot does not assume that all correct trajectories share one direction,
one endpoint, or one deterministic future.

## 2. Why This Is Not a VAE

A standard VAE learns coordinates that reconstruct the current hidden state:

\[
\mathcal L_{\mathrm{VAE}}
=
\mathbb E[-\log p_\theta(h\mid z)]
+
\beta D_{\mathrm{KL}}(q_\phi(z\mid h)\Vert p(z)).
\]

That objective can preserve token identity, syntax, response style, and length
because those factors dominate hidden-state variance. This pilot instead keeps
directions only when correct past windows predict correct future windows. The
scientific chart is consequently learned from dynamics rather than
reconstruction. VAE, time-lagged AE, Koopman AE, and nonlinear JEPA-style
encoders are later baselines, not part of Stage A.

## 3. Two Input Tiers

No model forward pass is required. Use:

```text
data/gsm8k_v2_custom.npz
data/gsm8k_v2_5shot.npz
```

Both existing files provide the legacy state-only arrays:

```text
sv_clouds                  token hidden states, response -> (N, L, D)
cloud_sizes                stored token count for each kept semantic step
cloud_layers               actual cloud layer IDs
problem_ids, sample_idx
is_correct, format_ok
responses
```

These files predate exact generation-trace storage. Their preflight mode is
`legacy_cloud_order`: token-cloud order is preserved, but model token IDs and
absolute omitted-token gaps are unavailable. They support an exploratory
state-only test with global correct-only centering. They do not support the
lexical quotient in Section 5 and cannot pass the full confirmatory gate.

New artifacts produced by the current extractor additionally provide:

```text
input_ids                  exact teacher-forced model input IDs
time_axis_token_ranges     inclusive absolute ranges used to build the cloud
```

For this `exact_trace` tier, the loader reconstructs cloud token IDs as

\[
\operatorname{ids}_{cloud}
=
\operatorname{concat}_{j}
\operatorname{input\_ids}[a_j:b_j+1].
\]

Every range length must equal the corresponding `cloud_sizes` entry, and the
concatenation length must equal `sv_clouds.shape[0]`. The program fails rather
than re-tokenizing response text or silently dropping a mismatched sample.
The exact tier is required to confirm lexical-independent predictive geometry.

## 4. Label-Free Computational Sketch

For selected layer \(l\), hidden dimension \(D\), and fixed random matrix
\(P_l\in\mathbb R^{D\times d}\), compute

\[
\tilde h_i^{(l)}=h_i^{(l)}P_l,
\qquad
(P_l)_{ab}\sim\mathcal N(0,1/d).
\]

The matrix is fixed by `--seed` and never uses correctness labels. It is only a
GPU-efficient Johnson-Lindenstrauss-style preconditioner. It is not the learned
latent manifold and must not be credited with detector supervision.

## 5. Cross-Fitted Lexical Quotient

Within each problem-group fold, use only correct training responses. Estimate
the response-equal token conditional mean

\[
\mu_v
=
\frac{\sum_{r}\frac{1}{N_r}\sum_{i\in r}
\mathbf 1[token_i=v]\tilde h_i}
{\sum_{r}\frac{1}{N_r}\sum_{i\in r}\mathbf 1[token_i=v]}.
\]

Tokens observed fewer than `min_token_count` times use the global correct-state
mean. The primary channel is

\[
x_i=\operatorname{standardize}_{train,correct}(\tilde h_i-\mu_{token_i}).
\]

An otherwise identical raw channel omits token-ID subtraction. This makes the
increment from lexical quotienting directly testable. On legacy artifacts,
only this raw channel is available; the implementation does not invent token
IDs by re-tokenizing decoded text.

## 6. Fixed Token Windows

Use fixed windows of \(W\) stored tokens, independent of semantic-step length.
A window is excluded if omitted input tokens make its absolute span exceed
`W + max_skipped_tokens`. The observation is

\[
o_t=
\left[
\frac{1}{W}\sum_{i\in t}x_i,
x_{t,end},
x_{t,end}-x_{t,start}
\right].
\]

No absolute token position, relative response position, number of steps, or
response length is supplied to the predictor. Those variables appear only in
post-hoc nuisance controls. A transition is also excluded when any adjacent
window on its context-to-target path crosses more than `max_skipped_tokens`
unstored input tokens; a long omitted text span is never treated as one-step
dynamics.

## 7. Correct-Only Predictive Chart

For each horizon \(k\), fit response-equal ridge dynamics on correct training
responses:

\[
B_k
=
\arg\min_B
\sum_r\frac{1}{M_r}
\sum_t\lVert o_{t+k}-c_tB\rVert_2^2
+\lambda\lVert B\rVert_F^2,
\]

where \(c_t\) concatenates the previous `context_windows` observations. Let
\(\widehat Y=XB_k\). The top \(d_z\) eigenvectors of
\(\operatorname{Cov}(\widehat Y)\) define \(V_k\). Thus

\[
z_{t+k}=o_{t+k}V_k,
\qquad
\hat z_{t+k}=c_tB_kV_k.
\]

These are directions of future variation that are actually predictable from a
correct prefix. They are not directions of maximum static variance.

## 8. Innovation Geometry

On correct training transitions, estimate residual covariance with shrinkage:

\[
r_t=z_{t+k}-\hat z_{t+k},
\qquad
\widehat\Sigma_k
=(1-\gamma)\operatorname{Cov}(r_t)
+\gamma\frac{\operatorname{tr}(\operatorname{Cov}(r_t))}{d_z}I.
\]

The primary transition score is normalized Mahalanobis innovation:

\[
I_{t,k}
=
\frac{1}{d_z}r_t^\top\widehat\Sigma_k^{-1}r_t.
\]

Residual covariance eigenvectors explaining `tangent_variance` define the
high-variance admissible branch directions. The complementary low-variance
innovation is reported as `transverse_mean`; it is exploratory until it beats
the full Mahalanobis score.

Each horizon is averaged within response and horizons are then equally
averaged. Longer responses do not receive more fitting weight or more weight
in the response score.

## 9. Mandatory Falsification Models

The same ordered target chart is reused for the two chronology nulls:

1. `null.shuffle`: future windows are cyclically permuted within each correct
   training response.
2. `null.same_problem_mismatch`: each correct prefix receives a future from a
   different correct response to the same problem at a matched transition
   index.
3. `static`: target states are scored under a correct-only Gaussian density in
   the same latent chart, without a transition predictor.
4. `token_bigram_nll`: exact-tier-only correct-trained lexical bigram surprise
   over adjacent stored token IDs.
5. `fixed_window_consensus`: the previous debiased directional baseline.

If ordered innovation does not beat these controls, there is no evidence that
temporal predictive state adds information.

## 10. Leakage-Safe Evaluation

All responses from one problem stay in one fold. Projection is fixed globally,
while feature scales, transition models, charts, and covariances are fitted on
correct responses from training problems only. Token means and bigram counts
are additionally fitted in the exact tier only.

The primary report includes:

- same-problem paired AUROC;
- problem-bootstrap confidence intervals;
- within-problem label permutation and BH correction;
- token-length-matched AUROC;
- cross-fitted residualization against step count, response characters,
  stored token count, and valid-window count;
- paired AUROC deltas against every mandatory null.

## 11. Frozen Stage-A Gate

The current legacy files first run a Stage-A0 exploration. They report all
state-dynamics, chronology-null, static, length, and consensus comparisons,
but `exact_lexical_control_available=False` forces the overall decision to
`FAIL`. A weak legacy result rejects the current global predictive-state
hypothesis without spending GPU time on re-extraction. A strong legacy result
only justifies creating exact-trace artifacts.

The full Stage-A1 gate below applies only when preflight reports
`alignment_mode=exact_trace`.

The pilot passes on one dataset only if all conditions hold:

\[
\operatorname{CI}_{low}
(\operatorname{AUC}_{ordered,length-residual})>0.5,
\qquad q<0.05,
\]

\[
\Delta\operatorname{AUC}_{ordered-shuffle}^{length-residual}\ge 0.03,
\quad
\operatorname{CI}_{low}(\Delta)>0,
\]

\[
\Delta\operatorname{AUC}_{ordered-consensus}\ge 0.03,
\quad
\operatorname{CI}_{low}(\Delta)>0.
\]

Every discrimination delta used by the gate is computed after the same
cross-fitted length residualization. The same \(0.03\) increment and
positive-CI rule applies against the same-problem mismatched-future model and
static latent density. Ordered length-residualized innovation must also beat
lexical bigram NLL with a positive delta confidence interval. Raw deltas remain
diagnostics only. The ordered model must have lower held-out innovation than
both chronology nulls on correct responses, and finite score coverage must be
at least 80 percent.

The exact same hyperparameters must pass independently on exact-trace versions
of both `custom` and `5shot` before a nonlinear predictive encoder is
justified. A failed gate retires the current global predictive-state
hypothesis; it must not be rescued with a VAE or a supervised ensemble.

## 12. Commands

Preflight the available alignment tier:

```bash
cd /gz-data/research/demo
python audit_predictive_state.py \
  --input data/gsm8k_v2_custom.npz \
  --preflight
```

Run the legacy state-only GPU audit on the existing artifact:

```bash
python audit_predictive_state.py \
  --input data/gsm8k_v2_custom.npz \
  --output outputs/predictive_state/gsm8k_custom_scores.npz \
  --output_dir outputs/predictive_state/gsm8k_custom_audit \
  --vector_key sv_vec_step_exp \
  --cloud_layers all \
  --label_policy answer_format_ok \
  --projection_dim 96 \
  --window_tokens 16 \
  --window_stride 16 \
  --horizons 1,2 \
  --latent_dim 16 \
  --compute_device cuda \
  --bootstrap 2000 \
  --permutations 2000
```

Replicate without tuning:

```bash
python audit_predictive_state.py \
  --input data/gsm8k_v2_5shot.npz \
  --output outputs/predictive_state/gsm8k_5shot_scores.npz \
  --output_dir outputs/predictive_state/gsm8k_5shot_audit \
  --vector_key sv_vec_step_exp \
  --cloud_layers all \
  --label_policy answer_format_ok \
  --projection_dim 96 \
  --window_tokens 16 \
  --window_stride 16 \
  --horizons 1,2 \
  --latent_dim 16 \
  --compute_device cuda \
  --bootstrap 2000 \
  --permutations 2000
```

## 13. Claim Boundary

Passing the exact-trace gate supports a response-level statement that compact correct-only
predictive dynamics add error discrimination beyond lexical, static,
chronology-null, length, and directional-consensus controls. It does not prove
first-error localization, prompt-conditioned state sufficiency, a global
reasoning manifold, causal error generation, model self-awareness, or
output-logit sensitivity. A legacy state-only result is exploratory: it can
falsify the dynamic hypothesis or motivate exact-trace extraction, but cannot
establish independence from lexical content.
