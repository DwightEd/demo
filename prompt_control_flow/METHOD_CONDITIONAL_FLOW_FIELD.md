# Conditional Spherical Feasible-Flow Field

## Status

This is the geometry-only validation that replaces the failed single
low-rank tangent model. It uses the existing same-problem multisample hidden
states and requires no new teacher-forcing extraction. Logits remain a later
stage and are blocked unless the geometric distribution passes both gates.

## Hypothesis

For problem \(q\), causal transition \(t\), and layer \(\ell\), correct
reasoning updates need not occupy one linear tangent. They may form a
multimodal distribution on the unit sphere:

\[
u_{i,t}^{(\ell)}=
\frac{h_{i,t+1}^{(\ell)}-h_{i,t}^{(\ell)}}
{\left\|h_{i,t+1}^{(\ell)}-h_{i,t}^{(\ell)}\right\|_2},
\qquad
u_{i,t}^{(\ell)}\sim\mu_{q,t,\ell}.
\]

Errors are hypothesized to create persistent low-density excursions from this
conditional feasible-flow distribution.

## Why This Is Not Another Scalar Heuristic

The model object is the complete empirical donor distribution
\(\mu_{q,t,\ell}\), not a mean, PCA rank, or fitted correctness direction.
Given chordal distance \(d_{\mathbb S}(u,v)=\lVert u-v\rVert_2\) on the unit
sphere, a target update \(u\),
and \(K\) healthy references \(v_1,\ldots,v_K\), the transition score is the
energy proper score

\[
\mathcal E(u;\mu)=
\frac1K\sum_j d_{\mathbb S}(u,v_j)
-\frac{1}{2K(K-1)}\sum_{j\ne k}d_{\mathbb S}(v_j,v_k).
\]

The second term corrects for the spread of the predictive distribution. The
score has no kernel bandwidth, uses a negative-type Euclidean metric required
by the energy scoring rule, and is invariant to a global orthogonal change of
hidden-state coordinates. It is calibrated against leave-one-out energy
scores of the same healthy donors using a robust median/MAD scale.

## Matched Donor Protocol

Every target from the same problem receives

\[
K_q=\min(K_{\max},N_q^{\mathrm{correct}}-1)
\]

healthy donors. Incorrect targets are downsampled to the same \(K_q\) used by
correct leave-one-out targets. Consequently, correctness cannot alter donor
count, Gram rank, or calibration variance.

The preregistered field uses exact causal-step alignment. A separate
state-conditioned variant searches only within a fixed causal window and is
reported as a secondary subgate. It cannot silently replace a failed phase
field. Time-shuffled and control-matched wrong-problem fields use the same
donor count.

## Dynamic Response Scores

The audit retains the transition sequence and reports:

- mean and late energy score;
- locally calibrated mean and late score;
- normalized risk-sensitive free energy;
- positive anomaly area;
- drift-corrected CUSUM.

The preregistered response score is calibrated free energy after cross-fitted
step-count and character-length residualization:

\[
F_i=\frac1\beta\log\left(
\frac1{T_i}\sum_t e^{\beta z_{i,t}}
\right).
\]

This smooth maximum preserves strong local events without allowing response
length to increase the score merely by adding more terms.

## Gates

1. On held-out correct responses, the same-problem phase field must beat both
   time-shuffled and matched wrong-problem fields with adequate coverage.
2. The length-residualized response score must separate wrong from correct
   responses within the same problem and beat the response scores induced by
   both structural null fields.

Only after both gates pass should the same trajectories be replayed to obtain
output Fisher/JVP quantities such as

\[
e_t^\top J_t^\top
\left(\operatorname{diag}(p_t)-p_tp_t^\top\right)J_t e_t.
\]

## Run

```bash
cd /gz-data/research/demo
python audit_conditional_flow_field.py \
  --input data/gsm8k_v2_custom.npz \
  --output_dir outputs/conditional_flow_field/gsm8k_custom_l16 \
  --vector_key sv_vec_step_exp \
  --layers 16 \
  --label_policy answer_format_ok \
  --min_donors 6 \
  --max_donors 11 \
  --state_window 2 \
  --wrong_problem_draws 3 \
  --device cuda \
  --bootstrap 2000 \
  --permutations 1000
```

Run the same command with `--preflight` first. The compact outputs are
`conditional_flow_field_scores.npz`, `chain_scores.csv`, `summary.json`, and
`summary.md`.
