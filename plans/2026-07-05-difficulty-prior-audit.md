# Difficulty Prior Audit: D0 and Value Innovation

## Motivation

The paper *The LLM Already Knows: Estimating LLM-Perceived Question Difficulty
via Hidden Representations* suggests that the model's initial hidden state
contains a prior estimate of whether the question is easy or hard for that
model.

For this project, the useful hypothesis is narrower:

```text
D0 is not the main step-error detector.
D0 is a risk baseline that can separate hard-but-correct healthy divergence
from real online reasoning breaks.
```

This directly addresses the current weakness of `spread/resultant` and
`anchor_uncertainty`: correct hard reasoning can also be diffuse, so raw spread
can over-alarm on hard-correct chains.

## Code Added

```text
difficulty_prior_audit.py
```

The script uses only existing `full_*.npz` fields:

```text
qvec                 -> D0 = P(chain has error before generation)
stepcloud/resultant  -> spread
stepvec + qvec       -> anchor_loss / step_direction_jump
tok_U_D              -> U_D_mean
gold_error_step      -> first-error label
```

No hidden shards are required for this first version.

## Method

### 1. Chain-Level Difficulty Prior

Train an OOF logistic probe on unit-normalized `qvec`:

```text
D0_error_prior = P(chain_error | qvec)
```

Also train a qvec-norm-only baseline to check whether D0 is really directional.

### 2. D0-Conditioned Innovation

For each step feature, fit on non-error steps only:

```text
E[feature_t | D0, logN, pos]
```

Then define:

```text
feature_innov_t = feature_t - E[feature_t | D0, logN, pos]
```

Currently implemented for:

```text
spread_innov
anchor_loss_innov
U_D_mean_innov
d_spread_innov
d_anchor_loss_innov
cz_*_innov
geom_value_innov = z(spread_innov) + z(anchor_loss_innov) + z(U_D_mean_innov)
```

### 3. Step-Level OOF Groups

Compare:

```text
controls
D0_only
anchor_uncertainty
anchor_plus_D0
anchor_D0_interactions
innovation_only
hazard_value
```

The important comparison is not D0 alone, but:

```text
anchor_plus_D0 / innovation_only / hazard_value vs anchor_uncertainty
```

### 4. Online Alarm Readout

The script reports OOF threshold alarms:

```text
FPR
hard_correct_FPR
recall
median_delay
early_warn
```

`hard_correct_FPR` is the key D0-specific metric.  If D0 helps, it should reduce
false alarms on correct chains that have high D0 risk.

## Local Selftest

Commands:

```bash
python -m py_compile difficulty_prior_audit.py
python difficulty_prior_audit.py --selftest --folds 3 --n_boot 50 --output_dir outputs/difficulty_prior_selftest
```

Selftest summary:

```text
chains 240 | error chains 93
D0 qvec prior AUROC 0.643 | norm-only AUROC 0.500

anchor_uncertainty AUROC 1.000
hazard_value AUROC 1.000

At eps=0.10:
anchor_uncertainty hardFPR       0.273
anchor_D0_interactions hardFPR   0.182
hazard_value hardFPR             0.205
```

## Result Analysis

The selftest is synthetic and not research evidence.  It verifies the intended
failure mode:

1. D0 can be learned from qvec direction while qvec norm-only is uninformative.
2. Raw `anchor_uncertainty` over-alarms high-D0 correct chains.
3. D0 interactions and innovation features can reduce hard-correct false alarms
   without reducing recall in the synthetic setting.

The real test is whether this happens on GSM8K/MATH/OmniMath.  If not, D0 should
stay a baseline/control rather than become a method claim.

## Real-Data Decision Rules

Promote the branch only if at least one of these holds:

```text
1. anchor_plus_D0 or hazard_value has positive cluster-bootstrap increment over
   anchor_uncertainty on at least two datasets.

2. Global AUROC is flat, but hard_correct_FPR drops at matched eps without a
   recall collapse.

3. Low-entropy confident-wrong recall is preserved or improved.
```

Kill or downgrade the branch if:

```text
D0_only is weak and D0-conditioned groups neither improve AUROC nor reduce
hard_correct_FPR.
```

## Follow-Up Research Direction

If useful, this should evolve into a value/hazard model:

```text
D0 = prior P(final error | prompt hidden)
h_t = P(first error at step t | no earlier error, D0, obs_<=t)
V_t = P(final correctness survives | D0, obs_<=t)
innovation_t = observed badness - expected badness under D0
```

Only after this audit shows utility should we consider TD-style value learning
or HMM/HSMM/CRF variants.

## Optimization Suggestions

1. Keep D0 separate from online break signals in every table.
2. Report hard-correct FPR and matched-FPR recall, not only AUROC.
3. Replace qvec-only D0 with prompt-span hidden / real prompt-anchor vectors
   when those are available.
4. Do not use CRF as the first model; use first-event hazard filtering first.

## Remote GPU Commands

Layer 14 triage:

```bash
cd /gz-data/research/demo
git pull

for d in gsm8k math omnimath; do
  python difficulty_prior_audit.py \
    --dataset $d \
    --data_dir /gz-data/research/demo/data \
    --layer 14 \
    --folds 5 \
    --prior_pca 32 \
    --n_boot 200 \
    --output_dir outputs/difficulty_prior_l14
done
```

Optional layer sweep:

```bash
for d in gsm8k math omnimath; do
  for l in 10 14 18 22; do
    python difficulty_prior_audit.py \
      --dataset $d \
      --data_dir /gz-data/research/demo/data \
      --layer $l \
      --folds 5 \
      --prior_pca 32 \
      --n_boot 200 \
      --output_dir outputs/difficulty_prior_layers
  done
done
```
