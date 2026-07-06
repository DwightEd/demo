# Premise Constraint Audit

## Hypothesis

The verified kappa signal captures local representational incoherence inside a
reasoning step, but it should miss coherent-but-wrong chains whose hidden
geometry and entropy remain healthy.  A complementary channel should therefore
test whether each generated step preserves explicit premises and local numeric
constraints.

This audit treats a chain as an online premise ledger:

- seed the ledger with numbers from the question, when present;
- scan each generated step for numeric literals and explicit equations;
- flag invalid equations such as `7 + 5 = 13`;
- flag newly introduced numbers that are not copied or derivable from known
  numbers by a simple arithmetic operation;
- propagate taint when later steps depend on already unsupported numbers.

This is a first local verifier, not a full premise parser.  Low equation
coverage means the text is insufficient for this specific audit, not that the
chain has no premise error.

## Why This Is Not Another Geometry Probe

The score is computed from explicit textual constraints and arithmetic
consistency.  It does not train a classifier on hidden states, does not reuse
second moments/spectra/trajectory drift, and does not assume that wrong chains
must be diffuse.  Its key target is the case where geometry/entropy baselines
rank an incorrect sample as safe.

The main complementarity metric is `baseline_miss_rescue`: among same-problem
error/correct pairs where the strongest saved geometry/entropy/length baseline
fails, count how often the premise-constraint score ranks the error sample as
riskier.

## Implemented Entry Point

```bash
python premise_constraint_audit.py --selftest
```

Selftest constructs low-entropy, low-spread coherent arithmetic errors.  The
expected behavior is that geometry/entropy baselines are weak while the
constraint score rescues the missed same-problem pairs.

Run on remote multisample data:

```bash
cd /gz-data/research/demo
git pull

python premise_constraint_audit.py \
  --input data/gsm8k_v2_custom.npz \
  --policy answer_format_ok \
  --bands mid,deep \
  --bootstrap 1000 \
  --output_dir outputs/premise_constraint_audit_custom

python premise_constraint_audit.py \
  --input data/gsm8k_v2_5shot.npz \
  --policy answer_format_ok \
  --bands mid,deep \
  --bootstrap 1000 \
  --output_dir outputs/premise_constraint_audit_5shot
```

If the ProcessBench full files include `steps_text` and question/prompt fields,
the same script can be run on them, but the first intended target is
same-problem multisampling because it controls question difficulty.

## Readout Gates

- Primary: same-problem paired AUROC for `constraint_*` scores.
- Complementarity: positive bootstrap CI over the best saved baseline.
- Coherent-wrong relevance: high `baseline_miss_rescue` rate.
- Coverage: enough samples with numeric equations.  If equation coverage is
  low, the next step is an LLM/premise parser or a data extraction pass that
  stores proposition-level steps.

## Current Local Verification

Selftest passed on 2026-07-06:

- best constraint: `constraint_risk_max`, within AUROC 1.000;
- best baseline: `baseline_ae_deep_mean`, within AUROC 0.594;
- increment over best baseline: 0.406, bootstrap CI roughly [0.307, 0.513];
- rescued baseline misses: 39 / 39.

Related tests:

```bash
python -m pytest \
  tests/test_premise_constraint_audit.py \
  tests/test_trajectory_difference_audit.py \
  tests/test_second_moment_dynamics_audit.py -q
```

Result: 5 passed.
