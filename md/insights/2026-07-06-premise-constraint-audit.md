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

## Remote GSM8K Result

Remote runs on 2026-07-06 do **not** support the lightweight regex arithmetic
ledger as a useful next detector.

`gsm8k_v2_custom.npz`, `answer_format_ok`:

- samples: 3452; errors: 532; contrastive problems: 147;
- text coverage: 3600 samples with parsed steps, 2903 samples with equations,
  9110 equation-bearing steps out of 32645 total steps;
- best constraint: `constraint_tainted_dep_sum`, within AUROC 0.562, cross
  AUROC 0.704;
- best saved baseline: `baseline_cloud_resultant_late_mean`, within AUROC
  0.659, cross AUROC 0.769;
- increment over best baseline: -0.097, bootstrap CI [-0.143, -0.048];
- baseline-miss rescue: 331 / 978 = 0.338.

`gsm8k_v2_5shot.npz`, `answer_format_ok`:

- samples: 2035; errors: 279; contrastive problems: 94;
- text coverage: 2646 samples with parsed steps, 2509 samples with equations,
  7127 equation-bearing steps out of 14754 total steps;
- best constraint: `constraint_invalid_eq_mean`, within AUROC 0.548, cross
  AUROC 0.542;
- best saved baseline: `baseline_cloud_resultant_max`, within AUROC 0.634,
  cross AUROC 0.826;
- increment over best baseline: -0.085, bootstrap CI [-0.159, -0.016];
- baseline-miss rescue: 49 / 341 = 0.144.

Interpretation:

- The current constraint score is not a same-problem detector.  On
  `gsm8k_v2_custom`, its cross AUROC is much higher than within AUROC, which is
  the signature of difficulty/style/length confounding rather than a
  question-conditioned failure mechanism.
- `constraint_tainted_dep_sum` is cumulative and therefore inherits the old
  late-chain/length risk.  It should not be promoted as an online trigger.
- Equation coverage is high enough that the negative result is meaningful for
  this parser family.  The failure is not simply "no equations were available."
- The premise-consistency research direction remains plausible, but the local
  regex arithmetic ledger is too shallow.  GSM8K errors often preserve explicit
  arithmetic syntax while using the wrong operation, wrong semantic binding, or
  wrong problem interpretation.  Those are graph/premise failures, not simple
  equation-evaluation failures.

Decision:

- Keep `premise_constraint_audit.py` as a negative baseline and coverage
  diagnostic.
- Do not spend more time tuning regex arithmetic scores.
- The next non-geometric implementation should test a stronger orthogonal
  channel: same-problem counterfactual step value / operation-template
  consistency / premise graph extraction, not scalar arithmetic invalidity.
