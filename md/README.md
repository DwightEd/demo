# Markdown Index

The canonical repository-wide evidence ledger is
[`../EXPERIMENT_LEDGER.md`](../EXPERIMENT_LEDGER.md). It supersedes optimistic
historical interpretations when later same-problem or frozen cross-dataset
controls disagree.

This folder keeps project documentation grouped by role.

## `progress/`

Plans, paper-progress notes, implementation reports, and stage summaries.

- `AAAI27_PAPER_PROGRESS_REPORT.md` - current paper-facing progress summary.
- `IMPLEMENTATION_ANALYSIS.md` - implementation-level analysis.
- `README_ANALYSIS.md` - analysis overview.
- `SMCD_SUMMARY.md` - summary of the SMCD/geometry line.
- `results_summary.md` - older result summary.
- `plans/` - dated plans and evidence gates.

## `insights/`

Research findings, literature notes, falsification records, and hypothesis/insight documents.

- `FINDINGS.md` - main audit and negative/positive findings.
- `SPECTRAL_GEOMETRY_ANCHORS.md` - spectral/anchor/transport research notes.
- `MULTISAMPLE_DATA_AND_METHOD_NOTES.md` - same-problem multisample data and method notes.
- `DYNAMICS_README.md` - dynamic-vs-static analysis notes.
- `TRAJECTORY_DESIGN.md` and `TRAJECTORY_README.md` - trajectory-phase-transition branch notes.
- `2026-07-06-pluggable-llm-monitor-research-plan.md` - literature-derived monitor/intervention plan.
- `2026-07-06-localized-trajectory-difference-methods.md` - localized trajectory-difference methods for same-problem samples.
- `2026-07-06-within-problem-functional-shape-model.md` - same-problem functional modeling plan after the full-data dynamic negative result.

## `guides/`

Operational guides, data/cache format references, and package-specific usage notes.

- `PROJECT_README.md` - current whole-layer layer-time geometry entry, runnable gates, and GPU handoff.
- `DATA.md` - dataset/cache label conventions and current data warnings.
- `FEATURES.md` - feature reference.
- `EXPLAIN_DATA_STRUCTURE.md` - data structure explanation.
- `STEP_AND_CACHE_FORMAT.md` - step/cache format guide.
- `NTS_README.md` - original `nts/README.md`.
- `PROJECT_WORKFLOW_MEMORY.md` - local/remote workflow and result-loop rules.

## Code Organization Notes

Python files are not moved in this cleanup because many are root-level experiment entrypoints and may rely on running from the demo root.  A safe future cleanup should package or wrap them before moving.

Suggested code groups:

- Extraction/data: `extract_features.py`, `10_sample_and_extract.py`, `data_loading*.py`, `hidden_io.py`, `npz_to_json.py`, `responses_to_json.py`.
- Main audits: `mainline_validation_suite.py`, `chain_dynamics_audit.py`, `trajectory_relative_audit.py`, `within_problem_trajectory_audit.py`, `within_problem_regime_hsmm_audit.py`, `within_problem_path_kernel_audit.py`, `validate_phase_instability.py`.
- Multisample audits: `multisample_*_audit.py`, `multisample_feature_distribution.py`, `multisample_to_json.py`.
- Geometry/spectral experiments: `trajectory_geometry.py`, `phase_transition.py`, `spectral_*`, `dir_*`, `kappa_*`, `step_*`, `seq_gram.py`, `step_gram.py`.
- Online/intervention prototypes: `online_intervene.py`, `intervene_prototype.py`, `intervene_bestofn.py`, `select_bestof.py`, `selfconsistency_cusum.py`, `conformal_cusum.py`.
- Heavy branches: `anchorflow/`, `nts/`, `hypergraph_*`, `latent_constraint_em_audit.py`.
- Diagnostics/checks: `check_*`, `diagnose_*`, `inspect_*`, `precheck_wcz.py`.
