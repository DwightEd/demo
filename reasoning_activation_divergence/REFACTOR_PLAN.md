# Raw Residual Dynamics Refactor Plan

## 1. Scope and invariants

This refactor covers only the production path that analyzes row-aligned raw
residual-stream shards:

`manifest -> provenance gate -> matching -> event windows -> cross-fit operator field -> paired statistics -> artifacts`

The numerical method and evidence boundary do not change. In particular:

- only manifests declaring `raw_residual_stream` are accepted;
- generator filtering remains row-aligned and fail-closed;
- matching, boundary filtering, component-grouped cross-fitting, sklearn
  randomized SVD/Ridge, spectral diagnostics, and paired inference remain real
  implementations;
- no proxy-data or missing-dependency fallback is introduced.

## 2. Problems in the current layout

- `raw_residual.py` mixes manifest parsing, provenance validation, matching,
  shard I/O, window construction, and metadata serialization.
- `layer_time.py` mixes domain records, numerical kernels, cross-fitting,
  diagnostics, and a legacy derived-geometry loader.
- experiment orchestration passes a long list of scalar arguments and then
  constructs large nested dictionaries inline.
- metadata is an untyped `dict[str, Any]`; callers unpack and repack the entire
  dictionary, which obscures required fields and permits accidental schema
  drift.
- the CLI has no stage/fold progress contract and gives no interpreter or
  dependency provenance before a long run.
- raw-state production code imports reporting helpers from a legacy
  geometry-proxy experiment module.

## 3. Target architecture

```text
functional_divergence/
  config.py       immutable RunConfig and SourceConfig value objects
  domain.py       datasets, provenance, cohort summary, result records
  progress.py     small ProgressReporter interface + tqdm implementation
  source.py       audited NPZ manifest repository and mmap shard reader
  matching.py     first-error/control matching and leakage components
  operators.py    projection, affine fields, spectra, plaquette metrics
  analysis.py     component-grouped cross-fit application service
  statistics.py   paired summaries and paired method comparisons
  reporting.py    JSON/CSV/figure artifact writer
  cli.py          argument adapter and environment preflight
```

Dependencies point inward: CLI/reporting/source are adapters; analysis is the
application service; domain/config/operators/statistics contain the stable
research contracts.

## 4. New method-facing components

### `RunConfig`

Owns offsets, layers, rank, folds, bootstrap count, seed, ridge strength, pair
limit, and generator selector. Validation happens once. The service receives
one object instead of ten loosely related scalars.

### `SourceProvenance` and `CohortSummary`

Replace the metadata blob with explicit immutable records. Provenance answers
what representation was measured; cohort summary answers what rows survived
filtering/matching. They serialize themselves only at the reporting boundary.

### `RawResidualRepository`

Owns manifest schema detection, strict provenance checks, row-aligned generator
selection, resolved shard paths, and memory-mapped shard reads. It exposes
small typed properties instead of passing manifest dictionaries downstream.

### `MatchedWindowBuilder`

Combines a matching strategy with the repository to construct `[sample, time,
layer, hidden]` windows. It reports pair progress and returns a typed dataset;
it does not know about JSON or plots.

### `OperatorFieldAnalyzer`

Runs component-grouped cross-fitting. Each fold fits one control-only shared
coordinate gauge, depth/time affine operators, held-out scores, spectral
rotation/non-normality diagnostics, and plaquette disagreement. Fold progress
is emitted through the reporter interface.

### `ExperimentRunner`

Coordinates load, analyze, paired inference, and artifact writing. This is the
single production use case and replaces the large procedural experiment
function while retaining a compatibility wrapper for existing callers.

### Progress and dependency provenance

The CLI uses required `tqdm` progress bars for dataset, matched-pair loading,
cross-validation folds, statistics, and artifact stages. Before computation it
prints `sys.executable`, active conda environment, and installed sklearn path
and version. A missing required package fails immediately; there is no degraded
implementation.

## 5. Research-method highlights preserved and clarified

1. **Rotation-aware dynamics:** operator eigenphase, proper polar rotation,
   orientation reversal, spectral radius, condition, effective rank, and
   Henrici non-normality complement radial activation change.
2. **Joint token-times-layer geometry:** the plaquette score measures whether
   depth-then-time and time-then-depth transport agree locally.
3. **Leakage-resistant estimation:** projections and affine maps are learned
   only from training controls; reused rows/problem groups stay in one fold.
4. **Auditable cohort provenance:** manifest total, generator-selected records,
   class balance, candidate/retained/dropped pairs, and layer semantics are
   separate typed records rather than an opaque metadata payload.
5. **Operational transparency:** foreground runs show exactly which dataset,
   pair, fold, and reporting stage is active and which Python environment is
   executing it.

## 6. Incremental implementation and acceptance gates

1. Add contract tests for config validation, typed metadata serialization, and
   progress events; confirm they fail against the current implementation.
2. Introduce domain/config/progress components without changing numerical
   outputs; run focused tests.
3. Extract repository and matching/window builder; verify exact row/shard
   alignment and existing raw-loader fixtures.
4. Extract operator analyzer and statistics; verify scores and diagnostics
   against deterministic fixtures.
5. Replace orchestration/reporting/CLI, add environment diagnostics and tqdm;
   verify a full synthetic raw-state experiment.
6. Remove only modules/functions proven unreachable or replaced by the new
   raw-state path; update documentation and remote foreground command.
7. Run the complete subproject test suite, `compileall`, shell syntax check,
   and code review before commit/push.

## 7. Non-goals

- no change to scientific claims before real remote results exist;
- no mixing of non-Llama response generators into the primary cohort;
- no autograd Jacobian/Fisher claim;
- no re-extraction or mutation of remote residual-state data;
- no refactor of sibling projects in the shared repository.
