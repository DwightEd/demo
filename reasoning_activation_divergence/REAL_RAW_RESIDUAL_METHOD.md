# Real Llama-3.1-8B residual-stream experiment

## What the experiment tests

The unit of comparison is a matched pair: one Llama-3.1-8B response at its
first annotated reasoning error and one fully-correct Llama-3.1-8B response at
a matched step. For token offsets `[-2,-1,0,1]` and stored post-block depths
`[8,10,12,14,16,18,20,22]`, the experiment asks whether first-error windows
depart more strongly from a control-only local dynamical field than correct
windows do. It separates four signals:

1. radial state change;
2. residual from the learned depth transport;
3. residual from the learned token-time transport;
4. disagreement between depth-then-time and time-then-depth paths (a
   plaquette/non-commutativity score).

The operators are empirical cross-fitted affine maps on stored residual states.
They are not autograd Jacobians and the output is not a model-native Fisher.

## Method

1. Filter manifest rows by the normalized response-generator name
   `llama3.1-8b`; every row-aligned label, range and shard path is filtered by
   the same mask.
2. Label `gold_error_step >= 0` as first-error and `< 0` as fully correct.
3. Match error and control events using relative step position, step token
   length and chain step count, with a penalty for leaving the same problem
   when a within-problem control exists.
4. Load only the small event window from each mmap shard, producing
   `[sample,time,layer,4096]` without materializing the full response.
5. Keep connected pairs/problems in one cross-validation fold. In each fold,
   use training controls only to fit a shared centered/scaled randomized-SVD
   coordinate system and ridge affine depth/time maps.
6. Score held-out error and control samples. Aggregate with matched-pair AUROC,
   paired bootstrap confidence intervals, sign-flip tests and AUROC deltas.
7. Report operator eigenphase, proper polar rotation, reflection, spectral
   radius, condition number, singular effective rank and Henrici
   non-normality. Since stored depths differ by two blocks, depth operators are
   sparse depth-interval maps, not single-block maps.

## Input and output files

- `selected/trace.raw_residual_stream.npz`: row-aligned labels, problem ids,
  step/token ranges, response generator, shard paths, stored layer ids/counts,
  and audited snapshot provenance.
- `selected/trace.response_states.*/row_*.npy`: real response-token residual
  states with shape `[response_token,8,4096]`.
- `results.json`: configuration, evidence boundary, dataset diagnostics,
  spectral cells and paired statistics.
- `pair_scores.csv`: error/control/difference values for every matched pair.
- `metric_comparison.png`: paired-AUROC comparison with bootstrap intervals.

## Function map for the real-data call path

### `run_raw_remote.sh`

- `run_raw`: invokes the Python module with the repository `PYTHONPATH`.
- `exact-pilot`: uses the verified **full** manifests, filters Llama-3.1-8B
  responses, then analyzes at most 20 already matched pairs per subset.
- `exact-full`: performs the same analysis without the pair cap. Both modes
  fail if the audited manifest is absent.

### `raw_residual_experiment.py`

- `run_raw_residual_experiment`: orchestrates loading, cross-fitting,
  statistics, comparisons and the three output artifacts.
- `_parse_ints`: validates comma-separated token offsets/layer ids.
- `main`: defines the CLI, runs provenance preflight, or starts the experiment.

### `raw_residual.py`

- `_RawSource`: validated, optionally generator-filtered manifest view.
- `_Match`: one error row/event paired with one control row/event.
- `_scalar`: reads an NPZ scalar while preserving non-scalars.
- `_ranges`: validates one record's `[step,2]` token ranges.
- `_normalized_model_name`: removes punctuation/case differences for model-name
  matching.
- `_response_generators`: reads response-generator provenance from the
  top-level manifest or per-row metadata.
- `_resolve_source`: validates the manifest, applies one row mask to all
  aligned arrays, resolves shard paths and rejects unverified snapshots.
- `_load_shard`: mmap-loads one shard and checks token/layer/count shape.
- `inspect_raw_residual_source`: reports post-filter class counts, generator
  values, layers and the first real shard shape without running analysis.
- `_match_events`: constructs the matching cost, solves the one-to-one Hungarian
  assignment, then reuses the nearest control only for excess error rows.
- `_layer_positions`: maps requested physical layer ids to shard columns.
- `_pair_components`: unions pairs sharing a row or problem, preventing fold
  leakage.
- `load_matched_raw_residual`: slices event windows, drops invalid boundaries
  and returns the dense time-layer dataset plus provenance diagnostics.

### `layer_time.py`

- `LayerTimeDataset`: carries `[sample,time,layer,feature]`, labels, pair/fold
  groups, row ids and metadata.
- `AffineMap.apply`: applies `xA+b`.
- `operator_spectral_metrics`: computes eigenphase, polar rotation/reflection,
  spectral radius, conditioning, effective rank and non-normality.
- `affine_plaquette_discrepancy`: compares the two linear operator compositions
  around one time-layer cell.
- `project_jvp_operator`: projects externally computed JVP columns; available
  for a future model-native Jacobian experiment and unused here.
- `_fit_shared_projection`: fits the fold-specific control-only shared
  randomized-SVD basis and coordinate scales.
- `_project_states`: applies that shared coordinate gauge.
- `_fit_affine`: fits one sklearn ridge affine map.
- `_fit_operator_field`: fits every adjacent stored-depth and token-time map.
- `_mean_norm`: averages edge-wise per-sample scores.
- `_score_operator_field`: computes radial, depth residual, time residual and
  observed plaquette scores on held-out samples.
- `_field_diagnostics`: aggregates per-cell spectral and linear plaquette
  diagnostics.
- `crossfit_layer_time_scores`: performs component-grouped folds, trains only
  on controls, scores held-out rows and verifies complete assignment/no leakage.
- `_event_window` and `load_matched_layer_time_geometry`: load the older derived
  geometry-proxy format; they are not used by this raw-residual experiment.

### `core.py`

- `connected_component_folds`: assigns whole connected components to folds;
  this is used by the real-data path.
- `paired_summary`: computes paired AUROC, bootstrap intervals, mean paired
  difference and a sign-flip p-value; this is used.
- `_paired_wins` and `paired_auc_difference`: compare a candidate metric's
  paired AUROC with a baseline; these are used.
- `MatchedDataset`, `_component_ids`, `_event_vector`, `load_matched_geometry`,
  `_scale_fit` and `crossfit_transport_fisher`: implement the older flattened
  geometry/task-probe experiment and are not called here.
- `categorical_pullback_fisher_energy`: evaluates a categorical Fisher quadratic
  form from supplied output JVPs; it is not called because stored activations do
  not contain model-native JVPs.

### `layer_time_experiment.py`

- `_pair_rows`: converts matched scores to auditable CSV rows; used here.
- `_write_plot`: plots paired AUROC and confidence intervals; used here.
- `run_layer_time_experiment` and `main`: run the older derived-geometry CLI;
  not used for raw shards.

### `output.py`

- `versioned_paths`: preserves an existing result under a timestamp before
  writing the new latest artifact.
