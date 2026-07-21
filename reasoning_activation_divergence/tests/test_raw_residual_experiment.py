from __future__ import annotations

import json

import numpy as np

from functional_divergence.raw_residual_experiment import run_raw_residual_experiment
from functional_divergence.progress import RecordingProgress


def _write_canonical_fixture(root, n_pairs: int = 6):
    hidden = root / "hidden"
    hidden.mkdir()
    n_records = n_pairs * 2
    ranges = np.empty(n_records, dtype=object)
    files = []
    gold = []
    for row in range(n_records):
        ranges[row] = np.asarray([[20, 21], [22, 24], [25, 27]])
        states = np.empty((8, 3, 16), dtype=np.float32)
        base = np.arange(16, dtype=np.float32) * 0.01 + row
        for token in range(8):
            for layer in range(3):
                states[token, layer] = base + token * 0.1 + layer * 0.2
        if row % 2 == 0:
            states[2, 1] += np.linspace(-2.0, 2.0, 16)
            gold.append(1)
        else:
            gold.append(-1)
        name = f"gsm8k-{row}.npy"
        np.save(hidden / name, states)
        files.append(name)
    manifest = root / "full_gsm8k.npz"
    np.savez(
        manifest,
        gold_error_step=np.asarray(gold),
        problem_ids=np.arange(n_records),
        step_token_ranges=ranges,
        hidden_files=np.asarray(files, dtype=object),
        hidden_layers=np.asarray([10, 14, 18]),
        hidden_stored=np.asarray(True),
    )
    return manifest, hidden


def test_raw_experiment_runs_high_dimensional_hidden_states_and_writes_outputs(tmp_path) -> None:
    # Ten pairs leave five train controls per two-fold split, so rank four is
    # identifiable after the n_control - 1 safety cap.
    manifest, hidden = _write_canonical_fixture(tmp_path, n_pairs=10)
    output = tmp_path / "raw_results"

    progress = RecordingProgress()
    result = run_raw_residual_experiment(
        input_path=manifest,
        hidden_dir=hidden,
        output_dir=output,
        offsets=(-1, 0, 1),
        rank=4,
        n_splits=2,
        n_boot=50,
        ridge_alpha=1e-3,
        seed=7,
        progress=progress,
    )

    assert result["schema_version"] == "raw_residual_layer_time_operator_v1"
    assert result["dataset"]["hidden_dim"] == 16
    assert result["dataset"]["representation_scope"] == "raw_residual_stream"
    assert result["dataset"]["diagnostics"]["projection_rank"] == 4
    assert (output / "results.json").is_file()
    assert (output / "pair_scores.csv").is_file()
    saved = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert saved["dataset"]["source_format"] == "canonical_full_hidden_shards_v1"
    stages = [event[1] for event in progress.events if event[0] == "stage"]
    assert stages == ["load", "analyze", "statistics", "write"]
    loops = [event[1] for event in progress.events if event[0] == "start"]
    assert "matched pairs" in loops
    assert "cross-validation folds" in loops
