from __future__ import annotations

from dataclasses import replace

import numpy as np

from functional_divergence.hidden_state_geometry.config import RawFunctionalConfig
from functional_divergence.hidden_state_geometry.contracts import ChainSample
from functional_divergence.hidden_state_geometry.method import FoldInput
from functional_divergence.hidden_state_geometry.methods import load_builtin_methods
from functional_divergence.hidden_state_geometry.registry import create_method
from functional_divergence.hidden_state_geometry.tasks import build_whole_chain_task
from functional_divergence.progress import RecordingProgress


def _samples(tmp_path, count: int, *, identical_hidden: bool) -> tuple[ChainSample, ...]:
    samples = []
    for row in range(count):
        label = row % 2
        states = np.empty((4, 2, 6), dtype=np.float32)
        for step in range(4):
            for layer in range(2):
                states[step, layer] = step + layer + np.arange(6) * 0.1
        if not identical_hidden and label:
            states[-1, 1, 0] += 8.0
        path = tmp_path / f"chain_{row}.npy"
        np.save(path, states)
        samples.append(
            ChainSample(
                chain_id=row,
                manifest_row=row,
                problem_group=str(row),
                dataset="train" if row < count - 8 else "test",
                generator="llama",
                observer_model="llama",
                state_path=path,
                state_count=4,
                response_start=10,
                step_ranges=np.asarray([[10 + i, 10 + i] for i in range(4)]),
                layer_ids=np.asarray([8, 10]),
                output_steps=np.column_stack(
                    [np.arange(4) + row * 0.01, np.arange(4)[::-1] - row * 0.01]
                ).astype(np.float32),
                output_feature_names=("entropy", "nll"),
                first_error_step=2 if label else -1,
            )
        )
    return tuple(samples)


def _fold(tmp_path, *, identical_hidden: bool) -> FoldInput:
    task = build_whole_chain_task(_samples(tmp_path, 32, identical_hidden=identical_hidden))
    train = np.arange(24)
    test = np.arange(24, 32)
    return FoldInput(
        task_name=task.name,
        train_examples=tuple(task.examples[i] for i in train),
        train_labels=task.labels[train],
        train_groups=task.groups[train],
        test_examples=tuple(task.examples[i] for i in test),
        seed=13,
    )


def test_raw_functional_method_is_a_plugin_and_returns_common_arms(tmp_path):
    load_builtin_methods()
    method = create_method(
        "raw_functional_probe",
        RawFunctionalConfig(
            pca_dim=2,
            time_basis=2,
            layer_basis=2,
            positions_per_chain=4,
            restarts=2,
            max_iter=200,
            null_repeats=2,
        ),
    )

    result = method.fit_predict(_fold(tmp_path, identical_hidden=False))

    assert set(result.probabilities) == {
        "nuisance",
        "output_only",
        "hidden_only",
        "output_plus_hidden",
        "output_plus_time_null_r0",
        "output_plus_layer_null_r0",
        "output_plus_time_null_r1",
        "output_plus_layer_null_r1",
    }
    assert all(values.shape == (8,) for values in result.probabilities.values())
    assert all(np.isfinite(values).all() for values in result.probabilities.values())
    assert result.diagnostics["projection_dim"] == 2
    assert "output_plus_hidden" in result.factors


def test_constant_hidden_has_exactly_no_increment_over_matching_static_arm(tmp_path):
    load_builtin_methods()
    method = create_method(
        "raw_functional_probe",
        RawFunctionalConfig(
            pca_dim=2,
            time_basis=2,
            layer_basis=2,
            positions_per_chain=4,
            restarts=2,
            max_iter=200,
            null_repeats=1,
        ),
    )

    result = method.fit_predict(_fold(tmp_path, identical_hidden=True))

    assert np.array_equal(
        result.probabilities["output_plus_hidden"],
        result.probabilities["output_only"],
    )
    assert np.array_equal(
        result.probabilities["hidden_only"], result.probabilities["nuisance"]
    )


def test_raw_method_reports_projection_encoding_and_probe_progress(tmp_path):
    load_builtin_methods()
    reporter = RecordingProgress()
    method = create_method(
        "raw_functional_probe",
        RawFunctionalConfig(
            pca_dim=2,
            time_basis=2,
            layer_basis=2,
            positions_per_chain=4,
            restarts=1,
            max_iter=20,
            null_repeats=1,
        ),
    )

    method.fit_predict(
        replace(_fold(tmp_path, identical_hidden=False), progress=reporter)
    )

    stages = [event[1] for event in reporter.events if event[0] == "stage"]
    loops = [event[1] for event in reporter.events if event[0] == "start"]
    assert {"projection", "encode", "fit"}.issubset(stages)
    assert "PCA chains" in loops
    assert "train examples" in loops
    assert "test examples" in loops
