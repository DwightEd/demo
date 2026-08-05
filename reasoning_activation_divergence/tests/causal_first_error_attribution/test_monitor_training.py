from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from functional_divergence.causal_first_error_attribution.monitor_data import (
    MonitorBoundaryRow,
    ProcessBenchMonitorData,
    _BoundaryStateStore,
)
from functional_divergence.causal_first_error_attribution.monitor_training import (
    MonitorTrainingConfig,
    build_inner_group_split,
    fit_state_normalizer,
    train_monitor_arm,
)


def _data(tmp_path: Path) -> ProcessBenchMonitorData:
    values = np.arange(18 * 3 * 4, dtype=np.float32).reshape(18, 3, 4) / 20.0
    state_path = tmp_path / "states.npy"
    np.save(state_path, values)
    rows = []
    row_index = 0
    for domain in ("gsm8k", "math", "omnimath"):
        for group in range(3):
            first_error = 1 if group < 2 else -1
            for step in range(2):
                rows.append(
                    MonitorBoundaryRow(
                        chain_id=f"{domain}::{group}",
                        problem_hash=f"{domain}::p{group}",
                        sibling_group=f"{domain}::g{group}",
                        domain=domain,
                        candidate_step=step,
                        first_error_step=first_error,
                        decision_position=2 + step,
                        label=int(first_error == step),
                        nuisance=np.asarray([step, group, 0.0, 0.0, step == 0]),
                        output_context=np.asarray([step, group, step, group, step > 0]),
                        store_index=0,
                        state_index=row_index,
                    )
                )
                row_index += 1
    return ProcessBenchMonitorData(
        rows=tuple(rows),
        stores=(
            _BoundaryStateStore(
                path=state_path,
                values=np.load(state_path, mmap_mode="r"),
            ),
        ),
        layer_ids=np.asarray([1, 2, 3]),
        hidden_size=4,
        output_feature_names=("entropy", "nll"),
    )


def test_inner_split_keeps_every_problem_group_together(tmp_path) -> None:
    data = _data(tmp_path)
    outer_train = np.asarray(
        [i for i, row in enumerate(data.rows) if row.domain != "omnimath"]
    )

    train, validation = build_inner_group_split(
        data.rows, outer_train, validation_fraction=0.34, seed=17
    )

    assert set(train).isdisjoint(validation)
    assert set(train).union(validation) == set(outer_train)
    outer_set = set(outer_train)
    for group in {data.rows[i].sibling_group for i in outer_train}:
        positions = {
            i
            for i, row in enumerate(data.rows)
            if row.sibling_group == group and i in outer_set
        }
        assert positions.issubset(set(train)) or positions.issubset(set(validation))
    assert {data.rows[i].domain for i in validation} == {"gsm8k", "math"}


def test_state_normalizer_is_fit_only_on_requested_rows(tmp_path) -> None:
    data = _data(tmp_path)

    normalizer = fit_state_normalizer(data, np.asarray([0, 1]))

    expected = np.stack([data.state(0), data.state(1)])
    np.testing.assert_allclose(normalizer.mean, expected.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(
        normalizer.scale,
        np.maximum(expected.std(axis=0), 1e-4),
        rtol=1e-5,
        atol=1e-6,
    )


def test_training_scores_each_requested_boundary_for_every_arm(tmp_path) -> None:
    data = _data(tmp_path)
    train = np.arange(0, 12, dtype=np.int64)
    validation = np.arange(12, 18, dtype=np.int64)
    config = MonitorTrainingConfig(
        width=8,
        message_passing_steps=1,
        epochs=1,
        patience=1,
        batch_size=4,
        learning_rate=1e-3,
        device="cpu",
        show_progress=False,
    )

    for arm in (
        "nuisance",
        "output_history",
        "layer_set",
        "depth_graph_shuffled",
        "depth_graph",
    ):
        trained = train_monitor_arm(
            data,
            train,
            validation,
            arm=arm,
            config=config,
            seed=19,
        )
        scores = trained.predict(data, validation, batch_size=3, device="cpu")
        assert scores.shape == (len(validation),)
        assert np.all((0.0 <= scores) & (scores <= 1.0))
        assert trained.arm == arm


def test_inner_split_fails_when_event_and_correct_validation_is_impossible(
    tmp_path,
) -> None:
    data = _data(tmp_path)
    only_error_groups = np.asarray(
        [
            i
            for i, row in enumerate(data.rows)
            if row.domain != "omnimath" and row.first_error_step >= 0
        ]
    )

    with pytest.raises(ValueError, match="fully-correct chain"):
        build_inner_group_split(
            data.rows,
            only_error_groups,
            validation_fraction=0.34,
            seed=17,
        )
