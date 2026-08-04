from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reasoning_error_detection.data import (
    FeatureConfig,
    ProcessBenchFeatureLoader,
    StepFeatureDataset,
)
from reasoning_error_detection.detector import DetectorConfig, ProcessBenchErrorDetector
from reasoning_error_detection.main import build_parser


def _write_processbench_features(path: Path) -> None:
    chain_idx = np.asarray([10, 11, 12], dtype=np.int64)
    n_steps = np.asarray([2, 3, 2], dtype=np.int64)
    state_chain_idx = np.repeat(chain_idx, n_steps)
    state_step_idx = np.concatenate(
        [np.arange(count, dtype=np.int64) for count in n_steps]
    )
    layers = np.asarray([8, 16], dtype=np.int64)
    pre = np.arange(7 * 2 * 4, dtype=np.float32).reshape(7, 2, 4) / 10.0
    end = pre.copy()
    end[:, 0, 0] += np.asarray([0.1, 0.2, 0.1, 2.0, 0.3, 3.0, 0.2])
    end[:, 1, 1] += np.asarray([0.2, 0.1, 0.2, 1.5, 0.4, 2.5, 0.1])
    np.save(path.parent / "pre_states.npy", pre)
    np.save(path.parent / "end_states.npy", end)

    ranges = np.full((3, 3, 2), -1, dtype=np.int64)
    ranges[0, :2] = [[10, 11], [12, 15]]
    ranges[1, :3] = [[20, 21], [22, 24], [25, 29]]
    ranges[2, :2] = [[30, 33], [34, 35]]
    step_scores = np.full((3, 3, 3), np.nan, dtype=np.float32)
    step_scores[0, :2] = [[0.2, 2.0, 0.1], [0.3, 4.0, 0.2]]
    step_scores[1, :3] = [[0.2, 2.0, 0.1], [0.9, 3.0, 0.8], [0.7, 5.0, 0.7]]
    step_scores[2, :2] = [[0.8, 4.0, 0.9], [0.6, 2.0, 0.6]]

    np.savez_compressed(
        path,
        chain_idx=chain_idx,
        problem_group_ids=np.asarray([100, 101, 102], dtype=np.int64),
        gold_error_step=np.asarray([-1, 1, 0], dtype=np.int64),
        n_steps=n_steps,
        step_token_ranges=ranges,
        step_score_names=np.asarray(
            ["token_entropy", "step_len", "icr_mean"], dtype=object
        ),
        step_scores=step_scores,
        step_pre_state_memmap_path=np.asarray("pre_states.npy", dtype=object),
        step_pre_state_memmap_count=np.asarray(7, dtype=np.int64),
        step_pre_state_vector_chain_idx=state_chain_idx,
        step_pre_state_vector_step_idx=state_step_idx,
        step_end_state_memmap_path=np.asarray("end_states.npy", dtype=object),
        step_end_state_memmap_count=np.asarray(7, dtype=np.int64),
        step_end_state_vector_chain_idx=state_chain_idx,
        step_end_state_vector_step_idx=state_step_idx,
        step_layer_state_vector_layers=layers,
    )


def test_loader_builds_label_safe_first_error_rows_from_processbench_features(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.npz"
    _write_processbench_features(path)

    data = ProcessBenchFeatureLoader(
        FeatureConfig(layers=(8, 16), projection_dim=2, seed=7, device="cpu")
    ).load(path)

    assert data.chain_idx.tolist() == [10, 10, 11, 11, 11, 12, 12]
    assert data.step_idx.tolist() == [0, 1, 0, 1, 2, 0, 1]
    assert data.onset_label.tolist() == [0, 0, 0, 1, -1, 1, -1]
    assert data.onset_eligible.tolist() == [True, True, True, True, False, True, False]
    assert data.chain_error.tolist() == [0, 0, 1, 1, 1, 1, 1]
    assert data.control_names == (
        "control.log1p_step_index",
        "control.log1p_step_tokens",
        "control.log1p_previous_step_tokens",
        "control.log1p_cumulative_tokens",
    )
    assert data.control_features[0, 0] == pytest.approx(0.0)
    assert data.control_features[1, 0] == pytest.approx(np.log(2.0))
    assert data.output_names == ("output.token_entropy",)
    assert data.routing_names == ("routing.icr_mean",)
    assert data.residual_features.shape == (7, 12)
    assert np.isfinite(data.residual_features).all()


def test_loader_fails_when_processbench_residual_state_views_are_missing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.npz"
    np.savez_compressed(
        path,
        chain_idx=np.asarray([0]),
        problem_ids=np.asarray([0]),
        gold_error_step=np.asarray([-1]),
        n_steps=np.asarray([1]),
        step_token_ranges=np.asarray([[[1, 2]]]),
    )

    with pytest.raises(ValueError, match="step_pre_state"):
        ProcessBenchFeatureLoader(FeatureConfig(device="cpu")).load(path)


def _synthetic_detection_dataset() -> StepFeatureDataset:
    rng = np.random.default_rng(23)
    chain_idx: list[int] = []
    groups: list[int] = []
    step_idx: list[int] = []
    gold: list[int] = []
    onset: list[int] = []
    chain_error: list[int] = []
    control: list[np.ndarray] = []
    output: list[np.ndarray] = []
    routing: list[np.ndarray] = []
    residual: list[np.ndarray] = []
    for chain in range(24):
        first_error = 1 + (chain // 2) % 3 if chain % 2 else -1
        for step in range(4):
            is_onset = int(first_error == step)
            chain_idx.append(chain)
            groups.append(chain)
            step_idx.append(step)
            gold.append(first_error)
            onset.append(0 if first_error < 0 or step < first_error else (1 if is_onset else -1))
            chain_error.append(int(first_error >= 0))
            control.append(np.asarray([step, 1.0, 1.0, step + 1.0]))
            output.append(np.asarray([0.4 * is_onset + 0.2 * rng.normal()]))
            routing.append(np.asarray([0.6 * is_onset + 0.2 * rng.normal()]))
            residual.append(
                np.asarray(
                    [
                        3.0 * is_onset + 0.15 * rng.normal(),
                        -2.0 * is_onset + 0.15 * rng.normal(),
                    ]
                )
            )
    labels = np.asarray(onset, dtype=np.int8)
    data = StepFeatureDataset(
        source_path="synthetic",
        chain_idx=np.asarray(chain_idx, dtype=np.int64),
        problem_groups=np.asarray(groups, dtype=np.int64),
        step_idx=np.asarray(step_idx, dtype=np.int64),
        gold_error_step=np.asarray(gold, dtype=np.int64),
        onset_label=labels,
        onset_eligible=labels >= 0,
        chain_error=np.asarray(chain_error, dtype=np.int8),
        control_features=np.asarray(control, dtype=np.float32),
        output_features=np.asarray(output, dtype=np.float32),
        routing_features=np.asarray(routing, dtype=np.float32),
        residual_features=np.asarray(residual, dtype=np.float32),
        control_names=("step", "length", "previous_length", "cumulative_length"),
        output_names=("entropy",),
        routing_names=("icr",),
        residual_names=("delta_x", "delta_y"),
        selected_layers=(16,),
    )
    data.validate()
    return data


def test_detector_crossfits_by_problem_and_localizes_injected_first_errors(
    tmp_path: Path,
) -> None:
    detector = ProcessBenchErrorDetector(
        DetectorConfig(folds=4, logistic_c=1.0, seed=29)
    )

    report = detector.run(_synthetic_detection_dataset(), tmp_path / "report")

    assert report["split"]["problem_group_overlap"] == 0
    assert report["feature_sets"]["residual"]["onset"]["auroc"] > 0.95
    assert (
        report["feature_sets"]["residual"]["onset"]["auroc"]
        > report["feature_sets"]["controls"]["onset"]["auroc"] + 0.2
    )
    assert report["feature_sets"]["joint"]["localization"]["top1"] > 0.9
    assert report["feature_sets"]["joint"]["chain_detection"]["auroc"] > 0.9
    assert (tmp_path / "report" / "summary.json").exists()
    predictions = np.load(tmp_path / "report" / "oof_predictions.npz")
    assert np.isfinite(predictions["probabilities"]).all()


def test_main_parser_exposes_processbench_feature_run_parameters() -> None:
    args = build_parser().parse_args(
        [
            "--input",
            "/data/processbench/gsm8k/geometry/trace.npz",
            "--output_dir",
            "/results/gsm8k",
            "--layers",
            "8,16,24",
            "--projection_dim",
            "32",
            "--device",
            "cuda:0",
        ]
    )

    assert args.input.endswith("geometry/trace.npz")
    assert args.layers == (8, 16, 24)
    assert args.projection_dim == 32
    assert args.device == "cuda:0"


def test_remote_runner_uses_geometry_manifests_and_active_python() -> None:
    runner = (
        Path(__file__).parents[1]
        / "reasoning_error_detection"
        / "run_processbench.sh"
    ).read_text(encoding="utf-8")

    assert "${subset}/geometry/trace.npz" in runner
    assert "gsm8k math olympiadbench omnimath" in runner
    assert 'PYTHON_BIN="${PYTHON_BIN:-python}"' in runner
    assert "transformers" not in runner
    assert "selected/trace.npz" not in runner
