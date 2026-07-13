from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from prompt_control_flow.directional_consensus import (
    DirectionalConsensusAuditConfig,
    DirectionalConsensusConfig,
    directional_statistics,
    inspect_directional_cloud_source,
    load_directional_cloud_dataset,
    run_directional_consensus_audit,
    write_directional_consensus_outputs,
)


def _object_array(values: list[np.ndarray]) -> np.ndarray:
    output = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        output[index] = value
    return output


def _write_synthetic_multisample(path: Path, n_problems: int = 12) -> None:
    rng = np.random.default_rng(23)
    layers = np.asarray([8, 16], dtype=np.int64)
    hidden = 8
    trajectories: list[np.ndarray] = []
    clouds: list[np.ndarray] = []
    cloud_sizes: list[np.ndarray] = []
    problem_ids: list[int] = []
    sample_idx: list[int] = []
    is_correct: list[int] = []
    responses: list[str] = []

    for problem in range(n_problems):
        for sample in range(4):
            correct = sample < 2
            n_tokens = 12 + ((problem * 3 + sample * 5) % 5)
            first = n_tokens // 3
            second = n_tokens // 3
            sizes = np.asarray([first, second, n_tokens - first - second], dtype=np.int64)

            if correct:
                base = np.zeros((n_tokens, layers.size, hidden), dtype=np.float32)
                base[..., 0] = 1.0
                cloud = base + rng.normal(0.0, 0.025, size=base.shape).astype(np.float32)
            else:
                cloud = rng.normal(
                    0.0,
                    1.0,
                    size=(n_tokens, layers.size, hidden),
                ).astype(np.float32)
            trajectory = rng.normal(0.0, 1.0, size=(3, layers.size, hidden)).astype(np.float32)

            trajectories.append(trajectory)
            clouds.append(cloud)
            cloud_sizes.append(sizes)
            problem_ids.append(problem)
            sample_idx.append(sample)
            is_correct.append(int(correct))
            responses.append("x" * (40 + 3 * n_tokens + ((problem + sample) % 7)))

    correct_array = np.asarray(is_correct, dtype=np.int64)
    np.savez(
        path,
        sv_vec_step_exp=_object_array(trajectories),
        sv_clouds=_object_array(clouds),
        cloud_sizes=_object_array(cloud_sizes),
        problem_ids=np.asarray(problem_ids, dtype=np.int64),
        sample_idx=np.asarray(sample_idx, dtype=np.int64),
        is_correct=correct_array,
        is_correct_strict=correct_array,
        format_ok=np.ones(correct_array.size, dtype=bool),
        responses=np.asarray(responses, dtype=object),
        layers_used=layers,
        cloud_layers=layers,
        model_name=np.asarray("synthetic"),
    )


def test_directional_statistics_matches_explicit_off_diagonal_cosine() -> None:
    vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=torch.float32,
    )
    resultant, concentration = directional_statistics(vectors)
    unit = torch.nn.functional.normalize(vectors, dim=-1)
    similarity = unit @ unit.T
    expected = (similarity.sum() - similarity.diag().sum()) / 12.0

    np.testing.assert_allclose(concentration.numpy(), expected.numpy(), atol=1e-7)
    np.testing.assert_allclose(
        resultant.numpy(),
        torch.linalg.vector_norm(unit.mean(dim=0)).numpy(),
        atol=1e-7,
    )


def test_same_problem_audit_recovers_injected_directional_dispersion(
    tmp_path: Path,
) -> None:
    source = tmp_path / "multisample.npz"
    _write_synthetic_multisample(source)
    preflight = inspect_directional_cloud_source(source)
    assert preflight["ready"] is True
    assert preflight["contrastive_problems"] == 12
    assert preflight["cloud_layers"] == [8, 16]

    dataset = load_directional_cloud_dataset(source)
    report, packed = run_directional_consensus_audit(
        dataset,
        DirectionalConsensusConfig(
            fixed_window_tokens=4,
            batch_size=16,
            max_batch_tokens=256,
            compute_device="cpu",
        ),
        DirectionalConsensusAuditConfig(
            folds=4,
            bootstrap=60,
            permutations=39,
            length_match_ratio=1.25,
            seed=5,
        ),
    )
    scores = {row["name"]: row for row in report["scores"]}
    primary = scores["consensus.debiased_dispersion.step_mean.length_residual"]
    assert primary["same_problem_auroc"] > 0.95
    assert primary["token_length_matched_auroc"] > 0.95
    assert scores["consensus.fixed_window_dispersion.mean"]["same_problem_auroc"] > 0.95
    assert report["decision_gate"]["passes"] is True
    assert np.isnan(scores["control.log1p_n_steps"]["same_problem_permutation_p"])
    assert report["meta"]["contrastive_problems"] == 12
    assert packed["scores"].shape[0] == dataset.n_samples

    paths = write_directional_consensus_outputs(
        report,
        packed,
        output=tmp_path / "scores.npz",
        output_dir=tmp_path / "audit",
        render_plots=False,
    )
    for value in paths.values():
        assert Path(value).exists()
    saved = np.load(paths["scores_npz"], allow_pickle=True)
    assert "step_pairwise_concentration" in saved.files
    assert "response_tokens" in saved.files
    assert saved["step_pairwise_concentration"].shape == (dataset.n_samples,)
