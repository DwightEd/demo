from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from prompt_control_flow.causal_pullback.features import build_pullback_features
from prompt_control_flow.causal_pullback.schema import (
    CausalPullbackArtifact,
    CausalPullbackConfig,
    CausalPullbackItem,
    FieldWitnesses,
)
from prompt_control_flow.data import ChainRecord


def _synthetic_item(index: int = 0) -> CausalPullbackItem:
    n_steps = 4
    transitions = n_steps - 1
    fisher = np.full((3, transitions, n_steps), np.nan, dtype=np.float32)
    chosen = np.full_like(fisher, np.nan)
    entropy = np.full_like(fisher, np.nan)
    half = np.full((transitions, n_steps), np.nan, dtype=np.float32)
    for direction, scale in enumerate((2.0, 0.8, 0.6)):
        fisher[direction, 0, 2:] = scale * np.asarray([1.0, 0.5])
        fisher[direction, 1, 3] = scale * 0.7
        chosen[direction] = np.where(np.isfinite(fisher[direction]), 0.1 * scale, np.nan)
        entropy[direction] = np.where(np.isfinite(fisher[direction]), -0.05 * scale, np.nan)
    half[:] = fisher[0]
    return CausalPullbackItem(
        chain_idx=index,
        original_index=index,
        problem_id=index // 2,
        sample_idx=index,
        is_correct=index % 2,
        n_steps=n_steps,
        response_chars=80 + index,
        layer=1,
        donor_count=6,
        replay_kind="synthetic",
        replay_cosine=np.full(n_steps, 0.999, dtype=np.float32),
        baseline_step_features=np.arange(n_steps * 4, dtype=np.float32).reshape(n_steps, 4),
        field_energy=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        field_calibrated_energy=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        witness_norms=np.asarray(
            [[1.0, 0.8, 0.4], [0.7, 0.6, 0.3], [1.0, 0.8, 0.4]],
            dtype=np.float32,
        ),
        fisher_transfer=fisher,
        chosen_logprob_transfer=chosen,
        entropy_transfer=entropy,
        primary_half_fisher_transfer=half,
        perturbation_scale=np.full(transitions, 0.1, dtype=np.float32),
        metadata={"maximum_acausal_fisher_leakage": 0.0},
    )


def test_spherical_energy_witness_is_rotation_equivariant() -> None:
    pytest.importorskip("torch")
    from prompt_control_flow.causal_pullback.field import spherical_energy_witness

    rng = np.random.default_rng(5)
    target = rng.normal(size=12).astype(np.float32)
    references = rng.normal(size=(7, 12)).astype(np.float32)
    permutation = rng.permutation(12)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=12)
    witness, norm = spherical_energy_witness(target, references)
    rotated, rotated_norm = spherical_energy_witness(
        target[permutation] * signs,
        references[:, permutation] * signs[None, :],
    )
    np.testing.assert_allclose(
        rotated, witness[permutation] * signs, atol=2e-5, rtol=2e-5
    )
    np.testing.assert_allclose(rotated_norm, norm, atol=2e-5, rtol=2e-5)
    unit_target = target / np.linalg.norm(target)
    assert abs(float(np.dot(witness, unit_target))) < 2e-5


def test_pullback_artifact_and_phase_features_round_trip(tmp_path) -> None:
    artifact = CausalPullbackArtifact(
        items=[_synthetic_item(0), _synthetic_item(1)],
        metadata={
            "config": {"replay_cosine_threshold": 0.98},
            "evidence_tier": "synthetic",
        },
    )
    path = tmp_path / "pullback.npz"
    artifact.save(path)
    loaded = CausalPullbackArtifact.load(path)
    assert loaded.n_items == 2
    np.testing.assert_allclose(
        loaded.items[0].fisher_transfer,
        artifact.items[0].fisher_transfer,
        equal_nan=True,
    )
    features = build_pullback_features(loaded, phase_grid=3)
    assert features.x_output.shape[0] == 2
    assert features.x_field.shape[0] == 2
    assert features.x_pullback.shape[0] == 2
    assert features.valid.all()
    assert np.all(
        features.direct_scores["field_consequential_mean"]
        > features.direct_scores["random_consequential_mean"]
    )


def test_resume_retries_skips_and_replaces_stale_failure() -> None:
    pytest.importorskip("torch")
    from prompt_control_flow.causal_pullback.extraction import CausalPullbackAccumulator

    accumulator = CausalPullbackAccumulator()
    accumulator.add_skip(
        original_index=3,
        chain_idx=30,
        problem_id=7,
        reason="OutOfMemoryError",
        detail="first attempt",
    )
    assert 3 not in accumulator.completed_original_indices
    accumulator.add_skip(
        original_index=3,
        chain_idx=30,
        problem_id=7,
        reason="RuntimeError",
        detail="second attempt",
    )
    assert len(accumulator.skipped) == 1
    assert accumulator.skipped[0]["reason"] == "RuntimeError"
    item = _synthetic_item(3)
    accumulator.add(item)
    assert 3 in accumulator.completed_original_indices
    assert accumulator.skipped == []

    accumulator.add(_synthetic_item(4))
    accumulator.add_skip(
        original_index=5,
        chain_idx=50,
        problem_id=9,
        reason="RuntimeError",
        detail="outside the new cohort",
    )
    accumulator.retain_original_indices({4})
    assert accumulator.completed_original_indices == {4}
    assert accumulator.skipped == []


def test_pilot_target_selection_preserves_same_problem_contrasts() -> None:
    pytest.importorskip("torch")
    from prompt_control_flow.causal_pullback.extraction import select_replay_targets

    problem_ids = np.repeat(np.arange(6), 4)
    y_error = np.tile(np.asarray([1, 0, 0, 0], dtype=np.int8), 6)
    sample_idx = np.tile(np.arange(4), 6)
    eligible = np.arange(len(problem_ids), dtype=np.int64)
    selected = select_replay_targets(
        problem_ids,
        y_error,
        sample_idx,
        eligible,
        max_targets=8,
        seed=19,
    )
    assert len(selected) == 8
    selected_problems = np.unique(problem_ids[selected])
    assert len(selected_problems) == 4
    for problem in selected_problems:
        local = selected[problem_ids[selected] == problem]
        assert set(y_error[local].tolist()) == {0, 1}
    np.testing.assert_array_equal(
        select_replay_targets(
            problem_ids,
            y_error,
            sample_idx,
            eligible,
            max_targets=0,
            seed=19,
        ),
        eligible,
    )


def test_torch_step_exp_pool_matches_legacy_numpy_implementation() -> None:
    torch = pytest.importorskip("torch")
    from prompt_control_flow.causal_pullback.replay import _step_exp_pool
    from utils.step_vector import step_vector

    rng = np.random.default_rng(23)
    cloud = rng.normal(size=(9, 13)).astype(np.float32)
    ranges = [(0, 0), (1, 3), (4, 8)]
    pooled = _step_exp_pool(torch.as_tensor(cloud), ranges).cpu().numpy()
    expected = np.stack(
        [
            step_vector(cloud[start : stop + 1], mode="step_exp", l2_normalize=False)
            for start, stop in ranges
        ]
    )
    np.testing.assert_allclose(pooled, expected, atol=2e-6, rtol=2e-6)


def test_exact_trace_replay_does_not_require_prompt_reconstruction() -> None:
    pytest.importorskip("torch")
    from prompt_control_flow.causal_pullback.data import prepare_record_trace

    record = ChainRecord(
        chain_idx=0,
        problem_id=0,
        problem="",
        steps=["first", "second"],
        response="first\nsecond",
        exact_input_ids=[10, 11, 12, 13, 14],
        exact_attention_mask=[1, 1, 1, 1, 1],
        exact_token_offsets=[(0, 0), (0, 0), (0, 5), (5, 6), (6, 12)],
        exact_step_token_ranges=[(2, 2), (4, 4)],
        exact_response_start_token=2,
    )
    trace, prompt = prepare_record_trace(
        record,
        object(),
        prompt_style="",
        ordered_questions=None,
        max_seq_len=32,
    )
    assert prompt == ""
    assert trace["input_ids"] == record.exact_input_ids
    assert trace["replay_kind"] == "exact_artifact_ids"


def test_replay_operator_is_strictly_causal() -> None:
    torch = pytest.importorskip("torch")
    from prompt_control_flow.causal_pullback.replay import (
        _step_exp_pool,
        compute_causal_pullback,
    )

    class CausalBlock(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(width, width, bias=False)
            torch.nn.init.eye_(self.projection.weight)

        def forward(self, hidden):
            position = torch.arange(
                1, hidden.shape[1] + 1, device=hidden.device, dtype=hidden.dtype
            )[None, :, None]
            causal_mean = torch.cumsum(hidden, dim=1) / position
            return hidden + 0.25 * self.projection(causal_mean)

    class ToyBackbone(torch.nn.Module):
        def __init__(self, vocab: int, width: int) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(vocab, width)
            self.layers = torch.nn.ModuleList([CausalBlock(width), CausalBlock(width)])

        def forward(
            self,
            input_ids,
            attention_mask=None,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        ):
            del attention_mask, use_cache, return_dict
            hidden = self.embed(input_ids)
            history = [hidden]
            for layer in self.layers:
                hidden = layer(hidden)
                history.append(hidden)
            return SimpleNamespace(
                last_hidden_state=hidden,
                hidden_states=tuple(history) if output_hidden_states else None,
            )

    class ToyLM(torch.nn.Module):
        base_model_prefix = "model"

        def __init__(self) -> None:
            super().__init__()
            self.model = ToyBackbone(vocab=31, width=10)
            self.lm_head = torch.nn.Linear(10, 31, bias=False)

        def get_output_embeddings(self):
            return self.lm_head

    torch.manual_seed(13)
    model = ToyLM().eval()
    ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    ranges = [(1, 2), (3, 4), (5, 6), (7, 7)]
    baseline = model.model(
        ids[None, :], output_hidden_states=True, return_dict=True
    )
    stored = _step_exp_pool(baseline.hidden_states[1][0], ranges).detach().numpy()
    transitions = len(ranges) - 1
    rng = np.random.default_rng(17)
    field = rng.normal(size=(transitions, stored.shape[1])).astype(np.float32)
    field /= np.linalg.norm(field, axis=1, keepdims=True)
    shuffle = np.roll(field, 1, axis=0)
    random = rng.normal(size=field.shape).astype(np.float32)
    random /= np.linalg.norm(random, axis=1, keepdims=True)
    witnesses = FieldWitnesses(
        field_direction=field,
        shuffle_direction=shuffle,
        random_direction=random,
        field_witness_norm=np.ones(transitions, dtype=np.float32),
        shuffle_witness_norm=np.ones(transitions, dtype=np.float32),
        field_energy=np.ones(transitions, dtype=np.float32),
        field_calibrated_energy=np.ones(transitions, dtype=np.float32),
        donor_count=6,
    )
    result = compute_causal_pullback(
        model,
        {
            "input_ids": ids.tolist(),
            "attention_mask": [1] * len(ids),
            "step_token_ranges": ranges,
        },
        stored,
        witnesses,
        CausalPullbackConfig(
            layer=1,
            min_donors=3,
            max_donors=6,
            epsilon_fraction=0.01,
            replay_cosine_threshold=0.999,
            variant_batch_size=4,
            logit_token_chunk=3,
        ),
    )
    assert np.nanmin(result.replay_cosine) > 0.999
    # Transition zero ends at step one, so only steps two and three are causal.
    assert np.isnan(result.fisher_transfer[:, 0, :2]).all()
    assert np.isfinite(result.fisher_transfer[:, 0, 2:]).all()
    # The final transition ends at the final step and has no measurable future.
    assert np.isnan(result.fisher_transfer[:, -1]).all()
    assert result.metadata["maximum_acausal_fisher_leakage"] < 1e-8


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="local Windows research env has a broken native LAPACK Ridge backend",
)
def test_end_to_end_audit_writes_grouped_oof_report(tmp_path) -> None:
    pytest.importorskip("sklearn")
    from prompt_control_flow.causal_pullback.audit import (
        CausalPullbackAuditConfig,
        run_causal_pullback_audit,
    )
    from prompt_control_flow.ocgpi.models import CrossFitConfig

    items = [_synthetic_item(index) for index in range(48)]
    artifact = CausalPullbackArtifact(
        items=items,
        metadata={
            "config": {"replay_cosine_threshold": 0.98},
            "evidence_tier": "synthetic",
            "source_model": "toy",
            "observer_model": "toy",
        },
    )
    artifact_path = tmp_path / "synthetic_pullback.npz"
    output_dir = tmp_path / "audit"
    artifact.save(artifact_path)
    report = run_causal_pullback_audit(
        artifact_path,
        output_dir,
        CausalPullbackAuditConfig(
            phase_grid=3,
            bootstrap=10,
            min_coverage=0.5,
            min_contrastive_problems=2,
            crossfit=CrossFitConfig(outer_folds=3, inner_folds=2, seed=11),
        ),
    )
    assert report["preflight"]["contrastive_problems"] == 24
    assert set(report["conditional_increment"]) == {
        "field_only",
        "causal_pullback_only",
        "field_plus_causal_pullback",
    }
    assert not report["validation"]["confirmatory_ready"]
    assert (output_dir / "summary.md").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "oof_predictions.npz").is_file()
