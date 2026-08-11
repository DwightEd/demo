from __future__ import annotations

import numpy as np
import torch

from functional_divergence.hidden_state_geometry.contracts import ChainSample
from functional_divergence.hidden_state_geometry.method import FoldInput
from functional_divergence.hidden_state_geometry.methods import load_builtin_methods
from functional_divergence.hidden_state_geometry.methods.predictive_state_monitor import (
    PredictiveStateConfig,
    _arm_sequence,
)
from functional_divergence.hidden_state_geometry.methods.token_attention import (
    TokenAttentionPool,
    _sinusoidal_positions,
)
from functional_divergence.hidden_state_geometry.registry import create_method
from functional_divergence.hidden_state_geometry.tasks import (
    TaskExample,
    build_strict_prefix_task,
)


def _sample(tmp_path, chain_id: int, *, error: bool) -> ChainSample:
    states = np.zeros((4, 2, 6), dtype=np.float32)
    direction = 1.0 if error else -1.0
    for step in range(4):
        states[step, :, 0] = direction * (step + 1)
        states[step, :, 1] = direction * (-1.0 if step % 2 else 1.0)
        states[step, :, 2:] = chain_id * 0.01
    path = tmp_path / f"chain_{chain_id}.npy"
    np.save(path, states)
    return ChainSample(
        chain_id=chain_id,
        manifest_row=chain_id,
        problem_group=f"problem_sha256:p{chain_id}",
        dataset="train" if chain_id < 24 else "test",
        generator="llama31-8b",
        observer_model="llama31-8b",
        state_path=path,
        state_count=4,
        response_start=10,
        step_ranges=np.asarray([[10 + step, 10 + step] for step in range(4)]),
        layer_ids=np.asarray([8, 10]),
        output_steps=np.column_stack(
            [np.linspace(0.1, 0.4, 4), np.linspace(0.4, 0.1, 4)]
        ).astype(np.float32),
        output_feature_names=("token_entropy", "token_nll"),
        first_error_step=3 if error else -1,
    )


def _fold(tmp_path) -> FoldInput:
    samples = tuple(_sample(tmp_path, i, error=bool(i % 2)) for i in range(32))
    task = build_strict_prefix_task(samples)
    train = np.asarray(
        [i for i, row in enumerate(task.examples) if row.sample.dataset == "train"]
    )
    test = np.asarray(
        [i for i, row in enumerate(task.examples) if row.sample.dataset == "test"]
    )
    return FoldInput(
        task_name=task.name,
        train_examples=tuple(task.examples[i] for i in train),
        train_labels=task.labels[train],
        train_groups=task.groups[train],
        test_examples=tuple(task.examples[i] for i in test),
        seed=13,
    )


def test_state_arm_sequences_share_current_and_only_shuffle_visible_past(tmp_path):
    sample = _sample(tmp_path, 3, error=True)
    example = TaskExample(sample, visible_steps=4, boundary_step=4, task_name="strict_prefix")
    sequence = np.arange(20, dtype=np.float32).reshape(4, 5)

    initial = _arm_sequence(sequence, example, "initial_state", seed=7)
    current = _arm_sequence(sequence, example, "current_state", seed=7)
    ordered = _arm_sequence(sequence, example, "ordered_history", seed=7)
    shuffled = _arm_sequence(sequence, example, "shuffled_history", seed=7)

    assert np.array_equal(initial, sequence[:1])
    assert np.array_equal(current, sequence[-1:])
    assert np.array_equal(ordered, sequence)
    assert np.array_equal(shuffled[-1], sequence[-1])
    assert sorted(map(tuple, shuffled[:-1])) == sorted(map(tuple, sequence[:-1]))
    assert not np.array_equal(shuffled[:-1], sequence[:-1])


def test_single_state_prefix_is_identical_for_current_ordered_and_shuffled(tmp_path):
    sample = _sample(tmp_path, 1, error=True)
    example = TaskExample(sample, visible_steps=1, boundary_step=1, task_name="strict_prefix")
    sequence = np.arange(5, dtype=np.float32).reshape(1, 5)

    assert np.array_equal(
        _arm_sequence(sequence, example, "current_state", seed=3),
        _arm_sequence(sequence, example, "ordered_history", seed=3),
    )
    assert np.array_equal(
        _arm_sequence(sequence, example, "current_state", seed=3),
        _arm_sequence(sequence, example, "shuffled_history", seed=3),
    )


def test_token_state_arms_keep_all_tokens_and_only_shuffle_complete_past_steps(
    tmp_path,
):
    sample = _sample(tmp_path, 3, error=True)
    example = TaskExample(
        sample, visible_steps=4, boundary_step=4, task_name="strict_prefix"
    )
    steps = (
        np.asarray([[10.0], [11.0]], dtype=np.float32),
        np.asarray([[20.0], [21.0], [22.0]], dtype=np.float32),
        np.asarray([[30.0]], dtype=np.float32),
        np.asarray([[40.0], [41.0]], dtype=np.float32),
    )

    initial = _arm_sequence(steps, example, "initial_state", seed=7)
    current = _arm_sequence(steps, example, "current_state", seed=7)
    ordered = _arm_sequence(steps, example, "ordered_history", seed=7)
    shuffled = _arm_sequence(steps, example, "shuffled_history", seed=7)

    assert initial[:, 0].tolist() == [10.0, 11.0]
    assert current[:, 0].tolist() == [40.0, 41.0]
    assert ordered[:, 0].tolist() == [
        10.0,
        11.0,
        20.0,
        21.0,
        22.0,
        30.0,
        40.0,
        41.0,
    ]
    assert shuffled[-2:, 0].tolist() == [40.0, 41.0]
    assert sorted(shuffled[:-2, 0].tolist()) == [
        10.0,
        11.0,
        20.0,
        21.0,
        22.0,
        30.0,
    ]
    assert shuffled[:-2, 0].tolist() != ordered[:-2, 0].tolist()
    for step in steps[:-1]:
        positions = [
            int(np.flatnonzero(shuffled[:, 0] == token)[0]) for token in step[:, 0]
        ]
        assert positions == list(range(positions[0], positions[0] + len(step)))


def test_attention_arm_sequence_marks_step_boundaries_without_pooling_tokens(tmp_path):
    sample = _sample(tmp_path, 3, error=True)
    example = TaskExample(
        sample, visible_steps=2, boundary_step=2, task_name="strict_prefix"
    )
    steps = (
        np.asarray([[10.0], [11.0]], dtype=np.float32),
        np.asarray([[20.0], [21.0], [22.0]], dtype=np.float32),
    )

    sequence = _arm_sequence(
        steps,
        example,
        "ordered_history",
        seed=7,
        add_step_markers=True,
    )

    assert sequence[:, 0].tolist() == [10.0, 11.0, 20.0, 21.0, 22.0]
    assert sequence[:, 1:].tolist() == [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ]


def test_token_attention_has_direct_access_to_early_tokens_and_masks_padding():
    torch.manual_seed(5)
    model = TokenAttentionPool(
        input_dim=5,
        context_dim=3,
        width=8,
        heads=2,
        queries=2,
    )
    states = torch.randn(2, 257, 5, requires_grad=True)
    lengths = torch.tensor([257, 13], dtype=torch.int64)
    context = torch.zeros(2, 3)

    logits, attention = model.forward_with_attention(states, lengths, context)
    logits.sum().backward()

    assert attention.shape == (2, 2, 2, 257)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2, 2, 2), atol=1e-6)
    assert torch.count_nonzero(attention[1, :, :, 13:]) == 0
    assert float(states.grad[0, 0].abs().sum()) > 1e-8


def test_attention_positions_keep_current_suffix_fixed_when_history_is_removed():
    current_only = _sinusoidal_positions(
        3,
        8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with_history = _sinusoidal_positions(
        11,
        8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(current_only, with_history[-3:])


def test_predictive_state_plugin_returns_capacity_matched_raw_history_arms(tmp_path):
    load_builtin_methods()
    method = create_method(
        "predictive_state_monitor",
        PredictiveStateConfig(
            pca_dim=2,
            positions_per_chain=4,
            sequence_unit="token",
            sequence_encoder="attention_pool",
            attention_heads=2,
            attention_queries=2,
            width=4,
            epochs=2,
            patience=1,
            batch_size=16,
            validation_fraction=0.25,
            device="cpu",
            show_progress=False,
        ),
    )

    result = method.fit_predict(_fold(tmp_path))

    assert set(result.probabilities) == {
        "output_only",
        "initial_state",
        "current_state",
        "ordered_history",
        "shuffled_history",
        "ordered_model_current_ablation",
        "ordered_model_shuffled_ablation",
    }
    assert all(values.shape == (24,) for values in result.probabilities.values())
    assert all(np.isfinite(values).all() for values in result.probabilities.values())
    counts = result.diagnostics["state_arm_parameter_counts"]
    assert len(set(counts.values())) == 1
    assert result.diagnostics["pca_fit_scope"] == "outer_train_unique_chains"
    assert result.diagnostics["target_alignment"] == "completed_prefix_predicts_next_step_first_error"
    assert result.diagnostics["uses_final_response_length"] is False
    assert result.diagnostics["sequence_unit"] == "token"
    assert result.diagnostics["step_pooling"] == "none"
    assert result.diagnostics["all_visible_step_tokens_preserved"] is True
    assert result.diagnostics["sequence_encoder"] == "attention_pool"
    assert 0.0 <= result.diagnostics["ordered_history_attention_mass_mean"] <= 1.0
    assert -1.0 <= result.diagnostics[
        "ordered_history_attention_excess_over_token_fraction_mean"
    ] <= 1.0
    assert np.isfinite(
        result.diagnostics["same_model_history_ablation_max_abs_probability_change"]
    )
