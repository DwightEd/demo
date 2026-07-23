from __future__ import annotations

import numpy as np

from functional_divergence.hidden_state_geometry.contracts import ChainSample
from functional_divergence.hidden_state_geometry.representation import (
    ChainBalancedPCA,
    FunctionalEncoder,
    sample_keyed_permutation,
)
from functional_divergence.hidden_state_geometry.tasks import TaskExample


def _example(tmp_path, chain_id: int, steps: int, offset: float) -> TaskExample:
    state = np.empty((steps, 2, 6), dtype=np.float32)
    for step in range(steps):
        for layer in range(2):
            state[step, layer] = offset + step + layer * 10 + np.arange(6)
    path = tmp_path / f"state_{chain_id}.npy"
    np.save(path, state)
    ranges = np.asarray([[10 + step, 10 + step] for step in range(steps)])
    sample = ChainSample(
        chain_id=chain_id,
        manifest_row=chain_id,
        problem_group=str(chain_id),
        dataset="gsm8k",
        generator="llama",
        observer_model="llama",
        state_path=path,
        state_count=steps,
        response_start=10,
        step_ranges=ranges,
        layer_ids=np.asarray([8, 10]),
        output_steps=np.arange(steps * 2, dtype=np.float32).reshape(steps, 2),
        output_feature_names=("entropy", "nll"),
        first_error_step=-1,
    )
    return TaskExample(sample, visible_steps=steps, boundary_step=None)


def test_chain_balanced_pca_gives_short_and_long_chains_equal_rows(tmp_path):
    short = _example(tmp_path, 1, 2, 0.0)
    long = _example(tmp_path, 2, 7, 100.0)
    projector = ChainBalancedPCA(dim=2, positions_per_chain=4, seed=3)

    projector.fit((short, long))

    assert projector.sampled_rows_per_chain == {("gsm8k", 1): 4, ("gsm8k", 2): 4}
    # Two steps x two layers from the short chain fixes four rows per chain.
    assert projector.training_rows == 8


def test_functional_encoder_uses_dct_for_whole_chain_and_two_histories_for_prefix(tmp_path):
    whole = _example(tmp_path, 3, 5, 0.0)
    prefix = TaskExample(whole.sample, visible_steps=3, boundary_step=3)
    projector = ChainBalancedPCA(dim=2, positions_per_chain=4, seed=4).fit((whole,))
    encoder = FunctionalEncoder(projector, time_basis=3, layer_basis=2)

    whole_tensor = encoder.hidden_tensor(whole)
    prefix_tensor = encoder.hidden_tensor(prefix)
    whole_output = encoder.output_features(whole)
    prefix_output = encoder.output_features(prefix)

    assert whole_tensor.shape == (3, 2, 2)
    assert prefix_tensor.shape == (2, 2, 2)
    assert whole_output.shape == (3 * 2,)
    assert prefix_output.shape == (2 * 2,)
    assert np.isfinite(whole_tensor).all()
    assert np.isfinite(prefix_tensor).all()


def test_time_and_layer_nulls_preserve_shape_but_change_organized_field(tmp_path):
    example = _example(tmp_path, 4, 6, 0.0)
    projector = ChainBalancedPCA(dim=2, positions_per_chain=6, seed=5).fit((example,))
    encoder = FunctionalEncoder(projector, time_basis=3, layer_basis=2)

    ordered = encoder.hidden_tensor(example)
    time_null = encoder.hidden_tensor(example, null="time")
    layer_null = encoder.hidden_tensor(example, null="layer")

    assert ordered.shape == time_null.shape == layer_null.shape
    assert not np.allclose(ordered, time_null)
    assert np.isfinite(layer_null).all()


def test_axis_null_uses_sample_keyed_permutations_not_one_global_reversal(tmp_path):
    examples = [_example(tmp_path, chain_id, 6, 0.0) for chain_id in range(10, 18)]

    orders = {
        tuple(sample_keyed_permutation(row, length=6, axis="time", seed=17))
        for row in examples
    }

    assert len(orders) > 1
    assert tuple(range(5, -1, -1)) not in orders or len(orders) > 2

    two_layer_orders = {
        tuple(sample_keyed_permutation(row, length=2, axis="layer", seed=17))
        for row in examples
    }
    assert two_layer_orders == {(0, 1), (1, 0)}
