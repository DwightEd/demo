from __future__ import annotations

import numpy as np
import torch

from functional_divergence.causal_first_error_attribution.monitor_models import (
    DepthGraphMonitor,
    LayerSetMonitor,
    depth_neighbor_mean,
    fixed_layer_permutation,
)


def test_depth_neighbor_message_uses_the_layer_chain() -> None:
    values = torch.tensor([[[0.0], [2.0], [8.0], [18.0]]])

    observed = depth_neighbor_mean(values)

    torch.testing.assert_close(
        observed,
        torch.tensor([[[2.0], [4.0], [10.0], [8.0]]]),
    )


def test_layer_set_control_is_invariant_to_layer_order() -> None:
    torch.manual_seed(3)
    model = LayerSetMonitor(hidden_size=5, context_size=4, width=8)
    model.eval()
    states = torch.randn(6, 4, 5)
    context = torch.randn(6, 4)
    permutation = torch.tensor([2, 0, 3, 1])

    original = model(states, context)
    shuffled = model(states[:, permutation], context)

    torch.testing.assert_close(original, shuffled)


def test_depth_graph_monitor_consumes_full_layer_hidden_tensor() -> None:
    torch.manual_seed(5)
    model = DepthGraphMonitor(
        hidden_size=7,
        layer_count=5,
        context_size=3,
        width=12,
        message_passing_steps=2,
    )
    states = torch.randn(9, 5, 7)
    context = torch.randn(9, 3)

    logits = model(states, context)

    assert logits.shape == (9,)
    assert model.input_projection.in_features == 7
    assert model.layer_position.num_embeddings == 5
    assert len(model.depth_blocks) == 2


def test_fixed_layer_permutation_is_reproducible_and_nontrivial() -> None:
    first = fixed_layer_permutation(layer_count=8, seed=17)
    second = fixed_layer_permutation(layer_count=8, seed=17)

    np.testing.assert_array_equal(first, second)
    assert sorted(first.tolist()) == list(range(8))
    assert not np.array_equal(first, np.arange(8))
