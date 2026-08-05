from __future__ import annotations

import torch

from functional_divergence.causal_first_error_attribution.monitor_models import (
    StaticLayerSetMonitor,
    TwoBoundaryBagMonitor,
    TwoBoundaryInnovationMonitor,
)


def test_two_boundary_monitors_are_layer_permutation_invariant() -> None:
    torch.manual_seed(2)
    models = (
        StaticLayerSetMonitor(hidden_size=5, context_size=3, width=8),
        TwoBoundaryBagMonitor(hidden_size=5, context_size=3, width=8),
        TwoBoundaryInnovationMonitor(hidden_size=5, context_size=3, width=8),
    )
    boundaries = torch.randn(4, 2, 6, 5)
    context = torch.randn(4, 3)

    for model in models:
        model.eval()
        original = model(boundaries, context)
        permuted = model(boundaries[:, :, [3, 0, 5, 1, 4, 2]], context)
        torch.testing.assert_close(original, permuted)


def test_bag_discards_boundary_direction_but_innovation_preserves_it() -> None:
    torch.manual_seed(11)
    bag = TwoBoundaryBagMonitor(hidden_size=4, context_size=2, width=7)
    innovation = TwoBoundaryInnovationMonitor(
        hidden_size=4, context_size=2, width=7
    )
    bag.eval()
    innovation.eval()
    boundaries = torch.randn(3, 2, 5, 4)
    context = torch.randn(3, 2)

    torch.testing.assert_close(
        bag(boundaries, context), bag(boundaries.flip(1), context)
    )
    assert not torch.allclose(
        innovation(boundaries, context),
        innovation(boundaries.flip(1), context),
    )


def test_two_boundary_controls_have_identical_capacity() -> None:
    models = (
        StaticLayerSetMonitor(hidden_size=9, context_size=4, width=6),
        TwoBoundaryBagMonitor(hidden_size=9, context_size=4, width=6),
        TwoBoundaryInnovationMonitor(hidden_size=9, context_size=4, width=6),
    )

    parameter_counts = {
        sum(value.numel() for value in model.parameters()) for model in models
    }
    assert len(parameter_counts) == 1
