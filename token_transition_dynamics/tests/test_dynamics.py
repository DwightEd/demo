from __future__ import annotations

import numpy as np

from token_transition_dynamics.dynamics import transition_coordinates


def test_spherical_log_map_is_tangent_to_start_state() -> None:
    projected = np.asarray(
        [
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
            [[0.0, 2.0, 0.0], [0.0, 1.0, 2.0], [1.0, 0.0, 1.0]],
        ]
    )

    state, displacement = transition_coordinates(projected, "spherical")

    assert state.shape == (2, 2, 3)
    assert displacement.shape == (2, 2, 3)
    np.testing.assert_allclose(
        np.sum(state * displacement, axis=-1), np.zeros((2, 2)), atol=1e-7
    )
