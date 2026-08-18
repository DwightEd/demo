from __future__ import annotations

import numpy as np

from token_transition_dynamics.window_geometry import window_features


def test_window_features_are_computed_per_layer_in_raw_coordinates() -> None:
    token = np.linspace(-1.0, 1.0, 12)
    states = np.zeros((12, 2, 6), dtype=np.float64)
    states[:, 0, 0] = token
    states[:, 0, 1] = token**2
    states[:, 1, 2] = np.sin(token)
    states[:, 1, 4] = np.cos(token)

    features = window_features(states, neighbors=4, tle_centers=6)

    assert features.shape == (2, 4)
    assert np.all(np.isfinite(features))
    assert np.all(features[:, 0] > 0.0)
    assert np.all(features[:, 1] >= 0.0)
    assert np.all(features[:, 2] >= 0.0)
    assert np.all((features[:, 3] >= 0.0) & (features[:, 3] <= 1.0))


def test_order_sensitive_features_change_when_token_order_is_shuffled() -> None:
    token = np.linspace(0.0, 2.0 * np.pi, 16)
    states = np.zeros((16, 1, 8), dtype=np.float64)
    states[:, 0, 0] = np.cos(token)
    states[:, 0, 1] = np.sin(token)
    shuffled = states[np.asarray([0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15])]

    ordered_features = window_features(states, neighbors=5, tle_centers=6)
    shuffled_features = window_features(shuffled, neighbors=5, tle_centers=6)

    np.testing.assert_allclose(ordered_features[:, :2], shuffled_features[:, :2])
    assert not np.allclose(ordered_features[:, 2:], shuffled_features[:, 2:])
