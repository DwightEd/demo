from __future__ import annotations

import numpy as np

from token_transition_dynamics.baseline import CorrectOnlyBaseline


def test_correct_only_baseline_scores_departures_from_domain_layer_position_norms() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(80, 2, 4))
    domains = np.full(80, "math", dtype=object)
    bins = np.repeat(np.arange(4), 20)
    baseline = CorrectOnlyBaseline(position_bins=4, min_bin_samples=10).fit(
        features, domains, bins
    )

    ordinary = baseline.standardize(features[:4], domains[:4], bins[:4])
    anomalous_features = features[:4].copy()
    anomalous_features[:, :, 0] += 12.0
    anomalous = baseline.standardize(anomalous_features, domains[:4], bins[:4])

    assert ordinary.shape == features[:4].shape
    assert np.mean(anomalous[:, :, 0] ** 2) > np.mean(ordinary[:, :, 0] ** 2)
    assert baseline.used_error_labels is False
