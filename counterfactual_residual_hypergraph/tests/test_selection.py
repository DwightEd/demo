from __future__ import annotations

import numpy as np

from crwh.selection import select_review_candidates


def test_review_selector_prioritizes_detector_disagreement() -> None:
    unsupervised = np.asarray([0.01, 0.99, 0.95, 0.10])
    semisupervised = np.asarray([0.02, 0.01, 0.90, 0.12])

    selection = select_review_candidates(
        unsupervised_anomaly_percentile=unsupervised,
        semisupervised_probability=semisupervised,
        unsupervised_threshold=0.95,
        semisupervised_threshold=0.5,
        budget=1,
    )

    assert selection.tolist() == [1]


def test_review_selector_does_not_compare_incommensurate_raw_levels() -> None:
    selection = select_review_candidates(
        unsupervised_anomaly_percentile=np.asarray([0.50, 0.60]),
        semisupervised_probability=np.asarray([0.10, 0.20]),
        unsupervised_threshold=0.95,
        semisupervised_threshold=0.5,
        budget=2,
    )

    assert selection.size == 0
