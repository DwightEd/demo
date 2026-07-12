import numpy as np

from anchorflow.phase import (
    calibrate_chain_fpr_threshold,
    causal_boundary_events,
    causal_change_scores,
)


def test_boundary_free_change_is_causal_and_localizes_injected_shift():
    prefix = np.zeros((7, 3))
    prefix[6] = 2.0
    base = causal_change_scores(prefix, min_history=4, ridge=0.05)
    extended = causal_change_scores(np.vstack([prefix, [-4.0, -4.0, -4.0]]), min_history=4, ridge=0.05)
    assert int(np.nanargmax(base["change_score"])) == 6
    for key in base:
        np.testing.assert_allclose(base[key], extended[key][: len(prefix)], equal_nan=True)


def test_chain_level_threshold_and_online_events():
    threshold = calibrate_chain_fpr_threshold(
        [[np.nan, 0.2, 0.4], [0.1, 0.3], [0.5]],
        target_fpr=0.34,
    )
    events = causal_boundary_events([0.1, threshold, threshold + 1.0, 0.0], threshold, refractory=2)
    assert events.tolist() == [False, True, False, False]
