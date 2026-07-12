import numpy as np

from anchorflow.lookback import compact_attention_lookback, compact_hidden_lookback


def test_hidden_lookback_detects_anchor_transport_shift_and_is_causal():
    anchors = np.eye(2)
    states = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    base = compact_hidden_lookback(states, anchors, tau=0.05)
    extended = compact_hidden_lookback(np.vstack([states, [1.0, 0.0]]), anchors, tau=0.05)
    assert int(np.nanargmax(base["transport_shift"])) == 3
    for key in ("max_similarity", "anchor_entropy", "detach", "transport_shift"):
        np.testing.assert_allclose(base[key], extended[key][: len(states)], equal_nan=True)


def test_attention_lookback_reduces_heads_to_prompt_mass():
    attention = np.array([
        [[0.30, 0.20, 0.50], [0.10, 0.10, 0.80]],
        [[0.20, 0.20, 0.60], [0.15, 0.05, 0.80]],
    ])
    out = compact_attention_lookback(attention, [True, True, False])
    np.testing.assert_allclose(out["prompt_mass"], [0.45, 0.20])
    assert np.all((out["prompt_concentration"] >= 0) & (out["prompt_concentration"] <= 1))


def test_attention_lookback_reports_agreement_persistence_and_topk_churn():
    # Two identical layers x two identical heads; the preferred prompt token
    # flips between consecutive queries.
    one = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    attention = np.tile(one[None, None, :, :], (2, 2, 1, 1))
    out = compact_attention_lookback(attention, [True, True, False], top_k=1)
    np.testing.assert_allclose(out["head_agreement"], [1.0, 1.0])
    np.testing.assert_allclose(out["layer_persistence"], [1.0, 1.0])
    assert np.isnan(out["topk_churn"][0])
    assert out["topk_churn"][1] == 1.0
