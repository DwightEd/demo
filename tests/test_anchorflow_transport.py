import numpy as np

from anchorflow.anchor_repr import build_anchor_bank
from anchorflow.anchors import fallback_anchors
from anchorflow.data import Trace
from anchorflow.transport import transport_features


def _trace():
    q = np.zeros(16)
    q[:4] = 1.0
    stepvec = np.tile(q, (5, 1)).astype(float)
    stepvec[3] = np.r_[np.zeros(8), np.ones(8)]
    return Trace(
        idx=0,
        chain_id="c0",
        problem_id=0,
        dataset="self",
        correct=False,
        gold_error_step=3,
        step_token_ranges=np.array([[0, 3], [4, 7], [8, 11], [12, 15], [16, 19]]),
        steps_text=[],
        response_text="",
        prompt_text="",
        features={"logN": np.ones(5), "pos": np.linspace(0, 1, 5), "spread": np.linspace(0.1, 0.5, 5)},
        stepvec=stepvec,
        qvec=q,
        sv_layers=[14],
        hidden_path=None,
        layer=14,
    )


def test_transport_features_have_expected_shape_and_jump():
    tr = _trace()
    bank = build_anchor_bank(tr, fallback_anchors(), max_anchors=4)
    feats = transport_features(tr, bank)
    assert feats["af_anchor_entropy"].shape == (5,)
    assert feats["af_transport_jump"].shape == (5,)
    assert np.isfinite(feats["af_transport_jump"][3])
