import numpy as np

from anchorflow.hazard import (
    DiscreteHazardReadout,
    discrete_hazard_nll,
    hazard_to_event_cdf,
    make_first_error_hazard_targets,
)


def test_first_error_targets_right_censor_correct_and_mask_post_error():
    out = make_first_error_hazard_targets([4, 5], [-1, 2])
    assert out.event_observed.tolist() == [False, True]
    assert out.at_risk[0].tolist() == [True, True, True, True, False]
    assert out.target[0].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert out.at_risk[1].tolist() == [True, True, True, False, False]
    assert out.target[1].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_hazard_loss_ignores_post_error_logits_and_cdf_is_monotone():
    targets = make_first_error_hazard_targets([4], [1])
    a = np.array([[-2.0, 2.0, -100.0, 100.0]])
    b = np.array([[-2.0, 2.0, 100.0, -100.0]])
    assert discrete_hazard_nll(a, targets.target, targets.at_risk) == discrete_hazard_nll(
        b, targets.target, targets.at_risk
    )
    cdf = hazard_to_event_cdf(np.array([0.1, 0.2, 0.4]))
    assert np.all(np.diff(cdf) >= 0)


def test_hazard_readout_trains_only_on_at_risk_first_error_positions():
    seqs = []
    lengths = []
    first_error = []
    for i in range(20):
        x = np.zeros((4, 1))
        event = 2 if i % 2 else -1
        if event >= 0:
            x[event, 0] = 5.0
            x[event + 1, 0] = -100.0  # post-error value must not enter fitting
        seqs.append(x)
        lengths.append(4)
        first_error.append(event)
    model = DiscreteHazardReadout().fit(seqs, lengths, first_error)
    hazard = model.predict_hazard(np.array([[0.0], [0.0], [5.0]]))
    assert hazard[2] > hazard[0]
