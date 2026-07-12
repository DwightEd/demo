import numpy as np

from anchorflow.hazard import grouped_oof_hazard


def test_grouped_oof_hazard_keeps_problem_groups_out_of_training_fold():
    rng = np.random.default_rng(3)
    sequences = []
    lengths = []
    errors = []
    groups = []
    for problem in range(8):
        for event in (False, True):
            length = 6
            x = 0.05 * rng.normal(size=(length, 2))
            error = 3 if event else None
            if event:
                x[3:, 0] += 4.0
            sequences.append(x)
            lengths.append(length)
            errors.append(error)
            groups.append(problem)

    result = grouped_oof_hazard(sequences, lengths, errors, groups, folds=4)

    assert not result["skipped_folds"]
    assert np.all(result["fold_id"] >= 0)
    assert all(np.isfinite(x).all() for x in result["hazard"])
    for i, error in enumerate(errors):
        if error is not None:
            assert not result["at_risk"][i, error + 1 :].any()
            assert result["hazard"][i][error] > result["hazard"][i][0]
