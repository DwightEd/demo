import argparse
import os

import premise_constraint_audit as pca


def _args(tmp_path):
    return argparse.Namespace(
        input=None,
        policy="answer_format_ok",
        min_per_class=1,
        bands="mid",
        bootstrap=30,
        seed=13,
        output_dir=str(tmp_path),
        selftest=False,
    )


def test_arithmetic_equation_parser_flags_invalid_step():
    pairs = pca.equation_pairs("Step 1: Combine: 7 + 5 = 13.")
    assert len(pairs) == 1
    assert not pca.close_num(pairs[0][0], pairs[0][1])

    bank = pca.KnownNumberBank()
    bank.seed([7, 5])
    metrics = pca.step_constraint_metrics("Step 1: Combine: 7 + 5 = 13.", bank)
    assert metrics.invalid_equations == 1
    assert metrics.risk > 1.0


def test_premise_constraint_selftest_rescues_baseline_misses(tmp_path):
    npz = tmp_path / "premise_constraint_selftest.npz"
    pca.make_selftest(str(npz), seed=4, n_problems=16, samples_per_problem=4)
    res = pca.run(str(npz), _args(tmp_path))
    pca.assert_selftest(res)

    best_constraint = res["headline"]["best_constraint"]
    rescue = res["headline"]["baseline_miss_rescue"]
    assert best_constraint["within_pair_auroc_error_high"] >= 0.95
    assert rescue["baseline_missed_pairs"] > 0
    assert rescue["rescue_rate_among_baseline_misses"] >= 0.90

    jpath, mpath = pca.write_outputs(res, str(tmp_path), "premise_constraint_selftest")
    assert os.path.exists(jpath)
    assert os.path.exists(mpath)
