from argparse import Namespace

import numpy as np

from whole_chain_gram_metrics_audit import (
    assert_selftest,
    build_arg_parser,
    gram_metrics_from_matrix,
    make_selftest,
    prefix_gram_traces,
    run,
)


def test_strict_hs_me_match_identity_scale():
    H = 2.0 * np.eye(4, dtype=np.float64)
    out = gram_metrics_from_matrix(H, rel_tol=1e-12)
    assert np.isclose(out["hs"], np.log(4.0))
    assert np.isclose(out["me"], np.log(4.0))
    assert np.isclose(out["lam1"], 0.25)
    assert out["hs_full_rank"] == 1.0


def test_strict_hs_nan_when_rank_deficient():
    H = np.ones((5, 2), dtype=np.float64)
    out = gram_metrics_from_matrix(H, rel_tol=1e-12)
    assert np.isnan(out["hs"])
    assert out["rank"] < 5
    assert np.isfinite(out["me"])


def test_prefix_traces_are_whole_chain_prefix_grams():
    H = np.eye(5, dtype=np.float64)
    traces = prefix_gram_traces(H, np.asarray([1, 3, 4]), rel_tol=1e-12)
    assert np.allclose(traces["prefix_me"], np.log([2.0, 4.0, 5.0]))
    assert np.allclose(traces["prefix_hs"], 0.0)


def test_selftest_recovers_whole_chain_and_prefix_signal(tmp_path):
    path = tmp_path / "whole_chain_selftest.npz"
    make_selftest(str(path), seed=4, n_problems=8, samples_per_problem=6, dim=48)
    args = build_arg_parser().parse_args(
        [
            "--input",
            str(path),
            "--layer",
            "16",
            "--bootstrap",
            "20",
            "--no_progress",
        ]
    )
    res = run(str(path), args)
    assert_selftest(res)
    assert res["headline"]["paper_hs_coverage"] == 1.0
    assert res["headline"]["attention_score_status"] == "available"

