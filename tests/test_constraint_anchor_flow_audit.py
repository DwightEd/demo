import argparse
import os

import numpy as np

import constraint_anchor_flow_audit as caf


def _args(tmp_path, npz):
    return argparse.Namespace(
        input=str(npz),
        layer=16,
        nearest_layer=False,
        hidden_dir="",
        no_mmap=False,
        control_pool="pre_and_correct",
        anchor_rank=4,
        posterior_temp=0.08,
        other_score=0.02,
        center_chain=True,
        raw_projection=False,
        kappa_beta=1.0,
        min_tokens=4,
        min_per_class=1,
        min_feature_coverage=0.55,
        coherent_spread_q=0.50,
        pos_match_step_gap=0,
        folds=5,
        bootstrap=10,
        seed=11,
        max_chains=0,
        examples_per_class=2,
        output_dir=str(tmp_path),
        no_progress=True,
        selftest=False,
    )


def test_probability_helpers_are_stable():
    p = caf.normalize_prob(np.array([0.0, np.nan, 2.0]))
    assert np.isfinite(p).all()
    assert np.isclose(p.sum(), 1.0)
    assert caf.kl_div(np.array([1.0, 0.0]), np.array([1.0, 0.0])) < 1e-6
    assert caf.js_div(np.array([1.0, 0.0]), np.array([0.0, 1.0])) > 0.1


def test_constraint_anchor_flow_selftest_runs_and_writes_outputs(tmp_path):
    npz = tmp_path / "constraint_anchor_flow_selftest.npz"
    caf.make_selftest(str(npz), seed=5, n_problems=12, samples_per_problem=5)
    res = caf.run(str(npz), _args(tmp_path, npz))
    caf.assert_selftest(res)

    best = res["headline"]["best_anchor"]
    assert best["within_pair_auroc_error_high"] >= 0.80
    assert "pos_matched_within_auroc_error_high" in best
    assert res["headline"]["coherent_slice"]["ok"]
    assert "text_hidden_kl" in [r["score"] for r in res["single_scores"][:6]]

    jpath, mpath = caf.write_outputs(res, str(tmp_path), "constraint_anchor_flow_selftest")
    assert os.path.exists(jpath)
    assert os.path.exists(mpath)


def test_text_anchor_posterior_detects_unsupported_numbers():
    p, feats = caf.text_anchor_posterior(
        "Use 7 and 9 to get 23.",
        question_numbers=[7.0, 9.0],
        earlier_numbers=[],
        recent_numbers=[],
    )
    assert np.isclose(p.sum(), 1.0)
    assert feats["text_unsupported_numbers"] == 1.0
    assert p[caf.ANCHORS.index("question")] > p[caf.ANCHORS.index("other")]
