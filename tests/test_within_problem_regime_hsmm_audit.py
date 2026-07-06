import argparse
import os

import numpy as np

import within_problem_regime_hsmm_audit as hsmm


def _args(tmp_path, npz):
    return argparse.Namespace(
        input=str(npz),
        policy="answer_format_ok",
        channels="cloud_spread,out_entropy,pr_mid,ae_mid",
        bands="mid",
        require_channels=False,
        min_channel_coverage=0.8,
        min_per_class=1,
        max_problems=0,
        grid=12,
        problem_center="problem_median",
        include_abs_delta=False,
        states=3,
        max_duration=4,
        em_iters=4,
        folds=2,
        smooth=1e-2,
        min_var=1e-3,
        prefix_fracs="0.50,0.80,1.00",
        permutations=5,
        seed=2,
        output_dir=str(tmp_path),
        selftest=False,
    )


def test_regime_hsmm_selftest_core(tmp_path):
    npz = tmp_path / "regime_selftest.npz"
    hsmm.make_selftest(str(npz), seed=2, n_problems=12, samples_per_problem=4)
    args = _args(tmp_path, npz)

    res = hsmm.run(str(npz), args)

    assert res["meta"]["n_contrastive_problems"] == 12
    assert "pos" not in ",".join(res["meta"]["obs_names"])
    assert np.isfinite(res["headline"]["hsmm_full_same_problem_auroc"])
    assert res["headline"]["hsmm_full_same_problem_auroc"] >= 0.60
    assert res["final_model"]["transition_difference"]["transition_l1"] > 0.1
    assert res["final_model"]["transition_difference"]["duration_l1"] > 0.1

    jpath, mpath = hsmm.write_outputs(res, str(tmp_path), "regime_selftest")
    assert os.path.exists(jpath)
    assert os.path.exists(mpath)
