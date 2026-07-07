import argparse
import os

import kappa_rank_joint_trajectory_audit as krj


def _args(tmp_path, npz):
    return argparse.Namespace(
        input=str(npz),
        policy="gold_error_step",
        layer=16,
        nearest_layer=False,
        hidden_dir="",
        no_mmap=False,
        kappa_beta=1.0,
        min_tokens=4,
        top_k=8,
        max_pair_tokens=64,
        length_bins=3,
        kappa_bins=3,
        pos_bins=2,
        control_pool="pre_and_correct",
        residual_ref="pre_and_correct",
        ridge=1e-3,
        quadrant_q=0.75,
        pre_window=3,
        post_window=3,
        profile_bins=4,
        bootstrap=10,
        seed=7,
        max_chains=0,
        output_dir=str(tmp_path),
        no_progress=True,
        selftest=False,
    )


def test_joint_trajectory_selftest_runs_and_writes_outputs(tmp_path):
    npz = tmp_path / "kappa_rank_joint_selftest.npz"
    krj.make_selftest(str(npz), seed=8)
    res = krj.run(str(npz), _args(tmp_path, npz))
    krj.assert_selftest(res)

    assert res["scores"]["joint_strict_zsum"]["auroc_error_high"] >= 0.85
    assert "dual_high_spread_high_rank" in res["headline"]["quadrants"]["quadrants"]
    assert res["transition_stats"]["joint_strict_zsum"]["pre_to_first_error"]["mean"] > 0

    paths = krj.write_outputs(res, str(tmp_path), "kappa_rank_joint_selftest")
    for path in paths:
        assert os.path.exists(path)


def test_z_against_does_not_explode_on_tiny_reference_variance():
    import numpy as np

    x = np.array([0.0] * 20 + [10.0])
    ref = np.array([0.0] * 20)
    z = krj.z_against(x, ref)
    assert np.isfinite(z).all()
    assert z[-1] < 100.0
