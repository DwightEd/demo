import argparse
import os

import trajectory_difference_audit as tda


def test_trajectory_difference_selftest_core(tmp_path):
    npz = tmp_path / "trajectory_difference_selftest.npz"
    tda.make_selftest(str(npz), seed=3, n_problems=18, samples_per_problem=4)
    args = argparse.Namespace(
        policy="answer_format_ok",
        channels="cloud_spread,out_entropy,out_committal,pr_mid,ae_mid",
        require_channels=True,
        min_channel_coverage=0.8,
        min_per_class=1,
        bands="mid",
        include_mahal=False,
        grid=20,
        permutations=80,
        cluster_t=2.0,
        alpha=0.10,
        signature_order=2,
        folds=3,
        bootstrap=40,
        alarm_eps="0.05,0.20",
        mean_alarm_reference=False,
        seed=11,
        output_dir=str(tmp_path),
        input=str(npz),
        selftest=False,
    )
    res = tda.run(str(npz), args)
    tda.assert_selftest(res)
    jpath, mpath = tda.write_outputs(res, str(tmp_path), "trajectory_difference_selftest")
    assert os.path.exists(jpath)
    assert os.path.exists(mpath)

