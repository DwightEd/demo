import argparse
import os

import within_problem_path_kernel_audit as pka


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
        grid=18,
        censor_frac=0.80,
        dct_components=5,
        signature_order=1,
        folds=3,
        bandwidth_points=120,
        score_permutations=10,
        mmd_permutations=10,
        seed=4,
        output_dir=str(tmp_path),
        selftest=False,
    )


def test_path_kernel_selftest_recovers_shape_not_static_level(tmp_path):
    npz = tmp_path / "path_kernel_selftest.npz"
    pka.make_selftest(str(npz), seed=4, n_problems=12, samples_per_problem=4)
    res = pka.run(str(npz), _args(tmp_path, npz))

    assert res["headline"]["best_shape_same_problem_auroc"] >= 0.85
    assert (
        res["headline"]["best_shape_same_problem_auroc"]
        - res["headline"]["best_static_same_problem_auroc"]
        >= 0.30
    )
    assert any(
        name.startswith("shape_") and row.get("p_ge", 1.0) <= 0.2
        for name, row in res["conditional_mmd"].items()
    )

    jpath, mpath = pka.write_outputs(res, str(tmp_path), "path_kernel_selftest")
    assert os.path.exists(jpath)
    assert os.path.exists(mpath)
