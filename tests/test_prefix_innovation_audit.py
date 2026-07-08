import argparse
import os

import numpy as np

import prefix_innovation_audit as pia


def _args(tmp_path, npz):
    return argparse.Namespace(
        input=str(npz),
        layer=16,
        nearest_layer=False,
        hidden_dir="",
        no_mmap=False,
        control_pool="pre_and_correct",
        rank=8,
        beta=1.0,
        raw_hidden=False,
        center_chain=True,
        z_top_k=16,
        z_thresh=3.0,
        std_floor_frac=0.10,
        dim_top_k=12,
        min_per_class=1,
        min_feature_coverage=0.60,
        folds=5,
        bootstrap=10,
        seed=2,
        max_chains=0,
        output_dir=str(tmp_path),
        no_progress=True,
        selftest=False,
    )


def test_off_subspace_energy_detects_orthogonal_direction():
    rng = np.random.default_rng(0)
    base = np.eye(5)[:2]
    X_in = rng.normal(size=(8, 2)) @ base
    X_out = X_in + 2.0 * np.eye(5)[3]
    V = pia.basis_from_rows(X_in, rank=2)
    assert pia.off_subspace_energy(X_in, V) < 1e-8
    assert pia.off_subspace_energy(X_out, V) > 0.50


def test_prefix_innovation_selftest_runs_and_writes_outputs(tmp_path):
    npz = tmp_path / "prefix_innovation_selftest.npz"
    pia.make_selftest(str(npz), seed=4, n_problems=10, samples_per_problem=5)
    res = pia.run(str(npz), _args(tmp_path, npz))
    pia.assert_selftest(res)

    assert res["headline"]["best_transition"]["within_pair_auroc_error_high"] >= 0.85
    dim = res["meta"]["dim_activation"]
    assert dim["ok"]
    assert dim["top_positive"][0]["dim"] == 7

    jpath, mpath = pia.write_outputs(res, str(tmp_path), "prefix_innovation_selftest")
    assert os.path.exists(jpath)
    assert os.path.exists(mpath)
