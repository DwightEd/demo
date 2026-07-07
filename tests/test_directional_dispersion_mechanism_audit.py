import argparse
import os

import numpy as np

import directional_dispersion_mechanism_audit as ddm


def _args(tmp_path, npz):
    return argparse.Namespace(
        input=str(npz),
        policy="gold_error_step",
        label_mode="auto",
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
        low_kappa_q=0.30,
        min_feature_coverage=0.70,
        bootstrap=10,
        seed=3,
        max_chains=0,
        examples_per_class=2,
        output_dir=str(tmp_path),
        no_progress=True,
        selftest=False,
    )


def test_residual_scatter_identity_holds():
    rng = np.random.default_rng(1)
    H = rng.normal(size=(13, 31))
    feats = ddm.morphology_features(H, beta=1.0, min_tokens=4, top_k=8, max_pair_tokens=64)

    assert feats["res_identity_error"] < 1e-10
    assert np.isclose(feats["residual_energy"], 1.0 - feats["kappa"] ** 2)


def test_high_rank_and_bipolar_morphologies_separate():
    rng = np.random.default_rng(2)
    hi = ddm._make_cloud(rng, n=24, dim=32, kappa=0.45, mode="high_rank")[:, 0, :]
    bi = ddm._make_cloud(rng, n=24, dim=32, kappa=0.45, mode="bipolar")[:, 0, :]

    f_hi = ddm.morphology_features(hi, beta=1.0, min_tokens=4, top_k=8, max_pair_tokens=64)
    f_bi = ddm.morphology_features(bi, beta=1.0, min_tokens=4, top_k=8, max_pair_tokens=64)

    assert f_hi["res_eff_rank"] > f_bi["res_eff_rank"] + 2.0
    assert f_bi["bipolarity"] > f_hi["bipolarity"]


def test_directional_dispersion_selftest_runs_and_writes_outputs(tmp_path):
    npz = tmp_path / "directional_dispersion_selftest.npz"
    ddm.make_selftest(str(npz), seed=4, n_chains=45, dim=40)
    res = ddm.run(str(npz), _args(tmp_path, npz))
    ddm.assert_selftest(res)

    h1a = res["headline"]["hypotheses"]["H1a_high_rank_dispersion"]["conditional"]["point"]
    assert h1a["conditional_best_direction"] >= 0.75
    assert "taxonomy_enrichment" in res["headline"]

    jpath, mpath, tax_path, feat_path, ex_path = ddm.write_outputs(
        res, str(tmp_path), "directional_dispersion_selftest"
    )
    for path in (jpath, mpath, tax_path, feat_path, ex_path):
        assert os.path.exists(path)
