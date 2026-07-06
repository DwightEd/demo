import argparse
import os

import second_moment_dynamics_audit as smd


def _args(tmp_path, npz):
    return argparse.Namespace(
        input=str(npz),
        policy="answer_format_ok",
        layer=16,
        nearest_layer=False,
        min_per_class=1,
        min_steps=2,
        min_tokens=4,
        min_feature_coverage=0.70,
        max_baseline_features=12,
        max_problems=0,
        kappa_beta=1.0,
        top_k=8,
        alpha_k=8,
        spectral_backend="cpu",
        spectral_device="",
        spectral_batch_size=8,
        folds=3,
        bootstrap=20,
        min_increment=0.02,
        seed=5,
        output_dir=str(tmp_path),
        no_progress=True,
        selftest=False,
    )


def test_second_moment_selftest_uses_direct_token_matrix(tmp_path):
    npz = tmp_path / "second_moment_selftest.npz"
    smd.make_selftest(str(npz), seed=5, n_problems=12, samples_per_problem=6)
    res = smd.run(str(npz), _args(tmp_path, npz))

    assert res["headline"]["best_group"].startswith("token_")
    assert res["headline"]["best_group_increment_over_baseline"]["point"] >= 0.10
    assert res["headline"]["best_gram_scalar"].startswith(("tok_raw_", "tok_cen_", "kappa_x_tok_", "spread_x_tok_"))
    assert "unit_direction_ablation" in res["meta"]["gram_groups"]

    jpath, mpath = smd.write_outputs(res, str(tmp_path), "second_moment_selftest")
    assert os.path.exists(jpath)
    assert os.path.exists(mpath)


def test_small_gram_eigvals_match_svd_squared():
    import numpy as np

    rng = np.random.default_rng(3)
    H = rng.normal(size=(7, 31))
    s2 = np.linalg.svd(H, compute_uv=False) ** 2
    gram = smd.small_gram_eigvals(H)
    np.testing.assert_allclose(gram[: len(s2)], s2, rtol=1e-10, atol=1e-10)
