import argparse
import os

import numpy as np

import token_stream_geometry_audit as tsga


def _args(tmp_path):
    return argparse.Namespace(
        input=None,
        policy="answer_format_ok",
        layer=16,
        nearest_layer=False,
        hidden_dir="",
        no_mmap=False,
        windows="8,16,32",
        alpha_windows="16,32",
        decay=0.08,
        min_window=5,
        min_tokens=8,
        alpha_k=8,
        alpha_stride=4,
        no_alpha=False,
        min_feature_coverage=0.70,
        min_per_class=1,
        folds=3,
        bootstrap=20,
        alarm_fpr=0.05,
        alarm_warmup_tokens=6,
        min_increment=0.02,
        max_problems=0,
        max_tokens=0,
        shape_max_traces=24,
        save_profiles=True,
        profile_max_points=32,
        profile_max_traces=12,
        seed=11,
        output_dir=str(tmp_path),
        no_progress=True,
        selftest=False,
    )


def test_sliding_resultant_separates_constant_from_alternating():
    e0 = np.array([1.0, 0.0])
    e1 = np.array([0.0, 1.0])
    U_const = np.tile(e0[None, :], (16, 1))
    U_alt = np.vstack([e0 if i % 2 == 0 else e1 for i in range(16)])

    r_const = tsga.sliding_resultant(U_const, window=8, decay=0.0, min_window=4)
    r_alt = tsga.sliding_resultant(U_alt, window=8, decay=0.0, min_window=4)

    assert np.nanmin(r_const[7:]) > 0.99
    assert np.nanmean(r_alt[7:]) < 0.80


def test_multi_window_cpu_matches_single_window():
    rng = np.random.default_rng(4)
    H = rng.normal(size=(23, 17))
    _, U = tsga.normalize_rows(H)
    multi = tsga.sliding_resultants_multi_cpu(U, windows=[5, 11], decay=0.07, min_window=4)

    for W in (5, 11):
        one = tsga.sliding_resultant(U, window=W, decay=0.07, min_window=4)
        np.testing.assert_allclose(multi[W], one, rtol=1e-5, atol=1e-6, equal_nan=True)


def test_token_stream_selftest_runs_and_writes_outputs(tmp_path):
    npz = tmp_path / "token_stream_selftest.npz"
    tsga.make_selftest(str(npz), seed=5, n_problems=12, samples_per_problem=6)
    res = tsga.run(str(npz), _args(tmp_path))
    tsga.assert_selftest(res)

    assert res["headline"]["best_stream_group"].startswith("token_stream_")
    assert res["headline"]["best_alarm"]["error_recall"] >= 0.50
    assert any(k.startswith("eff_rank_") for k in res["profiles"][0]["traces"])
    assert "trajectory_shape" in res and res["trajectory_shape"]

    jpath, mpath = tsga.write_outputs(res, str(tmp_path), "token_stream_selftest")
    assert os.path.exists(jpath)
    assert os.path.exists(mpath)
    assert os.path.exists(tmp_path / "token_stream_selftest.profiles.jsonl")
