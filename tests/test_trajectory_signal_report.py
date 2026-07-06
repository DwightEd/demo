import argparse
import json
import os

import trajectory_signal_report as tsr


def test_profile_report_writes_distribution_and_shape_files(tmp_path):
    profile = tmp_path / "toy.profiles.jsonl"
    rows = [
        {
            "idx": 0,
            "problem_id": 0,
            "y_err": 0,
            "n_tokens": 8,
            "n_steps": 2,
            "error_token": None,
            "traces": {
                "spread_w4": [0.10, 0.12, 0.15, 0.18, 0.17, 0.15, 0.13, 0.11],
                "eff_rank_raw_w4": [1.0, 1.5, 2.5, 3.0, 2.6, 2.0, 1.4, 1.1],
            },
        },
        {
            "idx": 1,
            "problem_id": 0,
            "y_err": 1,
            "n_tokens": 8,
            "n_steps": 2,
            "error_token": 5,
            "traces": {
                "spread_w4": [0.20, 0.22, 0.27, 0.33, 0.42, 0.50, 0.55, 0.60],
                "eff_rank_raw_w4": [1.0, 1.2, 1.8, 2.2, 2.4, 2.5, 2.7, 2.8],
            },
        },
    ]
    with open(profile, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    args = argparse.Namespace(
        profiles=str(profile),
        input="",
        policy="answer_format_ok",
        layer=16,
        nearest_layer=False,
        hidden_dir="",
        no_mmap=False,
        stream_backend="cpu",
        stream_device="",
        windows="4",
        alpha_windows="4",
        decay=0.0,
        min_window=2,
        min_tokens=4,
        alpha_k=4,
        alpha_stride=1,
        no_alpha=False,
        max_tokens=0,
        max_samples=0,
        max_problems=0,
        signals="spread_w*,eff_rank_raw_w*",
        max_signals=8,
        bins=4,
        output_dir=str(tmp_path / "out"),
        no_progress=True,
    )

    res = tsr.make_result(args)

    assert res["n_samples"] == 2
    assert "spread_w4" in res["signals"]
    assert "eff_rank_raw_w4" in res["signals"]
    for path in res["files"].values():
        assert os.path.exists(path)
    assert any(r["signal"] == "eff_rank_raw_w4" for r in res["trajectory_shape"])
