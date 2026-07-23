from __future__ import annotations

from pathlib import Path

from functional_divergence.hidden_state_geometry.cli import (
    _method_config,
    build_parser,
    trace_sources,
)
from functional_divergence.hidden_state_geometry.config import RawFunctionalConfig


def test_cli_resolves_each_domain_to_selected_real_trace(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--data-root",
            str(tmp_path),
            "--domains",
            "gsm8k,math,omnimath",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    sources = trace_sources(args.data_root, args.domains, args.manifest_name)

    assert [source.dataset for source in sources] == ["gsm8k", "math", "omnimath"]
    assert sources[0].manifest == Path(tmp_path / "gsm8k/selected/trace.raw_residual_stream.npz")
    assert sources[0].exact_trace == Path(tmp_path / "gsm8k/selected/trace.npz")
    assert sources[0].acquisition_mode == "observer_teacher_forcing_replay"
    assert args.max_records_per_domain == 0
    assert args.method == "raw_functional_probe"


def test_cli_preserves_plugin_default_config_unless_json_is_explicit(tmp_path):
    base = [
        "run",
        "--data-root",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "out"),
        "--method",
        "future_plugin",
    ]
    parser = build_parser()

    assert _method_config(parser.parse_args(base)) is None
    assert _method_config(
        parser.parse_args([*base, "--method-config-json", '{"rank": 4}'])
    ) == {"rank": 4}

    raw = _method_config(
        parser.parse_args(
            [
                "run",
                "--data-root",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "raw"),
                "--pca-dim",
                "8",
            ]
        )
    )
    assert isinstance(raw, RawFunctionalConfig)
    assert raw.pca_dim == 8


def test_remote_runner_exposes_foreground_full_tensor_ridge_modes():
    runner = Path(__file__).resolve().parents[2] / "run_hidden_geometry_remote.sh"
    script = runner.read_text(encoding="utf-8")

    assert "ridge-smoke)" in script
    assert "ridge-full)" in script
    assert "--method full_tensor_ridge" in script
    assert "nohup" not in script
    assert "screen -dmS" not in script
    assert "tmux" not in script
