from __future__ import annotations

from pathlib import Path

from functional_divergence.hidden_state_geometry import cli
from functional_divergence.hidden_state_geometry.cli import (
    _method_config,
    build_parser,
    trace_sources,
)
from functional_divergence.hidden_state_geometry.config import RawFunctionalConfig


def _concise_result() -> dict:
    return {
        "run_id": "run-123",
        "execution": {"max_records_per_domain": 32, "bootstrap_replicates": 200},
        "method": {"name": "full_tensor_ridge"},
        "data": {"records": 128, "domains": ["gsm8k", "math"]},
        "tasks": {
            "whole_chain": {
                "rows": 128,
                "events": 89,
                "summary": {
                    "arms": {
                        "hidden_only": {
                            "macro": {"auroc": 0.7, "auprc": 0.8, "nll_nats": 0.6}
                        }
                    },
                    "increments": {
                        "hidden_given_output_summary_nll": {
                            "point": -0.02,
                            "ci_low": -0.04,
                            "ci_high": -0.01,
                            "inference_scope": "conditional_test_problem_group_cluster_bootstrap",
                        }
                    },
                },
            }
        },
    }


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
    assert script.count('"max_iter":2000') == 2
    assert "nohup" not in script
    assert "screen -dmS" not in script
    assert "tmux" not in script


def test_cli_run_prints_concise_summary_not_saved_diagnostics(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_hidden_geometry_experiment", lambda **_: _concise_result())

    cli.main(
        [
            "run",
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "output"),
            "--method",
            "full_tensor_ridge",
        ]
    )

    text = capsys.readouterr().out
    assert "output_dir:" in text
    assert "run_id: run-123" in text
    assert "bootstrap_replicates: 200" in text
    assert "domain-macro metrics (problem-group balanced):" in text
    assert "hidden_only: AUROC=0.7000" in text
    assert "worse_on_evaluated_domains" in text
    assert "fold_diagnostics" not in text
    assert "optimizer" not in text
    assert "gradient_inf_norm" not in text
    assert "\"tasks\"" not in text


def test_cli_preflight_prints_compact_domain_provenance(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "inspect_hidden_geometry_sources",
        lambda **_: {
            "run_id": "preflight-123",
            "shards_validated": 32,
            "domains": [
                {
                    "dataset": "gsm8k",
                    "selected_records": 32,
                    "error_records": 20,
                    "correct_records": 12,
                    "layers": [0, 1],
                    "hidden_dimension": 4096,
                    "response_generators": ["llama3.1-8b"],
                    "observer_models": ["llama3.1-8b"],
                }
            ],
        },
    )

    cli.main(["preflight", "--data-root", str(tmp_path / "data")])

    text = capsys.readouterr().out
    assert "preflight: run_id=preflight-123 | validated shards=32" in text
    assert "gsm8k: selected=32" in text
    assert "\"domains\"" not in text
