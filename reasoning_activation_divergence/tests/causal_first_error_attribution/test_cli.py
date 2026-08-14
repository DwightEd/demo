from __future__ import annotations

import numpy as np
import pytest

from functional_divergence.causal_first_error_attribution import main as cli


def test_cli_exposes_separate_audit_extract_intervene_and_evaluate_commands(
    tmp_path,
) -> None:
    parser = cli.build_parser()
    extract = parser.parse_args(
        [
            "extract",
            "--data-root",
            str(tmp_path),
            "--model-dir",
            str(tmp_path / "model"),
            "--layers",
            "8,12",
        ]
    )
    intervene = parser.parse_args(
        [
            "intervene",
            "--data-root",
            str(tmp_path),
            "--model-dir",
            str(tmp_path / "model"),
            "--layers",
            "8,12",
        ]
    )

    assert extract.layers == (8, 12)
    assert intervene.layers == (8, 12)
    assert parser.parse_args(["summarize", "--data-root", str(tmp_path)]).command == (
        "summarize"
    )
    assert parser.parse_args(
        ["evaluate-monitor", "--scores", str(tmp_path / "scores.jsonl")]
    ).command == "evaluate-monitor"


def test_cli_exposes_a_direct_processbench_monitor_training_command(tmp_path) -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "train-monitor",
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "results"),
            "--arms",
            "output_history,static_layer_set,two_boundary_bag,two_boundary_innovation",
            "--epochs",
            "3",
            "--device",
            "cpu",
        ]
    )

    assert args.command == "train-monitor"
    assert args.arms == (
        "output_history",
        "static_layer_set",
        "two_boundary_bag",
        "two_boundary_innovation",
    )
    assert args.epochs == 3
    assert args.device == "cpu"


def test_cli_exposes_source_message_fisher_experiment(tmp_path) -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "message-fisher",
            "--data-root",
            str(tmp_path / "data"),
            "--model-dir",
            str(tmp_path / "model"),
            "--layers",
            "8,16",
            "--output-dir",
            str(tmp_path / "results"),
            "--epsilon",
            "0.05",
            "--perturbation-batch-size",
            "3",
            "--bootstrap",
            "200",
        ]
    )

    assert args.command == "message-fisher"
    assert args.layers == (8, 16)
    assert args.epsilon == 0.05
    assert args.perturbation_batch_size == 3
    assert args.bootstrap == 200


def test_extract_checks_pair_availability_before_loading_model(
    tmp_path, monkeypatch, capsys
) -> None:
    selected = tmp_path / "gsm8k" / "selected"
    selected.mkdir(parents=True)
    np.savez_compressed(
        selected / "trace.npz",
        gold_error_step=np.asarray([0, -1], dtype=np.int32),
        problem_ids=np.asarray(["error", "correct"]),
    )

    def forbidden_model_load(_args):
        raise AssertionError("model must not load without verified onset pairs")

    monkeypatch.setattr(cli, "_load_model", forbidden_model_load)

    with pytest.raises(SystemExit, match="no verified onset pairs"):
        cli.main(
            [
                "extract",
                "--data-root",
                str(tmp_path),
                "--domains",
                "gsm8k",
                "--model-dir",
                str(tmp_path / "model"),
                "--layers",
                "8,12",
            ]
        )

    assert "run causal-audit first" in capsys.readouterr().out


@pytest.mark.parametrize("layers", ["0", "8,8", "-1,8"])
def test_cli_rejects_invalid_layers_before_execution(tmp_path, layers) -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "extract",
                "--data-root",
                str(tmp_path),
                "--model-dir",
                str(tmp_path / "model"),
                "--layers",
                layers,
            ]
        )


def test_cli_rejects_negative_case_limit(tmp_path) -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "extract",
                "--data-root",
                str(tmp_path),
                "--model-dir",
                str(tmp_path / "model"),
                "--layers",
                "8",
                "--max-cases-per-domain",
                "-1",
            ]
        )
