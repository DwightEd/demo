from __future__ import annotations

import json

from crwh.cli import main


def test_synthetic_cli_runs_end_to_end_and_writes_auditable_artifacts(
    tmp_path,
) -> None:
    output_dir = tmp_path / "smoke"

    exit_code = main(
        [
            "synthetic",
            "--output-dir",
            str(output_dir),
            "--examples",
            "8",
            "--epochs",
            "1",
            "--seed",
            "11",
        ]
    )

    assert exit_code == 0
    metrics = json.loads((output_dir / "metrics.json").read_text())
    scores = json.loads((output_dir / "scores.json").read_text())
    config = json.loads((output_dir / "config.json").read_text())

    assert metrics["examples"] == 8
    assert metrics["finite_scores"] is True
    assert len(scores) == 16
    assert config["seed"] == 11
    assert (output_dir / "review_queue.json").exists()
