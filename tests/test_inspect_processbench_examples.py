from __future__ import annotations

import csv
import io
import json

import pytest

from prompt_control_flow.cli.inspect_processbench_examples import (
    DEFAULT_SUBSETS,
    collect_examples,
    main,
    render_csv,
    render_html,
    render_text,
)


def _row(subset: str, *, label: int, suffix: str) -> dict:
    return {
        "id": f"{subset}-{suffix}",
        "generator": "test-generator",
        "problem": f"Problem for {subset} ({suffix})",
        "steps": [f"{subset} step 0", f"{subset} step 1"],
        "final_answer_correct": label == -1,
        "label": label,
    }


def _write_fixture(data_dir) -> None:
    data_dir.mkdir()
    for position, subset in enumerate(DEFAULT_SUBSETS):
        rows = [
            _row(subset, label=-1, suffix="correct"),
            _row(subset, label=1, suffix="error"),
        ]
        if position % 2 == 0:
            (data_dir / f"{subset}.json").write_text(json.dumps(rows), encoding="utf-8")
        else:
            (data_dir / f"{subset}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )


def test_collects_one_error_example_per_processbench_subset(tmp_path):
    data_dir = tmp_path / "ProcessBench"
    _write_fixture(data_dir)

    payload = collect_examples(data_dir, DEFAULT_SUBSETS, kind="error", index=0)

    assert [example["dataset"] for example in payload["examples"]] == list(
        DEFAULT_SUBSETS
    )
    for example in payload["examples"]:
        assert example["gold_error_step"] == 1
        assert example["process_correct"] is False
        assert example["steps"][0]["is_first_error"] is False
        assert example["steps"][1]["is_first_error"] is True

    rendered = render_text(payload)
    assert rendered.count("<-- FIRST ERROR") == len(DEFAULT_SUBSETS)
    assert "Problem for gsm8k (error)" in rendered


def test_correct_filter_uses_process_label_not_only_final_answer(tmp_path):
    data_dir = tmp_path / "ProcessBench"
    _write_fixture(data_dir)

    payload = collect_examples(data_dir, ("gsm8k",), kind="correct", index=0)

    example = payload["examples"][0]
    assert example["gold_error_step"] == -1
    assert example["process_correct"] is True
    assert all(not step["is_first_error"] for step in example["steps"])


def test_cli_can_emit_machine_readable_json(tmp_path, capsys):
    data_dir = tmp_path / "ProcessBench"
    _write_fixture(data_dir)

    main(
        [
            "--data_dir",
            str(data_dir),
            "--subsets",
            "gsm8k",
            "math",
            "--kind",
            "any",
            "--format",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["selection"] == {"kind": "any", "index": 0}
    assert [row["dataset"] for row in payload["examples"]] == ["gsm8k", "math"]


def test_csv_expands_each_reasoning_step_and_marks_first_error(tmp_path):
    data_dir = tmp_path / "ProcessBench"
    _write_fixture(data_dir)
    payload = collect_examples(data_dir, ("gsm8k",), kind="error", index=0)

    rows = list(csv.DictReader(io.StringIO(render_csv(payload))))

    assert len(rows) == 2
    assert rows[0]["dataset"] == "gsm8k"
    assert rows[0]["step_index"] == "0"
    assert rows[0]["is_first_error"] == "False"
    assert rows[1]["step_index"] == "1"
    assert rows[1]["is_first_error"] == "True"
    assert rows[1]["step_text"] == "gsm8k step 1"


def test_html_is_self_contained_escaped_and_highlights_first_error(tmp_path):
    data_dir = tmp_path / "ProcessBench"
    _write_fixture(data_dir)
    payload = collect_examples(data_dir, ("gsm8k",), kind="error", index=0)
    payload["examples"][0]["problem"] = "Find <x> & verify it."

    rendered = render_html(payload)

    assert "<!doctype html>" in rendered.lower()
    assert "Find &lt;x&gt; &amp; verify it." in rendered
    assert "Find <x> & verify it." not in rendered
    assert 'class="first-error"' in rendered
    assert "First error" in rendered


def test_cli_output_dir_writes_json_csv_and_html_bundle(tmp_path, capsys):
    data_dir = tmp_path / "ProcessBench"
    _write_fixture(data_dir)
    output_dir = tmp_path / "report"

    main(
        [
            "--data_dir",
            str(data_dir),
            "--subsets",
            "gsm8k",
            "--kind",
            "error",
            "--output_dir",
            str(output_dir),
        ]
    )

    stdout = capsys.readouterr().out
    basename = "processbench_examples_error_0000"
    json_path = output_dir / f"{basename}.json"
    csv_path = output_dir / f"{basename}.csv"
    html_path = output_dir / f"{basename}.html"
    assert json_path.is_file()
    assert csv_path.is_file()
    assert html_path.is_file()
    assert str(json_path) in stdout
    assert str(csv_path) in stdout
    assert str(html_path) in stdout
    assert (
        json.loads(json_path.read_text(encoding="utf-8"))["examples"][0]["dataset"]
        == "gsm8k"
    )
    assert "gsm8k step 1" in csv_path.read_text(encoding="utf-8-sig")
    assert "ProcessBench Example Report" in html_path.read_text(encoding="utf-8")


def test_filtered_index_reports_available_count(tmp_path):
    data_dir = tmp_path / "ProcessBench"
    _write_fixture(data_dir)

    with pytest.raises(IndexError, match="contains 1 error record"):
        collect_examples(data_dir, ("gsm8k",), kind="error", index=2)
