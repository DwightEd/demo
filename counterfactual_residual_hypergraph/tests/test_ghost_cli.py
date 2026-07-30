from __future__ import annotations

import json
from pathlib import Path

import pytest

from crwh.ghost_cli import build_parser, select_processbench_cohort
from crwh.ghost_hf import tokenize_chat_record


class CharacterTokenizer:
    is_fast = True

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        assert messages[0]["role"] == "user"
        prefix = "<prompt><assistant>"
        if len(messages) == 1:
            assert add_generation_prompt is True
            rendered = prefix
        else:
            assert add_generation_prompt is False
            response = messages[1]["content"]
            rendered = prefix + response + "<eot>"
        if tokenize:
            return list(range(len(rendered)))
        return rendered

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        return_tensors,
    ):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        assert return_tensors is None
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _write_processbench(path: Path, *, count: int = 12) -> None:
    rows = [
        {
            "id": f"trace-{index}",
            "problem_id": f"problem-{index}",
            "problem": f"Question {index}?",
            "steps": [f"step {index} a", f"step {index} b"],
            "label": -1 if index % 2 == 0 else 1,
            "generator": f"generator-{index % 3}",
        }
        for index in range(count)
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_balanced_selection_is_token_aware_and_writes_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "gsm8k.json"
    selected = tmp_path / "selected.jsonl"
    manifest = tmp_path / "cohort.json"
    _write_processbench(source)

    result = select_processbench_cohort(
        source=source,
        output=selected,
        manifest_path=manifest,
        tokenizer=CharacterTokenizer(),
        mode="balanced_unique",
        limit=8,
        max_tokens=128,
        seed=17,
    )

    rows = [
        json.loads(line)
        for line in selected.read_text(encoding="utf-8").splitlines()
    ]
    persisted = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(rows) == 8
    assert result == persisted
    assert persisted["response_class_counts"] == {"0": 4, "1": 4}
    assert persisted["selected_unique_problem_ids"] == 8
    assert persisted["truncation_policy"] == "forbidden"
    assert persisted["prompt_format"] == "observer_chat_template"


def test_cli_exposes_separate_select_extract_and_evaluate_stages() -> None:
    parser = build_parser()

    assert parser.parse_args(
        [
            "select",
            "--input",
            "data.json",
            "--model",
            "model",
            "--output",
            "selected.jsonl",
            "--manifest",
            "cohort.json",
        ]
    ).command == "select"
    assert parser.parse_args(
        [
            "extract",
            "--input",
            "selected.jsonl",
            "--model",
            "model",
            "--output",
            "embeddings.npz",
            "--manifest",
            "extract.json",
        ]
    ).command == "extract"
    assert parser.parse_args(
        [
            "evaluate",
            "--embeddings",
            "embeddings.npz",
            "--output-dir",
            "evaluation",
        ]
    ).command == "evaluate"


def test_chat_tokenization_fails_closed_when_response_boundary_changes() -> None:
    class BoundaryChangingTokenizer(CharacterTokenizer):
        def apply_chat_template(
            self,
            messages,
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ):
            values = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
            if tokenize and len(messages) == 2 and messages[1]["content"]:
                values[3] += 7
            return values

    record = type(
        "Record",
        (),
        {"question": "Question?", "steps": ("first step", "second step")},
    )()

    with pytest.raises(ValueError, match="canonical"):
        tokenize_chat_record(BoundaryChangingTokenizer(), record)
