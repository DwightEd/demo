from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from prompt_control_flow.data import ChainRecord, load_chain_records


DEFAULT_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
SUBSET_TITLES = {
    "gsm8k": "GSM8K",
    "math": "MATH",
    "olympiadbench": "OlympiadBench",
    "omnimath": "Omni-MATH",
}


def _contains_subset(path: Path, subset: str) -> bool:
    return any(
        (path / f"{subset}{suffix}").is_file()
        for suffix in (".json", ".jsonl")
    )


def resolve_data_dir(
    data_dir: str | Path | None,
    subsets: Sequence[str] = DEFAULT_SUBSETS,
) -> Path:
    """Resolve a canonical ProcessBench directory with explicit diagnostics."""

    repo_root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    if data_dir:
        candidates.append(Path(data_dir).expanduser())
    else:
        environment_path = os.environ.get("PROCESSBENCH_DATA_DIR")
        if environment_path:
            candidates.append(Path(environment_path).expanduser())
        candidates.extend(
            [
                Path.cwd() / "data" / "hf_datasets" / "ProcessBench",
                Path.cwd() / "data" / "processbench",
                repo_root / "data" / "hf_datasets" / "ProcessBench",
                repo_root / "data" / "processbench",
                repo_root.parent / "data" / "processbench",
            ]
        )

    checked: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved).casefold()
        if key in seen:
            continue
        seen.add(key)
        checked.append(str(resolved))
        if resolved.is_dir() and all(
            _contains_subset(resolved, subset) for subset in subsets
        ):
            return resolved

    expected = ", ".join(f"{subset}.json[.l]" for subset in subsets)
    locations = "\n  - ".join(checked) if checked else "<none>"
    raise FileNotFoundError(
        "Could not find a ProcessBench directory containing "
        f"{expected}. Checked:\n  - {locations}\n"
        "Pass --data_dir or set PROCESSBENCH_DATA_DIR."
    )


def _matches_kind(record: ChainRecord, kind: str) -> bool:
    if kind == "any":
        return True
    if kind == "error":
        return record.gold_error_step >= 0
    if kind == "correct":
        return record.gold_error_step == -1
    raise ValueError(f"unknown selection kind={kind!r}")


def _record_payload(record: ChainRecord, *, selected_index: int) -> dict[str, Any]:
    first_error = int(record.gold_error_step)
    if first_error == -1:
        label_description = "all reasoning steps are correct"
    else:
        label_description = f"first erroneous step is {first_error} (0-based)"

    return {
        "dataset": str(record.dataset or ""),
        "selected_index": int(selected_index),
        "chain_idx": int(record.chain_idx),
        "problem_id": int(record.problem_id),
        "generator": record.generator,
        "gold_error_step": first_error,
        "process_correct": (
            None if record.process_correct is None else bool(record.process_correct)
        ),
        "final_answer_correct": (
            None
            if record.final_answer_correct is None
            else bool(record.final_answer_correct)
        ),
        "label_description": label_description,
        "n_steps": len(record.steps),
        "problem": record.problem,
        "response": record.response,
        "steps": [
            {
                "index": step_index,
                "text": step,
                "is_first_error": step_index == first_error,
            }
            for step_index, step in enumerate(record.steps)
        ],
    }


def collect_examples(
    data_dir: str | Path | None,
    subsets: Sequence[str] = DEFAULT_SUBSETS,
    *,
    kind: str = "any",
    index: int = 0,
) -> dict[str, Any]:
    """Select one record from each subset after applying a process-label filter."""

    if index < 0:
        raise ValueError("index must be non-negative")
    resolved = resolve_data_dir(data_dir, subsets)
    examples: list[dict[str, Any]] = []
    for subset in subsets:
        records = load_chain_records(
            resolved,
            input_format="processbench_source",
            subset=subset,
        )
        selected = [record for record in records if _matches_kind(record, kind)]
        if index >= len(selected):
            noun = "record" if len(selected) == 1 else "records"
            raise IndexError(
                f"ProcessBench/{subset} contains {len(selected)} {kind} {noun}; "
                f"cannot select filtered index {index}"
            )
        examples.append(_record_payload(selected[index], selected_index=index))

    return {
        "data_dir": str(resolved),
        "selection": {"kind": kind, "index": int(index)},
        "examples": examples,
    }


def _indented_lines(value: str, prefix: str = "    ") -> list[str]:
    lines = str(value).splitlines() or [""]
    return [prefix + line for line in lines]


def render_text(payload: dict[str, Any]) -> str:
    """Render selected examples without hiding multiline reasoning steps."""

    selection = payload["selection"]
    lines = [
        "ProcessBench examples",
        f"data_dir: {payload['data_dir']}",
        f"selection: kind={selection['kind']} index={selection['index']}",
    ]
    for example in payload["examples"]:
        title = SUBSET_TITLES.get(example["dataset"], example["dataset"])
        lines.extend(
            [
                "",
                "=" * 80,
                f"{title} | chain_idx={example['chain_idx']} | "
                f"problem_id={example['problem_id']}",
                f"generator: {example['generator'] or 'unknown'}",
                f"gold_error_step: {example['gold_error_step']} "
                f"({example['label_description']})",
                f"process_correct: {example['process_correct']}",
                f"final_answer_correct: {example['final_answer_correct']}",
                f"n_steps: {example['n_steps']}",
                "",
                "Problem:",
            ]
        )
        lines.extend(_indented_lines(example["problem"]))
        lines.extend(["", "Response steps:"])
        for step in example["steps"]:
            marker = " <-- FIRST ERROR" if step["is_first_error"] else ""
            lines.append(f"  [{step['index']}]{marker}")
            lines.extend(_indented_lines(step["text"]))
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print one canonical ProcessBench example from each requested subset, "
            "including its process label and first-error marker."
        )
    )
    parser.add_argument(
        "--data_dir",
        default="",
        help=(
            "Directory containing ProcessBench subset .json/.jsonl files. "
            "When omitted, use PROCESSBENCH_DATA_DIR or known project locations."
        ),
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=DEFAULT_SUBSETS,
        default=list(DEFAULT_SUBSETS),
        help="Subsets to display (default: all four).",
    )
    parser.add_argument(
        "--kind",
        choices=("any", "error", "correct"),
        default="any",
        help="Filter by ProcessBench process label before applying --index.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based record index after filtering, shared across subsets.",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output file; the same content is also printed to stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    payload = collect_examples(
        args.data_dir or None,
        tuple(args.subsets),
        kind=args.kind,
        index=args.index,
    )
    if args.output_format == "json":
        rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    else:
        rendered = render_text(payload)
    print(rendered, end="")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
