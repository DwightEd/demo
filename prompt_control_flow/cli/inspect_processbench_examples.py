from __future__ import annotations

import argparse
import csv
import io
import json
import os
from html import escape
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

CSV_FIELDS = (
    "dataset",
    "selected_index",
    "chain_idx",
    "problem_id",
    "generator",
    "process_correct",
    "final_answer_correct",
    "gold_error_step",
    "n_steps",
    "problem",
    "step_index",
    "is_first_error",
    "step_text",
)

HTML_STYLE = """
:root {
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  color: #18202a;
  background: #eef1f4;
}
* { box-sizing: border-box; }
body { margin: 0; background: #eef1f4; }
main { width: min(1180px, calc(100% - 32px)); margin: 28px auto 56px; }
header { border-top: 5px solid #1f6f78; padding: 24px 0 18px; }
h1 { margin: 0 0 8px; font-size: 30px; line-height: 1.2; }
h2 { margin: 0; font-size: 21px; line-height: 1.3; }
p { line-height: 1.58; }
.subtitle { margin: 0; color: #536170; }
.example { margin-top: 22px; background: #fff; border: 1px solid #d6dce2; border-radius: 6px; overflow: hidden; }
.example-heading { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 16px 18px; border-bottom: 1px solid #d6dce2; background: #f8fafb; }
.badge { display: inline-block; padding: 4px 8px; border-radius: 4px; background: #dceef0; color: #15515a; font-size: 13px; font-weight: 700; }
.badge.error { background: #f9dfdf; color: #8a2020; }
.content { padding: 18px; }
.problem { margin: 8px 0 18px; padding: 14px 16px; border-left: 4px solid #d69b2d; background: #fffaf0; white-space: pre-wrap; }
table { width: 100%; border-collapse: collapse; table-layout: fixed; }
th, td { padding: 10px 12px; border: 1px solid #dfe4e8; text-align: left; vertical-align: top; line-height: 1.48; }
th { background: #f3f5f7; color: #36414d; font-size: 13px; }
.metadata th { width: 180px; }
.steps th:first-child, .steps td:first-child { width: 86px; text-align: center; }
.steps td:last-child { white-space: pre-wrap; overflow-wrap: anywhere; }
.steps tr.first-error td { background: #fff0f0; border-color: #e7b9b9; }
.error-label { color: #9b2525; font-weight: 700; }
@media (max-width: 720px) {
  main { width: min(100% - 20px, 1180px); margin-top: 12px; }
  .example-heading { align-items: flex-start; flex-direction: column; }
  .metadata th { width: 128px; }
  th, td { padding: 8px; }
}
""".strip()


def _contains_subset(path: Path, subset: str) -> bool:
    return any((path / f"{subset}{suffix}").is_file() for suffix in (".json", ".jsonl"))


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


def render_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _step_rows(payload: dict[str, Any]):
    for example in payload["examples"]:
        common = {
            "dataset": example["dataset"],
            "selected_index": example["selected_index"],
            "chain_idx": example["chain_idx"],
            "problem_id": example["problem_id"],
            "generator": example["generator"] or "",
            "process_correct": example["process_correct"],
            "final_answer_correct": example["final_answer_correct"],
            "gold_error_step": example["gold_error_step"],
            "n_steps": example["n_steps"],
            "problem": example["problem"],
        }
        for step in example["steps"]:
            yield {
                **common,
                "step_index": step["index"],
                "is_first_error": step["is_first_error"],
                "step_text": step["text"],
            }


def render_csv(payload: dict[str, Any]) -> str:
    """Render one row per reasoning step for spreadsheet inspection."""

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(_step_rows(payload))
    return stream.getvalue()


def _html_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def render_html(payload: dict[str, Any]) -> str:
    """Render a self-contained report with first-error rows highlighted."""

    selection = payload["selection"]
    sections: list[str] = []
    for example in payload["examples"]:
        title = SUBSET_TITLES.get(example["dataset"], example["dataset"])
        is_error = example["gold_error_step"] >= 0
        badge_class = "badge error" if is_error else "badge"
        badge_text = (
            "First error: step " + str(example["gold_error_step"])
            if is_error
            else "All steps correct"
        )
        metadata = (
            ("Generator", example["generator"] or "unknown"),
            ("Chain index", example["chain_idx"]),
            ("Problem ID", example["problem_id"]),
            ("Process correct", _html_value(example["process_correct"])),
            ("Final answer correct", _html_value(example["final_answer_correct"])),
            ("Number of steps", example["n_steps"]),
        )
        metadata_rows = "".join(
            f"<tr><th>{escape(str(label))}</th><td>{escape(str(value))}</td></tr>"
            for label, value in metadata
        )
        step_rows: list[str] = []
        for step in example["steps"]:
            row_class = ' class="first-error"' if step["is_first_error"] else ""
            step_label = (
                f'<span class="error-label">{step["index"]}<br>First error</span>'
                if step["is_first_error"]
                else str(step["index"])
            )
            step_rows.append(
                f"<tr{row_class}><td>{step_label}</td>"
                f"<td>{escape(str(step['text']))}</td></tr>"
            )
        sections.append(
            '<section class="example">'
            '<div class="example-heading">'
            f"<h2>{escape(str(title))}</h2>"
            f'<span class="{badge_class}">{escape(badge_text)}</span>'
            '</div><div class="content">'
            f'<table class="metadata"><tbody>{metadata_rows}</tbody></table>'
            "<h3>Problem</h3>"
            f"<div class=\"problem\">{escape(str(example['problem']))}</div>"
            "<h3>Reasoning steps</h3>"
            '<table class="steps"><thead><tr><th>Step</th><th>Text</th>'
            f"</tr></thead><tbody>{''.join(step_rows)}</tbody></table>"
            "</div></section>"
        )

    data_dir = escape(str(payload["data_dir"]))
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>ProcessBench Example Report</title>"
        f"<style>{HTML_STYLE}</style></head><body><main>"
        "<header><h1>ProcessBench Example Report</h1>"
        f"<p class=\"subtitle\">Selection: {escape(str(selection['kind']))}, "
        f"index {selection['index']} | Source: {data_dir}</p></header>"
        f"{''.join(sections)}</main></body></html>\n"
    )


def render_payload(payload: dict[str, Any], output_format: str) -> str:
    renderers = {
        "text": render_text,
        "json": render_json,
        "csv": render_csv,
        "html": render_html,
    }
    try:
        renderer = renderers[output_format]
    except KeyError as exc:
        raise ValueError(f"unsupported output format {output_format!r}") from exc
    return renderer(payload)


def _write_rendered(path: Path, content: str, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = "utf-8-sig" if output_format == "csv" else "utf-8"
    path.write_text(content, encoding=encoding)


def write_report_bundle(
    payload: dict[str, Any], output_dir: str | Path
) -> dict[str, Path]:
    directory = Path(output_dir)
    selection = payload["selection"]
    basename = (
        f"processbench_examples_{selection['kind']}_{int(selection['index']):04d}"
    )
    paths: dict[str, Path] = {}
    for output_format in ("json", "csv", "html"):
        path = directory / f"{basename}.{output_format}"
        _write_rendered(path, render_payload(payload, output_format), output_format)
        paths[output_format] = path
    return paths


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
        choices=("text", "json", "csv", "html"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output file; the same content is also printed to stdout.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Optional report directory; writes JSON, CSV, and HTML together.",
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
    rendered = render_payload(payload, args.output_format)
    bundle_paths: dict[str, Path] = {}
    if args.output_dir:
        bundle_paths = write_report_bundle(payload, args.output_dir)
        for output_format, path in bundle_paths.items():
            print(f"saved {output_format}: {path}")
    else:
        print(rendered, end="")
    if args.output:
        output = Path(args.output)
        _write_rendered(output, rendered, args.output_format)


if __name__ == "__main__":
    main()
