"""Materialize auditable ProcessBench generator cohorts before GPU extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .extract import canonical_record, load_processbench_rows


COHORT_SCHEMA = "processbench_generator_cohort_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != content:
            raise FileExistsError(
                f"refusing to overwrite a different cohort artifact: {path}"
            )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _validate_write_target(path: Path, content: bytes) -> None:
    """Fail before either member of the cohort/report pair is changed."""

    if path.is_file() and path.read_bytes() != content:
        raise FileExistsError(
            f"refusing to overwrite a different cohort artifact: {path}"
        )


def materialize_generator_cohort(
    input_path: str | Path,
    output_path: str | Path,
    *,
    generator_model: str,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Filter by the dataset's exact generator tag while preserving row order."""

    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    selector = str(generator_model).strip()
    if not selector:
        raise ValueError("generator_model cannot be empty")
    rows = load_processbench_rows(str(source))
    selected: list[Mapping[str, Any]] = []
    selected_indices: list[int] = []
    input_generators: Counter[str] = Counter()
    selected_labels: Counter[str] = Counter()
    selector_key = selector.casefold()
    for index, row in enumerate(rows):
        record = canonical_record(row, index)
        generator = record.get("generator_model")
        generator_label = "<missing>" if generator in (None, "") else str(generator)
        input_generators[generator_label] += 1
        if generator_label.strip().casefold() != selector_key:
            continue
        selected.append(row)
        selected_indices.append(index)
        selected_labels["positive" if float(record["response_y"]) >= 0.5 else "negative"] += 1
    if not selected:
        raise ValueError(
            f"no ProcessBench rows match generator_model={selector!r} in {source}"
        )

    rendered = (
        json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    cohort_sha256 = hashlib.sha256(rendered).hexdigest()
    report = {
        "schema": COHORT_SCHEMA,
        "source_input": str(source),
        "source_input_sha256": _sha256_file(source),
        "cohort_output": str(destination),
        "cohort_sha256": cohort_sha256,
        "generator_model": selector,
        "match_policy": "exact_case_insensitive_dataset_tag",
        "num_input_rows": len(rows),
        "num_selected_rows": len(selected),
        "source_row_indices": selected_indices,
        "generator_distribution_input": dict(sorted(input_generators.items())),
        "response_label_counts_selected": {
            key: int(selected_labels.get(key, 0))
            for key in ("negative", "positive")
        },
    }
    report_destination = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else destination.with_suffix(destination.suffix + ".report.json")
    )
    if report_destination == destination:
        raise ValueError("cohort output and report must use different paths")
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _validate_write_target(destination, rendered)
    _validate_write_target(report_destination, report_bytes)
    _atomic_write(destination, rendered)
    _atomic_write(report_destination, report_bytes)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--report")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = materialize_generator_cohort(
            args.input,
            args.output,
            generator_model=args.generator_model,
            report_path=args.report,
        )
    except (FileNotFoundError, FileExistsError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COHORT_SCHEMA",
    "build_parser",
    "main",
    "materialize_generator_cohort",
]
