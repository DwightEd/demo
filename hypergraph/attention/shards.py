#!/usr/bin/env python3
"""Audit deterministic attention-extraction shards without loading dense arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


SCOPE_KEYS = (
    "input_sha256",
    "input_num_rows",
    "pre_shard_num_rows",
    "selected_num_rows",
    "requested_limit",
    "num_shards",
    "shard_index",
    "skip_invalid",
    "max_seq_len",
    "max_attention_gib",
    "allow_large_attention",
)


def _strict_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _canonical_scope(scope: Mapping[str, Any]) -> tuple[Dict[str, Any], str, str]:
    missing = [key for key in SCOPE_KEYS if key not in scope]
    if missing:
        raise ValueError(f"extraction scope lacks required keys: {missing}")
    canonical = {key: scope[key] for key in SCOPE_KEYS}
    input_sha256 = str(canonical["input_sha256"])
    if len(input_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in input_sha256
    ):
        raise ValueError("input_sha256 must be a 64-character hexadecimal digest")
    input_num_rows = _strict_int(
        canonical["input_num_rows"], name="input_num_rows", minimum=1
    )
    pre_shard_num_rows = _strict_int(
        canonical["pre_shard_num_rows"], name="pre_shard_num_rows", minimum=1
    )
    selected_num_rows = _strict_int(
        canonical["selected_num_rows"], name="selected_num_rows", minimum=1
    )
    num_shards = _strict_int(canonical["num_shards"], name="num_shards", minimum=1)
    shard_index = _strict_int(canonical["shard_index"], name="shard_index")
    if shard_index >= num_shards:
        raise ValueError("shard_index must be smaller than num_shards")
    if pre_shard_num_rows > input_num_rows:
        raise ValueError("pre_shard_num_rows cannot exceed input_num_rows")
    requested_limit = canonical["requested_limit"]
    if requested_limit is not None:
        requested_limit = _strict_int(
            requested_limit, name="requested_limit", minimum=1
        )
        if pre_shard_num_rows != min(requested_limit, input_num_rows):
            raise ValueError("pre_shard_num_rows is inconsistent with requested_limit")
    elif pre_shard_num_rows != input_num_rows:
        raise ValueError("pre_shard_num_rows must equal input_num_rows without a limit")
    expected_selected = sum(
        index % num_shards == shard_index for index in range(pre_shard_num_rows)
    )
    if selected_num_rows != expected_selected:
        raise ValueError(
            "selected_num_rows is inconsistent with modulo sharding: "
            f"expected {expected_selected}, got {selected_num_rows}"
        )
    for key in ("skip_invalid", "allow_large_attention"):
        if not isinstance(canonical[key], bool):
            raise ValueError(f"{key} must be a JSON boolean")
    scope_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(scope_json.encode("utf-8")).hexdigest()
    return canonical, scope_json, fingerprint


def audit_scope_records(
    records: Sequence[Mapping[str, Any]],
    *,
    allow_incomplete: bool = False,
    allow_multiple_inputs: bool = False,
) -> Dict[str, Any]:
    """Validate shard cohort identity, membership, uniqueness, and coverage."""

    if not records:
        raise ValueError("no extraction-scope records were supplied")
    cohorts: Dict[str, Dict[str, Any]] = {}
    seen_source_rows: set[tuple[str, int]] = set()
    missing_metadata: list[str] = []

    for position, record in enumerate(records):
        trace_id = str(record.get("trace_id", f"record-{position}"))
        scope_value = record.get("scope")
        source_sha = record.get("source_input_sha256")
        source_row = record.get("source_row_index")
        scope_fingerprint = record.get("extraction_scope_fingerprint")
        if (
            not isinstance(scope_value, Mapping)
            or source_sha in (None, "")
            or source_row in (None, "")
            or scope_fingerprint in (None, "")
        ):
            missing_metadata.append(trace_id)
            continue
        scope, _, computed_fingerprint = _canonical_scope(scope_value)
        if str(scope_fingerprint) != computed_fingerprint:
            raise ValueError(
                f"{trace_id}: extraction_scope_fingerprint does not match scope"
            )
        input_sha = str(scope["input_sha256"])
        if str(source_sha) != input_sha:
            raise ValueError(f"{trace_id}: source_input_sha256 does not match scope")
        row_index = _strict_int(source_row, name=f"{trace_id} source_row_index")
        if row_index >= int(scope["pre_shard_num_rows"]):
            raise ValueError(
                f"{trace_id}: source row {row_index} is outside the selected input prefix"
            )
        if row_index % int(scope["num_shards"]) != int(scope["shard_index"]):
            raise ValueError(
                f"{trace_id}: source row {row_index} is not a member of shard "
                f"{scope['shard_index']}/{scope['num_shards']}"
            )
        source_key = (input_sha, row_index)
        if source_key in seen_source_rows:
            raise ValueError(
                f"duplicate source row input={input_sha} index={row_index}"
            )
        seen_source_rows.add(source_key)

        cohort = cohorts.setdefault(
            input_sha,
            {
                "signature": None,
                "scopes": {},
                "observed_rows": set(),
                "failed_rows": [],
            },
        )
        signature = {
            key: value
            for key, value in scope.items()
            if key not in {"selected_num_rows", "shard_index"}
        }
        if cohort["signature"] is None:
            cohort["signature"] = signature
        elif cohort["signature"] != signature:
            raise ValueError(
                f"input {input_sha} has incompatible extraction shard configurations"
            )
        shard_index = int(scope["shard_index"])
        existing_scope = cohort["scopes"].get(shard_index)
        if existing_scope is not None and existing_scope != scope:
            raise ValueError(
                f"input {input_sha} has conflicting declarations for shard {shard_index}"
            )
        cohort["scopes"][shard_index] = scope
        status = str(record.get("status", "ok"))
        if status == "ok":
            cohort["observed_rows"].add(row_index)
        else:
            cohort["failed_rows"].append(
                {"source_row_index": row_index, "status": status, "trace_id": trace_id}
            )

    if missing_metadata and not allow_incomplete:
        raise ValueError(
            f"{len(missing_metadata)} traces lack auditable extraction scope metadata; "
            "use --allow-incomplete-extraction-scope only for a diagnostic"
        )
    if len(cohorts) > 1 and not allow_multiple_inputs:
        raise ValueError(
            "multiple source input SHA256 values would be mixed; pass "
            "--allow-multiple-input-datasets only for a declared combined-dataset run"
        )

    reports = []
    globally_complete = not missing_metadata
    for input_sha, cohort in sorted(cohorts.items()):
        signature = cohort["signature"]
        expected_rows = set(range(int(signature["pre_shard_num_rows"])))
        observed_rows = set(cohort["observed_rows"])
        missing_rows = sorted(expected_rows - observed_rows)
        extra_rows = sorted(observed_rows - expected_rows)
        expected_shards = set(range(int(signature["num_shards"])))
        observed_shards = set(cohort["scopes"])
        missing_shards = sorted(expected_shards - observed_shards)
        complete = not (
            missing_rows or extra_rows or missing_shards or cohort["failed_rows"]
        )
        globally_complete = globally_complete and complete
        reports.append(
            {
                "input_sha256": input_sha,
                "input_num_rows": int(signature["input_num_rows"]),
                "expected_rows": len(expected_rows),
                "observed_rows": len(observed_rows),
                "num_shards": int(signature["num_shards"]),
                "observed_shards": sorted(observed_shards),
                "missing_shards": missing_shards,
                "missing_row_count": len(missing_rows),
                "missing_row_examples": missing_rows[:20],
                "extra_rows": extra_rows[:20],
                "failed_rows": cohort["failed_rows"][:20],
                "complete": complete,
            }
        )

    if not globally_complete and not allow_incomplete:
        raise ValueError(
            "extraction shard cohort is incomplete; inspect the shard audit and use "
            "--allow-incomplete-extraction-scope only for a diagnostic"
        )
    return {
        "complete": bool(globally_complete),
        "incomplete_explicitly_allowed": bool(
            not globally_complete and allow_incomplete
        ),
        "multiple_inputs_explicitly_allowed": bool(
            len(cohorts) > 1 and allow_multiple_inputs
        ),
        "missing_metadata_count": len(missing_metadata),
        "missing_metadata_examples": missing_metadata[:20],
        "cohorts": reports,
    }


def audit_extraction_manifests(
    inputs: Sequence[str],
    *,
    allow_incomplete: bool = False,
    allow_multiple_inputs: bool = False,
) -> Dict[str, Any]:
    records = []
    manifests = []
    for raw_input in inputs:
        path = Path(raw_input).expanduser().resolve()
        if path.is_dir():
            path = path / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"extraction manifest does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("extraction_config")
        traces = payload.get("traces")
        if not isinstance(config, Mapping) or not isinstance(traces, list):
            raise ValueError(f"{path}: invalid extraction manifest structure")
        scope = {key: config.get(key) for key in SCOPE_KEYS}
        scope_fingerprint = payload.get(
            "extraction_scope_fingerprint",
            config.get("extraction_scope_fingerprint"),
        )
        for item in traces:
            if not isinstance(item, Mapping):
                raise ValueError(f"{path}: every trace manifest row must be an object")
            status = str(item.get("status", ""))
            row_index = item.get("index")
            trace_id = str(item.get("sample_id", f"row-{row_index}"))
            if status == "ok":
                filename = item.get("file")
                if not isinstance(filename, str) or not (path.parent / filename).is_file():
                    raise ValueError(f"{path}: missing trace file for {trace_id}")
            records.append(
                {
                    "trace_id": trace_id,
                    "scope": scope,
                    "source_input_sha256": config.get("input_sha256"),
                    "source_row_index": row_index,
                    "extraction_scope_fingerprint": scope_fingerprint,
                    "status": status,
                }
            )
        manifests.append(str(path))
    result = audit_scope_records(
        records,
        allow_incomplete=allow_incomplete,
        allow_multiple_inputs=allow_multiple_inputs,
    )
    result["manifests"] = manifests
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="extraction directories or manifest.json files")
    parser.add_argument("--output", help="optional JSON audit output")
    parser.add_argument("--allow-incomplete-extraction-scope", action="store_true")
    parser.add_argument("--allow-multiple-input-datasets", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = audit_extraction_manifests(
            args.inputs,
            allow_incomplete=bool(args.allow_incomplete_extraction_scope),
            allow_multiple_inputs=bool(args.allow_multiple_input_datasets),
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, output)
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
