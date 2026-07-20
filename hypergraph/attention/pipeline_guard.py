"""Fail-closed request gates for reusable attention extraction caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .trace_contract import model_identity_matches


TRACE_REQUEST_SCHEMA = "attention_trace_pipeline_request_v2"
LEGACY_CODE_KEY = "method_code_sha256"


def dataset_generator_matches_observer(generator: str, observer: str) -> bool:
    """Match a dataset tag to the observer with one curated Meta-Llama alias.

    This is an orchestration policy rather than trace-extraction semantics, so it
    deliberately lives outside :mod:`trace_contract` and its extraction hash.
    """

    if model_identity_matches(generator, observer):
        return True
    generator_leaf = str(generator).strip().replace("\\", "/").rstrip("/").lower()
    observer_leaf = str(observer).strip().replace("\\", "/").rstrip("/").lower()
    generator_leaf = generator_leaf.rsplit("/", 1)[-1]
    observer_leaf = observer_leaf.rsplit("/", 1)[-1]
    if observer_leaf.startswith("meta-llama-"):
        observer_leaf = observer_leaf[len("meta-") :]
    if generator_leaf.startswith("meta-llama-"):
        generator_leaf = generator_leaf[len("meta-") :]
    return bool(generator_leaf and generator_leaf == observer_leaf)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_digest(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA256")
    return normalized


def _request_from_items(items: Sequence[str]) -> dict[str, str]:
    request: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid request entry: {item!r}")
        key, value = item.split("=", 1)
        if not key or key in request:
            raise ValueError(f"duplicate or empty request key: {key!r}")
        request[key] = value
    if not request:
        raise ValueError("trace request cannot be empty")
    return request


def _differences(
    stored: Mapping[str, Any], requested: Mapping[str, Any]
) -> list[str]:
    return [
        f"  {key}: stored={stored.get(key)!r}, requested={requested.get(key)!r}"
        for key in sorted(set(stored) | set(requested))
        if stored.get(key) != requested.get(key)
    ]


def validate_or_initialize_trace_request(
    path: str | Path,
    *,
    request: Mapping[str, str],
    extraction_code_sha256: str,
) -> dict[str, Any]:
    """Validate a v2 request or safely admit an unchanged legacy request.

    Legacy files are never rewritten.  Their monolithic code hash is retained as
    historical producer provenance, while all effective extraction fields must
    exactly match.  Callers must additionally re-audit manifests and the selected
    training cohort before reuse.
    """

    destination = Path(path).expanduser().resolve()
    code_digest = _validate_digest(
        extraction_code_sha256, name="extraction_code_sha256"
    )
    normalized_request = {str(key): str(value) for key, value in request.items()}
    if not normalized_request or any(not key for key in normalized_request):
        raise ValueError("trace request must contain non-empty string keys")
    v2_payload = {
        "schema": TRACE_REQUEST_SCHEMA,
        "extraction_code_sha256": code_digest,
        "request": normalized_request,
    }

    if not destination.is_file():
        if destination.parent.exists() and any(destination.parent.iterdir()):
            raise ValueError(
                f"non-empty trace directory has no request gate: {destination.parent}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(
            v2_payload, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, destination)
        return {
            "mode": "initialized_v2",
            "request_file_sha256": _sha256_bytes(destination.read_bytes()),
            "legacy_method_code_sha256": None,
        }

    raw = destination.read_bytes()
    try:
        stored = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid trace request JSON: {destination}") from exc
    if not isinstance(stored, Mapping):
        raise ValueError(f"trace request must be a JSON object: {destination}")

    if stored.get("schema") == TRACE_REQUEST_SCHEMA:
        differences = _differences(stored, v2_payload)
        if differences:
            raise ValueError(
                f"trace request mismatch for {destination}:\n" + "\n".join(differences)
            )
        mode = "validated_v2"
        legacy_digest = None
    elif LEGACY_CODE_KEY in stored and "schema" not in stored:
        legacy_digest = _validate_digest(
            str(stored[LEGACY_CODE_KEY]), name=f"legacy {LEGACY_CODE_KEY}"
        )
        effective = {
            str(key): str(value)
            for key, value in stored.items()
            if key != LEGACY_CODE_KEY
        }
        differences = _differences(effective, normalized_request)
        if differences:
            raise ValueError(
                f"legacy trace request mismatch for {destination}:\n"
                + "\n".join(differences)
            )
        mode = "validated_legacy_without_rewrite"
    else:
        raise ValueError(
            f"unsupported trace request schema for {destination}; refusing cache reuse"
        )

    return {
        "mode": mode,
        "request_file_sha256": _sha256_bytes(raw),
        "legacy_method_code_sha256": legacy_digest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--extraction-code-sha256", required=True)
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("request", nargs="+")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_or_initialize_trace_request(
            args.path,
            request=_request_from_items(args.request),
            extraction_code_sha256=args.extraction_code_sha256,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.shell:
        print(
            result["mode"],
            result["request_file_sha256"],
            result["legacy_method_code_sha256"] or "-",
        )
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "LEGACY_CODE_KEY",
    "TRACE_REQUEST_SCHEMA",
    "build_parser",
    "dataset_generator_matches_observer",
    "main",
    "validate_or_initialize_trace_request",
]
