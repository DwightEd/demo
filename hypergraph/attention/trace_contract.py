"""Stable identity primitives shared by trace extraction and validation.

This module is deliberately small.  Extraction hashes it directly, while
training-only changes in :mod:`hypergraph.attention.data` must not invalidate
an otherwise compatible attention cache.
"""

from __future__ import annotations

import re
from typing import Any


TRACE_CONTRACT = "exact_prompt_response_attention_v1"
MODEL_COMMIT_SOURCES = frozenset(
    {
        "remote_resolved_model_commit",
        "remote_pinned_requested_commit",
        "remote_tokenizer_metadata_only",
        "local_model_metadata_commit",
        "local_tokenizer_metadata_only",
        "local_declared_commit",
        "unavailable",
    }
)
VERIFIED_MODEL_COMMIT_SOURCES = frozenset(
    {
        "remote_resolved_model_commit",
        "remote_pinned_requested_commit",
    }
)


def model_identity_matches(source: str, requested: str) -> bool:
    """Match an exact model identity, allowing a local path to alias its leaf."""

    source_key = str(source).strip().replace("\\", "/").rstrip("/").lower()
    requested_key = str(requested).strip().replace("\\", "/").rstrip("/").lower()
    if not source_key or not requested_key:
        return False
    if source_key == requested_key:
        return True
    source_local = source_key.startswith("/") or (
        len(source_key) >= 3 and source_key[1:3] == ":/"
    )
    requested_local = requested_key.startswith("/") or (
        len(requested_key) >= 3 and requested_key[1:3] == ":/"
    )
    return bool(
        (source_local or requested_local)
        and source_key.rsplit("/", 1)[-1] == requested_key.rsplit("/", 1)[-1]
    )


def is_immutable_commit_hash(value: Any) -> bool:
    """Return whether ``value`` is a usable abbreviated/full hexadecimal commit."""

    return bool(re.fullmatch(r"[0-9a-fA-F]{7,64}", str(value).strip()))


def commit_hashes_match(left: Any, right: Any) -> bool:
    """Compare full or abbreviated immutable commit hashes."""

    if not is_immutable_commit_hash(left) or not is_immutable_commit_hash(right):
        return False
    left_key, right_key = str(left).strip().lower(), str(right).strip().lower()
    return left_key.startswith(right_key) or right_key.startswith(left_key)


__all__ = [
    "MODEL_COMMIT_SOURCES",
    "TRACE_CONTRACT",
    "VERIFIED_MODEL_COMMIT_SOURCES",
    "commit_hashes_match",
    "is_immutable_commit_hash",
    "model_identity_matches",
]
