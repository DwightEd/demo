from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
import zipfile


BINARY_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".safetensors"}
REPORT_SUFFIXES = {".json", ".csv", ".md", ".html", ".png"}


def _npz_keys(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return sorted(
            name[:-4]
            for name in archive.namelist()
            if name.endswith(".npy") and "/" not in name
        )


def _roles(keys: set[str]) -> list[str]:
    roles: list[str] = []
    if {"sv_clouds", "cloud_sizes"}.issubset(keys):
        roles.append("same_problem_raw_token_hidden_clouds")
    if any(key.startswith("sv_vec_") for key in keys):
        roles.append("projected_or_pooled_multisample_step_vectors")
    if "stepvec" in keys:
        roles.append("processbench_step_hidden_states")
    if {"input_ids", "attention_mask", "step_token_ranges"}.issubset(keys):
        roles.append("exact_teacher_forcing_trace")
    if {"problem_ids", "sample_idx", "is_correct"}.issubset(keys):
        roles.append("same_problem_sampling_axis")
    if "gold_error_step" in keys:
        roles.append("process_error_labels")
    if "fisher_transfer" in keys or "metadata_json" in keys and "item_metadata" in keys:
        roles.append("causal_pullback_result")
    if "hidden_files" in keys or "layer_state_memmap_path" in keys:
        roles.append("external_hidden_shard_index")
    return roles


def _sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def inventory_artifacts(
    roots: Iterable[str | Path],
    *,
    include_reports: bool = True,
    compute_hash: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    suffixes = set(BINARY_SUFFIXES)
    if include_reports:
        suffixes.update(REPORT_SUFFIXES)
    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            stat = path.stat()
            keys: list[str] = []
            error = ""
            if path.suffix.lower() == ".npz":
                try:
                    keys = _npz_keys(path)
                except (OSError, zipfile.BadZipFile) as exc:
                    error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "path": path.as_posix(),
                    "bytes": int(stat.st_size),
                    "size_gib": float(stat.st_size / (1024**3)),
                    "modified_unix": float(stat.st_mtime),
                    "suffix": path.suffix.lower(),
                    "npz_keys": keys,
                    "roles": _roles(set(keys)),
                    "sha256": _sha256(path) if compute_hash else "",
                    "inspection_error": error,
                }
            )
    return sorted(rows, key=lambda row: (-int(row["bytes"]), str(row["path"])))


def write_inventory(rows: list[dict[str, Any]], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "artifact_inventory.json"
    csv_path = output_dir / "artifact_inventory.csv"
    md_path = output_dir / "artifact_inventory.md"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "path",
                "bytes",
                "size_gib",
                "modified_unix",
                "suffix",
                "roles",
                "npz_keys",
                "sha256",
                "inspection_error",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "roles": ";".join(row["roles"]),
                    "npz_keys": ";".join(row["npz_keys"]),
                }
            )
    lines = [
        "# Remote Research Artifact Inventory",
        "",
        "Files: "
        f"`{len(rows)}`; total size: "
        f"`{sum(row['bytes'] for row in rows) / (1024**3):.3f} GiB`.",
        "",
        "| GiB | artifact | inferred roles |",
        "|---:|---|---|",
    ]
    for row in rows:
        roles = ", ".join(row["roles"]) or "unclassified"
        lines.append(f"| {row['size_gib']:.3f} | `{row['path']}` | {roles} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
    }
