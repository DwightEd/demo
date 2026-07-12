from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np


@dataclass
class ChainRecord:
    """A single reasoning chain with text and labels."""

    chain_idx: int
    problem_id: int
    problem: str
    steps: List[str]
    response: str
    gold_error_step: int = -1
    is_correct: Optional[int] = None
    sample_idx: Optional[int] = None
    generator: Optional[str] = None
    dataset: Optional[str] = None


def _as_list_of_str(x: Any) -> List[str]:
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    if x is None:
        return []
    return [str(x)]


def _get_array(npz: np.lib.npyio.NpzFile, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in npz.files:
            return npz[name]
    return default


def is_processbench_full(path: str | Path, npz: np.lib.npyio.NpzFile | None = None) -> bool:
    name = Path(path).name.lower()
    if name.startswith("full_"):
        return True
    z = npz if npz is not None else np.load(path, allow_pickle=True)
    return ("gold_error_step" in z.files or "labels" in z.files) and "steps_text" in z.files


def is_multisample(path: str | Path, npz: np.lib.npyio.NpzFile | None = None) -> bool:
    name = Path(path).name.lower()
    if "multisample" in name:
        return True
    z = npz if npz is not None else np.load(path, allow_pickle=True)
    return "sample_idx" in z.files and ("is_correct" in z.files or "is_correct_strict" in z.files)


def _problem_id_from_record(raw_id: Any, fallback: int) -> int:
    text = str(raw_id)
    m = re.search(r"(\d+)$", text)
    return int(m.group(1)) if m else int(fallback)


def _dataset_from_path(path: Path) -> str:
    name = path.stem.lower()
    for suffix in ("_multisample_sv", "_features", "_sv", "_full"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name.startswith("full_"):
        name = name[len("full_") :]
    return name


def _load_processbench_jsonl(path: Path, max_chains: int = 0) -> List[ChainRecord]:
    rows: List[ChainRecord] = []
    dataset = _dataset_from_path(path)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_chains and len(rows) >= int(max_chains):
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            steps = _as_list_of_str(rec.get("steps", rec.get("steps_text", [])))
            response = str(rec.get("response", "\n\n".join(steps)))
            gold = int(rec.get("label", rec.get("gold_error_step", -1)))
            final_correct = rec.get("final_answer_correct", None)
            is_correct = None if final_correct is None else int(bool(final_correct))
            rows.append(
                ChainRecord(
                    chain_idx=len(rows),
                    problem_id=_problem_id_from_record(rec.get("id", i), i),
                    problem=str(rec.get("problem", rec.get("question", ""))),
                    steps=steps,
                    response=response,
                    gold_error_step=gold,
                    is_correct=is_correct,
                    sample_idx=None,
                    generator=rec.get("generator"),
                    dataset=dataset,
                )
            )
    return rows


def load_chain_records(
    path: str | Path,
    max_chains: int = 0,
    input_format: str = "auto",
) -> List[ChainRecord]:
    """Load text chains from a ProcessBench full or multisample npz.

    This loader intentionally avoids inferring same-problem response labels from
    ProcessBench full data.  The caller chooses the evaluation mode later.
    """

    path = Path(path)
    fmt = input_format.lower()
    if fmt not in {"auto", "npz", "processbench_jsonl", "jsonl"}:
        raise ValueError(f"unknown input_format={input_format!r}")
    if fmt in {"processbench_jsonl", "jsonl"} or (fmt == "auto" and path.suffix.lower() == ".jsonl"):
        return _load_processbench_jsonl(path, max_chains=max_chains)

    z = np.load(path, allow_pickle=True)
    steps_arr = _get_array(z, ["steps_text", "steps"], None)
    if steps_arr is None:
        raise ValueError(f"{path}: expected `steps_text` or `steps` for prompt-flow extraction")

    n = int(len(steps_arr))
    if max_chains and max_chains > 0:
        n = min(n, int(max_chains))

    problems = _get_array(z, ["problems", "problem", "questions", "question"], None)
    responses = _get_array(z, ["responses", "response"], None)
    problem_ids = _get_array(z, ["problem_ids"], np.arange(len(steps_arr), dtype=np.int64))
    sample_idx = _get_array(z, ["sample_idx"], None)
    generators = _get_array(z, ["generator", "generators", "source_model", "model_name"], None)
    datasets = _get_array(z, ["dataset", "datasets", "subset"], None)

    gold = _get_array(z, ["gold_error_step", "labels"], None)
    if gold is None:
        gold = np.full(len(steps_arr), -1, dtype=np.int64)

    correct = _get_array(z, ["is_correct_strict", "is_correct"], None)

    rows: List[ChainRecord] = []
    for i in range(n):
        steps = _as_list_of_str(steps_arr[i])
        response = str(responses[i]) if responses is not None else "\n\n".join(steps)
        problem = str(problems[i]) if problems is not None else ""
        rows.append(
            ChainRecord(
                chain_idx=i,
                problem_id=int(problem_ids[i]) if problem_ids is not None else i,
                problem=problem,
                steps=steps,
                response=response,
                gold_error_step=int(gold[i]) if gold is not None else -1,
                is_correct=int(correct[i]) if correct is not None else None,
                sample_idx=int(sample_idx[i]) if sample_idx is not None else None,
                generator=str(generators[i]) if generators is not None else None,
                dataset=str(datasets[i]) if datasets is not None else _dataset_from_path(path),
            )
        )
    return rows
