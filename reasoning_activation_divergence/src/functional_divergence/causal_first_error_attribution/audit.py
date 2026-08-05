from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .pairs import load_pair_file


class PairAuditor:
    """Inventory ProcessBench labels and verified onset-pair availability."""

    def __init__(
        self,
        data_root: str | Path,
        domains: Iterable[str],
        *,
        pair_directory: str = "causal_first_error_v1",
    ) -> None:
        self.data_root = Path(data_root).expanduser()
        self.domains = tuple(str(domain) for domain in domains)
        self.pair_directory = str(pair_directory)
        if not self.domains:
            raise ValueError("at least one domain is required")

    def run(self) -> dict[str, Any]:
        totals: defaultdict[str, int] = defaultdict(int)
        domain_reports = []
        missing_pair_paths: list[Path] = []
        for domain in self.domains:
            report = self._audit_domain(domain)
            domain_reports.append(report)
            for name in (
                "natural_chains",
                "error_chains",
                "correct_chains",
                "same_problem_error_correct_candidates",
                "valid_target_correction_pairs",
                "valid_controlled_root_pairs",
                "invalid_pair_records",
            ):
                totals[name] += int(report[name])
            if not report["pair_file_exists"]:
                missing_pair_paths.append(Path(report["pair_file"]))

        controlled = totals["valid_controlled_root_pairs"]
        target = totals["valid_target_correction_pairs"]
        next_path = (
            missing_pair_paths[0]
            if missing_pair_paths
            else self.data_root
            / self.domains[0]
            / "selected"
            / self.pair_directory
            / "onset_pairs_v1.jsonl"
        )
        return {
            **dict(totals),
            "domains": domain_reports,
            "root_cause_claim": (
                "controlled_pairs_available"
                if controlled
                else "not_identifiable_from_available_data"
            ),
            "natural_repair_claim": (
                "target_pairs_available" if target else "not_runnable_without_pairs"
            ),
            "model_required": False,
            "next_required_artifact": next_path.as_posix(),
        }

    def _audit_domain(self, domain: str) -> dict[str, Any]:
        selected = self.data_root / domain / "selected"
        trace_path = selected / "trace.npz"
        pair_path = selected / self.pair_directory / "onset_pairs_v1.jsonl"
        natural, errors, correct, candidates = self._trace_counts(trace_path)
        pairs, invalid = load_pair_file(pair_path)
        return {
            "dataset": domain,
            "trace_file": trace_path.as_posix(),
            "trace_file_exists": trace_path.is_file(),
            "pair_file": pair_path.as_posix(),
            "pair_file_exists": pair_path.is_file(),
            "natural_chains": natural,
            "error_chains": errors,
            "correct_chains": correct,
            "same_problem_error_correct_candidates": candidates,
            "valid_target_correction_pairs": sum(
                pair.pair_kind == "target_correction" for pair in pairs
            ),
            "valid_controlled_root_pairs": sum(
                pair.pair_kind == "controlled_root" for pair in pairs
            ),
            "invalid_pair_records": len(invalid),
            "pair_errors": list(invalid),
        }

    @staticmethod
    def _trace_counts(path: Path) -> tuple[int, int, int, int]:
        if not path.is_file():
            return 0, 0, 0, 0
        with np.load(path, allow_pickle=True) as archive:
            if "gold_error_step" not in archive.files:
                raise ValueError(f"{path}: trace lacks gold_error_step")
            gold = np.asarray(archive["gold_error_step"], dtype=np.int64).reshape(-1)
            group_name = next(
                (
                    name
                    for name in ("problem_hashes", "problem_ids", "problem_group_id")
                    if name in archive.files
                ),
                None,
            )
            groups = (
                np.asarray(archive[group_name], dtype=object).reshape(-1)
                if group_name is not None
                else np.arange(len(gold), dtype=object)
            )
        if groups.shape != gold.shape:
            raise ValueError(f"{path}: problem grouping is not record-aligned")
        candidates = 0
        for group in np.unique(groups):
            labels = gold[groups == group]
            candidates += int(np.any(labels >= 0) and np.any(labels == -1))
        return len(gold), int(np.sum(gold >= 0)), int(np.sum(gold == -1)), candidates
