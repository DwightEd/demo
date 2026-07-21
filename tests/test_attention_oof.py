import csv
import json
from pathlib import Path

import pytest

from hypergraph.attention.aggregate_oof import aggregate_oof_run


def _write_predictions(
    root: Path,
    *,
    fold: int,
    seed: int,
    rows: list[dict[str, object]],
) -> None:
    run = root / f"fold{fold}_seed{seed}"
    run.mkdir(parents=True)
    with (run / "predictions_test.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "trace_id",
                "group_id",
                "generator_model",
                "label",
                "score",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_aggregate_oof_reports_one_final_test_metric(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    (root / "aggregate_results.json").write_text(
        json.dumps({"test_aggregate": {"auroc": {"mean": 0.5}}}),
        encoding="utf-8",
    )
    traces = [
        ("t0", "p0", 0, 0.10),
        ("t1", "p1", 1, 0.90),
        ("t2", "p2", 0, 0.20),
        ("t3", "p3", 1, 0.80),
        ("t4", "p4", 0, 0.30),
        ("t5", "p5", 1, 0.70),
    ]
    for seed, fold_members in {
        17: ((0, 1), (2, 3), (4, 5)),
        23: ((2, 5), (0, 3), (4, 1)),
    }.items():
        for fold, members in enumerate(fold_members):
            rows = []
            for index in members:
                trace_id, group_id, label, score = traces[index]
                rows.append(
                    {
                        "trace_id": trace_id,
                        "group_id": group_id,
                        "generator_model": "Llama-3.1-8B-Instruct",
                        "label": label,
                        "score": score + (0.02 if seed == 23 and label else 0.0),
                    }
                )
            _write_predictions(root, fold=fold, seed=seed, rows=rows)

    result = aggregate_oof_run(root, folds=3, seeds=[17, 23], write_outputs=True)

    assert result["protocol"] == "pooled_out_of_fold_test"
    assert result["integrity"]["unique_traces_per_seed"] == 6
    assert result["integrity"]["each_trace_tested_once_per_seed"] is True
    assert result["by_seed"]["17"]["auroc"] == pytest.approx(1.0)
    assert result["by_seed"]["23"]["auroc"] == pytest.approx(1.0)
    assert result["seed_ensemble"]["n"] == 6
    assert result["seed_ensemble"]["auroc"] == pytest.approx(1.0)
    assert (root / "pooled_oof_results.json").is_file()
    assert (root / "predictions_pooled_oof.csv").is_file()
    assert (root / "predictions_pooled_oof_seed_ensemble.csv").is_file()
    aggregate = json.loads((root / "aggregate_results.json").read_text(encoding="utf-8"))
    assert aggregate["pooled_oof_test"]["seed_ensemble"]["auroc"] == pytest.approx(1.0)
    assert aggregate["test_aggregate"]["auroc"]["mean"] == pytest.approx(0.5)


def test_aggregate_oof_rejects_duplicate_test_prediction(tmp_path):
    root = tmp_path / "runs"
    duplicate = {
        "trace_id": "t0",
        "group_id": "p0",
        "generator_model": "model",
        "label": 0,
        "score": 0.1,
    }
    _write_predictions(root, fold=0, seed=17, rows=[duplicate])
    _write_predictions(root, fold=1, seed=17, rows=[duplicate])

    with pytest.raises(ValueError, match="more than one held-out fold"):
        aggregate_oof_run(root, folds=2, seeds=[17], write_outputs=False)


def test_aggregate_oof_rejects_different_trace_sets_across_seeds(tmp_path):
    root = tmp_path / "runs"
    for seed, trace_id in ((17, "t0"), (23, "other")):
        _write_predictions(
            root,
            fold=0,
            seed=seed,
            rows=[
                {
                    "trace_id": trace_id,
                    "group_id": trace_id,
                    "generator_model": "model",
                    "label": 0,
                    "score": 0.1,
                }
            ],
        )

    with pytest.raises(ValueError, match="trace identity set differs"):
        aggregate_oof_run(root, folds=1, seeds=[17, 23], write_outputs=False)
