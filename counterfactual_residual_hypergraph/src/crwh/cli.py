from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .detector import MultiViewOneClassDetector
from .model import MultiViewHypergraphDetector
from .selection import select_review_candidates
from .synthetic import make_synthetic_examples
from .training import TrainingConfig, train_semisupervised


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crwh")
    subparsers = parser.add_subparsers(dest="command", required=True)
    synthetic = subparsers.add_parser(
        "synthetic",
        help="run a CPU-only end-to-end contract smoke test",
    )
    synthetic.add_argument("--output-dir", required=True)
    synthetic.add_argument("--examples", type=int, default=16)
    synthetic.add_argument("--epochs", type=int, default=3)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--labeled-fraction", type=float, default=0.5)
    synthetic.add_argument("--review-fraction", type=float, default=0.2)
    synthetic.add_argument(
        "--unsupervised-threshold",
        type=float,
        default=0.95,
    )
    synthetic.add_argument(
        "--semisupervised-threshold",
        type=float,
        default=0.5,
    )
    return parser


def _run_synthetic(args: argparse.Namespace) -> int:
    if not 0.0 <= args.review_fraction <= 1.0:
        raise ValueError("review_fraction must lie in [0, 1]")
    examples = make_synthetic_examples(
        count=args.examples,
        seed=args.seed,
        labeled_fraction=args.labeled_fraction,
    )
    reference = [
        example
        for index, example in enumerate(examples)
        if index % 2 == 0
    ]
    one_class = MultiViewOneClassDetector(shrinkage=0.2).fit(reference)
    torch.manual_seed(args.seed)
    node_dim = examples[0].factual.node_features.shape[1]
    edge_dim = examples[0].factual.edge_features.shape[1]
    model = MultiViewHypergraphDetector(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=16,
        num_layers=1,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=1e-3,
        seed=args.seed,
    )
    history = train_semisupervised(
        model,
        examples,
        config=training_config,
        device=torch.device("cpu"),
    )

    records: list[dict[str, object]] = []
    unsupervised_percentiles: list[float] = []
    semisupervised_probabilities: list[float] = []
    model.eval()
    with torch.no_grad():
        for example in examples:
            unsupervised = one_class.score(example)
            neural = torch.sigmoid(model(example).logits).cpu().numpy()
            labels = example.factual.token_labels
            for token_index in range(len(neural)):
                label = int(labels[token_index]) if labels is not None else -1
                records.append(
                    {
                        "sample_id": example.sample_id,
                        "token_index": token_index,
                        "label": label,
                        "unsupervised_score": float(
                            unsupervised.fused[token_index]
                        ),
                        "unsupervised_state_score": float(
                            unsupervised.state[token_index]
                        ),
                        "unsupervised_relation_score": float(
                            unsupervised.relation[token_index]
                        ),
                        "unsupervised_context_state_score": float(
                            unsupervised.context_state[token_index]
                        ),
                        "unsupervised_context_relation_score": float(
                            unsupervised.context_relation[token_index]
                        ),
                        "unsupervised_anomaly_percentile": float(
                            unsupervised.fused_percentile[token_index]
                        ),
                        "semisupervised_probability": float(neural[token_index]),
                    }
                )
                unsupervised_percentiles.append(
                    float(unsupervised.fused_percentile[token_index])
                )
                semisupervised_probabilities.append(float(neural[token_index]))

    unsupervised_array = np.asarray(unsupervised_percentiles)
    semisupervised_array = np.asarray(semisupervised_probabilities)
    review_budget = int(np.ceil(args.review_fraction * len(records)))
    review_indices = select_review_candidates(
        unsupervised_anomaly_percentile=unsupervised_array,
        semisupervised_probability=semisupervised_array,
        unsupervised_threshold=args.unsupervised_threshold,
        semisupervised_threshold=args.semisupervised_threshold,
        budget=review_budget,
    )
    review_queue = [
        {
            **records[int(index)],
            "reason": "thresholded_detector_decision_disagreement",
        }
        for index in review_indices
    ]
    finite_scores = bool(
        np.isfinite(unsupervised_array).all()
        and np.isfinite(semisupervised_array).all()
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "command": "synthetic",
        "examples": args.examples,
        "epochs": args.epochs,
        "seed": args.seed,
        "labeled_fraction": args.labeled_fraction,
        "review_fraction": args.review_fraction,
        "unsupervised_threshold": args.unsupervised_threshold,
        "semisupervised_threshold": args.semisupervised_threshold,
    }
    metrics = {
        "examples": len(examples),
        "tokens": len(records),
        "finite_scores": finite_scores,
        "review_queue_size": len(review_queue),
        "last_epoch": asdict(history[-1]),
        "note": "Synthetic smoke metrics are not scientific results.",
    }
    for name, value in (
        ("config.json", config),
        ("metrics.json", metrics),
        ("scores.json", records),
        ("review_queue.json", review_queue),
    ):
        (output_dir / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "synthetic":
        return _run_synthetic(args)
    raise AssertionError(f"unhandled command: {args.command}")
