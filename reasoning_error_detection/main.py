from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import FeatureConfig, ProcessBenchFeatureLoader
from .detector import DetectorConfig, ProcessBenchErrorDetector


def _layers(value: str) -> tuple[int, ...]:
    text = value.strip()
    if not text or text.lower() == "all":
        return ()
    try:
        layers = tuple(int(item.strip()) for item in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "layers must be comma-separated integers or 'all'"
        ) from error
    if not layers or len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("layers must be non-empty and unique")
    return layers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-fit a first-error detector from existing ProcessBench "
            "output, routing, and residual-stream features."
        )
    )
    parser.add_argument("--input", required=True, help="ProcessBench geometry trace.npz")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layers", type=_layers, default=())
    parser.add_argument("--projection_dim", type=int, default=16)
    parser.add_argument("--projection_batch_size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--logistic_c", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = ProcessBenchFeatureLoader(
        FeatureConfig(
            layers=args.layers,
            projection_dim=args.projection_dim,
            projection_batch_size=args.projection_batch_size,
            seed=args.seed,
            device=args.device,
        )
    ).load(args.input)
    report = ProcessBenchErrorDetector(
        DetectorConfig(
            folds=args.folds,
            logistic_c=args.logistic_c,
            seed=args.seed,
        )
    ).run(data, Path(args.output_dir))
    concise = {
        name: {
            "onset_auroc": values["onset"]["auroc"],
            "onset_auprc": values["onset"]["auprc"],
            "localization_top1": values["localization"]["top1"],
            "chain_auroc": values["chain_detection"]["auroc"],
        }
        for name, values in report["feature_sets"].items()
    }
    print(json.dumps(concise, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
