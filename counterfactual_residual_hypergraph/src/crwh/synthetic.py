from __future__ import annotations

import numpy as np

from .builder import ResidualWriteHypergraphBuilder
from .contracts import CounterfactualExample, ResidualWriteTrace, TraceProvenance


SYNTHETIC_PROVENANCE = TraceProvenance(
    model_id="synthetic-model",
    model_revision="v1",
    tokenizer_id="synthetic-tokenizer",
    tokenizer_revision="v1",
    prompt_format_id="synthetic-template-v1",
    layer_id=4,
    extractor_id="synthetic_supplied_write_v1",
    projection_id="identity-r3",
    output_anchor_id="fixed-evidence-direction",
    teacher_forcing_offset=0,
)


def _attention(label: int, *, view: str, rng: np.random.Generator) -> np.ndarray:
    if label == 0:
        base = np.asarray(
            [
                [0.32, 0.28, 0.16, 0.08, 0.10, 0.00],
                [0.25, 0.25, 0.15, 0.07, 0.18, 0.10],
            ]
        )
    else:
        base = np.asarray(
            [
                [0.06, 0.05, 0.12, 0.14, 0.50, 0.00],
                [0.05, 0.05, 0.10, 0.10, 0.55, 0.15],
            ]
        )
    heads = np.stack((base, 0.9 * base), axis=0)
    if view == "counterfactual":
        factor = 0.1 if label == 0 else 0.9
        heads[:, :, :2] *= factor
        heads[:, :, 2:4] *= 1.2
    elif view == "paraphrase":
        heads *= np.clip(rng.normal(1.0, 0.015, size=heads.shape), 0.95, 1.05)
    heads[:, 0, 5] = 0.0
    return heads


def _trace(
    *,
    sample_id: str,
    view: str,
    label: int,
    labeled: bool,
    base_nodes: np.ndarray,
    rng: np.random.Generator,
) -> ResidualWriteTrace:
    node_features = base_nodes.copy()
    if view == "paraphrase":
        node_features[:4] += rng.normal(0.0, 0.01, size=(4, 4))
    elif view == "counterfactual":
        node_features[:2] = -node_features[:2]
        # Grounded responses receive a larger fixed-response embedding shift
        # when their evidence is replaced; synthetic confabulations change less.
        context_shift = 0.8 if label == 0 else 0.05
        node_features[4:, 0] += context_shift
    attention = _attention(label, view=view, rng=rng)
    content = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.8, 0.2],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    if view == "counterfactual":
        content = content.copy()
        content[:2] *= -1.0
    source_writes = attention[..., None] * content[None, None, :, :]
    source_writes[1] *= 0.9
    labels = np.asarray([label, label] if labeled else [-1, -1])
    return ResidualWriteTrace(
        sample_id=sample_id,
        view_name=view,
        perturbation_kind={
            "factual": "factual",
            "paraphrase": "paraphrase",
            "counterfactual": "evidence_replacement",
        }[view],
        perturbation_id=f"{sample_id}/{view}",
        perturbation_seed=0,
        perturbation_validated=True,
        provenance=SYNTHETIC_PROVENANCE,
        node_features=node_features,
        attention=attention,
        source_writes=source_writes,
        expected_updates=source_writes.sum(axis=2),
        output_directions=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        receiver_positions=np.asarray([4, 5]),
        response_token_ids=np.asarray([1001, 1002]),
        source_roles=np.asarray([0, 0, 1, 1, 2, 2]),
        token_labels=labels,
    )


def make_synthetic_examples(
    *,
    count: int,
    seed: int = 42,
    labeled_fraction: float = 0.5,
) -> list[CounterfactualExample]:
    if count < 2:
        raise ValueError("count must be at least two")
    if not 0.0 <= labeled_fraction <= 1.0:
        raise ValueError("labeled_fraction must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    builder = ResidualWriteHypergraphBuilder(
        attention_threshold=0.04,
        min_sources=2,
    )
    examples: list[CounterfactualExample] = []
    for index in range(count):
        label = index % 2
        labeled = index < round(count * labeled_fraction)
        sample_id = f"synthetic-{index:04d}"
        base_nodes = rng.normal(0.0, 0.5, size=(6, 4))
        traces = {
            view: _trace(
                sample_id=sample_id,
                view=view,
                label=label,
                labeled=labeled,
                base_nodes=base_nodes,
                rng=rng,
            )
            for view in ("factual", "counterfactual", "paraphrase")
        }
        examples.append(
            CounterfactualExample(
                sample_id=sample_id,
                factual=builder.build(traces["factual"]),
                counterfactual=builder.build(traces["counterfactual"]),
                paraphrase=builder.build(traces["paraphrase"]),
                normal_reference=label == 0,
                contrastive_eligible=np.full(2, label == 0, dtype=bool),
            )
        )
    return examples
