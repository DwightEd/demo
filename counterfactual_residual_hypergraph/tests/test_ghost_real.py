from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from crwh.ghost import GhostTraceEmbedding
from crwh.ghost_eval import (
    EvaluationConfig,
    _normal_tail_scores,
    evaluate_ghost,
)
from crwh.ghost_hf import (
    HookedResponseStateExtractor,
    extract_processbench_embeddings,
    load_embedding_artifact,
    save_embedding_artifact,
)


MID_DEPTHS = (2, 3, 4, 5, 6, 7)
ALL_DEPTHS = (*MID_DEPTHS, 8)


def _traces(count: int = 60) -> tuple[GhostTraceEmbedding, ...]:
    rng = np.random.default_rng(21)
    rows = []
    for index in range(count):
        label = index % 2
        center = 3.0 if label else 0.0
        mean = center + rng.normal(
            0.0,
            0.05,
            size=(len(ALL_DEPTHS), 6),
        )
        rows.append(
            GhostTraceEmbedding(
                trace_id=f"trace-{index:03d}",
                problem_id=f"problem-{index:03d}",
                response_label=label,
                first_error=1 if label else -1,
                generator_model=f"generator-{index % 3}",
                layer_depths=ALL_DEPTHS,
                response_mean=mean,
                response_last=mean + 0.01,
                token_count=20 + index % 5,
                response_token_count=10 + index % 3,
                step_count=3,
            )
        )
    return tuple(rows)


def test_embedding_artifact_round_trip_is_pickle_free(tmp_path: Path) -> None:
    destination = tmp_path / "embeddings.npz"
    source = _traces(6)

    save_embedding_artifact(
        destination,
        source,
        metadata={"observer_model": "/models/llama", "protocol": "fixture"},
    )
    restored, metadata = load_embedding_artifact(destination)

    assert metadata["observer_model"] == "/models/llama"
    assert [trace.trace_id for trace in restored] == [
        trace.trace_id for trace in source
    ]
    assert np.allclose(restored[2].response_mean, source[2].response_mean)
    with np.load(destination, allow_pickle=False) as archive:
        assert archive["schema_version"].item() == "ghost_embeddings_v1"


def test_discrete_nuisance_scores_use_tie_aware_midranks() -> None:
    calibration = np.asarray([3, 3, 3, 3], dtype=np.float64)
    test = np.asarray([3, 2, 4], dtype=np.float64)

    scores = _normal_tail_scores(calibration, test)

    assert np.allclose(scores["upper_tail"], [0.5, 0.0, 1.0])
    assert np.allclose(scores["lower_tail"], [0.5, 1.0, 0.0])
    assert np.allclose(scores["two_sided"], [0.0, 1.0, 1.0])


def test_hook_extractor_captures_only_requested_depths_in_one_forward(
    capsys,
) -> None:
    torch = pytest.importorskip("torch")

    class Block(torch.nn.Module):
        def __init__(self, increment: float) -> None:
            super().__init__()
            self.increment = increment

        def forward(self, hidden):
            return (hidden + self.increment,)

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(1.0) for _ in range(8)])
            self.config = SimpleNamespace(num_hidden_layers=8)
            self.last_kwargs: dict[str, object] = {}

        def forward(self, input_ids, **kwargs: object) -> SimpleNamespace:
            self.last_kwargs = kwargs
            hidden = input_ids.to(torch.float32)[..., None].repeat(1, 1, 3)
            for block in self.layers:
                hidden = block(hidden)[0]
            return SimpleNamespace(last_hidden_state=hidden)

    model = FakeModel()
    extractor = HookedResponseStateExtractor(
        model,
        mid_depths=MID_DEPTHS,
        final_depth=8,
    )

    response_mean, response_last = extractor.extract(
        input_ids=torch.arange(7, dtype=torch.int64)[None, :],
        response_positions=np.asarray([2, 3, 5, 6], dtype=np.int64),
    )
    extractor.close()

    assert response_mean.shape == (len(ALL_DEPTHS), 3)
    assert response_last.shape == response_mean.shape
    assert np.allclose(response_mean[0], np.mean([2, 3, 5, 6]) + 2)
    assert np.allclose(response_mean[-1], np.mean([2, 3, 5, 6]) + 8)
    assert model.last_kwargs["use_cache"] is False
    assert model.last_kwargs["output_hidden_states"] is False
    assert model.last_kwargs["output_attentions"] is False

    empty = extract_processbench_embeddings(
        (),
        model=model,
        tokenizer=object(),
        mid_depths=MID_DEPTHS,
        final_depth=8,
        max_tokens=16,
        show_progress=True,
    )
    captured = capsys.readouterr()
    assert empty == ()
    assert "Extracting hidden states" in captured.err
    assert "extracted 1:" not in captured.err


def test_real_evaluation_fits_only_normal_train_and_calibration_groups(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"

    summary = evaluate_ghost(
        _traces(),
        output_dir=output,
        config=EvaluationConfig(
            split_seed=17,
            validation_ratio=0.2,
            test_ratio=0.2,
            threshold_quantile=0.95,
            bootstrap_replicates=25,
        ),
    )

    audit = json.loads((output / "fit-audit.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    split = json.loads((output / "split.json").read_text(encoding="utf-8"))
    with (output / "scores-test.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    with (output / "anomaly-scores-test.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        unlabeled = list(csv.DictReader(stream))

    assert summary["result_kind"] == "real_processbench_ghost_style_one_class"
    assert summary["training_epochs"] == 0
    assert audit["fit_positive_count"] == 0
    assert audit["calibration_positive_count"] == 0
    assert audit["frozen_score_artifact_written_before_metric_computation"] is True
    assert audit["problem_group_intersections"] == {
        "fit_calibration": [],
        "fit_test": [],
        "calibration_test": [],
    }
    assert set(split["fit_normal_trace_ids"]).isdisjoint(
        split["test_trace_ids"]
    )
    assert metrics["response_mean"]["mid_fused"]["auroc"] > 0.95
    assert metrics["response_mean"]["mid_fused"]["aupr"] > 0.95
    assert metrics["response_mean"]["mid_fused"]["calibration_count"] == audit[
        "calibration_trace_count"
    ]
    assert set(metrics["nuisance_only"]["response_token_count"]) == {
        "upper_tail",
        "lower_tail",
        "two_sided",
    }
    assert metrics["response_mean"]["mid_minus_final_bootstrap"][
        "replicates"
    ] == 25
    assert len(
        metrics["response_mean"]["per_mid_layer_minus_final_bootstrap"]
    ) == len(MID_DEPTHS)
    assert len(predictions) == metrics["response_mean"]["mid_fused"]["n"]
    assert "label" not in unlabeled[0]
    assert "first_error" not in unlabeled[0]
    assert "response_mean_mid_fused_rank_score" in unlabeled[0]
    assert "response_mean_depth_2_distance" in unlabeled[0]
    assert metrics["response_mean"]["generator_macro_mid_fused"][
        "defined_auroc_generators"
    ] >= 1
    assert (output / "reference-model-response_mean.npz").is_file()
    assert (output / "reference-model-response_last.npz").is_file()
    assert (output / "summary.json").is_file()
