from __future__ import annotations

import numpy as np

from functional_divergence.causal_first_error_attribution import (
    message_fisher_experiment as experiment_module,
)
from functional_divergence.causal_first_error_attribution.message_fisher import (
    FFN_DIRECTION_ID,
    SourceMessageFisherResult,
)
from functional_divergence.causal_first_error_attribution.message_fisher_experiment import (
    MessageFisherExperiment,
    MessageFisherExperimentConfig,
    first_error_boundary_pairs,
    paired_fisher_summary,
)


def test_first_error_pairs_use_only_consecutive_future_free_boundaries(
    tmp_path,
) -> None:
    trace = tmp_path / "trace.npz"
    np.savez_compressed(
        trace,
        full_input_ids=np.asarray(
            [
                [10, 11, 20, 21, 30, 31, 0],
                [10, 12, 40, 41, 0, 0, 0],
                [10, 13, 50, 51, 60, 61, 0],
            ]
        ),
        full_attention_mask=np.asarray(
            [
                [1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0],
            ],
            dtype=np.int8,
        ),
        prompt_token_counts=np.asarray([2, 2, 2]),
        step_token_ranges=np.asarray(
            [
                [[2, 3], [4, 5]],
                [[2, 3], [-1, -1]],
                [[2, 3], [4, 5]],
            ]
        ),
        n_steps=np.asarray([2, 1, 2]),
        gold_error_step=np.asarray([1, 0, -1]),
        chain_idx=np.asarray([101, 102, 103]),
    )

    pairs = first_error_boundary_pairs(trace, max_cases=0, seed=17)

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.record_index == 0
    assert pair.chain_id == 101
    assert pair.first_error_step == 1
    assert pair.control_step == 0
    assert pair.control_decision_position == 1
    assert pair.event_decision_position == 3


def test_paired_summary_bootstraps_event_minus_control_by_case() -> None:
    rows = []
    for case, domain, control, event in (
        ("a", "gsm8k", 1.0, 3.0),
        ("b", "gsm8k", 2.0, 5.0),
        ("c", "math", 4.0, 8.0),
        ("d", "math", 3.0, 8.0),
    ):
        rows.extend(
            (
                {
                    "case_id": case,
                    "domain": domain,
                    "boundary_role": "previous_correct",
                    "layer": 8,
                    "largest_eigenvalue": control,
                },
                {
                    "case_id": case,
                    "domain": domain,
                    "boundary_role": "first_error",
                    "layer": 8,
                    "largest_eigenvalue": event,
                },
            )
        )

    summary = paired_fisher_summary(
        rows,
        metrics=("largest_eigenvalue",),
        bootstrap_repeats=500,
        seed=9,
    )

    result = summary["pooled"]["8"]["largest_eigenvalue"]
    assert result["n_pairs"] == 4
    assert result["mean_event_minus_control"] == 3.5
    assert result["ci_low"] > 0.0
    assert result["positive_fraction"] == 1.0


def test_experiment_extracts_paired_boundaries_and_writes_auditable_artifacts(
    tmp_path, monkeypatch
) -> None:
    selected = tmp_path / "data" / "gsm8k" / "selected"
    selected.mkdir(parents=True)
    np.savez_compressed(
        selected / "trace.npz",
        full_input_ids=np.asarray([[10, 11, 20, 21, 30, 31]]),
        full_attention_mask=np.ones((1, 6), dtype=np.int8),
        prompt_token_counts=np.asarray([2]),
        step_token_ranges=np.asarray([[[2, 3], [4, 5]]]),
        n_steps=np.asarray([2]),
        gold_error_step=np.asarray([1]),
        chain_idx=np.asarray([101]),
        dataset=np.asarray(["gsm8k"]),
    )
    calls = []

    class FakeRunner:
        def __init__(self, **_kwargs) -> None:
            pass

        def run(self, *, model, input_ids, source_step_ids):
            del model
            calls.append((input_ids.copy(), source_step_ids.copy()))
            scale = float(len(input_ids))
            messages = np.asarray([[[1.0, 0.0], [0.0, 1.0]]])
            attention = messages.sum(axis=1)
            mlp = np.asarray([[[0.5, 0.5]]]).reshape(1, 2)
            pre = np.asarray([[1.0, 1.0]])
            post = pre + attention + mlp
            result = SourceMessageFisherResult(
                layers=np.asarray([8]),
                source_ids=np.asarray([-1, 0]),
                direction_source_ids=np.asarray([-1, 0, FFN_DIRECTION_ID]),
                source_messages=messages,
                attention_mass=np.asarray([[[0.5, 0.5]]]),
                attention_output=attention,
                mlp_output=mlp,
                residual_pre=pre,
                residual_post=post,
                fisher_gram=np.asarray([np.diag([scale, 1.0, 0.5])]),
                euclidean_gram=np.asarray([np.diag([2.0, 1.0, 0.5])]),
                observed_symmetric_kl=np.asarray([[0.1, 0.1, 0.1]]),
                predicted_quadratic_kl=np.asarray([[0.1, 0.1, 0.1]]),
                quadratic_relative_error=np.zeros((1, 3)),
                attention_reconstruction_error=np.zeros(1),
                block_reconstruction_error=np.zeros(1),
                baseline_entropy=1.0,
                epsilon=0.05,
            )
            result.validate()
            return result

    monkeypatch.setattr(experiment_module, "SourceMessageFisherRunner", FakeRunner)
    output = tmp_path / "results"
    report = MessageFisherExperiment(
        MessageFisherExperimentConfig(
            data_root=tmp_path / "data",
            domains=("gsm8k",),
            layers=(8,),
            output_dir=output,
            max_cases_per_domain=1,
            bootstrap_repeats=50,
        )
    ).run(object())

    assert [tuple(values[0]) for values in calls] == [(10, 11), (10, 11, 20, 21)]
    assert [tuple(values[1]) for values in calls] == [(-1, -1), (-1, -1, 0, 0)]
    assert report["paired_cases"] == 1
    assert report["boundary_runs"] == 2
    assert (output / "events.jsonl").is_file()
    assert (output / "summary.json").is_file()
    artifacts = sorted((output / "artifacts" / "gsm8k").glob("*.npz"))
    assert len(artifacts) == 2
    with np.load(artifacts[0], allow_pickle=False) as archive:
        assert "source_messages" in archive.files
        assert "attention_mass" in archive.files
        assert "fisher_gram" in archive.files
        assert "euclidean_gram" in archive.files
        assert "input_ids" in archive.files
