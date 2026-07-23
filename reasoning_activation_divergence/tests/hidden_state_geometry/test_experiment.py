from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from functional_divergence.hidden_state_geometry.contracts import TraceSource
from functional_divergence.hidden_state_geometry.config import RawFunctionalConfig
from functional_divergence.hidden_state_geometry.experiment import (
    _jsonable,
    inspect_hidden_geometry_sources,
    run_hidden_geometry_experiment,
)
from functional_divergence.hidden_state_geometry.method import MethodFoldResult
from functional_divergence.hidden_state_geometry.registry import ContrastSpec, register_method
from functional_divergence.progress import RecordingProgress


@register_method(
    "experiment_dummy",
    contrasts=(
        ContrastSpec("hidden_given_output_nll", "output_only", "output_plus_hidden"),
    ),
)
class ExperimentDummy:
    def __init__(self, config):
        self.config = config

    def fit_predict(self, fold):
        probability = np.asarray(
            [0.8 if row.sample.chain_id % 2 else 0.2 for row in fold.test_examples]
        )
        return MethodFoldResult(
            probabilities={
                "nuisance": np.full(len(probability), 0.5),
                "output_only": probability * 0.7 + 0.15,
                "hidden_only": probability,
                "output_plus_hidden": probability,
                "output_plus_time_null": np.full(len(probability), 0.5),
                "output_plus_layer_null": np.full(len(probability), 0.5),
            },
            diagnostics={"ok": True},
            factors={"coefficient": np.ones((1, 1, 1))},
        )


def _source(root, domain: str, domain_index: int) -> TraceSource:
    selected = root / domain / "selected"
    shards = selected / "states"
    shards.mkdir(parents=True)
    count = 6
    files = []
    ranges = np.empty(count, dtype=object)
    scores = np.empty((count, 3, 2), dtype=np.float32)
    metadata = []
    for row in range(count):
        chain = domain_index * 100 + row
        path = shards / f"row_{chain}.npy"
        state = np.empty((6, 2, 4), dtype=np.float32)
        for token in range(6):
            for layer in range(2):
                state[token, layer] = np.asarray(
                    [
                        chain * 0.01 + token,
                        layer + token * 0.2,
                        np.sin(chain + token + layer),
                        (chain % 5) * 0.1 + token * layer,
                    ]
                )
        if row % 2:
            state[1:4, 1, 2] += 2.0
        np.save(path, state)
        files.append(str(path.relative_to(selected)))
        ranges[row] = np.asarray([[10, 11], [12, 13], [14, 15]])
        scores[row] = row + np.arange(6, dtype=np.float32).reshape(3, 2)
        metadata.append(
            json.dumps({"loaded_model": "meta-llama/Llama-3.1-8B-Instruct"})
        )
    manifest = selected / "trace.raw_residual_stream.npz"
    np.savez(
        manifest,
        chain_idx=np.asarray([domain_index * 100 + row for row in range(count)]),
        gold_error_step=np.asarray([1 if row % 2 else -1 for row in range(count)]),
        problem_group_id=np.asarray(
            [f"problem_sha256:{domain}-p{row}" for row in range(count)]
        ),
        generator=np.asarray(["Meta-Llama-3.1-8B-Instruct"] * count, dtype=object),
        metadata_json=np.asarray(metadata, dtype=object),
        dataset=np.asarray([domain] * count, dtype=object),
        n_steps=np.full(count, 3),
        step_token_ranges=ranges,
        step_scores=scores,
        step_score_names=np.asarray(["token_entropy", "token_nll"]),
        response_token_ranges=np.asarray([[10, 16]] * count),
        response_token_state_files=np.asarray(files, dtype=object),
        response_token_state_layers=np.asarray([8, 10]),
        response_token_state_counts=np.full(count, 6),
        response_token_state_storage_kind=np.asarray("per_chain_npy_shards_v1"),
        response_token_state_snapshot_kind=np.asarray("raw_residual_stream"),
    )
    return TraceSource(domain, manifest, "observer_teacher_forcing_replay")


def test_experiment_separates_tasks_and_writes_auditable_artifacts(tmp_path):
    sources = tuple(
        _source(tmp_path / "data", domain, index)
        for index, domain in enumerate(("gsm8k", "math", "olympiad", "omnimath"))
    )
    output = tmp_path / "outputs"
    progress = RecordingProgress()

    result = run_hidden_geometry_experiment(
        sources=sources,
        output_dir=output,
        response_generator="llama3.1-8b",
        observer_model="llama3.1-8b",
        output_features=("token_entropy", "token_nll"),
        tasks=("whole_chain", "strict_prefix"),
        method_name="experiment_dummy",
        method_config={},
        n_boot=20,
        seed=3,
        progress=progress,
    )

    assert set(result["tasks"]) == {"whole_chain", "strict_prefix"}
    assert result["tasks"]["whole_chain"]["claim_scope"] == "retrospective_information_ceiling"
    assert result["tasks"]["strict_prefix"]["claim_scope"] == "prospective_first_error_association"
    assert result["data"]["full_vocab_logits_stored"] is False
    assert result["data"]["generation_matched_online_states"] is False
    assert json.loads((output / "results.json").read_text(encoding="utf-8"))
    for name in (
        "preflight.json",
        "results.json",
        "oof_predictions.csv",
        "fold_audit.csv",
        "model_factors.npz",
        "artifact_manifest.json",
    ):
        assert (output / name).is_file()
    with (output / "oof_predictions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["task"] for row in rows} == {"whole_chain", "strict_prefix"}
    saved_result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    saved_preflight = json.loads(
        (output / "preflight.json").read_text(encoding="utf-8")
    )
    artifact_manifest = json.loads(
        (output / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert saved_result["run_id"] == saved_preflight["run_id"]
    assert artifact_manifest["run_id"] == saved_result["run_id"]
    stages = [event[1] for event in progress.events if event[0] == "stage"]
    assert stages[:3] == ["load", "build", "evaluate"]
    assert "bootstrap" in stages
    loops = [event[1] for event in progress.events if event[0] == "start"]
    assert "problem-group bootstrap" in loops


def test_full_preflight_checks_every_shard_and_one_global_layer_schema(tmp_path):
    sources = tuple(
        _source(tmp_path / "data", domain, index)
        for index, domain in enumerate(("gsm8k", "math"))
    )
    with np.load(sources[1].manifest, allow_pickle=True) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["response_token_state_layers"] = np.asarray([9, 11])
    np.savez(sources[1].manifest, **payload)

    with pytest.raises(ValueError, match="global hidden-state schema"):
        inspect_hidden_geometry_sources(
            sources=sources,
            response_generator="llama3.1-8b",
            observer_model="llama3.1-8b",
        )

    sources = tuple(
        _source(tmp_path / "missing", domain, index)
        for index, domain in enumerate(("gsm8k", "math"))
    )
    missing = sources[1].manifest.parent / "states" / "row_101.npy"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="row_101"):
        inspect_hidden_geometry_sources(
            sources=sources,
            response_generator="llama3.1-8b",
            observer_model="llama3.1-8b",
        )


def test_json_boundary_encodes_nonfinite_metrics_as_standard_json_null():
    assert _jsonable(float("nan")) is None
    assert _jsonable(np.float64("inf")) is None


def test_production_plugin_runs_through_generic_runner_on_raw_shards(tmp_path):
    sources = tuple(
        _source(tmp_path / "data", domain, index)
        for index, domain in enumerate(("gsm8k", "math", "olympiad", "omnimath"))
    )

    result = run_hidden_geometry_experiment(
        sources=sources,
        output_dir=tmp_path / "production",
        response_generator="llama3.1-8b",
        observer_model="llama3.1-8b",
        tasks=("whole_chain", "strict_prefix"),
        method_name="raw_functional_probe",
        method_config=RawFunctionalConfig(
            pca_dim=2,
            time_basis=2,
            layer_basis=2,
            positions_per_chain=4,
            restarts=1,
            max_iter=300,
            null_repeats=1,
        ),
        n_boot=5,
        seed=11,
    )

    assert "hidden_given_output_summary_nll" in result["tasks"]["whole_chain"][
        "summary"
    ]["increments"]
    assert result["method"]["name"] == "raw_functional_probe"
