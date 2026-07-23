from __future__ import annotations

import json

import numpy as np
import pytest

from functional_divergence.hidden_state_geometry.contracts import TraceSource
from functional_divergence.hidden_state_geometry.data import (
    load_hidden_geometry_dataset,
    load_step_end_states,
)


def _write_trace_fixture(root):
    selected = root / "gsm8k" / "selected"
    shards = selected / "trace.response_states.test"
    shards.mkdir(parents=True)
    chain_ids = np.asarray([10, 11, 12, 13], dtype=np.int64)
    files = []
    ranges = np.empty(4, dtype=object)
    metadata = []
    for row, chain_id in enumerate(chain_ids):
        state = np.empty((6, 3, 8), dtype=np.float32)
        for token in range(6):
            state[token] = chain_id * 100 + token * 10 + np.arange(3)[:, None]
        path = shards / f"row_{chain_id}.npy"
        np.save(path, state)
        files.append(str(path.relative_to(selected)))
        # One padded row mirrors the packed extraction artifact.
        ranges[row] = np.asarray([[10, 11], [12, 13], [14, 15], [-1, -1]])
        metadata.append(
            json.dumps(
                {
                    "loaded_model": "meta-llama/Llama-3.1-8B-Instruct",
                    "response_generator": (
                        "Meta-Llama-3.1-8B-Instruct" if row != 2 else "Qwen2.5-7B"
                    ),
                }
            )
        )
    manifest = selected / "trace.raw_residual_stream.npz"
    np.savez(
        manifest,
        chain_idx=chain_ids,
        gold_error_step=np.asarray([1, -1, 2, -1]),
        problem_group_id=np.asarray(
            [f"problem_sha256:{value}" for value in (100, 101, 102, 103)]
        ),
        generator=np.asarray(
            [
                "Meta-Llama-3.1-8B-Instruct",
                "meta-llama/Llama-3.1-8B-Instruct",
                "Qwen2.5-7B",
                "Meta-Llama-3.1-8B-Instruct",
            ],
            dtype=object,
        ),
        metadata_json=np.asarray(metadata, dtype=object),
        dataset=np.asarray(["gsm8k"] * 4, dtype=object),
        n_steps=np.asarray([3, 3, 3, 3]),
        step_token_ranges=ranges,
        response_token_ranges=np.asarray([[10, 16]] * 4),
        response_token_state_files=np.asarray(files, dtype=object),
        response_token_state_layers=np.asarray([8, 10, 12]),
        response_token_state_counts=np.full(4, 6),
        response_token_state_storage_kind=np.asarray("per_chain_npy_shards_v1"),
        response_token_state_snapshot_kind=np.asarray("raw_residual_stream"),
    )

    # Exact/output artifact is deliberately stored in a different row order.
    order = np.asarray([13, 10, 12, 11], dtype=np.int64)
    step_scores = np.empty((4, 3, 2), dtype=np.float32)
    for row, chain_id in enumerate(order):
        step_scores[row, :, 0] = chain_id + np.arange(3)
        step_scores[row, :, 1] = chain_id * 2 + np.arange(3)
    exact = selected / "trace.npz"
    metadata_by_chain = {int(chain): metadata[row] for row, chain in enumerate(chain_ids)}
    ranges_by_chain = {int(chain): ranges[row] for row, chain in enumerate(chain_ids)}
    np.savez(
        exact,
        chain_idx=order,
        n_steps=np.full(4, 3),
        gold_error_step=np.asarray([-1, 1, 2, -1]),
        is_correct=np.asarray([1, 0, 0, 1]),
        generator=np.asarray(
            [
                "Meta-Llama-3.1-8B-Instruct",
                "Meta-Llama-3.1-8B-Instruct",
                "Qwen2.5-7B",
                "meta-llama/Llama-3.1-8B-Instruct",
            ],
            dtype=object,
        ),
        dataset=np.asarray(["gsm8k"] * 4, dtype=object),
        metadata_json=np.asarray([metadata_by_chain[int(chain)] for chain in order], dtype=object),
        step_token_ranges=np.asarray(
            [ranges_by_chain[int(chain)][:3] for chain in order], dtype=object
        ),
        step_scores=step_scores,
        step_score_names=np.asarray(["token_entropy", "token_nll"]),
    )
    return manifest, exact


def test_loader_filters_llama_rows_joins_output_by_chain_and_trims_padding(tmp_path):
    manifest, exact = _write_trace_fixture(tmp_path)

    dataset = load_hidden_geometry_dataset(
        [TraceSource("gsm8k", manifest, "observer_teacher_forcing_replay", exact)],
        response_generator="llama3.1-8b",
        observer_model="llama3.1-8b",
        output_features=("token_entropy", "token_nll"),
    )

    assert dataset.labels.tolist() == [1, 0, 0]
    assert [sample.chain_id for sample in dataset.samples] == [10, 11, 13]
    assert all(sample.step_ranges.shape == (3, 2) for sample in dataset.samples)
    assert dataset.evidence.output_evidence_kind == "teacher_forced_step_summary"
    assert dataset.evidence.full_vocab_logits_stored is False
    assert dataset.evidence.selected_records == 3
    assert np.array_equal(dataset.samples[0].output_steps[:, 0], [10, 11, 12])


def test_step_end_loader_uses_response_relative_real_shard_positions(tmp_path):
    manifest, exact = _write_trace_fixture(tmp_path)
    dataset = load_hidden_geometry_dataset(
        [TraceSource("gsm8k", manifest, "observer_teacher_forcing_replay", exact)],
        response_generator="llama3.1-8b",
        observer_model="llama3.1-8b",
        output_features=("token_entropy", "token_nll"),
    )

    states = load_step_end_states(dataset.samples[0])

    assert states.shape == (3, 3, 8)
    # Inclusive absolute step ends 11,13,15 map to response-shard rows 1,3,5.
    assert states[:, 0, 0].tolist() == [1010.0, 1030.0, 1050.0]


def test_loader_fails_when_requested_output_summary_is_not_real(tmp_path):
    manifest, exact = _write_trace_fixture(tmp_path)

    with pytest.raises(ValueError, match="output summaries"):
        load_hidden_geometry_dataset(
            [TraceSource("gsm8k", manifest, "observer_teacher_forcing_replay", exact)],
            response_generator="llama3.1-8b",
            observer_model="llama3.1-8b",
            output_features=("top1_top2_margin",),
        )


def test_loader_accepts_explicit_local_problem_ids_without_inventing_hashes(tmp_path):
    manifest, exact = _write_trace_fixture(tmp_path)
    with np.load(manifest, allow_pickle=True) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload.pop("problem_group_id")
    payload["problem_ids"] = np.asarray([100, 101, 102, 103])
    np.savez(manifest, **payload)

    dataset = load_hidden_geometry_dataset(
        [TraceSource("gsm8k", manifest, "observer_teacher_forcing_replay", exact)],
        response_generator="llama3.1-8b",
        observer_model="llama3.1-8b",
    )

    assert [sample.problem_group for sample in dataset.samples] == ["100", "101", "103"]
    assert all(sample.problem_hash is None for sample in dataset.samples)


def test_loader_fails_when_observer_is_not_the_requested_model(tmp_path):
    manifest, exact = _write_trace_fixture(tmp_path)
    with np.load(manifest, allow_pickle=True) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["metadata_json"] = np.asarray(
        [json.dumps({"loaded_model": "Qwen2.5-7B"})] * 4, dtype=object
    )
    np.savez(manifest, **payload)

    with pytest.raises(ValueError, match="observer"):
        load_hidden_geometry_dataset(
            [TraceSource("gsm8k", manifest, "observer_teacher_forcing_replay", exact)],
            response_generator="llama3.1-8b",
            observer_model="llama3.1-8b",
            output_features=("token_entropy", "token_nll"),
        )


def test_loader_rejects_duplicate_chain_ids_and_misaligned_output_boundaries(tmp_path):
    manifest, exact = _write_trace_fixture(tmp_path)
    with np.load(manifest, allow_pickle=True) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["chain_idx"] = np.asarray([10, 10, 12, 13])
    np.savez(manifest, **payload)
    source = TraceSource(
        "gsm8k", manifest, "observer_teacher_forcing_replay", exact
    )

    with pytest.raises(ValueError, match="chain_idx.*unique"):
        load_hidden_geometry_dataset(
            [source],
            response_generator="llama3.1-8b",
            observer_model="llama3.1-8b",
        )

    manifest, exact = _write_trace_fixture(tmp_path / "second")
    with np.load(exact, allow_pickle=True) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["step_token_ranges"] = payload["step_token_ranges"].copy()
    payload["step_token_ranges"][1] = np.asarray([[10, 11], [12, 13], [99, 99]])
    np.savez(exact, **payload)

    with pytest.raises(ValueError, match="step ranges disagree"):
        load_hidden_geometry_dataset(
            [TraceSource("gsm8k", manifest, "observer_teacher_forcing_replay", exact)],
            response_generator="llama3.1-8b",
            observer_model="llama3.1-8b",
        )
