from __future__ import annotations

import numpy as np

from prompt_control_flow.config import ExtractionConfig, MetricNames
from prompt_control_flow.data import (
    ChainRecord,
    load_chain_records,
    process_correct_from_gold,
)
from prompt_control_flow.evaluate import auprc, evaluate_all, evaluate_response
from prompt_control_flow.extraction import (
    ChainExtraction,
    extract_chain_mechanisms,
    pack_extractions,
)
from prompt_control_flow.extractors import ICRResidualMismatchExtractor, UncertaintyExtractor
from prompt_control_flow.geometry import orthonormal_basis, projection_energy_fraction
from prompt_control_flow.metrics import (
    compute_step_boundary_state_vectors,
    compute_step_prompt_flow_metrics,
    compute_response_token_layer_states,
    compute_step_residual_vectors,
    summarize_step_metrics,
)
from prompt_control_flow.replay_protocols import (
    EXACT_ARTIFACT_REPLAY,
    PROCESSBENCH_OBSERVER_CHAT_V1,
    render_processbench_observer_prompt,
    stable_problem_group_id,
)
from prompt_control_flow.representation_geometry import GeometryAuditConfig, append_geometry_audit
from prompt_control_flow.schema import inspect_npz_schema
from prompt_control_flow.spectral_chain_dynamics import SpectralChainConfig, append_spectral_chain_dynamics, canonicalize_spectral_input
from prompt_control_flow.storage import (
    FixedStateMemmap,
    ResponseStateShardWriter,
    StateStorageError,
    cast_state_array,
)
from prompt_control_flow.teacher_forcing import ForwardCache, build_prompt_response
from prompt_control_flow.visualize import response_error_labels, write_first_error_aligned_csv, write_separability_csv, write_trajectory_csv


def test_projection_fraction_respects_known_subspace() -> None:
    x = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    basis = orthonormal_basis(x, k=2, center=False).basis

    inside = projection_energy_fraction(np.asarray([[2.0, 3.0, 0.0]]), basis)
    outside = projection_energy_fraction(np.asarray([[0.0, 0.0, 4.0]]), basis)

    assert np.allclose(inside, [1.0])
    assert np.allclose(outside, [0.0])


def test_build_prompt_response_is_stable_for_processbench_steps() -> None:
    prompt, response = build_prompt_response("What is 1+1?", ["1+1=2", "Answer is 2."])

    assert prompt == "Problem: What is 1+1?\n\nSolution:\n\n"
    assert response == "1+1=2\n\nAnswer is 2."


def test_chat_observer_prompt_records_the_exact_problem_span() -> None:
    class ChatTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return f"<user>{messages[0]['content']}</user><assistant>"

    rendered = render_processbench_observer_prompt(
        ChatTokenizer(), "What is 1+1?", protocol=PROCESSBENCH_OBSERVER_CHAT_V1
    )

    a, b = rendered.question_char_span
    assert rendered.rendered_prompt[a:b] == "What is 1+1?"
    assert rendered.protocol == PROCESSBENCH_OBSERVER_CHAT_V1
    assert rendered.rendered_prompt_sha256


def test_processbench_jsonl_loader_preserves_generator_and_dataset(tmp_path) -> None:
    path = tmp_path / "gsm8k.jsonl"
    path.write_text(
        '{"id":"gsm8k-12","generator":"Llama-3.1-8B-Instruct","problem":"q","steps":["s0"],"final_answer_correct":false,"label":0}\n',
        encoding="utf-8",
    )

    rows = load_chain_records(path, input_format="processbench_jsonl")

    assert len(rows) == 1
    assert rows[0].problem_id == 12
    assert rows[0].generator == "Llama-3.1-8B-Instruct"
    assert rows[0].dataset == "gsm8k"
    assert rows[0].is_correct == 0
    assert rows[0].process_correct == 0
    assert rows[0].final_answer_correct == 0


def test_processbench_duplicate_questions_share_a_stable_group_id(tmp_path) -> None:
    path = tmp_path / "gsm8k.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id":"sample-a","problem":"same question","steps":["ok"],"label":-1}',
                '{"id":"sample-b","problem":"same   question","steps":["wrong"],"label":0}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_chain_records(path, input_format="processbench_jsonl")

    assert rows[0].problem_group_id == rows[1].problem_group_id
    assert rows[0].problem_group_id == stable_problem_group_id("same question")
    assert [row.process_correct for row in rows] == [1, 0]


def test_processbench_loader_rejects_out_of_range_process_label(tmp_path) -> None:
    path = tmp_path / "gsm8k.jsonl"
    path.write_text(
        '{"id":"bad","problem":"q","steps":["only"],"label":1}\n',
        encoding="utf-8",
    )

    with np.testing.assert_raises(ValueError):
        load_chain_records(path, input_format="processbench_jsonl")


def test_prompt_flow_detects_prompt_aligned_residual_update() -> None:
    seq_len = 7
    dim = 4
    h0 = np.zeros((seq_len, dim), dtype=np.float64)
    h0[0, 0] = 1.0
    h0[1, 1] = 1.0

    h1 = h0.copy()
    # Step 0 predicts target tokens 2 and 3 from positions 1 and 2.
    h1[1, 0] += 1.0
    h1[2, 1] += 1.0
    # Step 1 predicts target tokens 4 and 5 from positions 3 and 4.
    h1[3, 2] += 1.0
    h1[4, 3] += 1.0

    scores = compute_step_prompt_flow_metrics(
        [h0, h1],
        logits=None,
        prompt_token_indices=np.asarray([0, 1], dtype=np.int64),
        question_token_indices=np.asarray([0], dtype=np.int64),
        response_token_start=2,
        step_ranges=[(2, 3), (4, 5)],
        layers=[1],
        subspace_k=2,
        prefix_k=2,
        rng=np.random.default_rng(0),
        center_subspaces=False,
    )

    prompt_frac = scores[MetricNames.PROMPT_FRAC]
    off_prompt = scores[MetricNames.OFF_PROMPT]

    assert prompt_frac[0] > 0.99
    assert prompt_frac[1] < 0.01
    assert off_prompt[0] < 0.01
    assert off_prompt[1] > 0.99
    assert scores[MetricNames.QUESTION_FRAC][0] > 0.49


def test_summary_contains_chain_level_prompt_scores() -> None:
    series = {
        MetricNames.PROMPT_FRAC: np.asarray([0.9, 0.2]),
        MetricNames.PREFIX_LOCK_RATIO: np.asarray([0.1, 2.0]),
    }
    summary = summarize_step_metrics(series)

    assert np.isclose(summary[f"mean_{MetricNames.PROMPT_FRAC}"], 0.55)
    assert np.isclose(summary[f"max_{MetricNames.PROMPT_FRAC}"], 0.9)
    assert 0.0 <= summary["survival_prefix_lock"] <= 1.0


def test_icr_extractor_returns_finite_step_scores_on_synthetic_cache() -> None:
    h0 = np.zeros((5, 4), dtype=np.float32)
    h1 = h0.copy()
    h0[0, 0] = 1.0
    h0[1, 1] = 1.0
    h0[2, 2] = 1.0
    h0[3, 3] = 1.0
    h1[2] += np.asarray([1.0, 0.0, 0.5, 0.0], dtype=np.float32)
    h1[3] += np.asarray([0.0, 1.0, 0.0, 0.5], dtype=np.float32)
    attn = np.zeros((1, 5, 5), dtype=np.float32)
    for i in range(5):
        attn[0, i, : i + 1] = 1.0 / (i + 1)
    cache = ForwardCache(
        input_ids=np.arange(5),
        attention_mask=np.ones(5, dtype=np.int64),
        offset_mapping=[(0, 1)] * 5,
        prompt_len_tokens=2,
        prompt_token_range=(0, 2),
        question_token_range=(0, 2),
        response_start_token=2,
        response_token_range=(2, 5),
        step_token_ranges=[(2, 3)],
        hidden_states=[h0, h1],
        attentions=[attn],
        logits=None,
        seq_len=5,
    )
    cfg = ExtractionConfig(layers=(1,))
    rec = ChainRecord(chain_idx=0, problem_id=0, problem="q", steps=["s"], response="s")

    scores = ICRResidualMismatchExtractor().compute(cache, rec, cfg)

    assert scores[MetricNames.ICR_MEAN].shape == (1,)
    assert np.isfinite(scores[MetricNames.ICR_MEAN][0])


def test_step_residual_vectors_are_layer_concatenated_means() -> None:
    h0 = np.zeros((5, 3), dtype=np.float32)
    h1 = h0.copy()
    h2 = h0.copy()
    h1[1] += np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    h1[2] += np.asarray([3.0, 0.0, 0.0], dtype=np.float32)
    h2[1] += np.asarray([0.0, 2.0, 0.0], dtype=np.float32)
    h2[2] += np.asarray([0.0, 4.0, 0.0], dtype=np.float32)

    vec = compute_step_residual_vectors([h0, h1, h2], step_ranges=[(2, 3)], layers=[1, 2])

    assert vec.shape == (1, 6)
    assert np.allclose(vec[0], [2.0, 0.0, 0.0, -2.0, 3.0, 0.0])


def test_step_boundary_states_preserve_pre_step_and_step_end() -> None:
    h0 = np.zeros((5, 2), dtype=np.float32)
    h1 = np.arange(10, dtype=np.float32).reshape(5, 2)

    pre, end = compute_step_boundary_state_vectors(
        [h0, h1],
        step_ranges=[(2, 3), (4, 4)],
        layers=[1],
    )

    assert pre.shape == (2, 1, 2)
    assert np.allclose(pre[:, 0], h1[[1, 3]])
    assert np.allclose(end[:, 0], h1[[3, 4]])


def test_uncertainty_uses_only_causal_prediction_positions() -> None:
    logits = np.zeros((5, 3), dtype=np.float64)
    # Token 2 is predicted at position 1 and is highly confident.
    logits[1] = np.asarray([0.0, 0.0, 10.0])
    # Position 2 predicts token 3 and is deliberately uniform.
    logits[2] = np.asarray([0.0, 0.0, 0.0])
    # Position 3 is after the step and must never affect this one-token step.
    logits[3] = np.asarray([20.0, -20.0, -20.0])
    cache = ForwardCache(
        input_ids=np.asarray([0, 1, 2, 1, 0]),
        attention_mask=np.ones(5, dtype=np.int64),
        offset_mapping=[(i, i + 1) for i in range(5)],
        prompt_len_tokens=2,
        prompt_token_range=(0, 2),
        question_token_range=(0, 2),
        response_start_token=2,
        response_token_range=(2, 5),
        step_token_ranges=[(2, 2)],
        logits=logits,
        seq_len=5,
    )
    cfg = ExtractionConfig(layers=(1,))
    rec = ChainRecord(chain_idx=0, problem_id=0, problem="q", steps=["s"], response="s")

    scores = UncertaintyExtractor().compute(cache, rec, cfg)

    assert scores[MetricNames.TOKEN_ENTROPY][0] < 0.01
    assert scores[MetricNames.TOKEN_ENTROPY_FIRST][0] == scores[MetricNames.TOKEN_ENTROPY_LAST][0]
    assert scores[MetricNames.TOKEN_NLL][0] < 0.01


def test_response_token_states_preserve_selected_depths_and_token_axis() -> None:
    h0 = np.zeros((6, 2), dtype=np.float32)
    h1 = np.arange(12, dtype=np.float32).reshape(6, 2)
    h2 = h1 + 100.0

    states = compute_response_token_layer_states(
        [h0, h1, h2], response_token_range=(2, 6), layers=[1, 2]
    )

    assert states.shape == (4, 2, 2)
    assert np.allclose(states[:, 0], h1[2:6])
    assert np.allclose(states[:, 1], h2[2:6])


def test_torch_metric_backend_matches_numpy_semantics_when_available() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    h0 = np.zeros((6, 3), dtype=np.float32)
    h1 = h0.copy()
    h2 = h0.copy()
    h1[1:4, 0] = np.asarray([1.0, 2.0, 3.0])
    h2[1:4, 1] = np.asarray([2.0, 4.0, 6.0])
    ranges = [(2, 3), (4, 4)]
    layers = [1, 2]

    expected_residual = compute_step_residual_vectors(
        [h0, h1, h2], step_ranges=ranges, layers=layers
    )
    actual_residual = compute_step_residual_vectors(
        [torch.from_numpy(h0), torch.from_numpy(h1), torch.from_numpy(h2)],
        step_ranges=ranges,
        layers=layers,
    )
    expected_pre, expected_end = compute_step_boundary_state_vectors(
        [h0, h1, h2], step_ranges=ranges, layers=layers
    )
    actual_pre, actual_end = compute_step_boundary_state_vectors(
        [torch.from_numpy(h0), torch.from_numpy(h1), torch.from_numpy(h2)],
        step_ranges=ranges,
        layers=layers,
    )

    assert np.allclose(actual_residual, expected_residual)
    assert np.allclose(actual_pre, expected_pre, equal_nan=True)
    assert np.allclose(actual_end, expected_end, equal_nan=True)


def test_compact_output_summaries_preserve_causal_token_axis_when_available() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from prompt_control_flow.teacher_forcing import (
        compute_compact_token_output_summaries,
    )

    logits = torch.zeros((5, 4), dtype=torch.float32)
    input_ids = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
    logits[1, 2] = 12.0  # position 1 predicts target token at position 2

    summary = compute_compact_token_output_summaries(
        logits, input_ids, token_chunk_size=2
    )

    assert np.isnan(np.asarray(summary["nll"])[0])
    assert float(np.asarray(summary["nll"])[2]) < 1e-3
    assert np.shares_memory(
        np.asarray(summary["topk_mass"]), np.asarray(summary["top10_mass"])
    )

    top_five = compute_compact_token_output_summaries(
        logits, input_ids, token_chunk_size=2, top_k=5
    )
    assert "topk_mass" in top_five
    assert "top10_mass" not in top_five


def test_chain_extraction_uses_the_frozen_chat_replay_protocol(monkeypatch) -> None:
    captured = {}

    class ChatTokenizer:
        name_or_path = "toy-tokenizer"
        init_kwargs = {}

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return f"<user>{messages[0]['content']}</user><assistant>"

    class Model:
        config = type("Config", (), {"_name_or_path": "toy-model", "_commit_hash": None})()

    class ScoreExtractor:
        name = "score"
        requires_hidden = False
        requires_attention = False
        requires_logits = False

        def compute(self, cache, record, cfg):
            return {"score": np.asarray([1.0])}

    def fake_forward(_model, _tokenizer, prompt, response, **kwargs):
        captured.update(prompt=prompt, response=response, kwargs=kwargs)
        return ForwardCache(
            input_ids=np.asarray([1, 2, 3]),
            attention_mask=np.ones(3, dtype=np.int64),
            offset_mapping=[(0, 1), (1, 2), (2, 3)],
            prompt_len_tokens=2,
            prompt_token_range=(0, 2),
            question_token_range=(0, 1),
            response_start_token=2,
            response_token_range=(2, 3),
            step_token_ranges=[(2, 2)],
            seq_len=3,
            replay_kind="single_axis_no_special_fallback",
            replay_protocol=kwargs["replay_protocol"],
            prompt_provenance=kwargs["prompt_provenance"],
            messages_json=kwargs["messages_json"],
        )

    monkeypatch.setattr("prompt_control_flow.extraction.run_teacher_forcing", fake_forward)
    record = ChainRecord(
        chain_idx=1,
        problem_id=2,
        problem="What is 1+1?",
        steps=["2"],
        response="2",
    )
    item = extract_chain_mechanisms(
        Model(),
        ChatTokenizer(),
        record,
        ExtractionConfig(layers=(1,), replay_protocol=PROCESSBENCH_OBSERVER_CHAT_V1),
        [ScoreExtractor()],
    )

    assert item is not None
    assert "What is 1+1?" in captured["prompt"]
    assert captured["kwargs"]["question_text"] == "What is 1+1?"
    assert item.metadata["replay_protocol"] == PROCESSBENCH_OBSERVER_CHAT_V1


def test_exact_chain_extraction_never_replaces_the_stored_prompt(monkeypatch) -> None:
    captured = {}

    class Tokenizer:
        name_or_path = "toy-tokenizer"
        init_kwargs = {}

        def apply_chat_template(self, *_args, **_kwargs):
            raise AssertionError("exact replay must not render a new prompt")

    class Model:
        config = type("Config", (), {"_name_or_path": "toy-model", "_commit_hash": None})()

    class ScoreExtractor:
        name = "score"
        requires_hidden = False
        requires_attention = False
        requires_logits = False

        def compute(self, cache, record, cfg):
            return {"score": np.asarray([1.0])}

    def fake_forward(_model, _tokenizer, prompt, response, **kwargs):
        captured.update(prompt=prompt, kwargs=kwargs)
        return ForwardCache(
            input_ids=np.asarray([1, 2]),
            attention_mask=np.ones(2, dtype=np.int64),
            offset_mapping=[(0, 1), (1, 2)],
            prompt_len_tokens=1,
            prompt_token_range=(0, 1),
            question_token_range=(-1, -1),
            response_start_token=1,
            response_token_range=(1, 2),
            step_token_ranges=[(1, 1)],
            seq_len=2,
            replay_kind="exact_artifact_ids",
            replay_protocol=kwargs["replay_protocol"],
            prompt_provenance=kwargs["prompt_provenance"],
        )

    monkeypatch.setattr("prompt_control_flow.extraction.run_teacher_forcing", fake_forward)
    record = ChainRecord(
        chain_idx=1,
        problem_id=2,
        problem="q",
        steps=["a"],
        response="a",
        rendered_prompt="stored prompt",
        exact_input_ids=[1, 2],
        exact_attention_mask=[1, 1],
        exact_token_offsets=[(0, 1), (1, 2)],
        exact_step_token_ranges=[(1, 1)],
        exact_response_start_token=1,
    )
    item = extract_chain_mechanisms(
        Model(), Tokenizer(), record, ExtractionConfig(layers=(1,)), [ScoreExtractor()]
    )

    assert item is not None
    assert captured["prompt"] == "stored prompt"
    assert captured["kwargs"]["replay_protocol"] == EXACT_ARTIFACT_REPLAY


def test_pack_extractions_can_store_flat_step_vector_bank() -> None:
    rows = [
        ChainExtraction(
            record=ChainRecord(
                chain_idx=5,
                problem_id=9,
                problem="",
                steps=["a", "b"],
                response="a\n\nb",
                gold_error_step=-1,
                problem_group_id="problem_sha256:abc",
                process_correct=1,
                final_answer_correct=0,
                is_correct=1,
                sample_idx=0,
            ),
            step_scores={name: np.asarray([0.1, 0.2], dtype=np.float32) for name in [MetricNames.PROMPT_FRAC]},
            chain_scores={f"mean_{MetricNames.PROMPT_FRAC}": 0.15},
            n_steps=2,
            step_token_ranges=[(2, 2), (3, 3)],
            step_vectors=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            step_state_vectors=np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            layers=(8,),
        )
    ]

    packed = pack_extractions(rows)

    assert packed["step_vectors"].shape == (2, 2)
    assert packed["step_state_vectors"].shape == (2, 2)
    assert packed["step_vector_layers"].tolist() == [8]
    assert packed["step_state_vector_layers"].tolist() == [8]
    assert packed["step_vector_chain_idx"].tolist() == [5, 5]
    assert packed["step_state_vector_chain_idx"].tolist() == [5, 5]
    assert packed["step_vector_step_idx"].tolist() == [0, 1]
    assert packed["step_state_vector_step_idx"].tolist() == [0, 1]
    assert packed["problem_group_id"].tolist() == ["problem_sha256:abc"]
    assert packed["process_correct"].tolist() == [1]
    assert packed["final_answer_correct"].tolist() == [0]
    for key in [
        "chain_idx",
        "problem_id",
        "gold_error_step",
        "is_correct",
        "sample_idx",
        "generator",
        "dataset",
        "n_steps",
        "step_token_ranges",
        "step_scores",
        "step_score_names",
        "chain_scores",
        "chain_score_names",
        "layers",
    ]:
        assert key in packed


def test_state_memmap_and_response_shards_finalize_atomically(tmp_path) -> None:
    store = FixedStateMemmap(
        tmp_path / "states.partial.npy",
        tmp_path / "states.npy",
        capacity=3,
        tail_shape=(2, 2),
        dtype="float32",
    )
    store.append(np.ones((2, 2, 2), dtype=np.float32), chain_idx=4)
    store.finalize()
    saved = np.load(tmp_path / "states.npy", mmap_mode="r")
    assert saved.shape == (3, 2, 2)
    assert store.count == 2
    assert store.chain_idx == [4, 4]

    shards = ResponseStateShardWriter(
        tmp_path / "responses.partial",
        tmp_path / "responses",
        dtype="float16",
    )
    relative = shards.write(np.ones((3, 2, 2), dtype=np.float32), chain_idx=9)
    shards.finalize()
    assert relative == "responses/row_00000000_chain_00000009.npy"
    assert np.load(tmp_path / relative).shape == (3, 2, 2)


def test_response_state_shards_do_not_collide_on_duplicate_chain_ids(tmp_path) -> None:
    shards = ResponseStateShardWriter(
        tmp_path / "responses.partial",
        tmp_path / "responses",
        dtype="float32",
    )
    first = shards.write(np.zeros((1, 1, 2), dtype=np.float32), chain_idx=7)
    second = shards.write(np.ones((1, 1, 2), dtype=np.float32), chain_idx=7)

    assert first != second
    shards.finalize()
    assert (tmp_path / first).exists()
    assert (tmp_path / second).exists()


def test_float16_state_storage_fails_instead_of_overflowing() -> None:
    with np.testing.assert_raises(StateStorageError):
        cast_state_array(np.asarray([70000.0], dtype=np.float32), "float16")


def test_representation_geometry_appends_crossfit_scores() -> None:
    rng = np.random.default_rng(0)
    n_chains = 8
    n_steps = np.full(n_chains, 3, dtype=np.int64)
    gold = np.asarray([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int64)
    is_correct = np.asarray([0 if g >= 0 else 1 for g in gold], dtype=np.int64)
    chain_idx = np.arange(n_chains, dtype=np.int64)
    step_vecs = []
    flat_chain = []
    flat_step = []
    for c in range(n_chains):
        for s in range(3):
            if gold[c] >= 0 and s == gold[c]:
                center = np.asarray([5.0, 5.0, 4.0, 4.0])
            elif gold[c] >= 0 and s > gold[c]:
                center = np.asarray([3.0, 3.0, 2.0, 2.0])
            else:
                center = np.asarray([0.0, 0.0, 0.0, 0.0])
            step_vecs.append(center + 0.03 * rng.normal(size=4))
            flat_chain.append(c)
            flat_step.append(s)
    metrics = {
        "chain_idx": chain_idx,
        "problem_id": chain_idx,
        "gold_error_step": gold,
        "is_correct": is_correct,
        "sample_idx": np.full(n_chains, -1, dtype=np.int64),
        "generator": np.asarray([""] * n_chains, dtype=object),
        "dataset": np.asarray(["toy"] * n_chains, dtype=object),
        "n_steps": n_steps,
        "step_token_ranges": np.zeros((n_chains, 3, 2), dtype=np.int32),
        "step_scores": np.zeros((n_chains, 3, 1), dtype=np.float32),
        "step_score_names": np.asarray(["step_len"], dtype=object),
        "chain_scores": np.zeros((n_chains, 1), dtype=np.float32),
        "chain_score_names": np.asarray(["mean_step_len"], dtype=object),
        "layers": np.asarray([0, 1], dtype=np.int64),
        "step_state_vectors": np.asarray(step_vecs, dtype=np.float32),
        "step_state_vector_chain_idx": np.asarray(flat_chain, dtype=np.int64),
        "step_state_vector_step_idx": np.asarray(flat_step, dtype=np.int64),
        "step_state_vector_layers": np.asarray([0, 1], dtype=np.int64),
    }

    enriched = append_geometry_audit(
        metrics,
        GeometryAuditConfig(
            n_folds=4,
            knn_k=3,
            pca_var=0.9,
            max_pca_rank=2,
            random_projection_dim=3,
            layer_projection_dim=2,
            random_seed=3,
        ),
    )
    names = [str(x) for x in enriched["step_score_names"].tolist()]
    assert "geom_boundary_proj" in names
    assert "geom_compartment_score" in names
    k = names.index("geom_boundary_proj")
    assert np.isfinite(enriched["step_scores"][:, :, k]).any()
    out = evaluate_all(enriched)
    assert "representation_geometry" in out["response"]["ablation_best"]


def test_spectral_chain_dynamics_accepts_canonical_full_stepvec() -> None:
    rng = np.random.default_rng(1)
    stepvec = np.empty(10, dtype=object)
    gold = np.asarray([2, -1, 2, -1, 2, -1, 2, -1, 2, -1], dtype=np.int64)
    for c in range(stepvec.size):
        rows = []
        for s in range(4):
            healthy = np.asarray([float(s), 0.2 * s, 0.0, 0.0], dtype=np.float32)
            if gold[c] >= 0 and s >= gold[c]:
                base = np.asarray([float(s), 2.0 + s, 1.5, 1.0], dtype=np.float32)
            else:
                base = healthy
            rows.append((base + 0.02 * rng.normal(size=4)).reshape(2, 2))
        stepvec[c] = np.asarray(rows, dtype=np.float32)
    metrics = {
        "stepvec": stepvec,
        "gold_error_step": gold,
        "problem_ids": np.arange(stepvec.size, dtype=np.int64),
        "is_correct": (gold < 0).astype(np.int64),
        "sv_layers": np.asarray([10, 14], dtype=np.int64),
    }

    packed = canonicalize_spectral_input(metrics)
    assert "step_state_vectors" in packed
    assert packed["step_state_vectors"].shape == (40, 4)

    enriched = append_spectral_chain_dynamics(
        metrics,
        SpectralChainConfig(
            n_folds=5,
            n_modes=3,
            low_modes=1,
            max_landmarks=24,
            kernel_k=4,
            committor_k=5,
            tube_k=4,
            tangent_k=4,
            tangent_rank=2,
            random_seed=7,
        ),
    )
    step_names = [str(x) for x in enriched["step_score_names"].tolist()]
    chain_names = [str(x) for x in enriched["chain_score_names"].tolist()]
    assert "sd_tube_dist" in step_names
    assert "sd_committor" in step_names
    assert "sd_curve_efficiency" in chain_names
    out = evaluate_all(enriched)
    assert "spectral_chain_dynamics" in out["response"]["ablation_best"]


def test_data_loader_preserves_processbench_and_multisample_labels(tmp_path) -> None:
    path = tmp_path / "toy_multisample.npz"
    np.savez(
        path,
        problems=np.asarray(["q0", "q0"], dtype=object),
        problem_ids=np.asarray([7, 7], dtype=np.int64),
        sample_idx=np.asarray([0, 1], dtype=np.int64),
        steps_text=np.asarray([["s0", "s1"], ["s0"]], dtype=object),
        is_correct=np.asarray([1, 0], dtype=np.int64),
    )

    rows = load_chain_records(path)

    assert len(rows) == 2
    assert rows[0].problem_id == 7
    assert rows[1].sample_idx == 1
    assert rows[0].gold_error_step == -2
    assert rows[1].is_correct == 0
    assert rows[0].process_correct == -1
    assert rows[1].process_correct == -1


def test_process_correct_mapping_never_treats_unavailable_as_correct() -> None:
    assert process_correct_from_gold(-2) == -1
    assert process_correct_from_gold(-1) == 1
    assert process_correct_from_gold(0) == 0


def test_npz_loader_never_collapses_missing_problem_text_into_one_group(tmp_path) -> None:
    path = tmp_path / "legacy_multisample.npz"
    np.savez(
        path,
        problem_ids=np.asarray([7, 8], dtype=np.int64),
        sample_idx=np.asarray([0, 0], dtype=np.int64),
        steps_text=np.asarray([["a"], ["b"]], dtype=object),
        is_correct=np.asarray([1, 0], dtype=np.int64),
    )

    rows = load_chain_records(path)

    assert [row.problem_group_id for row in rows] == ["problem_id:7", "problem_id:8"]


def test_response_eval_prefers_is_correct_when_available() -> None:
    metrics = {
        "chain_scores": np.asarray([[0.1], [0.9], [0.8]], dtype=np.float64),
        "chain_score_names": np.asarray(["risk"], dtype=object),
        "gold_error_step": np.asarray([-1, -1, -1], dtype=np.int64),
        "is_correct": np.asarray([1, 0, 0], dtype=np.int64),
    }

    out = evaluate_response(metrics)

    assert out["single"]["risk"] == 1.0


def test_response_eval_excludes_unknown_labels() -> None:
    metrics = {
        "chain_scores": np.asarray([[0.1], [0.9], [100.0]], dtype=np.float64),
        "chain_score_names": np.asarray(["risk"], dtype=object),
        "gold_error_step": np.asarray([-1, 0, -2], dtype=np.int64),
        "is_correct": np.asarray([1, 0, -1], dtype=np.int64),
    }

    out = evaluate_response(metrics)

    assert out["n"] == 2
    assert out["single"]["risk"] == 1.0


def test_auprc_is_tie_invariant_and_constant_score_equals_prevalence() -> None:
    y = np.asarray([1, 0, 1, 0], dtype=np.int64)
    score = np.zeros(4, dtype=np.float64)
    assert auprc(y, score) == 0.5
    assert auprc(y[[2, 1, 0, 3]], score) == 0.5


def test_response_ablation_excludes_low_coverage_metric() -> None:
    metrics = {
        "chain_scores": np.asarray(
            [
                [0.9, 0.9],
                [0.8, np.nan],
                [0.4, np.nan],
                [0.3, np.nan],
                [0.2, 0.1],
            ],
            dtype=np.float64,
        ),
        "chain_score_names": np.asarray(
            ["mean_ltg_dense", "mean_ltg_sparse"], dtype=object
        ),
        "gold_error_step": np.asarray([1, 1, -1, -1, -1], dtype=np.int64),
        "is_correct": np.asarray([0, 0, 1, 1, 1], dtype=np.int64),
    }
    out = evaluate_response(metrics)
    assert out["metric_stats"]["mean_ltg_sparse"]["coverage"] == 0.4
    assert out["ablation_best"]["layer_time_geometry"]["best_metric"] == "mean_ltg_dense"


def test_first_error_rank_uses_expected_tie_breaking() -> None:
    metrics = {
        "step_scores": np.asarray([[[1.0], [1.0], [0.0]]], dtype=np.float64),
        "step_score_names": np.asarray(["discrete"], dtype=object),
        "gold_error_step": np.asarray([0], dtype=np.int64),
        "n_steps": np.asarray([3], dtype=np.int64),
    }
    from prompt_control_flow.evaluate import rank_first_errors

    out = rank_first_errors(metrics, "discrete")
    assert out["top1"] == 0.5
    assert out["mean_rank"] == 1.5
    assert out["mean_candidates"] == 3.0


def test_evaluate_all_first_error_and_response() -> None:
    metrics = {
        "chain_idx": np.asarray([0, 1], dtype=np.int64),
        "problem_id": np.asarray([10, 11], dtype=np.int64),
        "n_steps": np.asarray([3, 2], dtype=np.int64),
        "gold_error_step": np.asarray([1, -1], dtype=np.int64),
        "is_correct": np.asarray([0, 1], dtype=np.int64),
        "step_score_names": np.asarray(["off_prompt"], dtype=object),
        "step_scores": np.asarray([[[0.1], [0.9], [0.2]], [[0.2], [0.3], [np.nan]]], dtype=np.float64),
        "chain_score_names": np.asarray(["max_off_prompt"], dtype=object),
        "chain_scores": np.asarray([[0.9], [0.3]], dtype=np.float64),
    }

    out = evaluate_all(metrics)

    assert out["first_error"]["rows"] == 3
    assert out["first_error"]["pos"] == 1
    assert out["rank"]["off_prompt"]["top1"] == 1.0
    assert out["response"]["single"]["max_off_prompt"] == 1.0


def test_schema_inspection_distinguishes_text_from_prompt_hidden(tmp_path) -> None:
    path = tmp_path / "old_features.npz"
    np.savez(
        path,
        problems=np.asarray(["q"], dtype=object),
        steps_text=np.asarray([["s"]], dtype=object),
        stepvec=np.asarray([np.zeros((1, 1, 4), dtype=np.float32)], dtype=object),
    )

    info = inspect_npz_schema(path)

    assert info["can_reconstruct_prompt_text"] is True
    assert info["has_step_vectors"] is True
    assert info["can_compute_prompt_svd_without_reextract"] is False
    assert info["needs_teacher_forcing_reextract"] is True


def test_visualization_csv_writers_show_response_and_step_splits(tmp_path) -> None:
    metrics = {
        "chain_idx": np.asarray([0, 1], dtype=np.int64),
        "problem_id": np.asarray([10, 11], dtype=np.int64),
        "n_steps": np.asarray([3, 3], dtype=np.int64),
        "gold_error_step": np.asarray([1, -1], dtype=np.int64),
        "is_correct": np.asarray([0, 1], dtype=np.int64),
        "step_score_names": np.asarray(["prompt_frac", "off_prompt"], dtype=object),
        "step_scores": np.asarray(
            [
                [[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]],
                [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
            ],
            dtype=np.float64,
        ),
        "chain_score_names": np.asarray(["mean_off_prompt"], dtype=object),
        "chain_scores": np.asarray([[0.7], [0.3]], dtype=np.float64),
    }

    assert response_error_labels(metrics).tolist() == [1, 0]

    sep_rows = write_separability_csv(metrics, tmp_path / "sep.csv")
    traj_rows = write_trajectory_csv(metrics, tmp_path / "traj.csv", metric_names=("off_prompt",), grid_size=3)
    aligned_rows = write_first_error_aligned_csv(metrics, tmp_path / "aligned.csv", metric_names=("off_prompt",), radius=1)

    assert (tmp_path / "sep.csv").exists()
    assert (tmp_path / "traj.csv").exists()
    assert (tmp_path / "aligned.csv").exists()
    assert any(r["level"] == "response" and r["metric"] == "mean_off_prompt" for r in sep_rows)
    assert any(r["group"] == "error_response" for r in traj_rows)
    assert any(r["offset_from_first_error"] == 0 and np.isclose(r["mean"], 0.8) for r in aligned_rows)
