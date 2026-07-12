import numpy as np

from anchorflow.anchor_repr import build_anchor_bank
from anchorflow.anchors import parse_anchors
from anchorflow.data import load_traces, make_labels
from anchorflow_validation_suite import audit_npz


def _objects(values):
    out = np.empty(len(values), dtype=object)
    out[:] = values
    return out


def test_exact_trace_schema_loads_semantic_anchors_and_step_clouds(tmp_path):
    prompt = "Demo 99. Problem: Tom has 7 and loses 2. How many remain?\nSolution:"
    question = "Tom has 7 and loses 2. How many remain?"
    q0 = prompt.index(question)
    offsets = np.asarray([(i, i + 1) for i in range(len(prompt) + 6)], int)
    prompt_hidden = np.zeros((len(prompt), 1, 3), dtype=np.float32)
    prompt_hidden[:, 0, 0] = 1.0
    prompt_hidden[q0 : q0 + len(question), 0, 1] = 2.0
    clouds = np.arange(18, dtype=float).reshape(6, 1, 3)

    path = tmp_path / "trace.npz"
    np.savez(
        path,
        trace_schema_version=np.array("exact_generation_trace_v1"),
        time_axis_kind=np.array("kept_step_index"),
        gold_error_step=np.array([2]),
        problem_ids=np.array([4]),
        ids=np.array(["c0"], dtype=object),
        step_token_ranges=_objects([np.array([[len(prompt), len(prompt) + 1],
                                                   [len(prompt) + 2, len(prompt) + 5]])]),
        kept_steps=_objects([np.array([0, 2])]),
        steps_text=_objects([["a", "b"]]),
        prompts=np.array([prompt], dtype=object),
        responses=np.array(["abcdef"], dtype=object),
        token_offsets=_objects([offsets]),
        prompt_token_counts=np.array([len(prompt)]),
        question_char_spans=np.array([[q0, q0 + len(question)]]),
        prompt_hidden=_objects([prompt_hidden]),
        prompt_hidden_layers=np.array([16]),
        qvec=_objects([np.array([0.0, 1.0, 0.0])]),
        sv_clouds=_objects([clouds]),
        cloud_sizes=_objects([np.array([2, 4])]),
        cloud_layers=np.array([16]),
    )

    traces, meta = load_traces(str(path), layer=16)
    trace = traces[0]

    assert trace.gold_error_step == 1
    assert trace.qvec.tolist() == [0.0, 1.0, 0.0]
    assert trace.prompt_hidden.shape == (len(prompt), 3)
    assert len(trace.step_clouds) == 2
    assert [len(x) for x in trace.step_clouds] == [2, 4]
    assert meta["has_exact_trace"] is True

    anchors = parse_anchors(prompt, char_span=trace.question_char_span)
    bank = build_anchor_bank(trace, anchors)
    assert bank.semantic
    assert not any("99" in anchor.text for anchor in bank.anchors)
    automatic_bank = build_anchor_bank(trace)
    assert automatic_bank.semantic
    assert not any("99" in anchor.text for anchor in automatic_bank.anchors)

    y, mask = make_labels(trace)
    assert y.tolist() == [0, 1]
    assert mask.tolist() == [True, True]

    audit = audit_npz(str(path), layer=16)
    assert audit["status"] == "passed"
    assert audit["semantic_anchor_ready"] == 1
    assert audit["conditional_geometry_ready"] == 1


def test_unlocalized_incorrect_chain_is_not_treated_as_clean_risk_set(tmp_path):
    path = tmp_path / "chain_only.npz"
    np.savez(
        path,
        is_correct=np.array([0]),
        problem_ids=np.array([0]),
        ids=np.array(["bad"], dtype=object),
        step_token_ranges=_objects([np.array([[0, 1], [2, 3]])]),
    )

    trace = load_traces(str(path))[0][0]
    y, mask = make_labels(trace)
    assert y.tolist() == [0, 0]
    assert mask.tolist() == [False, False]
