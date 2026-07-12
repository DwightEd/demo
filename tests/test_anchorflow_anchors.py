import numpy as np

from anchorflow.anchor_repr import _text_seed_vector, build_anchor_bank
from anchorflow.anchors import Anchor
from anchorflow.anchors import parse_anchors
from anchorflow.data import Trace


def test_parse_anchors_extracts_goal_numbers_constraints():
    text = "Tom has 12 apples. He gives 3 to Ana. How many apples are left?"
    anchors = parse_anchors(text)
    kinds = {a.kind for a in anchors}
    assert "goal" in kinds
    assert "number" in kinds
    assert "constraint" in kinds
    assert any(a.text == "12" for a in anchors)
    assert any(a.text == "3" for a in anchors)


def _trace(prompt: str, dim: int = 4) -> Trace:
    return Trace(
        idx=0,
        chain_id="stable-chain",
        problem_id=0,
        dataset="self",
        correct=True,
        gold_error_step=-1,
        step_token_ranges=np.array([[0, 1]]),
        steps_text=["done"],
        response_text="done",
        prompt_text=prompt,
        features={},
        stepvec=np.ones((1, dim)),
        qvec=np.ones(dim),
        sv_layers=[14],
        hidden_path=None,
        layer=14,
    )


def test_text_seed_vector_has_stable_digest_value():
    got = _text_seed_vector("goal:abc", 8, seed=7)
    expected = np.array([
        -0.020410763094601,
        -0.155893436172144,
        0.093858620522502,
    ])
    np.testing.assert_allclose(got[:3], expected, atol=1e-9)


def test_prompt_span_hidden_anchor_is_real_and_offset_aligned():
    prompt = "Tom has 12 apples"
    offsets = np.array([[0, 3], [4, 7], [8, 10], [11, 17]])
    hidden = np.eye(4)
    anchors = [Anchor(0, "number", "12", (8, 10))]
    bank = build_anchor_bank(
        _trace(prompt),
        anchors,
        prompt_offsets=offsets,
        prompt_hidden=hidden,
    )
    assert bank.mode == "prompt_span_hidden"
    assert bank.semantic
    assert bank.anchors[0].token_span == (2, 3)
    np.testing.assert_allclose(bank.vectors[0], np.array([0.0, 0.0, 1.0, 0.0]))


def test_missing_prompt_hidden_is_loudly_marked_fallback():
    bank = build_anchor_bank(_trace("Tom has 12 apples"), [Anchor(0, "number", "12", (8, 10))])
    assert bank.mode == "q_partition_fallback"
    assert bank.fallback_mask.tolist() == [True]
    assert not bank.semantic


def test_trace_schema_multilayer_hidden_uses_target_question_span():
    prompt = "Example has 99.\nQuestion: Tom has 12 apples left?"
    q0 = prompt.index("Tom")
    trace = _trace(prompt, dim=3)
    trace.layer = 14
    trace.features["question_char_span"] = np.array([q0, len(prompt)])
    trace.features["token_offsets"] = np.array([[i, i + 1] for i in range(len(prompt))])
    hidden = np.zeros((len(prompt), 2, 3))
    hidden[:, 0, 2] = 1.0
    hidden[:, 1, 1] = 1.0
    n0 = prompt.index("12")
    hidden[n0 : n0 + 2, 0] = np.array([1.0, 0.0, 0.0])
    trace.features["prompt_hidden"] = hidden
    trace.features["prompt_hidden_layers"] = np.array([14, 16])

    bank = build_anchor_bank(trace)
    assert bank.mode == "prompt_span_hidden"
    assert all(a.text != "99" for a in bank.anchors)
    number = next(i for i, a in enumerate(bank.anchors) if a.text == "12")
    np.testing.assert_allclose(bank.vectors[number], [1.0, 0.0, 0.0])
