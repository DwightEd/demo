from anchorflow.intervention import (
    apply_micro_replay,
    build_micro_replay,
    build_text_micro_replay,
    select_low_risk_candidate,
)


def test_token_micro_replay_is_pure_and_rolls_back_before_trigger():
    tokens = [10, 11, 12, 13, 14]
    before = list(tokens)
    plan = build_micro_replay(tokens, 4, rollback=1, repair_instruction=[99])
    assert tokens == before
    assert plan.cut_index == 3
    assert plan.prefix == (10, 11, 12)
    assert plan.removed_suffix == (13, 14)
    assert plan.model_input == (10, 11, 12, 99)
    assert apply_micro_replay(plan, [20, 21]) == (10, 11, 12, 20, 21)


def test_text_micro_replay_and_candidate_selection():
    text = "good\nbad\nlater"
    spans = [(0, 5), (5, 9), (9, 14)]
    plan = build_text_micro_replay(text, spans, 1, repair_instruction="\nredo\n")
    assert plan.prefix == "good\n"
    assert plan.removed_suffix == "bad\nlater"
    candidate, idx = select_low_risk_candidate(["a", "b", "c"], [0.8, 0.2, 0.2])
    assert (candidate, idx) == ("b", 1)
