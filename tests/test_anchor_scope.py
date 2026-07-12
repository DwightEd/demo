from anchorflow.anchors import parse_anchors


def test_parse_anchors_restricts_few_shot_prompt_to_target_question():
    demo = "Example: Alice has 99 apples. How many remain?\n"
    target = "Tom has 7 books and gives away 2. How many books are left?"
    prompt = demo + "Problem: " + target + "\nSolution:"
    start = prompt.index(target)

    anchors = parse_anchors(prompt, char_span=(start, start + len(target)))

    assert all(a.char_span is not None for a in anchors)
    assert all(start <= a.char_span[0] < a.char_span[1] <= start + len(target)
               for a in anchors)
    assert any(a.kind == "number" and a.text == "7" for a in anchors)
    assert any(a.kind == "number" and a.text == "2" for a in anchors)
    assert not any("99" in a.text or "Alice" in a.text for a in anchors)


def test_goal_span_preserves_full_prompt_offsets_with_leading_whitespace():
    question = "   First sentence. What is 3 plus 4?   "
    anchors = parse_anchors(question)
    goal = anchors[0]

    assert goal.kind == "goal"
    assert goal.text == "What is 3 plus 4?"
    assert question[goal.char_span[0]:goal.char_span[1]] == goal.text
