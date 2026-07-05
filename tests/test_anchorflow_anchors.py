import numpy as np

from anchorflow.anchors import parse_anchors


def test_parse_anchors_extracts_goal_numbers_constraints():
    text = "Tom has 12 apples. He gives 3 to Ana. How many apples are left?"
    anchors = parse_anchors(text)
    kinds = {a.kind for a in anchors}
    assert "goal" in kinds
    assert "number" in kinds
    assert "constraint" in kinds
    assert any(a.text == "12" for a in anchors)
    assert any(a.text == "3" for a in anchors)
