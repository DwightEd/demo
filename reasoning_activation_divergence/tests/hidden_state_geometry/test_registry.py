from __future__ import annotations

import pytest

from functional_divergence.hidden_state_geometry.registry import (
    ContrastSpec,
    RandomizationSpec,
    available_methods,
    create_method,
    method_spec,
    register_method,
)


def test_method_registry_adds_a_new_strategy_without_changing_the_runner():
    contrast = ContrastSpec("gain", "base", "candidate")
    randomization = RandomizationSpec("axis", "null_r", "candidate", 2)

    @register_method(
        "test_dummy_strategy",
        contrasts=(contrast,),
        randomizations=(randomization,),
        arm_definitions={"base": "controls", "candidate": "controls + signal"},
        default_config=lambda: {"rank": 2},
    )
    class DummyMethod:
        def __init__(self, config):
            self.config = config

    method = create_method("test_dummy_strategy", {"rank": 1})

    assert method.config == {"rank": 1}
    assert create_method("test_dummy_strategy", None).config == {"rank": 2}
    assert method_spec("test_dummy_strategy").contrasts == (contrast,)
    assert method_spec("test_dummy_strategy").randomizations == (randomization,)
    assert "test_dummy_strategy" in available_methods()


def test_method_registry_rejects_duplicate_names():
    with pytest.raises(ValueError, match="already registered"):

        @register_method("test_dummy_strategy")
        class DuplicateMethod:
            pass
