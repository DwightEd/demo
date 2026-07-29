from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT.parent
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from hypergraph.attention.cct.hypergraph import CausalHypergraphBuilder
from hypergraph.attention.cct.training import TrainingConfig


@pytest.mark.parametrize("invalid_value", (float("nan"), float("inf"), float("-inf")))
@pytest.mark.parametrize("field", ("min_effect", "min_synergy"))
def test_cct_hypergraph_builder_rejects_nonfinite_thresholds(
    field: str,
    invalid_value: float,
) -> None:
    arguments = {"min_effect": 0.01, "min_synergy": 0.01}
    arguments[field] = invalid_value

    with pytest.raises(ValueError):
        CausalHypergraphBuilder(**arguments)


@pytest.mark.parametrize("invalid_value", (float("nan"), float("inf"), float("-inf")))
@pytest.mark.parametrize(
    "field",
    ("learning_rate", "weight_decay", "gradient_clip"),
)
def test_cct_training_config_rejects_nonfinite_optimizer_values(
    field: str,
    invalid_value: float,
) -> None:
    arguments = {
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
    }
    arguments[field] = invalid_value

    with pytest.raises(ValueError):
        TrainingConfig(**arguments).validate()
