"""Same-problem feasible-tangent validation.

This package deliberately stops before output-logit conditioning.  It first
tests whether independently sampled correct responses define a reproducible,
low-rank transition space and whether erroneous responses leave that space.
"""

from .evaluate import FeasibleTangentValidationConfig, evaluate_feasible_tangent
from .report import write_feasible_tangent_report
from .schema import (
    CHAIN_SCORE_NAMES,
    TRANSITION_SCORE_NAMES,
    FeasibleTangentConfig,
    FeasibleTangentResult,
)
from .scoring import run_feasible_tangent_gate

__all__ = [
    "CHAIN_SCORE_NAMES",
    "TRANSITION_SCORE_NAMES",
    "FeasibleTangentConfig",
    "FeasibleTangentResult",
    "FeasibleTangentValidationConfig",
    "evaluate_feasible_tangent",
    "run_feasible_tangent_gate",
    "write_feasible_tangent_report",
]
