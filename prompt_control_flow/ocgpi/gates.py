from __future__ import annotations

from typing import Mapping


def _ci_lower(summary: Mapping[str, object], key: str) -> float:
    value = summary.get(key, {})
    if not isinstance(value, Mapping):
        return float("nan")
    return float(value.get("ci_low", float("nan")))


def evaluate_claim_gate(
    response: Mapping[str, object],
    forecast: Mapping[str, object],
    preflight: Mapping[str, object],
) -> dict[str, object]:
    """Separate empirical mechanism, detector, and confirmatory claims."""

    response_increment = response["increment"]
    forecast_increment = forecast["increment"]
    if not isinstance(response_increment, Mapping) or not isinstance(
        forecast_increment, Mapping
    ):
        raise TypeError("increment summaries must be mappings")
    geometry = preflight.get("geometry", {})
    if not isinstance(geometry, Mapping):
        geometry = {}
    conditions = {
        "response_usable_information_ci_above_zero": _ci_lower(
            response_increment, "conditional_usable_information"
        )
        > 0.0,
        "response_beats_length_matched_null": _ci_lower(
            response_increment, "delta_auroc_vs_null"
        )
        > 0.0,
        "future_output_partial_r2_ci_above_zero": _ci_lower(
            forecast_increment, "partial_r2"
        )
        > 0.0,
        "future_output_beats_length_matched_null": _ci_lower(
            forecast_increment, "partial_r2_vs_null"
        )
        > 0.0,
        "problem_groups_at_least_100": int(response.get("problem_groups", 0)) >= 100,
        "whole_layer_geometry": bool(geometry.get("mainline_geometry_ready", False)),
        "observer_model_identity_verified": bool(
            preflight.get("observer_model_match", False)
        ),
    }
    return {
        "conditions": conditions,
        "mechanism_supported": bool(
            conditions["future_output_partial_r2_ci_above_zero"]
            and conditions["future_output_beats_length_matched_null"]
        ),
        "detector_increment_supported": bool(
            conditions["response_usable_information_ci_above_zero"]
            and conditions["response_beats_length_matched_null"]
        ),
        "confirmatory_ready": bool(all(conditions.values())),
    }
