from __future__ import annotations

from functional_divergence.causal_first_error_attribution.analysis import (
    InterventionRecord,
    summarize_interventions,
)


def _records(pair_kind: str, *, attention: float, random: float):
    records = []
    for domain in ("gsm8k", "math"):
        for index in range(6):
            records.append(
                InterventionRecord(
                    case_id=f"{domain}-{index}",
                    domain=domain,
                    problem_hash=f"{domain}-problem-{index}",
                    pair_kind=pair_kind,
                    attention_effect=attention,
                    ffn_effect=0.0,
                    interaction_effect=0.0,
                    pre_state_effect=0.0,
                    random_donor_attention=random,
                    wrong_source_attention=random,
                    random_donor_ffn=0.0,
                    token_rescue=True,
                    step_rescue=True,
                )
            )
    return records


def test_random_donor_and_wrong_source_do_not_pass_routing_gate() -> None:
    supported = summarize_interventions(
        _records("controlled_root", attention=2.0, random=0.0),
        n_boot=200,
        seed=3,
    )
    matched_by_null = summarize_interventions(
        _records("controlled_root", attention=2.0, random=2.0),
        n_boot=200,
        seed=3,
    )

    assert supported["claims"]["routing_root_cause"] == "supported"
    assert supported["claims"]["ffn_root_cause"] == "unsupported"
    assert matched_by_null["claims"]["routing_root_cause"] == "unsupported"


def test_natural_rescue_never_becomes_root_cause_claim() -> None:
    report = summarize_interventions(
        _records("target_correction", attention=4.0, random=0.0),
        n_boot=200,
        seed=5,
    )

    assert report["claims"]["routing_root_cause"] == "not_identifiable"
    assert report["claims"]["ffn_root_cause"] == "not_identifiable"
    assert report["claims"]["natural_rescue"] == "supported"
    assert report["cases"] == 12
    assert len(report["individual_effects"]) == 12

