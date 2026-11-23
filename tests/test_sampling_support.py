from __future__ import annotations

from dataclasses import replace

import pytest

from damage_engine.errors import InputValidationError
from damage_engine.sampling_support import (
    CandidateSupportObservation,
    SupportReachabilityEvidence,
    audit_sampling_support,
    plan_outcome_blind_expansion,
)


def _row(
    candidate_id: str,
    *,
    stratum: str = "approach-a|profile-a",
    admitted: bool = True,
    proxy_complete: bool | None = True,
    rho: float = 0.25,
) -> CandidateSupportObservation:
    return CandidateSupportObservation(
        candidate_id=candidate_id,
        design_seed=17,
        stratum=stratum,
        proposal_id="sphere-5d-reference",
        s_contact_admitted=admitted,
        proxy_domain_complete=proxy_complete if admitted else None,
        inclusion_probability=rho,
        refusal_code=None if admitted else "no_forward_armor_intersection",
    )


def test_support_audit_reports_contact_proxy_and_positivity_separately() -> None:
    rows = [
        _row("a1", rho=0.25),
        _row("a2", proxy_complete=False, rho=0.10),
        _row("a3", admitted=False, rho=0.05),
        _row("a4", admitted=False, rho=0.00),
    ]

    report = audit_sampling_support(
        rows,
        required_strata=("approach-a|profile-a",),
        minimum_attempt_count=4,
        minimum_contact_count=2,
        minimum_proxy_complete_count=1,
        minimum_inclusion_probability=0.05,
    )
    cell = report["cells"]["approach-a|profile-a"]

    assert cell["attempt_count"] == 4
    assert cell["s_contact_admitted_count"] == 2
    assert cell["proxy_complete_count"] == 1
    assert cell["contact_rate"] == 0.5
    assert cell["proxy_complete_rate_given_contact"] == 0.5
    assert cell["positive_inclusion_count"] == 3
    assert cell["zero_inclusion_count"] == 1
    assert cell["minimum_positive_inclusion_probability"] == 0.05
    assert cell["checks"]["positive_support_for_every_candidate"] is False
    assert report["overall_pass"] is False


@pytest.mark.parametrize(
    ("geometry_reachable", "proposal_overlap", "expected"),
    [
        (False, None, "structurally_unreachable"),
        (True, False, "proposal_range_error"),
        (True, True, "budget_insufficient"),
        (None, None, "indeterminate_support_gap"),
    ],
)
def test_support_gap_classification_requires_the_right_evidence(
    geometry_reachable: bool | None,
    proposal_overlap: bool | None,
    expected: str,
) -> None:
    rows = [_row(f"r{index}", admitted=False) for index in range(8)]
    evidence = SupportReachabilityEvidence(
        stratum="approach-a|profile-a",
        geometry_reachable=geometry_reachable,
        proposal_overlaps_reachable_region=proposal_overlap,
        method_id="frozen-geometry-audit@1.0.0",
        evidence_hash="sha256:test-evidence",
    )

    report = audit_sampling_support(
        rows,
        required_strata=("approach-a|profile-a",),
        minimum_attempt_count=8,
        minimum_contact_count=4,
        minimum_proxy_complete_count=2,
        minimum_inclusion_probability=0.01,
        reachability_evidence=(evidence,),
    )

    assert report["cells"]["approach-a|profile-a"]["gap_classification"] == expected


def test_adequate_cell_is_not_mislabeled_as_budget_gap() -> None:
    rows = [_row(f"r{index}") for index in range(8)]
    evidence = SupportReachabilityEvidence(
        stratum="approach-a|profile-a",
        geometry_reachable=True,
        proposal_overlaps_reachable_region=True,
        method_id="frozen-geometry-audit@1.0.0",
        evidence_hash="sha256:test-evidence",
    )

    report = audit_sampling_support(
        rows,
        required_strata=("approach-a|profile-a",),
        minimum_attempt_count=8,
        minimum_contact_count=4,
        minimum_proxy_complete_count=2,
        minimum_inclusion_probability=0.01,
        reachability_evidence=(evidence,),
    )

    assert report["cells"]["approach-a|profile-a"]["gap_classification"] == "adequate"
    assert report["overall_pass"] is True


def test_expansion_plan_is_outcome_blind_deterministic_and_capped() -> None:
    rows = [
        *[_row(f"hit-{index}") for index in range(2)],
        *[_row(f"miss-{index}", admitted=False) for index in range(8)],
    ]
    evidence = SupportReachabilityEvidence(
        stratum="approach-a|profile-a",
        geometry_reachable=True,
        proposal_overlaps_reachable_region=True,
        method_id="frozen-geometry-audit@1.0.0",
        evidence_hash="sha256:test-evidence",
    )
    report = audit_sampling_support(
        rows,
        required_strata=("approach-a|profile-a",),
        minimum_attempt_count=10,
        minimum_contact_count=8,
        minimum_proxy_complete_count=6,
        minimum_inclusion_probability=0.01,
        reachability_evidence=(evidence,),
    )

    first = plan_outcome_blind_expansion(
        report,
        maximum_additional_attempts_per_stratum=20,
        safety_factor=1.25,
    )
    repeated = plan_outcome_blind_expansion(
        report,
        maximum_additional_attempts_per_stratum=20,
        safety_factor=1.25,
    )

    assert first == repeated
    allocation = first["allocations"][0]
    assert allocation["stratum"] == "approach-a|profile-a"
    assert allocation["additional_attempts"] == 20
    assert allocation["capped"] is True
    assert allocation["uses_teacher_outcomes"] is False


def test_support_observation_contract_fails_closed() -> None:
    row = _row("a1")

    with pytest.raises(InputValidationError, match="duplicate"):
        audit_sampling_support(
            [row, row],
            required_strata=("approach-a|profile-a",),
            minimum_attempt_count=1,
            minimum_contact_count=1,
            minimum_proxy_complete_count=1,
            minimum_inclusion_probability=0.01,
        )

    with pytest.raises(InputValidationError, match="inclusion_probability"):
        audit_sampling_support(
            [replace(row, inclusion_probability=1.1)],
            required_strata=("approach-a|profile-a",),
            minimum_attempt_count=1,
            minimum_contact_count=1,
            minimum_proxy_complete_count=1,
            minimum_inclusion_probability=0.01,
        )
