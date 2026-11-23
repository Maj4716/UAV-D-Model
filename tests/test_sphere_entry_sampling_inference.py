from __future__ import annotations

from dataclasses import replace

import pytest

from damage_engine.errors import InputValidationError
from damage_engine.sampling_inference import (
    SamplingObservation,
    coupled_stratified_multiplier_bootstrap,
    evaluate_coverage_admission,
    evaluate_reference_replacement,
    sampling_metric_summary,
)


def _observation(
    *,
    arm: str,
    ray: str,
    candidate_group: str | None = None,
    domain_complete: bool = True,
    tree_saturated: bool | None = False,
    rho: float = 1.0,
    stratum: str = "approach-a|profile-a",
    proxy_domain_complete: bool = True,
) -> SamplingObservation:
    return SamplingObservation(
        arm=arm,
        physical_ray_id=ray,
        candidate_group_id=candidate_group or ray,
        design_seed=17,
        stratum=stratum,
        inclusion_probability=rho,
        design_weight_raw=1.0 / rho,
        domain_complete=domain_complete,
        tree_saturated=tree_saturated,
        positive_component_count=3.0 if domain_complete else None,
        elapsed_seconds=0.25,
        proxy_domain_complete=proxy_domain_complete,
    )


def test_summary_separates_raw_selected_from_hajek_frame() -> None:
    rows = [
        _observation(arm="candidate", ray="r1", rho=1.0),
        _observation(
            arm="candidate",
            ray="r2",
            domain_complete=False,
            tree_saturated=None,
            rho=0.25,
        ),
    ]

    summary = sampling_metric_summary(rows)

    assert summary["raw_selected"]["domain_complete_fraction"] == 0.5
    assert summary["raw_selected"]["complete_non_saturated_yield"] == 0.5
    assert summary["hajek_frame"]["domain_complete_fraction"] == 0.2
    assert summary["hajek_frame"]["complete_non_saturated_yield"] == 0.2
    assert summary["hajek_frame"]["tree_saturation_given_complete"] == 0.0


def test_domain_censored_is_zero_in_primary_yield_not_dropped() -> None:
    rows = [
        _observation(arm="candidate", ray="r1"),
        _observation(
            arm="candidate",
            ray="r2",
            domain_complete=False,
            tree_saturated=None,
        ),
    ]

    summary = sampling_metric_summary(rows)

    assert summary["raw_selected"]["complete_non_saturated_yield"] == 0.5
    assert summary["raw_selected"]["tree_saturation_given_complete"] == 0.0


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"inclusion_probability": 0.0}, "inclusion_probability"),
        ({"inclusion_probability": 1.1}, "inclusion_probability"),
        ({"design_weight_raw": 9.0}, "reciprocal"),
    ],
)
def test_invalid_probability_or_weight_fails_closed(mutation, match) -> None:
    row = replace(_observation(arm="candidate", ray="r1"), **mutation)

    with pytest.raises(InputValidationError, match=match):
        sampling_metric_summary([row])


def test_duplicate_arm_cluster_key_fails_closed() -> None:
    row = _observation(arm="candidate", ray="r1")

    with pytest.raises(InputValidationError, match="duplicate"):
        sampling_metric_summary([row, row])


def test_same_physical_ray_can_appear_in_distinct_profile_candidate_groups() -> None:
    rows = [
        _observation(
            arm="candidate",
            ray="shared-geometry",
            candidate_group="profile-a-candidate",
            stratum="approach-a|profile-a",
        ),
        _observation(
            arm="candidate",
            ray="shared-geometry",
            candidate_group="profile-b-candidate",
            stratum="approach-a|profile-b",
        ),
    ]

    summary = sampling_metric_summary(rows)

    assert summary["row_count"] == 2
    assert summary["candidate_cluster_count"] == 2
    assert summary["unique_physical_ray_count"] == 1


def test_common_clusters_share_multiplier_across_arms() -> None:
    reference = [
        _observation(arm="reference", ray="r1"),
        _observation(
            arm="reference", ray="r2", tree_saturated=True, rho=0.5
        ),
    ]
    candidate = [replace(row, arm="candidate") for row in reference]

    report = coupled_stratified_multiplier_bootstrap(
        {"reference": reference, "candidate": candidate},
        contrasts=(("candidate", "reference"),),
        draws=64,
        seed=29,
        retain_draws=True,
    )
    metric = report["contrasts"]["candidate_minus_reference"]["metrics"][
        "complete_non_saturated_yield"
    ]

    assert metric["hajek_difference"] == 0.0
    assert metric["percentile_95_interval"] == [0.0, 0.0]
    assert set(metric["draw_values"]) == {0.0}


def test_stratified_multiplier_preserves_each_stratum_mass() -> None:
    reference = [
        _observation(
            arm="reference",
            ray="r1",
            tree_saturated=True,
            stratum="approach-a|profile-a",
        ),
        _observation(
            arm="reference",
            ray="r2",
            tree_saturated=False,
            stratum="approach-b|profile-a",
        ),
    ]
    candidate = [
        replace(reference[0], arm="candidate", tree_saturated=False),
        replace(reference[1], arm="candidate"),
    ]

    report = coupled_stratified_multiplier_bootstrap(
        {"reference": reference, "candidate": candidate},
        contrasts=(("candidate", "reference"),),
        draws=32,
        seed=41,
        retain_draws=True,
    )
    metric = report["contrasts"]["candidate_minus_reference"]["metrics"][
        "complete_non_saturated_yield"
    ]

    assert metric["hajek_difference"] == 0.5
    assert set(metric["draw_values"]) == {0.5}


def test_cluster_stratum_mismatch_across_arms_fails_closed() -> None:
    reference = [_observation(arm="reference", ray="shared")]
    candidate = [
        _observation(
            arm="candidate",
            ray="shared",
            stratum="approach-b|profile-a",
        )
    ]

    with pytest.raises(InputValidationError, match="stratum"):
        coupled_stratified_multiplier_bootstrap(
            {"reference": reference, "candidate": candidate},
            contrasts=(("candidate", "reference"),),
            draws=16,
            seed=3,
        )


def test_bootstrap_seed_replay_is_stable() -> None:
    arms = {
        "reference": [
            _observation(arm="reference", ray="r1"),
            _observation(arm="reference", ray="r2", tree_saturated=True),
        ],
        "candidate": [
            _observation(arm="candidate", ray="r1"),
            _observation(arm="candidate", ray="r3"),
        ],
    }
    kwargs = {
        "contrasts": (("candidate", "reference"),),
        "draws": 32,
        "seed": 101,
        "retain_draws": True,
    }

    assert coupled_stratified_multiplier_bootstrap(
        arms, **kwargs
    ) == coupled_stratified_multiplier_bootstrap(arms, **kwargs)


def _passing_report() -> dict:
    reference = [
        _observation(arm="reference", ray="r1", tree_saturated=True),
        _observation(arm="reference", ray="r2", tree_saturated=True),
    ]
    candidate = [
        _observation(arm="candidate", ray="r1"),
        _observation(arm="candidate", ray="r2"),
    ]
    return coupled_stratified_multiplier_bootstrap(
        {"reference": reference, "candidate": candidate},
        contrasts=(("candidate", "reference"),),
        draws=32,
        seed=7,
    )


def _coverage_contract() -> dict:
    return {
        "candidate_arm": "candidate",
        "reference_arm": "reference",
        "primary_metric": "complete_non_saturated_yield",
        "primary_lower_bound_min": 0.0,
        "domain_noninferiority_margin": 0.03,
        "minimum_ess_fraction": 0.5,
        "maximum_weight_cv": 2.0,
        "minimum_inclusion_probability": 0.08,
        "minimum_arm_count": 2,
        "required_strata": ["approach-a|profile-a"],
        "s_contact_all_pass": True,
        "key_component_guardrails_pass": True,
        "cost_guardrail_pass": True,
    }


def test_coverage_admission_uses_primary_superiority_and_guardrails() -> None:
    admission = evaluate_coverage_admission(
        _passing_report(), _coverage_contract()
    )

    assert admission["checks"]["primary_interval_superior"] is True
    assert admission["checks"]["domain_noninferior"] is True
    assert admission["overall_pass"] is True


def test_coverage_admission_requires_contact_component_and_cost_guardrails() -> None:
    contract = _coverage_contract()
    contract["s_contact_all_pass"] = False

    admission = evaluate_coverage_admission(_passing_report(), contract)

    assert admission["checks"]["s_contact_all_pass"] is False
    assert admission["overall_pass"] is False


def _replacement_contract(*, positive_support: int) -> dict:
    return {
        "target_measure_hash_match": True,
        "estimand_hash_match": True,
        "sealed_test_unused_for_tuning": True,
        "physical_domain_not_shrunk": True,
        "independent_frame_count": 3,
        "minimum_independent_frames": 3,
        "support_audit": {
            "proxy_partial_candidate_count": 7,
            "proxy_partial_positive_inclusion_count": positive_support,
            "minimum_inclusion_probability": (
                0.10 if positive_support == 7 else 0.0
            ),
            "outcome_blind": True,
        },
    }


def test_reference_replacement_fixed_false_when_proxy_partial_support_is_zero() -> None:
    coverage = evaluate_coverage_admission(
        _passing_report(), _coverage_contract()
    )

    admission = evaluate_reference_replacement(
        coverage, _replacement_contract(positive_support=0)
    )

    assert admission["checks"]["proxy_partial_positive_support"] is False
    assert admission["overall_pass"] is False
    assert "zero_support_for_proxy_partial_candidates" in admission["reason_codes"]


def test_reference_replacement_passes_only_with_complete_positive_audit_support() -> None:
    coverage = evaluate_coverage_admission(
        _passing_report(), _coverage_contract()
    )

    admission = evaluate_reference_replacement(
        coverage, _replacement_contract(positive_support=7)
    )

    assert admission["checks"]["proxy_partial_positive_support"] is True
    assert admission["overall_pass"] is True
