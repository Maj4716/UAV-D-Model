from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.evaluate_multicondition_penetration import (
    FORMAL_MIN_REPLICATES,
    _monotonic_checks,
    aggregate_equal_weight_conditions,
    evaluate_condition_gates,
    materialize_condition,
    resolve_replicate_budgets,
    resolve_condition_sets,
    select_admissible_candidate,
    study_mode_contract,
    source_identity_drift,
    validate_formal_study_plan,
    validate_replicate_budgets,
)


ROOT = Path(__file__).resolve().parents[1]


def test_formal_and_diagnostic_modes_have_distinct_replicate_contracts():
    assert FORMAL_MIN_REPLICATES == 299
    assert resolve_replicate_budgets("formal", None, None, None) == (
        299,
        299,
        299,
    )
    assert resolve_replicate_budgets("diagnostic", None, None, None) == (
        32,
        64,
        128,
    )
    assert validate_replicate_budgets(299, 299, 299, mode="formal") == (
        299,
        299,
        299,
    )
    with pytest.raises(ValueError, match="formal selection replicate count"):
        validate_replicate_budgets(298, 299, 299, mode="formal")
    assert validate_replicate_budgets(32, 64, 128, mode="diagnostic") == (
        32,
        64,
        128,
    )


def test_diagnostic_mode_contract_is_explicitly_non_promotable():
    diagnostic = study_mode_contract("diagnostic")
    formal = study_mode_contract("formal")

    assert diagnostic["promotion_permitted"] is False
    assert diagnostic["interpretation_status"] == "diagnostic_only_non_promotable"
    assert formal["promotion_permitted"] is True
    assert formal["interpretation_status"] == "formal_software_admission"


def test_formal_plan_requires_sealed_holdout_and_independent_phase_seeds():
    plan = {
        "holdout_evidence_status": "sealed_unobserved",
        "conditions": [
            {"id": "s", "role": "selection", "selection_seed": 11},
            {"id": "c", "role": "selection", "confirmation_seed": 29},
            {"id": "h", "role": "holdout", "holdout_seed": 47},
        ],
        "selection_condition_ids": ["s"],
        "confirmation_condition_ids": ["c"],
        "holdout_condition_ids": ["h"],
    }

    validate_formal_study_plan(plan)

    consumed = deepcopy(plan)
    consumed["holdout_evidence_status"] = "consumed_by_prior_soft_gate_run"
    with pytest.raises(ValueError, match="sealed_unobserved"):
        validate_formal_study_plan(consumed)

    missing_holdout_seed = deepcopy(plan)
    del missing_holdout_seed["conditions"][2]["holdout_seed"]
    with pytest.raises(ValueError, match="holdout_seed"):
        validate_formal_study_plan(missing_holdout_seed)

    reused_seed = deepcopy(plan)
    reused_seed["conditions"][2]["holdout_seed"] = 29
    with pytest.raises(ValueError, match="independent"):
        validate_formal_study_plan(reused_seed)


def test_materialized_conditions_are_replayable_and_phase_seeds_are_disjoint(case_data):
    spec = json.loads(
        (ROOT / "configs" / "sampling" / "contact_pose.smoke.v1.json")
        .read_text(encoding="utf-8")
    )
    condition = {
        "id": "t-left",
        "role": "selection",
        "profile_id": "medium_front",
        "latent": [1 / 12, 0.5, 0.5, 0.5, 0.25, 0.5],
        "selection_seed": 11,
        "confirmation_seed": 29,
    }

    selection = materialize_condition(
        deepcopy(case_data), spec["contact_manifold"], condition, phase="selection"
    )
    replay = materialize_condition(
        deepcopy(case_data), spec["contact_manifold"], condition, phase="selection"
    )
    confirmation = materialize_condition(
        deepcopy(case_data), spec["contact_manifold"], condition, phase="confirmation"
    )

    assert selection == replay
    assert selection["munition"]["profile_id"] == "medium_front"
    assert selection["analysis"]["aleatory_seed"] == 11
    assert confirmation["analysis"]["aleatory_seed"] == 29
    assert selection["analysis"]["epistemic_sample_id"] != confirmation["analysis"][
        "epistemic_sample_id"
    ]


def test_multicondition_aggregation_is_equal_weight_not_fragment_count_weighted():
    objective = {
        "armor_penetration_rate_target": 0.5,
        "component_penetration_rate_target": 0.5,
        "damage_positive_given_hit_target": 0.5,
        "mean_degraded_given_hit_target": 0.4,
        "mean_failed_given_hit_target": 0.2,
    }
    base = {
        "armor_penetration_rate": 0.2,
        "component_penetration_rate": 0.3,
        "damage_positive_given_hit": 0.4,
        "mean_degraded_given_hit": 0.3,
        "mean_failed_given_hit": 0.1,
        "saturated_damage_fraction": 0.0,
        "ordinal_violation_count": 0,
        "ordinal_projection_max_correction": 0.0,
        "energy_closure_max_abs_j": 0.0,
        "expected_hit_components_per_replicate": 3.0,
        "expected_hit_component_observations": 48.0,
    }
    large = {**base, "armor_penetration_rate": 0.8, "mean_armor_hits": 300.0}
    small = {**base, "armor_penetration_rate": 0.2, "mean_armor_hits": 10.0}
    rows = [
        {"condition_id": "small", "metrics": small},
        {"condition_id": "large", "metrics": large},
    ]

    aggregate = aggregate_equal_weight_conditions(rows, objective)

    assert aggregate["equal_weight_metrics"]["armor_penetration_rate"] == pytest.approx(0.5)
    assert aggregate["condition_count"] == 2


def test_multicondition_gate_rejects_degenerate_or_nonordinal_results():
    gates = {
        "armor_penetration_rate_range": [0.05, 0.95],
        "component_penetration_rate_range": [0.02, 0.98],
        "damage_positive_given_hit_range": [0.10, 0.90],
        "saturated_damage_fraction_max": 0.50,
        "ordinal_projection_max_correction_max": 1.0e-12,
        "energy_closure_max_abs_j": 1.0e-8,
        "expected_hit_component_observations_min": 30.0,
    }
    valid = {
        "armor_penetration_rate": 0.5,
        "component_penetration_rate": 0.4,
        "damage_positive_given_hit": 0.6,
        "saturated_damage_fraction": 0.1,
        "ordinal_projection_max_correction": 0.0,
        "energy_closure_max_abs_j": 0.0,
        "expected_hit_components_per_replicate": 4.0,
        "expected_hit_component_observations": 40.0,
        "penetration_affected_replicate_fraction": 0.0,
        "scientific_completion_status": "complete_model_domain",
    }

    assert evaluate_condition_gates(valid, gates)["passed"] is True
    wrong_completion_enum = {**valid, "scientific_completion_status": "complete"}
    assert evaluate_condition_gates(wrong_completion_enum, gates)["passed"] is False
    invalid = {**valid, "armor_penetration_rate": 1.0, "ordinal_projection_max_correction": 0.01}
    result = evaluate_condition_gates(invalid, gates)
    assert result["passed"] is False
    assert set(result["failed_gates"]) == {
        "armor_penetration_rate_range",
        "ordinal_projection_max_correction_max",
    }


def test_multicondition_gate_rejects_any_penetration_domain_censoring():
    gates = {
        "armor_penetration_rate_range": [0.05, 0.95],
        "component_penetration_rate_range": [0.02, 0.98],
        "damage_positive_given_hit_range": [0.10, 0.90],
        "saturated_damage_fraction_max": 0.50,
        "ordinal_projection_max_correction_max": 1.0e-12,
        "energy_closure_max_abs_j": 1.0e-8,
        "expected_hit_component_observations_min": 30.0,
        "out_of_domain_impact_fraction_max": 0.05,
    }
    censored = {
        "armor_penetration_rate": 0.5,
        "component_penetration_rate": 0.4,
        "damage_positive_given_hit": 0.6,
        "saturated_damage_fraction": 0.1,
        "ordinal_projection_max_correction": 0.0,
        "energy_closure_max_abs_j": 0.0,
        "expected_hit_component_observations": 40.0,
        "out_of_domain_impact_fraction": 0.001,
        "penetration_affected_replicate_fraction": 0.03125,
        "scientific_completion_status": "partial_domain_censored",
    }

    result = evaluate_condition_gates(censored, gates)

    assert result["passed"] is False
    assert result["checks"]["penetration_domain_complete"] is False
    assert result["failed_gates"] == ["penetration_domain_complete"]


def test_multicondition_source_identity_detects_tool_and_plan_drift():
    start = {
        "engine_code_hash": "engine-a",
        "evaluation_tool_hash": "tool-a",
        "sweep_tool_hash": "sweep-a",
        "plan_hash": "plan-a",
    }
    end = {**start, "evaluation_tool_hash": "tool-b", "plan_hash": "plan-b"}

    assert source_identity_drift(start, start) == []
    assert source_identity_drift(start, end) == [
        "evaluation_tool_hash",
        "plan_hash",
    ]


def test_selection_does_not_choose_a_diagnostic_leader_when_none_is_admissible():
    ranking = ["diagnostic-leader", "second"]
    results = {
        "diagnostic-leader": {
            "aggregate": {"all_condition_gates_passed": False}
        },
        "second": {"aggregate": {"all_condition_gates_passed": False}},
    }

    assert select_admissible_candidate(ranking, results) is None
    results["second"]["aggregate"]["all_condition_gates_passed"] = True
    assert select_admissible_candidate(ranking, results) == "second"


def test_condition_sets_keep_confirmation_and_holdout_sealed_and_disjoint():
    plan = {
        "holdout_evidence_status": "sealed_unobserved",
        "conditions": [
            {"id": "s", "role": "selection"},
            {"id": "c", "role": "selection"},
            {"id": "h", "role": "holdout"},
        ],
        "selection_condition_ids": ["s"],
        "confirmation_condition_ids": ["c"],
        "holdout_condition_ids": ["h"],
    }

    selection, confirmation, holdout = resolve_condition_sets(plan)
    assert [row["id"] for row in selection] == ["s"]
    assert [row["id"] for row in confirmation] == ["c"]
    assert [row["id"] for row in holdout] == ["h"]

    invalid = deepcopy(plan)
    invalid["confirmation_condition_ids"] = ["h"]
    with pytest.raises(ValueError, match="disjoint"):
        resolve_condition_sets(invalid)

    missing_holdout_status = deepcopy(plan)
    del missing_holdout_status["holdout_evidence_status"]
    with pytest.raises(ValueError, match="holdout_evidence_status"):
        resolve_condition_sets(missing_holdout_status)


def test_monotonic_check_is_unavailable_when_either_candidate_is_domain_censored():
    def candidate(value: float, domain_complete: bool):
        return {
            "aggregate": {"equal_weight_metrics": {"rate": value}},
            "conditions": [
                {
                    "gates": {
                        "checks": {
                            "penetration_domain_complete": domain_complete,
                        }
                    }
                }
            ],
        }

    check = {
        "lower_candidate_id": "low",
        "higher_candidate_id": "high",
        "metric": "rate",
        "expected_relation": "lower_ge_higher",
    }
    censored = {
        "low": candidate(0.7, True),
        "high": candidate(0.5, False),
    }
    result = _monotonic_checks(censored, [check])[0]
    assert result["status"] == "unavailable"
    assert result["passed"] is None
    assert result["reason"] == "domain_censored"

    complete = {
        "low": candidate(0.7, True),
        "high": candidate(0.5, True),
    }
    result = _monotonic_checks(complete, [check])[0]
    assert result["status"] == "available"
    assert result["passed"] is True
