from __future__ import annotations

import json
from pathlib import Path

import pytest

from damage_engine.registry import PhysicsRegistry
from tools.sweep_penetration_parameters import (
    apply_candidate,
    is_formally_admissible,
    load_candidate_plan,
    score_metrics,
    summarize_result,
)


ROOT = Path(__file__).resolve().parents[1]


def test_parameter_sweep_score_rejects_saturated_or_nonordinal_damage():
    objective = {
        "armor_penetration_rate_target": 0.55,
        "component_penetration_rate_target": 0.45,
        "damage_positive_given_hit_target": 0.65,
        "mean_degraded_given_hit_target": 0.35,
        "mean_failed_given_hit_target": 0.15,
    }
    balanced = {
        "armor_penetration_rate": 0.55,
        "component_penetration_rate": 0.45,
        "damage_positive_given_hit": 0.65,
        "mean_degraded_given_hit": 0.35,
        "mean_failed_given_hit": 0.15,
        "saturated_damage_fraction": 0.0,
        "ordinal_violation_count": 0,
        "energy_closure_max_abs_j": 0.0,
    }
    saturated = {
        **balanced,
        "mean_degraded_given_hit": 0.98,
        "mean_failed_given_hit": 0.97,
        "saturated_damage_fraction": 0.9,
    }
    nonordinal = {**balanced, "ordinal_violation_count": 1}

    assert score_metrics(balanced, objective) < score_metrics(saturated, objective)
    assert score_metrics(balanced, objective) < score_metrics(nonordinal, objective)


def test_selected_candidate_is_promoted_into_the_default_effective_registry():
    physics_path = ROOT / "configs" / "physics" / "loitering_impact.json"
    candidate_path = ROOT / "examples" / "penetration_sweep.selected.provisional.json"
    base = json.loads(physics_path.read_text(encoding="utf-8"))
    _, candidates = load_candidate_plan(candidate_path)

    assert len(candidates) == 1
    materialized = apply_candidate(base, candidates[0]["parameters"])
    assert PhysicsRegistry(materialized).hash == PhysicsRegistry(base).hash


def test_tuning_summary_uses_fixed_component_replicate_estimands():
    def stats(mean):
        return {"mean": mean}

    result = {
        "replicate_count": 10,
        "scientific_completion_status": "partial_domain_censored",
        "penetration_domain_coverage": {
            "affected_replicate_fraction": 0.2,
        },
        "summary": {
            "total_armor_hits": stats(10.0),
            "total_armor_penetrations": stats(5.0),
            "total_component_hits": stats(4.0),
            "total_component_penetrations": stats(2.0),
            "total_existing_hole_reuses": stats(0.0),
        },
        "components": {
            "common": {
                "fragment_metrics": {
                    "hits": stats(2.0),
                    "hit_indicator": stats(1.0),
                    "damage_positive_indicator": stats(0.5),
                    "saturated_damage_indicator": stats(0.1),
                    "hit_conditioned_degraded_probability": stats(0.2),
                    "hit_conditioned_failed_probability": stats(0.1),
                    "ordinal_projection_max_correction": stats(0.02),
                },
                "damage_states": {
                    "fragment": {
                        "degraded": stats(0.2),
                        "failed": stats(0.1),
                    }
                },
            },
            "rare": {
                "fragment_metrics": {
                    "hits": stats(0.01),
                    "hit_indicator": stats(0.01),
                    "damage_positive_indicator": stats(0.01),
                    "saturated_damage_indicator": stats(0.0),
                    "hit_conditioned_degraded_probability": stats(0.005),
                    "hit_conditioned_failed_probability": stats(0.001),
                    "ordinal_projection_max_correction": stats(0.0),
                },
                "damage_states": {
                    "fragment": {
                        "degraded": stats(0.005),
                        "failed": stats(0.001),
                    }
                },
            },
        },
    }

    summary = summarize_result(result)

    assert summary["damage_positive_given_hit"] == pytest.approx(0.51 / 1.01)
    assert summary["mean_degraded_given_hit"] == pytest.approx(0.205 / 1.01)
    assert summary["mean_failed_given_hit"] == pytest.approx(0.101 / 1.01)
    assert summary["saturated_damage_fraction"] == pytest.approx(0.1 / 1.01)
    assert summary["ordinal_projection_max_correction"] == pytest.approx(0.02)
    assert summary["expected_hit_components_per_replicate"] == pytest.approx(1.01)
    assert summary["expected_hit_component_observations"] == pytest.approx(10.1)
    assert summary["penetration_affected_replicate_fraction"] == pytest.approx(0.2)
    assert summary["scientific_completion_status"] == "partial_domain_censored"


def test_tuning_summary_requires_explicit_penetration_domain_evidence():
    missing_coverage = {
        "replicate_count": 1,
        "scientific_completion_status": "complete_model_domain",
        "summary": {},
        "components": {},
    }

    with pytest.raises(KeyError, match="penetration_domain_coverage"):
        summarize_result(missing_coverage)

    missing_status = {
        "replicate_count": 1,
        "penetration_domain_coverage": {"affected_replicate_fraction": 0.0},
        "summary": {},
        "components": {},
    }
    with pytest.raises(KeyError, match="scientific_completion_status"):
        summarize_result(missing_status)


def test_single_point_sweep_never_promotes_domain_incomplete_diagnostic():
    diagnostic = {
        "scientific_completion_status": (
            "observed_domain_complete_not_statistically_demonstrated"
        ),
        "penetration_affected_replicate_fraction": 0.0,
        "energy_closure_max_abs_j": 0.0,
    }
    complete = {
        **diagnostic,
        "scientific_completion_status": "complete_model_domain",
    }

    assert is_formally_admissible(diagnostic) is False
    assert is_formally_admissible(complete) is True


def test_single_point_formal_admission_enforces_declared_distribution_gates():
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
    admissible = {
        "scientific_completion_status": "complete_model_domain",
        "penetration_affected_replicate_fraction": 0.0,
        "armor_penetration_rate": 0.55,
        "component_penetration_rate": 0.45,
        "damage_positive_given_hit": 0.65,
        "saturated_damage_fraction": 0.20,
        "ordinal_projection_max_correction": 0.0,
        "energy_closure_max_abs_j": 0.0,
        "expected_hit_component_observations": 100.0,
        "out_of_domain_impact_fraction": 0.0,
    }

    assert is_formally_admissible(admissible, gates) is True
    assert is_formally_admissible(
        {**admissible, "damage_positive_given_hit": 0.95}, gates
    ) is False
    assert is_formally_admissible(
        {**admissible, "saturated_damage_fraction": 0.75}, gates
    ) is False
