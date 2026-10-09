from __future__ import annotations

from copy import deepcopy

import pytest

from damage_engine.engine import DamageEngine
from damage_engine.errors import OutOfDomainError, UnverifiedModelError
from damage_engine.utils import content_hash


def test_engine_is_deterministic_for_same_seed(target_data, case_data, registry):
    engine = DamageEngine(target_data, registry)
    case = deepcopy(case_data)
    case["analysis"]["replicates"] = 4
    case["analysis"]["adaptive_ci_half_width"] = None
    first = engine.run(case).to_dict()
    second = engine.run(case).to_dict()
    assert first["components"] == second["components"]
    assert first["damage_tree"] == second["damage_tree"]
    assert first["summary"] == second["summary"]


def test_probabilities_are_bounded_and_ordinal(target_data, case_data, registry):
    engine = DamageEngine(target_data, registry)
    case = deepcopy(case_data)
    case["analysis"]["replicates"] = 4
    case["analysis"]["adaptive_ci_half_width"] = None
    result = engine.run(case).to_dict()
    states = result["components"]["flight-control"]["damage_states"]
    for mechanism in ("fragment", "blast", "combined"):
        degraded = states[mechanism]["degraded"]["mean"]
        failed = states[mechanism]["failed"]["mean"]
        assert 0.0 <= failed <= degraded <= 1.0


def test_l3_rejects_unverified_registry(target_data, case_data, registry):
    engine = DamageEngine(target_data, registry)
    case = deepcopy(case_data)
    case["analysis"]["fidelity_level"] = "L3"
    with pytest.raises(UnverifiedModelError):
        engine.run(case)


def test_non_strict_mode_records_refusal(target_data, case_data, registry):
    engine = DamageEngine(target_data, registry)
    case = deepcopy(case_data)
    case["analysis"]["fidelity_level"] = "L3"
    result = engine.run(case, fail_closed=False).to_dict()
    assert result["status"] == "refused"
    assert result["refusal"]["code"] == "REF_MODEL_UNVERIFIED"


def test_epistemic_realization_is_case_stable(target_data, case_data, registry):
    target = deepcopy(target_data)
    parameter = target["components"][0]["response"]["equivalent_thickness"]
    parameter["uncertainty"] = {
        "kind": "uniform",
        "parameters": {"minimum": 1.0, "maximum": 5.0},
        "epistemic": True,
    }
    engine = DamageEngine(target, registry)
    first_case = deepcopy(case_data)
    second_case = deepcopy(case_data)
    target_hash = content_hash(target)
    first_case["target_ref"]["content_hash"] = target_hash
    second_case["target_ref"]["content_hash"] = target_hash
    first_case["analysis"]["replicates"] = 1
    second_case["analysis"]["replicates"] = 1
    first_case["analysis"]["adaptive_ci_half_width"] = None
    second_case["analysis"]["adaptive_ci_half_width"] = None
    second_case["analysis"]["aleatory_seed"] += 1
    first = engine.run(first_case).to_dict()["parameter_realization"]
    second = engine.run(second_case).to_dict()["parameter_realization"]
    assert first == second


def test_blast_table_refuses_extrapolation(target_data, case_data, registry):
    engine = DamageEngine(target_data, registry)
    case = deepcopy(case_data)
    case["encounter"]["detonation_position_m"] = [-20.0, 0.0, 0.0]
    case["analysis"]["replicates"] = 1
    case["analysis"]["adaptive_ci_half_width"] = None
    with pytest.raises(OutOfDomainError):
        engine.run(case)


def test_result_declares_dependency_and_convergence_scope(
    target_data,
    case_data,
    registry,
):
    engine = DamageEngine(target_data, registry)
    case = deepcopy(case_data)
    case["target_ref"]["content_hash"] = content_hash(target_data)
    case["analysis"]["replicates"] = 32
    case["analysis"]["adaptive_ci_half_width"] = 0.1
    result = engine.run(case).to_dict()

    assert result["combined_dependency_assumption"] == (
        "conditional_independence_given_replicate"
    )
    assert result["damage_tree"]["dependency_assumption"] == "explicit_per_gate"
    assert result["damage_tree"]["dependency_models"] == engine.damage_tree.dependency_models
    assert result["convergence"]["physical_metrics_in_scope"] is False
    assert result["convergence"]["minimum_required_replicates"] == 32
    assert isinstance(result["convergence"]["target_met"], bool)
    assert result["convergence"]["reached_replicate_limit"] is True
    assert "armor-front" in result["armor"]


def test_engine_outputs_requested_evaluation_standard_views(
    target_data, case_data, registry
):
    target = deepcopy(target_data)
    target["damage_tree"]["standards"] = [
        {"id": "all-effects", "outputs": ["mission_loss", "mission_degraded"]},
        {"id": "loss-only", "outputs": ["mission_loss"]},
    ]
    engine = DamageEngine(target, registry)
    case = deepcopy(case_data)
    case["target_ref"]["content_hash"] = engine.target_hash
    case["analysis"]["evaluation_standard_ids"] = ["loss-only", "all-effects"]
    case["analysis"]["replicates"] = 1
    case["analysis"]["adaptive_ci_half_width"] = None

    result = engine.run(case).to_dict()
    standards = result["damage_tree"]["evaluation_standards"]
    assert list(standards) == ["loss-only", "all-effects"]
    assert list(standards["loss-only"]) == ["mission_loss"]
    assert list(standards["all-effects"]) == ["mission_loss", "mission_degraded"]
