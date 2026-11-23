from __future__ import annotations

from copy import deepcopy

import pytest

from damage_engine.engine import DamageEngine
from damage_engine.errors import OutOfDomainError, UnverifiedModelError
from damage_engine.registry import PhysicsRegistry
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
    assert first["replicate_statistics_storage"]["mode"] == (
        "online_sufficient_moments"
    )
    assert first["replicate_statistics_storage"]["raw_trace_retained"] is False
    assert first["replicate_statistics_storage"] == second[
        "replicate_statistics_storage"
    ]


def test_raw_replicates_are_retained_only_when_trace_is_requested(
    target_data,
    case_data,
    registry,
):
    engine = DamageEngine(target_data, registry)
    case = deepcopy(case_data)
    case["analysis"]["replicates"] = 2
    case["analysis"]["adaptive_ci_half_width"] = None
    case["analysis"]["capture_replicate_trace"] = True

    result = engine.run(case).to_dict()

    assert result["replicate_statistics_storage"]["raw_trace_retained"] is True
    assert result["replicate_trace"]["replicate_count"] == 2


def test_component_state_trace_is_lightweight_and_independent_of_raw_trace(
    target_data,
    case_data,
    registry,
):
    engine = DamageEngine(target_data, registry)
    case = deepcopy(case_data)
    case["analysis"]["replicates"] = 2
    case["analysis"]["adaptive_ci_half_width"] = None
    case["analysis"]["capture_component_state_trace"] = True

    result = engine.run(case).to_dict()

    storage = result["replicate_statistics_storage"]
    assert storage["raw_trace_retained"] is False
    assert storage["component_state_trace_retained"] is True
    assert storage["mode"] == (
        "online_sufficient_moments_plus_component_state_trace"
    )
    assert "replicate_trace" not in result
    trace = result["component_state_trace"]
    assert trace["schema_version"] == "1.0.0"
    assert trace["semantics"] == (
        "joint_per_replicate_component_mechanism_states_before_tree_aggregation"
    )
    assert trace["replicate_count"] == 2
    assert trace["source_identity"]["case_id"] == case["case_id"]
    assert trace["source_identity"]["target_id"] == target_data["target_id"]
    assert set(trace["records"][0]) == {
        "replicate_index",
        "parameter_realization_hash",
        "component_mechanism_states",
        "system_outputs",
        "domain_censoring",
    }
    assert "fragment_metrics" not in trace["records"][0]


def test_fragment_audit_exposes_coherent_stochastic_perforation_outcomes(
    target_data,
    case_data,
    registry,
):
    engine = DamageEngine(target_data, registry)
    case = deepcopy(case_data)
    case["analysis"]["mechanisms"] = ["fragment"]
    case["analysis"]["replicates"] = 1
    case["analysis"]["adaptive_ci_half_width"] = None
    case["analysis"]["capture_fragment_audit"] = True

    result = engine.run(case).to_dict()
    events = [
        event
        for event in result["fragment_audit"][0]["events"]
        if event["record_type"] == "intersection"
    ]

    assert events
    assert all("sampled_ballistic_limit_mps" in event for event in events)
    assert all("perforation_class" in event for event in events)
    assert all(
        event["residual_speed_mps"] > 0.0
        for event in events
        if event["penetrated"]
    )
    assert all(
        event["incident_energy_j"]
        == pytest.approx(event["residual_energy_j"] + event["absorbed_energy_j"])
        for event in events
    )


def test_nonperforating_component_impact_can_cause_absorbed_energy_damage(
    target_data,
    case_data,
    registry,
):
    registry_data = deepcopy(registry.data)
    armor_model = registry_data["fragment_response_models"]["demo_armor_response"]
    armor_model["ballistic_limit"]["coefficient"] = 1.0
    component_model = registry_data["fragment_response_models"][
        "demo_component_response"
    ]
    component_model["ballistic_limit"]["coefficient"] = 1.0e9
    component_model["damage_energy_model"] = {
        "kind": "absorbed_plus_residual",
        "absorbed_weight": 1.0,
        "residual_weight": 0.1,
        "nonperforation_absorbed_fraction": 0.5,
    }
    for state_model in component_model["damage_states"].values():
        state_model["damage_energy_50_j"] = 1.0

    source_model = registry_data["fragment_source_models"]["demo_cone_source"]
    source_model["direction"]["half_angle_deg"] = 1.0

    engine = DamageEngine(target_data, PhysicsRegistry(registry_data))
    case = deepcopy(case_data)
    case["analysis"]["mechanisms"] = ["fragment"]
    case["analysis"]["replicates"] = 1
    case["analysis"]["adaptive_ci_half_width"] = None
    case["warhead"]["fragment_source"]["total_count"] = 128

    result = engine.run(case).to_dict()["components"]["flight-control"]

    assert result["fragment_metrics"]["hits"]["mean"] > 0.0
    assert result["fragment_metrics"]["penetrations"]["mean"] == 0.0
    assert result["fragment_metrics"]["deposited_energy_j"]["mean"] > 0.0
    assert result["damage_states"]["fragment"]["degraded"]["mean"] > 0.0
    conditional_energy = result["fragment_metrics"][
        "hit_conditioned_log1p_deposited_energy_j"
    ]
    conditional_hits = result["fragment_metrics"][
        "hit_conditioned_log1p_hit_count"
    ]
    assert conditional_energy["sample_count"] == 1
    assert conditional_energy["conditioning"] == "replicate_has_component_hit"
    assert conditional_energy["mean"] > 0.0
    assert conditional_hits["mean"] > 0.0


def test_audit_separates_geometry_envelope_from_material_traversal(
    target_data,
    case_data,
    registry,
):
    target = deepcopy(target_data)
    target["armor"][0]["geometry"]["extrusion_distance_m"] = 0.5
    target["armor"][0]["geometry_semantics"] = (
        "thin_shell_collision_envelope"
    )
    target["components"][0]["geometry_semantics"] = (
        "logical_component_envelope"
    )

    registry_data = deepcopy(registry.data)
    registry_data["fragment_response_models"]["demo_armor_response"][
        "ballistic_limit"
    ]["coefficient"] = 1.0
    registry_data["fragment_source_models"]["demo_cone_source"]["direction"][
        "half_angle_deg"
    ] = 1.0

    case = deepcopy(case_data)
    case["target_ref"]["content_hash"] = content_hash(target)
    case["analysis"]["mechanisms"] = ["fragment"]
    case["analysis"]["replicates"] = 1
    case["analysis"]["adaptive_ci_half_width"] = None
    case["analysis"]["capture_fragment_audit"] = True
    case["warhead"]["fragment_source"]["total_count"] = 128

    result = DamageEngine(target, PhysicsRegistry(registry_data)).run(case).to_dict()
    events = [
        event
        for event in result["fragment_audit"][0]["events"]
        if event["record_type"] == "intersection"
    ]
    armor_event = next(event for event in events if event["role"] == "armor")
    component_event = next(event for event in events if event["role"] == "component")

    assert armor_event["geometry_path_length_m"] > 0.49
    assert armor_event["material_path_length_m"] == pytest.approx(
        armor_event["equivalent_thickness_m"]
        / armor_event["incidence_cosine"],
    )
    assert armor_event["material_path_length_m"] < 0.03
    assert component_event["free_flight_distance_m"] > 0.25
    assert armor_event["geometry_semantics"] == "thin_shell_collision_envelope"
    assert component_event["geometry_semantics"] == "logical_component_envelope"


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


def test_engine_distinguishes_perforation_events_from_opening_primitives(
    target_data, case_data, registry
):
    case = deepcopy(case_data)
    case["analysis"]["replicates"] = 1
    case["analysis"]["adaptive_ci_half_width"] = None

    result = DamageEngine(target_data, registry).run(case).to_dict()

    opening_total = sum(
        record["fragment_panel_damage"]["opening_primitive_count"]["mean"]
        for record in result["armor"].values()
    )
    event_total = sum(
        record["fragment_panel_damage"]["perforation_event_count"]["mean"]
        for record in result["armor"].values()
    )
    connected_total = sum(
        record["fragment_panel_damage"][
            "opening_connected_component_count"
        ]["mean"]
        for record in result["armor"].values()
    )
    assert result["summary"]["total_unique_armor_holes"]["mean"] == pytest.approx(
        connected_total
    )
    assert result["summary"]["total_perforation_events"]["mean"] == pytest.approx(
        event_total
    )
    assert event_total >= opening_total
    assert opening_total >= connected_total


def test_l3_rejects_unverified_registry(target_data, case_data, registry):
    engine = DamageEngine(target_data, registry)
    case = deepcopy(case_data)
    case["analysis"]["fidelity_level"] = "L3"
    with pytest.raises(UnverifiedModelError):
        engine.run(case)


def test_engine_marks_unresolved_penetration_domain_coverage(
    target_data, case_data, registry
):
    registry_data = deepcopy(registry.data)
    for model in registry_data["fragment_response_models"].values():
        model["ballistic_limit"]["maximum_obliquity_deg"] = 0.1
    case = deepcopy(case_data)
    case["analysis"]["replicates"] = 1
    case["analysis"]["adaptive_ci_half_width"] = None

    result = DamageEngine(target_data, PhysicsRegistry(registry_data)).run(case).to_dict()

    coverage = result["penetration_domain_coverage"]
    assert coverage["status"] == "partial_unresolved"
    assert coverage["mean_unresolved_impact_count"] > 0.0
    assert 0.0 < coverage["unresolved_impact_fraction"] <= 1.0
    assert coverage["affected_replicate_fraction"] == 1.0
    assert coverage["probability_bound_method"] == (
        "affected-replicate worst-case bounded-outcome interval"
    )
    assert result["status"] == "ok"
    assert result["scientific_completion_status"] == "partial_domain_censored"
    assert result["convergence"]["domain_censoring_blocks_convergence"] is True
    assert result["convergence"]["penetration_domain_complete"] is False
    fragment_state = result["components"]["flight-control"]["damage_states"]["fragment"]["degraded"]
    bounds = coverage["probability_bounds"]["components"]["flight-control"][
        "fragment"
    ]["degraded"]
    assert bounds["lower"] == fragment_state["mean"]
    assert bounds["upper"] == 1.0
    tree_bounds = coverage["probability_bounds"]["damage_tree_outputs"][
        "mission_degraded"
    ]
    assert tree_bounds["lower"] == 0.0
    assert tree_bounds["upper"] == 1.0
    assert any("unresolved events" in warning for warning in result["warnings"])


def test_component_ood_contact_keeps_hit_indicator_point_identified(
    target_data, case_data, registry
):
    registry_data = deepcopy(registry.data)
    registry_data["fragment_response_models"]["demo_armor_response"][
        "ballistic_limit"
    ]["maximum_obliquity_deg"] = 89.0
    registry_data["fragment_response_models"]["demo_component_response"][
        "ballistic_limit"
    ]["maximum_obliquity_deg"] = 0.1
    case = deepcopy(case_data)
    case["analysis"]["replicates"] = 1
    case["analysis"]["aleatory_seed"] = 1
    case["analysis"]["adaptive_ci_half_width"] = None

    result = DamageEngine(target_data, PhysicsRegistry(registry_data)).run(case).to_dict()

    assert result["summary"]["total_out_of_domain_armor_impacts"]["mean"] == 0.0
    assert result["summary"]["total_out_of_domain_component_impacts"]["mean"] > 0.0
    component = result["components"]["flight-control"]
    assert component["fragment_metrics"]["hit_indicator"]["mean"] == 1.0
    bounds = result["penetration_domain_coverage"]["probability_bounds"][
        "components"
    ]["flight-control"]
    hit_bound = bounds["fragment_metrics"]["hit_indicator"]
    assert hit_bound["lower"] == hit_bound["upper"] == 1.0
    assert hit_bound["width"] == 0.0
    assert hit_bound["affected_replicate_fraction"] == 0.0
    degraded_bound = bounds["fragment"]["degraded"]
    assert degraded_bound["lower"] == 0.0
    assert degraded_bound["upper"] == 1.0
    assert degraded_bound["affected_replicate_fraction"] == 1.0
    assert degraded_bound["method"].startswith("component_specific")


def test_sequential_engine_applies_blast_failure_only_to_later_fragments(
    target_data, case_data, registry
):
    registry_data = deepcopy(registry.data)
    registry_data["blast_response_models"]["demo_structural_blast"][
        "damage_states"
    ] = {
        "failed": {
            "peak_50_pa": 1.0e6,
            "impulse_50_pa_s": 1.0e300,
            "peak_exponent": 1.0,
            "impulse_exponent": 1.0,
            "log_slope": 8.0,
        }
    }
    target = deepcopy(target_data)
    target["armor"][0]["response"]["blast_response"][
        "peak_overpressure_threshold"
    ] = {
        "value": 1.0,
        "unit": "kPa",
        "source_id": "TEST",
        "calibration_id": None,
        "validation_status": "unverified",
        "valid_range": [1.0, 1.0e9],
        "uncertainty": {"kind": "none", "parameters": {}, "epistemic": True},
    }
    case = deepcopy(case_data)
    case["target_ref"]["content_hash"] = content_hash(target)
    case["warhead"]["fragment_source"]["gurney_constant_mps"] = 150.0
    case["analysis"]["replicates"] = 1
    case["analysis"]["adaptive_ci_half_width"] = None
    case["analysis"]["capture_fragment_audit"] = True

    result = DamageEngine(target, PhysicsRegistry(registry_data)).run(case).to_dict()

    armor_events = [
        event
        for replicate in result["fragment_audit"]
        for event in replicate["events"]
        if event.get("record_type") == "intersection"
        and event.get("role") == "armor"
    ]
    assert armor_events
    assert all(event["blast_damage_active"] for event in armor_events)
    assert all(event["blast_survival_factor"] < 1.0 for event in armor_events)
    assert all(
        event["arrival_time_s"] >= event["blast_damage_effective_time_s"]
        for event in armor_events
    )
    assert result["coupling_semantics"] == (
        "time_ordered_two_way_armor_state_with_post_pulse_blast_failure"
    )

    early_case = deepcopy(case)
    early_case["case_id"] = "fragment-before-blast-response"
    early_case["warhead"]["fragment_source"]["gurney_constant_mps"] = 2200.0
    early_result = DamageEngine(target, PhysicsRegistry(registry_data)).run(
        early_case
    ).to_dict()
    early_armor_events = [
        event
        for replicate in early_result["fragment_audit"]
        for event in replicate["events"]
        if event.get("record_type") == "intersection"
        and event.get("role") == "armor"
    ]
    assert early_armor_events
    assert all(not event["blast_damage_active"] for event in early_armor_events)
    assert all(event["blast_survival_factor"] == 1.0 for event in early_armor_events)
    assert all(
        event["arrival_time_s"] < event["blast_damage_effective_time_s"]
        for event in early_armor_events
    )


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
        "conditional_independence_within_minimum_shared_support_sensitivity"
    )
    assert result["combined_dependency_identification"] == "not_identified"
    assert result["combined_dependency_scenarios"] == [
        "frechet_lower",
        "conditional_independence",
        "frechet_upper",
    ]
    assert result["damage_tree"]["dependency_assumption"] == "explicit_per_gate"
    assert result["damage_tree"]["dependency_models"] == engine.damage_tree.dependency_models
    sensitivity = result["damage_tree"]["mechanism_dependency_sensitivity"]
    assert set(sensitivity["scenarios"]) == {
        "frechet_lower",
        "conditional_independence",
        "frechet_upper",
    }
    assert sensitivity["identification_status"] == "not_identified"
    assert sensitivity["point_estimate_assumption"] == (
        "conditional_independence_within_minimum_shared_support_sensitivity"
    )
    assert result["combined_dependency_scenario_semantics"][
        "conditional_independence"
    ] == "conditional_independence_within_minimum_shared_support_sensitivity"
    assert result["blast_load_semantics"]["transmitted_positive_duration_s"] == (
        "equivalent_duration_preserving_peak_impulse_representability"
    )
    gate_sensitivity = result["damage_tree"]["gate_dependency_sensitivity"]
    assert set(gate_sensitivity["scenarios"]) == {
        "declared",
        "independent",
        "fully_dependent",
    }
    assert gate_sensitivity["identification_status"] == "not_identified"
    assert gate_sensitivity["point_estimate_assumption"] == "declared"
    assert gate_sensitivity["scenarios"]["independent"]["status"] == "unavailable"
    assert gate_sensitivity["scenarios"]["independent"]["outputs"] == {
        "mission_degraded": None,
        "mission_loss": None,
    }
    assert "independent" not in gate_sensitivity["available_scenarios"]
    assert result["convergence"]["physical_metrics_in_scope"] is False
    assert result["convergence"]["minimum_required_replicates"] == 32
    assert result["convergence"]["interval_method"] == (
        "finite_horizon_simultaneous_hoeffding"
    )
    assert result["convergence"]["coverage_status"] == (
        "familywise_optional_stopping_valid_for_bounded_probability_means"
    )
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
