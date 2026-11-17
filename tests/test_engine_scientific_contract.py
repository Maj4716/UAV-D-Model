from __future__ import annotations

from copy import deepcopy

import pytest

from damage_engine.blast import simulate_blast
from damage_engine.engine import DamageEngine
from damage_engine.geometry import compile_target
from damage_engine.parameters import nominal_target_parameters
from damage_engine.panel_damage import PanelDamageState
from damage_engine.registry import PhysicsRegistry
from damage_engine.utils import content_hash


def test_engine_owns_an_immutable_target_snapshot(target_data, case_data, registry):
    original_target_id = target_data["target_id"]
    engine = DamageEngine(target_data, registry)
    frozen_hash = engine.target_hash

    target_data["target_id"] = "mutated-caller-object"
    exported = engine.target_data
    exported["target_id"] = "mutated-exported-object"

    case = deepcopy(case_data)
    case["analysis"]["mechanisms"] = ["blast"]
    case["analysis"]["replicates"] = 1
    case["analysis"]["adaptive_ci_half_width"] = None
    result = engine.run(case).to_dict()

    assert engine.target_hash == frozen_hash
    assert engine.target_data["target_id"] == original_target_id
    assert result["target_id"] == original_target_id
    assert result["provenance"]["target_hash"] == frozen_hash


def test_prepare_case_returns_an_owned_copy_for_plain_cases(
    target_data, case_data, registry
):
    engine = DamageEngine(target_data, registry)
    source = deepcopy(case_data)
    frozen = deepcopy(source)

    prepared = engine.prepare_case(source)
    prepared["environment"]["temperature_k"] += 1.0

    assert source == frozen
    assert prepared is not source
    assert prepared["environment"] is not source["environment"]


def test_compiled_damage_tree_is_an_immutable_execution_plan(
    target_data, registry
):
    engine = DamageEngine(target_data, registry)
    node_id = engine.damage_tree.order[0]

    with pytest.raises(TypeError):
        engine.damage_tree.nodes[node_id]["id"] = "mutated"

    assert engine.damage_tree.nodes[node_id]["id"] == node_id


def test_blast_failure_quantile_creates_a_shared_discrete_panel_state(
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
    target = compile_target(target_data)
    response_registry = PhysicsRegistry(registry_data)
    parameters = nominal_target_parameters(target_data)

    failed = simulate_blast(
        case_data,
        target,
        response_registry,
        parameter_realization=parameters,
        armor_integrity={"armor-front": 1.0},
        armor_failure_quantiles={"armor-front": 0.0},
    )
    survived = simulate_blast(
        case_data,
        target,
        response_registry,
        parameter_realization=parameters,
        armor_integrity={"armor-front": 1.0},
        armor_failure_quantiles={"armor-front": 1.0},
    )

    probability = failed.armor_metrics["armor-front"][
        "blast_failure_probability"
    ]
    assert 0.0 < probability < 1.0
    assert failed.armor_metrics["armor-front"]["blast_failure_realized"] == 1.0
    assert survived.armor_metrics["armor-front"]["blast_failure_realized"] == 0.0
    assert failed.armor_integrity["armor-front"] == 0.0
    assert survived.armor_integrity["armor-front"] == 1.0


def test_realized_panel_failure_uses_a_bounded_mean_interval(
    target_data, case_data, registry
):
    case = deepcopy(case_data)
    case["analysis"]["mechanisms"] = ["blast"]
    case["analysis"]["replicates"] = 8
    case["analysis"]["adaptive_ci_half_width"] = None

    result = DamageEngine(target_data, registry).run(case).to_dict()
    statistic = result["armor"]["armor-front"]["blast_metrics"][
        "blast_failure_realized"
    ]

    assert statistic["interval_method"] == "fixed_sample_hoeffding_bounded_mean"
    assert 0.0 <= statistic["ci95_lower"] <= statistic["ci95_upper"] <= 1.0


def test_small_perforation_produces_partial_not_full_blast_transmission(
    target_data, case_data, registry
):
    target = compile_target(target_data)
    parameters = nominal_target_parameters(target_data)
    closed = simulate_blast(
        case_data,
        target,
        registry,
        parameter_realization=parameters,
    )
    small_state = PanelDamageState({"armor-front": 4.0})
    small_state.record_perforation(
        "armor-front",
        [-0.5, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        hole_area_m2=0.01,
        formed_time_s=0.0,
    )
    open_state = PanelDamageState({"armor-front": 4.0})
    open_state.record_perforation(
        "armor-front",
        [-0.5, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        hole_area_m2=4.0,
        formed_time_s=0.0,
    )
    small = simulate_blast(
        case_data,
        target,
        registry,
        parameter_realization=parameters,
        panel_damage_state=small_state,
    )
    opened = simulate_blast(
        case_data,
        target,
        registry,
        parameter_realization=parameters,
        panel_damage_state=open_state,
    )
    key = "effective_peak_pressure_pa"
    closed_peak = closed.component_metrics["flight-control"][key]
    small_peak = small.component_metrics["flight-control"][key]
    open_peak = opened.component_metrics["flight-control"][key]

    assert closed_peak < small_peak < open_peak


def test_remote_perforation_does_not_raise_load_on_an_unaligned_component_ray(
    target_data, case_data, registry
):
    target = compile_target(target_data)
    parameters = nominal_target_parameters(target_data)
    closed = simulate_blast(
        case_data,
        target,
        registry,
        parameter_realization=parameters,
    )
    remote_state = PanelDamageState({"armor-front": 4.0})
    remote_state.record_perforation(
        "armor-front",
        [-0.5, 0.4, 0.0],
        [-1.0, 0.0, 0.0],
        hole_area_m2=0.01,
        formed_time_s=0.0,
    )
    remote = simulate_blast(
        case_data,
        target,
        registry,
        parameter_realization=parameters,
        panel_damage_state=remote_state,
    )

    for key in ("effective_peak_pressure_pa", "effective_impulse_pa_s"):
        assert remote.component_metrics["flight-control"][key] == pytest.approx(
            closed.component_metrics["flight-control"][key]
        )


def test_zero_observed_domain_censoring_is_not_claimed_as_proven_complete(
    target_data, case_data, registry
):
    case = deepcopy(case_data)
    case["analysis"]["mechanisms"] = ["blast"]
    case["analysis"]["replicates"] = 16
    case["analysis"]["adaptive_ci_half_width"] = None

    result = DamageEngine(target_data, registry).run(case).to_dict()
    coverage = result["penetration_domain_coverage"]

    assert result["scientific_completion_status"] == "not_applicable_no_fragment"
    assert coverage["status"] == "not_applicable"
    assert coverage["zero_event_upper_confidence_bound"] is None
    assert result["convergence"]["penetration_domain_complete"] is None


def test_result_exposes_static_model_coverage_and_unresolved_pi_surface(
    target_data, case_data, registry
):
    target = deepcopy(target_data)
    extra = deepcopy(target["components"][0])
    extra["id"] = "unreferenced-component"
    extra["name"] = "Unreferenced component"
    extra["geometry"]["center_m"] = [2.0, 0.0, 0.0]
    target["components"].append(extra)
    case = deepcopy(case_data)
    case["analysis"]["mechanisms"] = ["blast"]
    case["analysis"]["replicates"] = 1
    case["analysis"]["adaptive_ci_half_width"] = None
    case["target_ref"]["content_hash"] = content_hash(target)

    result = DamageEngine(target, registry).run(case).to_dict()
    coverage = result["model_coverage"]

    assert "unreferenced-component" in coverage[
        "damage_tree_unreferenced_component_ids"
    ]
    assert "unreferenced-component" in coverage[
        "component_centers_not_inside_armor_material_ids"
    ]
    assert "component_centers_outside_all_armor_envelopes" not in coverage[
        "static_coverage_blockers"
    ]
    assert "component_aabb_overlap_candidates_present" not in coverage[
        "static_coverage_blockers"
    ]
    assert coverage["external_physical_validation_receipt_bound"] is False
    assert coverage["research_admission_ready"] is False
    assert "external_physical_validation_evidence_not_bound" in coverage[
        "research_admission_blockers"
    ]
    component = result["components"]["flight-control"]
    assert component["pressure_impulse_surface_id"] == "DEMO-PI-SURFACE"
    assert component["pressure_impulse_surface_status"] == (
        "declared_but_not_resolved_by_registry"
    )


def test_model_coverage_accepts_registered_blast_threshold_when_override_is_absent(
    target_data, case_data, registry
):
    target = deepcopy(target_data)
    del target["components"][0]["response"]["blast_response"][
        "peak_overpressure_threshold"
    ]
    case = deepcopy(case_data)
    case["analysis"]["mechanisms"] = ["blast"]
    case["analysis"]["replicates"] = 1
    case["analysis"]["adaptive_ci_half_width"] = None
    case["target_ref"]["content_hash"] = content_hash(target)

    coverage = DamageEngine(target, registry).run(case).to_dict()["model_coverage"]

    assert "flight-control" not in coverage[
        "components_missing_blast_peak_threshold_ids"
    ]
