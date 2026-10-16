from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest

from damage_engine.blast import (
    BlastLoad,
    _free_field_load,
    _state_probability,
    _transmit,
    simulate_blast,
)
from damage_engine.errors import InputValidationError, OutOfDomainError
from damage_engine.fragment_audit import summarize_component_reachability
from damage_engine.fragments import Fragment, generate_fragments, simulate_fragments
from damage_engine.geometry import compile_target
from damage_engine.parameters import nominal_target_parameters, sample_target_realization
from damage_engine.registry import PhysicsRegistry


def test_target_frame_fragment_speed_must_remain_in_model_domain(case_data, registry):
    case = deepcopy(case_data)
    # Registered source speed remains valid, but encounter translation pushes
    # every target-frame fragment speed above the calibrated upper bound.
    case["encounter"]["velocity_target_mps"] = [6000.0, 0.0, 0.0]
    with pytest.raises(OutOfDomainError, match="target-frame"):
        generate_fragments(case, registry, np.random.default_rng(7))


def test_penetrating_fragment_continues_through_serial_components(
    monkeypatch,
    target_data,
    case_data,
    registry,
):
    target = deepcopy(target_data)
    downstream = deepcopy(target["components"][0])
    downstream["id"] = "downstream-control"
    downstream["name"] = "Downstream control"
    downstream["geometry"]["center_m"] = [0.6, 0.0, 0.0]
    target["components"].append(downstream)

    registry_data = deepcopy(registry.data)
    for model in registry_data["fragment_response_models"].values():
        model["ballistic_limit"]["coefficient"] = 1.0
        model["ballistic_limit"]["transition_width_ratio"] = 0.001
    permissive_registry = PhysicsRegistry(registry_data)
    compiled = compile_target(target)
    parameters = nominal_target_parameters(target)
    fragment = Fragment(
        fragment_id=0,
        mass_kg=0.005,
        origin_m=np.array([-2.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        initial_speed_mps=1000.0,
        area_m2=1.0e-4,
        drag_coefficient=0.9,
    )
    monkeypatch.setattr(
        "damage_engine.fragments.generate_fragments",
        lambda case, registry, rng: [fragment],
    )

    result = simulate_fragments(
        case_data,
        compiled,
        permissive_registry,
        np.random.default_rng(3),
        parameter_realization=parameters,
        capture_audit=True,
    )

    assert result.total_component_hits == 2
    assert result.total_component_penetrations == 2
    assert result.component_metrics["flight-control"]["hits"] == 1.0
    assert result.component_metrics["downstream-control"]["hits"] == 1.0
    assert (
        result.component_metrics["downstream-control"]["mean_arrival_speed_mps"]
        < result.component_metrics["flight-control"]["mean_arrival_speed_mps"]
    )
    component_events = [event for event in result.audit if event["role"] == "component"]
    assert [event["object_id"] for event in component_events] == [
        "flight-control",
        "downstream-control",
    ]
    trace_records = [
        event for event in result.audit if event["record_type"] == "fragment_trace"
    ]
    assert len(trace_records) == 1
    trace = trace_records[0]
    assert trace["origin_target_m"] == [-2.0, 0.0, 0.0]
    assert trace["direction_target_xyz"] == [1.0, 0.0, 0.0]
    assert [item["object_id"] for item in trace["geometry_candidate_path"]] == [
        "armor-front",
        "flight-control",
        "downstream-control",
    ]
    assert [item["object_id"] for item in trace["processed_path"]] == [
        "armor-front",
        "flight-control",
        "downstream-control",
    ]
    assert trace["blocking_object_id"] is None
    assert trace["blocking_reason"] is None
    for event in component_events:
        assert event["entry_distance_m"] < event["exit_distance_m"]
        assert len(event["entry_point_target_m"]) == 3
        assert len(event["exit_point_target_m"]) == 3
        assert event["response_model"] == {
            "id": "demo_component_response",
            "source_id": "DEMO-NOT-VALIDATED",
        }
        assert event["equivalent_thickness_m"] > 0.0
        assert event["v50_mps"] > 0.0
        assert event["penetration_margin"] > 0.0
        assert set(event["damage_state_response_probabilities"]) == {
            "degraded",
            "failed",
        }
    json.dumps(result.audit, allow_nan=False)

    # Audit capture is observational: with an identical RNG state all physical
    # outputs must remain bit-for-bit equal when capture is disabled.
    without_audit = simulate_fragments(
        case_data,
        compiled,
        permissive_registry,
        np.random.default_rng(3),
        parameter_realization=parameters,
        capture_audit=False,
    )
    assert without_audit.audit == []
    assert without_audit.component_state_probabilities == result.component_state_probabilities
    assert without_audit.component_metrics == result.component_metrics
    assert without_audit.armor_integrity == result.armor_integrity
    assert without_audit.total_fragments == result.total_fragments
    assert without_audit.total_armor_hits == result.total_armor_hits
    assert without_audit.total_armor_penetrations == result.total_armor_penetrations
    assert without_audit.total_component_hits == result.total_component_hits
    assert without_audit.total_component_penetrations == result.total_component_penetrations


def test_fragment_audit_records_upstream_blocker_and_unprocessed_component(
    monkeypatch,
    target_data,
    case_data,
    registry,
):
    registry_data = deepcopy(registry.data)
    registry_data["fragment_response_models"]["demo_armor_response"]["ballistic_limit"][
        "coefficient"
    ] = 1.0e9
    blocking_registry = PhysicsRegistry(registry_data)
    fragment = Fragment(
        fragment_id=0,
        mass_kg=0.005,
        origin_m=np.array([-2.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        initial_speed_mps=1000.0,
        area_m2=1.0e-4,
        drag_coefficient=0.9,
    )
    monkeypatch.setattr(
        "damage_engine.fragments.generate_fragments",
        lambda case, registry, rng: [fragment],
    )

    result = simulate_fragments(
        case_data,
        compile_target(target_data),
        blocking_registry,
        np.random.default_rng(8),
        parameter_realization=nominal_target_parameters(target_data),
        capture_audit=True,
    )

    trace = next(item for item in result.audit if item["record_type"] == "fragment_trace")
    assert [item["object_id"] for item in trace["geometry_candidate_path"]] == [
        "armor-front",
        "flight-control",
    ]
    assert [item["object_id"] for item in trace["processed_path"]] == [
        "armor-front"
    ]
    assert trace["blocking_object_id"] == "armor-front"
    assert trace["blocking_reason"] == "armor_not_penetrated"
    assert trace["completed_geometry_path"] is False
    report = summarize_component_reachability(result.audit, ["flight-control"])
    assert report["components"]["flight-control"]["counts"][
        "blocked_before_component"
    ] == 1


@pytest.mark.parametrize("seed", [0, 3, 19])
def test_fragment_audit_capture_does_not_change_probabilistic_physics(
    seed,
    target_data,
    case_data,
    registry,
):
    compiled = compile_target(target_data)
    parameters = nominal_target_parameters(target_data)
    without_audit = simulate_fragments(
        case_data,
        compiled,
        registry,
        np.random.default_rng(seed),
        parameter_realization=parameters,
        capture_audit=False,
    )
    with_audit = simulate_fragments(
        case_data,
        compiled,
        registry,
        np.random.default_rng(seed),
        parameter_realization=parameters,
        capture_audit=True,
    )

    without_payload = asdict(without_audit)
    with_payload = asdict(with_audit)
    assert without_payload.pop("audit") == []
    assert with_payload.pop("audit")
    assert without_payload == with_payload


def test_blast_response_is_stable_for_extreme_load_ratios():
    state_model = {
        "peak_50_pa": 1.0e-300,
        "impulse_50_pa_s": 1.0e-300,
        "peak_exponent": 2.0,
        "impulse_exponent": 2.0,
        "duration_50_s": 1.0,
        "duration_exponent": 1.0,
        "log_slope": 4.0,
    }
    assert _state_probability(1.0e300, 1.0e300, 1.0, state_model, 1.0) == 1.0
    assert _state_probability(0.0, 0.0, 0.0, state_model, 1.0) == 0.0


def test_blast_arrival_time_uses_case_temperature(registry):
    model = registry.blast_field("demo_scaled_distance_field")
    cold = _free_field_load(1.0, 1.0, model, 101325.0, 250.0)
    hot = _free_field_load(1.0, 1.0, model, 101325.0, 350.0)
    assert cold.peak_pressure_pa == hot.peak_pressure_pa
    assert hot.arrival_time_s < cold.arrival_time_s


def test_perforated_armor_transmission_is_area_weighted(target_data, registry):
    target = compile_target(target_data)
    parameters = nominal_target_parameters(target_data)
    load = BlastLoad(100.0, 50.0, 0.01, 0.005)
    origin = np.array([-2.0, 0.0, 0.0])
    destination = target.components[0].geometry.center

    intact, _, _ = _transmit(
        load,
        target,
        origin,
        destination,
        registry,
        {"armor-front": 1.0},
        parameters,
    )
    half_open, _, _ = _transmit(
        load,
        target,
        origin,
        destination,
        registry,
        {"armor-front": 0.5},
        parameters,
    )
    open_panel, _, _ = _transmit(
        load,
        target,
        origin,
        destination,
        registry,
        {"armor-front": 0.0},
        parameters,
    )

    assert intact.peak_pressure_pa < half_open.peak_pressure_pa < open_panel.peak_pressure_pa
    assert open_panel.peak_pressure_pa == load.peak_pressure_pa
    assert half_open.peak_pressure_pa == pytest.approx(
        0.5 * (intact.peak_pressure_pa + open_panel.peak_pressure_pa)
    )


def test_armor_blast_failure_reduces_integrity_and_shielding(
    target_data,
    case_data,
    registry,
):
    registry_data = deepcopy(registry.data)
    registry_data["blast_response_models"]["demo_structural_blast"]["damage_states"] = {
        "failed": {
            "peak_50_pa": 1.0e6,
            "impulse_50_pa_s": 1.0e300,
            "peak_exponent": 1.0,
            "impulse_exponent": 1.0,
            "log_slope": 8.0,
        }
    }
    response_registry = PhysicsRegistry(registry_data)

    low_threshold_target = deepcopy(target_data)
    low_threshold_target["armor"][0]["response"]["blast_response"][
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
    high_threshold_target = deepcopy(low_threshold_target)
    high_threshold_target["armor"][0]["response"]["blast_response"][
        "peak_overpressure_threshold"
    ]["value"] = 1.0e9

    low = simulate_blast(
        case_data,
        compile_target(low_threshold_target),
        response_registry,
        np.random.default_rng(4),
        parameter_realization=nominal_target_parameters(low_threshold_target),
        armor_integrity={"armor-front": 1.0},
    )
    high = simulate_blast(
        case_data,
        compile_target(high_threshold_target),
        response_registry,
        np.random.default_rng(4),
        parameter_realization=nominal_target_parameters(high_threshold_target),
        armor_integrity={"armor-front": 1.0},
    )

    assert low.armor_metrics["armor-front"]["blast_failure_probability"] > high.armor_metrics[
        "armor-front"
    ]["blast_failure_probability"]
    assert low.armor_integrity["armor-front"] < high.armor_integrity["armor-front"]
    assert (
        low.component_metrics["flight-control"]["effective_peak_pressure_pa"]
        > high.component_metrics["flight-control"]["effective_peak_pressure_pa"]
    )


def test_registry_rejects_nonphysical_blast_distance_trend(registry):
    data = deepcopy(registry.data)
    table = data["blast_field_models"]["demo_scaled_distance_field"]["scaled_distance_table"]
    table[2]["peak_pressure_pa"] = 800000.0
    with pytest.raises(InputValidationError, match="must not increase"):
        PhysicsRegistry(data)


def test_fixed_epistemic_parameters_do_not_consume_aleatory_rng(target_data):
    target = deepcopy(target_data)
    vulnerable = target["components"][0]["response"]["vulnerable_area_ratio"]
    vulnerable["uncertainty"] = {
        "kind": "uniform",
        "parameters": {"minimum": 0.1, "maximum": 0.9},
        "epistemic": False,
    }
    nominal = nominal_target_parameters(target)
    epistemic = {
        object_id: {
            "equivalent_thickness_m": values["equivalent_thickness_m"],
            "vulnerable_area_ratio": (
                None if object_id == "flight-control" else values["vulnerable_area_ratio"]
            ),
            "blast_peak_threshold_pa": values["blast_peak_threshold_pa"],
        }
        for object_id, values in nominal.items()
    }
    expected = float(np.random.default_rng(123).uniform(0.1, 0.9))
    realized = sample_target_realization(
        target,
        np.random.default_rng(123),
        epistemic,
    )
    assert realized["flight-control"]["vulnerable_area_ratio"] == expected


def test_nonpositive_equivalent_thickness_is_rejected(target_data):
    target = deepcopy(target_data)
    parameter = target["components"][0]["response"]["equivalent_thickness"]
    parameter["value"] = 0.0
    parameter["valid_range"] = [0.0, 5.0]
    with pytest.raises(InputValidationError, match="equivalent thickness"):
        nominal_target_parameters(target)


def test_target_impulse_and_duration_thresholds_override_component_pi_response(
    target_data,
    case_data,
    registry,
):
    def parameter(value, unit, valid_range):
        return {
            "value": value,
            "unit": unit,
            "source_id": "TEST",
            "calibration_id": None,
            "validation_status": "unverified",
            "valid_range": valid_range,
            "uncertainty": {"kind": "none", "parameters": {}, "epistemic": False},
        }

    low_threshold_target = deepcopy(target_data)
    response = low_threshold_target["components"][0]["response"]["blast_response"]
    response["peak_overpressure_threshold"] = parameter(
        1.0e300, "Pa", [1.0, 1.0e300]
    )
    response["impulse_threshold"] = parameter(1.0, "Pa*s", [1.0, 1.0e9])
    response["duration_threshold"] = parameter(1.0, "ms", [1.0, 1000.0])

    high_threshold_target = deepcopy(low_threshold_target)
    high_response = high_threshold_target["components"][0]["response"]["blast_response"]
    high_response["impulse_threshold"]["value"] = 1.0e9
    high_response["duration_threshold"]["value"] = 1000.0

    low = simulate_blast(
        case_data,
        compile_target(low_threshold_target),
        registry,
        np.random.default_rng(5),
        parameter_realization=nominal_target_parameters(low_threshold_target),
    )
    high = simulate_blast(
        case_data,
        compile_target(high_threshold_target),
        registry,
        np.random.default_rng(5),
        parameter_realization=nominal_target_parameters(high_threshold_target),
    )

    assert low.component_state_probabilities["flight-control"]["degraded"] > high.component_state_probabilities[
        "flight-control"
    ]["degraded"]
    realized = nominal_target_parameters(low_threshold_target)["flight-control"]
    assert realized["blast_impulse_threshold_pa_s"] == 1.0
    assert realized["blast_duration_threshold_s"] == 0.001
