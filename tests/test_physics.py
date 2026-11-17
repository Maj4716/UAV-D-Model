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
from damage_engine.fragments import (
    Fragment,
    armor_reference_area_m2,
    _fragment_physical_keys,
    generate_fragments,
    simulate_fragments,
)
from damage_engine.geometry import compile_target
from damage_engine.parameters import nominal_target_parameters, sample_target_realization
from damage_engine.panel_damage import PanelDamageState
from damage_engine.registry import PhysicsRegistry


def test_target_frame_fragment_speed_must_remain_in_model_domain(case_data, registry):
    case = deepcopy(case_data)
    # Registered source speed remains valid, but encounter translation pushes
    # every target-frame fragment speed above the calibrated upper bound.
    case["encounter"]["velocity_target_mps"] = [6000.0, 0.0, 0.0]
    with pytest.raises(OutOfDomainError, match="target-frame"):
        generate_fragments(case, registry, np.random.default_rng(7))


def test_armor_integrity_denominator_uses_physical_panel_face(target_data):
    armor = compile_target(target_data).armor[0]

    assert armor_reference_area_m2(armor) == pytest.approx(4.0)


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


def test_simultaneous_fragments_do_not_reuse_a_hole_by_loop_order(
    monkeypatch,
    target_data,
    case_data,
    registry,
):
    registry_data = deepcopy(registry.data)
    registry_data["fragment_response_models"]["demo_armor_response"][
        "ballistic_limit"
    ]["coefficient"] = 1.0
    registry_data["fragment_response_models"]["demo_armor_response"][
        "ballistic_limit"
    ]["transition_width_ratio"] = 0.001
    permissive_registry = PhysicsRegistry(registry_data)
    fragments = [
        Fragment(
            fragment_id=index,
            mass_kg=0.005,
            origin_m=np.array([-2.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            initial_speed_mps=1000.0,
            area_m2=1.0e-4,
            drag_coefficient=0.9,
        )
        for index in range(2)
    ]
    monkeypatch.setattr(
        "damage_engine.fragments.generate_fragments",
        lambda case, registry, rng: fragments,
    )

    result = simulate_fragments(
        case_data,
        compile_target(target_data),
        permissive_registry,
        np.random.default_rng(3),
        parameter_realization=nominal_target_parameters(target_data),
        capture_audit=True,
    )
    armor_events = [
        event
        for event in result.audit
        if event.get("record_type") == "intersection"
        and event.get("object_id") == "armor-front"
    ]

    assert len(armor_events) == 2
    assert [event["arrival_time_s"] for event in armor_events] == pytest.approx(
        [armor_events[0]["arrival_time_s"]] * 2
    )
    assert not any(event["existing_hole_reused"] for event in armor_events)
    assert result.total_existing_hole_reuses == 0
    assert len({event["fragment_physical_key"] for event in armor_events}) == 2
    assert len({event["event_random_key_hash"] for event in armor_events}) == 2
    assert {
        event["event_random_contract"] for event in armor_events
    } == {"fragment-event-key-v2"}
    assert armor_events[0]["sampled_ballistic_limit_mps"] != pytest.approx(
        armor_events[1]["sampled_ballistic_limit_mps"],
        rel=0.0,
        abs=0.0,
    )


def test_duplicate_fragment_keys_are_stable_to_unrelated_insertion_and_reordering():
    duplicate_a = Fragment(
        fragment_id=7,
        mass_kg=0.005,
        origin_m=np.array([-2.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        initial_speed_mps=1000.0,
        area_m2=1.0e-4,
        drag_coefficient=0.9,
    )
    duplicate_b = Fragment(
        fragment_id=9,
        mass_kg=duplicate_a.mass_kg,
        origin_m=duplicate_a.origin_m.copy(),
        direction=duplicate_a.direction.copy(),
        initial_speed_mps=duplicate_a.initial_speed_mps,
        area_m2=duplicate_a.area_m2,
        drag_coefficient=duplicate_a.drag_coefficient,
    )
    unrelated = Fragment(
        fragment_id=1,
        mass_kg=0.003,
        origin_m=np.array([-1.0, 0.5, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        initial_speed_mps=700.0,
        area_m2=8.0e-5,
        drag_coefficient=0.8,
    )

    baseline = dict(
        zip(
            (7, 9),
            _fragment_physical_keys([duplicate_a, duplicate_b]),
        )
    )
    reordered_keys = _fragment_physical_keys(
        [unrelated, duplicate_b, duplicate_a]
    )
    reordered = dict(zip((1, 9, 7), reordered_keys))

    assert baseline[7] != baseline[9]
    assert reordered[7] == baseline[7]
    assert reordered[9] == baseline[9]


def test_fragment_cannot_reuse_through_hole_before_first_fragment_exits_material(
    monkeypatch,
    target_data,
    case_data,
    registry,
):
    registry_data = deepcopy(registry.data)
    armor_model = registry_data["fragment_response_models"]["demo_armor_response"]
    armor_model["ballistic_limit"]["coefficient"] = 1.0
    armor_model["ballistic_limit"]["transition_width_ratio"] = 0.001
    permissive_registry = PhysicsRegistry(registry_data)
    fragments = [
        Fragment(
            fragment_id=0,
            source_fragment_id="first",
            mass_kg=0.005,
            origin_m=np.array([-0.501, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            initial_speed_mps=1000.0,
            area_m2=1.0e-4,
            drag_coefficient=1.0e-9,
        ),
        Fragment(
            fragment_id=1,
            source_fragment_id="during-transit",
            mass_kg=0.005,
            origin_m=np.array([-0.506, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            initial_speed_mps=1000.0,
            area_m2=1.0e-4,
            drag_coefficient=1.0e-9,
        ),
    ]
    monkeypatch.setattr(
        "damage_engine.fragments.generate_fragments",
        lambda case, registry, rng: fragments,
    )

    result = simulate_fragments(
        case_data,
        compile_target(target_data),
        permissive_registry,
        np.random.default_rng(3),
        parameter_realization=nominal_target_parameters(target_data),
        capture_audit=True,
    )
    events = [
        event
        for event in result.audit
        if event.get("record_type") == "intersection"
        and event.get("object_id") == "armor-front"
    ]

    assert [event["source_fragment_id"] for event in events] == [
        "first",
        "during-transit",
    ]
    assert events[0]["arrival_time_s"] < events[1]["arrival_time_s"]
    assert events[1]["arrival_time_s"] < events[0]["hole_formation_time_s"]
    assert events[1]["existing_hole_reused"] is False


def test_fragment_impacts_follow_global_arrival_time_across_downstream_panels(
    monkeypatch,
    target_data,
    case_data,
    registry,
):
    target = deepcopy(target_data)
    target["components"][0]["geometry"]["center_m"] = [0.0, 10.0, 0.0]
    back = deepcopy(target["armor"][0])
    back["id"] = "armor-back"
    back["name"] = "Synthetic back panel"
    back["geometry"]["vertices_m"] = [
        [0.5, -1.0, -1.0],
        [0.5, 1.0, -1.0],
        [0.5, 1.0, 1.0],
        [0.5, -1.0, 1.0],
    ]
    target["armor"].append(back)

    registry_data = deepcopy(registry.data)
    armor_model = registry_data["fragment_response_models"]["demo_armor_response"]
    armor_model["ballistic_limit"]["coefficient"] = 1.0
    armor_model["ballistic_limit"]["transition_width_ratio"] = 0.001
    permissive_registry = PhysicsRegistry(registry_data)
    fragments = [
        Fragment(
            fragment_id=0,
            source_fragment_id="slow-near",
            mass_kg=0.005,
            origin_m=np.array([-0.6, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            initial_speed_mps=100.0,
            area_m2=1.0e-4,
            drag_coefficient=1.0e-9,
        ),
        Fragment(
            fragment_id=1,
            source_fragment_id="fast-far",
            mass_kg=0.005,
            origin_m=np.array([-2.5, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            initial_speed_mps=1000.0,
            area_m2=1.0e-4,
            drag_coefficient=1.0e-9,
        ),
    ]
    monkeypatch.setattr(
        "damage_engine.fragments.generate_fragments",
        lambda case, registry, rng: fragments,
    )

    result = simulate_fragments(
        case_data,
        compile_target(target),
        permissive_registry,
        np.random.default_rng(3),
        parameter_realization=nominal_target_parameters(target),
        capture_audit=True,
    )
    back_events = [
        event
        for event in result.audit
        if event.get("record_type") == "intersection"
        and event.get("object_id") == "armor-back"
    ]

    assert [event["source_fragment_id"] for event in back_events] == [
        "fast-far",
        "slow-near",
    ]
    assert back_events[0]["arrival_time_s"] < back_events[1]["arrival_time_s"]
    assert back_events[0]["existing_hole_reused"] is False
    assert back_events[1]["existing_hole_reused"] is True


def test_unrelated_fragment_does_not_reassign_existing_event_random_draw(
    monkeypatch,
    target_data,
    case_data,
    registry,
):
    target = deepcopy(target_data)
    target["components"][0]["geometry"]["center_m"] = [0.0, 10.0, 0.0]
    compiled = compile_target(target)
    parameters = nominal_target_parameters(target)
    existing = Fragment(
        fragment_id=17,
        source_fragment_id="existing-event",
        mass_kg=0.005,
        origin_m=np.array([-2.5, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        initial_speed_mps=700.0,
        area_m2=1.0e-4,
        drag_coefficient=1.0e-9,
    )
    unrelated = Fragment(
        fragment_id=3,
        source_fragment_id="unrelated-event",
        mass_kg=0.005,
        origin_m=np.array([-0.6, 0.5, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        initial_speed_mps=700.0,
        area_m2=1.0e-4,
        drag_coefficient=1.0e-9,
    )
    active = [existing]
    monkeypatch.setattr(
        "damage_engine.fragments.generate_fragments",
        lambda case, registry, rng: list(active),
    )

    baseline = simulate_fragments(
        case_data,
        compiled,
        registry,
        np.random.default_rng(91),
        parameter_realization=parameters,
        capture_audit=True,
    )
    active[:] = [unrelated, existing]
    extended = simulate_fragments(
        case_data,
        compiled,
        registry,
        np.random.default_rng(91),
        parameter_realization=parameters,
        capture_audit=True,
    )

    baseline_event = next(
        event
        for event in baseline.audit
        if event.get("source_fragment_id") == "existing-event"
        and event.get("record_type") == "intersection"
    )
    extended_event = next(
        event
        for event in extended.audit
        if event.get("source_fragment_id") == "existing-event"
        and event.get("record_type") == "intersection"
    )
    assert extended_event["sampled_ballistic_limit_mps"] == pytest.approx(
        baseline_event["sampled_ballistic_limit_mps"], rel=0.0, abs=0.0
    )
    assert extended_event["penetrated"] is baseline_event["penetrated"]
    assert extended_event["residual_speed_mps"] == pytest.approx(
        baseline_event["residual_speed_mps"], rel=0.0, abs=0.0
    )


def test_fragment_engine_passes_component_hit_locations_to_spatial_damage_model(
    monkeypatch,
    target_data,
    case_data,
    registry,
):
    registry_data = deepcopy(registry.data)
    for model in registry_data["fragment_response_models"].values():
        model["ballistic_limit"]["coefficient"] = 1.0
        model["ballistic_limit"]["transition_width_ratio"] = 0.001
    component_model = registry_data["fragment_response_models"][
        "demo_component_response"
    ]
    component_model["damage_accumulation_model"] = (
        "spatial_cumulative_damage_energy"
    )
    component_model["damage_correlation_length_m"] = 0.02
    spatial_registry = PhysicsRegistry(registry_data)
    fragments = [
        Fragment(
            fragment_id=index,
            source_fragment_id=f"spatial-{index}",
            mass_kg=0.005,
            origin_m=np.array([-2.0, y, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            initial_speed_mps=1000.0,
            area_m2=1.0e-4,
            drag_coefficient=1.0e-9,
        )
        for index, y in enumerate((-0.05, 0.05))
    ]
    monkeypatch.setattr(
        "damage_engine.fragments.generate_fragments",
        lambda case, registry, rng: fragments,
    )

    result = simulate_fragments(
        case_data,
        compile_target(target_data),
        spatial_registry,
        np.random.default_rng(7),
        parameter_realization=nominal_target_parameters(target_data),
    )

    assert result.component_metrics["flight-control"][
        "spatial_damage_cluster_count"
    ] == 2.0


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
    without_panel_state = without_payload.pop("panel_damage_state")
    with_panel_state = with_payload.pop("panel_damage_state")
    assert without_panel_state is not None
    assert with_panel_state is not None
    assert without_panel_state.summary() == with_panel_state.summary()
    assert without_payload == with_payload


def test_out_of_domain_fragment_contacts_are_reported_without_aborting_replicate(
    target_data,
    case_data,
    registry,
):
    registry_data = deepcopy(registry.data)
    for model in registry_data["fragment_response_models"].values():
        model["ballistic_limit"]["maximum_obliquity_deg"] = 0.1
    restricted = PhysicsRegistry(registry_data)

    result = simulate_fragments(
        case_data,
        compile_target(target_data),
        restricted,
        np.random.default_rng(0),
        parameter_realization=nominal_target_parameters(target_data),
        capture_audit=True,
    )

    assert result.total_out_of_domain_impacts > 0
    assert any(
        event["record_type"] == "out_of_domain_intersection"
        for event in result.audit
    )


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


def test_blast_positive_phase_waveform_matches_peak_duration_and_impulse():
    load = BlastLoad(100.0, 25.0, 1.0, 0.0)

    assert load.positive_phase_shape_exponent == pytest.approx(3.0)
    assert load.impulse_fraction(0.0, 0.5) == pytest.approx(0.9375)
    assert load.impulse_fraction(0.5, 1.0) == pytest.approx(0.0625)
    assert load.impulse_fraction(0.0, 1.0) == pytest.approx(1.0)


def test_thick_shield_returns_a_representable_equivalent_positive_phase(
    target_data,
    case_data,
    registry,
):
    target = deepcopy(target_data)
    thickness = target["armor"][0]["response"]["equivalent_thickness"]
    thickness["value"] = 100.0
    thickness["valid_range"] = [1.0, 200.0]

    result = simulate_blast(
        case_data,
        compile_target(target),
        registry,
        parameter_realization=nominal_target_parameters(target),
    )
    metrics = result.component_metrics["flight-control"]

    assert metrics["effective_peak_pressure_pa"] > 0.0
    assert metrics["effective_impulse_pa_s"] > 0.0
    assert metrics["positive_duration_s"] >= (
        metrics["effective_impulse_pa_s"]
        / metrics["effective_peak_pressure_pa"]
    )
    assert np.isfinite(metrics["positive_phase_shape_exponent"])


def test_late_local_opening_contributes_the_true_post_opening_pressure_peak(
    target_data,
    case_data,
    registry,
):
    target = compile_target(target_data)
    parameters = nominal_target_parameters(target_data)
    origin = np.asarray(
        case_data["encounter"]["detonation_position_m"], dtype=np.float64
    )
    component_center = target.components[0].geometry.center
    panel_hit = target.armor_on_segment(origin, component_center)[0]
    baseline = simulate_blast(
        case_data,
        target,
        registry,
        parameter_realization=parameters,
    )
    panel_metrics = baseline.armor_metrics["armor-front"]

    registry_data = deepcopy(registry.data)
    transmission = registry_data["transmission_models"]["demo_steel"]
    transmission["peak_half_thickness_m"] = 1.0e-12
    transmission["impulse_half_thickness_m"] = 1.0e-12
    transmission["minimum_transmission"] = 0.0
    opaque_registry = PhysicsRegistry(registry_data)
    panel_state = PanelDamageState({"armor-front": 4.0})
    panel_state.record_perforation(
        "armor-front",
        panel_hit.entry_point,
        panel_hit.entry_normal,
        hole_area_m2=4.0,
        formed_time_s=(
            panel_metrics["arrival_time_s"]
            + 0.25 * panel_metrics["positive_duration_s"]
        ),
    )

    result = simulate_blast(
        case_data,
        target,
        opaque_registry,
        parameter_realization=parameters,
        panel_damage_state=panel_state,
    )
    metrics = result.component_metrics["flight-control"]

    assert 0.0 < metrics["effective_peak_pressure_pa"] < metrics[
        "incident_peak_pressure_pa"
    ]
    assert metrics["effective_impulse_pa_s"] > 0.0
    assert metrics["positive_duration_s"] >= (
        metrics["effective_impulse_pa_s"]
        / metrics["effective_peak_pressure_pa"]
    )


def test_perforated_armor_transmission_is_area_weighted(target_data, registry):
    target = compile_target(target_data)
    parameters = nominal_target_parameters(target_data)
    load = BlastLoad(100.0, 0.5, 0.01, 0.005)
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


def test_blast_transmission_uses_bounded_panel_open_area_fraction(
    target_data,
    registry,
):
    target = compile_target(target_data)
    parameters = nominal_target_parameters(target_data)
    load = BlastLoad(100.0, 0.5, 0.01, 0.005)
    origin = np.array([-2.0, 0.0, 0.0])
    destination = target.components[0].geometry.center
    hit = target.armor_on_segment(origin, destination)[0]

    on_path = PanelDamageState({"armor-front": 1.0})
    on_path.record_perforation(
        "armor-front",
        hit.entry_point,
        hit.entry_normal,
        hole_area_m2=1.0e-4,
    )
    through_hole, shielding, _ = _transmit(
        load,
        target,
        origin,
        destination,
        registry,
        {"armor-front": 1.0},
        parameters,
        on_path,
        panel_arrival_time_s={"armor-front": load.arrival_time_s},
    )

    off_path = PanelDamageState({"armor-front": 1.0})
    off_path.record_perforation(
        "armor-front",
        hit.entry_point + np.array([0.0, 0.2, 0.0]),
        hit.entry_normal,
        hole_area_m2=1.0e-4,
    )
    shielded, _, _ = _transmit(
        load,
        target,
        origin,
        destination,
        registry,
        {"armor-front": 1.0},
        parameters,
        off_path,
        panel_arrival_time_s={"armor-front": load.arrival_time_s},
    )

    intact, _, _ = _transmit(
        load,
        target,
        origin,
        destination,
        registry,
        {"armor-front": 1.0},
        parameters,
    )

    assert intact.peak_pressure_pa < through_hole.peak_pressure_pa < load.peak_pressure_pa
    assert intact.impulse_pa_s < through_hole.impulse_pa_s < load.impulse_pa_s
    assert shielding > 0.0
    assert shielded.peak_pressure_pa == pytest.approx(intact.peak_pressure_pa)
    assert shielded.impulse_pa_s == pytest.approx(intact.impulse_pa_s)


def test_blast_transmission_only_uses_holes_formed_before_wave_arrival(
    target_data,
    registry,
):
    target = compile_target(target_data)
    parameters = nominal_target_parameters(target_data)
    origin = np.array([-2.0, 0.0, 0.0])
    destination = target.components[0].geometry.center
    hit = target.armor_on_segment(origin, destination)[0]
    panel_state = PanelDamageState({"armor-front": 1.0})
    panel_state.record_perforation(
        "armor-front",
        hit.entry_point,
        hit.entry_normal,
        hole_area_m2=1.0e-4,
        formed_time_s=0.01,
    )

    early_wave, _, _ = _transmit(
        BlastLoad(100.0, 0.5, 0.01, 0.02),
        target,
        origin,
        destination,
        registry,
        {"armor-front": 1.0},
        parameters,
        panel_state,
        panel_arrival_time_s={"armor-front": 0.005},
    )
    late_wave, _, _ = _transmit(
        BlastLoad(100.0, 0.5, 0.01, 0.02),
        target,
        origin,
        destination,
        registry,
        {"armor-front": 1.0},
        parameters,
        panel_state,
        panel_arrival_time_s={"armor-front": 0.02},
    )
    fully_early_wave, _, _ = _transmit(
        BlastLoad(100.0, 0.2, 0.004, 0.009),
        target,
        origin,
        destination,
        registry,
        {"armor-front": 1.0},
        parameters,
        panel_state,
        panel_arrival_time_s={"armor-front": 0.005},
    )

    assert early_wave.peak_pressure_pa < 100.0
    assert fully_early_wave.peak_pressure_pa == early_wave.peak_pressure_pa
    assert fully_early_wave.impulse_pa_s < early_wave.impulse_pa_s < 0.5
    assert early_wave.peak_pressure_pa < late_wave.peak_pressure_pa < 100.0
    assert early_wave.impulse_pa_s < late_wave.impulse_pa_s < 0.5


def test_armor_blast_failure_is_post_pulse_state_without_same_pulse_feedback(
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
        parameter_realization=nominal_target_parameters(low_threshold_target),
        armor_integrity={"armor-front": 1.0},
    )
    high = simulate_blast(
        case_data,
        compile_target(high_threshold_target),
        response_registry,
        parameter_realization=nominal_target_parameters(high_threshold_target),
        armor_integrity={"armor-front": 1.0},
    )

    assert low.armor_metrics["armor-front"]["blast_failure_probability"] > high.armor_metrics[
        "armor-front"
    ]["blast_failure_probability"]
    assert low.armor_integrity["armor-front"] < high.armor_integrity["armor-front"]
    assert low.component_metrics["flight-control"]["effective_peak_pressure_pa"] == pytest.approx(
        high.component_metrics["flight-control"]["effective_peak_pressure_pa"]
    )
    assert low.component_metrics["flight-control"]["effective_impulse_pa_s"] == pytest.approx(
        high.component_metrics["flight-control"]["effective_impulse_pa_s"]
    )
    low_metrics = low.armor_metrics["armor-front"]
    assert low_metrics["same_pulse_failure_feedback_applied"] == 0.0
    assert low_metrics["pre_pulse_shielding_survival"] == 1.0
    assert low_metrics["blast_failure_effective_time_s"] == pytest.approx(
        low_metrics["arrival_time_s"] + low_metrics["positive_duration_s"]
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
        parameter_realization=nominal_target_parameters(low_threshold_target),
    )
    high = simulate_blast(
        case_data,
        compile_target(high_threshold_target),
        registry,
        parameter_realization=nominal_target_parameters(high_threshold_target),
    )

    assert low.component_state_probabilities["flight-control"]["degraded"] > high.component_state_probabilities[
        "flight-control"
    ]["degraded"]
    realized = nominal_target_parameters(low_threshold_target)["flight-control"]
    assert realized["blast_impulse_threshold_pa_s"] == 1.0
    assert realized["blast_duration_threshold_s"] == 0.001
