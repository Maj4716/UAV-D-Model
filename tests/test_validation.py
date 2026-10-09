from __future__ import annotations

from copy import deepcopy

import pytest

from damage_engine.errors import InputValidationError
from damage_engine.registry import PhysicsRegistry
from damage_engine.utils import content_hash
from damage_engine.validation import _validate_schema, validate_inputs


def _case_for(target: dict, case_data: dict) -> dict:
    case = deepcopy(case_data)
    case["target_ref"]["content_hash"] = content_hash(target)
    return case


def test_semantic_guard_accepts_demo_contract(target_data, case_data, registry):
    validate_inputs(target_data, _case_for(target_data, case_data), registry)


def test_semantic_guard_rejects_ratio_outside_physical_bounds(
    target_data, case_data, registry
):
    target = deepcopy(target_data)
    ratio = target["components"][0]["response"]["vulnerable_area_ratio"]
    ratio["value"] = 1.2
    ratio["valid_range"] = [0.0, 1.2]
    with pytest.raises(InputValidationError, match="physical upper bound"):
        validate_inputs(target, _case_for(target, case_data), registry)


def test_semantic_guard_rejects_unordered_parameter_range(
    target_data, case_data, registry
):
    target = deepcopy(target_data)
    target["components"][0]["response"]["equivalent_thickness"]["valid_range"] = [5.0, 1.0]
    with pytest.raises(InputValidationError, match="valid_range must be ordered"):
        validate_inputs(target, _case_for(target, case_data), registry)


def test_semantic_guard_rejects_nonunit_component_orientation(
    target_data, case_data, registry
):
    target = deepcopy(target_data)
    target["components"][0]["geometry"]["orientation_xyzw"] = [0.0, 0.0, 0.0, 2.0]
    with pytest.raises(InputValidationError, match="unit norm in xyzw order"):
        validate_inputs(target, _case_for(target, case_data), registry)


def test_semantic_guard_rejects_nonunit_extrusion_direction(
    target_data, case_data, registry
):
    target = deepcopy(target_data)
    target["armor"][0]["geometry"]["extrusion_direction"] = [2.0, 0.0, 0.0]
    with pytest.raises(InputValidationError, match="must have unit norm"):
        validate_inputs(target, _case_for(target, case_data), registry)


def test_semantic_guard_rejects_unknown_evaluation_standard(
    target_data, case_data, registry
):
    case = _case_for(target_data, case_data)
    case["analysis"]["evaluation_standard_ids"] = ["unavailable-standard"]
    with pytest.raises(InputValidationError, match="not declared by the target damage tree"):
        validate_inputs(target_data, case, registry)


def test_semantic_guard_accepts_multiple_declared_evaluation_standards(
    target_data, case_data, registry
):
    target = deepcopy(target_data)
    target["damage_tree"]["standards"] = [
        {
            "id": "mission-effectiveness",
            "outputs": ["mission_loss", "mission_degraded"],
        },
        {"id": "mission-loss-only", "outputs": ["mission_loss"]},
    ]
    case = _case_for(target, case_data)
    case["analysis"]["evaluation_standard_ids"] = [
        "mission-effectiveness",
        "mission-loss-only",
    ]
    validate_inputs(target, case, registry)


def test_semantic_guard_rejects_parameter_registry_mismatch(
    target_data, case_data, registry
):
    case = _case_for(target_data, case_data)
    case["parameter_set_id"] = "wrong-registry"
    with pytest.raises(InputValidationError, match="does not match the physics registry"):
        validate_inputs(target_data, case, registry)


def test_semantic_guard_rejects_missing_fragment_list_reference(
    target_data, case_data, registry
):
    case = _case_for(target_data, case_data)
    case["warhead"]["fragment_source"]["fragment_list_ref"] = "missing-fragments.csv"
    with pytest.raises(InputValidationError, match="does not resolve"):
        validate_inputs(target_data, case, registry)


def test_semantic_guard_requires_response_model_state_coverage(
    target_data, case_data, registry
):
    registry_data = deepcopy(registry.data)
    del registry_data["blast_response_models"]["demo_electronics_blast"]["damage_states"]["failed"]
    incomplete = PhysicsRegistry(registry_data)
    with pytest.raises(InputValidationError, match="missing from a referenced response model") as error:
        validate_inputs(target_data, _case_for(target_data, case_data), incomplete)
    assert error.value.context["missing_states"] == ["failed"]


def test_schema_contract_accepts_complete_centimetre_geometry(target_data):
    target = deepcopy(target_data)
    target["coordinate_frame"]["length_unit"] = "cm"
    armor_geometry = target["armor"][0]["geometry"]
    armor_geometry["vertices_cm"] = [
        [coordinate * 100.0 for coordinate in vertex]
        for vertex in armor_geometry.pop("vertices_m")
    ]
    armor_geometry["extrusion_distance_cm"] = armor_geometry.pop("extrusion_distance_m") * 100.0
    component_geometry = target["components"][0]["geometry"]
    component_geometry["center_cm"] = [
        coordinate * 100.0 for coordinate in component_geometry.pop("center_m")
    ]
    component_geometry["size_lwh_cm"] = [
        coordinate * 100.0 for coordinate in component_geometry.pop("size_lwh_m")
    ]
    _validate_schema(target, "target.schema.json")


def test_schema_contract_rejects_mixed_geometry_units(target_data):
    target = deepcopy(target_data)
    target["components"][0]["geometry"]["center_cm"] = [0.0, 0.0, 0.0]
    with pytest.raises(InputValidationError) as error:
        _validate_schema(target, "target.schema.json")
    assert any(
        detail["path"] == "components/0/geometry"
        for detail in error.value.context["errors"]
    )


def test_schema_contract_rejects_adaptive_run_below_minimum_sample_count(
    case_data,
):
    case = deepcopy(case_data)
    case["analysis"]["replicates"] = 31
    case["analysis"]["adaptive_ci_half_width"] = 0.05
    with pytest.raises(InputValidationError):
        _validate_schema(case, "simulation_case.schema.json")
