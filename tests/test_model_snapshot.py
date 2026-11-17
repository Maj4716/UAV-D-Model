from __future__ import annotations

from copy import deepcopy
import pickle

import pytest

from damage_engine.geometry import compile_target
from damage_engine.registry import PhysicsRegistry


def test_physics_registry_detaches_constructor_input_and_public_snapshots(registry):
    source = deepcopy(registry.data)
    frozen = PhysicsRegistry(source)
    original_hash = frozen.hash
    original_coefficient = frozen.fragment_response("demo_armor_response")[
        "ballistic_limit"
    ]["coefficient"]

    source["fragment_response_models"]["demo_armor_response"]["ballistic_limit"][
        "coefficient"
    ] = 999999.0
    exported = frozen.data
    exported["fragment_response_models"]["demo_armor_response"]["ballistic_limit"][
        "coefficient"
    ] = 888888.0
    selected = frozen.fragment_response("demo_armor_response")
    with pytest.raises(TypeError, match="immutable"):
        selected["ballistic_limit"]["coefficient"] = 777777.0

    assert frozen.hash == original_hash
    assert (
        frozen.fragment_response("demo_armor_response")["ballistic_limit"][
            "coefficient"
        ]
        == original_coefficient
    )


def test_compiled_target_detaches_response_and_source_from_input(target_data):
    source = deepcopy(target_data)
    compiled = compile_target(source)
    first = compiled.objects[0]
    original_model_id = first.response["fragment_response_model_id"]
    original_name = first.source["name"]

    source["armor"][0]["response"]["fragment_response_model_id"] = "mutated"
    source["armor"][0]["name"] = "mutated"

    assert first.response["fragment_response_model_id"] == original_model_id
    assert first.source["name"] == original_name
    with pytest.raises(TypeError, match="immutable"):
        first.response["fragment_response_model_id"] = "second-mutation"
    with pytest.raises(TypeError, match="immutable"):
        first.source["response"]["material_id"] = "second-mutation"

    component_center = compiled.components[0].geometry.center
    with pytest.raises(ValueError):
        component_center[0] = component_center[0] + 1.0
    with pytest.raises(TypeError, match="immutable"):
        compiled.components[0].geometry.center_value = component_center.copy()
    with pytest.raises(AttributeError):
        compiled.target_id = "mutated"


def test_immutable_model_snapshots_round_trip_through_process_serialization(
    registry, target_data
):
    restored_registry = pickle.loads(pickle.dumps(registry))
    restored_target = pickle.loads(pickle.dumps(compile_target(target_data)))

    assert restored_registry.hash == registry.hash
    assert (
        restored_registry.fragment_response("demo_armor_response")[
            "ballistic_limit"
        ]["coefficient"]
        == registry.fragment_response("demo_armor_response")["ballistic_limit"][
            "coefficient"
        ]
    )
    assert restored_target.component_order == compile_target(target_data).component_order
    with pytest.raises(TypeError, match="immutable"):
        restored_target.objects[0].source["name"] = "mutated"
