from __future__ import annotations

from copy import deepcopy

import pytest

from damage_engine.errors import InputValidationError
from damage_engine.registry import PhysicsRegistry
from damage_engine.utils import content_hash
from damage_engine.validation import (
    clear_validation_caches,
    validate_inputs,
    validation_cache_info,
)


def test_static_cache_reuses_contract_but_revalidates_runtime_state(
    target_data, case_data, registry
):
    clear_validation_caches()
    validate_inputs(target_data, case_data, registry)
    first = validation_cache_info()
    assert first["static_misses"] == 1
    assert first["static_hits"] == 0

    second_case = deepcopy(case_data)
    second_case["case_id"] = "same-contract-new-runtime-state"
    second_case["analysis"]["aleatory_seed"] += 1
    second_case["analysis"]["epistemic_sample_id"] = "epistemic-cache-check"
    second_case["encounter"]["detonation_position_m"][0] += 0.01
    second_case["metadata"]["split_group"] = "per-candidate-metadata-does-not-affect-static-contract"
    validate_inputs(target_data, second_case, registry)
    second = validation_cache_info()
    assert second["static_misses"] == 1
    assert second["static_hits"] == 1
    assert second["schema_hits"] > first["schema_hits"]

    # A cache hit may never bypass quaternion validation on the runtime state.
    invalid_runtime = deepcopy(second_case)
    invalid_runtime["encounter"]["body_to_target_orientation_xyzw"] = [0.0, 0.0, 0.0, 2.0]
    with pytest.raises(InputValidationError, match="unit norm"):
        validate_inputs(target_data, invalid_runtime, registry)


def test_static_cache_key_binds_target_and_registry_content(
    target_data, case_data, registry
):
    clear_validation_caches()
    validate_inputs(target_data, case_data, registry)

    changed_target = deepcopy(target_data)
    changed_target["target_revision"] = "demo-cache-key-2"
    target_case = deepcopy(case_data)
    target_case["target_ref"]["target_revision"] = changed_target["target_revision"]
    target_case["target_ref"]["content_hash"] = content_hash(changed_target)
    validate_inputs(changed_target, target_case, registry)
    assert validation_cache_info()["static_misses"] == 2

    changed_registry_data = deepcopy(registry.data)
    changed_registry_data["config_id"] = "synthetic-cache-key-2"
    changed_registry = PhysicsRegistry(changed_registry_data)
    registry_case = deepcopy(case_data)
    registry_case["parameter_set_id"] = changed_registry_data["config_id"]
    validate_inputs(target_data, registry_case, changed_registry)
    assert validation_cache_info()["static_misses"] == 3

    # Mutating a previously validated registry changes the hash, forces a miss,
    # and reaches semantic validation instead of inheriting the old success.
    changed_registry.data["fragment_source_models"]["demo_cone_source"][
        "valid_speed_mps"
    ] = [4000.0, 1000.0]
    with pytest.raises(InputValidationError, match="strictly increasing"):
        validate_inputs(target_data, registry_case, changed_registry)
    assert validation_cache_info()["static_misses"] == 4


def test_invalid_target_mutation_cannot_reuse_prior_success(
    target_data, case_data, registry
):
    clear_validation_caches()
    validate_inputs(target_data, case_data, registry)

    changed_target = deepcopy(target_data)
    changed_target["coordinate_frame"]["x_axis"] = changed_target["coordinate_frame"][
        "y_axis"
    ]
    changed_case = deepcopy(case_data)
    changed_case["target_ref"]["content_hash"] = content_hash(changed_target)
    with pytest.raises(InputValidationError, match="axis descriptions must be distinct"):
        validate_inputs(changed_target, changed_case, registry)
    assert validation_cache_info()["static_misses"] == 2


def test_external_fragment_list_is_verified_on_every_case(
    target_data, case_data, registry, monkeypatch
):
    clear_validation_caches()
    case = deepcopy(case_data)
    case["warhead"]["fragment_source"]["fragment_list_ref"] = "changing-external.jsonl"
    calls = 0

    def reject_external(_source):
        nonlocal calls
        calls += 1
        raise InputValidationError("external bytes checked")

    monkeypatch.setattr("damage_engine.validation.load_fragment_list", reject_external)
    for _ in range(2):
        with pytest.raises(InputValidationError, match="external bytes checked"):
            validate_inputs(target_data, case, registry)
    assert calls == 2
    assert validation_cache_info()["static_entries"] == 0
