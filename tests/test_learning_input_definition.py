from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import damage_engine.archive_validation as archive_validation
from damage_engine.learning_input_identity import (
    LEARNING_INPUT_IDENTITY_VERSION,
    LOITERING_IMPACT_LEARNING_INPUT_IDENTITY_VERSION,
    LOITERING_IMPACT_MODEL_INPUT_DEFINITION,
    LOITERING_IMPACT_SAMPLING_MODEL_INPUT_CONTRACT,
    MODEL_INPUT_AXIS_NAME_FIELDS,
    MODEL_INPUT_DEFINITIONS,
    MODEL_INPUT_EXCLUDED_TENSOR_FIELDS,
    MODEL_INPUT_MASK_PAIRS,
    MODEL_INPUT_METADATA_FIELDS,
    MODEL_INPUT_SEMANTIC_SHAPES,
    UAV_SAMPLING_MODEL_INPUT_CONTRACT,
    UAV_MODEL_INPUT_DEFINITION,
    learning_input_fingerprint,
    legacy_learning_input_fingerprint,
    loitering_impact_learning_input_fingerprint,
    loitering_impact_state_value,
    model_input_definition_for_sampling_contract,
    model_input_definition_for_tensor_field,
    model_input_definition_for_version,
    uav_state_value,
)
from damage_engine.utils import content_hash, load_json


ROOT = Path(__file__).resolve().parents[1]


def test_model_input_definition_owns_every_persisted_feature_attribute():
    for definition in (
        UAV_MODEL_INPUT_DEFINITION,
        LOITERING_IMPACT_MODEL_INPUT_DEFINITION,
    ):
        size = len(definition.feature_names)
        assert size > 0
        assert {len(values) for values in definition.metadata_fields.values()} == {
            size
        }
        document = definition.manifest_document(
            fixed_scenario_hash="sha256:" + "0" * 64
        )
        assert document["tensor_field"] == definition.tensor_field
        assert document["mask_field"] == definition.mask_field
        assert [feature["name"] for feature in document["features"]] == list(
            definition.feature_names
        )
        assert document["excluded_tensor_fields"] == list(
            MODEL_INPUT_EXCLUDED_TENSOR_FIELDS
        )
        assert definition.tensor_shapes[definition.tensor_field] == definition.shape


def test_reader_axis_mask_and_metadata_contracts_derive_from_definitions():
    expected_metadata = {
        field
        for definition in MODEL_INPUT_DEFINITIONS
        for field in definition.metadata_fields
    }
    assert set(MODEL_INPUT_METADATA_FIELDS) == expected_metadata
    for definition in MODEL_INPUT_DEFINITIONS:
        names_field = f"{definition.metadata_prefix}_feature_names"
        assert MODEL_INPUT_AXIS_NAME_FIELDS[definition.feature_axis] == names_field
        assert MODEL_INPUT_MASK_PAIRS[definition.tensor_field] == definition.mask_field
        assert MODEL_INPUT_SEMANTIC_SHAPES[definition.tensor_field] == (
            "B",
            definition.feature_axis,
        )
        assert MODEL_INPUT_SEMANTIC_SHAPES[definition.mask_field] == (
            "B",
            definition.feature_axis,
        )
        assert archive_validation._AXIS_NAMES[definition.feature_axis] == names_field
        assert archive_validation._MASK_PAIRS[definition.tensor_field] == (
            definition.mask_field
        )


def test_all_model_input_lookup_paths_resolve_the_same_definition_objects():
    for definition in (
        UAV_MODEL_INPUT_DEFINITION,
        LOITERING_IMPACT_MODEL_INPUT_DEFINITION,
    ):
        assert model_input_definition_for_version(definition.version) is definition
        assert (
            model_input_definition_for_sampling_contract(
                definition.sampling_contract
            )
            is definition
        )
        assert (
            model_input_definition_for_tensor_field(definition.tensor_field)
            is definition
        )


def test_default_learner_contract_excludes_teacher_and_geometry_tensors():
    forbidden = {
        "case_x",
        "case_physics_x",
        "solver_context_x",
        "component_x",
        "armor_x",
        "component_armor_path_x",
        "armor_vertices_m",
        "component_vertices_m",
        "tree_edge_index",
    }
    assert forbidden.issubset(MODEL_INPUT_EXCLUDED_TENSOR_FIELDS)
    for definition in (
        UAV_MODEL_INPUT_DEFINITION,
        LOITERING_IMPACT_MODEL_INPUT_DEFINITION,
    ):
        assert definition.tensor_field not in forbidden


def test_definition_learning_identity_matches_legacy_payload_and_hash_bytes(
    case_data,
):
    loitering_case = load_json(
        ROOT / "raw_data" / "case.loitering_impact.v1.json"
    )
    expected_uav = content_hash(
        {
            "version": LEARNING_INPUT_IDENTITY_VERSION,
            "uav_state": uav_state_value(case_data),
        }
    )
    expected_loitering = content_hash(
        {
            "version": LOITERING_IMPACT_LEARNING_INPUT_IDENTITY_VERSION,
            "loitering_impact_state": loitering_impact_state_value(
                loitering_case
            ),
        }
    )
    assert UAV_MODEL_INPUT_DEFINITION.learning_input_payload_key == "uav_state"
    assert (
        LOITERING_IMPACT_MODEL_INPUT_DEFINITION.learning_input_payload_key
        == "loitering_impact_state"
    )
    assert (
        UAV_MODEL_INPUT_DEFINITION.learning_input_fingerprint(case_data)
        == expected_uav
        == "sha256:48dd7190774c527dba9fa69e3775441cbfb3a64b33cec3335ab5be3afcd69309"
    )
    assert (
        LOITERING_IMPACT_MODEL_INPUT_DEFINITION.learning_input_fingerprint(
            loitering_case
        )
        == expected_loitering
        == "sha256:0660aa7fd01c6ee83f5355ee6cafd788120cf98af97c74e209dad4c357fc5641"
    )
    assert learning_input_fingerprint(case_data) == expected_uav
    assert (
        loitering_impact_learning_input_fingerprint(loitering_case)
        == expected_loitering
    )
    assert learning_input_fingerprint(loitering_case) == expected_loitering


@pytest.mark.parametrize(
    ("definition", "contract", "case_factory"),
    (
        (
            UAV_MODEL_INPUT_DEFINITION,
            UAV_SAMPLING_MODEL_INPUT_CONTRACT,
            lambda case: deepcopy(case),
        ),
        (
            LOITERING_IMPACT_MODEL_INPUT_DEFINITION,
            LOITERING_IMPACT_SAMPLING_MODEL_INPUT_CONTRACT,
            lambda _case: load_json(
                ROOT / "raw_data" / "case.loitering_impact.v1.json"
            ),
        ),
    ),
)
def test_learning_identity_entry_resolves_declared_definition(
    case_data,
    definition,
    contract,
    case_factory,
):
    case = case_factory(case_data)
    case.setdefault("metadata", {})["sampling"] = {
        "sampling_spec_document": {
            "schema_version": "1.2.0",
            "model_input_contract": contract,
        }
    }
    assert learning_input_fingerprint(case) == (
        definition.learning_input_fingerprint(case)
    )
def test_definition_errors_and_generic_legacy_fallback_remain_distinct(
    case_data,
):
    malformed_uav = deepcopy(case_data)
    malformed_uav.pop("encounter")
    with pytest.raises(ValueError, match="case is missing encounter"):
        UAV_MODEL_INPUT_DEFINITION.learning_input_fingerprint(malformed_uav)
    assert learning_input_fingerprint(malformed_uav) == (
        legacy_learning_input_fingerprint(malformed_uav)
    )

    malformed_loitering = load_json(
        ROOT / "raw_data" / "case.loitering_impact.v1.json"
    )
    malformed_loitering["munition"]["profile_id"] = "unknown"
    with pytest.raises(
        ValueError,
        match="must be one of the three versioned loitering profiles",
    ):
        LOITERING_IMPACT_MODEL_INPUT_DEFINITION.learning_input_fingerprint(
            malformed_loitering
        )
    with pytest.raises(
        ValueError,
        match="must be one of the three versioned loitering profiles",
    ):
        loitering_impact_learning_input_fingerprint(malformed_loitering)
    assert learning_input_fingerprint(malformed_loitering) == (
        legacy_learning_input_fingerprint(malformed_loitering)
    )


def test_malformed_declared_contract_keeps_legacy_identity(case_data):
    case = deepcopy(case_data)
    case.setdefault("metadata", {})["sampling"] = {
        "sampling_spec_document": {
            "schema_version": "1.2.0",
            "model_input_contract": ["not", "hashable"],
        }
    }
    assert learning_input_fingerprint(case) == (
        legacy_learning_input_fingerprint(case)
    )
