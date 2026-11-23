from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import pytest

from damage_engine.engine import DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.learning_input_identity import (
    LOITERING_IMPACT_CONTACT_POSE_STATE_CONTRACT_VERSION,
    LOITERING_IMPACT_FIRST_ARMOR_STATE_CONTRACT_VERSION,
    LOITERING_IMPACT_FIRST_ARMOR_STATE_FEATURE_NAMES,
    LOITERING_IMPACT_FIRST_ARMOR_STATE_FEATURE_UNITS,
    LOITERING_IMPACT_FIRST_ARMOR_STATE_MODEL_INPUT_DEFINITION,
    LOITERING_IMPACT_STATE_CONTRACT_VERSION,
    loitering_impact_contact_pose_learning_input_fingerprint,
    loitering_impact_first_armor_state_learning_input_fingerprint,
    loitering_impact_first_armor_state_value,
    loitering_impact_first_armor_state_vector,
    loitering_impact_learning_input_fingerprint,
    model_input_definition_for_tensor_field,
    model_input_definition_for_version,
)
from damage_engine.registry import PhysicsRegistry
from damage_engine.training_view import (
    model_input_training_transform_for_version,
    transform_loitering_impact_first_armor_state,
)
from damage_engine.utils import load_json


ROOT = Path(__file__).resolve().parents[1]


def _engine() -> DamageEngine:
    return DamageEngine(
        load_json(ROOT / "configs" / "targets" / "target.rebuilt.json"),
        PhysicsRegistry.from_file(
            ROOT / "configs" / "physics" / "loitering_impact.json"
        ),
    )


def _prepared_case(*, roll_rad: float = 0.0) -> dict:
    case = load_json(ROOT / "configs" / "cases" / "loitering_impact.v1.json")
    case["encounter"]["body_to_target_orientation_xyzw"] = [
        math.sin(roll_rad / 2.0),
        0.0,
        0.0,
        math.cos(roll_rad / 2.0),
    ]
    return _engine().prepare_case(case)


def test_prepared_first_armor_geometry_maps_to_the_frozen_16d_state() -> None:
    prepared = _prepared_case()

    assert loitering_impact_first_armor_state_vector(prepared) == pytest.approx(
        [
            1.0,
            0.0,
            0.0,
            -1.6302649974822998,
            0.0,
            1.2,
            -1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            120.0,
            0.0,
            1.0,
            0.0,
        ]
    )
    value = loitering_impact_first_armor_state_value(prepared)
    assert tuple(value) == LOITERING_IMPACT_FIRST_ARMOR_STATE_FEATURE_NAMES
    assert len(value) == 16


def test_first_armor_state_encodes_roll_with_body_y_and_is_quaternion_sign_invariant() -> None:
    no_roll = _prepared_case()
    quarter_turn = _prepared_case(roll_rad=math.pi / 2.0)
    opposite_quaternion = deepcopy(quarter_turn)
    opposite_quaternion["encounter"]["body_to_target_orientation_xyzw"] = [
        -component
        for component in quarter_turn["encounter"][
            "body_to_target_orientation_xyzw"
        ]
    ]

    no_roll_vector = loitering_impact_first_armor_state_vector(no_roll)
    quarter_turn_vector = loitering_impact_first_armor_state_vector(quarter_turn)
    assert quarter_turn_vector[:13] == pytest.approx(no_roll_vector[:13])
    assert no_roll_vector[13:] == pytest.approx([0.0, 1.0, 0.0])
    assert quarter_turn_vector[13:] == pytest.approx([0.0, 0.0, 1.0])
    assert loitering_impact_first_armor_state_vector(
        opposite_quaternion
    ) == pytest.approx(quarter_turn_vector)
    assert loitering_impact_first_armor_state_learning_input_fingerprint(
        opposite_quaternion
    ) == loitering_impact_first_armor_state_learning_input_fingerprint(
        quarter_turn
    )


def test_first_armor_state_requires_the_post_intersection_deployment_stage() -> None:
    raw_launch_case = load_json(
        ROOT / "configs" / "cases" / "loitering_impact.v1.json"
    )
    raw_launch_case["encounter"].pop("impact_model_id", None)
    with pytest.raises(ValueError, match="requires prepared"):
        loitering_impact_first_armor_state_vector(raw_launch_case)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda case: case["munition"].__setitem__(
                "fuze_type", "timed"
            ),
            "munition.fuze_type",
        ),
        (
            lambda case: case["encounter"].pop(
                "collision_normal_target_xyz"
            ),
            "collision_normal_target_xyz",
        ),
        (
            lambda case: case["encounter"]["detonation_position_m"].__setitem__(
                0, float("nan")
            ),
            "detonation_position_m",
        ),
        (
            lambda case: case["encounter"].__setitem__(
                "velocity_target_mps", [0.0, 0.0, 0.0]
            ),
            "velocity_target_mps must be nonzero",
        ),
        (
            lambda case: case["encounter"].__setitem__(
                "body_to_target_orientation_xyzw", [0.0, 0.0, 0.0, 0.0]
            ),
            "body_to_target_orientation_xyzw must be nonzero",
        ),
        (
            lambda case: case["encounter"]["detonation_position_m"].__setitem__(
                0, case["encounter"]["detonation_position_m"][0] + 0.1
            ),
            "first armor hit point",
        ),
    ),
)
def test_first_armor_state_fails_closed_on_missing_nonfinite_or_inconsistent_geometry(
    mutation,
    message,
) -> None:
    case = _prepared_case()
    mutation(case)
    with pytest.raises(ValueError, match=message):
        loitering_impact_first_armor_state_vector(case)


def test_first_armor_state_contract_is_independent_and_registered() -> None:
    definition = LOITERING_IMPACT_FIRST_ARMOR_STATE_MODEL_INPUT_DEFINITION
    assert definition.version == LOITERING_IMPACT_FIRST_ARMOR_STATE_CONTRACT_VERSION
    assert len(definition.feature_names) == 16
    assert definition.feature_names == LOITERING_IMPACT_FIRST_ARMOR_STATE_FEATURE_NAMES
    assert definition.feature_units == LOITERING_IMPACT_FIRST_ARMOR_STATE_FEATURE_UNITS
    assert "deployment" in definition.evidence_boundary
    assert "ray" in definition.evidence_boundary
    assert model_input_definition_for_version(definition.version) is definition
    assert (
        model_input_definition_for_tensor_field(definition.tensor_field)
        is definition
    )


def test_first_armor_training_transform_is_strict_identity_16d() -> None:
    vector = np.asarray(
        loitering_impact_first_armor_state_vector(_prepared_case()),
        dtype=np.float64,
    )
    transform = model_input_training_transform_for_version(
        LOITERING_IMPACT_FIRST_ARMOR_STATE_CONTRACT_VERSION
    )
    assert transform is transform_loitering_impact_first_armor_state

    values, mask, names, units, transforms = transform(
        vector,
        np.ones(16, dtype=bool),
        LOITERING_IMPACT_FIRST_ARMOR_STATE_FEATURE_NAMES,
        LOITERING_IMPACT_FIRST_ARMOR_STATE_FEATURE_UNITS,
    )
    np.testing.assert_array_equal(values, vector.astype(np.float32))
    assert mask.tolist() == [True] * 16
    assert names == list(LOITERING_IMPACT_FIRST_ARMOR_STATE_FEATURE_NAMES)
    assert units == list(LOITERING_IMPACT_FIRST_ARMOR_STATE_FEATURE_UNITS)
    assert all(item["kind"] == "identity" for item in transforms)

    missing = np.ones(16, dtype=bool)
    missing[4] = False
    with pytest.raises(InputValidationError, match="requires every feature"):
        transform(
            vector,
            missing,
            LOITERING_IMPACT_FIRST_ARMOR_STATE_FEATURE_NAMES,
            LOITERING_IMPACT_FIRST_ARMOR_STATE_FEATURE_UNITS,
        )
    nonfinite = vector.copy()
    nonfinite[12] = float("inf")
    with pytest.raises(InputValidationError, match="finite"):
        transform(
            nonfinite,
            np.ones(16, dtype=bool),
            LOITERING_IMPACT_FIRST_ARMOR_STATE_FEATURE_NAMES,
            LOITERING_IMPACT_FIRST_ARMOR_STATE_FEATURE_UNITS,
        )


def test_existing_preimpact_contracts_keep_their_versions_and_hashes() -> None:
    source = load_json(ROOT / "configs" / "cases" / "loitering_impact.v1.json")
    assert LOITERING_IMPACT_STATE_CONTRACT_VERSION == (
        "loitering-impact-state-input@1.0.0"
    )
    assert LOITERING_IMPACT_CONTACT_POSE_STATE_CONTRACT_VERSION == (
        "loitering-impact-contact-pose-input@1.0.0"
    )
    assert loitering_impact_learning_input_fingerprint(source) == (
        "sha256:0660aa7fd01c6ee83f5355ee6cafd788120cf98af97c74e209dad4c357fc5641"
    )
    assert loitering_impact_contact_pose_learning_input_fingerprint(source) == (
        "sha256:e8a0d79981767b919c7f7d26bc60f86fe17a4a30c9284986f5f32638ec54c994"
    )
