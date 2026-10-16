"""Continuous first-contact impact fuze resolution for armored targets."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .errors import InputValidationError, OutOfDomainError
from .geometry import CompiledTarget
from .utils import EPS, content_hash, quaternion_matrix_xyzw


IMPACT_TRIGGER = "first_armor_contact"
IMPACT_MODEL_ID = "continuous-ray-first-armor-v1"
IMPACT_POSITION_TOLERANCE_M = 1.0e-8
IMPACT_TIME_TOLERANCE_S = 1.0e-10
IMPACT_ALIGNMENT_COSINE_MINIMUM = 1.0 - 1.0e-6
IMPACT_VALIDATION_DOCUMENT = {
    "impact_model_id": IMPACT_MODEL_ID,
    "trigger": IMPACT_TRIGGER,
    "trajectory_reference_point": "nose_fuze",
    "body_forward_axis": "+X",
    "position_tolerance_m": IMPACT_POSITION_TOLERANCE_M,
    "time_tolerance_s": IMPACT_TIME_TOLERANCE_S,
    "alignment_cosine_minimum": IMPACT_ALIGNMENT_COSINE_MINIMUM,
}
IMPACT_VALIDATION_HASH = content_hash(IMPACT_VALIDATION_DOCUMENT)


def is_impact_case(case: dict) -> bool:
    return (case.get("encounter") or {}).get("detonation_trigger") == IMPACT_TRIGGER


def _impact_solution(case: dict, target: CompiledTarget) -> dict:
    encounter = case.get("encounter") or {}
    munition = case.get("munition")
    if not isinstance(munition, dict):
        raise InputValidationError(
            "first_armor_contact requires a versioned munition profile"
        )
    if munition.get("fuze_type") != "impact_contact":
        raise InputValidationError("impact detonation requires an impact_contact fuze")
    if munition.get("trajectory_reference_point") != "nose_fuze":
        raise InputValidationError(
            "impact trajectory origin must reference the munition nose fuze"
        )
    try:
        origin = np.asarray(encounter["trajectory_origin_m"], dtype=np.float64)
        velocity = np.asarray(encounter["velocity_target_mps"], dtype=np.float64)
    except KeyError as exc:
        raise InputValidationError(
            "first_armor_contact requires trajectory_origin_m and velocity_target_mps"
        ) from exc
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise InputValidationError("trajectory_origin_m must be a finite 3-vector")
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise InputValidationError("velocity_target_mps must be a finite 3-vector")
    speed = float(np.linalg.norm(velocity))
    if speed <= EPS:
        raise InputValidationError("impact trajectory velocity must be non-zero")
    direction = velocity / speed

    containing_armor = [
        item.object_id for item in target.armor if item.geometry.contains(origin)
    ]
    if containing_armor:
        raise OutOfDomainError(
            "munition trajectory origin must be outside target armor",
            context={"containing_armor_ids": containing_armor},
        )

    rotation = quaternion_matrix_xyzw(
        encounter["body_to_target_orientation_xyzw"]
    )
    body_forward = rotation @ np.asarray([1.0, 0.0, 0.0])
    alignment = float(np.dot(body_forward, direction))
    if alignment < IMPACT_ALIGNMENT_COSINE_MINIMUM:
        raise InputValidationError(
            "munition body +X axis must align with the impact velocity",
            context={"alignment_cosine": alignment},
        )

    armor_hits = [hit for hit in target.ray_hits(origin, direction) if hit.role == "armor"]
    if not armor_hits:
        raise OutOfDomainError(
            "munition trajectory does not collide with target armor",
            context={
                "trajectory_origin_m": origin.tolist(),
                "velocity_target_mps": velocity.tolist(),
            },
        )
    hit = min(armor_hits, key=lambda item: (item.entry_t, item.object_id))
    if hit.entry_t < -IMPACT_POSITION_TOLERANCE_M:
        raise OutOfDomainError("first armor collision lies behind the trajectory origin")
    distance = max(0.0, float(hit.entry_t))
    return {
        "detonation_position_m": [float(value) for value in hit.entry_point],
        "detonation_time_s": distance / speed,
        "detonation_delay_s": 0.0,
        "collision_distance_m": distance,
        "collision_armor_id": hit.object_id,
        "collision_normal_target_xyz": [float(value) for value in hit.entry_normal],
        "impact_model_id": IMPACT_MODEL_ID,
        "trajectory_reference_point": "nose_fuze",
    }


def resolve_impact_detonation(case: dict, target: CompiledTarget) -> dict:
    """Return a copy with impact position/time derived from continuous ray contact."""

    if not is_impact_case(case):
        return case
    resolved = deepcopy(case)
    resolved["encounter"].update(_impact_solution(resolved, target))
    return resolved


def validate_impact_detonation(case: dict, target: CompiledTarget) -> None:
    """Verify that persisted derived collision fields equal a fresh solution."""

    if not is_impact_case(case):
        return
    expected = _impact_solution(case, target)
    encounter = case["encounter"]
    missing = sorted(set(expected) - set(encounter))
    if missing:
        raise InputValidationError(
            "impact case is missing derived collision fields",
            context={"missing": missing},
        )
    if encounter["collision_armor_id"] != expected["collision_armor_id"]:
        raise InputValidationError("collision_armor_id is inconsistent with the trajectory")
    if encounter["impact_model_id"] != IMPACT_MODEL_ID:
        raise InputValidationError("impact_model_id is unsupported")
    observed_position = np.asarray(encounter["detonation_position_m"], dtype=np.float64)
    expected_position = np.asarray(expected["detonation_position_m"], dtype=np.float64)
    if not np.allclose(
        observed_position,
        expected_position,
        rtol=0.0,
        atol=IMPACT_POSITION_TOLERANCE_M,
    ):
        raise InputValidationError(
            "detonation_position_m is not the first armor-contact point"
        )
    scalar_fields = {
        "detonation_time_s": IMPACT_TIME_TOLERANCE_S,
        "detonation_delay_s": IMPACT_TIME_TOLERANCE_S,
        "collision_distance_m": IMPACT_POSITION_TOLERANCE_M,
    }
    for field, tolerance in scalar_fields.items():
        if not math.isclose(
            float(encounter[field]),
            float(expected[field]),
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise InputValidationError(
                f"{field} is inconsistent with first armor contact"
            )
    observed_normal = np.asarray(
        encounter["collision_normal_target_xyz"], dtype=np.float64
    )
    expected_normal = np.asarray(
        expected["collision_normal_target_xyz"], dtype=np.float64
    )
    if not np.allclose(
        observed_normal,
        expected_normal,
        rtol=0.0,
        atol=IMPACT_POSITION_TOLERANCE_M,
    ):
        raise InputValidationError(
            "collision_normal_target_xyz is inconsistent with first armor contact"
        )
