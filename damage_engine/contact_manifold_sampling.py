"""Versioned target-facing contact-manifold construction.

The sampler owns the coupled geometry that independent JSON-pointer variables
cannot express safely: an approach direction, a ray origin on its transverse
aim plane, a positive speed, and a body attitude whose +X axis is exactly the
velocity direction.  The remaining attitude degree of freedom is a roll about
body +X.

This module only constructs pre-outcome inputs.  It does not decide whether a
ray is a valid first-armour contact; :mod:`damage_engine.contact_contract`
remains the fail-closed admission authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from typing import Any

import numpy as np

from .errors import InputValidationError
from .utils import content_hash


CONTACT_MANIFOLD_SCHEMA_VERSION = "1.0.0"
CONTACT_MANIFOLD_SAMPLER_ID = "target-facing-ray-manifold@1.0.0"
CONTACT_MANIFOLD_MODE = "target_facing_ray_with_body_x_roll"
CONTACT_MANIFOLD_APPROACH_WEIGHT_SEMANTICS = (
    "declared_design_mass_not_hit_probability_or_operational_prevalence"
)
CONTACT_MANIFOLD_LATENT_NAMES = (
    "approach_selector",
    "aim_u_quantile",
    "aim_v_quantile",
    "standoff_quantile",
    "speed_quantile",
    "body_x_roll_quantile",
)
CONTACT_MANIFOLD_LATENT_DIMENSION = len(CONTACT_MANIFOLD_LATENT_NAMES)

_VECTOR_TOLERANCE = 1.0e-10
_ORTHOGONALITY_TOLERANCE = 1.0e-10
_RIGHT_HANDED_TOLERANCE = 1.0e-10

CONTACT_MANIFOLD_IMPLEMENTATION_DOCUMENT = {
    "schema_version": CONTACT_MANIFOLD_SCHEMA_VERSION,
    "sampler_id": CONTACT_MANIFOLD_SAMPLER_ID,
    "mode": CONTACT_MANIFOLD_MODE,
    "approach_weight_semantics": CONTACT_MANIFOLD_APPROACH_WEIGHT_SEMANTICS,
    "latent_names": list(CONTACT_MANIFOLD_LATENT_NAMES),
    "latent_domain": "open_unit_hypercube",
    "construction": {
        "aim_point": "aim_center + aim_u*tangent_u + aim_v*tangent_v",
        "trajectory_origin": "aim_point - standoff*approach_direction",
        "velocity": "speed*approach_direction",
        "body_x": "approach_direction",
        "body_y": "cos(roll)*tangent_u + sin(roll)*tangent_v",
        "body_z": "-sin(roll)*tangent_u + cos(roll)*tangent_v",
        "orientation": "right_handed_body_to_target_rotation_as_xyzw",
    },
    "numerical_tolerances": {
        "unit_vector_absolute": _VECTOR_TOLERANCE,
        "orthogonality_absolute": _ORTHOGONALITY_TOLERANCE,
        "right_handed_absolute": _RIGHT_HANDED_TOLERANCE,
    },
    "admission_authority": "s-contact-first-armor@1.0.0",
    "evidence_boundary": (
        "The construction is an outcome-blind design distribution. First-armour "
        "contact must still be recomputed and admitted by S-CONTACT; approach "
        "weights are not operational prevalence claims."
    ),
}
CONTACT_MANIFOLD_IMPLEMENTATION_HASH = content_hash(
    CONTACT_MANIFOLD_IMPLEMENTATION_DOCUMENT
)


def _fail(message: str, **context: Any) -> None:
    raise InputValidationError(
        message,
        context={
            "sampler_id": CONTACT_MANIFOLD_SAMPLER_ID,
            **context,
        },
    )


def _finite_vector3(value: Any, *, field: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        _fail(f"{field} must be a finite 3-vector", field=field)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        _fail(f"{field} must be a finite 3-vector", field=field)
    return vector


def _finite_range(
    value: Any,
    *,
    field: str,
    positive_lower: bool = False,
    maximum_width: float | None = None,
) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        _fail(f"{field} must contain two finite increasing values", field=field)
    try:
        low, high = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, OverflowError):
        _fail(f"{field} must contain two finite increasing values", field=field)
    if not math.isfinite(low) or not math.isfinite(high) or not low < high:
        _fail(f"{field} must contain two finite increasing values", field=field)
    if positive_lower and low <= 0.0:
        _fail(f"{field} lower bound must be positive", field=field)
    if maximum_width is not None and high - low > maximum_width + 1.0e-12:
        _fail(
            f"{field} width exceeds the supported periodic interval",
            field=field,
            observed_width=high - low,
            maximum_width=maximum_width,
        )
    return low, high


def validate_contact_manifold(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical, semantically validated sampler declaration.

    JSON Schema owns the structural check.  This function checks the geometry
    that JSON Schema cannot express: unit, mutually orthogonal, right-handed
    frames; positive weights/speeds/standoffs; and unique approach identities.
    """

    if not isinstance(document, Mapping):
        _fail("contact_manifold must be an object")
    try:
        canonical = json.loads(
            json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError("contact_manifold must be finite JSON") from exc
    identity = {
        "schema_version": canonical.get("schema_version"),
        "sampler_id": canonical.get("sampler_id"),
        "mode": canonical.get("mode"),
        "approach_weight_semantics": canonical.get(
            "approach_weight_semantics"
        ),
    }
    expected = {
        "schema_version": CONTACT_MANIFOLD_SCHEMA_VERSION,
        "sampler_id": CONTACT_MANIFOLD_SAMPLER_ID,
        "mode": CONTACT_MANIFOLD_MODE,
        "approach_weight_semantics": (
            CONTACT_MANIFOLD_APPROACH_WEIGHT_SEMANTICS
        ),
    }
    if identity != expected:
        _fail(
            "contact-manifold sampler identity is unsupported",
            expected=expected,
            observed=identity,
        )
    declared_hash = canonical.get("implementation_hash")
    if declared_hash not in {None, CONTACT_MANIFOLD_IMPLEMENTATION_HASH}:
        _fail(
            "contact-manifold implementation hash is inconsistent",
            expected=CONTACT_MANIFOLD_IMPLEMENTATION_HASH,
            observed=declared_hash,
        )

    approaches = canonical.get("approaches")
    if not isinstance(approaches, list) or len(approaches) < 2:
        _fail("contact_manifold requires at least two approach frames")
    identifiers: set[str] = set()
    weight_total = 0.0
    for index, approach in enumerate(approaches):
        if not isinstance(approach, dict):
            _fail("each contact-manifold approach must be an object", index=index)
        approach_id = approach.get("approach_id")
        if not isinstance(approach_id, str) or not approach_id:
            _fail("approach_id must be a nonempty string", index=index)
        if approach_id in identifiers:
            _fail("contact-manifold approach IDs must be unique", approach_id=approach_id)
        identifiers.add(approach_id)
        try:
            weight = float(approach.get("selection_weight"))
        except (TypeError, ValueError, OverflowError):
            _fail("approach selection_weight must be finite and positive", approach_id=approach_id)
        if not math.isfinite(weight) or weight <= 0.0:
            _fail("approach selection_weight must be finite and positive", approach_id=approach_id)
        weight_total += weight

        direction = _finite_vector3(
            approach.get("direction_target_xyz"),
            field=f"approaches[{index}].direction_target_xyz",
        )
        tangent_u = _finite_vector3(
            approach.get("tangent_u_target_xyz"),
            field=f"approaches[{index}].tangent_u_target_xyz",
        )
        tangent_v = _finite_vector3(
            approach.get("tangent_v_target_xyz"),
            field=f"approaches[{index}].tangent_v_target_xyz",
        )
        for name, vector in (
            ("direction_target_xyz", direction),
            ("tangent_u_target_xyz", tangent_u),
            ("tangent_v_target_xyz", tangent_v),
        ):
            norm = float(np.linalg.norm(vector))
            if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_VECTOR_TOLERANCE):
                _fail(
                    "contact-manifold frame vectors must have unit norm",
                    approach_id=approach_id,
                    vector=name,
                    norm=norm,
                )
        dot_products = {
            "direction_dot_u": float(np.dot(direction, tangent_u)),
            "direction_dot_v": float(np.dot(direction, tangent_v)),
            "u_dot_v": float(np.dot(tangent_u, tangent_v)),
        }
        if any(abs(value) > _ORTHOGONALITY_TOLERANCE for value in dot_products.values()):
            _fail(
                "contact-manifold frame vectors must be mutually orthogonal",
                approach_id=approach_id,
                **dot_products,
            )
        handedness_error = float(
            np.linalg.norm(np.cross(direction, tangent_u) - tangent_v)
        )
        if handedness_error > _RIGHT_HANDED_TOLERANCE:
            _fail(
                "contact-manifold frame must be right handed",
                approach_id=approach_id,
                handedness_error=handedness_error,
            )
        _finite_vector3(
            approach.get("aim_center_target_m"),
            field=f"approaches[{index}].aim_center_target_m",
        )
        _finite_range(
            approach.get("aim_u_range_m"),
            field=f"approaches[{index}].aim_u_range_m",
        )
        _finite_range(
            approach.get("aim_v_range_m"),
            field=f"approaches[{index}].aim_v_range_m",
        )
        _finite_range(
            approach.get("standoff_range_m"),
            field=f"approaches[{index}].standoff_range_m",
            positive_lower=True,
        )
        _finite_range(
            approach.get("speed_range_mps"),
            field=f"approaches[{index}].speed_range_mps",
            positive_lower=True,
        )
        _finite_range(
            approach.get("body_x_roll_range_rad"),
            field=f"approaches[{index}].body_x_roll_range_rad",
            maximum_width=2.0 * math.pi,
        )
    if not math.isfinite(weight_total) or weight_total <= 0.0:
        _fail("contact-manifold approach weights must have a positive finite sum")
    canonical["implementation_hash"] = CONTACT_MANIFOLD_IMPLEMENTATION_HASH
    return canonical


def _linear(range_value: Sequence[float], quantile: float) -> float:
    low, high = float(range_value[0]), float(range_value[1])
    return low + quantile * (high - low)


def _matrix_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    """Convert a proper 3x3 rotation to a canonical unit xyzw quaternion."""

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(max(0.0, trace + 1.0))
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = 2.0 * math.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]))
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
            w = (rotation[2, 1] - rotation[1, 2]) / scale
        elif axis == 1:
            scale = 2.0 * math.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]))
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
            w = (rotation[0, 2] - rotation[2, 0]) / scale
        else:
            scale = 2.0 * math.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]))
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
            w = (rotation[1, 0] - rotation[0, 1]) / scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 0.0:
        _fail("constructed body-to-target rotation has no finite quaternion")
    quaternion /= norm
    # Canonical sign is for byte-stable identities only; the learner later uses
    # the q/-q-invariant rotation-6D representation.
    for component in reversed(quaternion.tolist()):
        if abs(component) > 1.0e-15:
            if component < 0.0:
                quaternion *= -1.0
            break
    return [float(value) for value in quaternion]


def apply_contact_manifold(
    case: dict[str, Any],
    document: Mapping[str, Any],
    latent: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Jointly write target-facing origin, velocity and attitude into ``case``.

    The returned record is pre-outcome construction evidence.  A caller must
    still invoke S-CONTACT before admitting a formal labelled case.
    """

    canonical = validate_contact_manifold(document)
    coordinates = np.asarray(latent, dtype=np.float64)
    if (
        coordinates.shape != (CONTACT_MANIFOLD_LATENT_DIMENSION,)
        or not np.isfinite(coordinates).all()
        or np.any(coordinates <= 0.0)
        or np.any(coordinates >= 1.0)
    ):
        _fail(
            "contact-manifold latent must lie in the open unit hypercube",
            expected_dimension=CONTACT_MANIFOLD_LATENT_DIMENSION,
        )
    encounter = case.get("encounter")
    if not isinstance(encounter, dict):
        _fail("contact-manifold construction requires an encounter object")

    approaches = canonical["approaches"]
    weights = np.asarray(
        [float(item["selection_weight"]) for item in approaches], dtype=np.float64
    )
    cumulative = np.cumsum(weights) / float(np.sum(weights))
    approach_index = min(
        int(np.searchsorted(cumulative, coordinates[0], side="right")),
        len(approaches) - 1,
    )
    approach = approaches[approach_index]
    direction = np.asarray(approach["direction_target_xyz"], dtype=np.float64)
    tangent_u = np.asarray(approach["tangent_u_target_xyz"], dtype=np.float64)
    tangent_v = np.asarray(approach["tangent_v_target_xyz"], dtype=np.float64)
    center = np.asarray(approach["aim_center_target_m"], dtype=np.float64)
    aim_u = _linear(approach["aim_u_range_m"], float(coordinates[1]))
    aim_v = _linear(approach["aim_v_range_m"], float(coordinates[2]))
    standoff = _linear(approach["standoff_range_m"], float(coordinates[3]))
    speed = _linear(approach["speed_range_mps"], float(coordinates[4]))
    roll = _linear(approach["body_x_roll_range_rad"], float(coordinates[5]))

    aim_point = center + aim_u * tangent_u + aim_v * tangent_v
    origin = aim_point - standoff * direction
    velocity = speed * direction
    cosine, sine = math.cos(roll), math.sin(roll)
    body_y = cosine * tangent_u + sine * tangent_v
    body_z = -sine * tangent_u + cosine * tangent_v
    rotation = np.column_stack((direction, body_y, body_z))
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-10):
        _fail("constructed body-to-target rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1.0e-10):
        _fail("constructed body-to-target rotation is not proper")
    quaternion = _matrix_to_quaternion_xyzw(rotation)

    encounter["trajectory_origin_m"] = [float(value) for value in origin]
    encounter["velocity_target_mps"] = [float(value) for value in velocity]
    encounter["body_to_target_orientation_xyzw"] = quaternion
    record = {
        "schema_version": CONTACT_MANIFOLD_SCHEMA_VERSION,
        "sampler_id": CONTACT_MANIFOLD_SAMPLER_ID,
        "implementation_hash": CONTACT_MANIFOLD_IMPLEMENTATION_HASH,
        "mode": CONTACT_MANIFOLD_MODE,
        "status": "constructed_pending_s_contact",
        "approach_id": approach["approach_id"],
        "approach_index": approach_index,
        "approach_design_mass": float(
            weights[approach_index] / np.sum(weights)
        ),
        "approach_weight_semantics": CONTACT_MANIFOLD_APPROACH_WEIGHT_SEMANTICS,
        "aim_u_m": aim_u,
        "aim_v_m": aim_v,
        "aim_point_target_m": [float(value) for value in aim_point],
        "standoff_m": standoff,
        "speed_mps": speed,
        "body_x_roll_rad": roll,
        "latent_names": list(CONTACT_MANIFOLD_LATENT_NAMES),
        "latent": [float(value) for value in coordinates],
        "evidence_boundary": (
            "Outcome-blind construction record only; formal label admission "
            "requires a separate successful S-CONTACT audit."
        ),
    }
    case.setdefault("metadata", {})["contact_manifold_sampling"] = record
    return record


__all__ = [
    "CONTACT_MANIFOLD_APPROACH_WEIGHT_SEMANTICS",
    "CONTACT_MANIFOLD_IMPLEMENTATION_DOCUMENT",
    "CONTACT_MANIFOLD_IMPLEMENTATION_HASH",
    "CONTACT_MANIFOLD_LATENT_DIMENSION",
    "CONTACT_MANIFOLD_LATENT_NAMES",
    "CONTACT_MANIFOLD_MODE",
    "CONTACT_MANIFOLD_SAMPLER_ID",
    "CONTACT_MANIFOLD_SCHEMA_VERSION",
    "apply_contact_manifold",
    "validate_contact_manifold",
]
