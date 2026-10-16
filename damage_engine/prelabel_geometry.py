"""Outcome-blind geometry proxies used only for sampling design.

These features are derived from frozen inputs and the fixed physics registry
before any simulator label is evaluated.  They may define conditional training
strata, but are not evidence of damage and must never enter the frozen
reference validation/test selection rule.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .errors import InputValidationError
from .utils import EPS, quaternion_matrix_xyzw


PRELABEL_GEOMETRY_VERSION = "prelabel-geometry@1.0.0"
PRELABEL_GEOMETRY_FIELDS = frozenset(
    {
        "uav_threat_range_m",
        "fragment_cone_axis_to_uav_cosine",
        "fragment_cone_axis_to_uav_angle_deg",
        "inside_nominal_fragment_cone",
        "uav_radial_velocity_from_fixed_threat_mps",
    }
)


def _finite_three(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise InputValidationError(f"{name} must be a finite three-vector")
    return array


def prelabel_geometry_features(case: dict, physics_registry) -> dict[str, Any]:
    """Compute fixed-threat/UAV geometry without executing any damage model."""

    try:
        encounter = case["encounter"]
        source = case["warhead"]["fragment_source"]
        source_model = physics_registry.fragment_source(source["model_id"])
        direction = source_model["direction"]
    except (KeyError, TypeError) as exc:
        raise InputValidationError(
            "pre-label geometry requires encounter and fragment-source definitions"
        ) from exc

    threat_in_uav_frame = _finite_three(
        encounter.get("detonation_position_m"), "encounter.detonation_position_m"
    )
    distance = float(np.linalg.norm(threat_in_uav_frame))
    if distance <= EPS:
        raise InputValidationError(
            "pre-label geometry is undefined at zero threat/UAV separation"
        )
    threat_to_uav = -threat_in_uav_frame / distance

    axis_body = _finite_three(
        direction.get("axis_body", [1.0, 0.0, 0.0]),
        "fragment direction axis_body",
    )
    axis_norm = float(np.linalg.norm(axis_body))
    if axis_norm <= EPS:
        raise InputValidationError("fragment direction axis_body must be nonzero")
    axis_body /= axis_norm
    rotation = quaternion_matrix_xyzw(
        encounter.get("body_to_target_orientation_xyzw")
    )
    axis_uav_frame = rotation @ axis_body
    cosine = float(np.clip(np.dot(axis_uav_frame, threat_to_uav), -1.0, 1.0))
    angle_deg = float(math.degrees(math.acos(cosine)))

    half_angle = direction.get("half_angle_deg")
    inside_cone = None
    if direction.get("kind") == "cone":
        if not isinstance(half_angle, (int, float)) or not math.isfinite(float(half_angle)):
            raise InputValidationError("cone direction requires finite half_angle_deg")
        inside_cone = angle_deg <= float(half_angle)

    uav_velocity = -_finite_three(
        encounter.get("velocity_target_mps"), "encounter.velocity_target_mps"
    )
    uav_position = -threat_in_uav_frame
    radial_velocity = float(np.dot(uav_velocity, uav_position / distance))

    return {
        "schema_version": PRELABEL_GEOMETRY_VERSION,
        "uav_threat_range_m": distance,
        "fragment_cone_axis_to_uav_cosine": cosine,
        "fragment_cone_axis_to_uav_angle_deg": angle_deg,
        "inside_nominal_fragment_cone": inside_cone,
        "uav_radial_velocity_from_fixed_threat_mps": radial_velocity,
        "source": "frozen_case_and_physics_registry_before_labels",
    }
