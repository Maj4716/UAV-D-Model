"""Longitudinally canonical sphere-entry rays for contact-conditioned sampling.

This module changes only the point chosen on an already-declared directed ray.
It does not change the transverse ray measure and it does not admit labels;
S-CONTACT remains the admission authority after impact fields are recomputed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .errors import InputValidationError, OutOfDomainError
from .geometry import CompiledTarget, EnclosingSphere
from .utils import EPS, content_hash, normalize


SPHERE_ENTRY_RAY_SCHEMA_VERSION = "1.0.0"
SPHERE_ENTRY_RAY_SAMPLER_ID = "sphere-entry-ray-manifold@1.0.0"
SPHERE_ENTRY_RAY_MODE = "sphere_entry_ray_with_body_x_roll"
PHYSICAL_RAY_IDENTITY_VERSION = "physical-directed-ray@1.0.0"
PHYSICAL_RAY_IDENTITY_DECIMALS = 12

PAIRED_LEGACY_STANDOFF_6D = "paired_legacy_standoff_6d"
REDUCED_PHYSICAL_RAY_5D = "reduced_physical_ray_5d"

_PHYSICAL_LATENT_NAMES = (
    "approach_selector",
    "aim_u_quantile",
    "aim_v_quantile",
    "speed_quantile",
    "body_x_roll_quantile",
)
_PAIRED_LATENT_NAMES = (
    "approach_selector",
    "aim_u_quantile",
    "aim_v_quantile",
    "standoff_quantile",
    "speed_quantile",
    "body_x_roll_quantile",
)
_LAYOUTS = {
    PAIRED_LEGACY_STANDOFF_6D: {
        "latent_names": _PAIRED_LATENT_NAMES,
        "no_op_latent_names": ("standoff_quantile",),
        "purpose": "paired_equivalence_control_with_legacy_six_dimensional_design",
    },
    REDUCED_PHYSICAL_RAY_5D: {
        "latent_names": _PHYSICAL_LATENT_NAMES,
        "no_op_latent_names": (),
        "purpose": "production_design_without_longitudinal_gauge_coordinate",
    },
}

SPHERE_ENTRY_RAY_IMPLEMENTATION_DOCUMENT = {
    "schema_version": SPHERE_ENTRY_RAY_SCHEMA_VERSION,
    "sampler_id": SPHERE_ENTRY_RAY_SAMPLER_ID,
    "mode": SPHERE_ENTRY_RAY_MODE,
    "sphere_algorithm": "target-aabb-circumsphere@1.0.0",
    "physical_ray_identity_version": PHYSICAL_RAY_IDENTITY_VERSION,
    "physical_ray_identity_decimals": PHYSICAL_RAY_IDENTITY_DECIMALS,
    "origin_formula": "c+b-sqrt(R^2-||b||^2)*d",
    "transverse_offset_formula": "b=(I-dd^T)(p-c)",
    "latent_layouts": {
        name: {
            "latent_names": list(value["latent_names"]),
            "no_op_latent_names": list(value["no_op_latent_names"]),
            "purpose": value["purpose"],
        }
        for name, value in _LAYOUTS.items()
    },
    "admission_authority": "s-contact-first-armor@1.0.0",
}
SPHERE_ENTRY_RAY_IMPLEMENTATION_HASH = content_hash(
    SPHERE_ENTRY_RAY_IMPLEMENTATION_DOCUMENT
)

_DERIVED_IMPACT_FIELDS = (
    "detonation_position_m",
    "detonation_time_s",
    "detonation_delay_s",
    "collision_distance_m",
    "collision_armor_id",
    "collision_normal_target_xyz",
    "impact_model_id",
    "trajectory_reference_point",
)


def _finite_vector3(value: Any, *, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(f"{name} must be a finite 3-vector") from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise InputValidationError(f"{name} must be a finite 3-vector")
    return vector


def _canonical_vector(value: np.ndarray) -> list[float]:
    rounded = np.round(np.asarray(value, dtype=np.float64), PHYSICAL_RAY_IDENTITY_DECIMALS)
    threshold = 0.5 * 10.0 ** (-PHYSICAL_RAY_IDENTITY_DECIMALS)
    rounded[np.abs(rounded) < threshold] = 0.0
    return [float(item) for item in rounded]


def sphere_entry_latent_declaration(layout: str) -> dict[str, Any]:
    """Return the closed public declaration for one supported latent layout."""

    if layout not in _LAYOUTS:
        raise InputValidationError(
            "sphere-entry latent layout is unsupported",
            context={"layout": layout, "supported": sorted(_LAYOUTS)},
        )
    definition = _LAYOUTS[layout]
    return {
        "layout": layout,
        "latent_dimension": len(definition["latent_names"]),
        "latent_names": list(definition["latent_names"]),
        "physical_latent_names": list(_PHYSICAL_LATENT_NAMES),
        "no_op_latent_names": list(definition["no_op_latent_names"]),
        "latent_domain": "open_unit_hypercube",
        "purpose": definition["purpose"],
    }


@dataclass(frozen=True)
class SphereEntryLatent:
    layout: str
    raw_latent: tuple[float, ...]
    physical_latent: tuple[float, float, float, float, float]
    legacy_contact_manifold_latent: tuple[float, float, float, float, float, float]
    no_op_latent_names: tuple[str, ...]

    def as_document(self) -> dict[str, Any]:
        return {
            **sphere_entry_latent_declaration(self.layout),
            "latent": list(self.raw_latent),
            "physical_latent": list(self.physical_latent),
            "legacy_contact_manifold_latent": list(
                self.legacy_contact_manifold_latent
            ),
        }


def map_sphere_entry_latent(
    latent: Sequence[float] | np.ndarray,
    *,
    layout: str,
    legacy_standoff_quantile: float = 0.5,
) -> SphereEntryLatent:
    """Map either public layout to one physical 5D and one legacy 6D vector."""

    declaration = sphere_entry_latent_declaration(layout)
    try:
        values = np.asarray(latent, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(
            "sphere-entry latent must lie in the open unit hypercube"
        ) from exc
    if (
        values.shape != (declaration["latent_dimension"],)
        or not np.isfinite(values).all()
        or np.any(values <= 0.0)
        or np.any(values >= 1.0)
    ):
        raise InputValidationError(
            "sphere-entry latent must lie in the open unit hypercube",
            context={"expected_dimension": declaration["latent_dimension"]},
        )
    raw = tuple(float(value) for value in values)
    if layout == PAIRED_LEGACY_STANDOFF_6D:
        physical = (raw[0], raw[1], raw[2], raw[4], raw[5])
        legacy = raw
    else:
        dummy = float(legacy_standoff_quantile)
        if not math.isfinite(dummy) or not 0.0 < dummy < 1.0:
            raise InputValidationError(
                "legacy_standoff_quantile must lie in the open unit interval"
            )
        physical = (raw[0], raw[1], raw[2], raw[3], raw[4])
        legacy = (raw[0], raw[1], raw[2], dummy, raw[3], raw[4])
    return SphereEntryLatent(
        layout=layout,
        raw_latent=raw,
        physical_latent=physical,
        legacy_contact_manifold_latent=legacy,
        no_op_latent_names=tuple(declaration["no_op_latent_names"]),
    )


def physical_ray_identity(
    *,
    line_point_target_m: Sequence[float] | np.ndarray,
    direction_target_xyz: Sequence[float] | np.ndarray,
    target_hash: str,
) -> dict[str, Any]:
    """Return a target-bound, longitudinally translation-invariant ray ID."""

    if not isinstance(target_hash, str) or not target_hash:
        raise InputValidationError("physical-ray target_hash must be nonempty")
    point = _finite_vector3(line_point_target_m, name="ray line point")
    direction = normalize(direction_target_xyz, name="ray direction")
    moment = np.cross(point, direction)
    identity = {
        "identity_version": PHYSICAL_RAY_IDENTITY_VERSION,
        "identity_quantization_decimals": PHYSICAL_RAY_IDENTITY_DECIMALS,
        "target_hash": target_hash,
        "direction_target_xyz": _canonical_vector(direction),
        "moment_target_m": _canonical_vector(moment),
    }
    return {**identity, "physical_ray_id": content_hash(identity)}


@dataclass(frozen=True)
class SphereEntryRay:
    origin_target_m: tuple[float, float, float]
    direction_target_xyz: tuple[float, float, float]
    transverse_offset_target_m: tuple[float, float, float]
    transverse_radius_m: float
    sphere_entry_to_closest_approach_m: float
    physical_ray_identity: Mapping[str, Any]

    @property
    def physical_ray_id(self) -> str:
        return str(self.physical_ray_identity["physical_ray_id"])

    def as_document(self) -> dict[str, Any]:
        return {
            "origin_target_m": list(self.origin_target_m),
            "direction_target_xyz": list(self.direction_target_xyz),
            "transverse_offset_target_m": list(self.transverse_offset_target_m),
            "transverse_radius_m": self.transverse_radius_m,
            "sphere_entry_to_closest_approach_m": (
                self.sphere_entry_to_closest_approach_m
            ),
            "physical_ray_identity": dict(self.physical_ray_identity),
            "physical_ray_id": self.physical_ray_id,
        }


def reanchor_ray_to_sphere(
    *,
    line_point_target_m: Sequence[float] | np.ndarray,
    direction_target_xyz: Sequence[float] | np.ndarray,
    sphere: EnclosingSphere,
    target_hash: str,
) -> SphereEntryRay:
    """Return the upstream intersection of a directed line and a sphere."""

    if not isinstance(sphere, EnclosingSphere):
        raise InputValidationError("sphere must be a public EnclosingSphere")
    point = _finite_vector3(line_point_target_m, name="ray line point")
    direction = normalize(direction_target_xyz, name="ray direction")
    center = np.asarray(sphere.center_target_m, dtype=np.float64)
    relative = point - center
    transverse = relative - float(np.dot(relative, direction)) * direction
    transverse_squared = float(np.dot(transverse, transverse))
    radius_squared = sphere.radius_m * sphere.radius_m
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, radius_squared)
    if transverse_squared > radius_squared + tolerance:
        raise OutOfDomainError(
            "directed ray does not intersect the enclosing sphere",
            context={
                "transverse_radius_m": math.sqrt(transverse_squared),
                "sphere_radius_m": sphere.radius_m,
                "target_id": sphere.target_id,
            },
        )
    upstream_distance = math.sqrt(max(0.0, radius_squared - transverse_squared))
    origin = center + transverse - upstream_distance * direction
    identity = physical_ray_identity(
        line_point_target_m=point,
        direction_target_xyz=direction,
        target_hash=target_hash,
    )
    return SphereEntryRay(
        origin_target_m=tuple(float(value) for value in origin),
        direction_target_xyz=tuple(float(value) for value in direction),
        transverse_offset_target_m=tuple(float(value) for value in transverse),
        transverse_radius_m=math.sqrt(transverse_squared),
        sphere_entry_to_closest_approach_m=upstream_distance,
        physical_ray_identity=identity,
    )


def apply_sphere_entry_reanchor(
    case: dict[str, Any],
    target: CompiledTarget,
    *,
    target_hash: str,
    clearance_m: float,
    object_scope: str,
    latent_layout: str,
    latent: Sequence[float] | np.ndarray,
    line_point_target_m: Sequence[float] | np.ndarray | None = None,
    legacy_standoff_quantile: float = 0.5,
) -> dict[str, Any]:
    """Rewrite only a case's longitudinal ray origin and record provenance."""

    if not isinstance(case, dict):
        raise InputValidationError("sphere-entry application requires a mutable case")
    if not isinstance(target, CompiledTarget):
        raise InputValidationError("sphere-entry application requires a CompiledTarget")
    encounter = case.get("encounter")
    if not isinstance(encounter, dict):
        raise InputValidationError("sphere-entry application requires an encounter object")
    old_origin = _finite_vector3(
        encounter.get("trajectory_origin_m"), name="trajectory origin"
    )
    velocity = _finite_vector3(
        encounter.get("velocity_target_mps"), name="target velocity"
    )
    speed = float(np.linalg.norm(velocity))
    if speed <= EPS:
        raise InputValidationError("sphere-entry target velocity must be nonzero")
    mapped = map_sphere_entry_latent(
        latent,
        layout=latent_layout,
        legacy_standoff_quantile=legacy_standoff_quantile,
    )
    line_point = (
        old_origin
        if line_point_target_m is None
        else _finite_vector3(line_point_target_m, name="ray line point")
    )
    sphere = target.enclosing_sphere(
        object_scope=object_scope,  # type: ignore[arg-type]
        clearance_m=clearance_m,
    )
    ray = reanchor_ray_to_sphere(
        line_point_target_m=line_point,
        direction_target_xyz=velocity / speed,
        sphere=sphere,
        target_hash=target_hash,
    )
    encounter["trajectory_origin_m"] = list(ray.origin_target_m)
    cleared_fields = [name for name in _DERIVED_IMPACT_FIELDS if name in encounter]
    for name in cleared_fields:
        encounter.pop(name, None)

    record = {
        "schema_version": SPHERE_ENTRY_RAY_SCHEMA_VERSION,
        "sampler_id": SPHERE_ENTRY_RAY_SAMPLER_ID,
        "implementation_hash": SPHERE_ENTRY_RAY_IMPLEMENTATION_HASH,
        "mode": SPHERE_ENTRY_RAY_MODE,
        "status": "reanchored_pending_s_contact",
        "latent_layout": mapped.layout,
        "latent": list(mapped.raw_latent),
        "physical_latent": list(mapped.physical_latent),
        "legacy_contact_manifold_latent": list(
            mapped.legacy_contact_manifold_latent
        ),
        "no_op_latent_names": list(mapped.no_op_latent_names),
        "sphere": sphere.as_document(),
        "line_point_target_m": [float(value) for value in line_point],
        "original_trajectory_origin_m": [float(value) for value in old_origin],
        "trajectory_origin_m": list(ray.origin_target_m),
        "direction_target_xyz": list(ray.direction_target_xyz),
        "transverse_offset_target_m": list(ray.transverse_offset_target_m),
        "transverse_radius_m": ray.transverse_radius_m,
        "physical_ray_identity": dict(ray.physical_ray_identity),
        "physical_ray_id": ray.physical_ray_id,
        "cleared_derived_impact_fields": cleared_fields,
        "evidence_boundary": (
            "Longitudinal coordinate normalization only. S-CONTACT must recompute "
            "and admit first-armor contact before formal labelling."
        ),
    }
    metadata = case.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise InputValidationError("case metadata must be an object")
    metadata["sphere_entry_ray_sampling"] = record
    contact_record = metadata.get("contact_manifold_sampling")
    if isinstance(contact_record, dict):
        contact_record["status"] = "reanchored_pending_s_contact"
        contact_record["trajectory_origin_superseded_by"] = SPHERE_ENTRY_RAY_SAMPLER_ID
    return record


__all__ = [
    "PAIRED_LEGACY_STANDOFF_6D",
    "PHYSICAL_RAY_IDENTITY_DECIMALS",
    "PHYSICAL_RAY_IDENTITY_VERSION",
    "REDUCED_PHYSICAL_RAY_5D",
    "SPHERE_ENTRY_RAY_IMPLEMENTATION_DOCUMENT",
    "SPHERE_ENTRY_RAY_IMPLEMENTATION_HASH",
    "SPHERE_ENTRY_RAY_MODE",
    "SPHERE_ENTRY_RAY_SAMPLER_ID",
    "SPHERE_ENTRY_RAY_SCHEMA_VERSION",
    "SphereEntryLatent",
    "SphereEntryRay",
    "apply_sphere_entry_reanchor",
    "map_sphere_entry_latent",
    "physical_ray_identity",
    "reanchor_ray_to_sphere",
    "sphere_entry_latent_declaration",
]
