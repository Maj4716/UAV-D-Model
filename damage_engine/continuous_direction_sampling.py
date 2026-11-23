"""Continuous sphere-entry directions with an explicit solid-angle measure.

The sampler replaces the five fixed approach vectors used by SamplingSpec 1.6
with five continuous direction regions.  Four horizontal sectors cover a
declared azimuth centre plus/minus a half-width and a downward dive interval;
one top cap covers directions within a declared angle of vertical.  Direction
coordinates are transformed so that area is uniform on the unit sphere.

This module remains pre-outcome.  It constructs and sphere-reanchors a directed
ray, but S-CONTACT is still the authority that admits a case for labelling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from typing import Any

import numpy as np

from .contact_manifold_sampling import (
    CONTACT_MANIFOLD_APPROACH_WEIGHT_SEMANTICS,
    _matrix_to_quaternion_xyzw,
)
from .errors import InputValidationError
from .geometry import CompiledTarget
from .sphere_entry_ray_sampling import reanchor_ray_to_sphere
from .utils import content_hash


CONTINUOUS_DIRECTION_SCHEMA_VERSION = "1.0.0"
CONTINUOUS_DIRECTION_SAMPLER_ID = (
    "sphere-entry-continuous-direction-manifold@1.0.0"
)
CONTINUOUS_DIRECTION_MODE = (
    "sphere_entry_continuous_direction_with_body_x_roll"
)
CONTINUOUS_REDUCED_7D = "continuous_direction_reduced_7d"
CONTINUOUS_LEGACY_STANDOFF_8D = "continuous_direction_legacy_standoff_8d"
CONTINUOUS_DIRECTION_LATENT_NAMES = (
    "direction_region_selector",
    "direction_coordinate_1",
    "direction_coordinate_2",
    "aim_u_quantile",
    "aim_v_quantile",
    "speed_quantile",
    "body_x_roll_quantile",
)
_LEGACY_LATENT_NAMES = (
    "direction_region_selector",
    "direction_coordinate_1",
    "direction_coordinate_2",
    "aim_u_quantile",
    "aim_v_quantile",
    "standoff_quantile",
    "speed_quantile",
    "body_x_roll_quantile",
)

CONTINUOUS_DIRECTION_MEASURE_CONTRACT = {
    "schema_version": "continuous-direction-measure@1.0.0",
    "base_measure": "solid_angle_sr",
    "region_mixture": "declared_selection_weight_normalized_over_regions",
    "within_region_measure": "uniform_solid_angle",
    "horizontal_sector_transform": {
        "coordinate_1": "azimuth_uniform",
        "coordinate_2": "sin_dive_uniform",
        "solid_angle_element": "cos(dive)*d(dive)*d(azimuth)",
    },
    "top_cap_transform": {
        "coordinate_1": "azimuth_uniform",
        "coordinate_2": "cos(offset_from_downward_vertical)_uniform",
        "solid_angle_element": "sin(offset)*d(offset)*d(azimuth)",
    },
    "direction_semantics": "target_frame_velocity_unit_vector",
    "tangent_basis": (
        "project_least_aligned_positive_target_axis_then_normalize; "
        "v=cross(direction,u)"
    ),
    "bottom_up_policy": "excluded_from_reference_and_reserved_for_stress",
}
CONTINUOUS_DIRECTION_MEASURE_HASH = content_hash(
    CONTINUOUS_DIRECTION_MEASURE_CONTRACT
)
CONTINUOUS_DIRECTION_IMPLEMENTATION_DOCUMENT = {
    "schema_version": CONTINUOUS_DIRECTION_SCHEMA_VERSION,
    "sampler_id": CONTINUOUS_DIRECTION_SAMPLER_ID,
    "mode": CONTINUOUS_DIRECTION_MODE,
    "latent_layouts": {
        CONTINUOUS_REDUCED_7D: {
            "latent_names": list(CONTINUOUS_DIRECTION_LATENT_NAMES),
            "no_op_latent_names": [],
        },
        CONTINUOUS_LEGACY_STANDOFF_8D: {
            "latent_names": list(_LEGACY_LATENT_NAMES),
            "no_op_latent_names": ["standoff_quantile"],
        },
    },
    "measure_contract_hash": CONTINUOUS_DIRECTION_MEASURE_HASH,
    "sphere_algorithm": "target-aabb-circumsphere@1.0.0",
    "admission_authority": "s-contact-first-armor@1.0.0",
}
CONTINUOUS_DIRECTION_IMPLEMENTATION_HASH = content_hash(
    CONTINUOUS_DIRECTION_IMPLEMENTATION_DOCUMENT
)

# Aim-window tangent bases.  The legacy basis is kept bit-identical so that
# archived SamplingSpecs (which do not declare ``aim_tangent_basis``) keep
# their measure and implementation hashes.  Its ``v`` axis flips with the
# approach direction, so one declared ``aim_v_range_m`` lands above the target
# for some sectors and below it for the opposite ones.
LEGACY_AIM_TANGENT_BASIS = "least_aligned_target_axis_projection@1.0.0"
REFERENCE_AXIS_AIM_TANGENT_BASIS = "target_up_horizontal_target_forward_top@1.0.0"
AIM_TANGENT_BASES = (LEGACY_AIM_TANGENT_BASIS, REFERENCE_AXIS_AIM_TANGENT_BASIS)

REFERENCE_AXIS_MEASURE_CONTRACT = {
    **CONTINUOUS_DIRECTION_MEASURE_CONTRACT,
    "schema_version": "continuous-direction-measure@1.1.0",
    "aim_tangent_basis_id": REFERENCE_AXIS_AIM_TANGENT_BASIS,
    "tangent_basis": (
        "horizontal_sector: v=normalize(+z_target projected orthogonal to direction); "
        "top_cap: v=normalize(+y_target projected orthogonal to direction); "
        "u=cross(v,direction)"
    ),
}
REFERENCE_AXIS_MEASURE_HASH = content_hash(REFERENCE_AXIS_MEASURE_CONTRACT)
REFERENCE_AXIS_IMPLEMENTATION_DOCUMENT = {
    **CONTINUOUS_DIRECTION_IMPLEMENTATION_DOCUMENT,
    "aim_tangent_basis": REFERENCE_AXIS_AIM_TANGENT_BASIS,
    "measure_contract_hash": REFERENCE_AXIS_MEASURE_HASH,
}
REFERENCE_AXIS_IMPLEMENTATION_HASH = content_hash(
    REFERENCE_AXIS_IMPLEMENTATION_DOCUMENT
)

_BASIS_CONTRACTS = {
    LEGACY_AIM_TANGENT_BASIS: (
        CONTINUOUS_DIRECTION_MEASURE_CONTRACT,
        CONTINUOUS_DIRECTION_MEASURE_HASH,
        CONTINUOUS_DIRECTION_IMPLEMENTATION_HASH,
    ),
    REFERENCE_AXIS_AIM_TANGENT_BASIS: (
        REFERENCE_AXIS_MEASURE_CONTRACT,
        REFERENCE_AXIS_MEASURE_HASH,
        REFERENCE_AXIS_IMPLEMENTATION_HASH,
    ),
}

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


def _fail(message: str, **context: Any) -> None:
    raise InputValidationError(message, context=context or None)


def _finite_range(
    value: Any,
    *,
    field: str,
    positive_lower: bool = False,
    maximum_width: float | None = None,
) -> tuple[float, float]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(f"{field} must be a finite increasing pair") from exc
    if array.shape != (2,) or not np.isfinite(array).all():
        _fail(f"{field} must be a finite increasing pair")
    low, high = float(array[0]), float(array[1])
    if not low < high or (positive_lower and low <= 0.0):
        _fail(f"{field} must be a finite increasing pair")
    if maximum_width is not None and high - low > maximum_width + 1.0e-12:
        _fail(f"{field} width exceeds its supported interval")
    return low, high


def _finite_vector3(value: Any, *, field: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(f"{field} must be a finite 3-vector") from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        _fail(f"{field} must be a finite 3-vector")
    return vector


def _finite_scalar(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def continuous_direction_latent_declaration(layout: str) -> dict[str, Any]:
    """Return the stable public declaration for a supported latent layout."""

    if layout == CONTINUOUS_REDUCED_7D:
        names = CONTINUOUS_DIRECTION_LATENT_NAMES
        no_op: tuple[str, ...] = ()
    elif layout == CONTINUOUS_LEGACY_STANDOFF_8D:
        names = _LEGACY_LATENT_NAMES
        no_op = ("standoff_quantile",)
    else:
        _fail(
            "continuous-direction latent layout is unsupported",
            layout=layout,
            supported=[CONTINUOUS_REDUCED_7D, CONTINUOUS_LEGACY_STANDOFF_8D],
        )
    return {
        "layout": layout,
        "latent_dimension": len(names),
        "latent_names": list(names),
        "physical_latent_names": list(CONTINUOUS_DIRECTION_LATENT_NAMES),
        "no_op_latent_names": list(no_op),
        "latent_domain": "open_unit_hypercube",
    }


def _physical_latent(
    latent: Sequence[float] | np.ndarray, *, layout: str
) -> tuple[np.ndarray, np.ndarray]:
    declaration = continuous_direction_latent_declaration(layout)
    try:
        raw = np.asarray(latent, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(
            "continuous-direction latent must lie in the open unit hypercube"
        ) from exc
    if (
        raw.shape != (declaration["latent_dimension"],)
        or not np.isfinite(raw).all()
        or np.any(raw <= 0.0)
        or np.any(raw >= 1.0)
    ):
        _fail(
            "continuous-direction latent must lie in the open unit hypercube",
            expected_dimension=declaration["latent_dimension"],
        )
    if layout == CONTINUOUS_REDUCED_7D:
        return raw, raw.copy()
    return raw, raw[[0, 1, 2, 3, 4, 6, 7]]


def dynamic_tangent_basis(
    direction_target_xyz: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a deterministic right-handed orthonormal tangent frame."""

    direction = _finite_vector3(
        direction_target_xyz, field="direction_target_xyz"
    )
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        _fail("direction_target_xyz must be nonzero")
    direction = direction / norm
    axes = np.eye(3, dtype=np.float64)
    reference = axes[int(np.argmin(np.abs(axes @ direction)))]
    tangent_u = reference - float(np.dot(reference, direction)) * direction
    tangent_u /= float(np.linalg.norm(tangent_u))
    tangent_v = np.cross(direction, tangent_u)
    tangent_v /= float(np.linalg.norm(tangent_v))
    return tangent_u, tangent_v


def reference_axis_tangent_basis(
    direction_target_xyz: Sequence[float] | np.ndarray,
    region_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Aim-window frame whose ``v`` axis keeps one target-frame meaning.

    Horizontal sectors use target ``+z`` (up) projected orthogonal to the
    approach direction as ``v``; the top cap uses target ``+y`` (forward).
    ``u = cross(v, direction)`` so ``(direction, u, v)`` is right-handed and
    ``cross(direction, u) == v``.  For the four cardinal horizontal
    directions and straight-down top attack this reproduces the explicit
    frames of the earlier ``contact_pose`` approach declarations, so a
    declared ``aim_v_range_m`` means the same height band in every sector.
    """

    direction = _finite_vector3(
        direction_target_xyz, field="direction_target_xyz"
    )
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        _fail("direction_target_xyz must be nonzero")
    direction = direction / norm
    if region_kind == "horizontal_sector":
        reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    elif region_kind == "top_cap":
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        _fail("reference-axis tangent basis region kind is unsupported", kind=region_kind)
    projected = reference - float(np.dot(reference, direction)) * direction
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm < 1.0e-6:
        _fail(
            "reference axis is parallel to the approach direction",
            region_kind=region_kind,
        )
    tangent_v = projected / projected_norm
    tangent_u = np.cross(tangent_v, direction)
    tangent_u /= float(np.linalg.norm(tangent_u))
    return tangent_u, tangent_v


def aim_tangent_basis(
    direction_target_xyz: Sequence[float] | np.ndarray,
    region_kind: str,
    basis_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the declared aim-window tangent basis."""

    if basis_id == LEGACY_AIM_TANGENT_BASIS:
        return dynamic_tangent_basis(direction_target_xyz)
    if basis_id == REFERENCE_AXIS_AIM_TANGENT_BASIS:
        return reference_axis_tangent_basis(direction_target_xyz, region_kind)
    _fail(
        "continuous-direction aim tangent basis is unsupported",
        aim_tangent_basis=basis_id,
        supported=list(AIM_TANGENT_BASES),
    )
    raise AssertionError("unreachable")


def _horizontal_direction(
    region: Mapping[str, Any], first: float, second: float
) -> tuple[np.ndarray, float]:
    center = math.radians(float(region["azimuth_center_deg"]))
    half_width = math.radians(float(region["azimuth_half_width_deg"]))
    dive_low_deg, dive_high_deg = region["dive_range_deg"]
    dive_low = math.radians(float(dive_low_deg))
    dive_high = math.radians(float(dive_high_deg))
    azimuth = center + (2.0 * first - 1.0) * half_width
    sine_dive = math.sin(dive_low) + second * (
        math.sin(dive_high) - math.sin(dive_low)
    )
    dive = math.asin(float(np.clip(sine_dive, -1.0, 1.0)))
    cosine_dive = math.cos(dive)
    direction = np.asarray(
        [
            cosine_dive * math.cos(azimuth),
            cosine_dive * math.sin(azimuth),
            -math.sin(dive),
        ],
        dtype=np.float64,
    )
    solid_angle = 2.0 * half_width * (
        math.sin(dive_high) - math.sin(dive_low)
    )
    return direction, solid_angle


def _top_direction(
    region: Mapping[str, Any], first: float, second: float
) -> tuple[np.ndarray, float]:
    azimuth_low_deg, azimuth_high_deg = region["azimuth_range_deg"]
    offset_low_deg, offset_high_deg = region["vertical_offset_range_deg"]
    azimuth_low = math.radians(float(azimuth_low_deg))
    azimuth_high = math.radians(float(azimuth_high_deg))
    offset_low = math.radians(float(offset_low_deg))
    offset_high = math.radians(float(offset_high_deg))
    azimuth = azimuth_low + first * (azimuth_high - azimuth_low)
    cosine_offset = math.cos(offset_low) - second * (
        math.cos(offset_low) - math.cos(offset_high)
    )
    offset = math.acos(float(np.clip(cosine_offset, -1.0, 1.0)))
    direction = np.asarray(
        [
            math.sin(offset) * math.cos(azimuth),
            math.sin(offset) * math.sin(azimuth),
            -math.cos(offset),
        ],
        dtype=np.float64,
    )
    solid_angle = (azimuth_high - azimuth_low) * (
        math.cos(offset_low) - math.cos(offset_high)
    )
    return direction, solid_angle


def validate_continuous_direction_manifold(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a canonical, semantically validated SamplingSpec 1.7 manifold."""

    if not isinstance(document, Mapping):
        _fail("continuous-direction contact_manifold must be an object")
    try:
        canonical = json.loads(
            json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(
            "continuous-direction contact_manifold must be finite JSON"
        ) from exc
    expected = {
        "schema_version": CONTINUOUS_DIRECTION_SCHEMA_VERSION,
        "sampler_id": CONTINUOUS_DIRECTION_SAMPLER_ID,
        "mode": CONTINUOUS_DIRECTION_MODE,
        "approach_weight_semantics": CONTACT_MANIFOLD_APPROACH_WEIGHT_SEMANTICS,
        "latent_layout": CONTINUOUS_REDUCED_7D,
        "direction_measure": "uniform_solid_angle_within_region",
        "bottom_up_policy": "stress_only",
    }
    observed = {key: canonical.get(key) for key in expected}
    if observed != expected:
        _fail(
            "continuous-direction contact-manifold identity is unsupported",
            expected=expected,
            observed=observed,
        )
    basis_declared = "aim_tangent_basis" in canonical
    basis_id = canonical.get("aim_tangent_basis", LEGACY_AIM_TANGENT_BASIS)
    if basis_id not in _BASIS_CONTRACTS:
        _fail(
            "continuous-direction aim tangent basis is unsupported",
            aim_tangent_basis=basis_id,
            supported=list(AIM_TANGENT_BASES),
        )
    measure_contract, measure_hash, implementation_hash = _BASIS_CONTRACTS[basis_id]
    if canonical.get("implementation_hash") not in {None, implementation_hash}:
        _fail(
            "continuous-direction implementation hash is inconsistent",
            expected=implementation_hash,
            observed=canonical.get("implementation_hash"),
            aim_tangent_basis=basis_id,
        )
    if canonical.get("measure_contract_hash") not in {None, measure_hash}:
        _fail(
            "continuous-direction measure hash is inconsistent",
            expected=measure_hash,
            observed=canonical.get("measure_contract_hash"),
            aim_tangent_basis=basis_id,
        )
    sphere = canonical.get("sphere")
    if (
        not isinstance(sphere, Mapping)
        or sphere.get("algorithm_id") != "target-aabb-circumsphere@1.0.0"
        or sphere.get("object_scope") not in {"all", "armor"}
    ):
        _fail("continuous-direction sphere declaration is unsupported")
    clearance = _finite_scalar(
        sphere.get("clearance_m", -1.0), field="sphere.clearance_m"
    )
    if not math.isfinite(clearance) or clearance < 0.0:
        _fail("continuous-direction sphere clearance_m must be non-negative")

    regions = canonical.get("regions")
    if not isinstance(regions, list) or len(regions) < 2:
        _fail("continuous-direction manifold requires at least two regions")
    identifiers: set[str] = set()
    horizontal_centres: list[float] = []
    top_count = 0
    for index, region in enumerate(regions):
        if not isinstance(region, Mapping):
            _fail("each continuous-direction region must be an object", index=index)
        identifier = region.get("region_id")
        if not isinstance(identifier, str) or not identifier:
            _fail("continuous-direction region_id must be nonempty", index=index)
        if identifier in identifiers:
            _fail("continuous-direction region IDs must be unique", region_id=identifier)
        identifiers.add(identifier)
        weight = _finite_scalar(
            region.get("selection_weight", 0.0),
            field=f"regions[{index}].selection_weight",
        )
        if not math.isfinite(weight) or weight <= 0.0:
            _fail("continuous-direction selection_weight must be positive", region_id=identifier)
        kind = region.get("kind")
        if kind == "horizontal_sector":
            centre = _finite_scalar(
                region.get("azimuth_center_deg"),
                field=f"regions[{index}].azimuth_center_deg",
            )
            half_width = _finite_scalar(
                region.get("azimuth_half_width_deg"),
                field=f"regions[{index}].azimuth_half_width_deg",
            )
            if not math.isfinite(centre) or not math.isclose(
                half_width, 30.0, rel_tol=0.0, abs_tol=1.0e-12
            ):
                _fail("horizontal sectors require an azimuth half-width of 30 degrees")
            dive = _finite_range(
                region.get("dive_range_deg"), field=f"regions[{index}].dive_range_deg"
            )
            if not math.isclose(dive[0], 0.0, abs_tol=1.0e-12) or not math.isclose(
                dive[1], 45.0, abs_tol=1.0e-12
            ):
                _fail("horizontal sectors require a 0 to 45 degree dive interval")
            horizontal_centres.append(centre % 360.0)
        elif kind == "top_cap":
            azimuth = _finite_range(
                region.get("azimuth_range_deg"),
                field=f"regions[{index}].azimuth_range_deg",
            )
            offset = _finite_range(
                region.get("vertical_offset_range_deg"),
                field=f"regions[{index}].vertical_offset_range_deg",
            )
            if not (
                math.isclose(azimuth[0], 0.0, abs_tol=1.0e-12)
                and math.isclose(azimuth[1], 360.0, abs_tol=1.0e-12)
                and math.isclose(offset[0], 0.0, abs_tol=1.0e-12)
                and math.isclose(offset[1], 45.0, abs_tol=1.0e-12)
            ):
                _fail("top cap requires full azimuth and 0 to 45 degree vertical offset")
            top_count += 1
        else:
            _fail(
                "continuous-direction region kind is unsupported",
                region_id=identifier,
                kind=kind,
            )
        _finite_vector3(
            region.get("aim_center_target_m"),
            field=f"regions[{index}].aim_center_target_m",
        )
        _finite_range(region.get("aim_u_range_m"), field=f"regions[{index}].aim_u_range_m")
        _finite_range(region.get("aim_v_range_m"), field=f"regions[{index}].aim_v_range_m")
        _finite_range(
            region.get("standoff_range_m"),
            field=f"regions[{index}].standoff_range_m",
            positive_lower=True,
        )
        _finite_range(
            region.get("speed_range_mps"),
            field=f"regions[{index}].speed_range_mps",
            positive_lower=True,
        )
        _finite_range(
            region.get("body_x_roll_range_rad"),
            field=f"regions[{index}].body_x_roll_range_rad",
            maximum_width=2.0 * math.pi,
        )
    if sorted(round(value, 12) for value in horizontal_centres) != [
        0.0,
        90.0,
        180.0,
        270.0,
    ] or top_count != 1:
        _fail(
            "reference continuous directions require four cardinal horizontal sectors and one top cap"
        )
    canonical["implementation_hash"] = implementation_hash
    canonical["measure_contract_hash"] = measure_hash
    canonical["measure_contract"] = measure_contract
    if basis_declared:
        canonical["aim_tangent_basis"] = basis_id
    return canonical


def apply_continuous_direction_manifold(
    case: dict[str, Any],
    document: Mapping[str, Any],
    latent: Sequence[float] | np.ndarray,
    target: CompiledTarget,
    *,
    target_hash: str,
) -> dict[str, Any]:
    """Construct, sphere-reanchor and record one continuous directed ray."""

    if not isinstance(case, dict):
        _fail("continuous-direction application requires a mutable case")
    if not isinstance(target, CompiledTarget):
        _fail("continuous-direction application requires a CompiledTarget")
    canonical = validate_continuous_direction_manifold(document)
    raw, physical = _physical_latent(latent, layout=canonical["latent_layout"])
    regions = canonical["regions"]
    weights = np.asarray(
        [float(region["selection_weight"]) for region in regions], dtype=np.float64
    )
    masses = weights / float(np.sum(weights))
    cumulative = np.cumsum(masses)
    region_index = min(
        int(np.searchsorted(cumulative, physical[0], side="right")),
        len(regions) - 1,
    )
    region = regions[region_index]
    if region["kind"] == "horizontal_sector":
        direction, solid_angle = _horizontal_direction(
            region, float(physical[1]), float(physical[2])
        )
    else:
        direction, solid_angle = _top_direction(
            region, float(physical[1]), float(physical[2])
        )
    basis_id = str(canonical.get("aim_tangent_basis", LEGACY_AIM_TANGENT_BASIS))
    _, measure_hash, implementation_hash = _BASIS_CONTRACTS[basis_id]
    tangent_u, tangent_v = aim_tangent_basis(direction, str(region["kind"]), basis_id)

    def linear(field: str, quantile: float) -> float:
        low, high = map(float, region[field])
        return low + quantile * (high - low)

    center = np.asarray(region["aim_center_target_m"], dtype=np.float64)
    aim_u = linear("aim_u_range_m", float(physical[3]))
    aim_v = linear("aim_v_range_m", float(physical[4]))
    aim_point = center + aim_u * tangent_u + aim_v * tangent_v
    speed = linear("speed_range_mps", float(physical[5]))
    roll = linear("body_x_roll_range_rad", float(physical[6]))
    standoff = 0.5 * sum(map(float, region["standoff_range_m"]))
    origin = aim_point - standoff * direction
    cosine, sine = math.cos(roll), math.sin(roll)
    body_y = cosine * tangent_u + sine * tangent_v
    body_z = -sine * tangent_u + cosine * tangent_v
    rotation = np.column_stack((direction, body_y, body_z))
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-10):
        _fail("continuous-direction body frame is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-10):
        _fail("continuous-direction body frame is not proper")
    encounter = case.get("encounter")
    if not isinstance(encounter, dict):
        _fail("continuous-direction construction requires an encounter object")
    encounter["trajectory_origin_m"] = [float(value) for value in origin]
    encounter["velocity_target_mps"] = [float(value) for value in speed * direction]
    encounter["body_to_target_orientation_xyzw"] = _matrix_to_quaternion_xyzw(rotation)

    sphere_declaration = canonical["sphere"]
    sphere = target.enclosing_sphere(
        object_scope=str(sphere_declaration["object_scope"]),  # type: ignore[arg-type]
        clearance_m=float(sphere_declaration["clearance_m"]),
    )
    ray = reanchor_ray_to_sphere(
        line_point_target_m=aim_point,
        direction_target_xyz=direction,
        sphere=sphere,
        target_hash=target_hash,
    )
    encounter["trajectory_origin_m"] = list(ray.origin_target_m)
    cleared_fields = [field for field in _DERIVED_IMPACT_FIELDS if field in encounter]
    for field in cleared_fields:
        encounter.pop(field, None)

    region_mass = float(masses[region_index])
    density = region_mass / solid_angle
    common = {
        "schema_version": CONTINUOUS_DIRECTION_SCHEMA_VERSION,
        "sampler_id": CONTINUOUS_DIRECTION_SAMPLER_ID,
        "implementation_hash": implementation_hash,
        "mode": CONTINUOUS_DIRECTION_MODE,
        "status": "reanchored_pending_s_contact",
        "latent_layout": canonical["latent_layout"],
        "latent_names": list(
            continuous_direction_latent_declaration(canonical["latent_layout"])[
                "latent_names"
            ]
        ),
        "latent": [float(value) for value in raw],
        "physical_latent": [float(value) for value in physical],
        "no_op_latent_names": list(
            continuous_direction_latent_declaration(canonical["latent_layout"])[
                "no_op_latent_names"
            ]
        ),
        "region_id": str(region["region_id"]),
        "region_index": region_index,
        "region_kind": str(region["kind"]),
        "region_design_mass": region_mass,
        "region_solid_angle_sr": float(solid_angle),
        "direction_density_sr_inv": float(density),
        "measure_contract_hash": measure_hash,
        "direction_target_xyz": [float(value) for value in direction],
        "tangent_u_target_xyz": [float(value) for value in tangent_u],
        "tangent_v_target_xyz": [float(value) for value in tangent_v],
        "aim_u_m": aim_u,
        "aim_v_m": aim_v,
        "aim_point_target_m": [float(value) for value in aim_point],
        "speed_mps": speed,
        "body_x_roll_rad": roll,
        "bottom_up_policy": "stress_only",
        "approach_weight_semantics": CONTACT_MANIFOLD_APPROACH_WEIGHT_SEMANTICS,
        "evidence_boundary": (
            "Outcome-blind continuous-direction construction only; formal label "
            "admission requires a successful S-CONTACT audit."
        ),
    }
    if basis_id != LEGACY_AIM_TANGENT_BASIS:
        # Legacy records stay byte-identical so archived case hashes reproduce.
        common["aim_tangent_basis"] = basis_id
    sphere_record = {
        **common,
        "sphere": sphere.as_document(),
        "trajectory_origin_m": list(ray.origin_target_m),
        "transverse_offset_target_m": list(ray.transverse_offset_target_m),
        "transverse_radius_m": ray.transverse_radius_m,
        "physical_ray_identity": dict(ray.physical_ray_identity),
        "physical_ray_id": ray.physical_ray_id,
        "cleared_derived_impact_fields": cleared_fields,
    }
    contact_record = {
        **common,
        "approach_id": str(region["region_id"]),
        "approach_index": region_index,
        "approach_design_mass": region_mass,
        "standoff_m": standoff,
        "trajectory_origin_superseded_by": CONTINUOUS_DIRECTION_SAMPLER_ID,
    }
    metadata = case.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        _fail("case metadata must be an object")
    metadata["continuous_direction_sampling"] = common
    metadata["contact_manifold_sampling"] = contact_record
    metadata["sphere_entry_ray_sampling"] = sphere_record
    return sphere_record


__all__ = [
    "AIM_TANGENT_BASES",
    "LEGACY_AIM_TANGENT_BASIS",
    "REFERENCE_AXIS_AIM_TANGENT_BASIS",
    "REFERENCE_AXIS_IMPLEMENTATION_HASH",
    "REFERENCE_AXIS_MEASURE_CONTRACT",
    "REFERENCE_AXIS_MEASURE_HASH",
    "aim_tangent_basis",
    "reference_axis_tangent_basis",
    "CONTINUOUS_DIRECTION_IMPLEMENTATION_DOCUMENT",
    "CONTINUOUS_DIRECTION_IMPLEMENTATION_HASH",
    "CONTINUOUS_DIRECTION_LATENT_NAMES",
    "CONTINUOUS_DIRECTION_MEASURE_CONTRACT",
    "CONTINUOUS_DIRECTION_MEASURE_HASH",
    "CONTINUOUS_DIRECTION_MODE",
    "CONTINUOUS_DIRECTION_SAMPLER_ID",
    "CONTINUOUS_DIRECTION_SCHEMA_VERSION",
    "CONTINUOUS_LEGACY_STANDOFF_8D",
    "CONTINUOUS_REDUCED_7D",
    "apply_continuous_direction_manifold",
    "continuous_direction_latent_declaration",
    "dynamic_tangent_basis",
    "validate_continuous_direction_manifold",
]
