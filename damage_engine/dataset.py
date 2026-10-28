from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .archive_validation import verify_monolithic_archive
from .content_contract import is_sha256_identity, validate_json_schema
from .engine import ENGINE_CODE_HASH, DamageEngine, EngineResult
from .errors import InputValidationError
from .estimands import positive_effective_sample_size, scale_positive_weights
from .geometry import BoxGeometry, CylinderGeometry, ExtrudedPolygonGeometry
from .learning_input_identity import (
    model_input_definition_for_sampling_contract,
)
from .registry import PhysicsRegistry
from .quality import synthetic_label_admission, validate_precomputed_results
from .result_ledger import write_result_ledger
from .utils import content_hash, file_hash, length_to_m, pressure_to_pa, ratio_value


CASE_FEATURE_NAMES = [
    "detonation_x_m", "detonation_y_m", "detonation_z_m", "detonation_time_s",
    "orientation_qx", "orientation_qy", "orientation_qz", "orientation_qw",
    "velocity_x_mps", "velocity_y_mps", "velocity_z_mps",
    "charge_mass_kg", "casing_mass_kg", "blast_equivalent_mass_kg",
    "fragment_total_count",
    "munition_profile_small_front", "munition_profile_medium_front", "munition_profile_large_front",
    "projectile_length_m", "projectile_diameter_m",
    "warhead_length_m", "warhead_diameter_m", "wall_thickness_m",
    "fragment_ring_count", "fragment_axial_coverage_ratio",
    "impact_first_armor_contact", "collision_distance_m",
    "mass_family_fixed", "mass_family_lognormal", "mass_family_weibull", "mass_family_empirical",
    "fragment_mass_mean_kg", "fragment_mass_stddev_kg",
    "fragment_mass_q10_kg", "fragment_mass_q50_kg", "fragment_mass_q90_kg",
    "shape_sphere", "shape_cube", "shape_cylinder", "shape_irregular", "shape_empirical",
    "fragment_density_kg_m3",
    "initial_velocity_fixed", "initial_velocity_gurney",
    "fixed_initial_velocity_mps", "gurney_constant_mps", "detonation_velocity_mps",
    "direction_isotropic", "direction_cone", "direction_taylor_ring",
    "direction_axis_body_x", "direction_axis_body_y", "direction_axis_body_z",
    "direction_half_angle_deg", "direction_spread_sigma_deg",
    "drag_coefficient", "area_shape_factor",
    "air_density_kg_m3", "ambient_pressure_pa", "temperature_k",
    "mechanism_fragment_enabled", "mechanism_blast_enabled",
    "coupling_none", "coupling_sequential", "coupling_fully_coupled",
    "fidelity_l0", "fidelity_l1", "fidelity_l2", "fidelity_l3",
    "requested_replicates", "adaptive_ci_half_width",
]

CASE_FEATURE_UNITS = [
    "m", "m", "m", "s",
    "1", "1", "1", "1",
    "m/s", "m/s", "m/s",
    "kg", "kg", "kg", "count",
    "1", "1", "1", "m", "m", "m", "m", "m", "count", "1", "1", "m",
    "1", "1", "1", "1",
    "kg", "kg", "kg", "kg", "kg",
    "1", "1", "1", "1", "1",
    "kg/m^3", "1", "1", "m/s", "m/s", "m/s",
    "1", "1", "1", "1", "1", "1", "deg", "deg",
    "1", "1", "kg/m^3", "Pa", "K",
    "1", "1", "1", "1", "1", "1", "1", "1", "1",
    "count", "1",
]

# Contract 1.5 separates variables available to a deployed surrogate from
# solver/acquisition controls.  ``case_x`` remains in 1.5 archives as a legacy
# compatibility bundle, but the Torch loader defaults to ``case_physics_x``.
SOLVER_CONTEXT_FEATURE_NAMES = [
    "mechanism_fragment_enabled", "mechanism_blast_enabled",
    "coupling_none", "coupling_sequential", "coupling_fully_coupled",
    "fidelity_l0", "fidelity_l1", "fidelity_l2", "fidelity_l3",
    "requested_replicates", "adaptive_ci_half_width",
]
CASE_PHYSICS_FEATURE_NAMES = [
    name for name in CASE_FEATURE_NAMES if name not in SOLVER_CONTEXT_FEATURE_NAMES
]
_CASE_UNIT_BY_NAME = dict(zip(CASE_FEATURE_NAMES, CASE_FEATURE_UNITS))
CASE_PHYSICS_FEATURE_UNITS = [_CASE_UNIT_BY_NAME[name] for name in CASE_PHYSICS_FEATURE_NAMES]
SOLVER_CONTEXT_FEATURE_UNITS = [_CASE_UNIT_BY_NAME[name] for name in SOLVER_CONTEXT_FEATURE_NAMES]
_CASE_PHYSICS_INDICES = [CASE_FEATURE_NAMES.index(name) for name in CASE_PHYSICS_FEATURE_NAMES]
_SOLVER_CONTEXT_INDICES = [CASE_FEATURE_NAMES.index(name) for name in SOLVER_CONTEXT_FEATURE_NAMES]


COMPONENT_FEATURE_NAMES = [
    "shape_box", "shape_cylinder", "shape_extruded_polygon",
    "center_x_m", "center_y_m", "center_z_m",
    "extent_x_m", "extent_y_m", "extent_z_m",
    "principal_axis_x", "principal_axis_y", "principal_axis_z",
    "rotation_r00", "rotation_r01", "rotation_r02",
    "rotation_r10", "rotation_r11", "rotation_r12",
    "rotation_r20", "rotation_r21", "rotation_r22",
    "box_size_l_m", "box_size_w_m", "box_size_h_m",
    "cylinder_radius_m", "cylinder_height_m",
    "extrusion_distance_m", "polygon_area_m2", "polygon_perimeter_m", "vertex_count",
    "equivalent_thickness_m", "vulnerable_area_ratio", "blast_peak_threshold_pa",
    "blast_impulse_threshold_pa_s", "blast_duration_threshold_s",
    "exposure_external", "exposure_semi_exposed", "exposure_internal",
]

COMPONENT_FEATURE_UNITS = [
    "1", "1", "1", "m", "m", "m", "m", "m", "m",
    "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1",
    "m", "m", "m", "m", "m", "m", "m^2", "m", "count",
    "m", "1", "Pa", "Pa*s", "s", "1", "1", "1",
]

ARMOR_FEATURE_NAMES = [
    "center_x_m", "center_y_m", "center_z_m",
    "extent_x_m", "extent_y_m", "extent_z_m",
    "extrusion_direction_x", "extrusion_direction_y", "extrusion_direction_z",
    "extrusion_distance_m", "polygon_area_m2", "polygon_perimeter_m", "vertex_count",
    "equivalent_thickness_m", "vulnerable_area_ratio", "blast_peak_threshold_pa",
    "blast_impulse_threshold_pa_s", "blast_duration_threshold_s",
]

ARMOR_FEATURE_UNITS = [
    "m", "m", "m", "m", "m", "m", "1", "1", "1", "m", "m^2", "m", "count",
    "m", "1", "Pa", "Pa*s", "s",
]

ARMOR_PATH_FEATURE_NAMES = [
    "intersects_path", "entry_distance_m", "path_length_m", "incidence_cosine"
]
ARMOR_PATH_FEATURE_UNITS = ["1", "m", "m", "1"]

ARMOR_LABEL_NAMES = [
    "post_fragment_integrity",
    "post_combined_integrity",
    "blast_distance_m",
    "blast_incident_peak_pressure_pa",
    "blast_incident_impulse_pa_s",
    "blast_reflected_peak_pressure_pa",
    "blast_reflected_impulse_pa_s",
    "blast_positive_duration_s",
    "blast_arrival_time_s",
    "blast_incidence_cosine",
    "blast_failure_probability",
    "blast_pre_integrity",
    "blast_post_integrity",
]

ARMOR_LABEL_UNITS = [
    "1", "1", "m", "Pa", "Pa*s", "Pa", "Pa*s", "s", "s", "1", "1", "1", "1",
]

_ARMOR_BLAST_LABEL_KEYS = [
    "distance_m",
    "incident_peak_pressure_pa",
    "incident_impulse_pa_s",
    "reflected_peak_pressure_pa",
    "reflected_impulse_pa_s",
    "positive_duration_s",
    "arrival_time_s",
    "incidence_cosine",
    "blast_failure_probability",
    "pre_blast_integrity",
    "post_blast_integrity",
]

FRECHET_BOUND_NAMES = ["lower", "upper"]
FRECHET_BOUND_UNITS = ["1", "1"]


def dataset_identity_hash(
    *,
    target_hash: str,
    physics_registry_hash: str,
    ordered_case_hashes: list[str],
    split_seed: int,
    split_ratios: tuple[float, float, float] | list[float],
    tensor_contract_version: str,
    engine_code_hash: str,
    damage_tree_definition_hash: str | None = None,
    label_spec_hash: str | None = None,
    sampling_design_hash: str | None = None,
    stratification_spec_hash: str | None = None,
    split_registry_hash: str | None = None,
) -> str:
    """Return the storage-layout-independent identity of teacher labels."""

    identity = {
            "target_hash": target_hash,
            "physics_registry_hash": physics_registry_hash,
            "engine_code_hash": engine_code_hash,
            "ordered_cases": ordered_case_hashes,
            "split_seed": int(split_seed),
            "split_ratios": [float(value) for value in split_ratios],
            "tensor_contract_version": str(tensor_contract_version),
        }
    optional = {
        "damage_tree_definition_hash": damage_tree_definition_hash,
        "label_spec_hash": label_spec_hash,
        "sampling_design_hash": sampling_design_hash,
        "stratification_spec_hash": stratification_spec_hash,
        "split_registry_hash": split_registry_hash,
    }
    identity.update({key: value for key, value in optional.items() if value is not None})
    return content_hash(identity)

AUXILIARY_LABEL_NAMES = [
    "fragment_hits", "fragment_penetrations", "fragment_mean_arrival_speed_mps",
    "fragment_mean_penetration_margin", "fragment_deposited_energy_j",
    "blast_distance_m", "blast_incident_peak_pressure_pa", "blast_incident_impulse_pa_s",
    "blast_effective_peak_pressure_pa", "blast_effective_impulse_pa_s",
    "blast_positive_duration_s", "blast_arrival_time_s",
    "blast_shielding_effective_thickness_m", "blast_shielding_panel_count",
]
AUXILIARY_LABEL_UNITS = [
    "count", "count", "m/s", "1", "J", "m", "Pa", "Pa*s", "Pa", "Pa*s",
    "s", "s", "m", "count",
]

_FRAGMENT_AUXILIARY_KEYS = [
    "hits", "penetrations", "mean_arrival_speed_mps", "mean_penetration_margin",
    "deposited_energy_j",
]
_BLAST_AUXILIARY_KEYS = [
    "distance_m", "incident_peak_pressure_pa", "incident_impulse_pa_s",
    "effective_peak_pressure_pa", "effective_impulse_pa_s", "positive_duration_s",
    "arrival_time_s", "shielding_effective_thickness_m", "shielding_panel_count",
]


_WORKER_ENGINE: DamageEngine | None = None


def _initialize_worker(target: dict, registry_data: dict) -> None:
    global _WORKER_ENGINE
    _WORKER_ENGINE = DamageEngine(target, PhysicsRegistry(registry_data))


def _refused_internal_result(engine: DamageEngine, case: dict, error: Exception) -> EngineResult:
    return EngineResult(
        {
            "status": "refused",
            "case_id": case.get("case_id"),
            "refusal": {
                "code": "REF_CASE_EXECUTION",
                "message": str(error),
                "context": {"exception_type": type(error).__name__},
            },
            "provenance": engine._provenance(case, replicate_count=0),
        }
    )


def _run_case(engine: DamageEngine, case: dict, fail_fast: bool) -> EngineResult:
    if fail_fast:
        return engine.run(case, fail_closed=True)
    try:
        return engine.run(case, fail_closed=False)
    except Exception as error:
        # A single malformed or numerically pathological case must not discard a
        # long batch.  The exception type is retained for diagnosis, while
        # KeyboardInterrupt/SystemExit are deliberately not caught.
        return _refused_internal_result(engine, case, error)


def _run_case_in_worker(arguments: tuple[dict, bool]) -> dict:
    case, fail_fast = arguments
    if _WORKER_ENGINE is None:
        raise RuntimeError("dataset worker was not initialized")
    return _run_case(_WORKER_ENGINE, case, fail_fast).to_dict()


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet export requires the optional pyarrow dependency"
        ) from exc
    return pa, pq


def _write_parquet(path: Path, rows: list[dict]) -> None:
    pa, pq = _require_pyarrow()
    from .parquet_schema import table_schema, validate_rows
    schema = table_schema(path.stem)
    validate_rows(rows, schema)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    index = int(np.searchsorted(cumulative, probability, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _mass_distribution_summary(distribution: dict) -> tuple[float, float, float, float, float]:
    family = distribution["family"]
    parameters = distribution["parameters"]
    if family == "fixed":
        value = float(parameters["value_kg"])
        return value, 0.0, value, value, value
    if family == "lognormal":
        median = float(parameters["median_kg"])
        sigma = float(parameters["log_sigma"])
        mean = median * math.exp(0.5 * sigma * sigma)
        variance = median * median * math.exp(sigma * sigma) * (math.exp(sigma * sigma) - 1.0)
        q10 = median * math.exp(-1.2815515655446004 * sigma)
        q90 = median * math.exp(1.2815515655446004 * sigma)
        return mean, math.sqrt(max(0.0, variance)), q10, median, q90
    if family == "weibull":
        scale = float(parameters["scale_kg"])
        shape = float(parameters["shape"])
        mean = scale * math.gamma(1.0 + 1.0 / shape)
        second = scale * scale * math.gamma(1.0 + 2.0 / shape)
        quantile = lambda probability: scale * (-math.log1p(-probability)) ** (1.0 / shape)
        return (
            mean,
            math.sqrt(max(0.0, second - mean * mean)),
            quantile(0.1),
            quantile(0.5),
            quantile(0.9),
        )
    empirical = np.asarray(parameters["values_kg"], dtype=np.float64)
    probabilities = parameters.get("probabilities")
    if probabilities is None:
        weights = np.full(len(empirical), 1.0 / len(empirical), dtype=np.float64)
    else:
        weights = np.asarray(probabilities, dtype=np.float64)
        weights = weights / np.sum(weights)
    mean = float(np.sum(empirical * weights))
    stddev = float(np.sqrt(np.sum((empirical - mean) ** 2 * weights)))
    return (
        mean,
        stddev,
        _weighted_quantile(empirical, weights, 0.1),
        _weighted_quantile(empirical, weights, 0.5),
        _weighted_quantile(empirical, weights, 0.9),
    )


def _optional_parameter_si(parameter: dict | None, quantity: str) -> tuple[float, bool]:
    if parameter is None:
        return 0.0, False
    if quantity == "pressure":
        return pressure_to_pa(parameter), True
    value = float(parameter["value"])
    unit = str(parameter["unit"])
    if quantity == "impulse":
        factors = {"Pa*s": 1.0, "Pa s": 1.0, "kPa*s": 1.0e3, "kPa s": 1.0e3, "MPa*s": 1.0e6, "MPa s": 1.0e6}
    elif quantity == "duration":
        factors = {"s": 1.0, "ms": 1.0e-3, "us": 1.0e-6}
    else:
        raise ValueError(f"unsupported parameter quantity: {quantity}")
    if unit not in factors:
        raise InputValidationError(f"unsupported {quantity} unit: {unit}")
    return value * factors[unit], True


def _case_features(case: dict, engine: DamageEngine) -> tuple[list[float], list[bool]]:
    encounter = case["encounter"]
    warhead = case["warhead"]
    environment = case["environment"]
    source = warhead["fragment_source"]
    munition = case.get("munition") or {}
    has_munition = bool(munition)
    distribution = source["mass_distribution"]
    family_names = ["fixed", "lognormal", "weibull", "empirical"]
    family = distribution["family"]
    family_one_hot = [float(family == name) for name in family_names]
    mass_mean, mass_stddev, mass_q10, mass_q50, mass_q90 = _mass_distribution_summary(
        distribution
    )
    shape_names = ["sphere", "cube", "cylinder", "irregular", "empirical"]
    shape_one_hot = [float(source["shape_class"] == name) for name in shape_names]
    source_model = engine.registry.fragment_source(source["model_id"])
    initial_kind = source_model["initial_velocity"]["kind"]
    initial_one_hot = [float(initial_kind == "fixed"), float(initial_kind == "gurney_cylinder")]
    fixed_velocity = float(source_model["initial_velocity"].get("velocity_mps", 0.0))
    direction = source_model["direction"]
    direction_names = ["isotropic", "cone", "taylor_ring"]
    direction_one_hot = [float(direction["kind"] == name) for name in direction_names]
    half_angle = float(direction.get("half_angle_deg", 0.0))
    spread_sigma = float(direction.get("spread_sigma_deg", 0.0))
    direction_axis = list(map(float, direction.get("axis_body", [0.0, 0.0, 0.0])))
    drag = source_model["drag"]
    mechanisms = set(case["analysis"]["mechanisms"])
    coupling = case["analysis"]["coupling_model"]
    fidelity = str(case["analysis"]["fidelity_level"]).lower()
    adaptive_ci = case["analysis"].get("adaptive_ci_half_width")
    entries: dict[str, tuple[float, bool]] = {}

    def add(name: str, value: float, valid: bool = True) -> None:
        entries[name] = (float(value), bool(valid))

    for axis, value in zip("xyz", encounter["detonation_position_m"]):
        add(f"detonation_{axis}_m", value)
    add("detonation_time_s", encounter.get("detonation_time_s", 0.0), "detonation_time_s" in encounter)
    for axis, value in zip(("qx", "qy", "qz", "qw"), encounter["body_to_target_orientation_xyzw"]):
        add(f"orientation_{axis}", value)
    for axis, value in zip("xyz", encounter["velocity_target_mps"]):
        add(f"velocity_{axis}_mps", value)
    add("charge_mass_kg", warhead.get("charge_mass_kg") or 0.0, warhead.get("charge_mass_kg") is not None)
    add("casing_mass_kg", warhead.get("casing_mass_kg") or 0.0, warhead.get("casing_mass_kg") is not None)
    add("blast_equivalent_mass_kg", warhead["blast_source"]["equivalent_mass_kg"])
    add("fragment_total_count", source["total_count"])
    profile_id = munition.get("profile_id")
    for name in ("small_front", "medium_front", "large_front"):
        add(
            f"munition_profile_{name}",
            profile_id == name,
            has_munition,
        )
    for feature_name, source_name in (
        ("projectile_length_m", "projectile_length_m"),
        ("projectile_diameter_m", "projectile_diameter_m"),
        ("warhead_length_m", "warhead_length_m"),
        ("warhead_diameter_m", "warhead_diameter_m"),
        ("wall_thickness_m", "wall_thickness_m"),
        ("fragment_ring_count", "fragment_ring_count"),
        ("fragment_axial_coverage_ratio", "fragment_axial_coverage_ratio"),
    ):
        add(
            feature_name,
            munition.get(source_name, 0.0),
            source_name in munition,
        )
    is_impact = encounter.get("detonation_trigger") == "first_armor_contact"
    add("impact_first_armor_contact", is_impact)
    add(
        "collision_distance_m",
        encounter.get("collision_distance_m", 0.0),
        "collision_distance_m" in encounter,
    )
    for name, value in zip(family_names, family_one_hot):
        add(f"mass_family_{name}", value)
    for name, value in zip(
        ("mean", "stddev", "q10", "q50", "q90"),
        (mass_mean, mass_stddev, mass_q10, mass_q50, mass_q90),
    ):
        add(f"fragment_mass_{name}_kg", value)
    for name, value in zip(shape_names, shape_one_hot):
        add(f"shape_{name}", value)
    add("fragment_density_kg_m3", source["density_kg_m3"])
    add("initial_velocity_fixed", initial_one_hot[0])
    add("initial_velocity_gurney", initial_one_hot[1])
    add("fixed_initial_velocity_mps", fixed_velocity, initial_kind == "fixed")
    add("gurney_constant_mps", source.get("gurney_constant_mps") or 0.0, source.get("gurney_constant_mps") is not None)
    add("detonation_velocity_mps", source.get("detonation_velocity_mps") or 0.0, source.get("detonation_velocity_mps") is not None)
    for name, value in zip(direction_names, direction_one_hot):
        add(f"direction_{name}", value)
    for axis, value in zip("xyz", direction_axis):
        add(f"direction_axis_body_{axis}", value, "axis_body" in direction)
    add("direction_half_angle_deg", half_angle, "half_angle_deg" in direction)
    add("direction_spread_sigma_deg", spread_sigma, "spread_sigma_deg" in direction)
    add("drag_coefficient", drag["coefficient"])
    add("area_shape_factor", drag.get("area_shape_factor", 1.0))
    add("air_density_kg_m3", environment["air_density_kg_m3"])
    add("ambient_pressure_pa", environment["ambient_pressure_pa"])
    add("temperature_k", environment["temperature_k"])
    add("mechanism_fragment_enabled", "fragment" in mechanisms)
    add("mechanism_blast_enabled", "blast" in mechanisms)
    for name in ("none", "sequential", "fully_coupled"):
        add(f"coupling_{name}", coupling == name)
    for name in ("l0", "l1", "l2", "l3"):
        add(f"fidelity_{name}", fidelity == name)
    add("requested_replicates", case["analysis"]["replicates"])
    add("adaptive_ci_half_width", adaptive_ci or 0.0, adaptive_ci is not None)
    if set(entries) != set(CASE_FEATURE_NAMES):
        raise RuntimeError("case feature implementation and contract are inconsistent")
    return (
        [entries[name][0] for name in CASE_FEATURE_NAMES],
        [entries[name][1] for name in CASE_FEATURE_NAMES],
    )


def _component_features(
    item,
    realized: dict[str, float | None] | None = None,
) -> tuple[list[float], list[bool]]:
    geometry = item.geometry
    shape = [0.0, 0.0, 0.0]
    box_size = [0.0, 0.0, 0.0]
    cylinder_size = [0.0, 0.0]
    polygon_summary = [0.0, 0.0, 0.0, 0.0]
    shape_parameter_mask = [False] * 9
    rotation_valid = True
    if isinstance(geometry, BoxGeometry):
        shape[0] = 1.0
        axis = geometry.rotation @ np.array([1.0, 0.0, 0.0])
        box_size = list(map(float, geometry.size))
        shape_parameter_mask[:3] = [True] * 3
    elif isinstance(geometry, CylinderGeometry):
        shape[1] = 1.0
        axis = geometry.rotation @ np.array([0.0, 0.0, 1.0])
        cylinder_size = [float(geometry.radius), float(geometry.height)]
        shape_parameter_mask[3:5] = [True] * 2
    elif isinstance(geometry, ExtrudedPolygonGeometry):
        shape[2] = 1.0
        axis = geometry.direction
        rotation_valid = False
        vertices = geometry.vertices
        shifted = np.roll(vertices, -1, axis=0)
        area_vector = 0.5 * np.sum(np.cross(vertices, shifted), axis=0)
        polygon_summary = [
            float(geometry.distance),
            abs(float(np.dot(area_vector, geometry.direction))),
            float(np.sum(np.linalg.norm(shifted - vertices, axis=1))),
            float(len(vertices)),
        ]
        shape_parameter_mask[5:] = [True] * 4
    else:
        raise InputValidationError("unsupported compiled component geometry")
    exposure_name = item.source.get("exposure_class", "internal")
    exposure = [
        float(exposure_name == "external"),
        float(exposure_name == "semi_exposed"),
        float(exposure_name == "internal"),
    ]
    response = item.source["response"]
    blast_response = response["blast_response"]
    threshold = blast_response.get("peak_overpressure_threshold")
    impulse_threshold = blast_response.get("impulse_threshold")
    duration_threshold = blast_response.get("duration_threshold")
    realized = realized or {}
    thickness_value = realized.get("equivalent_thickness_m")
    vulnerable_value = realized.get("vulnerable_area_ratio")
    threshold_value = realized.get("blast_peak_threshold_pa")
    peak_value, peak_valid = _optional_parameter_si(threshold, "pressure")
    impulse_value, impulse_valid = _optional_parameter_si(impulse_threshold, "impulse")
    duration_value, duration_valid = _optional_parameter_si(duration_threshold, "duration")
    if threshold_value is not None:
        peak_value, peak_valid = float(threshold_value), True
    rotation = geometry.rotation.reshape(-1) if rotation_valid else np.zeros(9, dtype=np.float64)
    values = [
        *shape,
        *map(float, geometry.center),
        *map(float, geometry.representative_size()),
        *map(float, axis),
        *map(float, rotation),
        *box_size,
        *cylinder_size,
        *polygon_summary,
        float(thickness_value) if thickness_value is not None else length_to_m(response["equivalent_thickness"]),
        float(vulnerable_value) if vulnerable_value is not None else ratio_value(response["vulnerable_area_ratio"]),
        peak_value,
        float(realized.get("blast_impulse_threshold_pa_s")) if realized.get("blast_impulse_threshold_pa_s") is not None else impulse_value,
        float(realized.get("blast_duration_threshold_s")) if realized.get("blast_duration_threshold_s") is not None else duration_value,
        *exposure,
    ]
    mask = [True] * len(values)
    mask[12:21] = [rotation_valid] * 9
    mask[21:30] = shape_parameter_mask
    mask[32] = peak_valid
    mask[33] = impulse_valid or realized.get("blast_impulse_threshold_pa_s") is not None
    mask[34] = duration_valid or realized.get("blast_duration_threshold_s") is not None
    if len(values) != len(COMPONENT_FEATURE_NAMES):
        raise RuntimeError("component feature implementation and contract are inconsistent")
    return values, mask


def _armor_features(item, realized: dict[str, float | None] | None = None) -> tuple[list[float], list[bool]]:
    geometry = item.geometry
    if not isinstance(geometry, ExtrudedPolygonGeometry):
        raise InputValidationError("armor geometry must be an extruded polygon")
    response = item.source["response"]
    blast_response = response["blast_response"]
    realized = realized or {}
    vertices = geometry.vertices
    shifted = np.roll(vertices, -1, axis=0)
    area_vector = 0.5 * np.sum(np.cross(vertices, shifted), axis=0)
    area = abs(float(np.dot(area_vector, geometry.direction)))
    perimeter = float(np.sum(np.linalg.norm(shifted - vertices, axis=1)))
    peak_value, peak_valid = _optional_parameter_si(
        blast_response.get("peak_overpressure_threshold"), "pressure"
    )
    impulse_value, impulse_valid = _optional_parameter_si(
        blast_response.get("impulse_threshold"), "impulse"
    )
    duration_value, duration_valid = _optional_parameter_si(
        blast_response.get("duration_threshold"), "duration"
    )
    if realized.get("blast_peak_threshold_pa") is not None:
        peak_value, peak_valid = float(realized["blast_peak_threshold_pa"]), True
    values = [
        *map(float, geometry.center),
        *map(float, geometry.representative_size()),
        *map(float, geometry.direction),
        float(geometry.distance), area, perimeter, float(len(vertices)),
        float(realized.get("equivalent_thickness_m")) if realized.get("equivalent_thickness_m") is not None else length_to_m(response["equivalent_thickness"]),
        float(realized.get("vulnerable_area_ratio")) if realized.get("vulnerable_area_ratio") is not None else ratio_value(response["vulnerable_area_ratio"]),
        peak_value,
        float(realized.get("blast_impulse_threshold_pa_s")) if realized.get("blast_impulse_threshold_pa_s") is not None else impulse_value,
        float(realized.get("blast_duration_threshold_s")) if realized.get("blast_duration_threshold_s") is not None else duration_value,
    ]
    mask = [True] * len(values)
    mask[15] = peak_valid
    mask[16] = impulse_valid or realized.get("blast_impulse_threshold_pa_s") is not None
    mask[17] = duration_valid or realized.get("blast_duration_threshold_s") is not None
    return values, mask


def _armor_path_features(case: dict, engine: DamageEngine) -> tuple[np.ndarray, np.ndarray]:
    components = engine.compiled_target.components
    armor = engine.compiled_target.armor
    values = np.zeros((len(components), len(armor), len(ARMOR_PATH_FEATURE_NAMES)), dtype=np.float32)
    masks = np.zeros_like(values, dtype=bool)
    masks[..., 0] = True
    origin = np.asarray(case["encounter"]["detonation_position_m"], dtype=np.float64)
    armor_index = {item.object_id: index for index, item in enumerate(armor)}
    for component_index, component in enumerate(components):
        ray = component.geometry.center - origin
        distance = float(np.linalg.norm(ray))
        if distance <= 1.0e-12:
            continue
        direction = ray / distance
        for hit in engine.compiled_target.ray_hits(
            origin, direction, maximum_distance_m=max(0.0, distance - 1.0e-9)
        ):
            if hit.object_id not in armor_index:
                continue
            index = armor_index[hit.object_id]
            values[component_index, index] = [
                1.0,
                float(hit.entry_t),
                float(hit.path_length_m),
                abs(float(np.dot(direction, hit.entry_normal))),
            ]
            masks[component_index, index] = True
    return values, masks


def _state_names(engine: DamageEngine) -> list[str]:
    values: list[str] = []
    for item in engine.compiled_target.components:
        for state in item.source["damage_states"]:
            if state != "intact" and state not in values:
                values.append(state)
    return values


def _physical_case_key(case: dict) -> str:
    """Hash inputs that can change the learned mapping, excluding MC controls.

    Case IDs, aleatory seeds and requested replicate counts must not create new
    groups: doing so leaks near-identical Monte Carlo repeats across splits.
    """
    from .learning_input_identity import learning_input_fingerprint
    return learning_input_fingerprint(case)


def _split_name(
    case: dict,
    *,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[str, str, str, str]:
    metadata = case.get("metadata") or {}
    leakage_key = _physical_case_key(case)
    frozen_split = metadata.get("frozen_split")
    if frozen_split is not None:
        registry_hash = metadata.get("split_registry_hash")
        group_id = metadata.get("split_group")
        sampling = metadata.get("sampling")
        if frozen_split not in {"train", "validation", "test"}:
            raise InputValidationError("frozen_split must be train, validation or test")
        if (
            not is_sha256_identity(
                registry_hash, acceptance="legacy_prefix_length"
            )
            or not isinstance(group_id, str)
            or not group_id
            or not isinstance(sampling, dict)
            or sampling.get("lineage_id") != group_id
            or sampling.get("split_registry_hash") != registry_hash
            or sampling.get("frozen_split") != frozen_split
        ):
            raise InputValidationError(
                "explicit split assignment is not bound to a frozen split registry and lineage"
            )
        if sampling.get("role") == "adaptive_train" and frozen_split != "train":
            raise InputValidationError("adaptive_train samples cannot enter validation or test")
        from .sampling_validation import load_verified_split_registry, validate_registry_assignments
        registry = load_verified_split_registry(
            sampling.get("split_registry_document"), expected_hash=registry_hash,
            expected_id=sampling.get("split_registry_id"),
        )
        validate_registry_assignments(registry, [sampling])
        return group_id, "frozen_split_registry", leakage_key, str(frozen_split)
    if metadata.get("split_group") is not None:
        group_source = "split_group"
        group_id = str(metadata["split_group"])
    elif metadata.get("condition_cluster") is not None:
        group_source = "condition_cluster"
        group_id = str(metadata["condition_cluster"])
    else:
        group_source = "physical_case_hash"
        group_id = leakage_key
    digest = int(
        content_hash({"split_seed": int(seed), "group_id": group_id}).split(":", 1)[1][:16],
        16,
    ) / float(16**16)
    train_boundary = ratios[0]
    validation_boundary = ratios[0] + ratios[1]
    if digest < train_boundary:
        split = "train"
    elif digest < validation_boundary:
        split = "validation"
    else:
        split = "test"
    return group_id, group_source, leakage_key, split


def _walk_parameters(value: Any):
    if isinstance(value, dict):
        if {"value", "unit", "source_id", "validation_status"}.issubset(value):
            yield value
        for child in value.values():
            yield from _walk_parameters(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_parameters(child)


_SAMPLING_REQUIRED_FIELDS_V1_0 = {
    "schema_version", "candidate_id", "lineage_id", "design_group_id", "round",
    "stream", "role", "stratum", "selection_probability", "design_weight_raw",
    "role_names", "stratum_names",
    "weight_mask", "sampling_spec_id", "sampling_spec_hash", "p0_id", "p0_hash",
    "label_spec_id", "label_spec_hash", "stratification_spec_id",
    "stratification_spec_hash", "split_registry_id", "split_registry_hash",
    "frozen_split", "epistemic_group_id", "screen_seed", "formal_seed",
    "damage_tree_definition_hash", "target_hash", "physics_registry_hash",
    "engine_code_hash",
    "split_registry_document", "label_spec_document",
}
_SAMPLING_REQUIRED_FIELDS_V1_1 = _SAMPLING_REQUIRED_FIELDS_V1_0 | {
    "sampling_spec_document", "training_design_document", "training_design_hash",
}
_SAMPLING_REQUIRED_FIELDS_V1_2 = _SAMPLING_REQUIRED_FIELDS_V1_1 | {
    "p0_density", "adaptive_density", "proposal_density",
    "p0_hierarchy_probability", "proposal_hierarchy_probability",
    "acquisition_model_hash", "adaptive_proposal_hash", "proposal_hash",
    "round_lineage_id", "parent_round_lineage_id", "candidate_case_hash",
    "learning_input_hash", "selected_case_set_hash", "design_weight_formula",
}
_SAMPLING_REQUIRED_FIELDS_V1_3 = _SAMPLING_REQUIRED_FIELDS_V1_1 | {
    "contact_conditioning",
}
_SAMPLING_REQUIRED_FIELDS_BY_VERSION = {
    "1.0.0": _SAMPLING_REQUIRED_FIELDS_V1_0,
    "1.1.0": _SAMPLING_REQUIRED_FIELDS_V1_1,
    "1.2.0": _SAMPLING_REQUIRED_FIELDS_V1_2,
    "1.3.0": _SAMPLING_REQUIRED_FIELDS_V1_3,
    "1.4.0": _SAMPLING_REQUIRED_FIELDS_V1_3,
}


def _sampling_context(cases: list[dict]) -> list[dict] | None:
    records = [((case.get("metadata") or {}).get("sampling")) for case in cases]
    if all(record is None for record in records):
        return None
    if any(not isinstance(record, dict) for record in records):
        raise InputValidationError(
            "sampling-aware datasets require a sampling envelope on every input case"
        )
    fixed_fields = {
        "schema_version",
        "sampling_spec_id", "sampling_spec_hash", "p0_id", "p0_hash",
        "label_spec_id", "label_spec_hash", "stratification_spec_id",
        "stratification_spec_hash", "split_registry_id", "split_registry_hash",
        "damage_tree_definition_hash", "target_hash", "physics_registry_hash",
        "engine_code_hash", "role_names", "stratum_names",
        "split_registry_document", "label_spec_document",
    }
    expected_fixed: dict[str, Any] | None = None
    candidate_ids: set[str] = set()
    validated: list[dict] = []
    for case, raw in zip(cases, records):
        assert isinstance(raw, dict)
        version = raw.get("schema_version")
        required_fields = _SAMPLING_REQUIRED_FIELDS_BY_VERSION.get(version)
        if required_fields is None:
            raise InputValidationError("unsupported sampling envelope schema version")
        if set(raw) != required_fields:
            raise InputValidationError(
                "sampling envelope has missing or unknown fields",
                context={
                    "case_id": case.get("case_id"),
                    "schema_version": version,
                    "missing": sorted(required_fields - set(raw)),
                    "unknown": sorted(set(raw) - required_fields),
                },
            )
        record = dict(raw)
        if any(type(record[field]) not in {int, float} for field in ("selection_probability", "design_weight_raw")):
            raise InputValidationError("sampling probabilities and raw weights must be numeric scalars")
        probability = float(record["selection_probability"])
        weight = float(record["design_weight_raw"])
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise InputValidationError("sampling selection probability must lie in (0, 1]")
        if not math.isfinite(weight) or weight <= 0.0:
            raise InputValidationError("raw sampling design weight must be finite and positive")
        if type(record["weight_mask"]) is not bool:
            raise InputValidationError("sampling weight mask must be boolean")
        if version == "1.2.0":
            adaptive_numbers = (
                "p0_density", "adaptive_density", "proposal_density",
                "p0_hierarchy_probability", "proposal_hierarchy_probability",
            )
            if any(type(record[field]) not in {int, float} for field in adaptive_numbers):
                raise InputValidationError("adaptive sampling density factors must be numeric scalars")
            factors = {field: float(record[field]) for field in adaptive_numbers}
            if any(not math.isfinite(value) or value <= 0.0 for value in factors.values()):
                raise InputValidationError("adaptive sampling density factors must be finite and positive")
            if (
                factors["p0_hierarchy_probability"] > 1.0
                or factors["proposal_hierarchy_probability"] > 1.0
            ):
                raise InputValidationError("adaptive hierarchy probabilities must lie in (0, 1]")
            expected_weight = (
                factors["p0_density"] * factors["p0_hierarchy_probability"]
                / (
                    factors["proposal_density"]
                    * factors["proposal_hierarchy_probability"]
                    * probability
                )
            )
            if not math.isclose(weight, expected_weight, rel_tol=1.0e-12, abs_tol=1.0e-12):
                raise InputValidationError("adaptive raw design weight violates the general proposal formula")
        elif not math.isclose(
            weight, 1.0 / probability, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise InputValidationError("raw design weight must equal inverse inclusion probability")
        if type(record["round"]) is not int or record["round"] < 0:
            raise InputValidationError("sampling round must be a nonnegative integer")
        if any(type(record[field]) is not int or record[field] < 0 for field in ("formal_seed", "screen_seed")):
            raise InputValidationError("sampling seeds must be nonnegative integers")
        if any(not isinstance(record[field], str) or not record[field] for field in (
            "candidate_id", "lineage_id", "design_group_id", "stream", "role", "stratum",
            "sampling_spec_id", "sampling_spec_hash", "p0_id", "p0_hash",
            "label_spec_id", "label_spec_hash", "stratification_spec_id",
            "stratification_spec_hash", "split_registry_id", "split_registry_hash",
            "frozen_split", "epistemic_group_id", "damage_tree_definition_hash",
            "target_hash", "physics_registry_hash", "engine_code_hash",
        )):
            raise InputValidationError("sampling string identities must be nonempty")
        if version == "1.2.0":
            adaptive_identity_fields = (
                "acquisition_model_hash", "adaptive_proposal_hash", "proposal_hash",
                "round_lineage_id", "parent_round_lineage_id", "candidate_case_hash",
                "learning_input_hash", "selected_case_set_hash", "design_weight_formula",
            )
            if any(
                not isinstance(record[field], str) or not record[field]
                for field in adaptive_identity_fields
            ):
                raise InputValidationError("adaptive sampling identities must be nonempty strings")
            for field in (
                "acquisition_model_hash", "adaptive_proposal_hash", "proposal_hash",
                "candidate_case_hash", "learning_input_hash", "selected_case_set_hash",
            ):
                value = record[field]
                if not is_sha256_identity(value):
                    raise InputValidationError(f"adaptive sampling {field} must be a sha256 identity")
            if (
                record["design_weight_formula"]
                != "p0_density*p0_hierarchy_probability/(proposal_density*proposal_hierarchy_probability*inclusion_probability)"
                or record["round"] < 1
                or record["stream"] != "adaptive_train"
                or record["role"] != "adaptive_train"
                or record["frozen_split"] != "train"
                or record["weight_mask"] is not True
            ):
                raise InputValidationError("adaptive sampling role, split or weight contract is invalid")
            from .learning_input_identity import learning_input_fingerprint

            if learning_input_fingerprint(case) != record["learning_input_hash"]:
                raise InputValidationError("adaptive learning-input hash differs from the executed case")
        for field, active in (("role_names", record["role"]), ("stratum_names", record["stratum"])):
            names = record[field]
            if (
                not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not name for name in names)
                or len(names) != len(set(names)) or active not in names
            ):
                raise InputValidationError(f"sampling {field} must be a unique vocabulary containing the active value")
        if record["candidate_id"] in candidate_ids:
            raise InputValidationError("sampling candidate_id values must be unique")
        candidate_ids.add(record["candidate_id"])
        if int(record["formal_seed"]) != int(case["analysis"]["aleatory_seed"]):
            raise InputValidationError("formal seed in sampling envelope differs from the executed case")
        active_fixed_fields = set(fixed_fields)
        if version in {"1.1.0", "1.2.0", "1.3.0", "1.4.0"}:
            active_fixed_fields.update(
                {"sampling_spec_document", "training_design_document", "training_design_hash"}
            )
        if version == "1.2.0":
            active_fixed_fields.update(
                {
                    "acquisition_model_hash", "adaptive_proposal_hash", "proposal_hash",
                    "round_lineage_id", "parent_round_lineage_id",
                    "selected_case_set_hash", "design_weight_formula",
                }
            )
        fixed = {field: record[field] for field in active_fixed_fields}
        if expected_fixed is None:
            expected_fixed = fixed
        elif fixed != expected_fixed:
            raise InputValidationError("sampling-aware dataset mixes incompatible plan identities")
        validated.append(record)
    from .sampling_validation import load_verified_split_registry, validate_candidate_set, validate_registry_assignments
    registry = load_verified_split_registry(
        validated[0]["split_registry_document"], expected_hash=validated[0]["split_registry_hash"],
        expected_id=validated[0]["split_registry_id"],
    )
    validate_candidate_set(validated)
    validate_registry_assignments(registry, validated)
    from .label_spec import LabelSpec
    spec = LabelSpec(validated[0]["label_spec_document"])
    if spec.hash != validated[0]["label_spec_hash"] or spec.spec_id != validated[0]["label_spec_id"]:
        raise InputValidationError("label specification document differs from the declared identity")
    if spec.data["source"]["damage_tree_definition_hash"] != validated[0]["damage_tree_definition_hash"]:
        raise InputValidationError("label specification document is bound to a different tree definition")
    if validated[0]["schema_version"] in {
        "1.1.0",
        "1.2.0",
        "1.3.0",
        "1.4.0",
    }:
        from .sampling_spec import SamplingSpec

        sampling_spec = SamplingSpec(validated[0]["sampling_spec_document"])
        if (
            sampling_spec.hash != validated[0]["sampling_spec_hash"]
            or sampling_spec.spec_id != validated[0]["sampling_spec_id"]
        ):
            raise InputValidationError(
                "sampling specification document differs from the declared identity"
            )
        if (
            sampling_spec.training_design_document
            != validated[0]["training_design_document"]
            or sampling_spec.training_design_hash != validated[0]["training_design_hash"]
        ):
            raise InputValidationError(
                "training design document differs from the declared identity"
            )
        if sampling_spec.has_contact_conditioning:
            declaration = sampling_spec.contact_conditioning
            if declaration is None:
                raise InputValidationError(
                    "contact-conditioned sampling specification lacks its declaration"
                )
            expected_binding = {
                "contract_id": declaration["contract_id"],
                "contract_hash": declaration["contract_hash"],
                "mode": declaration["mode"],
                "estimand": declaration["estimand"],
                "selection_declaration_hash": declaration[
                    "selection_declaration_hash"
                ],
            }
            for record in validated:
                binding = record.get("contact_conditioning")
                if not isinstance(binding, dict) or {
                    key: binding.get(key) for key in expected_binding
                } != expected_binding:
                    raise InputValidationError(
                        "case contact-conditioning binding differs from SamplingSpec"
                    )
                eligibility = binding.get("eligibility_record")
                if (
                    not isinstance(eligibility, dict)
                    or eligibility.get("status")
                    != "eligible_contact_conditioned"
                    or eligibility.get("formal_label_allowed") is not True
                    or eligibility.get("contract_hash")
                    != declaration["contract_hash"]
                ):
                    raise InputValidationError(
                        "formal case lacks successful S-CONTACT eligibility evidence"
                    )
    return validated


class DatasetWriter:
    def __init__(
        self,
        output_directory: str | Path,
        *,
        split_seed: int = 0,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        tensor_compression: str = "compressed",
        result_compression: str = "auto",
    ):
        self.output_directory = Path(output_directory)
        self.split_seed = int(split_seed)
        if self.split_seed < 0:
            raise ValueError("split_seed must be non-negative")
        self.split_ratios = tuple(float(value) for value in split_ratios)
        if len(self.split_ratios) != 3 or any(value < 0.0 for value in self.split_ratios):
            raise ValueError("split_ratios must contain three non-negative values")
        if not math.isclose(sum(self.split_ratios), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("split_ratios must sum to one")
        if tensor_compression not in {"compressed", "none"}:
            raise ValueError("tensor_compression must be 'compressed' or 'none'")
        self.tensor_compression = tensor_compression
        if result_compression not in {"auto", "none", "gzip"}:
            raise ValueError("result_compression must be 'auto', 'none', or 'gzip'")
        self.result_compression = result_compression

    def run_cases(
        self,
        engine: DamageEngine,
        cases: Iterable[dict],
        *,
        fail_closed: bool = False,
        workers: int = 1,
        chunksize: int | None = None,
        cases_base_directory: str | Path | None = None,
    ) -> dict:
        from .input_identity import freeze_case_inputs
        prepared_cases = (engine.prepare_case(case) for case in cases)
        with freeze_case_inputs(
            prepared_cases, cases_base_directory=cases_base_directory
        ) as frozen:
            return self._run_frozen_cases(
                engine, list(frozen), fail_closed=fail_closed, workers=workers,
                chunksize=chunksize, input_snapshot=frozen.metadata,
            )

    def _run_frozen_cases(
        self, engine: DamageEngine, cases: Iterable[dict], *, fail_closed: bool,
        workers: int, chunksize: int | None, input_snapshot: dict,
        _captured_inputs: bool = False,
        _case_hash: Callable[[dict], str] | None = None,
    ) -> dict:
        case_list = list(cases)
        if not case_list:
            raise ValueError("cases must contain at least one case")
        if workers < 1:
            raise ValueError("workers must be at least one")
        if chunksize is not None and chunksize < 1:
            raise ValueError("chunksize must be at least one")
        case_ids = [case.get("case_id") for case in case_list]
        if any(not isinstance(value, str) or not value for value in case_ids):
            raise InputValidationError("dataset case_id values must be nonempty strings")
        non_null_ids = [str(value) for value in case_ids if value is not None]
        if len(non_null_ids) != len(set(non_null_ids)):
            raise InputValidationError("dataset case_id values must be unique")
        effective_chunksize = chunksize or max(1, len(case_list) // (workers * 4))
        if workers == 1:
            results = [_run_case(engine, case, fail_closed) for case in case_list]
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_worker,
                initargs=(engine.target_data, engine.registry.data),
            ) as executor:
                payloads = list(
                    executor.map(
                        _run_case_in_worker,
                        ((case, fail_closed) for case in case_list),
                        chunksize=effective_chunksize,
                    )
                )
            results = [EngineResult(payload) for payload in payloads]
        generation = {
            "workers": int(workers),
            "chunksize": int(effective_chunksize),
            "failure_policy": "fail_fast" if fail_closed else "isolate_and_record",
            "input_order_preserved": True,
            "tensor_compression": self.tensor_compression,
            "result_compression": self.result_compression,
            "input_snapshot": input_snapshot,
        }
        if not _captured_inputs:
            return self.write(
                engine,
                case_list,
                results,
                generation=generation,
            )

        # ShardedDatasetWriter uses this private branch only after an internal
        # compact sampling source has copied and fully verified every finite
        # input.  Preserve the normal per-shard exclusive lock and all result,
        # case-id and schema validation, but do not serialize the hydrated
        # O(N) split registry into another temporary input spool.
        from .input_identity import capture_environment_identity, single_writer_lock

        generation.setdefault("environment", capture_environment_identity())
        with single_writer_lock(self.output_directory):
            return self._write_frozen(
                engine,
                case_list,
                results,
                generation=generation,
                _case_hash=_case_hash,
            )

    def write(
        self,
        engine: DamageEngine,
        cases: list[dict],
        results: list[EngineResult],
        *,
        generation: dict | None = None,
        cases_base_directory: str | Path | None = None,
    ) -> dict:
        from copy import deepcopy
        from .input_identity import FrozenCase, capture_environment_identity, case_identity_hash, freeze_case_inputs
        if len(cases) != len(results):
            raise InputValidationError("cases and results must have equal length")

        def preflight_hash(case: dict) -> str:
            warhead = case.get("warhead", {})
            if not isinstance(warhead, dict):
                raise InputValidationError("precomputed case warhead must be an object")
            source = warhead.get("fragment_source", {})
            if not isinstance(source, dict):
                raise InputValidationError("precomputed case fragment_source must be an object")
            reference = source.get("fragment_list_ref")
            if reference is not None and (not isinstance(reference, str) or not reference.strip()):
                raise InputValidationError("precomputed case fragment_list_ref must be a nonempty path string")
            if reference is not None and not isinstance(case, FrozenCase) and not Path(reference).is_absolute():
                if cases_base_directory is None:
                    raise InputValidationError("relative external input references require cases_base_directory")
                case = deepcopy(case)
                case["warhead"]["fragment_source"]["fragment_list_ref"] = str(Path(cases_base_directory) / reference)
            return case_identity_hash(case)

        # Precomputed-entry preflight is read-only, including external hashing.
        # Repeat it against the frozen snapshot to catch edits during capture.
        validate_precomputed_results(
            engine, cases, [result.to_dict() for result in results], case_hash=preflight_hash,
            engine_code_hash=ENGINE_CODE_HASH, fragment_metrics=_FRAGMENT_AUXILIARY_KEYS,
            blast_metrics=_BLAST_AUXILIARY_KEYS, armor_metrics=_ARMOR_BLAST_LABEL_KEYS,
        )
        with freeze_case_inputs(cases, cases_base_directory=cases_base_directory) as frozen:
            execution = dict(generation or {
                "workers": 1, "chunksize": 1, "failure_policy": "precomputed_results",
                "input_order_preserved": True, "tensor_compression": self.tensor_compression,
                "result_compression": self.result_compression,
            })
            execution.setdefault("input_snapshot", frozen.metadata)
            execution.setdefault("environment", capture_environment_identity())
            from .input_identity import single_writer_lock
            with single_writer_lock(self.output_directory):
                return self._write_frozen(engine, list(frozen), results, generation=execution)

    def _write_frozen(
        self, engine: DamageEngine, cases: list[dict], results: list[EngineResult], *,
        generation: dict,
        _case_hash: Callable[[dict], str] | None = None,
    ) -> dict:
        if len(cases) != len(results):
            raise InputValidationError("cases and results must have equal length")
        from .input_identity import case_identity_hash
        hash_case = _case_hash or case_identity_hash
        payloads = [result.to_dict() for result in results]
        validate_precomputed_results(
            engine, cases, payloads, case_hash=hash_case,
            engine_code_hash=ENGINE_CODE_HASH,
            fragment_metrics=_FRAGMENT_AUXILIARY_KEYS, blast_metrics=_BLAST_AUXILIARY_KEYS,
            armor_metrics=_ARMOR_BLAST_LABEL_KEYS,
        )
        sampling_records = _sampling_context(cases)
        sampling_enabled = sampling_records is not None
        model_input_definition = None
        sampling_spec_document = (
            sampling_records[0].get("sampling_spec_document")
            if sampling_records
            else None
        )
        if (
            isinstance(sampling_spec_document, dict)
            and sampling_spec_document.get("schema_version")
            in {"1.2.0", "1.3.0", "1.4.0"}
        ):
            candidate_definition = model_input_definition_for_sampling_contract(
                sampling_spec_document.get("model_input_contract")
            )
            if candidate_definition is not None:
                try:
                    for case in cases:
                        candidate_definition.vector(case)
                except (KeyError, TypeError, ValueError):
                    # Artificial/legacy SamplingSpec fixtures can use an
                    # abstract case schema. They keep their historical tensor
                    # contract; only executable cases enter contract 1.8.
                    pass
                else:
                    model_input_definition = candidate_definition
        versioned_model_contract_enabled = model_input_definition is not None
        fixed_scenario_hash: str | None = None
        if versioned_model_contract_enabled:
            fixed_scenario_hashes = {
                model_input_definition.fixed_scenario_fingerprint(case)
                for case in cases
            }
            if len(fixed_scenario_hashes) != 1:
                raise InputValidationError(
                    "versioned model-input archives require exactly one frozen scenario identity",
                    context={
                        "fixed_scenario_hashes": sorted(fixed_scenario_hashes),
                        "case_count": len(cases),
                    },
                )
            fixed_scenario_hash = next(iter(fixed_scenario_hashes))
        label_spec = None
        if sampling_records:
            from .label_spec import LabelSpec
            first_sampling = sampling_records[0]
            label_spec = LabelSpec(first_sampling["label_spec_document"])
            label_spec.validate_output_vocabulary(engine.damage_tree.outputs)
            if label_spec.data["source"]["damage_tree_revision"] != engine.damage_tree.version:
                raise InputValidationError("label specification source revision differs from the current tree")
            expected_tree_hash = content_hash(engine.target_data["damage_tree"])
            identity_checks = {
                "target_hash": engine.target_hash,
                "physics_registry_hash": engine.registry.hash,
                "engine_code_hash": ENGINE_CODE_HASH,
                "damage_tree_definition_hash": expected_tree_hash,
            }
            for field, expected in identity_checks.items():
                if first_sampling[field] != expected:
                    raise InputValidationError(
                        "sampling envelope is bound to a different teacher identity",
                        context={"field": field, "declared": first_sampling[field], "actual": expected},
                    )
            all_input_splits = []
            for case in cases:
                group_id, _, leakage_key, split = _split_name(case, seed=self.split_seed, ratios=self.split_ratios)
                all_input_splits.append({"group_id": group_id, "leakage_key": leakage_key, "split": split})
            if not self._check_split_leakage(all_input_splits):
                raise InputValidationError("duplicate declared learning inputs or groups cross split boundaries")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        incomplete_path = self.output_directory / ".dataset.incomplete"
        incomplete_path.write_text(
            json.dumps(
                {"started_at": datetime.now(timezone.utc).isoformat(), "case_count": len(cases)},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        result_compression = self.result_compression
        if result_compression == "auto":
            result_compression = "gzip" if sampling_enabled else "none"
        raw_path, raw_format = write_result_ledger(
            self.output_directory,
            (result.to_dict() for result in results),
            compression=result_compression,
        )
        generation["result_compression"] = result_compression
        valid_pairs = [
            (case, payload)
            for case, payload in zip(cases, payloads)
            if payload.get("status") == "ok"
        ]
        valid_sampling_records = (
            [record for record, payload in zip(sampling_records, payloads) if payload.get("status") == "ok"]
            if sampling_records is not None
            else []
        )
        refused = [payload for payload in payloads if payload.get("status") != "ok"]
        state_names = _state_names(engine)
        mechanism_names = ["fragment", "blast", "combined"]
        output_names = list(engine.damage_tree.outputs)
        evaluation_standard_ids = list(engine.damage_tree.standards)
        output_index = {output_id: index for index, output_id in enumerate(output_names)}
        standard_index = {
            standard_id: index
            for index, standard_id in enumerate(evaluation_standard_ids)
        }
        standard_output_mask = np.zeros(
            (len(evaluation_standard_ids), len(output_names)), dtype=bool
        )
        for standard_id, standard_outputs in engine.damage_tree.standards.items():
            for output_id in standard_outputs:
                standard_output_mask[
                    standard_index[standard_id], output_index[output_id]
                ] = True
        component_count = len(engine.compiled_target.components)
        armor_count = len(engine.compiled_target.armor)
        case_feature_rows: list[list[float]] = []
        case_feature_masks: list[list[bool]] = []
        component_features = []
        component_feature_masks = []
        armor_features = []
        armor_feature_masks = []
        armor_labels = []
        armor_label_stddev = []
        armor_label_ci95 = []
        armor_label_masks = []
        armor_path_features = []
        armor_path_feature_masks = []
        component_labels = []
        component_label_stddev = []
        component_label_ci95 = []
        component_label_masks = []
        component_frechet_labels = []
        component_frechet_stddev = []
        component_frechet_ci95 = []
        component_frechet_masks = []
        component_auxiliary_labels = []
        component_auxiliary_stddev = []
        component_auxiliary_ci95 = []
        component_auxiliary_masks = []
        system_labels = []
        system_label_stddev = []
        system_label_ci95 = []
        system_label_masks = []
        system_source_available_masks = []
        requested_standard_masks = []
        case_rows: list[dict] = []
        component_rows: list[dict] = []
        armor_rows: list[dict] = []
        armor_label_rows: list[dict] = []
        label_rows: list[dict] = []
        auxiliary_rows: list[dict] = []
        tree_rows: list[dict] = []
        edge_rows: list[dict] = []
        split_rows: list[dict] = []
        for case, result in valid_pairs:
            case_x, case_mask = _case_features(case, engine)
            case_feature_rows.append(case_x)
            case_feature_masks.append(case_mask)
            component_feature_template = [
                _component_features(
                    item,
                    result["parameter_realization"].get(item.object_id),
                )
                for item in engine.compiled_target.components
            ]
            component_features.append([item[0] for item in component_feature_template])
            component_feature_masks.append([item[1] for item in component_feature_template])
            armor_feature_template = [
                _armor_features(
                    item,
                    result["parameter_realization"].get(item.object_id),
                )
                for item in engine.compiled_target.armor
            ]
            armor_features.append([item[0] for item in armor_feature_template])
            armor_feature_masks.append([item[1] for item in armor_feature_template])
            path_x, path_mask = _armor_path_features(case, engine)
            armor_path_features.append(path_x)
            armor_path_feature_masks.append(path_mask)
            armor_case_labels = np.zeros(
                (armor_count, len(ARMOR_LABEL_NAMES)), dtype=np.float32
            )
            armor_case_stddev = np.zeros_like(armor_case_labels)
            armor_case_ci95 = np.zeros(
                (*armor_case_labels.shape, 2), dtype=np.float32
            )
            armor_case_mask = np.zeros_like(armor_case_labels, dtype=bool)
            labels = np.zeros((component_count, len(mechanism_names), len(state_names)), dtype=np.float32)
            label_stddev = np.zeros_like(labels)
            label_ci95 = np.zeros((*labels.shape, 2), dtype=np.float32)
            label_mask = np.zeros_like(labels, dtype=bool)
            frechet = np.zeros(
                (component_count, len(state_names), len(FRECHET_BOUND_NAMES)),
                dtype=np.float32,
            )
            frechet_stddev = np.zeros_like(frechet)
            frechet_ci95 = np.zeros((*frechet.shape, 2), dtype=np.float32)
            frechet_mask = np.zeros_like(frechet, dtype=bool)
            auxiliary = np.zeros((component_count, len(AUXILIARY_LABEL_NAMES)), dtype=np.float32)
            auxiliary_stddev = np.zeros_like(auxiliary)
            auxiliary_ci95 = np.zeros((*auxiliary.shape, 2), dtype=np.float32)
            auxiliary_mask = np.zeros_like(auxiliary, dtype=bool)
            enabled_mechanisms = set(case["analysis"]["mechanisms"])
            for component_index, item in enumerate(engine.compiled_target.components):
                component_result = result["components"][item.object_id]
                states_available = [
                    state for state in item.source["damage_states"] if state != "intact"
                ]
                for mechanism_index, mechanism in enumerate(mechanism_names):
                    for state_index, state in enumerate(state_names):
                        if state in states_available:
                            labels[component_index, mechanism_index, state_index] = component_result[
                                "damage_states"
                            ][mechanism][state]["mean"]
                            label_stddev[component_index, mechanism_index, state_index] = component_result[
                                "damage_states"
                            ][mechanism][state]["stddev"]
                            label_ci95[component_index, mechanism_index, state_index] = [
                                component_result["damage_states"][mechanism][state]["ci95_lower"],
                                component_result["damage_states"][mechanism][state]["ci95_upper"],
                            ]
                            label_mask[component_index, mechanism_index, state_index] = (
                                {"fragment", "blast"}.issubset(enabled_mechanisms)
                                if mechanism == "combined" else mechanism in enabled_mechanisms
                            )
                for state_index, state in enumerate(state_names):
                    if state not in states_available:
                        continue
                    combined_stats = component_result["damage_states"]["combined"][state]
                    for bound_index, bound in enumerate(FRECHET_BOUND_NAMES):
                        stats = combined_stats[f"frechet_union_{bound}_bound"]
                        frechet[component_index, state_index, bound_index] = stats["mean"]
                        frechet_stddev[component_index, state_index, bound_index] = stats[
                            "stddev"
                        ]
                        frechet_ci95[component_index, state_index, bound_index] = [
                            stats["ci95_lower"],
                            stats["ci95_upper"],
                        ]
                        frechet_mask[component_index, state_index, bound_index] = {
                            "fragment",
                            "blast",
                        }.issubset(enabled_mechanisms)
                auxiliary_items = [
                    ("fragment", key, component_result["fragment_metrics"][key])
                    for key in _FRAGMENT_AUXILIARY_KEYS
                ] + [
                    ("blast", key, component_result["blast_metrics"][key])
                    for key in _BLAST_AUXILIARY_KEYS
                ]
                for auxiliary_index, (mechanism, key, stats) in enumerate(auxiliary_items):
                    auxiliary[component_index, auxiliary_index] = stats["mean"]
                    auxiliary_stddev[component_index, auxiliary_index] = stats["stddev"]
                    auxiliary_ci95[component_index, auxiliary_index] = [
                        stats["ci95_lower"], stats["ci95_upper"]
                    ]
                    is_available = mechanism in enabled_mechanisms
                    auxiliary_mask[component_index, auxiliary_index] = is_available
                    auxiliary_rows.append(
                        {
                            "case_id": case["case_id"],
                            "component_id": item.object_id,
                            "mechanism": mechanism,
                            "metric": AUXILIARY_LABEL_NAMES[auxiliary_index],
                            "unit": AUXILIARY_LABEL_UNITS[auxiliary_index],
                            "available": is_available,
                            **stats,
                        }
                    )
                component_rows.append(
                    {
                        "case_id": case["case_id"],
                        "component_id": item.object_id,
                        "component_name": item.source["name"],
                        "component_type": item.source["component_type"],
                        "features": component_feature_template[component_index][0],
                        "feature_mask": component_feature_template[component_index][1],
                        "material_id": item.response["material_id"],
                        "fragment_response_model_id": item.response["fragment_response_model_id"],
                        "blast_response_model_id": item.response["blast_response"]["model_id"],
                    }
                )
                for mechanism in mechanism_names:
                    for state in states_available:
                        stats = component_result["damage_states"][mechanism][state]
                        label_rows.append(
                            {
                                "case_id": case["case_id"],
                                "component_id": item.object_id,
                                "mechanism": mechanism,
                                "damage_state": state,
                                "available": ({"fragment", "blast"}.issubset(enabled_mechanisms)
                                              if mechanism == "combined" else mechanism in enabled_mechanisms),
                                **stats,
                            }
                        )
            for armor_index, item in enumerate(engine.compiled_target.armor):
                armor_rows.append(
                    {
                        "case_id": case["case_id"],
                        "armor_id": item.object_id,
                        "armor_name": item.source["name"],
                        "features": armor_feature_template[armor_index][0],
                        "feature_mask": armor_feature_template[armor_index][1],
                        "material_id": item.response["material_id"],
                        "fragment_response_model_id": item.response["fragment_response_model_id"],
                        "blast_response_model_id": item.response["blast_response"]["model_id"],
                    }
                )
                armor_result = result["armor"][item.object_id]
                armor_stats = [
                    armor_result["post_fragment_integrity"],
                    armor_result["post_combined_integrity"],
                    *[
                        armor_result["blast_metrics"][key]
                        for key in _ARMOR_BLAST_LABEL_KEYS
                    ],
                ]
                for label_index, (label_name, unit, stats) in enumerate(
                    zip(ARMOR_LABEL_NAMES, ARMOR_LABEL_UNITS, armor_stats)
                ):
                    if label_name == "post_fragment_integrity":
                        available = "fragment" in enabled_mechanisms
                    elif label_name == "post_combined_integrity":
                        available = bool(enabled_mechanisms)
                    else:
                        available = "blast" in enabled_mechanisms
                    armor_case_labels[armor_index, label_index] = stats["mean"]
                    armor_case_stddev[armor_index, label_index] = stats["stddev"]
                    armor_case_ci95[armor_index, label_index] = [
                        stats["ci95_lower"],
                        stats["ci95_upper"],
                    ]
                    armor_case_mask[armor_index, label_index] = available
                    armor_label_rows.append(
                        {
                            "case_id": case["case_id"],
                            "armor_id": item.object_id,
                            "metric": label_name,
                            "unit": unit,
                            "available": available,
                            **stats,
                        }
                    )
            armor_labels.append(armor_case_labels)
            armor_label_stddev.append(armor_case_stddev)
            armor_label_ci95.append(armor_case_ci95)
            armor_label_masks.append(armor_case_mask)
            component_labels.append(labels)
            component_label_stddev.append(label_stddev)
            component_label_ci95.append(label_ci95)
            component_label_masks.append(label_mask)
            component_frechet_labels.append(frechet)
            component_frechet_stddev.append(frechet_stddev)
            component_frechet_ci95.append(frechet_ci95)
            component_frechet_masks.append(frechet_mask)
            component_auxiliary_labels.append(auxiliary)
            component_auxiliary_stddev.append(auxiliary_stddev)
            component_auxiliary_ci95.append(auxiliary_ci95)
            component_auxiliary_masks.append(auxiliary_mask)
            system = [result["damage_tree"]["outputs"][name]["mean"] for name in output_names]
            system_stddev = [result["damage_tree"]["outputs"][name]["stddev"] for name in output_names]
            system_ci95 = [
                [
                    result["damage_tree"]["outputs"][name]["ci95_lower"],
                    result["damage_tree"]["outputs"][name]["ci95_upper"],
                ]
                for name in output_names
            ]
            system_labels.append(system)
            system_label_stddev.append(system_stddev)
            system_label_ci95.append(system_ci95)
            requested_standard_ids = list(case["analysis"]["evaluation_standard_ids"])
            unsupported_standards = sorted(
                set(requested_standard_ids) - set(evaluation_standard_ids)
            )
            if unsupported_standards:
                raise InputValidationError(
                    "dataset case requests evaluation standards not declared by the target: "
                    + ", ".join(unsupported_standards)
                )
            requested_mask = np.zeros(len(evaluation_standard_ids), dtype=bool)
            for standard_id in requested_standard_ids:
                requested_mask[standard_index[standard_id]] = True
            requested_standard_masks.append(requested_mask.tolist())
            from .label_spec import system_output_availability
            component_availability = {
                (component.object_id, mechanism, state): bool(label_mask[c, m, s])
                for c, component in enumerate(engine.compiled_target.components)
                for m, mechanism in enumerate(mechanism_names) for s, state in enumerate(state_names)
            }
            source_flags = system_output_availability(
                engine.damage_tree.nodes, output_names, enabled_mechanisms,
                component_availability=component_availability,
            )
            source_mask = np.asarray([source_flags[name] for name in output_names], dtype=bool)
            system_source_available_masks.append(source_mask.tolist())
            requested_outputs = (np.any(standard_output_mask[requested_mask], axis=0)
                                 if requested_standard_ids else np.zeros(len(output_names), dtype=bool))
            system_label_masks.append((requested_outputs & source_mask).tolist())
            group_id, group_source, leakage_key, split = _split_name(
                case, seed=self.split_seed, ratios=self.split_ratios
            )
            split_rows.append(
                {
                    "case_id": case["case_id"],
                    "group_id": group_id,
                    "group_source": group_source,
                    "leakage_key": leakage_key,
                    "split": split,
                }
            )
            case_row = {
                    "case_id": case["case_id"],
                    "target_id": result["target_id"],
                    "features": case_x,
                    "feature_mask": case_mask,
                    "replicate_count": result["replicate_count"],
                    "aleatory_seed": case["analysis"]["aleatory_seed"],
                    "epistemic_sample_id": case["analysis"]["epistemic_sample_id"],
                    "group_id": group_id,
                    "group_source": group_source,
                    "split": split,
                    "case_hash": result["provenance"]["case_hash"],
                }
            sampling_record = (case.get("metadata") or {}).get("sampling")
            if isinstance(sampling_record, dict):
                case_row["sampling"] = {
                    key: sampling_record[key]
                    for key in (
                        "candidate_id", "lineage_id", "round", "stream", "role", "stratum",
                        "selection_probability", "design_weight_raw", "sampling_spec_hash",
                        "label_spec_hash", "split_registry_hash",
                        "weight_mask", "epistemic_group_id", "frozen_split",
                    )
                }
            case_rows.append(case_row)
            for standard_id in requested_standard_ids:
                for output_name in engine.damage_tree.standards[standard_id]:
                    tree_rows.append(
                        {
                            "case_id": case["case_id"],
                            "evaluation_standard_id": standard_id,
                            "output_node": output_name,
                            "available": bool(source_flags[output_name]),
                            **result["damage_tree"]["outputs"][output_name],
                        }
                    )
        graph_node_ids = list(engine.compiled_target.component_order) + list(engine.damage_tree.order)
        graph_node_types = ["component"] * component_count + ["damage_tree"] * len(
            engine.damage_tree.order
        )
        graph_node_index = {
            (node_type, identifier): index
            for index, (node_type, identifier) in enumerate(zip(graph_node_types, graph_node_ids))
        }
        graph_edges: list[tuple[int, int]] = []
        graph_edge_types: list[int] = []
        for node_id, node in engine.damage_tree.nodes.items():
            if node["node_type"] == "leaf":
                edge_rows.append(
                    {
                        "source_id": str(node["component_id"]),
                        "target_id": node_id,
                        "source_type": "component",
                        "target_type": "damage_tree",
                        "edge_type": "component_state_to_leaf",
                    }
                )
                graph_edges.append(
                    (graph_node_index[("component", str(node["component_id"]))],
                     graph_node_index[("damage_tree", node_id)])
                )
                graph_edge_types.append(0)
            for dependency in node["inputs"]:
                edge_rows.append(
                    {
                        "source_id": dependency,
                        "target_id": node_id,
                        "source_type": "damage_tree",
                        "target_type": "damage_tree",
                        "edge_type": "damage_tree_dependency",
                    }
                )
                graph_edges.append((graph_node_index[("damage_tree", dependency)],
                                    graph_node_index[("damage_tree", node_id)]))
                graph_edge_types.append(1)

        armor_vertices: list[list[float]] = []
        armor_vertex_offsets = [0]
        for item in engine.compiled_target.armor:
            armor_vertices.extend(item.geometry.vertices.tolist())
            armor_vertex_offsets.append(len(armor_vertices))
        component_vertices: list[list[float]] = []
        component_vertex_offsets = [0]
        for item in engine.compiled_target.components:
            if isinstance(item.geometry, ExtrudedPolygonGeometry):
                component_vertices.extend(item.geometry.vertices.tolist())
            component_vertex_offsets.append(len(component_vertices))

        valid_case_ids = [case["case_id"] for case, _ in valid_pairs]
        valid_index_by_id = {case_id: index for index, case_id in enumerate(valid_case_ids)}
        input_case_ids = [str(case.get("case_id") or "") for case in cases]
        input_status = [str(payload.get("status", "refused")) for payload in payloads]
        input_valid_mask = [status == "ok" for status in input_status]
        input_to_valid_index = [
            valid_index_by_id.get(case_id, -1) if is_valid else -1
            for case_id, is_valid in zip(input_case_ids, input_valid_mask)
        ]

        def batch_array(rows: list, tail_shape: tuple[int, ...], dtype) -> np.ndarray:
            return np.asarray(rows, dtype=dtype).reshape((len(rows), *tail_shape))

        tensor_values = {
            "case_x": batch_array(case_feature_rows, (len(CASE_FEATURE_NAMES),), np.float32),
            "case_feature_mask": batch_array(case_feature_masks, (len(CASE_FEATURE_NAMES),), bool),
            "component_x": batch_array(component_features, (component_count, len(COMPONENT_FEATURE_NAMES)), np.float32),
            "component_feature_mask": batch_array(component_feature_masks, (component_count, len(COMPONENT_FEATURE_NAMES)), bool),
            "armor_x": batch_array(armor_features, (armor_count, len(ARMOR_FEATURE_NAMES)), np.float32),
            "armor_feature_mask": batch_array(armor_feature_masks, (armor_count, len(ARMOR_FEATURE_NAMES)), bool),
            "armor_y": batch_array(armor_labels, (armor_count, len(ARMOR_LABEL_NAMES)), np.float32),
            "armor_y_stddev": batch_array(armor_label_stddev, (armor_count, len(ARMOR_LABEL_NAMES)), np.float32),
            "armor_y_ci95": batch_array(armor_label_ci95, (armor_count, len(ARMOR_LABEL_NAMES), 2), np.float32),
            "armor_label_mask": batch_array(armor_label_masks, (armor_count, len(ARMOR_LABEL_NAMES)), bool),
            "component_armor_path_x": batch_array(armor_path_features, (component_count, armor_count, len(ARMOR_PATH_FEATURE_NAMES)), np.float32),
            "component_armor_path_mask": batch_array(armor_path_feature_masks, (component_count, armor_count, len(ARMOR_PATH_FEATURE_NAMES)), bool),
            "component_y": batch_array(component_labels, (component_count, len(mechanism_names), len(state_names)), np.float32),
            "component_y_stddev": batch_array(component_label_stddev, (component_count, len(mechanism_names), len(state_names)), np.float32),
            "component_y_ci95": batch_array(component_label_ci95, (component_count, len(mechanism_names), len(state_names), 2), np.float32),
            "component_label_mask": batch_array(component_label_masks, (component_count, len(mechanism_names), len(state_names)), bool),
            "component_frechet_y": batch_array(component_frechet_labels, (component_count, len(state_names), len(FRECHET_BOUND_NAMES)), np.float32),
            "component_frechet_y_stddev": batch_array(component_frechet_stddev, (component_count, len(state_names), len(FRECHET_BOUND_NAMES)), np.float32),
            "component_frechet_y_ci95": batch_array(component_frechet_ci95, (component_count, len(state_names), len(FRECHET_BOUND_NAMES), 2), np.float32),
            "component_frechet_label_mask": batch_array(component_frechet_masks, (component_count, len(state_names), len(FRECHET_BOUND_NAMES)), bool),
            "component_aux_y": batch_array(component_auxiliary_labels, (component_count, len(AUXILIARY_LABEL_NAMES)), np.float32),
            "component_aux_y_stddev": batch_array(component_auxiliary_stddev, (component_count, len(AUXILIARY_LABEL_NAMES)), np.float32),
            "component_aux_y_ci95": batch_array(component_auxiliary_ci95, (component_count, len(AUXILIARY_LABEL_NAMES), 2), np.float32),
            "component_aux_label_mask": batch_array(component_auxiliary_masks, (component_count, len(AUXILIARY_LABEL_NAMES)), bool),
            "system_y": batch_array(system_labels, (len(output_names),), np.float32),
            "system_y_stddev": batch_array(system_label_stddev, (len(output_names),), np.float32),
            "system_y_ci95": batch_array(system_label_ci95, (len(output_names), 2), np.float32),
            "system_label_mask": batch_array(system_label_masks, (len(output_names),), bool),
            "system_source_available_mask": batch_array(system_source_available_masks, (len(output_names),), bool),
            "requested_standard_mask": batch_array(
                requested_standard_masks, (len(evaluation_standard_ids),), bool
            ),
            "case_ids": np.asarray(valid_case_ids, dtype=str),
            "case_hashes": np.asarray(
                [result["provenance"]["case_hash"] for _, result in valid_pairs],
                dtype=str,
            ),
            "split_names": np.asarray([row["split"] for row in split_rows], dtype=str),
            "split_group_ids": np.asarray(
                [row["group_id"] for row in split_rows], dtype=str
            ),
            "split_group_sources": np.asarray(
                [row["group_source"] for row in split_rows], dtype=str
            ),
            "leakage_keys": np.asarray(
                [row["leakage_key"] for row in split_rows], dtype=str
            ),
            "aleatory_seeds": np.asarray([case["analysis"]["aleatory_seed"] for case, _ in valid_pairs], dtype=np.int64),
            "epistemic_sample_ids": np.asarray(
                [case["analysis"]["epistemic_sample_id"] for case, _ in valid_pairs],
                dtype=str,
            ),
            "replicate_counts": np.asarray([result["replicate_count"] for _, result in valid_pairs], dtype=np.int64),
            "input_case_ids": np.asarray(input_case_ids, dtype=str),
            "input_case_status": np.asarray(input_status, dtype=str),
            "input_case_valid_mask": np.asarray(input_valid_mask, dtype=bool),
            "input_to_valid_index": np.asarray(input_to_valid_index, dtype=np.int64),
            "component_ids": np.asarray(list(engine.compiled_target.component_order), dtype=str),
            "armor_ids": np.asarray([item.object_id for item in engine.compiled_target.armor], dtype=str),
            "mechanism_names": np.asarray(mechanism_names),
            "state_names": np.asarray(state_names),
            "frechet_bound_names": np.asarray(FRECHET_BOUND_NAMES),
            "frechet_bound_units": np.asarray(FRECHET_BOUND_UNITS),
            "system_output_names": np.asarray(output_names),
            "evaluation_standard_ids": np.asarray(evaluation_standard_ids),
            "standard_output_mask": standard_output_mask,
            "case_feature_names": np.asarray(CASE_FEATURE_NAMES),
            "case_feature_units": np.asarray(CASE_FEATURE_UNITS),
            "component_feature_names": np.asarray(COMPONENT_FEATURE_NAMES),
            "component_feature_units": np.asarray(COMPONENT_FEATURE_UNITS),
            "armor_feature_names": np.asarray(ARMOR_FEATURE_NAMES),
            "armor_feature_units": np.asarray(ARMOR_FEATURE_UNITS),
            "armor_label_names": np.asarray(ARMOR_LABEL_NAMES),
            "armor_label_units": np.asarray(ARMOR_LABEL_UNITS),
            "armor_path_feature_names": np.asarray(ARMOR_PATH_FEATURE_NAMES),
            "armor_path_feature_units": np.asarray(ARMOR_PATH_FEATURE_UNITS),
            "component_aux_label_names": np.asarray(AUXILIARY_LABEL_NAMES),
            "component_aux_label_units": np.asarray(AUXILIARY_LABEL_UNITS),
            "component_material_ids": np.asarray([item.response["material_id"] for item in engine.compiled_target.components]),
            "component_type_ids": np.asarray([item.source["component_type"] for item in engine.compiled_target.components]),
            "fragment_response_model_ids": np.asarray([item.response["fragment_response_model_id"] for item in engine.compiled_target.components]),
            "blast_response_model_ids": np.asarray([item.response["blast_response"]["model_id"] for item in engine.compiled_target.components]),
            "armor_material_ids": np.asarray([item.response["material_id"] for item in engine.compiled_target.armor]),
            "armor_fragment_response_model_ids": np.asarray([item.response["fragment_response_model_id"] for item in engine.compiled_target.armor]),
            "armor_blast_response_model_ids": np.asarray([item.response["blast_response"]["model_id"] for item in engine.compiled_target.armor]),
            "armor_vertices_m": np.asarray(armor_vertices, dtype=np.float32).reshape((-1, 3)),
            "armor_vertex_offsets": np.asarray(armor_vertex_offsets, dtype=np.int64),
            "component_vertices_m": np.asarray(component_vertices, dtype=np.float32).reshape((-1, 3)),
            "component_vertex_offsets": np.asarray(component_vertex_offsets, dtype=np.int64),
            "graph_node_ids": np.asarray(graph_node_ids),
            "graph_node_types": np.asarray(graph_node_types),
            "tree_edge_index": np.asarray(graph_edges, dtype=np.int64).reshape((-1, 2)).T,
            "tree_edge_type": np.asarray(graph_edge_types, dtype=np.int16),
            "tree_edge_type_names": np.asarray(["component_state_to_leaf", "damage_tree_dependency"]),
        }
        if sampling_enabled:
            assert sampling_records is not None
            role_names = list(sampling_records[0]["role_names"])
            stratum_names = list(sampling_records[0]["stratum_names"])
            role_index = {name: index for index, name in enumerate(role_names)}
            stratum_index = {name: index for index, name in enumerate(stratum_names)}
            tensor_values.update(
                {
                    "case_physics_x": tensor_values["case_x"][:, _CASE_PHYSICS_INDICES].copy(),
                    "case_physics_feature_mask": tensor_values["case_feature_mask"][:, _CASE_PHYSICS_INDICES].copy(),
                    "solver_context_x": tensor_values["case_x"][:, _SOLVER_CONTEXT_INDICES].copy(),
                    "solver_context_feature_mask": tensor_values["case_feature_mask"][:, _SOLVER_CONTEXT_INDICES].copy(),
                    "case_physics_feature_names": np.asarray(CASE_PHYSICS_FEATURE_NAMES),
                    "case_physics_feature_units": np.asarray(CASE_PHYSICS_FEATURE_UNITS),
                    "solver_context_feature_names": np.asarray(SOLVER_CONTEXT_FEATURE_NAMES),
                    "solver_context_feature_units": np.asarray(SOLVER_CONTEXT_FEATURE_UNITS),
                    "sampling_selection_probability": np.asarray(
                        [record["selection_probability"] for record in valid_sampling_records], dtype=np.float64
                    ),
                    "sampling_design_weight_raw": np.asarray(
                        [record["design_weight_raw"] for record in valid_sampling_records], dtype=np.float64
                    ),
                    "sampling_weight_mask": np.asarray(
                        [record["weight_mask"] for record in valid_sampling_records], dtype=bool
                    ),
                    "sampling_round": np.asarray(
                        [record["round"] for record in valid_sampling_records], dtype=np.int64
                    ),
                    "sampling_stratum_index": np.asarray(
                        [stratum_index[record["stratum"]] for record in valid_sampling_records], dtype=np.int64
                    ),
                    "sampling_role_index": np.asarray(
                        [role_index[record["role"]] for record in valid_sampling_records], dtype=np.int64
                    ),
                    "sampling_stratum_names": np.asarray(stratum_names),
                    "sampling_role_names": np.asarray(role_names),
                    "sampling_candidate_ids": np.asarray(
                        [record["candidate_id"] for record in valid_sampling_records], dtype=str
                    ),
                    "sampling_lineage_ids": np.asarray(
                        [record["lineage_id"] for record in valid_sampling_records], dtype=str
                    ),
                    "sampling_epistemic_group_ids": np.asarray(
                        [record["epistemic_group_id"] for record in valid_sampling_records], dtype=str
                    ),
                }
            )
            if model_input_definition is not None:
                # Complete teacher tensors remain audit-only.  This versioned
                # definition is the sole source of learner-visible field and
                # feature metadata.
                tensor_values.update(
                    {
                        model_input_definition.tensor_field: batch_array(
                            [
                                model_input_definition.vector(case)
                                for case, _ in valid_pairs
                            ],
                            (len(model_input_definition.feature_names),),
                            np.float32,
                        ),
                        model_input_definition.mask_field: np.ones(
                            (
                                len(valid_pairs),
                                len(model_input_definition.feature_names),
                            ),
                            dtype=bool,
                        ),
                        **{
                            field: np.asarray(values)
                            for field, values in model_input_definition.metadata_fields.items()
                        },
                    }
                )
            for field, key in {
                "input_sampling_candidate_ids": "candidate_id", "input_sampling_lineage_ids": "lineage_id",
                "input_sampling_roles": "role", "input_sampling_epistemic_group_ids": "epistemic_group_id",
                "input_frozen_splits": "frozen_split",
            }.items():
                tensor_values[field] = np.asarray([record[key] for record in sampling_records], dtype=str)
            if sampling_records[0]["schema_version"] == "1.2.0":
                tensor_values["input_sampling_selection_probability"] = np.asarray(
                    [record["selection_probability"] for record in sampling_records],
                    dtype=np.float64,
                )
                tensor_values["input_sampling_design_weight_raw"] = np.asarray(
                    [record["design_weight_raw"] for record in sampling_records],
                    dtype=np.float64,
                )
                tensor_values["input_sampling_weight_mask"] = np.asarray(
                    [record["weight_mask"] for record in sampling_records], dtype=bool
                )
                adaptive_numeric_fields = {
                    "sampling_p0_density": "p0_density",
                    "sampling_adaptive_density": "adaptive_density",
                    "sampling_proposal_density": "proposal_density",
                    "sampling_p0_hierarchy_probability": "p0_hierarchy_probability",
                    "sampling_proposal_hierarchy_probability": "proposal_hierarchy_probability",
                }
                for field, key in adaptive_numeric_fields.items():
                    tensor_values[field] = np.asarray(
                        [record[key] for record in valid_sampling_records], dtype=np.float64
                    )
                    tensor_values["input_" + field] = np.asarray(
                        [record[key] for record in sampling_records], dtype=np.float64
                    )
                adaptive_identity_fields = {
                    "sampling_acquisition_model_hashes": "acquisition_model_hash",
                    "sampling_proposal_hashes": "proposal_hash",
                    "sampling_round_lineage_ids": "round_lineage_id",
                    "sampling_selected_case_set_hashes": "selected_case_set_hash",
                }
                for field, key in adaptive_identity_fields.items():
                    tensor_values[field] = np.asarray(
                        [record[key] for record in valid_sampling_records], dtype=str
                    )
                    tensor_values["input_" + field] = np.asarray(
                        [record[key] for record in sampling_records], dtype=str
                    )
            assert label_spec is not None
            task_masks = label_spec.validate_system_outputs(output_names, tensor_values["system_y"], tensor_values["system_label_mask"])
            task_names = sorted(task_masks)
            tensor_values["label_task_names"] = np.asarray(task_names, dtype=str)
            tensor_values["label_task_available_mask"] = np.stack([task_masks[name] for name in task_names], axis=-1)
        # Missing uncertainty is represented by NaN and an independent mask.
        # Label availability still means its mean is usable as a synthetic label.
        uncertainty_prefixes = {
            "armor": ("armor_y", "armor_label_mask"),
            "component": ("component_y", "component_label_mask"),
            "component_frechet": ("component_frechet_y", "component_frechet_label_mask"),
            "component_aux": ("component_aux_y", "component_aux_label_mask"),
            "system": ("system_y", "system_label_mask"),
        }
        for prefix, (label_name, availability_name) in uncertainty_prefixes.items():
            shape = tensor_values[label_name].shape
            counts = tensor_values["replicate_counts"].reshape((shape[0],) + (1,) * (len(shape) - 1))
            sample_count = np.where(tensor_values[availability_name],
                                    np.broadcast_to(counts, shape), 0).astype(np.int64)
            uncertainty_mask = (
                tensor_values[availability_name] & (sample_count >= 2)
                & np.isfinite(tensor_values[label_name + "_stddev"])
                & np.all(np.isfinite(tensor_values[label_name + "_ci95"]), axis=-1)
            )
            tensor_values[prefix + "_sample_count"] = sample_count
            tensor_values[prefix + "_uncertainty_mask"] = uncertainty_mask
            tensor_values[label_name + "_stddev"][~uncertainty_mask] = np.nan
            tensor_values[label_name + "_ci95"][~uncertainty_mask] = np.nan
        arrays_path = self.output_directory / "tensors.npz"
        tensor_writer = np.savez_compressed if self.tensor_compression == "compressed" else np.savez
        tensor_writer(arrays_path, **tensor_values)
        paths_and_rows = []
        for name, rows in (
            ("cases", case_rows),
            ("components", component_rows),
            ("armor", armor_rows),
            ("armor_labels", armor_label_rows),
            ("labels", label_rows),
            ("component_aux", auxiliary_rows),
            ("edges", edge_rows),
            ("splits", split_rows),
        ):
            path = self.output_directory / f"{name}.parquet"
            _write_parquet(path, rows)
            paths_and_rows.append((name, path, len(rows)))
        tree_path = self.output_directory / "tree_outputs.parquet"
        _write_parquet(tree_path, tree_rows)
        refusal_counts: dict[str, int] = {}
        for item in refused:
            code = item["refusal"]["code"]
            refusal_counts[code] = refusal_counts.get(code, 0) + 1
        parameters = list(_walk_parameters(engine.target_data))
        unverified_fraction = (
            sum(parameter["validation_status"] != "validated" for parameter in parameters)
            / max(1, len(parameters))
        )
        partitions = [
            {
                "name": name,
                "format": "parquet",
                "path_glob": path.name,
                "row_count": row_count,
                "content_hash": file_hash(path),
            }
            for name, path, row_count in paths_and_rows
        ]
        partitions.extend(
            [
                {
                    "name": "tree_outputs",
                    "format": "parquet",
                    "path_glob": tree_path.name,
                    "row_count": len(tree_rows),
                    "content_hash": file_hash(tree_path),
                },
                {
                    "name": "audits",
                    "format": raw_format,
                    "path_glob": raw_path.name,
                    "row_count": len(results),
                    "content_hash": file_hash(raw_path),
                },
                {
                    "name": "tensors",
                    "format": "npz",
                    "path_glob": arrays_path.name,
                    "row_count": len(valid_pairs),
                    "content_hash": file_hash(arrays_path),
                },
            ]
        )
        tensor_shapes = {
            "case_x": "[B,F_case]", "case_feature_mask": "[B,F_case]",
            "component_x": "[B,C,F_component]", "component_feature_mask": "[B,C,F_component]",
            "armor_x": "[B,A,F_armor]", "armor_feature_mask": "[B,A,F_armor]",
            "armor_y": "[B,A,K]", "armor_y_stddev": "[B,A,K]",
            "armor_y_ci95": "[B,A,K,2]", "armor_label_mask": "[B,A,K]",
            "component_armor_path_x": "[B,C,A,F_path]",
            "component_armor_path_mask": "[B,C,A,F_path]",
            "component_y": "[B,C,M,S]", "component_y_stddev": "[B,C,M,S]",
            "component_y_ci95": "[B,C,M,S,2]", "component_label_mask": "[B,C,M,S]",
            "component_frechet_y": "[B,C,S,D]",
            "component_frechet_y_stddev": "[B,C,S,D]",
            "component_frechet_y_ci95": "[B,C,S,D,2]",
            "component_frechet_label_mask": "[B,C,S,D]",
            "component_aux_y": "[B,C,U]", "component_aux_y_stddev": "[B,C,U]",
            "component_aux_y_ci95": "[B,C,U,2]", "component_aux_label_mask": "[B,C,U]",
            "system_y": "[B,O]", "system_y_stddev": "[B,O]", "system_y_ci95": "[B,O,2]",
            "system_label_mask": "[B,O]", "requested_standard_mask": "[B,E_standard]",
            "system_source_available_mask": "[B,O]",
            "case_ids": "[B]", "case_hashes": "[B]",
            "split_names": "[B]", "split_group_ids": "[B]", "split_group_sources": "[B]",
            "leakage_keys": "[B]", "aleatory_seeds": "[B]", "epistemic_sample_ids": "[B]",
            "replicate_counts": "[B]", "input_case_ids": "[N]", "input_case_status": "[N]",
            "input_case_valid_mask": "[N]", "input_to_valid_index": "[N]",
            "component_ids": "[C]", "armor_ids": "[A]", "mechanism_names": "[M]",
            "state_names": "[S]", "frechet_bound_names": "[D]",
            "frechet_bound_units": "[D]", "system_output_names": "[O]",
            "evaluation_standard_ids": "[E_standard]", "standard_output_mask": "[E_standard,O]",
            "case_feature_names": "[F_case]", "case_feature_units": "[F_case]",
            "component_feature_names": "[F_component]", "component_feature_units": "[F_component]",
            "armor_feature_names": "[F_armor]", "armor_feature_units": "[F_armor]",
            "armor_label_names": "[K]", "armor_label_units": "[K]",
            "armor_path_feature_names": "[F_path]", "armor_path_feature_units": "[F_path]",
            "component_aux_label_names": "[U]", "component_aux_label_units": "[U]",
            "component_material_ids": "[C]", "component_type_ids": "[C]",
            "fragment_response_model_ids": "[C]", "blast_response_model_ids": "[C]",
            "armor_material_ids": "[A]", "armor_fragment_response_model_ids": "[A]",
            "armor_blast_response_model_ids": "[A]", "armor_vertices_m": "[V,3]",
            "armor_vertex_offsets": "[A+1]", "graph_node_ids": "[G]", "graph_node_types": "[G]",
            "component_vertices_m": "[Vc,3]", "component_vertex_offsets": "[C+1]",
            "tree_edge_index": "[2,E_edge]", "tree_edge_type": "[E_edge]", "tree_edge_type_names": "[2]",
        }
        if sampling_enabled:
            tensor_shapes.update(
                {
                    "case_physics_x": "[B,F_case_physics]",
                    "case_physics_feature_mask": "[B,F_case_physics]",
                    "solver_context_x": "[B,F_solver]",
                    "solver_context_feature_mask": "[B,F_solver]",
                    "case_physics_feature_names": "[F_case_physics]",
                    "case_physics_feature_units": "[F_case_physics]",
                    "solver_context_feature_names": "[F_solver]",
                    "solver_context_feature_units": "[F_solver]",
                    "sampling_selection_probability": "[B]",
                    "sampling_design_weight_raw": "[B]",
                    "sampling_weight_mask": "[B]",
                    "sampling_round": "[B]",
                    "sampling_stratum_index": "[B]",
                    "sampling_role_index": "[B]",
                    "sampling_stratum_names": "[Q]",
                    "sampling_role_names": "[R]",
                    "sampling_candidate_ids": "[B]",
                    "sampling_lineage_ids": "[B]",
                    "sampling_epistemic_group_ids": "[B]",
                    "system_source_available_mask": "[B,O]",
                    "input_sampling_candidate_ids": "[N]", "input_sampling_lineage_ids": "[N]",
                    "input_sampling_roles": "[N]", "input_sampling_epistemic_group_ids": "[N]",
                    "input_frozen_splits": "[N]", "label_task_names": "[T]",
                    "label_task_available_mask": "[B,T]",
                }
            )
            if model_input_definition is not None:
                tensor_shapes.update(model_input_definition.tensor_shapes)
            if sampling_records[0]["schema_version"] == "1.2.0":
                tensor_shapes["input_sampling_selection_probability"] = "[N]"
                tensor_shapes["input_sampling_design_weight_raw"] = "[N]"
                tensor_shapes["input_sampling_weight_mask"] = "[N]"
                for name in (
                    "sampling_p0_density", "sampling_adaptive_density",
                    "sampling_proposal_density", "sampling_p0_hierarchy_probability",
                    "sampling_proposal_hierarchy_probability",
                    "sampling_acquisition_model_hashes", "sampling_proposal_hashes",
                    "sampling_round_lineage_ids", "sampling_selected_case_set_hashes",
                ):
                    tensor_shapes[name] = "[B]"
                    tensor_shapes["input_" + name] = "[N]"
        for prefix, (label_name, _) in uncertainty_prefixes.items():
            tensor_shapes[prefix + "_sample_count"] = tensor_shapes[label_name]
            tensor_shapes[prefix + "_uncertainty_mask"] = tensor_shapes[label_name]
        label_fields = {
            "component_y", "component_frechet_y", "armor_y", "system_y",
            "component_aux_y",
        }
        uncertainty_fields = {
            "component_y_stddev", "component_y_ci95", "system_y_stddev", "system_y_ci95",
            "component_frechet_y_stddev", "component_frechet_y_ci95",
            "armor_y_stddev", "armor_y_ci95", "component_aux_y_stddev", "component_aux_y_ci95",
        }
        mask_fields = {name for name in tensor_values if name.endswith("mask")}
        feature_fields = {
            "case_x", "case_physics_x", "solver_context_x", "uav_state_x",
            "loitering_impact_x", "loitering_impact_contact_pose_x",
            "component_x", "armor_x",
            "component_armor_path_x",
        }
        index_fields = {
            "aleatory_seeds", "replicate_counts", "input_to_valid_index", "armor_vertex_offsets",
            "component_vertex_offsets", "tree_edge_index", "tree_edge_type",
            "sampling_round", "sampling_stratum_index", "sampling_role_index",
        }
        index_fields.update(prefix + "_sample_count" for prefix in uncertainty_prefixes)
        field_units = {
            "case_x": "mixed; see case_feature_units",
            "case_physics_x": "mixed; see case_physics_feature_units",
            "solver_context_x": "mixed; see solver_context_feature_units",
            "uav_state_x": "mixed; see uav_state_feature_units",
            "loitering_impact_x": (
                "mixed; see loitering_impact_feature_units"
            ),
            "loitering_impact_contact_pose_x": (
                "mixed; see loitering_impact_contact_pose_feature_units"
            ),
            "component_x": "mixed; see component_feature_units",
            "armor_x": "mixed; see armor_feature_units",
            "component_armor_path_x": "mixed; see armor_path_feature_units",
            "component_y": "1", "component_y_stddev": "1", "component_y_ci95": "1",
            "component_frechet_y": "1", "component_frechet_y_stddev": "1",
            "component_frechet_y_ci95": "1",
            "armor_y": "mixed; see armor_label_units",
            "armor_y_stddev": "mixed; see armor_label_units",
            "armor_y_ci95": "mixed; see armor_label_units",
            "system_y": "1", "system_y_stddev": "1", "system_y_ci95": "1",
            "component_aux_y": "mixed; see component_aux_label_units",
            "component_aux_y_stddev": "mixed; see component_aux_label_units",
            "component_aux_y_ci95": "mixed; see component_aux_label_units",
            "armor_vertices_m": "m", "component_vertices_m": "m",
            "aleatory_seeds": "1", "replicate_counts": "count",
            "sampling_selection_probability": "1",
            "sampling_design_weight_raw": "1",
            "sampling_p0_density": "density",
            "sampling_adaptive_density": "density",
            "sampling_proposal_density": "density",
            "sampling_p0_hierarchy_probability": "1",
            "sampling_proposal_hierarchy_probability": "1",
            "input_sampling_p0_density": "density",
            "input_sampling_adaptive_density": "density",
            "input_sampling_proposal_density": "density",
            "input_sampling_p0_hierarchy_probability": "1",
            "input_sampling_proposal_hierarchy_probability": "1",
            "input_sampling_selection_probability": "1",
            "input_sampling_design_weight_raw": "1",
            "sampling_round": "count",
            "sampling_stratum_index": "1",
            "sampling_role_index": "1",
        }
        field_units.update({prefix + "_sample_count": "count" for prefix in uncertainty_prefixes})

        def dtype_name(array: np.ndarray) -> str:
            if array.dtype.kind == "b":
                return "bool"
            if array.dtype.kind in {"U", "S"}:
                return "string"
            if array.dtype == np.dtype("int16"):
                return "int16"
            if array.dtype.kind in {"i", "u"}:
                return "int64"
            if array.dtype == np.dtype("float64"):
                return "float64"
            return "float32"

        tensor_fields = []
        for name, array in tensor_values.items():
            if name in feature_fields:
                role = "feature"
            elif name in label_fields:
                role = "label"
            elif name in uncertainty_fields:
                role = "uncertainty"
            elif name in mask_fields:
                role = "mask"
            elif name in index_fields:
                role = "index"
            else:
                role = "metadata"
            tensor_fields.append(
                {
                    "name": name,
                    "dtype": dtype_name(array),
                    "shape": tensor_shapes[name],
                    "semantic_role": role,
                    "unit": field_units.get(name),
                    "normalization_id": None,
                }
            )

        fidelity_levels = sorted(
            {
                str(case.get("analysis", {}).get("fidelity_level"))
                for case, _ in valid_pairs
                if case.get("analysis", {}).get("fidelity_level") is not None
            }
        )
        fidelity_level = fidelity_levels[0] if len(fidelity_levels) == 1 else (
            "mixed" if fidelity_levels else "unknown"
        )
        provenance = next(
            (payload.get("provenance", {}) for payload in payloads if payload.get("provenance")),
            {},
        )
        sampling_fixed = sampling_records[0] if sampling_records else None
        contract_version = (
            "1.8.0"
            if versioned_model_contract_enabled
            else (
                "1.7.0"
                if sampling_fixed and sampling_fixed["schema_version"] == "1.2.0"
                else ("1.6.0" if sampling_enabled else "1.4.0")
            )
        )
        manifest_version = (
            "1.7.0"
            if versioned_model_contract_enabled
            else (
                "1.6.0"
                if sampling_fixed and sampling_fixed["schema_version"] == "1.2.0"
                else (
                    "1.5.0"
                    if sampling_fixed and sampling_fixed["schema_version"] == "1.1.0"
                    else ("1.4.0" if sampling_enabled else "1.2.0")
                )
            )
        )
        identity_hash = dataset_identity_hash(
            target_hash=engine.target_hash,
            physics_registry_hash=engine.registry.hash,
            engine_code_hash=ENGINE_CODE_HASH,
            ordered_case_hashes=[hash_case(case) for case in cases],
            split_seed=self.split_seed,
            split_ratios=self.split_ratios,
            tensor_contract_version=contract_version,
            damage_tree_definition_hash=(
                sampling_fixed["damage_tree_definition_hash"] if sampling_fixed else None
            ),
            label_spec_hash=(sampling_fixed["label_spec_hash"] if sampling_fixed else None),
            sampling_design_hash=(sampling_fixed["sampling_spec_hash"] if sampling_fixed else None),
            stratification_spec_hash=(
                sampling_fixed["stratification_spec_hash"] if sampling_fixed else None
            ),
            split_registry_hash=(sampling_fixed["split_registry_hash"] if sampling_fixed else None),
        )
        source_contracts = {
            "target_schema": "target.schema.json@1.0.0",
            "case_schema": "simulation_case.schema.json@1.0.0",
            "damage_tree_versions": [engine.damage_tree.version],
            "evaluation_standards": {
                standard_id: list(output_ids)
                for standard_id, output_ids in engine.damage_tree.standards.items()
            },
            "parameter_set_hashes": [engine.registry.hash],
            "parameter_set_ids": sorted(
                {
                    str(case["parameter_set_id"])
                    for case, _ in valid_pairs
                    if case.get("parameter_set_id") is not None
                }
            ),
            "target_hashes": [engine.target_hash],
        }
        if versioned_model_contract_enabled:
            assert fixed_scenario_hash is not None
            source_contracts.update(
                {
                    "fixed_scenario_hash": fixed_scenario_hash,
                    "fixed_scenario_identity_version": (
                        model_input_definition.fixed_scenario_identity_version
                    ),
                    "learning_input_identity_version": (
                        model_input_definition.learning_input_identity_version
                    ),
                }
            )
        generation_record = dict(generation)
        from .parquet_schema import PARQUET_SCHEMA_VERSION
        generation_record["storage_guarantees"] = {
            **generation_record.get("storage_guarantees", {}),
            "parquet_table_schema_version": PARQUET_SCHEMA_VERSION,
            "parquet_unknown_fields_refused": True,
        }
        if sampling_fixed:
            source_contracts.update(
                {
                    "damage_tree_definition_hash": sampling_fixed["damage_tree_definition_hash"],
                    "label_spec_id": sampling_fixed["label_spec_id"],
                    "label_spec_hash": sampling_fixed["label_spec_hash"],
                    "label_spec_document": sampling_fixed["label_spec_document"],
                    "sampling_spec_id": sampling_fixed["sampling_spec_id"],
                    "sampling_design_hash": sampling_fixed["sampling_spec_hash"],
                    "stratification_spec_id": sampling_fixed["stratification_spec_id"],
                    "stratification_spec_hash": sampling_fixed["stratification_spec_hash"],
                    "split_registry_id": sampling_fixed["split_registry_id"],
                    "split_registry_hash": sampling_fixed["split_registry_hash"],
                    "label_task_semantics": "externally_versioned; see label_spec",
                }
            )
            if sampling_fixed["schema_version"] in {
                "1.1.0",
                "1.2.0",
                "1.3.0",
                "1.4.0",
            }:
                source_contracts.update(
                    {
                        "sampling_spec_document": sampling_fixed[
                            "sampling_spec_document"
                        ],
                        "training_design_document": sampling_fixed[
                            "training_design_document"
                        ],
                        "training_design_hash": sampling_fixed[
                            "training_design_hash"
                        ],
                    }
                )
            generation_record["sampling"] = {
                "sampling_envelope_version": sampling_fixed["schema_version"],
                "rounds": sorted({record["round"] for record in sampling_records}),
                "streams": sorted({record["stream"] for record in sampling_records}),
                "roles": list(sampling_fixed["role_names"]),
                "strata": list(sampling_fixed["stratum_names"]),
                "selection_probability_field": "sampling_selection_probability",
                "raw_design_weight_field": "sampling_design_weight_raw",
                "raw_design_weight_semantics": (
                    "general_two_stage_reference_joint_design"
                    if sampling_fixed["schema_version"] == "1.2.0"
                    else "inverse_inclusion_probability_for_weight_mask_true_reference_cases"
                ),
                "split_registry_document": sampling_fixed["split_registry_document"],
            }
            if sampling_fixed["schema_version"] == "1.2.0":
                generation_record["sampling"].update(
                    {
                        "design_weight_formula": sampling_fixed["design_weight_formula"],
                        "proposal_hash": sampling_fixed["proposal_hash"],
                        "adaptive_proposal_hash": sampling_fixed[
                            "adaptive_proposal_hash"
                        ],
                        "acquisition_model_hash": sampling_fixed[
                            "acquisition_model_hash"
                        ],
                        "round_lineage_id": sampling_fixed["round_lineage_id"],
                        "parent_round_lineage_id": sampling_fixed[
                            "parent_round_lineage_id"
                        ],
                        "selected_case_set_hash": sampling_fixed[
                            "selected_case_set_hash"
                        ],
                        "adaptive_factor_fields": {
                            "p0_density": "sampling_p0_density",
                            "adaptive_density": "sampling_adaptive_density",
                            "proposal_density": "sampling_proposal_density",
                            "p0_hierarchy_probability": "sampling_p0_hierarchy_probability",
                            "proposal_hierarchy_probability": "sampling_proposal_hierarchy_probability",
                        },
                    }
                )
        manifest = {
            "schema_version": manifest_version,
            "dataset_id": "damage-dataset-" + identity_hash.split(":", 1)[1][:24],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "engine": {
                "version": provenance.get("engine_version", "unknown"),
                "code_hash": provenance.get("engine_code_hash", "unknown"),
                "configuration_hash": engine.registry.hash,
                "fidelity_level": fidelity_level,
                "fidelity_levels": fidelity_levels,
                "validation_record_id": None,
            },
            "source_contracts": source_contracts,
            "generation": generation_record,
            "semantic_vocabulary": {
                "version": "1.0.0",
                "names": {"state_names": state_names, "mechanism_names": mechanism_names,
                          "system_output_names": output_names,
                          "evaluation_standard_ids": evaluation_standard_ids},
                "content_hash": content_hash({"version": "1.0.0", "names": {
                    "state_names": state_names, "mechanism_names": mechanism_names,
                    "system_output_names": output_names,
                    "evaluation_standard_ids": evaluation_standard_ids}}),
            },
            "label_admission": synthetic_label_admission(),
            "capabilities": {
                "storage_layout": "monolithic",
                "bounded_tensor_batch": False,
                "checkpoint_resume": False,
                "sharded_writer_available": True,
                "single_target_topology_per_archive": True,
                "semantic_vocabulary_frozen": True,
                "independent_scientific_validation": False,
            },
            "randomness": {
                "split_seed": self.split_seed,
                "aleatory_seed_field": "analysis.aleatory_seed",
                "epistemic_sample_id_field": "analysis.epistemic_sample_id",
                "parallel_result_order_deterministic": True,
            },
            "partitions": partitions,
            "tensor_contract": {
                "version": contract_version,
                "fields": tensor_fields,
                "axes": [
                    {"axis_id": "B", "size": len(valid_pairs), "names_field": "case_ids", "units_field": None},
                    {"axis_id": "N", "size": len(cases), "names_field": "input_case_ids", "units_field": None},
                    {"axis_id": "C", "size": component_count, "names_field": "component_ids", "units_field": None},
                    {"axis_id": "A", "size": armor_count, "names_field": "armor_ids", "units_field": None},
                    {"axis_id": "M", "size": len(mechanism_names), "names_field": "mechanism_names", "units_field": None},
                    {"axis_id": "S", "size": len(state_names), "names_field": "state_names", "units_field": None},
                    {"axis_id": "O", "size": len(output_names), "names_field": "system_output_names", "units_field": None},
                    {"axis_id": "E_standard", "size": len(evaluation_standard_ids), "names_field": "evaluation_standard_ids", "units_field": None},
                    {"axis_id": "G", "size": len(graph_node_ids), "names_field": None, "units_field": None},
                    {"axis_id": "E_edge", "size": len(graph_edges), "names_field": None, "units_field": None},
                    {"axis_id": "V", "size": len(armor_vertices), "names_field": None, "units_field": None},
                    {"axis_id": "Vc", "size": len(component_vertices), "names_field": None, "units_field": None},
                    {"axis_id": "F_case", "size": len(CASE_FEATURE_NAMES), "names_field": "case_feature_names", "units_field": "case_feature_units"},
                    {"axis_id": "F_component", "size": len(COMPONENT_FEATURE_NAMES), "names_field": "component_feature_names", "units_field": "component_feature_units"},
                    {"axis_id": "F_armor", "size": len(ARMOR_FEATURE_NAMES), "names_field": "armor_feature_names", "units_field": "armor_feature_units"},
                    {"axis_id": "K", "size": len(ARMOR_LABEL_NAMES), "names_field": "armor_label_names", "units_field": "armor_label_units"},
                    {"axis_id": "D", "size": len(FRECHET_BOUND_NAMES), "names_field": "frechet_bound_names", "units_field": "frechet_bound_units"},
                    {"axis_id": "F_path", "size": len(ARMOR_PATH_FEATURE_NAMES), "names_field": "armor_path_feature_names", "units_field": "armor_path_feature_units"},
                    {"axis_id": "U", "size": len(AUXILIARY_LABEL_NAMES), "names_field": "component_aux_label_names", "units_field": "component_aux_label_units"},
                ],
            },
            "split_policy": {
                "method": "grouped",
                "group_keys": ["split_group", "condition_cluster", "physical_case_hash"],
                "seed": self.split_seed,
                "ratios": {
                    "train": self.split_ratios[0],
                    "validation": self.split_ratios[1],
                    "test": self.split_ratios[2],
                },
                "leakage_audit_passed": self._check_split_leakage(split_rows),
                "independent_unit_status": "caller_defined_or_exact_input_hash_only",
                "split_registry_status": "not_supplied",
                "independence_guaranteed": False,
                "coverage_guaranteed": False,
                "split_counts": {
                    name: sum(row["split"] == name for row in split_rows)
                    for name in ("train", "validation", "test")
                },
            },
            "quality_summary": {
                "n_total": len(results),
                "n_valid": len(valid_pairs),
                "n_valid_semantics": "execution_success_count",
                "execution_success_count": len(valid_pairs),
                "purpose_admission_counts": {
                    "software_development": len(valid_pairs), "research_training": 0,
                    "research_validation": 0, "research_test": 0,
                },
                "n_refused": len(refused),
                "refusal_counts": refusal_counts,
                "unverified_parameter_fraction": unverified_fraction,
                "mc_ci_half_width_quantiles": self._ci_quantiles(valid_pairs, output_names),
                "coverage_report_id": None,
            },
            "known_limitations": [
                "A dataset generated with an unvalidated registry is suitable only for software development.",
                "Damage-tree gates support only explicitly declared independent or fully-dependent (comonotonic) inputs; other dependence structures require a future calibrated model.",
                "Fragment/blast component-state probabilities are still combined using conditional independence within each replicate, even when sequential armor coupling is enabled.",
                "The monolithic writer materializes one batch of cases, results, arrays and Parquet rows; use ShardedDatasetWriter for checkpoint/resume and bounded tensor batches.",
                "Grouped splitting checks declared groups and exact physical input keys; it does not establish independent research units or enforce a cross-dataset split registry.",
                "Hash split ratios are assignment probabilities, not guaranteed realized coverage ratios.",
                "State labels are synthetic teacher outputs; state names and tensor shape alone do not establish mutually exclusive classes or a normalized probability distribution.",
                "Monte Carlo intervals are approximate intervals for a conditional replicate mean, not total epistemic or real-world predictive uncertainty; adaptive-stop coverage has not been validated.",
                "The parameter realization hash covers the nominal feature realization; non-epistemic target parameters may vary within replicates and are not all represented by that hash.",
                "Each dense NPZ still describes one target topology; TorchMultiTargetDataset and collate_damage_batch provide padding, name alignment, and disjoint-graph batching across those target-specific datasets.",
                "Armor labels now expose integrity and blast-response outputs, but remain tied to the same unvalidated physical registry as the other teacher labels.",
            ],
            "prohibited_uses": [
                "Do not use unvalidated outputs as certified real-world effectiveness data.",
                "Do not use the surrogate in an autonomous control loop without independent safety validation.",
            ],
        }
        if model_input_definition is not None:
            assert fixed_scenario_hash is not None
            manifest["model_input_contract"] = (
                model_input_definition.manifest_document(
                    fixed_scenario_hash=fixed_scenario_hash
                )
            )
            manifest["known_limitations"].append(
                model_input_definition.known_limitation
            )
        if sampling_enabled:
            assert sampling_fixed is not None
            manifest["tensor_contract"]["axes"].extend(
                [
                    {
                        "axis_id": "F_case_physics", "size": len(CASE_PHYSICS_FEATURE_NAMES),
                        "names_field": "case_physics_feature_names", "units_field": "case_physics_feature_units",
                    },
                    {
                        "axis_id": "F_solver", "size": len(SOLVER_CONTEXT_FEATURE_NAMES),
                        "names_field": "solver_context_feature_names", "units_field": "solver_context_feature_units",
                    },
                    {
                        "axis_id": "Q", "size": len(sampling_fixed["stratum_names"]),
                        "names_field": "sampling_stratum_names", "units_field": None,
                    },
                    {
                        "axis_id": "R", "size": len(sampling_fixed["role_names"]),
                        "names_field": "sampling_role_names", "units_field": None,
                    },
                    {"axis_id": "T", "size": len(task_names), "names_field": "label_task_names", "units_field": None},
                ]
            )
            if model_input_definition is not None:
                manifest["tensor_contract"]["axes"].append(
                    model_input_definition.axis_document
                )
            manifest["split_policy"].update(
                {
                    "split_registry_status": "frozen_pre_outcome",
                    "split_registry_hash": sampling_fixed["split_registry_hash"],
                    "independent_unit_status": "explicit_registry; domain_independence_unverified",
                    "independence_guaranteed": False,
                    "coverage_guaranteed": False,
                }
            )
            usable_weights = [record for record in valid_sampling_records if record["weight_mask"]]
            weights = np.asarray([record["design_weight_raw"] for record in usable_weights], dtype=np.float64)
            probabilities = np.asarray(
                [record["selection_probability"] for record in usable_weights], dtype=np.float64
            )
            scaled_weights = (
                scale_positive_weights(weights, field="sampling design weights")[0]
                if weights.size
                else weights
            )
            ess = positive_effective_sample_size(weights)
            manifest["quality_summary"]["sampling_diagnostics"] = {
                "positivity_min": float(np.min(probabilities)) if probabilities.size else None,
                "effective_sample_size": ess,
                "ess_fraction": ess / len(weights) if weights.size else 0.0,
                "weight_cv": (
                    float(np.std(scaled_weights) / np.mean(scaled_weights)) if weights.size else None
                ),
                "raw_weight_preserved": True,
                "candidate_ledger_external": True,
                "diagnostic_population": "successful_inputs_with_weight_mask_true",
                "usable_weight_count": len(usable_weights),
            }
            manifest["known_limitations"].append(
                f"Tensor contract {contract_version} preserves raw inclusion weights and verifies declared definitions; sampling diagnostics do not establish physical validity or operational representativeness."
            )
        # Refuse publication when a writer change and the executable JSON
        # contract drift apart.  Archive loaders validate tensor semantics,
        # while this gate validates the complete manifest envelope itself.
        validate_json_schema(manifest, "dataset_manifest.schema.json")
        manifest_path = self.output_directory / "manifest.json"
        manifest_temp_path = self.output_directory / "manifest.json.tmp"
        with manifest_temp_path.open("w", encoding="utf-8") as stream:
            json.dump(
                manifest,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        manifest_temp_path.replace(manifest_path)
        # Publication is complete only after the exact bytes can be reopened
        # through the same verifier used by downstream loaders.  The marker is
        # intentionally retained until this check succeeds.
        verify_monolithic_archive(
            self.output_directory,
            allow_split_leakage=True,
            allow_incomplete_marker=True,
        )
        incomplete_path.unlink(missing_ok=True)
        return manifest

    @staticmethod
    def _check_split_leakage(rows: list[dict]) -> bool:
        observed_groups: dict[str, str] = {}
        observed_cases: dict[str, str] = {}
        for row in rows:
            group = row["group_id"]
            leakage_key = row["leakage_key"]
            split = row["split"]
            if group in observed_groups and observed_groups[group] != split:
                return False
            if leakage_key in observed_cases and observed_cases[leakage_key] != split:
                return False
            observed_groups[group] = split
            observed_cases[leakage_key] = split
        return True

    @staticmethod
    def _ci_quantiles(valid_pairs: list[tuple[dict, dict]], outputs: list[str]) -> list[float]:
        widths = []
        for _, result in valid_pairs:
            for output in outputs:
                stats = result["damage_tree"]["outputs"][output]
                if stats["ci95_upper"] is None or stats["ci95_lower"] is None:
                    continue
                widths.append(0.5 * (stats["ci95_upper"] - stats["ci95_lower"]))
        if not widths:
            return []
        return [float(value) for value in np.quantile(widths, [0.5, 0.9, 0.99])]
