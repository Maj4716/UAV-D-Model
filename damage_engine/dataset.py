from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .engine import ENGINE_CODE_HASH, DamageEngine, EngineResult
from .errors import InputValidationError
from .geometry import BoxGeometry, CylinderGeometry, ExtrudedPolygonGeometry
from .registry import PhysicsRegistry
from .utils import content_hash, file_hash, length_to_m, pressure_to_pa, ratio_value


CASE_FEATURE_NAMES = [
    "detonation_x_m", "detonation_y_m", "detonation_z_m", "detonation_time_s",
    "orientation_qx", "orientation_qy", "orientation_qz", "orientation_qw",
    "velocity_x_mps", "velocity_y_mps", "velocity_z_mps",
    "charge_mass_kg", "casing_mass_kg", "blast_equivalent_mass_kg",
    "fragment_total_count",
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
    "1", "1", "1", "1",
    "kg", "kg", "kg", "kg", "kg",
    "1", "1", "1", "1", "1",
    "kg/m^3", "1", "1", "m/s", "m/s", "m/s",
    "1", "1", "1", "1", "1", "1", "deg", "deg",
    "1", "1", "kg/m^3", "Pa", "K",
    "1", "1", "1", "1", "1", "1", "1", "1", "1",
    "count", "1",
]


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
) -> str:
    """Return the storage-layout-independent identity of teacher labels."""

    return content_hash(
        {
            "target_hash": target_hash,
            "physics_registry_hash": physics_registry_hash,
            "engine_code_hash": engine_code_hash,
            "ordered_cases": ordered_case_hashes,
            "split_seed": int(split_seed),
            "split_ratios": [float(value) for value in split_ratios],
            "tensor_contract_version": str(tensor_contract_version),
        }
    )

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
    table = pa.Table.from_pylist(rows)
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
    analysis = case.get("analysis") or {}
    return content_hash(
        {
            "target_ref": case.get("target_ref"),
            "encounter": case.get("encounter"),
            "warhead": case.get("warhead"),
            "environment": case.get("environment"),
            "parameter_set_id": case.get("parameter_set_id"),
            "analysis": {
                "fidelity_level": analysis.get("fidelity_level"),
                "mechanisms": analysis.get("mechanisms"),
                "epistemic_sample_id": analysis.get("epistemic_sample_id"),
                "coupling_model": analysis.get("coupling_model"),
            },
        }
    )


def _split_name(
    case: dict,
    *,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[str, str, str, str]:
    metadata = case.get("metadata") or {}
    leakage_key = _physical_case_key(case)
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


class DatasetWriter:
    def __init__(
        self,
        output_directory: str | Path,
        *,
        split_seed: int = 0,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        tensor_compression: str = "compressed",
    ):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
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

    def run_cases(
        self,
        engine: DamageEngine,
        cases: Iterable[dict],
        *,
        fail_closed: bool = False,
        workers: int = 1,
        chunksize: int | None = None,
    ) -> dict:
        case_list = list(cases)
        if not case_list:
            raise ValueError("cases must contain at least one case")
        if workers < 1:
            raise ValueError("workers must be at least one")
        if chunksize is not None and chunksize < 1:
            raise ValueError("chunksize must be at least one")
        case_ids = [case.get("case_id") for case in case_list]
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
        return self.write(
            engine,
            case_list,
            results,
            generation={
                "workers": int(workers),
                "chunksize": int(effective_chunksize),
                "failure_policy": "fail_fast" if fail_closed else "isolate_and_record",
                "input_order_preserved": True,
                "tensor_compression": self.tensor_compression,
            },
        )

    def write(
        self,
        engine: DamageEngine,
        cases: list[dict],
        results: list[EngineResult],
        *,
        generation: dict | None = None,
    ) -> dict:
        if len(cases) != len(results):
            raise ValueError("cases and results must have equal length")
        incomplete_path = self.output_directory / ".dataset.incomplete"
        incomplete_path.write_text(
            json.dumps(
                {"started_at": datetime.now(timezone.utc).isoformat(), "case_count": len(cases)},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        raw_path = self.output_directory / "results.jsonl"
        with raw_path.open("w", encoding="utf-8") as stream:
            for result in results:
                stream.write(json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
        payloads = [result.to_dict() for result in results]
        valid_pairs = [
            (case, payload)
            for case, payload in zip(cases, payloads)
            if payload.get("status") == "ok"
        ]
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
                                mechanism == "combined" or mechanism in enabled_mechanisms
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
                                "available": mechanism == "combined" or mechanism in enabled_mechanisms,
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
            if requested_standard_ids:
                system_label_masks.append(
                    np.any(standard_output_mask[requested_mask], axis=0).tolist()
                )
            else:  # The case schema requires at least one standard; retain fail-safe behavior.
                system_label_masks.append([False] * len(output_names))
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
            case_rows.append(
                {
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
            )
            for standard_id in requested_standard_ids:
                for output_name in engine.damage_tree.standards[standard_id]:
                    tree_rows.append(
                        {
                            "case_id": case["case_id"],
                            "evaluation_standard_id": standard_id,
                            "output_node": output_name,
                            **result["damage_tree"]["outputs"][output_name],
                        }
                    )
        graph_node_ids = list(engine.compiled_target.component_order) + list(engine.damage_tree.order)
        graph_node_types = ["component"] * component_count + ["damage_tree"] * len(
            engine.damage_tree.order
        )
        graph_node_index = {identifier: index for index, identifier in enumerate(graph_node_ids)}
        graph_edges: list[tuple[int, int]] = []
        graph_edge_types: list[int] = []
        for node_id, node in engine.damage_tree.nodes.items():
            if node["node_type"] == "leaf":
                edge_rows.append(
                    {
                        "source_id": str(node["component_id"]),
                        "target_id": node_id,
                        "edge_type": "component_state_to_leaf",
                    }
                )
                graph_edges.append(
                    (graph_node_index[str(node["component_id"])], graph_node_index[node_id])
                )
                graph_edge_types.append(0)
            for dependency in node["inputs"]:
                edge_rows.append(
                    {
                        "source_id": dependency,
                        "target_id": node_id,
                        "edge_type": "damage_tree_dependency",
                    }
                )
                graph_edges.append((graph_node_index[dependency], graph_node_index[node_id]))
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
            "component_ids": np.asarray(list(engine.compiled_target.component_order)),
            "armor_ids": np.asarray([item.object_id for item in engine.compiled_target.armor]),
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
                    "format": "jsonl",
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
            "system_label_mask": "[B,O]", "requested_standard_mask": "[B,E]",
            "case_ids": "[B]", "case_hashes": "[B]",
            "split_names": "[B]", "split_group_ids": "[B]", "split_group_sources": "[B]",
            "leakage_keys": "[B]", "aleatory_seeds": "[B]", "epistemic_sample_ids": "[B]",
            "replicate_counts": "[B]", "input_case_ids": "[N]", "input_case_status": "[N]",
            "input_case_valid_mask": "[N]", "input_to_valid_index": "[N]",
            "component_ids": "[C]", "armor_ids": "[A]", "mechanism_names": "[M]",
            "state_names": "[S]", "frechet_bound_names": "[D]",
            "frechet_bound_units": "[D]", "system_output_names": "[O]",
            "evaluation_standard_ids": "[E]", "standard_output_mask": "[E,O]",
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
            "tree_edge_index": "[2,E]", "tree_edge_type": "[E]", "tree_edge_type_names": "[2]",
        }
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
        feature_fields = {"case_x", "component_x", "armor_x", "component_armor_path_x"}
        index_fields = {
            "aleatory_seeds", "replicate_counts", "input_to_valid_index", "armor_vertex_offsets",
            "component_vertex_offsets", "tree_edge_index", "tree_edge_type",
        }
        field_units = {
            "case_x": "mixed; see case_feature_units",
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
        }

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
        identity_hash = dataset_identity_hash(
            target_hash=engine.target_hash,
            physics_registry_hash=engine.registry.hash,
            engine_code_hash=ENGINE_CODE_HASH,
            ordered_case_hashes=[content_hash(case) for case in cases],
            split_seed=self.split_seed,
            split_ratios=self.split_ratios,
            tensor_contract_version="1.3.0",
        )
        manifest = {
            "schema_version": "1.1.0",
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
            "source_contracts": {
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
            },
            "generation": generation or {
                "workers": 1,
                "chunksize": 1,
                "failure_policy": "precomputed_results",
                "input_order_preserved": True,
                "tensor_compression": self.tensor_compression,
            },
            "randomness": {
                "split_seed": self.split_seed,
                "aleatory_seed_field": "analysis.aleatory_seed",
                "epistemic_sample_id_field": "analysis.epistemic_sample_id",
                "parallel_result_order_deterministic": True,
            },
            "partitions": partitions,
            "tensor_contract": {
                "version": "1.3.0",
                "fields": tensor_fields,
                "axes": [
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
                "split_counts": {
                    name: sum(row["split"] == name for row in split_rows)
                    for name in ("train", "validation", "test")
                },
            },
            "quality_summary": {
                "n_total": len(results),
                "n_valid": len(valid_pairs),
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
                "Generation still materializes cases, results, tensor arrays and Parquet rows in memory; use bounded batches because sharding and checkpoint/resume are not implemented.",
                "Each dense NPZ still describes one target topology; TorchMultiTargetDataset and collate_damage_batch provide padding, name alignment, and disjoint-graph batching across those target-specific datasets.",
                "Armor labels now expose integrity and blast-response outputs, but remain tied to the same unvalidated physical registry as the other teacher labels.",
            ],
            "prohibited_uses": [
                "Do not use unvalidated outputs as certified real-world effectiveness data.",
                "Do not use the surrogate in an autonomous control loop without independent safety validation.",
            ],
        }
        manifest_path = self.output_directory / "manifest.json"
        manifest_temp_path = self.output_directory / "manifest.json.tmp"
        with manifest_temp_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        manifest_temp_path.replace(manifest_path)
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
                widths.append(0.5 * (stats["ci95_upper"] - stats["ci95_lower"]))
        if not widths:
            return []
        return [float(value) for value in np.quantile(widths, [0.5, 0.9, 0.99])]
