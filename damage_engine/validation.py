from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from jsonschema import Draft202012Validator

from .damage_tree import compile_damage_tree
from .errors import InputValidationError, UnverifiedModelError
from .fragments import load_fragment_list, validate_fragment_mass_budget
from .normalization import normalize_target
from .registry import PhysicsRegistry
from .schema_resources import load_schema
from .utils import content_hash


_QUATERNION_TOLERANCE = 1.0e-6
_UNIT_VECTOR_TOLERANCE = 1.0e-6


def _validate_schema(instance: dict, schema_name: str) -> None:
    schema = load_schema(schema_name)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(instance),
        key=lambda error: tuple(str(value) for value in error.absolute_path),
    )
    if errors:
        details = []
        for error in errors[:20]:
            path = "/".join(str(value) for value in error.absolute_path) or "$"
            details.append(
                {
                    "path": path,
                    "message": error.message,
                    "validator": error.validator,
                }
            )
        raise InputValidationError(
            f"{schema_name} validation failed",
            context={"errors": details, "error_count": len(errors)},
        )


def _walk_parameters(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        if {"value", "unit", "source_id", "validation_status"}.issubset(value):
            yield path, value
        for key, child in value.items():
            yield from _walk_parameters(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_parameters(child, (*path, str(index)))


def _walk_numeric_values(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_numeric_values(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_numeric_values(child, (*path, str(index)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield path, value


def _validate_finite_numbers(value: Any, *, document: str) -> None:
    for path, number in _walk_numeric_values(value):
        try:
            finite = math.isfinite(float(number))
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            raise InputValidationError(
                "input contains a non-finite numeric value",
                context={"document": document, "path": "/".join(path) or "$", "value": repr(number)},
            )


def _finite_vector(value: Any, name: str, size: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(
            f"{name} must be a finite {size}-vector",
            context={"value": repr(value)},
        ) from exc
    if array.shape != (size,) or not np.isfinite(array).all():
        raise InputValidationError(
            f"{name} must be a finite {size}-vector",
            context={"shape": list(array.shape)},
        )
    return array


def _validate_unit_vector(value: Any, name: str) -> np.ndarray:
    vector = _finite_vector(value, name, 3)
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > _UNIT_VECTOR_TOLERANCE:
        raise InputValidationError(
            f"{name} must have unit norm",
            context={"norm": norm, "tolerance": _UNIT_VECTOR_TOLERANCE},
        )
    return vector


def _validate_quaternion(value: Any, name: str) -> None:
    quaternion = _finite_vector(value, name, 4)
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > _QUATERNION_TOLERANCE:
        raise InputValidationError(
            f"{name} must have unit norm in xyzw order",
            context={"norm": norm, "tolerance": _QUATERNION_TOLERANCE},
        )


def _require_positive(value: float, *, path: str) -> None:
    if value <= 0.0:
        raise InputValidationError(
            "physical parameter must be positive",
            context={"path": path, "value": value},
        )


def _validate_uncertainty(parameter: dict, *, path: str) -> None:
    uncertainty = parameter["uncertainty"]
    kind = uncertainty["kind"]
    parameters = uncertainty["parameters"]
    if kind == "none":
        if parameters:
            raise InputValidationError(
                "uncertainty kind 'none' cannot contain sampling parameters",
                context={"path": f"{path}/uncertainty/parameters", "keys": sorted(parameters)},
            )
        return

    def positive_field(field: str) -> float:
        if field not in parameters:
            raise InputValidationError(
                "uncertainty distribution is missing a required parameter",
                context={"path": f"{path}/uncertainty/parameters", "kind": kind, "missing": field},
            )
        value = float(parameters[field])
        if not math.isfinite(value) or value <= 0.0:
            raise InputValidationError(
                "uncertainty scale must be finite and positive",
                context={"path": f"{path}/uncertainty/parameters/{field}", "value": value},
            )
        return value

    if kind == "normal":
        positive_field("stddev")
    elif kind == "lognormal":
        positive_field("log_sigma")
        if float(parameter["value"]) <= 0.0:
            raise InputValidationError(
                "lognormal uncertainty requires a positive nominal value",
                context={"path": path, "value": parameter["value"]},
            )
    elif kind in {"uniform", "interval"}:
        missing = [field for field in ("minimum", "maximum") if field not in parameters]
        if missing:
            raise InputValidationError(
                "bounded uncertainty is missing required endpoints",
                context={"path": f"{path}/uncertainty/parameters", "kind": kind, "missing": missing},
            )
        minimum = float(parameters["minimum"])
        maximum = float(parameters["maximum"])
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
            raise InputValidationError(
                "bounded uncertainty requires finite minimum < maximum",
                context={"path": f"{path}/uncertainty/parameters", "minimum": minimum, "maximum": maximum},
            )
    elif kind == "empirical":
        values = parameters.get("values")
        if not isinstance(values, list) or not values:
            raise InputValidationError(
                "empirical uncertainty requires a non-empty values array",
                context={"path": f"{path}/uncertainty/parameters/values"},
            )
        numeric_values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(numeric_values).all():
            raise InputValidationError(
                "empirical uncertainty values must be finite",
                context={"path": f"{path}/uncertainty/parameters/values"},
            )
        probabilities = parameters.get("probabilities")
        if probabilities is not None:
            if not isinstance(probabilities, list) or len(probabilities) != len(values):
                raise InputValidationError(
                    "empirical uncertainty probabilities must match values length",
                    context={"path": f"{path}/uncertainty/parameters/probabilities"},
                )
            weights = np.asarray(probabilities, dtype=np.float64)
            if (
                not np.isfinite(weights).all()
                or np.any(weights < 0.0)
                or not math.isclose(float(np.sum(weights)), 1.0, rel_tol=0.0, abs_tol=1.0e-9)
            ):
                raise InputValidationError(
                    "empirical uncertainty probabilities must be non-negative and sum to one",
                    context={
                        "path": f"{path}/uncertainty/parameters/probabilities",
                        "sum": float(np.sum(weights)),
                    },
                )


def _validate_parameter(
    parameter: dict,
    *,
    path: str,
    allowed_units: dict[str, float],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    positive: bool = False,
) -> None:
    unit = parameter["unit"]
    if unit not in allowed_units:
        raise InputValidationError(
            "parameter uses an unsupported or dimensionally inconsistent unit",
            context={"path": f"{path}/unit", "unit": unit, "allowed_units": sorted(allowed_units)},
        )
    scale = allowed_units[unit]
    value = float(parameter["value"])
    valid_range = parameter["valid_range"]
    minimum = float(valid_range[0])
    maximum = float(valid_range[1])
    if minimum > maximum:
        raise InputValidationError(
            "parameter valid_range must be ordered [minimum, maximum]",
            context={"path": f"{path}/valid_range", "valid_range": valid_range},
        )
    if not minimum <= value <= maximum:
        raise InputValidationError(
            "nominal parameter value lies outside valid_range",
            context={"path": path, "value": value, "valid_range": valid_range},
        )
    si_value = value * scale
    si_minimum = minimum * scale
    si_maximum = maximum * scale
    if positive:
        _require_positive(si_value, path=f"{path}/value")
        _require_positive(si_minimum, path=f"{path}/valid_range/0")
    if lower_bound is not None and si_minimum < lower_bound:
        raise InputValidationError(
            "parameter valid_range falls below the physical lower bound",
            context={"path": path, "minimum": si_minimum, "lower_bound": lower_bound},
        )
    if upper_bound is not None and si_maximum > upper_bound:
        raise InputValidationError(
            "parameter valid_range exceeds the physical upper bound",
            context={"path": path, "maximum": si_maximum, "upper_bound": upper_bound},
        )
    _validate_uncertainty(parameter, path=path)


def _validate_geometry(geometry: dict, *, object_id: str, frame_unit: str) -> None:
    shape = geometry["shape"]
    suffix = "m" if any(key.endswith("_m") for key in geometry) else "cm"
    if suffix != frame_unit:
        raise InputValidationError(
            "geometry field units must match coordinate_frame.length_unit before normalization",
            context={"object_id": object_id, "geometry_unit": suffix, "coordinate_frame_unit": frame_unit},
        )
    if shape in {"box", "cylinder"}:
        _validate_quaternion(
            geometry["orientation_xyzw"],
            f"target object {object_id} orientation_xyzw",
        )
        return
    if shape != "extruded_polygon":
        raise InputValidationError(
            "unsupported target geometry shape",
            context={"object_id": object_id, "shape": shape},
        )
    direction = _validate_unit_vector(
        geometry["extrusion_direction"],
        f"target object {object_id} extrusion_direction",
    )
    vertices_key = f"vertices_{suffix}"
    vertices = np.asarray(geometry[vertices_key], dtype=np.float64)
    if not np.isfinite(vertices).all():
        raise InputValidationError(
            "extruded-polygon vertices must be finite",
            context={"object_id": object_id, "field": vertices_key},
        )
    unique_vertices = np.unique(vertices, axis=0)
    if len(unique_vertices) != len(vertices):
        raise InputValidationError(
            "extruded-polygon vertices must not repeat (including a repeated closing vertex)",
            context={"object_id": object_id, "vertex_count": len(vertices), "unique_count": len(unique_vertices)},
        )
    origin = vertices[0]
    plane_offsets = (vertices - origin) @ direction
    scale = max(1.0, float(np.ptp(vertices, axis=0).max()))
    maximum_offset = float(np.max(np.abs(plane_offsets)))
    if maximum_offset > 1.0e-8 * scale:
        raise InputValidationError(
            "extruded-polygon base vertices must be coplanar normal to extrusion_direction",
            context={"object_id": object_id, "maximum_plane_offset": maximum_offset},
        )


def _validate_fragment_mass_distribution(source: dict) -> None:
    distribution = source["mass_distribution"]
    path = "warhead/fragment_source/mass_distribution"
    valid_range = distribution.get("valid_range_kg")
    if valid_range is not None:
        minimum, maximum = map(float, valid_range)
        if minimum >= maximum:
            raise InputValidationError(
                "fragment mass valid_range_kg must be strictly increasing",
                context={"path": f"{path}/valid_range_kg", "valid_range_kg": valid_range},
            )
    parameters = distribution["parameters"]
    family = distribution["family"]
    if family == "empirical" and parameters.get("probabilities") is not None:
        values = parameters["values_kg"]
        probabilities = parameters["probabilities"]
        if len(probabilities) != len(values):
            raise InputValidationError(
                "empirical fragment-mass probabilities must match values_kg length",
                context={"path": f"{path}/parameters/probabilities"},
            )
        total = float(np.sum(np.asarray(probabilities, dtype=np.float64)))
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise InputValidationError(
                "empirical fragment-mass probabilities must sum to one",
                context={"path": f"{path}/parameters/probabilities", "sum": total},
            )
    deterministic_values: Iterable[float] = ()
    if family == "fixed":
        deterministic_values = (float(parameters["value_kg"]),)
    elif family == "empirical":
        deterministic_values = (float(value) for value in parameters["values_kg"])
    if valid_range is not None:
        outside = [value for value in deterministic_values if not minimum <= value <= maximum]
        if outside:
            raise InputValidationError(
                "declared fragment masses lie outside valid_range_kg",
                context={"path": path, "outside_values_kg": outside, "valid_range_kg": valid_range},
            )


def _validate_registry_semantics(registry: PhysicsRegistry) -> None:
    for model_id, model in registry.data["fragment_source_models"].items():
        speed_range = model["valid_speed_mps"]
        if float(speed_range[0]) >= float(speed_range[1]):
            raise InputValidationError(
                "fragment-source valid_speed_mps must be strictly increasing",
                context={"model_id": model_id, "valid_speed_mps": speed_range},
            )
        direction = model["direction"]
        if direction.get("axis_body") is not None:
            _validate_unit_vector(
                direction["axis_body"],
                f"fragment source model {model_id} direction.axis_body",
            )
    for model_id, model in registry.data["blast_field_models"].items():
        reflection = model["reflection"]
        if float(reflection["maximum_factor"]) < float(reflection["minimum_factor"]):
            raise InputValidationError(
                "blast reflection maximum_factor must be >= minimum_factor",
                context={"model_id": model_id, "reflection": reflection},
            )


def validate_inputs(target: dict, case: dict, registry: PhysicsRegistry) -> None:
    target = normalize_target(target)
    _validate_schema(target, "target.schema.json")
    _validate_schema(case, "simulation_case.schema.json")
    _validate_finite_numbers(target, document="target")
    _validate_finite_numbers(case, document="simulation_case")
    _validate_finite_numbers(registry.data, document="physics_registry")
    _validate_registry_semantics(registry)

    target_hash = content_hash(target)
    reference = case["target_ref"]
    if reference["target_id"] != target["target_id"]:
        raise InputValidationError(
            "case target_id does not match target model",
            context={"case_target_id": reference["target_id"], "target_id": target["target_id"]},
        )
    if reference["target_revision"] != target["target_revision"]:
        raise InputValidationError(
            "case target revision does not match target model",
            context={
                "case_target_revision": reference["target_revision"],
                "target_revision": target["target_revision"],
            },
        )
    if reference["content_hash"] != target_hash:
        raise InputValidationError(
            "case target hash does not match canonical target content",
            context={"expected": target_hash, "observed": reference["content_hash"]},
        )
    if case["parameter_set_id"] != registry.data["config_id"]:
        raise InputValidationError(
            "case parameter_set_id does not match the physics registry",
            context={
                "parameter_set_id": case["parameter_set_id"],
                "registry_config_id": registry.data["config_id"],
            },
        )

    requested_standards = case["analysis"]["evaluation_standard_ids"]
    compiled_tree = compile_damage_tree(target)
    unsupported_standards = sorted(set(requested_standards) - set(compiled_tree.standards))
    if unsupported_standards:
        raise InputValidationError(
            "case requests evaluation standards not declared by the target damage tree",
            context={
                "requested": requested_standards,
                "unsupported": unsupported_standards,
                "supported": sorted(compiled_tree.standards),
            },
        )

    frame = target["coordinate_frame"]
    axis_names = [frame["x_axis"], frame["y_axis"], frame["z_axis"]]
    if len(set(axis_names)) != 3:
        raise InputValidationError(
            "coordinate-frame axis descriptions must be distinct",
            context={"axes": axis_names},
        )
    frame_unit = frame["length_unit"]

    object_ids = [str(item["id"]) for item in target["armor"] + target["components"]]
    if len(object_ids) != len(set(object_ids)):
        raise InputValidationError("armor and component IDs must be globally unique")

    for collection_name, items in (("armor", target["armor"]), ("components", target["components"])):
        for item in items:
            _validate_target_item(
                item,
                collection_name=collection_name,
                registry=registry,
                frame_unit=frame_unit,
            )

    source = case["warhead"]["fragment_source"]
    _validate_fragment_mass_distribution(source)
    source_model = registry.fragment_source(source["model_id"])
    fragment_list = (
        load_fragment_list(source)
        if source.get("fragment_list_ref") is not None
        else None
    )
    if fragment_list is not None:
        validate_fragment_mass_budget(
            case["warhead"],
            source_model,
            fragment_list.total_mass_kg,
        )
        valid_speed = source_model.get("valid_speed_mps")
        if valid_speed is not None:
            lower_speed, upper_speed = map(float, valid_speed)
            outside = [
                record.initial_velocity_mps
                for record in fragment_list.records
                if not lower_speed <= record.initial_velocity_mps <= upper_speed
            ]
            if outside:
                raise InputValidationError(
                    "fragment-list body-frame initial speed is outside source model validity",
                    context={
                        "outside_count": len(outside),
                        "minimum_speed_mps": min(outside),
                        "maximum_speed_mps": max(outside),
                        "valid_speed_mps": [lower_speed, upper_speed],
                    },
                )
    declared_models = {
        "direction_model_id": source_model["direction"]["kind"],
        "drag_model_id": source_model["drag"]["kind"],
        "initial_velocity_model_id": source_model["initial_velocity"]["kind"],
    }
    for field, expected in declared_models.items():
        if source[field] != expected:
            raise InputValidationError(
                f"fragment source {field} does not match the registered source model",
                context={"field": field, "declared": source[field], "expected": expected},
            )
    if (
        fragment_list is None
        and source_model["initial_velocity"]["kind"] == "gurney_cylinder"
    ):
        missing = [
            field
            for field in ("charge_mass_kg", "casing_mass_kg")
            if case["warhead"].get(field) is None
        ]
        if missing:
            raise InputValidationError(
                "Gurney fragment source requires charge_mass_kg and casing_mass_kg",
                context={"missing": missing},
            )
        if source.get("gurney_constant_mps") is None:
            raise InputValidationError("Gurney fragment source requires gurney_constant_mps")
    if (
        fragment_list is None
        and source_model["direction"]["kind"] == "taylor_ring"
        and source.get("detonation_velocity_mps") is None
    ):
        raise InputValidationError("taylor_ring fragment source requires detonation_velocity_mps")

    registry.blast_field(case["warhead"]["blast_source"]["model_id"])
    _finite_vector(case["encounter"]["detonation_position_m"], "detonation_position_m", 3)
    _finite_vector(case["encounter"]["velocity_target_mps"], "velocity_target_mps", 3)
    _validate_quaternion(
        case["encounter"]["body_to_target_orientation_xyzw"],
        "encounter body_to_target_orientation_xyzw",
    )

    if case["analysis"]["fidelity_level"] == "L3":
        registry.require_validated()
        unverified = [
            parameter["source_id"]
            for _, parameter in _walk_parameters(target)
            if parameter["validation_status"] != "validated"
        ]
        provenance = [item["provenance"] for item in target["armor"] + target["components"]]
        unapproved = [item["source_id"] for item in provenance if item["status"] != "approved"]
        if unverified or unapproved:
            raise UnverifiedModelError(
                "L3 execution requires validated parameters and approved provenance",
                context={
                    "unverified_parameter_sources": sorted(set(unverified)),
                    "unapproved_sources": sorted(set(unapproved)),
                },
            )

    # Compilation above validates the full DAG and its evaluation-standard views.


def _validate_target_item(
    item: dict,
    *,
    collection_name: str,
    registry: PhysicsRegistry,
    frame_unit: str,
) -> None:
    object_id = str(item["id"])
    _validate_geometry(item["geometry"], object_id=object_id, frame_unit=frame_unit)
    response = item["response"]
    path = f"target/{collection_name}/{object_id}/response"
    _validate_parameter(
        response["equivalent_thickness"],
        path=f"{path}/equivalent_thickness",
        allowed_units={"m": 1.0, "cm": 1.0e-2, "mm": 1.0e-3},
        positive=True,
    )
    _validate_parameter(
        response["vulnerable_area_ratio"],
        path=f"{path}/vulnerable_area_ratio",
        allowed_units={"1": 1.0, "%": 1.0e-2},
        lower_bound=0.0,
        upper_bound=1.0,
    )
    blast_response = response["blast_response"]
    threshold_specs = (
        ("peak_overpressure_threshold", {"Pa": 1.0, "kPa": 1.0e3, "MPa": 1.0e6}),
        ("impulse_threshold", {"Pa*s": 1.0, "kPa*ms": 1.0, "kPa*s": 1.0e3, "MPa*ms": 1.0e3}),
        ("duration_threshold", {"s": 1.0, "ms": 1.0e-3}),
    )
    for field, units in threshold_specs:
        if blast_response.get(field) is not None:
            _validate_parameter(
                blast_response[field],
                path=f"{path}/blast_response/{field}",
                allowed_units=units,
                positive=True,
            )

    fragment_model_id = response["fragment_response_model_id"]
    fragment_model = registry.fragment_response(fragment_model_id)
    blast_model = registry.blast_response(blast_response["model_id"])
    if collection_name == "components":
        states = item["damage_states"]
        if states[0] != "intact":
            raise InputValidationError(
                "the first component damage state must be intact",
                context={"component_id": object_id, "damage_states": states},
            )
        modeled_states = states[1:]
        for mechanism, model in (("fragment", fragment_model), ("blast", blast_model)):
            missing = [state for state in modeled_states if state not in model["damage_states"]]
            if missing:
                raise InputValidationError(
                    "component damage states are missing from a referenced response model",
                    context={
                        "component_id": object_id,
                        "mechanism": mechanism,
                        "model_id": fragment_model_id if mechanism == "fragment" else blast_response["model_id"],
                        "missing_states": missing,
                    },
                )
        scale_groups = (
            (
                "fragment residual-energy",
                [float(fragment_model["damage_states"][state]["residual_energy_50_j"]) for state in modeled_states],
            ),
            (
                "blast peak-pressure",
                [float(blast_model["damage_states"][state]["peak_50_pa"]) for state in modeled_states],
            ),
            (
                "blast impulse",
                [float(blast_model["damage_states"][state]["impulse_50_pa_s"]) for state in modeled_states],
            ),
        )
        for name, values in scale_groups:
            if any(second < first for first, second in zip(values, values[1:])):
                raise InputValidationError(
                    f"{name} state scales must be non-decreasing with severity",
                    context={"component_id": object_id, "values": values, "damage_states": modeled_states},
                )
    else:
        registry.transmission(response["material_id"])
