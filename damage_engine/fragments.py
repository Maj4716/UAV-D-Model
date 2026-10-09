from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import InputValidationError, OutOfDomainError
from .geometry import CompiledObject, CompiledTarget, polygon_area_3d
from .registry import PhysicsRegistry
from .utils import EPS, normalize, quaternion_matrix_xyzw, sigmoid


@dataclass(frozen=True)
class Fragment:
    fragment_id: int
    mass_kg: float
    origin_m: np.ndarray
    direction: np.ndarray
    initial_speed_mps: float
    area_m2: float
    drag_coefficient: float
    source_fragment_id: str | None = None
    source_multiplicity_index: int = 1
    source_multiplicity: int = 1


@dataclass(frozen=True)
class FragmentListRecord:
    source_fragment_id: str | None
    mass_kg: float
    direction_body_xyz: tuple[float, float, float]
    initial_velocity_mps: float
    multiplicity: int


@dataclass(frozen=True)
class FragmentListData:
    records: tuple[FragmentListRecord, ...]
    expanded_count: int
    total_mass_kg: float
    provenance: dict[str, Any]


@dataclass
class FragmentSimulation:
    component_state_probabilities: dict[str, dict[str, float]]
    component_metrics: dict[str, dict[str, float]]
    armor_integrity: dict[str, float]
    total_fragments: int
    total_armor_hits: int
    total_armor_penetrations: int
    total_component_hits: int
    total_component_penetrations: int
    audit: list[dict] = field(default_factory=list)
    fragment_list_provenance: dict[str, Any] | None = None


_FRAGMENT_LIST_FIELDS = {
    "id",
    "mass_kg",
    "direction_body_xyz",
    "direction_body_x",
    "direction_body_y",
    "direction_body_z",
    "initial_velocity_mps",
    "multiplicity",
}


def _fragment_list_error(
    message: str,
    *,
    path: Path,
    record_index: int | None = None,
    field_name: str | None = None,
    value: Any = None,
) -> InputValidationError:
    context: dict[str, Any] = {"fragment_list_path": str(path)}
    if record_index is not None:
        context["record_index"] = record_index
    if field_name is not None:
        context["field"] = field_name
    if value is not None:
        context["value"] = value
    return InputValidationError(message, context=context)


def _load_fragment_list_documents(path: Path, text: str) -> tuple[list[dict], str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        reader = csv.reader(io.StringIO(text, newline=""))
        try:
            header = next(reader)
        except StopIteration as exc:
            raise _fragment_list_error("fragment-list CSV is empty", path=path) from exc
        header = [value.strip() for value in header]
        duplicates = sorted({name for name in header if header.count(name) > 1})
        if duplicates:
            raise _fragment_list_error(
                "fragment-list CSV headers must be unique",
                path=path,
                value=duplicates,
            )
        unknown = sorted(set(header) - _FRAGMENT_LIST_FIELDS)
        if unknown:
            raise _fragment_list_error(
                "fragment-list CSV contains unsupported columns",
                path=path,
                value=unknown,
            )
        required = {"mass_kg", "initial_velocity_mps"}
        missing = sorted(required - set(header))
        vector_column = "direction_body_xyz" in header
        split_field_names = {
            "direction_body_x",
            "direction_body_y",
            "direction_body_z",
        }
        present_split_columns = split_field_names.intersection(header)
        split_columns = present_split_columns == split_field_names
        invalid_direction_columns = (
            vector_column and bool(present_split_columns)
        ) or (not vector_column and not split_columns)
        if missing or invalid_direction_columns:
            raise _fragment_list_error(
                "fragment-list CSV requires mass_kg, initial_velocity_mps, and exactly one direction representation",
                path=path,
                value={
                    "missing": missing,
                    "has_direction_body_xyz": vector_column,
                    "has_split_direction_columns": split_columns,
                },
            )
        documents: list[dict] = []
        for line_number, values in enumerate(reader, start=2):
            if not values or all(not value.strip() for value in values):
                continue
            if len(values) != len(header):
                raise _fragment_list_error(
                    "fragment-list CSV row length does not match its header",
                    path=path,
                    record_index=line_number,
                    value={"expected": len(header), "observed": len(values)},
                )
            record = dict(zip(header, values, strict=True))
            if vector_column:
                raw_direction = record["direction_body_xyz"].strip()
                try:
                    record["direction_body_xyz"] = json.loads(raw_direction)
                except json.JSONDecodeError as exc:
                    raise _fragment_list_error(
                        "direction_body_xyz in CSV must be a JSON array",
                        path=path,
                        record_index=line_number,
                        field_name="direction_body_xyz",
                        value=raw_direction,
                    ) from exc
            else:
                record["direction_body_xyz"] = [
                    record.pop("direction_body_x"),
                    record.pop("direction_body_y"),
                    record.pop("direction_body_z"),
                ]
            documents.append(record)
        return documents, "csv"
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise _fragment_list_error(
                "fragment-list JSON is not valid JSON",
                path=path,
                value={"line": exc.lineno, "column": exc.colno},
            ) from exc
        if isinstance(payload, dict):
            if set(payload) != {"fragments"}:
                raise _fragment_list_error(
                    "fragment-list JSON object must contain only a fragments array",
                    path=path,
                    value=sorted(payload),
                )
            payload = payload["fragments"]
        if not isinstance(payload, list):
            raise _fragment_list_error(
                "fragment-list JSON root must be an array or a fragments object",
                path=path,
            )
        return payload, "json"
    if suffix == ".jsonl":
        documents = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _fragment_list_error(
                    "fragment-list JSONL line is not valid JSON",
                    path=path,
                    record_index=line_number,
                    value={"column": exc.colno},
                ) from exc
            documents.append(value)
        return documents, "jsonl"
    raise _fragment_list_error(
        "fragment_list_ref must use .csv, .json, or .jsonl",
        path=path,
        value=suffix,
    )


def _parse_positive_float(value: Any, *, path: Path, index: int, field_name: str) -> float:
    if isinstance(value, bool):
        raise _fragment_list_error(
            "fragment-list numeric values must not be booleans",
            path=path,
            record_index=index,
            field_name=field_name,
            value=value,
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise _fragment_list_error(
            "fragment-list value must be numeric",
            path=path,
            record_index=index,
            field_name=field_name,
            value=value,
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise _fragment_list_error(
            "fragment-list value must be finite and positive",
            path=path,
            record_index=index,
            field_name=field_name,
            value=value,
        )
    return result


def _parse_multiplicity(value: Any, *, path: Path, index: int) -> int:
    if value in (None, ""):
        return 1
    if isinstance(value, bool):
        raise _fragment_list_error(
            "fragment multiplicity must be a positive integer",
            path=path,
            record_index=index,
            field_name="multiplicity",
            value=value,
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise _fragment_list_error(
            "fragment multiplicity must be a positive integer",
            path=path,
            record_index=index,
            field_name="multiplicity",
            value=value,
        ) from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 1:
        raise _fragment_list_error(
            "fragment multiplicity must be a positive integer",
            path=path,
            record_index=index,
            field_name="multiplicity",
            value=value,
        )
    return int(numeric)


def load_fragment_list(source: dict) -> FragmentListData:
    """Load a deterministic joint fragment list referenced by a simulation case.

    Relative references are resolved against the process working directory.  The
    file is read once so that the returned digest identifies the exact bytes that
    were parsed.
    """

    reference = source.get("fragment_list_ref")
    if not isinstance(reference, str) or not reference.strip():
        raise InputValidationError("fragment_list_ref must be a non-empty path string")
    requested_path = Path(reference).expanduser()
    try:
        path = requested_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise InputValidationError(
            "fragment_list_ref does not resolve to an existing local file",
            context={"fragment_list_ref": reference},
        ) from exc
    if not path.is_file():
        raise InputValidationError(
            "fragment_list_ref must resolve to a regular file",
            context={"fragment_list_ref": reference, "resolved_path": str(path)},
        )
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise _fragment_list_error(
            "fragment list must be UTF-8 encoded",
            path=path,
            value={"byte_offset": exc.start},
        ) from exc
    except OSError as exc:
        raise _fragment_list_error("fragment list could not be read", path=path) from exc
    documents, format_name = _load_fragment_list_documents(path, text)
    if not documents:
        raise _fragment_list_error("fragment list must contain at least one record", path=path)

    mass_range = source.get("mass_distribution", {}).get("valid_range_kg")
    if mass_range is not None:
        lower_mass, upper_mass = map(float, mass_range)
        if (
            not math.isfinite(lower_mass)
            or not math.isfinite(upper_mass)
            or lower_mass <= 0.0
            or lower_mass > upper_mass
        ):
            raise InputValidationError(
                "fragment mass validity range must be finite, positive, and ordered",
                context={"valid_range_kg": mass_range},
            )
    else:
        lower_mass = upper_mass = None
    records: list[FragmentListRecord] = []
    observed_ids: set[str] = set()
    explicit_id_count = 0
    multiplicity_used = False
    for index, document in enumerate(documents, start=1):
        if not isinstance(document, dict):
            raise _fragment_list_error(
                "each fragment-list record must be an object",
                path=path,
                record_index=index,
                value=type(document).__name__,
            )
        unknown = sorted(set(document) - _FRAGMENT_LIST_FIELDS)
        if unknown:
            raise _fragment_list_error(
                "fragment-list record contains unsupported fields",
                path=path,
                record_index=index,
                value=unknown,
            )
        if any(
            field in document
            for field in ("direction_body_x", "direction_body_y", "direction_body_z")
        ):
            raise _fragment_list_error(
                "JSON and JSONL records must use direction_body_xyz",
                path=path,
                record_index=index,
            )
        missing = sorted(
            field
            for field in ("mass_kg", "direction_body_xyz", "initial_velocity_mps")
            if field not in document or document[field] in (None, "")
        )
        if missing:
            raise _fragment_list_error(
                "fragment-list record is missing required fields",
                path=path,
                record_index=index,
                value=missing,
            )
        raw_id = document.get("id")
        source_id: str | None
        if raw_id in (None, ""):
            source_id = None
        elif isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
            raise _fragment_list_error(
                "fragment id must be a non-empty string or integer",
                path=path,
                record_index=index,
                field_name="id",
                value=raw_id,
            )
        else:
            source_id = str(raw_id).strip()
            if not source_id:
                raise _fragment_list_error(
                    "fragment id must not be blank",
                    path=path,
                    record_index=index,
                    field_name="id",
                )
            if source_id in observed_ids:
                raise _fragment_list_error(
                    "fragment ids must be unique across source records",
                    path=path,
                    record_index=index,
                    field_name="id",
                    value=source_id,
                )
            observed_ids.add(source_id)
            explicit_id_count += 1
        mass = _parse_positive_float(
            document["mass_kg"], path=path, index=index, field_name="mass_kg"
        )
        if lower_mass is not None and not lower_mass <= mass <= upper_mass:
            raise OutOfDomainError(
                "fragment-list mass is outside the source validity range",
                context={
                    "fragment_list_path": str(path),
                    "record_index": index,
                    "mass_kg": mass,
                    "valid_range_kg": [lower_mass, upper_mass],
                },
            )
        direction = document["direction_body_xyz"]
        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            raise _fragment_list_error(
                "direction_body_xyz must contain exactly three numeric values",
                path=path,
                record_index=index,
                field_name="direction_body_xyz",
                value=direction,
            )
        try:
            direction_array = np.asarray(direction, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise _fragment_list_error(
                "direction_body_xyz must contain exactly three numeric values",
                path=path,
                record_index=index,
                field_name="direction_body_xyz",
                value=direction,
            ) from exc
        norm = float(np.linalg.norm(direction_array))
        if not np.isfinite(direction_array).all() or not math.isfinite(norm) or norm <= EPS:
            raise _fragment_list_error(
                "direction_body_xyz must be a finite non-zero vector",
                path=path,
                record_index=index,
                field_name="direction_body_xyz",
                value=direction,
            )
        if abs(norm - 1.0) > 1.0e-6:
            raise _fragment_list_error(
                "direction_body_xyz must have unit norm",
                path=path,
                record_index=index,
                field_name="direction_body_xyz",
                value={"vector": direction, "norm": norm},
            )
        direction_array /= norm
        speed = _parse_positive_float(
            document["initial_velocity_mps"],
            path=path,
            index=index,
            field_name="initial_velocity_mps",
        )
        multiplicity = _parse_multiplicity(
            document.get("multiplicity"), path=path, index=index
        )
        multiplicity_used |= multiplicity != 1
        records.append(
            FragmentListRecord(
                source_fragment_id=source_id,
                mass_kg=mass,
                direction_body_xyz=tuple(float(value) for value in direction_array),
                initial_velocity_mps=speed,
                multiplicity=multiplicity,
            )
        )

    expanded_count = sum(record.multiplicity for record in records)
    expected_count = int(source["total_count"])
    if expanded_count != expected_count:
        raise InputValidationError(
            "fragment-list expanded count does not match fragment_source.total_count",
            context={
                "fragment_list_path": str(path),
                "record_count": len(records),
                "expanded_count": expanded_count,
                "total_count": expected_count,
            },
        )
    total_mass = math.fsum(record.mass_kg * record.multiplicity for record in records)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    provenance = {
        "reference": reference,
        "resolved_path": str(path),
        "format": format_name,
        "content_hash": digest,
        "record_count": len(records),
        "expanded_fragment_count": expanded_count,
        "total_fragment_mass_kg": total_mass,
        "explicit_id_count": explicit_id_count,
        "multiplicity_used": multiplicity_used,
        "joint_fields": [
            "mass_kg",
            "direction_body_xyz",
            "initial_velocity_mps",
        ],
    }
    return FragmentListData(
        records=tuple(records),
        expanded_count=expanded_count,
        total_mass_kg=total_mass,
        provenance=provenance,
    )


def validate_fragment_mass_budget(
    warhead: dict,
    source_model: dict,
    fragment_mass_kg: float,
) -> None:
    casing_mass = float(warhead.get("casing_mass_kg") or 0.0)
    basis = source_model["casing_mass_basis"]
    total_metal_mass = (
        casing_mass if basis == "includes_fragments" else casing_mass + fragment_mass_kg
    )
    if total_metal_mass <= 0:
        raise InputValidationError("fragment source requires a positive total metal mass")
    fragment_fraction = fragment_mass_kg / total_metal_mass
    maximum_fraction = float(source_model["maximum_fragment_mass_fraction_of_total_metal"])
    if not math.isfinite(maximum_fraction) or not 0.0 < maximum_fraction <= 1.0:
        raise InputValidationError(
            "maximum fragment mass fraction must be finite and lie in (0,1]"
        )
    if fragment_fraction > maximum_fraction + 1.0e-12:
        raise OutOfDomainError(
            "fragment mass violates the configured metal mass budget",
            context={
                "fragment_mass_kg": fragment_mass_kg,
                "total_metal_mass_kg": total_metal_mass,
                "fragment_fraction": fragment_fraction,
                "maximum_fraction": maximum_fraction,
            },
        )


def _sample_masses(source: dict, rng: np.random.Generator) -> np.ndarray:
    count = int(source["total_count"])
    if count <= 0:
        raise InputValidationError("fragment total_count must be positive")
    distribution = source["mass_distribution"]
    family = distribution["family"]
    parameters = distribution["parameters"]
    if family == "fixed":
        value = float(parameters["value_kg"])
        if not math.isfinite(value) or value <= 0:
            raise InputValidationError("fixed fragment mass must be finite and positive")
        values = np.full(count, value, dtype=np.float64)
    elif family == "lognormal":
        median = float(parameters["median_kg"])
        sigma = float(parameters["log_sigma"])
        if not math.isfinite(median) or median <= 0 or not math.isfinite(sigma) or sigma < 0:
            raise InputValidationError(
                "lognormal fragment median must be positive and log_sigma non-negative"
            )
        values = rng.lognormal(math.log(median), sigma, count)
    elif family == "weibull":
        scale = float(parameters["scale_kg"])
        shape = float(parameters["shape"])
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(shape) or shape <= 0:
            raise InputValidationError(
                "Weibull fragment scale and shape must be finite and positive"
            )
        values = scale * rng.weibull(shape, count)
    elif family == "empirical":
        empirical = np.asarray(parameters["values_kg"], dtype=np.float64)
        probabilities = parameters.get("probabilities")
        if empirical.ndim != 1 or empirical.size == 0:
            raise InputValidationError("empirical fragment masses must be a non-empty vector")
        if not np.isfinite(empirical).all() or np.any(empirical <= 0):
            raise InputValidationError("empirical fragment masses must be finite and positive")
        if probabilities is not None:
            probabilities = np.asarray(probabilities, dtype=np.float64)
            if probabilities.shape != empirical.shape:
                raise InputValidationError(
                    "empirical fragment probabilities must match the mass vector"
                )
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
                raise InputValidationError(
                    "empirical fragment probabilities must be finite and non-negative"
                )
            total_probability = float(np.sum(probabilities))
            if total_probability <= 0 or not math.isclose(
                total_probability, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-12
            ):
                raise InputValidationError(
                    "empirical fragment probabilities must sum to one",
                    context={"probability_sum": total_probability},
                )
            probabilities = probabilities / total_probability
        values = rng.choice(empirical, count, p=probabilities)
    else:
        raise InputValidationError(f"unsupported fragment mass distribution: {family}")
    if np.any(values <= 0) or not np.isfinite(values).all():
        raise InputValidationError("fragment masses must be finite and positive")
    valid_range = distribution.get("valid_range_kg")
    if valid_range:
        lower, upper = map(float, valid_range)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower <= 0 or lower > upper:
            raise InputValidationError(
                "fragment mass validity range must be finite, positive, and ordered"
            )
        if np.any(values < lower) or np.any(values > upper):
            raise OutOfDomainError(
                "sampled fragment mass is outside the source validity range",
                context={"valid_range_kg": [lower, upper]},
            )
    return values


def _fragment_area(mass_kg: float, density: float, shape_factor: float) -> float:
    if (
        not math.isfinite(mass_kg)
        or mass_kg <= 0
        or not math.isfinite(density)
        or density <= 0
        or not math.isfinite(shape_factor)
        or shape_factor <= 0
    ):
        raise InputValidationError(
            "fragment mass, density, and area shape factor must be finite and positive"
        )
    equivalent_diameter = (6.0 * mass_kg / (math.pi * density)) ** (1.0 / 3.0)
    return float(shape_factor * math.pi * equivalent_diameter**2 / 4.0)


def _sample_directions(
    count: int,
    source_model: dict,
    rotation: np.ndarray,
    rng: np.random.Generator,
    *,
    gurney_speed: float,
    detonation_velocity_mps: float | None,
) -> np.ndarray:
    direction = source_model["direction"]
    kind = direction["kind"]
    axis = normalize(np.asarray(direction.get("axis_body", [1.0, 0.0, 0.0])), name="fragment axis")
    reference = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    basis1 = normalize(np.cross(axis, reference), name="fragment direction basis")
    basis2 = normalize(np.cross(axis, basis1), name="fragment direction basis")
    azimuth = rng.uniform(0.0, 2.0 * math.pi, count)
    if kind == "isotropic":
        cos_theta = rng.uniform(-1.0, 1.0, count)
        theta = np.arccos(cos_theta)
    elif kind == "cone":
        half_angle = math.radians(float(direction["half_angle_deg"]))
        cos_theta = rng.uniform(math.cos(half_angle), 1.0, count)
        theta = np.arccos(cos_theta)
    elif kind == "taylor_ring":
        if not detonation_velocity_mps or detonation_velocity_mps <= 0:
            raise InputValidationError("taylor_ring requires detonation_velocity_mps")
        base = math.atan2(2.0 * gurney_speed, detonation_velocity_mps)
        sigma = math.radians(float(direction.get("spread_sigma_deg", 0.0)))
        theta = np.clip(rng.normal(base, sigma, count), 0.0, math.pi)
    else:
        raise InputValidationError(f"unsupported direction model: {kind}")
    values = (
        np.cos(theta)[:, None] * axis
        + (np.sin(theta) * np.cos(azimuth))[:, None] * basis1
        + (np.sin(theta) * np.sin(azimuth))[:, None] * basis2
    )
    values = (rotation @ values.T).T
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def generate_fragments(
    case: dict,
    registry: PhysicsRegistry,
    rng: np.random.Generator,
    *,
    fragment_list_data: FragmentListData | None = None,
) -> list[Fragment]:
    warhead = case["warhead"]
    source = warhead["fragment_source"]
    source_model = registry.fragment_source(source["model_id"])
    list_reference = source.get("fragment_list_ref")
    source_ids: list[str | None]
    multiplicity_indices: list[int]
    multiplicities: list[int]
    if list_reference is not None:
        list_data = fragment_list_data or load_fragment_list(source)
        masses_list: list[float] = []
        body_directions_list: list[tuple[float, float, float]] = []
        body_speeds_list: list[float] = []
        source_ids = []
        multiplicity_indices = []
        multiplicities = []
        for record in list_data.records:
            for multiplicity_index in range(1, record.multiplicity + 1):
                masses_list.append(record.mass_kg)
                body_directions_list.append(record.direction_body_xyz)
                body_speeds_list.append(record.initial_velocity_mps)
                source_ids.append(record.source_fragment_id)
                multiplicity_indices.append(multiplicity_index)
                multiplicities.append(record.multiplicity)
        masses = np.asarray(masses_list, dtype=np.float64)
        body_directions = np.asarray(body_directions_list, dtype=np.float64)
        body_speeds = np.asarray(body_speeds_list, dtype=np.float64)
    else:
        list_data = None
        masses = _sample_masses(source, rng)
        source_ids = [None] * len(masses)
        multiplicity_indices = [1] * len(masses)
        multiplicities = [1] * len(masses)
    fragment_mass = float(np.sum(masses))
    validate_fragment_mass_budget(warhead, source_model, fragment_mass)
    basis = source_model["casing_mass_basis"]
    valid_speed = source_model.get("valid_speed_mps")
    rotation = quaternion_matrix_xyzw(case["encounter"]["body_to_target_orientation_xyzw"])
    if list_data is not None:
        if valid_speed:
            lower_speed, upper_speed = map(float, valid_speed)
            outside = (body_speeds < lower_speed) | (body_speeds > upper_speed)
            if np.any(outside):
                raise OutOfDomainError(
                    "fragment-list body-frame initial speed is outside source model validity",
                    context={
                        "minimum_speed_mps": float(np.min(body_speeds)),
                        "maximum_speed_mps": float(np.max(body_speeds)),
                        "outside_count": int(np.count_nonzero(outside)),
                        "valid_speed_mps": [lower_speed, upper_speed],
                    },
                )
        directions = (rotation @ body_directions.T).T
        velocity_vectors = directions * body_speeds[:, None]
    else:
        initial = source_model["initial_velocity"]
        kind = initial["kind"]
        if kind == "fixed":
            gurney_speed = float(initial["velocity_mps"])
        elif kind == "gurney_cylinder":
            charge = float(warhead["charge_mass_kg"])
            casing = float(warhead["casing_mass_kg"])
            if basis == "excludes_fragments":
                casing += fragment_mass
            if charge <= 0 or casing <= 0:
                raise InputValidationError("Gurney model requires positive charge and casing masses")
            constant = float(source["gurney_constant_mps"])
            if not math.isfinite(constant) or constant <= 0:
                raise InputValidationError("Gurney constant must be finite and positive")
            gurney_speed = constant / math.sqrt(casing / charge + 0.5)
        else:
            raise InputValidationError(f"unsupported initial velocity model: {kind}")
        if valid_speed and not (
            float(valid_speed[0]) <= gurney_speed <= float(valid_speed[1])
        ):
            raise OutOfDomainError(
                "fragment initial speed is outside source model validity",
                context={"speed_mps": gurney_speed, "valid_speed_mps": valid_speed},
            )
        directions = _sample_directions(
            len(masses),
            source_model,
            rotation,
            rng,
            gurney_speed=gurney_speed,
            detonation_velocity_mps=source.get("detonation_velocity_mps"),
        )
        velocity_vectors = directions * gurney_speed
    projectile_velocity = np.asarray(case["encounter"]["velocity_target_mps"], dtype=np.float64)
    if projectile_velocity.shape != (3,) or not np.isfinite(projectile_velocity).all():
        raise InputValidationError("velocity_target_mps must be a finite 3-vector")
    velocity_vectors = velocity_vectors + projectile_velocity
    speeds = np.linalg.norm(velocity_vectors, axis=1)
    if not np.isfinite(speeds).all() or np.any(speeds <= EPS):
        finite_speeds = speeds[np.isfinite(speeds)]
        raise OutOfDomainError(
            "fragment target-frame initial speed must be finite and positive",
            context={
                "minimum_finite_speed_mps": (
                    float(np.min(finite_speeds)) if finite_speeds.size else None
                ),
                "maximum_finite_speed_mps": (
                    float(np.max(finite_speeds)) if finite_speeds.size else None
                ),
                "nonfinite_count": int(np.count_nonzero(~np.isfinite(speeds))),
            },
        )
    if valid_speed:
        lower_speed, upper_speed = map(float, valid_speed)
        outside = (speeds < lower_speed) | (speeds > upper_speed)
        if np.any(outside):
            raise OutOfDomainError(
                "fragment target-frame initial speed is outside source model validity",
                context={
                    "minimum_speed_mps": float(np.min(speeds)),
                    "maximum_speed_mps": float(np.max(speeds)),
                    "outside_count": int(np.count_nonzero(outside)),
                    "valid_speed_mps": [lower_speed, upper_speed],
                },
            )
    directions = velocity_vectors / speeds[:, None]
    density = float(source["density_kg_m3"])
    drag = source_model["drag"]
    if drag["kind"] != "constant_cd":
        raise InputValidationError("only constant_cd drag is implemented")
    drag_coefficient = float(drag["coefficient"])
    shape_factor = float(drag.get("area_shape_factor", 1.0))
    if not math.isfinite(drag_coefficient) or drag_coefficient <= 0:
        raise InputValidationError("fragment drag coefficient must be finite and positive")
    origin = np.asarray(case["encounter"]["detonation_position_m"], dtype=np.float64)
    return [
        Fragment(
            fragment_id=index,
            mass_kg=float(mass),
            origin_m=origin.copy(),
            direction=directions[index].copy(),
            initial_speed_mps=float(speeds[index]),
            area_m2=_fragment_area(float(mass), density, shape_factor),
            drag_coefficient=drag_coefficient,
            source_fragment_id=source_ids[index],
            source_multiplicity_index=multiplicity_indices[index],
            source_multiplicity=multiplicities[index],
        )
        for index, mass in enumerate(masses)
    ]


def _arrival_speed(
    speed: float,
    distance_m: float,
    fragment: Fragment,
    air_density: float,
) -> float:
    if not math.isfinite(speed) or speed < 0 or not math.isfinite(distance_m):
        raise InputValidationError("fragment speed and flight distance must be finite and non-negative")
    if not math.isfinite(air_density) or air_density <= 0:
        raise InputValidationError("air density must be finite and positive")
    coefficient = (
        air_density
        * fragment.drag_coefficient
        * fragment.area_m2
        / (2.0 * fragment.mass_kg)
    )
    return float(speed * math.exp(-coefficient * max(0.0, distance_m)))


def _ballistic_response(
    speed_mps: float,
    fragment: Fragment,
    normal: np.ndarray,
    direction: np.ndarray,
    thickness_m: float,
    model: dict,
) -> tuple[float, float, float, float]:
    if not math.isfinite(speed_mps) or speed_mps < 0:
        raise InputValidationError("fragment impact speed must be finite and non-negative")
    if not math.isfinite(thickness_m) or thickness_m <= 0:
        raise InputValidationError("equivalent thickness must be finite and positive")
    ballistic = model["ballistic_limit"]
    cosine = float(np.clip(abs(np.dot(direction, normal)), 1.0e-3, 1.0))
    impact_angle = math.acos(cosine)
    maximum_angle = math.radians(float(ballistic.get("maximum_obliquity_deg", 89.0)))
    if impact_angle > maximum_angle:
        return 0.0, math.inf, 0.0, impact_angle
    mass_g = fragment.mass_kg * 1000.0
    thickness_mm = thickness_m * 1000.0
    v50 = (
        float(ballistic["coefficient"])
        * thickness_mm ** float(ballistic["thickness_exponent"])
        * mass_g ** float(ballistic["mass_exponent"])
        * (1.0 / cosine) ** float(ballistic["obliquity_exponent"])
    )
    if not math.isfinite(v50) or v50 <= 0:
        raise InputValidationError("ballistic-limit model produced a non-positive or non-finite V50")
    margin = speed_mps / max(v50, EPS)
    transition = float(ballistic.get("transition_width_ratio", 0.08))
    if not math.isfinite(transition) or transition <= 0:
        raise InputValidationError("ballistic transition width must be finite and positive")
    probability = sigmoid((margin - 1.0) / max(transition, 1.0e-6))
    return probability, float(v50), float(margin), float(impact_angle)


def _residual_speed(speed_mps: float, v50_mps: float) -> float:
    if speed_mps <= v50_mps:
        return 0.0
    return float(math.sqrt(max(0.0, speed_mps**2 - v50_mps**2)))


def _armor_area(item: CompiledObject) -> float:
    geometry = item.source["geometry"]
    if geometry["shape"] == "extruded_polygon":
        return max(
            polygon_area_3d(geometry["vertices_m"], np.asarray(geometry["extrusion_direction"])),
            EPS,
        )
    size = np.sort(item.geometry.representative_size())
    return max(float(size[-1] * size[-2]), EPS)


def simulate_fragments(
    case: dict,
    target: CompiledTarget,
    registry: PhysicsRegistry,
    rng: np.random.Generator,
    *,
    parameter_realization: dict[str, dict[str, float | None]],
    capture_audit: bool = False,
    fragment_list_data: FragmentListData | None = None,
) -> FragmentSimulation:
    if (
        fragment_list_data is None
        and case["warhead"]["fragment_source"].get("fragment_list_ref") is not None
    ):
        fragment_list_data = load_fragment_list(case["warhead"]["fragment_source"])
    fragments = (
        generate_fragments(
            case,
            registry,
            rng,
            fragment_list_data=fragment_list_data,
        )
        if fragment_list_data is not None
        else generate_fragments(case, registry, rng)
    )
    air_density = float(case["environment"]["air_density_kg_m3"])
    if not math.isfinite(air_density) or air_density <= 0:
        raise InputValidationError("air density must be finite and positive")
    component_survival: dict[str, dict[str, float]] = {}
    component_metrics: dict[str, dict[str, float]] = {}
    for component in target.components:
        states = [state for state in component.source["damage_states"] if state != "intact"]
        component_survival[component.object_id] = {state: 1.0 for state in states}
        component_metrics[component.object_id] = {
            "hits": 0.0,
            "penetrations": 0.0,
            "mean_arrival_speed_mps": 0.0,
            "mean_penetration_margin": 0.0,
            "deposited_energy_j": 0.0,
        }
    armor_holes = {item.object_id: 0.0 for item in target.armor}
    armor_hits = armor_penetrations = component_hits = component_penetrations = 0
    audit: list[dict] = []
    for fragment in fragments:
        speed = fragment.initial_speed_mps
        previous_distance = 0.0
        hits = target.ray_hits(fragment.origin_m, fragment.direction)
        for hit in hits:
            speed = _arrival_speed(
                speed,
                hit.entry_t - previous_distance,
                fragment,
                air_density,
            )
            if speed <= EPS:
                break
            model_id = hit.response.get("fragment_response_model_id")
            if not model_id:
                raise InputValidationError(
                    "fragment response model is required",
                    context={"object_id": hit.object_id},
                )
            model = registry.fragment_response(model_id)
            parameters = parameter_realization[hit.object_id]
            thickness = float(parameters["equivalent_thickness_m"])
            probability, v50, margin, angle = _ballistic_response(
                speed,
                fragment,
                hit.entry_normal,
                fragment.direction,
                thickness,
                model,
            )
            penetrated = bool(rng.random() < probability)
            before_speed = speed
            if hit.role == "armor":
                armor_hits += 1
                if penetrated:
                    armor_penetrations += 1
                    speed = _residual_speed(speed, v50)
                    equivalent_diameter = math.sqrt(4.0 * fragment.area_m2 / math.pi)
                    hole_multiplier = float(model.get("hole_area_multiplier", 1.0))
                    armor_holes[hit.object_id] += hole_multiplier * math.pi * equivalent_diameter**2 / 4.0
                if capture_audit:
                    audit.append(
                        {
                            "fragment_id": fragment.fragment_id,
                            "source_fragment_id": fragment.source_fragment_id,
                            "source_multiplicity_index": fragment.source_multiplicity_index,
                            "source_multiplicity": fragment.source_multiplicity,
                            "object_id": hit.object_id,
                            "role": hit.role,
                            "distance_m": hit.entry_t,
                            "arrival_speed_mps": before_speed,
                            "impact_angle_rad": angle,
                            "penetration_probability": probability,
                            "penetrated": penetrated,
                            "residual_speed_mps": speed if penetrated else 0.0,
                        }
                    )
                if not penetrated:
                    break
                previous_distance = max(previous_distance, hit.exit_t)
                continue

            component_hits += 1
            metrics = component_metrics[hit.object_id]
            metrics["hits"] += 1.0
            metrics["mean_arrival_speed_mps"] += before_speed
            metrics["mean_penetration_margin"] += margin
            if penetrated:
                component_penetrations += 1
                metrics["penetrations"] += 1.0
                residual = _residual_speed(speed, v50)
                deposited = 0.5 * fragment.mass_kg * max(0.0, speed**2 - residual**2)
                metrics["deposited_energy_j"] += deposited
                vulnerability = float(parameters["vulnerable_area_ratio"])
                for state, state_model in model.get("damage_states", {}).items():
                    if state not in component_survival[hit.object_id]:
                        continue
                    energy_50 = float(state_model["residual_energy_50_j"])
                    slope = float(state_model["log_slope"])
                    if residual <= 0 or energy_50 <= 0:
                        single_probability = 0.0
                    else:
                        residual_energy = 0.5 * fragment.mass_kg * residual**2
                        single_probability = sigmoid(slope * math.log(residual_energy / energy_50))
                    single_probability *= vulnerability
                    component_survival[hit.object_id][state] *= 1.0 - single_probability
            if capture_audit:
                audit.append(
                    {
                        "fragment_id": fragment.fragment_id,
                        "source_fragment_id": fragment.source_fragment_id,
                        "source_multiplicity_index": fragment.source_multiplicity_index,
                        "source_multiplicity": fragment.source_multiplicity,
                        "object_id": hit.object_id,
                        "role": hit.role,
                        "distance_m": hit.entry_t,
                        "arrival_speed_mps": before_speed,
                        "impact_angle_rad": angle,
                        "penetration_probability": probability,
                        "penetrated": penetrated,
                        "residual_speed_mps": residual if penetrated else 0.0,
                    }
                )
            if not penetrated or residual <= EPS:
                break
            speed = residual
            previous_distance = max(previous_distance, hit.exit_t)

    probabilities: dict[str, dict[str, float]] = {}
    for component in target.components:
        identifier = component.object_id
        raw = {state: 1.0 - survival for state, survival in component_survival[identifier].items()}
        ordered = [state for state in component.source["damage_states"] if state != "intact"]
        previous = 1.0
        probabilities[identifier] = {}
        for state in ordered:
            value = min(previous, float(np.clip(raw.get(state, 0.0), 0.0, 1.0)))
            probabilities[identifier][state] = value
            previous = value
        metrics = component_metrics[identifier]
        if metrics["hits"] > 0:
            metrics["mean_arrival_speed_mps"] /= metrics["hits"]
            metrics["mean_penetration_margin"] /= metrics["hits"]
    armor_integrity = {}
    for item in target.armor:
        perforation_fraction = min(1.0, armor_holes[item.object_id] / _armor_area(item))
        armor_integrity[item.object_id] = 1.0 - perforation_fraction
    return FragmentSimulation(
        component_state_probabilities=probabilities,
        component_metrics=component_metrics,
        armor_integrity=armor_integrity,
        total_fragments=len(fragments),
        total_armor_hits=armor_hits,
        total_armor_penetrations=armor_penetrations,
        total_component_hits=component_hits,
        total_component_penetrations=component_penetrations,
        audit=audit,
        fragment_list_provenance=(
            fragment_list_data.provenance if fragment_list_data is not None else None
        ),
    )
