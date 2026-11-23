from __future__ import annotations

import csv
import heapq
import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .component_damage import ComponentDamageAccumulator
from .errors import InputValidationError, OutOfDomainError
from .geometry import CompiledObject, CompiledTarget, polygon_area_3d
from .layered_penetration import (
    resolve_existing_hole_passage,
    resolve_blast_failed_panel_passage,
    resolve_penetration_path,
)
from .panel_damage import PanelDamageState, perforation_opening_area_m2
from .penetration import FragmentImpact
from .registry import PhysicsRegistry
from .utils import EPS, normalize, quaternion_matrix_xyzw


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


TAYLOR_SHAPIRO_LEGACY_ANGLE_CONVENTION = (
    "body_plus_x_polar_from_nose_local_normal_signed"
)
"""Legacy sign: deflects toward the initiation end (contrary to Taylor 1941).

Kept only so registries that declare it reproduce archived datasets."""

TAYLOR_SHAPIRO_WAVE_DIRECTED_ANGLE_CONVENTION = (
    "body_plus_x_polar_deflection_toward_detonation_propagation"
)
"""Taylor-consistent sign: deflects toward the detonation-wave travel direction."""

TAYLOR_SHAPIRO_ANGLE_CONVENTIONS = (
    TAYLOR_SHAPIRO_LEGACY_ANGLE_CONVENTION,
    TAYLOR_SHAPIRO_WAVE_DIRECTED_ANGLE_CONVENTION,
)


@dataclass(frozen=True)
class FragmentRingLayout:
    """One authoritative casing-ring layout shared by origins and directions."""

    source_offsets_body_m: np.ndarray
    local_normals_body: np.ndarray
    wave_directions_body: np.ndarray
    body_axis: np.ndarray
    ring_indices: np.ndarray
    azimuth_rad: np.ndarray
    ring_sizes: tuple[int, ...]
    angle_convention: str = TAYLOR_SHAPIRO_LEGACY_ANGLE_CONVENTION


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
    total_existing_hole_reuses: int
    total_out_of_domain_impacts: int
    unresolved_component_hit_affected: dict[str, bool] = field(default_factory=dict)
    unresolved_component_state_affected: dict[str, bool] = field(default_factory=dict)
    out_of_domain_impacts_by_role: dict[str, int] = field(default_factory=dict)
    panel_damage_state: PanelDamageState | None = None
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


def _taylor_ratio_and_bound(
    fragment_speed_mps: float,
    detonation_velocity_mps: float,
    phi1_rad: float,
    phi2_rad: float,
) -> tuple[float, float, float, float]:
    values = (
        float(fragment_speed_mps),
        float(detonation_velocity_mps),
        float(phi1_rad),
        float(phi2_rad),
    )
    if not all(math.isfinite(value) for value in values):
        raise InputValidationError("Taylor-Shapiro inputs must be finite")
    speed, detonation_velocity, phi1, phi2 = values
    if speed <= 0.0 or detonation_velocity <= 0.0:
        raise InputValidationError(
            "Taylor-Shapiro fragment and detonation velocities must be positive"
        )
    ratio = speed / (2.0 * detonation_velocity)
    if ratio > 1.0 + 1.0e-12:
        raise OutOfDomainError(
            "Taylor-Shapiro velocity ratio is outside the Taylor domain",
            context={
                "fragment_speed_mps": speed,
                "detonation_velocity_mps": detonation_velocity,
                "velocity_ratio": ratio,
                "required_maximum": 1.0,
            },
        )
    ratio = min(1.0, ratio)
    return ratio, math.asin(ratio), phi1, phi2


def taylor_shapiro_angles(
    fragment_speed_mps: float,
    detonation_velocity_mps: float,
    phi1_rad: float,
    phi2_rad: float,
) -> tuple[float, float]:
    """Return the Taylor bound and the legacy signed Shapiro angle in radians.

    ``phi1`` is the body-axis polar angle of the local casing normal and
    ``phi2`` is the body-axis polar angle of the detonation-wave direction.
    The legacy convention applies the returned angle from the local normal in
    the direction of increasing polar angle.  With that application the
    fragment leans toward the initiation end, which is the opposite of
    Taylor (1941): fragments lean toward the direction the detonation wave
    travels.  This function is kept bit-identical only so that registries
    declaring ``TAYLOR_SHAPIRO_LEGACY_ANGLE_CONVENTION`` reproduce archived
    datasets.  New registries use :func:`taylor_shapiro_deflection`.
    """

    ratio, taylor_bound, phi1, phi2 = _taylor_ratio_and_bound(
        fragment_speed_mps, detonation_velocity_mps, phi1_rad, phi2_rad
    )
    shapiro = math.atan(ratio * math.cos(math.pi / 2.0 + phi2 - phi1))
    shapiro = float(np.clip(shapiro, -taylor_bound, taylor_bound))
    return taylor_bound, shapiro


def taylor_shapiro_deflection(
    fragment_speed_mps: float,
    detonation_velocity_mps: float,
    phi1_rad: float,
    phi2_rad: float,
) -> tuple[float, float]:
    """Return the Taylor bound and the wave-directed Shapiro deflection.

    The deflection is measured from the local casing normal toward
    increasing body-axis polar angle.  In the meridional plane the unit
    tangent toward increasing polar angle is ``(-sin phi1, cos phi1)`` and
    the wave direction is ``(cos phi2, sin phi2)``, so the wave's tangential
    component along the casing is ``sin(phi2 - phi1)``.  Taylor's steady-flow
    construction turns the casing material toward that component, giving

        tan(delta) = V / (2 D) * sin(phi2 - phi1),   |delta| <= asin(V / 2D).

    For a nose initiator the wave travels aft, so fragments lean aft.
    """

    ratio, taylor_bound, phi1, phi2 = _taylor_ratio_and_bound(
        fragment_speed_mps, detonation_velocity_mps, phi1_rad, phi2_rad
    )
    deflection = math.atan(ratio * math.sin(phi2 - phi1))
    deflection = float(np.clip(deflection, -taylor_bound, taylor_bound))
    return taylor_bound, deflection


def build_taylor_shapiro_ring_layout(
    case: dict,
    source_model: dict,
    count: int,
) -> FragmentRingLayout:
    """Build cylindrical inner-wall ring geometry relative to the initiator.

    The v2 model intentionally supports one declared geometry: a front-axis
    point initiator and an axisymmetric cylindrical inner casing wall.  New
    geometries require a new model version rather than implicit fallbacks.
    """

    direction = source_model.get("direction", {})
    if direction.get("kind") != "taylor_shapiro_ring":
        raise InputValidationError(
            "Taylor-Shapiro ring layout requires taylor_shapiro_ring direction"
        )
    munition = case.get("munition")
    if not isinstance(munition, dict):
        raise InputValidationError(
            "taylor_shapiro_ring requires versioned munition geometry"
        )
    if direction.get("casing_surface") != "cylindrical_inner_wall":
        raise InputValidationError(
            "taylor_shapiro_ring supports only cylindrical_inner_wall geometry"
        )
    if direction.get("azimuth_layout") != "uniform_per_ring":
        raise InputValidationError(
            "taylor_shapiro_ring requires uniform_per_ring azimuth layout"
        )
    angle_convention = direction.get("angle_convention")
    if angle_convention not in TAYLOR_SHAPIRO_ANGLE_CONVENTIONS:
        raise InputValidationError(
            "taylor_shapiro_ring angle convention is missing or unsupported",
            context={
                "angle_convention": angle_convention,
                "supported": list(TAYLOR_SHAPIRO_ANGLE_CONVENTIONS),
            },
        )
    if direction.get("dispersion_model") != "tangent_plane_gaussian":
        raise InputValidationError(
            "taylor_shapiro_ring requires tangent_plane_gaussian dispersion"
        )
    if direction.get("taylor_limit_policy") != "reject_ratio_gt_one":
        raise InputValidationError(
            "taylor_shapiro_ring requires reject_ratio_gt_one Taylor policy"
        )

    axis = normalize(
        np.asarray(direction.get("axis_body"), dtype=np.float64),
        name="Taylor-Shapiro body axis",
    )
    if not np.allclose(axis, np.array([1.0, 0.0, 0.0]), atol=1.0e-12, rtol=0.0):
        raise InputValidationError(
            "taylor_shapiro_ring v2 requires body +X as the nose polar axis"
        )
    initiation = np.asarray(
        direction.get("initiation_point_body_m"), dtype=np.float64
    )
    if initiation.shape != (3,) or not np.isfinite(initiation).all():
        raise InputValidationError(
            "taylor_shapiro_ring initiation_point_body_m must be a finite 3-vector"
        )
    radial_initiation = initiation - float(np.dot(initiation, axis)) * axis
    if float(np.linalg.norm(radial_initiation)) > 1.0e-12:
        raise InputValidationError(
            "taylor_shapiro_ring v2 requires an on-axis point initiator"
        )

    try:
        ring_count = int(munition["fragment_ring_count"])
        ring_sizes = tuple(int(value) for value in munition["fragments_per_ring"])
        warhead_length = float(munition["warhead_length_m"])
        outer_radius = float(munition["warhead_diameter_m"]) / 2.0
        wall_thickness = float(munition["wall_thickness_m"])
        coverage = float(munition["fragment_axial_coverage_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InputValidationError(
            "taylor_shapiro_ring munition geometry is incomplete"
        ) from exc
    if (
        ring_count < 1
        or len(ring_sizes) != ring_count
        or any(size < 1 for size in ring_sizes)
        or sum(ring_sizes) != int(count)
    ):
        raise InputValidationError(
            "taylor_shapiro_ring fragment ring layout must match total_count",
            context={
                "ring_count": ring_count,
                "ring_sizes": list(ring_sizes),
                "fragment_count": int(count),
            },
        )
    configured_ring_count = direction.get("ring_count")
    if configured_ring_count is None or int(configured_ring_count) != ring_count:
        raise InputValidationError(
            "taylor_shapiro_ring source and munition ring_count must match",
            context={
                "source_ring_count": configured_ring_count,
                "munition_ring_count": ring_count,
            },
        )
    configured_coverage = direction.get("axial_coverage_ratio")
    if (
        configured_coverage is None
        or not math.isclose(
            float(configured_coverage), coverage, rel_tol=0.0, abs_tol=1.0e-12
        )
    ):
        raise InputValidationError(
            "taylor_shapiro_ring source and munition axial coverage must match",
            context={
                "source_axial_coverage_ratio": configured_coverage,
                "munition_axial_coverage_ratio": coverage,
            },
        )
    inner_radius = outer_radius - wall_thickness
    axial_span = warhead_length * coverage
    if (
        not all(
            math.isfinite(value)
            for value in (
                warhead_length,
                outer_radius,
                wall_thickness,
                coverage,
                inner_radius,
                axial_span,
            )
        )
        or warhead_length <= 0.0
        or wall_thickness <= 0.0
        or inner_radius <= 0.0
        or not 0.0 < coverage <= 1.0
        or axial_span <= 0.0
    ):
        raise InputValidationError(
            "taylor_shapiro_ring cylindrical geometry must be finite and positive"
        )

    basis1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    basis2 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    offsets: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    waves: list[np.ndarray] = []
    ring_indices: list[int] = []
    azimuths: list[float] = []
    for ring_index, ring_size in enumerate(ring_sizes):
        axial_offset = axial_span * (ring_index + 0.5) / ring_count
        for fragment_index in range(ring_size):
            azimuth = 2.0 * math.pi * fragment_index / ring_size
            radial = math.cos(azimuth) * basis1 + math.sin(azimuth) * basis2
            source_offset = -axial_offset * axis + inner_radius * radial
            offsets.append(source_offset)
            normals.append(radial)
            waves.append(normalize(source_offset, name="Taylor-Shapiro wave direction"))
            ring_indices.append(ring_index)
            azimuths.append(azimuth)
    return FragmentRingLayout(
        source_offsets_body_m=np.asarray(offsets, dtype=np.float64),
        local_normals_body=np.asarray(normals, dtype=np.float64),
        wave_directions_body=np.asarray(waves, dtype=np.float64),
        body_axis=axis,
        ring_indices=np.asarray(ring_indices, dtype=np.int64),
        azimuth_rad=np.asarray(azimuths, dtype=np.float64),
        ring_sizes=ring_sizes,
        angle_convention=str(angle_convention),
    )


def _sample_taylor_shapiro_directions(
    layout: FragmentRingLayout,
    rotation: np.ndarray,
    rng: np.random.Generator,
    *,
    gurney_speed: float,
    detonation_velocity_mps: float | None,
    spread_sigma_deg: float,
    spread_distribution: str = "normal",
) -> np.ndarray:
    if detonation_velocity_mps is None:
        raise InputValidationError(
            "taylor_shapiro_ring requires detonation_velocity_mps"
        )
    if layout.angle_convention == TAYLOR_SHAPIRO_WAVE_DIRECTED_ANGLE_CONVENTION:
        deflection_function = taylor_shapiro_deflection
    elif layout.angle_convention == TAYLOR_SHAPIRO_LEGACY_ANGLE_CONVENTION:
        deflection_function = taylor_shapiro_angles
    else:
        raise InputValidationError(
            "Taylor-Shapiro layout angle convention is unsupported",
            context={"angle_convention": layout.angle_convention},
        )
    axis = layout.body_axis
    normals = layout.local_normals_body
    waves = layout.wave_directions_body
    count = len(normals)
    nominal = np.empty_like(normals)
    for index in range(count):
        normal = normals[index]
        wave = waves[index]
        phi1 = math.acos(float(np.clip(np.dot(axis, normal), -1.0, 1.0)))
        phi2 = math.acos(float(np.clip(np.dot(axis, wave), -1.0, 1.0)))
        _, shapiro = deflection_function(
            gurney_speed,
            float(detonation_velocity_mps),
            phi1,
            phi2,
        )
        radial = normal - float(np.dot(axis, normal)) * axis
        radial = normalize(radial, name="Taylor-Shapiro radial direction")
        polar_increase = -math.sin(phi1) * axis + math.cos(phi1) * radial
        nominal[index] = normalize(
            math.cos(shapiro) * normal + math.sin(shapiro) * polar_increase,
            name="Taylor-Shapiro nominal fragment direction",
        )

    sigma = math.radians(float(spread_sigma_deg))
    if not math.isfinite(sigma) or sigma < 0.0:
        raise InputValidationError(
            "taylor_shapiro_ring spread_sigma_deg must be finite and non-negative"
        )
    values = nominal.copy()
    if sigma > 0.0:
        noise = _sample_direction_noise(
            rng,
            sigma,
            size=(count, 2),
            distribution=spread_distribution,
        )
        for index in range(count):
            azimuth_tangent = normalize(
                np.cross(axis, normals[index]),
                name="Taylor-Shapiro azimuth tangent",
            )
            meridional_tangent = normalize(
                np.cross(azimuth_tangent, nominal[index]),
                name="Taylor-Shapiro meridional tangent",
            )
            tangent_delta = (
                noise[index, 0] * meridional_tangent
                + noise[index, 1] * azimuth_tangent
            )
            angular_radius = float(np.linalg.norm(tangent_delta))
            if angular_radius > EPS:
                values[index] = (
                    math.cos(angular_radius) * nominal[index]
                    + math.sin(angular_radius) * tangent_delta / angular_radius
                )
    values = (rotation @ values.T).T
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def _sample_direction_noise(
    rng: np.random.Generator,
    sigma_rad: float,
    *,
    size: int | tuple[int, ...] | None = None,
    distribution: str = "normal",
) -> np.ndarray | float:
    """Sample zero-mean angular noise with the declared standard deviation."""

    sigma = float(sigma_rad)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise InputValidationError(
            "fragment direction spread sigma must be finite and non-negative"
        )
    name = str(distribution)
    if name == "normal":
        return rng.normal(0.0, sigma, size=size)
    if name == "uniform":
        half_width = math.sqrt(3.0) * sigma
        return rng.uniform(-half_width, half_width, size=size)
    if name == "laplace":
        return rng.laplace(0.0, sigma / math.sqrt(2.0), size=size)
    raise InputValidationError(
        "unsupported fragment direction spread distribution",
        context={
            "spread_distribution": name,
            "allowed": ["normal", "uniform", "laplace"],
        },
    )


def _sample_directions(
    count: int,
    source_model: dict,
    rotation: np.ndarray,
    rng: np.random.Generator,
    *,
    gurney_speed: float,
    detonation_velocity_mps: float | None,
    impact_inward_axis_target: np.ndarray | None = None,
    taylor_shapiro_layout: FragmentRingLayout | None = None,
) -> np.ndarray:
    direction = source_model["direction"]
    kind = direction["kind"]
    if kind == "taylor_shapiro_ring":
        if taylor_shapiro_layout is None:
            raise InputValidationError(
                "taylor_shapiro_ring requires a shared fragment ring layout"
            )
        if len(taylor_shapiro_layout.local_normals_body) != count:
            raise InputValidationError(
                "Taylor-Shapiro layout size must match fragment count"
            )
        return _sample_taylor_shapiro_directions(
            taylor_shapiro_layout,
            rotation,
            rng,
            gurney_speed=gurney_speed,
            detonation_velocity_mps=detonation_velocity_mps,
            spread_sigma_deg=float(direction.get("spread_sigma_deg", 0.0)),
            spread_distribution=str(
                direction.get("spread_distribution", "normal")
            ),
        )
    if impact_inward_axis_target is None:
        axis = normalize(
            np.asarray(direction.get("axis_body", [1.0, 0.0, 0.0])),
            name="fragment axis",
        )
    else:
        axis = normalize(impact_inward_axis_target, name="impact inward fragment axis")
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
        spread_distribution = str(
            direction.get("spread_distribution", "normal")
        )
        ring_count = direction.get("ring_count")
        if ring_count is None:
            if spread_distribution == "normal":
                # Preserve the legacy random stream and bit pattern when the
                # new distribution field is absent or explicitly normal.
                theta = np.clip(rng.normal(base, sigma, count), 0.0, math.pi)
            else:
                theta = np.clip(
                    base
                    + _sample_direction_noise(
                        rng,
                        sigma,
                        size=count,
                        distribution=spread_distribution,
                    ),
                    0.0,
                    math.pi,
                )
        else:
            ring_count = int(ring_count)
            if ring_count < 1 or ring_count > count:
                raise InputValidationError(
                    "taylor_ring ring_count must lie between one and total_count"
                )
            correction = math.radians(
                float(direction.get("axial_angle_correction_deg", 0.0))
            )
            base_per_ring, remainder = divmod(count, ring_count)
            ring_counts = [base_per_ring] * ring_count
            ring_counts[-1] += remainder
            theta_values: list[float] = []
            azimuth_values: list[float] = []
            for ring_index, ring_size in enumerate(ring_counts):
                normalized_axial = (
                    ring_index / max(1, ring_count - 1)
                    if ring_count > 1
                    else 0.5
                )
                ring_theta = base + correction * normalized_axial
                for fragment_index in range(ring_size):
                    theta_values.append(
                        ring_theta
                        + float(
                            _sample_direction_noise(
                                rng,
                                sigma,
                                distribution=spread_distribution,
                            )
                        )
                    )
                    azimuth_values.append(
                        2.0 * math.pi * fragment_index / ring_size
                        + float(
                            _sample_direction_noise(
                                rng,
                                sigma,
                                distribution=spread_distribution,
                            )
                        )
                    )
            theta = np.clip(np.asarray(theta_values), 0.0, math.pi)
            azimuth = np.asarray(azimuth_values)
    else:
        raise InputValidationError(f"unsupported direction model: {kind}")
    values = (
        np.cos(theta)[:, None] * axis
        + (np.sin(theta) * np.cos(azimuth))[:, None] * basis1
        + (np.sin(theta) * np.sin(azimuth))[:, None] * basis2
    )
    if impact_inward_axis_target is None:
        values = (rotation @ values.T).T
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def _fragment_origins(
    case: dict,
    count: int,
    rotation: np.ndarray,
    *,
    taylor_shapiro_layout: FragmentRingLayout | None = None,
) -> np.ndarray:
    detonation = np.asarray(
        case["encounter"]["detonation_position_m"], dtype=np.float64
    )
    if detonation.shape != (3,) or not np.isfinite(detonation).all():
        raise InputValidationError("detonation_position_m must be a finite 3-vector")
    if taylor_shapiro_layout is not None:
        if len(taylor_shapiro_layout.source_offsets_body_m) != count:
            raise InputValidationError(
                "Taylor-Shapiro layout size must match fragment count"
            )
        return detonation[None, :] + (
            rotation @ taylor_shapiro_layout.source_offsets_body_m.T
        ).T
    munition = case.get("munition")
    if munition is None:
        return np.repeat(detonation[None, :], count, axis=0)

    ring_count = int(munition["fragment_ring_count"])
    ring_sizes = [int(value) for value in munition["fragments_per_ring"]]
    if ring_count < 1 or len(ring_sizes) != ring_count or sum(ring_sizes) != count:
        raise InputValidationError(
            "munition fragment ring layout must match the expanded fragment count",
            context={
                "ring_count": ring_count,
                "ring_sizes": ring_sizes,
                "fragment_count": count,
            },
        )
    warhead_length = float(munition["warhead_length_m"])
    coverage = float(munition["fragment_axial_coverage_ratio"])
    axial_span = warhead_length * coverage
    if (
        not math.isfinite(axial_span)
        or axial_span <= 0.0
        or not 0.0 < coverage <= 1.0
    ):
        raise InputValidationError(
            "munition fragment axial coverage must define a positive finite span"
        )

    collision_normal = case["encounter"].get("collision_normal_target_xyz")
    if collision_normal is not None:
        backward_axis = normalize(
            np.asarray(collision_normal, dtype=np.float64),
            name="fragment source backward axis",
        )
    else:
        backward_axis = -normalize(
            rotation @ np.array([1.0, 0.0, 0.0]),
            name="fragment source backward axis",
        )
    offsets: list[float] = []
    for ring_index, ring_size in enumerate(ring_sizes):
        ring_offset = axial_span * (ring_index + 0.5) / ring_count
        offsets.extend([ring_offset] * ring_size)
    return detonation[None, :] + np.asarray(offsets)[:, None] * backward_axis


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
    taylor_shapiro_layout: FragmentRingLayout | None = None
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
        if source_model["direction"]["kind"] == "taylor_shapiro_ring":
            taylor_shapiro_layout = build_taylor_shapiro_ring_layout(
                case,
                source_model,
                len(masses),
            )
        axis_mode = str(source_model["direction"].get("axis_mode", "body_vector"))
        impact_inward_axis_target = None
        if axis_mode == "impact_inward":
            collision_normal = case["encounter"].get("collision_normal_target_xyz")
            if collision_normal is None:
                raise InputValidationError(
                    "impact_inward fragment direction requires collision_normal_target_xyz"
                )
            impact_inward_axis_target = -normalize(
                np.asarray(collision_normal, dtype=np.float64),
                name="collision normal",
            )
        directions = _sample_directions(
            len(masses),
            source_model,
            rotation,
            rng,
            gurney_speed=gurney_speed,
            detonation_velocity_mps=source.get("detonation_velocity_mps"),
            impact_inward_axis_target=impact_inward_axis_target,
            taylor_shapiro_layout=taylor_shapiro_layout,
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
    origins = _fragment_origins(
        case,
        len(masses),
        rotation,
        taylor_shapiro_layout=taylor_shapiro_layout,
    )
    return [
        Fragment(
            fragment_id=index,
            mass_kg=float(mass),
            origin_m=origins[index].copy(),
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


def propagate_fragment_speed(
    speed: float,
    distance_m: float,
    fragment: Fragment,
    air_density: float,
) -> float:
    """Propagate fragment speed through still air with the registered drag law."""

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


# Compatibility alias for downstream callers written before the propagation
# kernel became part of the public sampling-proxy seam.
_arrival_speed = propagate_fragment_speed


def _free_flight_time(
    speed: float,
    distance_m: float,
    fragment: Fragment,
    air_density: float,
) -> float:
    if not math.isfinite(speed) or speed <= 0.0 or not math.isfinite(distance_m):
        raise InputValidationError(
            "fragment speed must be positive and flight distance finite"
        )
    distance = max(0.0, float(distance_m))
    coefficient = (
        air_density
        * fragment.drag_coefficient
        * fragment.area_m2
        / (2.0 * fragment.mass_kg)
    )
    if coefficient <= EPS:
        return distance / speed
    exponent = coefficient * distance
    if exponent > 700.0:
        raise OutOfDomainError(
            "fragment free-flight time overflowed the drag model domain",
            context={"drag_exponent": exponent, "distance_m": distance},
        )
    return float(math.expm1(exponent) / (coefficient * speed))


def armor_reference_area_m2(item: CompiledObject) -> float:
    """Return the physical shell area used by the panel-integrity state.

    A thin-shell collision envelope is an extruded numerical volume.  Its back
    face and narrow closure edges do not represent additional load-bearing
    panel area, so the reference area is the generating face.  Solid-envelope
    semantics retain their complete exterior surface area.
    """

    geometry = item.source["geometry"]
    semantics = item.source.get(
        "geometry_semantics",
        "thin_shell_collision_envelope" if item.role == "armor" else None,
    )
    if geometry["shape"] == "extruded_polygon":
        vertices = np.asarray(geometry["vertices_m"], dtype=np.float64)
        base_area = polygon_area_3d(
            vertices,
            np.asarray(geometry["extrusion_direction"], dtype=np.float64),
        )
        if semantics == "thin_shell_collision_envelope":
            return max(float(base_area), EPS)
        perimeter = float(
            np.sum(
                np.linalg.norm(
                    np.roll(vertices, -1, axis=0) - vertices,
                    axis=1,
                )
            )
        )
        distance = float(geometry["extrusion_distance_m"])
        return max(2.0 * base_area + perimeter * distance, EPS)
    if geometry["shape"] == "box":
        length, width, height = map(float, geometry["size_lwh_m"])
        return max(
            2.0 * (length * width + length * height + width * height),
            EPS,
        )
    if geometry["shape"] == "cylinder":
        radius = float(geometry["radius_m"])
        height = float(geometry["height_m"])
        return max(2.0 * math.pi * radius * (radius + height), EPS)
    size = np.sort(item.geometry.representative_size())
    return max(float(size[-1] * size[-2]), EPS)


_armor_area = armor_reference_area_m2


def _json_finite(value: float) -> float | None:
    """Keep opt-in audit payloads compatible with strict JSON writers."""

    number = float(value)
    return number if math.isfinite(number) else None


def _geometry_path_record(fragment: Fragment, hit) -> dict[str, Any]:
    """Return geometry-only information for one exact ray intersection."""

    exit_point = fragment.origin_m + float(hit.exit_t) * fragment.direction
    return {
        "object_id": hit.object_id,
        "role": hit.role,
        "entry_distance_m": float(hit.entry_t),
        "exit_distance_m": float(hit.exit_t),
        "path_length_m": float(hit.path_length_m),
        "geometry_path_length_m": float(hit.path_length_m),
        "entry_point_target_m": np.asarray(hit.entry_point, dtype=np.float64).tolist(),
        "exit_point_target_m": np.asarray(exit_point, dtype=np.float64).tolist(),
        "entry_normal_target_xyz": np.asarray(hit.entry_normal, dtype=np.float64).tolist(),
        "fragment_response_model_id": hit.response.get("fragment_response_model_id"),
    }


@dataclass
class _FragmentTransit:
    fragment: Fragment
    hits: list[Any]
    physical_key: str
    geometry_path: list[dict[str, Any]]
    path_index: int = 0
    speed_mps: float = 0.0
    previous_distance_m: float = 0.0
    elapsed_time_s: float = 0.0
    processed_path: list[dict[str, Any]] = field(default_factory=list)
    blocking_object_id: str | None = None
    blocked_before_object_id: str | None = None
    blocking_reason: str | None = None


@dataclass(order=True, frozen=True)
class _ScheduledFragmentImpact:
    arrival_time_s: float
    physical_key: str
    path_index: int
    cursor_index: int
    free_flight_distance_m: float = field(compare=False)
    arrival_speed_mps: float = field(compare=False)


def _fragment_physical_key(fragment: Fragment) -> str:
    payload = {
        "source_fragment_id": fragment.source_fragment_id,
        "source_multiplicity_index": fragment.source_multiplicity_index,
        "source_multiplicity": fragment.source_multiplicity,
        "mass_kg": float(fragment.mass_kg),
        "origin_m": [float(value) for value in fragment.origin_m],
        "direction": [float(value) for value in fragment.direction],
        "initial_speed_mps": float(fragment.initial_speed_mps),
        "area_m2": float(fragment.area_m2),
        "drag_coefficient": float(fragment.drag_coefficient),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fragment_physical_keys(fragments: list[Fragment]) -> list[str]:
    """Assign unique stable keys to the members of a physical multiset.

    Exact duplicates are exchangeable. Their deterministic ordinal is scoped
    to the shared physical signature, so adding or reordering a fragment with
    another signature does not reassign existing event streams.
    """

    signatures = [_fragment_physical_key(fragment) for fragment in fragments]
    groups: dict[str, list[tuple[int, int]]] = {}
    for index, (fragment, signature) in enumerate(zip(fragments, signatures)):
        groups.setdefault(signature, []).append((int(fragment.fragment_id), index))
    ordinals: dict[int, int] = {}
    for members in groups.values():
        for ordinal, (_, index) in enumerate(sorted(members)):
            ordinals[index] = ordinal
    keys: list[str] = []
    for index, signature in enumerate(signatures):
        encoded = json.dumps(
            {
                "physical_signature": signature,
                "duplicate_ordinal": ordinals[index],
                "rng_contract": "fragment-physical-key-v2",
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        keys.append(hashlib.sha256(encoded).hexdigest())
    return keys


def _event_random_generator(
    timeline_entropy: tuple[int, ...],
    *,
    fragment_physical_key: str,
    object_id: str,
    path_index: int,
    event_kind: str,
) -> tuple[np.random.Generator, str]:
    identity = {
        "fragment_physical_key": fragment_physical_key,
        "object_id": str(object_id),
        "path_index": int(path_index),
        "event_kind": str(event_kind),
        "rng_contract": "fragment-event-key-v2",
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    identity_words = np.frombuffer(digest[:16], dtype="<u4").astype(np.uint32)
    seed = np.random.SeedSequence(
        [*timeline_entropy, *(int(value) for value in identity_words)]
    )
    return np.random.default_rng(seed), hashlib.sha256(encoded).hexdigest()


def _schedule_next_fragment_impact(
    queue: list[_ScheduledFragmentImpact],
    cursor_index: int,
    cursor: _FragmentTransit,
    air_density: float,
) -> None:
    if cursor.path_index >= len(cursor.hits):
        return
    hit = cursor.hits[cursor.path_index]
    free_flight_distance = max(
        0.0,
        float(hit.entry_t) - cursor.previous_distance_m,
    )
    travel_time = _free_flight_time(
        cursor.speed_mps,
        free_flight_distance,
        cursor.fragment,
        air_density,
    )
    arrival_speed = _arrival_speed(
        cursor.speed_mps,
        free_flight_distance,
        cursor.fragment,
        air_density,
    )
    if arrival_speed <= EPS:
        cursor.blocked_before_object_id = hit.object_id
        cursor.blocking_reason = "arrival_speed_exhausted_before_object"
        return
    heapq.heappush(
        queue,
        _ScheduledFragmentImpact(
            arrival_time_s=cursor.elapsed_time_s + travel_time,
            physical_key=cursor.physical_key,
            path_index=cursor.path_index,
            cursor_index=cursor_index,
            free_flight_distance_m=free_flight_distance,
            arrival_speed_mps=arrival_speed,
        ),
    )


def simulate_fragments(
    case: dict,
    target: CompiledTarget,
    registry: PhysicsRegistry,
    rng: np.random.Generator,
    *,
    parameter_realization: dict[str, dict[str, float | None]],
    capture_audit: bool = False,
    fragment_list_data: FragmentListData | None = None,
    armor_blast_damage_events: dict[str, dict[str, float]] | None = None,
) -> FragmentSimulation:
    timeline_entropy = tuple(
        int(value)
        for value in rng.integers(0, 2**32, size=4, dtype=np.uint32)
    )
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
    blast_damage_events = armor_blast_damage_events or {}
    known_armor_ids = {item.object_id for item in target.armor}
    unknown_event_ids = sorted(set(blast_damage_events) - known_armor_ids)
    if unknown_event_ids:
        raise InputValidationError(
            "blast damage timeline references unknown armor",
            context={"unknown_armor_ids": unknown_event_ids},
        )
    for armor_id, event in blast_damage_events.items():
        if set(event) != {"effective_time_s", "survival_factor"}:
            raise InputValidationError(
                "blast damage timeline event fields are invalid",
                context={"object_id": armor_id, "fields": sorted(event)},
            )
        effective_time = float(event["effective_time_s"])
        survival_factor = float(event["survival_factor"])
        if (
            not math.isfinite(effective_time)
            or effective_time < 0.0
            or not math.isfinite(survival_factor)
            or not 0.0 <= survival_factor <= 1.0
        ):
            raise InputValidationError(
                "blast damage timeline event is outside its numeric domain",
                context={"object_id": armor_id, **event},
            )
    component_accumulators: dict[str, ComponentDamageAccumulator] = {}
    component_metrics: dict[str, dict[str, float]] = {}
    for component in target.components:
        states = tuple(
            state for state in component.source["damage_states"] if state != "intact"
        )
        response_model = registry.fragment_response(
            component.response["fragment_response_model_id"]
        )
        component_accumulators[component.object_id] = ComponentDamageAccumulator(
            ordered_states=states,
            state_models=response_model["damage_states"],
            vulnerable_ratio=float(
                parameter_realization[component.object_id]["vulnerable_area_ratio"]
            ),
            accumulation_model=str(
                response_model.get("damage_accumulation_model", "independent_hits")
            ),
            spatial_correlation_length_m=response_model.get(
                "damage_correlation_length_m"
            ),
        )
        component_metrics[component.object_id] = {
            "hits": 0.0,
            "penetrations": 0.0,
            "mean_arrival_speed_mps": 0.0,
            "mean_penetration_margin": 0.0,
            "deposited_energy_j": 0.0,
            "hit_conditioned_log1p_deposited_energy_j": 0.0,
            "hit_conditioned_log1p_hit_count": 0.0,
            "damage_energy_j": 0.0,
            "ordinal_projection_max_correction": 0.0,
            "spatial_damage_cluster_count": 0.0,
            "spatial_damage_cluster_max_diameter_m": 0.0,
        }
    panel_damage = PanelDamageState(
        {item.object_id: armor_reference_area_m2(item) for item in target.armor}
    )
    armor_hits = armor_penetrations = component_hits = component_penetrations = 0
    existing_hole_reuses = 0
    out_of_domain_impacts = 0
    component_ids = tuple(component.object_id for component in target.components)
    unresolved_component_hit_affected = {
        component_id: False for component_id in component_ids
    }
    unresolved_component_state_affected = {
        component_id: False for component_id in component_ids
    }
    out_of_domain_impacts_by_role = {"armor": 0, "component": 0}
    resolved_penetration_margin_counts = {
        component_id: 0 for component_id in component_ids
    }
    audit: list[dict] = []
    cursors: list[_FragmentTransit] = []
    event_queue: list[_ScheduledFragmentImpact] = []
    fragment_physical_keys = _fragment_physical_keys(fragments)
    for fragment, fragment_physical_key in zip(fragments, fragment_physical_keys):
        hits = target.ray_hits(fragment.origin_m, fragment.direction)
        cursor = _FragmentTransit(
            fragment=fragment,
            hits=hits,
            physical_key=fragment_physical_key,
            geometry_path=(
                [_geometry_path_record(fragment, hit) for hit in hits]
                if capture_audit
                else []
            ),
            speed_mps=float(fragment.initial_speed_mps),
        )
        cursor_index = len(cursors)
        cursors.append(cursor)
        if hits:
            _schedule_next_fragment_impact(
                event_queue,
                cursor_index,
                cursor,
                air_density,
            )

    global_event_index = 0
    while event_queue:
        scheduled = heapq.heappop(event_queue)
        cursor = cursors[scheduled.cursor_index]
        if scheduled.path_index != cursor.path_index:
            raise InputValidationError(
                "fragment event queue path index drifted from transit state",
                context={
                    "scheduled_path_index": scheduled.path_index,
                    "current_path_index": cursor.path_index,
                },
            )
        fragment = cursor.fragment
        path_index = cursor.path_index
        hit = cursor.hits[path_index]
        cursor.elapsed_time_s = float(scheduled.arrival_time_s)
        cursor.speed_mps = float(scheduled.arrival_speed_mps)
        before_speed = cursor.speed_mps
        free_flight_distance = scheduled.free_flight_distance_m

        model_id = hit.response.get("fragment_response_model_id")
        if not model_id:
            raise InputValidationError(
                "fragment response model is required",
                context={"object_id": hit.object_id},
            )
        model = registry.fragment_response(model_id)
        parameters = parameter_realization[hit.object_id]
        nominal_thickness = float(parameters["equivalent_thickness_m"])
        blast_event = blast_damage_events.get(hit.object_id) if hit.role == "armor" else None
        blast_damage_active = bool(
            blast_event is not None
            and cursor.elapsed_time_s >= float(blast_event["effective_time_s"])
        )
        blast_survival_factor = (
            float(blast_event["survival_factor"])
            if blast_damage_active and blast_event is not None
            else 1.0
        )
        thickness = nominal_thickness * blast_survival_factor
        blast_failed_panel_passage = bool(
            hit.role == "armor"
            and blast_damage_active
            and blast_survival_factor <= EPS
        )
        impact = FragmentImpact(
            speed_mps=before_speed,
            mass_kg=fragment.mass_kg,
            direction=fragment.direction,
            surface_normal=hit.entry_normal,
            equivalent_thickness_m=(
                nominal_thickness if blast_failed_panel_passage else thickness
            ),
        )
        equivalent_radius = math.sqrt(fragment.area_m2 / math.pi)
        existing_hole_reused = bool(
            hit.role == "armor"
            and panel_damage.is_open(
                hit.object_id,
                hit.entry_point,
                hit.entry_normal,
                clearance_radius_m=equivalent_radius,
                query_time_s=cursor.elapsed_time_s,
            )
        )
        if existing_hole_reused:
            existing_hole_reuses += 1
        event_rng, event_random_key_hash = _event_random_generator(
            timeline_entropy,
            fragment_physical_key=cursor.physical_key,
            object_id=hit.object_id,
            path_index=path_index,
            event_kind="penetration",
        )
        try:
            impact_outcome = (
                resolve_existing_hole_passage(impact)
                if existing_hole_reused
                else resolve_blast_failed_panel_passage(impact)
                if blast_failed_panel_passage
                else resolve_penetration_path(
                    impact,
                    hit.response,
                    registry,
                    event_rng,
                )
            )
        except OutOfDomainError as error:
            if error.context.get("reason_code") != "obliquity_out_of_domain":
                raise
            # A trajectory outside the registered ballistic-response domain is
            # an unresolved event, not a known stop or known perforation.  End
            # this fragment path without assigning energy or damage, retain a
            # separate count, and preserve the exact refusal in event audit.
            out_of_domain_impacts += 1
            out_of_domain_impacts_by_role[hit.role] += 1
            if hit.role == "armor":
                # An unresolved armor response can change its opening state and
                # therefore any later fragment path.  Until that event graph is
                # narrowed, every component remains conservatively reachable.
                for component_id in component_ids:
                    unresolved_component_hit_affected[component_id] = True
                    unresolved_component_state_affected[component_id] = True
            else:
                # The current component contact is geometrically observed even
                # though its response is unresolved.  Its state is censored;
                # only later components on this same stopped ray have censored
                # hit indicators.
                component_hits += 1
                metrics = component_metrics[hit.object_id]
                metrics["hits"] += 1.0
                metrics["mean_arrival_speed_mps"] += before_speed
                unresolved_component_state_affected[hit.object_id] = True
                for downstream_hit in cursor.hits[path_index + 1 :]:
                    if downstream_hit.role != "armor":
                        downstream_id = downstream_hit.object_id
                        unresolved_component_hit_affected[downstream_id] = True
                        unresolved_component_state_affected[downstream_id] = True
            cursor.blocking_object_id = hit.object_id
            cursor.blocking_reason = "penetration_model_out_of_domain"
            if capture_audit:
                audit.append(
                    {
                        "record_type": "out_of_domain_intersection",
                        "global_event_index": global_event_index,
                        "fragment_physical_key": cursor.physical_key,
                        "event_random_key_hash": event_random_key_hash,
                        "event_random_contract": "fragment-event-key-v2",
                        "fragment_id": fragment.fragment_id,
                        "candidate_path_index": path_index,
                        "object_id": hit.object_id,
                        "role": hit.role,
                        "entry_distance_m": float(hit.entry_t),
                        "entry_point_target_m": np.asarray(
                            hit.entry_point, dtype=np.float64
                        ).tolist(),
                        "entry_normal_target_xyz": np.asarray(
                            hit.entry_normal, dtype=np.float64
                        ).tolist(),
                        "direction_target_xyz": np.asarray(
                            fragment.direction, dtype=np.float64
                        ).tolist(),
                        "arrival_speed_mps": before_speed,
                        "arrival_time_s": cursor.elapsed_time_s,
                        "fragment_response_model_id": model_id,
                        "out_of_domain": error.to_dict(),
                    }
                )
                cursor.processed_path.append(
                    {
                        "candidate_path_index": path_index,
                        "object_id": hit.object_id,
                        "role": hit.role,
                        "disposition": "penetration_model_out_of_domain",
                        "penetrated": None,
                        "residual_speed_mps": None,
                    }
                )
            global_event_index += 1
            continue
        probability = impact_outcome.penetration_probability
        v50 = impact_outcome.v50_mps
        margin = before_speed / max(v50, EPS) if math.isfinite(v50) else 0.0
        angle = impact_outcome.impact_angle_rad
        penetrated = impact_outcome.perforated
        residual = 0.0
        deposited = 0.0
        hole_formation_time_s: float | None = None
        opening_area_m2: float | None = None
        state_response_probabilities: dict[str, float] = {}
        state_response_parameters: dict[str, dict[str, float]] = {}
        if capture_audit:
            state_response_probabilities = {
                state: 0.0
                for state in model.get("damage_states", {})
                if hit.object_id in component_accumulators
                and state in component_accumulators[hit.object_id].ordered_states
            }
            state_response_parameters = {
                state: {
                    "residual_energy_50_j": float(
                        state_model["residual_energy_50_j"]
                    ),
                    "damage_energy_50_j": float(
                        state_model.get(
                            "damage_energy_50_j",
                            state_model["residual_energy_50_j"],
                        )
                    ),
                    "log_slope": float(state_model["log_slope"]),
                }
                for state, state_model in model.get("damage_states", {}).items()
                if hit.object_id in component_accumulators
                and state in component_accumulators[hit.object_id].ordered_states
            }
        if hit.role == "armor":
            armor_hits += 1
            if penetrated:
                armor_penetrations += 1
                residual = impact_outcome.residual_speed_mps
                if not existing_hole_reused and not blast_failed_panel_passage:
                    hole_formation_time_s = (
                        cursor.elapsed_time_s
                        + impact_outcome.material_transit_time_s
                    )
                    hole_multiplier = float(model.get("hole_area_multiplier", 1.0))
                    opening_area_m2 = perforation_opening_area_m2(
                        projectile_area_m2=fragment.area_m2,
                        incidence_cosine=impact_outcome.incidence_cosine,
                        incident_energy_j=impact_outcome.incident_energy_j,
                        residual_energy_j=impact_outcome.residual_energy_j,
                        hole_area_multiplier=hole_multiplier,
                    )
                    panel_damage.record_perforation(
                        hit.object_id,
                        hit.entry_point,
                        hit.entry_normal,
                        hole_area_m2=opening_area_m2,
                        formed_time_s=hole_formation_time_s,
                        update_union_area=False,
                    )
        else:
            component_hits += 1
            metrics = component_metrics[hit.object_id]
            metrics["hits"] += 1.0
            metrics["mean_arrival_speed_mps"] += before_speed
            metrics["mean_penetration_margin"] += margin
            resolved_penetration_margin_counts[hit.object_id] += 1
            deposited = impact_outcome.absorbed_energy_j
            metrics["deposited_energy_j"] += deposited
            metrics["damage_energy_j"] += impact_outcome.damage_energy_j
            if penetrated:
                component_penetrations += 1
                metrics["penetrations"] += 1.0
                residual = impact_outcome.residual_speed_mps
            state_response_probabilities = component_accumulators[
                hit.object_id
            ].record_impact(
                impact_outcome.damage_energy_j,
                location_m=hit.entry_point,
            )

        if capture_audit:
            exit_point = fragment.origin_m + float(hit.exit_t) * fragment.direction
            audit.append(
                {
                    "record_type": "intersection",
                    "global_event_index": global_event_index,
                    "fragment_physical_key": cursor.physical_key,
                    "event_random_key_hash": event_random_key_hash,
                    "event_random_contract": "fragment-event-key-v2",
                    "fragment_id": fragment.fragment_id,
                    "source_fragment_id": fragment.source_fragment_id,
                    "source_multiplicity_index": fragment.source_multiplicity_index,
                    "source_multiplicity": fragment.source_multiplicity,
                    "candidate_path_index": path_index,
                    "object_id": hit.object_id,
                    "role": hit.role,
                    "geometry_semantics": hit.source.get(
                        "geometry_semantics",
                        (
                            "thin_shell_collision_envelope"
                            if hit.role == "armor"
                            else "logical_component_envelope"
                        ),
                    ),
                    "geometry_semantics_source": (
                        "declared"
                        if "geometry_semantics" in hit.source
                        else "legacy_role_inference"
                    ),
                    "distance_m": float(hit.entry_t),
                    "entry_distance_m": float(hit.entry_t),
                    "exit_distance_m": float(hit.exit_t),
                    "path_length_m": float(hit.path_length_m),
                    "geometry_path_length_m": float(hit.path_length_m),
                    "material_path_length_m": impact_outcome.material_path_length_m,
                    "material_transit_time_s": impact_outcome.material_transit_time_s,
                    "material_exit_distance_m": (
                        float(hit.entry_t) + impact_outcome.material_path_length_m
                    ),
                    "free_flight_distance_m": free_flight_distance,
                    "entry_point_target_m": np.asarray(
                        hit.entry_point, dtype=np.float64
                    ).tolist(),
                    "exit_point_target_m": np.asarray(
                        exit_point, dtype=np.float64
                    ).tolist(),
                    "entry_normal_target_xyz": np.asarray(
                        hit.entry_normal, dtype=np.float64
                    ).tolist(),
                    "fragment_response_model_id": model_id,
                    "fragment_response_model_source_id": model.get("source_id"),
                    "response_model": {
                        "id": model_id,
                        "source_id": model.get("source_id"),
                    },
                    "equivalent_thickness_m": thickness,
                    "nominal_equivalent_thickness_m": nominal_thickness,
                    "blast_damage_active": blast_damage_active,
                    "blast_survival_factor": blast_survival_factor,
                    "blast_damage_effective_time_s": (
                        float(blast_event["effective_time_s"])
                        if blast_event is not None
                        else None
                    ),
                    "arrival_speed_mps": before_speed,
                    "arrival_time_s": cursor.elapsed_time_s,
                    "hole_formation_time_s": hole_formation_time_s,
                    "opening_area_m2": opening_area_m2,
                    "opening_area_model": (
                        "projected_geometric_plus_residual_energy_ratio_v1"
                        if opening_area_m2 is not None
                        else None
                    ),
                    "impact_angle_rad": angle,
                    "incidence_cosine": impact_outcome.incidence_cosine,
                    "v50_mps": _json_finite(v50),
                    "penetration_margin": margin,
                    "penetration_probability": probability,
                    "realized_branch_probability": (
                        impact_outcome.realized_branch_probability
                    ),
                    "penetration_probability_semantics": (
                        impact_outcome.penetration_probability_semantics
                    ),
                    "penetrated": penetrated,
                    "perforation_class": impact_outcome.perforation_class,
                    "existing_hole_reused": existing_hole_reused,
                    "layer_count": impact_outcome.layer_count,
                    "declared_layer_count": impact_outcome.declared_layer_count,
                    "resolved_layer_count": impact_outcome.resolved_layer_count,
                    "stopped_layer_index": impact_outcome.stopped_layer_index,
                    "layer_path": [
                        {
                            "layer_index": layer.layer_index,
                            "declared_layer_indices": list(
                                layer.declared_layer_indices
                            ),
                            "material_id": layer.material_id,
                            "fragment_response_model_id": (
                                layer.fragment_response_model_id
                            ),
                            "thickness_fraction": layer.thickness_fraction,
                            "v50_mps": _json_finite(layer.outcome.v50_mps),
                            "penetration_probability": (
                                layer.outcome.penetration_probability
                            ),
                            "sampled_ballistic_limit_mps": _json_finite(
                                layer.outcome.sampled_ballistic_limit_mps
                            ),
                            "perforated": layer.outcome.perforated,
                            "residual_speed_mps": layer.outcome.residual_speed_mps,
                            "absorbed_energy_j": layer.outcome.absorbed_energy_j,
                            "damage_energy_j": layer.outcome.damage_energy_j,
                        }
                        for layer in impact_outcome.layers
                    ],
                    "sampled_ballistic_limit_mps": _json_finite(
                        impact_outcome.sampled_ballistic_limit_mps
                    ),
                    "residual_speed_mps": residual if penetrated else 0.0,
                    "incident_energy_j": impact_outcome.incident_energy_j,
                    "residual_energy_j": impact_outcome.residual_energy_j,
                    "absorbed_energy_j": impact_outcome.absorbed_energy_j,
                    "damage_energy_j": impact_outcome.damage_energy_j,
                    "deposited_energy_j": deposited,
                    "damage_state_response_probabilities": (
                        state_response_probabilities
                    ),
                    "damage_state_response_parameters": state_response_parameters,
                }
            )
            disposition = "penetrated_continue"
            if not penetrated:
                disposition = "not_penetrated_block"
            elif residual <= EPS:
                disposition = "residual_speed_exhausted_block"
            cursor.processed_path.append(
                {
                    "candidate_path_index": path_index,
                    "object_id": hit.object_id,
                    "role": hit.role,
                    "disposition": disposition,
                    "penetrated": penetrated,
                    "residual_speed_mps": residual if penetrated else 0.0,
                }
            )
        global_event_index += 1

        if not penetrated:
            cursor.blocking_object_id = hit.object_id
            cursor.blocking_reason = (
                "armor_not_penetrated"
                if hit.role == "armor"
                else "component_not_penetrated"
            )
            continue
        if residual <= EPS:
            cursor.blocking_object_id = hit.object_id
            cursor.blocking_reason = "residual_speed_exhausted_after_penetration"
            continue

        cursor.speed_mps = residual
        cursor.elapsed_time_s += impact_outcome.material_transit_time_s
        cursor.previous_distance_m = max(
            cursor.previous_distance_m,
            float(hit.entry_t) + impact_outcome.material_path_length_m,
        )
        cursor.path_index += 1
        _schedule_next_fragment_impact(
            event_queue,
            scheduled.cursor_index,
            cursor,
            air_density,
        )

    if capture_audit:
        for cursor in sorted(
            cursors,
            key=lambda item: (item.fragment.fragment_id, item.physical_key),
        ):
            fragment = cursor.fragment
            audit.append(
                {
                    "record_type": "fragment_trace",
                    "fragment_id": fragment.fragment_id,
                    "fragment_physical_key": cursor.physical_key,
                    "source_fragment_id": fragment.source_fragment_id,
                    "source_multiplicity_index": fragment.source_multiplicity_index,
                    "source_multiplicity": fragment.source_multiplicity,
                    "role": "fragment_trace",
                    "origin_target_m": np.asarray(
                        fragment.origin_m, dtype=np.float64
                    ).tolist(),
                    "direction_target_xyz": np.asarray(
                        fragment.direction, dtype=np.float64
                    ).tolist(),
                    "initial_speed_mps": float(fragment.initial_speed_mps),
                    "mass_kg": float(fragment.mass_kg),
                    "geometry_candidate_path": cursor.geometry_path,
                    "processed_path": cursor.processed_path,
                    "blocking_object_id": cursor.blocking_object_id,
                    "blocked_before_object_id": cursor.blocked_before_object_id,
                    "blocking_reason": cursor.blocking_reason,
                    "completed_geometry_path": cursor.blocking_reason is None,
                }
            )

    probabilities: dict[str, dict[str, float]] = {}
    for component in target.components:
        identifier = component.object_id
        damage_result = component_accumulators[identifier].finalize()
        probabilities[identifier] = damage_result.probabilities
        metrics = component_metrics[identifier]
        metrics["damage_energy_j"] = damage_result.total_damage_energy_j
        metrics["ordinal_projection_max_correction"] = (
            damage_result.ordinal_projection_max_correction
        )
        metrics["spatial_damage_cluster_count"] = float(
            damage_result.spatial_cluster_count
        )
        metrics["spatial_damage_cluster_max_diameter_m"] = float(
            damage_result.spatial_cluster_max_diameter_m
        )
        hit_observed = metrics["hits"] > 0.0
        ordered_probabilities = [
            damage_result.probabilities[state]
            for state in component_accumulators[identifier].ordered_states
        ]
        degraded_probability = damage_result.probabilities.get(
            "degraded",
            ordered_probabilities[0] if ordered_probabilities else 0.0,
        )
        failed_probability = damage_result.probabilities.get(
            "failed",
            ordered_probabilities[-1] if ordered_probabilities else 0.0,
        )
        metrics["hit_indicator"] = float(hit_observed)
        metrics["hit_conditioned_log1p_deposited_energy_j"] = (
            math.log1p(metrics["deposited_energy_j"]) if hit_observed else 0.0
        )
        metrics["hit_conditioned_log1p_hit_count"] = (
            math.log1p(metrics["hits"]) if hit_observed else 0.0
        )
        metrics["damage_positive_indicator"] = float(
            hit_observed and degraded_probability > 0.01
        )
        metrics["saturated_damage_indicator"] = float(
            hit_observed
            and (degraded_probability > 0.95 or failed_probability > 0.95)
        )
        metrics["hit_conditioned_degraded_probability"] = (
            float(degraded_probability) if hit_observed else 0.0
        )
        metrics["hit_conditioned_failed_probability"] = (
            float(failed_probability) if hit_observed else 0.0
        )
        if metrics["hits"] > 0:
            metrics["mean_arrival_speed_mps"] /= metrics["hits"]
        resolved_margin_count = resolved_penetration_margin_counts[identifier]
        if resolved_margin_count > 0:
            metrics["mean_penetration_margin"] /= resolved_margin_count
    armor_integrity = panel_damage.integrity_map()
    return FragmentSimulation(
        component_state_probabilities=probabilities,
        component_metrics=component_metrics,
        armor_integrity=armor_integrity,
        total_fragments=len(fragments),
        total_armor_hits=armor_hits,
        total_armor_penetrations=armor_penetrations,
        total_component_hits=component_hits,
        total_component_penetrations=component_penetrations,
        total_existing_hole_reuses=existing_hole_reuses,
        total_out_of_domain_impacts=out_of_domain_impacts,
        unresolved_component_hit_affected=unresolved_component_hit_affected,
        unresolved_component_state_affected=unresolved_component_state_affected,
        out_of_domain_impacts_by_role=out_of_domain_impacts_by_role,
        panel_damage_state=panel_damage,
        audit=audit,
        fragment_list_provenance=(
            fragment_list_data.provenance if fragment_list_data is not None else None
        ),
    )
