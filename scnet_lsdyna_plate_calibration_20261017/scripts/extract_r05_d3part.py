"""Fail-closed extraction of preregistered R05 capture ledgers from D3PART.

Real-file readability remains distinct from separately hash-bound integration,
run-integrity, numerical, and physical acceptance.
"""
from __future__ import annotations

import argparse
import csv
import enum
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
VENDOR_DIR = PROJECT / ".vendor" / "lasso-python-2.0.4"


def _load_array_type() -> type:
    source = VENDOR_DIR / "lasso" / "dyna" / "array_type.py"
    spec = importlib.util.spec_from_file_location("_r05_lasso_array_type", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load vendored ArrayType from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ArrayType


ArrayType = _load_array_type()


class D3plotFiletype(enum.Enum):
    """Reader-independent filetype values used by the pure validation seam."""

    D3PLOT = 1
    D3PART = 5

from r05_capture_ledger import CaptureThresholds, build_capture_ledger  # noqa: E402


EXTRACTOR_NAME = "extract_r05_d3part"
EXTRACTOR_VERSION = "1.1.0"
INTEGRATION_STATUS = "real_d3part_exercised_pending_independent_integration_acceptance"
SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
TEMPORAL_GATE_SCHEMA_VERSION = "hf-p0-d3part-temporal-gate@1.0.0"
CROSSING_SPATIAL_RESOLUTION_MESH_FRACTION = 0.5
TIME_WORD_ULP_TOLERANCE = 4


def _finite_float(value: Any, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _integer_array(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{label} must be numeric")
    if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
        raise ValueError(f"{label} must contain finite integers")
    return array.astype(np.int64, copy=False)


def validate_solver_realizable_timeline(
    *,
    d3part_times_ms: Any,
    solver_state_times_ms: Any,
    solver_time_steps_ms: Any,
    nominal_output_interval_ms: float,
    termination_time_ms: float,
) -> dict:
    """Validate cycle-aligned output without fitting a tolerance to returned states.

    LS-DYNA can emit a requested database state only on an explicit solver state.
    Consequently, one D3PART interval may exceed the nominal output cadence, but
    only by a positive explicit step reported by GLSTAT inside that interval.  A
    genuinely missing database state is not excused by a percentage tolerance.
    """

    times = np.asarray(d3part_times_ms, dtype=float)
    solver_times = np.asarray(solver_state_times_ms, dtype=float)
    solver_steps = np.asarray(solver_time_steps_ms, dtype=float)
    nominal = _finite_float(nominal_output_interval_ms, "nominal_output_interval_ms")
    termination = _finite_float(termination_time_ms, "termination_time_ms")
    if nominal <= 0.0 or termination <= 0.0:
        raise ValueError("D3PART interval and termination time must be positive")
    if (
        times.ndim != 1
        or times.size < 2
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("timesteps must be finite, strictly increasing, and contain at least two states")
    if (
        solver_times.ndim != 1
        or solver_steps.ndim != 1
        or solver_times.size == 0
        or solver_times.shape != solver_steps.shape
        or not np.all(np.isfinite(solver_times))
        or not np.all(np.isfinite(solver_steps))
        or np.any(solver_steps <= 0.0)
        or (solver_times.size > 1 and np.any(np.diff(solver_times) <= 0.0))
    ):
        raise ValueError(
            "solver state times and explicit time steps must be equal-length finite vectors "
            "with strictly increasing times and positive steps"
        )

    time_scale = np.float32(max(nominal, termination))
    time_tolerance = max(
        np.finfo(np.float64).eps * max(nominal, termination) * TIME_WORD_ULP_TOLERANCE,
        abs(float(np.spacing(time_scale))) * TIME_WORD_ULP_TOLERANCE,
    )
    if abs(float(times[0])) > time_tolerance:
        raise ValueError("D3PART timeline must start near zero")
    if float(solver_times[0]) > float(times[0]) + time_tolerance:
        raise ValueError("solver-step evidence does not cover the first D3PART state")
    if float(solver_times[-1] + solver_steps[-1]) + time_tolerance < float(times[-1]):
        raise ValueError("solver-step evidence does not cover the final D3PART state")

    deltas = np.diff(times)
    realizable_bounds: list[float] = []
    for index, delta in enumerate(deltas):
        interval_mask = (
            (solver_times >= times[index] - time_tolerance)
            & (solver_times < times[index + 1] - time_tolerance)
        )
        if not np.any(interval_mask):
            raise ValueError(
                f"solver-step evidence is absent for D3PART interval {index}"
            )
        local_step = float(np.max(solver_steps[interval_mask]))
        realizable_bound = nominal + local_step
        if float(delta) > realizable_bound + time_tolerance:
            raise ValueError(
                "D3PART state interval is not realizable from the nominal output interval "
                f"and local explicit solver step at interval {index}"
            )
        realizable_bounds.append(realizable_bound)

    final_solver_step = float(solver_steps[-1])
    if float(times[-1]) + nominal + final_solver_step + time_tolerance < termination:
        raise ValueError(
            "D3PART timeline does not cover termination within one solver-realizable output interval"
        )

    return {
        "schema_version": TEMPORAL_GATE_SCHEMA_VERSION,
        "status": "pass",
        "schedule_rule": (
            "each actual D3PART interval must be no greater than the nominal output interval "
            "plus the maximum positive GLSTAT explicit solver step in that interval"
        ),
        "crossing_spatial_resolution_rule": (
            "an alive-to-alive element-centroid capture-plane crossing is resolved only when "
            "its directional cross-step travel is no greater than 0.5 times that case mesh; "
            "otherwise its reference mass is temporal_ambiguity"
        ),
        "crossing_spatial_resolution_mesh_fraction": CROSSING_SPATIAL_RESOLUTION_MESH_FRACTION,
        "time_word_tolerance_ulps": TIME_WORD_ULP_TOLERANCE,
        "time_word_tolerance_ms": time_tolerance,
        "nominal_output_interval_ms": nominal,
        "solver_state_count": int(solver_times.size),
        "maximum_explicit_solver_step_ms": float(np.max(solver_steps)),
        "maximum_actual_state_interval_ms": float(np.max(deltas)),
        "maximum_solver_realizable_interval_ms": float(np.max(realizable_bounds)),
        "maximum_interval_to_solver_realizable_bound_ratio": float(
            np.max(deltas / np.asarray(realizable_bounds, dtype=float))
        ),
    }


def resolve_node_state_coordinates(
    reference_coordinates: np.ndarray, node_state_array: np.ndarray
) -> tuple[np.ndarray, str]:
    """Resolve LASSO's ambiguous ``node_displacement`` payload without guessing."""
    reference = np.asarray(reference_coordinates, dtype=float)
    states = np.asarray(node_state_array, dtype=float)
    if states.ndim != 3 or reference.shape != states.shape[1:]:
        raise ValueError("node coordinate arrays must have shapes [node,3] and [state,node,3]")
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("node coordinate arrays require three components")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(states)):
        raise ValueError("node coordinate arrays must contain only finite values")
    if np.allclose(states[0], reference, rtol=1e-9, atol=1e-9):
        return states, "absolute_coordinates"
    if np.allclose(states[0], 0.0, rtol=0.0, atol=1e-9):
        return reference[None, :, :] + states, "displacement_from_reference"
    raise ValueError(
        "node_displacement is neither absolute coordinates nor displacement from reference"
    )


def hexahedron_reference_volumes(
    reference_coordinates: np.ndarray, connectivity: np.ndarray
) -> np.ndarray:
    """Integrate trilinear eight-node hexahedron reference volumes at 2x2x2 points."""
    coordinates = np.asarray(reference_coordinates, dtype=float)
    node_indexes = _integer_array(connectivity, "element_solid_node_indexes")
    if node_indexes.ndim != 2 or node_indexes.shape[1] != 8 or node_indexes.shape[0] == 0:
        raise ValueError("element_solid_node_indexes must have shape [solid,8]")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("node_coordinates must have shape [node,3]")
    if node_indexes.min() < 0 or node_indexes.max() >= coordinates.shape[0]:
        raise ValueError("solid connectivity references a missing node index")

    signs = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    gauss = 1.0 / math.sqrt(3.0)
    points = np.asarray(
        [(x, y, z) for x in (-gauss, gauss) for y in (-gauss, gauss) for z in (-gauss, gauss)]
    )
    derivatives = np.empty((8, 8, 3), dtype=float)
    for point_index, (xi, eta, zeta) in enumerate(points):
        derivatives[point_index, :, 0] = (
            signs[:, 0] * (1.0 + signs[:, 1] * eta) * (1.0 + signs[:, 2] * zeta) / 8.0
        )
        derivatives[point_index, :, 1] = (
            signs[:, 1] * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 2] * zeta) / 8.0
        )
        derivatives[point_index, :, 2] = (
            signs[:, 2] * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 1] * eta) / 8.0
        )
    element_coordinates = coordinates[node_indexes]
    jacobians = np.einsum("gnp,enc->egpc", derivatives, element_coordinates)
    determinants = np.linalg.det(jacobians)
    if not np.all(np.isfinite(determinants)) or np.any(determinants <= 0.0):
        raise ValueError("solid reference geometry contains non-positive Jacobians")
    volumes = determinants.sum(axis=1)
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise ValueError("solid reference volumes must be finite and positive")
    return volumes


def summarize_capture_crossing_resolution(
    *,
    case_id: str,
    node_positions_mm: np.ndarray,
    element_connectivity: np.ndarray,
    element_alive: np.ndarray,
    capture_plane_z_mm: float,
    mesh_mm: float,
) -> dict:
    """Report the exact spatial seam used to accept or defer crossing events."""

    element_z = np.asarray(node_positions_mm, dtype=float)[:, element_connectivity, 2].mean(axis=2)
    alive = np.asarray(element_alive, dtype=bool)
    previous_z = element_z[:-1]
    current_z = element_z[1:]
    alive_crossing = (
        alive[:-1]
        & alive[1:]
        & (previous_z > capture_plane_z_mm)
        & (current_z <= capture_plane_z_mm)
    )
    crossing_steps = (previous_z - current_z)[alive_crossing]
    spatial_limit = CROSSING_SPATIAL_RESOLUTION_MESH_FRACTION * mesh_mm
    overlimit = (crossing_steps <= 0.0) | (crossing_steps > spatial_limit)
    return {
        "case_id": case_id,
        "spatial_resolution_limit_mm": spatial_limit,
        "alive_crossing_count": int(crossing_steps.size),
        "overlimit_alive_crossing_count": int(np.count_nonzero(overlimit)),
        "maximum_alive_crossing_step_mm": (
            float(np.max(crossing_steps)) if crossing_steps.size else None
        ),
    }


def capture_thresholds_from_design(design: Mapping[str, Any]) -> tuple[CaptureThresholds, float]:
    capture = design["qualification_gates"]["capture_ledger"]
    classification = design["qualification_gates"]["numerical_perforation_classification"]
    thresholds = CaptureThresholds(
        initial_discrete_mass_relative_error_max=_finite_float(
            capture["initial_discrete_mass_relative_error_max"],
            "initial_discrete_mass_relative_error_max",
        ),
        mass_accounting_relative_error_max=_finite_float(
            capture["mass_accounting_relative_error_max"],
            "mass_accounting_relative_error_max",
        ),
        temporal_ambiguity_mass_fraction_max=_finite_float(
            capture["temporal_ambiguity_mass_fraction_max"],
            "temporal_ambiguity_mass_fraction_max",
        ),
        deleted_upstream_mass_fraction_max=_finite_float(
            capture["deleted_upstream_mass_fraction_max"],
            "deleted_upstream_mass_fraction_max",
        ),
        tail_increment_qM_max=_finite_float(capture["tail_increment_qM_max"], "tail_increment_qM_max"),
        tail_increment_qP_max=_finite_float(capture["tail_increment_qP_max"], "tail_increment_qP_max"),
        tail_increment_qE_max=_finite_float(capture["tail_increment_qE_max"], "tail_increment_qE_max"),
        tail_qV_relative_range_max=_finite_float(
            capture["tail_qV_relative_range_max"], "tail_qV_relative_range_max"
        ),
        confirmed_perforation_qM_min=_finite_float(
            classification["confirmed_perforation_qM_min"], "confirmed_perforation_qM_min"
        ),
        confirmed_perforation_qP_min=_finite_float(
            classification["confirmed_perforation_qP_min"], "confirmed_perforation_qP_min"
        ),
        confirmed_nonperforation_qM_max=_finite_float(
            classification["confirmed_nonperforation_qM_max"],
            "confirmed_nonperforation_qM_max",
        ),
        confirmed_nonperforation_qP_max=_finite_float(
            classification["confirmed_nonperforation_qP_max"],
            "confirmed_nonperforation_qP_max",
        ),
        confirmed_nonperforation_worst_case_crossing_mass_max=_finite_float(
            classification["confirmed_nonperforation_worst_case_crossing_mass_max"],
            "confirmed_nonperforation_worst_case_crossing_mass_max",
        ),
    )
    tail_window_fraction = _finite_float(capture["tail_window_fraction"], "tail_window_fraction")
    return thresholds, tail_window_fraction


def extract_case_ledgers_from_arrays(
    *,
    arrays: Mapping[str, Any],
    header: Any,
    design: Mapping[str, Any],
    case_rows: Sequence[Mapping[str, Any]],
    material: Mapping[str, Any],
    solver_state_times_ms: Any,
    solver_time_steps_ms: Any,
) -> dict:
    """Validate one D3PART population and build all identity-bound case ledgers."""
    filetype = getattr(header, "filetype", None)
    filetype_value = getattr(filetype, "value", filetype)
    if filetype_value != D3plotFiletype.D3PART.value:
        raise ValueError("result file header is not D3PART")
    for flag in (
        "has_node_displacement",
        "has_node_velocity",
        "has_element_deletion_data",
    ):
        if getattr(header, flag, False) is not True:
            raise ValueError(f"D3PART header flag {flag} is required")

    required_keys = {
        ArrayType.global_timesteps,
        ArrayType.node_coordinates,
        ArrayType.node_displacement,
        ArrayType.node_velocity,
        ArrayType.element_solid_node_indexes,
        ArrayType.element_solid_part_indexes,
        ArrayType.element_solid_is_alive,
        ArrayType.part_ids,
    }
    missing = sorted(required_keys - set(arrays))
    if missing:
        raise ValueError(f"D3PART required arrays are missing: {missing}")

    times = np.asarray(arrays[ArrayType.global_timesteps], dtype=float)
    expected_interval = _finite_float(
        design["measurement_contract"]["fragment_d3part_interval_ms"],
        "fragment_d3part_interval_ms",
    )
    termination_time = _finite_float(
        design["common"]["termination_time_ms"],
        "termination_time_ms",
    )
    temporal_gate = validate_solver_realizable_timeline(
        d3part_times_ms=times,
        solver_state_times_ms=solver_state_times_ms,
        solver_time_steps_ms=solver_time_steps_ms,
        nominal_output_interval_ms=expected_interval,
        termination_time_ms=termination_time,
    )
    deltas = np.diff(times)

    reference = np.asarray(arrays[ArrayType.node_coordinates], dtype=float)
    positions, coordinate_mode = resolve_node_state_coordinates(
        reference, arrays[ArrayType.node_displacement]
    )
    velocities = np.asarray(arrays[ArrayType.node_velocity], dtype=float)
    if velocities.shape != positions.shape or not np.all(np.isfinite(velocities)):
        raise ValueError("node_velocity must be finite and have shape [state,node,3]")
    if positions.shape[0] != times.size:
        raise ValueError("node state count differs from timesteps")

    connectivity = _integer_array(
        arrays[ArrayType.element_solid_node_indexes], "element_solid_node_indexes"
    )
    volumes = hexahedron_reference_volumes(reference, connectivity)
    part_indexes = _integer_array(
        arrays[ArrayType.element_solid_part_indexes], "element_solid_part_indexes"
    )
    part_ids = _integer_array(arrays[ArrayType.part_ids], "part_ids")
    if part_ids.ndim != 1 or part_ids.size == 0 or np.unique(part_ids).size != part_ids.size:
        raise ValueError("part_ids must be a nonempty unique vector")
    if part_indexes.shape != (connectivity.shape[0],):
        raise ValueError("element_solid_part_indexes must have shape [solid]")
    if part_indexes.min() < 0 or part_indexes.max() >= part_ids.size:
        raise ValueError("element_solid_part_indexes contains an out-of-range part index")
    solid_part_ids = part_ids[part_indexes]
    used_solid_part_ids = np.unique(solid_part_ids)
    used_solid_part_id_set = set(used_solid_part_ids.tolist())
    zero_solid_catalog_part_ids = np.asarray(
        [part_id for part_id in part_ids.tolist() if part_id not in used_solid_part_id_set],
        dtype=np.int64,
    )

    alive_raw = np.asarray(arrays[ArrayType.element_solid_is_alive])
    if not np.issubdtype(alive_raw.dtype, np.number) and not np.issubdtype(
        alive_raw.dtype, np.bool_
    ):
        raise ValueError("element_solid_is_alive must be numeric or boolean")
    if not np.all(np.isfinite(alive_raw)) or np.any(alive_raw < 0):
        raise ValueError("element_solid_is_alive must contain finite nonnegative values")
    if alive_raw.shape != (times.size, connectivity.shape[0]):
        raise ValueError("element_solid_is_alive must have shape [state,solid]")
    alive = alive_raw != 0
    if not np.all(alive[0]):
        raise ValueError("all fragment solids must be alive in the initial D3PART state")

    design_cases = {case["case_id"]: case for case in design["cases"]}
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    fragment_part_ids: list[int] = []
    for row in case_rows:
        case_id = str(row["case_id"])
        if case_id in rows_by_id:
            raise ValueError(f"duplicate case-map case_id: {case_id}")
        rows_by_id[case_id] = row
        fragment_part_ids.append(int(row["fragment_part_id"]))
    if set(rows_by_id) != set(design_cases):
        raise ValueError("case-map case IDs differ from the frozen design")
    if len(fragment_part_ids) != len(set(fragment_part_ids)):
        raise ValueError("fragment part IDs must be unique across cases")
    if set(used_solid_part_ids.tolist()) != set(fragment_part_ids):
        raise ValueError("D3PART used solid PID set differs from the exact fragment PID set")

    density = _finite_float(material["strength_model"]["density_kg_mm3"], "density_kg_mm3")
    if density <= 0.0:
        raise ValueError("density_kg_mm3 must be positive")
    element_masses = volumes * density
    thresholds, tail_window_fraction = capture_thresholds_from_design(design)
    capture_plane = _finite_float(
        design["measurement_contract"]["capture_plane"]["coordinate_mm"],
        "capture_plane.coordinate_mm",
    )

    results: list[dict] = []
    crossing_resolution: list[dict] = []
    for frozen_case in design["cases"]:
        case_id = frozen_case["case_id"]
        row = rows_by_id[case_id]
        role = str(row["role"])
        speed = _finite_float(row["impact_speed_mps"], f"{case_id}.impact_speed_mps")
        mesh = _finite_float(row["mesh_mm"], f"{case_id}.mesh_mm")
        row_plane = _finite_float(row["capture_plane_z_mm"], f"{case_id}.capture_plane_z_mm")
        expected_mass = _finite_float(
            row["fragment_initial_reference_mass_kg"],
            f"{case_id}.fragment_initial_reference_mass_kg",
        )
        if role != frozen_case["role"] or speed != float(frozen_case["impact_speed_mps"]):
            raise ValueError(f"{case_id} role or impact speed differs from the frozen design")
        if mesh != float(frozen_case["mesh_mm"]) or mesh <= 0.0:
            raise ValueError(f"{case_id} mesh differs from the frozen design or is non-positive")
        if row_plane != capture_plane:
            raise ValueError(f"{case_id} capture plane differs from the frozen design")
        if expected_mass <= 0.0:
            raise ValueError(f"{case_id} expected reference mass must be positive")

        fragment_pid = int(row["fragment_part_id"])
        element_mask = solid_part_ids == fragment_pid
        expected_elements = int(row["fragment_elements"])
        actual_elements = int(np.count_nonzero(element_mask))
        if expected_elements <= 0 or actual_elements != expected_elements:
            raise ValueError(
                f"{case_id} fragment element count mismatch: expected {expected_elements}, got {actual_elements}"
            )

        crossing_resolution.append(
            summarize_capture_crossing_resolution(
                case_id=case_id,
                node_positions_mm=positions,
                element_connectivity=connectivity[element_mask],
                element_alive=alive[:, element_mask],
                capture_plane_z_mm=capture_plane,
                mesh_mm=mesh,
            )
        )

        ledger = build_capture_ledger(
            times_ms=times,
            node_positions_mm=positions,
            node_velocities_mm_per_ms=velocities,
            element_connectivity=connectivity[element_mask],
            element_alive=alive[:, element_mask],
            element_reference_mass_kg=element_masses[element_mask],
            expected_reference_mass_kg=expected_mass,
            capture_plane_z_mm=capture_plane,
            impact_speed_mps=speed,
            maximum_unambiguous_step_mm=CROSSING_SPATIAL_RESOLUTION_MESH_FRACTION * mesh,
            tail_window_fraction=tail_window_fraction,
            thresholds=thresholds,
        )
        ledger_schema_version = ledger.pop("schema_version")
        results.append(
            {
                "ledger_schema_version": ledger_schema_version,
                "case_id": case_id,
                "role": role,
                "fragment_part_id": fragment_pid,
                "impact_speed_mps": speed,
                "mesh_mm": mesh,
                "run_integrity_gate": "not_assessed_by_d3part_extractor",
                **ledger,
            }
        )

    temporal_gate["case_crossing_resolution"] = crossing_resolution
    return {
        "d3part_summary": {
            "coordinate_mode": coordinate_mode,
            "alive_encoding": "zero_deleted_nonzero_alive",
            "state_count": int(times.size),
            "node_count": int(reference.shape[0]),
            "solid_count": int(connectivity.shape[0]),
            "part_ids": used_solid_part_ids.tolist(),
            "catalog_part_ids": part_ids.tolist(),
            "zero_solid_catalog_part_ids": zero_solid_catalog_part_ids.tolist(),
            "first_time_ms": float(times[0]),
            "last_time_ms": float(times[-1]),
            "minimum_state_interval_ms": float(deltas.min()),
            "maximum_state_interval_ms": float(deltas.max()),
            "expected_state_interval_ms": expected_interval,
            "termination_time_ms": termination_time,
            "termination_coverage_tolerance_ms": (
                expected_interval + temporal_gate["maximum_explicit_solver_step_ms"]
            ),
            "temporal_gate": temporal_gate,
        },
        "cases": results,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value using the assessor's canonical serialization contract."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase sha256 digest")
    return value


def _require_exact_mapping(
    value: Any, *, label: str, expected_keys: set[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is required and must be an object")
    if set(value) != expected_keys:
        missing = sorted(expected_keys - set(value))
        extra = sorted(set(value) - expected_keys)
        raise ValueError(f"{label} fields mismatch; missing={missing}, extra={extra}")
    return value


def parser_source_sha256_from_extraction(extraction_payload: Mapping[str, Any]) -> str:
    """Bind the exact reader and extractor identities accepted by integration review."""
    reader = extraction_payload.get("reader_identity")
    extractor = extraction_payload.get("extractor_identity")
    if not isinstance(reader, Mapping) or not isinstance(extractor, Mapping):
        raise ValueError("extraction reader_identity and extractor_identity must be objects")
    return canonical_sha256(
        {"reader_identity": dict(reader), "extractor_identity": dict(extractor)}
    )


def bridge_extraction_to_assessor_payload(
    *,
    extraction_payload: Mapping[str, Any],
    design_path: Path,
    package_manifest_path: Path,
    returned_artifact_path: Path,
    run_integrity_evidence: Mapping[str, Any] | None,
    integration_acceptance: Mapping[str, Any] | None,
) -> dict:
    """Promote an extraction only after separately bound run and integration acceptance.

    Real D3PART readability has been exercised at the adapter seam, but the extraction
    deliberately remains pending independent integration acceptance.  This bridge is
    the only seam that changes each case's run-integrity result to ``pass``, and it does
    so only when both external evidence documents are explicit, exact, and hash-bound.
    """
    extraction_keys = {
        "schema_version",
        "package_revision",
        "integration_status",
        "artifact_identity",
        "run_identity",
        "reader_identity",
        "extractor_identity",
        "d3part_summary",
        "cases",
        "claim_boundary",
    }
    extraction = _require_exact_mapping(
        extraction_payload,
        label="extraction payload",
        expected_keys=extraction_keys,
    )
    if extraction["schema_version"] != "hf-p0-r05-d3part-extraction@1.0.0":
        raise ValueError("extraction payload schema is not supported")
    if extraction["integration_status"] != INTEGRATION_STATUS:
        raise ValueError("extraction payload integration status is not the fail-closed sentinel")

    run_evidence = _require_exact_mapping(
        run_integrity_evidence,
        label="run-integrity evidence",
        expected_keys={
            "schema_version",
            "evidence_id",
            "status",
            "run_id",
            "solver_build",
            "d3part_family_sha256",
            "returned_artifact_sha256",
        },
    )
    if run_evidence["schema_version"] != "hf-p0-r05-run-integrity-evidence@1.0.0":
        raise ValueError("run-integrity evidence schema is not supported")
    if run_evidence["status"] != "pass":
        raise ValueError("run-integrity evidence status must be pass")
    if not str(run_evidence["evidence_id"]).strip():
        raise ValueError("run-integrity evidence_id must be nonempty")

    integration = _require_exact_mapping(
        integration_acceptance,
        label="integration-acceptance evidence",
        expected_keys={
            "schema_version",
            "evidence_id",
            "status",
            "accepted_extraction_sha256",
            "accepted_parser_source_sha256",
        },
    )
    if integration["schema_version"] != "hf-p0-r05-d3part-integration-acceptance@1.0.0":
        raise ValueError("integration-acceptance evidence schema is not supported")
    if integration["status"] != "pass":
        raise ValueError("integration-acceptance evidence status must be pass")
    if not str(integration["evidence_id"]).strip():
        raise ValueError("integration-acceptance evidence_id must be nonempty")

    artifact = _require_exact_mapping(
        extraction["artifact_identity"],
        label="extraction artifact_identity",
        expected_keys={
            "package_revision",
            "design_sha256",
            "deck_sha256",
            "case_map_sha256",
            "material_sha256",
        },
    )
    for field in ("design_sha256", "deck_sha256", "case_map_sha256", "material_sha256"):
        _require_sha256(artifact[field], f"artifact_identity.{field}")
    if artifact["package_revision"] != extraction["package_revision"]:
        raise ValueError("extraction package revision does not match artifact identity")

    try:
        design = json.loads(Path(design_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read bridge design: {exc}") from exc
    if not isinstance(design, dict):
        raise ValueError("bridge design must be a JSON object")
    if design.get("package_revision") != extraction["package_revision"]:
        raise ValueError("bridge design package revision does not match extraction")
    if sha256_file(Path(design_path)) != artifact["design_sha256"]:
        raise ValueError("bridge design raw hash does not match extraction artifact identity")
    design_canonical_hash = canonical_sha256(design)

    run = _require_exact_mapping(
        extraction["run_identity"],
        label="extraction run_identity",
        expected_keys={
            "run_id",
            "solver_build",
            "d3part_family_sha256",
            "d3part_family",
        },
    )
    family_hash = _require_sha256(run["d3part_family_sha256"], "run_identity.d3part_family_sha256")
    for field in ("run_id", "solver_build"):
        if not str(run[field]).strip():
            raise ValueError(f"run_identity.{field} must be nonempty")
        if run_evidence[field] != run[field]:
            raise ValueError(f"run-integrity evidence {field} does not match extraction")
    if run_evidence["d3part_family_sha256"] != family_hash:
        raise ValueError("run-integrity evidence d3part family does not match extraction")

    returned_hash = sha256_file(Path(returned_artifact_path))
    if run_evidence["returned_artifact_sha256"] != returned_hash:
        raise ValueError("run-integrity evidence returned artifact hash does not match raw artifact")
    _require_sha256(returned_hash, "returned artifact hash")

    extraction_hash = canonical_sha256(extraction)
    if integration["accepted_extraction_sha256"] != extraction_hash:
        raise ValueError("integration acceptance is not bound to this extraction payload")
    parser_source_hash = parser_source_sha256_from_extraction(extraction)
    if integration["accepted_parser_source_sha256"] != parser_source_hash:
        raise ValueError("integration acceptance is not bound to this parser source identity")

    raw_cases = extraction["cases"]
    if not isinstance(raw_cases, list) or any(not isinstance(case, Mapping) for case in raw_cases):
        raise ValueError("extraction cases must be a list of objects")
    expected_case_ids = [str(case["case_id"]) for case in design.get("cases", [])]
    observed_case_ids = [str(case.get("case_id", "")) for case in raw_cases]
    if observed_case_ids != expected_case_ids or len(set(observed_case_ids)) != len(observed_case_ids):
        raise ValueError("extraction cases do not exactly match the ordered design cases")
    promoted_cases = []
    for case in raw_cases:
        if case.get("run_integrity_gate") != "not_assessed_by_d3part_extractor":
            raise ValueError("extraction case run_integrity_gate is not the fail-closed sentinel")
        promoted = dict(case)
        promoted["run_integrity_gate"] = "pass"
        promoted_cases.append(promoted)

    reader = extraction["reader_identity"]
    extractor = extraction["extractor_identity"]
    if not isinstance(reader, Mapping) or not isinstance(extractor, Mapping):
        raise ValueError("reader and extractor identities must be objects")
    reader_name = str(reader.get("name", "")).strip()
    reader_version = str(reader.get("version", "")).strip()
    extractor_name = str(extractor.get("name", "")).strip()
    extractor_version = str(extractor.get("version", "")).strip()
    if not all((reader_name, reader_version, extractor_name, extractor_version)):
        raise ValueError("reader and extractor names and versions must be nonempty")

    manifest_hash = sha256_file(Path(package_manifest_path))
    _require_sha256(manifest_hash, "package manifest hash")
    return {
        "schema_version": "hf-p0-r05-case-results@1.0.0",
        "artifact_identity": {
            "package_revision": artifact["package_revision"],
            "design_sha256": artifact["design_sha256"],
            "design_canonical_sha256": design_canonical_hash,
            "deck_sha256": artifact["deck_sha256"],
            "case_map_sha256": artifact["case_map_sha256"],
            "package_manifest_sha256": manifest_hash,
        },
        "run_identity": {
            "run_id": run["run_id"],
            "solver_build": run["solver_build"],
            "returned_artifact_sha256": returned_hash,
        },
        "parser_identity": {
            "name": f"{extractor_name}+{reader_name}",
            "version": f"{extractor_version}+{reader_version}",
            "source_sha256": parser_source_hash,
            "acceptance_status": "pass",
        },
        "cases": promoted_cases,
    }


def artifact_identity_from_paths(
    *,
    package_revision: str,
    design_path: Path,
    deck_path: Path,
    case_map_path: Path,
    material_path: Path,
) -> dict:
    revision = str(package_revision).strip()
    if not revision:
        raise ValueError("package_revision must be nonempty")
    return {
        "package_revision": revision,
        "design_sha256": sha256_file(design_path),
        "deck_sha256": sha256_file(deck_path),
        "case_map_sha256": sha256_file(case_map_path),
        "material_sha256": sha256_file(material_path),
    }


def _natural_sort_key(name: str) -> list[int | str]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"([0-9]+)", name)]


def d3part_family_identity(
    base_path: Path, *, run_id: str, solver_build: str
) -> dict:
    base = base_path.resolve()
    run = str(run_id).strip()
    solver = str(solver_build).strip()
    if not run or not solver:
        raise ValueError("run_id and solver_build must be nonempty")
    if not base.is_file():
        raise FileNotFoundError(f"D3PART base file not found: {base}")
    continuation_pattern = re.compile(re.escape(base.name) + r"[0-9]+$")
    continuations = sorted(
        (
            path
            for path in base.parent.iterdir()
            if path.is_file() and continuation_pattern.fullmatch(path.name)
        ),
        key=lambda path: _natural_sort_key(path.name),
    )
    entries = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (base, *continuations)
    ]
    canonical_entries = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "run_id": run,
        "solver_build": solver,
        "d3part_family_sha256": _sha256_bytes(canonical_entries),
        "d3part_family": entries,
    }


def assemble_output_payload(
    *,
    package_revision: str,
    extraction: Mapping[str, Any],
    artifact_identity: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    reader_identity: Mapping[str, Any],
    extractor_identity: Mapping[str, Any],
) -> dict:
    revision = str(package_revision)
    if artifact_identity.get("package_revision") != revision:
        raise ValueError("artifact identity package revision mismatch")
    if not str(run_identity.get("run_id", "")).strip():
        raise ValueError("one nonempty run identity is required")
    if set(extraction) != {"d3part_summary", "cases"}:
        raise ValueError("extraction must contain exactly d3part_summary and cases")
    return {
        "schema_version": "hf-p0-r05-d3part-extraction@1.0.0",
        "package_revision": revision,
        "integration_status": INTEGRATION_STATUS,
        "artifact_identity": dict(artifact_identity),
        "run_identity": dict(run_identity),
        "reader_identity": dict(reader_identity),
        "extractor_identity": dict(extractor_identity),
        "d3part_summary": extraction["d3part_summary"],
        "cases": extraction["cases"],
        "claim_boundary": (
            "Real D3PART readability has been exercised at the adapter seam for the R05 family, "
            "but this hash-bound extraction remains pending independent integration acceptance "
            "and does not itself establish run integrity, numerical qualification, or physical "
            "validity."
        ),
    }


def strict_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _reader_version() -> str:
    metadata_path = VENDOR_DIR / "lasso_python-2.0.4.dist-info" / "METADATA"
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Version: "):
            return line.removeprefix("Version: ").strip()
    raise RuntimeError(f"lasso-python version is missing from {metadata_path}")


def _reader_identity() -> dict:
    d3plot_source = VENDOR_DIR / "lasso" / "dyna" / "d3plot.py"
    binout_source = VENDOR_DIR / "lasso" / "dyna" / "binout.py"
    array_type_source = VENDOR_DIR / "lasso" / "dyna" / "array_type.py"
    return {
        "name": "lasso-python",
        "version": _reader_version(),
        "api": "lasso.dyna.D3plot+Binout",
        "d3plot_source_sha256": sha256_file(d3plot_source),
        "binout_source_sha256": sha256_file(binout_source),
        "array_type_source_sha256": sha256_file(array_type_source),
    }


def _extractor_identity() -> dict:
    return {
        "name": EXTRACTOR_NAME,
        "version": EXTRACTOR_VERSION,
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }


def _append_vendor_import_path() -> None:
    """Expose the vendored reader without shadowing environment binary packages."""

    vendor = str(VENDOR_DIR)
    sys.path[:] = [entry for entry in sys.path if entry != vendor]
    sys.path.append(vendor)


def _require_vendored_lasso_source() -> None:
    import lasso

    source = Path(lasso.__file__).resolve()
    if not source.is_relative_to(VENDOR_DIR.resolve()):
        raise RuntimeError(f"lasso-python resolved outside the project vendor directory: {source}")


def _load_lasso_d3plot() -> type:
    _append_vendor_import_path()
    try:
        from lasso.dyna import D3plot as LassoD3plot
    except ImportError as exc:
        raise RuntimeError(
            "cannot import the project-local lasso-python reader; use its compatible "
            "Python 3.12 environment with the vendored binary dependencies"
        ) from exc
    _require_vendored_lasso_source()
    return LassoD3plot


def _load_lasso_binout() -> type:
    _append_vendor_import_path()
    try:
        from lasso.dyna import Binout as LassoBinout
    except ImportError as exc:
        raise RuntimeError(
            "cannot import the project-local lasso-python BINOUT reader; use its compatible "
            "Python 3.12 environment with the vendored binary dependencies"
        ) from exc
    _require_vendored_lasso_source()
    return LassoBinout


def load_solver_timeline_from_binout(binout_pattern: Path | str) -> dict[str, np.ndarray]:
    """Read the explicit GLSTAT timeline that makes cycle-aligned D3PART output auditable."""

    binout = _load_lasso_binout()(str(binout_pattern))
    return {
        "solver_state_times_ms": np.asarray(binout.read("glstat", "time"), dtype=float),
        "solver_time_steps_ms": np.asarray(binout.read("glstat", "time_step"), dtype=float),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract fail-closed R05 capture ledgers from a real D3PART family"
    )
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--d3part", type=Path, required=True, help="base d3part file, not a glob")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--solver-build", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_dir = args.package_dir.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing extraction: {output_path}")

    design_path = package_dir / "p0_cases_r05.json"
    deck_path = package_dir / "p0_velocity_mesh_calibration.k"
    case_map_path = package_dir / "p0_case_part_map.csv"
    material_path = package_dir / "jc4340_reference.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    material = json.loads(material_path.read_text(encoding="utf-8"))
    with case_map_path.open("r", encoding="utf-8", newline="") as handle:
        case_rows = list(csv.DictReader(handle))

    run_identity = d3part_family_identity(
        args.d3part,
        run_id=args.run_id,
        solver_build=args.solver_build,
    )
    reader_class = _load_lasso_d3plot()
    d3part = reader_class(
        str(args.d3part.resolve()),
        state_array_filter=[
            ArrayType.global_timesteps,
            ArrayType.node_displacement,
            ArrayType.node_velocity,
            ArrayType.element_solid_is_alive,
        ],
        buffered_reading=True,
    )
    solver_timeline = load_solver_timeline_from_binout(args.d3part.resolve().parent / "binout*")
    extraction = extract_case_ledgers_from_arrays(
        arrays=d3part.arrays,
        header=d3part.header,
        design=design,
        case_rows=case_rows,
        material=material,
        **solver_timeline,
    )
    artifact_identity = artifact_identity_from_paths(
        package_revision=design["package_revision"],
        design_path=design_path,
        deck_path=deck_path,
        case_map_path=case_map_path,
        material_path=material_path,
    )
    payload = assemble_output_payload(
        package_revision=design["package_revision"],
        extraction=extraction,
        artifact_identity=artifact_identity,
        run_identity=run_identity,
        reader_identity=_reader_identity(),
        extractor_identity=_extractor_identity(),
    )
    serialized = strict_json(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8", newline="\n")
    print(
        strict_json(
            {
                "output": str(output_path),
                "integration_status": payload["integration_status"],
                "case_count": len(payload["cases"]),
                "run_id": payload["run_identity"]["run_id"],
            }
        ),
        end="",
    )


if __name__ == "__main__":
    main()

