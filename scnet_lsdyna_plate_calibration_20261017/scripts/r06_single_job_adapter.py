"""Fail-closed R06 single-job capture and MAT_015 HVAR6 adapter.

Returned solver measurements remain separate from P0, dataset, and physical
admission. Missing contact evidence and invalid history values are represented
by a null value plus an explicit reason; they are never coerced to zero.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from extract_r05_d3part import (
    ArrayType,
    D3plotFiletype,
    _load_lasso_binout,
    _load_lasso_d3plot,
    extract_case_ledgers_from_arrays,
    load_solver_timeline_from_binout,
)


HVAR6_ONE_BASED = 6
HVAR6_ZERO_BASED = 5
SENTINEL_ABS_MIN = 1.0e29
WARNING_20340 = re.compile(
    r"(?:Warning\s+20340|STR\+340).*binary interface force data is unavailable",
    re.IGNORECASE,
)


def _strict_json(path: Path) -> dict:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=no_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value in {path}: {value}")
        ),
    )


def strict_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _finite_float(value: Any, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _equal_number(left: Any, right: Any, label: str) -> None:
    if not math.isclose(
        _finite_float(left, f"{label}.left"),
        _finite_float(right, f"{label}.right"),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{label} differs across R06f artifacts")


def normalize_r06_single_job_capture_inputs(
    *,
    job_contract: Mapping[str, Any],
    design: Mapping[str, Any],
    measurement_contract: Mapping[str, Any],
    case_row: Mapping[str, Any],
) -> tuple[dict, list[dict]]:
    """Map one identity-bound R06 job into the reusable capture-ledger seam."""

    revision = str(design["package_revision"])
    if job_contract["package_revision"] != revision:
        raise ValueError("job/design package revision mismatch")
    boundary = measurement_contract["version_boundary"]
    if boundary["applies_to_package_revision"] != revision:
        raise ValueError("measurement contract is not bound to this package revision")
    experiments = design["paired_experiments"]
    jobs = [job for experiment in experiments for job in experiment["jobs"]]
    if len(jobs) != 1:
        raise ValueError("R06 single-job adapter requires exactly one frozen job")
    job = jobs[0]
    if job["job_id"] != job_contract["job_id"] or case_row["job_id"] != job["job_id"]:
        raise ValueError("job identity differs across design, contract, and case map")

    frozen_case = job_contract["case"]
    identity_fields = (
        "case_id",
        "plate_part_id",
        "fragment_part_id",
        "plate_element_count",
        "fragment_element_count",
        "impact_speed_mps",
        "mesh_mm",
        "capture_plane_z_mm",
        "fragment_initial_reference_mass_kg",
        "termination_time_ms",
    )
    map_names = {
        "plate_element_count": "plate_elements",
        "fragment_element_count": "fragment_elements",
    }
    for field in identity_fields:
        map_field = map_names.get(field, field)
        if field == "case_id":
            if str(frozen_case[field]) != str(case_row[map_field]):
                raise ValueError("case_id differs across job contract and case map")
        else:
            _equal_number(frozen_case[field], case_row[map_field], field)

    measure = design["shared_measurement_upgrade"]
    contract_binary = measurement_contract["d3part_keyword"]["binary_card"]
    expected_binary = [
        measure["fragment_d3part_interval_ms"],
        0,
        0,
        0,
        measure["fragment_part_set_id"],
    ]
    if contract_binary != expected_binary:
        raise ValueError("D3PART binary card differs between design and measurement contract")
    if measure.get("d3part_extent_cards") != measurement_contract["d3part_keyword"]["extent_cards"]:
        raise ValueError("D3PART extent cards differ between design and measurement contract")

    mapped_design = {
        "measurement_contract": {
            "fragment_d3part_interval_ms": float(measure["fragment_d3part_interval_ms"]),
            "capture_plane": dict(measure["capture_plane"]),
        },
        "common": {"termination_time_ms": float(frozen_case["termination_time_ms"])},
        "qualification_gates": measurement_contract["capture_ledger_gates"],
        "cases": [
            {
                "case_id": frozen_case["case_id"],
                "role": job_contract["role"],
                "impact_speed_mps": float(frozen_case["impact_speed_mps"]),
                "mesh_mm": float(frozen_case["mesh_mm"]),
            }
        ],
    }
    mapped_row = dict(case_row)
    mapped_row["role"] = job_contract["role"]
    return mapped_design, [mapped_row]


def _integer_vector(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{label} must be a numeric vector")
    if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
        raise ValueError(f"{label} must contain finite integers")
    return array.astype(np.int64, copy=False)


def _null_stats(reason: str) -> dict:
    return {
        "status": "unavailable",
        "reason": reason,
        "minimum": None,
        "mean": None,
        "p50": None,
        "p95": None,
        "p99": None,
        "maximum": None,
    }


def _stats(values: np.ndarray) -> dict:
    return {
        "status": "available",
        "reason": None,
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def extract_mat015_hvar6_from_arrays(
    *,
    arrays: Mapping[str, Any],
    header: Any,
    case: Mapping[str, Any],
    sentinel_abs_min: float = SENTINEL_ABS_MIN,
) -> dict:
    """Extract MAT_015 HVAR6 from full D3PLOT with a mandatory alive mask."""

    filetype = getattr(header, "filetype", None)
    filetype_value = getattr(filetype, "value", filetype)
    if filetype_value != D3plotFiletype.D3PLOT.value:
        raise ValueError("MAT_015 HVAR6 source must be the full D3PLOT family")
    required = {
        ArrayType.global_timesteps,
        ArrayType.element_solid_history_variables,
        ArrayType.element_solid_is_alive,
        ArrayType.element_solid_ids,
        ArrayType.element_solid_part_indexes,
        ArrayType.part_ids,
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"D3PLOT arrays missing: {missing}")

    times = np.asarray(arrays[ArrayType.global_timesteps], dtype=float)
    if (
        times.ndim != 1
        or times.size == 0
        or not np.all(np.isfinite(times))
        or not math.isclose(float(times[0]), 0.0, rel_tol=0.0, abs_tol=1e-12)
        or (times.size > 1 and not np.all(np.diff(times) > 0.0))
    ):
        raise ValueError("D3PLOT times must be finite, start at zero, and increase strictly")
    history = np.asarray(arrays[ArrayType.element_solid_history_variables])
    if history.ndim != 4 or history.shape[2] != 1 or history.shape[3] < HVAR6_ONE_BASED:
        raise ValueError("solid history must have shape [state,solid,1,nvar] with nvar >= 6")
    alive_raw = np.asarray(arrays[ArrayType.element_solid_is_alive])
    if (
        alive_raw.shape != history.shape[:2]
        or not np.issubdtype(alive_raw.dtype, np.number)
        or not np.all(np.isfinite(alive_raw))
        or np.any(alive_raw < 0)
    ):
        raise ValueError("element_solid_is_alive must be finite/nonnegative [state,solid]")
    alive = alive_raw != 0
    if not np.all(alive[0]):
        raise ValueError("all solids must be alive in the initial D3PLOT state")
    if np.any(alive[1:] & ~alive[:-1]):
        raise ValueError("element alive history is non-monotone: a deleted solid revived")
    if history.shape[0] != times.size:
        raise ValueError("D3PLOT history state count differs from time count")

    element_ids = _integer_vector(arrays[ArrayType.element_solid_ids], "element_solid_ids")
    part_indexes = _integer_vector(
        arrays[ArrayType.element_solid_part_indexes], "element_solid_part_indexes"
    )
    part_ids = _integer_vector(arrays[ArrayType.part_ids], "part_ids")
    solid_count = history.shape[1]
    if (
        element_ids.shape != (solid_count,)
        or np.unique(element_ids).size != solid_count
        or part_indexes.shape != (solid_count,)
        or part_ids.size == 0
        or np.unique(part_ids).size != part_ids.size
        or np.any(part_indexes < 0)
        or np.any(part_indexes >= part_ids.size)
    ):
        raise ValueError("D3PLOT element/PID identity arrays are inconsistent")
    used_part_ids = part_ids[part_indexes]
    expected = {
        int(case["plate_part_id"]): int(case["plate_element_count"]),
        int(case["fragment_part_id"]): int(case["fragment_element_count"]),
    }
    if set(np.unique(used_part_ids).tolist()) != set(expected):
        raise ValueError("D3PLOT used solid PID set differs from the job contract")
    for pid, count in expected.items():
        if int(np.count_nonzero(used_part_ids == pid)) != count:
            raise ValueError(f"D3PLOT element count differs for PID {pid}")

    hvar6 = np.asarray(history[:, :, 0, HVAR6_ZERO_BASED], dtype=float)
    states: list[dict] = []
    parts: list[dict] = []
    overall_invalid_alive = 0
    for pid, label in (
        (int(case["plate_part_id"]), "plate"),
        (int(case["fragment_part_id"]), "fragment"),
    ):
        part_mask = used_part_ids == pid
        part_invalid_alive = 0
        all_valid: list[np.ndarray] = []
        for state_index, time_ms in enumerate(times):
            state_alive = alive[state_index] & part_mask
            values = hvar6[state_index, state_alive]
            invalid = ~np.isfinite(values) | (np.abs(values) >= sentinel_abs_min)
            invalid_count = int(np.count_nonzero(invalid))
            part_invalid_alive += invalid_count
            overall_invalid_alive += invalid_count
            if values.size == 0:
                stats = _null_stats("no_alive_elements")
            elif invalid_count:
                stats = _null_stats("alive_hvar6_contains_nonfinite_or_sentinel")
            else:
                stats = _stats(values)
                all_valid.append(values)
            states.append(
                {
                    "state_index": state_index,
                    "time_ms": float(time_ms),
                    "part": label,
                    "part_id": pid,
                    "alive_element_count": int(np.count_nonzero(state_alive)),
                    "deleted_element_count": int(np.count_nonzero(part_mask))
                    - int(np.count_nonzero(state_alive)),
                    "invalid_alive_value_count": invalid_count,
                    **stats,
                }
            )
        if part_invalid_alive:
            aggregate = _null_stats("one_or_more_alive_hvar6_values_are_invalid")
        elif all_valid:
            aggregate = _stats(np.concatenate(all_valid))
        else:
            aggregate = _null_stats("no_valid_alive_hvar6_values")
        parts.append(
            {
                "part": label,
                "part_id": pid,
                "element_count": int(np.count_nonzero(part_mask)),
                "invalid_alive_value_count": part_invalid_alive,
                **aggregate,
            }
        )

    last_alive: list[dict] = []
    for solid_index, element_id in enumerate(element_ids):
        live_indexes = np.flatnonzero(alive[:, solid_index])
        last_index = int(live_indexes[-1])
        value = float(hvar6[last_index, solid_index])
        invalid = not math.isfinite(value) or abs(value) >= sentinel_abs_min
        first_deleted = np.flatnonzero(~alive[:, solid_index])
        last_alive.append(
            {
                "element_id": int(element_id),
                "part_id": int(used_part_ids[solid_index]),
                "last_alive_state_index": last_index,
                "last_alive_time_ms": float(times[last_index]),
                "last_alive_hvar6": None if invalid else value,
                "last_alive_hvar6_status": (
                    "unavailable_nonfinite_or_sentinel" if invalid else "available"
                ),
                "first_deleted_state_index": int(first_deleted[0]) if first_deleted.size else None,
                "first_deleted_time_ms": (
                    float(times[int(first_deleted[0])]) if first_deleted.size else None
                ),
            }
        )

    return {
        "schema_version": "hf-p0-r06-mat015-hvar6@1.0.0",
        "status": "unavailable" if overall_invalid_alive else "available",
        "source": "full_d3plot",
        "history_variable_number_one_based": HVAR6_ONE_BASED,
        "array_index_zero_based": HVAR6_ZERO_BASED,
        "alive_mask_required": True,
        "post_deletion_values_ignored": True,
        "sentinel_abs_min": sentinel_abs_min,
        "state_count": int(times.size),
        "solid_count": solid_count,
        "invalid_alive_value_count": overall_invalid_alive,
        "parts": parts,
        "states": states,
        "element_last_alive": last_alive,
        "claim_boundary": "HVAR6 is a returned MAT_015 measurement channel, not material calibration or physical validation.",
    }


NCFORC_STATE_VARIABLES = (
    "ids",
    "x",
    "y",
    "z",
    "pressure",
    "x_force",
    "y_force",
    "z_force",
)


def summarize_ragged_ncforc_states(states: Sequence[Mapping[str, Any]]) -> dict:
    """Validate ragged NCFORC states while retaining every state's node IDs."""

    if not states:
        raise ValueError("NCFORC contains no states")
    records: list[dict] = []
    union_ids: set[int] = set()
    intersection_ids: set[int] | None = None
    ranges = {
        name: [math.inf, -math.inf]
        for name in ("pressure", "x_force", "y_force", "z_force")
    }
    previous_time: float | None = None
    for state_index, state in enumerate(states):
        state_name = str(state.get("state_name", f"state_{state_index:06d}"))
        time_ms = _finite_float(state["time_ms"], f"{state_name}.time_ms")
        if previous_time is not None and time_ms <= previous_time:
            raise ValueError("NCFORC times must increase strictly")
        previous_time = time_ms
        arrays = {name: np.asarray(state[name]) for name in NCFORC_STATE_VARIABLES}
        ids = _integer_vector(arrays["ids"], f"{state_name}.ids")
        node_count = int(ids.size)
        if node_count == 0 or np.unique(ids).size != node_count:
            raise ValueError(f"{state_name}.ids must be nonempty and unique")
        for name, array in arrays.items():
            if array.ndim != 1 or array.size != node_count:
                raise ValueError(f"{state_name}.{name} has inconsistent vector length")
            if name != "ids" and (
                not np.issubdtype(array.dtype, np.number)
                or not np.all(np.isfinite(array))
            ):
                raise ValueError(f"{state_name}.{name} contains a non-finite value")
        id_set = set(int(value) for value in ids)
        union_ids.update(id_set)
        intersection_ids = id_set if intersection_ids is None else intersection_ids & id_set
        for name in ranges:
            values = np.asarray(arrays[name], dtype=float)
            ranges[name][0] = min(ranges[name][0], float(np.min(values)))
            ranges[name][1] = max(ranges[name][1], float(np.max(values)))
        records.append(
            {
                "state_index": state_index,
                "state_name": state_name,
                "time_ms": time_ms,
                "node_count": node_count,
                "node_ids": [int(value) for value in ids],
            }
        )
    counts = np.asarray([record["node_count"] for record in records], dtype=np.int64)
    return {
        "status": "available_ragged_pending_semantic_validation",
        "state_count": len(records),
        "first_time_ms": records[0]["time_ms"],
        "last_time_ms": records[-1]["time_ms"],
        "minimum_nodes_per_state": int(np.min(counts)),
        "maximum_nodes_per_state": int(np.max(counts)),
        "union_node_count": len(union_ids),
        "all_state_intersection_node_count": len(intersection_ids or set()),
        "node_count_change_count": int(np.count_nonzero(np.diff(counts))),
        "all_ids_unique_per_state": True,
        "all_coordinates_forces_pressure_finite": True,
        "raw_model_unit_ranges": ranges,
        "states": records,
        "claim_boundary": (
            "Units, sign convention, contact-side mapping, impulse and energy closure "
            "are not yet validated."
        ),
    }


def load_binout_inventory(return_dir: Path) -> dict[str, list[str]]:
    """Enumerate logical BINOUT databases and their exposed children."""

    binout = _load_lasso_binout()(str((return_dir / "binout*").resolve()))
    return {
        str(directory): [str(value) for value in binout.read(directory)]
        for directory in binout.read()
    }


def read_ragged_ncforc(return_dir: Path) -> dict | None:
    """Read NCFORC directly from LSDA symbols to avoid rectangular coercion."""

    binout = _load_lasso_binout()(str((return_dir / "binout*").resolve()))
    logical_names = {str(name).lower() for name in binout.read()}
    if "ncforc" not in logical_names:
        return None
    surfaces = [str(name) for name in binout.read("ncforc")]
    if surfaces != ["surfa_1"]:
        raise ValueError(f"expected exactly ncforc/surfa_1, found {surfaces}")
    surface = binout.lsda_root.get("ncforc").get("surfa_1")
    raw_states: list[dict] = []
    for state_name, state_symbol in surface.children.items():
        if state_name == "metadata":
            continue
        time_symbol = state_symbol.get(b"time")
        if time_symbol is None:
            raise ValueError(f"NCFORC state {state_name} has no time")
        time_values = time_symbol.read()
        if len(time_values) != 1:
            raise ValueError(f"NCFORC state {state_name} has a non-scalar time")
        state = {
            "state_name": state_name,
            "time_ms": time_values[0],
        }
        for variable in NCFORC_STATE_VARIABLES:
            symbol = state_symbol.get(variable.encode())
            if symbol is None:
                raise ValueError(f"NCFORC state {state_name} has no {variable}")
            state[variable] = np.asarray(symbol.read())
        raw_states.append(state)
    raw_states.sort(key=lambda item: float(item["time_ms"]))
    result = summarize_ragged_ncforc_states(raw_states)
    result["path"] = "ncforc/surfa_1"
    return result


def assess_r06_eroding_contact_outputs(
    *,
    message_sources: Mapping[str, str],
    returned_names: Sequence[str],
    binout_inventory: Mapping[str, Sequence[str]],
    intfor_requested: bool | None = None,
) -> dict:
    """Report capability evidence separately from actual returned files."""

    basenames = [Path(name).name.lower() for name in returned_names]
    intfor_names = sorted(name for name in basenames if name.startswith("intfor"))
    ncforc_names = sorted(name for name in basenames if name.startswith("ncforc"))
    warning_evidence: list[dict] = []
    occurrence_count = 0
    for source, source_text in message_sources.items():
        matches = list(WARNING_20340.finditer(source_text))
        if matches:
            occurrence_count += len(matches)
            warning_evidence.append(
                {
                    "source": source,
                    "occurrences": len(matches),
                    "reason_code": "LS_DYNA_WARNING_20340_STR_340",
                }
            )
    if intfor_names:
        intfor = {
            "status": "present_pending_reader_validation",
            "value": None,
            "reason": "INTFOR family exists but content has not been validated",
            "files": intfor_names,
        }
    elif warning_evidence:
        intfor = {
            "status": "unavailable_by_solver_capability",
            "value": None,
            "reason": "LS-DYNA Warning 20340/STR+340 declares binary INTFOR unavailable for the eroding contact",
            "files": [],
        }
    elif intfor_requested is False:
        intfor = {
            "status": "not_requested_under_contract",
            "value": None,
            "reason": "the R06f measurement contract does not request unsupported binary INTFOR",
            "files": [],
        }
    else:
        intfor = {
            "status": "missing_unexplained",
            "value": None,
            "reason": "no INTFOR family and no solver capability warning were found",
            "files": [],
        }

    logical_names = {str(name).lower() for name in binout_inventory}
    ncforc_present = bool(ncforc_names) or "ncforc" in logical_names
    ncforc = {
        "status": (
            "present_pending_reader_validation"
            if ncforc_present
            else "unavailable_not_returned_best_effort"
        ),
        "value": None,
        "reason": (
            "NCFORC artifact exists but content has not been validated"
            if ncforc_present
            else "best-effort NCFORC was not returned; absence is nonblocking and not zero"
        ),
        "files": ncforc_names,
    }
    sleout_present = any(name.startswith("sleout") for name in basenames) or "sleout" in logical_names
    return {
        "intfor": intfor,
        "ncforc": ncforc,
        "sleout": {
            "status": "available_energy_only" if sleout_present else "missing",
            "direct_contact_substitute": False,
        },
        "warning_20340": {
            "occurrence_count": occurrence_count,
            "evidence": warning_evidence,
        },
        "missing_is_zero": False,
    }


def _read_case_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError("R06 single-job case map must contain exactly one row")
    return rows[0]


def _message_sources(return_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(return_dir.iterdir()):
        lower = path.name.lower()
        if path.is_file() and (
            lower.startswith("mes")
            or lower.startswith("std.out")
            or lower == "d3hsp"
            or lower.endswith(".out")
        ):
            result[path.name] = path.read_text(encoding="utf-8", errors="replace")
    return result


def analyze_r06_single_job(
    *,
    package_dir: Path,
    return_dir: Path,
    output_path: Path,
    platform_job_id: str,
    solver_build: str,
) -> dict:
    package_dir = package_dir.resolve()
    return_dir = return_dir.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis: {output_path}")
    if output_path.is_relative_to(return_dir):
        raise ValueError("analysis output must be outside the raw return directory")

    manifest = _strict_json(package_dir / "manifest.json")
    design = _strict_json(package_dir / manifest["design"]["path"])
    contract = _strict_json(package_dir / manifest["successor_contract"]["path"])
    job_contract = _strict_json(package_dir / manifest["job_contract"]["path"])
    material = _strict_json(package_dir / manifest["material"]["path"])
    case_row = _read_case_row(package_dir / manifest["case_map"]["path"])
    mapped_design, mapped_rows = normalize_r06_single_job_capture_inputs(
        job_contract=job_contract,
        design=design,
        measurement_contract=contract,
        case_row=case_row,
    )

    reader = _load_lasso_d3plot()
    d3part = reader(
        str((return_dir / "d3part").resolve()),
        state_array_filter=[
            ArrayType.global_timesteps,
            ArrayType.node_displacement,
            ArrayType.node_velocity,
            ArrayType.element_solid_is_alive,
        ],
        buffered_reading=True,
    )
    timeline = load_solver_timeline_from_binout(return_dir / "binout*")
    capture = extract_case_ledgers_from_arrays(
        arrays=d3part.arrays,
        header=d3part.header,
        design=mapped_design,
        case_rows=mapped_rows,
        material=material,
        **timeline,
    )
    d3plot = reader(
        str((return_dir / "d3plot").resolve()),
        state_array_filter=[
            ArrayType.global_timesteps,
            ArrayType.element_solid_history_variables,
            ArrayType.element_solid_is_alive,
        ],
        buffered_reading=True,
    )
    damage = extract_mat015_hvar6_from_arrays(
        arrays=d3plot.arrays,
        header=d3plot.header,
        case=job_contract["case"],
    )
    returned_names = [
        str(path.relative_to(return_dir))
        for path in return_dir.rglob("*")
        if path.is_file()
    ]
    binout_inventory = load_binout_inventory(return_dir)
    contact = assess_r06_eroding_contact_outputs(
        message_sources=_message_sources(return_dir),
        returned_names=returned_names,
        binout_inventory=binout_inventory,
        intfor_requested=bool(
            contract["measurement_sources"]["direct_contact"]["intfor"][
                "request_in_r06f"
            ]
        ),
    )
    contact["binout_inventory"] = binout_inventory
    ncforc_validation = read_ragged_ncforc(return_dir)
    if ncforc_validation is not None:
        contact["ncforc"] = {
            "status": ncforc_validation["status"],
            "value": None,
            "reason": (
                "ragged NCFORC is readable with per-state node identities; physical "
                "semantics remain pending"
            ),
            "files": contact["ncforc"]["files"],
            "validation": ncforc_validation,
        }
    payload = {
        "schema_version": "hf-p0-r06-single-job-analysis@1.0.0",
        "package_revision": design["package_revision"],
        "job_id": job_contract["job_id"],
        "platform_job_id": str(platform_job_id),
        "solver_build": solver_build,
        "status": "diagnostic_only_pending_independent_review",
        "capture": capture,
        "mat015_hvar6": damage,
        "direct_contact_outputs": contact,
        "admission": {
            "p0_response_calibration": "not_assessed_by_adapter",
            "p1_submission": "denied",
            "engine_fitting": "denied",
            "reference_dataset_generation": "denied",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(strict_json_dumps(payload), encoding="utf-8", newline="\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one returned R06f solver job")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--return-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform-job-id", required=True)
    parser.add_argument("--solver-build", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze_r06_single_job(
        package_dir=args.package_dir,
        return_dir=args.return_dir,
        output_path=args.output,
        platform_job_id=args.platform_job_id,
        solver_build=args.solver_build,
    )
    print(
        strict_json_dumps(
            {
                "output": str(args.output.resolve()),
                "job_id": payload["job_id"],
                "status": payload["status"],
            }
        ),
        end="",
    )


if __name__ == "__main__":
    main()
