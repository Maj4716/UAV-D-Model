"""Reproducible, diagnostic-only failure-mode analysis for the frozen R05 return.

The solver return is treated as immutable evidence.  This module distinguishes
explicit negative-volume messages from otherwise-unspecified ``solid element
... failed`` messages, deduplicates rank observations, binds element IDs to the
frozen case map, and relates fragment failures to the D3PART capture ledger.
For the two preregistered focus cases it also reads the complete D3PLOT family
to report alive-element effective plastic strain and derived solid hydrostatic
pressure histories, while keeping direct contact quantities fail-closed when
the corresponding output flags/databases are absent.

At Stage B kickoff, the original frozen extractor observed two software
blockers on the real R05 family.  This module records that historical result; it
does not assess the concurrent adapter repair.  Any ledger generated here is
diagnostic-only and must never be bridged to the response assessor or presented
as numerical/physical admission, including retroactively after a software fix.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from extract_r05_d3part import (
    ArrayType,
    VENDOR_DIR,
    capture_thresholds_from_design,
    d3part_family_identity,
    hexahedron_reference_volumes,
    resolve_node_state_coordinates,
    sha256_file,
)
from parse_lsdyna_ascii import NUMBER, load_cases, number
from r05_capture_ledger import build_capture_ledger


SCHEMA_VERSION = "hf-p0-r05-failure-mode-analysis@1.1.0"
SCRIPT_VERSION = "1.1.0"
NEGATIVE_VOLUME = re.compile(
    rf"negative volume in solid element #\s*(\d+)\s+cycle\s*(\d+)\s+time\s*({NUMBER})",
    flags=re.IGNORECASE,
)
OTHER_SOLID_FAILURE = re.compile(
    rf"solid element\s+(\d+)\s+failed at time\s+({NUMBER})",
    flags=re.IGNORECASE,
)
LEDGER_STATES = (
    "alive_crossed",
    "alive_upstream",
    "deleted_upstream",
    "deleted_after_crossing",
    "temporal_ambiguity",
)
OUTPUT_NAMES = {
    "mes_failure_observations.csv",
    "failure_events_unique.csv",
    "failure_summary_by_case.csv",
    "failure_summary_by_case_part_type.csv",
    "failure_time_bins_0p001ms.csv",
    "d3part_element_ledger_diagnostic.csv",
    "d3part_capture_ledger_diagnostic.json",
    "d3plot_focus_part_history.csv",
    "d3plot_first_failure_element_history.csv",
    "d3plot_focus_response_summary.json",
    "failure_mode_analysis.json",
    "failure_mode_report_zh.md",
    "stage_b_worklog.md",
    "stage_b_analysis_manifest.json",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _alive_population_statistics(
    values: np.ndarray, *, prefix: str, quantiles: Sequence[tuple[str, float]]
) -> dict[str, float | None]:
    selected = np.asarray(values, dtype=float).reshape(-1)
    keys = [f"{prefix}_mean", *(f"{prefix}_{name}" for name, _ in quantiles)]
    if selected.size == 0:
        return {key: None for key in keys}
    if not np.all(np.isfinite(selected)):
        raise ValueError(f"{prefix} contains non-finite values among alive elements")
    result: dict[str, float | None] = {f"{prefix}_mean": float(np.mean(selected))}
    for name, probability in quantiles:
        result[f"{prefix}_{name}"] = float(
            np.quantile(selected, probability, method="linear")
        )
    return result


def extract_focus_solid_history(
    *,
    times_ms: np.ndarray,
    element_stress_gpa: np.ndarray,
    element_effective_plastic_strain: np.ndarray,
    element_alive: np.ndarray,
    element_ids: np.ndarray,
    element_part_indexes: np.ndarray,
    part_ids: np.ndarray,
    focus_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Extract diagnostic solid histories for frozen focus cases.

    The returned statistics are unweighted across *alive* solid elements.  The
    R05 D3PLOT stores one solid result layer/integration record per element;
    refusing any other layer count avoids silently inventing a reduction rule.
    Hydrostatic pressure is derived from the three normal Cauchy stresses as
    ``p=-(sxx+syy+szz)/3`` and is not a contact-pressure measurement.
    """
    times = np.asarray(times_ms, dtype=float)
    stress = np.asarray(element_stress_gpa, dtype=float)
    plastic = np.asarray(element_effective_plastic_strain, dtype=float)
    alive = np.asarray(element_alive) != 0
    ids = np.asarray(element_ids, dtype=np.int64)
    part_index = np.asarray(element_part_indexes, dtype=np.int64)
    catalog_part_ids = np.asarray(part_ids, dtype=np.int64)

    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("D3PLOT timesteps must be a non-empty finite vector")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("D3PLOT timesteps must be strictly increasing")
    if stress.ndim != 4 or stress.shape[-1] != 6:
        raise ValueError("solid stress must have shape (state, element, layer, 6)")
    if plastic.ndim != 3:
        raise ValueError(
            "effective plastic strain must have shape (state, element, layer)"
        )
    if stress.shape[:3] != plastic.shape or stress.shape[0] != times.size:
        raise ValueError("D3PLOT solid stress/plastic-strain shapes disagree")
    if stress.shape[2] != 1:
        raise ValueError("R05 focus history requires exactly one solid result layer")
    n_elements = stress.shape[1]
    if alive.shape != (times.size, n_elements):
        raise ValueError("D3PLOT alive-state shape disagrees with solid state arrays")
    if ids.shape != (n_elements,) or np.unique(ids).size != ids.size:
        raise ValueError("D3PLOT solid element IDs must be unique")
    if part_index.shape != (n_elements,):
        raise ValueError("D3PLOT solid part-index shape is invalid")
    if (
        np.any(part_index < 0)
        or np.any(part_index >= catalog_part_ids.size)
        or catalog_part_ids.size == 0
    ):
        raise ValueError("D3PLOT solid part index is outside the part catalog")
    used_part_ids = catalog_part_ids[part_index]
    pressure = -(stress[:, :, 0, 0] + stress[:, :, 0, 1] + stress[:, :, 0, 2]) / 3.0
    pressure[pressure == 0.0] = 0.0
    plastic = plastic[:, :, 0]

    part_history_rows: list[dict[str, Any]] = []
    first_failure_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for case in focus_cases:
        case_id = str(case["case_id"])
        part_specs = (
            ("plate", int(case["plate_part_id"])),
            ("fragment", int(case["fragment_part_id"])),
        )
        case_part_summaries: list[dict[str, Any]] = []
        for part_kind, pid in part_specs:
            mask = used_part_ids == pid
            element_count = int(np.count_nonzero(mask))
            if element_count == 0:
                raise ValueError(f"{case_id} {part_kind} PID {pid} has no D3PLOT solids")
            if not np.all(alive[0, mask]):
                raise ValueError(f"{case_id} {part_kind} solids are not all initially alive")
            rows_for_part: list[dict[str, Any]] = []
            for state_index, time_ms in enumerate(times):
                state_mask = mask & alive[state_index]
                alive_count = int(np.count_nonzero(state_mask))
                row = {
                    "case_id": case_id,
                    "part_kind": part_kind,
                    "part_id": pid,
                    "d3plot_state_index": state_index,
                    "d3plot_time_ms": float(time_ms),
                    "total_element_count": element_count,
                    "alive_element_count": alive_count,
                    "deleted_element_count": element_count - alive_count,
                    "alive_element_fraction": alive_count / element_count,
                    **_alive_population_statistics(
                        plastic[state_index, state_mask],
                        prefix="effective_plastic_strain",
                        quantiles=(("p50", 0.50), ("p95", 0.95), ("p99", 0.99), ("max", 1.0)),
                    ),
                    **_alive_population_statistics(
                        pressure[state_index, state_mask],
                        prefix="derived_solid_hydrostatic_pressure_gpa",
                        quantiles=(
                            ("min", 0.0),
                            ("p05", 0.05),
                            ("p50", 0.50),
                            ("p95", 0.95),
                            ("max", 1.0),
                        ),
                    ),
                }
                rows_for_part.append(row)
                part_history_rows.append(row)

            eps_peak = max(
                (row for row in rows_for_part if row["effective_plastic_strain_max"] is not None),
                key=lambda row: float(row["effective_plastic_strain_max"]),
            )
            pressure_max = max(
                (
                    row
                    for row in rows_for_part
                    if row["derived_solid_hydrostatic_pressure_gpa_max"] is not None
                ),
                key=lambda row: float(row["derived_solid_hydrostatic_pressure_gpa_max"]),
            )
            pressure_min = min(
                (
                    row
                    for row in rows_for_part
                    if row["derived_solid_hydrostatic_pressure_gpa_min"] is not None
                ),
                key=lambda row: float(row["derived_solid_hydrostatic_pressure_gpa_min"]),
            )
            case_part_summaries.append(
                {
                    "part_kind": part_kind,
                    "part_id": pid,
                    "total_element_count": element_count,
                    "initial_alive_element_count": rows_for_part[0]["alive_element_count"],
                    "final_alive_element_count": rows_for_part[-1]["alive_element_count"],
                    "minimum_alive_element_count": min(
                        int(row["alive_element_count"]) for row in rows_for_part
                    ),
                    "maximum_effective_plastic_strain_among_alive": eps_peak[
                        "effective_plastic_strain_max"
                    ],
                    "maximum_effective_plastic_strain_time_ms": eps_peak[
                        "d3plot_time_ms"
                    ],
                    "maximum_derived_solid_hydrostatic_pressure_gpa_among_alive": pressure_max[
                        "derived_solid_hydrostatic_pressure_gpa_max"
                    ],
                    "maximum_derived_solid_hydrostatic_pressure_time_ms": pressure_max[
                        "d3plot_time_ms"
                    ],
                    "minimum_derived_solid_hydrostatic_pressure_gpa_among_alive": pressure_min[
                        "derived_solid_hydrostatic_pressure_gpa_min"
                    ],
                    "minimum_derived_solid_hydrostatic_pressure_time_ms": pressure_min[
                        "d3plot_time_ms"
                    ],
                }
            )

        failure_element_id = int(case["first_failure_element_id"])
        element_matches = np.flatnonzero(ids == failure_element_id)
        if element_matches.size != 1:
            raise ValueError(
                f"{case_id} first-failure EID {failure_element_id} is not unique in D3PLOT"
            )
        element_index = int(element_matches[0])
        failure_kind = str(case["first_failure_part_kind"])
        expected_pid = int(case[f"{failure_kind}_part_id"])
        actual_pid = int(used_part_ids[element_index])
        if actual_pid != expected_pid:
            raise ValueError(
                f"{case_id} first-failure EID {failure_element_id} PID {actual_pid} "
                f"does not match {failure_kind} PID {expected_pid}"
            )
        alive_series = alive[:, element_index]
        if not alive_series[0] or np.any(np.diff(alive_series.astype(np.int8)) > 0):
            raise ValueError(f"{case_id} first-failure element alive history is invalid")
        alive_indexes = np.flatnonzero(alive_series)
        if alive_indexes.size == 0:
            raise ValueError(f"{case_id} first-failure element has no alive D3PLOT state")
        failure_time = float(case["first_failure_time_ms"])
        element_rows: list[dict[str, Any]] = []
        for state_index in alive_indexes:
            normal = stress[state_index, element_index, 0, :3]
            if not np.all(np.isfinite(normal)) or not math.isfinite(
                float(plastic[state_index, element_index])
            ):
                raise ValueError(
                    f"{case_id} first-failure element has non-finite alive-state history"
                )
            row = {
                "case_id": case_id,
                "element_id": failure_element_id,
                "part_kind": failure_kind,
                "part_id": actual_pid,
                "failure_type": str(case["first_failure_type"]),
                "mes_first_failure_time_ms": failure_time,
                "d3plot_state_index": int(state_index),
                "d3plot_time_ms": float(times[state_index]),
                "state_relation_to_mes_failure": (
                    "before_mes_failure"
                    if float(times[state_index]) < failure_time
                    else "at_or_after_mes_failure_while_alive"
                ),
                "stress_xx_gpa": float(normal[0]),
                "stress_yy_gpa": float(normal[1]),
                "stress_zz_gpa": float(normal[2]),
                "derived_solid_hydrostatic_pressure_gpa": float(
                    pressure[state_index, element_index]
                ),
                "effective_plastic_strain": float(plastic[state_index, element_index]),
            }
            element_rows.append(row)
            first_failure_rows.append(row)
        dead_indexes = np.flatnonzero(~alive_series)
        first_deleted_index = int(dead_indexes[0]) if dead_indexes.size else None
        case_summaries.append(
            {
                "case_id": case_id,
                "parts": case_part_summaries,
                "first_failure_element": {
                    "element_id": failure_element_id,
                    "part_kind": failure_kind,
                    "part_id": actual_pid,
                    "failure_type": str(case["first_failure_type"]),
                    "mes_first_failure_time_ms": failure_time,
                    "alive_history_state_count": len(element_rows),
                    "last_alive_d3plot_time_ms": element_rows[-1]["d3plot_time_ms"],
                    "first_deleted_d3plot_time_ms": (
                        float(times[first_deleted_index])
                        if first_deleted_index is not None
                        else None
                    ),
                    "maximum_effective_plastic_strain_while_alive": max(
                        float(row["effective_plastic_strain"]) for row in element_rows
                    ),
                    "maximum_derived_solid_hydrostatic_pressure_gpa_while_alive": max(
                        float(row["derived_solid_hydrostatic_pressure_gpa"])
                        for row in element_rows
                    ),
                    "minimum_derived_solid_hydrostatic_pressure_gpa_while_alive": min(
                        float(row["derived_solid_hydrostatic_pressure_gpa"])
                        for row in element_rows
                    ),
                    "interpretation": (
                        "saved-state history while the element is alive; MES failure time and "
                        "the following saved deletion state do not establish a material mechanism"
                    ),
                },
            }
        )

    return {
        "state_count": int(times.size),
        "first_time_ms": float(times[0]),
        "last_time_ms": float(times[-1]),
        "solid_result_layer_count": int(stress.shape[2]),
        "statistics_population": "unweighted_alive_solid_elements",
        "effective_plastic_strain_definition": "D3PLOT element_solid_effective_plastic_strain",
        "derived_solid_hydrostatic_pressure": {
            "formula": "p=-(sxx+syy+szz)/3",
            "unit": "GPa",
            "source": "D3PLOT element_solid_stress normal components",
            "is_contact_pressure": False,
        },
        "cases": case_summaries,
        "part_history_rows": part_history_rows,
        "first_failure_element_history_rows": first_failure_rows,
    }


def assess_contact_output_availability(
    *, header: Any, d3plot_array_keys: Iterable[str], binout_inventory: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    """Report whether direct contact penetration/pressure histories were written."""
    array_keys = {str(key).lower() for key in d3plot_array_keys}
    logical = {
        str(directory).lower(): {str(variable).lower() for variable in variables}
        for directory, variables in binout_inventory.items()
    }
    penetration_flags = {
        "absolute": bool(
            getattr(header, "has_node_max_contact_penetration_absolute", False)
        ),
        "relative": bool(
            getattr(header, "has_node_max_contact_penetration_relative", False)
        ),
    }
    contact_energy_density_flag = bool(
        getattr(header, "has_node_contact_energy_density", False)
    )
    flattened_variables = {
        f"{directory}/{variable}"
        for directory, variables in logical.items()
        for variable in variables
    }
    penetration_written = (
        any(penetration_flags.values())
        or any("penetr" in key for key in array_keys)
        or any("penetr" in name for name in flattened_variables)
    )
    contact_pressure_written = any("contact_pressure" in key for key in array_keys) or any(
        "pressure" in name
        for name in flattened_variables
        if name.split("/", 1)[0] in {"intfor", "ncforc", "rcforc", "contact"}
    )
    return {
        "d3plot_header_flags": {
            "has_node_max_contact_penetration_absolute": penetration_flags["absolute"],
            "has_node_max_contact_penetration_relative": penetration_flags["relative"],
            "has_node_contact_energy_density": contact_energy_density_flag,
        },
        "binout_logical_directories": sorted(logical),
        "sleout_variables": sorted(logical.get("sleout", set())),
        "contact_penetration": {
            "status": (
                "written_but_extraction_not_implemented"
                if penetration_written
                else "unavailable_not_written"
            ),
            "reason": (
                "D3PLOT absolute/relative penetration flags are false and no BINOUT "
                "penetration variable is present"
                if not penetration_written
                else "contact penetration evidence exists but is not extracted here"
            ),
        },
        "contact_pressure": {
            "status": (
                "written_but_extraction_not_implemented"
                if contact_pressure_written
                else "unavailable_not_written"
            ),
            "reason": (
                "no D3PLOT contact-pressure array or contact-force/interface BINOUT "
                "pressure variable is present"
                if not contact_pressure_written
                else "contact pressure evidence exists but is not extracted here"
            ),
        },
        "derived_solid_hydrostatic_pressure": {
            "available": ArrayType.element_solid_stress.lower() in array_keys,
            "formula": "p=-(sxx+syy+szz)/3",
            "unit": "GPa",
            "is_contact_pressure": False,
        },
        "claim_boundary": (
            "SLEOUT energy channels and derived solid hydrostatic pressure do not replace "
            "direct contact penetration or contact-pressure histories."
        ),
    }


def _rank_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and re.fullmatch(r"mes\d{4}", path.name)
    )


def parse_mes_failure_events(directory: Path) -> dict[str, Any]:
    """Parse and deduplicate failure observations from all MPP rank messages.

    Deduplication is by failure type and global element ID.  The earliest time
    is retained as the event time while every source file/line remains listed.
    This prevents repeated rank observations from inflating element counts.
    """
    observations: list[dict[str, Any]] = []
    paths = _rank_paths(directory)
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), 1):
            negative = NEGATIVE_VOLUME.search(line)
            if negative:
                observations.append(
                    {
                        "failure_type": "negative_volume",
                        "element_id": int(negative.group(1)),
                        "cycle": int(negative.group(2)),
                        "time_ms": number(negative.group(3)),
                        "source_file": path.name,
                        "source_line": line_number,
                    }
                )
                continue
            other = OTHER_SOLID_FAILURE.search(line)
            if other:
                observations.append(
                    {
                        "failure_type": "other_solid_failure",
                        "element_id": int(other.group(1)),
                        "cycle": None,
                        "time_ms": number(other.group(2)),
                        "source_file": path.name,
                        "source_line": line_number,
                    }
                )

    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        groups[(row["failure_type"], row["element_id"])].append(row)

    unique: list[dict[str, Any]] = []
    for (failure_type, element_id), rows in groups.items():
        ordered = sorted(
            rows,
            key=lambda row: (
                row["time_ms"],
                row["source_file"],
                row["source_line"],
            ),
        )
        first = ordered[0]
        unique.append(
            {
                "failure_type": failure_type,
                "element_id": element_id,
                "cycle": first["cycle"],
                "time_ms": first["time_ms"],
                "last_observed_time_ms": ordered[-1]["time_ms"],
                "observation_count": len(ordered),
                "source_refs": ";".join(
                    f"{row['source_file']}:{row['source_line']}" for row in ordered
                ),
            }
        )
    unique.sort(key=lambda row: (row["time_ms"], row["failure_type"], row["element_id"]))
    overlap = set(
        row["element_id"] for row in unique if row["failure_type"] == "negative_volume"
    ) & set(
        row["element_id"]
        for row in unique
        if row["failure_type"] == "other_solid_failure"
    )
    return {
        "rank_file_count": len(paths),
        "rank_files": [path.name for path in paths],
        "raw_observation_count": len(observations),
        "raw_observations": observations,
        "unique_event_count": len(unique),
        "unique_element_count": len({row["element_id"] for row in unique}),
        "cross_type_element_overlap_count": len(overlap),
        "cross_type_element_overlap_ids": sorted(overlap),
        "unique_events": unique,
    }


def bind_events_to_cases(
    events: Sequence[Mapping[str, Any]], deck_path: Path, case_map_path: Path
) -> list[dict[str, Any]]:
    geometry = map_elements_to_initial_geometry(
        deck_path, {int(row["element_id"]) for row in events}
    )
    cases = load_cases(case_map_path)
    part_context: dict[int, tuple[str, str, str]] = {}
    for case in cases:
        part_context[int(case["plate_part_id"])] = (
            case["case_id"],
            case["role"],
            "plate",
        )
        part_context[int(case["fragment_part_id"])] = (
            case["case_id"],
            case["role"],
            "fragment",
        )

    bound: list[dict[str, Any]] = []
    for event in events:
        element_geometry = geometry.get(int(event["element_id"]))
        part_id = element_geometry["part_id"] if element_geometry is not None else None
        case_id, role, kind = part_context.get(part_id, (None, None, "unmapped"))
        bound.append(
            {
                **dict(event),
                "part_id": part_id,
                "case_id": case_id,
                "case_role": role,
                "part_kind": kind,
                "initial_deck_centroid_x_mm": (
                    element_geometry["initial_centroid_x_mm"]
                    if element_geometry is not None
                    else None
                ),
                "initial_deck_centroid_y_mm": (
                    element_geometry["initial_centroid_y_mm"]
                    if element_geometry is not None
                    else None
                ),
                "initial_deck_centroid_z_mm": (
                    element_geometry["initial_centroid_z_mm"]
                    if element_geometry is not None
                    else None
                ),
                "initial_position_semantics": (
                    "deck_reference_geometry_not_failure_time_position"
                    if element_geometry is not None
                    else None
                ),
            }
        )
    return bound


def map_elements_to_initial_geometry(
    deck_path: Path, element_ids: set[int]
) -> dict[int, dict[str, Any]]:
    """Read target solid connectivity and calculate reference centroids from the deck."""
    elements: dict[int, tuple[int, tuple[int, ...]]] = {}
    in_solids = False
    with deck_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            upper = stripped.upper()
            if upper == "*ELEMENT_SOLID":
                in_solids = True
                continue
            if stripped.startswith("*"):
                in_solids = False
                continue
            if not in_solids or not stripped or stripped.startswith("$"):
                continue
            fields = [field.strip() for field in stripped.split(",")]
            if len(fields) < 10:
                continue
            element_id = int(fields[0])
            if element_id in element_ids:
                elements[element_id] = (
                    int(fields[1]),
                    tuple(int(value) for value in fields[2:10]),
                )
    node_ids = {node_id for _, nodes in elements.values() for node_id in nodes}
    nodes: dict[int, tuple[float, float, float]] = {}
    in_nodes = False
    with deck_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            upper = stripped.upper()
            if upper == "*NODE":
                in_nodes = True
                continue
            if stripped.startswith("*"):
                in_nodes = False
                continue
            if not in_nodes or not stripped or stripped.startswith("$"):
                continue
            fields = [field.strip() for field in stripped.split(",")]
            if len(fields) < 4:
                continue
            node_id = int(fields[0])
            if node_id in node_ids:
                nodes[node_id] = (
                    number(fields[1]),
                    number(fields[2]),
                    number(fields[3]),
                )
    result: dict[int, dict[str, Any]] = {}
    for element_id, (part_id, connectivity) in elements.items():
        missing = [node_id for node_id in connectivity if node_id not in nodes]
        if missing:
            raise ValueError(
                f"deck nodes missing for target element {element_id}: {missing}"
            )
        coordinates = np.asarray([nodes[node_id] for node_id in connectivity], dtype=float)
        centroid = coordinates.mean(axis=0)
        result[element_id] = {
            "part_id": part_id,
            "node_ids": list(connectivity),
            "initial_centroid_x_mm": float(centroid[0]),
            "initial_centroid_y_mm": float(centroid[1]),
            "initial_centroid_z_mm": float(centroid[2]),
        }
    return result


def classify_fragment_elements(
    *,
    times_ms: np.ndarray,
    node_positions_mm: np.ndarray,
    node_velocities_mm_per_ms: np.ndarray,
    element_connectivity: np.ndarray,
    element_alive: np.ndarray,
    element_ids: np.ndarray,
    element_reference_mass_kg: np.ndarray,
    capture_plane_z_mm: float,
    maximum_unambiguous_step_mm: float,
) -> list[dict[str, Any]]:
    """Expose element-level states using the frozen capture-ledger semantics."""
    times = np.asarray(times_ms, dtype=float)
    positions = np.asarray(node_positions_mm, dtype=float)
    velocities = np.asarray(node_velocities_mm_per_ms, dtype=float)
    connectivity = np.asarray(element_connectivity, dtype=np.int64)
    alive = np.asarray(element_alive) != 0
    ids = np.asarray(element_ids, dtype=np.int64)
    masses = np.asarray(element_reference_mass_kg, dtype=float)
    capture_plane = float(capture_plane_z_mm)
    maximum_step = float(maximum_unambiguous_step_mm)

    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("times_ms must be strictly increasing")
    if positions.shape != velocities.shape or positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("node position/velocity arrays must have shape [state,node,3]")
    if positions.shape[0] != times.size or not np.all(np.isfinite(positions)):
        raise ValueError("node positions must be finite and match the state count")
    if not np.all(np.isfinite(velocities)):
        raise ValueError("node velocities must be finite")
    if connectivity.ndim != 2 or connectivity.shape[1] != 8:
        raise ValueError("element_connectivity must have shape [element,8]")
    count = connectivity.shape[0]
    if alive.shape != (times.size, count):
        raise ValueError("element_alive must have shape [state,element]")
    if ids.shape != (count,) or np.unique(ids).size != count:
        raise ValueError("element_ids must be a unique vector matching the elements")
    if masses.shape != (count,) or np.any(masses <= 0) or not np.all(np.isfinite(masses)):
        raise ValueError("element masses must be finite, positive, and match the elements")
    if connectivity.size and (connectivity.min() < 0 or connectivity.max() >= positions.shape[1]):
        raise ValueError("element connectivity references a missing node")
    if not math.isfinite(capture_plane) or not math.isfinite(maximum_step) or maximum_step <= 0:
        raise ValueError("capture plane and positive maximum step must be finite")

    element_positions = positions[:, connectivity, :].mean(axis=2)
    element_velocities = velocities[:, connectivity, :].mean(axis=2)
    result: list[dict[str, Any]] = []
    for element in range(count):
        state = ""
        crossed = False
        crossing_time = math.nan
        crossing_velocity = np.full(3, np.nan)
        deletion_index: int | None = None
        if not alive[0, element] or element_positions[0, element, 2] <= capture_plane:
            state = "temporal_ambiguity"
        else:
            for index in range(1, times.size):
                was_alive = bool(alive[index - 1, element])
                is_alive = bool(alive[index, element])
                previous_z = float(element_positions[index - 1, element, 2])
                current_z = float(element_positions[index, element, 2])
                if (
                    was_alive
                    and is_alive
                    and not crossed
                    and previous_z > capture_plane >= current_z
                ):
                    step = previous_z - current_z
                    if step <= 0.0 or step > maximum_step:
                        state = "temporal_ambiguity"
                        break
                    alpha = (previous_z - capture_plane) / step
                    interpolated = element_velocities[index - 1, element] + alpha * (
                        element_velocities[index, element]
                        - element_velocities[index - 1, element]
                    )
                    if interpolated[2] >= 0.0:
                        state = "temporal_ambiguity"
                        break
                    crossing_time = float(
                        times[index - 1] + alpha * (times[index] - times[index - 1])
                    )
                    crossing_velocity = interpolated
                    crossed = True
                if was_alive and not is_alive:
                    deletion_index = index
                    if crossed:
                        state = "deleted_after_crossing"
                    elif previous_z > capture_plane >= current_z:
                        state = "temporal_ambiguity"
                    else:
                        state = "deleted_upstream"
                    break
        if state == "":
            if crossed:
                state = "alive_crossed"
            elif alive[-1, element]:
                state = "alive_upstream"
            else:
                state = "deleted_upstream"
                deleted = np.flatnonzero(alive[:-1, element] & ~alive[1:, element])
                deletion_index = int(deleted[0] + 1) if deleted.size else None

        deletion_start = (
            float(times[deletion_index - 1]) if deletion_index is not None else None
        )
        deletion_end = float(times[deletion_index]) if deletion_index is not None else None
        last_alive_z = (
            float(element_positions[deletion_index - 1, element, 2])
            if deletion_index is not None
            else None
        )
        first_dead_z = (
            float(element_positions[deletion_index, element, 2])
            if deletion_index is not None
            else None
        )
        initial = element_positions[0, element]
        result.append(
            {
                "element_id": int(ids[element]),
                "ledger_state": state,
                "reference_mass_kg": float(masses[element]),
                "initial_centroid_x_mm": float(initial[0]),
                "initial_centroid_y_mm": float(initial[1]),
                "initial_centroid_z_mm": float(initial[2]),
                "crossing_time_ms": _finite_or_none(crossing_time),
                "crossing_velocity_x_mm_per_ms": _finite_or_none(crossing_velocity[0]),
                "crossing_velocity_y_mm_per_ms": _finite_or_none(crossing_velocity[1]),
                "crossing_velocity_z_mm_per_ms": _finite_or_none(crossing_velocity[2]),
                "d3part_deletion_interval_start_ms": deletion_start,
                "d3part_deletion_interval_end_ms": deletion_end,
                "last_alive_centroid_z_mm": last_alive_z,
                "first_dead_centroid_z_mm": first_dead_z,
                "alive_at_final_state": bool(alive[-1, element]),
            }
        )
    return result


def associate_failure_events_with_d3part(
    events: Sequence[Mapping[str, Any]], ledger_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    ledger_by_id = {int(row["element_id"]): dict(row) for row in ledger_rows}
    ledger_fields = (
        "ledger_state",
        "reference_mass_kg",
        "initial_centroid_x_mm",
        "initial_centroid_y_mm",
        "initial_centroid_z_mm",
        "crossing_time_ms",
        "d3part_deletion_interval_start_ms",
        "d3part_deletion_interval_end_ms",
        "last_alive_centroid_z_mm",
        "first_dead_centroid_z_mm",
    )
    result: list[dict[str, Any]] = []
    for source in events:
        event = dict(source)
        element_id = int(event["element_id"])
        ledger = ledger_by_id.get(element_id)
        if event.get("part_kind") == "plate":
            association = "not_in_fragment_only_d3part"
            ledger = None
        elif ledger is None:
            association = "fragment_element_missing_from_d3part"
        else:
            association = "matched_fragment_element"
        event["d3part_association"] = association
        for field in ledger_fields:
            event[field] = ledger.get(field) if ledger is not None else None
        if ledger is not None:
            deltas = [
                abs(
                    float(event[f"initial_deck_centroid_{axis}_mm"])
                    - float(ledger[f"initial_centroid_{axis}_mm"])
                )
                for axis in ("x", "y", "z")
            ]
            maximum_delta = max(deltas)
            event["fragment_initial_reference_centroid_max_abs_delta_mm"] = maximum_delta
            event["fragment_initial_reference_crosscheck_status"] = (
                "match" if maximum_delta <= 1e-6 else "mismatch"
            )
        else:
            event["fragment_initial_reference_centroid_max_abs_delta_mm"] = None
            event["fragment_initial_reference_crosscheck_status"] = (
                "not_applicable_fragment_only_d3part"
                if event.get("part_kind") == "plate"
                else "unavailable"
            )
        start = event.get("d3part_deletion_interval_start_ms")
        end = event.get("d3part_deletion_interval_end_ms")
        event["mes_time_within_d3part_deletion_interval"] = (
            bool(float(start) <= float(event["time_ms"]) <= float(end))
            if start is not None and end is not None
            else None
        )
        result.append(event)
    return result


def _count(
    events: Iterable[Mapping[str, Any]], *, kind: str, failure_type: str
) -> int:
    return sum(
        row.get("part_kind") == kind and row.get("failure_type") == failure_type
        for row in events
    )


def summarize_cases(
    events: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    ledgers: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        selected = [row for row in events if row.get("case_id") == case_id]
        selected.sort(key=lambda row: (row["time_ms"], row["failure_type"], row["element_id"]))
        first = selected[0] if selected else None
        ledger = ledgers.get(case_id)
        states = ledger.get("mass_fraction_by_ledger_state", {}) if ledger else {}
        fragment_spatial = [
            row
            for row in selected
            if row.get("part_kind") == "fragment"
            and row.get("d3part_association") == "matched_fragment_element"
        ]

        def bounds(field: str) -> tuple[float | None, float | None]:
            values = [float(row[field]) for row in fragment_spatial if row.get(field) is not None]
            return (min(values), max(values)) if values else (None, None)

        x_min, x_max = bounds("initial_centroid_x_mm")
        y_min, y_max = bounds("initial_centroid_y_mm")
        z_min, z_max = bounds("initial_centroid_z_mm")
        last_z_min, last_z_max = bounds("last_alive_centroid_z_mm")

        def deck_bounds(kind: str, axis: str) -> tuple[float | None, float | None]:
            field = f"initial_deck_centroid_{axis}_mm"
            values = [
                float(row[field])
                for row in selected
                if row.get("part_kind") == kind and row.get(field) is not None
            ]
            return (min(values), max(values)) if values else (None, None)

        plate_deck_bounds = {
            axis: deck_bounds("plate", axis) for axis in ("x", "y", "z")
        }
        fragment_deck_bounds = {
            axis: deck_bounds("fragment", axis) for axis in ("x", "y", "z")
        }
        deleted_upstream = states.get("deleted_upstream") if ledger else None
        summary = {
            "case_id": case_id,
            "role": case["role"],
            "impact_speed_mps": float(case["impact_speed_mps"]),
            "mesh_mm": float(case["mesh_mm"]),
            "plate_part_id": int(case["plate_part_id"]),
            "fragment_part_id": int(case["fragment_part_id"]),
            "unique_failure_element_count": len({row["element_id"] for row in selected}),
            "failure_event_count": len(selected),
            "first_failure_time_ms": first["time_ms"] if first else None,
            "first_failure_type": first["failure_type"] if first else None,
            "first_failure_part_kind": first["part_kind"] if first else None,
            "first_failure_element_id": first["element_id"] if first else None,
            "last_failure_time_ms": selected[-1]["time_ms"] if selected else None,
            "plate_negative_volume_count": _count(
                selected, kind="plate", failure_type="negative_volume"
            ),
            "fragment_negative_volume_count": _count(
                selected, kind="fragment", failure_type="negative_volume"
            ),
            "plate_other_failure_count": _count(
                selected, kind="plate", failure_type="other_solid_failure"
            ),
            "fragment_other_failure_count": _count(
                selected, kind="fragment", failure_type="other_solid_failure"
            ),
            "fragment_mes_time_in_d3part_deletion_interval_count": sum(
                row.get("part_kind") == "fragment"
                and row.get("mes_time_within_d3part_deletion_interval") is True
                for row in selected
            ),
            "fragment_mes_time_outside_d3part_deletion_interval_count": sum(
                row.get("part_kind") == "fragment"
                and row.get("mes_time_within_d3part_deletion_interval") is False
                for row in selected
            ),
            "fragment_event_ledger_state_counts": {
                state: sum(row.get("ledger_state") == state for row in fragment_spatial)
                for state in LEDGER_STATES
            },
            "deleted_upstream_mass_fraction": deleted_upstream,
            "primary_qoi_available": ledger.get("primary_qoi_available") if ledger else None,
            "qualification_use_forbidden": True,
            "capture_ledger_classification": ledger.get("classification") if ledger else None,
            "capture_ledger_unavailable_reasons": (
                ledger.get("unavailable_reasons", []) if ledger else ["ledger_unavailable"]
            ),
            "fragment_failure_initial_x_min_mm": x_min,
            "fragment_failure_initial_x_max_mm": x_max,
            "fragment_failure_initial_y_min_mm": y_min,
            "fragment_failure_initial_y_max_mm": y_max,
            "fragment_failure_initial_z_min_mm": z_min,
            "fragment_failure_initial_z_max_mm": z_max,
            "fragment_failure_last_alive_z_min_mm": last_z_min,
            "fragment_failure_last_alive_z_max_mm": last_z_max,
            **{
                f"plate_failure_initial_deck_{axis}_min_mm": values[0]
                for axis, values in plate_deck_bounds.items()
            },
            **{
                f"plate_failure_initial_deck_{axis}_max_mm": values[1]
                for axis, values in plate_deck_bounds.items()
            },
            **{
                f"fragment_failure_initial_deck_{axis}_min_mm": values[0]
                for axis, values in fragment_deck_bounds.items()
            },
            **{
                f"fragment_failure_initial_deck_{axis}_max_mm": values[1]
                for axis, values in fragment_deck_bounds.items()
            },
            "initial_deck_position_semantics": (
                "reference_geometry_only_not_failure_time_position"
            ),
            "plate_failure_deformed_position_status": (
                "unavailable_fragment_only_d3part"
            ),
        }
        summaries.append(summary)
    return summaries


def build_group_summary(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in events:
        groups[(str(row["case_id"]), str(row["part_kind"]), str(row["failure_type"]))].append(row)
    result: list[dict[str, Any]] = []
    for (case_id, part_kind, failure_type), rows in sorted(groups.items()):
        times = [float(row["time_ms"]) for row in rows]
        result.append(
            {
                "case_id": case_id,
                "part_kind": part_kind,
                "failure_type": failure_type,
                "unique_element_count": len({int(row["element_id"]) for row in rows}),
                "first_time_ms": min(times),
                "last_time_ms": max(times),
            }
        )
    return result


def build_time_bin_summary(
    events: Sequence[Mapping[str, Any]], *, width_ms: float = 0.001
) -> list[dict[str, Any]]:
    if width_ms <= 0 or not math.isfinite(width_ms):
        raise ValueError("time-bin width must be finite and positive")
    groups: dict[tuple[str, str, str, int], set[int]] = defaultdict(set)
    for row in events:
        bin_index = int(math.floor(float(row["time_ms"]) / width_ms + 1e-12))
        key = (
            str(row["case_id"]),
            str(row["part_kind"]),
            str(row["failure_type"]),
            bin_index,
        )
        groups[key].add(int(row["element_id"]))
    return [
        {
            "case_id": key[0],
            "part_kind": key[1],
            "failure_type": key[2],
            "bin_start_ms": key[3] * width_ms,
            "bin_end_ms": (key[3] + 1) * width_ms,
            "unique_element_count": len(element_ids),
        }
        for key, element_ids in sorted(groups.items())
    ]


def _load_real_d3part(base_path: Path) -> Any:
    # Append, rather than prepend, so compatible environment dependencies such
    # as h5py/pandas are not shadowed by incomplete vendored dependency folders.
    vendor = str(VENDOR_DIR.resolve())
    if vendor not in sys.path:
        sys.path.append(vendor)
    try:
        from lasso.dyna import D3plot
    except ImportError as exc:
        raise RuntimeError("cannot import project-local lasso-python D3plot reader") from exc
    return D3plot(
        str(base_path.resolve()),
        state_array_filter=[
            ArrayType.global_timesteps,
            ArrayType.node_displacement,
            ArrayType.node_velocity,
            ArrayType.element_solid_is_alive,
        ],
        buffered_reading=True,
    )


def _load_real_d3plot_focus(base_path: Path) -> Any:
    vendor = str(VENDOR_DIR.resolve())
    if vendor not in sys.path:
        sys.path.append(vendor)
    try:
        from lasso.dyna import D3plot
    except ImportError as exc:
        raise RuntimeError("cannot import project-local lasso-python D3plot reader") from exc
    return D3plot(
        str(base_path.resolve()),
        state_array_filter=[
            ArrayType.global_timesteps,
            ArrayType.element_solid_stress,
            ArrayType.element_solid_effective_plastic_strain,
            ArrayType.element_solid_is_alive,
        ],
        buffered_reading=True,
    )


def _load_binout_inventory(return_dir: Path) -> dict[str, list[str]]:
    vendor = str(VENDOR_DIR.resolve())
    if vendor not in sys.path:
        sys.path.append(vendor)
    try:
        from lasso.dyna import Binout
    except ImportError as exc:
        raise RuntimeError("cannot import project-local lasso-python Binout reader") from exc
    binout = Binout(str(return_dir / "binout*"))
    return {
        str(directory): sorted(str(variable) for variable in binout.read(directory))
        for directory in binout.read()
    }


def build_focus_d3plot_diagnostic(
    return_dir: Path, focus_cases: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    d3plot = _load_real_d3plot_focus(return_dir / "d3plot")
    arrays = d3plot.arrays
    required = {
        ArrayType.global_timesteps,
        ArrayType.element_solid_stress,
        ArrayType.element_solid_effective_plastic_strain,
        ArrayType.element_solid_is_alive,
        ArrayType.element_solid_ids,
        ArrayType.element_solid_part_indexes,
        ArrayType.part_ids,
    }
    missing = sorted(str(key) for key in required - set(arrays))
    if missing:
        raise ValueError(f"D3PLOT focus-history arrays missing: {missing}")
    if getattr(d3plot.header, "has_solid_stress", False) is not True:
        raise ValueError("D3PLOT header does not declare solid stress")
    history = extract_focus_solid_history(
        times_ms=arrays[ArrayType.global_timesteps],
        element_stress_gpa=arrays[ArrayType.element_solid_stress],
        element_effective_plastic_strain=arrays[
            ArrayType.element_solid_effective_plastic_strain
        ],
        element_alive=arrays[ArrayType.element_solid_is_alive],
        element_ids=arrays[ArrayType.element_solid_ids],
        element_part_indexes=arrays[ArrayType.element_solid_part_indexes],
        part_ids=arrays[ArrayType.part_ids],
        focus_cases=focus_cases,
    )
    binout_inventory = _load_binout_inventory(return_dir)
    contact = assess_contact_output_availability(
        header=d3plot.header,
        d3plot_array_keys=set(arrays),
        binout_inventory=binout_inventory,
    )
    return {
        "reader": "project-local lasso-python 2.0.4 D3plot",
        "d3plot_summary": {
            "state_count": history["state_count"],
            "first_time_ms": history["first_time_ms"],
            "last_time_ms": history["last_time_ms"],
            "node_count": int(d3plot.header.n_nodes),
            "solid_count": int(d3plot.header.n_solids),
            "solid_result_layer_count": history["solid_result_layer_count"],
            "solid_state_variable_count": int(d3plot.header.n_solid_vars),
            "solid_history_variable_count": int(d3plot.header.n_solid_history_vars),
            "has_solid_stress": bool(d3plot.header.has_solid_stress),
        },
        "statistics_population": history["statistics_population"],
        "effective_plastic_strain_definition": history[
            "effective_plastic_strain_definition"
        ],
        "derived_solid_hydrostatic_pressure": history[
            "derived_solid_hydrostatic_pressure"
        ],
        "cases": history["cases"],
        "part_history_rows": history["part_history_rows"],
        "first_failure_element_history_rows": history[
            "first_failure_element_history_rows"
        ],
        "contact_output_availability": contact,
    }


def build_d3part_diagnostic(
    return_dir: Path,
    design: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    material: Mapping[str, Any],
) -> dict[str, Any]:
    d3part = _load_real_d3part(return_dir / "d3part")
    arrays = d3part.arrays
    required = {
        ArrayType.global_timesteps,
        ArrayType.node_coordinates,
        ArrayType.node_displacement,
        ArrayType.node_velocity,
        ArrayType.element_solid_node_indexes,
        ArrayType.element_solid_part_indexes,
        ArrayType.element_solid_is_alive,
        ArrayType.element_solid_ids,
        ArrayType.part_ids,
    }
    missing = sorted(str(key) for key in required - set(arrays))
    if missing:
        raise ValueError(f"D3PART required arrays missing: {missing}")
    for flag in ("has_node_displacement", "has_node_velocity", "has_element_deletion_data"):
        if getattr(d3part.header, flag, False) is not True:
            raise ValueError(f"D3PART header flag {flag} is required")

    times = np.asarray(arrays[ArrayType.global_timesteps], dtype=float)
    reference = np.asarray(arrays[ArrayType.node_coordinates], dtype=float)
    positions, coordinate_mode = resolve_node_state_coordinates(
        reference, arrays[ArrayType.node_displacement]
    )
    velocities = np.asarray(arrays[ArrayType.node_velocity], dtype=float)
    connectivity = np.asarray(arrays[ArrayType.element_solid_node_indexes], dtype=np.int64)
    element_ids = np.asarray(arrays[ArrayType.element_solid_ids], dtype=np.int64)
    part_indexes = np.asarray(arrays[ArrayType.element_solid_part_indexes], dtype=np.int64)
    part_ids = np.asarray(arrays[ArrayType.part_ids], dtype=np.int64)
    alive = np.asarray(arrays[ArrayType.element_solid_is_alive])
    if velocities.shape != positions.shape or alive.shape != (times.size, connectivity.shape[0]):
        raise ValueError("D3PART state array shapes are inconsistent")
    used_part_ids = part_ids[part_indexes]
    expected_fragment_pids = {int(case["fragment_part_id"]) for case in cases}
    if set(np.unique(used_part_ids).tolist()) != expected_fragment_pids:
        raise ValueError("D3PART used solid PID set differs from the frozen fragment PID set")
    if element_ids.shape != (connectivity.shape[0],) or np.unique(element_ids).size != element_ids.size:
        raise ValueError("D3PART solid element IDs must be unique")
    if not np.all(alive[0] != 0):
        raise ValueError("all fragment solids must be alive in the initial D3PART state")

    density = float(material["strength_model"]["density_kg_mm3"])
    masses = hexahedron_reference_volumes(reference, connectivity) * density
    thresholds, tail_fraction = capture_thresholds_from_design(design)
    capture_plane = float(design["measurement_contract"]["capture_plane"]["coordinate_mm"])
    case_design = {row["case_id"]: row for row in design["cases"]}
    ledger_by_case: dict[str, dict[str, Any]] = {}
    element_rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = case["case_id"]
        pid = int(case["fragment_part_id"])
        mask = used_part_ids == pid
        expected_count = int(case["fragment_elements"])
        if int(np.count_nonzero(mask)) != expected_count:
            raise ValueError(f"{case_id} D3PART fragment element count mismatch")
        frozen = case_design[case_id]
        ledger = build_capture_ledger(
            times_ms=times,
            node_positions_mm=positions,
            node_velocities_mm_per_ms=velocities,
            element_connectivity=connectivity[mask],
            element_alive=alive[:, mask],
            element_reference_mass_kg=masses[mask],
            expected_reference_mass_kg=float(case["fragment_initial_reference_mass_kg"]),
            capture_plane_z_mm=capture_plane,
            impact_speed_mps=float(case["impact_speed_mps"]),
            maximum_unambiguous_step_mm=0.5 * float(case["mesh_mm"]),
            tail_window_fraction=tail_fraction,
            thresholds=thresholds,
        )
        rows = classify_fragment_elements(
            times_ms=times,
            node_positions_mm=positions,
            node_velocities_mm_per_ms=velocities,
            element_connectivity=connectivity[mask],
            element_alive=alive[:, mask],
            element_ids=element_ids[mask],
            element_reference_mass_kg=masses[mask],
            capture_plane_z_mm=capture_plane,
            maximum_unambiguous_step_mm=0.5 * float(case["mesh_mm"]),
        )
        for state in LEDGER_STATES:
            row_mass = sum(
                row["reference_mass_kg"] for row in rows if row["ledger_state"] == state
            )
            if not math.isclose(
                row_mass,
                float(ledger["mass_by_ledger_state_kg"][state]),
                rel_tol=1e-10,
                abs_tol=1e-15,
            ):
                raise AssertionError(f"{case_id} element ledger disagrees for {state}")
        ledger_by_case[case_id] = {
            "case_id": case_id,
            "role": case["role"],
            "fragment_part_id": pid,
            "impact_speed_mps": float(frozen["impact_speed_mps"]),
            "mesh_mm": float(frozen["mesh_mm"]),
            "diagnostic_only": True,
            **ledger,
        }
        for row in rows:
            element_rows.append(
                {
                    "case_id": case_id,
                    "fragment_part_id": pid,
                    **row,
                }
            )

    expected_interval = float(design["measurement_contract"]["fragment_d3part_interval_ms"])
    tolerance = max(1e-9, expected_interval * 1e-3)
    deltas = np.diff(times)
    used_set = set(np.unique(used_part_ids).tolist())
    zero_solid_catalog = sorted(set(part_ids.tolist()) - used_set)
    formal_blockers: list[dict[str, Any]] = []
    if float(deltas.max()) > expected_interval + tolerance:
        formal_blockers.append(
            {
                "gate": "maximum_state_interval",
                "expected_upper_bound_ms": expected_interval + tolerance,
                "actual_ms": float(deltas.max()),
                "exceeding_interval_count": int(
                    np.count_nonzero(deltas > expected_interval + tolerance)
                ),
            }
        )
    if set(part_ids.tolist()) != expected_fragment_pids:
        formal_blockers.append(
            {
                "gate": "exact_part_catalog",
                "catalog_part_ids": part_ids.tolist(),
                "used_solid_part_ids": sorted(used_set),
                "zero_solid_catalog_part_ids": zero_solid_catalog,
            }
        )
    return {
        "summary": {
            "reader": "project-local lasso-python 2.0.4 D3plot",
            "coordinate_mode": coordinate_mode,
            "state_count": int(times.size),
            "first_time_ms": float(times[0]),
            "last_time_ms": float(times[-1]),
            "minimum_state_interval_ms": float(deltas.min()),
            "maximum_state_interval_ms": float(deltas.max()),
            "node_count": int(reference.shape[0]),
            "solid_count": int(connectivity.shape[0]),
            "catalog_part_ids": part_ids.tolist(),
            "used_solid_part_ids": sorted(used_set),
            "zero_solid_catalog_part_ids": zero_solid_catalog,
        },
        "initial_frozen_adapter_observation": {
            "implementation_scope": (
                "original frozen adapter observed at Stage B kickoff, before the concurrent repair"
            ),
            "status_at_observation": "fail_closed" if formal_blockers else "not_assessed",
            "observed_software_blockers": formal_blockers,
            "current_repaired_adapter_status": "not_assessed_by_this_failure_mode_analyzer",
            "nonretroactivity": (
                "A later adapter repair cannot retroactively qualify this R05 run; "
                "run-integrity evidence is now separately hash-bound and passing, while "
                "independent integration acceptance remains required."
            ),
        },
        "diagnostic_normalizations": [
            "the formal frozen maximum-state-interval gate is reported but not applied to this diagnostic ledger",
            "zero-solid plate PID catalog metadata is retained in the report but only actually referenced fragment PIDs are analyzed",
        ],
        "case_ledgers": ledger_by_case,
        "element_rows": element_rows,
    }


def _hash_family(paths: Sequence[Path]) -> dict[str, Any]:
    entries = [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    canonical = json.dumps(
        entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "entries": entries,
        "family_sha256": "sha256:" + hashlib.sha256(canonical).hexdigest(),
    }


def _numbered_family_paths(base_path: Path) -> list[Path]:
    pattern = re.compile(rf"{re.escape(base_path.name)}(\d*)$")
    matches: list[tuple[int, Path]] = []
    for path in base_path.parent.iterdir():
        match = pattern.fullmatch(path.name)
        if path.is_file() and match:
            suffix = match.group(1)
            matches.append((-1 if suffix == "" else int(suffix), path))
    paths = [path for _, path in sorted(matches, key=lambda item: item[0])]
    if not paths or paths[0].name != base_path.name:
        raise FileNotFoundError(f"numbered family base file missing: {base_path}")
    return paths


def load_run_integrity_binding(
    output_dir: Path, *, expected_d3part_family_sha256: str
) -> dict[str, Any]:
    """Load and independently recheck the existing Stage A run binding."""
    evidence_path = output_dir / "run_integrity_evidence.json"
    manifest_path = output_dir / "run_artifact_manifest.json"
    if not evidence_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "run_integrity_evidence.json and run_artifact_manifest.json are required"
        )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "pass":
        raise ValueError("run-integrity evidence is not an explicit pass")
    if str(evidence.get("run_id")) != "72252362":
        raise ValueError("run-integrity evidence has the wrong run ID")
    if evidence.get("d3part_family_sha256") != expected_d3part_family_sha256:
        raise ValueError("run-integrity D3PART family hash mismatch")
    actual_manifest_sha256 = sha256_file(manifest_path)
    if evidence.get("returned_artifact_sha256") != actual_manifest_sha256:
        raise ValueError("returned artifact manifest hash mismatch")
    return {
        "schema_version": str(evidence.get("schema_version")),
        "evidence_id": str(evidence.get("evidence_id")),
        "status": "pass",
        "run_id": str(evidence["run_id"]),
        "solver_build": str(evidence["solver_build"]),
        "d3part_family_sha256": str(evidence["d3part_family_sha256"]),
        "returned_artifact_sha256": actual_manifest_sha256,
        "evidence_sha256": sha256_file(evidence_path),
        "claim_boundary": (
            "This existing pass establishes run/returned-evidence integrity only; "
            "it is not D3PART integration acceptance or physical admission."
        ),
    }


def build_input_identity(return_dir: Path) -> dict[str, Any]:
    d3part_identity = d3part_family_identity(
        return_dir / "d3part",
        run_id="72252362",
        solver_build="LS-DYNA MPP double R15.0.2-1-g1e7cfa42b1 I8R8 Intel-MPI 2018",
    )
    mes = _rank_paths(return_dir)
    d3plot_family = _hash_family(_numbered_family_paths(return_dir / "d3plot"))
    return {
        "return_directory": str(return_dir.resolve()),
        "design_sha256": sha256_file(return_dir / "p0_cases_r05.json"),
        "deck_sha256": sha256_file(return_dir / "p0_velocity_mesh_calibration.k"),
        "case_map_sha256": sha256_file(return_dir / "p0_case_part_map.csv"),
        "material_sha256": sha256_file(return_dir / "jc4340_reference.json"),
        "manifest_sha256": sha256_file(return_dir / "manifest.json"),
        "d3part_family_identity_contract": "extract_r05_d3part.d3part_family_identity",
        "d3part_family_sha256": d3part_identity["d3part_family_sha256"],
        "d3part_family": d3part_identity["d3part_family"],
        "d3plot_family_sha256": d3plot_family["family_sha256"],
        "d3plot_family": d3plot_family["entries"],
        "mes_family": _hash_family(mes),
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def build_report(payload: Mapping[str, Any]) -> str:
    case_rows = payload["case_summary"]
    table = [
        "| Case | 首失效 ms | 首类型/部件 | Plate 负体积/其他 | Fragment 负体积/其他 | `deleted_upstream` 质量比例 | 诊断账本状态 |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in case_rows:
        first = (
            f"{row['first_failure_type']} / {row['first_failure_part_kind']}"
            if row["first_failure_type"]
            else "无"
        )
        first_time = (
            f"{row['first_failure_time_ms']:.8g}"
            if row["first_failure_time_ms"] is not None
            else "—"
        )
        deleted = row["deleted_upstream_mass_fraction"]
        deleted_text = f"{deleted:.6%}" if deleted is not None else "—"
        table.append(
            f"| `{row['case_id']}` | {first_time} | {first} | "
            f"{row['plate_negative_volume_count']}/{row['plate_other_failure_count']} | "
            f"{row['fragment_negative_volume_count']}/{row['fragment_other_failure_count']} | "
            f"{deleted_text} | "
            f"{'diagnostic_available_only' if row['primary_qoi_available'] else 'unavailable'} |"
        )

    focus_lines: list[str] = []
    focus_ids = ("p0_v0800_h0p75", "p0_v2000_h0p75")
    for case_id in focus_ids:
        row = next(item for item in case_rows if item["case_id"] == case_id)
        focus_lines.extend(
            [
                f"### `{case_id}`",
                "",
                f"- 首失效：`{row['first_failure_time_ms']:.8g} ms`，"
                f"`{row['first_failure_type']}`，`{row['first_failure_part_kind']}`，"
                f"EID `{row['first_failure_element_id']}`。",
                f"- 部件/类型：plate 负体积 `{row['plate_negative_volume_count']}`、其他 `{row['plate_other_failure_count']}`；"
                f"fragment 负体积 `{row['fragment_negative_volume_count']}`、其他 `{row['fragment_other_failure_count']}`。",
                f"- D3PART 捕获前删除质量比例（`deleted_upstream`）："
                f"`{row['deleted_upstream_mass_fraction']:.9f}`；主 QoI 为 `unavailable`。",
                f"- MES 失效时刻落在 D3PART 删除保存区间内："
                f"`{row['fragment_mes_time_in_d3part_deletion_interval_count']}`；区间外："
                f"`{row['fragment_mes_time_outside_d3part_deletion_interval_count']}`。",
                "",
            ]
        )

    solid_history = payload["focus_solid_history"]
    solid_table = [
        "| Case | Part/PID | 存活单元 初始→末态 | 存活单元最大有效塑性应变（时刻 ms） | "
        "派生实体静水压力最小/最大 GPa |",
        "|---|---|---:|---:|---:|",
    ]
    first_failure_history_lines: list[str] = []
    for case in solid_history["cases"]:
        for part in case["parts"]:
            solid_table.append(
                f"| `{case['case_id']}` | {part['part_kind']}/`{part['part_id']}` | "
                f"{part['initial_alive_element_count']}→{part['final_alive_element_count']} | "
                f"{part['maximum_effective_plastic_strain_among_alive']:.9g} "
                f"(`{part['maximum_effective_plastic_strain_time_ms']:.9g}`) | "
                f"{part['minimum_derived_solid_hydrostatic_pressure_gpa_among_alive']:.9g} / "
                f"{part['maximum_derived_solid_hydrostatic_pressure_gpa_among_alive']:.9g} |"
            )
        first = case["first_failure_element"]
        deleted_time = (
            f"{first['first_deleted_d3plot_time_ms']:.9g} ms"
            if first["first_deleted_d3plot_time_ms"] is not None
            else "末态仍存活"
        )
        first_failure_history_lines.append(
            f"- `{case['case_id']}` 首失效 EID `{first['element_id']}`：MES 首报 "
            f"`{first['mes_first_failure_time_ms']:.9g} ms`；D3PLOT 最后存活状态 "
            f"`{first['last_alive_d3plot_time_ms']:.9g} ms`，首个删除状态 `{deleted_time}`；"
            f"存活历史最大有效塑性应变 "
            f"`{first['maximum_effective_plastic_strain_while_alive']:.9g}`，派生实体静水压力范围 "
            f"`{first['minimum_derived_solid_hydrostatic_pressure_gpa_while_alive']:.9g}`–"
            f"`{first['maximum_derived_solid_hydrostatic_pressure_gpa_while_alive']:.9g} GPa`。"
        )

    d3 = payload["d3part"]
    blocker_lines = [
        f"- `{item['gate']}`：{json.dumps(item, ensure_ascii=False, allow_nan=False)}"
        for item in d3["initial_frozen_adapter_observation"]["observed_software_blockers"]
    ]
    return "\n".join(
        [
            "# Job 72252362 R05 阶段 B：失效模式与捕获账本关联",
            "",
            "## 结论边界",
            "",
            "本报告是冻结返回文件上的可复现诊断，不是物理准入、材料标定或数据集晋级。"
            "报告记录的是阶段 B 启动时原冻结适配器的已观察软件阻塞；"
            "当前修复版适配器状态不由本报告判定。以下账本只用于定位数值失效模式。",
            "",
            "## 全工况摘要",
            "",
            *table,
            "",
            "## 重点工况",
            "",
            *focus_lines,
            "## D3PLOT 有效塑性应变与派生实体静水压力历史",
            "",
            f"- 完整 D3PLOT 共 `{solid_history['d3plot_summary']['state_count']}` 个保存状态，"
            f"覆盖 `{solid_history['d3plot_summary']['first_time_ms']:.9g}`–"
            f"`{solid_history['d3plot_summary']['last_time_ms']:.9g} ms`。",
            "- 压力仅按 `p=-(sxx+syy+szz)/3` 由实体应力张量派生；本模型单位为 `GPa`。"
            "它不是接触压力。统计总体为每个保存状态仍存活的实体单元，采用非加权算术统计。",
            "",
            *solid_table,
            "",
            "### 首失效单元的删除前保存状态历史",
            "",
            *first_failure_history_lines,
            "",
            "上述极值只描述离散保存状态，不直接归因于材料、接触、裂纹或穿孔机理。",
            "",
            "## 接触量可恢复性",
            "",
            "- 直接接触穿透：`unavailable_not_written`。D3PLOT 的绝对/相对穿透 flag 均为 false，"
            "BINOUT 无 penetration 变量。",
            "- 直接接触压力：`unavailable_not_written`。返回数据库无 contact-pressure 数组或"
            "接触力/界面压力变量；SLEOUT 能量通道不能替代接触压力。",
            "- `cont_profile.*` 是并行 CPU/clock profile，不是接触物理历史。",
            "",
            "## D3PART 适配器历史观察与非追溯边界",
            "",
            f"- 状态数 `{d3['summary']['state_count']}`，时间覆盖 "
            f"`{d3['summary']['first_time_ms']:.9g}`–`{d3['summary']['last_time_ms']:.9g} ms`。",
            "- 下列两项是原冻结实现于阶段 B 启动时的已观察阻塞，不代表并发修复后的当前状态：",
            *blocker_lines,
            "- 即使修复后的适配器随后通过，也不能追溯把本次诊断变成 R05 准入证据；"
            "run-integrity 已另行形成身份绑定的通过证据，但仍缺独立 integration acceptance。",
            "",
            "## 空间信息的可靠范围",
            "",
            "- 所有失效单元：由 deck 节点/连接可计算初始参考几何质心；"
            "这是初始位置，不是失效瞬时位置。",
            "- Fragment：D3PART 含全局单元 ID，其参考质心已与 deck 逐单元交叉核对；"
            "还可报告删除前最后存活状态质心 z、"
            "删除后首个保存状态质心 z 以及删除时间区间。",
            "- Plate：可报告 deck 初始质心和 PID/case；当前 D3PART 是 fragment part set，"
            "因此 plate 失效瞬时/变形后坐标仍不可用。",
            "- `solid element ... failed` 文本没有说明材料、接触或其他失效机理，故仅标为"
            " `other_solid_failure`，不作物理因果归因。",
            "",
            "## 不能判定",
            "",
            "- 不能仅由负体积或单元删除判定真实裂纹、塞块、穿孔机理或实验有效性。",
            "- 保存状态只能给出删除区间；MES 时刻与 D3PART 区间的一致不等于因果证明。",
            "- 本分析没有评估并发修复后的当前适配器，也没有生成 assessor 准入载荷。",
            "- 未访问密封测试集，未修改材料参数、仿真引擎或数据集生成器。",
            "",
        ]
    )


def validate_output_location(
    return_dir: Path, output_dir: Path, *, allow_replace_owned: bool = False
) -> None:
    """Allow a shared Stage A/B directory while refusing any Stage B overwrite."""
    source = return_dir.resolve()
    output = output_dir.resolve()
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("output directory must not be inside the frozen return directory")
    if output.exists():
        collisions = sorted(name for name in OUTPUT_NAMES if (output / name).exists())
        if collisions and not allow_replace_owned:
            raise FileExistsError(
                "refusing to overwrite existing Stage B artifacts: " + ", ".join(collisions)
            )


def run_analysis(
    return_dir: Path, output_dir: Path, *, allow_replace_owned: bool = False
) -> dict[str, Any]:
    return_dir = return_dir.resolve()
    output_dir = output_dir.resolve()
    validate_output_location(
        return_dir, output_dir, allow_replace_owned=allow_replace_owned
    )
    design = json.loads((return_dir / "p0_cases_r05.json").read_text(encoding="utf-8"))
    material = json.loads((return_dir / "jc4340_reference.json").read_text(encoding="utf-8"))
    cases = load_cases(return_dir / "p0_case_part_map.csv")
    parsed = parse_mes_failure_events(return_dir)
    if parsed["rank_file_count"] != 32:
        raise ValueError(f"expected 32 mes rank files, got {parsed['rank_file_count']}")
    bound = bind_events_to_cases(
        parsed["unique_events"],
        return_dir / "p0_velocity_mesh_calibration.k",
        return_dir / "p0_case_part_map.csv",
    )
    unmapped = [row for row in bound if row["part_kind"] == "unmapped"]
    if unmapped:
        raise ValueError(f"{len(unmapped)} unique failure events cannot be mapped to a frozen case")

    d3part = build_d3part_diagnostic(return_dir, design, cases, material)
    associated = associate_failure_events_with_d3part(bound, d3part["element_rows"])
    missing_fragments = [
        row
        for row in associated
        if row["d3part_association"] == "fragment_element_missing_from_d3part"
    ]
    if missing_fragments:
        raise ValueError(
            f"{len(missing_fragments)} fragment failure events are absent from D3PART"
        )
    reference_mismatches = [
        row
        for row in associated
        if row.get("part_kind") == "fragment"
        and row.get("fragment_initial_reference_crosscheck_status") != "match"
    ]
    if reference_mismatches:
        raise ValueError(
            f"{len(reference_mismatches)} fragment reference centroids disagree between deck and D3PART"
        )
    case_summary = summarize_cases(associated, cases, d3part["case_ledgers"])
    focus = {
        row["case_id"]: row
        for row in case_summary
        if row["case_id"] in {"p0_v0800_h0p75", "p0_v2000_h0p75"}
    }
    focus_specs = [
        {
            "case_id": row["case_id"],
            "plate_part_id": row["plate_part_id"],
            "fragment_part_id": row["fragment_part_id"],
            "first_failure_element_id": row["first_failure_element_id"],
            "first_failure_time_ms": row["first_failure_time_ms"],
            "first_failure_type": row["first_failure_type"],
            "first_failure_part_kind": row["first_failure_part_kind"],
        }
        for row in case_summary
        if row["case_id"] in {"p0_v0800_h0p75", "p0_v2000_h0p75"}
    ]
    if len(focus_specs) != 2:
        raise ValueError("both frozen focus cases are required for D3PLOT history extraction")
    focus_history = build_focus_d3plot_diagnostic(return_dir, focus_specs)
    input_identity = build_input_identity(return_dir)
    run_integrity = load_run_integrity_binding(
        output_dir,
        expected_d3part_family_sha256=input_identity["d3part_family_sha256"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    focus_part_csv = output_dir / "d3plot_focus_part_history.csv"
    first_failure_csv = output_dir / "d3plot_first_failure_element_history.csv"
    write_csv(focus_part_csv, focus_history["part_history_rows"])
    write_csv(
        first_failure_csv,
        focus_history["first_failure_element_history_rows"],
    )
    focus_summary = {
        "schema_version": "hf-p0-r05-focus-solid-history@1.0.0",
        "analysis_status": "diagnostic_history_complete_no_admission",
        "qualification_use_forbidden": True,
        "integration_acceptance_status": (
            "pending_independent_integration_acceptance_no_retroactive_admission"
        ),
        "artifact_identity": {
            "package_revision": design["package_revision"],
            "job_id": "72252362",
            "design_sha256": input_identity["design_sha256"],
            "deck_sha256": input_identity["deck_sha256"],
            "case_map_sha256": input_identity["case_map_sha256"],
            "material_sha256": input_identity["material_sha256"],
            "upload_manifest_sha256": input_identity["manifest_sha256"],
            "d3part_family_sha256": input_identity["d3part_family_sha256"],
            "d3plot_family_sha256": input_identity["d3plot_family_sha256"],
            "d3plot_family_file_count": len(input_identity["d3plot_family"]),
            "d3plot_family_identity_contract": (
                "sha256 of canonical JSON name/size_bytes/file_sha256 entries in numbered-family order"
            ),
            "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        },
        "run_integrity_binding": run_integrity,
        "reader": focus_history["reader"],
        "d3plot_summary": focus_history["d3plot_summary"],
        "statistics_contract": {
            "population": focus_history["statistics_population"],
            "quantile_method": "numpy.quantile linear",
            "effective_plastic_strain": focus_history[
                "effective_plastic_strain_definition"
            ],
            "derived_solid_hydrostatic_pressure": focus_history[
                "derived_solid_hydrostatic_pressure"
            ],
            "extrema_interpretation": (
                "diagnostic saved-state extrema only; they do not identify or prove a "
                "material, contact, or fracture mechanism"
            ),
        },
        "cases": focus_history["cases"],
        "contact_output_availability": focus_history[
            "contact_output_availability"
        ],
        "csv_artifacts": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (focus_part_csv, first_failure_csv)
        ],
        "claim_boundary": (
            "Diagnostic D3PLOT solid-state history only; no direct contact penetration or "
            "contact pressure is available, and no response, physical, engine, or dataset "
            "admission follows from these extrema."
        ),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "job_id": "72252362",
        "analysis_status": "diagnostic_complete_no_admission",
        "input_identity": input_identity,
        "run_integrity_binding": run_integrity,
        "mes": {
            "rank_file_count": parsed["rank_file_count"],
            "raw_observation_count": parsed["raw_observation_count"],
            "unique_event_count": parsed["unique_event_count"],
            "unique_element_count": parsed["unique_element_count"],
            "cross_type_element_overlap_count": parsed[
                "cross_type_element_overlap_count"
            ],
            "negative_volume_unique_count": sum(
                row["failure_type"] == "negative_volume" for row in associated
            ),
            "other_solid_failure_unique_count": sum(
                row["failure_type"] == "other_solid_failure" for row in associated
            ),
        },
        "d3part": {
            "summary": d3part["summary"],
            "initial_frozen_adapter_observation": d3part[
                "initial_frozen_adapter_observation"
            ],
            "diagnostic_normalizations": d3part["diagnostic_normalizations"],
        },
        "case_summary": case_summary,
        "focus_cases": focus,
        "focus_solid_history": focus_summary,
        "limitations": [
            "plate initial deck centroids are available, but plate failure-time/deformed coordinates are unavailable from the fragment-only D3PART",
            "other_solid_failure does not identify a physical or numerical failure mechanism",
            "D3PART saved states bracket deletion but do not prove exact deletion time or causality",
            "solid hydrostatic pressure is derived from D3PLOT stress and is not contact pressure",
            "direct contact penetration and contact pressure were not written in the returned databases",
            "this analyzer does not assess the concurrently repaired adapter or produce integration acceptance",
            "later software repair cannot retroactively promote this R05 diagnostic to admission evidence",
            "solver diagnostics do not establish experimental or physical validity",
        ],
        "claim_boundary": (
            "Failure-mode localization and diagnostic capture-ledger association only; "
            "not P0 response admission, material calibration, physical validation, engine fitting, "
            "or dataset-generator promotion."
        ),
    }

    (output_dir / "d3plot_focus_response_summary.json").write_text(
        _json(focus_summary), encoding="utf-8", newline="\n"
    )
    write_csv(output_dir / "mes_failure_observations.csv", parsed["raw_observations"])
    write_csv(output_dir / "failure_events_unique.csv", associated)
    write_csv(output_dir / "failure_summary_by_case.csv", case_summary)
    write_csv(output_dir / "failure_summary_by_case_part_type.csv", build_group_summary(associated))
    write_csv(output_dir / "failure_time_bins_0p001ms.csv", build_time_bin_summary(associated))
    write_csv(output_dir / "d3part_element_ledger_diagnostic.csv", d3part["element_rows"])
    (output_dir / "d3part_capture_ledger_diagnostic.json").write_text(
        _json(
            {
                "schema_version": "hf-p0-r05-d3part-ledger-diagnostic@1.0.0",
                "qualification_use_forbidden": True,
                "initial_frozen_adapter_observation": d3part[
                    "initial_frozen_adapter_observation"
                ],
                "diagnostic_normalizations": d3part["diagnostic_normalizations"],
                "cases": list(d3part["case_ledgers"].values()),
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "failure_mode_analysis.json").write_text(
        _json(payload), encoding="utf-8", newline="\n"
    )
    (output_dir / "failure_mode_report_zh.md").write_text(
        build_report(payload), encoding="utf-8", newline="\n"
    )
    (output_dir / "stage_b_worklog.md").write_text(
        "\n".join(
            [
                "# Stage B worklog",
                "",
                "1. Treated `05/` as immutable evidence and rejected output paths inside it.",
                "2. Parsed all 32 `mes` files and deduplicated by failure type plus global element ID.",
                "3. Bound every unique event through the deck EID-to-PID mapping and frozen case map.",
                "4. Read the real D3PART family with project-local lasso-python 2.0.4.",
                "5. Recorded two blockers observed in the original frozen adapter at Stage B kickoff; did not assess the concurrent repair.",
                "6. Associated fragment events with D3PART element identities, deletion brackets and ledger states.",
                "7. Rechecked the existing hash-bound run-integrity pass and bound it to this diagnostic without converting it into integration acceptance.",
                "8. Read all full-D3PLOT saved states for the two focus cases and summarized alive-element effective plastic strain and p=-(sxx+syy+szz)/3 solid hydrostatic pressure.",
                "9. Exported the first MES-failure element history only through saved states where that element remained alive.",
                "10. Confirmed direct contact penetration/contact pressure are unavailable because the corresponding D3PLOT flags and BINOUT databases/variables were not written.",
                "11. Wrote strict JSON/CSV/Markdown artifacts; no assessor bridge, retroactive admission, or physical-admission claim was produced.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    owned = sorted(
        output_dir / name
        for name in OUTPUT_NAMES
        if name != "stage_b_analysis_manifest.json" and (output_dir / name).is_file()
    )
    manifest = {
        "schema_version": "hf-p0-r05-stage-b-analysis-manifest@1.1.0",
        "analyzer": str(Path(__file__).resolve()),
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "source_identity": payload["input_identity"],
        "artifacts": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in owned
        ],
    }
    (output_dir / "stage_b_analysis_manifest.json").write_text(
        _json(manifest), encoding="utf-8", newline="\n"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze frozen R05 mes failures and diagnostic D3PART capture states"
    )
    parser.add_argument("--return-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--replace-owned",
        action="store_true",
        help="replace only this script's exact Stage B artifact names",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_analysis(
        args.return_dir,
        args.output_dir,
        allow_replace_owned=args.replace_owned,
    )
    print(
        _json(
            {
                "status": payload["analysis_status"],
                "job_id": payload["job_id"],
                "unique_failure_events": payload["mes"]["unique_event_count"],
                "output_dir": str(args.output_dir.resolve()),
            }
        ),
        end="",
    )


if __name__ == "__main__":
    main()
