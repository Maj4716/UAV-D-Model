"""Validate readable partial R06 D3PART/D3PLOT and ragged NCFORC evidence.

Unlike the completed-run validator, this helper does not require the frozen 802/402
state counts.  It is fail-closed on identity, cross-format time alignment, finite
history variables, and strictly increasing output time.  A pass means only that
the cancelled output is readable and internally consistent.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from extract_r05_d3part import ArrayType, _load_lasso_binout, _load_lasso_d3plot


def _as_int_vector(value: object, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
        raise ValueError(f"{label} is not a finite integer vector")
    return array.astype(np.int64, copy=False)


def _shared_state_indexes(part_times: np.ndarray, full_times: np.ndarray) -> np.ndarray:
    if part_times.size == 0 or full_times.size == 0:
        raise ValueError("D3PART/D3PLOT time arrays must be nonempty")
    if not np.all(np.diff(part_times) > 0) or not np.all(np.diff(full_times) > 0):
        raise ValueError("D3PART/D3PLOT times must be strictly increasing")
    result: list[int] = []
    for time_value in full_times:
        insertion = int(np.searchsorted(part_times, time_value))
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < part_times.size]
        best = min(candidates, key=lambda index: abs(float(part_times[index] - time_value)))
        if not math.isclose(float(part_times[best]), float(time_value), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"no matching D3PART state for full-D3PLOT time {time_value}")
        result.append(best)
    indexes = np.asarray(result, dtype=np.int64)
    if np.unique(indexes).size != indexes.size:
        raise ValueError("full-D3PLOT states do not map uniquely into D3PART")
    return indexes


def _summary(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values) & (np.abs(values) < 1.0e29)]
    if finite.size == 0:
        return {"status": "unavailable", "reason": "no_finite_values"}
    return {
        "status": "available",
        "count": int(finite.size),
        "minimum": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "maximum": float(np.max(finite)),
    }


def validate_d3part(return_dir: Path) -> dict:
    reader = _load_lasso_d3plot()
    requested = [
        ArrayType.global_timesteps,
        ArrayType.element_solid_history_variables,
        ArrayType.element_solid_is_alive,
    ]
    d3part = reader(str((return_dir / "d3part").resolve()), state_array_filter=requested, buffered_reading=True)
    full = reader(str((return_dir / "d3plot").resolve()), state_array_filter=requested, buffered_reading=True)

    part_times = np.asarray(d3part.arrays[ArrayType.global_timesteps], dtype=float)
    full_times = np.asarray(full.arrays[ArrayType.global_timesteps], dtype=float)
    shared_part_indexes = _shared_state_indexes(part_times, full_times)
    part_ids = _as_int_vector(d3part.arrays[ArrayType.element_solid_ids], "D3PART element ids")
    full_ids = _as_int_vector(full.arrays[ArrayType.element_solid_ids], "D3PLOT element ids")
    full_part_indexes = _as_int_vector(full.arrays[ArrayType.element_solid_part_indexes], "D3PLOT part indexes")
    full_catalog_part_ids = _as_int_vector(full.arrays[ArrayType.part_ids], "D3PLOT part ids")
    full_used_part_ids = full_catalog_part_ids[full_part_indexes]
    fragment_indexes = np.flatnonzero(full_used_part_ids == 2002)
    plate_indexes = np.flatnonzero(full_used_part_ids == 1002)
    if fragment_indexes.size != 686 or plate_indexes.size != 20216 or part_ids.size != 686:
        raise ValueError("solid part counts differ from the frozen R06 identity")
    index_by_id = {int(element_id): int(index) for index, element_id in enumerate(full_ids)}
    if len(index_by_id) != full_ids.size or any(int(element_id) not in index_by_id for element_id in part_ids):
        raise ValueError("D3PART element ids are not a unique subset of D3PLOT ids")
    aligned_full_indexes = np.asarray([index_by_id[int(element_id)] for element_id in part_ids])
    if set(aligned_full_indexes.tolist()) != set(fragment_indexes.tolist()):
        raise ValueError("D3PART ids do not equal the complete fragment solid set")

    part_history = np.asarray(d3part.arrays[ArrayType.element_solid_history_variables], dtype=float)
    full_history = np.asarray(full.arrays[ArrayType.element_solid_history_variables], dtype=float)
    part_alive = np.asarray(d3part.arrays[ArrayType.element_solid_is_alive]) != 0
    full_alive = np.asarray(full.arrays[ArrayType.element_solid_is_alive]) != 0
    if part_history.shape != (part_times.size, 686, 1, 7) or part_alive.shape != (part_times.size, 686):
        raise ValueError(f"unexpected partial D3PART shape {part_history.shape}/{part_alive.shape}")
    if full_history.shape != (full_times.size, 20902, 1, 7) or full_alive.shape != (full_times.size, 20902):
        raise ValueError(f"unexpected partial D3PLOT shape {full_history.shape}/{full_alive.shape}")

    selected_part_alive = part_alive[shared_part_indexes]
    selected_full_alive = full_alive[:, aligned_full_indexes]
    alive_mismatch_count = int(np.count_nonzero(selected_part_alive != selected_full_alive))
    common_alive = selected_part_alive & selected_full_alive
    part_values = part_history[shared_part_indexes, :, 0, :]
    full_values = full_history[:, aligned_full_indexes, 0, :]
    differences = np.abs(part_values[common_alive] - full_values[common_alive])
    hvar6_alive = part_history[:, :, 0, 5][part_alive]
    invalid_hvar6 = ~np.isfinite(hvar6_alive) | (np.abs(hvar6_alive) >= 1.0e29)
    final_fragment_values = part_history[-1, :, 0, 5][part_alive[-1]]
    final_plate_values = full_history[-1, plate_indexes, 0, 5][full_alive[-1, plate_indexes]]
    status = "pass" if alive_mismatch_count == 0 and not np.any(invalid_hvar6) else "fail"
    return {
        "status": status,
        "evidence_scope": "partial_cancelled_run_readability_only",
        "d3part_history_shape": list(part_history.shape),
        "full_d3plot_history_shape": list(full_history.shape),
        "d3part_state_count": int(part_times.size),
        "full_d3plot_state_count": int(full_times.size),
        "shared_state_count": int(shared_part_indexes.size),
        "d3part_first_time_ms": float(part_times[0]),
        "d3part_last_time_ms": float(part_times[-1]),
        "full_d3plot_last_time_ms": float(full_times[-1]),
        "fragment_element_id_count": int(part_ids.size),
        "alive_value_comparison_count": int(np.count_nonzero(common_alive)),
        "maximum_absolute_difference_all_7_channels": float(np.max(differences)),
        "alive_mask_mismatch_count": alive_mismatch_count,
        "maximum_time_mismatch_ms": float(np.max(np.abs(part_times[shared_part_indexes] - full_times))),
        "hvar6_alive_status": "available" if not np.any(invalid_hvar6) else "unavailable_invalid",
        "hvar6_invalid_alive_count": int(np.count_nonzero(invalid_hvar6)),
        "hvar6_all_alive_fragment": _summary(hvar6_alive),
        "initial_alive_fragment_count": int(np.count_nonzero(part_alive[0])),
        "final_alive_fragment_count": int(np.count_nonzero(part_alive[-1])),
        "final_fragment_hvar6": _summary(final_fragment_values),
        "final_alive_plate_count": int(np.count_nonzero(full_alive[-1, plate_indexes])),
        "final_plate_hvar6": _summary(final_plate_values),
    }


def validate_ncforc(return_dir: Path) -> dict:
    binout = _load_lasso_binout()(str((return_dir / "binout*").resolve()))
    databases = sorted(str(name) for name in binout.read())
    surface = binout.lsda_root.get("ncforc").get("surfa_1")
    records: list[dict] = []
    union_ids: set[int] = set()
    intersection_ids: set[int] | None = None
    ranges = {name: [math.inf, -math.inf] for name in ("pressure", "x_force", "y_force", "z_force")}
    variables = ("ids", "x", "y", "z", "pressure", "x_force", "y_force", "z_force")
    for name, state in surface.children.items():
        if name == "metadata":
            continue
        time_value = float(state.get(b"time").read()[0])
        arrays = {variable: np.asarray(state.get(variable.encode()).read()) for variable in variables}
        count = int(arrays["ids"].size)
        if any(value.ndim != 1 or value.size != count for value in arrays.values()):
            raise ValueError(f"NCFORC state {name} contains inconsistent vector lengths")
        if any(not np.all(np.isfinite(value)) for variable, value in arrays.items() if variable != "ids"):
            raise ValueError(f"NCFORC state {name} contains a non-finite value")
        ids = _as_int_vector(arrays["ids"], f"NCFORC ids at {name}")
        if np.unique(ids).size != ids.size:
            raise ValueError(f"NCFORC state {name} repeats a node id")
        id_set = set(int(value) for value in ids)
        union_ids.update(id_set)
        intersection_ids = id_set if intersection_ids is None else intersection_ids & id_set
        for variable in ranges:
            ranges[variable][0] = min(ranges[variable][0], float(np.min(arrays[variable])))
            ranges[variable][1] = max(ranges[variable][1], float(np.max(arrays[variable])))
        records.append({"time_ms": time_value, "node_count": count})
    records.sort(key=lambda item: item["time_ms"])
    times = np.asarray([item["time_ms"] for item in records], dtype=float)
    counts = np.asarray([item["node_count"] for item in records], dtype=np.int64)
    if times.size == 0 or not np.all(np.diff(times) > 0):
        raise ValueError("NCFORC times are empty or not strictly increasing")
    return {
        "status": "available_ragged_pending_semantic_validation",
        "path": "ncforc/surfa_1",
        "logical_databases": databases,
        "state_count": int(times.size),
        "first_time_ms": float(times[0]),
        "last_time_ms": float(times[-1]),
        "minimum_nodes_per_state": int(np.min(counts)),
        "maximum_nodes_per_state": int(np.max(counts)),
        "union_node_count": len(union_ids),
        "all_state_intersection_node_count": len(intersection_ids or set()),
        "node_count_change_count": int(np.count_nonzero(np.diff(counts))),
        "all_ids_unique_per_state": True,
        "all_coordinates_forces_pressure_finite": True,
        "raw_model_unit_ranges": ranges,
        "claim_boundary": "Units, sign convention, contact-side mapping, impulse and energy closure are not validated.",
    }


def build_validation(return_dir: Path) -> dict:
    return {
        "schema_version": "hf-p0-r06-partial-independent-measurement-validation@1.0.0",
        "run_completion_policy": "cancelled_partial_output_is_never_a_terminal_response",
        "d3part_full_d3plot_crosscheck": validate_d3part(return_dir),
        "ncforc": validate_ncforc(return_dir),
        "claim_boundary": "Readable partial output does not establish terminal response, numerical stability, physical validity, fitting or dataset admission.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("return_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.resolve().is_relative_to(args.return_dir.resolve()):
        raise ValueError("analysis output must be outside the immutable return directory")
    result = build_validation(args.return_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
