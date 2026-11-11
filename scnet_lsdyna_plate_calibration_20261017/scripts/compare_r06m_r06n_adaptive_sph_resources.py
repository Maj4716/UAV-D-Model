from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from build_r06g_tssfac02_return_audit import _negative_volume_topology, strict_json
from extract_r05_d3part import ArrayType, _load_lasso_d3plot, resolve_node_state_coordinates


TARGET_PLATE_ELEMENT_ID = 39948


def _read_run(return_dir: Path) -> dict:
    requested = [
        ArrayType.global_timesteps,
        ArrayType.node_displacement,
        ArrayType.element_solid_is_alive,
        ArrayType.element_solid_history_variables,
    ]
    d3plot = _load_lasso_d3plot()(
        str((return_dir / "d3plot").resolve()),
        state_array_filter=requested,
        buffered_reading=True,
    )
    arrays = d3plot.arrays
    times = np.asarray(arrays[ArrayType.global_timesteps], dtype=float)
    ids = np.asarray(arrays[ArrayType.element_solid_ids], dtype=np.int64)
    part_catalog = np.asarray(arrays[ArrayType.part_ids], dtype=np.int64)
    part_indexes = np.asarray(arrays[ArrayType.element_solid_part_indexes], dtype=np.int64)
    used_part_ids = part_catalog[part_indexes]
    alive = np.asarray(arrays[ArrayType.element_solid_is_alive], dtype=float) != 0.0
    history = np.asarray(arrays[ArrayType.element_solid_history_variables], dtype=float)
    if history.ndim != 4 or history.shape[2:] != (1, 7):
        raise ValueError(f"unexpected history shape {history.shape}")
    hvar6 = history[:, :, 0, 5]
    target_indexes = np.flatnonzero(ids == TARGET_PLATE_ELEMENT_ID)
    if target_indexes.size != 1:
        raise ValueError("target plate element identity is ambiguous")
    target_index = int(target_indexes[0])
    if int(used_part_ids[target_index]) != 1002:
        raise ValueError("target element is not on plate part 1002")

    reference = np.asarray(arrays[ArrayType.node_coordinates], dtype=float)
    connectivity = np.asarray(arrays[ArrayType.element_solid_node_indexes], dtype=np.int64)
    positions, coordinate_mode = resolve_node_state_coordinates(
        reference,
        np.asarray(arrays[ArrayType.node_displacement], dtype=float),
    )
    target_centroids = positions[:, connectivity[target_index], :].mean(axis=1)
    target_alive = alive[:, target_index]
    failed_states = np.flatnonzero(~target_alive)
    first_failed_state = int(failed_states[0]) if failed_states.size else None
    last_alive_state = int(np.flatnonzero(target_alive)[-1])

    fragment_mask = used_part_ids == 2002
    deleted_fragment_ids = ids[fragment_mask][~alive[-1, fragment_mask]]
    plate_mask = used_part_ids == 1002
    deleted_plate_ids = ids[plate_mask][~alive[-1, plate_mask]]
    return {
        "state_count": int(times.size),
        "last_time_ms": float(times[-1]),
        "coordinate_mode": coordinate_mode,
        "deleted_fragment_element_ids": deleted_fragment_ids.tolist(),
        "deleted_plate_element_ids": deleted_plate_ids.tolist(),
        "target_plate_element": {
            "element_id": TARGET_PLATE_ELEMENT_ID,
            "initial_centroid_mm": target_centroids[0].tolist(),
            "first_failed_state": first_failed_state,
            "first_failed_time_ms": float(times[first_failed_state]) if first_failed_state is not None else None,
            "last_alive_state": last_alive_state,
            "last_alive_time_ms": float(times[last_alive_state]),
            "last_alive_centroid_mm": target_centroids[last_alive_state].tolist(),
            "last_alive_hvar6": float(hvar6[last_alive_state, target_index]),
            "maximum_alive_hvar6": float(np.max(hvar6[target_alive, target_index])),
            "final_hvar6_raw": float(hvar6[-1, target_index]),
        },
    }


def _completion(return_dir: Path) -> dict:
    stdout_paths = list(return_dir.glob("std.out.*"))
    if len(stdout_paths) != 1:
        raise ValueError("expected one solver stdout")
    text = stdout_paths[0].read_text(encoding="utf-8", errors="replace")
    cycles = re.findall(r"Problem cycle\s*=\s*(\d+)", text)
    elapsed = re.findall(r"Elapsed time\s+(\d+) seconds", text)
    ranks = re.findall(r"MPP execution with\s+(\d+) procs", text)
    if not cycles or not elapsed or not ranks:
        raise ValueError("completion metrics missing")
    return {"cycle_count": int(cycles[-1]), "elapsed_seconds": int(elapsed[-1]), "ranks": int(ranks[-1])}


def build_comparison(
    *,
    r06m_return: Path,
    r06n_return: Path,
    r06m_ascii: Path,
    r06n_ascii: Path,
) -> dict:
    r06m_return = r06m_return.resolve()
    r06n_return = r06n_return.resolve()
    m = _read_run(r06m_return)
    n = _read_run(r06n_return)
    m_ascii = strict_json(r06m_ascii.resolve())
    n_ascii = strict_json(r06n_ascii.resolve())
    m_completion = _completion(r06m_return)
    n_completion = _completion(r06n_return)
    m_nv = _negative_volume_topology(r06m_return, r06m_return / "r06m_0800_adaptive_sph_canary.k")
    n_nv = _negative_volume_topology(
        r06n_return,
        r06n_return / "r06n_0800_adaptive_sph_32rank_confirmation.k",
    )
    m_converted = set(m["deleted_fragment_element_ids"])
    n_converted = set(n["deleted_fragment_element_ids"])
    m_nv_ids = {item["element_id"] for item in m_nv["elements"]}
    n_nv_ids = {item["element_id"] for item in n_nv["elements"]}
    union = m_converted | n_converted
    intersection = m_converted & n_converted
    return {
        "schema_version": "hf-p0-r06m-r06n-resource-sensitivity@1.0.0",
        "comparison_factor": {"r06m_observed_ranks": 64, "r06n_observed_ranks": 32},
        "r06m": {
            "completion": m_completion,
            "minimum_time_step_ms": m_ascii["global_statistics"]["minimum_time_step"],
            "final_sliding_energy_fraction": m_ascii["global_statistics"]["final_sliding_energy_fraction_of_initial"],
            "negative_volume": m_nv,
            "d3plot": m,
        },
        "r06n": {
            "completion": n_completion,
            "minimum_time_step_ms": n_ascii["global_statistics"]["minimum_time_step"],
            "final_sliding_energy_fraction": n_ascii["global_statistics"]["final_sliding_energy_fraction_of_initial"],
            "negative_volume": n_nv,
            "d3plot": n,
        },
        "paired_diagnostics": {
            "cycle_count_ratio_r06n_over_r06m": n_completion["cycle_count"] / m_completion["cycle_count"],
            "minimum_time_step_ratio_r06n_over_r06m": (
                n_ascii["global_statistics"]["minimum_time_step"]
                / m_ascii["global_statistics"]["minimum_time_step"]
            ),
            "converted_fragment_count_r06m": len(m_converted),
            "converted_fragment_count_r06n": len(n_converted),
            "converted_fragment_count_difference": len(n_converted) - len(m_converted),
            "converted_fragment_intersection_count": len(intersection),
            "converted_fragment_union_count": len(union),
            "converted_fragment_jaccard": len(intersection) / len(union),
            "converted_only_in_r06m": sorted(m_converted - n_converted),
            "converted_only_in_r06n": sorted(n_converted - m_converted),
            "r06m_negative_volume_equals_converted_set": m_nv_ids == m_converted,
            "r06n_negative_volume_equals_converted_set": n_nv_ids == n_converted,
            "r06n_plate_failure_is_not_in_negative_volume_set": TARGET_PLATE_ELEMENT_ID not in n_nv_ids,
        },
        "interpretation": (
            "The exact model is resource-decomposition sensitive. R06n preserves fragment conversion mass and energy gates, "
            "but the 32-rank trajectory produces a material-failed plate element and a cycle-count ratio outside the preregistered "
            "repeatability interval. This is a failed confirmation, not a calibrated response."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare R06m 64-rank and R06n 32-rank adaptive-SPH runs")
    parser.add_argument("--r06m-return", type=Path, required=True)
    parser.add_argument("--r06n-return", type=Path, required=True)
    parser.add_argument("--r06m-ascii", type=Path, required=True)
    parser.add_argument("--r06n-ascii", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = build_comparison(
        r06m_return=args.r06m_return,
        r06n_return=args.r06n_return,
        r06m_ascii=args.r06m_ascii,
        r06n_ascii=args.r06n_ascii,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result["paired_diagnostics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
