from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np

from build_r06g_tssfac02_return_audit import _negative_volume_topology, strict_json
from extract_r05_d3part import ArrayType, _load_lasso_d3plot, resolve_node_state_coordinates


TARGET_PLATE_ELEMENT_ID = 39948
CAPTURE_PLANE_Z_MM = -12.0


def _finite_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return {"count": int(values.size), "minimum": None, "p05": None, "p50": None, "p95": None, "maximum": None}
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def _read_run(return_dir: Path) -> dict:
    requested = [
        ArrayType.global_timesteps,
        ArrayType.node_displacement,
        ArrayType.node_velocity,
        ArrayType.element_solid_is_alive,
        ArrayType.element_solid_history_variables,
        ArrayType.sph_n_neighbors,
        ArrayType.sph_deletion,
    ]
    d3plot = _load_lasso_d3plot()(
        str((return_dir / "d3plot").resolve()),
        state_array_filter=requested,
        buffered_reading=True,
    )
    arrays = d3plot.arrays
    times = np.asarray(arrays[ArrayType.global_timesteps], dtype=float)
    reference = np.asarray(arrays[ArrayType.node_coordinates], dtype=float)
    positions, coordinate_mode = resolve_node_state_coordinates(
        reference,
        np.asarray(arrays[ArrayType.node_displacement], dtype=float),
    )
    velocities = np.asarray(arrays[ArrayType.node_velocity], dtype=float)
    ids = np.asarray(arrays[ArrayType.element_solid_ids], dtype=np.int64)
    parts = np.asarray(arrays[ArrayType.part_ids], dtype=np.int64)[
        np.asarray(arrays[ArrayType.element_solid_part_indexes], dtype=np.int64)
    ]
    connectivity = np.asarray(arrays[ArrayType.element_solid_node_indexes], dtype=np.int64)
    alive = np.asarray(arrays[ArrayType.element_solid_is_alive], dtype=float) != 0.0
    history = np.asarray(arrays[ArrayType.element_solid_history_variables], dtype=float)
    hvar6 = history[:, :, 0, 5]
    target_indexes = np.flatnonzero(ids == TARGET_PLATE_ELEMENT_ID)
    if target_indexes.size != 1:
        raise ValueError("target plate element identity is ambiguous")
    target = int(target_indexes[0])
    target_centroids = positions[:, connectivity[target], :].mean(axis=1)
    target_velocities = velocities[:, connectivity[target], :].mean(axis=1)
    target_alive = alive[:, target]

    fragment_mask = parts == 2002
    final_fragment_alive = alive[-1, fragment_mask]
    fragment_connectivity = connectivity[fragment_mask]
    final_solid_centroids = positions[-1, fragment_connectivity, :].mean(axis=1)
    final_solid_velocities = velocities[-1, fragment_connectivity, :].mean(axis=1)
    solid_z = final_solid_centroids[final_fragment_alive, 2]
    solid_vz = final_solid_velocities[final_fragment_alive, 2]
    solid_downward = solid_vz < 0.0
    solid_projected = times[-1] + (solid_z[solid_downward] - CAPTURE_PLANE_Z_MM) / (-solid_vz[solid_downward])

    sph_indexes = np.asarray(arrays[ArrayType.sph_node_indexes], dtype=np.int64)
    neighbours = np.asarray(arrays[ArrayType.sph_n_neighbors], dtype=float)
    sph_deletion = np.asarray(arrays[ArrayType.sph_deletion], dtype=float)
    final_active_sph = (neighbours[-1] > 0.0) & (sph_deletion[-1] == 0.0)
    sph_z = positions[-1, sph_indexes, 2][final_active_sph]
    sph_vz = velocities[-1, sph_indexes, 2][final_active_sph]
    sph_downward = sph_vz < 0.0
    sph_projected = times[-1] + (sph_z[sph_downward] - CAPTURE_PLANE_Z_MM) / (-sph_vz[sph_downward])

    stdout_paths = list(return_dir.glob("std.out.*"))
    if len(stdout_paths) != 1:
        raise ValueError("expected one solver stdout")
    stdout = stdout_paths[0].read_text(encoding="utf-8", errors="replace")
    cycles = re.findall(r"Problem cycle\s*=\s*(\d+)", stdout)
    elapsed = re.findall(r"Elapsed time\s+(\d+) seconds", stdout)
    if not cycles or not elapsed:
        raise ValueError("completion metrics missing")
    return {
        "state_count": int(times.size),
        "last_time_ms": float(times[-1]),
        "coordinate_mode": coordinate_mode,
        "cycle_count": int(cycles[-1]),
        "elapsed_seconds": int(elapsed[-1]),
        "final_deleted_fragment_element_ids": ids[fragment_mask][~final_fragment_alive].tolist(),
        "final_deleted_plate_element_ids": ids[parts == 1002][~alive[-1, parts == 1002]].tolist(),
        "target_plate_element": {
            "element_id": TARGET_PLATE_ELEMENT_ID,
            "terminal_alive": bool(target_alive[-1]),
            "final_centroid_mm": target_centroids[-1].tolist(),
            "final_velocity_mm_per_ms": target_velocities[-1].tolist(),
            "final_hvar6": float(hvar6[-1, target]),
            "maximum_alive_hvar6": float(np.max(hvar6[target_alive, target])),
        },
        "terminal_tail_diagnostic": {
            "capture_plane_z_mm": CAPTURE_PLANE_Z_MM,
            "alive_solid_fragment_count": int(np.count_nonzero(final_fragment_alive)),
            "downward_solid_fragment_count": int(np.count_nonzero(solid_downward)),
            "solid_fragment_z_mm": _finite_summary(solid_z),
            "downward_solid_fragment_vz_mm_per_ms": _finite_summary(solid_vz[solid_downward]),
            "linear_projected_solid_crossing_time_ms": _finite_summary(solid_projected),
            "active_sph_count": int(np.count_nonzero(final_active_sph)),
            "downward_active_sph_count": int(np.count_nonzero(sph_downward)),
            "active_sph_z_mm": _finite_summary(sph_z),
            "downward_active_sph_vz_mm_per_ms": _finite_summary(sph_vz[sph_downward]),
            "linear_projected_sph_crossing_time_ms": _finite_summary(sph_projected),
            "linear_projection_claim_boundary": (
                "Constant terminal velocity is used only to choose a conservative next horizon; it is not a physical "
                "trajectory prediction or a perforation result."
            ),
        },
    }


def build_comparison(
    *,
    r06n_summary: Path,
    r06o_return: Path,
    r06n_ascii: Path,
    r06o_ascii: Path,
) -> dict:
    r06o_return = r06o_return.resolve()
    frozen_n = strict_json(r06n_summary.resolve())
    frozen_n_run = frozen_n["r06n"]
    frozen_n_d3plot = frozen_n_run["d3plot"]
    n = {
        "cycle_count": frozen_n_run["completion"]["cycle_count"],
        "elapsed_seconds": frozen_n_run["completion"]["elapsed_seconds"],
        "final_deleted_fragment_element_ids": frozen_n_d3plot["deleted_fragment_element_ids"],
        "final_deleted_plate_element_ids": frozen_n_d3plot["deleted_plate_element_ids"],
        "target_plate_element": {
            "element_id": TARGET_PLATE_ELEMENT_ID,
            "terminal_alive": frozen_n_d3plot["target_plate_element"]["first_failed_state"] is None,
            "final_hvar6": frozen_n_d3plot["target_plate_element"]["final_hvar6_raw"],
        },
        "evidence_source": "frozen_r06m_r06n_resource_sensitivity_summary",
    }
    o = _read_run(r06o_return)
    n_ascii = strict_json(r06n_ascii.resolve())
    o_ascii = strict_json(r06o_ascii.resolve())
    n_nv = frozen_n_run["negative_volume"]
    o_nv = _negative_volume_topology(
        r06o_return,
        r06o_return / "r06o_0800_plate_erod1_isolation.k",
    )
    n_converted = set(n["final_deleted_fragment_element_ids"])
    o_converted = set(o["final_deleted_fragment_element_ids"])
    union = n_converted | o_converted
    intersection = n_converted & o_converted
    solid_projection = o["terminal_tail_diagnostic"]["linear_projected_solid_crossing_time_ms"]
    sph_projection = o["terminal_tail_diagnostic"]["linear_projected_sph_crossing_time_ms"]
    earliest_solid = solid_projection["minimum"]
    earliest_sph = sph_projection["minimum"]
    finite_earliest = [value for value in (earliest_solid, earliest_sph) if value is not None]
    earliest_projected = min(finite_earliest) if finite_earliest else None
    projection_targets = [value for value in (sph_projection["p50"], earliest_solid) if value is not None]
    if projection_targets:
        target_with_margin = min(projection_targets) + 0.04
        recommended_horizon = math.ceil(target_with_margin / 0.05 - 1e-12) * 0.05
    elif earliest_projected is not None:
        recommended_horizon = math.ceil((earliest_projected * 1.2) / 0.05 - 1e-12) * 0.05
    else:
        recommended_horizon = 0.1
    return {
        "schema_version": "hf-p0-r06n-r06o-plate-erod-horizon-comparison@1.0.0",
        "single_factor": {"plate_mat_johnson_cook_erod": {"r06n": 0, "r06o": 1}},
        "r06n": {
            "run": n,
            "minimum_time_step_ms": n_ascii["global_statistics"]["minimum_time_step"],
            "final_sliding_energy_fraction": n_ascii["global_statistics"]["final_sliding_energy_fraction_of_initial"],
            "negative_volume": n_nv,
        },
        "r06o": {
            "run": o,
            "minimum_time_step_ms": o_ascii["global_statistics"]["minimum_time_step"],
            "final_sliding_energy_fraction": o_ascii["global_statistics"]["final_sliding_energy_fraction_of_initial"],
            "negative_volume": o_nv,
        },
        "paired_diagnostics": {
            "cycle_count_ratio_r06o_over_r06n": o["cycle_count"] / n["cycle_count"],
            "minimum_time_step_ratio_r06o_over_r06n": (
                o_ascii["global_statistics"]["minimum_time_step"]
                / n_ascii["global_statistics"]["minimum_time_step"]
            ),
            "plate_deleted_count_r06n": len(n["final_deleted_plate_element_ids"]),
            "plate_deleted_count_r06o": len(o["final_deleted_plate_element_ids"]),
            "target_plate_element_terminal_alive_r06o": o["target_plate_element"]["terminal_alive"],
            "target_plate_element_final_hvar6_r06o": o["target_plate_element"]["final_hvar6"],
            "converted_fragment_count_r06n": len(n_converted),
            "converted_fragment_count_r06o": len(o_converted),
            "converted_fragment_count_difference": len(o_converted) - len(n_converted),
            "converted_fragment_jaccard": len(intersection) / len(union),
            "converted_only_in_r06n": sorted(n_converted - o_converted),
            "converted_only_in_r06o": sorted(o_converted - n_converted),
            "earliest_linear_projected_capture_crossing_time_ms": earliest_projected,
            "median_linear_projected_sph_crossing_time_ms": sph_projection["p50"],
            "earliest_linear_projected_solid_crossing_time_ms": earliest_solid,
            "recommended_next_horizon_ms": recommended_horizon,
        },
        "decision_support": {
            "plate_erod1_engineering_isolation": (
                "pass" if len(o["final_deleted_plate_element_ids"]) == 0 and o["target_plate_element"]["terminal_alive"] else "fail"
            ),
            "horizon_extension_candidate_ms": recommended_horizon,
            "reason": (
                "R06o retains the target plate element without timestep collapse. A single-factor horizon extension is now "
                "eligible, with the selected horizon based on terminal-position/velocity diagnostics only."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare R06n/R06o and choose the next response horizon")
    parser.add_argument("--r06n-summary", type=Path, required=True)
    parser.add_argument("--r06o-return", type=Path, required=True)
    parser.add_argument("--r06n-ascii", type=Path, required=True)
    parser.add_argument("--r06o-ascii", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = build_comparison(
        r06n_summary=args.r06n_summary,
        r06o_return=args.r06o_return,
        r06n_ascii=args.r06n_ascii,
        r06o_ascii=args.r06o_ascii,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"paired_diagnostics": result["paired_diagnostics"], "decision_support": result["decision_support"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
