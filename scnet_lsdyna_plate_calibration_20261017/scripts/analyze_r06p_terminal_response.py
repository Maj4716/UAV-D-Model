"""Build a fail-closed zero-crossing/terminal-separation response ledger for R06p.

The adaptive-SPH model transfers reference mass from failed fragment solids to
preallocated particles.  A terminal nonperforation result is admitted only if
the complete solid/SPH carrier history remains upstream of the capture plane,
the transfer closes mass state by state, the temporal resolution margin is
large, and every carrier in the final 10% window is moving away from the plane.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from extract_r05_d3part import (
    ArrayType,
    _load_lasso_d3plot,
    hexahedron_reference_volumes,
    resolve_node_state_coordinates,
)
from r05_capture_ledger import CaptureThresholds, classify_numerical_perforation


CAPTURE_PLANE_Z_MM = -12.0
EXPECTED_REFERENCE_MASS_KG = 0.002
FRAGMENT_PART_ID = 2002
PLATE_PART_ID = 1002
TARGET_PLATE_ELEMENT_ID = 39948
DENSITY_KG_MM3 = 7.83e-6
MAXIMUM_UNAMBIGUOUS_STEP_MM = 0.25
TAIL_WINDOW_FRACTION = 0.10


def _summary(values: np.ndarray) -> dict:
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


def _carrier_state_metrics(
    *,
    alive_fragment: np.ndarray,
    solid_z: np.ndarray,
    solid_vz: np.ndarray,
    active_sph: np.ndarray,
    sph_z: np.ndarray,
    sph_vz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_count = alive_fragment.shape[0]
    minimum_z = np.full(state_count, np.nan)
    minimum_vz = np.full(state_count, np.nan)
    downward_count = np.zeros(state_count, dtype=np.int64)
    for state in range(state_count):
        z_values = np.concatenate((solid_z[state, alive_fragment[state]], sph_z[state, active_sph[state]]))
        vz_values = np.concatenate((solid_vz[state, alive_fragment[state]], sph_vz[state, active_sph[state]]))
        if z_values.size == 0 or vz_values.size == 0:
            raise ValueError(f"no fragment mass carriers at state {state}")
        minimum_z[state] = np.min(z_values)
        minimum_vz[state] = np.min(vz_values)
        downward_count[state] = np.count_nonzero(vz_values < 0.0)
    return minimum_z, minimum_vz, downward_count


def build_terminal_response(
    return_dir: Path,
    *,
    schema_version: str = "hf-p0-r06p-terminal-response-ledger@1.0.0",
) -> dict:
    return_dir = return_dir.resolve()
    requested = [
        ArrayType.global_timesteps,
        ArrayType.node_displacement,
        ArrayType.node_velocity,
        ArrayType.element_solid_is_alive,
        ArrayType.element_solid_history_variables,
        ArrayType.sph_n_neighbors,
        ArrayType.sph_mass,
        ArrayType.sph_deletion,
    ]
    d3plot = _load_lasso_d3plot()(
        str((return_dir / "d3plot").resolve()),
        state_array_filter=requested,
        buffered_reading=True,
    )
    arrays = d3plot.arrays
    times = np.asarray(arrays[ArrayType.global_timesteps], dtype=float)
    if times.size < 2 or not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0.0):
        raise ValueError("D3PLOT time vector is invalid")
    if not math.isclose(float(times[-1]), 0.5, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError("R06p response ledger requires the completed 0.50 ms history")

    reference = np.asarray(arrays[ArrayType.node_coordinates], dtype=float)
    positions, coordinate_mode = resolve_node_state_coordinates(
        reference,
        np.asarray(arrays[ArrayType.node_displacement], dtype=float),
    )
    velocities = np.asarray(arrays[ArrayType.node_velocity], dtype=float)
    ids = np.asarray(arrays[ArrayType.element_solid_ids], dtype=np.int64)
    part_catalog = np.asarray(arrays[ArrayType.part_ids], dtype=np.int64)
    parts = part_catalog[np.asarray(arrays[ArrayType.element_solid_part_indexes], dtype=np.int64)]
    connectivity = np.asarray(arrays[ArrayType.element_solid_node_indexes], dtype=np.int64)
    alive = np.asarray(arrays[ArrayType.element_solid_is_alive], dtype=float) != 0.0
    history = np.asarray(arrays[ArrayType.element_solid_history_variables], dtype=float)
    fragment_mask = parts == FRAGMENT_PART_ID
    plate_mask = parts == PLATE_PART_ID
    if np.count_nonzero(fragment_mask) != 686 or np.count_nonzero(plate_mask) != 20216:
        raise ValueError("R06p solid topology differs from the frozen identity")
    alive_fragment = alive[:, fragment_mask]
    fragment_ids = ids[fragment_mask]
    fragment_connectivity = connectivity[fragment_mask]
    solid_positions = positions[:, fragment_connectivity, :].mean(axis=2)
    solid_velocities = velocities[:, fragment_connectivity, :].mean(axis=2)
    solid_z = solid_positions[:, :, 2]
    solid_vz = solid_velocities[:, :, 2]

    volumes = hexahedron_reference_volumes(reference, connectivity)
    reference_masses = volumes * DENSITY_KG_MM3
    fragment_reference_masses = reference_masses[fragment_mask]
    total_reference_mass = float(np.sum(fragment_reference_masses))
    if not math.isclose(total_reference_mass, EXPECTED_REFERENCE_MASS_KG, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("fragment reference mass differs from 0.002 kg")

    sph_indexes = np.asarray(arrays[ArrayType.sph_node_indexes], dtype=np.int64)
    node_ids = np.asarray(arrays[ArrayType.node_ids], dtype=np.int64)
    sph_node_ids = node_ids[sph_indexes]
    neighbors = np.asarray(arrays[ArrayType.sph_n_neighbors], dtype=float)
    sph_mass = np.asarray(arrays[ArrayType.sph_mass], dtype=float)
    sph_deletion = np.asarray(arrays[ArrayType.sph_deletion], dtype=float)
    active_sph = (neighbors > 0.0) & (sph_deletion == 0.0)
    sph_positions = positions[:, sph_indexes, :]
    sph_velocities = velocities[:, sph_indexes, :]
    sph_z = sph_positions[:, :, 2]
    sph_vz = sph_velocities[:, :, 2]

    deleted_fragment = ~alive_fragment
    deleted_counts = np.count_nonzero(deleted_fragment, axis=1)
    active_counts = np.count_nonzero(active_sph, axis=1)
    activation_correspondence = active_counts == 8 * deleted_counts
    alive_solid_mass = np.sum(alive_fragment * fragment_reference_masses[None, :], axis=1)
    converted_reference_mass = np.sum(deleted_fragment * fragment_reference_masses[None, :], axis=1)
    active_sph_mass = np.sum(active_sph * sph_mass, axis=1)
    combined_mass_error = (alive_solid_mass + active_sph_mass - total_reference_mass) / total_reference_mass
    conversion_mass_error = (active_sph_mass - converted_reference_mass) / total_reference_mass

    solid_crossings = (
        alive_fragment[:-1]
        & alive_fragment[1:]
        & (solid_z[:-1] > CAPTURE_PLANE_Z_MM)
        & (solid_z[1:] <= CAPTURE_PLANE_Z_MM)
    )
    sph_crossings = (
        active_sph[:-1]
        & active_sph[1:]
        & (sph_z[:-1] > CAPTURE_PLANE_Z_MM)
        & (sph_z[1:] <= CAPTURE_PLANE_Z_MM)
    )
    newly_active = active_sph & ~np.vstack((np.zeros((1, active_sph.shape[1]), dtype=bool), active_sph[:-1]))
    activation_below_plane = newly_active & (sph_z <= CAPTURE_PLANE_Z_MM)
    disappearing_solids = alive_fragment[:-1] & ~alive_fragment[1:]
    deletion_interval_plane_ambiguity = (
        disappearing_solids
        & (solid_z[:-1] > CAPTURE_PLANE_Z_MM)
        & (solid_z[1:] <= CAPTURE_PLANE_Z_MM)
    )
    endpoint_at_or_below_solid = alive_fragment & (solid_z <= CAPTURE_PLANE_Z_MM)
    endpoint_at_or_below_sph = active_sph & (sph_z <= CAPTURE_PLANE_Z_MM)

    solid_consecutive = alive_fragment[:-1] & alive_fragment[1:]
    sph_consecutive = active_sph[:-1] & active_sph[1:]
    solid_signed_step = (solid_z[:-1] - solid_z[1:])[solid_consecutive]
    sph_signed_step = (sph_z[:-1] - sph_z[1:])[sph_consecutive]
    solid_downward_matrix = np.where(
        solid_consecutive,
        np.maximum(solid_z[:-1] - solid_z[1:], 0.0),
        -np.inf,
    )
    sph_downward_matrix = np.where(
        sph_consecutive,
        np.maximum(sph_z[:-1] - sph_z[1:], 0.0),
        -np.inf,
    )
    solid_max_flat = int(np.argmax(solid_downward_matrix))
    sph_max_flat = int(np.argmax(sph_downward_matrix))
    solid_max_state, solid_max_index = np.unravel_index(solid_max_flat, solid_downward_matrix.shape)
    sph_max_state, sph_max_index = np.unravel_index(sph_max_flat, sph_downward_matrix.shape)
    solid_max_downward_step = float(solid_downward_matrix[solid_max_state, solid_max_index])
    sph_max_downward_step = float(sph_downward_matrix[sph_max_state, sph_max_index])
    maximum_observed_downward_step = float(
        max(np.max(np.maximum(solid_signed_step, 0.0)), np.max(np.maximum(sph_signed_step, 0.0)))
    )
    maximum_observed_absolute_step = float(
        max(np.max(np.abs(solid_signed_step)), np.max(np.abs(sph_signed_step)))
    )
    carrier_minimum_z, carrier_minimum_vz, carrier_downward_count = _carrier_state_metrics(
        alive_fragment=alive_fragment,
        solid_z=solid_z,
        solid_vz=solid_vz,
        active_sph=active_sph,
        sph_z=sph_z,
        sph_vz=sph_vz,
    )
    minimum_endpoint_z = float(np.min(carrier_minimum_z))
    minimum_capture_clearance = minimum_endpoint_z - CAPTURE_PLANE_Z_MM
    temporal_resolution_pass = bool(
        maximum_observed_downward_step <= MAXIMUM_UNAMBIGUOUS_STEP_MM
        and minimum_capture_clearance > MAXIMUM_UNAMBIGUOUS_STEP_MM
    )

    tail_start = float(times[-1] - TAIL_WINDOW_FRACTION * (times[-1] - times[0]))
    tail_mask = times >= tail_start
    tail_minimum_z = float(np.min(carrier_minimum_z[tail_mask]))
    tail_minimum_vz = float(np.min(carrier_minimum_vz[tail_mask]))
    tail_maximum_downward_count = int(np.max(carrier_downward_count[tail_mask]))
    terminal_separation_pass = bool(
        tail_minimum_z > CAPTURE_PLANE_Z_MM + MAXIMUM_UNAMBIGUOUS_STEP_MM
        and tail_minimum_vz >= 0.0
        and tail_maximum_downward_count == 0
    )

    plate_deleted_counts = np.count_nonzero(~alive[:, plate_mask], axis=1)
    target_indexes = np.flatnonzero(ids == TARGET_PLATE_ELEMENT_ID)
    if target_indexes.size != 1:
        raise ValueError("target plate element identity is ambiguous")
    target = int(target_indexes[0])
    target_alive = alive[:, target]
    target_hvar6 = history[:, target, 0, 5]

    terminal_alive_mass = float(alive_solid_mass[-1])
    terminal_sph_mass = float(active_sph_mass[-1])
    terminal_solid_velocity = solid_velocities[-1, alive_fragment[-1]]
    terminal_sph_velocity = sph_velocities[-1, active_sph[-1]]
    terminal_velocity_momentum = (
        np.sum(terminal_solid_velocity * fragment_reference_masses[alive_fragment[-1], None], axis=0)
        + np.sum(terminal_sph_velocity * sph_mass[-1, active_sph[-1], None], axis=0)
    )
    terminal_mass_weighted_velocity = terminal_velocity_momentum / total_reference_mass
    terminal_rebound_qp = float(max(terminal_velocity_momentum[2], 0.0) / (total_reference_mass * 800.0))

    crossing_count = int(np.count_nonzero(solid_crossings) + np.count_nonzero(sph_crossings))
    ambiguity_count = int(np.count_nonzero(activation_below_plane) + np.count_nonzero(deletion_interval_plane_ambiguity))
    endpoint_below_count = int(np.count_nonzero(endpoint_at_or_below_solid) + np.count_nonzero(endpoint_at_or_below_sph))
    mass_gate_pass = bool(
        np.all(activation_correspondence)
        and np.max(np.abs(combined_mass_error)) <= 0.01
        and np.max(np.abs(conversion_mass_error)) <= 0.01
        and not np.any(sph_deletion != 0.0)
    )
    plate_gate_pass = bool(np.max(plate_deleted_counts) == 0 and target_alive[-1])
    zero_crossing_gate_pass = bool(crossing_count == 0 and ambiguity_count == 0 and endpoint_below_count == 0)
    primary_available = bool(
        mass_gate_pass
        and plate_gate_pass
        and zero_crossing_gate_pass
        and temporal_resolution_pass
        and terminal_separation_pass
    )
    thresholds = CaptureThresholds()
    q_m = 0.0 if primary_available else None
    q_p = 0.0 if primary_available else None
    q_e = 0.0 if primary_available else None
    unresolved = 0.0 if primary_available else None
    classification = (
        classify_numerical_perforation(0.0, 0.0, unresolved_crossing_mass_fraction=0.0, thresholds=thresholds)
        if primary_available
        else "unavailable"
    )
    unavailable_reasons = []
    for passed, reason in (
        (mass_gate_pass, "solid_sph_mass_transfer"),
        (plate_gate_pass, "plate_retention"),
        (zero_crossing_gate_pass, "zero_crossing_identity"),
        (temporal_resolution_pass, "temporal_resolution"),
        (terminal_separation_pass, "terminal_tail_separation"),
    ):
        if not passed:
            unavailable_reasons.append(reason)

    converted_ids = fragment_ids[~alive_fragment[-1]].tolist()
    return {
        "schema_version": schema_version,
        "source": {
            "directory": return_dir.as_posix(),
            "evidence_policy": "read_only_immutable",
            "state_count": int(times.size),
            "first_time_ms": float(times[0]),
            "last_time_ms": float(times[-1]),
            "coordinate_mode": coordinate_mode,
        },
        "capture_contract": {
            "capture_plane_z_mm": CAPTURE_PLANE_Z_MM,
            "impact_speed_mps": 800.0,
            "expected_reference_mass_kg": EXPECTED_REFERENCE_MASS_KG,
            "maximum_unambiguous_step_mm": MAXIMUM_UNAMBIGUOUS_STEP_MM,
            "tail_window_fraction": TAIL_WINDOW_FRACTION,
            "classification_thresholds": {
                "confirmed_nonperforation_qM_max": thresholds.confirmed_nonperforation_qM_max,
                "confirmed_nonperforation_qP_max": thresholds.confirmed_nonperforation_qP_max,
                "confirmed_nonperforation_worst_case_crossing_mass_max": thresholds.confirmed_nonperforation_worst_case_crossing_mass_max,
            },
        },
        "solid_sph_transfer": {
            "initial_fragment_reference_mass_kg": total_reference_mass,
            "final_alive_solid_count": int(np.count_nonzero(alive_fragment[-1])),
            "final_converted_solid_count": int(deleted_counts[-1]),
            "final_converted_solid_element_ids": converted_ids,
            "final_active_sph_count": int(active_counts[-1]),
            "statewise_exact_8_to_1_activation": bool(np.all(activation_correspondence)),
            "final_alive_solid_mass_kg": terminal_alive_mass,
            "final_active_sph_mass_kg": terminal_sph_mass,
            "maximum_absolute_combined_mass_relative_error": float(np.max(np.abs(combined_mass_error))),
            "maximum_absolute_conversion_mass_relative_error": float(np.max(np.abs(conversion_mass_error))),
            "sph_deleted_particle_count": int(np.count_nonzero(sph_deletion)),
            "gate": "pass" if mass_gate_pass else "fail",
        },
        "capture_observations": {
            "solid_crossing_event_count": int(np.count_nonzero(solid_crossings)),
            "sph_crossing_event_count": int(np.count_nonzero(sph_crossings)),
            "newly_active_sph_below_plane_count": int(np.count_nonzero(activation_below_plane)),
            "solid_deletion_interval_plane_ambiguity_count": int(np.count_nonzero(deletion_interval_plane_ambiguity)),
            "active_or_alive_endpoint_at_or_below_plane_count": endpoint_below_count,
            "minimum_observed_carrier_z_mm": minimum_endpoint_z,
            "minimum_observed_capture_plane_clearance_mm": minimum_capture_clearance,
            "maximum_observed_consecutive_downward_carrier_z_step_mm": maximum_observed_downward_step,
            "maximum_observed_consecutive_absolute_carrier_z_step_mm": maximum_observed_absolute_step,
            "maximum_downward_step_event": (
                {
                    "carrier_type": "fragment_solid",
                    "carrier_id": int(fragment_ids[solid_max_index]),
                    "start_time_ms": float(times[solid_max_state]),
                    "end_time_ms": float(times[solid_max_state + 1]),
                    "start_z_mm": float(solid_z[solid_max_state, solid_max_index]),
                    "end_z_mm": float(solid_z[solid_max_state + 1, solid_max_index]),
                    "downward_step_mm": solid_max_downward_step,
                }
                if solid_max_downward_step >= sph_max_downward_step
                else {
                    "carrier_type": "active_sph",
                    "carrier_id": int(sph_node_ids[sph_max_index]),
                    "start_time_ms": float(times[sph_max_state]),
                    "end_time_ms": float(times[sph_max_state + 1]),
                    "start_z_mm": float(sph_z[sph_max_state, sph_max_index]),
                    "end_z_mm": float(sph_z[sph_max_state + 1, sph_max_index]),
                    "downward_step_mm": sph_max_downward_step,
                }
            ),
            "temporal_resolution_gate": "pass" if temporal_resolution_pass else "fail",
        },
        "terminal_tail": {
            "tail_start_time_ms": tail_start,
            "tail_state_count": int(np.count_nonzero(tail_mask)),
            "minimum_carrier_z_mm": tail_minimum_z,
            "minimum_carrier_vz_mm_per_ms": tail_minimum_vz,
            "maximum_downward_carrier_count": tail_maximum_downward_count,
            "final_minimum_carrier_z_mm": float(carrier_minimum_z[-1]),
            "final_minimum_carrier_vz_mm_per_ms": float(carrier_minimum_vz[-1]),
            "final_downward_carrier_count": int(carrier_downward_count[-1]),
            "final_mass_weighted_velocity_mm_per_ms": terminal_mass_weighted_velocity.tolist(),
            "diagnostic_rebound_qP": terminal_rebound_qp,
            "gate": "pass" if terminal_separation_pass else "fail",
        },
        "plate_retention": {
            "maximum_deleted_plate_element_count": int(np.max(plate_deleted_counts)),
            "final_deleted_plate_element_count": int(plate_deleted_counts[-1]),
            "target_element_id": TARGET_PLATE_ELEMENT_ID,
            "target_terminal_alive": bool(target_alive[-1]),
            "target_final_hvar6": float(target_hvar6[-1]),
            "target_maximum_alive_hvar6": float(np.max(target_hvar6[target_alive])),
            "gate": "pass" if plate_gate_pass else "fail",
        },
        "primary_response": {
            "available": primary_available,
            "unavailable_reasons": unavailable_reasons,
            "classification": classification,
            "qM": q_m,
            "qP": q_p,
            "qE": q_e,
            "qV_over_v0": None,
            "unresolved_crossing_mass_fraction": unresolved,
            "worst_case_crossing_mass_fraction": (q_m + unresolved) if primary_available else None,
        },
        "claim_boundary": (
            "The result is a deterministic numerical classification for this one 800 m/s, 0.75 mm-mesh run. "
            "It does not establish experimental ballistic limit, material calibration, mesh convergence, engine fitting, "
            "damage-tree fitting or physical validity."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R06p terminal response ledger")
    parser.add_argument("return_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--schema-version",
        default="hf-p0-r06p-terminal-response-ledger@1.0.0",
        help="identity label for an exact-model measurement repeat; analysis logic is shared",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    return_dir = args.return_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(return_dir):
        raise ValueError("analysis output must remain outside the immutable return directory")
    result = build_terminal_response(return_dir, schema_version=args.schema_version)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"primary_response": result["primary_response"], "terminal_tail": result["terminal_tail"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
