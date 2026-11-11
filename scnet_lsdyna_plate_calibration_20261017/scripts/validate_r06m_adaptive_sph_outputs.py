"""Validate R06m adaptive solid-to-SPH output without treating it as calibration.

The adaptive definition preallocates all possible SPH particles.  Therefore the
raw SPH array length and raw SPH mass sum are not active-debris mass.  For this
R15 return, activation is established only after an exact state-by-state check
that positive-neighbour particles equal eight times the converted-solid count.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from extract_r05_d3part import (
    ArrayType,
    _load_lasso_binout,
    _load_lasso_d3plot,
    hexahedron_reference_volumes,
    resolve_node_state_coordinates,
)
from validate_r06_partial_measurements import validate_d3part, validate_ncforc


EXPECTED_SOLID_COUNT = 20_902
EXPECTED_PLATE_COUNT = 20_216
EXPECTED_FRAGMENT_COUNT = 686
EXPECTED_PREALLOCATED_SPH_COUNT = EXPECTED_FRAGMENT_COUNT * 8
EXPECTED_PART_IDS = [1002, 2002, 2003]
CAPTURE_PLANE_Z_MM = -12.0
DENSITY_KG_MM3 = 7.83e-6


def _finite_range(values: np.ndarray, label: str) -> list[float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain finite values")
    return [float(np.min(array)), float(np.max(array))]


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise ValueError(f"{label} shape {array.shape} differs from {shape}")


def validate_adaptive_sph(
    return_dir: Path,
    *,
    strict_gates: bool = True,
    required_problem_time_ms: float = 0.04,
) -> dict:
    reader = _load_lasso_d3plot()
    requested = [
        ArrayType.global_timesteps,
        ArrayType.node_displacement,
        ArrayType.node_velocity,
        ArrayType.element_solid_is_alive,
        ArrayType.sph_radius,
        ArrayType.sph_pressure,
        ArrayType.sph_effective_plastic_strain,
        ArrayType.sph_density,
        ArrayType.sph_internal_energy,
        ArrayType.sph_n_neighbors,
        ArrayType.sph_mass,
        ArrayType.sph_deletion,
    ]
    d3plot = reader(
        str((return_dir / "d3plot").resolve()),
        state_array_filter=requested,
        buffered_reading=True,
    )
    arrays = d3plot.arrays
    times = np.asarray(arrays[ArrayType.global_timesteps], dtype=float)
    if times.size == 0 or not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise ValueError("D3PLOT time vector is empty, nonfinite or nonmonotonic")
    state_count = int(times.size)
    if not math.isclose(float(times[-1]), required_problem_time_ms, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError(
            f"D3PLOT does not cover the required {required_problem_time_ms:g} ms problem time"
        )

    part_ids = np.asarray(arrays[ArrayType.part_ids], dtype=np.int64)
    if part_ids.tolist() != EXPECTED_PART_IDS:
        raise ValueError(f"part catalog differs: {part_ids.tolist()}")
    solid_part_indexes = np.asarray(arrays[ArrayType.element_solid_part_indexes], dtype=np.int64)
    used_part_ids = part_ids[solid_part_indexes]
    if (
        used_part_ids.size != EXPECTED_SOLID_COUNT
        or int(np.count_nonzero(used_part_ids == 1002)) != EXPECTED_PLATE_COUNT
        or int(np.count_nonzero(used_part_ids == 2002)) != EXPECTED_FRAGMENT_COUNT
    ):
        raise ValueError("solid part topology differs from the frozen R06 identity")
    plate_mask = used_part_ids == 1002
    fragment_mask = used_part_ids == 2002

    alive_raw = np.asarray(arrays[ArrayType.element_solid_is_alive], dtype=float)
    _require_shape(alive_raw, (state_count, EXPECTED_SOLID_COUNT), "solid alive array")
    if not np.all(np.isfinite(alive_raw)) or np.any(alive_raw < 0.0):
        raise ValueError("solid alive array contains invalid values")
    alive = alive_raw != 0.0
    deleted_fragment = ~alive[:, fragment_mask]
    deleted_plate = ~alive[:, plate_mask]
    if strict_gates and np.any(deleted_plate):
        raise ValueError("plate deletion occurred in the adaptive-SPH canary")
    deleted_plate_counts = np.count_nonzero(deleted_plate, axis=1)
    plate_deletion_states = np.flatnonzero(deleted_plate_counts > 0)

    sph_indexes = np.asarray(arrays[ArrayType.sph_node_indexes], dtype=np.int64)
    sph_material_indexes = np.asarray(arrays[ArrayType.sph_node_material_index], dtype=np.int64)
    if sph_indexes.size != EXPECTED_PREALLOCATED_SPH_COUNT or sph_material_indexes.shape != sph_indexes.shape:
        raise ValueError("SPH preallocation differs from 686 times 8")
    if not np.all(sph_material_indexes == 2):
        raise ValueError("SPH material index differs from material 2")

    neighbors = np.asarray(arrays[ArrayType.sph_n_neighbors], dtype=float)
    sph_mass = np.asarray(arrays[ArrayType.sph_mass], dtype=float)
    sph_deletion = np.asarray(arrays[ArrayType.sph_deletion], dtype=float)
    for label, value in (
        ("SPH neighbour", neighbors),
        ("SPH mass", sph_mass),
        ("SPH deletion", sph_deletion),
    ):
        _require_shape(value, (state_count, EXPECTED_PREALLOCATED_SPH_COUNT), label)
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{label} contains nonfinite values")
    active_sph = neighbors > 0.0
    deleted_counts = np.count_nonzero(deleted_fragment, axis=1)
    active_counts = np.count_nonzero(active_sph, axis=1)
    activation_correspondence = active_counts == deleted_counts * 8
    if strict_gates and not np.all(activation_correspondence):
        mismatch = np.flatnonzero(~activation_correspondence).tolist()
        raise ValueError(f"active SPH count does not equal eight per converted solid at states {mismatch[:8]}")
    if strict_gates and np.any(sph_deletion != 0.0):
        raise ValueError("one or more preallocated SPH particles report deletion")
    activation_states = np.flatnonzero(active_counts > 0)
    if activation_states.size == 0:
        raise ValueError("adaptive conversion never activated SPH particles")

    reference_coordinates = np.asarray(arrays[ArrayType.node_coordinates], dtype=float)
    connectivity = np.asarray(arrays[ArrayType.element_solid_node_indexes], dtype=np.int64)
    volumes = hexahedron_reference_volumes(reference_coordinates, connectivity)
    reference_masses = volumes * DENSITY_KG_MM3
    fragment_reference_masses = reference_masses[fragment_mask]
    initial_fragment_mass = float(np.sum(fragment_reference_masses))
    if not math.isclose(initial_fragment_mass, 0.002, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("fragment reference mass differs from 0.002 kg")
    alive_solid_mass = np.sum(alive[:, fragment_mask] * fragment_reference_masses[None, :], axis=1)
    active_sph_mass = np.sum(active_sph * sph_mass, axis=1)
    converted_reference_mass = np.sum(deleted_fragment * fragment_reference_masses[None, :], axis=1)
    combined_mass_relative_error = (
        alive_solid_mass + active_sph_mass - initial_fragment_mass
    ) / initial_fragment_mass
    conversion_mass_relative_error = (
        active_sph_mass - converted_reference_mass
    ) / initial_fragment_mass
    max_combined_error = float(np.max(np.abs(combined_mass_relative_error)))
    max_conversion_error = float(np.max(np.abs(conversion_mass_relative_error)))
    mass_gate_pass = max_combined_error <= 0.01 and max_conversion_error <= 0.01
    if strict_gates and not mass_gate_pass:
        raise ValueError("adaptive solid plus active-SPH mass fails the preregistered 1% engineering gate")

    positions, coordinate_mode = resolve_node_state_coordinates(
        reference_coordinates,
        np.asarray(arrays[ArrayType.node_displacement], dtype=float),
    )
    velocities = np.asarray(arrays[ArrayType.node_velocity], dtype=float)
    sph_positions = positions[:, sph_indexes, :]
    sph_velocities = velocities[:, sph_indexes, :]
    all_active_positions_finite = bool(np.all(np.isfinite(sph_positions[active_sph])))
    all_active_velocities_finite = bool(np.all(np.isfinite(sph_velocities[active_sph])))
    if strict_gates and (not all_active_positions_finite or not all_active_velocities_finite):
        raise ValueError("active SPH position or velocity contains nonfinite values")
    fragment_connectivity = connectivity[fragment_mask]
    solid_centroids = positions[:, fragment_connectivity, :].mean(axis=2)
    solid_velocities = velocities[:, fragment_connectivity, :].mean(axis=2)
    solid_crossings = (
        alive[:-1, fragment_mask]
        & alive[1:, fragment_mask]
        & (solid_centroids[:-1, :, 2] > CAPTURE_PLANE_Z_MM)
        & (solid_centroids[1:, :, 2] <= CAPTURE_PLANE_Z_MM)
    )
    sph_crossings = (
        active_sph[:-1]
        & active_sph[1:]
        & (sph_positions[:-1, :, 2] > CAPTURE_PLANE_Z_MM)
        & (sph_positions[1:, :, 2] <= CAPTURE_PLANE_Z_MM)
    )
    final_alive_solid = alive[-1, fragment_mask]
    final_active_sph = active_sph[-1]
    final_downward_solid_count = int(np.count_nonzero(solid_velocities[-1, final_alive_solid, 2] < 0.0))
    final_downward_sph_count = int(np.count_nonzero(sph_velocities[-1, final_active_sph, 2] < 0.0))
    crossing_count = int(np.count_nonzero(solid_crossings) + np.count_nonzero(sph_crossings))
    if crossing_count > 0:
        capture_status = "crossing_observed_requires_combined_solid_sph_ledger"
        capture_reason = "crossing events are present; a de-duplicated reference-mass ledger is required before admitting a response"
    elif final_downward_solid_count or final_downward_sph_count:
        capture_status = "no_crossing_observed_but_terminal_horizon_not_qualified"
        capture_reason = (
            f"the {required_problem_time_ms:g} ms horizon ends with downward-moving retained mass before any capture-plane crossing"
        )
    else:
        capture_status = "no_crossing_observed_terminal_tail_requires_stable_state_review"
        capture_reason = "no crossing is observed and no retained mass is moving downward; stable no-perforation review is still required"

    active_fields = {
        "radius": np.asarray(arrays[ArrayType.sph_radius], dtype=float)[active_sph],
        "pressure": np.asarray(arrays[ArrayType.sph_pressure], dtype=float)[active_sph],
        "effective_plastic_strain": np.asarray(
            arrays[ArrayType.sph_effective_plastic_strain], dtype=float
        )[active_sph],
        "density": np.asarray(arrays[ArrayType.sph_density], dtype=float)[active_sph],
        "internal_energy": np.asarray(arrays[ArrayType.sph_internal_energy], dtype=float)[active_sph],
        "neighbors": neighbors[active_sph],
        "mass": sph_mass[active_sph],
    }
    active_ranges = {name: _finite_range(value, f"active SPH {name}") for name, value in active_fields.items()}

    binout = _load_lasso_binout()(str((return_dir / "binout*").resolve()))
    matsum_ids = np.asarray(binout.read("matsum", "ids"), dtype=np.int64)
    matsum_times = np.asarray(binout.read("matsum", "time"), dtype=float)
    matsum_kinetic = np.asarray(binout.read("matsum", "kinetic_energy"), dtype=float)
    matsum_internal = np.asarray(binout.read("matsum", "internal_energy"), dtype=float)
    if matsum_ids.tolist() != EXPECTED_PART_IDS:
        raise ValueError("MATSUM part catalog differs")
    if matsum_kinetic.shape != (matsum_times.size, 3) or matsum_internal.shape != (matsum_times.size, 3):
        raise ValueError("MATSUM part-energy shapes differ")
    initial_source_sph_kinetic_difference = float(abs(matsum_kinetic[0, 1] - matsum_kinetic[0, 2]))
    initial_source_sph_internal_difference = float(abs(matsum_internal[0, 1] - matsum_internal[0, 2]))
    source_sph_columns_duplicate_at_initial_state = bool(
        np.isclose(matsum_kinetic[0, 1], matsum_kinetic[0, 2], rtol=1e-12, atol=1e-9)
        and np.isclose(matsum_internal[0, 1], matsum_internal[0, 2], rtol=1e-12, atol=1e-9)
    )
    if not source_sph_columns_duplicate_at_initial_state:
        raise ValueError(
            "MATSUM source/SPH duplication diagnostic changed; the combined-energy policy requires review"
        )

    adaptive_gate_pass = bool(
        np.all(activation_correspondence)
        and not np.any(deleted_plate)
        and not np.any(sph_deletion != 0.0)
        and mass_gate_pass
        and all_active_positions_finite
        and all_active_velocities_finite
    )
    diagnostic_plate_fields = {}
    if not strict_gates:
        diagnostic_plate_fields = {
            "maximum_plate_deleted_element_count": int(np.max(deleted_plate_counts)),
            "first_plate_deletion_state": (
                int(plate_deletion_states[0]) if plate_deletion_states.size else None
            ),
            "first_plate_deletion_time_ms": (
                float(times[plate_deletion_states[0]]) if plate_deletion_states.size else None
            ),
            "final_deleted_plate_element_ids": np.asarray(
                arrays[ArrayType.element_solid_ids], dtype=np.int64
            )[plate_mask][deleted_plate[-1]].tolist(),
        }
    return {
        "status": (
            "pass_adaptive_sph_capability_and_mass_gate"
            if adaptive_gate_pass
            else "fail_one_or_more_preregistered_adaptive_sph_engineering_gates"
        ),
        "d3plot": {
            "state_count": state_count,
            "first_time_ms": float(times[0]),
            "last_time_ms": float(times[-1]),
            "solid_count": EXPECTED_SOLID_COUNT,
            "part_ids": part_ids.tolist(),
            "coordinate_mode": coordinate_mode,
        },
        "conversion": {
            "preallocated_sph_count": EXPECTED_PREALLOCATED_SPH_COUNT,
            "first_activation_state": int(activation_states[0]),
            "first_activation_time_ms": float(times[activation_states[0]]),
            "final_converted_solid_count": int(deleted_counts[-1]),
            "final_active_sph_count": int(active_counts[-1]),
            "particles_per_converted_solid": 8,
            "statewise_exact_activation_correspondence": bool(np.all(activation_correspondence)),
            "plate_deleted_element_count": int(deleted_plate_counts[-1]),
            **diagnostic_plate_fields,
            "sph_deleted_particle_count": int(np.count_nonzero(sph_deletion)),
        },
        "mass_accounting": {
            "initial_fragment_reference_mass_kg": initial_fragment_mass,
            "final_alive_solid_mass_kg": float(alive_solid_mass[-1]),
            "final_converted_reference_mass_kg": float(converted_reference_mass[-1]),
            "final_active_sph_mass_kg": float(active_sph_mass[-1]),
            "maximum_absolute_combined_mass_relative_error": max_combined_error,
            "maximum_absolute_conversion_mass_relative_error": max_conversion_error,
            "preregistered_maximum_absolute_relative_error": 0.01,
            "gate": "pass" if mass_gate_pass else "fail",
        },
        "active_sph": {
            "all_positions_finite": all_active_positions_finite,
            "all_velocities_finite": all_active_velocities_finite,
            "field_ranges": active_ranges,
        },
        "capture_plane": {
            "z_mm": CAPTURE_PLANE_Z_MM,
            "solid_crossing_event_count": int(np.count_nonzero(solid_crossings)),
            "sph_crossing_event_count": int(np.count_nonzero(sph_crossings)),
            "final_minimum_alive_solid_centroid_z_mm": float(
                np.min(solid_centroids[-1, final_alive_solid, 2])
            ),
            "final_minimum_active_sph_z_mm": float(np.min(sph_positions[-1, final_active_sph, 2])),
            "final_downward_alive_solid_count": final_downward_solid_count,
            "final_downward_active_sph_count": final_downward_sph_count,
            "status": capture_status,
            "primary_qoi_available": False,
            "reason": capture_reason,
        },
        "matsum": {
            "part_ids": matsum_ids.tolist(),
            "state_count": int(matsum_times.size),
            "last_time_ms": float(matsum_times[-1]),
            "sph_part_2003_present": True,
            "initial_source_sph_kinetic_energy_difference": initial_source_sph_kinetic_difference,
            "initial_source_sph_internal_energy_difference": initial_source_sph_internal_difference,
            "source_part_2002_and_sph_part_2003_duplicate_at_initial_state": (
                source_sph_columns_duplicate_at_initial_state
            ),
            "naive_sum_across_source_part_2002_and_sph_part_2003_is_admissible": False,
            "combined_energy_use": "denied_source_and_sph_columns_are_not_independent_at_initial_state",
        },
        "claim_boundary": (
            "Positive-neighbour SPH activation is admitted only because it matches eight particles per converted "
            "solid at every state and closes reference mass. MATSUM source/SPH columns must not be naively summed. "
            "No capture-plane response, physical validity, fitting or dataset admission is established."
        ),
    }


def build_validation(
    return_dir: Path,
    *,
    schema_version: str = "hf-p0-r06m-adaptive-sph-validation@1.0.0",
    strict_gates: bool = True,
    required_problem_time_ms: float = 0.04,
) -> dict:
    d3part = validate_d3part(return_dir)
    d3part["evidence_scope"] = "completed_run_cross_format_readability_and_hvar6_consistency"
    ncforc = validate_ncforc(return_dir)
    adaptive = validate_adaptive_sph(
        return_dir,
        strict_gates=strict_gates,
        required_problem_time_ms=required_problem_time_ms,
    )
    time_label = str(required_problem_time_ms).replace(".", "p")
    return {
        "schema_version": schema_version,
        "run_completion_policy": f"requires_normal_termination_and_problem_time_{time_label}_in_the_separate_return_audit",
        "adaptive_sph": adaptive,
        "d3part_full_d3plot_crosscheck": d3part,
        "ncforc": ncforc,
        "claim_boundary": (
            "This artifact validates output readability, adaptive activation and mass accounting only. "
            "Resource identity, solver termination, energy gates and research admission are separate audit layers."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate completed R06m adaptive-SPH output")
    parser.add_argument("return_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--schema-version",
        default="hf-p0-r06m-adaptive-sph-validation@1.0.0",
        help="machine-readable artifact schema label; validation logic is shared across exact-deck repeats",
    )
    parser.add_argument(
        "--diagnostic-mode",
        action="store_true",
        help="emit a complete failed-gate artifact instead of raising on plate deletion or other engineering gate failures",
    )
    parser.add_argument(
        "--required-problem-time-ms",
        type=float,
        default=0.04,
        help="required final D3PLOT time in ms (default preserves the R06m/R06n 0.04 ms contract)",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    return_dir = args.return_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(return_dir):
        raise ValueError("analysis output must be outside the immutable return directory")
    result = build_validation(
        return_dir,
        schema_version=args.schema_version,
        strict_gates=not args.diagnostic_mode,
        required_problem_time_ms=args.required_problem_time_ms,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
