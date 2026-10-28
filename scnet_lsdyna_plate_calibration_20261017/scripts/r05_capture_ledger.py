from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CaptureThresholds:
    initial_discrete_mass_relative_error_max: float = 0.005
    mass_accounting_relative_error_max: float = 1e-6
    temporal_ambiguity_mass_fraction_max: float = 0.01
    deleted_upstream_mass_fraction_max: float = 0.05
    tail_increment_qM_max: float = 0.01
    tail_increment_qP_max: float = 0.01
    tail_increment_qE_max: float = 0.01
    tail_qV_relative_range_max: float = 0.05
    confirmed_perforation_qM_min: float = 0.02
    confirmed_perforation_qP_min: float = 0.01
    confirmed_nonperforation_qM_max: float = 0.005
    confirmed_nonperforation_qP_max: float = 0.005
    confirmed_nonperforation_worst_case_crossing_mass_max: float = 0.005


def _relative_range(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size <= 1:
        return 0.0
    mean = float(np.mean(np.abs(finite)))
    return float(np.ptp(finite) / mean) if mean > 0.0 else 0.0


def classify_numerical_perforation(
    qM: float,
    qP: float,
    *,
    unresolved_crossing_mass_fraction: float = 0.0,
    thresholds: CaptureThresholds = CaptureThresholds(),
) -> str:
    if qM >= thresholds.confirmed_perforation_qM_min and qP >= thresholds.confirmed_perforation_qP_min:
        return "confirmed_perforation"
    worst_case_crossing_mass = qM + unresolved_crossing_mass_fraction
    if (
        qM <= thresholds.confirmed_nonperforation_qM_max
        and qP <= thresholds.confirmed_nonperforation_qP_max
        and worst_case_crossing_mass
        <= thresholds.confirmed_nonperforation_worst_case_crossing_mass_max
    ):
        return "confirmed_nonperforation"
    return "transition_or_indeterminate"


def build_capture_ledger(
    *,
    times_ms: np.ndarray,
    node_positions_mm: np.ndarray,
    node_velocities_mm_per_ms: np.ndarray,
    element_connectivity: np.ndarray,
    element_alive: np.ndarray,
    element_reference_mass_kg: np.ndarray,
    expected_reference_mass_kg: float,
    capture_plane_z_mm: float,
    impact_speed_mps: float,
    maximum_unambiguous_step_mm: float,
    tail_window_fraction: float = 0.10,
    thresholds: CaptureThresholds = CaptureThresholds(),
) -> dict:
    """Build a mutually exclusive initial-mass ledger from fragment-only states.

    The algorithm follows element identities and their initial reference mass. It never
    substitutes the terminal surviving subset for the original fragment population.
    A crossing is accepted only when the element is alive in both bracketing states,
    moves in negative z, and advances no farther than the frozen temporal-resolution
    bound. Deleted or temporally unresolved mass remains explicit in the ledger. The
    supplied element population is also checked against the independently expected
    reference mass so a truncated D3PART read cannot redefine the qM denominator.
    """
    times = np.asarray(times_ms, dtype=float)
    positions = np.asarray(node_positions_mm, dtype=float)
    velocities = np.asarray(node_velocities_mm_per_ms, dtype=float)
    connectivity_input = np.asarray(element_connectivity)
    alive_input = np.asarray(element_alive)
    masses = np.asarray(element_reference_mass_kg, dtype=float)
    expected_mass = float(expected_reference_mass_kg)
    capture_plane = float(capture_plane_z_mm)
    impact_speed = float(impact_speed_mps)
    maximum_step = float(maximum_unambiguous_step_mm)
    tail_fraction = float(tail_window_fraction)
    threshold_values = asdict(thresholds)

    if not np.all(np.isfinite(times)):
        raise ValueError("times_ms must contain only finite values")
    if not np.all(np.isfinite(positions)):
        raise ValueError("node_positions_mm must contain only finite values")
    if not np.all(np.isfinite(velocities)):
        raise ValueError("node_velocities_mm_per_ms must contain only finite values")
    if not np.all(np.isfinite(masses)):
        raise ValueError("element_reference_mass_kg must contain only finite values")
    if not all(np.isfinite(value) for value in threshold_values.values()):
        raise ValueError("capture thresholds must contain only finite values")
    if any(value < 0.0 for value in threshold_values.values()):
        raise ValueError("capture thresholds must be nonnegative")
    scalar_inputs = {
        "expected_reference_mass_kg": expected_mass,
        "capture_plane_z_mm": capture_plane,
        "impact_speed_mps": impact_speed,
        "maximum_unambiguous_step_mm": maximum_step,
        "tail_window_fraction": tail_fraction,
    }
    if not all(np.isfinite(value) for value in scalar_inputs.values()):
        raise ValueError("capture-ledger scalar inputs must contain only finite values")

    if not np.issubdtype(connectivity_input.dtype, np.number):
        raise ValueError("element_connectivity must be numeric")
    if not np.all(np.isfinite(connectivity_input)):
        raise ValueError("element_connectivity must contain only finite values")
    if not np.all(connectivity_input == np.floor(connectivity_input)):
        raise ValueError("element_connectivity must contain integer node indices")
    connectivity = connectivity_input.astype(np.int64, copy=False)

    if np.issubdtype(alive_input.dtype, np.bool_):
        alive = alive_input.astype(bool, copy=False)
    elif np.issubdtype(alive_input.dtype, np.number):
        if not np.all(np.isfinite(alive_input)):
            raise ValueError("element_alive must contain only finite values")
        alive = alive_input.astype(bool)
    else:
        raise ValueError("element_alive must be boolean or numeric")

    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("times_ms must be a strictly increasing vector with at least two states")
    if positions.shape != velocities.shape or positions.ndim != 3 or positions.shape[0] != times.size:
        raise ValueError("node position/velocity arrays must have shape [state,node,3]")
    if positions.shape[2] != 3:
        raise ValueError("node position/velocity arrays require three components")
    if connectivity.ndim != 2 or connectivity.shape[0] == 0 or connectivity.shape[1] != 8:
        raise ValueError("element_connectivity must have shape [element,8]")
    if alive.shape != (times.size, connectivity.shape[0]):
        raise ValueError("element_alive must have shape [state,element]")
    if masses.shape != (connectivity.shape[0],) or np.any(masses <= 0.0):
        raise ValueError("element reference masses must be positive and match the element count")
    if connectivity.min() < 0 or connectivity.max() >= positions.shape[1]:
        raise ValueError("element connectivity references a missing node index")
    if expected_mass <= 0.0:
        raise ValueError("expected_reference_mass_kg must be positive")
    if impact_speed <= 0.0 or maximum_step <= 0.0:
        raise ValueError("impact speed and temporal-resolution bound must be positive")
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("tail_window_fraction must be between zero and one")

    element_positions = positions[:, connectivity, :].mean(axis=2)
    element_velocities = velocities[:, connectivity, :].mean(axis=2)
    element_count = connectivity.shape[0]
    category = np.full(element_count, "", dtype=object)
    crossing_time = np.full(element_count, np.nan)
    crossing_velocity = np.full((element_count, 3), np.nan)

    for element in range(element_count):
        crossed = False
        if not alive[0, element] or element_positions[0, element, 2] <= capture_plane:
            category[element] = "temporal_ambiguity"
            continue
        for state in range(1, times.size):
            was_alive = alive[state - 1, element]
            is_alive = alive[state, element]
            previous_z = element_positions[state - 1, element, 2]
            current_z = element_positions[state, element, 2]
            if was_alive and is_alive and not crossed and previous_z > capture_plane >= current_z:
                step = previous_z - current_z
                if step <= 0.0 or step > maximum_step:
                    category[element] = "temporal_ambiguity"
                    break
                alpha = (previous_z - capture_plane) / step
                interpolated_velocity = (
                    element_velocities[state - 1, element]
                    + alpha * (element_velocities[state, element] - element_velocities[state - 1, element])
                )
                if interpolated_velocity[2] >= 0.0:
                    category[element] = "temporal_ambiguity"
                    break
                crossing_time[element] = times[state - 1] + alpha * (times[state] - times[state - 1])
                crossing_velocity[element] = interpolated_velocity
                crossed = True
            if was_alive and not is_alive:
                if crossed:
                    category[element] = "deleted_after_crossing"
                elif previous_z > capture_plane >= current_z:
                    # The element disappeared during the same saved-state interval in
                    # which its centroid may have crossed. There is no alive lower
                    # bracket from which to interpolate a defensible crossing event.
                    category[element] = "temporal_ambiguity"
                else:
                    category[element] = "deleted_upstream"
                break
        if category[element] == "":
            if crossed:
                category[element] = "alive_crossed"
            elif alive[-1, element]:
                category[element] = "alive_upstream"
            else:
                category[element] = "deleted_upstream"

    categories = (
        "alive_crossed",
        "alive_upstream",
        "deleted_upstream",
        "deleted_after_crossing",
        "temporal_ambiguity",
    )
    total_mass = float(masses.sum())
    if not np.isfinite(total_mass):
        raise ValueError("summed element reference mass is non-finite")
    initial_discrete_mass_relative_error = abs(total_mass - expected_mass) / expected_mass
    mass_by_category = {
        name: float(masses[category == name].sum()) for name in categories
    }
    # Qualification fractions use the independently identity-bound expected mass.
    # The observed D3PART population must never redefine its own denominator.
    fraction_by_category = {
        name: value / expected_mass for name, value in mass_by_category.items()
    }
    accounted_mass = sum(mass_by_category.values())
    accounting_error = abs(accounted_mass - total_mass) / total_mass

    captured = np.isin(category, ["alive_crossed", "deleted_after_crossing"])
    captured_mass = float(masses[captured].sum())
    directional_speed = np.maximum(-crossing_velocity[:, 2], 0.0)
    captured_momentum = float(np.sum(masses[captured] * directional_speed[captured]))
    with np.errstate(over="ignore", invalid="ignore"):
        captured_energy = float(
            np.sum(0.5 * masses[captured] * np.sum(crossing_velocity[captured] ** 2, axis=1))
        )
    initial_momentum = expected_mass * impact_speed
    initial_energy = 0.5 * expected_mass * impact_speed**2
    qM = captured_mass / expected_mass
    qP = captured_momentum / initial_momentum
    qE = captured_energy / initial_energy
    qV_over_v0 = (
        captured_momentum / captured_mass / impact_speed
        if qM >= thresholds.confirmed_perforation_qM_min
        else None
    )

    tail_start = times[-1] - tail_fraction * (times[-1] - times[0])
    tail_events = captured & (crossing_time >= tail_start)
    tail_increment_qM = float(masses[tail_events].sum() / expected_mass)
    tail_increment_qP = float(
        np.sum(masses[tail_events] * directional_speed[tail_events]) / initial_momentum
    )
    with np.errstate(over="ignore", invalid="ignore"):
        tail_increment_qE = float(
            np.sum(0.5 * masses[tail_events] * np.sum(crossing_velocity[tail_events] ** 2, axis=1))
            / initial_energy
        )
    cumulative_qv: list[float] = []
    for state_time in times[times >= tail_start]:
        seen = captured & (crossing_time <= state_time)
        seen_mass = float(masses[seen].sum())
        cumulative_qv.append(
            float(np.sum(masses[seen] * directional_speed[seen]) / seen_mass / impact_speed)
            if seen_mass > 0.0
            else np.nan
        )
    tail_qV_relative_range = _relative_range(np.asarray(cumulative_qv, dtype=float))

    missing_reference_mass_fraction = max(0.0, expected_mass - total_mass) / expected_mass
    unresolved_crossing_mass_fraction = (
        fraction_by_category["temporal_ambiguity"]
        + fraction_by_category["deleted_upstream"]
        + missing_reference_mass_fraction
    )
    worst_case_crossing_mass_fraction = qM + unresolved_crossing_mass_fraction
    derived_scalars = {
        "reference_mass_kg": total_mass,
        "initial_discrete_mass_relative_error": initial_discrete_mass_relative_error,
        "mass_accounting_relative_error": accounting_error,
        "qM": qM,
        "qP": qP,
        "qE": qE,
        "tail_start_time_ms": float(tail_start),
        "tail_increment_qM": tail_increment_qM,
        "tail_increment_qP": tail_increment_qP,
        "tail_increment_qE": tail_increment_qE,
        "tail_qV_relative_range": tail_qV_relative_range,
        "unresolved_crossing_mass_fraction": unresolved_crossing_mass_fraction,
        "missing_reference_mass_fraction": missing_reference_mass_fraction,
        "worst_case_crossing_mass_fraction": worst_case_crossing_mass_fraction,
        **{f"mass_{name}_kg": value for name, value in mass_by_category.items()},
        **{f"mass_fraction_{name}": value for name, value in fraction_by_category.items()},
    }
    if qV_over_v0 is not None:
        derived_scalars["qV_over_v0"] = qV_over_v0
    nonfinite_outputs = [
        name for name, value in derived_scalars.items() if not np.isfinite(value)
    ]
    if nonfinite_outputs:
        raise ValueError(
            "capture-ledger derived outputs must be finite; bad fields="
            + ",".join(nonfinite_outputs)
        )

    unavailable_reasons: list[str] = []
    if (
        initial_discrete_mass_relative_error
        > thresholds.initial_discrete_mass_relative_error_max
    ):
        unavailable_reasons.append("initial_discrete_mass_relative_error")
    if accounting_error > thresholds.mass_accounting_relative_error_max:
        unavailable_reasons.append("mass_accounting_error")
    if fraction_by_category["temporal_ambiguity"] > thresholds.temporal_ambiguity_mass_fraction_max:
        unavailable_reasons.append("temporal_ambiguity_mass")
    if fraction_by_category["deleted_upstream"] > thresholds.deleted_upstream_mass_fraction_max:
        unavailable_reasons.append("deleted_upstream_mass")
    if tail_increment_qM > thresholds.tail_increment_qM_max:
        unavailable_reasons.append("unstable_tail_qM")
    if tail_increment_qP > thresholds.tail_increment_qP_max:
        unavailable_reasons.append("unstable_tail_qP")
    if tail_increment_qE > thresholds.tail_increment_qE_max:
        unavailable_reasons.append("unstable_tail_qE")
    if tail_qV_relative_range > thresholds.tail_qV_relative_range_max:
        unavailable_reasons.append("unstable_tail_qV")

    available = not unavailable_reasons
    classification = (
        classify_numerical_perforation(
            qM,
            qP,
            unresolved_crossing_mass_fraction=unresolved_crossing_mass_fraction,
            thresholds=thresholds,
        )
        if available
        else "unavailable"
    )
    return {
        "schema_version": "hf-p0-capture-ledger-result@1.1.0",
        "provenance": {
            "finite_input_validation": "pass",
            "state_count": int(times.size),
            "node_count": int(positions.shape[1]),
            "element_count": int(element_count),
            "expected_reference_mass_kg": expected_mass,
            "capture_plane_z_mm": capture_plane,
            "impact_speed_mps": impact_speed,
            "maximum_unambiguous_step_mm": maximum_step,
            "tail_window_fraction": tail_fraction,
            "thresholds": threshold_values,
        },
        "primary_qoi_available": available,
        "unavailable_reasons": unavailable_reasons,
        "classification": classification,
        "qM": qM if available else None,
        "qP": qP if available else None,
        "qE": qE if available else None,
        "qV_over_v0": qV_over_v0 if available else None,
        "diagnostic_values_before_availability_gate": {
            "qM": qM,
            "qP": qP,
            "qE": qE,
            "qV_over_v0": qV_over_v0,
        },
        "reference_mass_kg": total_mass,
        "expected_reference_mass_kg": expected_mass,
        "initial_discrete_mass_relative_error": initial_discrete_mass_relative_error,
        "mass_by_ledger_state_kg": mass_by_category,
        "mass_fraction_by_ledger_state": fraction_by_category,
        "mass_accounting_relative_error": accounting_error,
        "unresolved_crossing_mass_fraction": unresolved_crossing_mass_fraction,
        "missing_reference_mass_fraction": missing_reference_mass_fraction,
        "worst_case_crossing_mass_fraction": worst_case_crossing_mass_fraction,
        "tail_window": {
            "start_time_ms": float(tail_start),
            "increment_qM": tail_increment_qM,
            "increment_qP": tail_increment_qP,
            "increment_qE": tail_increment_qE,
            "qV_relative_range": tail_qV_relative_range,
        },
        "crossing_event_count": int(np.count_nonzero(captured)),
        "element_count": int(element_count),
    }
