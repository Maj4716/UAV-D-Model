"""Stratified opt-in fragment micro-audit on frozen train cases only.

Two lexicographically fixed cases are selected from every region/profile cell.
The original case, seed and R=8 are retained; only fragment-audit capture is
enabled.  Archived train results are used to prove numerical equivalence.  Test
result lines are never decoded.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.engine import DamageEngine
from damage_engine.fragments import (
    build_taylor_shapiro_ring_layout,
    taylor_shapiro_angles,
)
from damage_engine.fragment_audit import (
    REACHABILITY_CATEGORIES,
    diagnose_front_axis_semantics,
    extract_warhead_target_matching_observations,
    summarize_component_reachability,
)
from damage_engine.registry import PhysicsRegistry
from damage_engine.utils import content_hash, file_hash, normalize, quaternion_matrix_xyzw


_ENGINE: DamageEngine | None = None
_REGISTRY: PhysicsRegistry | None = None


def _initialize_worker(target_path: str, physics_path: str) -> None:
    global _ENGINE, _REGISTRY
    target = json.loads(Path(target_path).read_text(encoding="utf-8-sig"))
    _REGISTRY = PhysicsRegistry.from_file(physics_path)
    _ENGINE = DamageEngine(target, _REGISTRY)


def _finite_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
            "minimum": None,
            "p10": None,
            "median": None,
            "p90": None,
            "maximum": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "minimum": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def _rayleigh_ks(values: list[float] | np.ndarray, sigma: float) -> float | None:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.size == 0 or sigma <= 0.0:
        return None
    empirical_upper = np.arange(1, array.size + 1) / array.size
    empirical_lower = np.arange(0, array.size) / array.size
    reference_cdf = 1.0 - np.exp(-np.square(array) / (2.0 * sigma**2))
    return float(
        max(
            np.max(np.abs(empirical_upper - reference_cdf)),
            np.max(np.abs(empirical_lower - reference_cdf)),
        )
    )


def _mean_or_zero(record: dict[str, Any]) -> float:
    value = record.get("mean")
    return 0.0 if value is None else float(value)


def _path_count_template() -> dict[str, int]:
    return {
        "fragment_trace": 0,
        "with_any_geometry_candidate": 0,
        "with_any_processed_intersection": 0,
        "with_armor_candidate": 0,
        "with_component_candidate": 0,
        "with_armor_processed": 0,
        "with_component_processed": 0,
        "blocked": 0,
        "completed_geometry_path": 0,
        "component_processed_before_first_armor": 0,
        "component_processed_with_no_armor": 0,
    }


def _accumulate_path_counts(counts: dict[str, int], event: dict[str, Any]) -> None:
    counts["fragment_trace"] += 1
    candidate_path = event["geometry_candidate_path"]
    processed_path = event["processed_path"]
    counts["with_any_geometry_candidate"] += int(bool(candidate_path))
    counts["with_any_processed_intersection"] += int(bool(processed_path))
    counts["with_armor_candidate"] += int(
        any(item["role"] == "armor" for item in candidate_path)
    )
    counts["with_component_candidate"] += int(
        any(item["role"] == "component" for item in candidate_path)
    )
    counts["with_armor_processed"] += int(
        any(item["role"] == "armor" for item in processed_path)
    )
    counts["with_component_processed"] += int(
        any(item["role"] == "component" for item in processed_path)
    )
    counts["blocked"] += int(event.get("blocking_object_id") is not None)
    counts["completed_geometry_path"] += int(
        event.get("completed_geometry_path") is True
    )
    roles = [item["role"] for item in processed_path]
    if "component" in roles and "armor" in roles:
        counts["component_processed_before_first_armor"] += int(
            roles.index("component") < roles.index("armor")
        )
    elif "component" in roles:
        counts["component_processed_with_no_armor"] += 1


def _gurney_speed(case: dict[str, Any], source_model: dict[str, Any]) -> float:
    fragment_source = case["warhead"]["fragment_source"]
    count = int(fragment_source["total_count"])
    charge = float(case["warhead"]["charge_mass_kg"])
    fragment_mass = count * float(
        fragment_source["mass_distribution"]["parameters"]["value_kg"]
    )
    casing = float(case["warhead"]["casing_mass_kg"])
    if source_model["casing_mass_basis"] == "excludes_fragments":
        casing += fragment_mass
    return float(fragment_source["gurney_constant_mps"]) / math.sqrt(
        casing / charge + 0.5
    )


def _taylor_shapiro_nominal_directions(
    case: dict[str, Any], source_model: dict[str, Any]
) -> np.ndarray:
    count = int(case["warhead"]["fragment_source"]["total_count"])
    layout = build_taylor_shapiro_ring_layout(case, source_model, count)
    axis = layout.body_axis
    nominal = np.empty_like(layout.local_normals_body)
    gurney_speed = _gurney_speed(case, source_model)
    detonation_speed = float(
        case["warhead"]["fragment_source"]["detonation_velocity_mps"]
    )
    for index, (normal, wave) in enumerate(
        zip(layout.local_normals_body, layout.wave_directions_body, strict=True)
    ):
        phi1 = math.acos(float(np.clip(np.dot(axis, normal), -1.0, 1.0)))
        phi2 = math.acos(float(np.clip(np.dot(axis, wave), -1.0, 1.0)))
        _, shapiro = taylor_shapiro_angles(
            gurney_speed, detonation_speed, phi1, phi2
        )
        radial = normalize(normal - float(np.dot(axis, normal)) * axis)
        polar_increase = -math.sin(phi1) * axis + math.cos(phi1) * radial
        nominal[index] = normalize(
            math.cos(shapiro) * normal + math.sin(shapiro) * polar_increase
        )
    rotation = quaternion_matrix_xyzw(
        case["encounter"]["body_to_target_orientation_xyzw"]
    )
    values = (rotation @ nominal.T).T
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _numerical_equivalence(archived: dict[str, Any], rerun: dict[str, Any]) -> dict[str, Any]:
    summary_error = max(
        abs(_mean_or_zero(rerun["summary"][name]) - _mean_or_zero(value))
        for name, value in archived["summary"].items()
    )
    component_error = 0.0
    missingness_mismatch = 0
    for component_id, archived_component in archived["components"].items():
        rerun_component = rerun["components"][component_id]
        for name, archived_metric in archived_component["fragment_metrics"].items():
            left = archived_metric.get("mean")
            right = rerun_component["fragment_metrics"][name].get("mean")
            if left is None or right is None:
                missingness_mismatch += int(left != right)
            else:
                component_error = max(component_error, abs(float(left) - float(right)))
        for state in ("degraded", "failed"):
            component_error = max(
                component_error,
                abs(
                    _mean_or_zero(
                        archived_component["damage_states"]["fragment"][state]
                    )
                    - _mean_or_zero(
                        rerun_component["damage_states"]["fragment"][state]
                    )
                ),
            )
    tree_error = max(
        abs(
            _mean_or_zero(archived["damage_tree"]["outputs"][name])
            - _mean_or_zero(rerun["damage_tree"]["outputs"][name])
        )
        for name in archived["damage_tree"]["outputs"]
    )
    return {
        "summary_max_abs_error": summary_error,
        "component_max_abs_error": component_error,
        "tree_max_abs_error": tree_error,
        "metric_missingness_mismatch_count": missingness_mismatch,
    }


def _worker(payload: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    if _ENGINE is None or _REGISTRY is None:
        raise RuntimeError("worker was not initialized")
    original, archived = payload
    if original["analysis"].get("capture_fragment_audit") is not False:
        raise RuntimeError("selected case did not start with capture_fragment_audit=false")
    if int(original["analysis"]["replicates"]) != 8:
        raise RuntimeError("micro-audit is restricted to formal train R8 cases")
    case = deepcopy(original)
    case["analysis"]["capture_fragment_audit"] = True
    rerun = _ENGINE.run(case, fail_closed=True).to_dict()
    audit = rerun["fragment_audit"]
    component_ids = list(_ENGINE.compiled_target.component_order)
    equivalence = _numerical_equivalence(archived, rerun)

    source = _REGISTRY.fragment_source(case["warhead"]["fragment_source"]["model_id"])
    axis = diagnose_front_axis_semantics(case, source)
    source_axis = np.asarray(axis["source_axis_target_xyz"], dtype=np.float64)
    expected_mass = float(
        case["warhead"]["fragment_source"]["mass_distribution"]["parameters"]["value_kg"]
    )
    valid_mass = case["warhead"]["fragment_source"]["mass_distribution"].get(
        "valid_range_kg"
    )
    valid_speed = source["valid_speed_mps"]
    projectile_velocity = np.asarray(
        case["encounter"]["velocity_target_mps"], dtype=np.float64
    )
    expected_gurney_speed = _gurney_speed(case, source)
    nominal_directions = _taylor_shapiro_nominal_directions(case, source)
    spread_sigma_rad = math.radians(float(source["direction"]["spread_sigma_deg"]))

    path_counts = _path_count_template()
    replicate_rows: list[dict[str, Any]] = []
    masses: list[float] = []
    target_frame_speeds: list[float] = []
    detonation_frame_speeds: list[float] = []
    detonation_axial_cosines: list[float] = []
    taylor_shapiro_angular_deviations_rad: list[float] = []
    direction_norm_errors: list[float] = []
    calibration = {
        role: {
            "count": 0,
            "brier_numerator": 0.0,
            "bernoulli_variance_numerator": 0.0,
            "predicted_sum": 0.0,
            "outcome_sum": 0.0,
            "bin_count": [0] * 10,
            "bin_predicted_sum": [0.0] * 10,
            "bin_outcome_sum": [0.0] * 10,
        }
        for role in ("armor", "component")
    }
    armor_log_v50: list[float] = []
    state_log_e50: dict[str, list[float]] = defaultdict(list)
    state_eligible: dict[str, int] = defaultdict(int)
    state_below: dict[str, int] = defaultdict(int)
    state_nonpositive: dict[str, int] = defaultdict(int)

    for wrapper in audit:
        replicate_path_counts = _path_count_template()
        replicate_calibration = {
            role: {
                "count": 0,
                "predicted_sum": 0.0,
                "outcome_sum": 0.0,
                "brier_numerator": 0.0,
                "bernoulli_variance_numerator": 0.0,
            }
            for role in ("armor", "component")
        }
        events = wrapper["events"]
        for event in events:
            record_type = event.get("record_type")
            if record_type == "fragment_trace":
                _accumulate_path_counts(path_counts, event)
                _accumulate_path_counts(replicate_path_counts, event)
                mass = float(event["mass_kg"])
                target_speed = float(event["initial_speed_mps"])
                target_direction = np.asarray(
                    event["direction_target_xyz"], dtype=np.float64
                )
                target_velocity = target_speed * target_direction
                detonation_velocity = target_velocity - projectile_velocity
                detonation_speed = float(np.linalg.norm(detonation_velocity))
                detonation_direction = detonation_velocity / detonation_speed
                fragment_id = int(event["fragment_id"])
                masses.append(mass)
                target_frame_speeds.append(target_speed)
                detonation_frame_speeds.append(detonation_speed)
                detonation_axial_cosines.append(
                    float(np.dot(detonation_direction, source_axis))
                )
                taylor_shapiro_angular_deviations_rad.append(
                    math.acos(
                        float(
                            np.clip(
                                np.dot(
                                    detonation_direction,
                                    nominal_directions[fragment_id],
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
                direction_norm_errors.append(
                    abs(float(np.linalg.norm(target_direction)) - 1.0)
                )
            elif record_type == "intersection":
                role = str(event["role"])
                if role not in calibration:
                    continue
                probability = float(event["penetration_probability"])
                outcome = float(event["penetrated"])
                bin_index = min(9, max(0, int(probability * 10.0)))
                item = calibration[role]
                item["count"] += 1
                item["brier_numerator"] += (probability - outcome) ** 2
                item["bernoulli_variance_numerator"] += probability * (1.0 - probability)
                item["predicted_sum"] += probability
                item["outcome_sum"] += outcome
                item["bin_count"][bin_index] += 1
                item["bin_predicted_sum"][bin_index] += probability
                item["bin_outcome_sum"][bin_index] += outcome
                replicate_item = replicate_calibration[role]
                replicate_item["count"] += 1
                replicate_item["predicted_sum"] += probability
                replicate_item["outcome_sum"] += outcome
                replicate_item["brier_numerator"] += (probability - outcome) ** 2
                replicate_item["bernoulli_variance_numerator"] += probability * (
                    1.0 - probability
                )
                if role == "armor":
                    speed = float(event["arrival_speed_mps"])
                    v50 = float(event["v50_mps"])
                    if speed > 0.0 and v50 > 0.0:
                        armor_log_v50.append(math.log(speed / v50))
                else:
                    energy = float(event["damage_energy_j"])
                    parameters = event["damage_state_response_parameters"]
                    for state, state_parameters in parameters.items():
                        threshold = float(state_parameters["damage_energy_50_j"])
                        state_eligible[state] += 1
                        state_below[state] += int(energy < threshold)
                        state_nonpositive[state] += int(energy <= 0.0)
                        if energy > 0.0:
                            state_log_e50[state].append(math.log(energy / threshold))

        replicate_reachability = summarize_component_reachability(
            [wrapper], component_ids
        )
        replicate_reachability_counts = {
            category: int(
                sum(
                    component["counts"][category]
                    for component in replicate_reachability["components"].values()
                )
            )
            for category in REACHABILITY_CATEGORIES
        }
        for replicate_item in replicate_calibration.values():
            denominator = float(replicate_item["count"])
            replicate_item.update(
                {
                    "predicted_mean": (
                        replicate_item["predicted_sum"] / denominator
                        if denominator > 0.0
                        else None
                    ),
                    "observed_penetration_rate": (
                        replicate_item["outcome_sum"] / denominator
                        if denominator > 0.0
                        else None
                    ),
                    "raw_brier": (
                        replicate_item["brier_numerator"] / denominator
                        if denominator > 0.0
                        else None
                    ),
                }
            )
        replicate_rows.append(
            {
                "replicate": int(wrapper["replicate"]),
                "fragment_trace_count": replicate_path_counts["fragment_trace"],
                "expected_fragment_trace_count": int(
                    case["warhead"]["fragment_source"]["total_count"]
                ),
                "path_counts": replicate_path_counts,
                "reachability_denominator": int(
                    replicate_path_counts["fragment_trace"] * len(component_ids)
                ),
                "reachability_counts": replicate_reachability_counts,
                "penetration_calibration_sufficient_statistics": replicate_calibration,
            }
        )

    reachability = summarize_component_reachability(audit, component_ids)
    reachability_counts = {category: 0 for category in REACHABILITY_CATEGORIES}
    for component in reachability["components"].values():
        for category, count in component["counts"].items():
            reachability_counts[category] += int(count)
    reachability_by_component = {
        component_id: {
            category: int(values["counts"][category])
            for category in REACHABILITY_CATEGORIES
        }
        for component_id, values in reachability["components"].items()
    }
    for item in calibration.values():
        denominator = float(item["count"])
        item.update(
            {
                "predicted_mean": item["predicted_sum"] / denominator if denominator else None,
                "observed_penetration_rate": item["outcome_sum"] / denominator if denominator else None,
                "calibration_in_large_observed_minus_predicted": (
                    item["outcome_sum"] - item["predicted_sum"]
                )
                / denominator if denominator else None,
                "raw_brier": item["brier_numerator"] / denominator if denominator else None,
                "mean_model_bernoulli_variance": item[
                    "bernoulli_variance_numerator"
                ]
                / denominator if denominator else None,
                "excess_brier": (
                    item["brier_numerator"]
                    - item["bernoulli_variance_numerator"]
                )
                / denominator if denominator else None,
            }
        )

    matching = extract_warhead_target_matching_observations(audit)
    trace_tree_error = 0.0
    records = rerun["component_state_trace"]["records"]
    for output_name, output in rerun["damage_tree"]["outputs"].items():
        trace_mean = float(np.mean([record["system_outputs"][output_name] for record in records]))
        trace_tree_error = max(trace_tree_error, abs(trace_mean - _mean_or_zero(output)))

    rayleigh_ks = _rayleigh_ks(
        taylor_shapiro_angular_deviations_rad, spread_sigma_rad
    )

    return {
        "case_id": case["case_id"],
        "profile": case["munition"]["profile_id"],
        "region": case["metadata"]["contact_manifold_sampling"]["region_id"],
        "replicate_count": int(case["analysis"]["replicates"]),
        "aleatory_seed": int(case["analysis"]["aleatory_seed"]),
        "audit_wrapper_count": len(audit),
        "source_contract": {
            "fragment_count_per_replicate": int(
                case["warhead"]["fragment_source"]["total_count"]
            ),
            "mass_expected_kg": expected_mass,
            "mass_valid_range_kg": valid_mass,
            "speed_valid_range_mps": valid_speed,
            "expected_gurney_speed_mps": expected_gurney_speed,
            "direction_contract": source["direction"],
            "source_id": source["source_id"],
            "axis_diagnostic": axis,
        },
        "source_actual": {
            "fragment_trace_count": path_counts["fragment_trace"],
            "expected_fragment_trace_count": int(
                case["analysis"]["replicates"]
                * case["warhead"]["fragment_source"]["total_count"]
            ),
            "mass_kg": _finite_summary(masses),
            "mass_max_abs_error_from_fixed_contract": max(
                abs(value - expected_mass) for value in masses
            ),
            "target_frame_speed_mps": _finite_summary(target_frame_speeds),
            "target_frame_speed_outside_implemented_valid_range_count": sum(
                not (float(valid_speed[0]) <= value <= float(valid_speed[1]))
                for value in target_frame_speeds
            ),
            "detonation_frame_speed_mps": _finite_summary(detonation_frame_speeds),
            "detonation_frame_speed_max_abs_error_from_gurney_mps": max(
                abs(value - expected_gurney_speed)
                for value in detonation_frame_speeds
            ),
            "detonation_frame_direction_dot_source_axis": _finite_summary(
                detonation_axial_cosines
            ),
            "taylor_shapiro_angular_deviation_rad": _finite_summary(
                taylor_shapiro_angular_deviations_rad
            ),
            "taylor_shapiro_spread_rayleigh_ks": rayleigh_ks,
            "direction_norm_max_abs_error": max(direction_norm_errors),
        },
        "_source_samples": {
            "mass_kg": masses,
            "target_frame_speed_mps": target_frame_speeds,
            "detonation_frame_speed_mps": detonation_frame_speeds,
            "detonation_frame_direction_dot_source_axis": detonation_axial_cosines,
            "taylor_shapiro_angular_deviation_rad": taylor_shapiro_angular_deviations_rad,
        },
        "path_counts": path_counts,
        "replicate_sufficient_statistics": replicate_rows,
        "penetration_calibration": calibration,
        "matching": {
            "armor_log_arrival_speed_over_v50": armor_log_v50,
            "component_state_log_damage_energy_over_e50": dict(state_log_e50),
            "component_state_eligible_event_count": dict(state_eligible),
            "component_state_below_threshold_count": dict(state_below),
            "component_state_nonpositive_energy_count": dict(state_nonpositive),
            "armor_penetrating_fragment_count": int(
                matching["armor_penetrating_fragment_count"]
            ),
            "component_reached_after_armor_count": int(
                matching["component_reached_after_armor_count"]
            ),
        },
        "reachability_counts": reachability_counts,
        "reachability_by_component": reachability_by_component,
        "reachability_denominator": int(
            reachability["fragment_trace_count"] * len(component_ids)
        ),
        "archive_equivalence": equivalence,
        "trace_tree_max_abs_error": trace_tree_error,
    }


def _resample_index_matrix(
    rng: np.random.Generator,
    n: int,
    draws: int,
    strata: list[str] | None,
) -> np.ndarray:
    if strata is None:
        return rng.integers(0, n, size=(draws, n))
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        grouped[str(stratum)].append(index)
    selected: list[np.ndarray] = []
    for stratum in sorted(grouped):
        indices = np.asarray(grouped[stratum], dtype=np.int64)
        selected.append(
            rng.choice(indices, size=(draws, indices.size), replace=True)
        )
    return np.concatenate(selected, axis=1)


def _bootstrap_ratio(
    numerators: np.ndarray,
    denominators: np.ndarray,
    *,
    draws: int,
    seed: int,
    strata: list[str] | None = None,
) -> list[float] | None:
    rng = np.random.default_rng(seed)
    n = numerators.size
    index = _resample_index_matrix(rng, n, draws, strata)
    numerator = np.sum(numerators[index], axis=1)
    denominator = np.sum(denominators[index], axis=1)
    values = np.divide(
        numerator,
        denominator,
        out=np.full(draws, np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if finite.size else None


def _combine_calibration(
    cases: list[dict[str, Any]],
    role: str,
    *,
    draws: int,
    seed: int,
    strata: list[str] | None = None,
) -> dict[str, Any]:
    count = np.asarray([item["penetration_calibration"][role]["count"] for item in cases], dtype=float)
    brier_sum = np.asarray(
        [item["penetration_calibration"][role]["brier_numerator"] for item in cases],
        dtype=float,
    )
    bernoulli_variance_sum = np.asarray(
        [
            item["penetration_calibration"][role][
                "bernoulli_variance_numerator"
            ]
            for item in cases
        ],
        dtype=float,
    )
    predicted_sum = np.asarray(
        [item["penetration_calibration"][role]["predicted_sum"] for item in cases],
        dtype=float,
    )
    outcome_sum = np.asarray(
        [item["penetration_calibration"][role]["outcome_sum"] for item in cases],
        dtype=float,
    )
    bin_count = np.asarray(
        [item["penetration_calibration"][role]["bin_count"] for item in cases],
        dtype=float,
    )
    bin_predicted = np.asarray(
        [item["penetration_calibration"][role]["bin_predicted_sum"] for item in cases],
        dtype=float,
    )
    bin_outcome = np.asarray(
        [item["penetration_calibration"][role]["bin_outcome_sum"] for item in cases],
        dtype=float,
    )

    total = float(np.sum(count))
    if total <= 0.0:
        return {
            "status": "no_intersection_events",
            "event_count": 0,
            "uncertainty_unit": "case_cluster",
        }
    bins = []
    ece = 0.0
    combined_count = np.sum(bin_count, axis=0)
    combined_predicted = np.sum(bin_predicted, axis=0)
    combined_outcome = np.sum(bin_outcome, axis=0)
    for index in range(10):
        mass = float(combined_count[index])
        if mass == 0.0:
            continue
        predicted = float(combined_predicted[index] / mass)
        observed = float(combined_outcome[index] / mass)
        ece += mass / total * abs(predicted - observed)
        bins.append(
            {
                "bin": index,
                "lower": index / 10.0,
                "upper": (index + 1) / 10.0,
                "event_count": int(mass),
                "predicted_mean": predicted,
                "observed_rate": observed,
            }
        )

    rng = np.random.default_rng(seed)
    selected = _resample_index_matrix(rng, len(cases), draws, strata)
    draw_count = np.sum(count[selected], axis=1)
    valid_draw = draw_count > 0.0
    brier_draws = np.divide(
        np.sum(brier_sum[selected], axis=1),
        draw_count,
        out=np.full(draws, np.nan),
        where=valid_draw,
    )
    variance_draws = np.divide(
        np.sum(bernoulli_variance_sum[selected], axis=1),
        draw_count,
        out=np.full(draws, np.nan),
        where=valid_draw,
    )
    excess_brier_draws = brier_draws - variance_draws
    calibration_in_large_draws = np.divide(
        np.sum(outcome_sum[selected] - predicted_sum[selected], axis=1),
        draw_count,
        out=np.full(draws, np.nan),
        where=valid_draw,
    )
    draw_bin_count = np.sum(bin_count[selected], axis=1)
    draw_bin_predicted = np.sum(bin_predicted[selected], axis=1)
    draw_bin_outcome = np.sum(bin_outcome[selected], axis=1)
    bin_gap = np.divide(
        np.abs(draw_bin_predicted - draw_bin_outcome),
        draw_bin_count,
        out=np.zeros_like(draw_bin_count),
        where=draw_bin_count > 0.0,
    )
    ece_draws = np.divide(
        np.sum(draw_bin_count * bin_gap, axis=1),
        draw_count,
        out=np.full(draws, np.nan),
        where=valid_draw,
    )

    return {
        "status": "estimated",
        "event_count": int(total),
        "brier": float(np.sum(brier_sum) / total),
        "raw_brier_including_bernoulli_variance": float(np.sum(brier_sum) / total),
        "mean_model_bernoulli_variance_p_times_one_minus_p": float(
            np.sum(bernoulli_variance_sum) / total
        ),
        "excess_brier_over_model_bernoulli_variance": float(
            (np.sum(brier_sum) - np.sum(bernoulli_variance_sum)) / total
        ),
        "brier_case_cluster_bootstrap_95_interval": np.quantile(
            brier_draws[np.isfinite(brier_draws)], [0.025, 0.975]
        ).tolist(),
        "ece": float(ece),
        "ece_case_cluster_bootstrap_95_interval": np.quantile(
            ece_draws[np.isfinite(ece_draws)], [0.025, 0.975]
        ).tolist(),
        "predicted_mean": float(np.sum(predicted_sum) / total),
        "observed_penetration_rate": float(np.sum(outcome_sum) / total),
        "calibration_in_large_observed_minus_predicted": float(
            (np.sum(outcome_sum) - np.sum(predicted_sum)) / total
        ),
        "calibration_in_large_case_cluster_bootstrap_95_interval": np.quantile(
            calibration_in_large_draws[np.isfinite(calibration_in_large_draws)],
            [0.025, 0.975],
        ).tolist(),
        "excess_brier_case_cluster_bootstrap_95_interval": np.quantile(
            excess_brier_draws[np.isfinite(excess_brier_draws)],
            [0.025, 0.975],
        ).tolist(),
        "calibration_bins": bins,
        "uncertainty_unit": "case_cluster",
    }


def _combine_margin(
    cases: list[dict[str, Any]],
    accessor,
    *,
    draws: int,
    seed: int,
    strata: list[str] | None = None,
) -> dict[str, Any]:
    per_case = [np.asarray(accessor(item), dtype=float) for item in cases]
    nonempty = [array for array in per_case if array.size]
    if not nonempty:
        return {
            "status": "no_observations",
            "distribution": _finite_summary([]),
            "pooled_mean_case_cluster_bootstrap_95_interval": None,
        }
    combined = np.concatenate(nonempty)
    sums = np.asarray([float(np.sum(array)) for array in per_case])
    counts = np.asarray([float(array.size) for array in per_case])
    return {
        "status": "estimated",
        "distribution": _finite_summary(combined.tolist()),
        "pooled_mean_case_cluster_bootstrap_95_interval": _bootstrap_ratio(
            sums, counts, draws=draws, seed=seed, strata=strata
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-per-cell", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026112201)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    split_by_case: dict[str, str] = {}
    result_location: dict[str, tuple[Path, int]] = {}
    test_case_count = 0
    for shard in sorted((dataset / "formal_dataset/shards").glob("part-*")):
        with np.load(shard / "tensors.npz", allow_pickle=False) as arrays:
            ids = arrays["case_ids"].astype(str)
            splits = arrays["split_names"].astype(str)
        for index, (case_id, split) in enumerate(zip(ids, splits, strict=True)):
            split_by_case[str(case_id)] = str(split)
            result_location[str(case_id)] = (shard / "results.jsonl.gz", index)
            test_case_count += int(split == "test")

    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(
        dataset / "rounds/round-000/formal_cases.jsonl.gz", "rt", encoding="utf-8"
    ) as stream:
        for line in stream:
            case = json.loads(line)
            case_id = str(case["case_id"])
            if split_by_case[case_id] != "train":
                continue
            profile = str(case["munition"]["profile_id"])
            region = str(case["metadata"]["contact_manifold_sampling"]["region_id"])
            cells[(profile, region)].append(case)
    if len(cells) != 15:
        raise RuntimeError(f"expected 15 profile-region cells, found {len(cells)}")
    selected_cases = []
    for key in sorted(cells):
        ordered = sorted(cells[key], key=lambda item: item["case_id"])
        if len(ordered) < args.cases_per_cell:
            raise RuntimeError(f"insufficient cases in cell {key}")
        selected_cases.extend(ordered[: args.cases_per_cell])

    selected_ids = {case["case_id"] for case in selected_cases}
    archived: dict[str, dict[str, Any]] = {}
    for shard in sorted((dataset / "formal_dataset/shards").glob("part-*")):
        with np.load(shard / "tensors.npz", allow_pickle=False) as arrays:
            ids = arrays["case_ids"].astype(str)
            splits = arrays["split_names"].astype(str)
        with gzip.open(shard / "results.jsonl.gz", "rt", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                case_id = str(ids[index])
                if case_id not in selected_ids:
                    continue
                if str(splits[index]) != "train":
                    raise RuntimeError("selected audit case is not train")
                archived[case_id] = json.loads(line)
    if set(archived) != selected_ids:
        raise RuntimeError("selected archived results are incomplete")

    payloads = [(case, archived[case["case_id"]]) for case in selected_cases]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_worker,
        initargs=(str(args.target.resolve()), str(args.physics.resolve())),
    ) as executor:
        futures = {executor.submit(_worker, payload): payload[0]["case_id"] for payload in payloads}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"completed {len(results)}/{len(payloads)} {result['case_id']}", flush=True)
    results.sort(key=lambda item: item["case_id"])
    case_strata = [f"{item['region']}::{item['profile']}" for item in results]

    calibration = {
        role: _combine_calibration(
            results,
            role,
            draws=args.bootstrap_draws,
            seed=args.seed + index,
            strata=case_strata,
        )
        for index, role in enumerate(("armor", "component"))
    }
    path_count_names = list(results[0]["path_counts"])
    path_counts = {name: int(sum(item["path_counts"][name] for item in results)) for name in path_count_names}
    trace_total = path_counts["fragment_trace"]
    path_fractions = {
        name: {
            "pooled_fraction": count / trace_total,
            "case_cluster_bootstrap_95_interval": _bootstrap_ratio(
                np.asarray([item["path_counts"][name] for item in results], dtype=float),
                np.asarray([item["path_counts"]["fragment_trace"] for item in results], dtype=float),
                draws=args.bootstrap_draws,
                seed=args.seed + 100 + index,
                strata=case_strata,
            ),
        }
        for index, (name, count) in enumerate(path_counts.items())
        if name != "fragment_trace"
    }

    reachability_counts = {
        category: int(sum(item["reachability_counts"][category] for item in results))
        for category in REACHABILITY_CATEGORIES
    }
    reachability_total = int(sum(reachability_counts.values()))
    reachability = {
        category: {
            "count": count,
            "fraction": count / reachability_total,
            "case_cluster_bootstrap_95_interval": _bootstrap_ratio(
                np.asarray([item["reachability_counts"][category] for item in results], dtype=float),
                np.asarray([item["reachability_denominator"] for item in results], dtype=float),
                draws=args.bootstrap_draws,
                seed=args.seed + 300 + index,
                strata=case_strata,
            ),
        }
        for index, (category, count) in enumerate(reachability_counts.items())
    }

    component_reachability: dict[str, Any] = {}
    component_denominators = np.asarray(
        [item["source_actual"]["fragment_trace_count"] for item in results],
        dtype=float,
    )
    for component_index, component_id in enumerate(
        sorted(results[0]["reachability_by_component"])
    ):
        categories: dict[str, Any] = {}
        for category_index, category in enumerate(REACHABILITY_CATEGORIES):
            values = np.asarray(
                [
                    item["reachability_by_component"][component_id][category]
                    for item in results
                ],
                dtype=float,
            )
            categories[category] = {
                "count": int(np.sum(values)),
                "fraction": float(np.sum(values) / np.sum(component_denominators)),
                "stratified_case_cluster_bootstrap_95_interval": _bootstrap_ratio(
                    values,
                    component_denominators,
                    draws=args.bootstrap_draws,
                    seed=args.seed + 1000 + 10 * component_index + category_index,
                    strata=case_strata,
                ),
            }
        component_reachability[component_id] = {
            "fragment_component_pair_count": int(np.sum(component_denominators)),
            "categories": categories,
        }

    profile_region_cells: dict[str, Any] = {}
    for cell_index, key in enumerate(sorted(set(case_strata))):
        cell_cases = [
            item for item, stratum in zip(results, case_strata, strict=True)
            if stratum == key
        ]
        cell_path_counts = {
            name: int(sum(item["path_counts"][name] for item in cell_cases))
            for name in path_count_names
        }
        cell_trace_total = cell_path_counts["fragment_trace"]
        cell_path_fractions = {
            name: {
                "fraction": count / cell_trace_total,
                "case_cluster_bootstrap_95_interval": _bootstrap_ratio(
                    np.asarray(
                        [item["path_counts"][name] for item in cell_cases],
                        dtype=float,
                    ),
                    np.asarray(
                        [
                            item["path_counts"]["fragment_trace"]
                            for item in cell_cases
                        ],
                        dtype=float,
                    ),
                    draws=args.bootstrap_draws,
                    seed=args.seed + 2000 + 100 * cell_index + name_index,
                ),
            }
            for name_index, (name, count) in enumerate(cell_path_counts.items())
            if name != "fragment_trace"
        }
        cell_reachability_denominator = int(
            sum(item["reachability_denominator"] for item in cell_cases)
        )
        cell_reachability = {}
        for category_index, category in enumerate(REACHABILITY_CATEGORIES):
            values = np.asarray(
                [item["reachability_counts"][category] for item in cell_cases],
                dtype=float,
            )
            denominators = np.asarray(
                [item["reachability_denominator"] for item in cell_cases],
                dtype=float,
            )
            cell_reachability[category] = {
                "count": int(np.sum(values)),
                "fraction": float(np.sum(values) / np.sum(denominators)),
                "case_cluster_bootstrap_95_interval": _bootstrap_ratio(
                    values,
                    denominators,
                    draws=args.bootstrap_draws,
                    seed=args.seed + 4000 + 100 * cell_index + category_index,
                ),
            }
        profile_region_cells[key] = {
            "case_count": len(cell_cases),
            "case_ids": [item["case_id"] for item in cell_cases],
            "replicate_count": int(
                sum(item["replicate_count"] for item in cell_cases)
            ),
            "fragment_trace_count": cell_trace_total,
            "path_counts": cell_path_counts,
            "path_fractions": cell_path_fractions,
            "fragment_component_pair_count": cell_reachability_denominator,
            "reachability": cell_reachability,
            "penetration_probability_calibration": {
                role: _combine_calibration(
                    cell_cases,
                    role,
                    draws=args.bootstrap_draws,
                    seed=args.seed + 6000 + 10 * cell_index + role_index,
                )
                for role_index, role in enumerate(("armor", "component"))
            },
        }

    pooled_angular_deviation = np.concatenate(
        [
            np.asarray(
                item["_source_samples"]["taylor_shapiro_angular_deviation_rad"],
                dtype=float,
            )
            for item in results
        ]
    )
    spread_sigma_deg_values = sorted(
        {
            float(item["source_contract"]["direction_contract"]["spread_sigma_deg"])
            for item in results
        }
    )
    source_contract = {
        "selected_case_count": len(results),
        "coordinate_semantics": {
            "audit_initial_speed_and_direction": "target_frame_after_projectile_velocity_addition",
            "contract_check": "detonation_frame_vector_equals_target_frame_vector_minus_encounter_velocity_target_mps",
            "taylor_shapiro_spread_reference": "radial angular deviation from nominal direction follows Rayleigh(sigma=registered tangent-plane Gaussian coordinate sigma)",
        },
        "fragment_trace_count": trace_total,
        "expected_fragment_trace_count": int(
            sum(item["source_actual"]["expected_fragment_trace_count"] for item in results)
        ),
        "fragment_count_mismatch_case_count": int(
            sum(
                item["source_actual"]["fragment_trace_count"]
                != item["source_actual"]["expected_fragment_trace_count"]
                for item in results
            )
        ),
        "mass_max_abs_error_from_fixed_contract": float(
            max(item["source_actual"]["mass_max_abs_error_from_fixed_contract"] for item in results)
        ),
        "target_frame_speed_outside_implemented_valid_range_count": int(
            sum(
                item["source_actual"][
                    "target_frame_speed_outside_implemented_valid_range_count"
                ]
                for item in results
            )
        ),
        "detonation_frame_speed_max_abs_error_from_gurney_mps": float(
            max(
                item["source_actual"][
                    "detonation_frame_speed_max_abs_error_from_gurney_mps"
                ]
                for item in results
            )
        ),
        "taylor_shapiro_spread_rayleigh_ks_by_case": _finite_summary(
            [
                item["source_actual"]["taylor_shapiro_spread_rayleigh_ks"]
                for item in results
            ]
        ),
        "taylor_shapiro_spread_sigma_deg_values": spread_sigma_deg_values,
        "taylor_shapiro_spread_pooled_descriptive_rayleigh_ks": _rayleigh_ks(
            pooled_angular_deviation,
            math.radians(spread_sigma_deg_values[0]),
        )
        if len(spread_sigma_deg_values) == 1
        else None,
        "direction_norm_max_abs_error": float(
            max(item["source_actual"]["direction_norm_max_abs_error"] for item in results)
        ),
        "mass_kg": _combine_margin(
            results,
            lambda item: item["_source_samples"]["mass_kg"],
            draws=args.bootstrap_draws,
            seed=args.seed + 499,
            strata=case_strata,
        ),
        "target_frame_speed_mps": _combine_margin(
            results,
            lambda item: item["_source_samples"]["target_frame_speed_mps"],
            draws=args.bootstrap_draws,
            seed=args.seed + 500,
            strata=case_strata,
        ),
        "detonation_frame_speed_mps": _combine_margin(
            results,
            lambda item: item["_source_samples"]["detonation_frame_speed_mps"],
            draws=args.bootstrap_draws,
            seed=args.seed + 501,
            strata=case_strata,
        ),
        "detonation_frame_direction_dot_source_axis": _combine_margin(
            results,
            lambda item: item["_source_samples"][
                "detonation_frame_direction_dot_source_axis"
            ],
            draws=args.bootstrap_draws,
            seed=args.seed + 502,
            strata=case_strata,
        ),
        "taylor_shapiro_angular_deviation_rad": _combine_margin(
            results,
            lambda item: item["_source_samples"][
                "taylor_shapiro_angular_deviation_rad"
            ],
            draws=args.bootstrap_draws,
            seed=args.seed + 503,
            strata=case_strata,
        ),
        "per_case_contracts": {
            item["case_id"]: item["source_contract"] for item in results
        },
    }

    margin_summary = {
        "armor_log_arrival_speed_over_v50": _combine_margin(
            results,
            lambda item: item["matching"]["armor_log_arrival_speed_over_v50"],
            draws=args.bootstrap_draws,
            seed=args.seed + 600,
            strata=case_strata,
        ),
        "component_state_log_damage_energy_over_e50": {},
    }
    states = sorted(
        {
            state
            for item in results
            for state in item["matching"]["component_state_log_damage_energy_over_e50"]
        }
    )
    for index, state in enumerate(states):
        eligible = sum(
            item["matching"]["component_state_eligible_event_count"].get(state, 0)
            for item in results
        )
        below = sum(
            item["matching"]["component_state_below_threshold_count"].get(state, 0)
            for item in results
        )
        nonpositive = sum(
            item["matching"]["component_state_nonpositive_energy_count"].get(state, 0)
            for item in results
        )
        margin_summary["component_state_log_damage_energy_over_e50"][state] = {
            **_combine_margin(
                results,
                lambda item, state=state: item["matching"][
                    "component_state_log_damage_energy_over_e50"
                ].get(state, []),
                draws=args.bootstrap_draws,
                seed=args.seed + 610 + index,
                strata=case_strata,
            ),
            "eligible_event_count": eligible,
            "below_threshold_fraction": below / eligible if eligible else None,
            "nonpositive_energy_count": nonpositive,
        }

    armor_penetrating = np.asarray(
        [item["matching"]["armor_penetrating_fragment_count"] for item in results],
        dtype=float,
    )
    reached_after = np.asarray(
        [item["matching"]["component_reached_after_armor_count"] for item in results],
        dtype=float,
    )
    post_armor_reach = {
        "armor_penetrating_fragment_count": int(np.sum(armor_penetrating)),
        "component_reached_after_armor_count": int(np.sum(reached_after)),
        "fraction": (
            float(np.sum(reached_after) / np.sum(armor_penetrating))
            if np.sum(armor_penetrating) > 0.0
            else None
        ),
        "case_cluster_bootstrap_95_interval": _bootstrap_ratio(
            reached_after,
            armor_penetrating,
            draws=args.bootstrap_draws,
            seed=args.seed + 700,
            strata=case_strata,
        ),
    }

    equivalence = {
        "summary_max_abs_error": max(
            item["archive_equivalence"]["summary_max_abs_error"] for item in results
        ),
        "component_max_abs_error": max(
            item["archive_equivalence"]["component_max_abs_error"] for item in results
        ),
        "tree_max_abs_error": max(
            item["archive_equivalence"]["tree_max_abs_error"] for item in results
        ),
        "metric_missingness_mismatch_count": sum(
            item["archive_equivalence"]["metric_missingness_mismatch_count"]
            for item in results
        ),
        "trace_tree_max_abs_error": max(item["trace_tree_max_abs_error"] for item in results),
    }

    for item in results:
        item.pop("_source_samples", None)

    report = {
        "schema_version": "stratified-fragment-micro-audit@1.0.0",
        "analysis_id": "pipeline-expected-actual-20261122-v2-micro-audit",
        "analysis_scope": "fixed_train_only_two_cases_per_region_profile_cell_R8",
        "selection": {
            "method": "lexicographically_first_case_ids_within_each_frozen_train_region_profile_cell",
            "design": "15_strata_equal_allocation_two_train_cases_per_stratum",
            "cases_per_cell": args.cases_per_cell,
            "cell_count": len(cells),
            "selected_case_count": len(results),
            "selected_case_ids": [item["case_id"] for item in results],
            "test_case_count_in_source": test_case_count,
            "train_result_json_lines_decoded": len(archived),
            "test_result_json_lines_decoded": 0,
        },
        "source": {
            "dataset": str(dataset),
            "target_path": str(args.target.resolve()),
            "target_sha256": file_hash(args.target),
            "physics_path": str(args.physics.resolve()),
            "physics_sha256": file_hash(args.physics),
            "replicate_count": 8,
            "aleatory_seed_values": sorted({item["aleatory_seed"] for item in results}),
            "audit_wrapper_count_values": sorted({item["audit_wrapper_count"] for item in results}),
            "capture_change": "capture_fragment_audit_false_to_true_only",
        },
        "archive_equivalence": equivalence,
        "fragment_source_contract": source_contract,
        "fragment_path_funnel": {
            "pooled_counts": path_counts,
            "fractions": path_fractions,
            "semantics": "fragment-level executed geometry and traversal trace",
        },
        "component_reachability": {
            "fragment_component_pair_count": reachability_total,
            "categories": reachability,
            "by_component": component_reachability,
            "semantics": "mutually exclusive categories over fragment-component pairs",
        },
        "profile_region_strata": profile_region_cells,
        "penetration_probability_calibration": calibration,
        "matching_margins": margin_summary,
        "post_armor_component_reach": post_armor_reach,
        "per_case": results,
        "bootstrap": {
            "unit": "case",
            "method": "resample_cases_with_replacement_within_each_of_15_region_profile_strata",
            "draws": args.bootstrap_draws,
            "seed": args.seed,
            "interval": "percentile_95",
            "reason": "fragments and intersections within a case are dependent",
            "scope": "case-resampling sensitivity conditional on the fixed train-case selection and original fixed QMC design",
        },
        "evidence_boundaries": {
            "physical_validation": False,
            "external_parameter_validation": False,
            "selection_is_fixed_development_diagnostic_not_confirmatory_sample": True,
            "equal_allocation_pooled_rates_are_not_formal_dataset_population_weighted_rates": True,
            "test_results_used": False,
            "audit_capture_numerically_equivalent_to_archived_results": equivalence[
                "summary_max_abs_error"
            ]
            == 0.0
            and equivalence["component_max_abs_error"] == 0.0
            and equivalence["tree_max_abs_error"] == 0.0,
        },
    }
    report["report_hash"] = content_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "report_hash": report["report_hash"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
