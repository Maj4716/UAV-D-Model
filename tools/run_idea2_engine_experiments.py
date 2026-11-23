#!/usr/bin/env python
"""Executable E0/E1/E2 pilots with the full-scale design available by flags."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from damage_engine.engine import ENGINE_CODE_HASH, ENGINE_VERSION, DamageEngine
from damage_engine.fragment_audit import (
    combine_warhead_target_matching_observations,
    extract_warhead_target_matching_observations,
    summarize_component_reachability,
    summarize_warhead_target_matching_observations,
)
from damage_engine.mechanism_diagnostics import (
    paired_tost_equivalence,
    variance_decomposition,
)
from damage_engine.registry import PhysicsRegistry
from damage_engine.utils import content_hash, file_hash, quaternion_matrix_xyzw


_E0_ROLLS_RAD = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
_E0_DISTANCES_M = (8.0, 10.0, 12.0)
_E1_CANDIDATE_REPLICATES = (4, 8, 16, 32, 64, 128, 256, 512, 1024)


def _matrix_to_quaternion(rotation: np.ndarray) -> list[float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            x, y, z, w = 0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[2, 1] - rotation[1, 2]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            x, y, z, w = (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            x, y, z, w = (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale, (rotation[1, 0] - rotation[0, 1]) / scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion.tolist()


def _load_cases(path: Path, count: int) -> list[dict[str, Any]]:
    cases = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            case = json.loads(line)
            role = case.get("metadata", {}).get("sampling", {}).get("role")
            if role in {"reference_train", "reference_validation"}:
                cases.append(case)
                if len(cases) >= count:
                    break
    if len(cases) < count:
        raise RuntimeError(f"requested {count} non-test cases, found {len(cases)}")
    return cases


def _roll_variant(case: dict[str, Any], roll_delta_rad: float) -> dict[str, Any]:
    variant = deepcopy(case)
    encounter = variant["encounter"]
    rotation = quaternion_matrix_xyzw(encounter["body_to_target_orientation_xyzw"])
    body_x, body_y, body_z = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    cosine, sine = math.cos(roll_delta_rad), math.sin(roll_delta_rad)
    new_y = cosine * body_y + sine * body_z
    new_z = -sine * body_y + cosine * body_z
    encounter["body_to_target_orientation_xyzw"] = _matrix_to_quaternion(
        np.column_stack([body_x, new_y, new_z])
    )
    return variant


def _distance_variant(case: dict[str, Any], collision_distance_m: float) -> dict[str, Any]:
    variant = deepcopy(case)
    encounter = variant["encounter"]
    velocity = np.asarray(encounter["velocity_target_mps"], dtype=np.float64)
    speed = float(np.linalg.norm(velocity))
    direction = velocity / speed
    contact = np.asarray(encounter["detonation_position_m"], dtype=np.float64)
    encounter["trajectory_origin_m"] = (contact - collision_distance_m * direction).tolist()
    encounter["collision_distance_m"] = float(collision_distance_m)
    encounter["detonation_time_s"] = float(collision_distance_m / speed)
    encounter["detonation_delay_s"] = 0.0
    return variant


def _configure_case(case: dict[str, Any], *, replicates: int, seed: int, audit: bool = False) -> dict[str, Any]:
    case = deepcopy(case)
    case["analysis"]["replicates"] = int(replicates)
    case["analysis"]["adaptive_ci_half_width"] = None
    # An empty adaptive-physical-CI object is invalid by schema.  Fixed-R
    # experiments disable the adaptive controller by omitting the object.
    case["analysis"].pop("adaptive_physical_ci", None)
    case["analysis"]["aleatory_seed"] = int(seed)
    case["analysis"]["capture_fragment_audit"] = bool(audit)
    case["analysis"]["capture_replicate_trace"] = False
    case["analysis"]["capture_component_state_trace"] = False
    return case


def _metrics(result: dict[str, Any]) -> dict[str, float]:
    output = {
        "total_component_hits": float(result["summary"]["total_component_hits"]["mean"]),
        "total_component_penetrations": float(result["summary"]["total_component_penetrations"]["mean"]),
        "total_component_energy_j": float(
            sum(item["fragment_metrics"]["deposited_energy_j"]["mean"] for item in result["components"].values())
        ),
        "domain_affected_fraction": float(result["penetration_domain_coverage"]["affected_replicate_fraction"]),
    }
    output.update(
        {
            f"tree:{name}": float(statistic["mean"])
            for name, statistic in result["damage_tree"]["outputs"].items()
        }
    )
    return output


def _e0_metric_normalization(
    metric_name: str,
    reference: np.ndarray,
) -> tuple[float, str]:
    """Return the preregistered TOST scale and margin interpretation.

    Tree outputs and the domain-affected fraction are probabilities, so the
    declared 0.02 margin is an absolute probability difference.  Extensive
    count/energy outputs retain a relative margin and are normalized by the
    reference magnitude.
    """

    if metric_name == "domain_affected_fraction" or metric_name.startswith("tree:"):
        return 1.0, "absolute_probability"
    return max(float(np.mean(np.abs(reference))), 1.0e-9), "relative_to_reference_mean_abs"


def run_e0(engine: DamageEngine, cases: list[dict[str, Any]], *, replicates: int, seed: int, relative_margin: float) -> dict[str, Any]:
    rolls = _E0_ROLLS_RAD
    distances = _E0_DISTANCES_M
    records = []
    for base_index, base in enumerate(cases):
        for roll_index, roll in enumerate(rolls):
            for distance_index, distance in enumerate(distances):
                variant = _distance_variant(_roll_variant(base, roll), distance)
                variant = _configure_case(variant, replicates=replicates, seed=seed + base_index)
                variant["case_id"] = f"{base['case_id']}::E0::r{roll_index}::d{distance_index}"
                result = engine.run(variant).to_dict()
                records.append(
                    {
                        "base_index": base_index,
                        "roll_index": roll_index,
                        "roll_rad": roll,
                        "distance_index": distance_index,
                        "distance_m": distance,
                        "metrics": _metrics(result),
                    }
                )
    names = tuple(records[0]["metrics"])
    equivalence = []
    for factor, levels, selector in (
        ("roll", range(1, len(rolls)), lambda row, level: row["roll_index"] == level and row["distance_index"] == 1),
        ("distance", (0, 2), lambda row, level: row["distance_index"] == level and row["roll_index"] == 0),
    ):
        reference_selector = (
            (lambda row: row["roll_index"] == 0 and row["distance_index"] == 1)
        )
        reference_rows = sorted((row for row in records if reference_selector(row)), key=lambda row: row["base_index"])
        for level in levels:
            comparison_rows = sorted((row for row in records if selector(row, level)), key=lambda row: row["base_index"])
            for name in names:
                reference = np.asarray([row["metrics"][name] for row in reference_rows])
                comparison = np.asarray([row["metrics"][name] for row in comparison_rows])
                scale, margin_semantics = _e0_metric_normalization(name, reference)
                report = paired_tost_equivalence(comparison / scale, reference / scale, margin=relative_margin)
                equivalence.append({
                    "factor": factor,
                    "level_index": int(level),
                    "metric": name,
                    "normalization_scale": scale,
                    "margin_semantics": margin_semantics,
                    **report,
                })
    return {
        "experiment_id": "E0",
        "base_case_count": len(cases),
        "replicates": replicates,
        "rolls_rad": list(rolls),
        "distances_m": list(distances),
        "equivalence_margin": relative_margin,
        "probability_margin_semantics": "absolute_probability_difference",
        "extensive_metric_margin_semantics": "relative_to_reference_mean_absolute_value",
        "records": records,
        "equivalence": equivalence,
        "all_equivalent": all(row["equivalent"] for row in equivalence),
    }


def _e1_target_report(
    means: np.ndarray,
    variances: np.ndarray,
    sample_counts: np.ndarray,
    component_ids: tuple[str, ...],
    *,
    rmse_tolerance: float,
    conditional_on_positive: bool = False,
    total_replicates: int | None = None,
) -> dict[str, Any]:
    """Decompose one E1 target while preserving conditional-target masks."""

    mean_array = np.asarray(means, dtype=np.float64)
    variance_array = np.asarray(variances, dtype=np.float64)
    count_array = np.asarray(sample_counts, dtype=np.float64)
    if (
        mean_array.shape != variance_array.shape
        or mean_array.shape != count_array.shape
        or mean_array.ndim != 2
        or mean_array.shape[1] != len(component_ids)
    ):
        raise ValueError("E1 target arrays must be aligned case-by-component matrices")
    if not math.isfinite(rmse_tolerance) or rmse_tolerance <= 0.0:
        raise ValueError("E1 RMSE tolerance must be finite and positive")
    if conditional_on_positive and (
        total_replicates is None or int(total_replicates) < 2
    ):
        raise ValueError("conditional E1 targets require total_replicates >= 2")

    component_reports: list[dict[str, Any]] = []
    current_rmse: list[float] = []
    required_replicates: list[int] = []
    for column, component_id in enumerate(component_ids):
        valid = (
            np.isfinite(mean_array[:, column])
            & np.isfinite(variance_array[:, column])
            & np.isfinite(count_array[:, column])
            & (variance_array[:, column] >= 0.0)
            & (count_array[:, column] >= 2.0)
        )
        valid_count = int(np.count_nonzero(valid))
        if valid_count < 2:
            component_reports.append(
                {
                    "component_id": component_id,
                    "valid_case_count": valid_count,
                    "status": "insufficient_conditioned_cases",
                    "decomposition": None,
                    "current_R_rmse": None,
                    "signal_to_mean_estimator_noise_ratio": None,
                    "minimum_candidate_R": None,
                }
            )
            continue
        decomposition = variance_decomposition(
            mean_array[valid, column],
            variance_array[valid, column],
            count_array[valid, column],
        )
        noise_variance = float(decomposition["mean_estimator_noise_variance"])
        corrected_signal = float(
            decomposition["noise_corrected_between_case_variance"]
        )
        rmse = math.sqrt(max(0.0, noise_variance))
        mean_within = float(np.mean(variance_array[valid, column]))
        minimum_observations = next(
            (
                candidate
                for candidate in _E1_CANDIDATE_REPLICATES
                if math.sqrt(mean_within / candidate) <= rmse_tolerance
            ),
            None,
        )
        pooled_positive_rate = None
        if conditional_on_positive:
            finite_counts = count_array[
                np.isfinite(count_array[:, column]), column
            ]
            pooled_positive_rate = float(
                np.sum(finite_counts)
                / (finite_counts.size * int(total_replicates))
            )
            minimum_r = (
                next(
                    (
                        candidate
                        for candidate in _E1_CANDIDATE_REPLICATES
                        if math.sqrt(
                            noise_variance
                            * int(total_replicates)
                            / candidate
                        )
                        <= rmse_tolerance
                    ),
                    None,
                )
                if minimum_observations is not None
                else None
            )
        else:
            minimum_r = minimum_observations
        current_rmse.append(rmse)
        if minimum_r is not None:
            required_replicates.append(int(minimum_r))
        component_reports.append(
            {
                "component_id": component_id,
                "valid_case_count": valid_count,
                "status": "evaluated",
                "decomposition": decomposition,
                "current_R_rmse": rmse,
                "signal_to_mean_estimator_noise_ratio": (
                    corrected_signal / noise_variance
                    if noise_variance > 0.0
                    else None
                ),
                "minimum_candidate_R": minimum_r,
                "minimum_positive_observations": (
                    minimum_observations if conditional_on_positive else None
                ),
                "pooled_positive_rate": pooled_positive_rate,
            }
        )

    evaluated = [item for item in component_reports if item["status"] == "evaluated"]
    return {
        "rmse_tolerance": float(rmse_tolerance),
        "target_semantics": (
            "conditional_on_positive_replicates"
            if conditional_on_positive
            else "per_replicate_target"
        ),
        "R_selection_basis": (
            "current conditional mean-estimator noise scaled by current_total_R "
            "over candidate_R, assuming the observed hit-rate distribution is "
            "stable; exploratory conversion and not a per-stratum guarantee"
            if conditional_on_positive
            else "mean within-case variance divided by total replicate count"
        ),
        "candidate_replicates": list(_E1_CANDIDATE_REPLICATES),
        "component_count": len(component_ids),
        "evaluated_component_count": len(evaluated),
        "insufficient_component_count": len(component_ids) - len(evaluated),
        "current_R_rmse_summary": (
            {
                "median": float(np.median(current_rmse)),
                "p90": float(np.quantile(current_rmse, 0.9)),
                "maximum": float(np.max(current_rmse)),
            }
            if current_rmse
            else None
        ),
        "recommended_R_for_all_evaluated_components": (
            max(required_replicates)
            if len(required_replicates) == len(evaluated) and evaluated
            else None
        ),
        "components": component_reports,
    }


def run_e1(
    engine: DamageEngine,
    cases: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
    probability_rmse_tolerance: float = 0.05,
    conditional_rmse_tolerance: float = 0.10,
) -> dict[str, Any]:
    component_ids = tuple(item.object_id for item in engine.compiled_target.components)
    target_names = (
        "p_hit",
        "p_degraded",
        "p_failed",
        "hit_conditioned_log1p_deposited_energy_j",
        "hit_conditioned_log1p_hit_count",
    )
    target_means = {name: [] for name in target_names}
    target_variances = {name: [] for name in target_names}
    target_counts = {name: [] for name in target_names}
    domain_affected_fractions: list[float] = []
    for index, base in enumerate(cases):
        case = _configure_case(base, replicates=replicates, seed=seed + index)
        case["case_id"] = f"{base['case_id']}::E1"
        result = engine.run(case).to_dict()
        domain_affected_fractions.append(
            float(
                result["penetration_domain_coverage"][
                    "affected_replicate_fraction"
                ]
            )
        )
        case_statistics: dict[str, list[dict[str, Any]]] = {
            "p_hit": [
                result["components"][name]["fragment_metrics"]["hit_indicator"]
                for name in component_ids
            ],
            "p_degraded": [
                result["components"][name]["damage_states"]["fragment"]["degraded"]
                for name in component_ids
            ],
            "p_failed": [
                result["components"][name]["damage_states"]["fragment"]["failed"]
                for name in component_ids
            ],
            "hit_conditioned_log1p_deposited_energy_j": [
                result["components"][name]["fragment_metrics"][
                    "hit_conditioned_log1p_deposited_energy_j"
                ]
                for name in component_ids
            ],
            "hit_conditioned_log1p_hit_count": [
                result["components"][name]["fragment_metrics"][
                    "hit_conditioned_log1p_hit_count"
                ]
                for name in component_ids
            ],
        }
        for target_name, statistics in case_statistics.items():
            means_row, variances_row, counts_row = [], [], []
            for statistic in statistics:
                sample_count = int(statistic.get("sample_count", 0))
                mean = statistic.get("mean")
                stddev = statistic.get("stddev")
                if (
                    sample_count < 2
                    or mean is None
                    or stddev is None
                    or not math.isfinite(float(mean))
                    or not math.isfinite(float(stddev))
                ):
                    means_row.append(float("nan"))
                    variances_row.append(float("nan"))
                    counts_row.append(float(sample_count))
                else:
                    means_row.append(float(mean))
                    variances_row.append(float(stddev) ** 2)
                    counts_row.append(float(sample_count))
            target_means[target_name].append(means_row)
            target_variances[target_name].append(variances_row)
            target_counts[target_name].append(counts_row)

    target_noise = {}
    for target_name in target_names:
        tolerance = (
            probability_rmse_tolerance
            if target_name in {"p_hit", "p_degraded", "p_failed"}
            else conditional_rmse_tolerance
        )
        target_noise[target_name] = _e1_target_report(
            np.asarray(target_means[target_name]),
            np.asarray(target_variances[target_name]),
            np.asarray(target_counts[target_name]),
            component_ids,
            rmse_tolerance=tolerance,
            conditional_on_positive=target_name.startswith("hit_conditioned_"),
            total_replicates=replicates,
        )

    p_hit_means = np.asarray(target_means["p_hit"])
    p_hit_variances = np.asarray(target_variances["p_hit"])
    p_hit_counts = np.asarray(target_counts["p_hit"])
    decomposition = variance_decomposition(
        p_hit_means,
        p_hit_variances,
        p_hit_counts,
    )
    return {
        "experiment_id": "E1",
        "case_count": len(cases),
        "replicates": replicates,
        "component_ids": list(component_ids),
        "p_hit_variance_decomposition": decomposition,
        "hit_positive_replicate_count": np.asarray(
            target_counts["hit_conditioned_log1p_deposited_energy_j"]
        ).tolist(),
        "target_noise": target_noise,
        "probability_rmse_tolerance": float(probability_rmse_tolerance),
        "conditional_log1p_rmse_tolerance": float(conditional_rmse_tolerance),
        "penetration_domain_scope": {
            "affected_case_count": int(
                np.count_nonzero(np.asarray(domain_affected_fractions) > 0.0)
            ),
            "affected_fraction_summary": {
                "mean": float(np.mean(domain_affected_fractions)),
                "maximum": float(np.max(domain_affected_fractions)),
            },
            "noise_decomposition_semantics": (
                "point-estimate Monte Carlo noise across all selected cases; "
                "penetration-domain identification uncertainty is reported "
                "separately and is not folded into these variances"
            ),
        },
        "R_selection_rule": "choose_smallest_R_whose_mean_estimator_noise_is_below_preregistered_task_tolerance",
    }


def run_e2(engine: DamageEngine, cases: list[dict[str, Any]], *, replicates: int, seed: int) -> dict[str, Any]:
    component_ids = tuple(item.object_id for item in engine.compiled_target.components)
    aggregate = {identifier: {name: 0 for name in ("no_geometric_intersection", "blocked_before_component", "hit_no_penetration", "penetrated_no_damage", "response_possible")} for identifier in component_ids}
    trace_count = 0
    matching_observations = []
    matching_by_cell: dict[str, list[dict[str, Any]]] = {}
    for index, base in enumerate(cases):
        case = _configure_case(base, replicates=replicates, seed=seed + index, audit=True)
        case["case_id"] = f"{base['case_id']}::E2"
        result = engine.run(case).to_dict()
        report = summarize_component_reachability(result["fragment_audit"], component_ids)
        matching = extract_warhead_target_matching_observations(
            result["fragment_audit"]
        )
        matching_observations.append(matching)
        direction = base.get("metadata", {}).get(
            "continuous_direction_sampling", {}
        ).get("region_id", "unclassified")
        profile = base.get("munition", {}).get("profile_id", "unclassified")
        cell = f"{direction}::{profile}"
        matching_by_cell.setdefault(cell, []).append(matching)
        trace_count += int(report["fragment_trace_count"])
        for identifier in component_ids:
            for category, count in report["components"][identifier]["counts"].items():
                aggregate[identifier][category] += int(count)
    components = {
        identifier: {
            "counts": counts,
            "fractions": {name: count / trace_count for name, count in counts.items()},
        }
        for identifier, counts in aggregate.items()
    }
    totals = {name: sum(row[name] for row in aggregate.values()) for name in next(iter(aggregate.values()))}
    return {
        "experiment_id": "E2",
        "case_count": len(cases),
        "replicates": replicates,
        "fragment_trace_count_per_component": trace_count,
        "components": components,
        "aggregate_component_fragment_counts": totals,
        "dominant_loss_category": max(totals, key=totals.get),
        "warhead_target_matching": summarize_warhead_target_matching_observations(
            combine_warhead_target_matching_observations(matching_observations)
        ),
        "warhead_target_matching_by_direction_profile": {
            cell: summarize_warhead_target_matching_observations(
                combine_warhead_target_matching_observations(rows)
            )
            for cell, rows in sorted(matching_by_cell.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["E0", "E1", "E2"])
    parser.add_argument("--cases", type=Path, default=Path("artifacts/validation/sampling_sphere_v1/no_bottom_impact_state_pilot/rounds/round-000/formal_cases.jsonl.gz"))
    parser.add_argument("--target", type=Path, default=Path("configs/targets/target.fragment_only.v1.json"))
    parser.add_argument("--physics", type=Path, default=Path("configs/physics/loitering_impact.json"))
    parser.add_argument("--case-count", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026112001)
    parser.add_argument("--relative-margin", type=float, default=0.02)
    parser.add_argument("--probability-rmse-tolerance", type=float, default=0.05)
    parser.add_argument("--conditional-rmse-tolerance", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recommended = {"E0": (200, 32), "E1": (500, 32), "E2": (500, 4)}
    case_count, replicates = recommended[args.experiment]
    case_count = args.case_count or case_count
    replicates = args.replicates or replicates
    if args.experiment == "E1" and case_count < 2:
        raise ValueError("E1 variance decomposition requires at least two cases")
    target = json.loads(args.target.read_text(encoding="utf-8"))
    engine = DamageEngine(target, PhysicsRegistry.from_file(args.physics))
    cases = _load_cases(args.cases, case_count)
    if args.experiment == "E0":
        result = run_e0(engine, cases, replicates=replicates, seed=args.seed, relative_margin=args.relative_margin)
    elif args.experiment == "E1":
        result = run_e1(
            engine,
            cases,
            replicates=replicates,
            seed=args.seed,
            probability_rmse_tolerance=args.probability_rmse_tolerance,
            conditional_rmse_tolerance=args.conditional_rmse_tolerance,
        )
    else:
        result = run_e2(engine, cases, replicates=replicates, seed=args.seed)
    config = {
        "experiment": args.experiment,
        "case_count": case_count,
        "replicates": replicates,
        "seed": args.seed,
        "relative_margin": args.relative_margin,
        "case_archive_hash": file_hash(args.cases),
        "target_hash": engine.target_hash,
        "physics_registry_hash": engine.registry.hash,
        "engine_version": ENGINE_VERSION,
        "engine_code_hash": ENGINE_CODE_HASH,
    }
    document = {
        "schema_version": "idea2-engine-experiment@1.0.0",
        "config": {**config, "config_hash": content_hash(config)},
        "result": result,
    }
    document["result_hash"] = content_hash(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "result_hash": document["result_hash"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
