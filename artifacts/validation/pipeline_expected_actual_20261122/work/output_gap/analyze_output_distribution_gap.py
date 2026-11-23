"""Train/validation-only output, label, and damage-tree distribution audit.

The script treats the frozen software contracts as expectations and the
development labels/traces as observations.  It skips every sealed-test trace
line before JSON decoding and does not run training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.damage_tree import compile_damage_tree
from damage_engine.distribution_metrics import jensen_shannon_binned
from damage_engine.utils import content_hash, file_hash


PROBABILITY_EDGES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.000000000001)
EPSILON = 0.01


def _summary(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    quantiles = np.quantile(array, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "population_standard_deviation": float(np.std(array)),
        "minimum": float(np.min(array)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "maximum": float(np.max(array)),
        "zero_mass": float(np.mean(array == 0.0)),
        "one_mass": float(np.mean(array == 1.0)),
    }


def _two_sample(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size == 0 or right.size == 0:
        return {"status": "unavailable", "left_count": int(left.size), "right_count": int(right.size)}
    return {
        "status": "estimated",
        "left_count": int(left.size),
        "right_count": int(right.size),
        "wasserstein1": float(wasserstein_distance(left, right)),
        "ks_distance": float(ks_2samp(left, right, method="asymp").statistic),
        "fixed_bin_jensen_shannon": float(
            jensen_shannon_binned(left, right, bin_edges=PROBABILITY_EDGES)
        ),
        "right_minus_left_mean": float(np.mean(right) - np.mean(left)),
        "right_minus_left_zero_mass": float(np.mean(right == 0.0) - np.mean(left == 0.0)),
    }


def _bootstrap_mean_differences(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    train = np.asarray(train, dtype=np.float64)
    validation = np.asarray(validation, dtype=np.float64)
    if train.ndim == 1:
        train = train[:, None]
    if validation.ndim == 1:
        validation = validation[:, None]
    if train.shape[1] != validation.shape[1]:
        raise ValueError("bootstrap matrices must have the same column count")
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, train.shape[1]), dtype=np.float64)
    for draw in range(draws):
        train_index = rng.integers(0, train.shape[0], size=train.shape[0])
        validation_index = rng.integers(0, validation.shape[0], size=validation.shape[0])
        samples[draw] = (
            np.mean(validation[validation_index], axis=0)
            - np.mean(train[train_index], axis=0)
        )
    point = np.mean(validation, axis=0) - np.mean(train, axis=0)
    lower, upper = np.quantile(samples, [0.025, 0.975], axis=0)
    return {
        "method": "independent_case_cluster_percentile_bootstrap",
        "draws": draws,
        "seed": seed,
        "point": point.tolist(),
        "percentile_95_lower": lower.tolist(),
        "percentile_95_upper": upper.tolist(),
        "scope_note": (
            "Conditional on the fixed QMC designs and different R8/R32 teacher protocols; "
            "not between-scramble or physical uncertainty."
        ),
    }


def _bootstrap_identified_mean_difference(
    train_case_interval: np.ndarray,
    validation_case_interval: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap an outer envelope for an interval-identified split mean gap."""

    train_case_interval = np.asarray(train_case_interval, dtype=np.float64)
    validation_case_interval = np.asarray(validation_case_interval, dtype=np.float64)
    if train_case_interval.ndim != 2 or train_case_interval.shape[1] != 2:
        raise ValueError("train_case_interval must have shape (case, 2)")
    if validation_case_interval.ndim != 2 or validation_case_interval.shape[1] != 2:
        raise ValueError("validation_case_interval must have shape (case, 2)")
    rng = np.random.default_rng(seed)
    lower_draws = np.empty(draws, dtype=np.float64)
    upper_draws = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        train_index = rng.integers(
            0, train_case_interval.shape[0], size=train_case_interval.shape[0]
        )
        validation_index = rng.integers(
            0,
            validation_case_interval.shape[0],
            size=validation_case_interval.shape[0],
        )
        lower_draws[draw] = float(
            np.mean(validation_case_interval[validation_index, 0])
            - np.mean(train_case_interval[train_index, 1])
        )
        upper_draws[draw] = float(
            np.mean(validation_case_interval[validation_index, 1])
            - np.mean(train_case_interval[train_index, 0])
        )
    point_lower = float(
        np.mean(validation_case_interval[:, 0]) - np.mean(train_case_interval[:, 1])
    )
    point_upper = float(
        np.mean(validation_case_interval[:, 1]) - np.mean(train_case_interval[:, 0])
    )
    return {
        "method": "independent_case_resampling_with_identification_outer_envelope",
        "draws": draws,
        "seed": seed,
        "identified_validation_minus_train_mean_interval": [point_lower, point_upper],
        "bootstrap_95_outer_envelope": [
            float(np.quantile(lower_draws, 0.025)),
            float(np.quantile(upper_draws, 0.975)),
        ],
        "scope_note": (
            "Uses all cases and a fixed denominator of all 67 components; identification "
            "bounds are propagated before case resampling."
        ),
    }


def _pair_correlation(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    keep = np.isfinite(left) & np.isfinite(right)
    left = left[keep]
    right = right[keep]
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return {"status": "undefined", "count": int(left.size)}
    return {
        "status": "estimated",
        "count": int(left.size),
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "spearman": float(spearmanr(left, right).statistic),
    }


def _component_correlation_summary(values: np.ndarray, masks: np.ndarray, component_ids: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for left_index in range(values.shape[1]):
        for right_index in range(left_index + 1, values.shape[1]):
            keep = masks[:, left_index] & masks[:, right_index]
            if int(np.sum(keep)) < 20:
                continue
            left = values[keep, left_index]
            right = values[keep, right_index]
            if np.std(left) == 0.0 or np.std(right) == 0.0:
                continue
            correlation = float(np.corrcoef(left, right)[0, 1])
            rows.append(
                {
                    "left": component_ids[left_index],
                    "right": component_ids[right_index],
                    "count": int(np.sum(keep)),
                    "pearson": correlation,
                }
            )
    if not rows:
        return {"pair_count": 0}
    coefficients = np.asarray([row["pearson"] for row in rows])
    maximum = max(rows, key=lambda row: abs(row["pearson"]))
    return {
        "pair_count": len(rows),
        "median_signed_pearson": float(np.median(coefficients)),
        "median_absolute_pearson": float(np.median(np.abs(coefficients))),
        "p95_absolute_pearson": float(np.quantile(np.abs(coefficients), 0.95)),
        "maximum_absolute_pair": maximum,
        "interpretation": "descriptive dependence in the software-teacher outputs; no independence target was declared",
    }


def _blockwise_rbf_mmd2(left: np.ndarray, right: np.ndarray, *, seed: int) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("MMD inputs must be matrices with a shared feature axis")
    combined = np.vstack([left, right])
    rng = np.random.default_rng(seed)
    sample_count = min(500, combined.shape[0])
    sample = combined[rng.choice(combined.shape[0], size=sample_count, replace=False)]
    sample_distances = np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=-1)
    positive = sample_distances[sample_distances > 0.0]
    bandwidth = math.sqrt(0.5 * float(np.median(positive))) if positive.size else 1.0
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)

    def kernel_mean(first: np.ndarray, second: np.ndarray, block: int = 256) -> float:
        total = 0.0
        count = 0
        for start_a in range(0, first.shape[0], block):
            a = first[start_a : start_a + block]
            for start_b in range(0, second.shape[0], block):
                b = second[start_b : start_b + block]
                distances = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
                total += float(np.sum(np.exp(-gamma * distances)))
                count += distances.size
        return total / count

    value = kernel_mean(left, left) + kernel_mean(right, right) - 2.0 * kernel_mean(left, right)
    return {
        "mmd2_biased": max(0.0, float(value)),
        "bandwidth": float(bandwidth),
        "bandwidth_estimation": "median positive squared distance on fixed-seed sample of at most 500 cases",
        "seed": seed,
    }


def _load_case_strata(formal_cases_path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with gzip.open(formal_cases_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            case = json.loads(line)
            metadata = case["metadata"]
            contact = metadata.get("contact_manifold_sampling") or metadata["continuous_direction_sampling"]
            result[str(case["case_id"])] = {
                "profile": str(case["munition"]["profile_id"]),
                "region": str(contact.get("region_id", contact.get("approach_id"))),
            }
    return result


def _component_support(values: np.ndarray, masks: np.ndarray, component_ids: list[str]) -> dict[str, Any]:
    rows = []
    for index, component_id in enumerate(component_ids):
        observed = values[masks[:, index], index]
        rows.append(
            {
                "component_id": component_id,
                "observed_count": int(observed.size),
                "positive_count": int(np.sum(observed > 0.0)),
                "zero_count": int(np.sum(observed == 0.0)),
                "mean": float(np.mean(observed)) if observed.size else None,
            }
        )
    observed_counts = np.asarray([row["observed_count"] for row in rows])
    positive_counts = np.asarray([row["positive_count"] for row in rows])
    valid_rows = [row for row in rows if row["mean"] is not None]
    return {
        "component_count": len(rows),
        "observed_count": _summary(observed_counts),
        "positive_count": _summary(positive_counts),
        "zero_positive_component_count": int(np.sum(positive_counts == 0)),
        "lowest_mean_components": sorted(valid_rows, key=lambda row: row["mean"])[:5],
        "highest_mean_components": sorted(valid_rows, key=lambda row: row["mean"], reverse=True)[:5],
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-v2", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--proxy-report", type=Path, required=True)
    parser.add_argument("--proxy-uncertainty", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026112201)
    args = parser.parse_args()

    core = args.dataset_v2 / "core"
    with np.load(core / "component_labels.npz", allow_pickle=False) as archive:
        labels = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(core / "sampling_weights.npz", allow_pickle=False) as archive:
        weights = {name: np.asarray(archive[name]) for name in archive.files}
    case_ids = weights["case_ids"].astype(str)
    split_names = weights["split"].astype(str)
    component_ids = labels["component_ids"].astype(str).tolist()
    if labels["case_ids"].astype(str).tolist() != case_ids.tolist():
        raise ValueError("label and sampling case order differs")
    development = split_names != "test"
    train = split_names == "train"
    validation = split_names == "validation"
    if (int(np.sum(train)), int(np.sum(validation)), int(np.sum(split_names == "test"))) != (2000, 300, 300):
        raise ValueError("unexpected split counts")
    if not np.all(weights["design_weight"][development] == 1.0):
        raise ValueError("this audit expects the frozen equal-weight Reference design")

    strata_map = _load_case_strata(args.dataset_root / "rounds/round-000/formal_cases.jsonl.gz")
    profiles = np.asarray([strata_map[case_id]["profile"] for case_id in case_ids])
    regions = np.asarray([strata_map[case_id]["region"] for case_id in case_ids])
    strata = np.asarray([f"{region}::{profile}" for region, profile in zip(regions, profiles, strict=True)])

    target_document = json.loads(args.target.read_text(encoding="utf-8-sig"))
    tree = compile_damage_tree(target_document)
    output_names = list(tree.outputs)
    development_indices = np.flatnonzero(development)
    row_for_case = {case_id: index for index, case_id in enumerate(case_ids)}
    fragment_degraded = np.full((case_ids.size, len(component_ids)), np.nan)
    fragment_failed = np.full_like(fragment_degraded, np.nan)
    tree_online = np.full((case_ids.size, len(output_names)), np.nan)
    tree_offline = np.full_like(tree_online, np.nan)

    trace_metrics = {
        "development_case_count": 0,
        "development_replicate_count": 0,
        "sealed_test_trace_lines_seen": 0,
        "sealed_test_trace_json_lines_decoded": 0,
        "blast_state_value_count": 0,
        "blast_nonzero_count": 0,
        "combined_fragment_comparison_count": 0,
        "combined_fragment_mismatch_count": 0,
        "combined_fragment_maximum_absolute_error": 0.0,
        "mechanism_state_value_count": 0,
        "mechanism_state_out_of_bounds_count": 0,
        "failed_degraded_comparison_count": 0,
        "failed_exceeds_degraded_violation_count": 0,
        "tree_output_comparison_count": 0,
        "tree_output_out_of_bounds_count": 0,
        "tree_output_mismatch_count": 0,
        "tree_output_maximum_absolute_error": 0.0,
    }
    trace_path = core / "component_state_trace.jsonl.gz"
    line_count = 0
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        for line_index, line in enumerate(stream):
            line_count += 1
            if line_index >= case_ids.size:
                raise ValueError("trace file has more lines than cases")
            if split_names[line_index] == "test":
                trace_metrics["sealed_test_trace_lines_seen"] += 1
                # Intentionally skip json.loads for the sealed-test line.
                continue
            wrapper = json.loads(line)
            case_id = str(wrapper["case_id"])
            if case_id != case_ids[line_index] or row_for_case[case_id] != line_index:
                raise ValueError("trace order differs from the frozen case order")
            trace = wrapper["component_state_trace"]
            records = trace["records"]
            expected_replicates = 8 if split_names[line_index] == "train" else 32
            if int(trace["replicate_count"]) != expected_replicates or len(records) != expected_replicates:
                raise ValueError("trace replicate count differs from the split contract")
            trace_metrics["development_case_count"] += 1
            trace_metrics["development_replicate_count"] += len(records)
            degraded_sum = np.zeros(len(component_ids), dtype=np.float64)
            failed_sum = np.zeros(len(component_ids), dtype=np.float64)
            online_sum = np.zeros(len(output_names), dtype=np.float64)
            offline_sum = np.zeros(len(output_names), dtype=np.float64)
            for expected_replicate, record in enumerate(records):
                if record["replicate_index"] != expected_replicate:
                    raise ValueError("replicate order is not contiguous")
                states = record["component_mechanism_states"]
                evaluation = tree.evaluate(states)["outputs"]
                for component_index, component_id in enumerate(component_ids):
                    fragment = states["fragment"][component_id]
                    blast = states["blast"][component_id]
                    combined = states["combined"][component_id]
                    degraded = float(fragment["degraded"])
                    failed = float(fragment["failed"])
                    degraded_sum[component_index] += degraded
                    failed_sum[component_index] += failed
                    for state_name in ("degraded", "failed"):
                        blast_value = float(blast[state_name])
                        trace_metrics["blast_state_value_count"] += 1
                        trace_metrics["blast_nonzero_count"] += int(blast_value != 0.0)
                        difference = abs(float(combined[state_name]) - float(fragment[state_name]))
                        trace_metrics["combined_fragment_comparison_count"] += 1
                        trace_metrics["combined_fragment_mismatch_count"] += int(difference > 1.0e-12)
                        trace_metrics["combined_fragment_maximum_absolute_error"] = max(
                            trace_metrics["combined_fragment_maximum_absolute_error"], difference
                        )
                    for mechanism in ("fragment", "blast", "combined"):
                        state = states[mechanism][component_id]
                        for state_name in ("degraded", "failed"):
                            state_value = float(state[state_name])
                            trace_metrics["mechanism_state_value_count"] += 1
                            trace_metrics["mechanism_state_out_of_bounds_count"] += int(
                                not 0.0 <= state_value <= 1.0
                            )
                        trace_metrics["failed_degraded_comparison_count"] += 1
                        trace_metrics["failed_exceeds_degraded_violation_count"] += int(
                            float(state["failed"]) > float(state["degraded"]) + 1.0e-12
                        )
                for output_index, output_name in enumerate(output_names):
                    online = float(record["system_outputs"][output_name])
                    offline = float(evaluation[output_name])
                    difference = abs(online - offline)
                    online_sum[output_index] += online
                    offline_sum[output_index] += offline
                    trace_metrics["tree_output_comparison_count"] += 1
                    trace_metrics["tree_output_out_of_bounds_count"] += int(
                        not 0.0 <= online <= 1.0 or not 0.0 <= offline <= 1.0
                    )
                    trace_metrics["tree_output_mismatch_count"] += int(difference > 1.0e-12)
                    trace_metrics["tree_output_maximum_absolute_error"] = max(
                        trace_metrics["tree_output_maximum_absolute_error"], difference
                    )
            fragment_degraded[line_index] = degraded_sum / len(records)
            fragment_failed[line_index] = failed_sum / len(records)
            tree_online[line_index] = online_sum / len(records)
            tree_offline[line_index] = offline_sum / len(records)
    if line_count != case_ids.size:
        raise ValueError("trace line count differs from case count")

    heads = {
        "p_hit": (labels["p_hit"], labels["p_hit_mask"]),
        "p_degraded": (labels["p_degraded"], labels["p_degraded_mask"]),
        "p_failed": (labels["p_failed"], labels["p_failed_mask"]),
    }
    label_contract = {
        "p_hit_success_trial_reconstruction_maximum_absolute_error": float(
            np.max(
                np.abs(
                    labels["p_hit"][development]
                    - labels["p_hit_success_count"][development]
                    / labels["p_hit_trial_count"][development]
                )
            )
        ),
        "p_degraded_trace_aggregation_maximum_absolute_error": float(
            np.max(np.abs(labels["p_degraded"][development] - fragment_degraded[development]))
        ),
        "p_failed_trace_aggregation_maximum_absolute_error": float(
            np.max(np.abs(labels["p_failed"][development] - fragment_failed[development]))
        ),
        "trial_count_by_split": {
            "train": sorted(set(labels["p_hit_trial_count"][train].ravel().tolist())),
            "validation": sorted(set(labels["p_hit_trial_count"][validation].ravel().tolist())),
        },
    }
    shared_mask = development[:, None] & labels["p_hit_mask"] & labels["p_degraded_mask"] & labels["p_failed_mask"]
    label_contract["probability_nesting"] = {
        "shared_observed_pair_count": int(np.sum(shared_mask)),
        "p_degraded_exceeds_p_hit_count": int(
            np.sum(shared_mask & (labels["p_degraded"] > labels["p_hit"] + 1.0e-12))
        ),
        "p_failed_exceeds_p_degraded_count": int(
            np.sum(shared_mask & (labels["p_failed"] > labels["p_degraded"] + 1.0e-12))
        ),
    }
    interval_checks = {}
    for head_name, (values, masks) in heads.items():
        ci = labels[f"{head_name}_ci95"]
        identification = labels[f"{head_name}_identification_interval"]
        selected = development[:, None]
        width = identification[..., 1] - identification[..., 0]
        interval_checks[head_name] = {
            "development_pair_count": int(np.sum(selected) * values.shape[1]),
            "ci_contains_point_violation_count": int(
                np.sum(selected & ((values < ci[..., 0] - 1.0e-12) | (values > ci[..., 1] + 1.0e-12)))
            ),
            "identification_contains_point_violation_count": int(
                np.sum(
                    selected
                    & (
                        (values < identification[..., 0] - 1.0e-12)
                        | (values > identification[..., 1] + 1.0e-12)
                    )
                )
            ),
            "mask_disagrees_with_width_le_epsilon_count": int(
                np.sum(selected & (masks != (width <= EPSILON + 1.0e-12)))
            ),
            "identification_width": _summary(width[development]),
        }
    label_contract["interval_contracts"] = interval_checks

    head_report: dict[str, Any] = {}
    complete_case_bootstrap: dict[str, Any] = {}
    case_means = np.empty((case_ids.size, len(heads)), dtype=np.float64)
    for head_index, (head_name, (values, masks)) in enumerate(heads.items()):
        for row in development_indices:
            observed = values[row, masks[row]]
            case_means[row, head_index] = float(np.mean(observed))
        train_case = case_means[train, head_index]
        validation_case = case_means[validation, head_index]
        observed_component_count = np.sum(masks, axis=1)
        complete_case = development & np.all(masks, axis=1)
        train_complete_case = train & complete_case
        validation_complete_case = validation & complete_case
        complete_case_mean = np.mean(values[complete_case], axis=1)
        train_complete_case_mean = np.mean(values[train_complete_case], axis=1)
        validation_complete_case_mean = np.mean(values[validation_complete_case], axis=1)
        complete_case_bootstrap[head_name] = _bootstrap_mean_differences(
            train_complete_case_mean,
            validation_complete_case_mean,
            draws=args.bootstrap_draws,
            seed=args.seed + 100 + head_index,
        )
        identification = labels[f"{head_name}_identification_interval"]
        fixed_denominator_case_interval = np.mean(identification, axis=1)
        fixed_denominator_identified_gap = _bootstrap_identified_mean_difference(
            fixed_denominator_case_interval[train],
            fixed_denominator_case_interval[validation],
            draws=args.bootstrap_draws,
            seed=args.seed + 200 + head_index,
        )
        head_report[head_name] = {
            "development_observed_pairs": _summary(values[development[:, None] & masks]),
            "train_observed_pairs": _summary(values[train[:, None] & masks]),
            "validation_observed_pairs": _summary(values[validation[:, None] & masks]),
            "flattened_observed_pair_train_validation_gap": _two_sample(
                values[train[:, None] & masks], values[validation[:, None] & masks]
            ),
            "fixed_67_component_identified_case_mean": {
                "train_mean_identification_interval": [
                    float(np.mean(fixed_denominator_case_interval[train, 0])),
                    float(np.mean(fixed_denominator_case_interval[train, 1])),
                ],
                "validation_mean_identification_interval": [
                    float(np.mean(fixed_denominator_case_interval[validation, 0])),
                    float(np.mean(fixed_denominator_case_interval[validation, 1])),
                ],
                **fixed_denominator_identified_gap,
            },
            "complete_67_component_case_analysis": {
                "development_case_count": int(np.sum(complete_case)),
                "train_case_count": int(np.sum(train_complete_case)),
                "validation_case_count": int(np.sum(validation_complete_case)),
                "development_case_mean": _summary(complete_case_mean),
                "train_case_mean": _summary(train_complete_case_mean),
                "validation_case_mean": _summary(validation_complete_case_mean),
                "train_validation_gap": _two_sample(
                    train_complete_case_mean, validation_complete_case_mean
                ),
                "validation_minus_train_mean_case_bootstrap": complete_case_bootstrap[head_name],
            },
            "missingness_sensitive_case_mean_analysis": {
                "development_case_mean_across_observed_components": _summary(
                    case_means[development, head_index]
                ),
                "train_case_mean_across_observed_components": _summary(train_case),
                "validation_case_mean_across_observed_components": _summary(validation_case),
                "train_validation_gap": _two_sample(train_case, validation_case),
                "observed_component_count": _summary(observed_component_count[development]),
                "observed_count_case_mean_correlation": {
                    "development": _pair_correlation(
                        observed_component_count[development],
                        case_means[development, head_index],
                    ),
                    "train": _pair_correlation(
                        observed_component_count[train],
                        case_means[train, head_index],
                    ),
                    "validation": _pair_correlation(
                        observed_component_count[validation],
                        case_means[validation, head_index],
                    ),
                },
                "interpretation": (
                    "Sensitivity diagnostic only: the denominator varies by case and is associated "
                    "with the case mean; use the complete-67 or flattened-pair result for the main split comparison."
                ),
            },
            "component_support": _component_support(values[development], masks[development], component_ids),
        }
    case_mean_bootstrap = _bootstrap_mean_differences(
        case_means[train],
        case_means[validation],
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    case_mean_bootstrap["columns"] = list(heads)
    case_mean_bootstrap["interpretation"] = (
        "Missingness-sensitive diagnostic because each case mean uses a variable observed-component denominator."
    )

    conditional_report = {}
    for name in ("conditional_log1p_energy", "conditional_log1p_hit_count"):
        values = labels[name]
        masks = labels[f"{name}_mask"]
        conditional_report[name] = {
            "development_observed_values": _summary(values[development[:, None] & masks]),
            "train_observed_values": _summary(values[train[:, None] & masks]),
            "validation_observed_values": _summary(values[validation[:, None] & masks]),
            "component_support": _component_support(values[development], masks[development], component_ids),
            "scope": "hit-conditioned and domain-point-identified values only",
        }

    flattened_correlations = {}
    head_names = list(heads)
    for left_index, left_name in enumerate(head_names):
        for right_name in head_names[left_index + 1 :]:
            left_values, left_masks = heads[left_name]
            right_values, right_masks = heads[right_name]
            keep = development[:, None] & left_masks & right_masks
            flattened_correlations[f"{left_name}__{right_name}"] = _pair_correlation(
                left_values[keep], right_values[keep]
            )
    all_three_complete = development.copy()
    for _, (_, masks) in heads.items():
        all_three_complete &= np.all(masks, axis=1)
    correlations = {
        "flattened_head_correlations": flattened_correlations,
        "p_hit_component_dependence": _component_correlation_summary(
            labels["p_hit"][development], labels["p_hit_mask"][development], component_ids
        ),
        "missingness_sensitive_case_mean_three_head_train_validation_mmd": _blockwise_rbf_mmd2(
            case_means[train], case_means[validation], seed=args.seed
        ),
        "complete_67_all_three_heads_case_count": int(np.sum(all_three_complete)),
        "complete_67_all_three_heads_train_validation_mmd": _blockwise_rbf_mmd2(
            case_means[train & all_three_complete],
            case_means[validation & all_three_complete],
            seed=args.seed + 1,
        ),
    }

    component_w1 = []
    for component_index, component_id in enumerate(component_ids):
        train_values = labels["p_hit"][train & labels["p_hit_mask"][:, component_index], component_index]
        validation_values = labels["p_hit"][
            validation & labels["p_hit_mask"][:, component_index], component_index
        ]
        if train_values.size and validation_values.size:
            component_w1.append(
                {
                    "component_id": component_id,
                    "train_count": int(train_values.size),
                    "validation_count": int(validation_values.size),
                    "wasserstein1": float(wasserstein_distance(train_values, validation_values)),
                    "ks_distance": float(ks_2samp(train_values, validation_values, method="asymp").statistic),
                }
            )
    component_w1_values = np.asarray([row["wasserstein1"] for row in component_w1])
    split_component_gap = {
        "component_count": len(component_w1),
        "p_hit_componentwise_wasserstein1": _summary(component_w1_values),
        "largest_component_gaps": sorted(component_w1, key=lambda row: row["wasserstein1"], reverse=True)[:10],
    }

    tree_report = {
        "output_names": output_names,
        "offline_recompute_contract": {
            "comparison_count": trace_metrics["tree_output_comparison_count"],
            "mismatch_count": trace_metrics["tree_output_mismatch_count"],
            "maximum_absolute_error": trace_metrics["tree_output_maximum_absolute_error"],
            "expected": "compile frozen damage tree and evaluate each replicate before averaging",
            "actual": "stored per-replicate system_outputs",
        },
        "outputs": {},
    }
    for output_index, output_name in enumerate(output_names):
        train_values = tree_online[train, output_index]
        validation_values = tree_online[validation, output_index]
        tree_report["outputs"][output_name] = {
            "development": _summary(tree_online[development, output_index]),
            "train": _summary(train_values),
            "validation": _summary(validation_values),
            "train_expected_equal_to_validation_observed_gap": _two_sample(
                train_values, validation_values
            ),
        }
    tree_bootstrap = _bootstrap_mean_differences(
        tree_online[train],
        tree_online[validation],
        draws=args.bootstrap_draws,
        seed=args.seed + 10,
    )
    tree_bootstrap["columns"] = output_names
    tree_report["validation_minus_train_mean_bootstrap"] = tree_bootstrap

    stratum_names = sorted(set(strata[development].tolist()))
    stratum_rows = []
    for stratum in stratum_names:
        selected = development & (strata == stratum)
        selected_train = train & (strata == stratum)
        selected_validation = validation & (strata == stratum)
        row = {
            "stratum": stratum,
            "case_count": int(np.sum(selected)),
            "train_case_count": int(np.sum(selected_train)),
            "validation_case_count": int(np.sum(selected_validation)),
            "missingness_sensitive_case_mean_labels": {
                head_name: float(np.mean(case_means[selected, head_index]))
                for head_index, head_name in enumerate(head_names)
            },
            "fixed_67_component_case_mean_identification_intervals": {
                head_name: np.mean(
                    labels[f"{head_name}_identification_interval"][selected],
                    axis=(0, 1),
                ).tolist()
                for head_name in head_names
            },
            "tree_output_means": {
                name: float(np.mean(tree_online[selected, index]))
                for index, name in enumerate(output_names)
            },
        }
        if np.sum(selected_train) >= 5 and np.sum(selected_validation) >= 5:
            row["p_hit_missingness_sensitive_case_mean_train_validation_gap"] = _two_sample(
                case_means[selected_train, 0], case_means[selected_validation, 0]
            )
            p_hit_identification = labels["p_hit_identification_interval"]
            train_interval = np.mean(p_hit_identification[selected_train], axis=(0, 1))
            validation_interval = np.mean(
                p_hit_identification[selected_validation], axis=(0, 1)
            )
            row["p_hit_fixed_67_identified_validation_minus_train_mean_interval"] = [
                float(validation_interval[0] - train_interval[1]),
                float(validation_interval[1] - train_interval[0]),
            ]
        stratum_rows.append(row)
    train_stratum_counts = Counter(strata[train].tolist())
    validation_stratum_counts = Counter(strata[validation].tolist())
    train_mass = np.asarray([train_stratum_counts[name] / np.sum(train) for name in stratum_names])
    validation_mass = np.asarray(
        [validation_stratum_counts[name] / np.sum(validation) for name in stratum_names]
    )
    stratum_report = {
        "stratum_count": len(stratum_names),
        "train_validation_stratum_mass_total_variation": float(
            0.5 * np.sum(np.abs(train_mass - validation_mass))
        ),
        "rows": stratum_rows,
    }

    proxy_report = json.loads(args.proxy_report.read_text(encoding="utf-8"))
    proxy_uncertainty = json.loads(args.proxy_uncertainty.read_text(encoding="utf-8"))
    proxy_metrics = proxy_report["metrics"]
    proxy_expected_actual = {
        "expectation": "FastSamplingProxy deterministic expectation",
        "actual": "software-teacher replicate mean on identified train/validation pairs",
        "protocol_warning": proxy_uncertainty.get("mixed_protocol_warning"),
        "identified_case_component_count": proxy_metrics["identified_case_component_count"],
        "brier": proxy_metrics["hit_probability"]["brier"],
        "ece": proxy_metrics["hit_probability"]["ece"],
        "jensen_shannon_fixed_bins": proxy_metrics["hit_probability"]["jensen_shannon_fixed_bins"],
        "hit_probability_correlation": proxy_metrics["hit_probability"]["correlation"],
        "positive_energy_log1p_wasserstein1": proxy_metrics["positive_energy"]["log1p_wasserstein1"],
        "positive_energy_scope": "pairs where proxy and teacher energy are both positive",
        "multivariate_hit_probability_mmd2": proxy_metrics["multivariate_hit_probability"]["mmd2"],
        "full_mixed_protocol_uncertainty": proxy_uncertainty["intervals"],
        "replicate_count_strata": proxy_uncertainty["split_scopes"],
        "interpretation": "software-teacher agreement, not physical truth",
    }

    report = {
        "schema_version": "output-label-tree-distribution-gap@1.0.0",
        "analysis_id": "pipeline-expected-actual-20261122-v1/output-gap",
        "analysis_status": "development_descriptive_exploratory",
        "scope": {
            "train_case_count": int(np.sum(train)),
            "validation_case_count": int(np.sum(validation)),
            "sealed_test_case_count": int(np.sum(split_names == "test")),
            "sealed_test_trace_lines_seen": trace_metrics["sealed_test_trace_lines_seen"],
            "sealed_test_trace_json_lines_decoded": trace_metrics[
                "sealed_test_trace_json_lines_decoded"
            ],
            "training_runs_started": 0,
            "design_weights_all_equal_one": True,
        },
        "expectation_classes": {
            "contractual": [
                "R8 train and R32 validation replicate counts",
                "p_hit equals success_count/trial_count",
                "failed probability does not exceed degraded probability",
                "fragment-only blast states are zero",
                "combined states equal fragment states",
                "frozen damage tree recomputes per replicate before averaging",
            ],
            "same_design_split_comparison": (
                "train and validation are intended to sample the same S-CONTACT-conditioned "
                "Reference estimand, but use different QMC streams and R8/R32 teacher precision"
            ),
            "proxy_expectation": "FastSamplingProxy compared with the software teacher",
            "not_available": "externally validated physical or operational target-output distribution",
        },
        "trace_contracts": trace_metrics,
        "label_contracts": label_contract,
        "actual_label_distributions": head_report,
        "conditional_positive_distributions": conditional_report,
        "correlation_structure": correlations,
        "train_validation_component_gap": split_component_gap,
        "missingness_sensitive_case_mean_validation_minus_train_bootstrap": case_mean_bootstrap,
        "complete_67_case_mean_validation_minus_train_bootstrap": complete_case_bootstrap,
        "damage_tree": tree_report,
        "profile_region_strata": stratum_report,
        "proxy_expected_vs_teacher_actual": proxy_expected_actual,
        "evidence_boundaries": {
            "capture_fragment_audit": False,
            "unobserved_full_scale_microfunnel": [
                "fragment_generation",
                "armor_intersection",
                "occlusion",
                "penetration_event_funnel",
            ],
            "software_teacher_is_not_physical_truth": True,
            "tree_prevalence_target_declared": False,
            "physical_distribution_gap": "not_estimable_without_external_observations",
        },
        "evidence": {
            "dataset_v2_manifest": {
                "path": str(args.dataset_v2 / "manifest.json"),
                "sha256": file_hash(args.dataset_v2 / "manifest.json"),
            },
            "component_labels": {
                "path": str(core / "component_labels.npz"),
                "sha256": file_hash(core / "component_labels.npz"),
            },
            "component_state_trace": {
                "path": str(trace_path),
                "sha256": file_hash(trace_path),
            },
            "target": {"path": str(args.target), "sha256": file_hash(args.target)},
            "proxy_report": {"path": str(args.proxy_report), "sha256": file_hash(args.proxy_report)},
            "proxy_uncertainty": {
                "path": str(args.proxy_uncertainty),
                "sha256": file_hash(args.proxy_uncertainty),
            },
        },
    }
    report["report_hash"] = content_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_hash": report["report_hash"],
                "development_cases": int(np.sum(development)),
                "sealed_test_trace_json_lines_decoded": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
