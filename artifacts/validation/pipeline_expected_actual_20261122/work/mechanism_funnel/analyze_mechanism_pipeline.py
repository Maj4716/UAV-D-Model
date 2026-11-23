"""Stage-wise expected-versus-teacher analysis on train/validation only.

The script uses the project's sealed-test-safe loader.  Test result lines are
skipped before JSON decoding.  FastSamplingProxy supplies the explicit expected
mechanism totals; archived stochastic-teacher replicate means supply actuals.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.distribution_metrics import (
    jensen_shannon_binned,
    weighted_brier,
    weighted_calibration,
)
from damage_engine.engine import DamageEngine
from damage_engine.fast_sampling_proxy import FastSamplingProxy
from damage_engine.hurdle_targets import extract_component_hurdle_targets
from damage_engine.registry import PhysicsRegistry
from damage_engine.utils import content_hash, file_hash
from tools.evaluate_proxy_teacher_distribution import (
    _P_HIT_JS_EDGES,
    load_development_evidence,
)


STAGE_SPECS = {
    "source_fragments": {
        "expected_key": "source_fragment_count",
        "actual_key": "total_fragments",
        "expectation_source": "frozen warhead fragment_source.total_count",
        "unit": "fragments_per_replicate",
    },
    "armor_hits": {
        "expected_key": "expected_armor_hits",
        "actual_key": "total_armor_hits",
        "expectation_source": "FastSamplingProxy weighted ray expectation",
        "unit": "intersections_per_replicate",
    },
    "armor_penetrations": {
        "expected_key": "expected_armor_penetrations",
        "actual_key": "total_armor_penetrations",
        "expectation_source": "FastSamplingProxy expected penetration paths",
        "unit": "penetrations_per_replicate",
    },
    "component_hits": {
        "expected_key": "expected_total_component_hits",
        "actual_key": "total_component_hits",
        "expectation_source": "FastSamplingProxy weighted component ray expectation",
        "unit": "component_intersections_per_replicate",
    },
    "component_penetrations": {
        "expected_key": "expected_total_component_penetrations",
        "actual_key": "total_component_penetrations",
        "expectation_source": "FastSamplingProxy expected component penetration paths",
        "unit": "component_penetrations_per_replicate",
    },
    "component_deposited_energy_j": {
        "expected_key": "expected_total_component_absorbed_energy_j",
        "actual_key": "component_deposited_energy_j",
        "expectation_source": "FastSamplingProxy expected absorbed energy",
        "unit": "joule_per_replicate_summed_over_components",
    },
    "component_damage_energy_j": {
        "expected_key": "expected_total_component_damage_energy_j",
        "actual_key": "component_damage_energy_j",
        "expectation_source": "FastSamplingProxy expected vulnerable-area-adjusted damage energy",
        "unit": "joule_per_replicate_summed_over_components",
    },
}


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "minimum": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(np.max(values)),
        "zero_fraction": float(np.mean(values == 0.0)),
    }


def _bootstrap_mean_interval(
    values: np.ndarray, *, draws: int, seed: int
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    chunk = 250
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = np.mean(values[indices], axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def _correlations(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return {"status": "undefined_constant_or_insufficient", "pearson": None, "spearman": None}
    return {
        "status": "estimated",
        "pearson": float(pearsonr(left, right).statistic),
        "spearman": float(spearmanr(left, right).statistic),
    }


def _log1p_wasserstein_equal_count(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(
                np.sort(np.log1p(np.maximum(left, 0.0)))
                - np.sort(np.log1p(np.maximum(right, 0.0)))
            )
        )
    )


def _paired_summary(
    expected: np.ndarray,
    actual: np.ndarray,
    mask: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    keep = mask & np.isfinite(expected) & np.isfinite(actual)
    expected = expected[keep]
    actual = actual[keep]
    difference = actual - expected
    expected_mean = float(np.mean(expected))
    actual_mean = float(np.mean(actual))
    return {
        "paired_case_count": int(expected.size),
        "expected": _summary(expected),
        "actual": _summary(actual),
        "actual_minus_expected_mean": float(np.mean(difference)),
        "actual_minus_expected_mean_case_bootstrap_95_interval": _bootstrap_mean_interval(
            difference, draws=draws, seed=seed
        ),
        "actual_to_expected_mean_ratio": None
        if expected_mean == 0.0
        else actual_mean / expected_mean,
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "root_mean_squared_error": float(np.sqrt(np.mean(np.square(difference)))),
        "log1p_wasserstein1_all_nonnegative": _log1p_wasserstein_equal_count(
            expected, actual
        ),
        "correlation": _correlations(expected, actual),
        "expected_zero_actual_positive_count": int(np.sum((expected == 0.0) & (actual > 0.0))),
        "actual_zero_expected_positive_count": int(np.sum((actual == 0.0) & (expected > 0.0))),
    }


def _soft_head_metrics(
    expected: np.ndarray,
    actual: np.ndarray,
    observed: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    case_count, component_count = expected.shape
    identified_count = int(np.sum(observed))
    if case_count == 0 or identified_count == 0:
        return {
            "status": "no_identified_case_component_pairs",
            "identified_case_component_count": identified_count,
            "shape": [case_count, component_count],
            "semantics": "proxy probability versus software-teacher replicate-mean soft label",
        }
    weights = np.ones_like(expected)
    calibration = weighted_calibration(
        expected[observed], actual[observed], weight=weights[observed]
    )
    brier = weighted_brier(expected[observed], actual[observed], weights[observed])
    js = jensen_shannon_binned(
        expected[observed], actual[observed], bin_edges=_P_HIT_JS_EDGES
    )
    expected_flat = expected[observed]
    actual_flat = actual[observed]

    case_denominator = np.sum(observed, axis=1).astype(np.float64)
    case_brier_numerator = np.sum(
        np.where(observed, np.square(expected - actual), 0.0), axis=1
    )
    case_difference_numerator = np.sum(
        np.where(observed, actual - expected, 0.0), axis=1
    )
    false_zero = observed & (expected == 0.0) & (actual > 0.0)
    expected_zero = observed & (expected == 0.0)
    case_false_zero = np.sum(false_zero, axis=1).astype(np.float64)
    case_expected_zero = np.sum(expected_zero, axis=1).astype(np.float64)
    rng = np.random.default_rng(seed)
    brier_draws = np.empty(draws)
    difference_draws = np.empty(draws)
    false_zero_draws = np.empty(draws)
    for draw in range(draws):
        index = rng.integers(0, case_count, size=case_count)
        denominator = float(np.sum(case_denominator[index]))
        brier_draws[draw] = float(np.sum(case_brier_numerator[index])) / denominator
        difference_draws[draw] = float(np.sum(case_difference_numerator[index])) / denominator
        zero_denominator = float(np.sum(case_expected_zero[index]))
        false_zero_draws[draw] = (
            float(np.sum(case_false_zero[index])) / zero_denominator
            if zero_denominator > 0.0
            else np.nan
        )

    false_zero_count = int(np.sum(false_zero))
    expected_zero_count = int(np.sum(expected_zero))
    return {
        "status": "estimated",
        "identified_case_component_count": identified_count,
        "brier": brier,
        "brier_case_cluster_bootstrap_95_interval": np.quantile(
            brier_draws, [0.025, 0.975]
        ).tolist(),
        "ece": calibration["ece"],
        "calibration_bins": calibration["bins"],
        "jensen_shannon_fixed_bins": js,
        "expected_mean": float(np.mean(expected_flat)),
        "actual_mean": float(np.mean(actual_flat)),
        "actual_minus_expected_mean": float(np.mean(actual_flat - expected_flat)),
        "actual_minus_expected_mean_case_cluster_bootstrap_95_interval": np.quantile(
            difference_draws, [0.025, 0.975]
        ).tolist(),
        "correlation": _correlations(expected_flat, actual_flat),
        "expected_zero_pair_count": expected_zero_count,
        "teacher_positive_among_expected_zero": false_zero_count,
        "expected_zero_teacher_positive_rate": None
        if expected_zero_count == 0
        else false_zero_count / expected_zero_count,
        "expected_zero_teacher_positive_case_cluster_bootstrap_95_interval": (
            np.quantile(
                false_zero_draws[np.isfinite(false_zero_draws)], [0.025, 0.975]
            ).tolist()
            if np.any(np.isfinite(false_zero_draws))
            else None
        ),
        "shape": [case_count, component_count],
        "semantics": "proxy probability versus software-teacher replicate-mean soft label",
    }


def _mean_or_zero(record: dict[str, Any]) -> float:
    value = record.get("mean")
    return 0.0 if value is None else float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026112201)
    args = parser.parse_args()

    evidence, source_audit = load_development_evidence(args.dataset)
    target = json.loads(args.target.read_text(encoding="utf-8-sig"))
    engine = DamageEngine(target, PhysicsRegistry.from_file(args.physics))
    proxy = FastSamplingProxy(engine)
    component_ids = list(engine.compiled_target.component_order)
    case_count = len(evidence)
    component_count = len(component_ids)

    split_names = np.empty(case_count, dtype="U16")
    profiles = np.empty(case_count, dtype="U32")
    regions = np.empty(case_count, dtype="U32")
    replicate_counts = np.zeros(case_count, dtype=np.int64)
    domain_complete = np.zeros(case_count, dtype=bool)
    proxy_domain_complete = np.zeros(case_count, dtype=bool)
    affected_fraction = np.zeros(case_count, dtype=np.float64)
    proxy_out_of_domain_fraction = np.zeros(case_count, dtype=np.float64)
    stage_expected = {name: np.zeros(case_count) for name in STAGE_SPECS}
    stage_actual = {name: np.zeros(case_count) for name in STAGE_SPECS}
    summary_actual = {
        name: np.zeros(case_count)
        for name in (
            "total_existing_hole_reuses",
            "total_opening_primitives",
            "total_out_of_domain_armor_impacts",
            "total_out_of_domain_component_impacts",
            "total_out_of_domain_impacts",
            "total_perforation_events",
            "total_unique_armor_holes",
        )
    }
    head_expected = {
        name: np.zeros((case_count, component_count))
        for name in ("p_hit", "p_degraded", "p_failed")
    }
    head_actual = {name: np.zeros_like(next(iter(head_expected.values()))) for name in head_expected}
    head_observed = {
        name: np.zeros((case_count, component_count), dtype=bool) for name in head_expected
    }
    tree_names: list[str] | None = None
    tree_expected: dict[str, np.ndarray] = {}
    tree_actual: dict[str, np.ndarray] = {}
    trace_max_component_error = 0.0
    trace_max_tree_error = 0.0
    trace_component_mismatch_count = 0
    trace_tree_mismatch_count = 0
    trace_record_count_mismatch_count = 0
    storage_mode_counts: dict[str, int] = {}
    raw_trace_retained_counts: dict[str, int] = {}
    component_trace_retained_counts: dict[str, int] = {}
    proxy_domain_status_counts: dict[str, int] = {}

    for row, item in enumerate(evidence):
        case = item.case
        result = item.result
        proxy_result = proxy.evaluate(deepcopy(case))
        split_names[row] = item.split
        profiles[row] = str(case["munition"]["profile_id"])
        regions[row] = str(case["metadata"]["contact_manifold_sampling"]["region_id"])
        replicate_counts[row] = int(case["analysis"]["replicates"])
        coverage = result["penetration_domain_coverage"]
        affected_fraction[row] = float(coverage["affected_replicate_fraction"])
        domain_complete[row] = affected_fraction[row] == 0.0
        proxy_status = str(proxy_result["proxy_domain_status"])
        proxy_domain_complete[row] = proxy_status == "complete_proxy_domain"
        proxy_out_of_domain_fraction[row] = float(
            proxy_result["proxy_domain_coverage"]["out_of_domain_packet_fraction"]
        )
        proxy_domain_status_counts[proxy_status] = proxy_domain_status_counts.get(proxy_status, 0) + 1

        stage_expected["source_fragments"][row] = float(
            case["warhead"]["fragment_source"]["total_count"]
        )
        stage_actual["source_fragments"][row] = _mean_or_zero(
            result["summary"]["total_fragments"]
        )
        for name, spec in STAGE_SPECS.items():
            if name == "source_fragments":
                continue
            if spec["actual_key"] in result["summary"]:
                stage_actual[name][row] = _mean_or_zero(
                    result["summary"][spec["actual_key"]]
                )
            stage_expected[name][row] = float(proxy_result[spec["expected_key"]])
        stage_actual["component_deposited_energy_j"][row] = sum(
            _mean_or_zero(component["fragment_metrics"]["deposited_energy_j"])
            for component in result["components"].values()
        )
        stage_actual["component_damage_energy_j"][row] = sum(
            _mean_or_zero(component["fragment_metrics"]["damage_energy_j"])
            for component in result["components"].values()
        )
        for name in summary_actual:
            summary_actual[name][row] = _mean_or_zero(result["summary"][name])

        targets = extract_component_hurdle_targets(result)
        for column, component_id in enumerate(component_ids):
            component_target = targets["components"][component_id]
            proxy_metrics = proxy_result["component_metrics"][component_id]
            head_expected["p_hit"][row, column] = -math.expm1(
                -max(0.0, float(proxy_metrics["expected_hits"]))
            )
            head_expected["p_degraded"][row, column] = float(
                proxy_result["component_state_probabilities"][component_id].get(
                    "degraded", 0.0
                )
            )
            head_expected["p_failed"][row, column] = float(
                proxy_result["component_state_probabilities"][component_id].get(
                    "failed", 0.0
                )
            )
            head_actual["p_hit"][row, column] = float(
                component_target["p_hit"]["value"]
            )
            head_actual["p_degraded"][row, column] = float(
                component_target["state_probabilities"]["p_degraded"]["value"]
            )
            head_actual["p_failed"][row, column] = float(
                component_target["state_probabilities"]["p_failed"]["value"]
            )
            head_observed["p_hit"][row, column] = bool(
                component_target["p_hit"]["mask"]
            )
            head_observed["p_degraded"][row, column] = bool(
                component_target["state_probabilities"]["p_degraded"]["mask"]
            )
            head_observed["p_failed"][row, column] = bool(
                component_target["state_probabilities"]["p_failed"]["mask"]
            )

        if tree_names is None:
            tree_names = sorted(proxy_result["tree_outputs"])
            tree_expected = {name: np.zeros(case_count) for name in tree_names}
            tree_actual = {name: np.zeros(case_count) for name in tree_names}
        for name in tree_names:
            tree_expected[name][row] = float(proxy_result["tree_outputs"][name])
            tree_actual[name][row] = _mean_or_zero(
                result["damage_tree"]["outputs"][name]
            )

        storage = result["replicate_statistics_storage"]
        mode = str(storage["mode"])
        storage_mode_counts[mode] = storage_mode_counts.get(mode, 0) + 1
        raw_key = str(bool(storage["raw_trace_retained"])).lower()
        component_key = str(bool(storage["component_state_trace_retained"])).lower()
        raw_trace_retained_counts[raw_key] = raw_trace_retained_counts.get(raw_key, 0) + 1
        component_trace_retained_counts[component_key] = component_trace_retained_counts.get(component_key, 0) + 1
        records = result["component_state_trace"]["records"]
        if len(records) != replicate_counts[row]:
            trace_record_count_mismatch_count += 1
        if records:
            for name in tree_names:
                trace_mean = float(np.mean([record["system_outputs"][name] for record in records]))
                error = abs(trace_mean - tree_actual[name][row])
                trace_max_tree_error = max(trace_max_tree_error, error)
                trace_tree_mismatch_count += int(error > 1.0e-12)
            for component_id in component_ids:
                for state in ("degraded", "failed"):
                    trace_mean = float(
                        np.mean(
                            [
                                record["component_mechanism_states"]["fragment"][component_id][state]
                                for record in records
                            ]
                        )
                    )
                    archived = _mean_or_zero(
                        result["components"][component_id]["damage_states"]["fragment"][state]
                    )
                    error = abs(trace_mean - archived)
                    trace_max_component_error = max(trace_max_component_error, error)
                    trace_component_mismatch_count += int(error > 1.0e-12)

        if (row + 1) % 250 == 0 or row + 1 == case_count:
            print(f"processed {row + 1}/{case_count}", flush=True)

    split_masks = {
        "all_mixed_R8_R32": np.ones(case_count, dtype=bool),
        "train_R8": split_names == "train",
        "validation_R32": split_names == "validation",
    }
    joint_domain_complete = domain_complete & proxy_domain_complete
    stage_comparisons: dict[str, Any] = {}
    for stage_index, (name, spec) in enumerate(STAGE_SPECS.items()):
        stage_comparisons[name] = {
            "unit": spec["unit"],
            "expectation_source": spec["expectation_source"],
            "all_cases_by_protocol": {},
            "domain_complete_cases_by_protocol": {},
            "proxy_domain_complete_cases_by_protocol": {},
            "teacher_and_proxy_domain_complete_cases_by_protocol": {},
            "all_case_expectation_semantics": (
                "proxy expectation is right-censored/lower-bound for cases with "
                "partial_proxy_domain_censored status"
            ),
        }
        for scope_index, (scope, scope_mask) in enumerate(split_masks.items()):
            stage_comparisons[name]["all_cases_by_protocol"][scope] = _paired_summary(
                stage_expected[name],
                stage_actual[name],
                scope_mask,
                draws=args.bootstrap_draws,
                seed=args.seed + 100 * stage_index + scope_index,
            )
            complete_mask = scope_mask & domain_complete
            stage_comparisons[name]["domain_complete_cases_by_protocol"][scope] = _paired_summary(
                stage_expected[name],
                stage_actual[name],
                complete_mask,
                draws=args.bootstrap_draws,
                seed=args.seed + 1000 + 100 * stage_index + scope_index,
            )
            proxy_complete_mask = scope_mask & proxy_domain_complete
            stage_comparisons[name]["proxy_domain_complete_cases_by_protocol"][scope] = _paired_summary(
                stage_expected[name],
                stage_actual[name],
                proxy_complete_mask,
                draws=args.bootstrap_draws,
                seed=args.seed + 2000 + 100 * stage_index + scope_index,
            )
            joint_complete_mask = scope_mask & joint_domain_complete
            stage_comparisons[name]["teacher_and_proxy_domain_complete_cases_by_protocol"][scope] = _paired_summary(
                stage_expected[name],
                stage_actual[name],
                joint_complete_mask,
                draws=args.bootstrap_draws,
                seed=args.seed + 3000 + 100 * stage_index + scope_index,
            )

    conversion_pairs = {
        "armor_hit_per_source_fragment": ("source_fragments", "armor_hits"),
        "armor_penetration_per_armor_hit": ("armor_hits", "armor_penetrations"),
        "component_interaction_per_source_fragment": ("source_fragments", "component_hits"),
        "component_penetration_per_component_hit": ("component_hits", "component_penetrations"),
    }
    conversion_rates: dict[str, Any] = {
        "all_cases": {},
        "teacher_and_proxy_domain_complete_cases": {},
    }
    for label, (denominator_name, numerator_name) in conversion_pairs.items():
        for selection_name, selection_mask in (
            ("all_cases", np.ones(case_count, dtype=bool)),
            ("teacher_and_proxy_domain_complete_cases", joint_domain_complete),
        ):
            conversion_rates[selection_name][label] = {}
            for scope, scope_mask in split_masks.items():
                mask = scope_mask & selection_mask
                expected_denominator = float(np.sum(stage_expected[denominator_name][mask]))
                actual_denominator = float(np.sum(stage_actual[denominator_name][mask]))
                conversion_rates[selection_name][label][scope] = {
                    "case_count": int(np.sum(mask)),
                    "expected_proxy_ratio_of_sums": None
                    if expected_denominator == 0.0
                    else float(np.sum(stage_expected[numerator_name][mask])) / expected_denominator,
                    "actual_teacher_ratio_of_sums": None
                    if actual_denominator == 0.0
                    else float(np.sum(stage_actual[numerator_name][mask])) / actual_denominator,
                    "semantics": "interaction_ratio_of_case_level_replicate_mean_totals_not_an_iid_fragment_probability",
                }

    head_metrics = {
        name: _soft_head_metrics(
            head_expected[name],
            head_actual[name],
            head_observed[name],
            draws=args.bootstrap_draws,
            seed=args.seed + 5000 + index,
        )
        for index, name in enumerate(("p_hit", "p_degraded", "p_failed"))
    }
    head_metrics_by_split = {}
    for split_index, (scope, scope_mask) in enumerate(split_masks.items()):
        head_metrics_by_split[scope] = {}
        for head_index, name in enumerate(("p_hit", "p_degraded", "p_failed")):
            head_metrics_by_split[scope][name] = _soft_head_metrics(
                head_expected[name][scope_mask],
                head_actual[name][scope_mask],
                head_observed[name][scope_mask],
                draws=args.bootstrap_draws,
                seed=args.seed + 6000 + 100 * split_index + head_index,
            )

    head_metrics_complete67_by_split: dict[str, Any] = {}
    head_metrics_joint_complete_by_split: dict[str, Any] = {}
    head_metrics_joint_complete67_by_split: dict[str, Any] = {}
    for split_index, (scope, scope_mask) in enumerate(split_masks.items()):
        head_metrics_complete67_by_split[scope] = {}
        head_metrics_joint_complete_by_split[scope] = {}
        head_metrics_joint_complete67_by_split[scope] = {}
        for head_index, name in enumerate(("p_hit", "p_degraded", "p_failed")):
            complete67 = np.all(head_observed[name], axis=1)
            complete67_mask = scope_mask & complete67
            joint_mask = scope_mask & joint_domain_complete
            joint_complete67_mask = joint_mask & complete67
            head_metrics_complete67_by_split[scope][name] = _soft_head_metrics(
                head_expected[name][complete67_mask],
                head_actual[name][complete67_mask],
                head_observed[name][complete67_mask],
                draws=args.bootstrap_draws,
                seed=args.seed + 8000 + 100 * split_index + head_index,
            )
            head_metrics_joint_complete_by_split[scope][name] = _soft_head_metrics(
                head_expected[name][joint_mask],
                head_actual[name][joint_mask],
                head_observed[name][joint_mask],
                draws=args.bootstrap_draws,
                seed=args.seed + 9000 + 100 * split_index + head_index,
            )
            head_metrics_joint_complete67_by_split[scope][name] = _soft_head_metrics(
                head_expected[name][joint_complete67_mask],
                head_actual[name][joint_complete67_mask],
                head_observed[name][joint_complete67_mask],
                draws=args.bootstrap_draws,
                seed=args.seed + 10000 + 100 * split_index + head_index,
            )

    tree_comparisons = {
        "teacher_domain_complete_cases": {
            name: _paired_summary(
                tree_expected[name], tree_actual[name], domain_complete,
                draws=args.bootstrap_draws, seed=args.seed + 7000 + index,
            )
            for index, name in enumerate(tree_names or [])
        },
        "teacher_and_proxy_domain_complete_cases": {
            name: _paired_summary(
                tree_expected[name], tree_actual[name], joint_domain_complete,
                draws=args.bootstrap_draws, seed=args.seed + 11000 + index,
            )
            for index, name in enumerate(tree_names or [])
        },
    }

    strata: dict[str, Any] = {}
    stratum_index = 0
    for profile in sorted(set(profiles.tolist())):
        for region in sorted(set(regions.tolist())):
            mask = (profiles == profile) & (regions == region)
            if not np.any(mask):
                continue
            key = f"{region}::{profile}"
            joint_stratum_mask = mask & joint_domain_complete
            strata[key] = {
                "case_count": int(np.sum(mask)),
                "split_counts": {
                    split: int(np.sum(mask & (split_names == split)))
                    for split in ("train", "validation")
                },
                "domain_complete_case_fraction": float(np.mean(domain_complete[mask])),
                "proxy_domain_complete_case_fraction": float(np.mean(proxy_domain_complete[mask])),
                "teacher_and_proxy_domain_complete_case_fraction": float(np.mean(joint_domain_complete[mask])),
                "teacher_and_proxy_domain_complete_case_count": int(np.sum(joint_stratum_mask)),
                "stages": {
                    name: {
                        "expected_mean": float(np.mean(stage_expected[name][mask])),
                        "actual_mean": float(np.mean(stage_actual[name][mask])),
                        "actual_minus_expected_mean": float(
                            np.mean(stage_actual[name][mask] - stage_expected[name][mask])
                        ),
                    }
                    for name in STAGE_SPECS
                },
                "p_hit_brier": weighted_brier(
                    head_expected["p_hit"][mask][head_observed["p_hit"][mask]],
                    head_actual["p_hit"][mask][head_observed["p_hit"][mask]],
                ),
                "teacher_and_proxy_domain_complete_stages": {
                    name: _paired_summary(
                        stage_expected[name],
                        stage_actual[name],
                        joint_stratum_mask,
                        draws=args.bootstrap_draws,
                        seed=args.seed + 12000 + 100 * stratum_index + stage_index,
                    )
                    for stage_index, name in enumerate(STAGE_SPECS)
                },
                "teacher_and_proxy_domain_complete_p_hit": _soft_head_metrics(
                    head_expected["p_hit"][joint_stratum_mask],
                    head_actual["p_hit"][joint_stratum_mask],
                    head_observed["p_hit"][joint_stratum_mask],
                    draws=args.bootstrap_draws,
                    seed=args.seed + 14000 + stratum_index,
                ),
            }
            stratum_index += 1

    actual_mechanism_distribution = {
        "summary_metrics": {name: _summary(values) for name, values in summary_actual.items()},
        "domain_censoring": {
            "teacher": {
                "affected_case_count": int(np.sum(~domain_complete)),
                "affected_case_fraction": float(np.mean(~domain_complete)),
                "affected_replicate_fraction": _summary(affected_fraction),
            },
            "proxy": {
                "affected_case_count": int(np.sum(~proxy_domain_complete)),
                "affected_case_fraction": float(np.mean(~proxy_domain_complete)),
                "out_of_domain_packet_fraction": _summary(proxy_out_of_domain_fraction),
            },
            "teacher_and_proxy_complete_case_count": int(np.sum(joint_domain_complete)),
            "teacher_and_proxy_complete_case_fraction": float(np.mean(joint_domain_complete)),
        },
        "identity_checks": {
            "opening_primitives_minus_perforation_events_max_abs": float(
                np.max(
                    np.abs(
                        summary_actual["total_opening_primitives"]
                        - summary_actual["total_perforation_events"]
                    )
                )
            ),
            "out_of_domain_total_minus_parts_max_abs": float(
                np.max(
                    np.abs(
                        summary_actual["total_out_of_domain_impacts"]
                        - summary_actual["total_out_of_domain_armor_impacts"]
                        - summary_actual["total_out_of_domain_component_impacts"]
                    )
                )
            ),
        },
    }

    report = {
        "schema_version": "mechanism-pipeline-expected-actual@1.0.0",
        "analysis_id": "pipeline-expected-actual-20261122-v2-mechanism",
        "analysis_scope": "train_validation_only_full_aggregate_mechanism_analysis",
        "source": {
            **source_audit,
            "target_path": str(args.target),
            "target_sha256": file_hash(args.target),
            "physics_path": str(args.physics),
            "physics_sha256": file_hash(args.physics),
            "test_result_json_lines_decoded": 0,
            "test_metrics_computed": False,
        },
        "population": {
            "case_count": case_count,
            "component_count": component_count,
            "split_counts": {scope: int(np.sum(mask)) for scope, mask in split_masks.items()},
            "replicate_counts_by_split": {
                split: sorted(set(replicate_counts[split_names == split].tolist()))
                for split in ("train", "validation")
            },
            "design_weights_unique": sorted(set(float(item.design_weight) for item in evidence)),
        },
        "bootstrap": {
            "unit": "case_with_all_components_and_mechanism_totals",
            "draws": args.bootstrap_draws,
            "seed": args.seed,
            "interval": "percentile_95",
            "scope": "case-resampling sensitivity conditional on the fixed QMC design; not repeated-QMC uncertainty",
        },
        "stage_comparisons": stage_comparisons,
        "conversion_rates": conversion_rates,
        "component_probability_heads": head_metrics,
        "component_probability_heads_by_protocol": head_metrics_by_split,
        "component_probability_heads_complete67_by_protocol": head_metrics_complete67_by_split,
        "component_probability_heads_teacher_and_proxy_complete_by_protocol": head_metrics_joint_complete_by_split,
        "component_probability_heads_teacher_and_proxy_complete67_by_protocol": head_metrics_joint_complete67_by_split,
        "damage_tree_outputs_domain_complete_cases": tree_comparisons,
        "actual_mechanism_distribution": actual_mechanism_distribution,
        "profile_region_strata": strata,
        "trace_aggregation_consistency": {
            "component_state_max_abs_error": trace_max_component_error,
            "component_state_mismatch_count_gt_1e_12": trace_component_mismatch_count,
            "system_output_max_abs_error": trace_max_tree_error,
            "system_output_mismatch_count_gt_1e_12": trace_tree_mismatch_count,
            "trace_record_count_mismatch_case_count": trace_record_count_mismatch_count,
        },
        "storage_and_observability": {
            "storage_mode_counts": storage_mode_counts,
            "raw_fragment_trace_retained_counts": raw_trace_retained_counts,
            "component_state_trace_retained_counts": component_trace_retained_counts,
            "proxy_domain_status_counts": proxy_domain_status_counts,
            "teacher_domain_complete_case_count": int(np.sum(domain_complete)),
            "proxy_domain_complete_case_count": int(np.sum(proxy_domain_complete)),
            "teacher_and_proxy_domain_complete_case_count": int(np.sum(joint_domain_complete)),
            "capture_fragment_audit": False,
            "directly_unobservable_full_scale_stages": [
                "fragment_level_geometry_candidate_path",
                "fragment_level_processed_path_and_blocker",
                "explicit_occlusion_or_blocked_before_component_counts",
                "intersection_level_penetration_probability_calibration",
                "intersection_level_v50_and_e50_margin_distributions",
            ],
        },
        "interpretation_guards": {
            "proxy_is_expected_reference_not_physical_truth": True,
            "partial_proxy_domain_expected_totals_are_right_censored_lower_bounds": True,
            "teacher_is_software_distribution_not_real_world_distribution": True,
            "component_hits_are_interactions_not_unique_fragments": True,
            "component_geometry_can_precede_armor_in_candidate_paths": True,
            "mixed_R8_R32_metrics_are_descriptive_and_split_metrics_are_provided": True,
            "sealed_test_results_decoded": 0,
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
