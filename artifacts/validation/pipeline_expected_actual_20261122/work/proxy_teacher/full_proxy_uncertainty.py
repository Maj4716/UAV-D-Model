"""Fast full-development case-cluster uncertainty for proxy/teacher metrics.

This companion analysis reuses the sealed-test-safe evidence loader.  It avoids
recomputing the quadratic MMD inside each bootstrap draw and instead bootstraps
the additive probability summaries and the false-unreached gate at case level.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from bootstrap_proxy_teacher import _arrays
from tools.evaluate_proxy_teacher_distribution import (
    _P_HIT_JS_EDGES,
    load_development_evidence,
)


CALIBRATION_EDGES = np.linspace(0.0, 1.0, 11)


def _interval(values: np.ndarray) -> dict[str, object]:
    return {
        "valid_resamples": int(values.size),
        "percentile_95_interval": np.quantile(values, [0.025, 0.975]).tolist(),
        "bootstrap_median": float(np.median(values)),
    }


def _js_from_counts(left: np.ndarray, right: np.ndarray) -> float:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    middle = 0.5 * (p + q)

    def kl(values: np.ndarray) -> float:
        keep = values > 0.0
        return float(np.sum(values[keep] * np.log(values[keep] / middle[keep])))

    return 0.5 * (kl(p) + kl(q))


def _wilson_upper(successes: float, trials: float) -> float | None:
    if trials <= 0.0:
        return None
    z = 1.959963984540054
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    half = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return min(1.0, center + half)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--primary-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026112201)
    args = parser.parse_args()
    if args.resamples < 100:
        raise ValueError("resamples must be at least 100")

    arrays = _arrays(args.pilot, args.target, args.physics)
    proxy_hit, teacher_hit, _, _, observed, weights = arrays[:6]
    source_audit = arrays[6]
    case_count, component_count = proxy_hit.shape
    development_evidence, split_audit = load_development_evidence(args.pilot)
    split_names = np.asarray([item.split for item in development_evidence])
    if split_names.shape != (case_count,):
        raise ValueError("development evidence order/count changed between safe reads")
    if split_audit.get("test_result_json_lines_decoded") != 0:
        raise ValueError("sealed-test decode count must remain zero")
    pair_weights = np.repeat(weights[:, None], component_count, axis=1)

    observed_weight = np.where(observed, pair_weights, 0.0)
    brier_numerator = np.sum(
        observed_weight * np.square(proxy_hit - teacher_hit), axis=1
    )
    brier_denominator = np.sum(observed_weight, axis=1)

    calibration_index = np.clip(
        np.searchsorted(CALIBRATION_EDGES, proxy_hit, side="right") - 1,
        0,
        len(CALIBRATION_EDGES) - 2,
    )
    calibration_mass = np.zeros((case_count, 10), dtype=np.float64)
    calibration_proxy_sum = np.zeros_like(calibration_mass)
    calibration_teacher_sum = np.zeros_like(calibration_mass)
    for bin_index in range(10):
        selected = observed & (calibration_index == bin_index)
        weighted = np.where(selected, pair_weights, 0.0)
        calibration_mass[:, bin_index] = np.sum(weighted, axis=1)
        calibration_proxy_sum[:, bin_index] = np.sum(weighted * proxy_hit, axis=1)
        calibration_teacher_sum[:, bin_index] = np.sum(weighted * teacher_hit, axis=1)

    js_proxy_counts = np.zeros((case_count, len(_P_HIT_JS_EDGES) - 1))
    js_teacher_counts = np.zeros_like(js_proxy_counts)
    for case_index in range(case_count):
        mask = observed[case_index]
        js_proxy_counts[case_index] = np.histogram(
            proxy_hit[case_index, mask], bins=_P_HIT_JS_EDGES
        )[0]
        js_teacher_counts[case_index] = np.histogram(
            teacher_hit[case_index, mask], bins=_P_HIT_JS_EDGES
        )[0]

    proxy_unreached = observed & (proxy_hit == 0.0)
    false_unreached = proxy_unreached & (teacher_hit > 0.0)
    unreached_by_case = np.sum(proxy_unreached, axis=1).astype(np.float64)
    false_by_case = np.sum(false_unreached, axis=1).astype(np.float64)

    def metrics(case_multiplicity: np.ndarray) -> dict[str, float]:
        total_weight = float(case_multiplicity @ brier_denominator)
        brier = float(case_multiplicity @ brier_numerator) / total_weight

        bin_mass = case_multiplicity @ calibration_mass
        proxy_sum = case_multiplicity @ calibration_proxy_sum
        teacher_sum = case_multiplicity @ calibration_teacher_sum
        nonempty = bin_mass > 0.0
        ece = float(
            np.sum(
                bin_mass[nonempty]
                / np.sum(bin_mass)
                * np.abs(
                    proxy_sum[nonempty] / bin_mass[nonempty]
                    - teacher_sum[nonempty] / bin_mass[nonempty]
                )
            )
        )

        proxy_hist = case_multiplicity @ js_proxy_counts
        teacher_hist = case_multiplicity @ js_teacher_counts
        false_count = float(case_multiplicity @ false_by_case)
        unreached_count = float(case_multiplicity @ unreached_by_case)
        return {
            "brier": brier,
            "ece": ece,
            "jensen_shannon_fixed_bins": _js_from_counts(proxy_hist, teacher_hist),
            "false_unreached_rate": false_count / unreached_count,
        }

    def point_for_indices(indices: np.ndarray) -> dict[str, float | int | None]:
        multiplicity = np.zeros(case_count, dtype=np.float64)
        multiplicity[indices] = 1.0
        point_result = metrics(multiplicity)
        unreached_count = float(np.sum(unreached_by_case[indices]))
        false_count = float(np.sum(false_by_case[indices]))
        point_result["case_count"] = int(indices.size)
        point_result["proxy_unreached_pair_count"] = int(unreached_count)
        point_result["teacher_positive_among_proxy_unreached"] = int(false_count)
        point_result["false_unreached_wilson_upper"] = _wilson_upper(
            false_count, unreached_count
        )
        point_result["identified_case_component_count"] = int(
            np.sum(observed[indices])
        )
        return point_result

    metric_names = (
        "brier",
        "ece",
        "jensen_shannon_fixed_bins",
        "false_unreached_rate",
    )

    def bootstrap_for_indices(indices: np.ndarray, seed: int) -> dict[str, object]:
        rng = np.random.default_rng(seed)
        draws = {name: np.empty(args.resamples) for name in metric_names}
        for draw_index in range(args.resamples):
            sampled = rng.choice(indices, size=indices.size, replace=True)
            multiplicity = np.bincount(sampled, minlength=case_count).astype(np.float64)
            result = metrics(multiplicity)
            for name in draws:
                draws[name][draw_index] = result[name]
        return {
            "point_estimates": point_for_indices(indices),
            "intervals": {name: _interval(values) for name, values in draws.items()},
        }

    all_indices = np.arange(case_count)
    full_scope = bootstrap_for_indices(all_indices, args.seed)
    point = full_scope["point_estimates"]
    train_indices = np.flatnonzero(split_names == "train")
    validation_indices = np.flatnonzero(split_names == "validation")
    if train_indices.size != 2000 or validation_indices.size != 300:
        raise ValueError("unexpected train/validation development split counts")
    split_scopes = {
        "train_R8": {
            "split": "train",
            "replicate_count": 8,
            **bootstrap_for_indices(train_indices, args.seed + 1),
        },
        "validation_R32": {
            "split": "validation",
            "replicate_count": 32,
            **bootstrap_for_indices(validation_indices, args.seed + 2),
        },
    }

    primary = json.loads(args.primary_report.read_text(encoding="utf-8"))
    primary_hit = primary["metrics"]["hit_probability"]
    cross_checks = {
        "brier_absolute_error": abs(point["brier"] - primary_hit["brier"]),
        "ece_absolute_error": abs(point["ece"] - primary_hit["ece"]),
        "js_absolute_error": abs(
            point["jensen_shannon_fixed_bins"]
            - primary_hit["jensen_shannon_fixed_bins"]
        ),
    }
    cross_checks["all_within_1e-12"] = all(
        value <= 1.0e-12 for value in cross_checks.values()
    )

    report = {
        "schema_version": "proxy-teacher-full-case-cluster-bootstrap@1.1.0",
        "analysis_role": "development_descriptive_uncertainty_not_qmc_design_variance",
        "source": {
            "dataset_id": source_audit.get("source_dataset_id"),
            "development_case_count": case_count,
            "component_count": component_count,
            "test_case_count": source_audit.get("test_row_count"),
            "test_result_json_lines_decoded": 0,
            "test_metrics_computed": False,
            "primary_report": str(args.primary_report),
        },
        "bootstrap": {
            "unit": "case_with_all_components",
            "method": "nonparametric_case_cluster_percentile_bootstrap",
            "resamples": args.resamples,
            "seed": args.seed,
            "scope_note": (
                "Finite development-case uncertainty conditional on one QMC design; "
                "not between-scramble or real-world uncertainty."
            ),
        },
        "mixed_protocol_warning": (
            "The full development summary mixes train R8 and validation R32. "
            "Use split_scopes for replicate-count-specific interpretation."
        ),
        "point_estimates": point,
        "intervals": full_scope["intervals"],
        "split_scopes": split_scopes,
        "primary_report_cross_checks": cross_checks,
    }
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8")
    report["report_sha256_without_self_field"] = "sha256:" + sha256(canonical).hexdigest()
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "point": point}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
