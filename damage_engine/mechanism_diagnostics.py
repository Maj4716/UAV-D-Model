"""Mechanism-level diagnostics for sparse, long-tailed damage labels."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .distribution_metrics import hill_tail_stability
from .errors import InputValidationError


def variance_decomposition(
    case_means: Any,
    within_case_sample_variance: Any,
    replicate_count: Any,
) -> dict[str, Any]:
    """Separate average within-case Monte Carlo variance from between-case signal."""

    means = np.asarray(case_means, dtype=np.float64)
    within = np.asarray(within_case_sample_variance, dtype=np.float64)
    count = np.asarray(replicate_count, dtype=np.float64)
    if means.shape != within.shape or means.shape != count.shape or means.ndim < 1:
        raise InputValidationError("variance-decomposition inputs must align")
    if not np.isfinite(means).all() or not np.isfinite(within).all() or not np.isfinite(count).all() or np.any(within < 0.0) or np.any(count < 2.0):
        raise InputValidationError("variance-decomposition inputs are outside their domain")
    axis = 0
    aleatory = np.mean(within, axis=axis)
    mean_estimator_noise = np.mean(within / count, axis=axis)
    observed_between = np.var(means, axis=axis, ddof=1)
    corrected_between = np.maximum(0.0, observed_between - mean_estimator_noise)
    total_single_replicate = corrected_between + aleatory
    noise_fraction = np.divide(
        aleatory,
        total_single_replicate,
        out=np.zeros_like(aleatory),
        where=total_single_replicate > 0.0,
    )
    return {
        "aleatory_within_case_variance": np.asarray(aleatory).tolist(),
        "mean_estimator_noise_variance": np.asarray(mean_estimator_noise).tolist(),
        "observed_between_case_mean_variance": np.asarray(observed_between).tolist(),
        "noise_corrected_between_case_variance": np.asarray(corrected_between).tolist(),
        "single_replicate_noise_fraction": np.asarray(noise_fraction).tolist(),
        "noise_ceiling_mse_for_current_R": np.asarray(mean_estimator_noise).tolist(),
    }


def paired_tost_equivalence(
    left: Any,
    right: Any,
    *,
    margin: float,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Two one-sided tests through the equivalent 90% paired-difference CI."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise InputValidationError("TOST inputs must be aligned finite vectors")
    if margin <= 0.0 or not 0.0 < alpha < 0.5:
        raise InputValidationError("TOST margin and alpha are invalid")
    difference = a - b
    mean = float(np.mean(difference))
    se = float(np.std(difference, ddof=1) / math.sqrt(difference.size))
    try:
        from scipy.stats import t
        critical = float(t.ppf(1.0 - alpha, difference.size - 1))
    except ImportError as exc:  # pragma: no cover
        raise InputValidationError("SciPy is required for TOST") from exc
    interval = [mean - critical * se, mean + critical * se]
    return {
        "paired_count": int(difference.size),
        "mean_difference": mean,
        "equivalence_margin": float(margin),
        "confidence_level": float(1.0 - 2.0 * alpha),
        "difference_ci": interval,
        "equivalent": bool(interval[0] > -margin and interval[1] < margin),
    }


def component_funnel(
    *,
    emitted: Any,
    armor_hits: Any,
    armor_penetrations: Any,
    component_hits: Any,
    damage_positive: Any,
) -> dict[str, Any]:
    arrays = [np.asarray(value, dtype=np.float64) for value in (emitted, armor_hits, armor_penetrations, component_hits, damage_positive)]
    if any(array.shape != arrays[0].shape for array in arrays) or arrays[0].ndim != 1 or not all(np.isfinite(array).all() for array in arrays) or any(np.any(array < 0.0) for array in arrays):
        raise InputValidationError("funnel stages must be aligned nonnegative vectors")
    totals = np.asarray([float(np.sum(array)) for array in arrays])
    names = ["emitted", "armor_hits", "armor_penetrations", "component_hits", "damage_positive"]
    retention = [1.0]
    for index in range(1, len(totals)):
        retention.append(None if totals[index - 1] <= 0.0 else float(totals[index] / totals[index - 1]))
    losses = totals[:-1] - totals[1:]
    dominant = int(np.argmax(losses))
    return {
        "stage_totals": {name: float(total) for name, total in zip(names, totals)},
        "stage_retention": {name: value for name, value in zip(names, retention)},
        "dominant_loss_transition": f"{names[dominant]}_to_{names[dominant + 1]}",
        "loss_counts": {f"{names[index]}_to_{names[index + 1]}": float(losses[index]) for index in range(len(losses))},
        "nested_count_warning": bool(np.any(losses < -1.0e-9)),
    }


def matching_metrics(
    *,
    arrival_speed_mps: Any,
    armor_v50_mps: Any,
    deposited_energy_j: Any,
    degraded_energy50_j: Any,
    failed_energy50_j: Any,
    reachable_indicator: Any,
) -> dict[str, np.ndarray]:
    values = [np.asarray(value, dtype=np.float64) for value in (arrival_speed_mps, armor_v50_mps, deposited_energy_j, degraded_energy50_j, failed_energy50_j, reachable_indicator)]
    if any(value.shape != values[0].shape for value in values) or not all(np.isfinite(value).all() for value in values):
        raise InputValidationError("matching-metric inputs must align and be finite")
    speed, v50, energy, e_deg, e_fail, reach = values
    if np.any(speed <= 0.0) or np.any(v50 <= 0.0) or np.any(energy < 0.0) or np.any(e_deg <= 0.0) or np.any(e_fail <= 0.0) or np.any((reach < 0.0) | (reach > 1.0)):
        raise InputValidationError("matching-metric inputs are outside their domains")
    tiny = np.finfo(np.float64).tiny
    return {
        "m_armor": np.log(speed / v50),
        "m_component_degraded": np.log(np.maximum(energy, tiny) / e_deg),
        "m_component_failed": np.log(np.maximum(energy, tiny) / e_fail),
        "m_reach": reach,
    }


def tree_distribution_metrics(values: Any, *, weight: Any | None = None) -> dict[str, float]:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 1 or probability.size < 1 or not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
        raise InputValidationError("tree outputs must be a probability vector")
    weights = np.ones_like(probability) if weight is None else np.asarray(weight, dtype=np.float64)
    if weights.shape != probability.shape or not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise InputValidationError("tree weights must align and be positive")
    clipped = np.clip(probability, 1.0e-15, 1.0 - 1.0e-15)
    entropy = -clipped * np.log(clipped) - (1.0 - clipped) * np.log1p(-clipped)
    return {
        "intermediate_fraction": float(np.average((probability > 0.01) & (probability < 0.99), weights=weights)),
        "saturation_fraction": float(np.average(probability >= 0.99, weights=weights)),
        "zero_fraction": float(np.average(probability <= 1.0e-6, weights=weights)),
        "weighted_binary_entropy": float(np.average(entropy, weights=weights)),
    }


def log_tail_regression(
    energy: Any,
    hits: Any,
    range_m: Any,
    mass_velocity2: Any,
    *,
    fixed_effects: Any | None = None,
    hill_k_values: Sequence[int] = (20, 50, 100),
) -> dict[str, Any]:
    energy = np.asarray(energy, dtype=np.float64)
    hits = np.asarray(hits, dtype=np.float64)
    range_m = np.asarray(range_m, dtype=np.float64)
    mass_velocity2 = np.asarray(mass_velocity2, dtype=np.float64)
    if any(value.shape != energy.shape for value in (hits, range_m, mass_velocity2)) or energy.ndim != 1:
        raise InputValidationError("tail-regression vectors must align")
    selected = (energy > 0.0) & (hits > 0.0) & (range_m > 0.0) & (mass_velocity2 > 0.0)
    if np.count_nonzero(selected) < 10:
        raise InputValidationError("tail regression requires at least ten positive observations")
    columns = [
        np.ones(np.count_nonzero(selected)),
        np.log(hits[selected]),
        -2.0 * np.log(range_m[selected]),
        np.log(mass_velocity2[selected]),
    ]
    names = ["intercept", "log_hits", "log_inverse_range_squared", "log_mass_velocity_squared"]
    if fixed_effects is not None:
        fixed = np.asarray(fixed_effects, dtype=np.float64)
        if fixed.shape[0] != energy.size or fixed.ndim != 2 or not np.isfinite(fixed).all():
            raise InputValidationError("tail fixed effects must be a case-by-feature matrix")
        columns.extend(fixed[selected, index] for index in range(fixed.shape[1]))
        names.extend(f"fixed_effect_{index}" for index in range(fixed.shape[1]))
    design = np.column_stack(columns)
    target = np.log(energy[selected])
    coefficient, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficient
    residual = target - fitted
    total = float(np.sum((target - np.mean(target)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0.0 else 0.0
    valid_k = [k for k in hill_k_values if 2 <= int(k) < target.size]
    if not valid_k:
        valid_k = [max(2, min(target.size - 1, target.size // 5))]
    return {
        "positive_observation_count": int(target.size),
        "rank": int(rank),
        "coefficients": {name: float(value) for name, value in zip(names, coefficient)},
        "r2": r2,
        "residual_stddev": float(np.std(residual, ddof=max(1, design.shape[1]))),
        "hill": hill_tail_stability(energy[selected], k_values=valid_k),
    }


__all__ = [
    "component_funnel",
    "log_tail_regression",
    "matching_metrics",
    "paired_tost_equivalence",
    "tree_distribution_metrics",
    "variance_decomposition",
]
