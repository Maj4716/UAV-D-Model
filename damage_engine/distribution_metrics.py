"""Low-dimensional, hurdle-aware distribution comparisons from idea2 §9."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .errors import InputValidationError


def _vector(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise InputValidationError(f"{name} must be a nonempty finite vector")
    return array


def weighted_brier(probability: Any, outcome: Any, weight: Any | None = None) -> float:
    probability = _vector(probability, name="probability")
    outcome = _vector(outcome, name="outcome")
    if probability.shape != outcome.shape or np.any((probability < 0.0) | (probability > 1.0)) or np.any((outcome < 0.0) | (outcome > 1.0)):
        raise InputValidationError("Brier inputs must align and lie in [0,1]")
    weights = np.ones_like(probability) if weight is None else _vector(weight, name="weight")
    if weights.shape != probability.shape or np.any(weights <= 0.0):
        raise InputValidationError("Brier weights must align and be positive")
    return float(np.average((probability - outcome) ** 2, weights=weights))


def weighted_calibration(
    probability: Any,
    outcome: Any,
    *,
    weight: Any | None = None,
    bin_count: int = 10,
) -> dict[str, Any]:
    probability = _vector(probability, name="probability")
    outcome = _vector(outcome, name="outcome")
    if probability.shape != outcome.shape or bin_count < 2:
        raise InputValidationError("calibration inputs or bin_count are invalid")
    weights = np.ones_like(probability) if weight is None else _vector(weight, name="weight")
    if weights.shape != probability.shape or np.any(weights <= 0.0):
        raise InputValidationError("calibration weights must align and be positive")
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    index = np.minimum(bin_count - 1, np.searchsorted(edges, probability, side="right") - 1)
    index = np.maximum(0, index)
    rows = []
    total = float(np.sum(weights))
    ece = 0.0
    for bin_index in range(bin_count):
        selected = index == bin_index
        mass = float(np.sum(weights[selected]))
        if mass <= 0.0:
            continue
        predicted = float(np.average(probability[selected], weights=weights[selected]))
        observed = float(np.average(outcome[selected], weights=weights[selected]))
        ece += mass / total * abs(predicted - observed)
        rows.append(
            {
                "bin": bin_index,
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "weight_mass": mass,
                "predicted_mean": predicted,
                "observed_mean": observed,
            }
        )
    return {"ece": float(ece), "brier": weighted_brier(probability, outcome, weights), "bins": rows}


def wasserstein1_log_positive(left: Any, right: Any) -> float:
    left = _vector(left, name="left")
    right = _vector(right, name="right")
    left = np.log1p(left[left > 0.0])
    right = np.log1p(right[right > 0.0])
    if left.size < 1 or right.size < 1:
        raise InputValidationError("positive-part Wasserstein distance requires positives on both sides")
    probabilities = (np.arange(max(left.size, right.size), dtype=np.float64) + 0.5) / max(left.size, right.size)
    left_q = np.quantile(left, probabilities)
    right_q = np.quantile(right, probabilities)
    return float(np.mean(np.abs(left_q - right_q)))


def rbf_mmd2(left: Any, right: Any, *, bandwidth: float | None = None) -> dict[str, float]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or x.size == 0 or y.size == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise InputValidationError("MMD inputs must be finite matrices with the same feature dimension")
    combined = np.vstack([x, y])
    distances = np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=-1)
    if bandwidth is None:
        positive = distances[distances > 0.0]
        bandwidth = math.sqrt(0.5 * float(np.median(positive))) if positive.size else 1.0
    if not math.isfinite(float(bandwidth)) or bandwidth <= 0.0:
        raise InputValidationError("MMD bandwidth must be finite and positive")
    gamma = 1.0 / (2.0 * float(bandwidth) ** 2)
    kxx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    kyy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    kxy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    value = float(np.mean(kxx) + np.mean(kyy) - 2.0 * np.mean(kxy))
    return {"mmd2": max(0.0, value), "bandwidth": float(bandwidth)}


def jensen_shannon_binned(
    left: Any,
    right: Any,
    *,
    bin_edges: Sequence[float],
) -> float:
    left = _vector(left, name="left")
    right = _vector(right, name="right")
    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 3 or np.any(np.diff(edges) <= 0.0):
        raise InputValidationError("JS bin edges must be strictly increasing")
    p = np.histogram(left, bins=edges)[0].astype(np.float64)
    q = np.histogram(right, bins=edges)[0].astype(np.float64)
    if p.sum() == 0.0 or q.sum() == 0.0:
        raise InputValidationError("JS bins must cover observations from both samples")
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    def kl(a):
        selected = a > 0.0
        return float(np.sum(a[selected] * np.log(a[selected] / m[selected])))
    return float(0.5 * (kl(p) + kl(q)))


def hill_tail_stability(values: Any, *, k_values: Sequence[int]) -> dict[str, Any]:
    values = np.sort(_vector(values, name="tail values"))
    if np.any(values <= 0.0):
        raise InputValidationError("Hill estimation requires strictly positive values")
    rows = []
    for k in sorted(set(int(item) for item in k_values)):
        if k < 2 or k >= values.size:
            raise InputValidationError("Hill k must satisfy 2 <= k < sample count")
        threshold = values[-k - 1]
        xi = float(np.mean(np.log(values[-k:] / threshold)))
        rows.append({"k": k, "threshold": float(threshold), "xi": xi, "alpha": None if xi <= 0.0 else 1.0 / xi})
    alpha = np.asarray([row["alpha"] for row in rows if row["alpha"] is not None])
    return {
        "estimates": rows,
        "alpha_median": float(np.median(alpha)) if alpha.size else None,
        "alpha_range": [float(np.min(alpha)), float(np.max(alpha))] if alpha.size else None,
        "stability_definition": "reported_across_preregistered_k_values_no_single_k_cherry_pick",
    }


__all__ = [
    "hill_tail_stability",
    "jensen_shannon_binned",
    "rbf_mmd2",
    "wasserstein1_log_positive",
    "weighted_brier",
    "weighted_calibration",
]
