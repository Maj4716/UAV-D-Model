"""Numerically stable primitives for declared weighted estimands.

This module owns weight eligibility and scale-invariant accumulation.  The
callers still own the scientific meaning of each estimand: a design-weight
ESS, a soft-event ESS, and a task weight are intentionally not interchangeable
just because they share the same Kish formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import InputValidationError


def scale_positive_weights(
    weights: np.ndarray, *, field: str = "weights"
) -> tuple[np.ndarray, float, float]:
    """Return positive weights divided by their maximum and their stable sum."""

    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise InputValidationError(f"{field} must be a nonempty vector")
    if np.any(~np.isfinite(values) | (values <= 0.0)):
        raise InputValidationError(f"{field} must contain finite positive values")
    scale = float(np.max(values))
    scaled = values / scale
    scaled_sum = float(np.sum(scaled, dtype=np.float64))
    if (
        np.any(~np.isfinite(scaled))
        or not math.isfinite(scaled_sum)
        or scaled_sum <= 0.0
    ):
        raise InputValidationError(f"{field} cannot be accumulated stably")
    return scaled, scale, scaled_sum


def normalized_positive_weights(
    weights: np.ndarray, *, field: str = "weights"
) -> np.ndarray:
    """Return finite positive weights normalized without summing raw magnitudes."""

    scaled, _, scaled_sum = scale_positive_weights(weights, field=field)
    normalized = scaled / scaled_sum
    if np.any(~np.isfinite(normalized)) or not math.isclose(
        float(np.sum(normalized, dtype=np.float64)),
        1.0,
        rel_tol=64.0 * np.finfo(np.float64).eps * normalized.size,
        abs_tol=64.0 * np.finfo(np.float64).eps * normalized.size,
    ):
        raise InputValidationError(f"{field} cannot be normalized stably")
    return normalized


def positive_weighted_mean(
    values: np.ndarray, weights: np.ndarray, *, field: str = "weights"
) -> float | np.ndarray:
    """Return a scale-invariant weighted mean along the first value axis.

    Both the weights and observations are scaled before accumulation.  This
    keeps a mathematically finite convex mean finite even when either input is
    close to the float64 limit.
    """

    observations = np.asarray(values, dtype=np.float64)
    design_weights = np.asarray(weights, dtype=np.float64)
    if observations.ndim == 0 or observations.shape[0] != design_weights.size:
        raise InputValidationError(
            "values and weights must be aligned along the first axis"
        )
    if np.any(~np.isfinite(observations)):
        raise InputValidationError("weighted-mean values must be finite")
    normalized = normalized_positive_weights(design_weights, field=field)
    value_scale = float(np.max(np.abs(observations))) if observations.size else 0.0
    scaled_values = observations if value_scale == 0.0 else observations / value_scale
    scaled_result = np.tensordot(normalized, scaled_values, axes=(0, 0))
    # A convex combination must remain in the component-wise input envelope;
    # clip only round-off beyond that mathematical bound before restoring scale.
    scaled_result = np.clip(scaled_result, -1.0, 1.0)
    result = scaled_result if value_scale == 0.0 else scaled_result * value_scale
    if np.any(~np.isfinite(result)):
        raise InputValidationError("weighted mean is not finite after stable scaling")
    if np.ndim(result) == 0:
        return float(result)
    return np.asarray(result, dtype=np.float64)


@dataclass
class StreamingPositiveWeightMoments:
    """Mergeable, scale-invariant moments for finite positive weights.

    The state is deliberately expressed in units of the largest weight seen so
    far.  It supports shard-by-shard archive assembly without retaining every
    raw weight and without allowing raw sums to overflow.
    """

    count: int = 0
    scale: float = 0.0
    scaled_sum: float = 0.0
    scaled_square_sum: float = 0.0

    def update(self, weights: np.ndarray, *, field: str = "weights") -> None:
        values = np.asarray(weights, dtype=np.float64)
        if values.ndim != 1:
            raise InputValidationError(f"{field} must be a one-dimensional array")
        if values.size == 0:
            return
        chunk_scaled, chunk_scale, _ = scale_positive_weights(values, field=field)
        if self.count == 0:
            self.scale = chunk_scale
            adjusted = chunk_scaled
        elif chunk_scale > self.scale:
            ratio = self.scale / chunk_scale
            self.scaled_sum *= ratio
            self.scaled_square_sum *= ratio * ratio
            self.scale = chunk_scale
            adjusted = chunk_scaled
        else:
            adjusted = values / self.scale
        self.scaled_sum += float(np.sum(adjusted, dtype=np.float64))
        self.scaled_square_sum += float(
            np.sum(adjusted * adjusted, dtype=np.float64)
        )
        self.count += int(values.size)
        if (
            not math.isfinite(self.scaled_sum)
            or not math.isfinite(self.scaled_square_sum)
            or self.scaled_sum <= 0.0
            or self.scaled_square_sum <= 0.0
        ):
            raise InputValidationError(f"{field} moments cannot be accumulated stably")

    @property
    def effective_sample_size(self) -> float:
        if self.count == 0:
            return 0.0
        result = self.scaled_sum * self.scaled_sum / self.scaled_square_sum
        tolerance = 64.0 * np.finfo(np.float64).eps * self.count
        if (
            not math.isfinite(result)
            or result < 1.0 - tolerance
            or result > self.count + tolerance
        ):
            raise InputValidationError(
                "streaming weight ESS lies outside its mathematical [1, n] bounds"
            )
        return float(np.clip(result, 1.0, float(self.count)))

    @property
    def coefficient_of_variation(self) -> float | None:
        if self.count == 0:
            return None
        mean = self.scaled_sum / self.count
        variance = max(
            0.0, self.scaled_square_sum / self.count - mean * mean
        )
        result = math.sqrt(variance) / mean
        if not math.isfinite(result):
            raise InputValidationError(
                "streaming weight coefficient of variation is not finite"
            )
        return float(result)


def restore_scaled_total(
    scaled_total: float, scale: float
) -> tuple[float | None, str]:
    """Restore a raw weighted total, reporting unrepresentable float64 totals."""

    value = float(scaled_total)
    if (
        not math.isfinite(value)
        or value < 0.0
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise InputValidationError("scaled weight total is invalid")
    if value == 0.0:
        return 0.0, "finite"
    if value > 1.0 and scale > np.finfo(np.float64).max / value:
        return None, "not_representable_float64_use_scale_and_scaled_total"
    restored = float(scale * value)
    if not math.isfinite(restored):
        raise InputValidationError("weight total reconstruction is non-finite")
    return restored, "finite"


def positive_effective_sample_size(weights: np.ndarray) -> float:
    """Return scale-invariant Kish ESS for finite positive weights.

    An empty vector has ESS zero.  This is the strict design-weight primitive;
    callers that require a nonempty sample must reject emptiness before calling.
    """

    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1:
        raise InputValidationError("ESS weights must be a one-dimensional array")
    if values.size == 0:
        return 0.0
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise InputValidationError("ESS weights must be finite and positive")
    scaled, _, scaled_sum = scale_positive_weights(values, field="ESS weights")
    denominator = float(np.sum(scaled * scaled, dtype=np.float64))
    result = scaled_sum * scaled_sum / denominator
    if not math.isfinite(result):
        raise InputValidationError("ESS is not finite after stable weight scaling")
    tolerance = 64.0 * np.finfo(np.float64).eps * values.size
    if result < 1.0 - tolerance or result > values.size + tolerance:
        raise InputValidationError("ESS lies outside its mathematical [1, n] bounds")
    return float(np.clip(result, 1.0, float(values.size)))


def nonnegative_effective_sample_size(contributions: np.ndarray) -> float:
    """Return Kish ESS over positive mass in a nonnegative contribution vector.

    Zeros mean that a row contributes no mass to this particular estimand; they
    are not invalid design weights.  All-zero input is therefore explicitly
    ``not estimable`` and represented by zero ESS.
    """

    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values) | (values < 0.0)):
        raise InputValidationError(
            "ESS contributions must be a finite nonnegative vector"
        )
    positive = values[values > 0.0]
    return positive_effective_sample_size(positive)


def validate_design_weight_contract(
    weights: np.ndarray,
    weight_mask: np.ndarray,
    selection_probability: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate aligned raw design weights and their estimand eligibility mask."""

    raw_mask = np.asarray(weight_mask)
    if raw_mask.dtype != np.dtype(bool) or raw_mask.ndim != 1:
        raise InputValidationError(
            "sampling_weight_mask must be a one-dimensional bool array"
        )
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.shape != raw_mask.shape:
        raise InputValidationError(
            "design weights and sampling_weight_mask must share one axis"
        )
    invalid_weights = raw_mask & (~np.isfinite(values) | (values <= 0.0))
    if np.any(invalid_weights):
        raise InputValidationError(
            "eligible design weights must be finite and positive",
            context={"violation_count": int(np.count_nonzero(invalid_weights))},
        )
    probabilities = None
    if selection_probability is not None:
        probabilities = np.asarray(selection_probability, dtype=np.float64)
        if probabilities.ndim != 1 or probabilities.shape != values.shape:
            raise InputValidationError(
                "selection probabilities must share the design-weight axis"
            )
        invalid_probability = raw_mask & (
            ~np.isfinite(probabilities)
            | (probabilities <= 0.0)
            | (probabilities > 1.0)
        )
        if np.any(invalid_probability):
            raise InputValidationError(
                "eligible selection probabilities must lie in (0, 1]",
                context={
                    "violation_count": int(np.count_nonzero(invalid_probability))
                },
            )
    return values, raw_mask, probabilities
