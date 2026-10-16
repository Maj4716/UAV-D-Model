"""Fail-closed numerical primitives shared by descriptive audit tools.

These helpers validate the estimand inputs before computing a diagnostic.  They
do not grant research admission and intentionally return explicit
``not_estimable`` states instead of encoding an undefined statistic as zero.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .errors import InputValidationError
from .estimands import (
    positive_effective_sample_size as stable_effective_sample_size,
    validate_design_weight_contract,
)


def stable_finite_summary_statistics(
    values: np.ndarray,
    quantile_probabilities: np.ndarray,
) -> dict[str, Any]:
    """Compute finite float64 location/scale statistics without sum overflow."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(quantile_probabilities, dtype=np.float64).reshape(-1)
    if array.size == 0 or np.any(~np.isfinite(array)):
        raise InputValidationError("summary values must be a non-empty finite array")
    if (
        probabilities.size == 0
        or np.any(~np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise InputValidationError("quantile probabilities must be finite values in [0, 1]")

    scale = float(np.max(np.abs(array)))
    scaled = array if scale == 0.0 else array / scale
    scaled_mean = float(np.mean(scaled))
    centered = scaled - scaled_mean
    scaled_std = min(math.sqrt(float(np.mean(centered * centered))), 1.0)
    scaled_quantiles = np.quantile(scaled, probabilities)
    mean = scaled_mean if scale == 0.0 else scaled_mean * scale
    standard_deviation = scaled_std if scale == 0.0 else scaled_std * scale
    quantiles = scaled_quantiles if scale == 0.0 else scaled_quantiles * scale
    if (
        not math.isfinite(mean)
        or not math.isfinite(standard_deviation)
        or np.any(~np.isfinite(quantiles))
    ):
        raise InputValidationError("scaled summary produced a non-finite result")
    coefficient = None
    coefficient_status = "not_estimable_zero_mean"
    if scaled_mean != 0.0:
        candidate = abs(scaled_std / scaled_mean)
        if math.isfinite(candidate):
            coefficient = candidate
            coefficient_status = "estimated"
        else:
            coefficient_status = "not_estimable_nonfinite_ratio"
    return {
        "mean": float(mean),
        "standard_deviation": float(standard_deviation),
        "quantiles": np.asarray(quantiles, dtype=np.float64),
        "coefficient_of_variation": coefficient,
        "coefficient_of_variation_status": coefficient_status,
        "numeric_method": "max_abs_scaled_float64",
        "numeric_scale": scale,
    }


def standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Describe a two-group mean contrast without hiding a zero denominator."""

    left_raw = np.asarray(left, dtype=np.float64).reshape(-1)
    right_raw = np.asarray(right, dtype=np.float64).reshape(-1)
    left_values = left_raw[np.isfinite(left_raw)]
    right_values = right_raw[np.isfinite(right_raw)]
    result: dict[str, Any] = {
        "left_total_count": int(left_raw.size),
        "right_total_count": int(right_raw.size),
        "left_finite_count": int(left_values.size),
        "right_finite_count": int(right_values.size),
        "left_nonfinite_count": int(left_raw.size - left_values.size),
        "right_nonfinite_count": int(right_raw.size - right_values.size),
        "left_missing_count": int(np.count_nonzero(np.isnan(left_raw))),
        "right_missing_count": int(np.count_nonzero(np.isnan(right_raw))),
        "left_invalid_count": int(np.count_nonzero(np.isinf(left_raw))),
        "right_invalid_count": int(np.count_nonzero(np.isinf(right_raw))),
        "raw_mean_difference": None,
        "raw_mean_difference_status": "not_computed",
        "pooled_standard_deviation": None,
        "pooled_standard_deviation_status": "not_computed",
        "absolute_standardized_mean_difference": None,
    }
    if not left_values.size or not right_values.size:
        result["status"] = "not_estimable_empty_group"
        result["pooled_standard_deviation_status"] = "not_estimable_empty_group"
        return result
    common_scale = float(
        max(np.max(np.abs(left_values)), np.max(np.abs(right_values)))
    )
    left_scaled = left_values if common_scale == 0.0 else left_values / common_scale
    right_scaled = right_values if common_scale == 0.0 else right_values / common_scale
    left_mean = float(np.mean(left_scaled))
    right_mean = float(np.mean(right_scaled))
    difference_scaled = left_mean - right_mean
    left_centered = left_scaled - left_mean
    right_centered = right_scaled - right_mean
    pooled_scaled = math.sqrt(
        (
            float(np.mean(left_centered * left_centered))
            + float(np.mean(right_centered * right_centered))
        )
        / 2.0
    )
    difference = difference_scaled if common_scale == 0.0 else difference_scaled * common_scale
    pooled = pooled_scaled if common_scale == 0.0 else pooled_scaled * common_scale
    result["numeric_method"] = "common_max_abs_scaled_float64"
    result["numeric_scale"] = common_scale
    result["raw_mean_difference_scaled"] = difference_scaled
    if math.isfinite(difference):
        result["raw_mean_difference"] = difference
        result["raw_mean_difference_status"] = "finite"
    else:
        result["raw_mean_difference_status"] = "nonfinite_unrepresentable"
    if not math.isfinite(pooled) or not math.isfinite(pooled_scaled):
        result["status"] = "not_estimable_nonfinite_denominator"
        result["pooled_standard_deviation_status"] = "nonfinite"
        return result
    result["pooled_standard_deviation"] = pooled
    result["pooled_standard_deviation_status"] = "finite_positive" if pooled > 0.0 else "finite_zero"
    if pooled_scaled > 0.0:
        result["status"] = "estimated"
        result["absolute_standardized_mean_difference"] = abs(difference_scaled) / pooled_scaled
    elif difference_scaled == 0.0:
        result["status"] = "same_constant"
        result["absolute_standardized_mean_difference"] = 0.0
    else:
        result["status"] = "not_estimable_zero_scale_different_constants"
    return result
