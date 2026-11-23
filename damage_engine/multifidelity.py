"""Auditable control-variate utilities for proxy/teacher experiments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .errors import InputValidationError
from .utils import content_hash


def _paired(proxy: Any, teacher: Any) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(proxy, dtype=np.float64)
    y = np.asarray(teacher, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise InputValidationError("proxy and teacher arrays must be aligned finite matrices")
    return x, y


def fit_control_variate(proxy: Any, teacher: Any) -> dict[str, Any]:
    """Fit one scalar control coefficient per output on paired pilot cases."""

    x, y = _paired(proxy, teacher)
    xc = x - np.mean(x, axis=0)
    yc = y - np.mean(y, axis=0)
    var_x = np.sum(xc * xc, axis=0) / (x.shape[0] - 1)
    var_y = np.sum(yc * yc, axis=0) / (x.shape[0] - 1)
    covariance = np.sum(xc * yc, axis=0) / (x.shape[0] - 1)
    beta = np.divide(covariance, var_x, out=np.zeros_like(covariance), where=var_x > 0.0)
    correlation = np.divide(
        covariance,
        np.sqrt(var_x * var_y),
        out=np.zeros_like(covariance),
        where=(var_x > 0.0) & (var_y > 0.0),
    )
    residual = y - beta[None, :] * x
    declaration = {
        "schema_version": "control-variate-fit@1.0.0",
        "paired_count": int(x.shape[0]),
        "output_dimension": int(x.shape[1]),
        "beta": beta.tolist(),
        "proxy_teacher_correlation": correlation.tolist(),
        "ideal_variance_reduction_fraction": np.clip(correlation**2, 0.0, 1.0).tolist(),
        "residual_sample_variance": np.var(residual, axis=0, ddof=1).tolist(),
        "scope": "software_teacher_estimation_not_external_physical_validation",
    }
    return {**declaration, "fit_hash": content_hash(declaration)}


def control_variate_mean(
    proxy_paired: Any,
    teacher_paired: Any,
    *,
    proxy_reference_mean: Any,
    fit: dict[str, Any],
) -> dict[str, Any]:
    """Estimate the teacher mean when a large proxy-reference mean is known."""

    x, y = _paired(proxy_paired, teacher_paired)
    if fit.get("schema_version") != "control-variate-fit@1.0.0":
        raise InputValidationError("unsupported control-variate fit contract")
    expected_hash = content_hash({key: value for key, value in fit.items() if key != "fit_hash"})
    if fit.get("fit_hash") != expected_hash:
        raise InputValidationError("control-variate fit hash mismatch")
    beta = np.asarray(fit["beta"], dtype=np.float64)
    proxy_mean = np.asarray(proxy_reference_mean, dtype=np.float64)
    if beta.shape != (x.shape[1],) or proxy_mean.shape != beta.shape or not np.isfinite(proxy_mean).all():
        raise InputValidationError("control-variate dimensions do not align")
    adjusted = y + beta[None, :] * (proxy_mean[None, :] - x)
    mean = np.mean(adjusted, axis=0)
    variance = np.var(adjusted, axis=0, ddof=1)
    se = np.sqrt(variance / adjusted.shape[0])
    return {
        "schema_version": "control-variate-estimate@1.0.0",
        "fit_hash": fit["fit_hash"],
        "estimate": mean.tolist(),
        "standard_error": se.tolist(),
        "ci95": np.column_stack([mean - 1.96 * se, mean + 1.96 * se]).tolist(),
        "paired_count": int(adjusted.shape[0]),
        "proxy_reference_mean_treated_as_fixed": True,
    }


def paired_efficiency_report(proxy: Any, teacher: Any) -> dict[str, Any]:
    x, y = _paired(proxy, teacher)
    fit = fit_control_variate(x, y)
    beta = np.asarray(fit["beta"])
    adjusted = y - beta[None, :] * (x - np.mean(x, axis=0))
    raw_variance = np.var(y, axis=0, ddof=1)
    adjusted_variance = np.var(adjusted, axis=0, ddof=1)
    reduction = np.divide(
        raw_variance - adjusted_variance,
        raw_variance,
        out=np.zeros_like(raw_variance),
        where=raw_variance > 0.0,
    )
    return {
        "fit": fit,
        "observed_variance_reduction_fraction": reduction.tolist(),
        "teacher_sample_equivalent_multiplier": [
            None if item >= 1.0 else float(1.0 / max(1.0e-12, 1.0 - item))
            for item in reduction
        ],
    }


__all__ = [
    "control_variate_mean",
    "fit_control_variate",
    "paired_efficiency_report",
]
