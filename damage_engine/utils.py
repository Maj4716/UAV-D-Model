from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .errors import InputValidationError, OutOfDomainError


EPS = 1.0e-12


# Two-sided 95% Student-t critical values for 1..30 degrees of freedom.
# Small Monte Carlo batches otherwise receive materially over-confident
# intervals when the asymptotic normal value (1.96) is used.
_T95 = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def source_tree_hash(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    source_files = sorted(
        file_path
        for file_path in root.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in {".py", ".json"}
        and "__pycache__" not in file_path.parts
    )
    for file_path in source_files:
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = file_path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return "sha256:" + digest.hexdigest()


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def normalize(vector: np.ndarray, *, name: str = "vector") -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise InputValidationError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(value))
    if norm <= EPS:
        raise InputValidationError(f"{name} must be non-zero")
    return value / norm


def quaternion_matrix_xyzw(quaternion: list[float] | np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise InputValidationError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.0) > 1.0e-6:
        raise InputValidationError(
            "quaternion must have unit norm",
            context={"norm": norm},
        )
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(np.clip(value, -60.0, 60.0)))))


def loglog_interpolate(
    x: float,
    x_points: np.ndarray,
    y_points: np.ndarray,
    *,
    quantity: str,
) -> float:
    if x <= 0:
        raise OutOfDomainError(f"{quantity} abscissa must be positive")
    xp = np.asarray(x_points, dtype=np.float64)
    yp = np.asarray(y_points, dtype=np.float64)
    if xp.ndim != 1 or yp.shape != xp.shape or xp.size < 2:
        raise InputValidationError(f"{quantity} table must contain at least two points")
    if np.any(xp <= 0) or np.any(yp <= 0) or np.any(np.diff(xp) <= 0):
        raise InputValidationError(
            f"{quantity} table must be positive and strictly increasing in x"
        )
    if x < xp[0] or x > xp[-1]:
        raise OutOfDomainError(
            f"{quantity} requested outside calibrated table",
            context={"x": x, "minimum": float(xp[0]), "maximum": float(xp[-1])},
        )
    return float(np.exp(np.interp(np.log(x), np.log(xp), np.log(yp))))


def parameter_value(parameter: dict, rng: np.random.Generator | None = None) -> float:
    value = float(parameter["value"])
    uncertainty = parameter.get("uncertainty") or {"kind": "none"}
    kind = uncertainty.get("kind", "none")
    if rng is None or kind == "none":
        sampled = value
    else:
        params = uncertainty.get("parameters") or {}
        if kind == "normal":
            sampled = rng.normal(value, float(params["stddev"]))
        elif kind == "lognormal":
            sigma = float(params["log_sigma"])
            sampled = rng.lognormal(math.log(value), sigma)
        elif kind == "uniform":
            sampled = rng.uniform(float(params["minimum"]), float(params["maximum"]))
        elif kind == "interval":
            raise InputValidationError(
                "interval specifies bounds only and cannot be sampled as a probability distribution; "
                "declare uniform explicitly if that assumption is justified",
                context={"kind": kind, "uncertainty_semantics_version": "2.0"},
            )
        elif kind == "empirical":
            values = np.asarray(params["values"], dtype=np.float64)
            probabilities = params.get("probabilities")
            sampled = rng.choice(values, p=probabilities)
        else:
            raise InputValidationError(f"unsupported uncertainty kind: {kind}")
    sampled = float(sampled)
    valid_range = parameter.get("valid_range")
    if valid_range and not (float(valid_range[0]) <= sampled <= float(valid_range[1])):
        raise OutOfDomainError(
            "sampled parameter is outside its validity range",
            context={"sample": sampled, "valid_range": valid_range},
        )
    return sampled


def length_to_m(parameter: dict, rng: np.random.Generator | None = None) -> float:
    value = parameter_value(parameter, rng)
    unit = parameter["unit"]
    if unit == "m":
        return value
    if unit == "mm":
        return value / 1000.0
    if unit == "cm":
        return value / 100.0
    raise InputValidationError(f"unsupported length unit: {unit}")


def pressure_to_pa(parameter: dict, rng: np.random.Generator | None = None) -> float:
    value = parameter_value(parameter, rng)
    unit = parameter["unit"]
    if unit == "Pa":
        return value
    if unit == "kPa":
        return value * 1.0e3
    if unit == "MPa":
        return value * 1.0e6
    raise InputValidationError(f"unsupported pressure unit: {unit}")


def impulse_to_pa_s(parameter: dict, rng: np.random.Generator | None = None) -> float:
    value = parameter_value(parameter, rng)
    unit = parameter["unit"]
    factors = {
        "Pa*s": 1.0,
        "kPa*ms": 1.0,
        "kPa*s": 1.0e3,
        "MPa*ms": 1.0e3,
    }
    try:
        return value * factors[unit]
    except KeyError as exc:
        raise InputValidationError(f"unsupported impulse unit: {unit}") from exc


def duration_to_s(parameter: dict, rng: np.random.Generator | None = None) -> float:
    value = parameter_value(parameter, rng)
    unit = parameter["unit"]
    if unit == "s":
        return value
    if unit == "ms":
        return value * 1.0e-3
    raise InputValidationError(f"unsupported duration unit: {unit}")


def ratio_value(parameter: dict, rng: np.random.Generator | None = None) -> float:
    value = parameter_value(parameter, rng)
    unit = parameter["unit"]
    if unit == "%":
        value /= 100.0
    elif unit != "1":
        raise InputValidationError(f"unsupported ratio unit: {unit}")
    if not 0.0 <= value <= 1.0:
        raise InputValidationError("ratio parameter must lie in [0,1]")
    return value


def confidence_interval(
    values: np.ndarray,
    *,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, float, float, float]:
    """Approximate fixed-sample mean interval, conditional on the observed series.

    A single observation has a mean but no estimable sample variance or mean
    interval; those three tuple positions are NaN. Use statistics_dict for
    JSON-safe nulls and explicit estimation status. No optional-stopping or
    simultaneous-coverage guarantee is implied.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return (math.nan, math.nan, math.nan, math.nan)
    if not np.isfinite(array).all():
        raise InputValidationError("confidence interval values must be finite")
    mean = float(np.mean(array))
    if array.size == 1:
        return (mean, math.nan, math.nan, math.nan)
    std = float(np.std(array, ddof=1))
    degrees_of_freedom = array.size - 1
    critical = _T95[degrees_of_freedom - 1] if degrees_of_freedom <= 30 else 1.96
    half = critical * std / math.sqrt(array.size)
    lower = mean - half
    upper = mean + half
    if bounds is not None:
        lower = max(float(bounds[0]), lower)
        upper = min(float(bounds[1]), upper)
    return (mean, std, lower, upper)


def statistics_dict(
    values: np.ndarray | list[float],
    *,
    probability: bool = False,
) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean, std, lower, upper = confidence_interval(
        array,
        bounds=(0.0, 1.0) if probability else None,
    )
    estimable = array.size >= 2
    return {
        "mean": mean if array.size else None,
        "stddev": std if estimable else None,
        "ci95_lower": lower if estimable else None,
        "ci95_upper": upper if estimable else None,
        "sample_count": int(array.size),
        "uncertainty_estimable": estimable,
        "insufficient_reason": None if estimable else "fewer_than_two_observations",
        "estimand": "conditional_replicate_mean",
        "stddev_estimand": "observed_replicate_sample_dispersion",
        "conditioning": "fixed_case_and_epistemic_realization_with_possible_within_replicate_parameter_variation",
        "random_level": "within_case_replicates",
        "interval_type": "approximate_pointwise_mean_ci",
        "interval_method": "student_t_df_le_30_else_normal_1.96",
        "probability_bounds_clipped": bool(probability),
        "coverage_status": "fixed_sample_approximation_not_optional_stopping_validated",
    }
