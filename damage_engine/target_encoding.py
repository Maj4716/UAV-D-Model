"""Train-only, reversible target encoders for dataset-v2 learning views.

Encoders change only the numerical representation presented to a learner.  The
archived physical labels, masks, estimands, and evaluation scale remain the
source of truth.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import InputValidationError
from .utils import content_hash


TARGET_ENCODER_SCHEMA_VERSION = "1.0.0"
_NORMAL = NormalDist()


def _training_values(values: Any, mask: Any | None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    available = np.ones(array.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if available.shape != array.shape:
        raise InputValidationError("target encoder mask shape must match values")
    selected = array[available]
    if selected.size < 1 or not np.isfinite(selected).all():
        raise InputValidationError(
            "target encoder requires at least one finite Train observation"
        )
    return selected


def _box_cox(values: np.ndarray, exponent: float, shift: float) -> np.ndarray:
    positive = values + shift
    if np.any(positive <= 0.0):
        raise InputValidationError("Box-Cox transform received a nonpositive value")
    if abs(exponent) <= 1.0e-10:
        return np.log(positive)
    return np.expm1(exponent * np.log(positive)) / exponent


def _fit_box_cox(values: np.ndarray) -> tuple[float, float]:
    scale = max(1.0, float(np.ptp(values)), abs(float(np.mean(values))))
    shift = max(0.0, -float(np.min(values))) + 1.0e-9 * scale
    positive = values + shift
    log_values = np.log(positive)
    best_exponent = 0.0
    best_score = -math.inf
    for exponent in np.linspace(-2.0, 2.0, 161):
        transformed = _box_cox(values, float(exponent), shift)
        variance = float(np.var(transformed))
        if variance <= 0.0:
            score = 0.0 if np.all(values == values[0]) else -math.inf
        else:
            score = (
                -0.5 * values.size * math.log(variance)
                + (float(exponent) - 1.0) * float(np.sum(log_values))
            )
        if score > best_score:
            best_score = score
            best_exponent = float(exponent)
    return best_exponent, shift


def fit_target_encoder(
    values: Any,
    *,
    mask: Any | None = None,
    method: str,
    task_name: str,
    source_split: str = "train",
    quantile_knots: int = 129,
) -> dict[str, Any]:
    """Fit one encoder from Train only and return a serializable contract."""

    if source_split not in {"train", "reference_train", "train_pilot"}:
        raise InputValidationError("target encoders may be fitted from Train only")
    if not isinstance(task_name, str) or not task_name:
        raise InputValidationError("target encoder task_name must be nonempty")
    selected = _training_values(values, mask)
    method = str(method)
    parameters: dict[str, Any]
    if method == "identity":
        parameters = {}
    elif method == "log1p":
        if np.any(selected < 0.0):
            raise InputValidationError("log1p target encoder requires nonnegative labels")
        parameters = {}
    elif method == "box_cox":
        exponent, shift = _fit_box_cox(selected)
        transformed = _box_cox(selected, exponent, shift)
        parameters = {
            "exponent": exponent,
            "shift": shift,
            "training_value_min": float(np.min(selected)),
            "training_value_max": float(np.max(selected)),
            "transformed_training_min": float(np.min(transformed)),
            "transformed_training_max": float(np.max(transformed)),
            "prediction_projection": "clip_to_train_encoded_range_before_inverse",
        }
    elif method == "empirical_quantile_normal":
        if quantile_knots < 3:
            raise InputValidationError("quantile target encoder requires at least three knots")
        knot_count = min(int(quantile_knots), selected.size)
        quantile_locations = np.linspace(0.0, 1.0, knot_count)
        knots = np.quantile(selected, quantile_locations)
        probabilities = (np.arange(knot_count, dtype=np.float64) + 0.5) / knot_count
        unique, first = np.unique(knots, return_index=True)
        probabilities = probabilities[first]
        if unique.size == 1:
            probabilities = np.asarray([0.5], dtype=np.float64)
        tail_clip = float(
            min(probabilities[0], 1.0 - probabilities[-1])
        )
        parameters = {
            "value_knots": [float(item) for item in unique],
            "probability_knots": [float(item) for item in probabilities],
            "tail_probability_clip": float(max(1.0e-6, tail_clip)),
            "interpolation": "piecewise_linear_with_clipped_tails",
        }
    else:
        raise InputValidationError(
            "unsupported target encoder method",
            context={"method": method},
        )
    contract = {
        "schema_version": TARGET_ENCODER_SCHEMA_VERSION,
        "task_name": task_name,
        "method": method,
        "fit_scope": "train_only",
        "source_split": source_split,
        "training_observation_count": int(selected.size),
        "parameters": parameters,
        "evaluation_scale": "original_physical_label_scale",
        "label_mutation": False,
    }
    return {**contract, "encoder_hash": content_hash(contract)}


def _validated_contract(contract: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(contract, Mapping):
        raise InputValidationError("target encoder contract must be an object")
    if contract.get("schema_version") != TARGET_ENCODER_SCHEMA_VERSION:
        raise InputValidationError("unsupported target encoder schema version")
    expected = content_hash({key: value for key, value in contract.items() if key != "encoder_hash"})
    if contract.get("encoder_hash") != expected:
        raise InputValidationError("target encoder hash does not match its content")
    parameters = contract.get("parameters")
    if not isinstance(parameters, Mapping):
        raise InputValidationError("target encoder parameters must be an object")
    return str(contract.get("method")), parameters


def transform_target(values: Any, contract: Mapping[str, Any]) -> np.ndarray:
    method, parameters = _validated_contract(contract)
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise InputValidationError("target transform requires finite values")
    if method == "identity":
        return array.copy()
    if method == "log1p":
        if np.any(array < 0.0):
            raise InputValidationError("log1p target transform requires nonnegative values")
        return np.log1p(array)
    if method == "box_cox":
        return _box_cox(array, float(parameters["exponent"]), float(parameters["shift"]))
    if method == "empirical_quantile_normal":
        x = np.asarray(parameters["value_knots"], dtype=np.float64)
        p = np.asarray(parameters["probability_knots"], dtype=np.float64)
        clip = float(parameters["tail_probability_clip"])
        probabilities = np.interp(array, x, p)
        probabilities = np.clip(probabilities, clip, 1.0 - clip)
        return np.asarray([_NORMAL.inv_cdf(float(item)) for item in probabilities.flat]).reshape(array.shape)
    raise InputValidationError("unsupported target encoder method")


def inverse_target(values: Any, contract: Mapping[str, Any]) -> np.ndarray:
    method, parameters = _validated_contract(contract)
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise InputValidationError("target inverse transform requires finite values")
    if method == "identity":
        return array.copy()
    if method == "log1p":
        return np.expm1(array)
    if method == "box_cox":
        exponent = float(parameters["exponent"])
        shift = float(parameters["shift"])
        if abs(exponent) <= 1.0e-10:
            return np.exp(array) - shift
        base = 1.0 + exponent * array
        if np.any(base <= 0.0):
            raise InputValidationError("Box-Cox inverse is outside its numeric domain")
        return np.exp(np.log(base) / exponent) - shift
    if method == "empirical_quantile_normal":
        x = np.asarray(parameters["value_knots"], dtype=np.float64)
        p = np.asarray(parameters["probability_knots"], dtype=np.float64)
        probabilities = np.asarray([_NORMAL.cdf(float(item)) for item in array.flat]).reshape(array.shape)
        return np.interp(probabilities, p, x)
    raise InputValidationError("unsupported target encoder method")


def project_encoded_prediction(
    values: Any, contract: Mapping[str, Any]
) -> np.ndarray:
    """Project learned encoded predictions to a registered inverse domain.

    A Box-Cox regression head is unconstrained, while its inverse has a finite
    numeric domain.  The fitted training range is an outcome-blind, Train-only
    projection contract that keeps evaluation finite and makes extrapolation
    behavior explicit.  Other encoders already have total inverse maps.
    """

    method, parameters = _validated_contract(contract)
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise InputValidationError(
            "encoded target prediction projection requires finite values"
        )
    if method != "box_cox":
        return array.copy()
    if (
        "transformed_training_min" in parameters
        and "transformed_training_max" in parameters
    ):
        lower = float(parameters["transformed_training_min"])
        upper = float(parameters["transformed_training_max"])
    else:
        exponent = float(parameters["exponent"])
        lower, upper = -math.inf, math.inf
        if exponent > 1.0e-10:
            lower = math.nextafter(-1.0 / exponent, math.inf)
        elif exponent < -1.0e-10:
            upper = math.nextafter(-1.0 / exponent, -math.inf)
    # A train split can contain a single distinct target value.  Its fitted
    # projection domain is then the one-point interval [lower, lower], which
    # is valid and prevents validation/test values from influencing the map.
    if lower > upper:
        raise InputValidationError("Box-Cox prediction projection range is invalid")
    return np.clip(array, lower, upper)


def default_hurdle_target_encoder_bundle(
    *,
    conditional_energy: Sequence[float],
    conditional_hits: Sequence[float],
    energy_mask: Sequence[bool] | None = None,
    hit_mask: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Fit the Section-9 preferred encoders while keeping probabilities raw."""

    encoders = {
        "p_hit": fit_target_encoder(
            [0.0, 1.0], method="identity", task_name="p_hit"
        ),
        "conditional_log1p_deposited_energy_j": fit_target_encoder(
            conditional_energy,
            mask=energy_mask,
            method="empirical_quantile_normal",
            task_name="conditional_log1p_deposited_energy_j",
        ),
        "conditional_log1p_hit_count": fit_target_encoder(
            conditional_hits,
            mask=hit_mask,
            method="box_cox",
            task_name="conditional_log1p_hit_count",
        ),
    }
    document = {
        "schema_version": "hurdle-target-encoder-bundle@1.0.0",
        "fit_scope": "train_only",
        "evaluation_scale": "original_physical_label_scale",
        "encoders": encoders,
    }
    return {**document, "bundle_hash": content_hash(document)}


__all__ = [
    "TARGET_ENCODER_SCHEMA_VERSION",
    "default_hurdle_target_encoder_bundle",
    "fit_target_encoder",
    "inverse_target",
    "project_encoded_prediction",
    "transform_target",
]
