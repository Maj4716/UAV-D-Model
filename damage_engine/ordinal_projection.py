"""Opt-in ordinal projection for provisional synthetic-teacher experiments.

This module is deliberately independent of :mod:`damage_engine.damage_tree`.
It does not redefine any legacy node and it refuses configurations that claim
real-world KMFC approval.  Its narrow purpose is to make the proposed
``continuous severity -> per-replicate hard ordinal gate`` calculation
deterministic, auditable, and fail-closed while the domain codebook and
physical calibration remain unresolved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import NormalDist
from typing import Any

import numpy as np

from .content_contract import is_sha256_identity
from .errors import InputValidationError
from .utils import content_hash


ORDINAL_PROJECTION_SCHEMA_VERSION = "1.0.0"
_PROVISIONAL_STATUS = "provisional"
_SOFTWARE_SCOPE = "software_only"
_CLAIM_SCOPE = "project_specific_simulator_emulation"
_PHYSICAL_STATUS = "not_established"
_THRESHOLD_STATUS = "not_approved_for_real_world_kmfc"
_ALLOWED_AGGREGATE_POWERS = frozenset({1.0, 2.0, 4.0})

_REQUIRED_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "projection_id",
        "category_id",
        "status",
        "execution_scope",
        "scientific_claim_scope",
        "physical_validity_status",
        "threshold_approval_status",
        "threshold_source_id",
        "input_names",
        "evidence_scales",
        "evidence_exponents",
        "maximum_slopes",
        "numeric_tolerance",
        "aggregate_weights",
        "aggregate_power",
        "tau1",
        "tau2",
        "minimum_threshold_gap",
    }
)
_OPTIONAL_CONFIG_FIELDS = frozenset({"calibration_artifact_hash"})


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise InputValidationError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(f"{name} must be a finite number") from exc
    if not isfinite(number):
        raise InputValidationError(f"{name} must be a finite number")
    return number


def _positive_vector(value: Any, *, name: str, size: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InputValidationError(f"{name} must contain one positive value per input")
    numbers = tuple(_finite_number(item, name=f"{name}[{index}]") for index, item in enumerate(value))
    if len(numbers) != size or any(item <= 0.0 for item in numbers):
        raise InputValidationError(f"{name} must contain one positive value per input")
    return numbers


def _require_exact(value: Any, expected: str, *, name: str) -> str:
    if value != expected:
        raise InputValidationError(
            f"{name} must be {expected!r} for this software-only ordinal projection"
        )
    return expected


@dataclass(frozen=True)
class OrdinalProjectionConfig:
    """Compiled software-only configuration with no physical-admission claim."""

    projection_id: str
    category_id: str
    threshold_source_id: str
    input_names: tuple[str, ...]
    evidence_scales: tuple[float, ...]
    evidence_exponents: tuple[float, ...]
    maximum_slopes: tuple[float, ...]
    numeric_tolerance: float
    aggregate_weights: tuple[float, ...]
    aggregate_power: float
    tau1: float
    tau2: float
    minimum_threshold_gap: float
    calibration_artifact_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ORDINAL_PROJECTION_SCHEMA_VERSION,
            "projection_id": self.projection_id,
            "category_id": self.category_id,
            "status": _PROVISIONAL_STATUS,
            "execution_scope": _SOFTWARE_SCOPE,
            "scientific_claim_scope": _CLAIM_SCOPE,
            "physical_validity_status": _PHYSICAL_STATUS,
            "threshold_approval_status": _THRESHOLD_STATUS,
            "threshold_source_id": self.threshold_source_id,
            "input_names": list(self.input_names),
            "evidence_scales": list(self.evidence_scales),
            "evidence_exponents": list(self.evidence_exponents),
            "maximum_slopes": list(self.maximum_slopes),
            "numeric_tolerance": self.numeric_tolerance,
            "aggregate_weights": list(self.aggregate_weights),
            "aggregate_power": self.aggregate_power,
            "tau1": self.tau1,
            "tau2": self.tau2,
            "minimum_threshold_gap": self.minimum_threshold_gap,
            "calibration_artifact_hash": self.calibration_artifact_hash,
        }

    @property
    def hash(self) -> str:
        return content_hash(self.to_dict())


def compile_ordinal_projection(config: Mapping[str, Any]) -> OrdinalProjectionConfig:
    """Validate and freeze a provisional ordinal-projection configuration.

    The accepted status fields are intentionally exact.  A configuration that
    claims approved physical thresholds must go through a future reviewed
    contract rather than silently widening this software-only interface.
    """

    if not isinstance(config, Mapping):
        raise InputValidationError("ordinal projection configuration must be an object")
    fields = set(config)
    missing = sorted(_REQUIRED_CONFIG_FIELDS - fields)
    unknown = sorted(fields - _REQUIRED_CONFIG_FIELDS - _OPTIONAL_CONFIG_FIELDS)
    if missing or unknown:
        raise InputValidationError(
            "ordinal projection configuration fields are incomplete or unknown",
            context={"missing": missing, "unknown": unknown},
        )
    if config["schema_version"] != ORDINAL_PROJECTION_SCHEMA_VERSION:
        raise InputValidationError("unsupported ordinal projection schema version")
    _require_exact(config["status"], _PROVISIONAL_STATUS, name="status")
    _require_exact(config["execution_scope"], _SOFTWARE_SCOPE, name="execution_scope")
    _require_exact(
        config["scientific_claim_scope"], _CLAIM_SCOPE, name="scientific_claim_scope"
    )
    _require_exact(
        config["physical_validity_status"], _PHYSICAL_STATUS, name="physical_validity_status"
    )
    _require_exact(
        config["threshold_approval_status"],
        _THRESHOLD_STATUS,
        name="threshold_approval_status",
    )

    strings: dict[str, str] = {}
    for name in ("projection_id", "category_id", "threshold_source_id"):
        value = config[name]
        if not isinstance(value, str) or not value.strip():
            raise InputValidationError(f"{name} must be a nonempty string")
        strings[name] = value

    names_value = config["input_names"]
    if isinstance(names_value, (str, bytes)) or not isinstance(names_value, Sequence):
        raise InputValidationError("input_names must be a nonempty sequence")
    input_names = tuple(names_value)
    if (
        not input_names
        or any(not isinstance(item, str) or not item for item in input_names)
        or len(set(input_names)) != len(input_names)
    ):
        raise InputValidationError("input_names must contain unique nonempty strings")

    size = len(input_names)
    scales = _positive_vector(config["evidence_scales"], name="evidence_scales", size=size)
    exponents = _positive_vector(
        config["evidence_exponents"], name="evidence_exponents", size=size
    )
    if any(item != 1.0 for item in exponents):
        raise InputValidationError(
            "KMFC-v0 evidence_exponents must be exactly 1 to keep the zero-anchor slope finite and frozen"
        )
    maximum_slopes = _positive_vector(
        config["maximum_slopes"], name="maximum_slopes", size=size
    )
    numeric_tolerance = _finite_number(
        config["numeric_tolerance"], name="numeric_tolerance"
    )
    if not 0.0 <= numeric_tolerance <= 1.0e-6:
        raise InputValidationError("numeric_tolerance must lie in [0, 1e-6]")
    analytic_slopes = tuple(1.0 / item for item in scales)
    if any(not isfinite(item) for item in analytic_slopes):
        raise InputValidationError("evidence scales imply a non-finite maximum slope")
    slope_violations = [
        index
        for index, (observed, declared) in enumerate(
            zip(analytic_slopes, maximum_slopes)
        )
        if observed / declared > 1.0 + numeric_tolerance
    ]
    if slope_violations:
        raise InputValidationError(
            "analytic Hill slope exceeds the declared maximum_slope",
            context={"input_indices": slope_violations},
        )
    weights = _positive_vector(
        config["aggregate_weights"], name="aggregate_weights", size=size
    )
    power = _finite_number(config["aggregate_power"], name="aggregate_power")
    if power not in _ALLOWED_AGGREGATE_POWERS:
        raise InputValidationError(
            "aggregate_power must be one of the preregistered software candidates 1, 2, or 4"
        )
    tau1 = _finite_number(config["tau1"], name="tau1")
    tau2 = _finite_number(config["tau2"], name="tau2")
    minimum_gap = _finite_number(
        config["minimum_threshold_gap"], name="minimum_threshold_gap"
    )
    if not 0.0 < tau1 < tau2 <= 1.0:
        raise InputValidationError("ordinal thresholds must satisfy 0 < tau1 < tau2 <= 1")
    if minimum_gap <= 0.0 or tau2 - tau1 < minimum_gap:
        raise InputValidationError(
            "ordinal threshold separation is below the positive declared minimum"
        )

    artifact_hash = config.get("calibration_artifact_hash")
    if artifact_hash is not None and (
        not is_sha256_identity(artifact_hash)
    ):
        raise InputValidationError(
            "calibration_artifact_hash must be null or a complete lowercase sha256 identity"
        )

    return OrdinalProjectionConfig(
        projection_id=strings["projection_id"],
        category_id=strings["category_id"],
        threshold_source_id=strings["threshold_source_id"],
        input_names=input_names,
        evidence_scales=scales,
        evidence_exponents=exponents,
        maximum_slopes=maximum_slopes,
        numeric_tolerance=numeric_tolerance,
        aggregate_weights=weights,
        aggregate_power=power,
        tau1=tau1,
        tau2=tau2,
        minimum_threshold_gap=minimum_gap,
        calibration_artifact_hash=artifact_hash,
    )


def zero_anchored_hill(
    values,
    scales,
    exponents,
    *,
    maximum_slopes,
    numeric_tolerance: float,
) -> np.ndarray:
    """Map nonnegative evidence to ``[0,1]`` with an exact zero anchor.

    The positive branch is evaluated as a stable logistic transform of the
    log-ratio, avoiding overflow for large evidence or exponents.
    """

    evidence = np.asarray(values)
    if evidence.dtype.kind not in "fiu" or evidence.ndim < 1:
        raise InputValidationError("Hill evidence must be a numeric array")
    evidence = evidence.astype(np.float64, copy=False)
    scale = np.asarray(scales, dtype=np.float64)
    exponent = np.asarray(exponents, dtype=np.float64)
    slope_cap = np.asarray(maximum_slopes, dtype=np.float64)
    tolerance = _finite_number(numeric_tolerance, name="numeric_tolerance")
    if (
        scale.ndim != 1
        or exponent.ndim != 1
        or slope_cap.ndim != 1
        or scale.shape != exponent.shape
        or scale.shape != slope_cap.shape
    ):
        raise InputValidationError(
            "Hill scales, exponents and maximum slopes must be aligned vectors"
        )
    if evidence.shape[-1] != scale.size:
        raise InputValidationError("Hill evidence final axis does not match its parameters")
    if (
        not np.all(np.isfinite(evidence))
        or np.any(evidence < 0.0)
        or not np.all(np.isfinite(scale))
        or not np.all(np.isfinite(exponent))
        or np.any(scale <= 0.0)
        or np.any(exponent != 1.0)
        or not np.all(np.isfinite(slope_cap))
        or np.any(slope_cap <= 0.0)
        or not 0.0 <= tolerance <= 1.0e-6
    ):
        raise InputValidationError(
            "Hill evidence must be finite and nonnegative; KMFC-v0 uses finite positive scales/caps and exponent 1"
        )
    analytic_slope = 1.0 / scale
    if np.any(~np.isfinite(analytic_slope)) or np.any(
        analytic_slope / slope_cap > 1.0 + tolerance
    ):
        raise InputValidationError("analytic Hill slope exceeds the declared maximum_slope")
    output = np.zeros(evidence.shape, dtype=np.float64)
    positive = evidence > 0.0
    if np.any(positive):
        broadcast_scale = np.broadcast_to(scale, evidence.shape)
        broadcast_exponent = np.broadcast_to(exponent, evidence.shape)
        log_ratio = broadcast_exponent[positive] * (
            np.log(evidence[positive]) - np.log(broadcast_scale[positive])
        )
        positive_output = np.empty(log_ratio.shape, dtype=np.float64)
        nonnegative = log_ratio >= 0.0
        positive_output[nonnegative] = 1.0 / (1.0 + np.exp(-log_ratio[nonnegative]))
        exp_value = np.exp(log_ratio[~nonnegative])
        positive_output[~nonnegative] = exp_value / (1.0 + exp_value)
        output[positive] = positive_output
    return output


def positive_weight_power_mean(values, weights, power: float) -> np.ndarray:
    """Return a bounded, zero-preserving, monotone positive-weight mean."""

    evidence = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    degree = _finite_number(power, name="power")
    if evidence.ndim < 1 or weight.ndim != 1 or evidence.shape[-1] != weight.size:
        raise InputValidationError("power-mean values and weights are not aligned")
    if (
        degree <= 0.0
        or not np.all(np.isfinite(weight))
        or np.any(weight <= 0.0)
        or not np.all(np.isfinite(evidence))
        or np.any(evidence < 0.0)
        or np.any(evidence > 1.0)
    ):
        raise InputValidationError(
            "power mean requires bounded finite values, positive finite weights and positive power"
        )
    # Only relative weights matter.  Scaling first avoids overflow for valid
    # finite weights near the float64 limit and preserves common-scale
    # invariance.
    scaled_weight = weight / float(np.max(weight))
    normalized_weight = scaled_weight / float(np.sum(scaled_weight))
    weighted = np.sum(normalized_weight * np.power(evidence, degree), axis=-1)
    return np.power(weighted, 1.0 / degree)


def _boolean_vector(value, *, size: int, name: str, default: bool) -> np.ndarray:
    if value is None:
        return np.full(size, default, dtype=bool)
    array = np.asarray(value)
    if array.dtype != np.dtype(bool) or array.shape != (size,):
        raise InputValidationError(f"{name} must be a bool vector with one value per replicate")
    return array.copy()


def hard_ordinal_gate(
    severity,
    *,
    tau1: float,
    tau2: float,
    necessary_level1=None,
    necessary_level2=None,
    availability=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply nested hard gates independently inside every replicate.

    Returned arrays contain exact ``0.0``/``1.0`` for available replicates and
    ``NaN`` for unavailable replicates.  Threshold equality is admitted via
    the explicit ``>=`` rule.  Level two is parented by level one, so the
    ordinal implication is structural rather than repaired after evaluation.
    """

    scores = np.asarray(severity, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 1:
        raise InputValidationError("severity must be a nonempty replicate vector")
    first = _finite_number(tau1, name="tau1")
    second = _finite_number(tau2, name="tau2")
    if not 0.0 < first < second <= 1.0:
        raise InputValidationError("ordinal thresholds must satisfy 0 < tau1 < tau2 <= 1")
    available = _boolean_vector(
        availability, size=scores.size, name="availability", default=True
    )
    if np.any(~np.isfinite(scores[available])) or np.any(
        (scores[available] < 0.0) | (scores[available] > 1.0)
    ):
        raise InputValidationError("available severity scores must be finite and lie in [0,1]")
    condition1 = _boolean_vector(
        necessary_level1,
        size=scores.size,
        name="necessary_level1",
        default=True,
    )
    condition2 = _boolean_vector(
        necessary_level2,
        size=scores.size,
        name="necessary_level2",
        default=True,
    )
    level1 = np.full(scores.shape, np.nan, dtype=np.float64)
    level2 = np.full(scores.shape, np.nan, dtype=np.float64)
    first_event = condition1[available] & (scores[available] >= first)
    second_event = first_event & condition2[available] & (scores[available] >= second)
    level1[available] = first_event.astype(np.float64)
    level2[available] = second_event.astype(np.float64)
    return level1, level2


def _nullable_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _nullable_json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_nullable_json(item) for item in value]
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return _nullable_json(value.item())
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class OrdinalProjectionResult:
    """Per-replicate projection with explicit missingness and audit values."""

    configuration: OrdinalProjectionConfig
    availability: np.ndarray
    normalized_evidence: np.ndarray
    severity_score: np.ndarray
    necessary_level1: np.ndarray
    necessary_level2: np.ndarray
    level_ge_1: np.ndarray
    level_ge_2: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ORDINAL_PROJECTION_SCHEMA_VERSION,
            "projection_id": self.configuration.projection_id,
            "category_id": self.configuration.category_id,
            "configuration_hash": self.configuration.hash,
            "availability": _nullable_json(self.availability),
            "normalized_evidence": _nullable_json(self.normalized_evidence),
            "severity_score": _nullable_json(self.severity_score),
            "necessary_level1": _nullable_json(self.necessary_level1),
            "necessary_level2": _nullable_json(self.necessary_level2),
            "level_ge_1": _nullable_json(self.level_ge_1),
            "level_ge_2": _nullable_json(self.level_ge_2),
            "threshold_rule": "greater_than_or_equal_per_replicate",
            "output_semantics": "project_specific_criterion_exceedance_not_physical_probability",
            "physical_validity_status": _PHYSICAL_STATUS,
        }


def project_ordinal_replicates(
    evidence,
    configuration: OrdinalProjectionConfig,
    *,
    evidence_availability=None,
    necessary_level1=None,
    necessary_level2=None,
    necessary_level1_availability=None,
    necessary_level2_availability=None,
) -> OrdinalProjectionResult:
    """Project ``[replicate,input]`` evidence without converting missing to zero."""

    if not isinstance(configuration, OrdinalProjectionConfig):
        raise InputValidationError("configuration must be compiled before ordinal projection")
    raw = np.asarray(evidence)
    if raw.dtype.kind not in "fiu" or raw.ndim != 2 or raw.shape[0] < 1:
        raise InputValidationError("ordinal evidence must be a numeric [replicate,input] array")
    if raw.shape[1] != len(configuration.input_names):
        raise InputValidationError("ordinal evidence input axis does not match the configuration")
    values = raw.astype(np.float64, copy=False)
    if evidence_availability is None:
        evidence_mask = np.ones(values.shape, dtype=bool)
    else:
        evidence_mask = np.asarray(evidence_availability)
        if evidence_mask.dtype != np.dtype(bool) or evidence_mask.shape != values.shape:
            raise InputValidationError(
                "evidence_availability must be bool and aligned to ordinal evidence"
            )
    if np.any(~np.isfinite(values[evidence_mask])) or np.any(values[evidence_mask] < 0.0):
        raise InputValidationError("available ordinal evidence must be finite and nonnegative")

    replicate_count = values.shape[0]
    condition1 = _boolean_vector(
        necessary_level1,
        size=replicate_count,
        name="necessary_level1",
        default=True,
    )
    condition2 = _boolean_vector(
        necessary_level2,
        size=replicate_count,
        name="necessary_level2",
        default=True,
    )
    condition1_available = _boolean_vector(
        necessary_level1_availability,
        size=replicate_count,
        name="necessary_level1_availability",
        default=True,
    )
    condition2_available = _boolean_vector(
        necessary_level2_availability,
        size=replicate_count,
        name="necessary_level2_availability",
        default=True,
    )
    available = (
        np.all(evidence_mask, axis=1)
        & condition1_available
        & condition2_available
    )
    normalized = np.full(values.shape, np.nan, dtype=np.float64)
    severity = np.full(replicate_count, np.nan, dtype=np.float64)
    if np.any(available):
        normalized[available] = zero_anchored_hill(
            values[available],
            configuration.evidence_scales,
            configuration.evidence_exponents,
            maximum_slopes=configuration.maximum_slopes,
            numeric_tolerance=configuration.numeric_tolerance,
        )
        severity[available] = positive_weight_power_mean(
            normalized[available],
            configuration.aggregate_weights,
            configuration.aggregate_power,
        )
    level1, level2 = hard_ordinal_gate(
        severity,
        tau1=configuration.tau1,
        tau2=configuration.tau2,
        necessary_level1=condition1,
        necessary_level2=condition2,
        availability=available,
    )
    return OrdinalProjectionResult(
        configuration=configuration,
        availability=available,
        normalized_evidence=normalized,
        severity_score=severity,
        necessary_level1=np.where(
            condition1_available, condition1.astype(np.float64), np.nan
        ),
        necessary_level2=np.where(
            condition2_available, condition2.astype(np.float64), np.nan
        ),
        level_ge_1=level1,
        level_ge_2=level2,
    )


def wilson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Two-sided Wilson score interval for an exact binary event count."""

    if (
        isinstance(successes, (bool, np.bool_))
        or isinstance(trials, (bool, np.bool_))
        or not isinstance(successes, (int, np.integer))
        or not isinstance(trials, (int, np.integer))
    ):
        raise InputValidationError("Wilson successes and trials must be integers")
    success_count, trial_count = int(successes), int(trials)
    level = _finite_number(confidence, name="confidence")
    if trial_count < 1 or success_count < 0 or success_count > trial_count:
        raise InputValidationError("Wilson counts must satisfy 0 <= successes <= trials and trials >= 1")
    if not 0.0 < level < 1.0:
        raise InputValidationError("Wilson confidence must lie strictly between zero and one")
    z_value = NormalDist().inv_cdf(0.5 + level / 2.0)
    estimate = success_count / trial_count
    denominator = 1.0 + z_value * z_value / trial_count
    center = (estimate + z_value * z_value / (2.0 * trial_count)) / denominator
    half_width = (
        z_value
        * sqrt(
            estimate * (1.0 - estimate) / trial_count
            + z_value * z_value / (4.0 * trial_count * trial_count)
        )
        / denominator
    )
    return {
        "successes": success_count,
        "trials": trial_count,
        "confidence": level,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
        "method": "wilson_score_two_sided_fixed_sample",
    }


def aggregate_ordinal_projection(result: OrdinalProjectionResult) -> dict[str, Any]:
    """Aggregate nested replicate events into exact ordinal counts and rates."""

    if not isinstance(result, OrdinalProjectionResult):
        raise InputValidationError("ordinal aggregation requires a projection result")
    available = np.asarray(result.availability)
    level1 = np.asarray(result.level_ge_1, dtype=np.float64)
    level2 = np.asarray(result.level_ge_2, dtype=np.float64)
    if available.dtype != np.dtype(bool) or level1.shape != available.shape or level2.shape != available.shape:
        raise InputValidationError("ordinal projection arrays are not aligned")
    selected1, selected2 = level1[available], level2[available]
    if available.ndim != 1 or not np.all(np.isin(selected1, (0.0, 1.0))) or not np.all(
        np.isin(selected2, (0.0, 1.0))
    ):
        raise InputValidationError("available ordinal gate values must be exact binary indicators")
    if np.any(selected2 > selected1):
        raise InputValidationError("ordinal level two cannot exceed level one")
    trials = int(np.count_nonzero(available))
    if trials < 1:
        raise InputValidationError("ordinal aggregation requires at least one available replicate")
    successes1 = int(np.sum(selected1, dtype=np.int64))
    successes2 = int(np.sum(selected2, dtype=np.int64))
    n2 = successes2
    n1 = successes1 - successes2
    n0 = trials - successes1
    q1, q2 = successes1 / trials, successes2 / trials
    return {
        "schema_version": ORDINAL_PROJECTION_SCHEMA_VERSION,
        "projection_id": result.configuration.projection_id,
        "category_id": result.configuration.category_id,
        "configuration_hash": result.configuration.hash,
        "trials": trials,
        "successes_level_ge_1": successes1,
        "successes_level_ge_2": successes2,
        "n0": n0,
        "n1": n1,
        "n2": n2,
        "q1": q1,
        "q2": q2,
        "class_probabilities": [n0 / trials, n1 / trials, n2 / trials],
        "level_ge_1_interval": wilson_interval(successes1, trials),
        "level_ge_2_interval": wilson_interval(successes2, trials),
        "interval_boundary": (
            "marginal fixed-sample Wilson intervals; joint three-class uncertainty "
            "requires a preregistered multinomial procedure"
        ),
        "output_semantics": "project_specific_criterion_exceedance_not_physical_probability",
        "physical_validity_status": _PHYSICAL_STATUS,
    }


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty(values.shape, dtype=np.float64)
    nonnegative = values >= 0.0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_value = np.exp(values[~nonnegative])
    output[~nonnegative] = exp_value / (1.0 + exp_value)
    return output


def nested_cumulative_probabilities(level1_logits, conditional_level2_logits) -> dict[str, np.ndarray]:
    """Map two logits to nested cumulative probabilities and three classes.

    ``q2 = q1 * sigmoid(conditional_level2_logit)`` makes ``q2 <= q1`` a
    structural property of the model output rather than an evaluation-time
    clipping operation.
    """

    first = np.asarray(level1_logits, dtype=np.float64)
    second = np.asarray(conditional_level2_logits, dtype=np.float64)
    if first.shape != second.shape or not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise InputValidationError("nested cumulative logits must be finite and shape-aligned")
    q1 = _stable_sigmoid(first)
    conditional_q2 = _stable_sigmoid(second)
    q2 = q1 * conditional_q2
    classes = np.stack((1.0 - q1, q1 - q2, q2), axis=-1)
    return {
        "q1": q1,
        "q2": q2,
        "conditional_q2_given_level1": conditional_q2,
        "class_probabilities": classes,
    }


def ordinal_ranked_probability_score(
    predicted_class_probabilities,
    target_class_probabilities,
    *,
    mask=None,
    sample_weight=None,
) -> float:
    """Return the normalized three-class ranked probability score.

    The score is the mean squared cumulative-probability error over the two
    nontrivial class boundaries, then the weighted mean over selected cases.
    Inputs may be hard one-hot outcomes or soft three-class targets.
    """

    predicted = np.asarray(predicted_class_probabilities, dtype=np.float64)
    target = np.asarray(target_class_probabilities, dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim < 2 or predicted.shape[-1] != 3:
        raise InputValidationError("RPS inputs must be aligned arrays with a final three-class axis")
    if (
        not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(target))
        or np.any(predicted < 0.0)
        or np.any(target < 0.0)
        or not np.allclose(np.sum(predicted, axis=-1), 1.0, rtol=0.0, atol=1.0e-12)
        or not np.allclose(np.sum(target, axis=-1), 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise InputValidationError("RPS class probabilities must be finite, nonnegative and sum to one")
    leading_shape = predicted.shape[:-1]
    selected = np.ones(leading_shape, dtype=bool) if mask is None else np.asarray(mask)
    if selected.dtype != np.dtype(bool) or selected.shape != leading_shape:
        raise InputValidationError("RPS mask must be bool and aligned to cases")
    if not np.any(selected):
        raise InputValidationError("RPS requires at least one selected case")
    weights = (
        np.ones(leading_shape, dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    if weights.shape != leading_shape or np.any(~np.isfinite(weights[selected])) or np.any(
        weights[selected] <= 0.0
    ):
        raise InputValidationError("selected RPS weights must be finite, positive and aligned")
    predicted_cdf = np.cumsum(predicted, axis=-1)[..., :-1]
    target_cdf = np.cumsum(target, axis=-1)[..., :-1]
    per_case = np.mean(np.square(predicted_cdf - target_cdf), axis=-1)
    selected_weights = weights[selected]
    scaled_weights = selected_weights / float(np.max(selected_weights))
    normalized_weights = scaled_weights / float(np.sum(scaled_weights))
    score = float(np.sum(per_case[selected] * normalized_weights))
    if not np.isfinite(score):
        raise InputValidationError("RPS computation produced a non-finite score")
    return score
