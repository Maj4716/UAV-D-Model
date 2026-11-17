"""Envelope-normalized, fixed-size sampling primitives.

The module operationalizes the reusable part of Idea 1 without turning the
damage tree into a label-shaping network.  A deterministic, pre-outcome proxy
provides component reach and state probabilities.  Those values define a
bounded component envelope ``U``, a continuous severity ``S`` and the
normalized utilization ``s = S / U``.  Sampling may use these quantities, but
they remain separate from formal teacher labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from .adaptive_sampling import inclusion_probabilities
from .errors import InputValidationError


NORMALIZED_SAMPLING_VERSION = "idea1-envelope-normalized@1.1.0"
DEFAULT_ZERO_TOLERANCE = 1.0e-12
DEFAULT_POSITIVE_QUANTILES = (0.35, 0.75)
ENVELOPE_BANDS = ("audit_zero", "low", "transition", "high")


def _finite_nonnegative(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(f"{field} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise InputValidationError(f"{field} must be finite and non-negative")
    return result


def _bounded_probability(value: Any, *, field: str) -> float:
    result = _finite_nonnegative(value, field=field)
    if result > 1.0 + 1.0e-12:
        raise InputValidationError(f"{field} must lie in [0, 1]")
    return float(np.clip(result, 0.0, 1.0))


def derive_normalized_envelope(proxy_result: Mapping[str, Any]) -> dict[str, Any]:
    """Derive sample-max-free ``U``, ``S`` and ``s`` from one proxy result.

    ``1-exp(-expected_count)`` is the Poisson support transform: it maps an
    expected event count to the probability of at least one event under the
    reference Poisson approximation.  Averaging over components gives a
    bounded, component-equal envelope suited to multi-component supervision.
    The transform uses no selected-sample statistic and therefore does not
    leak the sampling distribution into the normalization denominator.
    """

    metrics = proxy_result.get("component_metrics")
    states = proxy_result.get("component_state_probabilities")
    if not isinstance(metrics, Mapping) or not metrics:
        raise InputValidationError("proxy result requires nonempty component_metrics")
    if not isinstance(states, Mapping):
        raise InputValidationError(
            "proxy result requires component_state_probabilities"
        )
    component_ids = sorted(str(value) for value in metrics)
    if set(component_ids) != {str(value) for value in states}:
        raise InputValidationError(
            "proxy component metrics and state probabilities must align"
        )

    hit_support: dict[str, float] = {}
    penetration_support: dict[str, float] = {}
    severity: dict[str, float] = {}
    support_indicator: dict[str, int] = {}
    threshold = _finite_nonnegative(
        proxy_result.get("component_support_threshold", 0.05),
        field="component_support_threshold",
    )
    for identifier in component_ids:
        record = metrics[identifier]
        if not isinstance(record, Mapping):
            raise InputValidationError("each component metric must be an object")
        expected_hits = _finite_nonnegative(
            record.get("expected_hits"),
            field=f"component_metrics.{identifier}.expected_hits",
        )
        expected_penetrations = _finite_nonnegative(
            record.get("expected_penetrations"),
            field=f"component_metrics.{identifier}.expected_penetrations",
        )
        if expected_penetrations > expected_hits + 1.0e-9:
            raise InputValidationError(
                "expected component penetrations cannot exceed expected hits"
            )
        state_record = states[identifier]
        if not isinstance(state_record, Mapping):
            raise InputValidationError("each component state record must be an object")
        state_values = [
            _bounded_probability(
                value,
                field=f"component_state_probabilities.{identifier}.{state}",
            )
            for state, value in state_record.items()
        ]
        hit_support[identifier] = float(-math.expm1(-expected_hits))
        penetration_support[identifier] = float(
            -math.expm1(-expected_penetrations)
        )
        severity[identifier] = max(state_values, default=0.0)
        support_indicator[identifier] = int(expected_hits >= threshold)

    u_hit = float(np.mean(list(hit_support.values())))
    u_penetration = float(np.mean(list(penetration_support.values())))
    damage_severity = float(np.mean(list(severity.values())))
    utilization_raw = (
        0.0
        if u_penetration <= DEFAULT_ZERO_TOLERANCE
        else damage_severity / u_penetration
    )
    utilization = float(np.clip(utilization_raw, 0.0, 1.0))
    tree = proxy_result.get("tree_outputs", {})
    if not isinstance(tree, Mapping):
        raise InputValidationError("proxy tree_outputs must be an object")
    tree_values = [
        _bounded_probability(value, field=f"tree_outputs.{name}")
        for name, value in tree.items()
    ]
    tree_information = float(
        np.mean([4.0 * value * (1.0 - value) for value in tree_values])
    ) if tree_values else 0.0
    return {
        "schema_version": NORMALIZED_SAMPLING_VERSION,
        "component_count": len(component_ids),
        "component_ids": component_ids,
        "component_hit_support": hit_support,
        "component_penetration_support": penetration_support,
        "component_damage_severity": severity,
        "component_support_indicator": support_indicator,
        "supported_component_count": int(sum(support_indicator.values())),
        "u_hit": u_hit,
        "u_penetration": u_penetration,
        "damage_severity": damage_severity,
        "normalized_utilization_raw": float(utilization_raw),
        "normalized_utilization": utilization,
        "utilization_information": float(4.0 * utilization * (1.0 - utilization)),
        "tree_information": tree_information,
        "envelope_consistency_excess": float(
            max(0.0, damage_severity - u_penetration)
        ),
        "normalization_denominator_source": (
            "fixed_case_proxy_component_penetration_envelope_not_sample_maximum"
        ),
    }


def calibrate_envelope_policy(
    envelopes: Sequence[Mapping[str, Any]],
    *,
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
    positive_quantiles: tuple[float, float] = DEFAULT_POSITIVE_QUANTILES,
) -> dict[str, Any]:
    """Freeze positive-``U`` band edges and component support prevalence."""

    if not envelopes:
        raise InputValidationError("envelope calibration requires nonempty rows")
    if not math.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise InputValidationError("zero_tolerance must be finite and non-negative")
    q_low, q_high = map(float, positive_quantiles)
    if not 0.0 < q_low < q_high < 1.0:
        raise InputValidationError(
            "positive envelope quantiles must satisfy 0 < low < high < 1"
        )
    component_ids = list(envelopes[0].get("component_ids", []))
    if not component_ids:
        raise InputValidationError("calibration rows require component identities")
    values: list[float] = []
    support = np.zeros(len(component_ids), dtype=np.float64)
    for row in envelopes:
        if list(row.get("component_ids", [])) != component_ids:
            raise InputValidationError("calibration component identities must align")
        value = _bounded_probability(row.get("u_penetration"), field="u_penetration")
        values.append(value)
        indicator = row.get("component_support_indicator")
        if not isinstance(indicator, Mapping):
            raise InputValidationError(
                "calibration rows require component_support_indicator"
            )
        support += np.asarray(
            [
                _bounded_probability(
                    indicator.get(identifier),
                    field=f"component_support_indicator.{identifier}",
                )
                for identifier in component_ids
            ],
            dtype=np.float64,
        )
    u_values = np.asarray(values, dtype=np.float64)
    positive = np.sort(u_values[u_values > zero_tolerance])
    if positive.size < 2:
        raise InputValidationError(
            "envelope calibration requires at least two positive U values"
        )
    low_edge, high_edge = np.quantile(positive, [q_low, q_high]).tolist()
    unique = np.unique(positive)
    if not low_edge < high_edge:
        if unique.size < 2:
            raise InputValidationError(
                "positive U values do not identify distinct envelope bands"
            )
        low_index = min(int(math.floor(q_low * (unique.size - 1))), unique.size - 2)
        high_index = max(
            low_index + 1,
            min(int(math.ceil(q_high * (unique.size - 1))), unique.size - 1),
        )
        low_edge = float(unique[low_index])
        high_edge = float(unique[high_index])
    prevalence = support / float(len(envelopes))
    zero_mass = float(np.mean(u_values <= zero_tolerance))
    audit_zero_target = min(0.05, zero_mass)
    positive_target = 1.0 - audit_zero_target
    positive_reference = {
        "low": 0.20,
        "transition": 0.45,
        "high": 0.30,
    }
    positive_reference_total = sum(positive_reference.values())
    return {
        "schema_version": NORMALIZED_SAMPLING_VERSION,
        "calibration_count": len(envelopes),
        "zero_tolerance": float(zero_tolerance),
        "zero_mass": zero_mass,
        "positive_quantiles": [q_low, q_high],
        "positive_band_edges": {
            "low_to_transition": float(low_edge),
            "transition_to_high": float(high_edge),
        },
        "component_ids": component_ids,
        "component_support_prevalence": {
            identifier: float(prevalence[index])
            for index, identifier in enumerate(component_ids)
        },
        "target_band_fractions": {
            "audit_zero": audit_zero_target,
            **{
                band: positive_target * weight / positive_reference_total
                for band, weight in positive_reference.items()
            },
        },
        "target_band_policy": (
            "audit_zero_mass_capped_at_5pct_then_positive_targets_renormalized"
        ),
    }


def envelope_band(
    envelope: Mapping[str, Any], calibration: Mapping[str, Any]
) -> str:
    """Assign one of the four frozen envelope bands."""

    value = _bounded_probability(envelope.get("u_penetration"), field="u_penetration")
    tolerance = _finite_nonnegative(
        calibration.get("zero_tolerance"), field="zero_tolerance"
    )
    edges = calibration.get("positive_band_edges")
    if not isinstance(edges, Mapping):
        raise InputValidationError("calibration requires positive_band_edges")
    low = _bounded_probability(
        edges.get("low_to_transition"), field="low_to_transition"
    )
    high = _bounded_probability(
        edges.get("transition_to_high"), field="transition_to_high"
    )
    if not tolerance < low < high <= 1.0:
        raise InputValidationError("calibrated envelope edges are not increasing")
    if value <= tolerance:
        return "audit_zero"
    if value <= low:
        return "low"
    if value <= high:
        return "transition"
    return "high"


def idea1_selection_score(
    envelope: Mapping[str, Any], calibration: Mapping[str, Any]
) -> dict[str, float]:
    """Return the frozen pre-outcome score and its interpretable components."""

    component_ids = list(calibration.get("component_ids", []))
    prevalence = calibration.get("component_support_prevalence")
    support = envelope.get("component_hit_support")
    if not component_ids or not isinstance(prevalence, Mapping) or not isinstance(support, Mapping):
        raise InputValidationError(
            "Idea-1 score requires aligned component calibration and support"
        )
    floor = 1.0 / max(1, int(calibration.get("calibration_count", 1)))
    inverse_sqrt = np.asarray(
        [1.0 / math.sqrt(max(float(prevalence[identifier]), floor)) for identifier in component_ids],
        dtype=np.float64,
    )
    soft_support = np.asarray(
        [
            _bounded_probability(
                support.get(identifier),
                field=f"component_hit_support.{identifier}",
            )
            for identifier in component_ids
        ],
        dtype=np.float64,
    )
    rarity = float(np.dot(soft_support, inverse_sqrt) / np.sum(inverse_sqrt))
    utilization_information = _bounded_probability(
        envelope.get("utilization_information"),
        field="utilization_information",
    )
    tree_information = _bounded_probability(
        envelope.get("tree_information"), field="tree_information"
    )
    coverage = float(np.mean(soft_support))
    boundary_information = 0.5 * utilization_information + 0.5 * tree_information
    total = 0.50 * rarity + 0.30 * boundary_information + 0.20 * coverage
    return {
        "rarity_coverage": rarity,
        "boundary_information": float(boundary_information),
        "soft_component_coverage": coverage,
        "total": float(total + 1.0e-12),
    }


def fixed_size_inclusion_probabilities(
    scores: Sequence[float] | np.ndarray,
    *,
    sample_size: int,
    rho_min: float = 0.05,
) -> np.ndarray:
    """Build positive first-order probabilities summing to an exact integer."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise InputValidationError("fixed-size scores must be a finite vector")
    if np.any(values < 0.0):
        raise InputValidationError("fixed-size scores must be non-negative")
    if not isinstance(sample_size, (int, np.integer)) or isinstance(sample_size, (bool, np.bool_)):
        raise InputValidationError("sample_size must be an integer")
    count = int(sample_size)
    if not 0 < count <= values.size:
        raise InputValidationError("sample_size must lie in (0, candidate_count]")
    if count == values.size:
        return np.ones(values.size, dtype=np.float64)
    if not math.isfinite(rho_min) or not 0.0 < rho_min <= 1.0:
        raise InputValidationError("rho_min must lie in (0, 1]")
    admissible_floor = min(float(rho_min), 0.5 * count / values.size)
    probabilities = inclusion_probabilities(
        values,
        expected_count=float(count),
        rho_min=admissible_floor,
    )
    difference = float(count - np.sum(probabilities))
    if abs(difference) > 1.0e-12:
        candidates = np.flatnonzero(
            (probabilities > 1.0e-12) & (probabilities < 1.0 - 1.0e-12)
        )
        if candidates.size == 0:
            raise RuntimeError("cannot correct fixed-size inclusion probability sum")
        index = int(candidates[np.argmax(np.minimum(probabilities[candidates], 1.0 - probabilities[candidates]))])
        probabilities[index] += difference
    if (
        np.any(probabilities <= 0.0)
        or np.any(probabilities > 1.0)
        or not math.isclose(float(np.sum(probabilities)), count, rel_tol=0.0, abs_tol=1.0e-9)
    ):
        raise RuntimeError("fixed-size inclusion probabilities are invalid")
    return probabilities


def global_marginal_coverage_probabilities(
    support_matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    strata: Sequence[Sequence[int] | np.ndarray],
    quotas: Sequence[int],
    auxiliary_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    auxiliary_weights: Sequence[float] | np.ndarray | None = None,
    smoothing: float = 5.0,
    score_exponent: float = 1.0,
    iterations: int = 80,
    damping: float = 0.2,
    rho_min: float = 0.08,
) -> np.ndarray:
    """Allocate inclusion probabilities by global marginal component coverage.

    The probability vector is optimized jointly across all components while
    retaining an exact integer probability sum inside every declared stratum.
    A candidate receives high utility when it supports components whose current
    expected counts are small.  Optional pre-outcome auxiliary features are
    standardized inside each stratum and added with frozen signed weights.  This
    permits an explicit coverage/information trade-off without mixing teacher
    labels into the selector.  The update is deterministic; randomized pivotal
    sampling is applied separately after the probabilities are frozen.
    """

    support = np.asarray(support_matrix, dtype=np.float64)
    if (
        support.ndim != 2
        or support.shape[0] == 0
        or support.shape[1] == 0
        or not np.all(np.isfinite(support))
        or np.any(support < 0.0)
        or np.any(support > 1.0)
    ):
        raise InputValidationError(
            "support_matrix must be a finite nonempty matrix in [0, 1]"
        )
    if len(strata) != len(quotas) or not strata:
        raise InputValidationError("strata and quotas must be nonempty and aligned")
    if not math.isfinite(smoothing) or smoothing <= 0.0:
        raise InputValidationError("smoothing must be finite and positive")
    if not math.isfinite(score_exponent) or score_exponent <= 0.0:
        raise InputValidationError("score_exponent must be finite and positive")
    if not isinstance(iterations, (int, np.integer)) or int(iterations) <= 0:
        raise InputValidationError("iterations must be a positive integer")
    if not math.isfinite(damping) or not 0.0 < damping <= 1.0:
        raise InputValidationError("damping must lie in (0, 1]")
    if not math.isfinite(rho_min) or not 0.0 < rho_min <= 1.0:
        raise InputValidationError("rho_min must lie in (0, 1]")

    auxiliary: np.ndarray | None = None
    weights: np.ndarray | None = None
    if (auxiliary_features is None) != (auxiliary_weights is None):
        raise InputValidationError(
            "auxiliary_features and auxiliary_weights must be supplied together"
        )
    if auxiliary_features is not None and auxiliary_weights is not None:
        auxiliary = np.asarray(auxiliary_features, dtype=np.float64)
        weights = np.asarray(auxiliary_weights, dtype=np.float64)
        if (
            auxiliary.ndim != 2
            or auxiliary.shape[0] != support.shape[0]
            or auxiliary.shape[1] == 0
            or not np.all(np.isfinite(auxiliary))
        ):
            raise InputValidationError(
                "auxiliary_features must be a finite candidate-aligned matrix"
            )
        if (
            weights.ndim != 1
            or weights.size != auxiliary.shape[1]
            or not np.all(np.isfinite(weights))
        ):
            raise InputValidationError(
                "auxiliary_weights must be a finite feature-aligned vector"
            )

    normalized_strata: list[np.ndarray] = []
    seen: list[int] = []
    normalized_quotas: list[int] = []
    for raw_indices, raw_quota in zip(strata, quotas, strict=True):
        indices = np.asarray(raw_indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise InputValidationError("each coverage stratum must be nonempty")
        if np.any(indices < 0) or np.any(indices >= support.shape[0]):
            raise InputValidationError("coverage stratum index is out of range")
        if len(np.unique(indices)) != indices.size:
            raise InputValidationError("coverage stratum indices must be unique")
        if (
            not isinstance(raw_quota, (int, np.integer))
            or isinstance(raw_quota, (bool, np.bool_))
            or not 0 <= int(raw_quota) <= indices.size
        ):
            raise InputValidationError(
                "each coverage quota must lie in [0, stratum_size]"
            )
        normalized_strata.append(indices)
        normalized_quotas.append(int(raw_quota))
        seen.extend(map(int, indices))
    if sorted(seen) != list(range(support.shape[0])):
        raise InputValidationError(
            "coverage strata must partition every candidate exactly once"
        )

    probabilities = np.zeros(support.shape[0], dtype=np.float64)
    for indices, quota in zip(
        normalized_strata, normalized_quotas, strict=True
    ):
        probabilities[indices] = float(quota) / float(indices.size)

    for _ in range(int(iterations)):
        expected_counts = support.T @ probabilities
        marginal_utility = support @ (1.0 / (expected_counts + float(smoothing)))
        proposal = np.zeros_like(probabilities)
        for indices, quota in zip(
            normalized_strata, normalized_quotas, strict=True
        ):
            if quota == 0:
                continue
            if quota == indices.size:
                proposal[indices] = 1.0
                continue
            local = marginal_utility[indices]
            standard_deviation = float(np.std(local, ddof=0))
            standardized_coverage = (
                np.zeros_like(local)
                if standard_deviation <= 1.0e-15
                else (local - float(np.mean(local))) / standard_deviation
            )
            combined_utility = standardized_coverage
            if auxiliary is not None and weights is not None:
                local_auxiliary = auxiliary[indices]
                auxiliary_mean = np.mean(local_auxiliary, axis=0)
                auxiliary_standard_deviation = np.std(
                    local_auxiliary, axis=0, ddof=0
                )
                standardized_auxiliary = np.zeros_like(local_auxiliary)
                varying = auxiliary_standard_deviation > 1.0e-15
                standardized_auxiliary[:, varying] = (
                    local_auxiliary[:, varying] - auxiliary_mean[varying]
                ) / auxiliary_standard_deviation[varying]
                combined_utility = (
                    combined_utility + standardized_auxiliary @ weights
                )
            scores = np.exp(
                np.clip(float(score_exponent) * combined_utility, -6.0, 6.0)
            )
            proposal[indices] = fixed_size_inclusion_probabilities(
                scores,
                sample_size=quota,
                rho_min=float(rho_min),
            )
        probabilities = (
            (1.0 - float(damping)) * probabilities
            + float(damping) * proposal
        )

    for indices, quota in zip(
        normalized_strata, normalized_quotas, strict=True
    ):
        difference = float(quota - np.sum(probabilities[indices]))
        if abs(difference) <= 1.0e-12:
            continue
        adjustable = indices[
            (probabilities[indices] > 1.0e-12)
            & (probabilities[indices] < 1.0 - 1.0e-12)
        ]
        if adjustable.size == 0:
            raise RuntimeError("cannot correct coverage probability sum")
        index = int(
            adjustable[
                np.argmax(
                    np.minimum(
                        probabilities[adjustable],
                        1.0 - probabilities[adjustable],
                    )
                )
            ]
        )
        probabilities[index] += difference

    if (
        np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or any(
            not math.isclose(
                float(np.sum(probabilities[indices])),
                quota,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            for indices, quota in zip(
                normalized_strata, normalized_quotas, strict=True
            )
        )
    ):
        raise RuntimeError("global coverage inclusion probabilities are invalid")
    return probabilities


def pivotal_sample(
    probabilities: Sequence[float] | np.ndarray, *, seed: int
) -> np.ndarray:
    """Select a fixed-size sample while preserving first-order probabilities.

    The pairwise pivotal updates preserve each candidate's expectation and
    make at least one member of every processed pair integral.  An integer
    probability sum therefore terminates with exactly that many selections.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise InputValidationError("pivotal probabilities must lie in [0, 1]")
    target = int(round(float(np.sum(values))))
    if not math.isclose(float(np.sum(values)), target, rel_tol=0.0, abs_tol=1.0e-9):
        raise InputValidationError("pivotal probability sum must be an integer")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or int(seed) < 0:
        raise InputValidationError("pivotal seed must be a non-negative integer")

    working = values.copy()
    rng = np.random.default_rng(int(seed))
    active = [int(index) for index in np.flatnonzero((working > 1.0e-12) & (working < 1.0 - 1.0e-12))]
    rng.shuffle(active)
    while len(active) >= 2:
        first = active.pop()
        second = active.pop()
        left, right = float(working[first]), float(working[second])
        total = left + right
        draw = float(rng.random())
        if total < 1.0:
            if draw < left / total:
                working[first], working[second] = total, 0.0
            else:
                working[first], working[second] = 0.0, total
        else:
            denominator = 2.0 - total
            if denominator <= 1.0e-15:
                working[first], working[second] = 1.0, 1.0
            elif draw < (1.0 - right) / denominator:
                working[first], working[second] = 1.0, total - 1.0
            else:
                working[first], working[second] = total - 1.0, 1.0
        for index in (first, second):
            if 1.0e-12 < working[index] < 1.0 - 1.0e-12:
                active.append(index)
            elif working[index] <= 1.0e-12:
                working[index] = 0.0
            else:
                working[index] = 1.0
    if active:
        index = active.pop()
        working[index] = float(round(working[index]))
    selected = working >= 0.5
    if int(np.count_nonzero(selected)) != target:
        raise RuntimeError("pivotal sampler did not preserve the fixed sample size")
    return selected


def design_weight_quality(
    probabilities: Sequence[float] | np.ndarray,
    selected: Sequence[bool] | np.ndarray,
) -> dict[str, float]:
    """Summarize raw ``1/rho`` weight quality for the realized sample."""

    rho = np.asarray(probabilities, dtype=np.float64)
    mask = np.asarray(selected, dtype=bool)
    if rho.shape != mask.shape or rho.ndim != 1 or not np.any(mask):
        raise InputValidationError("probabilities and selected mask must align")
    if np.any(rho[mask] <= 0.0):
        raise InputValidationError("selected candidates require positive probability")
    weights = 1.0 / rho[mask]
    total = float(np.sum(weights))
    ess = total * total / float(np.sum(np.square(weights)))
    mean = float(np.mean(weights))
    cv = float(np.std(weights, ddof=0) / mean)
    return {
        "selected_count": float(np.count_nonzero(mask)),
        "ess": ess,
        "ess_fraction": ess / float(np.count_nonzero(mask)),
        "weight_cv": cv,
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "inclusion_probability_min_selected": float(np.min(rho[mask])),
    }


__all__ = [
    "DEFAULT_POSITIVE_QUANTILES",
    "DEFAULT_ZERO_TOLERANCE",
    "ENVELOPE_BANDS",
    "NORMALIZED_SAMPLING_VERSION",
    "calibrate_envelope_policy",
    "derive_normalized_envelope",
    "design_weight_quality",
    "envelope_band",
    "fixed_size_inclusion_probabilities",
    "global_marginal_coverage_probabilities",
    "idea1_selection_score",
    "pivotal_sample",
]
