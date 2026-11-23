"""Versioned component-level hurdle targets derived from engine statistics.

The primary support target is the per-replicate hit probability.  It is an
R-independent estimand even though its Monte Carlo uncertainty depends on R.
Binary support is only a derived, uncertainty-aware view.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Mapping

from .errors import InputValidationError


HURDLE_TARGET_SCHEMA_VERSION = "1.1.0"
HURDLE_TARGET_SPEC_ID = "component-fragment-hurdle-targets-v2"
DEFAULT_SUPPORT_PROBABILITY_THRESHOLD = 0.05
DEFAULT_DOMAIN_EPSILON = 0.01
SUPPORT_DECISION_RULE = (
    "monte_carlo_ci95_and_domain_identification_interval_exclude_fixed_"
    "probability_threshold"
)
HIT_PROBABILITY_ESTIMAND = (
    "P(component_has_at_least_one_fragment_hit_in_a_replicate_given_case)"
)
CONDITIONAL_METRICS = {
    "log1p_deposited_energy_j": (
        "hit_conditioned_log1p_deposited_energy_j",
        "conditional_mean_log1p_deposited_energy_j_given_component_hit",
    ),
    "log1p_hit_count": (
        "hit_conditioned_log1p_hit_count",
        "conditional_mean_log1p_hit_count_given_component_hit",
    ),
}
CONDITIONAL_METRIC_CONDITIONING = "replicate_has_component_hit"


def _refuse(message: str, **context: Any) -> None:
    raise InputValidationError(message, context=context)


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse("hurdle statistic must be numeric", field=field, value=repr(value))
    number = float(value)
    if not math.isfinite(number):
        _refuse("hurdle statistic must be finite", field=field, value=repr(value))
    return number


def _validated_statistics(
    value: Any,
    *,
    field: str,
    probability: bool,
    expected_sample_count: int | None,
    allow_empty: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _refuse("hurdle statistic record must be an object", field=field)
    required = {
        "mean",
        "stddev",
        "ci95_lower",
        "ci95_upper",
        "sample_count",
        "uncertainty_estimable",
        "estimand",
        "conditioning",
    }
    missing = sorted(required - set(value))
    if missing:
        _refuse("hurdle statistic record is incomplete", field=field, missing=missing)
    sample_count = value["sample_count"]
    minimum_sample_count = 0 if allow_empty else 1
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, Integral)
        or int(sample_count) < minimum_sample_count
    ):
        _refuse(
            (
                "hurdle statistic sample_count must be a non-negative integer"
                if allow_empty
                else "hurdle statistic sample_count must be a positive integer"
            ),
            field=field,
            sample_count=sample_count,
        )
    sample_count = int(sample_count)
    if expected_sample_count is not None and sample_count != expected_sample_count:
        _refuse(
            "hurdle statistic sample_count disagrees with engine replicate_count",
            field=field,
            sample_count=sample_count,
            replicate_count=expected_sample_count,
        )
    if sample_count == 0:
        if any(
            value[name] is not None
            for name in ("mean", "stddev", "ci95_lower", "ci95_upper")
        ):
            _refuse(
                "empty hurdle statistic value and uncertainty fields must be null",
                field=field,
            )
        if value["uncertainty_estimable"] is not False:
            _refuse(
                "empty hurdle statistic uncertainty_estimable must be false",
                field=field,
            )
        return {
            "value": None,
            "sample_count": 0,
            "sample_stddev": None,
            "standard_error": None,
            "ci95": None,
            "uncertainty_mask": False,
            "source_estimand": str(value["estimand"]),
            "source_conditioning": str(value["conditioning"]),
        }
    mean = _finite_number(value["mean"], field=f"{field}.mean")
    if probability and not 0.0 <= mean <= 1.0:
        _refuse(
            "probability hurdle statistic must lie in [0,1]",
            field=field,
            mean=mean,
        )
    if not probability and mean < 0.0:
        _refuse(
            "log1p hurdle statistic must be nonnegative",
            field=field,
            mean=mean,
        )

    uncertainty_estimable = value["uncertainty_estimable"]
    if not isinstance(uncertainty_estimable, bool):
        _refuse(
            "uncertainty_estimable must be boolean",
            field=field,
        )
    should_be_estimable = sample_count >= 2
    if uncertainty_estimable != should_be_estimable:
        _refuse(
            "uncertainty_estimable disagrees with sample_count",
            field=field,
            sample_count=sample_count,
        )
    if uncertainty_estimable:
        stddev = _finite_number(value["stddev"], field=f"{field}.stddev")
        lower = _finite_number(value["ci95_lower"], field=f"{field}.ci95_lower")
        upper = _finite_number(value["ci95_upper"], field=f"{field}.ci95_upper")
        if stddev < 0.0 or lower > mean or mean > upper:
            _refuse(
                "hurdle uncertainty must be ordered and contain its mean",
                field=field,
                mean=mean,
                stddev=stddev,
                lower=lower,
                upper=upper,
            )
        if probability and not (0.0 <= lower <= upper <= 1.0):
            _refuse(
                "probability hurdle interval must lie in [0,1]",
                field=field,
                lower=lower,
                upper=upper,
            )
        standard_error = stddev / math.sqrt(sample_count)
        ci95: list[float] | None = [lower, upper]
    else:
        if any(
            value[name] is not None
            for name in ("stddev", "ci95_lower", "ci95_upper")
        ):
            _refuse(
                "inestimable hurdle uncertainty fields must be null",
                field=field,
            )
        stddev = standard_error = None
        ci95 = None
    return {
        "value": mean,
        "sample_count": sample_count,
        "sample_stddev": stddev,
        "standard_error": standard_error,
        "ci95": ci95,
        "uncertainty_mask": uncertainty_estimable,
        "source_estimand": str(value["estimand"]),
        "source_conditioning": str(value["conditioning"]),
    }


def _unavailable_conditional(reason: str) -> dict[str, Any]:
    return {
        "value": None,
        "sample_count": 0,
        "sample_stddev": None,
        "standard_error": None,
        "ci95": None,
        "mask": False,
        "status": "unavailable",
        "reason": reason,
    }


def _conditional_target(
    fragment_metrics: Mapping[str, Any],
    *,
    public_name: str,
    source_name: str,
    expected_estimand: str,
    domain_point_identified: bool,
) -> dict[str, Any]:
    if source_name not in fragment_metrics:
        return _unavailable_conditional(
            "engine aggregate lacks an explicit hit-conditioned log1p statistic; "
            "unconditional totals are intentionally not substituted"
        )
    source = fragment_metrics[source_name]
    if not isinstance(source, Mapping):
        _refuse("conditional hurdle statistic must be an object", field=public_name)
    if source.get("conditioning") != CONDITIONAL_METRIC_CONDITIONING:
        _refuse(
            "conditional hurdle statistic has incompatible conditioning",
            field=public_name,
            expected=CONDITIONAL_METRIC_CONDITIONING,
            observed=source.get("conditioning"),
        )
    if source.get("estimand") != expected_estimand:
        _refuse(
            "conditional hurdle statistic has incompatible estimand",
            field=public_name,
            expected=expected_estimand,
            observed=source.get("estimand"),
        )
    target = _validated_statistics(
        source,
        field=public_name,
        probability=False,
        expected_sample_count=None,
        allow_empty=True,
    )
    if target["sample_count"] == 0:
        target.update(
            {
                "mask": False,
                "status": "unavailable",
                "reason": "zero hit-positive replicates; conditional target is undefined",
                "estimand": expected_estimand,
                "conditioning": CONDITIONAL_METRIC_CONDITIONING,
            }
        )
        return target
    if target["sample_count"] < 2 or not target["uncertainty_mask"]:
        target.update(
            {
                "mask": False,
                "status": "unavailable",
                "reason": (
                    "fewer than two hit-positive replicates; conditional "
                    "uncertainty is not estimable"
                ),
                "estimand": expected_estimand,
                "conditioning": CONDITIONAL_METRIC_CONDITIONING,
            }
        )
        return target
    if not domain_point_identified:
        target.update(
            {
                "mask": False,
                "status": "domain_censored",
                "reason": (
                    "an unresolved penetration event can change this component's "
                    "hit-conditioned numerator or conditioning set"
                ),
                "estimand": expected_estimand,
                "conditioning": CONDITIONAL_METRIC_CONDITIONING,
            }
        )
        return target
    target.update(
        {
            "mask": True,
            "status": "available",
            "estimand": expected_estimand,
            "conditioning": CONDITIONAL_METRIC_CONDITIONING,
        }
    )
    return target


def _state_target(
    component: Mapping[str, Any],
    *,
    state: str,
    replicate_count: int,
    domain_epsilon: float,
    identification_bound: Mapping[str, Any] | None = None,
    affected_fraction: float = 0.0,
) -> dict[str, Any]:
    try:
        source = component["damage_states"]["fragment"][state]
    except (KeyError, TypeError):
        return {
            "value": None,
            "sample_count": 0,
            "sample_stddev": None,
            "standard_error": None,
            "ci95": None,
            "uncertainty_mask": False,
            "mask": False,
            "status": "unavailable",
            "reason": f"fragment state {state!r} is absent from the engine result",
            "identification_interval": {
                "lower": 0.0,
                "upper": 1.0,
                "width": 1.0,
                "mask": False,
                "method": "unavailable_fragment_state",
                "uncertainty_type": (
                    "target_unavailable_not_monte_carlo_confidence"
                ),
            },
            "point_supervision_error_bound": 1.0,
        }
    target = _validated_statistics(
        source,
        field=f"fragment_state.{state}",
        probability=True,
        expected_sample_count=replicate_count,
    )
    interval = _identification_interval(
        target["value"],
        identification_bound,
        affected_fraction=(
            affected_fraction
            if identification_bound is None and affected_fraction > 0.0
            else None
        ),
    )
    epsilon_complete = interval["width"] <= domain_epsilon + 1.0e-12
    target.update(
        {
            "mask": epsilon_complete,
            "status": (
                "available_epsilon_complete"
                if epsilon_complete
                else "domain_censored_interval"
            ),
            "estimand": f"P(fragment_state_at_least_{state}_in_a_replicate_given_case)",
            "identification_interval": interval,
            "point_supervision_error_bound": interval["width"],
        }
    )
    return target


def _identification_interval(
    point: float,
    bound: Mapping[str, Any] | None,
    *,
    affected_fraction: float | None = None,
) -> dict[str, Any]:
    """Keep censoring identification separate from Monte Carlo confidence."""

    if bound is None:
        if affected_fraction is None:
            lower = upper = float(point)
            method = "complete_domain_point_identified"
        else:
            lower = float(point)
            upper = min(1.0, float(point) + float(affected_fraction))
            method = "affected_replicate_worst_case_hit_indicator"
    else:
        lower = _finite_number(bound.get("lower"), field="identification.lower")
        upper = _finite_number(bound.get("upper"), field="identification.upper")
        method = str(bound.get("method", "affected_replicate_worst_case_bounded_outcome"))
    if not 0.0 <= lower <= point <= upper <= 1.0:
        _refuse(
            "identification interval must be ordered in [0,1] and contain the point",
            lower=lower,
            point=point,
            upper=upper,
        )
    return {
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
        "mask": True,
        "method": method,
        "uncertainty_type": "domain_censoring_identification_not_monte_carlo_confidence",
    }


def _derived_support(
    p_hit: Mapping[str, Any],
    *,
    threshold: float,
    point_supervision_admissible: bool,
) -> dict[str, Any]:
    base = {
        "threshold": threshold,
        "decision_rule": SUPPORT_DECISION_RULE,
        "threshold_semantics": "support_means_p_hit_at_least_threshold",
        "uncertainty_source": (
            "p_hit_fixed_sample_ci95_and_domain_identification_interval"
        ),
        "evidence_ci95": p_hit["ci95"],
        "evidence_identification_interval": p_hit.get("identification_interval"),
    }
    if not point_supervision_admissible:
        return {
            **base,
            "value": None,
            "mask": False,
            "status": "domain_censored",
        }
    ci95 = p_hit["ci95"]
    if ci95 is None:
        return {
            **base,
            "value": None,
            "mask": False,
            "status": "uncertainty_unavailable",
        }
    lower, upper = ci95
    identification = p_hit.get("identification_interval") or {
        "lower": p_hit["value"],
        "upper": p_hit["value"],
    }
    identified_lower = float(identification["lower"])
    identified_upper = float(identification["upper"])
    if lower >= threshold and identified_lower >= threshold:
        return {**base, "value": 1, "mask": True, "status": "positive"}
    if upper < threshold and identified_upper < threshold:
        return {**base, "value": 0, "mask": True, "status": "negative"}
    return {
        **base,
        "value": None,
        "mask": False,
        "status": "ambiguous_ci_crosses_threshold",
    }


def extract_component_hurdle_targets(
    result: Mapping[str, Any],
    *,
    support_probability_threshold: float = DEFAULT_SUPPORT_PROBABILITY_THRESHOLD,
    domain_epsilon: float = DEFAULT_DOMAIN_EPSILON,
) -> dict[str, Any]:
    """Extract stable component estimands without changing the engine result."""

    if not isinstance(result, Mapping):
        _refuse("engine result must be an object")
    threshold = _finite_number(
        support_probability_threshold,
        field="support_probability_threshold",
    )
    if not 0.0 <= threshold <= 1.0:
        _refuse(
            "support probability threshold must lie in [0,1]",
            threshold=threshold,
        )
    epsilon = _finite_number(domain_epsilon, field="domain_epsilon")
    if not 0.0 <= epsilon <= 1.0:
        _refuse("domain_epsilon must lie in [0,1]", domain_epsilon=epsilon)
    replicate_count = result.get("replicate_count")
    if (
        isinstance(replicate_count, bool)
        or not isinstance(replicate_count, Integral)
        or int(replicate_count) < 1
    ):
        _refuse("engine result replicate_count must be a positive integer")
    replicate_count = int(replicate_count)
    components = result.get("components")
    if not isinstance(components, Mapping) or not components:
        _refuse("engine result must contain component aggregates")

    coverage = result.get("penetration_domain_coverage", {})
    if not isinstance(coverage, Mapping):
        _refuse("penetration_domain_coverage must be an object")
    affected_fraction = _finite_number(
        coverage.get("affected_replicate_fraction", 1.0),
        field="penetration_domain_coverage.affected_replicate_fraction",
    )
    if not 0.0 <= affected_fraction <= 1.0:
        _refuse("affected replicate fraction must lie in [0,1]")
    completion_status = str(result.get("scientific_completion_status", "missing"))
    domain_point_identified = affected_fraction == 0.0 and completion_status in {
        "complete_model_domain",
        "observed_domain_complete_not_statistically_demonstrated",
    }
    probability_bounds = coverage.get("probability_bounds") or {}
    if not isinstance(probability_bounds, Mapping):
        _refuse("penetration probability_bounds must be an object")
    component_bounds = probability_bounds.get("components", {})
    if not isinstance(component_bounds, Mapping):
        component_bounds = {}

    extracted: dict[str, Any] = {}
    probability_target_masks: list[bool] = []
    probability_interval_widths: list[float] = []
    for component_id, component in components.items():
        if not isinstance(component_id, str) or not component_id:
            _refuse("component identifiers must be non-empty strings")
        if not isinstance(component, Mapping):
            _refuse("component aggregate must be an object", component_id=component_id)
        fragment_metrics = component.get("fragment_metrics")
        if not isinstance(fragment_metrics, Mapping):
            _refuse(
                "component aggregate lacks fragment_metrics",
                component_id=component_id,
            )
        if "hit_indicator" not in fragment_metrics:
            _refuse(
                "component aggregate lacks the hit_indicator estimand",
                component_id=component_id,
            )
        p_hit = _validated_statistics(
            fragment_metrics["hit_indicator"],
            field=f"components.{component_id}.fragment_metrics.hit_indicator",
            probability=True,
            expected_sample_count=replicate_count,
        )
        bounds_for_component = component_bounds.get(component_id, {})
        if not isinstance(bounds_for_component, Mapping):
            bounds_for_component = {}
        metric_bounds = bounds_for_component.get("fragment_metrics", {})
        if not isinstance(metric_bounds, Mapping):
            metric_bounds = {}
        p_hit_interval = _identification_interval(
            p_hit["value"],
            metric_bounds.get("hit_indicator"),
            affected_fraction=(
                affected_fraction
                if metric_bounds.get("hit_indicator") is None
                and affected_fraction > 0.0
                else None
            ),
        )
        p_hit_epsilon_complete = (
            p_hit_interval["width"] <= epsilon + 1.0e-12
        )
        p_hit.update(
            {
                "mask": p_hit_epsilon_complete,
                "status": (
                    "available_epsilon_complete"
                    if p_hit_epsilon_complete
                    else "domain_censored_interval"
                ),
                "estimand": HIT_PROBABILITY_ESTIMAND,
                "binomial_observation": {
                    "success_count": int(round(p_hit["value"] * replicate_count)),
                    "trial_count": replicate_count,
                    "estimand_is_R_independent": True,
                },
                "identification_interval": p_hit_interval,
                "point_supervision_error_bound": p_hit_interval["width"],
            }
        )
        fragment_bounds = bounds_for_component.get("fragment", {})
        if not isinstance(fragment_bounds, Mapping):
            fragment_bounds = {}
        degraded = _state_target(
            component,
            state="degraded",
            replicate_count=replicate_count,
            domain_epsilon=epsilon,
            identification_bound=fragment_bounds.get("degraded"),
            affected_fraction=affected_fraction,
        )
        failed = _state_target(
            component,
            state="failed",
            replicate_count=replicate_count,
            domain_epsilon=epsilon,
            identification_bound=fragment_bounds.get("failed"),
            affected_fraction=affected_fraction,
        )
        if (
            degraded["value"] is not None
            and failed["value"] is not None
            and failed["value"] > degraded["value"] + 1.0e-12
        ):
            _refuse(
                "fragment failed probability exceeds degraded probability",
                component_id=component_id,
                degraded=degraded["value"],
                failed=failed["value"],
            )
        component_domain_affected = max(
            float(
                (fragment_bounds.get(state) or {}).get(
                    "affected_replicate_fraction",
                    affected_fraction,
                )
            )
            for state in ("degraded", "failed")
        )
        conditional = {
            public_name: _conditional_target(
                fragment_metrics,
                public_name=public_name,
                source_name=source_name,
                expected_estimand=estimand,
                domain_point_identified=component_domain_affected == 0.0,
            )
            for public_name, (source_name, estimand) in CONDITIONAL_METRICS.items()
        }
        extracted[component_id] = {
            "p_hit": p_hit,
            "state_probabilities": {
                "p_degraded": degraded,
                "p_failed": failed,
            },
            "conditional_positive_targets": conditional,
            "derived_support": _derived_support(
                p_hit,
                threshold=threshold,
                point_supervision_admissible=p_hit_epsilon_complete,
            ),
        }
        for probability_target in (p_hit, degraded, failed):
            interval = probability_target["identification_interval"]
            probability_target_masks.append(bool(probability_target["mask"]))
            probability_interval_widths.append(float(interval["width"]))

    epsilon_complete = bool(probability_target_masks) and all(
        probability_target_masks
    )
    point_supervision_fraction = (
        sum(probability_target_masks) / len(probability_target_masks)
        if probability_target_masks
        else 0.0
    )
    if epsilon_complete:
        censoring_supervision = "point_and_identification_interval"
    elif any(probability_target_masks):
        censoring_supervision = "partial_point_and_identification_interval"
    else:
        censoring_supervision = "identification_interval_only"

    return {
        "schema_version": HURDLE_TARGET_SCHEMA_VERSION,
        "target_spec_id": HURDLE_TARGET_SPEC_ID,
        "source": {
            "case_id": str(result.get("case_id", "")),
            "target_id": str(result.get("target_id", "")),
            "replicate_count": replicate_count,
            "scientific_completion_status": completion_status,
            "penetration_affected_replicate_fraction": affected_fraction,
            "response_admissible": epsilon_complete,
            "exact_domain_point_identified": domain_point_identified,
            "domain_epsilon": epsilon,
            "epsilon_complete": epsilon_complete,
            "probability_target_point_supervision_fraction": (
                point_supervision_fraction
            ),
            "maximum_probability_identification_width": (
                max(probability_interval_widths)
                if probability_interval_widths
                else None
            ),
            "censoring_supervision": censoring_supervision,
        },
        "estimands": {
            "p_hit": HIT_PROBABILITY_ESTIMAND,
            "state_probabilities": "per_replicate_fragment_state_probability_mean",
            "conditional_positive": (
                "explicit_hit_conditioned_log1p_means_only; unavailable when the "
                "engine aggregate lacks sufficient conditioned statistics"
            ),
        },
        "derived_support_spec": {
            "threshold": threshold,
            "decision_rule": SUPPORT_DECISION_RULE,
            "uncertainty_source": "p_hit_fixed_sample_ci95",
            "role": "derived_view_not_primary_label",
        },
        "uncertainty_separation": {
            "monte_carlo": "sample_stddev_standard_error_and_ci95",
            "domain_censoring": "identification_interval",
            "must_not_be_combined_as_one_confidence_interval": True,
        },
        "components": extracted,
    }
