"""Design-aware inference for contact-conditioned sampling comparisons.

The functions in this module operate on already-observed software-teacher
rows.  They keep selected-sample summaries separate from finite-frame Hajek
estimates, treat domain-censored candidates as zero useful-label yield, and
couple common candidate clusters across sampling arms during bootstrap.

This module does not implement candidate selection or claim exact pivotal
design variance.  The multiplier bootstrap estimates empirical candidate-frame
uncertainty while preserving covariance from candidates shared by two arms.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .errors import InputValidationError


INFERENCE_SCHEMA_VERSION = "sphere-entry-sampling-inference@1.0.0"
BOOTSTRAP_METHOD = "coupled_stratified_exponential_cluster_multiplier"
_METRIC_NAMES = (
    "complete_non_saturated_yield",
    "domain_complete_fraction",
    "tree_saturation_given_complete",
    "positive_component_count_given_complete",
)


@dataclass(frozen=True)
class SamplingObservation:
    """One selected candidate observation from one sampling arm.

    ``physical_ray_id`` identifies the longitudinally invariant geometry.
    ``candidate_group_id`` identifies the complete approach/profile candidate
    shared across arms.  The distinction allows one physical ray to be studied
    under multiple munition profiles without collapsing those observations.
    ``design_weight_raw`` must equal ``1 / inclusion_probability`` for this
    finite-frame comparison seam.
    """

    arm: str
    physical_ray_id: str
    candidate_group_id: str
    design_seed: int
    stratum: str
    inclusion_probability: float
    design_weight_raw: float
    domain_complete: bool
    tree_saturated: bool | None
    positive_component_count: float | None = None
    elapsed_seconds: float | None = None
    proxy_domain_complete: bool = True


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise InputValidationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise InputValidationError(f"{name} must be a finite number")
    return result


def _validate_observation(row: SamplingObservation) -> None:
    if not isinstance(row, SamplingObservation):
        raise InputValidationError(
            "sampling rows must be SamplingObservation instances"
        )
    if not isinstance(row.arm, str) or not row.arm.strip():
        raise InputValidationError("arm must be a nonempty string")
    if not isinstance(row.physical_ray_id, str) or not row.physical_ray_id.strip():
        raise InputValidationError("physical_ray_id must be a nonempty string")
    if (
        not isinstance(row.candidate_group_id, str)
        or not row.candidate_group_id.strip()
    ):
        raise InputValidationError("candidate_group_id must be a nonempty string")
    if not isinstance(row.stratum, str) or not row.stratum.strip():
        raise InputValidationError("stratum must be a nonempty string")
    if (
        not isinstance(row.design_seed, (int, np.integer))
        or isinstance(row.design_seed, (bool, np.bool_))
        or int(row.design_seed) < 0
    ):
        raise InputValidationError("design_seed must be a nonnegative integer")
    rho = _finite_float(
        row.inclusion_probability, name="inclusion_probability"
    )
    if not 0.0 < rho <= 1.0:
        raise InputValidationError("inclusion_probability must lie in (0, 1]")
    weight = _finite_float(row.design_weight_raw, name="design_weight_raw")
    if weight <= 0.0:
        raise InputValidationError("design_weight_raw must be positive")
    expected = 1.0 / rho
    if not math.isclose(weight, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise InputValidationError(
            "design_weight_raw must equal the reciprocal of inclusion_probability"
        )
    if not isinstance(row.domain_complete, (bool, np.bool_)):
        raise InputValidationError("domain_complete must be boolean")
    if not isinstance(row.proxy_domain_complete, (bool, np.bool_)):
        raise InputValidationError("proxy_domain_complete must be boolean")
    if row.domain_complete and not isinstance(
        row.tree_saturated, (bool, np.bool_)
    ):
        raise InputValidationError(
            "tree_saturated must be boolean for complete-domain observations"
        )
    if row.tree_saturated is not None and not isinstance(
        row.tree_saturated, (bool, np.bool_)
    ):
        raise InputValidationError("tree_saturated must be boolean or null")
    if row.positive_component_count is not None:
        positive_count = _finite_float(
            row.positive_component_count, name="positive_component_count"
        )
        if positive_count < 0.0:
            raise InputValidationError(
                "positive_component_count must be nonnegative"
            )
    if row.elapsed_seconds is not None:
        elapsed = _finite_float(row.elapsed_seconds, name="elapsed_seconds")
        if elapsed < 0.0:
            raise InputValidationError("elapsed_seconds must be nonnegative")


def _validate_rows(
    rows: Sequence[SamplingObservation], *, expected_arm: str | None = None
) -> tuple[SamplingObservation, ...]:
    values = tuple(rows)
    if not values:
        raise InputValidationError("sampling arm must contain at least one row")
    seen: set[tuple[int, str]] = set()
    observed_arm = values[0].arm
    for row in values:
        _validate_observation(row)
        if row.arm != observed_arm:
            raise InputValidationError("one sampling summary cannot mix arm names")
        if expected_arm is not None and row.arm != expected_arm:
            raise InputValidationError(
                "sampling observation arm disagrees with its arm mapping key"
            )
        key = (int(row.design_seed), row.candidate_group_id)
        if key in seen:
            raise InputValidationError(
                "duplicate design_seed and candidate_group_id within one arm"
            )
        seen.add(key)
    return values


def _weighted_metrics(
    rows: Sequence[SamplingObservation],
    *,
    use_design_weights: bool,
    multipliers: Mapping[tuple[int, str], float] | None = None,
) -> dict[str, float | None]:
    base = np.asarray(
        [row.design_weight_raw if use_design_weights else 1.0 for row in rows],
        dtype=np.float64,
    )
    if multipliers is not None:
        base = base * np.asarray(
            [
                multipliers[(int(row.design_seed), row.candidate_group_id)]
                for row in rows
            ],
            dtype=np.float64,
        )
    total = float(np.sum(base))
    if not math.isfinite(total) or total <= 0.0:
        raise InputValidationError("weighted sampling mass must be finite and positive")

    domain = np.asarray([bool(row.domain_complete) for row in rows], dtype=np.float64)
    saturated = np.asarray(
        [
            float(bool(row.tree_saturated)) if row.domain_complete else 0.0
            for row in rows
        ],
        dtype=np.float64,
    )
    useful = domain * (1.0 - saturated)
    domain_mass = float(np.sum(base * domain))
    positive_available = np.asarray(
        [
            bool(row.domain_complete and row.positive_component_count is not None)
            for row in rows
        ],
        dtype=bool,
    )
    positive_mass = float(np.sum(base[positive_available]))
    positive_mean = None
    if positive_mass > 0.0:
        positive_values = np.asarray(
            [
                0.0
                if row.positive_component_count is None
                else float(row.positive_component_count)
                for row in rows
            ],
            dtype=np.float64,
        )
        positive_mean = float(
            np.sum(base[positive_available] * positive_values[positive_available])
            / positive_mass
        )

    return {
        "complete_non_saturated_yield": float(np.sum(base * useful) / total),
        "domain_complete_fraction": float(domain_mass / total),
        "tree_saturation_given_complete": (
            None
            if domain_mass <= 0.0
            else float(np.sum(base * domain * saturated) / domain_mass)
        ),
        "positive_component_count_given_complete": positive_mean,
    }


def sampling_metric_summary(
    rows: Sequence[SamplingObservation],
) -> dict[str, Any]:
    """Summarize one arm without conflating selected and frame estimands."""

    values = _validate_rows(rows)
    weights = np.asarray(
        [row.design_weight_raw for row in values], dtype=np.float64
    )
    weight_sum = float(np.sum(weights))
    ess = weight_sum * weight_sum / float(np.sum(np.square(weights)))
    mean_weight = float(np.mean(weights))
    return {
        "arm": values[0].arm,
        "row_count": len(values),
        "strata": sorted({row.stratum for row in values}),
        "candidate_cluster_count": len(
            {(int(row.design_seed), row.candidate_group_id) for row in values}
        ),
        "unique_physical_ray_count": len(
            {(int(row.design_seed), row.physical_ray_id) for row in values}
        ),
        "raw_selected": _weighted_metrics(
            values, use_design_weights=False
        ),
        "hajek_frame": _weighted_metrics(values, use_design_weights=True),
        "design_weight_quality": {
            "ess": float(ess),
            "ess_fraction": float(ess / len(values)),
            "weight_cv": float(np.std(weights, ddof=0) / mean_weight),
            "weight_min": float(np.min(weights)),
            "weight_max": float(np.max(weights)),
            "inclusion_probability_min": float(
                min(row.inclusion_probability for row in values)
            ),
        },
        "proxy_domain_support": {
            "complete_count": int(
                sum(bool(row.proxy_domain_complete) for row in values)
            ),
            "partial_count": int(
                sum(not bool(row.proxy_domain_complete) for row in values)
            ),
        },
    }


def coupled_stratified_multiplier_bootstrap(
    arms: Mapping[str, Sequence[SamplingObservation]],
    *,
    contrasts: Iterable[tuple[str, str]],
    draws: int,
    seed: int,
    retain_draws: bool = False,
) -> dict[str, Any]:
    """Compare arms with shared multipliers for shared physical-ray clusters.

    Each contrast is ``(candidate_arm, reference_arm)`` and is reported as
    candidate minus reference.  Independent exponential multipliers are drawn
    for clusters inside their fixed strata; a cluster present in two arms uses
    exactly the same multiplier in both arms.  Candidate groups, rather than
    bare physical rays, prevent different profiles on one ray from aliasing.
    """

    if not isinstance(arms, Mapping) or len(arms) < 2:
        raise InputValidationError("bootstrap requires at least two sampling arms")
    if (
        not isinstance(draws, (int, np.integer))
        or isinstance(draws, (bool, np.bool_))
        or int(draws) < 2
    ):
        raise InputValidationError("bootstrap draws must be an integer of at least two")
    if (
        not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
        or int(seed) < 0
    ):
        raise InputValidationError("bootstrap seed must be a nonnegative integer")

    validated: dict[str, tuple[SamplingObservation, ...]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    cluster_strata: dict[tuple[int, str], str] = {}
    for arm in sorted(arms):
        if not isinstance(arm, str) or not arm.strip():
            raise InputValidationError("arm mapping keys must be nonempty strings")
        values = _validate_rows(arms[arm], expected_arm=arm)
        validated[arm] = values
        summaries[arm] = sampling_metric_summary(values)
        for row in values:
            key = (int(row.design_seed), row.candidate_group_id)
            previous = cluster_strata.setdefault(key, row.stratum)
            if previous != row.stratum:
                raise InputValidationError(
                    "one candidate group cannot change stratum across arms"
                )

    contrast_pairs = tuple(contrasts)
    if not contrast_pairs:
        raise InputValidationError("at least one arm contrast is required")
    if len(set(contrast_pairs)) != len(contrast_pairs):
        raise InputValidationError("arm contrasts must be unique")
    for candidate_arm, reference_arm in contrast_pairs:
        if candidate_arm == reference_arm:
            raise InputValidationError("contrast arms must be different")
        if candidate_arm not in validated or reference_arm not in validated:
            raise InputValidationError("contrast references an unknown sampling arm")

    ordered_clusters = sorted(
        cluster_strata,
        key=lambda key: (cluster_strata[key], key[0], key[1]),
    )
    rng = np.random.default_rng(int(seed))
    contrast_draws: dict[tuple[str, str], dict[str, list[float]]] = {
        pair: {name: [] for name in _METRIC_NAMES} for pair in contrast_pairs
    }
    for _ in range(int(draws)):
        generated = rng.exponential(scale=1.0, size=len(ordered_clusters))
        multipliers: dict[tuple[int, str], float] = {}
        for stratum in sorted(set(cluster_strata.values())):
            indices = [
                index
                for index, key in enumerate(ordered_clusters)
                if cluster_strata[key] == stratum
            ]
            stratum_values = generated[indices]
            stratum_mean = float(np.mean(stratum_values))
            for index in indices:
                multipliers[ordered_clusters[index]] = float(
                    generated[index] / stratum_mean
                )
        metrics_by_arm = {
            arm: _weighted_metrics(
                rows,
                use_design_weights=True,
                multipliers=multipliers,
            )
            for arm, rows in validated.items()
        }
        for pair in contrast_pairs:
            candidate_arm, reference_arm = pair
            for name in _METRIC_NAMES:
                candidate_value = metrics_by_arm[candidate_arm][name]
                reference_value = metrics_by_arm[reference_arm][name]
                if candidate_value is None or reference_value is None:
                    continue
                contrast_draws[pair][name].append(
                    float(candidate_value - reference_value)
                )

    contrast_reports: dict[str, Any] = {}
    for candidate_arm, reference_arm in contrast_pairs:
        identifier = f"{candidate_arm}_minus_{reference_arm}"
        metric_reports: dict[str, Any] = {}
        for name in _METRIC_NAMES:
            candidate_point = summaries[candidate_arm]["hajek_frame"][name]
            reference_point = summaries[reference_arm]["hajek_frame"][name]
            values = contrast_draws[(candidate_arm, reference_arm)][name]
            if candidate_point is None or reference_point is None or not values:
                metric_reports[name] = {
                    "status": "not_estimable",
                    "hajek_difference": None,
                    "percentile_95_interval": None,
                    "draw_count": len(values),
                }
                continue
            array = np.asarray(values, dtype=np.float64)
            record: dict[str, Any] = {
                "status": "estimable",
                "hajek_difference": float(candidate_point - reference_point),
                "percentile_95_interval": [
                    float(np.quantile(array, 0.025)),
                    float(np.quantile(array, 0.975)),
                ],
                "draw_count": len(values),
            }
            if retain_draws:
                record["draw_values"] = [float(value) for value in values]
            metric_reports[name] = record
        contrast_reports[identifier] = {
            "candidate_arm": candidate_arm,
            "reference_arm": reference_arm,
            "metrics": metric_reports,
        }

    return {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "stage": "sampling_arm_inference",
        "arms": summaries,
        "contrasts": contrast_reports,
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "draws": int(draws),
            "seed": int(seed),
            "cluster_fields": ["design_seed", "candidate_group_id"],
            "geometry_identity_field": "physical_ray_id",
            "stratum_field": "stratum",
            "within_stratum_multiplier_mean_normalized": True,
            "shared_cluster_multiplier_across_arms": True,
            "variance_scope": (
                "empirical_candidate_frame_not_exact_pivotal_design_variance"
            ),
        },
    }


def _require_mapping_fields(
    value: Mapping[str, Any], fields: Iterable[str], *, name: str
) -> None:
    if not isinstance(value, Mapping):
        raise InputValidationError(f"{name} must be an object")
    missing = sorted(set(fields) - set(value))
    if missing:
        raise InputValidationError(
            f"{name} is missing required fields", context={"missing": missing}
        )


def evaluate_coverage_admission(
    report: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the Coverage admission gate from a frozen inference report."""

    required = {
        "candidate_arm",
        "reference_arm",
        "primary_metric",
        "primary_lower_bound_min",
        "domain_noninferiority_margin",
        "minimum_ess_fraction",
        "maximum_weight_cv",
        "minimum_inclusion_probability",
        "minimum_arm_count",
        "required_strata",
        "s_contact_all_pass",
        "key_component_guardrails_pass",
        "cost_guardrail_pass",
    }
    _require_mapping_fields(contract, required, name="coverage contract")
    _require_mapping_fields(report, {"arms", "contrasts"}, name="inference report")
    candidate_arm = str(contract["candidate_arm"])
    reference_arm = str(contract["reference_arm"])
    arms = report["arms"]
    if candidate_arm not in arms or reference_arm not in arms:
        raise InputValidationError("coverage contract references an unknown arm")
    contrast_id = f"{candidate_arm}_minus_{reference_arm}"
    contrasts = report["contrasts"]
    if contrast_id not in contrasts:
        raise InputValidationError("inference report lacks the required contrast")
    metrics = contrasts[contrast_id]["metrics"]
    primary_name = str(contract["primary_metric"])
    if primary_name not in metrics:
        raise InputValidationError("coverage primary metric is absent from report")
    primary_interval = metrics[primary_name].get("percentile_95_interval")
    domain_interval = metrics["domain_complete_fraction"].get(
        "percentile_95_interval"
    )
    if primary_interval is None or domain_interval is None:
        raise InputValidationError("coverage gate metrics are not estimable")

    primary_floor = _finite_float(
        contract["primary_lower_bound_min"], name="primary_lower_bound_min"
    )
    domain_margin = _finite_float(
        contract["domain_noninferiority_margin"],
        name="domain_noninferiority_margin",
    )
    minimum_ess = _finite_float(
        contract["minimum_ess_fraction"], name="minimum_ess_fraction"
    )
    maximum_cv = _finite_float(
        contract["maximum_weight_cv"], name="maximum_weight_cv"
    )
    minimum_rho = _finite_float(
        contract["minimum_inclusion_probability"],
        name="minimum_inclusion_probability",
    )
    minimum_count = contract["minimum_arm_count"]
    if (
        not isinstance(minimum_count, (int, np.integer))
        or isinstance(minimum_count, (bool, np.bool_))
        or int(minimum_count) < 1
    ):
        raise InputValidationError("minimum_arm_count must be a positive integer")
    required_strata_raw = contract["required_strata"]
    if not isinstance(required_strata_raw, Sequence) or isinstance(
        required_strata_raw, (str, bytes)
    ):
        raise InputValidationError("required_strata must be an array of names")
    required_strata = {str(value) for value in required_strata_raw}
    if not required_strata or any(not value for value in required_strata):
        raise InputValidationError("required_strata must contain nonempty names")

    qualities = [
        arms[name]["design_weight_quality"]
        for name in (candidate_arm, reference_arm)
    ]
    checks = {
        "primary_interval_superior": float(primary_interval[0]) > primary_floor,
        "domain_noninferior": float(domain_interval[0]) > -domain_margin,
        "minimum_arm_count_met": all(
            int(arms[name]["row_count"]) >= int(minimum_count)
            for name in (candidate_arm, reference_arm)
        ),
        "required_strata_present": all(
            required_strata.issubset(set(arms[name]["strata"]))
            for name in (candidate_arm, reference_arm)
        ),
        "ess_fraction_adequate": all(
            float(quality["ess_fraction"]) >= minimum_ess
            for quality in qualities
        ),
        "weight_cv_adequate": all(
            float(quality["weight_cv"]) <= maximum_cv
            for quality in qualities
        ),
        "inclusion_probability_adequate": all(
            float(quality["inclusion_probability_min"]) >= minimum_rho
            for quality in qualities
        ),
        "s_contact_all_pass": contract["s_contact_all_pass"] is True,
        "key_component_guardrails_pass": (
            contract["key_component_guardrails_pass"] is True
        ),
        "cost_guardrail_pass": contract["cost_guardrail_pass"] is True,
    }
    reason_codes = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "gate": "coverage_admission",
        "candidate_arm": candidate_arm,
        "reference_arm": reference_arm,
        "checks": checks,
        "overall_pass": all(checks.values()),
        "reason_codes": reason_codes,
    }


def evaluate_reference_replacement(
    coverage_admission: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the stricter Reference-replacement gate.

    Proxy-partial candidates in the target frame must all have positive audit
    inclusion probability.  A zero-support audit cannot be repaired by
    weighting and therefore forces this gate to fail.
    """

    _require_mapping_fields(
        coverage_admission,
        {"overall_pass"},
        name="coverage admission",
    )
    required = {
        "target_measure_hash_match",
        "estimand_hash_match",
        "sealed_test_unused_for_tuning",
        "physical_domain_not_shrunk",
        "independent_frame_count",
        "minimum_independent_frames",
        "support_audit",
    }
    _require_mapping_fields(contract, required, name="replacement contract")
    support = contract["support_audit"]
    _require_mapping_fields(
        support,
        {
            "proxy_partial_candidate_count",
            "proxy_partial_positive_inclusion_count",
            "minimum_inclusion_probability",
            "outcome_blind",
        },
        name="support audit",
    )
    total = support["proxy_partial_candidate_count"]
    positive = support["proxy_partial_positive_inclusion_count"]
    for value, name in (
        (total, "proxy_partial_candidate_count"),
        (positive, "proxy_partial_positive_inclusion_count"),
    ):
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or int(value) < 0
        ):
            raise InputValidationError(f"{name} must be a nonnegative integer")
    if int(positive) > int(total):
        raise InputValidationError(
            "proxy_partial_positive_inclusion_count cannot exceed candidate count"
        )
    frame_count = contract["independent_frame_count"]
    minimum_frames = contract["minimum_independent_frames"]
    for value, name in (
        (frame_count, "independent_frame_count"),
        (minimum_frames, "minimum_independent_frames"),
    ):
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or int(value) < 1
        ):
            raise InputValidationError(f"{name} must be a positive integer")
    audit_rho = _finite_float(
        support["minimum_inclusion_probability"],
        name="support_audit.minimum_inclusion_probability",
    )
    positive_support = int(total) == 0 or (
        int(positive) == int(total) and 0.0 < audit_rho <= 1.0
    )
    checks = {
        "coverage_admission_pass": coverage_admission["overall_pass"] is True,
        "target_measure_hash_match": contract["target_measure_hash_match"] is True,
        "estimand_hash_match": contract["estimand_hash_match"] is True,
        "sealed_test_unused_for_tuning": (
            contract["sealed_test_unused_for_tuning"] is True
        ),
        "physical_domain_not_shrunk": (
            contract["physical_domain_not_shrunk"] is True
        ),
        "independent_frame_count_met": int(frame_count) >= int(minimum_frames),
        "support_audit_outcome_blind": support["outcome_blind"] is True,
        "proxy_partial_positive_support": positive_support,
    }
    reason_codes = [name for name, passed in checks.items() if not passed]
    if int(total) > 0 and int(positive) == 0:
        reason_codes.append("zero_support_for_proxy_partial_candidates")
    elif not positive_support:
        reason_codes.append("incomplete_support_for_proxy_partial_candidates")
    return {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "gate": "reference_replacement",
        "checks": checks,
        "overall_pass": all(checks.values()),
        "reason_codes": reason_codes,
    }


__all__ = [
    "BOOTSTRAP_METHOD",
    "INFERENCE_SCHEMA_VERSION",
    "SamplingObservation",
    "coupled_stratified_multiplier_bootstrap",
    "evaluate_coverage_admission",
    "evaluate_reference_replacement",
    "sampling_metric_summary",
]
