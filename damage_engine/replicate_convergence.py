"""Fixed-prefix replicate convergence diagnostics.

This module intentionally evaluates a fixed prefix schedule (for example
R=4/8/16/32/64/128).  It does not implement outcome-adaptive stopping, whose
coverage properties would require a separate validated sequential procedure.
Supplying a schedule to this function does not establish that it was externally
pre-registered before earlier outcomes were observed.
"""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Iterable

import numpy as np

from .errors import InputValidationError


# Two-sided 95% Student-t critical values for df 1..31.  Larger fixed prefixes
# use the standard Cornish-Fisher expansion below.  Keeping the exact
# small-sample table avoids the poor low-df approximation while allowing a
# fixed-prefix study to extend through R=64/128 (or a larger available
# prefix) without a hard-coded ceiling.
_T95 = (
    np.nan,
    12.706205, 4.302653, 3.182446, 2.776445, 2.570582, 2.446912,
    2.364624, 2.306004, 2.262157, 2.228139, 2.200985, 2.178813,
    2.160369, 2.144787, 2.131450, 2.119905, 2.109816, 2.100922,
    2.093024, 2.085963, 2.079614, 2.073873, 2.068658, 2.063899,
    2.059539, 2.055529, 2.051831, 2.048407, 2.045230, 2.042272,
    2.039513,
)


def _student_t_critical_95(degrees_of_freedom: int) -> float:
    """Return the two-sided 95% Student-t critical value.

    The third-order Cornish-Fisher expansion is used only above df=31, where
    it is already very accurate (about 1e-7 at df=63).  This project does not
    depend on SciPy, so importing a statistical package just to remove the old
    R=32 ceiling would make the core convergence audit less portable.
    """

    if degrees_of_freedom < 1:
        raise InputValidationError("Student-t degrees of freedom must be positive")
    if degrees_of_freedom < len(_T95):
        return float(_T95[degrees_of_freedom])
    z_value = NormalDist().inv_cdf(0.975)
    df = float(degrees_of_freedom)
    return float(
        z_value
        + (z_value**3 + z_value) / (4.0 * df)
        + (5.0 * z_value**5 + 16.0 * z_value**3 + 3.0 * z_value)
        / (96.0 * df**2)
        + (
            3.0 * z_value**7
            + 19.0 * z_value**5
            + 17.0 * z_value**3
            - 15.0 * z_value
        )
        / (384.0 * df**3)
    )


def _per_output_values(
    value: float | Iterable[float],
    *,
    output_count: int,
    name: str,
    lower: float,
    upper: float,
    lower_inclusive: bool,
) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 0 or raw.size == 1:
        raw = np.repeat(float(raw.reshape(-1)[0]), output_count)
    if raw.shape != (output_count,) or not np.all(np.isfinite(raw)):
        raise InputValidationError(
            f"{name} must be a finite scalar or one value per output"
        )
    lower_ok = raw >= lower if lower_inclusive else raw > lower
    if not np.all(lower_ok & (raw <= upper)):
        left = "[" if lower_inclusive else "("
        raise InputValidationError(f"{name} must lie in {left}{lower}, {upper}]")
    return raw


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling."""

    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranked[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranked


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2:
        return None
    a, b = _rank(left), _rank(right)
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator == 0.0 else float(np.dot(a, b) / denominator)


def replicate_convergence_report(
    traces,
    *,
    levels: Iterable[int] = (4, 8, 16, 32, 64, 128),
    ci_half_width_tolerance: float | Iterable[float] | None = None,
    decision_threshold: float | Iterable[float] = 0.5,
    threshold_neighborhood_half_width: float | Iterable[float] = 0.1,
) -> dict:
    """Summarize nested fixed prefixes for ``[case, replicate, output]`` data."""

    values = np.asarray(traces, dtype=np.float64)
    if values.ndim == 2:
        values = values[..., None]
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 2 or values.shape[2] < 1:
        raise InputValidationError(
            "replicate traces must have shape [case, replicate, output]"
        )
    if not np.all(np.isfinite(values)):
        raise InputValidationError("replicate traces must be finite")
    prefixes = tuple(int(level) for level in levels)
    if (
        not prefixes
        or any(level < 2 or level > values.shape[1] for level in prefixes)
        or tuple(sorted(set(prefixes))) != prefixes
    ):
        raise InputValidationError(
            "levels must be unique increasing prefixes of at least two and available in traces"
        )

    if ci_half_width_tolerance is None:
        tolerances = None
    else:
        raw = np.asarray(ci_half_width_tolerance, dtype=np.float64)
        if raw.ndim == 0:
            raw = np.repeat(raw, values.shape[2])
        if raw.shape != (values.shape[2],) or not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
            raise InputValidationError(
                "ci_half_width_tolerance must be positive scalar or one value per output"
            )
        tolerances = raw

    thresholds = _per_output_values(
        decision_threshold,
        output_count=values.shape[2],
        name="decision_threshold",
        lower=0.0,
        upper=1.0,
        lower_inclusive=True,
    )
    neighborhood_half_widths = _per_output_values(
        threshold_neighborhood_half_width,
        output_count=values.shape[2],
        name="threshold_neighborhood_half_width",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
    )

    final_means = values[:, : prefixes[-1], :].mean(axis=1)
    final_decisions = final_means >= thresholds[None, :]
    threshold_neighborhood = (
        np.abs(final_means - thresholds[None, :])
        <= neighborhood_half_widths[None, :]
    )
    threshold_neighborhood_counts = np.sum(threshold_neighborhood, axis=0)
    rows = []
    previous_means = None
    for level in prefixes:
        prefix = values[:, :level, :]
        means = prefix.mean(axis=1)
        stddev = prefix.std(axis=1, ddof=1)
        critical_value = _student_t_critical_95(level - 1)
        half_width = critical_value * stddev / sqrt(level)
        absolute_error = np.abs(means - final_means)
        decisions = means >= thresholds[None, :]
        flips = decisions != final_decisions
        neighborhood_flip_rates = []
        for output_index, count in enumerate(threshold_neighborhood_counts):
            neighborhood_flip_rates.append(
                None
                if int(count) == 0
                else float(
                    np.mean(flips[threshold_neighborhood[:, output_index], output_index])
                )
            )
        rank_stability = [
            _rank_correlation(means[:, index], final_means[:, index])
            for index in range(values.shape[2])
        ]
        row = {
            "replicates": level,
            "case_count": int(values.shape[0]),
            "output_count": int(values.shape[2]),
            "ci_method": "two_sided_student_t_95_fixed_prefix",
            "ci_critical_value_method": (
                "exact_table_for_df_1_through_31_else_"
                "third_order_cornish_fisher_approximation"
            ),
            "ci_critical_value": critical_value,
            "ci_half_width_median_by_output": np.median(half_width, axis=0).tolist(),
            "ci_half_width_p95_by_output": np.quantile(half_width, 0.95, axis=0).tolist(),
            "ci_half_width_max_by_output": np.max(half_width, axis=0).tolist(),
            "absolute_error_to_final_p95_by_output": np.quantile(
                absolute_error, 0.95, axis=0
            ).tolist(),
            "absolute_error_to_final_p99_by_output": np.quantile(
                absolute_error, 0.99, axis=0
            ).tolist(),
            "threshold_label_flip_rate_by_output": np.mean(
                flips, axis=0
            ).tolist(),
            "threshold_neighborhood_case_count_by_output": (
                threshold_neighborhood_counts.astype(int).tolist()
            ),
            "threshold_neighborhood_flip_rate_by_output": neighborhood_flip_rates,
            "rank_correlation_to_final_by_output": rank_stability,
            "relative_compute_cost": level / prefixes[0],
        }
        if previous_means is not None:
            shift = np.abs(means - previous_means)
            row["mean_shift_from_previous_median_by_output"] = np.median(
                shift, axis=0
            ).tolist()
            row["mean_shift_from_previous_max_by_output"] = np.max(
                shift, axis=0
            ).tolist()
        if tolerances is not None:
            row["ci_tolerance_by_output"] = tolerances.tolist()
            row["fraction_exceeding_tolerance_by_output"] = np.mean(
                half_width > tolerances[None, :], axis=0
            ).tolist()
        rows.append(row)
        previous_means = means

    return {
        "schema_version": "1.1.0",
        "design": "nested_fixed_prefix_numerical_diagnostic",
        "external_preregistration_status": "not_established_by_this_report",
        "levels": list(prefixes),
        "reference_prefix_replicates": prefixes[-1],
        "comparison_definition": {
            "reference": (
                "casewise mean over the largest supplied fixed prefix; "
                "it is a numerical convergence reference, not physical truth"
            ),
            "absolute_error": (
                "abs(prefix_mean - reference_prefix_mean), summarized across "
                "cases independently for each output"
            ),
            "absolute_error_quantiles": [0.95, 0.99],
            "decision_rule": "prefix_mean >= decision_threshold",
            "decision_threshold_by_output": thresholds.tolist(),
            "threshold_neighborhood_rule": (
                "abs(reference_prefix_mean - decision_threshold) <= "
                "threshold_neighborhood_half_width"
            ),
            "threshold_neighborhood_half_width_by_output": (
                neighborhood_half_widths.tolist()
            ),
            "threshold_flip": (
                "prefix decision differs from the decision at the reference prefix; "
                "the neighborhood rate is null when no reference case is near the threshold"
            ),
        },
        "results": rows,
        "evidence_boundary": (
            "This diagnoses conditional within-case replicate convergence only; "
            "it does not quantify epistemic or model-form uncertainty and does not "
            "authorize outcome-adaptive stopping. It does not establish external "
            "preregistration or the scientific adequacy of case selection."
        ),
    }
