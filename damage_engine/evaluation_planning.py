"""Outcome-independent planning helpers for reference evaluation sets.

The functions in this module size a *future* frozen reference set from a
pre-registered pilot prevalence.  They do not inspect candidate labels and do
not claim that an IID binomial approximation remains exact under grouping or
unequal design weights.
"""

from __future__ import annotations

from math import ceil, isfinite
from statistics import NormalDist
from typing import Iterable

from .errors import InputValidationError


def _probability(value: float, name: str, *, closed_one: bool = False) -> float:
    number = float(value)
    upper_ok = number <= 1.0 if closed_one else number < 1.0
    if not isfinite(number) or number <= 0.0 or not upper_ok:
        bound = "(0, 1]" if closed_one else "(0, 1)"
        raise InputValidationError(f"{name} must lie in {bound}")
    return number


def _positive_values(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not isfinite(value) or value <= 0.0 for value in result):
        raise InputValidationError(f"{name} must contain finite positive values")
    return result


def plan_reference_evaluation(
    prevalence: float,
    *,
    target_event_counts: Iterable[int] = (50, 100),
    absolute_half_widths: Iterable[float] = (0.01, 0.005),
    confidence: float = 0.95,
    effective_sample_fraction: float = 1.0,
) -> dict:
    """Return transparent IID and design-adjusted reference-test budgets.

    ``effective_sample_fraction`` is a conservative planning factor such as an
    anticipated ESS/N or independent-group fraction.  A value below one
    inflates the IID approximation; it is never inferred from outcomes here.
    """

    p = _probability(prevalence, "prevalence")
    conf = _probability(confidence, "confidence")
    eff = _probability(
        effective_sample_fraction, "effective_sample_fraction", closed_one=True
    )
    counts = tuple(int(value) for value in target_event_counts)
    if not counts or any(value < 1 for value in counts):
        raise InputValidationError("target_event_counts must contain positive integers")
    widths = _positive_values(absolute_half_widths, "absolute_half_widths")
    if any(value >= 1.0 for value in widths):
        raise InputValidationError("absolute_half_widths must be smaller than one")

    z_value = NormalDist().inv_cdf(0.5 + conf / 2.0)

    event_rows = []
    for event_count in counts:
        iid = ceil(event_count / p)
        event_rows.append(
            {
                "target_expected_events": event_count,
                "iid_case_count": iid,
                "design_adjusted_case_count": ceil(iid / eff),
            }
        )

    precision_rows = []
    for half_width in widths:
        iid = ceil((z_value * z_value * p * (1.0 - p)) / (half_width * half_width))
        precision_rows.append(
            {
                "target_absolute_half_width": half_width,
                "iid_case_count": iid,
                "design_adjusted_case_count": ceil(iid / eff),
            }
        )

    return {
        "schema_version": "1.0.0",
        "method": "two_sided_normal_binomial_planning_approximation",
        "prevalence": p,
        "confidence": conf,
        "z_value": z_value,
        "effective_sample_fraction": eff,
        "event_count_plans": event_rows,
        "precision_plans": precision_rows,
        "evidence_boundary": (
            "Planning approximation only. Freeze the reference distribution and "
            "split before labels; grouped dependence, unequal weights, prevalence "
            "drift, and model-comparison uncertainty can require a larger set."
        ),
    }
def plan_reference_soft_event_mass(
    mean_soft_label: float,
    *,
    target_soft_event_masses: Iterable[float] = (50.0, 100.0),
    effective_sample_fraction: float = 1.0,
) -> dict:
    """Plan Reference cases needed for an expected sum of soft labels.

    This complements event-count planning when the archived target is a Monte
    Carlo probability rather than a hard event.  The pilot mean must be frozen
    before the new Reference set is generated; no candidate outcome is read by
    this calculation.
    """

    mean = _probability(mean_soft_label, "mean_soft_label", closed_one=True)
    eff = _probability(
        effective_sample_fraction, "effective_sample_fraction", closed_one=True
    )
    targets = _positive_values(target_soft_event_masses, "target_soft_event_masses")
    rows = []
    for target in targets:
        iid = ceil(target / mean)
        rows.append(
            {
                "target_expected_soft_event_mass": target,
                "iid_case_count": iid,
                "design_adjusted_case_count": ceil(iid / eff),
            }
        )
    return {
        "schema_version": "1.0.0",
        "method": "expected_soft_event_mass_planning_approximation",
        "mean_soft_label": mean,
        "effective_sample_fraction": eff,
        "soft_event_mass_plans": rows,
        "evidence_boundary": (
            "Planning approximation only. It does not account for pilot uncertainty, "
            "group dependence, task subgroups, Monte Carlo label noise, or prevalence drift."
        ),
    }
