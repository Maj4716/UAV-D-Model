from __future__ import annotations

from dataclasses import dataclass
import math

from .errors import InputValidationError


@dataclass(frozen=True)
class ProbabilityConvergenceBound:
    half_width: float
    sample_count: int
    series_count: int
    maximum_looks: int
    familywise_alpha: float
    interval_method: str = "finite_horizon_simultaneous_hoeffding"
    coverage_status: str = (
        "familywise_optional_stopping_valid_for_bounded_probability_means"
    )

    def to_dict(self) -> dict:
        return {
            "half_width": self.half_width,
            "sample_count": self.sample_count,
            "series_count": self.series_count,
            "maximum_looks": self.maximum_looks,
            "familywise_alpha": self.familywise_alpha,
            "interval_method": self.interval_method,
            "coverage_status": self.coverage_status,
        }


def finite_horizon_probability_bound(
    *,
    sample_count: int,
    series_count: int,
    maximum_looks: int,
    familywise_alpha: float = 0.05,
) -> ProbabilityConvergenceBound:
    """A distribution-free simultaneous bound for adaptive probability means.

    Every monitored replicate value is required to lie in [0, 1].  Hoeffding's
    inequality is union-bounded over every probability series and every planned
    look up to ``maximum_looks``.  The resulting interval therefore remains
    valid when the run stops at the first look meeting its tolerance.  It is
    intentionally conservative; continuous unbounded physical metrics are not
    authorized to use this stopping rule.
    """

    if sample_count < 1:
        raise InputValidationError("probability convergence sample_count must be positive")
    if series_count < 1:
        raise InputValidationError("probability convergence series_count must be positive")
    if maximum_looks < sample_count:
        raise InputValidationError(
            "probability convergence maximum_looks must cover sample_count",
            context={
                "sample_count": sample_count,
                "maximum_looks": maximum_looks,
            },
        )
    if not math.isfinite(familywise_alpha) or not 0.0 < familywise_alpha < 1.0:
        raise InputValidationError(
            "probability convergence familywise_alpha must lie in (0,1)"
        )
    multiplicity = 2.0 * float(series_count) * float(maximum_looks)
    half_width = math.sqrt(
        math.log(multiplicity / familywise_alpha) / (2.0 * float(sample_count))
    )
    return ProbabilityConvergenceBound(
        half_width=float(min(1.0, half_width)),
        sample_count=int(sample_count),
        series_count=int(series_count),
        maximum_looks=int(maximum_looks),
        familywise_alpha=float(familywise_alpha),
    )
