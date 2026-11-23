from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Iterable

from .errors import InputValidationError
from .utils import statistics_dict_from_moments


StatisticPath = tuple[str, ...]

_BOUNDED_UNIT_INTERVAL_STATISTICS = {
    "blast_failure_realized",
    "blast_failure_sampling_applied",
    "pre_pulse_shielding_survival",
    "same_pulse_failure_feedback_applied",
}


def is_probability_statistic_path(path: Iterable[str]) -> bool:
    """Return whether a public statistic is bounded to the unit interval.

    The classification is centralized here so indicator metrics cannot silently
    fall back to an unbounded Student-t interval merely because their names do
    not end in ``probability``.
    """

    key = tuple(str(part).lower() for part in path)
    if not key:
        return False
    name = key[-1]
    return (
        name.endswith("probability")
        or name.endswith("integrity")
        or name.endswith("_indicator")
        or name.endswith("_fraction")
        or name.endswith("_rate")
        or name == "penetration_affected_replicate"
        or name in _BOUNDED_UNIT_INTERVAL_STATISTICS
    )


@dataclass
class RunningMoments:
    """Numerically stable exact first/second moments for one scalar series."""

    count: int = 0
    mean: float = 0.0
    sum_squared_deviations: float = 0.0

    def observe(self, value: float) -> None:
        sample = float(value)
        if not math.isfinite(sample):
            raise InputValidationError(
                "online statistic samples must be finite",
                context={"value": value},
            )
        self.count += 1
        delta = sample - self.mean
        self.mean += delta / self.count
        self.sum_squared_deviations += delta * (sample - self.mean)

    def statistics(self, *, probability: bool = False) -> dict[str, Any]:
        sample_variance = (
            self.sum_squared_deviations / (self.count - 1)
            if self.count >= 2
            else None
        )
        return statistics_dict_from_moments(
            count=self.count,
            mean=self.mean if self.count else None,
            sample_variance=sample_variance,
            probability=probability,
        )


class OnlineStatisticsStore:
    """Stream nested replicate scalars without retaining replicate dictionaries.

    Paths form the entire interface.  The store preserves exactly the moments
    required by published mean/std/CI fields while avoiding O(R) object-graph
    growth when a raw replicate trace was not explicitly requested.
    """

    def __init__(self) -> None:
        self._series: dict[StatisticPath, RunningMoments] = {}
        self._children: dict[StatisticPath, tuple[str, ...]] = {}
        self.replicate_count = 0

    def observe_replicate(self, replicate: dict[str, Any]) -> None:
        expected_before = set(self._series)
        samples: dict[StatisticPath, float] = {}

        def walk(value: Any, path: StatisticPath) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    walk(child, (*path, str(key)))
                return
            if isinstance(value, bool):
                return
            if isinstance(value, Real):
                sample = float(value)
                if not math.isfinite(sample):
                    raise InputValidationError(
                        "replicate contains a non-finite statistic",
                        context={"path": list(path), "value": value},
                    )
                samples[path] = sample

        walk(replicate, ())
        observed = set(samples)
        if self.replicate_count > 0 and observed != expected_before:
            raise InputValidationError(
                "replicate scalar statistic paths changed during a run",
                context={
                    "missing": [
                        list(path) for path in sorted(expected_before - observed)
                    ],
                    "unexpected": [
                        list(path) for path in sorted(observed - expected_before)
                    ],
                },
            )
        # Commit only after the complete replicate has passed value and shape
        # validation. A rejected replicate therefore cannot partially update
        # moments or create a new series.
        if self.replicate_count == 0:
            self._series = {path: RunningMoments() for path in samples}
            children: dict[StatisticPath, set[str]] = {}
            for path in samples:
                for index, child in enumerate(path):
                    children.setdefault(path[:index], set()).add(child)
            self._children = {
                prefix: tuple(sorted(values))
                for prefix, values in children.items()
            }
        for path, sample in samples.items():
            self._series[path].observe(sample)
        self.replicate_count += 1

    def statistics(
        self,
        path: Iterable[str],
        *,
        probability: bool = False,
    ) -> dict[str, Any]:
        key = tuple(str(part) for part in path)
        try:
            moments = self._series[key]
        except KeyError as exc:
            raise InputValidationError(
                "requested online statistic path was not observed",
                context={"path": list(key)},
            ) from exc
        return moments.statistics(probability=probability)

    def conditioned_statistics(
        self,
        value_path: Iterable[str],
        condition_path: Iterable[str],
        *,
        estimand: str,
        conditioning: str,
    ) -> dict[str, Any]:
        """Aggregate ``value`` only over replicates selected by a binary path.

        The caller supplies zero for ``value`` when ``condition`` is zero.
        Welford moments retain the first and second raw sums, so the selected
        subset's mean and variance can be reconstructed without an O(R) trace.
        """

        value_key = tuple(str(part) for part in value_path)
        condition_key = tuple(str(part) for part in condition_path)
        try:
            value = self._series[value_key]
            condition = self._series[condition_key]
        except KeyError as exc:
            raise InputValidationError(
                "requested conditioned statistic path was not observed",
                context={
                    "value_path": list(value_key),
                    "condition_path": list(condition_key),
                },
            ) from exc
        if value.count != condition.count or value.count != self.replicate_count:
            raise InputValidationError(
                "conditioned statistic paths do not share the replicate count"
            )

        condition_sum = condition.mean * condition.count
        condition_sum_squares = (
            condition.sum_squared_deviations
            + condition.count * condition.mean * condition.mean
        )
        tolerance = 1.0e-10 * max(1.0, abs(condition_sum))
        positive_count = int(round(condition_sum))
        if (
            positive_count < 0
            or positive_count > condition.count
            or abs(condition_sum - positive_count) > tolerance
            or abs(condition_sum_squares - condition_sum) > tolerance
        ):
            raise InputValidationError(
                "conditioned statistic requires a binary indicator path",
                context={"condition_path": list(condition_key)},
            )

        value_sum = value.mean * value.count
        value_sum_squares = (
            value.sum_squared_deviations + value.count * value.mean * value.mean
        )
        if positive_count == 0:
            mean = sample_variance = None
        else:
            mean = value_sum / positive_count
            sample_variance = (
                max(0.0, value_sum_squares - value_sum * value_sum / positive_count)
                / (positive_count - 1)
                if positive_count >= 2
                else None
            )
        result = statistics_dict_from_moments(
            count=positive_count,
            mean=mean,
            sample_variance=sample_variance,
            probability=False,
        )
        result.update(
            {
                "estimand": str(estimand),
                "conditioning": str(conditioning),
                "random_level": "within_case_hit_positive_replicates",
            }
        )
        return result

    def child_keys(self, path: Iterable[str]) -> tuple[str, ...]:
        prefix = tuple(str(part) for part in path)
        return self._children.get(prefix, ())
