from __future__ import annotations

import pytest

from damage_engine.convergence import finite_horizon_probability_bound
from damage_engine.errors import InputValidationError
from damage_engine.utils import statistics_dict


def test_finite_horizon_probability_bound_is_simultaneous_and_monotone():
    early = finite_horizon_probability_bound(
        sample_count=32,
        series_count=10,
        maximum_looks=128,
    )
    late = finite_horizon_probability_bound(
        sample_count=128,
        series_count=10,
        maximum_looks=128,
    )
    more_series = finite_horizon_probability_bound(
        sample_count=128,
        series_count=100,
        maximum_looks=128,
    )

    assert early.half_width > late.half_width
    assert more_series.half_width > late.half_width
    assert late.interval_method == "finite_horizon_simultaneous_hoeffding"
    assert late.coverage_status == (
        "familywise_optional_stopping_valid_for_bounded_probability_means"
    )
    assert late.familywise_alpha == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("sample_count", "series_count", "maximum_looks"),
    [(0, 1, 32), (1, 0, 32), (33, 1, 32)],
)
def test_finite_horizon_probability_bound_rejects_invalid_contract(
    sample_count, series_count, maximum_looks
):
    with pytest.raises(InputValidationError):
        finite_horizon_probability_bound(
            sample_count=sample_count,
            series_count=series_count,
            maximum_looks=maximum_looks,
        )


def test_all_zero_bounded_probability_samples_keep_a_nonzero_detection_limit():
    zero = statistics_dict([0.0, 0.0, 0.0, 0.0], probability=True)
    one = statistics_dict([1.0, 1.0, 1.0, 1.0], probability=True)

    assert zero["mean"] == 0.0
    assert zero["ci95_lower"] == 0.0
    assert 0.0 < zero["ci95_upper"] < 1.0
    assert one["mean"] == 1.0
    assert 0.0 < one["ci95_lower"] < 1.0
    assert one["ci95_upper"] == 1.0
    assert zero["interval_method"] == "fixed_sample_hoeffding_bounded_mean"
