from __future__ import annotations

import numpy as np
import pytest

from damage_engine.errors import InputValidationError
from damage_engine.estimands import (
    StreamingPositiveWeightMoments,
    nonnegative_effective_sample_size,
    normalized_positive_weights,
    positive_effective_sample_size,
    positive_weighted_mean,
    restore_scaled_total,
    scale_positive_weights,
    validate_design_weight_contract,
)


def test_positive_ess_is_scale_invariant_at_float64_extremes():
    expected = 9.0 / 5.0
    assert positive_effective_sample_size(np.asarray([1.0e-300, 2.0e-300])) == pytest.approx(expected)
    assert positive_effective_sample_size(np.asarray([1.0e300, 2.0e300])) == pytest.approx(expected)


def test_nonnegative_ess_preserves_zero_mass_semantics():
    assert nonnegative_effective_sample_size(np.zeros(4)) == 0.0
    assert nonnegative_effective_sample_size(np.asarray([0.0, 1.0, 2.0])) == pytest.approx(9.0 / 5.0)
    with pytest.raises(InputValidationError, match="nonnegative"):
        nonnegative_effective_sample_size(np.asarray([1.0, -1.0]))


def test_scaled_total_reports_unrepresentable_raw_total():
    scaled, scale, total = scale_positive_weights(
        np.asarray([1.0e308, 1.0e308]), field="design weights"
    )
    assert scaled.tolist() == [1.0, 1.0]
    assert (scale, total) == (1.0e308, 2.0)
    assert restore_scaled_total(total, scale) == (
        None,
        "not_representable_float64_use_scale_and_scaled_total",
    )


def test_normalization_and_weighted_mean_are_finite_at_float64_limit():
    weights = np.asarray([1.0e308, 1.0e308])
    np.testing.assert_allclose(normalized_positive_weights(weights), [0.5, 0.5])
    assert positive_weighted_mean(np.asarray([0.0, 1.0]), weights) == pytest.approx(0.5)
    np.testing.assert_allclose(
        positive_weighted_mean(
            np.asarray([[1.0e308, -1.0e308], [1.0e308, 1.0e308]]),
            weights,
        ),
        [1.0e308, 0.0],
    )


def test_streaming_weight_moments_are_scale_invariant_across_chunks():
    moments = StreamingPositiveWeightMoments()
    moments.update(np.asarray([1.0e-300, 2.0e-300]))
    moments.update(np.asarray([1.0e300, 2.0e300]))
    expected = positive_effective_sample_size(
        np.asarray([1.0e-300, 2.0e-300, 1.0e300, 2.0e300])
    )
    assert moments.count == 4
    assert moments.effective_sample_size == pytest.approx(expected)
    assert moments.coefficient_of_variation is not None


def test_design_weight_mask_controls_eligibility_without_relabeling_zeros():
    weights, mask, probabilities = validate_design_weight_contract(
        np.asarray([2.0, np.nan]),
        np.asarray([True, False]),
        np.asarray([0.5, np.nan]),
    )
    assert weights[0] == 2.0
    assert mask.tolist() == [True, False]
    assert probabilities is not None and probabilities[0] == 0.5
