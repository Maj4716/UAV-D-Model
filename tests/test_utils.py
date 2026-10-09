from __future__ import annotations

import numpy as np
import pytest

from damage_engine.errors import InputValidationError
from damage_engine.utils import confidence_interval


def test_small_sample_interval_uses_student_t():
    mean, stddev, lower, upper = confidence_interval(np.array([0.0, 1.0]))
    assert mean == pytest.approx(0.5)
    assert stddev == pytest.approx(2.0**-0.5)
    assert lower == pytest.approx(0.5 - 6.353, abs=1.0e-3)
    assert upper == pytest.approx(0.5 + 6.353, abs=1.0e-3)


def test_probability_interval_is_clipped_after_student_t():
    _, _, lower, upper = confidence_interval(
        np.array([0.0, 1.0]), bounds=(0.0, 1.0)
    )
    assert lower == 0.0
    assert upper == 1.0


def test_confidence_interval_rejects_non_finite_values():
    with pytest.raises(InputValidationError, match="must be finite"):
        confidence_interval(np.array([0.0, np.nan]))
