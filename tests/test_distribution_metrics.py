import numpy as np
import pytest

from damage_engine.distribution_metrics import (
    hill_tail_stability,
    jensen_shannon_binned,
    rbf_mmd2,
    wasserstein1_log_positive,
    weighted_brier,
    weighted_calibration,
)


def test_identical_distributions_have_zero_distances():
    values = np.asarray([0.0, 1.0, 2.0, 4.0, 8.0])
    assert wasserstein1_log_positive(values, values) == pytest.approx(0.0)
    assert rbf_mmd2(values, values)["mmd2"] == pytest.approx(0.0)
    assert jensen_shannon_binned(values, values, bin_edges=[-1, 0.5, 2.5, 10]) == pytest.approx(0.0)


def test_weighted_probability_scores_and_hill_stability():
    probability = [0.1, 0.9, 0.8, 0.2]
    outcome = [0.0, 1.0, 1.0, 0.0]
    weight = [1.0, 2.0, 2.0, 1.0]
    assert weighted_brier(probability, outcome, weight) == pytest.approx(0.025)
    report = weighted_calibration(probability, outcome, weight=weight, bin_count=4)
    assert report["brier"] == pytest.approx(0.025)
    tail = hill_tail_stability(np.arange(1.0, 101.0), k_values=[5, 10, 20])
    assert len(tail["estimates"]) == 3
    assert tail["alpha_median"] > 0.0
