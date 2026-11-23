import numpy as np
import pytest

from damage_engine.mechanism_diagnostics import (
    component_funnel,
    log_tail_regression,
    paired_tost_equivalence,
    tree_distribution_metrics,
    variance_decomposition,
)


def test_variance_decomposition_and_tost_contracts():
    means = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    within = np.full_like(means, 0.25)
    count = np.full_like(means, 4.0)
    result = variance_decomposition(means, within, count)
    assert result["aleatory_within_case_variance"][0] == pytest.approx(0.25)
    assert result["mean_estimator_noise_variance"][0] == pytest.approx(0.0625)
    tost = paired_tost_equivalence(np.ones(20), np.ones(20), margin=0.1)
    assert tost["equivalent"] is True


def test_funnel_tree_and_tail_metrics_are_executable():
    funnel = component_funnel(
        emitted=[100, 100],
        armor_hits=[50, 50],
        armor_penetrations=[20, 20],
        component_hits=[10, 10],
        damage_positive=[5, 5],
    )
    assert funnel["dominant_loss_transition"] == "emitted_to_armor_hits"
    tree = tree_distribution_metrics([0.0, 0.5, 1.0])
    assert tree["intermediate_fraction"] == pytest.approx(1.0 / 3.0)

    rng = np.random.default_rng(3)
    hits = rng.integers(1, 20, size=500).astype(float)
    ranges = rng.uniform(1.0, 4.0, size=500)
    mv2 = rng.uniform(100.0, 1000.0, size=500)
    energy = 0.2 * hits * mv2 / ranges**2
    report = log_tail_regression(energy, hits, ranges, mv2, hill_k_values=[20, 50])
    assert report["r2"] > 0.999
    assert report["coefficients"]["log_hits"] == pytest.approx(1.0, abs=1.0e-10)
