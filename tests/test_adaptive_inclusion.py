from __future__ import annotations

import numpy as np
import pytest

from damage_engine.adaptive_sampling import (
    hajek_mean,
    inclusion_probabilities,
    raw_design_weights,
    select_bernoulli,
)
from damage_engine.case_factory import randomized_lhs
from damage_engine.errors import InputValidationError
from damage_engine.sampling_audit import audit_sampling


def test_inclusion_probabilities_respect_positivity_and_expected_budget() -> None:
    scores = np.array([0.0, 0.25, 1.0, 4.0])

    probabilities = inclusion_probabilities(
        scores,
        expected_count=2.3,
        rho_min=0.2,
    )
    assert np.all(probabilities >= 0.2)
    assert np.all(probabilities <= 1.0)
    assert probabilities[0] == pytest.approx(0.2)
    assert np.all(np.diff(probabilities) >= 0.0)
    assert probabilities.sum() == pytest.approx(2.3, abs=1.0e-10)


def test_inclusion_probabilities_preserve_budget_when_positive_scores_saturate() -> None:
    # Two zero-score candidates stay selectable at the positivity floor.  The
    # positive candidate saturates at one, so any residual mass must be
    # redistributed without losing the requested expected-count invariant.
    probabilities = inclusion_probabilities(
        [0.0, 0.0, 1.0],
        expected_count=2.0,
        rho_min=0.1,
    )
    assert np.all((0.1 <= probabilities) & (probabilities <= 1.0))
    assert probabilities.sum() == pytest.approx(2.0, abs=1.0e-10)


def test_inclusion_budget_below_positivity_floor_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="positivity floor"):
        inclusion_probabilities([0.0, 1.0, 2.0], expected_count=1.0, rho_min=0.5)


def test_bernoulli_selection_replays_from_the_selection_seed() -> None:
    probabilities = np.array([0.2, 0.5, 0.9, 1.0])

    selected, draws = select_bernoulli(probabilities, seed=77)
    replay_selected, replay_draws = select_bernoulli(probabilities, seed=77)

    np.testing.assert_array_equal(selected, replay_selected)
    np.testing.assert_array_equal(draws, replay_draws)
    np.testing.assert_array_equal(selected, draws < probabilities)


def test_hajek_weights_recover_a_toy_finite_population_mean() -> None:
    # The selected low-valued unit represents three population units, while
    # the high-valued unit is a certainty selection.  The unweighted selected
    # mean is 5, but the four-unit finite-population mean is 2.5.
    selected_values = np.array([0.0, 10.0])
    weights = raw_design_weights([1.0 / 3.0, 1.0])

    np.testing.assert_allclose(weights, [3.0, 1.0])
    assert float(np.mean(selected_values)) == 5.0
    assert hajek_mean(selected_values, weights) == pytest.approx(2.5)


def test_hajek_mean_is_finite_when_raw_weight_sum_would_overflow() -> None:
    assert hajek_mean([0.0, 1.0], [1.0e308, 1.0e308]) == pytest.approx(0.5)


def test_sampling_audit_checks_inverse_weights_and_reports_software_boundary() -> None:
    latent = randomized_lhs(16, 2, seed=9)
    rows = [
        {
            "inclusion_probability": 1.0,
            "candidate_id": f"toy-{index}", "lineage_id": f"lineage-{index}",
            "epistemic_group_id": f"epistemic-{index}", "weight_mask": True,
            "selected": True,
            "design_weight_raw": 1.0,
            "split": "train" if index >= 4 else "validation",
            "role": "train_pilot" if index >= 4 else "reference_validation",
            "stream": "reference",
        }
        for index in range(16)
    ]
    gates = {"positivity_min": 0.1, "min_ess_fraction": 0.9, "max_weight_cv": 0.1}

    report = audit_sampling(rows, latent, quality_gates=gates)

    assert report["candidate_count"] == 16
    assert report["selected_count"] == 16
    assert report["effective_sample_size"] == pytest.approx(16.0)
    assert report["quality_gate_passed"] is True
    assert "do not validate damage physics" in report["evidence_boundary"]

    rows[0]["design_weight_raw"] = 2.0
    with pytest.raises(InputValidationError, match="inverse inclusion probabilities"):
        audit_sampling(rows, latent, quality_gates=gates)


def test_sampling_audit_normalizes_extreme_finite_weights_without_overflow() -> None:
    rows = [
        {
            "inclusion_probability": 1.0e-308,
            "candidate_id": f"extreme-{index}",
            "lineage_id": f"extreme-lineage-{index}",
            "epistemic_group_id": f"extreme-epistemic-{index}",
            "weight_mask": True,
            "selected": True,
            "design_weight_raw": 1.0e308,
            "split": "train",
            "role": "train_pilot",
            "stream": "reference",
        }
        for index in range(2)
    ]
    report = audit_sampling(
        rows,
        np.asarray([[0.0], [1.0]]),
        quality_gates={
            "positivity_min": 1.0e-309,
            "min_ess_fraction": 0.9,
            "max_weight_cv": 0.1,
        },
    )
    assert report["design_weight"]["maximum_normalized"] == pytest.approx(0.5)
    assert report["cross_round_drift"]["rounds"][0][
        "selected_weighted_mean_by_axis"
    ] == pytest.approx([0.5])


def test_adaptive_audit_accepts_unbounded_densities_without_fake_latent_coverage() -> None:
    rows = [
        {
            "inclusion_probability": 1.0,
            "candidate_id": f"adaptive-{index}",
            "lineage_id": f"adaptive-lineage-{index}",
            "epistemic_group_id": f"adaptive-epistemic-{index}",
            "weight_mask": True,
            "selected": True,
            "p0_density": 4.0 + index,
            "adaptive_density": 8.0 + index,
            "proposal_density": 6.0 + index,
            "p0_hierarchy_probability": 0.5,
            "proposal_hierarchy_probability": 0.5,
            "design_weight_raw": (4.0 + index) / (6.0 + index),
            "split": "train",
            "role": "adaptive_train",
            "stream": "adaptive_train",
            "round": 1,
        }
        for index in range(4)
    ]
    gates = {"positivity_min": 0.1, "min_ess_fraction": 0.9, "max_weight_cv": 0.1}

    report = audit_sampling(rows, None, quality_gates=gates)

    assert report["candidate_frame_unweighted"]["status"] == (
        "not_evaluated_without_prelabel_latent_design"
    )
    assert report["selected_weighted"]["ks_by_axis"] is None
    assert report["cross_round_drift"]["status"] == (
        "not_evaluated_without_prelabel_latent_design"
    )
