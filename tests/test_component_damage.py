from __future__ import annotations

import numpy as np
import pytest

from damage_engine.component_damage import ComponentDamageAccumulator


STATE_MODELS = {
    "degraded": {
        "residual_energy_50_j": 100.0,
        "damage_energy_50_j": 100.0,
        "log_slope": 1.0,
    },
    "failed": {
        "residual_energy_50_j": 200.0,
        "damage_energy_50_j": 200.0,
        "log_slope": 1.0,
    },
}


def test_cumulative_energy_applies_vulnerability_once_per_component_realization():
    accumulator = ComponentDamageAccumulator(
        ordered_states=("degraded", "failed"),
        state_models=STATE_MODELS,
        vulnerable_ratio=0.5,
        accumulation_model="cumulative_damage_energy",
    )
    accumulator.record_impact(50.0)
    accumulator.record_impact(50.0)

    result = accumulator.finalize()

    assert result.total_damage_energy_j == 100.0
    assert result.probabilities["degraded"] == pytest.approx(0.25)
    assert result.probabilities["failed"] < result.probabilities["degraded"]
    assert result.hit_count == 2


def test_legacy_independent_hit_adapter_remains_explicitly_available():
    accumulator = ComponentDamageAccumulator(
        ordered_states=("degraded", "failed"),
        state_models=STATE_MODELS,
        vulnerable_ratio=0.5,
        accumulation_model="independent_hits",
    )
    isolated = accumulator.isolated_hit_probabilities(50.0)
    accumulator.record_impact(50.0)
    accumulator.record_impact(50.0)

    result = accumulator.finalize()

    expected = 1.0 - (1.0 - isolated["degraded"]) ** 2
    assert result.probabilities["degraded"] == pytest.approx(expected)
    assert result.accumulation_model == "independent_hits"


def test_component_damage_rejects_unknown_accumulation_model():
    with pytest.raises(Exception, match="unsupported component damage accumulation"):
        ComponentDamageAccumulator(
            ordered_states=("degraded", "failed"),
            state_models=STATE_MODELS,
            vulnerable_ratio=1.0,
            accumulation_model="invented",
        )


def test_spatial_cumulative_damage_distinguishes_local_repeat_hits_from_separated_hits():
    def accumulator():
        return ComponentDamageAccumulator(
            ordered_states=("degraded", "failed"),
            state_models=STATE_MODELS,
            vulnerable_ratio=1.0,
            accumulation_model="spatial_cumulative_damage_energy",
            spatial_correlation_length_m=0.02,
        )

    local = accumulator()
    local.record_impact(50.0, location_m=np.array([0.0, 0.0, 0.0]))
    local.record_impact(50.0, location_m=np.array([0.0, 0.01, 0.0]))
    separated = accumulator()
    separated.record_impact(50.0, location_m=np.array([0.0, 0.0, 0.0]))
    separated.record_impact(50.0, location_m=np.array([0.0, 0.10, 0.0]))

    local_result = local.finalize()
    separated_result = separated.finalize()

    assert local_result.spatial_cluster_count == 1
    assert separated_result.spatial_cluster_count == 2
    assert local_result.total_damage_energy_j == pytest.approx(100.0)
    assert separated_result.total_damage_energy_j == pytest.approx(100.0)
    assert separated_result.probabilities["degraded"] != pytest.approx(
        local_result.probabilities["degraded"]
    )


def test_spatial_cumulative_damage_cannot_chain_beyond_correlation_length():
    accumulator = ComponentDamageAccumulator(
        ordered_states=("degraded", "failed"),
        state_models=STATE_MODELS,
        vulnerable_ratio=1.0,
        accumulation_model="spatial_cumulative_damage_energy",
        spatial_correlation_length_m=0.05,
    )
    # A-B and B-C are local, but A-C exceeds the declared spatial support.
    for y in (0.00, 0.04, 0.08):
        accumulator.record_impact(
            25.0,
            location_m=np.array([0.0, y, 0.0]),
        )

    result = accumulator.finalize()

    assert result.spatial_cluster_count == 2
    assert result.spatial_cluster_max_diameter_m == pytest.approx(0.04)
    assert result.spatial_cluster_max_diameter_m <= 0.05


def test_spatial_complete_link_clusters_are_invariant_to_impact_order():
    def evaluate(order):
        accumulator = ComponentDamageAccumulator(
            ordered_states=("degraded", "failed"),
            state_models=STATE_MODELS,
            vulnerable_ratio=1.0,
            accumulation_model="spatial_cumulative_damage_energy",
            spatial_correlation_length_m=0.05,
        )
        for y, energy in order:
            accumulator.record_impact(
                energy,
                location_m=np.array([0.0, y, 0.0]),
            )
        return accumulator.finalize()

    forward = evaluate([(0.00, 10.0), (0.04, 20.0), (0.08, 30.0)])
    reverse = evaluate([(0.08, 30.0), (0.04, 20.0), (0.00, 10.0)])

    assert reverse.spatial_cluster_count == forward.spatial_cluster_count
    assert reverse.spatial_cluster_max_diameter_m == pytest.approx(
        forward.spatial_cluster_max_diameter_m
    )
    assert reverse.probabilities == pytest.approx(forward.probabilities)
