from __future__ import annotations

import pytest

from damage_engine.mechanism_coupling import combine_mechanism_states


def test_mechanism_combination_exposes_identified_bounds_and_unidentified_point():
    component_states = {"component-a": ("degraded", "failed")}
    fragment = {"component-a": {"degraded": 0.4, "failed": 0.1}}
    blast = {"component-a": {"degraded": 0.3, "failed": 0.05}}

    result = combine_mechanism_states(component_states, fragment, blast)

    assert result.point_estimate["component-a"]["degraded"] == pytest.approx(0.58)
    assert result.scenarios["frechet_lower"]["component-a"]["degraded"] == 0.4
    assert result.scenarios["frechet_upper"]["component-a"]["degraded"] == 0.7
    assert result.identification_status == "not_identified"
    assert result.point_estimate_assumption == "conditional_independence"


def test_mechanism_combination_preserves_cumulative_state_order():
    component_states = {"component-a": ("degraded", "failed")}
    fragment = {"component-a": {"degraded": 0.8, "failed": 0.2}}
    blast = {"component-a": {"degraded": 0.6, "failed": 0.1}}

    result = combine_mechanism_states(component_states, fragment, blast)

    for scenario in result.scenarios.values():
        assert scenario["component-a"]["failed"] <= scenario["component-a"][
            "degraded"
        ]


def test_shared_vulnerable_region_is_not_counted_twice():
    result = combine_mechanism_states(
        {"c": ("failed",)},
        {"c": {"failed": 0.4}},
        {"c": {"failed": 0.3}},
        shared_vulnerable_ratio={"c": 0.5},
    )

    # Conditional risks are 0.8 and 0.6 inside the same 50% vulnerable
    # region: 0.5 * (1 - 0.2 * 0.4) = 0.46.
    assert result.point_estimate["c"]["failed"] == pytest.approx(0.46)
    assert result.frechet_bounds["c"]["failed"]["lower"] == pytest.approx(0.4)
    # Formal Fréchet bounds stay assumption-free; the shared-support result is
    # an explicitly labelled point sensitivity inside those bounds.
    assert result.frechet_bounds["c"]["failed"]["upper"] == pytest.approx(0.7)
    assert result.point_estimate_assumption == (
        "conditional_independence_within_minimum_shared_support_sensitivity"
    )
