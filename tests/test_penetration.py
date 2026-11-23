from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

from damage_engine import FragmentImpact, resolve_fragment_impact
from damage_engine.errors import OutOfDomainError


def test_random_ballistic_limit_keeps_perforation_residual_and_energy_coherent():
    impact = FragmentImpact(
        speed_mps=100.0,
        mass_kg=0.001,
        direction=np.array([1.0, 0.0, 0.0]),
        surface_normal=np.array([-1.0, 0.0, 0.0]),
        equivalent_thickness_m=0.001,
    )
    model = {
        "ballistic_limit": {
            "coefficient": 100.0,
            "thickness_exponent": 1.0,
            "mass_exponent": 0.0,
            "obliquity_exponent": 1.0,
            "transition_width_ratio": 0.1,
            "maximum_obliquity_deg": 80.0,
            "residual_velocity_coefficient": 0.9,
            "residual_velocity_exponent": 2.0,
        }
    }
    rng = np.random.default_rng(20261113)

    outcomes = [resolve_fragment_impact(impact, model, rng) for _ in range(4000)]
    perforated = [item for item in outcomes if item.perforated]

    assert 0.47 < len(perforated) / len(outcomes) < 0.53
    assert all(item.residual_speed_mps > 0.0 for item in perforated)
    assert all(
        item.residual_speed_mps == 0.0
        for item in outcomes
        if not item.perforated
    )
    assert all(
        abs(
            item.incident_energy_j
            - item.residual_energy_j
            - item.absorbed_energy_j
        )
        < 1.0e-10
        for item in outcomes
    )


def test_material_path_uses_equivalent_thickness_and_incidence_not_geometry():
    impact = FragmentImpact(
        speed_mps=500.0,
        mass_kg=0.001,
        direction=np.array([0.5, np.sqrt(0.75), 0.0]),
        surface_normal=np.array([-1.0, 0.0, 0.0]),
        equivalent_thickness_m=0.01,
    )
    model = {
        "ballistic_limit": {
            "coefficient": 100.0,
            "thickness_exponent": 1.0,
            "mass_exponent": 0.0,
            "obliquity_exponent": 1.0,
            "transition_width_ratio": 0.1,
            "maximum_obliquity_deg": 80.0,
        }
    }

    outcome = resolve_fragment_impact(
        impact,
        model,
        np.random.default_rng(20261113),
    )

    assert outcome.incidence_cosine == pytest.approx(0.5)
    assert outcome.material_path_length_m == pytest.approx(0.02)


def test_impact_beyond_ballistic_model_obliquity_is_explicitly_out_of_domain():
    impact = FragmentImpact(
        speed_mps=200.0,
        mass_kg=0.001,
        direction=np.array([0.5, np.sqrt(0.75), 0.0]),
        surface_normal=np.array([-1.0, 0.0, 0.0]),
        equivalent_thickness_m=0.01,
    )
    model = {
        "ballistic_limit": {
            "coefficient": 100.0,
            "thickness_exponent": 1.0,
            "mass_exponent": 0.0,
            "obliquity_exponent": 1.0,
            "transition_width_ratio": 0.1,
            "maximum_obliquity_deg": 30.0,
        },
        "damage_energy_model": {
            "kind": "absorbed_plus_residual",
            "absorbed_weight": 0.2,
            "residual_weight": 0.05,
            "nonperforation_absorbed_fraction": 0.25,
        },
    }

    with pytest.raises(OutOfDomainError, match="obliquity"):
        resolve_fragment_impact(
            impact,
            model,
            np.random.default_rng(20261113),
        )


def _high_obliquity_model() -> dict:
    return {
        "ballistic_limit": {
            "coefficient": 100.0,
            "thickness_exponent": 1.0,
            "mass_exponent": 0.0,
            "obliquity_exponent": 1.0,
            "transition_width_ratio": 0.01,
            "maximum_obliquity_deg": 30.0,
            "high_obliquity_nonperforation_bound": {
                "kind": "pathwise_monotone_ballistic_limit_lower_bound",
                "maximum_obliquity_deg": 89.0,
            },
        },
        "damage_energy_model": {
            "kind": "absorbed_plus_residual",
            "absorbed_weight": 0.2,
            "residual_weight": 0.05,
            "nonperforation_absorbed_fraction": 0.25,
        },
    }


def _impact_at_angle(angle_rad: float, *, speed_mps: float) -> FragmentImpact:
    return FragmentImpact(
        speed_mps=speed_mps,
        mass_kg=0.001,
        direction=np.array([math.cos(angle_rad), math.sin(angle_rad), 0.0]),
        surface_normal=np.array([-1.0, 0.0, 0.0]),
        equivalent_thickness_m=0.001,
    )


def test_high_obliquity_bound_certifies_only_pathwise_nonperforation() -> None:
    outcome = resolve_fragment_impact(
        _impact_at_angle(math.radians(60.0), speed_mps=10.0),
        _high_obliquity_model(),
        np.random.default_rng(7),
    )

    assert outcome.perforated is False
    assert outcome.perforation_class == (
        "certified_nonperforation_by_pathwise_obliquity_bound"
    )
    assert outcome.penetration_probability_semantics == (
        "boundary_domain_penetration_probability_upper_bound"
    )
    assert outcome.residual_speed_mps == 0.0
    assert outcome.absorbed_energy_j == pytest.approx(outcome.incident_energy_j)
    assert outcome.damage_energy_j == pytest.approx(0.25 * outcome.incident_energy_j)


def test_high_obliquity_bound_refuses_when_nonperforation_is_not_certified() -> None:
    with pytest.raises(OutOfDomainError) as exc_info:
        resolve_fragment_impact(
            _impact_at_angle(math.radians(60.0), speed_mps=1000.0),
            _high_obliquity_model(),
            np.random.default_rng(7),
        )

    assert exc_info.value.context["reason_code"] == "obliquity_out_of_domain"
    assert exc_info.value.context["detail_code"] == (
        "high_obliquity_nonperforation_not_certified"
    )


def test_high_obliquity_policy_does_not_change_in_domain_result() -> None:
    policy_model = _high_obliquity_model()
    legacy_model = deepcopy(policy_model)
    del legacy_model["ballistic_limit"]["high_obliquity_nonperforation_bound"]
    impact = _impact_at_angle(math.radians(20.0), speed_mps=100.0)

    policy = resolve_fragment_impact(
        impact, policy_model, np.random.default_rng(20261122)
    )
    legacy = resolve_fragment_impact(
        impact, legacy_model, np.random.default_rng(20261122)
    )

    assert policy == legacy


def test_next_float_above_domain_boundary_uses_bound_not_silent_clamp() -> None:
    boundary = math.radians(30.0)
    boundary_cosine = math.cos(boundary)
    outside_cosine = math.nextafter(boundary_cosine, 0.0)
    impact = FragmentImpact(
        speed_mps=10.0,
        mass_kg=0.001,
        direction=np.array(
            [outside_cosine, math.sqrt(1.0 - outside_cosine**2), 0.0]
        ),
        surface_normal=np.array([-1.0, 0.0, 0.0]),
        equivalent_thickness_m=0.001,
    )
    policy = resolve_fragment_impact(
        impact, _high_obliquity_model(), np.random.default_rng(7)
    )
    legacy_model = _high_obliquity_model()
    del legacy_model["ballistic_limit"]["high_obliquity_nonperforation_bound"]

    assert policy.impact_angle_rad > boundary
    assert policy.perforation_class.startswith("certified_nonperforation")
    with pytest.raises(OutOfDomainError):
        resolve_fragment_impact(impact, legacy_model, np.random.default_rng(7))


def test_fragment_impact_requires_the_incoming_side_of_the_surface():
    impact = FragmentImpact(
        speed_mps=500.0,
        mass_kg=0.001,
        direction=np.array([1.0, 0.0, 0.0]),
        surface_normal=np.array([1.0, 0.0, 0.0]),
        equivalent_thickness_m=0.01,
    )
    model = {
        "ballistic_limit": {
            "coefficient": 100.0,
            "thickness_exponent": 1.0,
            "mass_exponent": 0.0,
            "obliquity_exponent": 1.0,
            "transition_width_ratio": 0.1,
            "maximum_obliquity_deg": 80.0,
        }
    }

    with pytest.raises(OutOfDomainError, match="incoming surface side"):
        resolve_fragment_impact(
            impact,
            model,
            np.random.default_rng(20261113),
        )
