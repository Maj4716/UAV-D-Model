from __future__ import annotations

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
