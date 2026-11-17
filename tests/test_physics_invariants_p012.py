from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

from damage_engine.fragments import (
    Fragment,
    armor_reference_area_m2,
    propagate_fragment_speed,
)
from damage_engine.geometry import compile_target
from damage_engine.layered_penetration import resolve_penetration_path
from damage_engine.panel_damage import (
    PanelDamageState,
    perforation_opening_area_m2,
)
from damage_engine.penetration import FragmentImpact, resolve_fragment_impact
from damage_engine.registry import PhysicsRegistry


class _MedianQuantile:
    @staticmethod
    def random() -> float:
        return 0.5


def _impact(*, speed_mps: float = 2000.0, thickness_m: float = 1.0e-3):
    return FragmentImpact(
        speed_mps=speed_mps,
        mass_kg=0.005,
        direction=np.array([1.0, 0.0, 0.0]),
        surface_normal=np.array([-1.0, 0.0, 0.0]),
        equivalent_thickness_m=thickness_m,
    )


def _response_model(*, residual_coefficient: float = 0.7) -> dict:
    return {
        "ballistic_limit": {
            "coefficient": 100.0,
            "thickness_exponent": 1.0,
            "mass_exponent": 0.0,
            "obliquity_exponent": 1.0,
            "transition_width_ratio": 0.08,
            "maximum_obliquity_deg": 85.0,
            "residual_velocity_coefficient": residual_coefficient,
            "residual_velocity_exponent": 2.0,
        }
    }


def test_residual_law_recovers_incident_speed_in_the_zero_thickness_limit():
    outcome = resolve_fragment_impact(
        _impact(thickness_m=1.0e-12),
        _response_model(residual_coefficient=0.7),
        _MedianQuantile(),
    )

    assert outcome.perforated
    assert outcome.residual_speed_mps == pytest.approx(2000.0, rel=1.0e-12)
    assert outcome.absorbed_energy_j == pytest.approx(0.0, abs=1.0e-8)


def test_homogeneous_layer_subdivision_does_not_change_sampled_outcome(registry):
    registry_data = deepcopy(registry.data)
    registry_data["fragment_response_models"]["demo_component_response"][
        "ballistic_limit"
    ].update(_response_model(residual_coefficient=0.7)["ballistic_limit"])
    registry_data["layer_stack_models"] = {
        "homogeneous-split": {
            "layers": [
                {
                    "material_id": "demo_steel",
                    "fragment_response_model_id": "demo_component_response",
                    "thickness_fraction": 0.25,
                },
                {
                    "material_id": "demo_steel",
                    "fragment_response_model_id": "demo_component_response",
                    "thickness_fraction": 0.75,
                },
            ],
            "source_id": "TEST-ONLY",
        },
        "homogeneous-four": {
            "layers": [
                {
                    "material_id": "demo_steel",
                    "fragment_response_model_id": "demo_component_response",
                    "thickness_fraction": 0.25,
                }
                for _ in range(4)
            ],
            "source_id": "TEST-ONLY",
        },
    }
    local_registry = PhysicsRegistry(registry_data)
    single_response = {
        "material_id": "demo_steel",
        "fragment_response_model_id": "demo_component_response",
        "layer_stack_id": None,
    }
    single = resolve_penetration_path(
        _impact(), single_response, local_registry, np.random.default_rng(41)
    )
    for stack_id, declared_count in (
        ("homogeneous-split", 2),
        ("homogeneous-four", 4),
    ):
        split = resolve_penetration_path(
            _impact(),
            dict(single_response, layer_stack_id=stack_id),
            local_registry,
            np.random.default_rng(41),
        )

        assert split.declared_layer_count == declared_count
        assert split.resolved_layer_count == 1
        assert split.layers[0].declared_layer_indices == tuple(
            range(declared_count)
        )
        assert split.residual_speed_mps == pytest.approx(single.residual_speed_mps)
        assert split.absorbed_energy_j == pytest.approx(single.absorbed_energy_j)
        assert split.perforated is single.perforated


def test_exact_circle_union_is_order_invariant_and_matches_analytic_solution():
    radius = 0.02
    distance = 0.015
    disk_area = math.pi * radius**2
    overlap = (
        2.0 * radius**2 * math.acos(distance / (2.0 * radius))
        - 0.5 * distance * math.sqrt(4.0 * radius**2 - distance**2)
    )
    expected_union = 2.0 * disk_area - overlap
    centers = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, distance, 0.0]),
    ]
    normal = np.array([1.0, 0.0, 0.0])

    forward = PanelDamageState({"panel": 1.0})
    reverse = PanelDamageState({"panel": 1.0})
    for center in centers:
        forward.record_perforation(
            "panel", center, normal, hole_area_m2=disk_area
        )
    for center in reversed(centers):
        reverse.record_perforation(
            "panel", center, normal, hole_area_m2=disk_area
        )

    assert forward.covered_area_m2("panel") == pytest.approx(
        expected_union, rel=1.0e-12, abs=1.0e-15
    )
    assert reverse.covered_area_m2("panel") == pytest.approx(
        expected_union, rel=1.0e-12, abs=1.0e-15
    )


def test_deferred_circle_union_materializes_to_the_same_exact_area():
    radius = 0.02
    disk_area = math.pi * radius**2
    normal = np.array([1.0, 0.0, 0.0])
    centers = (
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.015, 0.0]),
        np.array([0.0, 0.0075, 0.012,]),
    )
    eager = PanelDamageState({"panel": 1.0})
    deferred = PanelDamageState({"panel": 1.0})
    for center in centers:
        eager.record_perforation(
            "panel", center, normal, hole_area_m2=disk_area
        )
        result = deferred.record_perforation(
            "panel",
            center,
            normal,
            hole_area_m2=disk_area,
            update_union_area=False,
        )
        assert result is None

    assert deferred.covered_area_m2("panel") == pytest.approx(
        eager.covered_area_m2("panel"), rel=1.0e-12, abs=1.0e-15
    )


def test_opening_area_is_finite_bounded_and_responds_to_energy_and_obliquity():
    base = perforation_opening_area_m2(
        projectile_area_m2=1.0e-4,
        incidence_cosine=1.0,
        incident_energy_j=100.0,
        residual_energy_j=0.0,
        hole_area_multiplier=1.25,
    )
    energetic = perforation_opening_area_m2(
        projectile_area_m2=1.0e-4,
        incidence_cosine=1.0,
        incident_energy_j=100.0,
        residual_energy_j=80.0,
        hole_area_multiplier=1.25,
    )
    oblique = perforation_opening_area_m2(
        projectile_area_m2=1.0e-4,
        incidence_cosine=0.5,
        incident_energy_j=100.0,
        residual_energy_j=80.0,
        hole_area_multiplier=1.25,
    )

    assert base == pytest.approx(1.25e-4)
    assert base < energetic < 2.0 * base
    assert oblique == pytest.approx(2.0 * energetic)


def test_thin_shell_reference_area_excludes_extrusion_edge_and_back_face(target_data):
    armor = compile_target(target_data).armor[0]

    assert armor_reference_area_m2(armor) == pytest.approx(4.0)


def test_public_drag_propagator_is_monotone_and_has_zero_distance_identity():
    fragment = Fragment(
        fragment_id=0,
        mass_kg=0.005,
        origin_m=np.zeros(3),
        direction=np.array([1.0, 0.0, 0.0]),
        initial_speed_mps=1000.0,
        area_m2=1.0e-4,
        drag_coefficient=0.9,
    )

    at_origin = propagate_fragment_speed(1000.0, 0.0, fragment, 1.225)
    downstream = propagate_fragment_speed(1000.0, 100.0, fragment, 1.225)

    assert at_origin == 1000.0
    assert 0.0 < downstream < at_origin
