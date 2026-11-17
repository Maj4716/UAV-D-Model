from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from damage_engine.layered_penetration import resolve_penetration_path
from damage_engine.penetration import FragmentImpact
from damage_engine.registry import PhysicsRegistry


def test_layer_stack_resolves_in_declared_order_and_closes_total_energy(registry):
    registry_data = deepcopy(registry.data)
    registry_data["layer_stack_models"] = {
        "two-layer-demo": {
            "layers": [
                {
                    "material_id": "demo_steel",
                    "fragment_response_model_id": "demo_component_response",
                    "thickness_fraction": 0.4,
                },
                {
                    "material_id": "demo_steel",
                    "fragment_response_model_id": "demo_armor_response",
                    "thickness_fraction": 0.6,
                },
            ],
            "source_id": "TEST-ONLY",
        }
    }
    registry_data["fragment_response_models"]["demo_component_response"][
        "damage_energy_model"
    ] = {
        "kind": "absorbed_plus_residual",
        "absorbed_weight": 0.2,
        "residual_weight": 0.05,
        "nonperforation_absorbed_fraction": 0.15,
    }
    registry_data["fragment_response_models"]["demo_armor_response"][
        "damage_energy_model"
    ] = deepcopy(
        registry_data["fragment_response_models"]["demo_component_response"][
            "damage_energy_model"
        ]
    )
    layered_registry = PhysicsRegistry(registry_data)
    response = {
        "material_id": "demo_steel",
        "fragment_response_model_id": "demo_component_response",
        "layer_stack_id": "two-layer-demo",
    }
    impact = FragmentImpact(
        speed_mps=1000.0,
        mass_kg=0.001,
        direction=np.array([1.0, 0.0, 0.0]),
        surface_normal=np.array([-1.0, 0.0, 0.0]),
        equivalent_thickness_m=0.001,
    )

    outcome = resolve_penetration_path(
        impact, response, layered_registry, np.random.default_rng(20261114)
    )

    assert outcome.layer_count == 2
    assert outcome.declared_layer_count == 2
    assert outcome.resolved_layer_count == 2
    assert [layer.thickness_fraction for layer in outcome.layers] == [0.4, 0.6]
    assert outcome.perforated
    assert outcome.perforation_class == "complete_layer_stack"
    assert outcome.material_path_length_m == pytest.approx(0.001)
    expected_transit_time = 0.0
    layer_incoming_speed = impact.speed_mps
    for layer in outcome.layers:
        expected_transit_time += (
            2.0
            * layer.outcome.material_path_length_m
            / (layer_incoming_speed + layer.outcome.residual_speed_mps)
        )
        layer_incoming_speed = layer.outcome.residual_speed_mps
    assert outcome.material_transit_time_s == pytest.approx(expected_transit_time)
    total_path_average_approximation = (
        2.0
        * outcome.material_path_length_m
        / (impact.speed_mps + outcome.residual_speed_mps)
    )
    assert outcome.material_transit_time_s != pytest.approx(
        total_path_average_approximation,
        rel=1.0e-8,
        abs=0.0,
    )
    assert outcome.incident_energy_j == pytest.approx(
        outcome.residual_energy_j + outcome.absorbed_energy_j
    )
    model = layered_registry.fragment_response("demo_component_response")[
        "damage_energy_model"
    ]
    expected_damage = sum(
        model["absorbed_weight"] * layer.outcome.absorbed_energy_j
        for layer in outcome.layers
    ) + model["residual_weight"] * outcome.residual_energy_j
    assert outcome.damage_energy_j == pytest.approx(expected_damage)


def test_single_layer_adapter_preserves_existing_penetration_interface(registry):
    response = {
        "material_id": "demo_steel",
        "fragment_response_model_id": "demo_component_response",
        "layer_stack_id": None,
    }
    impact = FragmentImpact(
        speed_mps=500.0,
        mass_kg=0.001,
        direction=np.array([1.0, 0.0, 0.0]),
        surface_normal=np.array([-1.0, 0.0, 0.0]),
        equivalent_thickness_m=0.001,
    )

    outcome = resolve_penetration_path(
        impact, response, registry, np.random.default_rng(20261114)
    )

    assert outcome.layer_count == 1
    assert outcome.layers[0].thickness_fraction == 1.0
    assert outcome.material_transit_time_s >= 0.0
    assert outcome.perforation_class in {"complete", "none"}


def test_stopped_multilayer_path_does_not_report_partial_path_probability_as_full_stack_probability(
    registry,
):
    registry_data = deepcopy(registry.data)
    registry_data["layer_stack_models"] = {
        "two-layer-probability-contract": {
            "layers": [
                {
                    "material_id": "demo_steel",
                    "fragment_response_model_id": "demo_armor_response",
                    "thickness_fraction": 0.5,
                },
                {
                    "material_id": "demo_steel",
                    "fragment_response_model_id": "demo_component_response",
                    "thickness_fraction": 0.5,
                },
            ],
            "source_id": "TEST-ONLY",
        }
    }
    layered_registry = PhysicsRegistry(registry_data)
    response = {
        "material_id": "demo_steel",
        "fragment_response_model_id": "demo_component_response",
        "layer_stack_id": "two-layer-probability-contract",
    }
    impact = FragmentImpact(
        speed_mps=100.0,
        mass_kg=0.001,
        direction=np.array([1.0, 0.0, 0.0]),
        surface_normal=np.array([-1.0, 0.0, 0.0]),
        equivalent_thickness_m=0.001,
    )

    class HighQuantile:
        @staticmethod
        def random():
            return 0.999999

    outcome = resolve_penetration_path(
        impact,
        response,
        layered_registry,
        HighQuantile(),
    )

    assert outcome.perforated is False
    assert outcome.resolved_layer_count == 1
    assert 0.0 < outcome.realized_branch_probability < 1.0
    assert outcome.penetration_probability == 0.0
    assert outcome.penetration_probability_semantics == (
        "complete_stack_bernoulli_observation"
    )
