from copy import deepcopy

import numpy as np
import pytest

from damage_engine import FragmentImpact, resolve_fragment_impact
from damage_engine.component_damage import ComponentDamageAccumulator
from damage_engine.damage_tree import probability_and, probability_k_of_n, probability_or
from damage_engine.engine import DamageEngine
from damage_engine.geometry import BoxGeometry
from damage_engine.registry import PhysicsRegistry
from damage_engine.utils import content_hash


BALLISTIC_MODEL = {
    "ballistic_limit": {
        "coefficient": 100.0,
        "thickness_exponent": 1.0,
        "mass_exponent": -0.35,
        "obliquity_exponent": 1.0,
        "transition_width_ratio": 0.1,
        "maximum_obliquity_deg": 80.0,
        "residual_velocity_coefficient": 0.9,
        "residual_velocity_exponent": 2.0,
    }
}


def _impact(*, speed=500.0, mass=0.001, thickness=0.01):
    return FragmentImpact(
        speed_mps=speed,
        mass_kg=mass,
        direction=np.asarray([1.0, 0.0, 0.0]),
        surface_normal=np.asarray([-1.0, 0.0, 0.0]),
        equivalent_thickness_m=thickness,
    )


def test_ballistic_probability_is_monotone_in_thickness_speed_and_mass():
    seed = 101
    thin = resolve_fragment_impact(_impact(thickness=0.005), BALLISTIC_MODEL, np.random.default_rng(seed))
    thick = resolve_fragment_impact(_impact(thickness=0.02), BALLISTIC_MODEL, np.random.default_rng(seed))
    slow = resolve_fragment_impact(_impact(speed=300.0), BALLISTIC_MODEL, np.random.default_rng(seed))
    fast = resolve_fragment_impact(_impact(speed=800.0), BALLISTIC_MODEL, np.random.default_rng(seed))
    light = resolve_fragment_impact(_impact(mass=0.0005), BALLISTIC_MODEL, np.random.default_rng(seed))
    heavy = resolve_fragment_impact(_impact(mass=0.005), BALLISTIC_MODEL, np.random.default_rng(seed))
    assert thin.penetration_probability >= thick.penetration_probability
    assert fast.penetration_probability >= slow.penetration_probability
    assert heavy.penetration_probability >= light.penetration_probability


def test_component_threshold_and_zero_vulnerability_limits():
    base = {
        "degraded": {"residual_energy_50_j": 100.0, "damage_energy_50_j": 100.0, "log_slope": 2.0},
        "failed": {"residual_energy_50_j": 200.0, "damage_energy_50_j": 200.0, "log_slope": 2.0},
    }
    low = ComponentDamageAccumulator(
        ordered_states=("degraded", "failed"), state_models=base, vulnerable_ratio=1.0, accumulation_model="cumulative_damage_energy"
    )
    high_models = deepcopy(base)
    for model in high_models.values():
        model["residual_energy_50_j"] *= 2.0
        model["damage_energy_50_j"] *= 2.0
    high = ComponentDamageAccumulator(
        ordered_states=("degraded", "failed"), state_models=high_models, vulnerable_ratio=1.0, accumulation_model="cumulative_damage_energy"
    )
    zero = ComponentDamageAccumulator(
        ordered_states=("degraded", "failed"), state_models=base, vulnerable_ratio=0.0, accumulation_model="cumulative_damage_energy"
    )
    for accumulator in (low, high, zero):
        accumulator.record_impact(150.0)
    assert low.finalize().probabilities["degraded"] >= high.finalize().probabilities["degraded"]
    assert zero.finalize().probabilities == {"degraded": 0.0, "failed": 0.0}


def test_monotone_gate_families_are_coordinatewise_monotone_randomized():
    rng = np.random.default_rng(20261120)
    for _ in range(500):
        lower = rng.uniform(0.0, 1.0, size=6)
        upper = lower + rng.uniform(0.0, 1.0 - lower)
        assert probability_or(lower) <= probability_or(upper) + 1.0e-12
        assert probability_and(lower) <= probability_and(upper) + 1.0e-12
        for k in (1, 3, 6):
            assert probability_k_of_n(lower, k) <= probability_k_of_n(upper, k) + 1.0e-12


def _fragment_only_case(case_data, target, *, replicates=32):
    case = deepcopy(case_data)
    case["target_ref"]["content_hash"] = content_hash(target)
    case["analysis"]["mechanisms"] = ["fragment"]
    case["analysis"]["replicates"] = replicates
    case["analysis"]["adaptive_ci_half_width"] = None
    case["analysis"].pop("adaptive_physical_ci", None)
    return case


def _fragment_summary(target, case_data, registry_data):
    result = DamageEngine(target, PhysicsRegistry(registry_data)).run(
        _fragment_only_case(case_data, target)
    ).to_dict()
    component = result["components"]["flight-control"]
    return {
        "penetrations": result["summary"]["total_component_penetrations"]["mean"],
        "degraded": component["damage_states"]["combined"]["degraded"]["mean"],
        "failed": component["damage_states"]["combined"]["failed"]["mean"],
    }


def test_end_to_end_fragment_chain_respects_controlled_monotone_variants(
    target_data, case_data, registry
):
    """Exercise monotonicity through the engine with geometry/directions fixed.

    The synthetic cone has zero direction spread, so common random numbers
    isolate armor thickness, fragment mass/speed and response-threshold effects.
    """

    base_target = deepcopy(target_data)
    base_registry = deepcopy(registry.data)

    thin_target = deepcopy(base_target)
    thick_target = deepcopy(base_target)
    thin_target["armor"][0]["response"]["equivalent_thickness"]["value"] = 5.0
    thick_target["armor"][0]["response"]["equivalent_thickness"]["value"] = 20.0
    thin = _fragment_summary(thin_target, case_data, base_registry)
    thick = _fragment_summary(thick_target, case_data, base_registry)
    assert thin["penetrations"] >= thick["penetrations"]
    assert thin["degraded"] >= thick["degraded"]
    assert thin["failed"] >= thick["failed"]

    light_case = deepcopy(case_data)
    heavy_case = deepcopy(case_data)
    light_case["warhead"]["fragment_source"]["mass_distribution"]["parameters"]["value_kg"] = 0.001
    heavy_case["warhead"]["fragment_source"]["mass_distribution"]["parameters"]["value_kg"] = 0.010
    light = _fragment_summary(base_target, light_case, base_registry)
    heavy = _fragment_summary(base_target, heavy_case, base_registry)
    assert heavy["penetrations"] >= light["penetrations"]
    assert heavy["degraded"] >= light["degraded"]
    assert heavy["failed"] >= light["failed"]

    slow_case = deepcopy(case_data)
    fast_case = deepcopy(case_data)
    slow_case["warhead"]["fragment_source"]["gurney_constant_mps"] = 1400.0
    fast_case["warhead"]["fragment_source"]["gurney_constant_mps"] = 3000.0
    slow = _fragment_summary(base_target, slow_case, base_registry)
    fast = _fragment_summary(base_target, fast_case, base_registry)
    assert fast["penetrations"] >= slow["penetrations"]
    assert fast["degraded"] >= slow["degraded"]
    assert fast["failed"] >= slow["failed"]

    low_threshold_registry = deepcopy(base_registry)
    high_threshold_registry = deepcopy(base_registry)
    for state in ("degraded", "failed"):
        low_threshold_registry["fragment_response_models"]["demo_component_response"]["damage_states"][state]["residual_energy_50_j"] *= 0.5
        high_threshold_registry["fragment_response_models"]["demo_component_response"]["damage_states"][state]["residual_energy_50_j"] *= 2.0
    low_threshold = _fragment_summary(base_target, case_data, low_threshold_registry)
    high_threshold = _fragment_summary(base_target, case_data, high_threshold_registry)
    assert low_threshold["degraded"] >= high_threshold["degraded"]
    assert low_threshold["failed"] >= high_threshold["failed"]


def test_synthetic_mirror_geometry_exchanges_component_intersections_exactly():
    left = BoxGeometry(
        np.asarray([0.0, -0.4, 0.0]), np.asarray([0.4, 0.2, 0.2]), np.eye(3)
    )
    right = BoxGeometry(
        np.asarray([0.0, 0.4, 0.0]), np.asarray([0.4, 0.2, 0.2]), np.eye(3)
    )
    origin = np.asarray([-2.0, -0.4, 0.0])
    direction = np.asarray([1.0, 0.0, 0.0])
    mirrored_origin = origin * np.asarray([1.0, -1.0, 1.0])
    mirrored_direction = direction * np.asarray([1.0, -1.0, 1.0])

    left_hit = left.intersect(origin, direction)
    right_hit = right.intersect(mirrored_origin, mirrored_direction)
    assert left_hit is not None and right_hit is not None
    np.testing.assert_allclose(left_hit[:2], right_hit[:2], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        left_hit[2] * np.asarray([1.0, -1.0, 1.0]),
        right_hit[2],
        rtol=0.0,
        atol=1.0e-12,
    )
