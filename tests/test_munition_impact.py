from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import pytest

from damage_engine.case_factory import CaseFactory
from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.errors import InputValidationError, OutOfDomainError
from damage_engine.impact import resolve_impact_detonation
from damage_engine.input_identity import case_identity_hash
from damage_engine.label_spec import LabelSpec
from damage_engine.munition_profiles import (
    MUNITION_PROFILE_IDS,
    MUNITION_PROFILES,
    apply_munition_profile,
)
from damage_engine.registry import PhysicsRegistry
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.utils import content_hash, load_json
from damage_engine.validation import validate_inputs


ROOT = Path(__file__).resolve().parents[1]


def _engine() -> DamageEngine:
    return DamageEngine(
        load_json(ROOT / "raw_data" / "target.rebuilt.json"),
        PhysicsRegistry.from_file(ROOT / "examples" / "physics.loitering_impact.json"),
    )


def _case() -> dict:
    return load_json(ROOT / "raw_data" / "case.loitering_impact.v1.json")


def test_catalog_contains_only_the_three_approved_profiles() -> None:
    assert MUNITION_PROFILE_IDS == (
        "small_front",
        "medium_front",
        "large_front",
    )
    assert "medium_rear" not in MUNITION_PROFILES
    expected = {
        "small_front": (0.30, 0.10, 0.12, 0.09, 0.0025, 1.0, 2440.0, 6900.0, 60, 0.005, 6),
        "medium_front": (0.45, 0.15, 0.20, 0.13, 0.0030, 3.0, 2440.0, 6900.0, 150, 0.008, 12),
        "large_front": (0.60, 0.20, 0.28, 0.18, 0.0040, 5.0, 2700.0, 7900.0, 300, 0.010, 18),
    }
    for profile_id, values in expected.items():
        profile = MUNITION_PROFILES[profile_id]
        assert (
            profile.projectile_length_m,
            profile.projectile_diameter_m,
            profile.warhead_length_m,
            profile.warhead_diameter_m,
            profile.wall_thickness_m,
            profile.charge_mass_kg,
            profile.gurney_constant_mps,
            profile.detonation_velocity_mps,
            profile.fragment_count,
            profile.fragment_mass_kg,
            profile.fragment_ring_count,
        ) == values
        assert profile.detonation_point == "front"
        assert profile.fragment_shape == "cube"
        assert profile.axial_coverage_ratio == 0.85
        assert profile.spread_sigma_deg == 2.0
        assert sum(profile.fragments_per_ring) == profile.fragment_count


def test_profile_application_updates_all_solver_inputs() -> None:
    case = _case()
    apply_munition_profile(case, "large_front")
    profile = MUNITION_PROFILES["large_front"]
    assert case["munition"]["profile_id"] == "large_front"
    assert case["warhead"]["charge_mass_kg"] == 5.0
    assert case["warhead"]["fragment_source"]["total_count"] == 300
    assert case["warhead"]["fragment_source"]["mass_distribution"]["parameters"]["value_kg"] == 0.010
    assert math.isclose(
        case["warhead"]["casing_mass_kg"],
        profile.casing_mass_kg,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )


def test_reference_front_axis_and_cube_drag_area_are_not_silently_reinterpreted() -> None:
    engine = _engine()
    for profile in MUNITION_PROFILES.values():
        source = engine.registry.fragment_source(profile.source_model_id)
        assert source["direction"]["axis_body"] == [-1.0, 0.0, 0.0]
        assert source["drag"]["coefficient"] == 1.05
        # Convert the engine's equal-volume-sphere projection to the reference
        # cube face area A=(m/rho)^(2/3).
        assert source["drag"]["area_shape_factor"] == pytest.approx(
            0.8271339879, rel=0.0, abs=1.0e-10
        )


def test_first_armor_contact_sets_detonation_time_and_position() -> None:
    engine = _engine()
    resolved = resolve_impact_detonation(_case(), engine.compiled_target)
    encounter = resolved["encounter"]
    expected_distance = 10.0 - 1.6302649974822998
    assert encounter["collision_armor_id"] == "armor-001"
    assert encounter["collision_distance_m"] == pytest.approx(expected_distance)
    assert encounter["detonation_time_s"] == pytest.approx(expected_distance / 120.0)
    assert encounter["detonation_delay_s"] == 0.0
    assert encounter["detonation_position_m"] == pytest.approx(
        [-1.6302649974822998, 0.0, 1.2]
    )
    start = np.asarray(encounter["trajectory_origin_m"])
    velocity = np.asarray(encounter["velocity_target_mps"])
    assert np.allclose(
        start + velocity * encounter["detonation_time_s"],
        encounter["detonation_position_m"],
    )


def test_impact_resolution_refuses_zero_speed_and_miss() -> None:
    engine = _engine()
    zero = _case()
    zero["encounter"]["velocity_target_mps"] = [0.0, 0.0, 0.0]
    with pytest.raises(InputValidationError, match="non-zero"):
        resolve_impact_detonation(zero, engine.compiled_target)

    miss = _case()
    miss["encounter"]["trajectory_origin_m"] = [-10.0, 20.0, 20.0]
    with pytest.raises(OutOfDomainError, match="does not collide"):
        resolve_impact_detonation(miss, engine.compiled_target)

    inside = _case()
    inside["encounter"]["trajectory_origin_m"] = [0.0, 0.0, 1.2]
    with pytest.raises(OutOfDomainError, match="must be outside"):
        resolve_impact_detonation(inside, engine.compiled_target)


def test_runtime_validation_rejects_tampered_collision_time() -> None:
    engine = _engine()
    case = engine.prepare_case(_case())
    case["encounter"]["detonation_time_s"] += 0.01
    with pytest.raises(InputValidationError, match="detonation_time_s"):
        validate_inputs(
            engine.target_data,
            case,
            engine.registry,
            compiled_target=engine.compiled_target,
        )


def test_case_factory_materializes_every_sample_before_identity() -> None:
    engine = _engine()
    spec = SamplingSpec.from_file(
        ROOT / "raw_data" / "sampling_spec.loitering_impact.smoke.v1.json"
    )
    label = LabelSpec.from_file(
        ROOT / "raw_data" / "label_spec.loitering_impact.v1.json"
    )
    plan = CaseFactory(
        _case(),
        spec,
        label,
        target_hash=engine.target_hash,
        damage_tree_definition_hash=content_hash(engine.target_data["damage_tree"]),
        physics_registry_hash=engine.registry.hash,
        engine_code_hash=ENGINE_CODE_HASH,
        physics_registry=engine.registry,
        compiled_target=engine.compiled_target,
    ).build_reference_plan()
    assert len(plan.cases) == 12
    assert {case["munition"]["profile_id"] for case in plan.cases} == set(
        MUNITION_PROFILE_IDS
    )
    for case in plan.cases:
        encounter = case["encounter"]
        assert encounter["collision_armor_id"] in {"armor-001", "armor-002"}
        assert encounter["detonation_time_s"] > 0.0
        validate_inputs(
            engine.target_data,
            case,
            engine.registry,
            compiled_target=engine.compiled_target,
        )


def test_engine_recomputes_stale_persisted_impact_fields() -> None:
    engine = _engine()
    case = deepcopy(_case())
    case["encounter"]["detonation_time_s"] = 999.0
    result = engine.run(case).to_dict()
    assert result["status"] == "ok"
    assert result["provenance"]["case_hash"] == case_identity_hash(
        engine.prepare_case(case)
    )
