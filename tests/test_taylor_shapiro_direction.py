from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from damage_engine.dataset import CASE_FEATURE_NAMES, _case_features
from damage_engine.engine import DamageEngine
from damage_engine.errors import InputValidationError, OutOfDomainError
from damage_engine.fast_sampling_proxy import _weighted_fragment_packets
from damage_engine.fragments import (
    TAYLOR_SHAPIRO_LEGACY_ANGLE_CONVENTION,
    TAYLOR_SHAPIRO_WAVE_DIRECTED_ANGLE_CONVENTION,
    _sample_taylor_shapiro_directions,
    build_taylor_shapiro_ring_layout,
    generate_fragments,
    taylor_shapiro_angles,
    taylor_shapiro_deflection,
)
from damage_engine.munition_profiles import (
    TAYLOR_SHAPIRO_PROFILE_REVISION,
    apply_munition_profile,
)
from damage_engine.registry import PhysicsRegistry
from damage_engine.utils import load_json
from damage_engine.validation import validate_inputs


ROOT = Path(__file__).resolve().parents[1]
V1_PHYSICS = ROOT / "configs" / "physics" / "loitering_impact.json"
V2_PHYSICS = (
    ROOT / "configs" / "physics" / "loitering_impact.taylor_shapiro.v2.json"
)
V3_PHYSICS = (
    ROOT
    / "configs"
    / "physics"
    / "loitering_impact.taylor_shapiro.obliquity_bound.v3.json"
)
V1_CASE = ROOT / "configs" / "cases" / "loitering_impact.fragment_only.v1.json"
V2_CASE = (
    ROOT
    / "configs"
    / "cases"
    / "loitering_impact.fragment_only.taylor_shapiro.v2.json"
)
V3_CASE = (
    ROOT
    / "configs"
    / "cases"
    / "loitering_impact.fragment_only.taylor_shapiro.obliquity_bound.v3.json"
)
V4_PHYSICS = (
    ROOT
    / "configs"
    / "physics"
    / "loitering_impact.taylor_shapiro.wave_directed.obliquity_bound.v4.json"
)
V4_CASE = (
    ROOT
    / "configs"
    / "cases"
    / "loitering_impact.fragment_only.taylor_shapiro.wave_directed.obliquity_bound.v4.json"
)
TARGET = ROOT / "configs" / "targets" / "target.fragment_only.v1.json"


def _engine(physics_path: Path) -> DamageEngine:
    return DamageEngine(load_json(TARGET), PhysicsRegistry.from_file(physics_path))


def test_taylor_shapiro_reference_angles_and_taylor_bound() -> None:
    speed = 250.0
    detonation_velocity = 1000.0
    taylor, forward = taylor_shapiro_angles(
        speed, detonation_velocity, math.pi / 2.0, 0.0
    )
    _, transverse = taylor_shapiro_angles(
        speed, detonation_velocity, math.pi / 2.0, math.pi / 2.0
    )
    _, rearward = taylor_shapiro_angles(
        speed, detonation_velocity, math.pi / 2.0, math.pi
    )

    assert math.degrees(taylor) == pytest.approx(7.180755781, abs=1.0e-9)
    assert math.degrees(forward) == pytest.approx(7.125016349, abs=1.0e-9)
    assert transverse == pytest.approx(0.0, abs=1.0e-15)
    assert rearward == pytest.approx(-forward, abs=1.0e-15)
    assert abs(forward) <= taylor
    assert abs(rearward) <= taylor


def test_taylor_shapiro_rejects_velocity_ratio_outside_taylor_domain() -> None:
    with pytest.raises(OutOfDomainError, match="outside the Taylor domain"):
        taylor_shapiro_angles(2001.0, 1000.0, math.pi / 2.0, math.pi)


def test_v2_profile_application_preserves_the_selected_revision() -> None:
    case = load_json(V2_CASE)
    apply_munition_profile(case, "medium_front")

    assert case["munition"]["profile_revision"] == TAYLOR_SHAPIRO_PROFILE_REVISION
    assert case["warhead"]["revision"] == TAYLOR_SHAPIRO_PROFILE_REVISION
    assert case["warhead"]["fragment_source"]["model_id"] == (
        "loitering_medium_front_taylor_shapiro_source"
    )
    assert case["warhead"]["fragment_source"]["direction_model_id"] == (
        "taylor_shapiro_ring"
    )


def test_v2_layout_shares_casing_origins_normals_and_meridional_directions() -> None:
    registry = PhysicsRegistry.from_file(V2_PHYSICS)
    case = load_json(V2_CASE)
    source = registry.fragment_source(
        case["warhead"]["fragment_source"]["model_id"]
    )
    layout = build_taylor_shapiro_ring_layout(
        case,
        source,
        case["warhead"]["fragment_source"]["total_count"],
    )
    directions = _sample_taylor_shapiro_directions(
        layout,
        np.eye(3),
        np.random.default_rng(17),
        gurney_speed=2000.0,
        detonation_velocity_mps=6900.0,
        spread_sigma_deg=0.0,
    )

    inner_radius = (
        case["munition"]["warhead_diameter_m"] / 2.0
        - case["munition"]["wall_thickness_m"]
    )
    radial = layout.source_offsets_body_m.copy()
    radial[:, 0] = 0.0
    assert np.linalg.norm(radial, axis=1) == pytest.approx(inner_radius)
    assert np.linalg.norm(layout.local_normals_body, axis=1) == pytest.approx(1.0)
    assert np.linalg.norm(directions, axis=1) == pytest.approx(1.0)
    assert len(np.unique(np.round(layout.source_offsets_body_m[:, 0], 12))) == 6

    azimuth_tangents = np.cross(
        np.repeat(layout.body_axis[None, :], len(directions), axis=0),
        layout.local_normals_body,
    )
    tangential_components = np.einsum("ij,ij->i", directions, azimuth_tangents)
    assert tangential_components == pytest.approx(0.0, abs=1.0e-12)
    # Legacy v2/v3 sign, kept only to reproduce archived datasets: the nose
    # initiator yields a forward (+X) lean, i.e. toward the initiation end.
    # Taylor (1941) requires the opposite (see the v4 tests below).
    assert layout.angle_convention == TAYLOR_SHAPIRO_LEGACY_ANGLE_CONVENTION
    assert np.all(directions[:, 0] > 0.0)


def test_taylor_shapiro_zero_spread_is_seed_independent() -> None:
    registry = PhysicsRegistry.from_file(V2_PHYSICS)
    case = load_json(V2_CASE)
    source = registry.fragment_source(
        case["warhead"]["fragment_source"]["model_id"]
    )
    layout = build_taylor_shapiro_ring_layout(
        case,
        source,
        case["warhead"]["fragment_source"]["total_count"],
    )
    first = _sample_taylor_shapiro_directions(
        layout,
        np.eye(3),
        np.random.default_rng(17),
        gurney_speed=2000.0,
        detonation_velocity_mps=6900.0,
        spread_sigma_deg=0.0,
    )
    second = _sample_taylor_shapiro_directions(
        layout,
        np.eye(3),
        np.random.default_rng(999),
        gurney_speed=2000.0,
        detonation_velocity_mps=6900.0,
        spread_sigma_deg=0.0,
    )

    np.testing.assert_array_equal(first, second)


def test_v2_registry_and_case_execute_and_fast_proxy_preserves_weight() -> None:
    engine = _engine(V2_PHYSICS)
    case = engine.prepare_case(load_json(V2_CASE))
    fragments = generate_fragments(
        case,
        engine.registry,
        np.random.default_rng(20261120),
    )
    source = engine.registry.fragment_source(
        case["warhead"]["fragment_source"]["model_id"]
    )
    packets = _weighted_fragment_packets(
        fragments,
        source,
        azimuths_per_ring=4,
    )

    assert len(fragments) == 60
    assert len(packets) == 24
    assert sum(weight for _, weight in packets) == pytest.approx(60.0)
    detonation = np.asarray(case["encounter"]["detonation_position_m"])
    offsets_body = np.asarray([fragment.origin_m - detonation for fragment in fragments])
    assert np.max(np.linalg.norm(offsets_body[:, 1:], axis=1)) > 0.04


def test_v3_obliquity_bound_is_versioned_and_reduces_only_certified_ood() -> None:
    v2_result = _engine(V2_PHYSICS).run(load_json(V2_CASE)).to_dict()
    v3_engine = _engine(V3_PHYSICS)
    v3_case = load_json(V3_CASE)
    v3_result = v3_engine.run(v3_case).to_dict()

    assert v3_case["parameter_set_id"] == v3_engine.registry.data["config_id"]
    assert v3_case["case_id"].endswith("template-v3")
    assert v2_result["penetration_domain_coverage"][
        "affected_replicate_fraction"
    ] > 0.0
    assert v3_result["penetration_domain_coverage"][
        "affected_replicate_fraction"
    ] == 0.0
    assert v3_result["scientific_completion_status"] == (
        "observed_domain_complete_not_statistically_demonstrated"
    )


def test_v2_layout_fails_closed_on_registry_profile_ring_drift() -> None:
    engine = _engine(V2_PHYSICS)
    registry_data = deepcopy(engine.registry.data)
    registry_data["fragment_source_models"][
        "loitering_small_front_taylor_shapiro_source"
    ]["direction"]["ring_count"] = 5
    drifted = DamageEngine(load_json(TARGET), PhysicsRegistry(registry_data))

    with pytest.raises(InputValidationError, match="ring_count must match"):
        validate_inputs(
            load_json(TARGET),
            drifted.prepare_case(load_json(V2_CASE)),
            drifted.registry,
            compiled_target=drifted.compiled_target,
        )


def test_v2_direction_has_an_explicit_case_feature_one_hot() -> None:
    v2_engine = _engine(V2_PHYSICS)
    v2_case = v2_engine.prepare_case(load_json(V2_CASE))
    values, _ = _case_features(v2_case, v2_engine)
    feature = dict(zip(CASE_FEATURE_NAMES, values))

    assert feature["direction_taylor_shapiro_ring"] == 1.0
    assert feature["direction_taylor_ring"] == 0.0
    assert sum(
        feature[name]
        for name in (
            "direction_isotropic",
            "direction_cone",
            "direction_taylor_ring",
            "direction_taylor_shapiro_ring",
        )
    ) == 1.0


def test_legacy_taylor_ring_seed_output_remains_bitwise_unchanged() -> None:
    registry = PhysicsRegistry.from_file(V1_PHYSICS)
    fragments = generate_fragments(
        load_json(V1_CASE),
        registry,
        np.random.default_rng(123456),
    )
    values = np.concatenate(
        [
            np.asarray(
                [[fragment.mass_kg, fragment.initial_speed_mps] for fragment in fragments]
            ),
            np.asarray([fragment.origin_m for fragment in fragments]),
            np.asarray([fragment.direction for fragment in fragments]),
        ],
        axis=1,
    ).astype("<f8")

    assert hashlib.sha256(values.tobytes()).hexdigest() == (
        "02f8a33c96ef112d4970ab4e64542af617915e69dfb74b7102243f1cb817cd33"
    )


def _taylor_lab_frame_signed_deflection(
    speed: float, detonation_velocity: float, wave_axial_sign: float
) -> float:
    """Independent oracle: Taylor's (1941) steady-flow construction.

    For a wave running along a side wall (phi1 = 90 deg), casing material in
    the wave frame enters at speed D along -t and turns outward through theta
    at constant speed, so V = 2 D sin(theta / 2) and the lab-frame velocity is
    D (1 - cos theta) t + D sin theta n.  Return its angle from n, signed
    toward increasing body polar angle (toward -X for a +X polar axis).
    """

    theta = 2.0 * math.asin(speed / (2.0 * detonation_velocity))
    axial = detonation_velocity * (1.0 - math.cos(theta)) * wave_axial_sign
    radial = detonation_velocity * math.sin(theta)
    return math.atan2(-axial, radial)


@pytest.mark.parametrize("wave_axial_sign", [1.0, -1.0])
def test_wave_directed_deflection_matches_taylor_lab_frame_construction(
    wave_axial_sign: float,
) -> None:
    speed, detonation_velocity = 250.0, 1000.0
    phi2 = 0.0 if wave_axial_sign > 0.0 else math.pi
    taylor, deflection = taylor_shapiro_deflection(
        speed, detonation_velocity, math.pi / 2.0, phi2
    )
    oracle = _taylor_lab_frame_signed_deflection(
        speed, detonation_velocity, wave_axial_sign
    )

    assert math.copysign(1.0, deflection) == math.copysign(1.0, oracle)
    # Shapiro uses tan(delta) = q while Taylor gives sin(delta) = q.
    assert deflection == pytest.approx(oracle, abs=2.0e-3)
    assert abs(deflection) <= taylor


def test_wave_directed_deflection_is_the_exact_mirror_of_the_legacy_sign() -> None:
    for phi1 in np.linspace(0.2, math.pi - 0.2, 7):
        for phi2 in np.linspace(0.0, math.pi, 9):
            _, legacy = taylor_shapiro_angles(1900.0, 6900.0, phi1, phi2)
            _, wave_directed = taylor_shapiro_deflection(1900.0, 6900.0, phi1, phi2)
            assert wave_directed == pytest.approx(-legacy, abs=1.0e-15)


@pytest.mark.parametrize("profile", ["small_front", "medium_front", "large_front"])
def test_v4_nose_initiated_fragments_lean_toward_detonation_propagation(
    profile: str,
) -> None:
    registry = PhysicsRegistry.from_file(V4_PHYSICS)
    case = load_json(V4_CASE)
    apply_munition_profile(case, profile)
    source_document = case["warhead"]["fragment_source"]
    source = registry.fragment_source(source_document["model_id"])
    layout = build_taylor_shapiro_ring_layout(
        case, source, source_document["total_count"]
    )
    speed = 2000.0
    detonation_velocity = float(source_document["detonation_velocity_mps"])
    directions = _sample_taylor_shapiro_directions(
        layout,
        np.eye(3),
        np.random.default_rng(3),
        gurney_speed=speed,
        detonation_velocity_mps=detonation_velocity,
        spread_sigma_deg=0.0,
    )

    assert layout.angle_convention == TAYLOR_SHAPIRO_WAVE_DIRECTED_ANGLE_CONVENTION
    normals = layout.local_normals_body
    waves = layout.wave_directions_body
    # The nose initiator sends the wave aft, so every fragment leans aft.
    assert np.all(waves @ layout.body_axis < 0.0)
    assert np.all(directions @ layout.body_axis < 0.0)
    # Convention-free statement of Taylor's result: the deviation from the
    # casing normal points along the wave's tangential component.
    wave_tangential = waves - np.einsum("ij,ij->i", waves, normals)[:, None] * normals
    deviation = directions - np.einsum("ij,ij->i", directions, normals)[:, None] * normals
    assert np.all(np.einsum("ij,ij->i", deviation, wave_tangential) > 0.0)
    taylor_bound = math.asin(speed / (2.0 * detonation_velocity))
    angles = np.arccos(np.clip(np.einsum("ij,ij->i", directions, normals), -1.0, 1.0))
    assert np.all(angles <= taylor_bound + 1.0e-12)


def test_v4_registry_and_case_execute_under_their_own_identity() -> None:
    v4_engine = _engine(V4_PHYSICS)
    v4_case = load_json(V4_CASE)
    result = v4_engine.run(v4_case).to_dict()

    assert v4_case["parameter_set_id"] == v4_engine.registry.data["config_id"]
    assert v4_case["case_id"].endswith("template-v4")
    assert v4_engine.registry.hash != _engine(V3_PHYSICS).registry.hash
    assert result["summary"]["total_fragments"]["mean"] == (
        v4_case["warhead"]["fragment_source"]["total_count"]
    )


def test_taylor_shapiro_layout_rejects_an_undeclared_angle_convention() -> None:
    registry = PhysicsRegistry.from_file(V4_PHYSICS)
    case = load_json(V4_CASE)
    source = deepcopy(
        registry.fragment_source(case["warhead"]["fragment_source"]["model_id"])
    )
    source["direction"]["angle_convention"] = "body_plus_x_polar_unsigned"
    with pytest.raises(InputValidationError, match="angle convention"):
        build_taylor_shapiro_ring_layout(
            case, source, case["warhead"]["fragment_source"]["total_count"]
        )
