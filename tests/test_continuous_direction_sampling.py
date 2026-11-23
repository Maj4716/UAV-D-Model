from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from damage_engine.continuous_direction_sampling import (
    CONTINUOUS_DIRECTION_IMPLEMENTATION_HASH,
    CONTINUOUS_DIRECTION_MEASURE_HASH,
    CONTINUOUS_REDUCED_7D,
    LEGACY_AIM_TANGENT_BASIS,
    REFERENCE_AXIS_AIM_TANGENT_BASIS,
    REFERENCE_AXIS_IMPLEMENTATION_HASH,
    REFERENCE_AXIS_MEASURE_HASH,
    _horizontal_direction,
    _top_direction,
    apply_continuous_direction_manifold,
    dynamic_tangent_basis,
    reference_axis_tangent_basis,
    validate_continuous_direction_manifold,
)
from damage_engine.case_factory import CaseFactory
from damage_engine.dataset import DatasetWriter, _sampling_context
from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.label_spec import LabelSpec
from damage_engine.registry import PhysicsRegistry
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.utils import content_hash


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _region(identifier: str, kind: str, centre: float | None = None) -> dict:
    value = {
        "region_id": identifier,
        "kind": kind,
        "selection_weight": 1,
        "aim_center_target_m": [0, 0, 0],
        "aim_u_range_m": [-1.0, 1.0],
        "aim_v_range_m": [-1.0, 1.0],
        "standoff_range_m": [8.0, 12.0],
        "speed_range_mps": [80.0, 160.0],
        "body_x_roll_range_rad": [-math.pi, math.pi],
    }
    if kind == "horizontal_sector":
        value.update(
            {
                "azimuth_center_deg": centre,
                "azimuth_half_width_deg": 30.0,
                "dive_range_deg": [0.0, 45.0],
            }
        )
    else:
        value.update(
            {
                "azimuth_range_deg": [0.0, 360.0],
                "vertical_offset_range_deg": [0.0, 45.0],
            }
        )
    return value


def _manifold() -> dict:
    return {
        "schema_version": "1.0.0",
        "sampler_id": "sphere-entry-continuous-direction-manifold@1.0.0",
        "mode": "sphere_entry_continuous_direction_with_body_x_roll",
        "approach_weight_semantics": "declared_design_mass_not_hit_probability_or_operational_prevalence",
        "latent_layout": CONTINUOUS_REDUCED_7D,
        "direction_measure": "uniform_solid_angle_within_region",
        "bottom_up_policy": "stress_only",
        "sphere": {
            "algorithm_id": "target-aabb-circumsphere@1.0.0",
            "object_scope": "all",
            "clearance_m": 0.01,
        },
        "regions": [
            _region("pos_x", "horizontal_sector", 0.0),
            _region("pos_y", "horizontal_sector", 90.0),
            _region("neg_x", "horizontal_sector", 180.0),
            _region("neg_y", "horizontal_sector", 270.0),
            _region("top", "top_cap"),
        ],
    }


def test_dynamic_tangent_basis_is_right_handed_and_orthonormal():
    direction = np.asarray([0.3, -0.4, -0.866025403784])
    direction /= np.linalg.norm(direction)
    tangent_u, tangent_v = dynamic_tangent_basis(direction)

    frame = np.column_stack((direction, tangent_u, tangent_v))
    assert frame.T @ frame == pytest.approx(np.eye(3), abs=1.0e-12)
    assert np.linalg.det(frame) == pytest.approx(1.0, abs=1.0e-12)
    assert np.cross(direction, tangent_u) == pytest.approx(tangent_v, abs=1.0e-12)


def test_continuous_manifold_records_solid_angle_measure_and_non_bottom_direction():
    target = _load(ROOT / "configs/targets/target.fragment_only.v1.json")
    registry = PhysicsRegistry.from_file(ROOT / "configs/physics/loitering_impact.json")
    engine = DamageEngine(target, registry)
    case = _load(ROOT / "configs/cases/loitering_impact.fragment_only.v1.json")
    record = apply_continuous_direction_manifold(
        case,
        _manifold(),
        [0.9, 0.25, 0.75, 0.4, 0.6, 0.5, 0.3],
        engine.compiled_target,
        target_hash=engine.target_hash,
    )

    assert record["region_id"] == "top"
    assert record["direction_target_xyz"][2] < 0.0
    assert record["region_solid_angle_sr"] == pytest.approx(
        2.0 * math.pi * (1.0 - math.cos(math.pi / 4.0))
    )
    assert record["measure_contract_hash"] == CONTINUOUS_DIRECTION_MEASURE_HASH
    assert record["bottom_up_policy"] == "stress_only"
    assert case["metadata"]["contact_manifold_sampling"]["status"] == (
        "reanchored_pending_s_contact"
    )


def test_sampling_spec_17_adds_two_physical_direction_coordinates_without_mutating_16():
    legacy_document = _load(
        ROOT / "configs/sampling/p0_sphere_entry_sobol_no_bottom_impact_state.v1.json"
    )
    legacy = SamplingSpec(legacy_document)
    document = json.loads(json.dumps(legacy_document))
    document["schema_version"] = "1.7.0"
    document["sampling_spec_id"] = "test-continuous-direction-v1"
    document["contact_conditioning"]["selection_declaration"][
        "selection_rule_id"
    ] = "test-continuous-direction-selection-v1"
    document["contact_manifold"] = _manifold()

    spec = SamplingSpec(document)

    assert spec.is_continuous_sphere_entry_ray is True
    assert spec.contact_manifold_latent_dimension == 7
    assert spec.latent_dimension == legacy.latent_dimension + 2
    assert spec.contact_manifold["measure_contract_hash"] == CONTINUOUS_DIRECTION_MEASURE_HASH
    assert legacy.schema_version == "1.6.0"


def test_reference_continuous_manifold_rejects_bottom_up_region_kind():
    document = _manifold()
    document["regions"][0]["kind"] = "bottom_up"
    with pytest.raises(Exception, match="region kind is unsupported"):
        validate_continuous_direction_manifold(document)


def test_case_factory_dispatches_sampling_spec_17_and_admits_all_five_regions():
    engine = DamageEngine(
        _load(ROOT / "configs/targets/target.rebuilt.json"),
        PhysicsRegistry.from_file(ROOT / "configs/physics/loitering_impact.json"),
    )
    base_case = _load(ROOT / "configs/cases/loitering_impact.v1.json")
    base_case["target_ref"]["content_hash"] = engine.target_hash
    document = _load(
        ROOT / "configs/sampling/p0_sphere_entry_continuous_direction_reference.v1.json"
    )
    document["budgets"].update(train_pilot=12, validation=3, test=3)
    for allocation in document["round_budgets"][0]["allocations"]:
        allocation["count"] = 12 if allocation["role"] == "reference_train" else 3
    spec = SamplingSpec(document)
    factory = CaseFactory(
        base_case,
        spec,
        LabelSpec.from_file(ROOT / "configs/labels/loitering_impact.v1.json"),
        target_hash=engine.target_hash,
        damage_tree_definition_hash=content_hash(engine.target_data["damage_tree"]),
        physics_registry_hash=engine.registry.hash,
        engine_code_hash=ENGINE_CODE_HASH,
        physics_registry=engine.registry,
        compiled_target=engine.compiled_target,
    )

    plan = factory.build_reference_plan()

    assert len(plan.cases) == 18
    assert plan.latent_design.shape[1] == 8
    assert {
        case["metadata"]["continuous_direction_sampling"]["region_id"]
        for case in plan.cases
    } == {"pos_x", "neg_x", "pos_y", "neg_y", "top_cap"}
    assert all(
        case["metadata"]["sphere_entry_ray_sampling"]["status"]
        == "admitted_s_contact"
        for case in plan.cases
    )


def test_sampling_17_taylor_shapiro_cases_build_contract_18_dataset(tmp_path):
    engine = DamageEngine(
        _load(ROOT / "configs/targets/target.fragment_only.v1.json"),
        PhysicsRegistry.from_file(
            ROOT / "configs/physics/loitering_impact.taylor_shapiro.v2.json"
        ),
    )
    document = _load(
        ROOT
        / "configs/sampling/p0_sphere_entry_continuous_direction_reference.v1.json"
    )
    document["sampling_spec_id"] += "-dataset-smoke"
    document["split_policy"]["registry_id"] += "-dataset-smoke"
    document["budgets"].update(train_pilot=12, validation=3, test=3)
    document["formal_policy"]["replicates"] = 2
    document["replicate_policy"]["default_replicates"] = 2
    for allocation in document["round_budgets"][0]["allocations"]:
        allocation["count"] = (
            12 if allocation["role"] == "reference_train" else 3
        )
    spec = SamplingSpec(document)
    label = LabelSpec.from_file(
        ROOT / "configs/labels/loitering_impact.fragment_only.hurdle.v2.json"
    )
    plan = CaseFactory(
        _load(
            ROOT
            / "configs/cases/loitering_impact.fragment_only.taylor_shapiro.v2.json"
        ),
        spec,
        label,
        target_hash=engine.target_hash,
        damage_tree_definition_hash=content_hash(engine.target_data["damage_tree"]),
        physics_registry_hash=engine.registry.hash,
        engine_code_hash=ENGINE_CODE_HASH,
        physics_registry=engine.registry,
        compiled_target=engine.compiled_target,
    ).build_reference_plan()

    records = _sampling_context(plan.cases)
    assert records is not None
    assert len(records) == 18
    assert {record["schema_version"] for record in records} == {"1.7.0"}
    assert all(record["sampling_spec_document"] == spec.data for record in records)

    manifest = DatasetWriter(tmp_path / "sampling-17-taylor-shapiro").run_cases(
        engine, plan.cases
    )
    assert manifest["schema_version"] == "1.7.0"
    assert manifest["tensor_contract"]["version"] == "1.8.0"
    assert manifest["generation"]["sampling"]["sampling_envelope_version"] == (
        "1.7.0"
    )
    assert manifest["source_contracts"]["sampling_spec_document"] == spec.data
    assert manifest["model_input_contract"]["tensor_field"] == (
        "loitering_impact_first_armor_state_x"
    )


# Frozen hashes of the archived 2026-11-22 recovery dataset plan.  The legacy
# basis must keep them so archived SamplingSpecs revalidate unchanged.
ARCHIVED_MEASURE_HASH = (
    "sha256:7eb2ff5ffbdf79cb155feac8c5839c8e1e419a166d099856acc96697255c7218"
)
ARCHIVED_IMPLEMENTATION_HASH = (
    "sha256:794b4c82170ab20a6743930a8901cb0bd52f03b8c8891cabc4e23c4b86392926"
)
RECOVERY_V1 = (
    ROOT
    / "configs/sampling"
    / "p0_sphere_entry_continuous_direction_recovery_train2000_eval300_role_r8_r32.v1.json"
)
RECOVERY_V2 = (
    ROOT
    / "configs/sampling"
    / "p0_sphere_entry_continuous_direction_recovery_train2000_eval300_role_r8_r32.v2.json"
)


def test_legacy_aim_basis_keeps_archived_measure_and_implementation_hashes():
    assert CONTINUOUS_DIRECTION_MEASURE_HASH == ARCHIVED_MEASURE_HASH
    assert CONTINUOUS_DIRECTION_IMPLEMENTATION_HASH == ARCHIVED_IMPLEMENTATION_HASH
    canonical = validate_continuous_direction_manifold(_manifold())
    assert "aim_tangent_basis" not in canonical
    assert canonical["measure_contract_hash"] == ARCHIVED_MEASURE_HASH


@pytest.mark.parametrize(
    ("direction", "kind"),
    [
        ([1, 0, 0], "horizontal_sector"),
        ([-1, 0, 0], "horizontal_sector"),
        ([0, 1, 0], "horizontal_sector"),
        ([0, -1, 0], "horizontal_sector"),
        ([0, 0, -1], "top_cap"),
    ],
)
def test_reference_axis_basis_reproduces_the_explicit_contact_pose_frames(
    direction, kind
):
    """The explicit approach frames declared in contact_pose.smoke.v1 are the oracle."""

    smoke = _load(ROOT / "configs/sampling/contact_pose.smoke.v1.json")
    declared = {
        tuple(approach["direction_target_xyz"]): approach
        for approach in smoke["contact_manifold"]["approaches"]
    }
    approach = declared[tuple(direction)]

    tangent_u, tangent_v = reference_axis_tangent_basis(direction, kind)

    assert tangent_u == pytest.approx(approach["tangent_u_target_xyz"], abs=1.0e-12)
    assert tangent_v == pytest.approx(approach["tangent_v_target_xyz"], abs=1.0e-12)


def test_reference_axis_basis_keeps_v_up_for_horizontal_and_forward_for_top():
    sector = {"azimuth_half_width_deg": 30.0, "dive_range_deg": [0.0, 45.0]}
    top = {"azimuth_range_deg": [0.0, 360.0], "vertical_offset_range_deg": [0.0, 45.0]}
    grid = np.linspace(0.01, 0.99, 9)
    floor = math.cos(math.radians(45.0)) - 1.0e-12
    for centre in (0.0, 90.0, 180.0, 270.0):
        for first in grid:
            for second in grid:
                direction, _ = _horizontal_direction(
                    {**sector, "azimuth_center_deg": centre}, first, second
                )
                tangent_u, tangent_v = reference_axis_tangent_basis(
                    direction, "horizontal_sector"
                )
                frame = np.column_stack((direction, tangent_u, tangent_v))
                assert frame.T @ frame == pytest.approx(np.eye(3), abs=1.0e-12)
                assert np.linalg.det(frame) == pytest.approx(1.0, abs=1.0e-12)
                assert np.cross(direction, tangent_u) == pytest.approx(
                    tangent_v, abs=1.0e-12
                )
                assert tangent_v[2] >= floor
                assert abs(tangent_u[2]) <= 1.0e-12
    for first in grid:
        for second in grid:
            direction, _ = _top_direction(top, first, second)
            tangent_u, tangent_v = reference_axis_tangent_basis(direction, "top_cap")
            assert np.cross(direction, tangent_u) == pytest.approx(
                tangent_v, abs=1.0e-12
            )
            assert tangent_v[1] >= floor


def test_legacy_basis_flips_v_across_mirror_sectors_and_reference_basis_mirrors():
    """Mirror symmetry x -> -x must map an aim-window frame to its mirror image."""

    pos_x = np.asarray(
        [math.cos(math.radians(20.0)), 0.3, -math.sin(math.radians(30.0))]
    )
    pos_x /= np.linalg.norm(pos_x)
    mirror = np.asarray([-1.0, 1.0, 1.0])
    neg_x = pos_x * mirror
    _, legacy_pos = dynamic_tangent_basis(pos_x)
    _, legacy_neg = dynamic_tangent_basis(neg_x)
    _, fixed_pos = reference_axis_tangent_basis(pos_x, "horizontal_sector")
    _, fixed_neg = reference_axis_tangent_basis(neg_x, "horizontal_sector")

    assert math.copysign(1.0, legacy_pos[2]) != math.copysign(1.0, legacy_neg[2])
    assert fixed_pos[2] > 0.0 and fixed_neg[2] > 0.0
    assert fixed_neg == pytest.approx(fixed_pos * mirror, abs=1.0e-12)


def test_reference_axis_manifold_records_its_own_contract_identity():
    document = _manifold()
    document["aim_tangent_basis"] = REFERENCE_AXIS_AIM_TANGENT_BASIS
    canonical = validate_continuous_direction_manifold(document)
    assert canonical["aim_tangent_basis"] == REFERENCE_AXIS_AIM_TANGENT_BASIS
    assert canonical["measure_contract_hash"] == REFERENCE_AXIS_MEASURE_HASH
    assert canonical["implementation_hash"] == REFERENCE_AXIS_IMPLEMENTATION_HASH
    assert REFERENCE_AXIS_MEASURE_HASH != CONTINUOUS_DIRECTION_MEASURE_HASH

    engine = DamageEngine(
        _load(ROOT / "configs/targets/target.fragment_only.v1.json"),
        PhysicsRegistry.from_file(ROOT / "configs/physics/loitering_impact.json"),
    )
    case = _load(ROOT / "configs/cases/loitering_impact.fragment_only.v1.json")
    record = apply_continuous_direction_manifold(
        case,
        document,
        [0.1, 0.25, 0.75, 0.4, 0.6, 0.5, 0.3],
        engine.compiled_target,
        target_hash=engine.target_hash,
    )
    assert record["aim_tangent_basis"] == REFERENCE_AXIS_AIM_TANGENT_BASIS
    assert record["measure_contract_hash"] == REFERENCE_AXIS_MEASURE_HASH
    assert record["tangent_v_target_xyz"][2] > 0.0


def test_manifold_rejects_mismatched_basis_hash_and_unknown_basis():
    document = _manifold()
    document["aim_tangent_basis"] = REFERENCE_AXIS_AIM_TANGENT_BASIS
    document["measure_contract_hash"] = CONTINUOUS_DIRECTION_MEASURE_HASH
    with pytest.raises(Exception, match="measure hash is inconsistent"):
        validate_continuous_direction_manifold(document)

    document = _manifold()
    document["aim_tangent_basis"] = "least_aligned_axis@0.0.1"
    with pytest.raises(Exception, match="aim tangent basis is unsupported"):
        validate_continuous_direction_manifold(document)
    assert LEGACY_AIM_TANGENT_BASIS != REFERENCE_AXIS_AIM_TANGENT_BASIS


def test_recovery_sampling_spec_v2_declares_the_reference_axis_basis():
    v1 = SamplingSpec.from_file(RECOVERY_V1)
    v2 = SamplingSpec.from_file(RECOVERY_V2)

    assert "aim_tangent_basis" not in v1.contact_manifold
    assert v1.contact_manifold["measure_contract_hash"] == ARCHIVED_MEASURE_HASH
    assert v2.contact_manifold["aim_tangent_basis"] == REFERENCE_AXIS_AIM_TANGENT_BASIS
    assert v2.contact_manifold["measure_contract_hash"] == REFERENCE_AXIS_MEASURE_HASH
    assert v2.data["seeds"] != v1.data["seeds"]
    assert v2.data["sampling_spec_id"] != v1.data["sampling_spec_id"]
