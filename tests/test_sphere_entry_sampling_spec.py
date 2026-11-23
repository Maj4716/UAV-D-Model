from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

from damage_engine.case_factory import CaseFactory
from damage_engine.contact_manifold_sampling import apply_contact_manifold
from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.label_spec import LabelSpec
from damage_engine.learning_input_identity import (
    LOITERING_IMPACT_FIRST_ARMOR_STATE_MODEL_INPUT_DEFINITION,
)
from damage_engine.registry import PhysicsRegistry
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.sphere_entry_ray_sampling import (
    PAIRED_LEGACY_STANDOFF_6D,
    REDUCED_PHYSICAL_RAY_5D,
    apply_sphere_entry_reanchor,
    map_sphere_entry_latent,
)
from damage_engine.utils import content_hash, load_json


ROOT = Path(__file__).resolve().parents[1]
LEGACY_SPEC_PATH = ROOT / "configs" / "sampling" / "p0_contact_reference.json"
SPHERE_SPEC_PATH = (
    ROOT / "configs" / "sampling" / "p0_sphere_entry_reference.v1.json"
)
REDUCED_SPEC_PATH = (
    ROOT / "configs" / "sampling" / "p0_sphere_entry_reduced_5d.v1.json"
)
SOBOL_SPEC_PATH = (
    ROOT / "configs" / "sampling" / "p0_sphere_entry_sobol_reference.v1.json"
)
NO_BOTTOM_IMPACT_STATE_SPEC_PATH = (
    ROOT
    / "configs"
    / "sampling"
    / "p0_sphere_entry_sobol_no_bottom_impact_state.v1.json"
)


def test_sampling_spec_15_binds_sphere_entry_without_removing_dummy_standoff() -> None:
    spec = SamplingSpec.from_file(SPHERE_SPEC_PATH)

    assert spec.schema_version == "1.5.0"
    assert spec.is_sphere_entry_ray
    assert spec.has_contact_manifold
    assert spec.independent_latent_dimension == 1
    assert spec.contact_manifold_latent_dimension == 6
    assert spec.latent_dimension == 7
    assert spec.contact_manifold is not None
    assert spec.contact_manifold["sampler_id"] == (
        "sphere-entry-ray-manifold@1.0.0"
    )
    assert spec.contact_manifold["latent_layout"] == (
        "paired_legacy_standoff_6d"
    )
    assert spec.contact_manifold["sphere"]["clearance_m"] == 0.01
    assert spec.training_design_document["contract_mode"] == (
        "loitering_impact_sphere_entry_contact_pose_1.5"
    )


def test_sampling_spec_15_dispatch_is_strict_and_legacy_hash_is_unchanged() -> None:
    legacy = SamplingSpec.from_file(LEGACY_SPEC_PATH)
    assert legacy.hash == (
        "sha256:4589489789fbb0389f7da8ce1c5db5bd0162b70ba1933838e8edfe76b9c1062b"
    )
    assert legacy.latent_dimension == 7
    assert not legacy.is_sphere_entry_ray
    assert legacy.training_design_document["contract_mode"] == (
        "loitering_impact_contact_pose_1.4"
    )

    document = load_json(SPHERE_SPEC_PATH)
    unknown = deepcopy(document)
    unknown["contact_manifold"]["sampler_id"] = "unknown-ray-sampler@1.0.0"
    with pytest.raises(InputValidationError, match="schema validation"):
        SamplingSpec(unknown)

    downgraded = deepcopy(document)
    downgraded["schema_version"] = "1.4.0"
    with pytest.raises(InputValidationError, match="schema validation"):
        SamplingSpec(downgraded)

    legacy_as_15 = load_json(LEGACY_SPEC_PATH)
    legacy_as_15["schema_version"] = "1.5.0"
    with pytest.raises(InputValidationError, match="schema validation"):
        SamplingSpec(legacy_as_15)


def test_case_factory_dispatches_15_through_legacy_transverse_measure_then_sphere() -> None:
    engine = DamageEngine(
        load_json(ROOT / "configs" / "targets" / "target.rebuilt.json"),
        PhysicsRegistry.from_file(
            ROOT / "configs" / "physics" / "loitering_impact.json"
        ),
    )
    base_case = load_json(ROOT / "configs" / "cases" / "loitering_impact.v1.json")
    base_case["target_ref"]["content_hash"] = engine.target_hash
    label_spec = LabelSpec.from_file(
        ROOT / "configs" / "labels" / "loitering_impact.v1.json"
    )
    document = load_json(
        ROOT / "configs" / "sampling" / "contact_pose.smoke.v1.json"
    )
    document["schema_version"] = "1.5.0"
    document["sampling_spec_id"] = "sphere-entry-case-factory-smoke-v1"
    document["revision"] = "1.0.0-phase1-paired"
    document["contact_conditioning"]["selection_declaration"][
        "selection_rule_id"
    ] = "sphere-entry-contact-input-admission-v1"
    approaches = document["contact_manifold"]["approaches"]
    document["contact_manifold"] = {
        "schema_version": "1.0.0",
        "sampler_id": "sphere-entry-ray-manifold@1.0.0",
        "mode": "sphere_entry_ray_with_body_x_roll",
        "approach_weight_semantics": (
            "declared_design_mass_not_hit_probability_or_operational_prevalence"
        ),
        "latent_layout": "paired_legacy_standoff_6d",
        "sphere": {
            "algorithm_id": "target-aabb-circumsphere@1.0.0",
            "object_scope": "all",
            "clearance_m": 0.01,
        },
        "approaches": approaches,
    }
    spec = SamplingSpec(document)
    factory = CaseFactory(
        base_case,
        spec,
        label_spec,
        target_hash=engine.target_hash,
        damage_tree_definition_hash=content_hash(engine.target_data["damage_tree"]),
        physics_registry_hash=engine.registry.hash,
        engine_code_hash=ENGINE_CODE_HASH,
        physics_registry=engine.registry,
        compiled_target=engine.compiled_target,
    )

    plan = factory.build_reference_plan()

    assert len(plan.cases) == 18
    assert plan.latent_design.shape == (len(plan.candidate_rows), 7)
    assert all(len(row["latent"]) == 7 for row in plan.candidate_rows)
    for case in plan.cases:
        sphere_record = case["metadata"]["sphere_entry_ray_sampling"]
        contact_record = case["metadata"]["contact_manifold_sampling"]
        assert sphere_record["status"] == "admitted_s_contact"
        assert contact_record["status"] == "admitted_s_contact"
        assert sphere_record["latent_layout"] == "paired_legacy_standoff_6d"
        assert sphere_record["no_op_latent_names"] == ["standoff_quantile"]
        assert contact_record["trajectory_origin_superseded_by"] == (
            "sphere-entry-ray-manifold@1.0.0"
        )
        assert sphere_record["s_contact_contract_hash"] == case["metadata"][
            "sampling"
        ]["contact_conditioning"]["contract_hash"]


def test_sampling_spec_16_removes_only_the_dummy_standoff_latent() -> None:
    paired = SamplingSpec.from_file(SPHERE_SPEC_PATH)
    reduced = SamplingSpec.from_file(REDUCED_SPEC_PATH)

    assert paired.hash == (
        "sha256:ca7bebc8e5af1b96f756f0b2fdbdfe05d118d87c01e5c04861d149eb52590024"
    )
    assert paired.latent_dimension == 7
    assert reduced.schema_version == "1.6.0"
    assert reduced.is_sphere_entry_ray
    assert reduced.is_reduced_sphere_entry_ray
    assert reduced.independent_latent_dimension == 1
    assert reduced.contact_manifold_latent_dimension == 5
    assert reduced.latent_dimension == 6
    assert reduced.contact_manifold is not None
    assert reduced.contact_manifold["latent_layout"] == REDUCED_PHYSICAL_RAY_5D
    assert reduced.training_design_document["contract_mode"] == (
        "loitering_impact_sphere_entry_contact_pose_1.6"
    )

    reduced_as_15 = load_json(REDUCED_SPEC_PATH)
    reduced_as_15["schema_version"] = "1.5.0"
    with pytest.raises(InputValidationError, match="schema validation"):
        SamplingSpec(reduced_as_15)

    paired_as_16 = load_json(SPHERE_SPEC_PATH)
    paired_as_16["schema_version"] = "1.6.0"
    with pytest.raises(InputValidationError, match="schema validation"):
        SamplingSpec(paired_as_16)


def test_sobol_sampling_revision_is_explicit_and_leaves_16_frozen() -> None:
    reduced = SamplingSpec.from_file(REDUCED_SPEC_PATH)
    sobol = SamplingSpec.from_file(SOBOL_SPEC_PATH)

    assert reduced.hash == (
        "sha256:dcb28b9a46d9802653150800fe8232f24aff411e5d1439c23684a5baf1dca056"
    )
    assert reduced.design_sequence is None
    assert sobol.schema_version == "1.6.0"
    assert sobol.design_sequence == {
        "schema_version": "unit-cube-design@1.0.0",
        "method": "scrambled_sobol",
        "outcome_blind": True,
        "non_power_of_two_policy": (
            "pad_to_next_power_of_two_then_take_prefix"
        ),
    }
    assert sobol.training_design_document["design_sequence"] == (
        sobol.design_sequence
    )

    invalid = load_json(SOBOL_SPEC_PATH)
    invalid["design_sequence"]["outcome_blind"] = False
    with pytest.raises(InputValidationError, match="schema validation"):
        SamplingSpec(invalid)


def test_explicit_design_sequence_reports_missing_sampling_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "damage_engine.design_sequences", None)

    with pytest.raises(InputValidationError, match=r"install.*\[sampling\]"):
        SamplingSpec(load_json(SOBOL_SPEC_PATH))


def test_case_factory_dispatches_reduced_5d_and_preserves_the_physical_ray() -> None:
    engine = DamageEngine(
        load_json(ROOT / "configs" / "targets" / "target.rebuilt.json"),
        PhysicsRegistry.from_file(
            ROOT / "configs" / "physics" / "loitering_impact.json"
        ),
    )
    base_case = load_json(ROOT / "configs" / "cases" / "loitering_impact.v1.json")
    base_case["target_ref"]["content_hash"] = engine.target_hash
    label_spec = LabelSpec.from_file(
        ROOT / "configs" / "labels" / "loitering_impact.v1.json"
    )
    document = load_json(REDUCED_SPEC_PATH)
    document["budgets"] = {
        "train_pilot": 6,
        "validation": 6,
        "test": 6,
        "rare_stress_test": 0,
    }
    for allocation in document["round_budgets"][0]["allocations"]:
        allocation["count"] = 6
    document["formal_policy"]["replicates"] = 2
    document["replicate_policy"]["default_replicates"] = 2
    document["legal_sampling"]["max_attempts_per_allocation"] = 120
    spec = SamplingSpec(document)
    factory = CaseFactory(
        base_case,
        spec,
        label_spec,
        target_hash=engine.target_hash,
        damage_tree_definition_hash=content_hash(engine.target_data["damage_tree"]),
        physics_registry_hash=engine.registry.hash,
        engine_code_hash=ENGINE_CODE_HASH,
        physics_registry=engine.registry,
        compiled_target=engine.compiled_target,
    )

    plan = factory.build_reference_plan()

    assert len(plan.cases) == 18
    assert plan.latent_design.shape == (len(plan.candidate_rows), 6)
    first_case = plan.cases[0]
    reduced_record = first_case["metadata"]["sphere_entry_ray_sampling"]
    assert reduced_record["latent_layout"] == REDUCED_PHYSICAL_RAY_5D
    assert reduced_record["no_op_latent_names"] == []

    physical = reduced_record["physical_latent"]
    paired_latent = [physical[0], physical[1], physical[2], 0.13, physical[3], physical[4]]
    paired_mapping = map_sphere_entry_latent(
        paired_latent,
        layout=PAIRED_LEGACY_STANDOFF_6D,
    )
    paired_case = load_json(
        ROOT / "configs" / "cases" / "loitering_impact.v1.json"
    )
    paired_manifold = load_json(SPHERE_SPEC_PATH)["contact_manifold"]
    legacy_manifold = {
        "schema_version": "1.0.0",
        "sampler_id": "target-facing-ray-manifold@1.0.0",
        "mode": "target_facing_ray_with_body_x_roll",
        "approach_weight_semantics": (
            "declared_design_mass_not_hit_probability_or_operational_prevalence"
        ),
        "approaches": paired_manifold["approaches"],
    }
    contact_record = apply_contact_manifold(
        paired_case,
        legacy_manifold,
        paired_mapping.legacy_contact_manifold_latent,
    )
    paired_record = apply_sphere_entry_reanchor(
        paired_case,
        engine.compiled_target,
        target_hash=engine.target_hash,
        clearance_m=0.01,
        object_scope="all",
        latent_layout=PAIRED_LEGACY_STANDOFF_6D,
        latent=paired_latent,
        line_point_target_m=contact_record["aim_point_target_m"],
    )
    assert paired_record["physical_ray_id"] == reduced_record["physical_ray_id"]
    assert paired_case["encounter"]["trajectory_origin_m"] == first_case["encounter"][
        "trajectory_origin_m"
    ]


def test_contact_conditioned_production_plan_uses_declared_scrambled_sobol() -> None:
    engine = DamageEngine(
        load_json(ROOT / "configs" / "targets" / "target.rebuilt.json"),
        PhysicsRegistry.from_file(
            ROOT / "configs" / "physics" / "loitering_impact.json"
        ),
    )
    base_case = load_json(ROOT / "configs" / "cases" / "loitering_impact.v1.json")
    base_case["target_ref"]["content_hash"] = engine.target_hash
    label_spec = LabelSpec.from_file(
        ROOT / "configs" / "labels" / "loitering_impact.v1.json"
    )
    document = load_json(SOBOL_SPEC_PATH)
    document["budgets"] = {
        "train_pilot": 6,
        "validation": 6,
        "test": 6,
        "rare_stress_test": 0,
    }
    for allocation in document["round_budgets"][0]["allocations"]:
        allocation["count"] = 6
    document["formal_policy"]["replicates"] = 2
    document["replicate_policy"]["default_replicates"] = 2
    document["legal_sampling"]["max_attempts_per_allocation"] = 120
    spec = SamplingSpec(document)

    def build_plan():
        return CaseFactory(
            base_case,
            spec,
            label_spec,
            target_hash=engine.target_hash,
            damage_tree_definition_hash=content_hash(
                engine.target_data["damage_tree"]
            ),
            physics_registry_hash=engine.registry.hash,
            engine_code_hash=ENGINE_CODE_HASH,
            physics_registry=engine.registry,
            compiled_target=engine.compiled_target,
        ).build_reference_plan()

    first = build_plan()
    repeated = build_plan()

    assert len(first.cases) == 18
    assert np.array_equal(first.latent_design, repeated.latent_design)
    assert [row["candidate_id"] for row in first.candidate_rows] == [
        row["candidate_id"] for row in repeated.candidate_rows
    ]
    assert np.all(first.latent_design > 0.0)
    assert np.all(first.latent_design < 1.0)
    assert all(
        row["design_sequence"]["method"] == "scrambled_sobol"
        and row["design_sequence"]["outcome_blind"] is True
        for row in first.candidate_rows
    )
    assert any(
        row["design_sequence"]["prefix_truncated"] is True
        and row["design_sequence"]["requested_sample_count"] == 6
        and row["design_sequence"]["generated_sample_count"] == 8
        for row in first.candidate_rows
    )
    assert all(
        case["metadata"]["sampling"]["design_sequence"]["method"]
        == "scrambled_sobol"
        for case in first.cases
    )

    hashes_by_split = {
        split: {
            row["learning_input_hash"]
            for row in first.candidate_rows
            if row["selected"] and row["split"] == split
        }
        for split in ("train", "validation", "test")
    }
    assert hashes_by_split["train"].isdisjoint(hashes_by_split["validation"])
    assert hashes_by_split["train"].isdisjoint(hashes_by_split["test"])
    assert hashes_by_split["validation"].isdisjoint(hashes_by_split["test"])


def test_no_bottom_reference_plan_uses_first_armor_identity_without_changing_six_direction_spec() -> None:
    six_direction = SamplingSpec.from_file(SOBOL_SPEC_PATH)
    no_bottom = SamplingSpec.from_file(NO_BOTTOM_IMPACT_STATE_SPEC_PATH)

    assert six_direction.hash == (
        "sha256:0ff42b8e380aa0f8b8af70fc0690dc2a0c29d2bae4c768b7cec02b929000c169"
    )
    assert no_bottom.hash == (
        "sha256:1c9aa0a04832000311e2c8934ddd406224b7cd66ea7a4fc0664c9bc87d29eaf6"
    )
    assert no_bottom.is_first_armor_state_model_input
    assert no_bottom.training_design_document["contract_mode"] == (
        "loitering_impact_sphere_entry_first_armor_state_1.6"
    )
    assert [
        approach["approach_id"]
        for approach in no_bottom.contact_manifold["approaches"]
    ] == [
        "left_to_right_pos_x",
        "right_to_left_neg_x",
        "rear_to_front_pos_y",
        "front_to_rear_neg_y",
        "top_down_neg_z",
    ]
    assert any(
        approach["approach_id"] == "bottom_up_pos_z"
        for approach in six_direction.contact_manifold["approaches"]
    )

    engine = DamageEngine(
        load_json(ROOT / "configs" / "targets" / "target.rebuilt.json"),
        PhysicsRegistry.from_file(
            ROOT / "configs" / "physics" / "loitering_impact.json"
        ),
    )
    base_case = load_json(ROOT / "configs" / "cases" / "loitering_impact.v1.json")
    base_case["target_ref"]["content_hash"] = engine.target_hash
    label_spec = LabelSpec.from_file(
        ROOT / "configs" / "labels" / "loitering_impact.v1.json"
    )
    document = no_bottom.data
    document["budgets"] = {
        "train_pilot": 6,
        "validation": 6,
        "test": 6,
        "rare_stress_test": 0,
    }
    for allocation in document["round_budgets"][0]["allocations"]:
        allocation["count"] = 6
    document["formal_policy"]["replicates"] = 2
    document["replicate_policy"]["default_replicates"] = 2
    document["legal_sampling"]["max_attempts_per_allocation"] = 120
    smoke_spec = SamplingSpec(document)
    plan = CaseFactory(
        base_case,
        smoke_spec,
        label_spec,
        target_hash=engine.target_hash,
        damage_tree_definition_hash=content_hash(engine.target_data["damage_tree"]),
        physics_registry_hash=engine.registry.hash,
        engine_code_hash=ENGINE_CODE_HASH,
        physics_registry=engine.registry,
        compiled_target=engine.compiled_target,
    ).build_reference_plan()

    definition = LOITERING_IMPACT_FIRST_ARMOR_STATE_MODEL_INPUT_DEFINITION
    assert len(plan.cases) == 18
    assert all(len(definition.vector(case)) == 16 for case in plan.cases)
    assert all(
        row["learning_input_hash"]
        == definition.learning_input_fingerprint(case)
        for row, case in zip(
            (row for row in plan.candidate_rows if row["selected"]),
            plan.cases,
            strict=True,
        )
    )
    hashes_by_split = {
        split: {
            row["learning_input_hash"]
            for row in plan.candidate_rows
            if row["selected"] and row["split"] == split
        }
        for split in ("train", "validation", "test")
    }
    assert hashes_by_split["train"].isdisjoint(hashes_by_split["validation"])
    assert hashes_by_split["train"].isdisjoint(hashes_by_split["test"])
    assert hashes_by_split["validation"].isdisjoint(hashes_by_split["test"])
