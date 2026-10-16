from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

from damage_engine.case_factory import CaseFactory, _validate_fixed_scenario_plan
from damage_engine.dataset import DatasetWriter
from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.label_spec import LabelSpec
from damage_engine.learning_input_identity import (
    UAV_SAMPLING_MODEL_INPUT_CONTRACT,
    UAV_STATE_FEATURE_NAMES,
    fixed_scenario_fingerprint,
    learning_input_fingerprint,
    uav_state_vector,
)
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.torch_dataset import TorchDamageDataset
from damage_engine.training_view import (
    TrainingViewDataset,
    UAV_ROTATION_6D_FEATURES,
    fit_training_view_artifact,
    transform_uav_state,
)
from damage_engine.utils import content_hash


def test_uav_state_is_relative_canonical_and_separate_from_scenario(case_data) -> None:
    case = deepcopy(case_data)
    case["encounter"]["detonation_position_m"] = [1.0, -2.0, 3.0]
    case["encounter"]["velocity_target_mps"] = [4.0, -5.0, 6.0]
    root_half = math.sqrt(0.5)
    case["encounter"]["body_to_target_orientation_xyzw"] = [
        0.0, 0.0, root_half, root_half
    ]

    actual = uav_state_vector(case)
    np.testing.assert_allclose(
        actual,
        [-1.0, 2.0, -3.0, -4.0, 5.0, -6.0, 0.0, 0.0, -root_half, root_half],
    )

    equivalent = deepcopy(case)
    equivalent["encounter"]["body_to_target_orientation_xyzw"] = [
        -value for value in equivalent["encounter"]["body_to_target_orientation_xyzw"]
    ]
    assert learning_input_fingerprint(equivalent) == learning_input_fingerprint(case)

    different_teacher = deepcopy(case)
    different_teacher["warhead"]["blast_source"]["equivalent_mass_kg"] *= 1.01
    assert learning_input_fingerprint(different_teacher) == learning_input_fingerprint(case)
    assert fixed_scenario_fingerprint(different_teacher) != fixed_scenario_fingerprint(case)


def test_uav_training_transform_is_position_velocity_plus_rotation6d(case_data) -> None:
    values = np.asarray(uav_state_vector(case_data), dtype=np.float32)
    transformed, mask, names, units, transforms = transform_uav_state(
        values,
        np.ones(len(values), dtype=bool),
        UAV_STATE_FEATURE_NAMES,
        ("m", "m", "m", "m/s", "m/s", "m/s", "1", "1", "1", "1"),
    )
    assert transformed.shape == (12,)
    assert mask.tolist() == [True] * 12
    assert names[:6] == list(UAV_STATE_FEATURE_NAMES[:6])
    assert names[6:] == list(UAV_ROTATION_6D_FEATURES)
    assert units == ["m", "m", "m", "m/s", "m/s", "m/s"] + ["1"] * 6
    assert transforms[-1]["kind"] == "quaternion_xyzw_to_rotation_matrix_first_two_columns"


def _label_spec(engine: DamageEngine) -> LabelSpec:
    return LabelSpec(
        {
            "schema_version": "1.0.0",
            "label_spec_id": "uav-input-labels",
            "revision": "1",
            "status": "provisional",
            "source": {
                "damage_tree_id": "demo-tree",
                "damage_tree_revision": engine.damage_tree.version,
                "damage_tree_definition_hash": content_hash(engine.target_data["damage_tree"]),
            },
            "tasks": {
                "system_events": {
                    "type": "multi_label_event_probability",
                    "outputs": list(engine.damage_tree.outputs),
                    "loss": "binary_cross_entropy",
                }
            },
            "provenance": {"approval_status": "not_approved", "evidence_ref": None},
        }
    )


def _sampling_spec() -> SamplingSpec:
    position_names = list(UAV_STATE_FEATURE_NAMES[:3])
    velocity_names = list(UAV_STATE_FEATURE_NAMES[3:6])
    return SamplingSpec(
        {
            "schema_version": "1.2.0",
            "model_input_contract": UAV_SAMPLING_MODEL_INPUT_CONTRACT,
            "sampling_spec_id": "uav-input-p0",
            "revision": "1",
            "status": "provisional",
            "reference_distribution": {
                "distribution_id": "uav-input-reference",
                "interpretation": "design_distribution",
                "variables": [
                    {
                        "name": "relative_position",
                        "path": "/encounter/detonation_position_m",
                        "distribution": "sphere_shell",
                        "center": [0.0, 0.0, 0.0],
                        "radius": 2.0,
                    },
                    {
                        "name": "uav_attitude",
                        "path": "/encounter/body_to_target_orientation_xyzw",
                        "distribution": "haar_quaternion",
                    },
                ],
                "constraints": [],
            },
            "budgets": {"train_pilot": 3, "validation": 1, "test": 1, "rare_stress_test": 0},
            "seeds": {"design": 11, "selection": 12, "split": 13, "screening": 14, "formal": 15},
            "formal_policy": {"replicates": 2, "adaptive_ci_half_width": None, "epistemic_policy": "unique_per_lineage"},
            "split_policy": {"registry_id": "uav-input-splits", "freeze_before_outcomes": True, "adaptive_train_only": True},
            "quality_gates": {"positivity_min": 0.1, "min_ess_fraction": 0.5, "max_weight_cv": 2.0},
            "feature_roles": [
                {
                    "feature_id": "relative_position",
                    "path": "/encounter/detonation_position_m",
                    "tensor_feature_names": position_names,
                    "role": "learnable_variable",
                    "availability_stage": "raw_input",
                    "deployment_available": True,
                },
                {
                    "feature_id": "uav_velocity",
                    "path": "/encounter/velocity_target_mps",
                    "tensor_feature_names": velocity_names,
                    "role": "fixed_context",
                    "availability_stage": "raw_input",
                    "deployment_available": True,
                },
                {
                    "feature_id": "uav_attitude",
                    "path": "/encounter/body_to_target_orientation_xyzw",
                    "tensor_feature_names": list(UAV_ROTATION_6D_FEATURES),
                    "role": "learnable_variable",
                    "availability_stage": "raw_input",
                    "deployment_available": True,
                },
            ],
            "variable_blocks": [],
            "geometry_strata": [],
            "stream_roles": [
                {"role": "reference_train", "stream": "reference", "split": "train", "train_eligible": True},
                {"role": "reference_validation", "stream": "reference", "split": "validation", "train_eligible": False},
                {"role": "reference_test", "stream": "reference", "split": "test", "train_eligible": False},
            ],
            "round_budgets": [
                {"round": 0, "allocations": [
                    {"role": "reference_validation", "stratum_id": "p0", "proposal_id": "uav-input-reference", "count": 1},
                    {"role": "reference_test", "stratum_id": "p0", "proposal_id": "uav-input-reference", "count": 1},
                    {"role": "reference_train", "stratum_id": "p0", "proposal_id": "uav-input-reference", "count": 3},
                ]}
            ],
            "replicate_policy": {"default_replicates": 2, "outcome_adaptive": False, "overrides": []},
            "key_projections": [
                {"projection_id": "uav-pose", "paths": ["/encounter/detonation_position_m", "/encounter/body_to_target_orientation_xyzw"], "purpose": "coverage_audit", "pre_label_only": True}
            ],
            "proxy_scope": {"scope": "fixed_target", "geometry_boundary": "end_to_end", "target_identity_policy": "fixed_target_hash", "deployment_features_match_training": True, "evidence_status": "unverified"},
            "legal_sampling": {"invalid_candidate_policy": "fail_closed", "max_attempts_per_allocation": 100},
        }
    )


def test_contract_18_default_loader_exposes_only_uav_state(
    tmp_path, target_data, case_data, registry
) -> None:
    engine = DamageEngine(target_data, registry)
    spec = _sampling_spec()
    plan = CaseFactory(
        deepcopy(case_data),
        spec,
        _label_spec(engine),
        target_hash=engine.target_hash,
        damage_tree_definition_hash=content_hash(engine.target_data["damage_tree"]),
        physics_registry_hash=engine.registry.hash,
        engine_code_hash=ENGINE_CODE_HASH,
    ).build_reference_plan()

    output = tmp_path / "uav-contract"
    manifest = DatasetWriter(output).run_cases(engine, plan.cases)
    assert manifest["schema_version"] == "1.7.0"
    assert manifest["tensor_contract"]["version"] == "1.8.0"
    assert manifest["model_input_contract"]["tensor_field"] == "uav_state_x"
    assert manifest["source_contracts"]["fixed_scenario_hash"]

    with np.load(output / "tensors.npz", allow_pickle=False) as tensors:
        assert tensors["uav_state_x"].shape == (5, 10)
        assert tensors["uav_state_feature_names"].tolist() == list(UAV_STATE_FEATURE_NAMES)
        assert not any("warhead" in name for name in tensors["uav_state_feature_names"].tolist())
        assert "case_physics_x" in tensors.files  # teacher/audit compatibility only

    dataset = TorchDamageDataset(output)
    sample = dataset[0]
    assert "uav_state_x" in sample
    assert "uav_state_feature_mask" in sample
    assert "case_x" not in sample
    assert "case_physics_x" not in sample
    assert "solver_context_x" not in sample
    for name in (
        "component_x",
        "component_feature_mask",
        "armor_x",
        "armor_feature_mask",
        "component_armor_path_x",
        "component_armor_path_mask",
        "armor_vertices_m",
        "armor_vertex_offsets",
        "component_vertices_m",
        "component_vertex_offsets",
        "tree_edge_index",
        "tree_edge_type",
    ):
        assert name not in sample
    assert dataset.include_static_graph is False
    assert dataset.metadata["uav_state_feature_names"] == UAV_STATE_FEATURE_NAMES

    # Fixed teacher/scenario tensors remain archived for explicit audit only;
    # requesting them is a deliberate opt-in rather than the model default.
    audit_dataset = TorchDamageDataset(
        output,
        fields=("uav_state_x", "component_x", "armor_x", "component_armor_path_x"),
        include_static_graph=True,
    )
    audit_sample = audit_dataset[0]
    assert "component_x" in audit_sample
    assert "armor_x" in audit_sample
    assert "component_armor_path_x" in audit_sample
    assert "tree_edge_index" in audit_sample

    train_dataset = TorchDamageDataset(output, split="train")
    artifact = fit_training_view_artifact(train_dataset, proxy_boundary="end_to_end")
    training_view = TrainingViewDataset(
        train_dataset,
        artifact,
        proxy_boundary="end_to_end",
    )
    training_sample = training_view[0]
    assert tuple(training_sample["training_x"].shape) == (12,)
    assert training_view.feature_names == (
        *UAV_STATE_FEATURE_NAMES[:6],
        *UAV_ROTATION_6D_FEATURES,
    )
    assert training_sample["training_proxy_boundary"] == "end_to_end"
    with pytest.raises(InputValidationError, match="end_to_end"):
        fit_training_view_artifact(
            train_dataset,
            proxy_boundary="solver_surrogate",
            derived_input_contract={"deployment_available": True},
        )


def test_prelabeled_geometry_is_limited_to_nonreference_allocations(
    target_data, case_data, registry
) -> None:
    engine = DamageEngine(target_data, registry)
    document = deepcopy(_sampling_spec().data)
    document["budgets"]["train_pilot"] = 4
    document["budgets"]["rare_stress_test"] = 1
    document["geometry_strata"] = [
        {
            "stratum_id": "near-fixed-threat",
            "description": "Outcome-blind range screening around the fixed threat",
            "pre_label": True,
            "predicate": {
                "all": [
                    {
                        "source": "prelabel_screening",
                        "path": "/uav_threat_range_m",
                        "operator": "le",
                        "value": 2.1,
                    }
                ]
            },
        }
    ]
    document["stream_roles"].extend(
        [
            {
                "role": "geometry_stratified_train",
                "stream": "geometry_stratified",
                "split": "train",
                "train_eligible": True,
            },
            {
                "role": "rare_stress_test",
                "stream": "stress",
                "split": "test",
                "train_eligible": False,
            },
        ]
    )
    document["round_budgets"][0]["allocations"].extend(
        [
            {
                "role": "geometry_stratified_train",
                "stratum_id": "near-fixed-threat",
                "proposal_id": "near-fixed-threat-train-v1",
                "count": 1,
            },
            {
                "role": "rare_stress_test",
                "stratum_id": "near-fixed-threat",
                "proposal_id": "near-fixed-threat-stress-v1",
                "count": 1,
            },
        ]
    )
    spec = SamplingSpec(document)
    factory_kwargs = {
        "target_hash": engine.target_hash,
        "damage_tree_definition_hash": content_hash(engine.target_data["damage_tree"]),
        "physics_registry_hash": engine.registry.hash,
        "engine_code_hash": ENGINE_CODE_HASH,
    }
    with pytest.raises(InputValidationError, match="read-only physics registry"):
        CaseFactory(
            deepcopy(case_data),
            spec,
            _label_spec(engine),
            **factory_kwargs,
        )

    plan = CaseFactory(
        deepcopy(case_data),
        spec,
        _label_spec(engine),
        physics_registry=engine.registry,
        **factory_kwargs,
    ).build_reference_plan()
    reference_rows = [
        row for row in plan.candidate_rows if row["role"].startswith("reference_")
    ]
    screened_rows = [
        row
        for row in plan.candidate_rows
        if row["role"] in {"geometry_stratified_train", "rare_stress_test"}
    ]
    assert reference_rows
    assert all(row["stratum"] == "p0" for row in reference_rows)
    assert {row["role"] for row in screened_rows} == {
        "geometry_stratified_train",
        "rare_stress_test",
    }
    assert all(row["stratum"] == "near-fixed-threat" for row in screened_rows)


def test_uav_sampling_contract_rejects_non_uav_tensor_roles() -> None:
    document = deepcopy(_sampling_spec().data)
    document["feature_roles"][0]["tensor_feature_names"][0] = "projectile_x_m"
    with pytest.raises(InputValidationError, match="position3, velocity3"):
        SamplingSpec(document)

    document = deepcopy(_sampling_spec().data)
    document["feature_roles"][0]["path"] = "/warhead/fragment_source/model_id"
    with pytest.raises(InputValidationError, match="raw UAV state"):
        SamplingSpec(document)


def test_uav_prelabel_rejects_impossible_fixed_sphere_shell(
    target_data, case_data, registry
) -> None:
    engine = DamageEngine(target_data, registry)
    document = deepcopy(_sampling_spec().data)
    document["budgets"]["train_pilot"] = 4
    document["geometry_strata"] = [
        {
            "stratum_id": "impossible-near-threat",
            "description": "The fixed two-metre shell cannot satisfy this range",
            "pre_label": True,
            "predicate": {
                "all": [
                    {
                        "source": "prelabel_screening",
                        "path": "/uav_threat_range_m",
                        "operator": "lt",
                        "value": 1.5,
                    }
                ]
            },
        }
    ]
    document["stream_roles"].append(
        {
            "role": "geometry_stratified_train",
            "stream": "geometry_stratified",
            "split": "train",
            "train_eligible": True,
        }
    )
    document["round_budgets"][0]["allocations"].append(
        {
            "role": "geometry_stratified_train",
            "stratum_id": "impossible-near-threat",
            "proposal_id": "impossible-near-threat-v1",
            "count": 1,
        }
    )
    document["legal_sampling"] = {
        "invalid_candidate_policy": "reject_and_resample",
        "max_attempts_per_allocation": 7,
    }
    spec = SamplingSpec(document)
    with pytest.raises(InputValidationError, match="unable to fill") as error:
        CaseFactory(
            deepcopy(case_data),
            spec,
            _label_spec(engine),
            target_hash=engine.target_hash,
            damage_tree_definition_hash=content_hash(engine.target_data["damage_tree"]),
            physics_registry_hash=engine.registry.hash,
            engine_code_hash=ENGINE_CODE_HASH,
            physics_registry=engine.registry,
        ).build_reference_plan()
    assert error.value.context["accepted"] == 0
    assert error.value.context["attempts"] == 7


def test_uav_plan_preflights_final_feature_role_semantics(
    target_data, case_data, registry
) -> None:
    engine = DamageEngine(target_data, registry)
    document = deepcopy(_sampling_spec().data)
    velocity_role = next(
        item for item in document["feature_roles"] if item["feature_id"] == "uav_velocity"
    )
    velocity_role["role"] = "learnable_variable"
    spec = SamplingSpec(document)
    with pytest.raises(InputValidationError, match="learnable_variable.*constant on Train"):
        CaseFactory(
            deepcopy(case_data),
            spec,
            _label_spec(engine),
            target_hash=engine.target_hash,
            damage_tree_definition_hash=content_hash(engine.target_data["damage_tree"]),
            physics_registry_hash=engine.registry.hash,
            engine_code_hash=ENGINE_CODE_HASH,
        ).build_reference_plan()


def test_uav_plan_rejects_mixed_fixed_scenarios_before_labels(case_data) -> None:
    first = deepcopy(case_data)
    second = deepcopy(case_data)
    second["warhead"]["blast_source"]["equivalent_mass_kg"] *= 1.01
    with pytest.raises(InputValidationError, match="one frozen threat/scenario"):
        _validate_fixed_scenario_plan([first, second], _sampling_spec())
