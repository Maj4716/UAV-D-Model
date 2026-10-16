from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from damage_engine.case_factory import CaseFactory
from damage_engine.dataset import DatasetWriter
from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.label_spec import LabelSpec
from damage_engine.learning_input_identity import (
    LOITERING_IMPACT_SAMPLING_MODEL_INPUT_CONTRACT,
    LOITERING_IMPACT_STATE_FEATURE_NAMES,
    loitering_impact_fixed_scenario_fingerprint,
    loitering_impact_learning_input_fingerprint,
    loitering_impact_state_vector,
)
from damage_engine.registry import PhysicsRegistry
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.torch_dataset import TorchDamageDataset
from damage_engine.training_view import (
    TrainingViewDataset,
    fit_training_view_artifact,
    transform_loitering_impact_state,
)
from damage_engine.utils import content_hash, load_json


ROOT = Path(__file__).resolve().parents[1]


def _case() -> dict:
    return load_json(ROOT / "raw_data" / "case.loitering_impact.v1.json")


def _engine() -> DamageEngine:
    return DamageEngine(
        load_json(ROOT / "raw_data" / "target.rebuilt.json"),
        PhysicsRegistry.from_file(
            ROOT / "examples" / "physics.loitering_impact.json"
        ),
    )


def _sampling_spec() -> SamplingSpec:
    document = load_json(
        ROOT / "raw_data" / "sampling_spec.loitering_impact.smoke.v1.json"
    )
    document["schema_version"] = "1.2.0"
    document["model_input_contract"] = (
        LOITERING_IMPACT_SAMPLING_MODEL_INPUT_CONTRACT
    )
    document["feature_roles"] = [
        {
            "feature_id": "munition_profile",
            "path": "/munition/profile_id",
            "tensor_feature_names": list(
                LOITERING_IMPACT_STATE_FEATURE_NAMES[:3]
            ),
            "role": "learnable_variable",
            "availability_stage": "raw_input",
            "deployment_available": True,
        },
        {
            "feature_id": "trajectory_origin_y",
            "path": "/encounter/trajectory_origin_m/1",
            "tensor_feature_names": [
                LOITERING_IMPACT_STATE_FEATURE_NAMES[3]
            ],
            "role": "learnable_variable",
            "availability_stage": "raw_input",
            "deployment_available": True,
        },
        {
            "feature_id": "trajectory_origin_z",
            "path": "/encounter/trajectory_origin_m/2",
            "tensor_feature_names": [
                LOITERING_IMPACT_STATE_FEATURE_NAMES[4]
            ],
            "role": "learnable_variable",
            "availability_stage": "raw_input",
            "deployment_available": True,
        },
        {
            "feature_id": "impact_speed",
            "path": "/encounter/velocity_target_mps/0",
            "tensor_feature_names": [
                LOITERING_IMPACT_STATE_FEATURE_NAMES[5]
            ],
            "role": "learnable_variable",
            "availability_stage": "raw_input",
            "deployment_available": True,
        },
    ]
    return SamplingSpec(document)


def test_loitering_state_and_identities_exclude_collision_solution() -> None:
    case = _case()
    assert loitering_impact_state_vector(case) == [
        1.0,
        0.0,
        0.0,
        0.0,
        1.2,
        120.0,
    ]
    changed_contact = deepcopy(case)
    changed_contact["encounter"]["detonation_time_s"] += 1.0
    changed_contact["encounter"]["collision_distance_m"] += 120.0
    assert loitering_impact_learning_input_fingerprint(
        changed_contact
    ) == loitering_impact_learning_input_fingerprint(case)

    changed_input = deepcopy(case)
    changed_input["encounter"]["trajectory_origin_m"][1] = 0.5
    assert loitering_impact_learning_input_fingerprint(
        changed_input
    ) != loitering_impact_learning_input_fingerprint(case)
    assert loitering_impact_fixed_scenario_fingerprint(
        changed_input
    ) == loitering_impact_fixed_scenario_fingerprint(case)

    changed_hidden_context = deepcopy(case)
    changed_hidden_context["encounter"]["trajectory_origin_m"][0] -= 1.0
    assert loitering_impact_fixed_scenario_fingerprint(
        changed_hidden_context
    ) != loitering_impact_fixed_scenario_fingerprint(case)


def test_loitering_transform_is_identity_6d() -> None:
    raw = np.asarray(loitering_impact_state_vector(_case()), dtype=np.float32)
    values, mask, names, units, transforms = transform_loitering_impact_state(
        raw,
        np.ones(6, dtype=bool),
        LOITERING_IMPACT_STATE_FEATURE_NAMES,
        ("1", "1", "1", "m", "m", "m/s"),
    )
    np.testing.assert_array_equal(values, raw)
    assert mask.tolist() == [True] * 6
    assert names == list(LOITERING_IMPACT_STATE_FEATURE_NAMES)
    assert units == ["1", "1", "1", "m", "m", "m/s"]
    assert all(item["kind"] == "identity" for item in transforms)


def test_contract_18_default_loader_exposes_only_loitering_state(
    tmp_path,
) -> None:
    engine = _engine()
    spec = _sampling_spec()
    label = LabelSpec.from_file(
        ROOT / "raw_data" / "label_spec.loitering_impact.v1.json"
    )
    plan = CaseFactory(
        _case(),
        spec,
        label,
        target_hash=engine.target_hash,
        damage_tree_definition_hash=content_hash(
            engine.target_data["damage_tree"]
        ),
        physics_registry_hash=engine.registry.hash,
        engine_code_hash=ENGINE_CODE_HASH,
        physics_registry=engine.registry,
        compiled_target=engine.compiled_target,
    ).build_reference_plan()

    output = tmp_path / "loitering-impact-contract"
    manifest = DatasetWriter(output).run_cases(engine, plan.cases)
    assert manifest["schema_version"] == "1.7.0"
    assert manifest["tensor_contract"]["version"] == "1.8.0"
    assert (
        manifest["model_input_contract"]["tensor_field"]
        == "loitering_impact_x"
    )

    with np.load(output / "tensors.npz", allow_pickle=False) as tensors:
        assert tensors["loitering_impact_x"].shape == (12, 6)
        assert tensors["loitering_impact_feature_names"].tolist() == list(
            LOITERING_IMPACT_STATE_FEATURE_NAMES
        )
        assert "case_physics_x" in tensors.files

    dataset = TorchDamageDataset(output)
    sample = dataset[0]
    assert "loitering_impact_x" in sample
    assert "loitering_impact_feature_mask" in sample
    assert "case_x" not in sample
    assert "case_physics_x" not in sample
    assert "solver_context_x" not in sample
    assert "component_x" not in sample
    assert "armor_x" not in sample
    assert dataset.include_static_graph is False

    train_dataset = TorchDamageDataset(output, split="train")
    artifact = fit_training_view_artifact(
        train_dataset, proxy_boundary="end_to_end"
    )
    view = TrainingViewDataset(
        train_dataset,
        artifact,
        proxy_boundary="end_to_end",
    )
    assert view.feature_names == LOITERING_IMPACT_STATE_FEATURE_NAMES
    assert tuple(view[0]["training_x"].shape) == (6,)


def test_fragment_only_target_admits_the_declared_system_labels(tmp_path) -> None:
    """The fragment-only task must not retain the old all-false label masks."""

    target = load_json(
        ROOT / "raw_data" / "target.rebuilt.fragment_only.v1.json"
    )
    engine = DamageEngine(
        target,
        PhysicsRegistry.from_file(
            ROOT / "examples" / "physics.loitering_impact.json"
        ),
    )
    document = load_json(
        ROOT
        / "raw_data"
        / "sampling_spec.5000.loitering_impact.fragment_only.v1.json"
    )
    document["sampling_spec_id"] += "-mask-smoke"
    document["split_policy"]["registry_id"] += "-mask-smoke"
    document["budgets"] = {
        "train_pilot": 3,
        "validation": 3,
        "test": 3,
        "rare_stress_test": 0,
    }
    for allocation in document["round_budgets"][0]["allocations"]:
        allocation["count"] = 3
    spec = SamplingSpec(document)
    label = LabelSpec.from_file(
        ROOT / "raw_data" / "label_spec.loitering_impact.fragment_only.v1.json"
    )
    plan = CaseFactory(
        load_json(ROOT / "raw_data" / "case.loitering_impact.fragment_only.v1.json"),
        spec,
        label,
        target_hash=engine.target_hash,
        damage_tree_definition_hash=content_hash(engine.target_data["damage_tree"]),
        physics_registry_hash=engine.registry.hash,
        engine_code_hash=ENGINE_CODE_HASH,
        physics_registry=engine.registry,
        compiled_target=engine.compiled_target,
    ).build_reference_plan()

    output = tmp_path / "fragment-only-label-mask"
    DatasetWriter(output).run_cases(engine, plan.cases)
    with np.load(output / "tensors.npz", allow_pickle=False) as tensors:
        assert tensors["system_source_available_mask"].all()
        assert tensors["system_label_mask"].all()
        assert tensors["label_task_available_mask"].all()
        assert tensors["loitering_impact_x"].shape == (9, 6)
