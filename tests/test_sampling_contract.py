from __future__ import annotations

from copy import deepcopy

import numpy as np

from damage_engine.case_factory import CaseFactory
from damage_engine.dataset import (
    CASE_FEATURE_NAMES,
    CASE_PHYSICS_FEATURE_NAMES,
    SOLVER_CONTEXT_FEATURE_NAMES,
    DatasetWriter,
)
from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.label_spec import LabelSpec
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.torch_dataset import TorchDamageDataset
from damage_engine.utils import content_hash


def _label_spec(engine: DamageEngine) -> LabelSpec:
    return LabelSpec(
        {
            "schema_version": "1.0.0",
            "label_spec_id": "sampling-contract-labels",
            "revision": "1",
            "status": "provisional",
            "source": {
                "damage_tree_id": "demo-tree",
                "damage_tree_revision": engine.damage_tree.version,
                "damage_tree_definition_hash": content_hash(
                    engine.target_data["damage_tree"]
                ),
            },
            "tasks": {
                "system_events": {
                    "type": "multi_label_event_probability",
                    "outputs": list(engine.damage_tree.outputs),
                    "loss": "binary_cross_entropy",
                }
            },
            "provenance": {
                "approval_status": "not_approved",
                "evidence_ref": None,
            },
        }
    )


def _sampling_spec() -> SamplingSpec:
    return SamplingSpec(
        {
            "schema_version": "1.0.0",
            "sampling_spec_id": "sampling-contract-p0",
            "revision": "1",
            "status": "provisional",
            "reference_distribution": {
                "distribution_id": "sampling-contract-reference",
                "interpretation": "design_distribution",
                "variables": [
                    {
                        "name": "temperature",
                        "path": "/environment/temperature_k",
                        "distribution": "uniform",
                        "low": 286.0,
                        "high": 290.0,
                    }
                ],
                "constraints": [],
            },
            "budgets": {
                "train_pilot": 1,
                "validation": 1,
                "test": 1,
                "rare_stress_test": 0,
            },
            "seeds": {
                "design": 101,
                "selection": 102,
                "split": 103,
                "screening": 104,
                "formal": 105,
            },
            "formal_policy": {
                "replicates": 2,
                "adaptive_ci_half_width": None,
                "epistemic_policy": "unique_per_lineage",
            },
            "split_policy": {
                "registry_id": "sampling-contract-splits",
                "freeze_before_outcomes": True,
                "adaptive_train_only": True,
            },
            "quality_gates": {
                "positivity_min": 0.1,
                "min_ess_fraction": 0.5,
                "max_weight_cv": 2.0,
            },
        }
    )


def _sampling_plan(
    engine: DamageEngine, case_data: dict
):
    label_spec = _label_spec(engine)
    sampling_spec = _sampling_spec()
    plan = CaseFactory(
        deepcopy(case_data),
        sampling_spec,
        label_spec,
        target_hash=engine.target_hash,
        damage_tree_definition_hash=content_hash(engine.target_data["damage_tree"]),
        physics_registry_hash=engine.registry.hash,
        engine_code_hash=ENGINE_CODE_HASH,
    ).build_reference_plan()
    return label_spec, sampling_spec, plan


def test_sampling_bound_writer_emits_contract_16_and_torch_defaults(
    tmp_path, target_data, case_data, registry
) -> None:
    engine = DamageEngine(target_data, registry)
    label_spec, sampling_spec, plan = _sampling_plan(engine, case_data)
    output = tmp_path / "sampling-bound"

    manifest = DatasetWriter(output, split_seed=999).run_cases(
        engine, plan.cases
    )

    assert manifest["schema_version"] == "1.4.0"
    assert manifest["tensor_contract"]["version"] == "1.6.0"
    assert manifest["source_contracts"]["label_spec_hash"] == label_spec.hash
    assert (
        manifest["source_contracts"]["sampling_design_hash"]
        == sampling_spec.hash
    )
    assert (
        manifest["source_contracts"]["split_registry_hash"]
        == plan.split_registry.definition_hash
    )
    assert manifest["split_policy"]["split_registry_status"] == "frozen_pre_outcome"
    assert manifest["split_policy"]["split_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    # This flag deliberately stays false: a frozen registry is a software
    # guarantee, not proof that the declared groups are independent in nature.
    assert manifest["split_policy"]["independence_guaranteed"] is False

    with np.load(output / "tensors.npz", allow_pickle=False) as tensors:
        assert tensors["split_names"].tolist() == ["validation", "test", "train"]
        assert tensors["split_group_sources"].tolist() == [
            "frozen_split_registry",
            "frozen_split_registry",
            "frozen_split_registry",
        ]

        physics_names = tensors["case_physics_feature_names"].tolist()
        solver_names = tensors["solver_context_feature_names"].tolist()
        assert physics_names == CASE_PHYSICS_FEATURE_NAMES
        assert solver_names == SOLVER_CONTEXT_FEATURE_NAMES
        assert set(physics_names).isdisjoint(solver_names)
        assert set(physics_names) | set(solver_names) == set(CASE_FEATURE_NAMES)
        assert "requested_replicates" not in physics_names
        assert "requested_replicates" in solver_names
        assert tensors["case_physics_x"].shape == (
            3,
            len(CASE_PHYSICS_FEATURE_NAMES),
        )
        assert tensors["solver_context_x"].shape == (
            3,
            len(SOLVER_CONTEXT_FEATURE_NAMES),
        )

        physics_indices = [CASE_FEATURE_NAMES.index(name) for name in physics_names]
        solver_indices = [CASE_FEATURE_NAMES.index(name) for name in solver_names]
        np.testing.assert_array_equal(
            tensors["case_physics_x"], tensors["case_x"][:, physics_indices]
        )
        np.testing.assert_array_equal(
            tensors["solver_context_x"], tensors["case_x"][:, solver_indices]
        )

        probability = tensors["sampling_selection_probability"]
        raw_weight = tensors["sampling_design_weight_raw"]
        assert probability.dtype == np.float64
        assert raw_weight.dtype == np.float64
        np.testing.assert_array_equal(probability, np.ones(3, dtype=np.float64))
        np.testing.assert_array_equal(raw_weight, 1.0 / probability)
        assert tensors["sampling_weight_mask"].tolist() == [True, True, True]
        assert tensors["sampling_round"].tolist() == [0, 0, 0]
        assert len(set(tensors["sampling_candidate_ids"].tolist())) == 3
        assert len(set(tensors["sampling_lineage_ids"].tolist())) == 3

    dataset = TorchDamageDataset(output, include_static_graph=False)
    assert len(dataset) == 3
    sample = dataset[0]
    assert "case_physics_x" in sample
    assert "case_x" not in sample
    assert "solver_context_x" not in sample
    assert "sampling_selection_probability" in sample
    assert "sampling_design_weight_raw" in sample
    assert sample["sampling_selection_probability"].dtype.is_floating_point
    assert sample["sampling_candidate_id"].startswith("candidate-")
    assert sample["sampling_lineage_id"].startswith("lineage-")
    assert sample["sampling_role"] == "reference_validation"
    assert sample["sampling_stratum"] == "p0"
    assert dataset.metadata["case_physics_feature_names"] == tuple(
        CASE_PHYSICS_FEATURE_NAMES
    )
    assert dataset.metadata["solver_context_feature_names"] == tuple(
        SOLVER_CONTEXT_FEATURE_NAMES
    )

    validation = TorchDamageDataset(
        output, split="validation", include_static_graph=False
    )
    assert len(validation) == 1
    assert validation[0]["split"] == "validation"


def test_plain_cases_remain_manifest_12_tensor_contract_14(
    tmp_path, target_data, case_data, registry
) -> None:
    engine = DamageEngine(target_data, registry)
    case = deepcopy(case_data)
    case["case_id"] = "plain-contract-case"
    case["target_ref"]["content_hash"] = engine.target_hash
    case["analysis"]["replicates"] = 2
    case["analysis"]["adaptive_ci_half_width"] = None
    output = tmp_path / "plain"

    manifest = DatasetWriter(output).run_cases(engine, [case])

    assert manifest["schema_version"] == "1.2.0"
    assert manifest["tensor_contract"]["version"] == "1.4.0"
    assert "sampling" not in manifest["generation"]
    assert "sampling_design_hash" not in manifest["source_contracts"]
    with np.load(output / "tensors.npz", allow_pickle=False) as tensors:
        assert "case_x" in tensors.files
        assert "case_physics_x" not in tensors.files
        assert "solver_context_x" not in tensors.files
        assert "sampling_selection_probability" not in tensors.files

    dataset = TorchDamageDataset(output, include_static_graph=False)
    sample = dataset[0]
    assert "case_x" in sample
    assert "case_physics_x" not in sample
    assert "sampling_selection_probability" not in sample
