from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from damage_engine.case_factory import CaseFactory
from damage_engine.errors import InputValidationError
from damage_engine.label_spec import LabelSpec
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.split_registry import SplitRegistry
from damage_engine.utils import content_hash


def _label_spec() -> LabelSpec:
    return LabelSpec(
        {
            "schema_version": "1.0.0",
            "label_spec_id": "labels-p0",
            "revision": "1",
            "status": "provisional",
            "source": {
                "damage_tree_id": "demo-tree",
                "damage_tree_revision": "1",
                "damage_tree_definition_hash": "sha256:" + "2" * 64,
            },
            "tasks": {
                "effects": {
                    "type": "multi_label_event_probability",
                    "outputs": ["effect-a", "effect-b"],
                    "loss": "binary_cross_entropy",
                }
            },
            "provenance": {"approval_status": "not_approved", "evidence_ref": None},
        }
    )
def _sampling_spec() -> SamplingSpec:
    return SamplingSpec(
        {
            "schema_version": "1.0.0",
            "sampling_spec_id": "sampling-p0",
            "revision": "1",
            "status": "provisional",
            "reference_distribution": {
                "distribution_id": "p0-reference",
                "interpretation": "design_distribution",
                "variables": [
                    {
                        "name": "temperature",
                        "path": "/environment/temperature_k",
                        "distribution": "uniform",
                        "low": 280.0,
                        "high": 300.0,
                    },
                    {
                        "name": "velocity_direction",
                        "path": "/encounter/velocity_target_mps",
                        "distribution": "unit_sphere_direction",
                    },
                    {
                        "name": "attitude",
                        "path": "/encounter/body_to_target_orientation_xyzw",
                        "distribution": "haar_quaternion",
                    },
                ],
                "constraints": [],
            },
            "budgets": {
                "train_pilot": 2,
                "validation": 1,
                "test": 1,
                "rare_stress_test": 0,
            },
            "seeds": {
                "design": 11,
                "selection": 22,
                "split": 33,
                "screening": 44,
                "formal": 55,
            },
            "formal_policy": {
                "replicates": 32,
                "adaptive_ci_half_width": None,
                "epistemic_policy": "unique_per_lineage",
            },
            "split_policy": {
                "registry_id": "split-p0",
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

def _register(
    registry: SplitRegistry,
    lineage_id: str,
    split: str,
    epistemic_group_id: str,
    *,
    role: str = "reference",
) -> None:
    registry.register(
        lineage_id=lineage_id,
        split=split,
        role=role,
        epistemic_group_id=epistemic_group_id,
    )


def test_split_registry_freezes_a_deterministic_assignment_definition(tmp_path) -> None:
    first = SplitRegistry("frozen-p0")
    _register(first, "lineage-b", "train", "group-b")
    _register(first, "lineage-a", "validation", "group-a")
    with pytest.raises(InputValidationError, match="must be frozen"):
        first.assignment("lineage-a")
    first_hash = first.freeze()

    reordered = SplitRegistry("frozen-p0")
    _register(reordered, "lineage-a", "validation", "group-a")
    _register(reordered, "lineage-b", "train", "group-b")
    assert reordered.freeze() == first_hash

    with pytest.raises(InputValidationError, match="is frozen"):
        _register(first, "lineage-c", "test", "group-c")

    path = tmp_path / "split-registry.json"
    first.write(path)
    restored = SplitRegistry.from_file(path)
    assert restored.definition_hash == first_hash
    assert restored.assignment("lineage-a").split == "validation"
    with pytest.raises(InputValidationError, match="is frozen"):
        _register(restored, "lineage-c", "test", "group-c")


def test_epistemic_group_cannot_cross_split_boundaries() -> None:
    registry = SplitRegistry("epistemic-isolation")
    _register(registry, "lineage-a", "train", "shared-epistemic")

    with pytest.raises(InputValidationError, match="epistemic group crosses"):
        _register(registry, "lineage-b", "test", "shared-epistemic")


def test_adaptive_lineages_are_train_only() -> None:
    registry = SplitRegistry("adaptive-train-only")

    with pytest.raises(InputValidationError, match="must be assigned to train"):
        _register(
            registry,
            "adaptive-lineage",
            "validation",
            "adaptive-epistemic",
            role="adaptive_train",
        )

    _register(
        registry,
        "adaptive-lineage",
        "train",
        "adaptive-epistemic",
        role="adaptive_train",
    )
    registry.freeze()
    assert registry.assignment("adaptive-lineage").split == "train"


def test_case_factory_replays_design_and_separates_screening_and_formal_seeds(
    case_data,
) -> None:
    sampling_spec = _sampling_spec()
    label_spec = _label_spec()
    kwargs = {
        "target_hash": "sha256:" + "3" * 64,
        "damage_tree_definition_hash": "sha256:" + "2" * 64,
        "physics_registry_hash": "sha256:" + "4" * 64,
        "engine_code_hash": "sha256:" + "5" * 64,
    }

    first = CaseFactory(
        deepcopy(case_data), sampling_spec, label_spec, **kwargs
    ).build_reference_plan()
    replay = CaseFactory(
        deepcopy(case_data), sampling_spec, label_spec, **kwargs
    ).build_reference_plan()

    np.testing.assert_array_equal(first.latent_design, replay.latent_design)
    assert first.cases == replay.cases
    assert first.candidate_rows == replay.candidate_rows
    assert content_hash(
        {
            "cases": first.cases,
            "candidate_rows": first.candidate_rows,
            "latent_design": first.latent_design.tolist(),
            "split_registry": first.split_registry.definition_document(),
        }
    ) == "sha256:f8895126fe4e7c10b1cfcc2d116503d2670ed1b733ec68b771fecbe8e835c48c"
    assert [row["split"] for row in first.candidate_rows] == [
        "validation",
        "test",
        "train",
        "train",
    ]
    assert len({row["lineage_id"] for row in first.candidate_rows}) == 4
    assert all(
        row["screen_seed"] != row["formal_seed"] for row in first.candidate_rows
    )
    assert all(
        case["analysis"]["aleatory_seed"] == row["formal_seed"]
        for case, row in zip(first.cases, first.candidate_rows)
    )
    assert all(
        case["metadata"]["split_registry_hash"]
        == first.split_registry.definition_hash
        for case in first.cases
    )
