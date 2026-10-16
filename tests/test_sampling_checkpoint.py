from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from damage_engine.archive_validation import ArchiveIntegrityError
from damage_engine.engine import DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.label_spec import LabelSpec
from damage_engine.sampling_checkpoint import (
    initialize_sampling_run,
    mark_pilot_complete,
    verify_sampling_checkpoint,
)
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.utils import content_hash, file_hash


def _label_spec(engine: DamageEngine) -> LabelSpec:
    return LabelSpec(
        {
            "schema_version": "1.0.0",
            "label_spec_id": "checkpoint-labels",
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
            "sampling_spec_id": "checkpoint-p0",
            "revision": "1",
            "status": "provisional",
            "reference_distribution": {
                "distribution_id": "checkpoint-reference",
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
                "design": 201,
                "selection": 202,
                "split": 203,
                "screening": 204,
                "formal": 205,
            },
            "formal_policy": {
                "replicates": 2,
                "adaptive_ci_half_width": None,
                "epistemic_policy": "unique_per_lineage",
            },
            "split_policy": {
                "registry_id": "checkpoint-splits",
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


def _initialize(
    root: Path, target_data: dict, case_data: dict, registry
):
    engine = DamageEngine(target_data, registry)
    base_case = deepcopy(case_data)
    base_case["target_ref"]["content_hash"] = engine.target_hash
    sampling_spec = _sampling_spec()
    label_spec = _label_spec(engine)
    checkpoint, plan = initialize_sampling_run(
        root,
        engine=engine,
        base_case=base_case,
        sampling_spec=sampling_spec,
        label_spec=label_spec,
    )
    return engine, base_case, sampling_spec, label_spec, checkpoint, plan


def _formal_manifest(checkpoint: dict) -> dict:
    identity = checkpoint["identity"]
    return {
        "schema_version": "1.3.0",
        "engine": {
            "configuration_hash": identity["physics_registry_hash"],
            "code_hash": identity["engine_code_hash"],
        },
        "source_contracts": {
            "target_hashes": [identity["target_hash"]],
            "damage_tree_definition_hash": identity[
                "damage_tree_definition_hash"
            ],
            "sampling_design_hash": identity["sampling_spec_hash"],
            "label_spec_hash": identity["label_spec_hash"],
            "split_registry_hash": checkpoint["split_registry_hash"],
        },
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_initialize_commits_complete_hash_inventory_and_verifies(
    tmp_path, target_data, case_data, registry
) -> None:
    root = tmp_path / "sampling-run"
    _, _, sampling_spec, label_spec, checkpoint, plan = _initialize(
        root, target_data, case_data, registry
    )

    assert checkpoint["status"] == "initialized"
    assert checkpoint["formal_dataset"] is None
    assert checkpoint["identity"]["sampling_spec_hash"] == sampling_spec.hash
    assert checkpoint["identity"]["label_spec_hash"] == label_spec.hash
    assert len(plan.cases) == 3
    assert [case["metadata"]["frozen_split"] for case in plan.cases] == [
        "validation",
        "test",
        "train",
    ]
    assert not (root / ".sampling.incomplete").exists()

    expected_paths = {
        "plan/sampling_spec.json",
        "plan/label_spec.json",
        "plan/base_case.json", "plan/input_snapshot.json", "plan/formal_input_identity.json",
        "plan/split_registry.json",
        "plan/frozen_hashes.json",
        "rounds/round-000/candidates.parquet",
        "rounds/round-000/formal_cases.jsonl.gz",
        "rounds/round-000/round_manifest.json",
        "audits/sampling_quality.json",
    }
    inventory = {
        record["path"]: record["content_hash"]
        for record in checkpoint["committed_files"]
    }
    assert set(inventory) == expected_paths
    for relative, expected_hash in inventory.items():
        assert file_hash(root / relative) == expected_hash

    verified = verify_sampling_checkpoint(root)
    assert verified["identity"] == checkpoint["identity"]
    assert verified["split_registry_hash"] == plan.split_registry.definition_hash


def test_resume_accepts_identical_plan_but_rejects_changed_spec(
    tmp_path, target_data, case_data, registry
) -> None:
    root = tmp_path / "sampling-run"
    engine, base_case, sampling_spec, label_spec, checkpoint, plan = _initialize(
        root, target_data, case_data, registry
    )

    resumed_checkpoint, resumed_plan = initialize_sampling_run(
        root,
        engine=engine,
        base_case=deepcopy(base_case),
        sampling_spec=SamplingSpec(deepcopy(sampling_spec.data)),
        label_spec=LabelSpec(deepcopy(label_spec.data)),
        resume=True,
    )
    assert resumed_checkpoint == checkpoint
    assert [row["candidate_id"] for row in resumed_plan.candidate_rows] == [
        row["candidate_id"] for row in plan.candidate_rows
    ]

    changed = deepcopy(sampling_spec.data)
    changed["reference_distribution"]["variables"][0]["high"] = 291.0
    with pytest.raises(InputValidationError, match="identity differs"):
        initialize_sampling_run(
            root,
            engine=engine,
            base_case=base_case,
            sampling_spec=SamplingSpec(changed),
            label_spec=label_spec,
            resume=True,
        )


def test_verify_rejects_tampered_committed_artifact(
    tmp_path, target_data, case_data, registry
) -> None:
    root = tmp_path / "sampling-run"
    _initialize(root, target_data, case_data, registry)
    committed = root / "plan" / "sampling_spec.json"
    committed.write_text(
        committed.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InputValidationError, match="committed file is missing or changed"):
        verify_sampling_checkpoint(root)


def test_mark_pilot_complete_rejects_escape_and_hash_only_incomplete_archive(
    tmp_path, target_data, case_data, registry
) -> None:
    root = tmp_path / "sampling-run"
    _, _, _, _, checkpoint, _ = _initialize(
        root, target_data, case_data, registry
    )
    manifest = _formal_manifest(checkpoint)

    outside = tmp_path / "outside-manifest.json"
    _write_json(outside, manifest)
    with pytest.raises(InputValidationError, match="inside"):
        mark_pilot_complete(root, outside)

    manifest_path = root / "formal_dataset" / "manifest.json"
    mismatched = deepcopy(manifest)
    mismatched["source_contracts"]["target_hashes"] = ["sha256:" + "0" * 64]
    _write_json(manifest_path, mismatched)
    with pytest.raises(ArchiveIntegrityError) as mismatched_archive:
        mark_pilot_complete(root, manifest_path)
    assert mismatched_archive.value.context["integrity_kind"] == "inventory_mismatch"

    _write_json(manifest_path, manifest)
    with pytest.raises(ArchiveIntegrityError) as incomplete_archive:
        mark_pilot_complete(root, manifest_path)
    assert incomplete_archive.value.context["integrity_kind"] == "inventory_mismatch"
    assert verify_sampling_checkpoint(root)["status"] == "initialized"
