from __future__ import annotations

from copy import deepcopy

import pytest

from damage_engine.errors import InputValidationError
from damage_engine.utils import content_hash
from tools.freeze_dataset_v2_task_admission import freeze_task_admission


def _hashed(document: dict, field: str) -> dict:
    document[field] = content_hash(document)
    return document


def _source_binary_target() -> dict:
    train_positive = {
        "component-000": 25,
        "component-001": 19,
        "component-002": 25,
    }
    train_negative = {
        "component-000": 25,
        "component-001": 25,
        "component-002": 25,
    }
    validation_positive = {
        "component-000": 6,
        "component-001": 6,
        "component-002": 6,
    }
    validation_negative = {
        "component-000": 6,
        "component-001": 6,
        "component-002": 4,
    }
    return {
        "support_kind": "binary_probability",
        "selection_scope": "label-availability gate only",
        "thresholds": {
            "train_positive_min": 20,
            "train_negative_min": 20,
            "validation_positive_min": 5,
            "validation_negative_min": 5,
        },
        "train_eligible_component_count": 2,
        "validation_evaluable_component_count": 2,
        "admitted_component_count": 1,
        "admitted_component_ids": ["component-000"],
        "rare_or_underpowered_component_ids": ["component-001"],
        "train_positive_count": train_positive,
        "train_negative_count": train_negative,
        "validation_positive_count": validation_positive,
        "validation_negative_count": validation_negative,
    }


def _source_conditional_target() -> dict:
    return {
        "support_kind": "observed_continuous",
        "selection_scope": "label-availability gate only",
        "thresholds": {
            "train_observed_min": 20,
            "validation_observed_min": 5,
        },
        "train_eligible_component_count": 2,
        "validation_evaluable_component_count": 2,
        "admitted_component_count": 1,
        "admitted_component_ids": ["component-000"],
        "rare_or_underpowered_component_ids": ["component-001"],
        "rejected_components": [
            {
                "component_id": "component-001",
                "reasons": ["train_observed_below_minimum"],
            },
            {
                "component_id": "component-002",
                "reasons": ["validation_observed_below_minimum"],
            },
        ],
        "train_observed_count": {
            "component-000": 25,
            "component-001": 19,
            "component-002": 25,
        },
        "validation_observed_count": {
            "component-000": 6,
            "component-001": 6,
            "component-002": 4,
        },
    }


def _inputs() -> tuple[dict, dict]:
    manifest = _hashed(
        {
            "schema_version": "dataset-v2@1.0.0",
            "dataset_id": "dataset-for-admission",
            "case_count": 2600,
        },
        "manifest_hash",
    )
    audit = {
        "schema_version": "dataset-v2-development-audit@1.0.0",
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_hash": manifest["manifest_hash"],
        "scope": "train_and_validation_only_with_opaque_sealed_test_labels",
        "sealed_test": {
            "test_case_count": 300,
            "test_labels_present_in_core": False,
            "test_open_count": 0,
            "test_result_json_lines_decoded": 0,
            "test_metrics_computed": False,
        },
        "component_task_admission": {
            **{
                target: _source_binary_target()
                for target in ("p_hit", "p_degraded", "p_failed")
            },
            **{
                target: _source_conditional_target()
                for target in (
                    "conditional_log1p_energy",
                    "conditional_log1p_hit_count",
                )
            },
        },
    }
    _hashed(audit, "report_hash")
    return audit, manifest


def test_freeze_task_admission_recomputes_rejections_and_hashes_artifact() -> None:
    audit, manifest = _inputs()
    artifact = freeze_task_admission(audit, manifest)

    assert artifact["dataset_identity"] == {
        "dataset_id": "dataset-for-admission",
        "dataset_manifest_hash": manifest["manifest_hash"],
        "audit_report_hash": audit["report_hash"],
    }
    assert artifact["sealed_test"]["sealed"] is True
    assert artifact["schema_version"] == "dataset-v2-task-admission@1.1.0"
    assert artifact["selection_contract"]["performance_metrics_used"] is False
    assert artifact["selection_contract"]["sealed_test_outcomes_used"] is False
    admission = artifact["component_task_admission"]["p_hit"]
    assert admission["admitted_component_ids"] == ["component-000"]
    assert admission["support_kind"] == "binary_probability"
    assert admission["rejected_components"] == [
        {
            "component_id": "component-001",
            "reasons": ["train_positive_below_minimum"],
            "counts": {
                "train_positive_count": 19,
                "train_negative_count": 25,
                "validation_positive_count": 6,
                "validation_negative_count": 6,
            },
        },
        {
            "component_id": "component-002",
            "reasons": ["validation_negative_below_minimum"],
            "counts": {
                "train_positive_count": 25,
                "train_negative_count": 25,
                "validation_positive_count": 6,
                "validation_negative_count": 4,
            },
        },
    ]
    conditional = artifact["component_task_admission"][
        "conditional_log1p_energy"
    ]
    assert conditional["support_kind"] == "observed_continuous"
    assert conditional["admitted_component_ids"] == ["component-000"]
    assert conditional["rejected_components"] == [
        {
            "component_id": "component-001",
            "reasons": ["train_observed_below_minimum"],
            "counts": {
                "train_observed_count": 19,
                "validation_observed_count": 6,
            },
        },
        {
            "component_id": "component-002",
            "reasons": ["validation_observed_below_minimum"],
            "counts": {
                "train_observed_count": 25,
                "validation_observed_count": 4,
            },
        },
    ]
    artifact_hash = artifact.pop("artifact_hash")
    assert artifact_hash == content_hash(artifact)


def test_freeze_task_admission_rejects_opened_test() -> None:
    audit, manifest = _inputs()
    audit.pop("report_hash")
    audit["sealed_test"]["test_result_json_lines_decoded"] = 1
    _hashed(audit, "report_hash")

    with pytest.raises(InputValidationError, match="unopened sealed test"):
        freeze_task_admission(audit, manifest)


def test_freeze_task_admission_rejects_tampered_audit() -> None:
    audit, manifest = _inputs()
    tampered = deepcopy(audit)
    tampered["component_task_admission"]["p_hit"]["admitted_component_ids"] = []

    with pytest.raises(InputValidationError, match="content hash mismatch"):
        freeze_task_admission(tampered, manifest)
