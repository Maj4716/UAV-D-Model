"""Content-bound component-task admission for dataset-v2 baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import InputValidationError
from .utils import content_hash, file_hash


TASK_ADMISSION_SCHEMA_VERSION = "dataset-v2-task-admission@1.1.0"
SUPPORTED_TASK_ADMISSION_SCHEMA_VERSIONS = frozenset(
    {
        "dataset-v2-task-admission@1.0.0",
        TASK_ADMISSION_SCHEMA_VERSION,
    }
)
PROBABILITY_TASKS = ("p_hit", "p_degraded", "p_failed")
CONDITIONAL_TASKS = (
    "conditional_log1p_energy",
    "conditional_log1p_hit_count",
)
BASELINE_TASKS = PROBABILITY_TASKS + CONDITIONAL_TASKS


def load_baseline_task_admission(
    path: str | Path | None,
    *,
    dataset_validation: Mapping[str, Any],
    component_ids: Sequence[str] | np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a frozen admission artifact without touching sealed-test labels.

    When ``path`` is ``None``, every component remains active.  A supplied
    artifact must bind to the validated dataset and explicitly define all
    three probability tasks.
    """

    identifiers = np.asarray(component_ids).astype(str)
    if identifiers.ndim != 1 or identifiers.size == 0:
        raise InputValidationError("task admission requires one-dimensional component IDs")
    if len(set(identifiers.tolist())) != int(identifiers.size):
        raise InputValidationError("dataset component IDs must be unique")
    all_active = np.ones(identifiers.size, dtype=bool)
    if path is None:
        return (
            {task: all_active.copy() for task in BASELINE_TASKS},
            {
                "mode": "all_dataset_components",
                "artifact_path": None,
                "artifact_sha256": None,
                "artifact_hash": None,
                "unspecified_component_task_policy": "all_components_active",
                "component_task_mask_source": {
                    task: "all_dataset_components" for task in BASELINE_TASKS
                },
                "derived_support_component_task_policy": (
                    "inherits_p_hit_admission"
                ),
                "admitted_component_count_by_task": {
                    task: int(identifiers.size) for task in BASELINE_TASKS
                },
            },
        )

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise InputValidationError(
            "task admission artifact does not exist",
            context={"path": str(artifact_path)},
        )
    try:
        document = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputValidationError(
            "task admission artifact is not valid UTF-8 JSON",
            context={"path": str(artifact_path)},
        ) from exc
    if not isinstance(document, dict):
        raise InputValidationError("task admission artifact must be a JSON object")
    schema_version = document.get("schema_version")
    if schema_version not in SUPPORTED_TASK_ADMISSION_SCHEMA_VERSIONS:
        raise InputValidationError(
            "unsupported task admission schema version",
            context={"schema_version": document.get("schema_version")},
        )
    artifact_hash = document.get("artifact_hash")
    hash_payload = dict(document)
    hash_payload.pop("artifact_hash", None)
    expected_hash = content_hash(hash_payload)
    if artifact_hash != expected_hash:
        raise InputValidationError(
            "task admission artifact hash mismatch",
            context={"expected": expected_hash, "actual": artifact_hash},
        )

    identity = document.get("dataset_identity")
    if not isinstance(identity, dict):
        raise InputValidationError("task admission lacks dataset_identity")
    expected_identity = {
        "dataset_id": dataset_validation.get("dataset_id"),
        "dataset_manifest_hash": dataset_validation.get("manifest_hash"),
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise InputValidationError(
                "task admission is bound to a different dataset",
                context={"field": key, "expected": expected, "actual": identity.get(key)},
            )

    rows = document.get("component_task_admission")
    if not isinstance(rows, dict):
        raise InputValidationError("task admission lacks component_task_admission")
    required_tasks = set(PROBABILITY_TASKS)
    if schema_version == TASK_ADMISSION_SCHEMA_VERSION:
        required_tasks.update(CONDITIONAL_TASKS)
    missing = sorted(required_tasks - set(rows))
    if missing:
        raise InputValidationError(
            "task admission lacks required component tasks",
            context={"missing": missing},
        )
    identifier_to_column = {
        identifier: column for column, identifier in enumerate(identifiers.tolist())
    }
    masks: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for task, row in rows.items():
        if not isinstance(task, str) or not isinstance(row, dict):
            raise InputValidationError("task admission rows must be named objects")
        admitted_ids = row.get("admitted_component_ids")
        if not isinstance(admitted_ids, list) or any(
            not isinstance(value, str) for value in admitted_ids
        ):
            raise InputValidationError(
                "task admission admitted_component_ids must be a string list",
                context={"task": task},
            )
        if len(set(admitted_ids)) != len(admitted_ids):
            raise InputValidationError(
                "task admission component IDs must be unique",
                context={"task": task},
            )
        unknown = sorted(set(admitted_ids) - set(identifier_to_column))
        if unknown:
            raise InputValidationError(
                "task admission references unknown components",
                context={"task": task, "unknown": unknown},
            )
        declared_count = row.get("admitted_component_count")
        if declared_count != len(admitted_ids):
            raise InputValidationError(
                "task admission component count mismatch",
                context={
                    "task": task,
                    "declared": declared_count,
                    "observed": len(admitted_ids),
                },
            )
        mask = np.zeros(identifiers.size, dtype=bool)
        for identifier in admitted_ids:
            mask[identifier_to_column[identifier]] = True
        masks[task] = mask
        counts[task] = int(np.sum(mask))

    for task in required_tasks:
        if not np.any(masks[task]):
            raise InputValidationError(
                "task admission leaves a required component task empty",
                context={"task": task},
            )
    sealed = document.get("sealed_test")
    if not isinstance(sealed, dict):
        raise InputValidationError("task admission lacks sealed_test evidence")
    if int(sealed.get("test_open_count", -1)) != 0 or int(
        sealed.get("test_result_json_lines_decoded", -1)
    ) != 0:
        raise InputValidationError("task admission was not frozen under a sealed test")

    task_mask_source = {task: "explicit" for task in masks}
    if schema_version == "dataset-v2-task-admission@1.0.0":
        for task in CONDITIONAL_TASKS:
            masks[task] = masks["p_hit"].copy()
            counts[task] = counts["p_hit"]
            task_mask_source[task] = "inherited_from_p_hit_v1_compatibility"

    return masks, {
        "mode": "frozen_component_task_subset",
        "schema_version": schema_version,
        "artifact_path": str(artifact_path),
        "artifact_sha256": file_hash(artifact_path),
        "artifact_hash": artifact_hash,
        "audit_report_hash": identity.get("audit_report_hash"),
        "component_task_mask_source": task_mask_source,
        "derived_support_component_task_policy": "inherits_p_hit_admission",
        "admitted_component_count_by_task": counts,
    }


__all__ = [
    "BASELINE_TASKS",
    "CONDITIONAL_TASKS",
    "PROBABILITY_TASKS",
    "SUPPORTED_TASK_ADMISSION_SCHEMA_VERSIONS",
    "TASK_ADMISSION_SCHEMA_VERSION",
    "load_baseline_task_admission",
]
