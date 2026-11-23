#!/usr/bin/env python
"""Freeze train/validation-only component-task admission for dataset-v2.

The input audit is expected to have been produced by
``audit_dataset_v2_development.py``. This command reads only that audit and
the dataset manifest. It never opens the sealed-test payload.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.errors import InputValidationError  # noqa: E402
from damage_engine.utils import content_hash  # noqa: E402


SCHEMA_VERSION = "dataset-v2-task-admission@1.1.0"
SELECTION_PROTOCOL_VERSION = "train-validation-label-support-gate@1.1.0"
_TARGETS = (
    "p_hit",
    "p_degraded",
    "p_failed",
    "conditional_log1p_energy",
    "conditional_log1p_hit_count",
)
_SUPPORT_CONTRACTS = {
    "binary_probability": {
        "threshold_keys": (
            "train_positive_min",
            "train_negative_min",
            "validation_positive_min",
            "validation_negative_min",
        ),
        "count_keys": (
            "train_positive_count",
            "train_negative_count",
            "validation_positive_count",
            "validation_negative_count",
        ),
        "comparisons": (
            (
                "train_positive_count",
                "train_positive_min",
                "train_positive_below_minimum",
            ),
            (
                "train_negative_count",
                "train_negative_min",
                "train_negative_below_minimum",
            ),
            (
                "validation_positive_count",
                "validation_positive_min",
                "validation_positive_below_minimum",
            ),
            (
                "validation_negative_count",
                "validation_negative_min",
                "validation_negative_below_minimum",
            ),
        ),
    },
    "observed_continuous": {
        "threshold_keys": (
            "train_observed_min",
            "validation_observed_min",
        ),
        "count_keys": (
            "train_observed_count",
            "validation_observed_count",
        ),
        "comparisons": (
            (
                "train_observed_count",
                "train_observed_min",
                "train_observed_below_minimum",
            ),
            (
                "validation_observed_count",
                "validation_observed_min",
                "validation_observed_below_minimum",
            ),
        ),
    },
}


def _fail(message: str, **context: Any) -> None:
    raise InputValidationError(message, context=context)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("task-admission input field must be an object", field=field)
    return value


def _verified_hash(
    document: Mapping[str, Any], *, hash_field: str, document_name: str
) -> str:
    observed = document.get(hash_field)
    if not isinstance(observed, str) or not observed.startswith("sha256:"):
        _fail(
            "task-admission input is missing a content hash",
            document=document_name,
            hash_field=hash_field,
        )
    plain = {str(key): value for key, value in document.items() if key != hash_field}
    expected = content_hash(plain)
    if observed != expected:
        _fail(
            "task-admission input content hash mismatch",
            document=document_name,
            hash_field=hash_field,
            expected=expected,
            observed=observed,
        )
    return observed


def _nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail("task-admission count must be a nonnegative integer", field=field)
    return int(value)


def _support_contract(
    source: Mapping[str, Any], *, target: str
) -> tuple[str, Mapping[str, Any]]:
    support_kind = source.get("support_kind")
    if support_kind not in _SUPPORT_CONTRACTS:
        _fail(
            "task-admission support kind is unsupported",
            target=target,
            support_kind=support_kind,
        )
    return str(support_kind), _SUPPORT_CONTRACTS[str(support_kind)]


def _thresholds(
    source: Mapping[str, Any], *, target: str, threshold_keys: tuple[str, ...]
) -> dict[str, int]:
    values = _mapping(source.get("thresholds"), field=f"{target}.thresholds")
    return {
        key: _nonnegative_integer(
            values.get(key), field=f"component_task_admission.{target}.thresholds.{key}"
        )
        for key in threshold_keys
    }


def _counts_by_component(
    source: Mapping[str, Any], *, target: str, count_keys: tuple[str, ...]
) -> tuple[list[str], dict[str, dict[str, int]]]:
    maps: dict[str, Mapping[str, Any]] = {
        key: _mapping(
            source.get(key), field=f"component_task_admission.{target}.{key}"
        )
        for key in count_keys
    }
    key_sets = {key: set(values) for key, values in maps.items()}
    reference = key_sets[count_keys[0]]
    if not reference:
        _fail("task-admission support maps are empty", target=target)
    if any(keys != reference for keys in key_sets.values()):
        _fail(
            "task-admission support maps use different component sets",
            target=target,
        )
    component_ids = sorted(str(value) for value in reference)
    counts: dict[str, dict[str, int]] = {}
    for component_id in component_ids:
        counts[component_id] = {
            key: _nonnegative_integer(
                maps[key].get(component_id),
                field=(
                    f"component_task_admission.{target}.{key}.{component_id}"
                ),
            )
            for key in count_keys
        }
    return component_ids, counts


def _target_admission(source: Mapping[str, Any], *, target: str) -> dict[str, Any]:
    support_kind, contract = _support_contract(source, target=target)
    threshold_keys = tuple(contract["threshold_keys"])
    count_keys = tuple(contract["count_keys"])
    comparisons = tuple(contract["comparisons"])
    thresholds = _thresholds(
        source, target=target, threshold_keys=threshold_keys
    )
    component_ids, counts = _counts_by_component(
        source, target=target, count_keys=count_keys
    )
    admitted: list[str] = []
    rejected: list[dict[str, Any]] = []
    for component_id in component_ids:
        reasons = [
            reason
            for count_key, threshold_key, reason in comparisons
            if counts[component_id][count_key] < thresholds[threshold_key]
        ]
        if reasons:
            rejected.append(
                {
                    "component_id": component_id,
                    "reasons": reasons,
                    "counts": counts[component_id],
                }
            )
        else:
            admitted.append(component_id)

    source_admitted = source.get("admitted_component_ids")
    if not isinstance(source_admitted, list) or not all(
        isinstance(value, str) for value in source_admitted
    ):
        _fail("source audit admitted component list is invalid", target=target)
    if sorted(source_admitted) != admitted:
        _fail(
            "source audit admission conflicts with recomputed support gate",
            target=target,
            source_admitted=sorted(source_admitted),
            recomputed_admitted=admitted,
        )
    source_count = _nonnegative_integer(
        source.get("admitted_component_count"),
        field=f"component_task_admission.{target}.admitted_component_count",
    )
    if source_count != len(admitted):
        _fail(
            "source audit admitted count conflicts with admitted component list",
            target=target,
            source_count=source_count,
            list_count=len(admitted),
        )

    return {
        "support_kind": support_kind,
        "thresholds": thresholds,
        "component_count": len(component_ids),
        "component_order_hash": content_hash({"component_ids": component_ids}),
        "admitted_component_count": len(admitted),
        "admitted_component_ids": admitted,
        "rejected_component_count": len(rejected),
        "rejected_components": rejected,
    }


def freeze_task_admission(
    audit: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a deterministic task-admission artifact from verified inputs."""

    audit_hash = _verified_hash(
        audit, hash_field="report_hash", document_name="development audit"
    )
    manifest_hash = _verified_hash(
        manifest, hash_field="manifest_hash", document_name="dataset manifest"
    )
    audit_dataset_id = audit.get("dataset_id")
    manifest_dataset_id = manifest.get("dataset_id")
    if audit_dataset_id != manifest_dataset_id:
        _fail(
            "development audit and dataset manifest identify different datasets",
            audit_dataset_id=audit_dataset_id,
            manifest_dataset_id=manifest_dataset_id,
        )
    if audit.get("dataset_manifest_hash") != manifest_hash:
        _fail(
            "development audit references a different manifest hash",
            audit_manifest_hash=audit.get("dataset_manifest_hash"),
            manifest_hash=manifest_hash,
        )

    sealed_source = _mapping(audit.get("sealed_test"), field="sealed_test")
    sealed_test = {
        "test_case_count": _nonnegative_integer(
            sealed_source.get("test_case_count"), field="sealed_test.test_case_count"
        ),
        "test_labels_present_in_core": sealed_source.get("test_labels_present_in_core"),
        "test_open_count": _nonnegative_integer(
            sealed_source.get("test_open_count"), field="sealed_test.test_open_count"
        ),
        "test_result_json_lines_decoded": _nonnegative_integer(
            sealed_source.get("test_result_json_lines_decoded"),
            field="sealed_test.test_result_json_lines_decoded",
        ),
        "test_metrics_computed": sealed_source.get("test_metrics_computed"),
    }
    if not isinstance(sealed_test["test_labels_present_in_core"], bool) or not isinstance(
        sealed_test["test_metrics_computed"], bool
    ):
        _fail("development audit sealed-test flags must be boolean")
    sealed_test["sealed"] = bool(
        not sealed_test["test_labels_present_in_core"]
        and sealed_test["test_open_count"] == 0
        and sealed_test["test_result_json_lines_decoded"] == 0
        and not sealed_test["test_metrics_computed"]
    )
    if not sealed_test["sealed"]:
        _fail("task admission requires an unopened sealed test", **sealed_test)

    source_admission = _mapping(
        audit.get("component_task_admission"), field="component_task_admission"
    )
    component_task_admission = {
        target: _target_admission(
            _mapping(
                source_admission.get(target),
                field=f"component_task_admission.{target}",
            ),
            target=target,
        )
        for target in _TARGETS
    }
    component_sets = {
        target: set(value["admitted_component_ids"])
        for target, value in component_task_admission.items()
    }
    admitted_union = sorted(set().union(*component_sets.values()))
    admitted_intersection = sorted(set.intersection(*component_sets.values()))

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_identity": {
            "dataset_id": manifest_dataset_id,
            "dataset_manifest_hash": manifest_hash,
            "audit_report_hash": audit_hash,
        },
        "selection_contract": {
            "protocol_version": SELECTION_PROTOCOL_VERSION,
            "scope": "train_and_validation_label_availability_only",
            "source_audit_schema_version": audit.get("schema_version"),
            "performance_metrics_used": False,
            "sealed_test_outcomes_used": False,
        },
        "sealed_test": sealed_test,
        "component_task_admission": component_task_admission,
        "summary": {
            "admitted_component_union_count": len(admitted_union),
            "admitted_component_union_ids": admitted_union,
            "admitted_component_intersection_count": len(admitted_intersection),
            "admitted_component_intersection_ids": admitted_intersection,
        },
    }
    artifact["artifact_hash"] = content_hash(artifact)
    return artifact


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputValidationError(
            "task-admission JSON is missing or unreadable",
            context={"path": str(path)},
        ) from exc
    return _mapping(value, field=str(path))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = freeze_task_admission(_load_json(args.audit), _load_json(args.manifest))
    _write_json(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_hash": artifact["artifact_hash"],
                "admitted_component_counts": {
                    target: admission["admitted_component_count"]
                    for target, admission in artifact[
                        "component_task_admission"
                    ].items()
                },
                "sealed_test_result_json_lines_decoded": artifact["sealed_test"][
                    "test_result_json_lines_decoded"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
