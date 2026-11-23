#!/usr/bin/env python
"""Package a sampler pilot while keeping sealed-test labels opaque.

The output keeps test inputs and split identities so existing validation and
training code can enforce the sealed-test contract.  Test label arrays and
component traces are placeholders with all masks disabled; the source test
result lines are never JSON-decoded.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.dataset_v2 import (  # noqa: E402
    TRACE_SCHEMA_VERSION,
    TRACE_SEMANTICS,
    validate_dataset_v2,
    write_dataset_v2,
)
from damage_engine.dataset_v2_sampler import (  # noqa: E402
    _json,
    _source_dataset_directory,
    _target_arrays,
)
from damage_engine.errors import InputValidationError  # noqa: E402
from damage_engine.utils import content_hash, file_hash  # noqa: E402


STRICT_CONVERSION_VERSION = "sampler-pilot-to-dataset-v2-strict-seal@1.0.0"
_VALID_SPLITS = {"train", "validation", "test"}


def _selected_npz(path: Path, fields: Sequence[str]) -> dict[str, np.ndarray]:
    """Load only named arrays; label tensors in the archive stay unopened."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = [name for name in fields if name not in archive.files]
            if missing:
                raise InputValidationError(
                    "sampler tensor shard lacks strict-seal source fields",
                    context={"path": str(path), "missing": missing},
                )
            return {name: np.asarray(archive[name]) for name in fields}
    except InputValidationError:
        raise
    except (OSError, ValueError) as exc:
        raise InputValidationError(
            "sampler tensor shard is unreadable",
            context={"path": str(path)},
        ) from exc


def _development_result_rows(
    path: Path,
    *,
    case_ids: Sequence[str],
    splits: Sequence[str],
) -> tuple[list[dict[str, Any]], list[int], dict[str, int]]:
    """Decode development rows only, using frozen tensor order for routing."""

    rows: list[dict[str, Any]] = []
    indices: list[int] = []
    nonempty_line_count = 0
    test_line_count = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                if nonempty_line_count >= len(case_ids):
                    raise ValueError("result shard has more rows than its tensor shard")
                row_index = nonempty_line_count
                nonempty_line_count += 1
                split = str(splits[row_index])
                if split == "test":
                    # Reading the opaque line is required to count ledger rows;
                    # parsing it would open the sealed test label payload.
                    test_line_count += 1
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} is not an object")
                expected_case_id = str(case_ids[row_index])
                if str(value.get("case_id")) != expected_case_id:
                    raise ValueError(
                        f"line {line_number} case_id differs from frozen tensor order"
                    )
                rows.append(value)
                indices.append(row_index)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise InputValidationError(
            "sampler development result rows are unreadable or misaligned",
            context={"path": str(path)},
        ) from exc
    if nonempty_line_count != len(case_ids):
        raise InputValidationError(
            "sampler result and tensor shard row counts differ",
            context={
                "path": str(path),
                "result_rows": nonempty_line_count,
                "tensor_rows": len(case_ids),
            },
        )
    return rows, indices, {
        "result_json_lines_seen": nonempty_line_count,
        "development_result_json_lines_decoded": len(rows),
        "test_result_json_lines_seen": test_line_count,
        "test_result_json_lines_decoded": 0,
    }


def _full_target_arrays(
    development_results: Sequence[Mapping[str, Any]],
    development_indices: Sequence[int],
    *,
    case_ids: np.ndarray,
    component_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Interleave observed development targets with masked test placeholders."""

    development = _target_arrays(development_results, component_ids.tolist())
    expected_development_ids = case_ids[np.asarray(development_indices, dtype=np.int64)]
    if development["case_ids"].astype(str).tolist() != expected_development_ids.tolist():
        raise InputValidationError(
            "development hurdle targets differ from frozen tensor order"
        )
    development_traces = development.pop("_traces")
    count = int(case_ids.size)
    full: dict[str, np.ndarray] = {
        "case_ids": case_ids.astype(str),
        "component_ids": component_ids.astype(str),
    }
    index = np.asarray(development_indices, dtype=np.int64)
    for name, value in development.items():
        if name in {"case_ids", "component_ids"}:
            continue
        array = np.asarray(value)
        shape = (count,) + array.shape[1:]
        target = np.zeros(shape, dtype=array.dtype)
        target[index] = array
        full[name] = target

    test_mask = np.ones(count, dtype=bool)
    test_mask[index] = False
    for name in (
        "p_hit_ci95",
        "p_hit_identification_interval",
        "p_degraded_ci95",
        "p_degraded_identification_interval",
        "p_failed_ci95",
        "p_failed_identification_interval",
    ):
        full[name][test_mask, ..., 0] = 0.0
        full[name][test_mask, ..., 1] = 1.0
    # The dataset-v2 count contract requires positive trial counts.  A masked
    # test placeholder therefore uses the neutral 0/1 pair and is never used
    # by loss, metrics, selection, or readiness calculations.
    full["p_hit_success_count"][test_mask] = 0
    full["p_hit_trial_count"][test_mask] = 1
    full["label_mask"][test_mask] = False
    full["conditional_log1p_energy_mask"][test_mask] = False
    full["conditional_log1p_hit_count_mask"][test_mask] = False
    full["derived_support_mask"][test_mask] = False

    trace_by_index = {
        int(row_index): trace
        for row_index, trace in zip(
            development_indices, development_traces, strict=True
        )
    }
    traces: list[dict[str, Any]] = []
    for row_index, case_id in enumerate(case_ids.astype(str).tolist()):
        if row_index in trace_by_index:
            traces.append(trace_by_index[row_index])
            continue
        traces.append(
            {
                "case_id": case_id,
                "component_state_trace": {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "semantics": TRACE_SEMANTICS,
                    "replicate_count": 1,
                    "records": [
                        {
                            "replicate_index": 0,
                            "component_mechanism_states": {},
                            "sealed_test_placeholder": True,
                        }
                    ],
                    "sealed_test_placeholder": True,
                },
            }
        )
    return full, traces


def convert_sampler_pilot_to_strict_development_dataset_v2(
    pilot_directory: str | Path,
    output_directory: str | Path,
    *,
    status: str = "provisional",
    provisional_reason: str = (
        "software-development dataset with opaque sealed-test labels; physical "
        "parameters and semantic authority remain unverified"
    ),
) -> dict[str, Any]:
    """Create a dataset-v2 package without decoding sealed-test result JSON."""

    source = _source_dataset_directory(Path(pilot_directory))
    manifest = _json(source / "manifest.json")
    input_contract = manifest.get("model_input_contract")
    if not isinstance(input_contract, Mapping):
        raise InputValidationError("sampler manifest lacks model_input_contract")
    input_field = str(input_contract.get("tensor_field", ""))
    mask_field = str(input_contract.get("mask_field", ""))
    if not input_field or not mask_field:
        raise InputValidationError("model input contract lacks tensor or mask field")
    feature_field = input_field.replace("_x", "_feature_names")
    selected_fields = [
        "case_ids",
        "split_names",
        "input_frozen_splits",
        input_field,
        mask_field,
        feature_field,
        "component_ids",
        "sampling_weight_mask",
        "sampling_selection_probability",
        "sampling_design_weight_raw",
    ]

    shard_directories = sorted((source / "shards").glob("part-*"))
    if not shard_directories:
        raise InputValidationError("sampler formal dataset contains no shards")
    shard_arrays: list[dict[str, np.ndarray]] = []
    feature_names: np.ndarray | None = None
    component_ids: np.ndarray | None = None
    development_results: list[dict[str, Any]] = []
    development_indices: list[int] = []
    shard_evidence: list[dict[str, Any]] = []
    global_offset = 0
    decoded_totals = {
        "result_json_lines_seen": 0,
        "development_result_json_lines_decoded": 0,
        "test_result_json_lines_seen": 0,
        "test_result_json_lines_decoded": 0,
    }
    for shard in shard_directories:
        tensor_path = shard / "tensors.npz"
        result_path = shard / "results.jsonl.gz"
        arrays = _selected_npz(tensor_path, selected_fields)
        case_ids = arrays["case_ids"].astype(str)
        splits = arrays["split_names"].astype(str)
        frozen = arrays["input_frozen_splits"].astype(str)
        if frozen.tolist() != splits.tolist():
            raise InputValidationError("frozen input split differs from dataset split")
        if case_ids.shape != splits.shape or not set(splits.tolist()).issubset(
            _VALID_SPLITS
        ):
            raise InputValidationError("sampler shard has invalid frozen split identity")
        names = arrays[feature_field].astype(str)
        components = arrays["component_ids"].astype(str)
        if feature_names is None:
            feature_names = names
            component_ids = components
        elif (
            names.tolist() != feature_names.tolist()
            or components.tolist() != component_ids.tolist()
        ):
            raise InputValidationError("sampler vocabulary differs across shards")
        rows, local_indices, access = _development_result_rows(
            result_path,
            case_ids=case_ids.tolist(),
            splits=splits.tolist(),
        )
        development_results.extend(rows)
        development_indices.extend(global_offset + value for value in local_indices)
        for name, value in access.items():
            decoded_totals[name] += int(value)
        shard_arrays.append(arrays)
        shard_evidence.append(
            {
                "shard": shard.name,
                "case_count": int(case_ids.size),
                "test_case_count": int(np.sum(splits == "test")),
                "tensors_sha256": file_hash(tensor_path),
                "results_sha256": file_hash(result_path),
            }
        )
        global_offset += int(case_ids.size)

    assert feature_names is not None and component_ids is not None

    def concatenate(name: str) -> np.ndarray:
        return np.concatenate([item[name] for item in shard_arrays], axis=0)

    case_ids = concatenate("case_ids").astype(str)
    splits = concatenate("split_names").astype(str)
    if len(set(case_ids.tolist())) != case_ids.size:
        raise InputValidationError("sampler dataset contains duplicate case_ids")
    if not np.any(splits == "train") or not np.any(splits == "validation"):
        raise InputValidationError("strict development package requires train and validation")
    if not np.any(splits == "test"):
        raise InputValidationError("strict development package requires a sealed test split")
    if not np.all(concatenate("sampling_weight_mask").astype(bool)):
        raise InputValidationError("all converted cases require known sampling weights")

    targets, traces = _full_target_arrays(
        development_results,
        development_indices,
        case_ids=case_ids,
        component_ids=component_ids,
    )
    sampling_document = manifest.get("source_contracts", {}).get(
        "sampling_spec_document", {}
    )
    design_seed = int(sampling_document.get("seeds", {}).get("design", 0))
    if design_seed <= 0:
        raise InputValidationError("sampler manifest lacks a positive design seed")
    source_code_hash = str(manifest.get("engine", {}).get("code_hash", ""))
    label_spec_id = str(manifest.get("source_contracts", {}).get("label_spec_id", ""))
    source_manifest_hash = file_hash(source / "manifest.json")
    test_case_ids = case_ids[splits == "test"].tolist()
    identity = {
        "conversion_version": STRICT_CONVERSION_VERSION,
        "source_dataset_id": manifest.get("dataset_id"),
        "source_manifest_hash": source_manifest_hash,
        "source_code_hash": source_code_hash,
        "case_ids": case_ids.tolist(),
        "test_case_id_order_hash": content_hash({"case_ids": test_case_ids}),
    }
    dataset_id = "damage-dataset-v2-" + content_hash(identity).split(":", 1)[1][:24]
    write_dataset_v2(
        output_directory,
        dataset_id=dataset_id,
        inputs={
            "case_ids": case_ids,
            "x": concatenate(input_field).astype(np.float64),
            "feature_names": feature_names,
            "input_mask": concatenate(mask_field).astype(bool),
        },
        component_state_traces=traces,
        component_labels=targets,
        sampling_weights={
            "case_ids": case_ids,
            "selection_probability": concatenate(
                "sampling_selection_probability"
            ).astype(np.float64),
            "design_weight": concatenate("sampling_design_weight_raw").astype(
                np.float64
            ),
            "target_measure_weight": concatenate(
                "sampling_design_weight_raw"
            ).astype(np.float64),
            "training_weight": np.ones(case_ids.size, dtype=np.float64),
            "split": splits,
            "design_seed": np.full(case_ids.size, design_seed, dtype=np.int64),
        },
        input_contract={
            **dict(input_contract),
            "source_dataset_id": manifest.get("dataset_id"),
            "source_manifest_sha256": source_manifest_hash,
            "conversion_version": STRICT_CONVERSION_VERSION,
            "sealed_test_label_policy": "opaque_source_with_masked_core_placeholders",
        },
        label_spec_id=label_spec_id,
        source_code_hash=source_code_hash,
        status=status,
        provisional_reason=provisional_reason,
    )
    output = Path(output_directory)
    split_counts = {
        name: int(np.sum(splits == name)) for name in sorted(_VALID_SPLITS)
    }
    sealing = {
        "schema_version": "dataset-v2-sealed-test@1.0.0",
        "dataset_id": dataset_id,
        "split_counts": split_counts,
        "test_packaging_operation": "opaque_source_with_masked_core_placeholders",
        "test_labels_present_in_core": False,
        "test_component_traces_present_in_core": False,
        "test_label_masks_all_false": True,
        "test_result_json_lines_decoded": 0,
        "development_result_json_lines_decoded": decoded_totals[
            "development_result_json_lines_decoded"
        ],
        "test_open_count": 0,
        "test_labels_evaluated_for_model_selection": False,
        "opening_requirement": "frozen_model_and_threshold_contract",
        "source_dataset_id": manifest.get("dataset_id"),
        "source_manifest_sha256": source_manifest_hash,
        "source_shards": shard_evidence,
        "test_case_id_order_hash": identity["test_case_id_order_hash"],
        "source_tensor_fields_read": selected_fields,
    }
    (output / "sealed_test.json").write_text(
        json.dumps(sealing, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validation = validate_dataset_v2(output, recompute_views=True)
    return {
        **validation,
        "source_dataset_id": manifest.get("dataset_id"),
        "split_counts": split_counts,
        "sealed_test": sealing,
        "result_access": decoded_totals,
        "conditional_energy_observation_count": int(
            np.sum(targets["conditional_log1p_energy_mask"])
        ),
        "conditional_hit_count_observation_count": int(
            np.sum(targets["conditional_log1p_hit_count_mask"])
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--status", default="provisional")
    parser.add_argument(
        "--provisional-reason",
        default=(
            "software-development dataset with opaque sealed-test labels; physical "
            "parameters and semantic authority remain unverified"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = convert_sampler_pilot_to_strict_development_dataset_v2(
        args.pilot,
        args.output,
        status=args.status,
        provisional_reason=args.provisional_reason,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "STRICT_CONVERSION_VERSION",
    "convert_sampler_pilot_to_strict_development_dataset_v2",
]
