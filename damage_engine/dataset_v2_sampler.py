"""Deterministic sampler-pilot to tree-independent dataset-v2 conversion.

The conversion reads archived engine results only to package immutable core
labels and replicate-level component states.  It does not compute model
selection metrics and marks any test rows as sealed for downstream consumers.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset_v2 import validate_dataset_v2, write_dataset_v2, write_tree_view
from .errors import InputValidationError
from .hurdle_targets import extract_component_hurdle_targets
from .utils import content_hash, file_hash


SAMPLER_CONVERSION_VERSION = "sampler-pilot-to-dataset-v2@1.0.0"


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputValidationError(
            "sampler dataset manifest is missing or unreadable",
            context={"path": str(path)},
        ) from exc
    if not isinstance(value, dict):
        raise InputValidationError("sampler dataset manifest must be an object")
    return value


def _result_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise InputValidationError(
            "sampler result shard is unreadable",
            context={"path": str(path)},
        ) from exc
    return rows


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise InputValidationError(
            "sampler tensor shard is unreadable",
            context={"path": str(path)},
        ) from exc


def _source_dataset_directory(pilot_directory: Path) -> Path:
    direct = pilot_directory / "manifest.json"
    nested = pilot_directory / "formal_dataset" / "manifest.json"
    if nested.is_file():
        return nested.parent
    if direct.is_file() and (pilot_directory / "shards").is_dir():
        return pilot_directory
    raise InputValidationError(
        "path is neither a sampler pilot nor a formal sharded dataset",
        context={"path": str(pilot_directory)},
    )


def _required_array(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in arrays:
        raise InputValidationError(
            "sampler tensor shard lacks a required dataset-v2 source field",
            context={"field": name},
        )
    return np.asarray(arrays[name])


def _finite_or_zero(value: Any) -> float:
    if value is None:
        return 0.0
    number = float(value)
    return number if np.isfinite(number) else 0.0


def _target_arrays(
    results: Sequence[Mapping[str, Any]], component_ids: Sequence[str]
) -> dict[str, np.ndarray]:
    shape = (len(results), len(component_ids))
    values = {
        "p_hit": np.zeros(shape, dtype=np.float64),
        "p_hit_std_error": np.zeros(shape, dtype=np.float64),
        "p_hit_ci95": np.zeros(shape + (2,), dtype=np.float64),
        "p_hit_success_count": np.zeros(shape, dtype=np.int64),
        "p_hit_trial_count": np.zeros(shape, dtype=np.int64),
        "p_hit_identification_interval": np.zeros(shape + (2,), dtype=np.float64),
        "p_hit_mask": np.zeros(shape, dtype=bool),
        "p_degraded": np.zeros(shape, dtype=np.float64),
        "p_degraded_std_error": np.zeros(shape, dtype=np.float64),
        "p_degraded_ci95": np.zeros(shape + (2,), dtype=np.float64),
        "p_degraded_identification_interval": np.zeros(shape + (2,), dtype=np.float64),
        "p_degraded_mask": np.zeros(shape, dtype=bool),
        "p_failed": np.zeros(shape, dtype=np.float64),
        "p_failed_std_error": np.zeros(shape, dtype=np.float64),
        "p_failed_ci95": np.zeros(shape + (2,), dtype=np.float64),
        "p_failed_identification_interval": np.zeros(shape + (2,), dtype=np.float64),
        "p_failed_mask": np.zeros(shape, dtype=bool),
        "conditional_log1p_energy": np.zeros(shape, dtype=np.float64),
        "conditional_log1p_energy_std_error": np.zeros(shape, dtype=np.float64),
        "conditional_log1p_energy_mask": np.zeros(shape, dtype=bool),
        "conditional_log1p_hit_count": np.zeros(shape, dtype=np.float64),
        "conditional_log1p_hit_count_std_error": np.zeros(shape, dtype=np.float64),
        "conditional_log1p_hit_count_mask": np.zeros(shape, dtype=bool),
        "derived_support": np.zeros(shape, dtype=np.int8),
        "derived_support_mask": np.zeros(shape, dtype=bool),
        "label_mask": np.zeros(shape, dtype=bool),
    }
    traces: list[dict[str, Any]] = []
    case_ids: list[str] = []
    for row_index, result in enumerate(results):
        case_id = str(result.get("case_id", ""))
        if not case_id:
            raise InputValidationError("sampler result lacks case_id")
        trace = result.get("component_state_trace")
        if not isinstance(trace, Mapping):
            raise InputValidationError(
                "dataset-v2 conversion requires capture_component_state_trace=true",
                context={"case_id": case_id},
            )
        targets = extract_component_hurdle_targets(dict(result))
        observed = targets["components"]
        if set(observed) != set(component_ids):
            raise InputValidationError(
                "component vocabulary differs between tensors and hurdle targets",
                context={"case_id": case_id},
            )
        for column, component_id in enumerate(component_ids):
            target = observed[component_id]
            hit = target["p_hit"]
            degraded = target["state_probabilities"]["p_degraded"]
            failed = target["state_probabilities"]["p_failed"]
            values["p_hit"][row_index, column] = float(hit["value"])
            values["p_hit_std_error"][row_index, column] = _finite_or_zero(
                hit["standard_error"]
            )
            values["p_hit_ci95"][row_index, column] = (
                hit["ci95"] if hit["ci95"] is not None else [0.0, 1.0]
            )
            counts = hit["binomial_observation"]
            values["p_hit_success_count"][row_index, column] = int(
                counts["success_count"]
            )
            values["p_hit_trial_count"][row_index, column] = int(
                counts["trial_count"]
            )
            values["p_hit_identification_interval"][row_index, column] = [
                hit["identification_interval"]["lower"],
                hit["identification_interval"]["upper"],
            ]
            values["p_hit_mask"][row_index, column] = bool(hit["mask"])
            for public, source in (("p_degraded", degraded), ("p_failed", failed)):
                values[public][row_index, column] = _finite_or_zero(source["value"])
                values[public + "_std_error"][row_index, column] = _finite_or_zero(
                    source["standard_error"]
                )
                values[public + "_ci95"][row_index, column] = (
                    source["ci95"] if source["ci95"] is not None else [0.0, 1.0]
                )
                values[public + "_identification_interval"][row_index, column] = [
                    source["identification_interval"]["lower"],
                    source["identification_interval"]["upper"],
                ]
                values[public + "_mask"][row_index, column] = bool(source["mask"])
            values["label_mask"][row_index, column] = bool(
                hit["mask"] and degraded["mask"] and failed["mask"]
            )
            conditional = target["conditional_positive_targets"]
            for public, source_name in (
                ("conditional_log1p_energy", "log1p_deposited_energy_j"),
                ("conditional_log1p_hit_count", "log1p_hit_count"),
            ):
                source = conditional[source_name]
                values[public][row_index, column] = _finite_or_zero(source["value"])
                values[public + "_std_error"][row_index, column] = _finite_or_zero(
                    source["standard_error"]
                )
                values[public + "_mask"][row_index, column] = bool(source["mask"])
            support = target["derived_support"]
            values["derived_support_mask"][row_index, column] = bool(support["mask"])
            values["derived_support"][row_index, column] = int(
                support["value"] if support["value"] is not None else 0
            )
        case_ids.append(case_id)
        traces.append({"case_id": case_id, "component_state_trace": dict(trace)})
    return {
        "case_ids": np.asarray(case_ids, dtype=str),
        "component_ids": np.asarray(component_ids, dtype=str),
        **values,
        "_traces": traces,
    }


def convert_sampler_pilot_to_dataset_v2(
    pilot_directory: str | Path,
    output_directory: str | Path,
    *,
    tree_candidate: Mapping[str, Any] | None = None,
    status: str = "provisional",
    provisional_reason: str = (
        "software-development dataset; physical parameters and semantic authority remain unverified"
    ),
) -> dict[str, Any]:
    """Convert all formal shards while preserving their frozen split identities."""

    source = _source_dataset_directory(Path(pilot_directory))
    manifest = _json(source / "manifest.json")
    input_contract = manifest.get("model_input_contract")
    if not isinstance(input_contract, Mapping):
        raise InputValidationError("sampler manifest lacks model_input_contract")
    input_field = str(input_contract.get("tensor_field", ""))
    mask_field = str(input_contract.get("mask_field", ""))
    if not input_field or not mask_field:
        raise InputValidationError("model input contract lacks tensor or mask field")

    shard_directories = sorted((source / "shards").glob("part-*"))
    if not shard_directories:
        raise InputValidationError("sampler formal dataset contains no shards")
    tensors: list[dict[str, np.ndarray]] = []
    results: list[dict[str, Any]] = []
    feature_names: np.ndarray | None = None
    component_ids: np.ndarray | None = None
    for shard in shard_directories:
        arrays = _load_npz(shard / "tensors.npz")
        rows = _result_rows(shard / "results.jsonl.gz")
        case_ids = _required_array(arrays, "case_ids").astype(str)
        if len(rows) != case_ids.size or [str(row.get("case_id")) for row in rows] != case_ids.tolist():
            raise InputValidationError(
                "sampler result and tensor shard case order differs",
                context={"shard": shard.name},
            )
        names = _required_array(arrays, input_field.replace("_x", "_feature_names")).astype(str)
        components = _required_array(arrays, "component_ids").astype(str)
        if feature_names is None:
            feature_names = names
            component_ids = components
        elif names.tolist() != feature_names.tolist() or components.tolist() != component_ids.tolist():
            raise InputValidationError("sampler vocabulary differs across shards")
        tensors.append(arrays)
        results.extend(rows)
    assert feature_names is not None and component_ids is not None

    def concatenate(name: str) -> np.ndarray:
        return np.concatenate([_required_array(item, name) for item in tensors], axis=0)

    case_ids = concatenate("case_ids").astype(str)
    if len(set(case_ids.tolist())) != case_ids.size:
        raise InputValidationError("sampler dataset contains duplicate case_ids")
    splits = concatenate("split_names").astype(str)
    if "input_frozen_splits" in tensors[0]:
        frozen = concatenate("input_frozen_splits").astype(str)
        if frozen.tolist() != splits.tolist():
            raise InputValidationError("frozen input split differs from dataset split")
    valid_splits = {"train", "validation", "test"}
    if not set(splits).issubset(valid_splits):
        raise InputValidationError(
            "dataset-v2 conversion accepts only frozen train/validation/test splits"
        )
    weight_mask = concatenate("sampling_weight_mask").astype(bool)
    if not np.all(weight_mask):
        raise InputValidationError("all converted cases require known sampling weights")
    target_arrays = _target_arrays(results, component_ids.tolist())
    if target_arrays["case_ids"].tolist() != case_ids.tolist():
        raise InputValidationError("hurdle target case order differs from input order")
    traces = target_arrays.pop("_traces")

    sampling_document = manifest.get("source_contracts", {}).get(
        "sampling_spec_document", {}
    )
    design_seed = int(sampling_document.get("seeds", {}).get("design", 0))
    if design_seed <= 0:
        raise InputValidationError("sampler manifest lacks a positive design seed")
    selection_probability = concatenate("sampling_selection_probability").astype(np.float64)
    design_weight = concatenate("sampling_design_weight_raw").astype(np.float64)
    source_code_hash = str(manifest.get("engine", {}).get("code_hash", ""))
    label_spec_id = str(
        manifest.get("source_contracts", {}).get("label_spec_id", "")
    )
    identity = {
        "conversion_version": SAMPLER_CONVERSION_VERSION,
        "source_dataset_id": manifest.get("dataset_id"),
        "source_manifest_hash": file_hash(source / "manifest.json"),
        "source_code_hash": source_code_hash,
        "case_ids": case_ids.tolist(),
    }
    dataset_id = "damage-dataset-v2-" + content_hash(identity).split(":", 1)[1][:24]
    report = write_dataset_v2(
        output_directory,
        dataset_id=dataset_id,
        inputs={
            "case_ids": case_ids,
            "x": concatenate(input_field).astype(np.float64),
            "feature_names": feature_names,
            "input_mask": concatenate(mask_field).astype(bool),
        },
        component_state_traces=traces,
        component_labels=target_arrays,
        sampling_weights={
            "case_ids": case_ids,
            "selection_probability": selection_probability,
            "design_weight": design_weight,
            "target_measure_weight": design_weight,
            "training_weight": np.ones(case_ids.size, dtype=np.float64),
            "split": splits,
            "design_seed": np.full(case_ids.size, design_seed, dtype=np.int64),
        },
        input_contract={
            **dict(input_contract),
            "source_dataset_id": manifest.get("dataset_id"),
            "source_manifest_sha256": file_hash(source / "manifest.json"),
            "conversion_version": SAMPLER_CONVERSION_VERSION,
        },
        label_spec_id=label_spec_id,
        source_code_hash=source_code_hash,
        status=status,
        provisional_reason=provisional_reason,
    )
    output = Path(output_directory)
    sealing = {
        "schema_version": "dataset-v2-sealed-test@1.0.0",
        "dataset_id": dataset_id,
        "split_counts": {
            name: int(np.sum(splits == name)) for name in sorted(valid_splits)
        },
        "test_packaging_operation": "deterministic_ingest_without_metric_computation",
        "test_labels_evaluated_for_model_selection": False,
        "test_open_count": 0,
        "opening_requirement": "frozen_model_and_threshold_contract",
        "source_manifest_sha256": file_hash(source / "manifest.json"),
    }
    (output / "sealed_test.json").write_text(
        json.dumps(sealing, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if tree_candidate is not None:
        report = write_tree_view(output, tree_candidate)
    final = validate_dataset_v2(output, recompute_views=True)
    return {
        **final,
        "source_dataset_id": manifest.get("dataset_id"),
        "split_counts": sealing["split_counts"],
        "sealed_test": sealing,
        "conditional_energy_observation_count": int(
            np.sum(target_arrays["conditional_log1p_energy_mask"])
        ),
        "conditional_hit_count_observation_count": int(
            np.sum(target_arrays["conditional_log1p_hit_count_mask"])
        ),
    }


__all__ = [
    "SAMPLER_CONVERSION_VERSION",
    "convert_sampler_pilot_to_dataset_v2",
]
