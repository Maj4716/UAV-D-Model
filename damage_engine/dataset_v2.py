"""Tree-independent dataset-v2 packages and replaceable offline-tree views."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from .errors import InputValidationError
from .offline_tree_candidates import compile_offline_tree_candidate
from .utils import content_hash, file_hash


DATASET_SCHEMA_VERSION = "dataset-v2@1.0.0"
TRACE_SCHEMA_VERSION = "1.0.0"
TRACE_SEMANTICS = (
    "joint_per_replicate_component_mechanism_states_before_tree_aggregation"
)
_CORE_FILES = {
    "inputs": "core/inputs.npz",
    "component_state_trace": "core/component_state_trace.jsonl.gz",
    "component_labels": "core/component_labels.npz",
    "sampling_weights": "core/sampling_weights.npz",
}
_INPUT_KEYS = {"case_ids", "x", "feature_names", "input_mask"}
_LABEL_REQUIRED_KEYS = {
    "case_ids",
    "component_ids",
    "p_hit",
    "p_hit_std_error",
    "p_hit_ci95",
    "p_degraded",
    "p_failed",
    "label_mask",
}
_LABEL_OPTIONAL_KEYS = {
    "p_hit_success_count",
    "p_hit_trial_count",
    "p_hit_identification_interval",
    "p_hit_mask",
    "p_degraded_std_error",
    "p_degraded_ci95",
    "p_degraded_identification_interval",
    "p_degraded_mask",
    "p_failed_std_error",
    "p_failed_ci95",
    "p_failed_identification_interval",
    "p_failed_mask",
    "conditional_log1p_energy",
    "conditional_log1p_energy_std_error",
    "conditional_log1p_energy_mask",
    "conditional_log1p_hit_count",
    "conditional_log1p_hit_count_std_error",
    "conditional_log1p_hit_count_mask",
    "derived_support",
    "derived_support_mask",
}
_LABEL_KEYS = _LABEL_REQUIRED_KEYS | _LABEL_OPTIONAL_KEYS
_WEIGHT_KEYS = {
    "case_ids",
    "selection_probability",
    "design_weight",
    "target_measure_weight",
    "training_weight",
    "split",
    "design_seed",
}
_VIEW_KEYS = {
    "case_ids",
    "output_names",
    "tree_outputs",
    "candidate_id",
    "candidate_hash",
    "candidate_status",
    "semantic_authority_status",
}


def _plain_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in manifest.items() if key != "manifest_hash"}


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    return content_hash(_plain_manifest(manifest))


def _json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    document = _plain_manifest(manifest)
    document["manifest_hash"] = _manifest_hash(document)
    temporary = path.with_name(path.name + ".tmp")
    _json_dump(temporary, document)
    os.replace(temporary, path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise InputValidationError(
            "dataset-v2 array archive is unreadable or contains object arrays",
            context={"path": str(path)},
        ) from exc


def _string_vector(value: Any, *, field: str, nonempty: bool = True) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise InputValidationError(f"{field} must be a one-dimensional string array")
    array = array.astype(str)
    if nonempty and (array.size == 0 or any(not item for item in array.tolist())):
        raise InputValidationError(f"{field} must contain nonempty strings")
    if len(set(array.tolist())) != array.size:
        raise InputValidationError(f"{field} must contain unique values")
    return array


def _finite_array(value: Any, *, field: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise InputValidationError(f"{field} must be a finite {ndim}-dimensional array")
    return array


def _probability_array(value: Any, *, field: str, ndim: int) -> np.ndarray:
    array = _finite_array(value, field=field, ndim=ndim)
    if np.any((array < 0.0) | (array > 1.0)):
        raise InputValidationError(f"{field} must lie in [0,1]")
    return array


def _normalize_inputs(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    unknown = set(raw) - _INPUT_KEYS
    missing = {"case_ids", "x", "feature_names"} - set(raw)
    if unknown or missing:
        raise InputValidationError(
            "dataset-v2 inputs have an invalid field set",
            context={"unknown": sorted(unknown), "missing": sorted(missing)},
        )
    case_ids = _string_vector(raw["case_ids"], field="inputs.case_ids")
    features = _string_vector(raw["feature_names"], field="inputs.feature_names")
    x = _finite_array(raw["x"], field="inputs.x", ndim=2)
    if x.shape != (case_ids.size, features.size):
        raise InputValidationError("inputs.x shape must match case_ids and feature_names")
    result = {"case_ids": case_ids, "x": x, "feature_names": features}
    if "input_mask" in raw:
        mask = np.asarray(raw["input_mask"])
        if mask.shape != x.shape or mask.dtype.kind != "b":
            raise InputValidationError("inputs.input_mask must be boolean and match inputs.x")
        result["input_mask"] = mask
    return result


def _normalize_labels(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    unknown = set(raw) - _LABEL_KEYS
    missing = _LABEL_REQUIRED_KEYS - set(raw)
    if unknown or missing:
        raise InputValidationError(
            "dataset-v2 component labels have an invalid field set",
            context={"unknown": sorted(unknown), "missing": sorted(missing)},
        )
    case_ids = _string_vector(raw["case_ids"], field="labels.case_ids")
    component_ids = _string_vector(raw["component_ids"], field="labels.component_ids")
    expected = (case_ids.size, component_ids.size)
    result: dict[str, np.ndarray] = {
        "case_ids": case_ids,
        "component_ids": component_ids,
    }
    for field in ("p_hit", "p_degraded", "p_failed"):
        array = _probability_array(raw[field], field=f"labels.{field}", ndim=2)
        if array.shape != expected:
            raise InputValidationError(f"labels.{field} shape must be {expected}")
        result[field] = array
    standard_error = _finite_array(
        raw["p_hit_std_error"], field="labels.p_hit_std_error", ndim=2
    )
    if standard_error.shape != expected or np.any(standard_error < 0.0):
        raise InputValidationError("labels.p_hit_std_error must be nonnegative and aligned")
    interval = _probability_array(raw["p_hit_ci95"], field="labels.p_hit_ci95", ndim=3)
    if interval.shape != expected + (2,) or np.any(interval[..., 0] > interval[..., 1]):
        raise InputValidationError("labels.p_hit_ci95 must have ordered bounds and shape [B,C,2]")
    mask = np.asarray(raw["label_mask"])
    if mask.shape != expected or mask.dtype.kind != "b":
        raise InputValidationError("labels.label_mask must be boolean and aligned")
    result.update(
        {
            "p_hit_std_error": standard_error,
            "p_hit_ci95": interval,
            "label_mask": mask,
        }
    )

    target_mask_fields = ("p_hit_mask", "p_degraded_mask", "p_failed_mask")
    target_mask_presence = [field in raw for field in target_mask_fields]
    if any(target_mask_presence) and not all(target_mask_presence):
        raise InputValidationError(
            "dataset-v2 probability target masks must be supplied together"
        )
    if all(target_mask_presence):
        target_masks: dict[str, np.ndarray] = {}
        for field in target_mask_fields:
            target_mask = np.asarray(raw[field])
            if target_mask.shape != expected or target_mask.dtype.kind != "b":
                raise InputValidationError(
                    f"labels.{field} must be boolean and aligned"
                )
            target_masks[field] = target_mask
            result[field] = target_mask
        combined = np.logical_and.reduce(list(target_masks.values()))
        if not np.array_equal(mask, combined):
            raise InputValidationError(
                "labels.label_mask must equal the intersection of probability target masks"
            )

    state_observed_mask = mask
    if all(target_mask_presence):
        state_observed_mask = target_masks["p_degraded_mask"] & target_masks[
            "p_failed_mask"
        ]
    if np.any(
        result["p_failed"][state_observed_mask]
        > result["p_degraded"][state_observed_mask] + 1.0e-12
    ):
        raise InputValidationError(
            "labels require p_failed <= p_degraded wherever both state targets are observed"
        )

    probability_fields = {
        "p_degraded_ci95": 3,
        "p_degraded_identification_interval": 3,
        "p_failed_ci95": 3,
        "p_failed_identification_interval": 3,
        "p_hit_identification_interval": 3,
    }
    for field, ndim in probability_fields.items():
        if field not in raw:
            continue
        array = _probability_array(raw[field], field=f"labels.{field}", ndim=ndim)
        if array.shape != expected + (2,) or np.any(array[..., 0] > array[..., 1]):
            raise InputValidationError(
                f"labels.{field} must contain ordered [lower,upper] intervals"
            )
        point_field = field.removesuffix("_identification_interval")
        if point_field != field and np.any(
            (result[point_field] < array[..., 0] - 1.0e-12)
            | (result[point_field] > array[..., 1] + 1.0e-12)
        ):
            raise InputValidationError(
                f"labels.{field} must contain labels.{point_field}"
            )
        result[field] = array

    for field in ("p_degraded_std_error", "p_failed_std_error"):
        if field not in raw:
            continue
        array = _finite_array(raw[field], field=f"labels.{field}", ndim=2)
        if array.shape != expected or np.any(array < 0.0):
            raise InputValidationError(f"labels.{field} must be nonnegative and aligned")
        result[field] = array

    if ("p_hit_success_count" in raw) != ("p_hit_trial_count" in raw):
        raise InputValidationError(
            "labels.p_hit_success_count and p_hit_trial_count must be supplied together"
        )
    if "p_hit_success_count" in raw:
        successes = np.asarray(raw["p_hit_success_count"])
        trials = np.asarray(raw["p_hit_trial_count"])
        if (
            successes.shape != expected
            or trials.shape != expected
            or successes.dtype.kind not in {"i", "u"}
            or trials.dtype.kind not in {"i", "u"}
            or np.any(trials < 1)
            or np.any(successes < 0)
            or np.any(successes > trials)
        ):
            raise InputValidationError(
                "labels binomial hit counts must be aligned integers with 0 <= success <= trial"
            )
        result["p_hit_success_count"] = successes.astype(np.int64)
        result["p_hit_trial_count"] = trials.astype(np.int64)

    conditional_pairs = (
        (
            "conditional_log1p_energy",
            "conditional_log1p_energy_std_error",
            "conditional_log1p_energy_mask",
        ),
        (
            "conditional_log1p_hit_count",
            "conditional_log1p_hit_count_std_error",
            "conditional_log1p_hit_count_mask",
        ),
    )
    for value_field, error_field, mask_field in conditional_pairs:
        present = {name in raw for name in (value_field, error_field, mask_field)}
        if len(present) != 1:
            raise InputValidationError(
                f"labels conditional target {value_field} requires value, standard error and mask"
            )
        if True not in present:
            continue
        value = _finite_array(raw[value_field], field=f"labels.{value_field}", ndim=2)
        error = _finite_array(raw[error_field], field=f"labels.{error_field}", ndim=2)
        mask = np.asarray(raw[mask_field])
        if (
            value.shape != expected
            or error.shape != expected
            or mask.shape != expected
            or mask.dtype.kind != "b"
            or np.any(value < 0.0)
            or np.any(error < 0.0)
        ):
            raise InputValidationError(
                f"labels conditional target {value_field} is invalid or misaligned"
            )
        result[value_field] = value
        result[error_field] = error
        result[mask_field] = mask

    if ("derived_support" in raw) != ("derived_support_mask" in raw):
        raise InputValidationError(
            "labels.derived_support and derived_support_mask must be supplied together"
        )
    if "derived_support" in raw:
        support = np.asarray(raw["derived_support"])
        support_mask = np.asarray(raw["derived_support_mask"])
        if (
            support.shape != expected
            or support_mask.shape != expected
            or support_mask.dtype.kind != "b"
            or support.dtype.kind not in {"i", "u", "b"}
            or np.any((support < 0) | (support > 1))
        ):
            raise InputValidationError("labels.derived_support is invalid or misaligned")
        result["derived_support"] = support.astype(np.int8)
        result["derived_support_mask"] = support_mask
    return result


def _normalize_weights(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    unknown = set(raw) - _WEIGHT_KEYS
    missing = _WEIGHT_KEYS - set(raw)
    if unknown or missing:
        raise InputValidationError(
            "dataset-v2 sampling weights have an invalid field set",
            context={"unknown": sorted(unknown), "missing": sorted(missing)},
        )
    case_ids = _string_vector(raw["case_ids"], field="weights.case_ids")
    count = case_ids.size
    result: dict[str, np.ndarray] = {"case_ids": case_ids}
    bounds = {
        "selection_probability": (0.0, 1.0, True),
        "design_weight": (0.0, math.inf, True),
        "target_measure_weight": (0.0, math.inf, False),
        "training_weight": (0.0, math.inf, False),
    }
    for field, (lower, upper, strict_lower) in bounds.items():
        array = _finite_array(raw[field], field=f"weights.{field}", ndim=1)
        invalid_lower = array <= lower if strict_lower else array < lower
        if array.shape != (count,) or np.any(invalid_lower) or np.any(array > upper):
            raise InputValidationError(f"weights.{field} has invalid values or shape")
        result[field] = array
    split = np.asarray(raw["split"])
    if split.shape != (count,) or split.dtype.kind not in {"U", "S"}:
        raise InputValidationError("weights.split must be a string vector aligned to cases")
    seed = np.asarray(raw["design_seed"])
    if seed.shape != (count,) or seed.dtype.kind not in {"i", "u"}:
        raise InputValidationError("weights.design_seed must be an integer vector aligned to cases")
    result["split"] = split.astype(str)
    result["design_seed"] = seed.astype(np.int64)
    return result


def validate_component_state_trace(trace: Mapping[str, Any]) -> None:
    if trace.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise InputValidationError("unsupported component-state trace schema version")
    if trace.get("semantics") != TRACE_SEMANTICS:
        raise InputValidationError("component-state trace semantics are incompatible")
    records = trace.get("records")
    if not isinstance(records, list) or not records:
        raise InputValidationError("component-state trace requires replicate records")
    if trace.get("replicate_count") != len(records):
        raise InputValidationError("component-state trace replicate_count is inconsistent")
    for expected_index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("replicate_index") != expected_index:
            raise InputValidationError(
                "component-state trace replicate indices must be contiguous and ordered"
            )
        if not isinstance(record.get("component_mechanism_states"), Mapping):
            raise InputValidationError("component-state trace record lacks mechanism states")


def _normalize_traces(
    traces: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    case_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if isinstance(traces, Mapping):
        wrappers = [
            {"case_id": case_id, "component_state_trace": traces[case_id]}
            for case_id in case_ids
            if case_id in traces
        ]
    else:
        wrappers = [dict(item) for item in traces]
    if len(wrappers) != len(case_ids):
        raise InputValidationError("component-state traces must contain exactly one row per case")
    for expected_case, wrapper in zip(case_ids, wrappers, strict=True):
        if wrapper.get("case_id") != expected_case:
            raise InputValidationError("component-state trace case order must match inputs")
        trace = wrapper.get("component_state_trace")
        if not isinstance(trace, Mapping):
            raise InputValidationError("component-state trace wrapper is invalid")
        validate_component_state_trace(trace)
    return wrappers


def _write_traces(path: Path, wrappers: Iterable[Mapping[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for wrapper in wrappers:
            stream.write(
                json.dumps(wrapper, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                + "\n"
            )


def load_component_state_traces(dataset_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(dataset_dir)
    path = root / _CORE_FILES["component_state_trace"]
    wrappers: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} is not an object")
                wrappers.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise InputValidationError(
            "dataset-v2 component-state trace file is unreadable",
            context={"path": str(path)},
        ) from exc
    return wrappers


def write_dataset_v2(
    output_dir: str | Path,
    *,
    dataset_id: str,
    inputs: Mapping[str, Any],
    component_state_traces: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    component_labels: Mapping[str, Any],
    sampling_weights: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    label_spec_id: str,
    source_code_hash: str,
    status: str = "provisional",
    provisional_reason: str | None = None,
) -> dict[str, Any]:
    """Write a new immutable core package.  Existing paths are never overwritten."""

    if not dataset_id or not label_spec_id or not source_code_hash or not status:
        raise InputValidationError("dataset identity fields must be nonempty")
    normalized_inputs = _normalize_inputs(inputs)
    normalized_labels = _normalize_labels(component_labels)
    normalized_weights = _normalize_weights(sampling_weights)
    case_ids = normalized_inputs["case_ids"].tolist()
    if normalized_labels["case_ids"].tolist() != case_ids:
        raise InputValidationError("component label case_ids must match inputs")
    if normalized_weights["case_ids"].tolist() != case_ids:
        raise InputValidationError("sampling-weight case_ids must match inputs")
    wrappers = _normalize_traces(component_state_traces, case_ids)

    destination = Path(output_dir)
    if destination.exists():
        raise InputValidationError(
            "dataset-v2 output path already exists",
            context={"path": str(destination)},
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        (temporary / "core").mkdir()
        (temporary / "views").mkdir()
        _write_npz(temporary / _CORE_FILES["inputs"], normalized_inputs)
        _write_npz(temporary / _CORE_FILES["component_labels"], normalized_labels)
        _write_npz(temporary / _CORE_FILES["sampling_weights"], normalized_weights)
        _write_traces(temporary / _CORE_FILES["component_state_trace"], wrappers)
        core_files = {
            key: {"path": relative, "sha256": file_hash(temporary / relative)}
            for key, relative in _CORE_FILES.items()
        }
        manifest: dict[str, Any] = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "status": status,
            "provisional_reason": provisional_reason,
            "core_tree_independent": True,
            "case_count": len(case_ids),
            "component_count": int(normalized_labels["component_ids"].size),
            "input_contract": dict(input_contract),
            "input_contract_hash": content_hash(input_contract),
            "label_spec_id": label_spec_id,
            "source_code_hash": source_code_hash,
            "core_files": core_files,
            "views": {},
        }
        _write_manifest(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_dataset_v2(destination)


def _evaluate_candidate_cases(
    candidate_document: Mapping[str, Any],
    wrappers: Sequence[Mapping[str, Any]],
) -> tuple[Any, np.ndarray]:
    candidate = compile_offline_tree_candidate(candidate_document)
    rows: list[list[float]] = []
    for wrapper in wrappers:
        trace = wrapper["component_state_trace"]
        validate_component_state_trace(trace)
        replicate_rows: list[list[float]] = []
        for record in trace["records"]:
            evaluation = candidate.evaluate(record["component_mechanism_states"])
            replicate_rows.append(
                [float(evaluation["outputs"][name]) for name in candidate.output_names]
            )
        rows.append(np.mean(np.asarray(replicate_rows, dtype=np.float64), axis=0).tolist())
    return candidate, np.asarray(rows, dtype=np.float64)


def write_tree_view(
    dataset_dir: str | Path,
    candidate_document: Mapping[str, Any],
) -> dict[str, Any]:
    """Add a derived view without changing any dataset-v2 core artifact."""

    root = Path(dataset_dir)
    validation = validate_dataset_v2(root, recompute_views=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    wrappers = load_component_state_traces(root)
    candidate, values = _evaluate_candidate_cases(candidate_document, wrappers)
    short_hash = candidate.definition_hash.split(":", 1)[-1]
    view_name = f"tree_{short_hash}"
    relative_path = f"views/{view_name}.npz"
    path = root / relative_path
    if path.exists() or view_name in manifest["views"]:
        raise InputValidationError(
            "dataset-v2 tree view already exists",
            context={"view": view_name},
        )
    arrays = {
        "case_ids": np.asarray(validation["case_ids"], dtype=str),
        "output_names": np.asarray(candidate.output_names, dtype=str),
        "tree_outputs": values,
        "candidate_id": np.asarray(candidate.candidate_id),
        "candidate_hash": np.asarray(candidate.definition_hash),
        "candidate_status": np.asarray(candidate.status),
        "semantic_authority_status": np.asarray(candidate.authority_status),
    }
    temporary = path.with_name(path.name + ".tmp")
    try:
        _write_npz(temporary, arrays)
        os.replace(temporary, path)
        manifest["views"][view_name] = {
            "path": relative_path,
            "sha256": file_hash(path),
            "candidate_id": candidate.candidate_id,
            "candidate_hash": candidate.definition_hash,
            "candidate_status": candidate.status,
            "semantic_authority_status": candidate.authority_status,
            "derived_only": True,
            "candidate_definition": candidate.document,
        }
        _write_manifest(root / "manifest.json", manifest)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        # The path was proven absent before this transaction, so removing it
        # restores the previous package if manifest replacement fails.
        if path.exists():
            path.unlink()
        raise
    return validate_dataset_v2(root, recompute_views=True)


def _assert_exact_keys(arrays: Mapping[str, np.ndarray], expected: set[str], name: str) -> None:
    if set(arrays) != expected:
        raise InputValidationError(
            f"dataset-v2 {name} archive has an invalid field set",
            context={"found": sorted(arrays), "expected": sorted(expected)},
        )


def validate_dataset_v2(
    dataset_dir: str | Path,
    *,
    recompute_views: bool = True,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputValidationError("dataset-v2 manifest is missing or unreadable") from exc
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise InputValidationError("unsupported dataset-v2 manifest schema")
    if manifest.get("core_tree_independent") is not True:
        raise InputValidationError("dataset-v2 core must declare tree independence")
    if manifest.get("manifest_hash") != _manifest_hash(manifest):
        raise InputValidationError("dataset-v2 manifest hash mismatch")
    if set(manifest.get("core_files", {})) != set(_CORE_FILES):
        raise InputValidationError("dataset-v2 manifest core file set is invalid")
    for name, expected_relative in _CORE_FILES.items():
        entry = manifest["core_files"][name]
        if entry.get("path") != expected_relative:
            raise InputValidationError("dataset-v2 core path is not canonical")
        path = root / expected_relative
        if not path.is_file() or entry.get("sha256") != file_hash(path):
            raise InputValidationError(
                "dataset-v2 core file hash mismatch", context={"file": name}
            )
    inputs = _load_npz(root / _CORE_FILES["inputs"])
    labels = _load_npz(root / _CORE_FILES["component_labels"])
    weights = _load_npz(root / _CORE_FILES["sampling_weights"])
    _assert_exact_keys(
        inputs,
        {"case_ids", "x", "feature_names"}
        | ({"input_mask"} if "input_mask" in inputs else set()),
        "inputs",
    )
    if not _LABEL_REQUIRED_KEYS.issubset(labels) or set(labels) - _LABEL_KEYS:
        raise InputValidationError(
            "dataset-v2 component labels archive has an invalid field set",
            context={
                "missing": sorted(_LABEL_REQUIRED_KEYS - set(labels)),
                "unknown": sorted(set(labels) - _LABEL_KEYS),
            },
        )
    _assert_exact_keys(weights, _WEIGHT_KEYS, "sampling weights")
    normalized_inputs = _normalize_inputs(inputs)
    normalized_labels = _normalize_labels(labels)
    normalized_weights = _normalize_weights(weights)
    case_ids = normalized_inputs["case_ids"].tolist()
    if normalized_labels["case_ids"].tolist() != case_ids:
        raise InputValidationError("dataset-v2 label cases are not aligned")
    if normalized_weights["case_ids"].tolist() != case_ids:
        raise InputValidationError("dataset-v2 sampling cases are not aligned")
    wrappers = load_component_state_traces(root)
    _normalize_traces(wrappers, case_ids)
    if manifest.get("case_count") != len(case_ids):
        raise InputValidationError("dataset-v2 manifest case_count is inconsistent")
    if manifest.get("component_count") != int(normalized_labels["component_ids"].size):
        raise InputValidationError("dataset-v2 manifest component_count is inconsistent")

    view_reports: dict[str, Any] = {}
    for view_name, entry in manifest.get("views", {}).items():
        expected_name = "tree_" + str(entry.get("candidate_hash", "")).split(":", 1)[-1]
        if view_name != expected_name or entry.get("path") != f"views/{view_name}.npz":
            raise InputValidationError("dataset-v2 tree view name is not content addressed")
        path = root / entry["path"]
        if not path.is_file() or entry.get("sha256") != file_hash(path):
            raise InputValidationError("dataset-v2 tree view hash mismatch")
        arrays = _load_npz(path)
        _assert_exact_keys(arrays, _VIEW_KEYS, "tree view")
        if arrays["case_ids"].astype(str).tolist() != case_ids:
            raise InputValidationError("dataset-v2 tree view cases are not aligned")
        candidate = compile_offline_tree_candidate(entry["candidate_definition"])
        if str(arrays["candidate_hash"].item()) != candidate.definition_hash:
            raise InputValidationError("dataset-v2 tree view candidate hash is inconsistent")
        if str(arrays["candidate_id"].item()) != candidate.candidate_id:
            raise InputValidationError("dataset-v2 tree view candidate id is inconsistent")
        output_names = arrays["output_names"].astype(str).tolist()
        values = _probability_array(arrays["tree_outputs"], field="view.tree_outputs", ndim=2)
        if output_names != list(candidate.output_names) or values.shape != (
            len(case_ids),
            len(output_names),
        ):
            raise InputValidationError("dataset-v2 tree view output contract is inconsistent")
        maximum_error = None
        if recompute_views:
            _, expected_values = _evaluate_candidate_cases(candidate.document, wrappers)
            maximum_error = float(np.max(np.abs(values - expected_values), initial=0.0))
            if maximum_error > 1.0e-12:
                raise InputValidationError(
                    "dataset-v2 tree view does not recompute from core traces",
                    context={"view": view_name, "maximum_error": maximum_error},
                )
        view_reports[view_name] = {
            "candidate_id": candidate.candidate_id,
            "candidate_hash": candidate.definition_hash,
            "maximum_recompute_error": maximum_error,
        }
    return {
        "valid": True,
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": manifest.get("dataset_id"),
        "status": manifest.get("status"),
        "case_count": len(case_ids),
        "component_count": int(normalized_labels["component_ids"].size),
        "case_ids": case_ids,
        "component_ids": normalized_labels["component_ids"].tolist(),
        "core_tree_independent": True,
        "view_count": len(view_reports),
        "views": view_reports,
        "manifest_hash": manifest["manifest_hash"],
    }


def load_dataset_v2_core(dataset_dir: str | Path) -> dict[str, Any]:
    """Load validated core arrays and traces for selection or analysis tools."""

    root = Path(dataset_dir)
    validate_dataset_v2(root, recompute_views=False)
    return {
        "inputs": _load_npz(root / _CORE_FILES["inputs"]),
        "component_labels": _load_npz(root / _CORE_FILES["component_labels"]),
        "sampling_weights": _load_npz(root / _CORE_FILES["sampling_weights"]),
        "component_state_traces": load_component_state_traces(root),
    }
