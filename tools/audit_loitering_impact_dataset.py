"""Audit a loitering-munition/armored-target dataset for research and ML use.

This is a read-only, task-specific audit.  It accepts either a sampling-run
directory (containing ``formal_dataset``) or the completed sharded dataset
itself.  The report deliberately separates software/archive consistency from
physical validity and research admission.

The audit uses the frozen formal-case ledger for the four designed inputs and
collision checks, and uses the tensor archive for labels, masks, Monte Carlo
uncertainty and design weights.  No conclusion in this report is empirical
validation of the imported target, munition parameters or damage physics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

# Keep direct script execution bound to this checkout.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from damage_engine.audit_statistics import (
    stable_effective_sample_size,
    stable_finite_summary_statistics,
    standardized_mean_difference,
    validate_design_weight_contract,
)
from damage_engine.torch_dataset import validated_committed_shard_tensor_paths


AUDIT_VERSION = "loitering-impact-dataset-audit@1.0.0"
PROFILE_ORDER = ("small_front", "medium_front", "large_front")
SPLIT_ORDER = ("train", "validation", "test")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _content_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _summary(values: Iterable[float] | np.ndarray) -> dict[str, Any]:
    raw_array = np.asarray(
        list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64
    )
    array = raw_array[np.isfinite(raw_array)]
    denominators = {
        "total_count": int(raw_array.size),
        "finite_count": int(array.size),
        "missing_count": int(np.count_nonzero(np.isnan(raw_array))),
        "invalid_count": int(np.count_nonzero(np.isinf(raw_array))),
    }
    if array.size == 0:
        return {"count": 0, **denominators, "status": "not_estimable_no_finite_values"}
    stable = stable_finite_summary_statistics(
        array, np.asarray([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    )
    quantiles = stable["quantiles"]
    unique, counts = np.unique(array, return_counts=True)
    mode_index = int(np.argmax(counts))
    mean = stable["mean"]
    std = stable["standard_deviation"]
    return {
        "count": int(array.size),
        **denominators,
        "status": "estimated",
        "unique": int(unique.size),
        "min": float(quantiles[0]),
        "p01": float(quantiles[1]),
        "p05": float(quantiles[2]),
        "p25": float(quantiles[3]),
        "median": float(quantiles[4]),
        "p75": float(quantiles[5]),
        "p95": float(quantiles[6]),
        "p99": float(quantiles[7]),
        "max": float(quantiles[8]),
        "mean": mean,
        "std": std,
        "coefficient_of_variation": stable["coefficient_of_variation"],
        "coefficient_of_variation_status": stable["coefficient_of_variation_status"],
        "numeric_method": stable["numeric_method"],
        "numeric_scale": stable["numeric_scale"],
        "mode": float(unique[mode_index]),
        "mode_count": int(counts[mode_index]),
        "mode_fraction": float(counts[mode_index] / array.size),
        "exact_zero_fraction": float(np.mean(array == 0.0)),
        "exact_one_fraction": float(np.mean(array == 1.0)),
    }


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if total <= 0:
        return {"successes": int(successes), "total": int(total), "rate": None, "ci95": [None, None]}
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "successes": int(successes),
        "total": int(total),
        "rate": float(p),
        "ci95": [float(max(0.0, center - half)), float(min(1.0, center + half))],
        "interval_semantics": (
            "Wilson interval for the finite synthetic design sample; it is not a real-world "
            "performance or physical-validity interval"
        ),
    }


def _counts(values: Iterable[str], order: Iterable[str] = ()) -> dict[str, int]:
    counter = Counter(map(str, values))
    result = {name: int(counter.pop(name, 0)) for name in order}
    result.update({key: int(counter[key]) for key in sorted(counter)})
    return result


def _resolve_roots(requested: Path) -> tuple[Path, Path]:
    requested = requested.resolve()
    if (requested / "formal_dataset" / "manifest.json").is_file():
        return requested, requested / "formal_dataset"
    if (requested / "manifest.json").is_file() and (requested / "shards").is_dir():
        parent = requested.parent
        sampling_root = parent if (parent / "sampling_checkpoint.json").is_file() else requested
        return sampling_root, requested
    raise FileNotFoundError(
        "--dataset-root must be a sampling run containing formal_dataset/manifest.json "
        "or a completed sharded dataset containing manifest.json and shards/"
    )


def _formal_ledger_path(sampling_root: Path, archive_root: Path) -> Path | None:
    roots = [sampling_root]
    if archive_root.parent not in roots:
        roots.append(archive_root.parent)
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "rounds" / "round-000" / "formal_cases.jsonl.gz",
                root / "rounds" / "round-000" / "formal_cases.jsonl",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield value


ROW_FIELDS = (
    "case_ids",
    "case_hashes",
    "split_names",
    "split_group_ids",
    "leakage_keys",
    "replicate_counts",
    "sampling_selection_probability",
    "sampling_design_weight_raw",
    "sampling_weight_mask",
    "sampling_lineage_ids",
    "sampling_epistemic_group_ids",
    "system_y",
    "system_y_stddev",
    "system_y_ci95",
    "system_label_mask",
    "system_source_available_mask",
    "system_sample_count",
    "system_uncertainty_mask",
    "component_aux_y",
    "component_aux_y_stddev",
    "component_aux_y_ci95",
    "component_aux_label_mask",
    "component_aux_sample_count",
    "component_aux_uncertainty_mask",
    "label_task_available_mask",
    "loitering_impact_x",
    "loitering_impact_feature_mask",
)

METADATA_FIELDS = (
    "system_output_names",
    "component_aux_label_names",
    "label_task_names",
    "loitering_impact_feature_names",
)


def _load_tensors(archive_root: Path) -> tuple[dict[str, np.ndarray], dict[str, list[str]], list[Path]]:
    _manifest, committed_paths = validated_committed_shard_tensor_paths(archive_root)
    shard_paths = list(committed_paths)
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    metadata: dict[str, list[str]] = {}
    first_fields: set[str] | None = None
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as shard:
            fields = set(shard.files)
            if first_fields is None:
                first_fields = fields
            for name in ROW_FIELDS:
                if name in fields:
                    chunks[name].append(np.asarray(shard[name]))
                elif first_fields is not None and name in first_fields:
                    raise ValueError(f"tensor field {name!r} is missing from shard {path}")
            for name in METADATA_FIELDS:
                if name not in fields:
                    continue
                current = list(map(str, np.asarray(shard[name]).tolist()))
                if name in metadata and metadata[name] != current:
                    raise ValueError(f"metadata field {name!r} changes across tensor shards")
                metadata[name] = current
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}
    if "case_ids" not in arrays or "split_names" not in arrays:
        raise ValueError("tensor archive lacks required case_ids or split_names")
    row_count = int(arrays["case_ids"].shape[0])
    inconsistent = {name: value.shape[0] for name, value in arrays.items() if value.shape[0] != row_count}
    if inconsistent:
        raise ValueError(f"case-level tensor row counts disagree: {inconsistent}")
    return arrays, metadata, shard_paths


def _sampling_spec(sampling_root: Path) -> dict[str, Any] | None:
    path = sampling_root / "plan" / "sampling_spec.json"
    return _load_json(path) if path.is_file() else None


def _design_bounds(spec: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not spec:
        return {}
    result: dict[str, dict[str, Any]] = {}
    reference = spec.get("reference_distribution", {})
    for variable in reference.get("variables", []):
        if not isinstance(variable, dict):
            continue
        name = str(variable.get("name", variable.get("path", "unnamed")))
        result[name] = variable
    return result


def _budget_audit(spec: dict[str, Any] | None, splits: np.ndarray) -> dict[str, Any]:
    observed = _counts(splits.tolist(), SPLIT_ORDER)
    if not spec:
        return {"available": False, "observed_successful_split_counts": observed}
    budgets = spec.get("budgets", {})
    expected = {
        "train": int(budgets.get("train_pilot", 0)),
        "validation": int(budgets.get("validation", 0)),
        "test": int(budgets.get("test", 0)),
    }
    delta = {split: observed.get(split, 0) - expected[split] for split in SPLIT_ORDER}
    return {
        "available": True,
        "expected_split_counts": expected,
        "observed_successful_split_counts": observed,
        "delta_observed_minus_expected": delta,
        "exact_match": all(value == 0 for value in delta.values()),
        "expected_total": int(sum(expected.values())),
        "sampling_spec_id": spec.get("sampling_spec_id"),
        "sampling_schema_version": spec.get("schema_version"),
        "model_input_contract": spec.get("model_input_contract"),
    }


def _hash_binding_audit(
    sampling_root: Path,
    manifest: dict[str, Any],
    formal_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    checkpoint_path = sampling_root / "sampling_checkpoint.json"
    frozen_path = sampling_root / "plan" / "frozen_hashes.json"
    label_path = sampling_root / "plan" / "label_spec.json"
    checkpoint = _load_json(checkpoint_path) if checkpoint_path.is_file() else {}
    frozen = _load_json(frozen_path) if frozen_path.is_file() else {}
    label_spec = _load_json(label_path) if label_path.is_file() else {}
    identity = checkpoint.get("identity", {}) if isinstance(checkpoint.get("identity", {}), dict) else {}
    source = manifest.get("source_contracts", {})
    keys = (
        "target_hash",
        "damage_tree_definition_hash",
        "label_spec_hash",
        "sampling_spec_hash",
    )
    comparisons: list[dict[str, Any]] = []
    for key in keys:
        checkpoint_value = identity.get(key)
        frozen_value = frozen.get(key)
        comparisons.append(
            {
                "binding": f"checkpoint.identity.{key}=plan.frozen_hashes.{key}",
                "left": checkpoint_value,
                "right": frozen_value,
                "match": checkpoint_value is not None and checkpoint_value == frozen_value,
            }
        )
    expected_target = identity.get("target_hash")
    expected_tree = identity.get("damage_tree_definition_hash")
    expected_label = identity.get("label_spec_hash")
    manifest_targets = source.get("target_hashes", [])
    comparisons.extend(
        [
            {
                "binding": "checkpoint target hash=manifest sole target hash",
                "left": expected_target,
                "right": manifest_targets[0] if len(manifest_targets) == 1 else manifest_targets,
                "match": len(manifest_targets) == 1 and manifest_targets[0] == expected_target,
            },
            {
                "binding": "checkpoint tree hash=manifest tree hash",
                "left": expected_tree,
                "right": source.get("damage_tree_definition_hash"),
                "match": source.get("damage_tree_definition_hash") == expected_tree and expected_tree is not None,
            },
            {
                "binding": "checkpoint label hash=manifest label hash",
                "left": expected_label,
                "right": source.get("label_spec_hash"),
                "match": source.get("label_spec_hash") == expected_label and expected_label is not None,
            },
            {
                "binding": "checkpoint label hash=canonical plan label_spec hash",
                "left": expected_label,
                "right": _content_hash(label_spec) if label_spec else None,
                "match": bool(label_spec) and _content_hash(label_spec) == expected_label,
            },
            {
                "binding": "label_spec source tree hash=checkpoint tree hash",
                "left": label_spec.get("source", {}).get("damage_tree_definition_hash") if label_spec else None,
                "right": expected_tree,
                "match": bool(label_spec)
                and label_spec.get("source", {}).get("damage_tree_definition_hash") == expected_tree,
            },
        ]
    )
    case_violations = {
        "target_hash": 0,
        "damage_tree_definition_hash": 0,
        "label_spec_hash": 0,
    }
    for case in formal_cases:
        sampling = case.get("metadata", {}).get("sampling", {})
        if case.get("target_ref", {}).get("content_hash") != expected_target:
            case_violations["target_hash"] += 1
        if sampling.get("damage_tree_definition_hash") != expected_tree:
            case_violations["damage_tree_definition_hash"] += 1
        if sampling.get("label_spec_hash") != expected_label:
            case_violations["label_spec_hash"] += 1
    return {
        "checkpoint_available": bool(checkpoint),
        "frozen_hashes_available": bool(frozen),
        "label_spec_available": bool(label_spec),
        "comparisons": comparisons,
        "formal_case_binding_violations": case_violations,
        "all_bindings_match": bool(comparisons)
        and all(item["match"] for item in comparisons)
        and not any(case_violations.values()),
    }


def _case_inputs(case: dict[str, Any]) -> dict[str, Any]:
    encounter = case.get("encounter", {})
    munition = case.get("munition", {})
    origin = encounter.get("trajectory_origin_m") or [None, None, None]
    velocity = encounter.get("velocity_target_mps") or [None, None, None]
    detonation = encounter.get("detonation_position_m") or [None, None, None]
    numeric_velocity = np.asarray(velocity, dtype=np.float64)
    speed = float(np.linalg.norm(numeric_velocity)) if np.all(np.isfinite(numeric_velocity)) else None
    return {
        "profile": str(munition.get("profile_id", "missing")),
        "origin_x_m": _json_safe(origin[0]),
        "origin_y_m": _json_safe(origin[1]),
        "origin_z_m": _json_safe(origin[2]),
        "velocity_x_mps": _json_safe(velocity[0]),
        "velocity_y_mps": _json_safe(velocity[1]),
        "velocity_z_mps": _json_safe(velocity[2]),
        "impact_speed_mps": _json_safe(speed),
        "detonation_x_m": _json_safe(detonation[0]),
        "detonation_y_m": _json_safe(detonation[1]),
        "detonation_z_m": _json_safe(detonation[2]),
        "detonation_time_s": _json_safe(encounter.get("detonation_time_s")),
        "collision_distance_m": _json_safe(encounter.get("collision_distance_m")),
        "collision_armor_id": encounter.get("collision_armor_id"),
        "collision_normal": encounter.get("collision_normal_target_xyz"),
        "detonation_trigger": encounter.get("detonation_trigger"),
        "detonation_delay_s": _json_safe(encounter.get("detonation_delay_s")),
        "impact_model_id": encounter.get("impact_model_id"),
        "validation_status": munition.get("validation_status"),
    }


def _finite_feature(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = [row.get(key) for row in rows]
    return np.asarray([float(value) for value in values if value is not None], dtype=np.float64)


def _bin_coverage(values: np.ndarray, low: float, high: float, bins: int = 20) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    in_range = finite[(finite >= low) & (finite <= high)]
    counts, edges = np.histogram(in_range, bins=bins, range=(low, high))
    return {
        "declared_range": [float(low), float(high)],
        "bins": int(bins),
        "observed": int(finite.size),
        "out_of_range": int(finite.size - in_range.size),
        "occupied_bins": int(np.sum(counts > 0)),
        "empty_bins": int(np.sum(counts == 0)),
        "min_bin_count": int(np.min(counts)),
        "median_bin_count": float(np.median(counts)),
        "max_bin_count": int(np.max(counts)),
        "counts": counts.astype(int).tolist(),
        "edges": edges.astype(float).tolist(),
    }


def _profile_report(profiles: np.ndarray, splits: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {"all": _counts(profiles.tolist(), PROFILE_ORDER), "by_split": {}}
    n = max(1, profiles.size)
    expected = n / len(PROFILE_ORDER)
    result["max_absolute_fraction_deviation_from_equal_design"] = float(
        max(abs(np.sum(profiles == profile) - expected) / n for profile in PROFILE_ORDER)
    )
    for split in SPLIT_ORDER:
        mask = splits == split
        result["by_split"][split] = _counts(profiles[mask].tolist(), PROFILE_ORDER)
    return result


def _split_drift(
    aligned_rows: list[dict[str, Any]], profiles: np.ndarray, splits: np.ndarray
) -> dict[str, Any]:
    comparisons = (("train", "validation"), ("train", "test"), ("validation", "test"))
    continuous: list[dict[str, Any]] = []
    maximum_smd: float | None = None
    for key in ("origin_y_m", "origin_z_m", "impact_speed_mps"):
        values = np.asarray([row.get(key, np.nan) for row in aligned_rows], dtype=np.float64)
        records: list[dict[str, Any]] = []
        for left, right in comparisons:
            a = values[splits == left]
            b = values[splits == right]
            diagnostic = standardized_mean_difference(a, b)
            smd = diagnostic["absolute_standardized_mean_difference"]
            if smd is not None:
                maximum_smd = smd if maximum_smd is None else max(maximum_smd, smd)
            records.append(
                {
                    "comparison": f"{left}_vs_{right}",
                    "absolute_standardized_mean_difference": smd,
                    "standardized_mean_difference_status": diagnostic["status"],
                    "raw_mean_difference": diagnostic["raw_mean_difference"],
                    "pooled_standard_deviation": diagnostic["pooled_standard_deviation"],
                    "pooled_standard_deviation_status": diagnostic[
                        "pooled_standard_deviation_status"
                    ],
                    "left_count": diagnostic["left_total_count"],
                    "right_count": diagnostic["right_total_count"],
                    "left_finite_count": diagnostic["left_finite_count"],
                    "right_finite_count": diagnostic["right_finite_count"],
                    "left_nonfinite_count": diagnostic["left_nonfinite_count"],
                    "right_nonfinite_count": diagnostic["right_nonfinite_count"],
                    "left_missing_count": diagnostic["left_missing_count"],
                    "right_missing_count": diagnostic["right_missing_count"],
                    "left_invalid_count": diagnostic["left_invalid_count"],
                    "right_invalid_count": diagnostic["right_invalid_count"],
                }
            )
        continuous.append({"feature": key, "comparisons": records})

    categorical: list[dict[str, Any]] = []
    maximum_tv = 0.0
    for left, right in comparisons:
        left_mask = splits == left
        right_mask = splits == right
        left_n = int(np.sum(left_mask))
        right_n = int(np.sum(right_mask))
        if left_n and right_n:
            left_probability = np.asarray(
                [np.mean(profiles[left_mask] == profile) for profile in PROFILE_ORDER], dtype=np.float64
            )
            right_probability = np.asarray(
                [np.mean(profiles[right_mask] == profile) for profile in PROFILE_ORDER], dtype=np.float64
            )
            total_variation = float(0.5 * np.sum(np.abs(left_probability - right_probability)))
            maximum_tv = max(maximum_tv, total_variation)
        else:
            total_variation = None
        categorical.append(
            {
                "comparison": f"{left}_vs_{right}",
                "profile_total_variation_distance": total_variation,
                "left_count": left_n,
                "right_count": right_n,
            }
        )
    return {
        "continuous": continuous,
        "profile": categorical,
        "maximum_absolute_standardized_mean_difference": maximum_smd,
        "maximum_profile_total_variation_distance": maximum_tv,
        "interpretation": (
            "Descriptive balance diagnostics only. Because splits are generated by a designed sampler, "
            "these are not randomization-test p-values."
        ),
    }


def _input_distribution(
    aligned_rows: list[dict[str, Any]], profiles: np.ndarray, splits: np.ndarray, spec: dict[str, Any] | None
) -> dict[str, Any]:
    keys = (
        "origin_x_m",
        "origin_y_m",
        "origin_z_m",
        "velocity_x_mps",
        "velocity_y_mps",
        "velocity_z_mps",
        "impact_speed_mps",
        "detonation_x_m",
        "detonation_y_m",
        "detonation_z_m",
        "detonation_time_s",
        "collision_distance_m",
    )
    continuous: dict[str, Any] = {}
    for key in keys:
        all_values = np.asarray(
            [float(row[key]) if row.get(key) is not None else np.nan for row in aligned_rows], dtype=np.float64
        )
        record: dict[str, Any] = {"all": _summary(all_values)}
        record["by_split"] = {
            split: _summary(all_values[splits == split]) for split in SPLIT_ORDER
        }
        record["by_profile"] = {
            profile: _summary(all_values[profiles == profile]) for profile in PROFILE_ORDER
        }
        continuous[key] = record

    variables = _design_bounds(spec)
    coverages: dict[str, Any] = {}
    name_to_key = {
        "trajectory_origin_y_m": "origin_y_m",
        "trajectory_origin_z_m": "origin_z_m",
        "impact_speed_mps": "impact_speed_mps",
    }
    for name, key in name_to_key.items():
        variable = variables.get(name)
        if variable and variable.get("distribution") == "uniform":
            values = np.asarray(
                [float(row[key]) if row.get(key) is not None else np.nan for row in aligned_rows], dtype=np.float64
            )
            coverages[name] = _bin_coverage(values, float(variable["low"]), float(variable["high"]))

    input_matrix = np.column_stack(
        [
            np.asarray([row.get("origin_y_m", np.nan) for row in aligned_rows], dtype=np.float64),
            np.asarray([row.get("origin_z_m", np.nan) for row in aligned_rows], dtype=np.float64),
            np.asarray([row.get("impact_speed_mps", np.nan) for row in aligned_rows], dtype=np.float64),
        ]
    )
    finite_rows = np.all(np.isfinite(input_matrix), axis=1)
    correlation = np.corrcoef(input_matrix[finite_rows], rowvar=False) if np.sum(finite_rows) > 1 else np.full((3, 3), np.nan)
    off_diagonal = np.abs(correlation[np.triu_indices(3, 1)])
    return {
        "profile": _profile_report(profiles, splits),
        "continuous": continuous,
        "declared_range_coverage": coverages,
        "split_drift": _split_drift(aligned_rows, profiles, splits),
        "continuous_design_correlation": {
            "feature_names": ["origin_y_m", "origin_z_m", "impact_speed_mps"],
            "pearson_matrix": [[_json_safe(x) for x in row] for row in correlation],
            "maximum_absolute_off_diagonal": _json_safe(np.max(off_diagonal)) if off_diagonal.size else None,
            "interpretation": "descriptive design diagnostic, not an independence test",
        },
    }


def _learner_input_audit(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, list[str]],
    aligned_rows: list[dict[str, Any]],
    splits: np.ndarray,
) -> dict[str, Any]:
    """Check the explicit six-value model-facing tensor, when present."""

    if "loitering_impact_x" not in arrays:
        return {
            "available": False,
            "expected_contract": "loitering_impact_state_only@1.0.0",
            "expected_tensor": "loitering_impact_x",
        }
    values = np.asarray(arrays["loitering_impact_x"], dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("loitering_impact_x must be a rank-two [B,F] tensor")
    mask = np.asarray(
        arrays.get("loitering_impact_feature_mask", np.ones(values.shape, dtype=bool)),
        dtype=bool,
    )
    if mask.shape != values.shape:
        raise ValueError("loitering_impact_feature_mask shape differs from loitering_impact_x")
    names = metadata.get("loitering_impact_feature_names", [])
    if len(names) != values.shape[1]:
        names = [f"feature_{index}" for index in range(values.shape[1])]

    expected_rows: list[list[float]] = []
    for row in aligned_rows:
        profile = str(row["profile"])
        expected_rows.append(
            [
                *(1.0 if profile == candidate else 0.0 for candidate in PROFILE_ORDER),
                float(row["origin_y_m"]),
                float(row["origin_z_m"]),
                float(row["impact_speed_mps"]),
            ]
        )
    expected = np.asarray(expected_rows, dtype=np.float64)
    comparable = expected.shape == values.shape
    max_error = (
        float(np.max(np.abs(values - expected))) if comparable and values.size else None
    )
    train = splits == "train"
    features: list[dict[str, Any]] = []
    constant_train: list[str] = []
    for index, name in enumerate(names):
        valid = mask[:, index] & np.isfinite(values[:, index])
        train_valid = valid & train
        train_stats = _summary(values[train_valid, index])
        if train_stats.get("count", 0) and train_stats.get("unique", 0) <= 1:
            constant_train.append(name)
        features.append(
            {
                "name": name,
                "mask_fraction": float(np.mean(mask[:, index])),
                "finite_fraction": float(np.mean(np.isfinite(values[:, index]))),
                "all": _summary(values[valid, index]),
                "train": train_stats,
            }
        )

    duplicate_signatures: dict[bytes, set[str]] = defaultdict(set)
    duplicate_counts: Counter[bytes] = Counter()
    for row, split in zip(values, splits, strict=True):
        signature = np.ascontiguousarray(row, dtype=np.float64).tobytes()
        duplicate_signatures[signature].add(str(split))
        duplicate_counts[signature] += 1
    duplicate_extra = int(sum(count - 1 for count in duplicate_counts.values() if count > 1))
    cross_split = int(sum(len(group) > 1 for group in duplicate_signatures.values()))

    train_std = np.asarray(
        [float(feature["train"].get("std", math.nan)) for feature in features],
        dtype=np.float64,
    )
    positive_std = train_std[np.isfinite(train_std) & (train_std > 0.0)]
    scale_ratio = float(np.max(positive_std) / np.min(positive_std)) if positive_std.size else None
    one_hot_valid = False
    if values.shape[1] >= 3:
        one_hot = values[:, :3]
        one_hot_valid = bool(
            np.all(np.isin(one_hot, [0.0, 1.0]))
            and np.all(np.sum(one_hot, axis=1) == 1.0)
        )
    return {
        "available": True,
        "shape": list(values.shape),
        "feature_names": names,
        "all_mask_true": bool(np.all(mask)),
        "all_values_finite": bool(np.all(np.isfinite(values[mask]))),
        "profile_one_hot_valid": one_hot_valid,
        "formal_case_reconstruction_comparable": comparable,
        "formal_case_reconstruction_max_abs_error": max_error,
        "constant_train_features": constant_train,
        "feature_statistics": features,
        "duplicate_extra_rows": duplicate_extra,
        "cross_split_duplicate_groups": cross_split,
        "nonzero_train_std_scale_ratio": scale_ratio,
        "normalization_guidance": (
            "Fit continuous-feature normalization on Train only and apply the frozen artifact to "
            "Validation/Test. Keep the three one-hot profile indicators unscaled or explicitly document "
            "their transform. Do not normalize using all splits."
        ),
    }


def _duplicate_and_leakage(
    arrays: dict[str, np.ndarray], aligned_rows: list[dict[str, Any]], splits: np.ndarray
) -> dict[str, Any]:
    signatures: list[str] = []
    for row in aligned_rows:
        payload = [
            row.get("profile"),
            row.get("origin_x_m"),
            row.get("origin_y_m"),
            row.get("origin_z_m"),
            row.get("velocity_x_mps"),
            row.get("velocity_y_mps"),
            row.get("velocity_z_mps"),
        ]
        signatures.append(hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest())

    def repeated(values: Iterable[str]) -> tuple[int, int]:
        counter = Counter(map(str, values))
        groups = sum(count > 1 for count in counter.values())
        extra = sum(count - 1 for count in counter.values() if count > 1)
        return int(groups), int(extra)

    def cross_split(values: Iterable[str]) -> dict[str, Any]:
        seen: dict[str, set[str]] = defaultdict(set)
        for value, split in zip(values, splits, strict=True):
            seen[str(value)].add(str(split))
        rows = [
            {"value": value, "splits": sorted(groups)}
            for value, groups in seen.items()
            if len(groups) > 1
        ]
        return {"count": len(rows), "examples": rows[:20]}

    signature_groups, signature_extra = repeated(signatures)
    case_hashes = arrays.get("case_hashes", np.asarray([], dtype=str)).astype(str)
    case_hash_groups, case_hash_extra = repeated(case_hashes.tolist()) if case_hashes.size else (0, 0)
    result: dict[str, Any] = {
        "designed_input_signature": {
            "fields": ["profile", "origin_xyz_m", "velocity_xyz_mps"],
            "duplicate_groups": signature_groups,
            "duplicate_extra_rows": signature_extra,
            "cross_split": cross_split(signatures),
        },
        "case_hash": {
            "duplicate_groups": case_hash_groups,
            "duplicate_extra_rows": case_hash_extra,
            "cross_split": cross_split(case_hashes.tolist()) if case_hashes.size else {"count": 0, "examples": []},
        },
    }
    for field in ("split_group_ids", "leakage_keys", "sampling_lineage_ids", "sampling_epistemic_group_ids"):
        if field in arrays:
            values = arrays[field].astype(str).tolist()
            groups, extra = repeated(values)
            result[field] = {
                "duplicate_groups": groups,
                "duplicate_extra_rows": extra,
                "cross_split": cross_split(values),
            }
    result["cross_split_violation_count"] = int(
        sum(block.get("cross_split", {}).get("count", 0) for block in result.values())
    )
    return result


def _event_breakdown(event: np.ndarray, splits: np.ndarray, profiles: np.ndarray) -> dict[str, Any]:
    event = np.asarray(event, dtype=bool)
    result: dict[str, Any] = {"all": _wilson(int(np.sum(event)), int(event.size)), "by_split": {}, "by_profile": {}}
    for split in SPLIT_ORDER:
        mask = splits == split
        result["by_split"][split] = _wilson(int(np.sum(event & mask)), int(np.sum(mask)))
    for profile in PROFILE_ORDER:
        mask = profiles == profile
        result["by_profile"][profile] = _wilson(int(np.sum(event & mask)), int(np.sum(mask)))
    return result


def _label_audit(
    arrays: dict[str, np.ndarray], metadata: dict[str, list[str]], splits: np.ndarray, profiles: np.ndarray
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    system_names = metadata.get("system_output_names", [])
    if {"system_y", "system_label_mask"}.issubset(arrays):
        values = np.asarray(arrays["system_y"], dtype=np.float64)
        mask = np.asarray(arrays["system_label_mask"], dtype=bool)
        outputs: list[dict[str, Any]] = []
        for index in range(values.shape[1]):
            name = system_names[index] if index < len(system_names) else f"system_output_{index}"
            available = mask[:, index]
            available_values = values[available, index]
            row: dict[str, Any] = {
                "name": name,
                "availability": _wilson(int(np.sum(available)), int(available.size)),
                "values": _summary(available_values),
                "nonzero_case_count": int(np.sum(available_values > 0.0)),
                "threshold_ge_0_5_case_count": int(np.sum(available_values >= 0.5)),
                "by_split": {},
            }
            for split in SPLIT_ORDER:
                selected = available & (splits == split)
                split_values = values[selected, index]
                row["by_split"][split] = {
                    "available": int(np.sum(selected)),
                    "values": _summary(split_values),
                    "nonzero_case_count": int(np.sum(split_values > 0.0)),
                    "threshold_ge_0_5_case_count": int(np.sum(split_values >= 0.5)),
                }
            outputs.append(row)
        report["system_outputs"] = outputs
        report["system_mask_overall_fraction"] = float(np.mean(mask)) if mask.size else None
        if "system_source_available_mask" in arrays:
            source_mask = np.asarray(arrays["system_source_available_mask"], dtype=bool)
            report["system_source_available_mask_overall_fraction"] = (
                float(np.mean(source_mask)) if source_mask.size else None
            )
            report["system_source_mask_matches_label_mask"] = bool(
                source_mask.shape == mask.shape and np.array_equal(source_mask, mask)
            )
        else:
            report["system_source_available_mask_overall_fraction"] = None
            report["system_source_mask_matches_label_mask"] = False
    else:
        report["system_outputs"] = []
        report["system_mask_overall_fraction"] = None

    task_names = metadata.get("label_task_names", [])
    if "label_task_available_mask" in arrays:
        task_mask = np.asarray(arrays["label_task_available_mask"], dtype=bool)
        report["task_availability"] = [
            {
                "name": task_names[index] if index < len(task_names) else f"task_{index}",
                "availability": _wilson(int(np.sum(task_mask[:, index])), int(task_mask.shape[0])),
            }
            for index in range(task_mask.shape[1])
        ]
        report["all_declared_tasks_fully_available"] = bool(
            task_mask.size and np.all(task_mask)
        )
    else:
        report["task_availability"] = []
        report["all_declared_tasks_fully_available"] = False

    aux_names = metadata.get("component_aux_label_names", [])
    aux_values = arrays.get("component_aux_y")
    aux_mask = arrays.get("component_aux_label_mask")
    component_events: dict[str, Any] = {}
    if aux_values is not None and aux_mask is not None:
        aux_values = np.asarray(aux_values, dtype=np.float64)
        aux_mask = np.asarray(aux_mask, dtype=bool)
        for label in ("fragment_hits", "fragment_penetrations"):
            if label not in aux_names:
                continue
            index = aux_names.index(label)
            valid = aux_mask[:, :, index]
            totals = np.sum(np.where(valid, aux_values[:, :, index], 0.0), axis=1)
            event = totals > 0.0
            component_events[label] = {
                "case_total": _summary(totals),
                "case_event": _event_breakdown(event, splits, profiles),
                "component_cell_availability_fraction": float(np.mean(valid)),
            }
    report["component_events"] = component_events
    return report


def _monte_carlo_audit(
    arrays: dict[str, np.ndarray], metadata: dict[str, list[str]]
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "replicate_counts": _summary(arrays.get("replicate_counts", np.asarray([], dtype=float))),
        "interpretation": (
            "CI and standard-deviation arrays quantify simulator Monte Carlo uncertainty only; "
            "they do not cover target-model, munition-parameter, structural-model or real-world discrepancy"
        ),
    }
    if {"system_y_ci95", "system_label_mask"}.issubset(arrays):
        ci = np.asarray(arrays["system_y_ci95"], dtype=np.float64)
        mask = np.asarray(arrays["system_label_mask"], dtype=bool)
        widths = ci[:, :, 1] - ci[:, :, 0]
        names = metadata.get("system_output_names", [])
        report["system_ci95_width"] = {
            "all_available": _summary(widths[mask]),
            "by_output": [
                {
                    "name": names[index] if index < len(names) else f"system_output_{index}",
                    "width": _summary(widths[mask[:, index], index]),
                }
                for index in range(widths.shape[1])
            ],
        }
    if "system_sample_count" in arrays:
        sample_count = np.asarray(arrays["system_sample_count"], dtype=np.int64)
        valid = np.asarray(arrays.get("system_label_mask", np.ones_like(sample_count)), dtype=bool)
        report["system_sample_count"] = _summary(sample_count[valid])
    if "system_uncertainty_mask" in arrays:
        uncertainty_mask = np.asarray(arrays["system_uncertainty_mask"], dtype=bool)
        valid = np.asarray(arrays.get("system_label_mask", np.ones_like(uncertainty_mask)), dtype=bool)
        report["system_uncertainty_estimable_fraction"] = (
            float(np.mean(uncertainty_mask[valid])) if np.any(valid) else None
        )
    return report


def _weight_audit(arrays: dict[str, np.ndarray], splits: np.ndarray) -> dict[str, Any]:
    if "sampling_design_weight_raw" not in arrays:
        return {"available": False}
    raw_weights = np.asarray(arrays["sampling_design_weight_raw"])
    raw_mask = arrays.get("sampling_weight_mask", np.ones(raw_weights.shape, dtype=bool))
    raw_probability = arrays.get("sampling_selection_probability")
    weights, eligible, probabilities = validate_design_weight_contract(
        raw_weights, raw_mask, raw_probability
    )
    if splits.ndim != 1 or splits.shape != weights.shape:
        raise ValueError("split_names must share the design-weight axis")

    def summarize(selected: np.ndarray) -> dict[str, Any]:
        selected = np.asarray(selected, dtype=bool)
        if selected.ndim != 1 or selected.shape != weights.shape:
            raise ValueError("weight-audit selection mask must share the design-weight axis")
        selected_eligible = selected & eligible
        current = weights[selected_eligible]
        total_count = int(np.count_nonzero(selected))
        finite_weight_count = int(np.count_nonzero(selected & np.isfinite(weights)))
        eligible_count = int(np.count_nonzero(selected_eligible))
        denominator_report: dict[str, Any] = {
            "total_count": total_count,
            "finite_weight_count": finite_weight_count,
            "eligible_count": eligible_count,
            "eligible_finite_weight_count": int(
                np.count_nonzero(selected_eligible & np.isfinite(weights))
            ),
            "eligible_invalid_weight_count": 0,
        }
        if probabilities is not None:
            denominator_report.update(
                {
                    "finite_selection_probability_count": int(
                        np.count_nonzero(selected & np.isfinite(probabilities))
                    ),
                    "eligible_finite_selection_probability_count": int(
                        np.count_nonzero(selected_eligible & np.isfinite(probabilities))
                    ),
                    "eligible_invalid_selection_probability_count": 0,
                }
            )
        if current.size == 0:
            return {
                **denominator_report,
                "count": 0,
                "ess": None,
                "ess_fraction": None,
                "ess_status": "not_estimable_no_eligible_rows",
                "weight": {"count": 0},
            }
        ess = stable_effective_sample_size(current)
        return {
            **denominator_report,
            "count": int(current.size),
            "ess": ess,
            "ess_fraction": float(ess / current.size),
            "ess_status": "estimated",
            "weight": _summary(current),
        }

    total_count = int(weights.size)
    report: dict[str, Any] = {
        "available": True,
        "contract_status": "validated_fail_closed",
        "total_count": total_count,
        "finite_weight_count": int(np.count_nonzero(np.isfinite(weights))),
        "eligible_count": int(np.count_nonzero(eligible)),
        "eligible_invalid_weight_count": 0,
        "eligible_fraction": (
            float(np.count_nonzero(eligible) / total_count) if total_count else None
        ),
        "all": summarize(np.ones(weights.shape, dtype=bool)),
        "by_split": {split: summarize(splits == split) for split in SPLIT_ORDER},
    }
    if probabilities is not None:
        report["selection_probability"] = _summary(probabilities[eligible])
        report["finite_selection_probability_count"] = int(
            np.count_nonzero(np.isfinite(probabilities))
        )
        report["eligible_finite_selection_probability_count"] = int(
            np.count_nonzero(eligible & np.isfinite(probabilities))
        )
        report["eligible_invalid_selection_probability_count"] = 0
        report["positivity_violations"] = 0
    return report


def _collision_audit(cases: list[dict[str, Any]]) -> dict[str, Any]:
    time_errors: list[float] = []
    position_errors: list[float] = []
    normal_errors: list[float] = []
    delays: list[float] = []
    triggers: list[str] = []
    armor_ids: list[str] = []
    impact_models: list[str] = []
    incomplete = 0
    for case in cases:
        encounter = case.get("encounter", {})
        try:
            origin = np.asarray(encounter["trajectory_origin_m"], dtype=np.float64)
            velocity = np.asarray(encounter["velocity_target_mps"], dtype=np.float64)
            detonation = np.asarray(encounter["detonation_position_m"], dtype=np.float64)
            time_s = float(encounter["detonation_time_s"])
            distance = float(encounter["collision_distance_m"])
            speed = float(np.linalg.norm(velocity))
            if origin.shape != (3,) or velocity.shape != (3,) or detonation.shape != (3,) or speed <= 0.0:
                raise ValueError
            time_errors.append(abs(time_s - distance / speed))
            position_errors.append(float(np.linalg.norm(detonation - (origin + velocity * time_s))))
            normal = np.asarray(encounter["collision_normal_target_xyz"], dtype=np.float64)
            normal_errors.append(abs(float(np.linalg.norm(normal)) - 1.0))
            delays.append(float(encounter.get("detonation_delay_s", math.nan)))
            triggers.append(str(encounter.get("detonation_trigger", "missing")))
            armor_ids.append(str(encounter.get("collision_armor_id", "missing")))
            impact_models.append(str(encounter.get("impact_model_id", "missing")))
        except (KeyError, TypeError, ValueError, IndexError):
            incomplete += 1
    tolerance_time_s = 1e-10
    tolerance_position_m = 1e-8
    tolerance_normal = 1e-8
    return {
        "checked_cases": int(len(time_errors)),
        "incomplete_cases": int(incomplete),
        "detonation_trigger_counts": _counts(triggers),
        "collision_armor_counts": _counts(armor_ids),
        "impact_model_counts": _counts(impact_models),
        "detonation_delay_s": _summary(np.asarray(delays)),
        "time_equals_distance_over_speed_absolute_error_s": _summary(np.asarray(time_errors)),
        "trajectory_position_absolute_error_m": _summary(np.asarray(position_errors)),
        "collision_normal_unit_error": _summary(np.asarray(normal_errors)),
        "tolerances": {
            "time_error_s": tolerance_time_s,
            "position_error_m": tolerance_position_m,
            "normal_norm_error": tolerance_normal,
        },
        "violations": {
            "trigger_not_first_armor_contact": int(sum(value != "first_armor_contact" for value in triggers)),
            "nonzero_or_missing_delay": int(sum(not math.isfinite(value) or abs(value) > tolerance_time_s for value in delays)),
            "time_consistency": int(sum(value > tolerance_time_s for value in time_errors)),
            "position_consistency": int(sum(value > tolerance_position_m for value in position_errors)),
            "normal_consistency": int(sum(value > tolerance_normal for value in normal_errors)),
            "missing_collision_armor": int(sum(value == "missing" for value in armor_ids)),
        },
    }


def _source_status(cases: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    validation_status = Counter()
    source_ids = Counter()
    mechanisms = Counter()
    targets = Counter()
    for case in cases:
        munition = case.get("munition", {})
        validation_status[str(munition.get("validation_status", "missing"))] += 1
        source_ids[str(munition.get("reference_source_id", "missing"))] += 1
        mechanisms.update(map(str, case.get("analysis", {}).get("mechanisms", [])))
        targets[str(case.get("target_ref", {}).get("target_id", "missing"))] += 1
    return {
        "munition_validation_status_counts": dict(sorted(validation_status.items())),
        "munition_reference_source_counts": dict(sorted(source_ids.items())),
        "mechanism_case_counts": dict(sorted(mechanisms.items())),
        "target_id_counts": dict(sorted(targets.items())),
        "manifest_independent_scientific_validation": bool(
            manifest.get("capabilities", {}).get("independent_scientific_validation", False)
        ),
        "purpose_admission_counts": manifest.get("quality_summary", {}).get("purpose_admission_counts", {}),
        "known_limitations": manifest.get("known_limitations", []),
        "prohibited_uses": manifest.get("prohibited_uses", []),
    }


def _issue(
    issues: list[dict[str, str]], issue_id: str, severity: str, finding: str, consequence: str, recommendation: str
) -> None:
    issues.append(
        {
            "id": issue_id,
            "severity": severity,
            "finding": finding,
            "consequence": consequence,
            "recommendation": recommendation,
        }
    )


def _derive_issues(report: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    counts = report["sample_counts"]
    if counts["formal_case_count"] != 5000:
        _issue(
            issues,
            "sample-count-not-5000",
            "info",
            f"当前审计对象包含 {counts['formal_case_count']} 条正式样本，不是目标规模 5000。",
            "小样本烟雾数据只能验证审计工具容错，不能用于判断 5000 条数据的稀有事件覆盖。",
            "在最终 5000 条数据归档完成后重新运行同一脚本，并以新报告作为验收依据。",
        )
    if counts["formal_case_count"] != counts["tensor_valid_case_count"] + counts["refused_case_count"]:
        _issue(
            issues,
            "case-accounting",
            "high",
            "正式样本数与成功张量样本数加拒绝数不一致。",
            "样本可能丢失、重复，或清单统计口径不一致。",
            "核对 sampling checkpoint、根 manifest 与各分片 input_case_status，修复后再训练。",
        )
    budget = report["budget_conformance"]
    if budget.get("available") and not budget.get("exact_match"):
        _issue(
            issues,
            "split-budget-mismatch",
            "high",
            f"成功样本划分与冻结预算不符：{budget['delta_observed_minus_expected']}。",
            "训练/验证/测试统计口径偏离预注册设计，且可能存在拒绝样本导致的选择偏差。",
            "采用 reject-and-resample 补足冻结预算，重新生成后要求 3000/500/1500 或 sampling spec 中声明的预算精确匹配。",
        )
    if not report["identity_binding"]["all_bindings_match"]:
        _issue(
            issues,
            "target-tree-label-binding",
            "critical",
            "target、damage-tree、label spec 与正式 case 的内容哈希未全部互相绑定。",
            "标签语义可能来自不同版本的目标或毁伤树，数据无法追溯。",
            "停止训练；重建冻结计划，并要求 checkpoint、manifest、label spec 与每条 formal case 的哈希逐项相等。",
        )
    unexpected_profiles = sorted(set(counts["profile_counts"]) - set(PROFILE_ORDER))
    if unexpected_profiles:
        _issue(
            issues,
            "unexpected-munition-profile",
            "critical",
            "出现未允许弹型：" + ", ".join(unexpected_profiles),
            "数据超出只保留 small_front、medium_front、large_front 的任务定义。",
            "在采样冻结和张量导出两处 fail closed，仅接受三个版本化弹型。",
        )
    leakage = report["duplicates_and_leakage"]
    if leakage["cross_split_violation_count"]:
        _issue(
            issues,
            "cross-split-leakage",
            "critical",
            f"发现 {leakage['cross_split_violation_count']} 个跨划分重复/组泄漏证据（最多列出 20 个示例/字段）。",
            "验证与测试指标会被污染并产生过度乐观估计。",
            "按 pre-outcome lineage/group 重新冻结划分，确保相同输入、case hash、leakage key 和 epistemic group 只属于一个划分。",
        )
    if leakage["designed_input_signature"]["duplicate_extra_rows"]:
        _issue(
            issues,
            "duplicate-designed-inputs",
            "medium",
            f"设计输入存在 {leakage['designed_input_signature']['duplicate_extra_rows']} 条额外重复记录。",
            "重复样本降低有效样本量，并可能对特定工况过度加权。",
            "在冻结计划前对 profile、起点和速度的规范化签名去重；若重复是独立 MC 重复，应改为同一 case 的 replicate 轴。",
        )
    learner = report["learner_input_contract"]
    if not learner.get("available"):
        _issue(
            issues,
            "explicit-learner-input-missing",
            "high",
            "归档中没有 loitering_impact_x 显式学习器输入张量。",
            "训练代码可能回退到包含固定场景、求解器或后验碰撞量的通用 teacher 特征，造成冗余或信息泄漏。",
            "使用 loitering_impact_state_only@1.0.0，仅暴露三弹型 one-hot、起点 y/z 和入射速度，并让默认 loader 拒绝隐式全特征回退。",
        )
    else:
        contract_failures: list[str] = []
        if learner.get("shape", [0, 0])[1] != 6:
            contract_failures.append("特征维度不是 6")
        if not learner.get("all_mask_true"):
            contract_failures.append("存在无效 feature mask")
        if not learner.get("all_values_finite"):
            contract_failures.append("存在非有限数值")
        if not learner.get("profile_one_hot_valid"):
            contract_failures.append("弹型 one-hot 非法")
        reconstruction_error = learner.get("formal_case_reconstruction_max_abs_error")
        if reconstruction_error is None or reconstruction_error > 1e-5:
            contract_failures.append("与冻结 formal case 重建不一致")
        if learner.get("cross_split_duplicate_groups", 0):
            contract_failures.append("存在跨划分重复输入")
        if contract_failures:
            _issue(
                issues,
                "learner-input-contract-invalid",
                "critical",
                "；".join(contract_failures) + "。",
                "神经网络实际输入与预注册设计变量不一致，可能导致训练失败、泄漏或不可复现。",
                "修复张量导出和默认 loader，逐行与 formal ledger 对账，并把契约检查设为训练前 fail-closed gate。",
            )
        if learner.get("constant_train_features"):
            _issue(
                issues,
                "constant-train-inputs",
                "critical",
                "Train 中常量输入：" + ", ".join(learner["constant_train_features"]),
                "对应变量不可学习，且可能表明弹型或连续设计未正确进入张量。",
                "修正 sampling design 或特征提取，使所有声明为 learnable_variable 的输入在 Train 中有变化。",
            )
    collision_violations = report["collision_consistency"]["violations"]
    collision_total = sum(collision_violations.values())
    if report["collision_consistency"]["incomplete_cases"] or collision_total:
        _issue(
            issues,
            "collision-detonation-consistency",
            "critical",
            f"碰撞/起爆一致性检查发现 {collision_total} 项违规，另有 {report['collision_consistency']['incomplete_cases']} 条信息不完整。",
            "起爆位置、起爆时刻或接触法向错误会使输入语义和毁伤标签错位。",
            "要求所有样本满足零延时 first_armor_contact，且 time=distance/speed、detonation=origin+velocity*time 后再归档。",
        )
    labels = report["labels"]
    availability = labels.get("system_mask_overall_fraction")
    if availability is None or availability < 1.0:
        _issue(
            issues,
            "system-label-unavailable",
            "critical",
            f"系统输出标签总体可用率为 {availability!r}，未达到 1.0。",
            "默认系统级神经网络任务无法形成完整监督信号；被 mask 的数值不能当作零标签。",
            "使 damage-tree 叶节点机制与实际仿真机制一致，或明确改为 component fragment 辅助任务；生成后验证每个任务的 mask。",
        )
    source_availability = labels.get("system_source_available_mask_overall_fraction")
    if source_availability is None or source_availability < 1.0 or not labels.get("all_declared_tasks_fully_available"):
        _issue(
            issues,
            "source-or-task-mask-incomplete",
            "critical",
            f"system source mask 可用率={source_availability!r}，全部声明任务完全可用={labels.get('all_declared_tasks_fully_available')}。",
            "即使 system_y 存在，也不能把来源不可用或任务不可用的数值作为监督标签。",
            "要求 system_source_available_mask、system_label_mask 与 label_task_available_mask 对目标训练任务 100% 为真并互相一致。",
        )
    rare_outputs: list[str] = []
    constant_outputs: list[str] = []
    for output in labels.get("system_outputs", []):
        train = output.get("by_split", {}).get("train", {})
        stats = train.get("values", {})
        n = int(train.get("available", 0))
        nonzero = int(train.get("nonzero_case_count", 0))
        if n and min(nonzero, n - nonzero) < 100:
            rare_outputs.append(output["name"])
        if n and int(stats.get("unique", 0)) <= 1:
            constant_outputs.append(output["name"])
    if rare_outputs:
        _issue(
            issues,
            "rare-system-labels",
            "high",
            "训练划分中至少一侧少于 100 条的系统标签：" + ", ".join(rare_outputs),
            "分类损失和评估指标高度不稳定，模型可能学成恒定预测器。",
            "先做分层稀有事件扩充；设计权重与 minibatch/task 权重分离，并在冻结验证/测试集报告 PR-AUC、Brier、校准曲线及置信区间。",
        )
    if constant_outputs:
        _issue(
            issues,
            "constant-system-labels",
            "critical",
            "训练划分中的常量系统标签：" + ", ".join(constant_outputs),
            "对应输出不可辨识，神经网络无法学习输入到标签的关系。",
            "扩大能触发状态变化的物理设计域，或从当前任务移除这些输出；不得用关闭 mask 或改标签数值绕过检查。",
        )
    mc = report["monte_carlo_uncertainty"]
    replicate = mc.get("replicate_counts", {})
    if replicate.get("count", 0) and float(replicate.get("min", 0.0)) < 30.0:
        _issue(
            issues,
            "low-monte-carlo-replicates",
            "high",
            f"每工况最少仅 {replicate.get('min')} 个 MC 重复，概率分辨率和方差估计不足。",
            "软标签呈粗量化，CI/标准差在 R=2 时尤其不稳，稀有事件可能被误判为零。",
            "先做独立收敛研究；按最难收敛输出确定 R，建议将 R=2 仅作为流水线烟雾配置，并保留预先声明且不窥视验证/测试结果的停止规则。",
        )
    weight = report["design_weights"]
    ess_fraction = weight.get("all", {}).get("ess_fraction")
    if (
        weight.get("available")
        and isinstance(ess_fraction, (int, float))
        and ess_fraction < 0.5
    ):
        _issue(
            issues,
            "low-weight-ess",
            "high",
            f"设计权重 ESS 比例仅 {weight['all']['ess_fraction']:.3f}。",
            "少量高权重样本主导总体估计，方差增大。",
            "调整 proposal/strata 提高 positivity，保留 raw design weight，并另设训练采样权重。",
        )
    source = report["scientific_evidence_boundary"]
    unverified = source.get("munition_validation_status_counts", {}).get("unverified", 0)
    admitted = source.get("purpose_admission_counts", {})
    independent = source.get("manifest_independent_scientific_validation", False)
    if unverified or not independent or not any(admitted.get(purpose, 0) for purpose in ("research_training", "research_validation", "research_test")):
        _issue(
            issues,
            "research-admission-not-established",
            "critical",
            "弹体/目标/物理响应仍含未验证来源，且归档未建立独立科研验证或 research_* 准入。",
            "当前数据最多证明软件链路可执行，不能支持真实毁伤精度、外推能力或论文中的物理有效性结论。",
            "使用可追溯实测/权威数据校准弹体与目标参数；预注册外部验证方案，保留冻结 holdout，并报告模型差异、不确定度和适用域。",
        )
    for split in ("validation", "test"):
        if report["sample_counts"]["split_counts"].get(split, 0) < 100:
            _issue(
                issues,
                f"small-{split}-split",
                "medium",
                f"{split} 划分少于 100 条。",
                "总体与稀有事件指标的置信区间会很宽。",
                "最终 5000 条设计应按预注册预算冻结足量 validation/test，并逐标签核对正负事件数。",
            )
    return issues


def build_report(dataset_root: Path) -> dict[str, Any]:
    sampling_root, archive_root = _resolve_roots(dataset_root)
    manifest = _load_json(archive_root / "manifest.json")
    arrays, metadata, shards = _load_tensors(archive_root)
    case_ids = arrays["case_ids"].astype(str)
    splits = arrays["split_names"].astype(str)

    ledger_path = _formal_ledger_path(sampling_root, archive_root)
    formal_cases = list(_iter_jsonl(ledger_path)) if ledger_path else []
    formal_by_id = {str(case.get("case_id")): case for case in formal_cases}
    duplicate_formal_ids = len(formal_cases) - len(formal_by_id)
    missing_case_ids = [case_id for case_id in case_ids if case_id not in formal_by_id]
    if missing_case_ids:
        raise ValueError(
            f"formal-case ledger is missing {len(missing_case_ids)} successful tensor case IDs; "
            f"examples={missing_case_ids[:5]}"
        )
    aligned_cases = [formal_by_id[case_id] for case_id in case_ids]
    aligned_rows = [_case_inputs(case) for case in aligned_cases]
    profiles = np.asarray([row["profile"] for row in aligned_rows], dtype=str)

    quality = manifest.get("quality_summary", {})
    formal_count = len(formal_cases) if formal_cases else int(quality.get("n_total", case_ids.size))
    refusal_count = int(quality.get("n_refused", max(0, formal_count - case_ids.size)))
    spec = _sampling_spec(sampling_root)
    report: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "requested_root": str(dataset_root.resolve()),
            "sampling_root": str(sampling_root),
            "archive_root": str(archive_root),
            "manifest": str((archive_root / "manifest.json").resolve()),
            "formal_case_ledger": str(ledger_path.resolve()) if ledger_path else None,
            "shard_count": len(shards),
            "dataset_id": manifest.get("dataset_id"),
            "schema_version": manifest.get("schema_version"),
        },
        "sample_counts": {
            "formal_case_count": int(formal_count),
            "tensor_valid_case_count": int(case_ids.size),
            "execution_success_count": int(quality.get("execution_success_count", case_ids.size)),
            "refused_case_count": refusal_count,
            "duplicate_formal_case_ids": int(duplicate_formal_ids),
            "split_counts": _counts(splits.tolist(), SPLIT_ORDER),
            "profile_counts": _counts(profiles.tolist(), PROFILE_ORDER),
        },
        "budget_conformance": _budget_audit(spec, splits),
        "identity_binding": _hash_binding_audit(sampling_root, manifest, formal_cases),
        "input_distribution": _input_distribution(aligned_rows, profiles, splits, spec),
        "learner_input_contract": _learner_input_audit(
            arrays, metadata, aligned_rows, splits
        ),
        "duplicates_and_leakage": _duplicate_and_leakage(arrays, aligned_rows, splits),
        "labels": _label_audit(arrays, metadata, splits, profiles),
        "monte_carlo_uncertainty": _monte_carlo_audit(arrays, metadata),
        "design_weights": _weight_audit(arrays, splits),
        "collision_consistency": _collision_audit(formal_cases),
        "scientific_evidence_boundary": _source_status(formal_cases, manifest),
    }
    report["issues"] = _derive_issues(report)
    critical_or_high = [issue for issue in report["issues"] if issue["severity"] in {"critical", "high"}]
    labels_available = report["labels"].get("system_mask_overall_fraction") == 1.0
    learner_available = report["learner_input_contract"].get("available", False)
    report["decision"] = {
        "archive_audit_completed": True,
        "default_system_task_training_ready": bool(
            learner_available and labels_available and not critical_or_high
        ),
        "research_admission": False,
        "research_admission_reason": (
            "This audit is descriptive and the archive does not contain independent empirical validation "
            "of target response, munition parameters, damage physics, or out-of-domain generalization."
        ),
        "issue_counts": _counts((issue["severity"] for issue in report["issues"]), ("critical", "high", "medium", "low", "info")),
    }
    return report


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _markdown(report: dict[str, Any]) -> str:
    counts = report["sample_counts"]
    decision = report["decision"]
    lines = [
        "# 巡飞弹撞击装甲车数据集科学性与神经网络训练审计",
        "",
        f"- 数据集：`{report['scope']['dataset_id']}`",
        f"- 审计版本：`{report['audit_version']}`",
        f"- 正式样本 / 成功张量 / 拒绝：{counts['formal_case_count']} / {counts['tensor_valid_case_count']} / {counts['refused_case_count']}",
        f"- 默认系统任务训练就绪：**{'是' if decision['default_system_task_training_ready'] else '否'}**",
        "- 科研准入：**否**。本报告是描述性软件与数据审计，不是实物毁伤或物理模型验证。",
        "",
        "## 1. 样本组成",
        "",
        "| 维度 | 计数 |",
        "|---|---:|",
    ]
    for split in SPLIT_ORDER:
        lines.append(f"| 划分 `{split}` | {counts['split_counts'].get(split, 0)} |")
    for profile in PROFILE_ORDER:
        lines.append(f"| 弹型 `{profile}` | {counts['profile_counts'].get(profile, 0)} |")

    lines.extend(["", "## 2. 设计输入分布", ""])
    lines.extend(
        [
            "| 输入 | 最小 | P05 | 中位数 | P95 | 最大 | 均值±标准差 | 唯一值 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    display_keys = (
        "origin_y_m",
        "origin_z_m",
        "impact_speed_mps",
        "collision_distance_m",
        "detonation_time_s",
    )
    continuous = report["input_distribution"]["continuous"]
    for key in display_keys:
        stats = continuous[key]["all"]
        lines.append(
            f"| `{key}` | {_fmt(stats.get('min'))} | {_fmt(stats.get('p05'))} | {_fmt(stats.get('median'))} | "
            f"{_fmt(stats.get('p95'))} | {_fmt(stats.get('max'))} | {_fmt(stats.get('mean'))} ± {_fmt(stats.get('std'))} | {stats.get('unique', 0)} |"
        )
    correlation = report["input_distribution"]["continuous_design_correlation"]
    lines.extend(
        [
            "",
            f"三个连续设计变量的最大绝对 Pearson 相关系数为 {_fmt(correlation['maximum_absolute_off_diagonal'])}。"
            "这是采样设计诊断，不是统计独立性的证明。",
        ]
    )
    coverage = report["input_distribution"]["declared_range_coverage"]
    if coverage:
        lines.extend(["", "| 声明均匀变量 | 区间 | 占用箱/总箱 | 越界 | 最小/中位/最大箱计数 |", "|---|---|---:|---:|---:|"])
        for name, block in coverage.items():
            lines.append(
                f"| `{name}` | {block['declared_range']} | {block['occupied_bins']}/{block['bins']} | {block['out_of_range']} | "
                f"{block['min_bin_count']}/{_fmt(block['median_bin_count'])}/{block['max_bin_count']} |"
            )

    learner = report["learner_input_contract"]
    lines.extend(["", "### 显式学习器输入契约", ""])
    if not learner.get("available"):
        lines.append("`loitering_impact_x` 不存在；当前归档不能证明默认训练入口只接收预注册的 6 个学习变量。")
    else:
        lines.extend(
            [
                f"- 张量形状：`{learner['shape']}`；特征：`{learner['feature_names']}`。",
                f"- mask 全真 / 数值全有限 / one-hot 合法：{learner['all_mask_true']} / {learner['all_values_finite']} / {learner['profile_one_hot_valid']}。",
                f"- 与冻结 formal case 重建的最大绝对误差：{_fmt(learner['formal_case_reconstruction_max_abs_error'])}。",
                f"- Train 常量特征：`{learner['constant_train_features']}`；跨划分重复输入组：{learner['cross_split_duplicate_groups']}。",
                f"- 连续/one-hot 非零 Train 标准差尺度比：{_fmt(learner['nonzero_train_std_scale_ratio'])}。",
                f"- 归一化要求：{learner['normalization_guidance']}",
            ]
        )

    duplicate = report["duplicates_and_leakage"]
    lines.extend(
        [
            "",
            "## 3. 重复与划分泄漏",
            "",
            f"- 设计输入额外重复行：{duplicate['designed_input_signature']['duplicate_extra_rows']}。",
            f"- case hash 额外重复行：{duplicate['case_hash']['duplicate_extra_rows']}。",
            f"- 跨划分违规证据计数：{duplicate['cross_split_violation_count']}。",
        ]
    )

    lines.extend(["", "## 4. 标签可用性、常量与稀有性", ""])
    lines.extend(
        [
            f"系统标签 mask 总体可用率：**{_fmt(report['labels'].get('system_mask_overall_fraction'))}**。",
            "",
            "| 系统输出 | 可用数 | 均值 | 标准差 | 唯一值 | 零值率 | 一值率 | Train 非零/可用 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for output in report["labels"].get("system_outputs", []):
        stats = output["values"]
        train = output["by_split"]["train"]
        lines.append(
            f"| `{output['name']}` | {output['availability']['successes']} | {_fmt(stats.get('mean'))} | {_fmt(stats.get('std'))} | "
            f"{stats.get('unique', 0)} | {_fmt(stats.get('exact_zero_fraction'))} | {_fmt(stats.get('exact_one_fraction'))} | "
            f"{train['nonzero_case_count']}/{train['available']} |"
        )
    if report["labels"].get("component_events"):
        lines.extend(["", "组件级碎片事件（每 case 至少一个组件发生事件）：", ""])
        for name, block in report["labels"]["component_events"].items():
            event = block["case_event"]["all"]
            lines.append(f"- `{name}`：{event['successes']}/{event['total']}，率 {_fmt(event['rate'])}，Wilson 95% CI {event['ci95']}。")

    mc = report["monte_carlo_uncertainty"]
    lines.extend(
        [
            "",
            "## 5. Monte Carlo 与设计权重",
            "",
            f"- replicate 数：min={_fmt(mc['replicate_counts'].get('min'))}，median={_fmt(mc['replicate_counts'].get('median'))}，max={_fmt(mc['replicate_counts'].get('max'))}。",
        ]
    )
    if "system_ci95_width" in mc:
        width = mc["system_ci95_width"]["all_available"]
        lines.append(f"- 系统标签 CI95 全宽：median={_fmt(width.get('median'))}，P95={_fmt(width.get('p95'))}，max={_fmt(width.get('max'))}。")
    weight = report["design_weights"]
    if weight.get("available"):
        lines.append(
            f"- 有效设计权重：n={weight['all']['count']}，ESS={_fmt(weight['all']['ess'])}，ESS 比例={_fmt(weight['all']['ess_fraction'])}。"
        )
    lines.append(f"- 注意：{mc['interpretation']}。")

    collision = report["collision_consistency"]
    lines.extend(
        [
            "",
            "## 6. 碰撞即起爆一致性",
            "",
            f"- 检查 {collision['checked_cases']} 条，信息不完整 {collision['incomplete_cases']} 条。",
            f"- 时间误差最大值：{_fmt(collision['time_equals_distance_over_speed_absolute_error_s'].get('max'))} s。",
            f"- 位置误差最大值：{_fmt(collision['trajectory_position_absolute_error_m'].get('max'))} m。",
            f"- 法向单位长度误差最大值：{_fmt(collision['collision_normal_unit_error'].get('max'))}。",
            f"- 违规计数：`{json.dumps(collision['violations'], ensure_ascii=False)}`。",
        ]
    )

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    issues = sorted(report["issues"], key=lambda row: (severity_order[row["severity"]], row["id"]))
    lines.extend(["", "## 7. 问题与专业修改建议", ""])
    for index, issue in enumerate(issues, 1):
        lines.extend(
            [
                f"### {index}. [{issue['severity'].upper()}] {issue['id']}",
                "",
                f"- 发现：{issue['finding']}",
                f"- 后果：{issue['consequence']}",
                f"- 修改建议：{issue['recommendation']}",
                "",
            ]
        )

    boundary = report["scientific_evidence_boundary"]
    lines.extend(
        [
            "## 8. 科研证据边界",
            "",
            "本审计可证明归档可读取、输入分布/划分/标签 mask/MC 数量及碰撞几何的一致性。它不能证明真实装甲毁伤精度、弹体参数真实性、响应模型校准、跨目标泛化或论文层面的科研有效性。",
            "",
            f"- 弹体验证状态：`{json.dumps(boundary['munition_validation_status_counts'], ensure_ascii=False)}`",
            f"- research 准入计数：`{json.dumps(boundary['purpose_admission_counts'], ensure_ascii=False)}`",
            f"- 独立科学验证：`{boundary['manifest_independent_scientific_validation']}`",
            "",
            "因此即使所有软件检查通过，`research_admission` 仍保持 `false`，直到完成独立校准与冻结外部验证。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit a loitering-munition/armored-target dataset for distribution, ML readiness and scientific boundaries."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)

    sampling_root, _ = _resolve_roots(args.dataset_root)
    output_json = args.output_json or sampling_root / "audits" / "loitering_impact_scientific_audit.json"
    output_md = args.output_md or sampling_root / "audits" / "loitering_impact_scientific_audit.md"
    report = build_report(args.dataset_root)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    output_md.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "dataset_id": report["scope"]["dataset_id"],
                "output_json": str(output_json.resolve()),
                "output_md": str(output_md.resolve()),
                "decision": report["decision"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
