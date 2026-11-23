#!/usr/bin/env python
"""Audit a strict dataset-v2 development package without opening test labels.

The audit reports train/validation coverage, hurdle-target support, design-weight
ESS, epsilon-completeness and a descriptive missingness diagnostic.  Test result
rows are classified from the frozen tensor split before ``json.loads``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gzip
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.dataset_v2 import validate_dataset_v2  # noqa: E402
from damage_engine.errors import InputValidationError  # noqa: E402
from damage_engine.utils import content_hash, file_hash  # noqa: E402


_TARGETS = ("p_hit", "p_degraded", "p_failed")
_CONDITIONAL_TARGETS = (
    "conditional_log1p_energy",
    "conditional_log1p_hit_count",
)
_KS_AXES = (
    "direction_coordinate_1",
    "direction_coordinate_2",
    "aim_u_m",
    "aim_v_m",
    "speed_mps",
    "body_x_roll_rad",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_dataset_directory(pilot: Path) -> Path:
    candidates = (pilot / "formal_dataset", pilot)
    for candidate in candidates:
        if (candidate / "manifest.json").is_file() and (
            candidate / "shards"
        ).is_dir():
            return candidate
    raise InputValidationError("pilot does not contain a completed formal dataset")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise InputValidationError(
            "dataset-v2 archive is missing or unreadable",
            context={"path": str(path)},
        ) from exc


def _scan_development_results(
    pilot: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    """Decode development rows only; test rows remain opaque byte strings."""

    source = _source_dataset_directory(pilot)
    results: dict[str, dict[str, Any]] = {}
    access = {
        "result_json_lines_seen": 0,
        "development_result_json_lines_decoded": 0,
        "test_result_json_lines_seen": 0,
        "test_result_json_lines_decoded": 0,
    }
    evidence: list[dict[str, Any]] = []
    shards = sorted((source / "shards").glob("part-*"))
    if not shards:
        raise InputValidationError("formal dataset contains no shards")
    for shard in shards:
        tensor_path = shard / "tensors.npz"
        result_path = shard / "results.jsonl.gz"
        try:
            with np.load(tensor_path, allow_pickle=False) as archive:
                case_ids = archive["case_ids"].astype(str)
                splits = archive["split_names"].astype(str)
                frozen = archive["input_frozen_splits"].astype(str)
        except (OSError, ValueError, KeyError) as exc:
            raise InputValidationError(
                "sampler shard lacks frozen case/split identity",
                context={"shard": shard.name},
            ) from exc
        if (
            case_ids.shape != splits.shape
            or frozen.tolist() != splits.tolist()
            or not set(splits.tolist()).issubset({"train", "validation", "test"})
        ):
            raise InputValidationError("sampler shard split identity is invalid")
        line_count = 0
        try:
            with gzip.open(result_path, "rt", encoding="utf-8") as stream:
                for row_index, line in enumerate(stream):
                    line_count += 1
                    if row_index >= case_ids.size:
                        raise InputValidationError(
                            "result ledger has more rows than the tensor shard"
                        )
                    access["result_json_lines_seen"] += 1
                    split = str(splits[row_index])
                    case_id = str(case_ids[row_index])
                    if split == "test":
                        access["test_result_json_lines_seen"] += 1
                        # The label payload is intentionally not parsed here.
                        continue
                    row = json.loads(line)
                    access["development_result_json_lines_decoded"] += 1
                    if str(row.get("case_id", "")) != case_id:
                        raise InputValidationError(
                            "development result case identity differs from tensors",
                            context={"expected": case_id, "actual": row.get("case_id")},
                        )
                    if case_id in results:
                        raise InputValidationError("duplicate development result case_id")
                    results[case_id] = row
        except (OSError, json.JSONDecodeError) as exc:
            raise InputValidationError(
                "development result ledger is missing or malformed",
                context={"shard": shard.name},
            ) from exc
        if line_count != case_ids.size:
            raise InputValidationError(
                "result ledger row count differs from the tensor shard",
                context={
                    "shard": shard.name,
                    "result_rows": line_count,
                    "tensor_rows": int(case_ids.size),
                },
            )
        evidence.append(
            {
                "shard": shard.name,
                "case_count": int(case_ids.size),
                "test_case_count": int(np.sum(splits == "test")),
                "tensors_sha256": file_hash(tensor_path),
                "results_sha256": file_hash(result_path),
            }
        )
    return results, access, evidence


def _case_documents(pilot: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    paths = sorted((pilot / "rounds").glob("round-*/formal_cases.jsonl.gz"))
    if not paths:
        raise InputValidationError("pilot lacks the frozen formal-case ledger")
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                case = json.loads(line)
                case_id = str(case.get("case_id", ""))
                if case_id in wanted:
                    if case_id in output:
                        raise InputValidationError("duplicate formal case identity")
                    output[case_id] = case
    missing = wanted - set(output)
    if missing:
        raise InputValidationError(
            "formal-case ledger is missing development cases",
            context={"missing_count": len(missing), "sample": sorted(missing)[:5]},
        )
    return output


def _metadata(case: Mapping[str, Any]) -> dict[str, Any]:
    metadata = case.get("metadata", {})
    sampling = metadata.get("continuous_direction_sampling")
    if not isinstance(sampling, Mapping):
        sampling = metadata.get("contact_manifold_sampling", {})
    if not isinstance(sampling, Mapping):
        sampling = {}
    names = list(map(str, sampling.get("latent_names", [])))
    latent = sampling.get("physical_latent", sampling.get("latent", []))
    latent_map = {
        name: float(value)
        for name, value in zip(names, latent, strict=False)
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    profile = str(case.get("munition", {}).get("profile_id", "unclassified"))
    region = str(
        sampling.get("region_id", sampling.get("approach_id", "unclassified"))
    )
    values = {
        "direction_coordinate_1": latent_map.get("direction_coordinate_1"),
        "direction_coordinate_2": latent_map.get("direction_coordinate_2"),
        "aim_u_m": sampling.get("aim_u_m"),
        "aim_v_m": sampling.get("aim_v_m"),
        "speed_mps": sampling.get("speed_mps"),
        "body_x_roll_rad": sampling.get("body_x_roll_rad"),
    }
    for name, value in list(values.items()):
        try:
            number = float(value)
        except (TypeError, ValueError):
            values[name] = None
        else:
            values[name] = number if math.isfinite(number) else None
    return {
        "region_id": region,
        "profile_id": profile,
        "cell_id": f"{region}::{profile}",
        **values,
    }


def _ess(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0.0):
        return 0.0
    denominator = float(np.sum(values * values))
    return float(np.sum(values) ** 2 / denominator) if denominator > 0.0 else 0.0


def _quantiles(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    q = np.quantile(array, [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        "count": int(array.size),
        "minimum": float(q[0]),
        "median": float(q[1]),
        "p90": float(q[2]),
        "p95": float(q[3]),
        "p99": float(q[4]),
        "maximum": float(q[5]),
        "mean": float(np.mean(array)),
    }


def _count_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.int64)
    return {
        "minimum": int(np.min(array)) if array.size else 0,
        "median": float(np.median(array)) if array.size else None,
        "maximum": int(np.max(array)) if array.size else 0,
        "zero_supported_component_count": int(np.sum(array == 0)),
        "below_20_component_count": int(np.sum(array < 20)),
    }


def _holm_adjust(records: list[dict[str, Any]], *, alpha: float) -> None:
    evaluable = [
        (index, float(record["p_value"]))
        for index, record in enumerate(records)
        if record.get("status") == "evaluated" and record.get("p_value") is not None
    ]
    order = sorted(evaluable, key=lambda item: item[1])
    count = len(order)
    running = 0.0
    for rank, (index, p_value) in enumerate(order):
        adjusted = min(1.0, (count - rank) * p_value)
        running = max(running, adjusted)
        records[index]["p_value_holm"] = running
        records[index]["significant_holm_0_05"] = bool(running < alpha)


def _ks_diagnostics(
    metadata_rows: Sequence[Mapping[str, Any]],
    split: np.ndarray,
    epsilon_complete: np.ndarray,
    *,
    alpha: float,
) -> dict[str, Any]:
    regions = np.asarray([str(row["region_id"]) for row in metadata_rows])
    profiles = np.asarray([str(row["profile_id"]) for row in metadata_rows])
    groups: list[tuple[str, np.ndarray]] = []
    for split_name in ("train", "validation"):
        split_mask = split == split_name
        groups.append((f"split::{split_name}", split_mask))
        cells = sorted(set(zip(regions[split_mask], profiles[split_mask], strict=False)))
        for region, profile in cells:
            groups.append(
                (
                    f"split_cell::{split_name}::{region}::{profile}",
                    split_mask & (regions == region) & (profiles == profile),
                )
            )
    for region, profile in sorted(set(zip(regions, profiles, strict=False))):
        groups.append(
            (
                f"cell::development::{region}::{profile}",
                (regions == region) & (profiles == profile),
            )
        )

    records: list[dict[str, Any]] = []
    for group_id, group_mask in groups:
        group_count = int(np.sum(group_mask))
        complete_mask = group_mask & epsilon_complete
        complete_count = int(np.sum(complete_mask))
        for axis in _KS_AXES:
            values = np.asarray(
                [
                    float(row[axis]) if row.get(axis) is not None else np.nan
                    for row in metadata_rows
                ],
                dtype=np.float64,
            )
            all_values = values[group_mask & np.isfinite(values)]
            complete_values = values[complete_mask & np.isfinite(values)]
            record: dict[str, Any] = {
                "group_id": group_id,
                "axis": axis,
                "group_case_count": group_count,
                "epsilon_complete_case_count": complete_count,
                "all_finite_count": int(all_values.size),
                "epsilon_complete_finite_count": int(complete_values.size),
                "comparison": "epsilon_complete_subset_vs_all_development_cases_in_group",
            }
            if all_values.size < 4 or complete_values.size < 2:
                record.update(
                    {
                        "status": "insufficient_sample",
                        "ks_statistic": None,
                        "p_value": None,
                    }
                )
            elif complete_values.size == all_values.size:
                record.update(
                    {
                        "status": "no_missingness_contrast",
                        "ks_statistic": 0.0,
                        "p_value": 1.0,
                    }
                )
            else:
                statistic, p_value = ks_2samp(
                    complete_values,
                    all_values,
                    alternative="two-sided",
                    method="auto",
                )
                record.update(
                    {
                        "status": "evaluated",
                        "ks_statistic": float(statistic),
                        "p_value": float(p_value),
                    }
                )
            records.append(record)
    _holm_adjust(records, alpha=alpha)
    evaluated = [record for record in records if record["status"] == "evaluated"]
    significant = [
        record for record in evaluated if record.get("significant_holm_0_05")
    ]
    return {
        "alpha": float(alpha),
        "multiple_comparison": "Holm family-wise adjustment across all evaluated group-axis tests",
        "subset_dependence_note": (
            "The epsilon-complete subset is contained in the all-case sample; KS p-values "
            "are descriptive missingness diagnostics, while the KS statistic is the primary effect size."
        ),
        "tested_record_count": len(records),
        "evaluated_record_count": len(evaluated),
        "significant_record_count": len(significant),
        "maximum_ks_statistic": (
            float(max(record["ks_statistic"] for record in evaluated))
            if evaluated
            else None
        ),
        "records": records,
    }


def _stratum_report(
    indices: np.ndarray,
    labels: Mapping[str, np.ndarray],
    weights: Mapping[str, np.ndarray],
    input_mask: np.ndarray,
    epsilon_complete: np.ndarray,
    interval_complete: Mapping[str, np.ndarray],
    affected_fraction: np.ndarray,
) -> dict[str, Any]:
    count = int(indices.size)
    design_weight = weights["design_weight"][indices].astype(np.float64)
    label_mask = labels["label_mask"][indices].astype(bool)
    target_masks = {
        target: labels.get(target + "_mask", labels["label_mask"])[indices].astype(bool)
        for target in _TARGETS
    }
    target_positive: dict[str, Any] = {}
    target_negative: dict[str, Any] = {}
    target_observed: dict[str, Any] = {}
    for target in _TARGETS:
        observed = np.sum(target_masks[target], axis=0)
        positive = np.sum(
            (labels[target][indices] > 0.0) & target_masks[target],
            axis=0,
        )
        negative = observed - positive
        target_observed[target] = _count_summary(observed)
        target_positive[target] = _count_summary(positive)
        target_negative[target] = _count_summary(negative)
    return {
        "case_count": count,
        "input_complete_case_rate": float(np.mean(np.all(input_mask[indices], axis=1))),
        "label_observation_rate": float(np.mean(label_mask)),
        "target_observation_rate": {
            target: float(np.mean(target_masks[target])) for target in _TARGETS
        },
        "fully_observed_case_rate": float(np.mean(np.all(label_mask, axis=1))),
        "epsilon_complete_case_rate": float(np.mean(epsilon_complete[indices])),
        "epsilon_complete_case_count": int(np.sum(epsilon_complete[indices])),
        "target_epsilon_complete_case_rate": {
            target: float(np.mean(interval_complete[target][indices]))
            for target in _TARGETS
        },
        "conditional_energy_observation_rate": float(
            np.mean(labels["conditional_log1p_energy_mask"][indices])
        ),
        "conditional_hit_count_observation_rate": float(
            np.mean(labels["conditional_log1p_hit_count_mask"][indices])
        ),
        "component_positive_support": target_positive,
        "component_negative_support": target_negative,
        "component_observed_support": target_observed,
        "design_weight": {
            "minimum": float(np.min(design_weight)),
            "maximum": float(np.max(design_weight)),
            "effective_sample_size": _ess(design_weight),
            "ess_fraction": float(_ess(design_weight) / count),
        },
        "affected_replicate_fraction": _quantiles(affected_fraction[indices]),
    }


def audit_development_dataset(
    dataset: str | Path,
    pilot: str | Path,
    *,
    epsilon: float = 0.01,
    ks_alpha: float = 0.05,
) -> dict[str, Any]:
    root = Path(dataset)
    pilot_root = Path(pilot)
    if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise InputValidationError("epsilon must lie in [0,1]")
    if not 0.0 < ks_alpha < 1.0:
        raise InputValidationError("KS alpha must lie in (0,1)")

    validation = validate_dataset_v2(root, recompute_views=False)
    seal = _json(root / "sealed_test.json")
    if (
        seal.get("test_labels_present_in_core") is not False
        or seal.get("test_result_json_lines_decoded") != 0
        or seal.get("test_open_count") != 0
    ):
        raise InputValidationError("dataset is not a strict unopened sealed-test package")

    manifest = _json(root / "manifest.json")
    inputs = _load_npz(root / "core" / "inputs.npz")
    labels = _load_npz(root / "core" / "component_labels.npz")
    weights = _load_npz(root / "core" / "sampling_weights.npz")
    case_ids = inputs["case_ids"].astype(str)
    split = weights["split"].astype(str)
    if case_ids.tolist() != weights["case_ids"].astype(str).tolist():
        raise InputValidationError("dataset core case identities are misaligned")
    test_mask = split == "test"
    target_mask_fields = (
        "label_mask",
        "p_hit_mask",
        "p_degraded_mask",
        "p_failed_mask",
        "conditional_log1p_energy_mask",
        "conditional_log1p_hit_count_mask",
    )
    exposed_test_masks = [
        field
        for field in target_mask_fields
        if field in labels and np.any(labels[field][test_mask])
    ]
    if not np.any(test_mask) or exposed_test_masks:
        raise InputValidationError("sealed test labels must exist only as masked placeholders")
    development_mask = np.isin(split, ["train", "validation"])
    development_indices = np.flatnonzero(development_mask)
    development_case_ids = case_ids[development_mask]
    if not development_indices.size:
        raise InputValidationError("dataset contains no development cases")

    decoded_results, access, shard_evidence = _scan_development_results(pilot_root)
    if access["test_result_json_lines_decoded"] != 0:
        raise InputValidationError("sealed test result payload was decoded")
    if set(decoded_results) != set(development_case_ids.tolist()):
        raise InputValidationError(
            "strict package and source development result identities differ",
            context={
                "package_count": int(development_case_ids.size),
                "decoded_count": len(decoded_results),
            },
        )

    case_documents = _case_documents(pilot_root, set(development_case_ids.tolist()))
    metadata_by_case = {
        case_id: _metadata(case_documents[case_id])
        for case_id in development_case_ids.tolist()
    }
    metadata_rows = [metadata_by_case[case_id] for case_id in development_case_ids]

    intervals: dict[str, np.ndarray] = {}
    interval_complete: dict[str, np.ndarray] = {}
    for target in _TARGETS:
        field = target + "_identification_interval"
        if field not in labels:
            raise InputValidationError(
                "component labels lack identification intervals",
                context={"field": field},
            )
        interval = labels[field][development_mask].astype(np.float64)
        width = interval[..., 1] - interval[..., 0]
        if not np.isfinite(width).all() or np.any(width < -1.0e-12):
            raise InputValidationError("identification intervals are invalid")
        intervals[target] = np.maximum(0.0, width)
        interval_complete[target] = np.all(
            intervals[target] <= epsilon + 1.0e-12,
            axis=1,
        )
        mask_field = target + "_mask"
        if mask_field in labels:
            expected_mask = intervals[target] <= epsilon + 1.0e-12
            observed_mask = labels[mask_field][development_mask].astype(bool)
            if not np.array_equal(observed_mask, expected_mask):
                raise InputValidationError(
                    "probability target mask disagrees with its identification interval",
                    context={"target": target, "epsilon": float(epsilon)},
                )
    epsilon_complete = np.logical_and.reduce(
        [interval_complete[target] for target in _TARGETS]
    )

    label_development = {
        name: np.asarray(value)[development_mask]
        if np.asarray(value).ndim >= 1 and np.asarray(value).shape[0] == case_ids.size
        else np.asarray(value)
        for name, value in labels.items()
    }
    weight_development = {
        name: np.asarray(value)[development_mask]
        if np.asarray(value).ndim >= 1 and np.asarray(value).shape[0] == case_ids.size
        else np.asarray(value)
        for name, value in weights.items()
    }
    input_mask = inputs.get("input_mask", np.ones_like(inputs["x"], dtype=bool))[
        development_mask
    ].astype(bool)
    affected_values = []
    for case_id in development_case_ids:
        value = (
            decoded_results[case_id]
            .get("penetration_domain_coverage", {})
            .get("affected_replicate_fraction", 0.0)
        )
        affected_values.append(float(value) if value is not None else 0.0)
    affected_fraction = np.asarray(affected_values, dtype=np.float64)
    split_development = split[development_mask]

    overall_indices = np.arange(development_indices.size, dtype=np.int64)
    split_reports = {
        name: _stratum_report(
            np.flatnonzero(split_development == name),
            label_development,
            weight_development,
            input_mask,
            epsilon_complete,
            interval_complete,
            affected_fraction,
        )
        for name in ("train", "validation")
    }
    overall_report = _stratum_report(
        overall_indices,
        label_development,
        weight_development,
        input_mask,
        epsilon_complete,
        interval_complete,
        affected_fraction,
    )

    cells = np.asarray([row["cell_id"] for row in metadata_rows])
    development_cell_reports: dict[str, Any] = {}
    for cell_id in sorted(set(cells.tolist())):
        indices = np.flatnonzero(cells == cell_id)
        development_cell_reports[cell_id] = _stratum_report(
            indices,
            label_development,
            weight_development,
            input_mask,
            epsilon_complete,
            interval_complete,
            affected_fraction,
        )
    cell_reports: dict[str, Any] = {}
    for split_name in ("train", "validation"):
        for cell_id in sorted(set(cells[split_development == split_name].tolist())):
            indices = np.flatnonzero(
                (split_development == split_name) & (cells == cell_id)
            )
            cell_reports[f"{split_name}::{cell_id}"] = _stratum_report(
                indices,
                label_development,
                weight_development,
                input_mask,
                epsilon_complete,
                interval_complete,
                affected_fraction,
            )

    feature_names = inputs["feature_names"].astype(str)
    x = inputs["x"][development_mask].astype(np.float64)
    input_coverage = {
        feature_names[column]: {
            "mask_rate": float(np.mean(input_mask[:, column])),
            **_quantiles(x[input_mask[:, column], column]),
            "unique_count": int(np.unique(x[input_mask[:, column], column]).size),
        }
        for column in range(feature_names.size)
    }

    component_ids = labels["component_ids"].astype(str)
    train_local = np.flatnonzero(split_development == "train")
    validation_local = np.flatnonzero(split_development == "validation")
    component_task_admission: dict[str, Any] = {}
    for target in _TARGETS:
        target_mask = label_development.get(
            target + "_mask", label_development["label_mask"]
        ).astype(bool)

        def split_support(local_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            observed = np.sum(target_mask[local_indices], axis=0)
            positive = np.sum(
                (label_development[target][local_indices] > 0.0)
                & target_mask[local_indices],
                axis=0,
            )
            return observed, positive, observed - positive

        train_observed, train_positive, train_negative = split_support(train_local)
        val_observed, val_positive, val_negative = split_support(validation_local)
        train_eligible = (train_positive >= 20) & (train_negative >= 20)
        validation_evaluable = (val_positive >= 5) & (val_negative >= 5)
        admitted = train_eligible & validation_evaluable
        rare = ~train_eligible
        component_task_admission[target] = {
            "support_kind": "binary_probability",
            "selection_scope": (
                "label-availability gate only; no validation performance or sealed-test "
                "outcome is used"
            ),
            "thresholds": {
                "train_positive_min": 20,
                "train_negative_min": 20,
                "validation_positive_min": 5,
                "validation_negative_min": 5,
            },
            "train_eligible_component_count": int(np.sum(train_eligible)),
            "validation_evaluable_component_count": int(
                np.sum(validation_evaluable)
            ),
            "admitted_component_count": int(np.sum(admitted)),
            "admitted_component_ids": component_ids[admitted].tolist(),
            "rare_or_underpowered_component_ids": component_ids[rare].tolist(),
            "train_positive_count": {
                component_id: int(value)
                for component_id, value in zip(component_ids, train_positive)
            },
            "train_negative_count": {
                component_id: int(value)
                for component_id, value in zip(component_ids, train_negative)
            },
            "validation_positive_count": {
                component_id: int(value)
                for component_id, value in zip(component_ids, val_positive)
            },
            "validation_negative_count": {
                component_id: int(value)
                for component_id, value in zip(component_ids, val_negative)
            },
        }

    for target in _CONDITIONAL_TARGETS:
        mask_field = target + "_mask"
        if mask_field not in label_development:
            raise InputValidationError(
                "component labels lack a conditional-target mask",
                context={"field": mask_field},
            )
        target_mask = label_development[mask_field].astype(bool)
        train_observed = np.sum(target_mask[train_local], axis=0)
        val_observed = np.sum(target_mask[validation_local], axis=0)
        train_eligible = train_observed >= 20
        validation_evaluable = val_observed >= 5
        admitted = train_eligible & validation_evaluable
        rejected_components = []
        for component_id, train_count, validation_count in zip(
            component_ids, train_observed, val_observed
        ):
            reasons = []
            if int(train_count) < 20:
                reasons.append("train_observed_below_minimum")
            if int(validation_count) < 5:
                reasons.append("validation_observed_below_minimum")
            if reasons:
                rejected_components.append(
                    {
                        "component_id": str(component_id),
                        "reasons": reasons,
                    }
                )
        component_task_admission[target] = {
            "support_kind": "observed_continuous",
            "selection_scope": (
                "label-availability gate only; no validation performance or sealed-test "
                "outcome is used"
            ),
            "thresholds": {
                "train_observed_min": 20,
                "validation_observed_min": 5,
            },
            "train_eligible_component_count": int(np.sum(train_eligible)),
            "validation_evaluable_component_count": int(
                np.sum(validation_evaluable)
            ),
            "admitted_component_count": int(np.sum(admitted)),
            "admitted_component_ids": component_ids[admitted].tolist(),
            "rare_or_underpowered_component_ids": component_ids[
                ~train_eligible
            ].tolist(),
            "rejected_components": rejected_components,
            "train_observed_count": {
                component_id: int(value)
                for component_id, value in zip(component_ids, train_observed)
            },
            "validation_observed_count": {
                component_id: int(value)
                for component_id, value in zip(component_ids, val_observed)
            },
        }

    train_report = split_reports["train"]
    validation_report = split_reports["validation"]
    readiness_gates = {
        "minimum_train_cases_2000": bool(train_report["case_count"] >= 2000),
        "train_epsilon_complete_rate_at_least_0_90": bool(
            train_report["epsilon_complete_case_rate"] >= 0.90
        ),
        "minimum_component_p_hit_positive_count_20": bool(
            train_report["component_positive_support"]["p_hit"]["minimum"] >= 20
        ),
        "minimum_component_p_degraded_positive_count_20": bool(
            train_report["component_positive_support"]["p_degraded"]["minimum"]
            >= 20
        ),
        "minimum_component_p_failed_positive_count_20": bool(
            train_report["component_positive_support"]["p_failed"]["minimum"]
            >= 20
        ),
        "minimum_component_p_hit_negative_count_20": bool(
            train_report["component_negative_support"]["p_hit"]["minimum"] >= 20
        ),
        "minimum_validation_component_p_hit_positive_count_5": bool(
            validation_report["component_positive_support"]["p_hit"]["minimum"]
            >= 5
        ),
        "minimum_validation_component_p_hit_negative_count_5": bool(
            validation_report["component_negative_support"]["p_hit"]["minimum"]
            >= 5
        ),
        "train_ess_fraction_at_least_0_50": bool(
            train_report["design_weight"]["ess_fraction"] >= 0.50
        ),
        "validation_has_cases": bool(split_reports["validation"]["case_count"] > 0),
        "sealed_test_result_json_lines_decoded_zero": bool(
            access["test_result_json_lines_decoded"] == 0
        ),
    }
    readiness = {
        "gates": readiness_gates,
        "formal_neural_training_ready": bool(all(readiness_gates.values())),
        "decision_scope": (
            "statistical/software training readiness for this synthetic teacher; "
            "physical validation and research admission are separate gates"
        ),
    }

    report: dict[str, Any] = {
        "schema_version": "dataset-v2-development-audit@1.0.0",
        "dataset_id": manifest.get("dataset_id"),
        "dataset_manifest_hash": manifest.get("manifest_hash"),
        "source_pilot": str(pilot_root),
        "epsilon_probability": float(epsilon),
        "scope": "train_and_validation_only_with_opaque_sealed_test_labels",
        "split_counts": {
            name: int(np.sum(split == name))
            for name in ("train", "validation", "test")
        },
        "sealed_test": {
            "test_case_count": int(np.sum(test_mask)),
            "test_labels_present_in_core": False,
            "test_open_count": 0,
            "test_result_json_lines_decoded": access[
                "test_result_json_lines_decoded"
            ],
            "test_metrics_computed": False,
        },
        "result_access": access,
        "source_shards": shard_evidence,
        "input_coverage": input_coverage,
        "identification_interval_width": {
            target: _quantiles(intervals[target]) for target in _TARGETS
        },
        "development": overall_report,
        "by_split": split_reports,
        "by_region_profile": development_cell_reports,
        "by_split_region_profile": cell_reports,
        "missingness_ks": _ks_diagnostics(
            metadata_rows,
            split_development,
            epsilon_complete,
            alpha=ks_alpha,
        ),
        "training_readiness": readiness,
        "component_task_admission": component_task_admission,
        "validation_identity": {
            "dataset_valid": bool(validation["valid"]),
            "dataset_case_count": len(validation["case_ids"]),
            "sealed_test_file_sha256": file_hash(root / "sealed_test.json"),
        },
    }
    report["report_hash"] = content_hash(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--ks-alpha", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_development_dataset(
        args.dataset,
        args.pilot,
        epsilon=args.epsilon,
        ks_alpha=args.ks_alpha,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_hash": report["report_hash"],
                "formal_neural_training_ready": report["training_readiness"][
                    "formal_neural_training_ready"
                ],
                "test_result_json_lines_decoded": report["sealed_test"][
                    "test_result_json_lines_decoded"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
