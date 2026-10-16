"""Audit statistical training readiness for a completed sharded dataset.

This audit is descriptive.  It verifies neither physical validity nor research
admission.  It deliberately reports reference-design coverage, event counts,
Monte Carlo resolution, and leakage indicators separately.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from damage_engine.result_ledger import (  # noqa: E402
    iter_result_ledger,
    resolve_result_ledger,
    result_partition,
)


def _number(value: Any) -> float | int | None:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    quantiles = np.quantile(array, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    unique, counts = np.unique(array, return_counts=True)
    mode_index = int(np.argmax(counts))
    return {
        "count": int(array.size),
        "min": float(quantiles[0]),
        "p01": float(quantiles[1]),
        "p05": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": float(quantiles[6]),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "unique": int(unique.size),
        "mode": float(unique[mode_index]),
        "mode_count": int(counts[mode_index]),
        "mode_fraction": float(counts[mode_index] / array.size),
        "exact_zero_fraction": float(np.mean(array == 0.0)),
        "exact_one_fraction": float(np.mean(array == 1.0)),
    }


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if total <= 0:
        return {"successes": successes, "total": total, "rate": None, "ci95": [None, None]}
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "successes": int(successes),
        "total": int(total),
        "rate": float(p),
        "ci95": [float(max(0.0, center - half)), float(min(1.0, center + half))],
        "interval_semantics": "Wilson binomial interval for this synthetic design sample; not a real-world validity interval",
    }


def _normalized_entropy(values: np.ndarray, bins: int = 20) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"bins": bins, "entropy_bits": None, "normalized_entropy": None}
    counts, _ = np.histogram(array, bins=bins, range=(0.0, 1.0))
    probabilities = counts[counts > 0] / counts.sum()
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    return {
        "bins": bins,
        "occupied_bins": int(np.sum(counts > 0)),
        "entropy_bits": entropy,
        "normalized_entropy": entropy / math.log2(bins),
        "counts": counts.astype(int).tolist(),
    }


def _row_hash(values: np.ndarray, mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(values, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(mask, dtype=np.bool_).tobytes())
    return digest.hexdigest()


def _load(root: Path) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    shard_paths = sorted((root / "shards").glob("part-*/tensors.npz"))
    if not shard_paths:
        raise RuntimeError(f"no tensor shards found under {root}")
    row_fields = [
        "case_physics_x",
        "case_physics_feature_mask",
        "component_aux_y",
        "component_aux_label_mask",
        "component_aux_y_ci95",
        "component_aux_uncertainty_mask",
        "system_y",
        "system_label_mask",
        "system_y_ci95",
        "system_uncertainty_mask",
        "split_names",
        "case_ids",
        "case_hashes",
        "replicate_counts",
        "sampling_selection_probability",
        "sampling_design_weight_raw",
        "sampling_weight_mask",
        "sampling_round",
        "sampling_stratum_index",
        "sampling_role_index",
        "sampling_candidate_ids",
        "sampling_lineage_ids",
        "sampling_epistemic_group_ids",
    ]
    metadata_fields = [
        "case_physics_feature_names",
        "component_aux_label_names",
        "system_output_names",
        "sampling_stratum_names",
        "sampling_role_names",
    ]
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in row_fields}
    metadata: dict[str, list[str]] = {}
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            for name in row_fields:
                pieces[name].append(np.asarray(shard[name]))
            for name in metadata_fields:
                current = list(map(str, shard[name].tolist()))
                if name in metadata and metadata[name] != current:
                    raise RuntimeError(f"metadata changed across shards: {name}")
                metadata[name] = current
    return {name: np.concatenate(chunks, axis=0) for name, chunks in pieces.items()}, metadata


def _event_report(event: np.ndarray, splits: np.ndarray) -> dict[str, Any]:
    result = {"all": _wilson(int(np.sum(event)), int(event.size))}
    for split in ("train", "validation", "test"):
        mask = splits == split
        result[split] = _wilson(int(np.sum(event & mask)), int(np.sum(mask)))
    return result


def _result_ledger_paths(root: Path):
    root_manifest_path = root / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    declared: dict[str, dict] = {}
    for partition in root_manifest.get("partitions", []):
        if partition.get("name") != "audits":
            continue
        relative = Path(str(partition.get("path_glob", "")))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or any(character in str(relative) for character in "*?[]")
            or len(relative.parts) != 3
            or relative.parts[0] != "shards"
        ):
            raise RuntimeError("root manifest declares an invalid shard result path")
        shard_key = relative.parent.as_posix()
        if shard_key in declared:
            raise RuntimeError(f"root manifest repeats the result ledger for {shard_key}")
        declared[shard_key] = partition

    shard_manifests = sorted((root / "shards").glob("part-*/manifest.json"))
    if not shard_manifests or len(declared) != len(shard_manifests):
        raise RuntimeError("root/shard manifests disagree on the result-ledger count")

    observed: set[str] = set()
    for manifest_path in shard_manifests:
        shard_directory = manifest_path.parent
        shard_key = shard_directory.relative_to(root).as_posix()
        if shard_key not in declared:
            raise RuntimeError(f"root manifest omits the result ledger for {shard_key}")
        child_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        child_partition = result_partition(child_manifest)
        root_partition = declared[shard_key]
        result_path = resolve_result_ledger(shard_directory, child_manifest)
        root_result_path = (root / Path(str(root_partition["path_glob"]))).resolve()
        if result_path != root_result_path:
            raise RuntimeError(f"root/shard result paths disagree for {shard_key}")
        for field in ("format", "row_count", "content_hash"):
            if root_partition.get(field) != child_partition.get(field):
                raise RuntimeError(
                    f"root/shard result partition field differs for {shard_key}: {field}"
                )
        observed.add(shard_key)
        yield result_path, int(root_partition["row_count"])

    if observed != set(declared):
        raise RuntimeError("root manifest contains undeclared or orphan result ledgers")


def _trace_audit(root: Path, system_names: list[str], aggregate_system: np.ndarray) -> dict[str, Any]:
    trace_counts: Counter[int] = Counter()
    distinct_parameter_hashes: Counter[int] = Counter()
    varying_cases = 0
    system_ranges: dict[str, list[float]] = {name: [] for name in system_names}
    aggregate_errors: dict[str, list[float]] = {name: [] for name in system_names}
    cases_with_any_fragment_hit = 0
    total_records = 0
    row_index = 0
    for path, expected_count in _result_ledger_paths(root):
        shard_count = 0
        for record in iter_result_ledger(path):
            trace = record.get("replicate_trace")
            if not isinstance(trace, dict) or not isinstance(trace.get("records"), list):
                raise RuntimeError(f"missing replicate trace in {path}")
            rows = trace["records"]
            trace_counts[len(rows)] += 1
            total_records += len(rows)
            hashes = {str(item.get("parameter_realization_hash")) for item in rows}
            distinct_parameter_hashes[len(hashes)] += 1
            any_varying = False
            any_hit = False
            for output_index, name in enumerate(system_names):
                values = np.asarray(
                    [item["system_outputs"][name] for item in rows], dtype=np.float64
                )
                value_range = float(np.max(values) - np.min(values))
                system_ranges[name].append(value_range)
                any_varying = any_varying or value_range > 0.0
                aggregate_errors[name].append(
                    abs(float(np.mean(values)) - float(aggregate_system[row_index, output_index]))
                )
            for item in rows:
                fragment_metrics = item.get("fragment_metrics", {})
                if any(float(metrics.get("hits", 0.0)) > 0.0 for metrics in fragment_metrics.values()):
                    any_hit = True
            varying_cases += int(any_varying)
            cases_with_any_fragment_hit += int(any_hit)
            row_index += 1
            shard_count += 1
        if shard_count != expected_count:
            raise RuntimeError(f"result trace count disagrees with manifest row_count in {path}")
    if row_index != aggregate_system.shape[0]:
        raise RuntimeError("result trace count does not match tensor rows")
    return {
        "case_count": row_index,
        "total_replicate_records": total_records,
        "replicate_count_distribution": {str(k): int(v) for k, v in sorted(trace_counts.items())},
        "distinct_parameter_realization_hashes_per_case": {
            str(k): int(v) for k, v in sorted(distinct_parameter_hashes.items())
        },
        "cases_with_any_system_output_variation_across_replicates": varying_cases,
        "fraction_with_any_system_output_variation": varying_cases / row_index,
        "cases_with_any_fragment_hit_in_any_replicate": cases_with_any_fragment_hit,
        "system_output_range_across_replicates": {
            name: _summary(np.asarray(values)) for name, values in system_ranges.items()
        },
        "trace_mean_vs_aggregate_absolute_error": {
            name: _summary(np.asarray(values)) for name, values in aggregate_errors.items()
        },
        "interpretation": "R=4 traces diagnose synthetic-teacher Monte Carlo variation only; they do not quantify model-form or real-world uncertainty.",
    }


def audit(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data, metadata = _load(root)
    splits = data["split_names"].astype(str)
    case_count = len(splits)
    feature_names = metadata["case_physics_feature_names"]
    aux_names = metadata["component_aux_label_names"]
    system_names = metadata["system_output_names"]
    aux_index = {name: index for index, name in enumerate(aux_names)}

    hits = data["component_aux_y"][:, 0, aux_index["fragment_hits"]]
    penetrations = data["component_aux_y"][:, 0, aux_index["fragment_penetrations"]]
    shielding_panels = data["component_aux_y"][:, 0, aux_index["blast_shielding_panel_count"]]
    any_hit = hits > 0.0
    any_penetration = penetrations > 0.0
    hit_without_penetration = any_hit & ~any_penetration
    partial_nonpenetration = hits > penetrations
    shielding = shielding_panels > 0.0

    weights = data["sampling_design_weight_raw"].astype(np.float64)
    weight_mask = data["sampling_weight_mask"].astype(bool)
    eligible_weights = weights[weight_mask]
    normalized_weights = eligible_weights / np.sum(eligible_weights)
    ess = float(np.sum(eligible_weights) ** 2 / np.sum(eligible_weights**2))

    feature_mask = data["case_physics_feature_mask"].astype(bool)
    constant_features = []
    variable_features = []
    missing_features = []
    for index, name in enumerate(feature_names):
        available = feature_mask[:, index]
        if not np.all(available):
            missing_features.append({"name": name, "available": int(np.sum(available))})
        values = data["case_physics_x"][available, index]
        if values.size and np.unique(values).size > 1:
            variable_features.append(name)
        else:
            constant_features.append(name)

    hashes = [
        _row_hash(row, mask)
        for row, mask in zip(data["case_physics_x"], feature_mask)
    ]
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    for row_hash, split in zip(hashes, splits):
        hash_to_splits[row_hash].add(str(split))
    duplicate_counts = Counter(hashes)

    position = data["case_physics_x"][:, :3].astype(np.float64)
    radius = np.linalg.norm(position, axis=1)
    unit = position / radius[:, None]
    azimuth = (np.arctan2(unit[:, 1], unit[:, 0]) + 2.0 * math.pi) % (2.0 * math.pi)
    equal_area_z = (unit[:, 2] + 1.0) / 2.0
    cells = np.minimum((equal_area_z * 10).astype(int), 9) * 20 + np.minimum(
        (azimuth / (2.0 * math.pi) * 20).astype(int), 19
    )
    cell_counts = np.bincount(cells, minlength=200)
    quaternions = data["case_physics_x"][:, 4:8].astype(np.float64)

    system_modes: list[float] = []
    system_stats: dict[str, Any] = {}
    for index, name in enumerate(system_names):
        values = data["system_y"][:, index]
        unique, counts = np.unique(values, return_counts=True)
        system_modes.append(float(unique[int(np.argmax(counts))]))
        system_stats[name] = {
            "summary": _summary(values),
            "histogram_entropy": _normalized_entropy(values),
        }
    joint_mode = np.all(data["system_y"] == np.asarray(system_modes, dtype=np.float32), axis=1)

    uncertainty: dict[str, Any] = {}
    for index, name in enumerate(system_names):
        mask = data["system_uncertainty_mask"][:, index].astype(bool)
        half_width = (
            data["system_y_ci95"][:, index, 1] - data["system_y_ci95"][:, index, 0]
        ) / 2.0
        uncertainty[name] = {
            "uncertainty_mask_count": int(np.sum(mask)),
            "positive_half_width_count": int(np.sum(mask & (half_width > 0.0))),
            "half_width": _summary(half_width[mask]),
            "half_width_gt_0_05": int(np.sum(mask & (half_width > 0.05))),
            "half_width_gt_0_10": int(np.sum(mask & (half_width > 0.10))),
        }

    role_names = metadata["sampling_role_names"]
    stratum_names = metadata["sampling_stratum_names"]
    role_counts = Counter(role_names[int(index)] for index in data["sampling_role_index"])
    stratum_counts = Counter(stratum_names[int(index)] for index in data["sampling_stratum_index"])

    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "scope": {
            "dataset_root": str(root),
            "dataset_id": manifest.get("dataset_id"),
            "engine_version": manifest.get("engine", {}).get("version"),
            "case_count": case_count,
            "split_counts": dict(Counter(map(str, splits))),
            "purpose_admission_counts": manifest.get("quality_summary", {}).get("purpose_admission_counts"),
            "parameter_set_ids": manifest.get("source_contracts", {}).get("parameter_set_ids"),
            "evidence_boundary": "Software and synthetic-design statistical audit only; physical validity and research admission remain unestablished.",
        },
        "sampling_design": {
            "role_counts": dict(sorted(role_counts.items())),
            "stratum_counts": dict(sorted(stratum_counts.items())),
            "round_counts": {str(k): int(v) for k, v in sorted(Counter(map(int, data["sampling_round"])).items())},
            "selection_probability": _summary(data["sampling_selection_probability"]),
            "design_weight_raw": _summary(eligible_weights),
            "effective_sample_size": ess,
            "ess_fraction": ess / len(eligible_weights),
            "weight_cv": float(np.std(eligible_weights) / np.mean(eligible_weights)),
            "max_normalized_weight": float(np.max(normalized_weights)),
            "note": "Raw design weights are neither class weights nor minibatch sampling probabilities.",
        },
        "input_features": {
            "raw_feature_count": len(feature_names),
            "variable_feature_count": len(variable_features),
            "variable_features": variable_features,
            "constant_feature_count": len(constant_features),
            "constant_features": constant_features,
            "features_with_missing_values": missing_features,
            "sphere_shell_radius_m": _summary(radius),
            "sphere_equal_area_20x10": {
                "occupied_cells": int(np.sum(cell_counts > 0)),
                "total_cells": 200,
                "occupancy_fraction": float(np.mean(cell_counts > 0)),
                "min_count": int(np.min(cell_counts)),
                "median_count": float(np.median(cell_counts)),
                "max_count": int(np.max(cell_counts)),
            },
            "quaternion_norm": _summary(np.linalg.norm(quaternions, axis=1)),
        },
        "leakage_and_duplicates": {
            "unique_exact_physics_rows": len(duplicate_counts),
            "duplicate_row_groups": int(sum(count > 1 for count in duplicate_counts.values())),
            "duplicate_rows_beyond_first": int(sum(count - 1 for count in duplicate_counts.values() if count > 1)),
            "exact_physics_rows_crossing_splits": int(sum(len(value) > 1 for value in hash_to_splits.values())),
            "candidate_id_unique": len(set(map(str, data["sampling_candidate_ids"]))) == case_count,
            "lineage_id_unique": len(set(map(str, data["sampling_lineage_ids"]))) == case_count,
            "epistemic_group_id_unique": len(set(map(str, data["sampling_epistemic_group_ids"]))) == case_count,
        },
        "mechanism_events": {
            "fragment_hit": _event_report(any_hit, splits),
            "fragment_penetration": _event_report(any_penetration, splits),
            "fragment_hit_without_penetration": _event_report(hit_without_penetration, splits),
            "fragment_hit_count_exceeds_penetration_count": _event_report(partial_nonpenetration, splits),
            "blast_shielding_panel": _event_report(shielding, splits),
            "shielding_and_fragment_hit": _event_report(shielding & any_hit, splits),
            "no_shielding_and_no_fragment_hit": _event_report(~shielding & ~any_hit, splits),
            "hit_to_penetration_fraction": float(np.sum(any_penetration) / np.sum(any_hit)) if np.any(any_hit) else None,
            "penetration_count_to_hit_count_ratio": float(np.sum(penetrations) / np.sum(hits)) if np.sum(hits) > 0 else None,
            "raw_hit_summary": _summary(hits),
            "raw_penetration_summary": _summary(penetrations),
        },
        "labels": {
            "system_outputs": system_stats,
            "joint_mode_vector": system_modes,
            "joint_mode_count": int(np.sum(joint_mode)),
            "joint_mode_fraction": float(np.mean(joint_mode)),
            "joint_unique_rows": int(np.unique(data["system_y"], axis=0).shape[0]),
            "joint_mode_fraction_given_no_fragment_hit": float(np.mean(joint_mode[~any_hit])),
            "joint_mode_fraction_given_fragment_hit": float(np.mean(joint_mode[any_hit])) if np.any(any_hit) else None,
            "mission_loss_le_mission_degraded_fraction": float(np.mean(data["system_y"][:, 1] <= data["system_y"][:, 0])),
        },
        "uncertainty": {
            "replicate_count": _summary(data["replicate_counts"]),
            "system": uncertainty,
            "caveat": "CI95 is a finite-R conditional Monte Carlo interval, not total uncertainty and not a real-world prediction interval.",
        },
    }
    report["replicate_trace"] = _trace_audit(root, system_names, data["system_y"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "case_count": report["scope"]["case_count"]}))


if __name__ == "__main__":
    main()
