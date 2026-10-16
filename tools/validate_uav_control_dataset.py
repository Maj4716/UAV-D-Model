"""Fail-closed acceptance audit for a completed UAV-control dataset.

The audit accepts either a sampling-run directory containing ``formal_dataset``
or the completed sharded dataset itself.  It deliberately proves only archive,
model-input and synthetic-design integrity.  Passing this tool never grants
physical validation or research admission.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from damage_engine.learning_input_identity import (  # noqa: E402
    FIXED_SCENARIO_IDENTITY_VERSION,
    LEARNING_INPUT_IDENTITY_VERSION,
    UAV_STATE_CONTRACT_VERSION,
    UAV_STATE_FEATURE_NAMES,
    UAV_TRAINING_FEATURE_NAMES,
)
from damage_engine.normalization import (  # noqa: E402
    save_training_normalization_artifact,
)
from damage_engine.result_ledger import (  # noqa: E402
    iter_result_ledger,
    resolve_result_ledger,
    result_partition,
)
from damage_engine.torch_dataset import TorchShardedDamageDataset  # noqa: E402
from damage_engine.training_view import (  # noqa: E402
    TrainingViewDataset,
    fit_training_view_artifact,
)
from damage_engine.utils import file_hash  # noqa: E402


REPORT_SCHEMA_VERSION = "uav-control-dataset-acceptance@1.0.0"
EXPECTED_TENSOR_CONTRACT = "1.8.0"
SPLITS = ("train", "validation", "test")
FORBIDDEN_DEFAULT_FIELDS = frozenset(
    {
        "case_x",
        "case_feature_mask",
        "case_physics_x",
        "case_physics_feature_mask",
        "solver_context_x",
        "solver_context_feature_mask",
        "component_x",
        "component_feature_mask",
        "armor_x",
        "armor_feature_mask",
        "component_armor_path_x",
        "component_armor_path_mask",
        "armor_vertices_m",
        "armor_vertex_offsets",
        "component_vertices_m",
        "component_vertex_offsets",
        "tree_edge_index",
        "tree_edge_type",
    }
)


def _resolve_dataset_root(path: str | Path) -> tuple[Path, str]:
    requested = Path(path).resolve()
    direct = requested / "manifest.json"
    nested = requested / "formal_dataset" / "manifest.json"
    if direct.is_file() and (requested / "checkpoint.json").is_file():
        return requested, "formal_dataset"
    if nested.is_file() and (nested.parent / "checkpoint.json").is_file():
        return nested.parent.resolve(), "sampling_run"
    raise FileNotFoundError(
        "expected a completed sharded dataset or a sampling run containing "
        f"formal_dataset: {requested}"
    )


def _require_external_artifact_path(path: Path, dataset_root: Path, *, name: str) -> None:
    resolved = path.resolve()
    if resolved == dataset_root or resolved.is_relative_to(dataset_root):
        raise ValueError(
            f"{name} must be outside the immutable formal dataset directory"
        )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _tensor_digest(values: np.ndarray, mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(values, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(mask, dtype=np.bool_).tobytes())
    return "sha256:" + digest.hexdigest()


def _weight_diagnostics(weights: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(weights, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        return {
            "eligible_n": 0,
            "ineligible_n": int(len(weights)),
            "ess": 0.0,
            "ess_fraction": None,
            "tail": {"available": False, "reason": "no weight-eligible rows"},
        }
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("eligible design weights must be finite and positive")
    total = float(np.sum(values))
    square_total = float(np.sum(values * values))
    ess = total * total / square_total
    sorted_desc = np.sort(values)[::-1]
    top_count = max(1, int(math.ceil(0.01 * len(sorted_desc))))
    quantiles = np.quantile(values, [0.5, 0.9, 0.95, 0.99, 1.0])
    mean = float(np.mean(values))
    return {
        "eligible_n": int(values.size),
        "ineligible_n": int(len(weights) - values.size),
        "ess": float(ess),
        "ess_fraction": float(ess / values.size),
        "tail": {
            "available": True,
            "min": float(np.min(values)),
            "median": float(quantiles[0]),
            "p90": float(quantiles[1]),
            "p95": float(quantiles[2]),
            "p99": float(quantiles[3]),
            "max": float(quantiles[4]),
            "mean": mean,
            "cv": float(np.std(values) / mean),
            "max_normalized_weight": float(sorted_desc[0] / total),
            "top_1_percent_weight_share": float(
                np.sum(sorted_desc[:top_count]) / total
            ),
        },
        "semantics": (
            "Raw design-weight ESS describes reference-risk support only; it is "
            "not an event count, class-loss weight or minibatch probability."
        ),
    }


def _event_summary(
    event: np.ndarray,
    available: np.ndarray,
    weights: np.ndarray,
    weight_mask: np.ndarray,
) -> dict[str, Any]:
    event = np.asarray(event, dtype=bool)
    available = np.asarray(available, dtype=bool)
    if event.shape != available.shape:
        raise ValueError("event and availability axes differ")
    available_n = int(np.sum(available))
    if available_n == 0:
        return {
            "available": False,
            "reason": "required archived fields are unavailable for every row",
            "available_n": 0,
            "unavailable_n": int(len(event)),
        }
    selected = available & np.asarray(weight_mask, dtype=bool)
    selected_weights = np.asarray(weights, dtype=np.float64)[selected]
    successes = int(np.sum(event & available))
    weighted_denominator = float(np.sum(selected_weights))
    weighted_numerator = float(
        np.sum(np.asarray(weights, dtype=np.float64)[selected & event])
    )
    return {
        "available": True,
        "available_n": available_n,
        "unavailable_n": int(len(event) - available_n),
        "count": successes,
        "rate": float(successes / available_n),
        "weight_eligible_n": int(np.sum(selected)),
        "weighted_success_sum": weighted_numerator,
        "weighted_rate": (
            float(weighted_numerator / weighted_denominator)
            if weighted_denominator > 0.0
            else None
        ),
    }


def _masked_finite(value: np.ndarray, mask: np.ndarray, *, field: str) -> None:
    if value.shape != mask.shape:
        raise ValueError(f"{field} and its mask have different shapes")
    if mask.dtype != np.dtype(bool):
        raise ValueError(f"{field} mask must be boolean")
    if not np.isfinite(value[mask]).all():
        raise ValueError(f"{field} contains nonfinite available values")


def _validate_sample(sample: dict[str, Any], training_sample: dict[str, Any]) -> None:
    for name in FORBIDDEN_DEFAULT_FIELDS:
        if name in sample:
            raise ValueError(f"default model loader exposes forbidden field {name!r}")

    raw = _as_numpy(sample["uav_state_x"])
    raw_mask = _as_numpy(sample["uav_state_feature_mask"])
    if raw.shape != (10,) or raw_mask.shape != (10,) or not np.all(raw_mask):
        raise ValueError("default UAV state must be an all-available 10D vector")
    _masked_finite(raw, raw_mask, field="uav_state_x")

    training = _as_numpy(training_sample["training_x"])
    training_mask = _as_numpy(training_sample["training_feature_mask"])
    if (
        training.shape != (12,)
        or training_mask.shape != (12,)
        or not np.all(training_mask)
    ):
        raise ValueError("training view must be an all-available 12D vector")
    _masked_finite(training, training_mask, field="training_x")


def _validate_contract(manifest: dict, shard_manifests: Iterable[dict]) -> dict[str, Any]:
    all_manifests = [manifest, *shard_manifests]
    versions = {
        item.get("tensor_contract", {}).get("version") for item in all_manifests
    }
    if versions != {EXPECTED_TENSOR_CONTRACT}:
        raise ValueError(
            f"UAV-control acceptance requires tensor contract {EXPECTED_TENSOR_CONTRACT}; "
            f"observed {sorted(map(str, versions))}"
        )
    contracts = [item.get("model_input_contract") for item in all_manifests]
    if any(not isinstance(item, dict) for item in contracts):
        raise ValueError("root/shard manifests must all declare model_input_contract")
    first = contracts[0]
    if any(item != first for item in contracts[1:]):
        raise ValueError("model_input_contract differs between root and shards")
    feature_names = [item.get("name") for item in first.get("features", [])]
    if (
        first.get("version") != UAV_STATE_CONTRACT_VERSION
        or first.get("tensor_field") != "uav_state_x"
        or first.get("mask_field") != "uav_state_feature_mask"
        or feature_names != list(UAV_STATE_FEATURE_NAMES)
        or len(feature_names) != 10
    ):
        raise ValueError("manifest does not declare the exact 10D UAV-only input")
    excluded = set(first.get("excluded_tensor_fields", ()))
    required_excluded = set(FORBIDDEN_DEFAULT_FIELDS)
    if not required_excluded.issubset(excluded):
        raise ValueError("model_input_contract does not exclude teacher/solver inputs")
    forbidden_tokens = ("warhead", "fragment", "blast", "projectile", "environment")
    if any(any(token in name.lower() for token in forbidden_tokens) for name in feature_names):
        raise ValueError("UAV model input names contain scenario/projectile semantics")
    return {
        "tensor_contract_version": EXPECTED_TENSOR_CONTRACT,
        "model_input_version": first["version"],
        "raw_tensor_field": first["tensor_field"],
        "raw_feature_count": len(feature_names),
        "raw_feature_names": feature_names,
        "training_transform": "position3 + velocity3 + quaternion_to_rotation6d",
        "training_feature_count": len(UAV_TRAINING_FEATURE_NAMES),
        "training_feature_names": list(UAV_TRAINING_FEATURE_NAMES),
        "excluded_teacher_or_solver_fields": sorted(required_excluded),
        "evidence_boundary": first.get("evidence_boundary"),
    }


def _collect_archive_rows(
    dataset: TorchShardedDamageDataset,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ledger_formats: set[str] = set()
    ledger_paths: list[str] = []
    ledger_count = 0
    result_case_ids: set[str] = set()
    fixed_hashes: set[str] = set()
    shard_manifests: list[dict] = []

    for shard_path in dataset.shard_paths:
        manifest_path = shard_path / "manifest.json"
        child = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_manifests.append(child)
        source = child.get("source_contracts", {})
        fixed_hash = source.get("fixed_scenario_hash")
        if not isinstance(fixed_hash, str) or not fixed_hash.startswith("sha256:"):
            raise ValueError(f"shard lacks fixed_scenario_hash: {shard_path}")
        fixed_hashes.add(fixed_hash)

        partition = result_partition(child)
        ledger_formats.add(str(partition["format"]))
        ledger_path = resolve_result_ledger(shard_path, child)
        ledger_paths.append(str(ledger_path))
        ledger_records = list(iter_result_ledger(ledger_path))
        if len(ledger_records) != int(partition.get("row_count", -1)):
            raise ValueError(f"result ledger row_count mismatch: {ledger_path}")

        with np.load(shard_path / "tensors.npz", allow_pickle=False) as arrays:
            input_ids = list(map(str, arrays["input_case_ids"].tolist()))
            input_status = list(map(str, arrays["input_case_status"].tolist()))
            ledger_ids = [str(item.get("case_id") or "") for item in ledger_records]
            ledger_status = [str(item.get("status") or "") for item in ledger_records]
            if ledger_ids != input_ids or ledger_status != input_status:
                raise ValueError(
                    f"result ledger order/status differs from full tensor input axis: {ledger_path}"
                )
            if any(status != "ok" for status in ledger_status):
                raise ValueError(
                    "UAV-control training acceptance refuses archives with failed executions"
                )
            if result_case_ids.intersection(ledger_ids):
                raise ValueError("result case IDs repeat across shard ledgers")
            result_case_ids.update(ledger_ids)
            ledger_count += len(ledger_records)

            case_ids = np.asarray(arrays["case_ids"]).astype(str)
            splits = np.asarray(arrays["split_names"]).astype(str)
            uav = np.asarray(arrays["uav_state_x"], dtype=np.float32)
            uav_mask = np.asarray(arrays["uav_state_feature_mask"], dtype=bool)
            roles = np.asarray(arrays["sampling_role_names"]).astype(str)[
                np.asarray(arrays["sampling_role_index"], dtype=np.int64)
            ]
            strata = np.asarray(arrays["sampling_stratum_names"]).astype(str)[
                np.asarray(arrays["sampling_stratum_index"], dtype=np.int64)
            ]
            rounds = np.asarray(arrays["sampling_round"], dtype=np.int64)
            weights = np.asarray(arrays["sampling_design_weight_raw"], dtype=np.float64)
            weight_mask = np.asarray(arrays["sampling_weight_mask"], dtype=bool)

            aux_names = list(map(str, arrays["component_aux_label_names"].tolist()))
            aux_values = np.asarray(arrays["component_aux_y"], dtype=np.float64)
            aux_mask = np.asarray(arrays["component_aux_label_mask"], dtype=bool)
            hit_index = aux_names.index("fragment_hits") if "fragment_hits" in aux_names else None
            penetration_index = (
                aux_names.index("fragment_penetrations")
                if "fragment_penetrations" in aux_names
                else None
            )
            if hit_index is not None:
                hit_available = np.all(aux_mask[:, :, hit_index], axis=1)
                hits = np.sum(
                    np.where(aux_mask[:, :, hit_index], aux_values[:, :, hit_index], 0.0),
                    axis=1,
                )
            else:
                hit_available = np.zeros(len(case_ids), dtype=bool)
                hits = np.zeros(len(case_ids), dtype=np.float64)
            if penetration_index is not None:
                penetration_available = np.all(
                    aux_mask[:, :, penetration_index], axis=1
                )
                penetrations = np.sum(
                    np.where(
                        aux_mask[:, :, penetration_index],
                        aux_values[:, :, penetration_index],
                        0.0,
                    ),
                    axis=1,
                )
            else:
                penetration_available = np.zeros(len(case_ids), dtype=bool)
                penetrations = np.zeros(len(case_ids), dtype=np.float64)

            path_names = list(map(str, arrays["armor_path_feature_names"].tolist()))
            if "intersects_path" in path_names:
                path_index = path_names.index("intersects_path")
                path_values = np.asarray(
                    arrays["component_armor_path_x"], dtype=np.float64
                )[..., path_index]
                path_mask = np.asarray(
                    arrays["component_armor_path_mask"], dtype=bool
                )[..., path_index]
                occlusion_available = np.all(path_mask, axis=(1, 2))
                occluded = np.any(path_mask & (path_values > 0.5), axis=(1, 2))
            else:
                occlusion_available = np.zeros(len(case_ids), dtype=bool)
                occluded = np.zeros(len(case_ids), dtype=bool)

            if not (
                len(case_ids)
                == len(splits)
                == len(uav)
                == len(roles)
                == len(strata)
                == len(rounds)
                == len(weights)
                == len(weight_mask)
            ):
                raise ValueError("shard row axes are inconsistent")
            for index, case_id in enumerate(case_ids):
                both_available = bool(hit_available[index] and penetration_available[index])
                tolerance = 1.0e-9 * max(1.0, abs(float(hits[index])))
                if both_available and (
                    hits[index] < -tolerance
                    or penetrations[index] < -tolerance
                    or penetrations[index] - hits[index] > tolerance
                ):
                    raise ValueError(
                        "archived aggregate fragment hit/penetration counts are inconsistent"
                    )
                any_hit = bool(hits[index] > tolerance)
                any_penetration = bool(penetrations[index] > tolerance)
                not_all = bool(hits[index] - penetrations[index] > tolerance)
                rows.append(
                    {
                        "case_id": str(case_id),
                        "split": str(splits[index]),
                        "role": str(roles[index]),
                        "stratum": str(strata[index]),
                        "round": int(rounds[index]),
                        "weight": float(weights[index]),
                        "weight_mask": bool(weight_mask[index]),
                        "learning_input_hash": _tensor_digest(
                            uav[index], uav_mask[index]
                        ),
                        "events": {
                            "fragment_hit": (any_hit, bool(hit_available[index])),
                            "hit_but_not_all_penetrated": (not_all, both_available),
                            "partial_penetration": (
                                any_penetration and not_all,
                                both_available,
                            ),
                            "full_penetration": (
                                any_hit and any_penetration and not not_all,
                                both_available,
                            ),
                            "occluded_and_fragment_hit": (
                                bool(occluded[index] and any_hit),
                                bool(occlusion_available[index] and hit_available[index]),
                            ),
                        },
                    }
                )

    root_fixed_hash = dataset.manifest.get("source_contracts", {}).get(
        "fixed_scenario_hash"
    )
    if isinstance(root_fixed_hash, str):
        fixed_hashes.add(root_fixed_hash)
    if len(fixed_hashes) != 1:
        raise ValueError(
            "archive does not bind exactly one fixed target/threat/environment scenario"
        )
    if len(rows) != int(dataset.manifest["quality_summary"]["n_valid"]):
        raise ValueError("collected valid rows disagree with root quality summary")
    if len(result_case_ids) != len(rows) or result_case_ids != {
        row["case_id"] for row in rows
    }:
        raise ValueError("result ledgers and valid tensor cases do not match exactly")
    return rows, {
        "shard_manifests": shard_manifests,
        "fixed_scenario_hashes": sorted(fixed_hashes),
        "result_ledger": {
            "loaded": True,
            "row_count": ledger_count,
            "formats": sorted(ledger_formats),
            "paths": ledger_paths,
            "gzip_compatible": "jsonl_gzip" in ledger_formats,
            "plain_compatible": "jsonl" in ledger_formats,
        },
    }


def _validate_no_cross_split_duplicates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    case_splits: dict[str, set[str]] = defaultdict(set)
    input_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        case_splits[row["case_id"]].add(row["split"])
        input_splits[row["learning_input_hash"]].add(row["split"])
    duplicate_case_ids = sorted(
        identifier for identifier, splits in case_splits.items() if len(splits) > 1
    )
    crossing_inputs = sorted(
        identifier for identifier, splits in input_splits.items() if len(splits) > 1
    )
    if len(case_splits) != len(rows) or duplicate_case_ids:
        raise ValueError("case_id values are not globally unique across splits")
    if crossing_inputs:
        raise ValueError(
            "identical archived UAV learning-input tensors cross split boundaries"
        )
    return {
        "case_id_count": len(case_splits),
        "case_ids_globally_unique": True,
        "learning_input_identity": (
            "sha256 over archived float32 uav_state_x plus availability mask; "
            "this is the exact learner-visible duplicate guard"
        ),
        "unique_learning_inputs": len(input_splits),
        "duplicate_learning_inputs_within_same_split": int(
            len(rows) - len(input_splits)
        ),
        "learning_inputs_crossing_splits": 0,
    }


def _group_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["role"], row["stratum"], row["round"])].append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(grouped):
        members = grouped[key]
        weights = np.asarray([row["weight"] for row in members], dtype=np.float64)
        weight_mask = np.asarray(
            [row["weight_mask"] for row in members], dtype=bool
        )
        events: dict[str, Any] = {}
        for event_name in next(iter(members))["events"]:
            event = np.asarray(
                [row["events"][event_name][0] for row in members], dtype=bool
            )
            available = np.asarray(
                [row["events"][event_name][1] for row in members], dtype=bool
            )
            events[event_name] = _event_summary(
                event, available, weights, weight_mask
            )
        result.append(
            {
                "split": key[0],
                "sampling_role": key[1],
                "stratum": key[2],
                "round": key[3],
                "n": len(members),
                "weights": _weight_diagnostics(weights, weight_mask),
                "events": events,
            }
        )
    return result


def validate_dataset(
    root: str | Path,
    *,
    normalization_output: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one completed UAV-control archive and return report/artifact."""

    dataset_root, input_kind = _resolve_dataset_root(root)
    normalization_path = (
        Path(normalization_output).resolve()
        if normalization_output is not None
        else None
    )
    if normalization_path is not None:
        _require_external_artifact_path(
            normalization_path, dataset_root, name="normalization_output"
        )

    datasets: dict[str, TorchShardedDamageDataset] = {}
    try:
        for split in SPLITS:
            datasets[split] = TorchShardedDamageDataset(
                dataset_root,
                split=split,
                copy_on_read=False,
                verify_integrity=True,
                allow_split_leakage=False,
                allow_legacy_manifest=False,
                purpose="software_development",
            )
            if len(datasets[split]) == 0:
                raise ValueError(f"UAV-control acceptance requires nonempty {split} split")

        train = datasets["train"]
        if train.include_static_graph:
            raise ValueError("tensor 1.8 default loader must hide fixed static graph inputs")
        if FORBIDDEN_DEFAULT_FIELDS.intersection(train.sample_fields):
            raise ValueError("default tensor field list exposes teacher/solver inputs")
        rows, archive = _collect_archive_rows(train)
        contract = _validate_contract(
            train.manifest, archive.pop("shard_manifests")
        )
        root_fields = {
            item.get("name") for item in train.manifest["tensor_contract"]["fields"]
        }
        retained_for_audit = sorted(
            {"case_physics_x", "solver_context_x"}.intersection(root_fields)
        )
        contract["default_loader_forbidden_fields_absent"] = sorted(
            FORBIDDEN_DEFAULT_FIELDS
        )
        contract["archive_only_teacher_audit_fields"] = retained_for_audit

        source = train.manifest.get("source_contracts", {})
        if source.get("fixed_scenario_identity_version") != FIXED_SCENARIO_IDENTITY_VERSION:
            raise ValueError("fixed-scenario identity version is missing or unsupported")
        if source.get("learning_input_identity_version") != LEARNING_INPUT_IDENTITY_VERSION:
            raise ValueError("learning-input identity version is missing or unsupported")

        artifact = fit_training_view_artifact(train, proxy_boundary="end_to_end")
        artifact_names = [item["name"] for item in artifact["features"]]
        if artifact_names != list(UAV_TRAINING_FEATURE_NAMES):
            raise ValueError("normalization artifact does not describe the exact 12D UAV view")
        if artifact.get("fit", {}).get("split") != "train":
            raise ValueError("normalization artifact was not fit on Train only")

        split_traversal: dict[str, Any] = {}
        for split, dataset in datasets.items():
            view = TrainingViewDataset(
                dataset,
                artifact,
                proxy_boundary="end_to_end",
            )
            if tuple(view.feature_names) != tuple(UAV_TRAINING_FEATURE_NAMES):
                raise ValueError("training view feature order differs from the contract")
            seen: set[str] = set()
            for index in range(len(dataset)):
                sample = dataset[index]
                training_sample = view[index]
                if sample.get("split") != split:
                    raise ValueError(f"{split} loader yielded a different split")
                _validate_sample(sample, training_sample)
                case_id = str(sample.get("case_id") or "")
                if not case_id or case_id in seen:
                    raise ValueError(f"{split} traversal yielded missing/duplicate case_id")
                seen.add(case_id)
            split_traversal[split] = {
                "n": len(seen),
                "raw_uav_state_shape": [10],
                "training_shape": [12],
                "all_rows_traversed": True,
                "available_values_finite": True,
                "masks_and_shapes_valid": True,
                "normalization_operation": "fit_then_apply" if split == "train" else "apply_only",
            }

        leakage = _validate_no_cross_split_duplicates(rows)
        split_counts = {
            split: sum(row["split"] == split for row in rows) for split in SPLITS
        }
        if split_counts != {
            split: split_traversal[split]["n"] for split in SPLITS
        }:
            raise ValueError("low-level archive rows disagree with split loaders")

        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "passed",
            "dataset": {
                "requested_root": str(Path(root).resolve()),
                "resolved_formal_dataset": str(dataset_root),
                "input_kind": input_kind,
                "dataset_id": train.manifest.get("dataset_id"),
                "manifest_hash": file_hash(dataset_root / "manifest.json"),
                "case_count": len(rows),
                "split_counts": split_counts,
                "shard_count": len(train.shard_paths),
            },
            "integrity": {
                "root_and_shard_hashes_verified": True,
                "complete_checkpoint_verified": True,
                "all_tensor_fields_validated": True,
                "all_result_ledgers_validated": True,
                "split_leakage_override_used": False,
                "legacy_override_used": False,
                "forensic_integrity_override_used": False,
            },
            "model_input_contract": contract,
            "fixed_scenario": {
                "policy": "single_frozen_scenario_per_archive",
                "unique_hash_count": len(archive["fixed_scenario_hashes"]),
                "hashes": archive["fixed_scenario_hashes"],
                "identity_version": source["fixed_scenario_identity_version"],
            },
            "normalization": {
                "normalization_id": artifact["normalization_id"],
                "artifact_path": str(normalization_path) if normalization_path else None,
                "fit_split": "train",
                "fit_case_count": artifact["fit"]["case_count"],
                "validation_operation": "apply_only",
                "test_operation": "apply_only",
                "target_or_label_statistics_used_for_fit": False,
            },
            "split_traversal": split_traversal,
            "leakage_and_duplicates": leakage,
            "result_ledgers": archive["result_ledger"],
            "event_definitions": {
                "fragment_hit": "sum(component.fragment_hits) > 0",
                "hit_but_not_all_penetrated": (
                    "sum(component.fragment_hits) > sum(component.fragment_penetrations)"
                ),
                "partial_penetration": (
                    "0 < sum(component.fragment_penetrations) < sum(component.fragment_hits)"
                ),
                "full_penetration": (
                    "fragment hit and positive penetrations with aggregate hit and "
                    "penetration counts equal within numerical tolerance"
                ),
                "occluded_and_fragment_hit": (
                    "any component_armor_path.intersects_path > 0.5 and fragment hit"
                ),
                "count_semantics": (
                    "Descriptive case counts from synthetic aggregate outputs; these are "
                    "not independent real-world Bernoulli trials."
                ),
            },
            "groups": _group_report(rows),
            "research_admission": {
                "admitted": False,
                "evidence_boundary": (
                    "Software/archive integrity, learner-input isolation and synthetic-design "
                    "coverage only. No physical calibration, external validation, operational "
                    "safety validation or research admission is established."
                ),
                "prohibited_inference": (
                    "A passed report must not be presented as proof of real UAV damage, "
                    "control effectiveness or autonomous-flight safety."
                ),
            },
        }
        if normalization_path is not None:
            # Publish only after every acceptance condition has succeeded, so
            # a failed archive cannot leave a seemingly usable normalizer.
            save_training_normalization_artifact(artifact, normalization_path)
        return report, artifact
    finally:
        for dataset in datasets.values():
            dataset.clear_cache()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strictly validate a completed UAV-control sharded dataset."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Sampling-run root or its completed formal_dataset directory.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Acceptance JSON path.")
    parser.add_argument(
        "--normalization-output",
        type=Path,
        help=(
            "Train-only normalization JSON path. Defaults beside --output with "
            "'.normalization.json' appended to the stem."
        ),
    )
    arguments = parser.parse_args(argv)
    output = arguments.output.resolve()
    normalization = (
        arguments.normalization_output.resolve()
        if arguments.normalization_output is not None
        else output.with_name(output.stem + ".normalization.json")
    )
    if output == normalization:
        parser.error("--output and --normalization-output must differ")
    dataset_root: Path | None = None
    output_is_external = True
    try:
        dataset_root, _ = _resolve_dataset_root(arguments.root)
        output_is_external = not (
            output == dataset_root or output.is_relative_to(dataset_root)
        )
        _require_external_artifact_path(output, dataset_root, name="output")
        report, _artifact = validate_dataset(
            arguments.root,
            normalization_output=normalization,
        )
        _write_json(output, report)
    except Exception as exc:
        failure = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "failed",
            "requested_root": str(arguments.root.resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "research_admission": {
                "admitted": False,
                "evidence_boundary": "Validation failed; no dataset acceptance is granted.",
            },
        }
        # Never mutate the immutable archive merely to report that the user
        # selected an invalid in-archive output path.
        if output_is_external:
            _write_json(output, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
