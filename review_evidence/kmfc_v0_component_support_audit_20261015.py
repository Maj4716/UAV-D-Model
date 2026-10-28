"""Read-only component-support audit for the proposed project-specific KMFC-v0 tree.

The audit reads only committed tensor shards from the frozen 5,000-case archive.
It uses the archived case-level mean ``fragment_deposited_energy_j`` field as a
reachability/support diagnostic.  It does not calibrate a tree and does not
execute the simulator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "raw_data/dataset_5000_loitering_impact_fragment_only_v1"
FORMAL_ROOT = DATASET_ROOT / "formal_dataset"
SHARD_ROOT = FORMAL_ROOT / "shards"
CHECKPOINT_PATH = FORMAL_ROOT / "checkpoint.json"
MANIFEST_PATH = FORMAL_ROOT / "manifest.json"
TARGET_PATH = REPO_ROOT / "raw_data/target.rebuilt.fragment_only.v1.json"
OUTPUT_PATH = REPO_ROOT / "review_evidence/kmfc_v0_component_support_audit_20261015.json"

ZERO_TOLERANCE_J = 0.0

BRANCHES: dict[str, dict[str, list[str]]] = {
    "K": {
        "fuel_store": ["component-003"],
        "ammunition_store": ["component-046", "component-053", "component-054"],
    },
    "M": {
        "powertrain": ["component-001", "component-002", "component-003"],
        "drive_and_idler": [
            "component-004",
            "component-005",
            "component-030",
            "component-031",
        ],
        "right_carrying": [
            *[f"component-{index:03d}" for index in range(6, 12)],
            *[f"component-{index:03d}" for index in range(18, 21)],
        ],
        "left_carrying": [
            *[f"component-{index:03d}" for index in range(12, 18)],
            *[f"component-{index:03d}" for index in range(21, 24)],
        ],
        "suspension": [f"component-{index:03d}" for index in range(24, 30)],
        "right_track": [f"component-{index:03d}" for index in range(32, 36)],
        "left_track": [f"component-{index:03d}" for index in range(36, 40)],
        "mobility_operator": ["component-058"],
    },
    "F": {
        "sensing": ["component-040", "component-041", "component-042"],
        "fire_control": ["component-043", "component-044"],
        "weapon": [
            "component-045",
            "component-050",
            "component-051",
            "component-052",
        ],
        "feed_and_ammunition": [
            "component-046",
            "component-049",
            "component-053",
            "component-054",
        ],
        "countermeasure": ["component-047", "component-048"],
    },
    "C": {
        "essential_crew": ["component-058", "component-059", "component-060"],
        "embarked_personnel": [
            f"component-{index:03d}" for index in range(61, 68)
        ],
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _committed_tensor_paths(checkpoint: dict[str, Any]) -> list[Path]:
    if checkpoint.get("status") != "complete":
        raise RuntimeError("formal checkpoint is not complete")
    completed = checkpoint.get("completed_shards")
    if not isinstance(completed, list) or not completed:
        raise RuntimeError("formal checkpoint has no committed shards")
    expected_directories = {
        (FORMAL_ROOT / str(item["path"])).resolve() for item in completed
    }
    observed_directories = {
        path.resolve() for path in SHARD_ROOT.glob("part-*") if path.is_dir()
    }
    if expected_directories != observed_directories:
        raise RuntimeError("shard directory inventory differs from committed checkpoint")

    paths: list[Path] = []
    for item in sorted(completed, key=lambda value: int(value["index"])):
        directory = FORMAL_ROOT / str(item["path"])
        shard_manifest_path = directory / "manifest.json"
        if _file_hash(shard_manifest_path) != item.get("manifest_hash"):
            raise RuntimeError(f"shard manifest hash mismatch: {shard_manifest_path}")
        shard_manifest = _read_json(shard_manifest_path)
        partitions = [
            row
            for row in shard_manifest.get("partitions", [])
            if isinstance(row, dict) and row.get("name") == "tensors"
        ]
        if len(partitions) != 1:
            raise RuntimeError(f"expected one tensors partition: {shard_manifest_path}")
        partition = partitions[0]
        path = directory / str(partition["path_glob"])
        if _file_hash(path) != partition.get("content_hash"):
            raise RuntimeError(f"tensor partition hash mismatch: {path}")
        if int(partition.get("row_count", -1)) != len(item.get("case_ids", [])):
            raise RuntimeError(f"tensor partition row count mismatch: {path}")
        paths.append(path)
    return paths


def _count_by_split(mask: np.ndarray, splits: np.ndarray) -> dict[str, int]:
    return {
        split: int(np.sum(mask & (splits == split)))
        for split in sorted(set(splits.tolist()))
    }


def main() -> None:
    checkpoint = _read_json(CHECKPOINT_PATH)
    manifest = _read_json(MANIFEST_PATH)
    target = _read_json(TARGET_PATH)
    if checkpoint.get("dataset_id") != manifest.get("dataset_id"):
        raise RuntimeError("checkpoint and manifest dataset IDs differ")
    if checkpoint.get("root_manifest_hash") != _file_hash(MANIFEST_PATH):
        raise RuntimeError("root manifest hash differs from committed checkpoint")

    tensor_paths = _committed_tensor_paths(checkpoint)
    arrays: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    splits: list[np.ndarray] = []
    case_ids: list[np.ndarray] = []
    component_ids: list[str] | None = None
    aux_names: list[str] | None = None
    for path in tensor_paths:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "component_aux_y",
                "component_aux_label_mask",
                "component_aux_label_names",
                "component_ids",
                "split_names",
                "case_ids",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise RuntimeError(f"{path} missing required arrays: {missing}")
            current_component_ids = archive["component_ids"].astype(str).tolist()
            current_aux_names = archive["component_aux_label_names"].astype(str).tolist()
            if component_ids is None:
                component_ids = current_component_ids
                aux_names = current_aux_names
            elif component_ids != current_component_ids or aux_names != current_aux_names:
                raise RuntimeError("component or auxiliary vocabulary differs across shards")
            arrays.append(np.asarray(archive["component_aux_y"], dtype=np.float64))
            masks.append(np.asarray(archive["component_aux_label_mask"], dtype=bool))
            splits.append(archive["split_names"].astype(str))
            case_ids.append(archive["case_ids"].astype(str))

    assert component_ids is not None and aux_names is not None
    values = np.concatenate(arrays, axis=0)
    available = np.concatenate(masks, axis=0)
    split_values = np.concatenate(splits, axis=0)
    case_values = np.concatenate(case_ids, axis=0)
    if len(set(case_values.tolist())) != len(case_values):
        raise RuntimeError("duplicate case IDs across committed shards")
    if not np.isfinite(values[available]).all():
        raise RuntimeError("non-finite available component auxiliary value")

    try:
        energy_index = aux_names.index("fragment_deposited_energy_j")
    except ValueError as error:
        raise RuntimeError("fragment_deposited_energy_j is absent") from error
    component_positions = {name: index for index, name in enumerate(component_ids)}
    target_components = {str(item["id"]): item for item in target["components"]}
    requested_components = {
        component_id
        for category in BRANCHES.values()
        for branch in category.values()
        for component_id in branch
    }
    missing_components = sorted(requested_components - set(component_positions))
    if missing_components:
        raise RuntimeError(f"KMFC-v0 component IDs absent from archive: {missing_components}")

    energy = values[:, :, energy_index]
    energy_available = available[:, :, energy_index]
    positive = energy_available & (energy > ZERO_TOLERANCE_J)

    component_support: dict[str, Any] = {}
    for component_id in sorted(requested_components):
        index = component_positions[component_id]
        component_mask = positive[:, index]
        positive_values = energy[component_mask, index]
        component_support[component_id] = {
            "name": str(target_components[component_id].get("name", "")),
            "positive_case_count": int(np.sum(component_mask)),
            "positive_case_count_by_split": _count_by_split(
                component_mask, split_values
            ),
            "positive_energy_distinct_value_count": int(
                np.unique(positive_values).size
            ),
            "positive_energy_minimum_j": (
                float(np.min(positive_values)) if positive_values.size else None
            ),
            "positive_energy_maximum_j": (
                float(np.max(positive_values)) if positive_values.size else None
            ),
        }

    category_support: dict[str, Any] = {}
    category_positive_masks: dict[str, np.ndarray] = {}
    for category, branches in BRANCHES.items():
        branch_masks: dict[str, np.ndarray] = {}
        for branch, ids in branches.items():
            indices = [component_positions[component_id] for component_id in ids]
            branch_masks[branch] = np.any(positive[:, indices], axis=1)
        active_matrix = np.column_stack(list(branch_masks.values()))
        active_count = np.sum(active_matrix, axis=1)
        any_positive = active_count > 0
        category_positive_masks[category] = any_positive
        category_support[category] = {
            "branch_component_ids": branches,
            "branch_positive_case_counts": {
                branch: int(np.sum(mask)) for branch, mask in branch_masks.items()
            },
            "any_positive_case_count": int(np.sum(any_positive)),
            "any_positive_case_count_by_split": _count_by_split(
                any_positive, split_values
            ),
            "active_branch_count_distribution": {
                str(count): int(np.sum(active_count == count))
                for count in range(active_matrix.shape[1] + 1)
                if np.any(active_count == count)
            },
            "exactly_one_active_branch_count": int(np.sum(active_count == 1)),
            "at_least_two_active_branches_count": int(np.sum(active_count >= 2)),
        }

    k_mask = category_positive_masks["K"]
    f_mask = category_positive_masks["F"]
    audit = {
        "schema_version": "1.0.0",
        "audit_date": "2026-10-15",
        "scope": "read_only_observed_archive_component_support_diagnostic",
        "audit_script": {
            "path": _relative(Path(__file__).resolve()),
            "sha256": _file_hash(Path(__file__).resolve()),
        },
        "sources": {
            "dataset_id": manifest["dataset_id"],
            "case_count": int(case_values.size),
            "checkpoint_status": checkpoint["status"],
            "strict_committed_shard_inventory_validated": True,
            "tensor_paths": [_relative(path) for path in tensor_paths],
            "target_path": _relative(TARGET_PATH),
            "target_sha256": _file_hash(TARGET_PATH),
            "metric": "fragment_deposited_energy_j",
            "metric_level": "archived_case_level_replicate_mean",
            "positive_rule": "available and value > 0 J",
        },
        "kmfc_v0_branches": BRANCHES,
        "component_support": component_support,
        "category_support": category_support,
        "cross_category_support": {
            "K_and_F_positive_count": int(np.sum(k_mask & f_mask)),
            "K_positive_F_zero_count": int(np.sum(k_mask & ~f_mask)),
            "F_positive_K_zero_count": int(np.sum(f_mask & ~k_mask)),
            "K_and_F_positive_case_sets_identical": bool(np.array_equal(k_mask, f_mask)),
        },
        "conclusion": {
            "M_structure_prototyping": "supported_for_independent_calibration",
            "K_fixed_threshold_calibration_from_current_archive": "rejected_sparse_single_branch_support",
            "F_fixed_threshold_calibration_from_current_archive": "rejected_sparse_single_branch_support",
            "C_fixed_threshold_calibration_from_current_archive": "rejected_structural_zero",
        },
        "evidence_boundary": (
            "This audit reads already-observed Train, Validation, and Test rows and therefore is "
            "diagnostic only. It measures positive support in archived case-level mean deposited "
            "energy, not replicate-level continuous dose, calibrated severity, physical validity, "
            "or admissible final thresholds. Its counts may justify a new calibration design but "
            "must not be used to fit the formal tree or evaluate a model."
        ),
    }
    payload = json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    OUTPUT_PATH.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
