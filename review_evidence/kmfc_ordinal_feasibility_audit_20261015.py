"""Read-only feasibility audit for direct KMFC ordinalization.

The script reads the frozen 5,000-case archive, the target definition, and the
label specification.  It does not execute the simulator or mutate the formal
dataset.  Its only write is the versioned JSON report beside this script.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "raw_data/dataset_5000_loitering_impact_fragment_only_v1"
SHARD_ROOT = DATASET_ROOT / "formal_dataset/shards"
MANIFEST_PATH = DATASET_ROOT / "formal_dataset/manifest.json"
CHECKPOINT_PATH = DATASET_ROOT / "formal_dataset/checkpoint.json"
FROZEN_HASHES_PATH = DATASET_ROOT / "plan/frozen_hashes.json"
TARGET_PATH = REPO_ROOT / "raw_data/target.rebuilt.fragment_only.v1.json"
LABEL_SPEC_PATH = REPO_ROOT / "raw_data/label_spec.loitering_impact.fragment_only.v1.json"
OUTPUT_PATH = REPO_ROOT / "review_evidence/kmfc_ordinal_feasibility_audit_20261015.json"

STRICT_TOLERANCE = 1.0e-12
LABEL_CONTRACT_TOLERANCE = 1.0e-6
ZERO_TOLERANCE = 1.0e-12


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
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


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _leaf_events(output_id: str, nodes: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    events: set[tuple[str, str, str]] = set()
    active: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in active:
            raise RuntimeError(f"cycle encountered while tracing {output_id}: {node_id}")
        try:
            node = nodes[node_id]
        except KeyError as exc:
            raise RuntimeError(f"missing tree node referenced by {output_id}: {node_id}") from exc
        if node.get("node_type") == "leaf":
            event = (
                str(node.get("component_id") or ""),
                str(node.get("mechanism") or ""),
                str(node.get("damage_state") or ""),
            )
            if not all(event):
                raise RuntimeError(f"incomplete leaf mapping: {node_id}")
            events.add(event)
            return
        dependencies = node.get("inputs")
        if not isinstance(dependencies, list) or not dependencies:
            raise RuntimeError(f"non-leaf node has no dependencies: {node_id}")
        active.add(node_id)
        for dependency in dependencies:
            visit(str(dependency))
        active.remove(node_id)

    visit(output_id)
    return [
        {"component_id": component, "mechanism": mechanism, "damage_state": state}
        for component, mechanism, state in sorted(events)
    ]


def _load_archive(
    checkpoint: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    if checkpoint.get("status") != "complete":
        raise RuntimeError("formal archive checkpoint is not complete")
    completed = checkpoint.get("completed_shards")
    if not isinstance(completed, list) or not completed:
        raise RuntimeError("formal archive checkpoint has no committed shards")
    indices = [item.get("index") for item in completed if isinstance(item, dict)]
    if len(indices) != len(completed) or len(set(indices)) != len(indices):
        raise RuntimeError("formal archive checkpoint has invalid or duplicate shard indices")

    expected_directories = {
        (DATASET_ROOT / "formal_dataset" / str(item["path"])).resolve()
        for item in completed
    }
    observed_directories = {path.resolve() for path in SHARD_ROOT.glob("part-*") if path.is_dir()}
    if observed_directories != expected_directories:
        raise RuntimeError(
            "formal shard directory inventory differs from the committed checkpoint; "
            f"missing={sorted(str(path) for path in expected_directories - observed_directories)}, "
            f"extra={sorted(str(path) for path in observed_directories - expected_directories)}"
        )

    shard_paths: list[Path] = []
    for item in sorted(completed, key=lambda value: int(value["index"])):
        shard_directory = DATASET_ROOT / "formal_dataset" / str(item["path"])
        manifest_path = shard_directory / "manifest.json"
        if _file_hash(manifest_path) != item.get("manifest_hash"):
            raise RuntimeError(f"committed shard manifest hash mismatch: {manifest_path}")
        shard_manifest = _read_json(manifest_path)
        tensors_partitions = [
            partition
            for partition in shard_manifest.get("partitions", [])
            if isinstance(partition, dict) and partition.get("name") == "tensors"
        ]
        if len(tensors_partitions) != 1:
            raise RuntimeError(f"shard manifest must contain one tensors partition: {manifest_path}")
        partition = tensors_partitions[0]
        tensor_path = shard_directory / str(partition.get("path_glob"))
        if _file_hash(tensor_path) != partition.get("content_hash"):
            raise RuntimeError(f"committed tensor hash mismatch: {tensor_path}")
        if int(partition.get("row_count", -1)) != len(item.get("case_ids", [])):
            raise RuntimeError(f"checkpoint/tensor row-count mismatch: {tensor_path}")
        shard_paths.append(tensor_path)

    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    splits: list[np.ndarray] = []
    case_ids: list[np.ndarray] = []
    expected_names: list[str] | None = None

    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as archive:
            required = {
                "system_y",
                "system_label_mask",
                "system_output_names",
                "split_names",
                "case_ids",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise RuntimeError(f"{shard_path} is missing fields: {missing}")
            names = archive["system_output_names"].astype(str).tolist()
            if expected_names is None:
                expected_names = names
            elif names != expected_names:
                raise RuntimeError("system output vocabulary differs across shards")
            shard_values = np.asarray(archive["system_y"], dtype=np.float64)
            shard_masks = np.asarray(archive["system_label_mask"])
            if shard_masks.dtype != np.dtype(bool) or shard_masks.shape != shard_values.shape:
                raise RuntimeError(f"invalid system label mask in {shard_path}")
            if shard_values.ndim != 2 or shard_values.shape[1] != len(names):
                raise RuntimeError(f"invalid system output tensor in {shard_path}")
            if not np.isfinite(shard_values[shard_masks]).all():
                raise RuntimeError(f"non-finite available system label in {shard_path}")
            values.append(shard_values)
            masks.append(shard_masks)
            splits.append(archive["split_names"].astype(str))
            case_ids.append(archive["case_ids"].astype(str))

    assert expected_names is not None
    all_values = np.concatenate(values, axis=0)
    all_masks = np.concatenate(masks, axis=0)
    all_splits = np.concatenate(splits, axis=0)
    all_case_ids = np.concatenate(case_ids, axis=0)
    if len(set(all_case_ids.tolist())) != len(all_case_ids):
        raise RuntimeError("case IDs are duplicated across committed shards")
    return (
        all_values,
        all_masks,
        all_splits,
        all_case_ids,
        expected_names,
        [path.relative_to(REPO_ROOT).as_posix() for path in shard_paths],
    )


def _support_record(
    output: str,
    values: np.ndarray,
    masks: np.ndarray,
    splits: np.ndarray,
    positions: dict[str, int],
) -> dict[str, Any]:
    index = positions[output]
    available = masks[:, index]
    selected = values[available, index]
    by_split: dict[str, Any] = {}
    for split in sorted(set(splits.tolist())):
        split_mask = available & (splits == split)
        split_values = values[split_mask, index]
        by_split[split] = {
            "available_count": int(split_values.size),
            "nonzero_count": int(np.sum(split_values > ZERO_TOLERANCE)),
            "exact_zero_count": int(np.sum(split_values == 0.0)),
            "soft_mass": float(np.sum(split_values, dtype=np.float64)),
        }
    return {
        "output": output,
        "available_count": int(selected.size),
        "nonzero_count": int(np.sum(selected > ZERO_TOLERANCE)),
        "exact_zero_count": int(np.sum(selected == 0.0)),
        "soft_mass": float(np.sum(selected, dtype=np.float64)),
        "minimum": float(np.min(selected)) if selected.size else None,
        "maximum": float(np.max(selected)) if selected.size else None,
        "by_split": by_split,
    }


def _structural_pair_record(
    level1_output: str,
    level2_output: str,
    leaves: dict[str, list[dict[str, str]]],
    component_state_order: dict[str, list[str]],
) -> dict[str, Any]:
    low_events = leaves[level1_output]
    high_events = leaves[level2_output]
    low_by_component_mechanism = {
        (event["component_id"], event["mechanism"]): event["damage_state"]
        for event in low_events
    }
    high_keys = {
        (event["component_id"], event["mechanism"])
        for event in high_events
    }
    low_keys = set(low_by_component_mechanism)
    covered = 0
    ordered = 0
    for event in high_events:
        key = (event["component_id"], event["mechanism"])
        low_state = low_by_component_mechanism.get(key)
        if low_state is None:
            continue
        covered += 1
        state_order = component_state_order[event["component_id"]]
        if state_order.index(low_state) <= state_order.index(event["damage_state"]):
            ordered += 1
    return {
        "level1_leaf_count": len(low_events),
        "level2_leaf_count": len(high_events),
        "component_mechanism_sets_equal": low_keys == high_keys,
        "level2_leaves_with_level1_component_mechanism": covered,
        "level2_leaves_with_ordered_level1_state": ordered,
        "structurally_nested_component_states": (
            bool(high_events)
            and low_keys == high_keys
            and covered == len(high_events)
            and ordered == len(high_events)
        ),
    }


def _empirical_pair_record(
    level1_output: str,
    level2_output: str,
    values: np.ndarray,
    masks: np.ndarray,
    splits: np.ndarray,
    positions: dict[str, int],
) -> dict[str, Any]:
    low_index = positions[level1_output]
    high_index = positions[level2_output]
    available = masks[:, low_index] & masks[:, high_index]
    low = values[available, low_index]
    high = values[available, high_index]
    selected_splits = splits[available]
    strict_violation = high > low + STRICT_TOLERANCE
    contract_violation = high > low + LABEL_CONTRACT_TOLERANCE
    by_split = {}
    for split in sorted(set(selected_splits.tolist())):
        rows = selected_splits == split
        by_split[split] = {
            "usable_count": int(np.sum(rows)),
            "strict_violation_count": int(np.sum(strict_violation[rows])),
            "label_contract_violation_count": int(np.sum(contract_violation[rows])),
        }
    violation_magnitudes = high - low
    return {
        "usable_count": int(low.size),
        "strict_tolerance": STRICT_TOLERANCE,
        "strict_violation_count": int(np.sum(strict_violation)),
        "label_contract_tolerance": LABEL_CONTRACT_TOLERANCE,
        "label_contract_violation_count": int(np.sum(contract_violation)),
        "maximum_level2_minus_level1": (
            float(np.max(violation_magnitudes)) if violation_magnitudes.size else None
        ),
        "level1_positive_level2_zero": int(
            np.sum((low > ZERO_TOLERANCE) & (high <= ZERO_TOLERANCE))
        ),
        "level2_positive_level1_zero": int(
            np.sum((high > ZERO_TOLERANCE) & (low <= ZERO_TOLERANCE))
        ),
        "both_positive": int(
            np.sum((low > ZERO_TOLERANCE) & (high > ZERO_TOLERANCE))
        ),
        "both_zero": int(
            np.sum((low <= ZERO_TOLERANCE) & (high <= ZERO_TOLERANCE))
        ),
        "by_split": by_split,
    }


def main() -> None:
    target = _read_json(TARGET_PATH)
    label_spec = _read_json(LABEL_SPEC_PATH)
    manifest = _read_json(MANIFEST_PATH)
    checkpoint = _read_json(CHECKPOINT_PATH)
    frozen_hashes = _read_json(FROZEN_HASHES_PATH)

    current_tree_hash = _content_hash(target["damage_tree"])
    current_label_hash = _content_hash(label_spec)
    archive_tree_hash = str(frozen_hashes["damage_tree_definition_hash"])
    archive_label_hash = str(frozen_hashes["label_spec_hash"])
    if current_tree_hash != archive_tree_hash:
        raise RuntimeError("current damage tree does not match the frozen archive")
    if current_label_hash != archive_label_hash:
        raise RuntimeError("current label specification does not match the frozen archive")

    if checkpoint.get("dataset_id") != manifest.get("dataset_id"):
        raise RuntimeError("formal checkpoint and root manifest dataset IDs differ")
    if checkpoint.get("root_manifest_hash") != _file_hash(MANIFEST_PATH):
        raise RuntimeError("formal checkpoint root manifest hash mismatch")

    values, masks, splits, case_ids, outputs, shard_paths = _load_archive(checkpoint)
    declared_outputs = list(target["damage_tree"]["outputs"])
    if outputs != declared_outputs:
        raise RuntimeError("archive system output order differs from the current frozen tree")
    positions = {name: index for index, name in enumerate(outputs)}
    nodes = {str(node["id"]): node for node in target["damage_tree"]["nodes"]}
    components = {str(item["id"]): item for item in target["components"]}
    component_state_order = {
        component_id: [str(state) for state in component["damage_states"] if state != "intact"]
        for component_id, component in components.items()
    }
    leaves = {output: _leaf_events(output, nodes) for output in outputs}

    pair_definitions = {
        "K": ("effect-k1", "effect-k2"),
        "M": ("effect-m1", "effect-m2"),
        "F": ("effect-f1", "effect-f2"),
        "C": ("effect-c1", "effect-c2"),
    }
    ordinal_pairs: list[dict[str, Any]] = []
    for category, (level1, level2) in pair_definitions.items():
        missing_outputs = [name for name in (level1, level2) if name not in positions]
        record: dict[str, Any] = {
            "category": category,
            "level1_output": level1,
            "level2_output": level2,
            "missing_outputs": missing_outputs,
        }
        if missing_outputs:
            record.update(
                {
                    "direct_ordinalization_status": "rejected_missing_output",
                    "reason": "the legacy seven-output vocabulary does not define both levels",
                }
            )
        else:
            structural = _structural_pair_record(
                level1, level2, leaves, component_state_order
            )
            empirical = _empirical_pair_record(
                level1, level2, values, masks, splits, positions
            )
            if not structural["structurally_nested_component_states"]:
                status = "rejected_structurally_distinct_outputs"
                reason = (
                    "legacy levels use different component/mechanism event sets; zero empirical "
                    "violations would therefore be vacuous rather than proof of ordered semantics"
                )
            elif empirical["label_contract_violation_count"]:
                status = "rejected_empirical_nonmonotonicity"
                reason = "legacy outputs violate the cumulative level2 <= level1 invariant"
            else:
                status = "structurally_compatible_semantics_unapproved"
                reason = (
                    "component-state nesting is compatible with an ordinal pair, but the label "
                    "specification remains provisional and not approved"
                )
            record.update(
                {
                    "direct_ordinalization_status": status,
                    "reason": reason,
                    "structural": structural,
                    "empirical": empirical,
                }
            )
        ordinal_pairs.append(record)

    required_eight = [
        "effect-k1",
        "effect-k2",
        "effect-m1",
        "effect-m2",
        "effect-f1",
        "effect-f2",
        "effect-c1",
        "effect-c2",
    ]
    audit = {
        "schema_version": "1.0.0",
        "audit_date": "2026-10-15",
        "scope": "frozen_archive_direct_ordinalization_feasibility_diagnostic",
        "audit_script": {
            "path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
            "sha256": _file_hash(Path(__file__).resolve()),
        },
        "sources": {
            "dataset_id": manifest["dataset_id"],
            "case_count": int(len(case_ids)),
            "shards": shard_paths,
            "target": {
                "path": TARGET_PATH.relative_to(REPO_ROOT).as_posix(),
                "file_sha256": _file_hash(TARGET_PATH),
                "damage_tree_definition_hash": current_tree_hash,
                "archive_frozen_damage_tree_definition_hash": archive_tree_hash,
                "matches_archive": current_tree_hash == archive_tree_hash,
            },
            "label_spec": {
                "path": LABEL_SPEC_PATH.relative_to(REPO_ROOT).as_posix(),
                "file_sha256": _file_hash(LABEL_SPEC_PATH),
                "label_spec_hash": current_label_hash,
                "archive_frozen_label_spec_hash": archive_label_hash,
                "matches_archive": current_label_hash == archive_label_hash,
                "status": label_spec.get("status"),
                "approval_status": label_spec.get("provenance", {}).get("approval_status"),
            },
            "manifest": {
                "path": MANIFEST_PATH.relative_to(REPO_ROOT).as_posix(),
                "file_sha256": _file_hash(MANIFEST_PATH),
            },
            "checkpoint": {
                "path": CHECKPOINT_PATH.relative_to(REPO_ROOT).as_posix(),
                "file_sha256": _file_hash(CHECKPOINT_PATH),
                "status": checkpoint["status"],
                "committed_shard_count": len(checkpoint["completed_shards"]),
                "strict_inventory_validated": True,
            },
        },
        "legacy_output_contract": {
            "outputs": outputs,
            "output_count": len(outputs),
            "required_four_category_two_level_outputs": required_eight,
            "required_output_count": len(required_eight),
            "missing_required_outputs": sorted(set(required_eight) - set(outputs)),
            "safe_in_place_relabel": False,
            "required_action": (
                "create a new damage-tree version and label-spec version, then regenerate a new "
                "archive; preserve this seven-output archive unchanged"
            ),
        },
        "output_support": [
            _support_record(output, values, masks, splits, positions)
            for output in outputs
        ],
        "leaf_events_by_output": leaves,
        "direct_ordinalization": ordinal_pairs,
        "conclusion": {
            "feasibility": "conditional_only",
            "existing_outputs_can_be_directly_reused_as_four_ordinal_pairs": False,
            "supported_software_claim": (
                "the archive can mechanically test an ordinal contract and shows that the legacy "
                "C pair is structurally compatible while M and K are distinct events and F2 is absent"
            ),
            "unsupported_scientific_claim": (
                "this audit does not define or validate KMFC semantics, component membership, "
                "weights, thresholds, activation functions, dependence, or real-world damage"
            ),
        },
        "evidence_boundary": (
            "Read-only diagnostic over an already-observed synthetic archive. It performs no new "
            "simulation and no physical calibration. Empirical zero violations for constant-zero "
            "pairs are vacuous; smooth gates must not manufacture nonzero labels from zero inputs."
        ),
    }
    payload = json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    OUTPUT_PATH.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
