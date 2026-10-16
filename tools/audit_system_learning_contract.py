"""Build a Train-selected neural-learning contract from a sharded archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.system_learning import (
    build_system_learning_contract,
    system_learning_archive_binding,
)
from damage_engine.normalization import normalize_target
from damage_engine.torch_dataset import validated_committed_shard_tensor_paths
from damage_engine.utils import content_hash, file_hash


def _load_arrays(dataset: Path) -> tuple[dict, dict[str, np.ndarray]]:
    # Reuse the production loader's completed-checkpoint, exact-inventory,
    # orphan/incomplete-marker, partition-hash and tensor-axis validation.  A
    # learning contract must never be generated from an independently
    # rediscovered or partially verified set of shards.
    manifest, tensor_paths = validated_committed_shard_tensor_paths(dataset)
    fields = (
        "system_y",
        "system_label_mask",
        "split_names",
        "sampling_role_index",
        "sampling_design_weight_raw",
        "sampling_weight_mask",
    )
    chunks = {field: [] for field in fields}
    role_names = None
    task_names = None
    for tensor_path in tensor_paths:
        with np.load(tensor_path, allow_pickle=False) as archive:
            for field in fields:
                if field not in archive.files:
                    raise RuntimeError(
                        f"committed shard is missing {field}: {tensor_path.parent.name}"
                    )
                chunks[field].append(np.asarray(archive[field]))
            current_roles = tuple(map(str, archive["sampling_role_names"].tolist()))
            current_tasks = tuple(map(str, archive["system_output_names"].tolist()))
            if role_names is None:
                role_names = current_roles
                task_names = current_tasks
            elif role_names != current_roles or task_names != current_tasks:
                raise RuntimeError("shard semantic vocabularies are inconsistent")
    if not chunks["system_y"]:
        raise RuntimeError("checkpoint contains no completed shards")
    arrays = {field: np.concatenate(parts, axis=0) for field, parts in chunks.items()}
    arrays["sampling_roles"] = np.asarray(
        [role_names[int(index)] for index in arrays["sampling_role_index"]]
    )
    arrays["system_output_names"] = np.asarray(task_names)
    return manifest, arrays


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--target",
        required=True,
        help="exact target JSON used by the archive; unresolved tree rows become blockers",
    )
    parser.add_argument(
        "--test-status",
        required=True,
        choices=("sealed_unobserved", "observed_diagnostic_only"),
    )
    arguments = parser.parse_args(argv)
    dataset = Path(arguments.dataset).resolve()
    manifest, arrays = _load_arrays(dataset)
    target_path = Path(arguments.target).resolve()
    target = json.loads(target_path.read_text(encoding="utf-8"))
    target_hash = content_hash(normalize_target(target))
    target_hashes = manifest.get("source_contracts", {}).get("target_hashes", ())
    if target_hash not in target_hashes:
        raise RuntimeError("target JSON is not bound to the dataset manifest")
    blockers: dict[str, list[dict]] = {}
    for row in target.get("metadata", {}).get("raw_damage_tree_records", ()):
        if row.get("apply_to_provisional_tree") is False or row.get("resolved_damage_state") is None:
            damage_type = str(row.get("damage_type", "")).strip().lower()
            task_name = f"effect-{damage_type}" if damage_type else ""
            if task_name:
                blockers.setdefault(task_name, []).append(
                    {
                        "component_id": row.get("component_id"),
                        "component_name": row.get("component_name"),
                        "source_id": row.get("source_id"),
                        "reason": "unresolved_damage_threshold_not_in_provisional_tree",
                    }
                )
    contract = build_system_learning_contract(
        arrays["system_y"],
        arrays["system_label_mask"],
        arrays["system_output_names"],
        split_names=arrays["split_names"],
        sampling_roles=arrays["sampling_roles"],
        dataset_id=str(manifest["dataset_id"]),
        dataset_manifest_hash=file_hash(dataset / "manifest.json"),
        test_status=arguments.test_status,
        design_weights=arrays["sampling_design_weight_raw"],
        design_weight_mask=arrays["sampling_weight_mask"],
        source_contract_hashes=system_learning_archive_binding(
            manifest, file_hash(dataset / "manifest.json")
        )["source_contract_hashes"],
        task_definition_blockers=blockers,
    )
    output = Path(arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "default_task_names": contract["selection_policy"]["default_task_names"],
        "exploratory_task_names": contract["selection_policy"]["exploratory_task_names"],
        "excluded_task_names": contract["selection_policy"]["excluded_task_names"],
        "blocked_task_names": contract["selection_policy"]["blocked_task_names"],
        "requires_new_sealed_reference_test": contract["test_policy"]["requires_new_sealed_reference_test"],
    }, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
