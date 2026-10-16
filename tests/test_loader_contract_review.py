"""Artificial software archives: no physics, engine execution or model training."""
from __future__ import annotations

from copy import deepcopy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from damage_engine.archive_validation import semantic_vocabulary
from damage_engine.errors import InputValidationError
from damage_engine.quality import synthetic_label_admission
from damage_engine.torch_dataset import (
    TorchDamageDataset, TorchMultiTargetDataset, TorchShardedDamageDataset,
    collate_damage_batch, make_damage_collator,
)
from damage_engine.utils import content_hash, file_hash


FIELDS = ("case_x", "system_y", "system_label_mask", "requested_standard_mask",
          "system_uncertainty_mask", "system_sample_count")


def toy_arrays(label="label-a", prefix="toy", count=2):
    return {
        "case_x": np.arange(count, dtype=np.float32).reshape(count, 1),
        "case_ids": np.asarray([f"{prefix}-{index}" for index in range(count)]),
        "case_hashes": np.asarray([f"hash-{prefix}-{index}" for index in range(count)]),
        "split_names": np.asarray(["train"] + ["test"] * (count - 1)),
        "split_group_ids": np.asarray([f"{prefix}-group-{index}" for index in range(count)]),
        "leakage_keys": np.asarray([f"{prefix}-key-{index}" for index in range(count)]),
        "case_feature_names": np.asarray(["toy-feature"]),
        "case_feature_units": np.asarray(["1"]),
        "component_ids": np.asarray(["toy-component"]),
        "armor_ids": np.asarray([], dtype=str),
        "mechanism_names": np.asarray(["mechanism-a"]),
        "state_names": np.asarray(["state-a"]),
        "system_output_names": np.asarray([label]),
        "evaluation_standard_ids": np.asarray(["standard-" + label]),
        "system_y": np.full((count, 1), 0.25, dtype=np.float32),
        "system_y_stddev": np.full((count, 1), 0.1, dtype=np.float32),
        "system_y_ci95": np.tile(np.asarray([0.1, 0.4], dtype=np.float32), (count, 1, 1)),
        "system_label_mask": np.ones((count, 1), dtype=bool),
        "system_uncertainty_mask": np.ones((count, 1), dtype=bool),
        "system_sample_count": np.full((count, 1), 2, dtype=np.int64),
        "requested_standard_mask": np.ones((count, 1), dtype=bool),
        "standard_output_mask": np.ones((1, 1), dtype=bool),
        "replicate_counts": np.full(count, 2, dtype=np.int64),
        "graph_node_ids": np.asarray(["toy-component", "leaf", "output"]),
        "graph_node_types": np.asarray(["component", "damage_tree", "damage_tree"]),
        "tree_edge_index": np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        "tree_edge_type": np.asarray([0, 1], dtype=np.int16),
        "component_vertices_m": np.zeros((0, 3), dtype=np.float32),
        "component_vertex_offsets": np.asarray([0, 0], dtype=np.int64),
    }


def write_toy(directory: Path, arrays=None, *, label="label-a", prefix="toy"):
    arrays = toy_arrays(label, prefix) if arrays is None else arrays
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "tensors.npz", **arrays)
    named_shapes = {
        "case_x": "[B,F_case]", "case_ids": "[B]", "case_hashes": "[B]", "split_names": "[B]",
        "split_group_ids": "[B]", "leakage_keys": "[B]", "case_feature_names": "[F_case]", "case_feature_units": "[F_case]",
        "component_ids": "[C]", "armor_ids": "[A]", "mechanism_names": "[M]", "state_names": "[S]",
        "system_output_names": "[O]", "evaluation_standard_ids": "[E_standard]",
        "system_y": "[B,O]", "system_y_stddev": "[B,O]", "system_y_ci95": "[B,O,2]",
        "system_label_mask": "[B,O]", "system_uncertainty_mask": "[B,O]", "system_sample_count": "[B,O]",
        "requested_standard_mask": "[B,E_standard]", "standard_output_mask": "[E_standard,O]",
        "replicate_counts": "[B]", "graph_node_ids": "[G]", "graph_node_types": "[G]",
        "tree_edge_index": "[2,E_edge]", "tree_edge_type": "[E_edge]",
        "component_vertices_m": "[Vc,3]", "component_vertex_offsets": "[C+1]",
    }
    manifest = {
        "dataset_id": "synthetic-" + prefix,
        "source_contracts": {"target_hashes": ["synthetic-target"]},
        "label_admission": synthetic_label_admission(),
        "split_policy": {"leakage_audit_passed": True},
        "quality_summary": {"n_valid": len(arrays["case_x"])},
        "partitions": [{"name": "tensors", "format": "npz", "path_glob": "tensors.npz", "row_count": len(arrays["case_x"]), "content_hash": file_hash(directory / "tensors.npz")}],
        "tensor_contract": {"version": "1.4.0", "fields": [
            {"name": name, "dtype": "string" if value.dtype.kind in "US" else str(value.dtype),
             "shape": named_shapes.get(name, "[" + ",".join(map(str, value.shape)) + "]")}
            for name, value in arrays.items()
        ], "axes": [{"axis_id": "F_case", "size": 1, "names_field": "case_feature_names", "units_field": "case_feature_units"}]},
        "semantic_vocabulary": semantic_vocabulary({name: tuple(map(str, arrays[name])) for name in ("mechanism_names", "state_names", "system_output_names", "evaluation_standard_ids")}),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def write_sharded(root: Path, mutate=None):
    records, partitions = [], []
    first = None
    for index in range(2):
        shard = root / "shards" / f"part-{index:06d}"
        arrays = toy_arrays(prefix=f"shard-{index}")
        if mutate is not None and index == 0:
            mutate(arrays)
        manifest = write_toy(shard, arrays, prefix=f"shard-{index}")
        first = first or manifest
        records.append({"index": index, "path": shard.relative_to(root).as_posix(),
                        "dataset_id": manifest["dataset_id"], "manifest_hash": file_hash(shard / "manifest.json"),
                        "case_ids": arrays["case_ids"].astype(str).tolist(),
                        "case_hashes": arrays["case_hashes"].astype(str).tolist()})
        for partition in manifest["partitions"]:
            partitions.append({**partition, "path_glob": (shard.relative_to(root) / partition["path_glob"]).as_posix()})
    root_manifest = {**first, "dataset_id": "synthetic-sharded", "partitions": partitions,
                     "quality_summary": {"n_valid": 4}}
    (root / "manifest.json").write_text(json.dumps(root_manifest), encoding="utf-8")
    identity = {"shard_size": 2}
    identity["identity_hash"] = content_hash(identity)
    checkpoint = {"schema_version": "2.0.0", "identity": identity,
                  "dataset_id": root_manifest["dataset_id"], "status": "complete", "completed_shards": records,
                  "root_manifest_hash": file_hash(root / "manifest.json")}
    (root / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")


def load(root, **kwargs):
    return TorchDamageDataset(root, fields=FIELDS, **kwargs)


def test_manifest_missing_requires_explicit_forensic_mode(tmp_path):
    write_toy(tmp_path)
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="manifest"):
        load(tmp_path)
    dataset = load(tmp_path, allow_legacy_manifest=True)
    assert "allow_legacy_manifest=True:missing_manifest" in dataset.audit_overrides


def test_hash_and_leakage_fail_closed_with_explicit_overrides(tmp_path):
    manifest = write_toy(tmp_path)
    manifest["split_policy"]["leakage_audit_passed"] = False
    manifest["partitions"][0]["content_hash"] = "sha256:wrong"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="split-leakage"):
        load(tmp_path)
    with pytest.raises(RuntimeError, match="partition"):
        load(tmp_path, allow_split_leakage=True)
    dataset = load(tmp_path, allow_split_leakage=True, verify_integrity=False)
    assert set(dataset.audit_overrides) == {"allow_split_leakage=True", "verify_integrity=False"}


def test_existing_instances_never_ignore_split_or_options(tmp_path):
    write_toy(tmp_path)
    unfiltered = load(tmp_path)
    with pytest.raises(ValueError, match="split differs"):
        TorchMultiTargetDataset([unfiltered], split="train")
    with pytest.raises(ValueError, match="split must"):
        TorchMultiTargetDataset([unfiltered], split="holdout")
    with pytest.raises(ValueError, match="copy_on_read"):
        TorchMultiTargetDataset([unfiltered], copy_on_read=False)
    with pytest.raises(ValueError, match="fields"):
        TorchMultiTargetDataset([unfiltered], fields=("case_x",))
    filtered = load(tmp_path, split="train")
    for dataset in (TorchMultiTargetDataset([filtered], split="train"), TorchMultiTargetDataset([tmp_path], split="train", fields=FIELDS)):
        assert [dataset[index]["split"] for index in range(len(dataset))] == ["train"]


def test_frozen_vocabulary_is_stable_across_subsets_shuffle_and_source_order(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    write_toy(first, label="label-a", prefix="a")
    write_toy(second, label="label-b", prefix="b")
    dataset = TorchMultiTargetDataset([first, second], fields=FIELDS)
    reverse = TorchMultiTargetDataset([second, first], fields=FIELDS)
    assert dataset.semantic_vocabulary == reverse.semantic_vocabulary
    for samples in ([dataset[0]], [dataset[2]], [dataset[0], dataset[2]], [dataset[2], dataset[0]]):
        batch = collate_damage_batch(samples)
        assert batch["system_output_names"] == ("label-a", "label-b")
        assert batch["evaluation_standard_ids"] == ("standard-label-a", "standard-label-b")
        assert batch["system_y"].shape == (len(samples), 2)
        assert batch["semantic_vocabulary_hash"] == dataset.semantic_vocabulary_hash
    assert dataset.collate_fn([dataset[0]])["system_label_mask"].tolist() == [[True, False]]
    path = dataset.save_semantic_vocabulary(tmp_path / "vocabulary.json")
    assert json.loads(path.read_text(encoding="utf-8")) == dataset.semantic_vocabulary
    collator = pickle.loads(pickle.dumps(make_damage_collator(dataset.semantic_vocabulary)))
    assert collator([dataset[0]])["system_y"].shape == (1, 2)
    unknown = deepcopy(dataset[0])
    unknown["_topology"]["system_output_names"] = ("unknown",)
    with pytest.raises(ValueError, match="unknown"):
        dataset.collate_fn([unknown])
    with pytest.raises(ValueError, match="frozen dataset vocabulary"):
        collate_damage_batch([{key: value for key, value in dataset[0].items() if key != "_semantic_vocabulary"}])


def test_default_copy_prevents_in_place_cache_pollution_in_samples_and_graph(tmp_path):
    write_toy(tmp_path)
    dataset = load(tmp_path)
    original = dataset[0]
    original["case_x"].add_(7)
    original["tree_edge_index"].add_(7)
    for _ in range(3):
        assert dataset[0]["case_x"].tolist() == [0.0]
        assert dataset[0]["tree_edge_index"].tolist() == [[0, 1], [1, 2]]
    shared = load(tmp_path, copy_on_read=False)
    shared[0]["case_x"].add_(7)
    assert shared[0]["case_x"].tolist() == [7.0]


@pytest.mark.parametrize("mutation, message", [
    (lambda arrays: arrays.update(case_feature_units=np.asarray(["1", "extra"])), "units"),
    (lambda arrays: arrays.update(case_ids=np.asarray(["duplicate", "duplicate"])), "unique"),
    (lambda arrays: arrays.update(system_label_mask=np.ones((2, 1), dtype=np.int64)), "bool"),
    (lambda arrays: arrays["tree_edge_index"].__setitem__((0, 0), 99), "outside"),
    (lambda arrays: arrays["tree_edge_type"].__setitem__(0, 1), "endpoints"),
    (lambda arrays: arrays["component_vertex_offsets"].__setitem__(1, 1), "offsets"),
    (lambda arrays: arrays["system_y_ci95"].__setitem__((0, 0), [0.4, 0.1]), "reversed"),
    (lambda arrays: arrays["system_y"].__setitem__((0, 0), np.nan), "nonfinite"),
    (lambda arrays: arrays["system_uncertainty_mask"].__setitem__((0, 0), False), "silently disabled"),
    (lambda arrays: arrays["requested_standard_mask"].__setitem__((0, 0), False), "requested standard"),
])
def test_corrupt_semantics_rejected_even_with_updated_file_hash(tmp_path, mutation, message):
    arrays = toy_arrays()
    mutation(arrays)
    write_toy(tmp_path, arrays)
    with pytest.raises(ValueError, match=message):
        load(tmp_path)


def test_dtype_contract_and_vocabulary_digest_rejected(tmp_path):
    manifest = write_toy(tmp_path)
    manifest["tensor_contract"]["fields"][0]["dtype"] = "float64"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dtype"):
        load(tmp_path)
    manifest = write_toy(tmp_path)
    manifest["semantic_vocabulary"]["content_hash"] = "sha256:wrong"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="content_hash"):
        load(tmp_path)


def test_single_observation_uncertainty_remains_unestimable(tmp_path):
    arrays = toy_arrays()
    arrays["replicate_counts"][0] = 1
    arrays["system_sample_count"][0] = 1
    arrays["system_uncertainty_mask"][0] = False
    arrays["system_y_stddev"][0] = np.nan
    arrays["system_y_ci95"][0] = np.nan
    write_toy(tmp_path, arrays)
    assert load(tmp_path)[0]["system_uncertainty_mask"].tolist() == [False]
    arrays["system_y_stddev"][0] = 0
    write_toy(tmp_path, arrays)
    with pytest.raises(ValueError, match="NaN"):
        load(tmp_path)


def test_sharded_admission_split_instances_and_uncommitted_directories(tmp_path):
    write_sharded(tmp_path)
    source = TorchShardedDamageDataset(tmp_path, fields=FIELDS, split="train")
    assert len(source) == 2
    assert len(source._cache) == 0
    wrapper = TorchMultiTargetDataset([source], split="train")
    assert [wrapper[index]["split"] for index in range(len(wrapper))] == ["train", "train"]
    path_wrapper = TorchMultiTargetDataset([tmp_path], split="train", fields=FIELDS)
    assert [path_wrapper[index]["split"] for index in range(len(path_wrapper))] == ["train", "train"]
    with pytest.raises(ValueError, match="split differs"):
        TorchMultiTargetDataset([source], split="test")
    restored = pickle.loads(pickle.dumps(source))
    assert restored[1]["case_id"] == source[1]["case_id"]
    (tmp_path / "shards" / "part-999999").mkdir()
    with pytest.raises(RuntimeError, match="inventory"):
        TorchShardedDamageDataset(tmp_path, fields=FIELDS)


@pytest.mark.parametrize("storage", ["monolithic", "sharded"])
def test_storage_formats_share_fail_closed_graph_and_label_admission(tmp_path, storage):
    mutate = lambda arrays: arrays["tree_edge_index"].__setitem__((1, 0), 99)
    if storage == "monolithic":
        arrays = toy_arrays()
        mutate(arrays)
        write_toy(tmp_path, arrays)
        constructor = load
    else:
        write_sharded(tmp_path, mutate=mutate)
        constructor = lambda path: TorchShardedDamageDataset(path, fields=FIELDS)
    with pytest.raises(ValueError, match="outside"):
        constructor(tmp_path)


def test_worker_reopening_rejects_changed_manifest(tmp_path):
    write_toy(tmp_path)
    dataset = load(tmp_path)
    serialized = pickle.dumps(dataset)
    path = tmp_path / "manifest.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        pickle.loads(serialized)


def test_sharded_copy_and_spawn_preserve_artificial_samples(tmp_path):
    import torch
    write_sharded(tmp_path)
    dataset = TorchShardedDamageDataset(tmp_path, fields=FIELDS)
    dataset[0]["case_x"].add_(7)
    assert dataset[0]["case_x"].tolist() == [0.0]
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=2, multiprocessing_context="spawn",
    )
    assert [item for batch in loader for item in batch["case_id"]] == ["shard-0-0", "shard-0-1", "shard-1-0", "shard-1-1"]


def test_explicit_forensic_sharded_snapshot_reopens_without_reenabling_hash_checks(tmp_path):
    write_sharded(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["root_manifest_hash"] = "sha256:deliberately-old-forensic-hash"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    source = TorchShardedDamageDataset(tmp_path, fields=FIELDS, verify_integrity=False)
    restored = pickle.loads(pickle.dumps(source))
    assert restored[0]["case_id"] == "shard-0-0"
    assert "verify_integrity=False" in restored.audit_overrides


def test_research_purpose_is_not_inferred_from_integrity(tmp_path):
    write_toy(tmp_path)
    with pytest.raises(InputValidationError, match="does not approve purpose"):
        load(tmp_path, purpose="research_training")


@pytest.mark.parametrize("override", [
    {"allow_legacy_manifest": True}, {"verify_integrity": False}, {"allow_split_leakage": True},
])
def test_research_loading_rejects_all_forensic_overrides(tmp_path, override):
    write_toy(tmp_path)
    if override.get("allow_legacy_manifest"):
        (tmp_path / "manifest.json").unlink()
    with pytest.raises(ValueError, match="forensic"):
        load(tmp_path, purpose="research_training", **override)


def test_names_are_not_coerced_and_probabilities_are_bounded(tmp_path):
    with pytest.raises(ValueError, match="without coercion"):
        semantic_vocabulary({"state_names": [1]})
    arrays = toy_arrays()
    arrays["component_ids"] = np.asarray([123], dtype=np.int64)
    write_toy(tmp_path, arrays)
    with pytest.raises(ValueError, match="strings without coercion"):
        load(tmp_path)
    arrays = toy_arrays()
    arrays["system_y"][0, 0] = 1.25
    write_toy(tmp_path, arrays)
    with pytest.raises(ValueError, match="within"):
        load(tmp_path)


def test_real_dataloader_spawn_shuffle_final_batch_and_persistent_workers(tmp_path):
    import torch
    first, second = tmp_path / "first", tmp_path / "second"
    write_toy(first, label="label-a", prefix="a")
    write_toy(second, label="label-b", prefix="b")
    dataset = TorchMultiTargetDataset([first, second], fields=FIELDS)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=3, shuffle=True, num_workers=2,
        multiprocessing_context="spawn", persistent_workers=True,
        collate_fn=dataset.collate_fn, generator=torch.Generator().manual_seed(7),
    )
    try:
        for _ in range(2):
            seen = []
            for batch in loader:
                assert batch["system_y"].shape[1] == 2
                assert batch["semantic_vocabulary_hash"] == dataset.semantic_vocabulary_hash
                seen.extend(batch["case_id"])
                batch["case_x"].add_(100)
            assert set(seen) == {"a-0", "a-1", "b-0", "b-1"}
        assert dataset[0]["case_x"].tolist() == [0.0]
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
