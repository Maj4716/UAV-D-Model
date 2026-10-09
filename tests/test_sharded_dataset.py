from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys
import types

import numpy as np
import pytest
from jsonschema import Draft202012Validator

import damage_engine.dataset as dataset_module
from damage_engine.dataset import DatasetWriter, dataset_identity_hash
from damage_engine.engine import DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.schema_resources import load_schema
from damage_engine.sharded_dataset import ShardedDatasetWriter


def _cases(engine, case_data, count=5):
    output = []
    for index in range(count):
        case = deepcopy(case_data)
        case["case_id"] = f"sharded-case-{index}"
        case["target_ref"]["content_hash"] = engine.target_hash
        case["analysis"]["aleatory_seed"] = 9000 + index
        case["analysis"]["replicates"] = 1
        case["analysis"]["adaptive_ci_half_width"] = None
        case["metadata"]["split_group"] = f"condition-{index // 2}"
        output.append(case)
    return output


def _concatenate_shards(root: Path, field: str) -> np.ndarray:
    values = []
    for shard in sorted((root / "shards").glob("part-*")):
        with np.load(shard / "tensors.npz", allow_pickle=False) as arrays:
            values.append(np.asarray(arrays[field]))
    return np.concatenate(values, axis=0)


def test_sharded_generation_is_bounded_and_matches_monolithic_output(
    tmp_path, target_data, case_data, registry
):
    engine = DamageEngine(target_data, registry)
    cases = _cases(engine, case_data)
    monolithic = tmp_path / "monolithic"
    sharded = tmp_path / "sharded"
    monolithic_manifest = DatasetWriter(monolithic).run_cases(engine, cases)
    manifest = ShardedDatasetWriter(sharded, shard_size=2).run_cases(engine, iter(cases))

    Draft202012Validator(load_schema("dataset_manifest.schema.json")).validate(manifest)
    assert manifest["quality_summary"]["n_total"] == 5
    assert manifest["quality_summary"]["n_valid"] == 5
    assert len(list((sharded / "shards").glob("part-*"))) == 3
    assert not (sharded / ".dataset.incomplete").exists()
    checkpoint = json.loads((sharded / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "complete"
    assert [len(item["case_ids"]) for item in checkpoint["completed_shards"]] == [2, 2, 1]
    np.testing.assert_allclose(
        manifest["quality_summary"]["mc_ci_half_width_quantiles"],
        monolithic_manifest["quality_summary"]["mc_ci_half_width_quantiles"],
    )
    child_manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((sharded / "shards").glob("part-*/manifest.json"))
    ]
    for partition in manifest["partitions"]:
        assert len(list(sharded.glob(partition["path_glob"]))) == 3
        assert partition["row_count"] == sum(
            next(
                item["row_count"]
                for item in child["partitions"]
                if item["name"] == partition["name"]
                and item["format"] == partition["format"]
            )
            for child in child_manifests
        )

    with np.load(monolithic / "tensors.npz", allow_pickle=False) as expected:
        for field in ("case_ids", "case_x", "component_y", "system_y"):
            np.testing.assert_array_equal(
                _concatenate_shards(sharded, field), expected[field], err_msg=field
            )


def test_sharded_resume_reuses_committed_shards_and_rejects_changed_input(
    tmp_path, target_data, case_data, registry, monkeypatch
):
    engine = DamageEngine(target_data, registry)
    cases = _cases(engine, case_data)
    output = tmp_path / "resume"
    original = DatasetWriter.run_cases
    calls = {"count": 0}

    def interrupted(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("injected interruption")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DatasetWriter, "run_cases", interrupted)
    with pytest.raises(RuntimeError, match="injected interruption"):
        ShardedDatasetWriter(output, shard_size=2).run_cases(engine, iter(cases))
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert len(checkpoint["completed_shards"]) == 1
    first_manifest_hash = checkpoint["completed_shards"][0]["manifest_hash"]

    monkeypatch.setattr(DatasetWriter, "run_cases", original)
    manifest = ShardedDatasetWriter(output, shard_size=2).run_cases(
        engine, iter(cases), resume=True
    )
    assert manifest["quality_summary"]["n_total"] == 5
    resumed = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert resumed["completed_shards"][0]["manifest_hash"] == first_manifest_hash
    assert resumed["status"] == "complete"

    changed = deepcopy(cases)
    changed[0]["environment"]["temperature_k"] += 1.0
    with pytest.raises(InputValidationError, match="do not match"):
        ShardedDatasetWriter(output, shard_size=2).run_cases(
            engine, iter(changed), resume=True
        )


def test_sharded_generation_rejects_cross_shard_duplicate_ids(
    tmp_path, target_data, case_data, registry
):
    engine = DamageEngine(target_data, registry)
    cases = _cases(engine, case_data, count=3)
    cases[2]["case_id"] = cases[0]["case_id"]
    with pytest.raises(InputValidationError, match="globally unique"):
        ShardedDatasetWriter(tmp_path / "duplicate", shard_size=1).run_cases(
            engine, iter(cases)
        )


def test_logical_dataset_id_is_independent_of_shard_layout(
    tmp_path, target_data, case_data, registry
):
    engine = DamageEngine(target_data, registry)
    cases = _cases(engine, case_data, count=3)
    one = ShardedDatasetWriter(
        tmp_path / "one", shard_size=1, tensor_compression="none"
    ).run_cases(engine, iter(cases))
    two = ShardedDatasetWriter(
        tmp_path / "two", shard_size=2, tensor_compression="compressed"
    ).run_cases(engine, iter(cases))
    monolithic = DatasetWriter(tmp_path / "monolithic").run_cases(
        engine, iter(cases)
    )
    assert one["dataset_id"] == two["dataset_id"]
    assert one["dataset_id"] == monolithic["dataset_id"]


def test_completed_resume_verifies_root_and_partition_hashes(
    tmp_path, target_data, case_data, registry
):
    engine = DamageEngine(target_data, registry)
    cases = _cases(engine, case_data, count=3)

    root_tamper = tmp_path / "root-tamper"
    ShardedDatasetWriter(root_tamper, shard_size=2).run_cases(engine, iter(cases))
    manifest_path = root_tamper / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_id"] = "forged"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(InputValidationError, match="root manifest"):
        ShardedDatasetWriter(root_tamper, shard_size=2).run_cases(
            engine, iter(cases), resume=True
        )
    assert not (root_tamper / ".dataset.incomplete").exists()

    partition_tamper = tmp_path / "partition-tamper"
    ShardedDatasetWriter(partition_tamper, shard_size=2).run_cases(
        engine, iter(cases)
    )
    tensor_path = partition_tamper / "shards" / "part-000000" / "tensors.npz"
    with tensor_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(InputValidationError, match="partition"):
        ShardedDatasetWriter(partition_tamper, shard_size=2).run_cases(
            engine, iter(cases), resume=True
        )


def test_completed_resume_clears_only_a_validated_stale_marker(
    tmp_path, target_data, case_data, registry
):
    engine = DamageEngine(target_data, registry)
    cases = _cases(engine, case_data, count=3)
    output = tmp_path / "stale-marker"
    writer = ShardedDatasetWriter(output, shard_size=2)
    expected = writer.run_cases(engine, iter(cases))
    marker = output / ".dataset.incomplete"
    marker.write_text("simulated crash after checkpoint publication", encoding="utf-8")

    observed = ShardedDatasetWriter(output, shard_size=2).run_cases(
        engine, iter(cases), resume=True
    )
    assert observed["dataset_id"] == expected["dataset_id"]
    assert not marker.exists()


def test_resume_execution_provenance_must_match(
    tmp_path, target_data, case_data, registry
):
    engine = DamageEngine(target_data, registry)
    cases = _cases(engine, case_data, count=2)
    output = tmp_path / "execution"
    ShardedDatasetWriter(output, shard_size=1).run_cases(engine, iter(cases))
    with pytest.raises(InputValidationError, match="configuration"):
        ShardedDatasetWriter(output, shard_size=1).run_cases(
            engine, iter(cases), workers=2, resume=True
        )
    with pytest.raises(InputValidationError, match="configuration"):
        ShardedDatasetWriter(output, shard_size=1).run_cases(
            engine, iter(cases), fail_closed=True, resume=True
        )


def test_global_leakage_audit_detects_conflict_across_shards(
    tmp_path, target_data, case_data, registry, monkeypatch
):
    engine = DamageEngine(target_data, registry)
    cases = _cases(engine, case_data, count=2)

    def conflicting_split(case, **_kwargs):
        split = "train" if case["case_id"].endswith("0") else "test"
        return "shared-group", "split_group", "shared-physical-case", split

    monkeypatch.setattr(dataset_module, "_split_name", conflicting_split)
    manifest = ShardedDatasetWriter(tmp_path / "leakage", shard_size=1).run_cases(
        engine, iter(cases)
    )
    assert manifest["split_policy"]["leakage_audit_passed"] is False

    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = np.asarray
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    from damage_engine.torch_dataset import TorchShardedDamageDataset

    with pytest.raises(RuntimeError, match="split-leakage"):
        TorchShardedDamageDataset(tmp_path / "leakage")
    assert len(
        TorchShardedDamageDataset(
            tmp_path / "leakage", allow_split_leakage=True
        )
    ) == 2


def test_refused_only_shard_keeps_contract_and_does_not_pollute_sources(
    tmp_path, target_data, case_data, registry
):
    engine = DamageEngine(target_data, registry)
    cases = _cases(engine, case_data, count=2)
    cases[1]["parameter_set_id"] = "missing-parameter-set"
    manifest = ShardedDatasetWriter(tmp_path / "refused", shard_size=1).run_cases(
        engine, iter(cases), fail_closed=False
    )
    assert manifest["quality_summary"]["n_valid"] == 1
    assert manifest["quality_summary"]["n_refused"] == 1
    assert "missing-parameter-set" not in manifest["source_contracts"][
        "parameter_set_ids"
    ]


def test_dataset_identity_tracks_engine_code_and_chunksize_zero_is_rejected(
    tmp_path, target_data, case_data, registry
):
    identity = {
        "target_hash": "sha256:" + "1" * 64,
        "physics_registry_hash": "sha256:" + "2" * 64,
        "ordered_case_hashes": ["sha256:" + "3" * 64],
        "split_seed": 0,
        "split_ratios": (0.8, 0.1, 0.1),
        "tensor_contract_version": "1.3.0",
    }
    assert dataset_identity_hash(**identity, engine_code_hash="sha256:" + "4" * 64) != (
        dataset_identity_hash(**identity, engine_code_hash="sha256:" + "5" * 64)
    )

    engine = DamageEngine(target_data, registry)
    with pytest.raises(ValueError, match="chunksize"):
        ShardedDatasetWriter(tmp_path / "bad-chunksize", shard_size=1).run_cases(
            engine, iter(_cases(engine, case_data, count=1)), chunksize=0
        )


def test_torch_sharded_dataset_is_lazy_split_aware_lru_and_spawn_safe(
    tmp_path, target_data, case_data, registry, monkeypatch
):
    engine = DamageEngine(target_data, registry)
    cases = _cases(engine, case_data, count=5)
    output = tmp_path / "torch-sharded"
    ShardedDatasetWriter(output, shard_size=2).run_cases(engine, iter(cases))

    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = np.asarray
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    from damage_engine.torch_dataset import TorchShardedDamageDataset

    dataset = TorchShardedDamageDataset(output, max_cached_shards=1)
    assert len(dataset) == 5
    assert dataset.cached_shard_indices == ()
    assert dataset[0]["case_id"] == "sharded-case-0"
    assert dataset.cached_shard_indices == (0,)
    assert dataset[-1]["case_id"] == "sharded-case-4"
    assert dataset.cached_shard_indices == (2,)
    assert dataset.target_hash == engine.target_hash

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.cached_shard_indices == ()
    assert restored[2]["case_id"] == "sharded-case-2"
    assert restored.cached_shard_indices == (1,)

    case_ids = _concatenate_shards(output, "case_ids").astype(str)
    split_names = _concatenate_shards(output, "split_names").astype(str)
    for split in ("train", "validation", "test"):
        expected = case_ids[split_names == split].tolist()
        split_dataset = TorchShardedDamageDataset(
            output, split=split, max_cached_shards=1
        )
        assert len(split_dataset) == len(expected)
        assert [split_dataset[index]["case_id"] for index in range(len(expected))] == expected


def test_torch_sharded_dataset_rejects_incomplete_root(
    tmp_path, target_data, case_data, registry, monkeypatch
):
    engine = DamageEngine(target_data, registry)
    output = tmp_path / "torch-incomplete"
    ShardedDatasetWriter(output, shard_size=2).run_cases(
        engine, iter(_cases(engine, case_data, count=2))
    )
    (output / ".dataset.incomplete").write_text("{}", encoding="utf-8")

    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = np.asarray
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    from damage_engine.torch_dataset import TorchShardedDamageDataset

    with pytest.raises(RuntimeError, match="incomplete"):
        TorchShardedDamageDataset(output)


def test_torch_sharded_dataset_uses_global_valid_axis_not_local_input_axis(
    tmp_path, target_data, case_data, registry, monkeypatch
):
    engine = DamageEngine(target_data, registry)
    cases = _cases(engine, case_data, count=5)
    cases[1]["target_ref"]["content_hash"] = "sha256:" + "0" * 64
    output = tmp_path / "torch-valid-axis"
    ShardedDatasetWriter(output, shard_size=2).run_cases(engine, iter(cases))

    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = np.asarray
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    from damage_engine.torch_dataset import TorchShardedDamageDataset

    dataset = TorchShardedDamageDataset(output, max_cached_shards=1)
    expected = _concatenate_shards(output, "case_ids").astype(str).tolist()
    assert expected == [
        "sharded-case-0", "sharded-case-2", "sharded-case-3", "sharded-case-4"
    ]
    assert len(dataset) == len(expected)
    assert [dataset[index]["case_id"] for index in range(len(dataset))] == expected


def test_torch_sharded_dataset_rejects_integrity_tampering(
    tmp_path, target_data, case_data, registry, monkeypatch
):
    engine = DamageEngine(target_data, registry)
    output = tmp_path / "torch-integrity"
    ShardedDatasetWriter(output, shard_size=2).run_cases(
        engine, iter(_cases(engine, case_data, count=2))
    )

    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = np.asarray
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    from damage_engine.torch_dataset import TorchShardedDamageDataset

    root_manifest = output / "manifest.json"
    root_bytes = root_manifest.read_bytes()
    root_manifest.write_bytes(root_bytes + b" ")
    with pytest.raises(RuntimeError, match="root manifest"):
        TorchShardedDamageDataset(output)
    root_manifest.write_bytes(root_bytes)

    shard = output / "shards" / "part-000000"
    shard_manifest = shard / "manifest.json"
    shard_bytes = shard_manifest.read_bytes()
    shard_manifest.write_bytes(shard_bytes + b" ")
    with pytest.raises(RuntimeError, match="shard manifest"):
        TorchShardedDamageDataset(output)
    shard_manifest.write_bytes(shard_bytes)

    tensors = shard / "tensors.npz"
    tensor_bytes = tensors.read_bytes()
    tensors.write_bytes(tensor_bytes + b"tampered")
    with pytest.raises(RuntimeError, match="partition"):
        TorchShardedDamageDataset(output)

