"""Cross-entry archive integrity contracts; synthetic software data only."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

import damage_engine.archive_validation as archive_validation_module
import damage_engine.dataset as dataset_module
import damage_engine.sharded_dataset as sharded_module
from damage_engine.archive_validation import (
    ArchiveIntegrityError,
    verify_monolithic_archive,
    verify_sharded_archive,
)
from damage_engine.dataset import DatasetWriter
from damage_engine.dataset_policy import DatasetUsePolicy
from damage_engine.engine import DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.sharded_dataset import ShardedDatasetWriter
from damage_engine.torch_dataset import TorchDamageDataset, TorchShardedDamageDataset
from damage_engine.utils import file_hash
from test_dataset_policy import document as policy_document
from test_sharded_dataset import _cases


def _policy_for(directory) -> DatasetUsePolicy:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    value = policy_document()
    value["dataset_id"] = manifest["dataset_id"]
    value["dataset_identity_hash"] = file_hash(directory / "manifest.json")
    return DatasetUsePolicy(value)


@pytest.mark.parametrize(
    "entry",
    ["public_verifier", "writer_resume", "torch_loader", "policy_loader"],
)
def test_partition_tamper_has_one_error_category_and_context(
    tmp_path, target_data, case_data, registry, entry,
):
    engine = DamageEngine(target_data, registry)
    cases = _cases(engine, case_data, count=2)
    output = tmp_path / entry
    ShardedDatasetWriter(output, shard_size=2).run_cases(engine, iter(cases))
    policy = _policy_for(output)
    tensor_path = output / "shards" / "part-000000" / "tensors.npz"
    with tensor_path.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ArchiveIntegrityError) as raised:
        if entry == "public_verifier":
            verify_sharded_archive(output, validate_tensors=False)
        elif entry == "writer_resume":
            ShardedDatasetWriter(output, shard_size=2).run_cases(
                engine, iter(cases), resume=True
            )
        elif entry == "torch_loader":
            TorchShardedDamageDataset(output)
        else:
            TorchShardedDamageDataset(
                output,
                policy=policy,
                operation="software_development",
            )

    context = raised.value.context
    assert context["integrity_kind"] == "partition_hash_mismatch"
    assert context["archive_scope"] == "shard"
    assert context["shard_index"] == 0
    assert context["partition"] == "tensors"
    assert context["path"].endswith("tensors.npz")


@pytest.mark.parametrize(
    "entry",
    ["public_verifier", "torch_loader", "policy_loader"],
)
def test_monolithic_partition_tamper_has_one_error_category_and_context(
    tmp_path, target_data, case_data, registry, entry,
):
    engine = DamageEngine(target_data, registry)
    output = tmp_path / f"monolithic-{entry}"
    DatasetWriter(output).run_cases(engine, iter(_cases(engine, case_data, count=2)))
    policy = _policy_for(output)
    tensor_path = output / "tensors.npz"
    with tensor_path.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ArchiveIntegrityError) as raised:
        if entry == "public_verifier":
            verify_monolithic_archive(output, validate_tensors=False)
        elif entry == "torch_loader":
            TorchDamageDataset(output)
        else:
            TorchDamageDataset(
                output,
                policy=policy,
                operation="software_development",
            )

    context = raised.value.context
    assert context["integrity_kind"] == "partition_hash_mismatch"
    assert context["archive_scope"] == "root"
    assert context["partition"] == "tensors"
    assert context["path"].endswith("tensors.npz")


@pytest.mark.parametrize(
    "entry",
    ["public_verifier", "torch_loader", "policy_loader"],
)
def test_monolithic_schema_tamper_has_one_error_category_and_context(
    tmp_path, target_data, case_data, registry, entry,
):
    engine = DamageEngine(target_data, registry)
    output = tmp_path / f"schema-{entry}"
    DatasetWriter(output).run_cases(engine, iter(_cases(engine, case_data, count=2)))
    policy = _policy_for(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("tensor_contract")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArchiveIntegrityError) as raised:
        if entry == "public_verifier":
            verify_monolithic_archive(output, validate_tensors=False)
        elif entry == "torch_loader":
            TorchDamageDataset(output)
        else:
            TorchDamageDataset(
                output,
                policy=policy,
                operation="software_development",
            )

    context = raised.value.context
    assert context["integrity_kind"] == "manifest_schema_invalid"
    assert context["archive_scope"] == "root"
    assert context["schema_name"] == "dataset_manifest.schema.json"
    assert context["path"].endswith("manifest.json")


@pytest.mark.parametrize(
    "entry",
    ["public_verifier", "torch_loader", "policy_loader"],
)
def test_hash_synchronized_tensor_contract_tamper_has_one_error_category(
    tmp_path, target_data, case_data, registry, entry,
):
    engine = DamageEngine(target_data, registry)
    output = tmp_path / f"tensor-contract-{entry}"
    DatasetWriter(output).run_cases(engine, iter(_cases(engine, case_data, count=2)))
    tensor_path = output / "tensors.npz"
    with np.load(tensor_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays["case_ids"][1] = arrays["case_ids"][0]
    np.savez_compressed(tensor_path, **arrays)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensor_partition = next(
        item for item in manifest["partitions"] if item["name"] == "tensors"
    )
    tensor_partition["content_hash"] = file_hash(tensor_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    policy = _policy_for(output)

    with pytest.raises(ArchiveIntegrityError) as raised:
        if entry == "public_verifier":
            verify_monolithic_archive(output)
        elif entry == "torch_loader":
            TorchDamageDataset(output)
        else:
            TorchDamageDataset(
                output,
                policy=policy,
                operation="software_development",
            )

    context = raised.value.context
    assert context["integrity_kind"] == "tensor_contract_invalid"
    assert context["archive_scope"] == "root"
    assert "case_ids must be unique" in context["cause_message"]
    assert context["path"].endswith("tensors.npz")


@pytest.mark.parametrize("storage_layout", ["monolithic", "sharded"])
def test_writer_reopens_committed_bytes_before_clearing_marker(
    tmp_path, target_data, case_data, registry, monkeypatch, storage_layout,
):
    engine = DamageEngine(target_data, registry)
    cases = _cases(engine, case_data, count=2)
    output = tmp_path / storage_layout
    observed = []

    if storage_layout == "monolithic":
        real_verify = dataset_module.verify_monolithic_archive

        def verify(directory, **kwargs):
            directory = output.__class__(directory)
            observed.append(
                {
                    "marker": (directory / ".dataset.incomplete").exists(),
                    "manifest": (directory / "manifest.json").is_file(),
                    "tensors": (directory / "tensors.npz").is_file(),
                }
            )
            return real_verify(directory, **kwargs)

        monkeypatch.setattr(dataset_module, "verify_monolithic_archive", verify)
        DatasetWriter(output).run_cases(engine, iter(cases))
    else:
        real_verify = sharded_module.verify_sharded_archive

        def verify(directory, **kwargs):
            directory = output.__class__(directory)
            checkpoint = json.loads(
                (directory / "checkpoint.json").read_text(encoding="utf-8")
            )
            observed.append(
                {
                    "marker": (directory / ".dataset.incomplete").exists(),
                    "manifest": (directory / "manifest.json").is_file(),
                    "checkpoint_complete": checkpoint["status"] == "complete",
                }
            )
            return real_verify(directory, **kwargs)

        monkeypatch.setattr(sharded_module, "verify_sharded_archive", verify)
        ShardedDatasetWriter(output, shard_size=1).run_cases(engine, iter(cases))

    assert observed == [
        {
            "marker": True,
            "manifest": True,
            **(
                {"tensors": True}
                if storage_layout == "monolithic"
                else {"checkpoint_complete": True}
            ),
        }
    ]
    assert not (output / ".dataset.incomplete").exists()


@pytest.mark.parametrize("storage_layout", ["monolithic", "sharded"])
def test_writer_reopen_refuses_tampered_commit_and_keeps_incomplete_marker(
    tmp_path, target_data, case_data, registry, monkeypatch, storage_layout,
):
    engine = DamageEngine(target_data, registry)
    cases = _cases(engine, case_data, count=2)
    output = tmp_path / f"writer-tamper-{storage_layout}"

    if storage_layout == "monolithic":
        real_verify = dataset_module.verify_monolithic_archive

        def verify(directory, **kwargs):
            tensor_path = output / "tensors.npz"
            with tensor_path.open("ab") as stream:
                stream.write(b"tampered-before-publication")
            return real_verify(directory, **kwargs)

        monkeypatch.setattr(dataset_module, "verify_monolithic_archive", verify)

        def run():
            return DatasetWriter(output).run_cases(engine, iter(cases))
    else:
        real_verify = sharded_module.verify_sharded_archive

        def verify(directory, **kwargs):
            tensor_path = output / "shards" / "part-000000" / "tensors.npz"
            with tensor_path.open("ab") as stream:
                stream.write(b"tampered-before-publication")
            return real_verify(directory, **kwargs)

        monkeypatch.setattr(sharded_module, "verify_sharded_archive", verify)

        def run():
            return ShardedDatasetWriter(output, shard_size=2).run_cases(
                engine, iter(cases)
            )

    with pytest.raises(ArchiveIntegrityError) as raised:
        run()
    assert raised.value.context["integrity_kind"] == "partition_hash_mismatch"
    assert (output / ".dataset.incomplete").exists()


def test_policy_binds_immutable_verified_receipt_and_rejects_stale_manifest(
    tmp_path, target_data, case_data, registry,
):
    engine = DamageEngine(target_data, registry)
    output = tmp_path / "policy"
    ShardedDatasetWriter(output, shard_size=2).run_cases(
        engine, iter(_cases(engine, case_data, count=2))
    )
    receipt = verify_sharded_archive(output, validate_tensors=False)
    policy = _policy_for(output)
    assert policy.assert_verified_archive_binding(receipt)["dataset_id"] == receipt.dataset_id
    assert receipt.checkpoint_hash == file_hash(output / "checkpoint.json")

    mutable_copy = receipt.manifest
    assert mutable_copy is not None
    mutable_copy["dataset_id"] = "forged-copy"
    assert receipt.manifest["dataset_id"] == receipt.dataset_id
    mutable_checkpoint = receipt.checkpoint
    assert mutable_checkpoint is not None
    mutable_checkpoint["status"] = "forged-copy"
    assert receipt.checkpoint["status"] == "complete"

    manifest_path = output / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    with pytest.raises(ArchiveIntegrityError) as raised:
        policy.assert_verified_archive_binding(receipt)
    assert raised.value.context["integrity_kind"] == "manifest_hash_mismatch"


@pytest.mark.parametrize(
    ("changed_path", "expected_kind"),
    [
        ("checkpoint.json", "checkpoint_hash_mismatch"),
        ("shards/part-000000/manifest.json", "manifest_hash_mismatch"),
        ("shards/part-000000/tensors.npz", "partition_hash_mismatch"),
    ],
)
def test_policy_receipt_rejects_checkpoint_or_partition_drift(
    tmp_path, target_data, case_data, registry, changed_path, expected_kind,
):
    engine = DamageEngine(target_data, registry)
    output = tmp_path / expected_kind
    ShardedDatasetWriter(output, shard_size=2).run_cases(
        engine, iter(_cases(engine, case_data, count=2))
    )
    receipt = verify_sharded_archive(output, validate_tensors=False)
    policy = _policy_for(output)
    path = output / changed_path
    with path.open("ab") as stream:
        stream.write(b"stale")

    with pytest.raises(ArchiveIntegrityError) as raised:
        policy.assert_verified_archive_binding(receipt)
    assert raised.value.context["integrity_kind"] == expected_kind


def test_lazy_sharded_loader_rejects_child_drift_after_initial_verification(
    tmp_path, target_data, case_data, registry,
):
    engine = DamageEngine(target_data, registry)
    output = tmp_path / "lazy-child-drift"
    ShardedDatasetWriter(output, shard_size=1).run_cases(
        engine, iter(_cases(engine, case_data, count=1))
    )
    dataset = TorchShardedDamageDataset(output)

    shard = output / "shards" / "part-000000"
    tensor_path = shard / "tensors.npz"
    with np.load(tensor_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    arrays["case_x"] = arrays["case_x"].copy()
    arrays["case_x"][0, 0] += 1.0
    np.savez_compressed(tensor_path, **arrays)

    manifest_path = shard / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensor_partition = next(
        item for item in manifest["partitions"] if item["name"] == "tensors"
    )
    tensor_partition["content_hash"] = file_hash(tensor_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArchiveIntegrityError) as raised:
        dataset[0]
    assert raised.value.context["integrity_kind"] == "manifest_hash_mismatch"
    assert raised.value.context["archive_scope"] == "shard"
    assert raised.value.context["shard_index"] == 0


def test_nested_tensor_input_validation_is_wrapped_as_archive_integrity_error(
    tmp_path, target_data, case_data, registry, monkeypatch,
):
    engine = DamageEngine(target_data, registry)
    output = tmp_path / "nested-tensor-validation"
    DatasetWriter(output).run_cases(
        engine, iter(_cases(engine, case_data, count=1))
    )

    def reject_nested_contract(*args, **kwargs):
        raise InputValidationError(
            "nested tensor validator refused the archive",
            context={"field": "input_sampling_candidate_ids"},
        )

    monkeypatch.setattr(
        archive_validation_module, "validate_tensor_arrays", reject_nested_contract
    )
    with pytest.raises(ArchiveIntegrityError) as raised:
        verify_monolithic_archive(output)
    assert raised.value.context["integrity_kind"] == "tensor_contract_invalid"
    assert raised.value.context["archive_scope"] == "root"
    assert raised.value.context["cause_message"] == (
        "nested tensor validator refused the archive"
    )


def test_sharded_worker_reopen_rejects_same_dataset_id_snapshot_drift(
    tmp_path, target_data, case_data, registry,
):
    engine = DamageEngine(target_data, registry)
    output = tmp_path / "worker-snapshot-drift"
    ShardedDatasetWriter(output, shard_size=1).run_cases(
        engine, iter(_cases(engine, case_data, count=1))
    )
    dataset = TorchShardedDamageDataset(output)
    serialized = pickle.dumps(dataset)
    original_dataset_id = dataset.checkpoint["dataset_id"]

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["known_limitations"].append("post-verification snapshot drift")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint_path = output / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["root_manifest_hash"] = file_hash(manifest_path)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    assert checkpoint["dataset_id"] == original_dataset_id

    with pytest.raises(ArchiveIntegrityError) as raised:
        pickle.loads(serialized)
    assert raised.value.context["integrity_kind"] in {
        "manifest_hash_mismatch",
        "checkpoint_hash_mismatch",
    }
    assert raised.value.context["archive_scope"] == "root"


def test_archive_integrity_error_pickle_roundtrip_preserves_machine_context():
    error = ArchiveIntegrityError(
        "partition changed",
        integrity_kind="partition_hash_mismatch",
        archive_scope="shard",
        path="shards/part-000003/tensors.npz",
        context={"shard_index": 3, "partition": "tensors"},
    )

    restored = pickle.loads(pickle.dumps(error))

    assert isinstance(restored, ArchiveIntegrityError)
    assert restored.to_dict() == error.to_dict()
