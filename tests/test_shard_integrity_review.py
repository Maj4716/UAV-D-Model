"""Generic synthetic I/O regressions; no engine simulation is called."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

import damage_engine.sharded_dataset as shard_module
from damage_engine.dataset import DatasetWriter
from damage_engine.errors import InputValidationError
from damage_engine.input_identity import (
    case_identity_hash,
    capture_environment_identity,
    freeze_case_inputs,
)
from damage_engine.quality import synthetic_label_admission
from damage_engine.sharded_dataset import ShardedDatasetWriter, _single_writer_lock, _write_json_atomic
from damage_engine.utils import content_hash, file_hash
from test_writer_integrity_review import synthetic_writer


def _engine():
    return SimpleNamespace(target_hash="sha256:" + "a" * 64, registry=SimpleNamespace(hash="sha256:" + "b" * 64))


def _cases(count=3):
    return [{"case_id": f"toy-{index}", "value": index} for index in range(count)]


def _fake_write(self, engine, cases, **kwargs):
    """Minimal opaque toy arrays for exercising publication mechanics only."""
    case_list = list(cases)
    vocabulary_names = {
        "mechanism_names": [],
        "state_names": [],
        "system_output_names": [],
        "evaluation_standard_ids": ["toy"],
    }
    self.output_directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        self.output_directory / "tensors.npz",
        split_group_ids=np.array([item["case_id"] for item in case_list]),
        leakage_keys=np.array([item["case_id"] for item in case_list]),
        split_names=np.array(["train"] * len(case_list)),
        system_output_names=np.array([], dtype=str),
    )
    (self.output_directory / "results.jsonl").write_text("", encoding="utf-8")
    manifest = {
        "schema_version": "1.1.0", "dataset_id": "toy-" + content_hash(case_list).split(":")[1][:16],
        "created_at": "2026-10-16T00:00:00+00:00",
        "engine": {"version": "toy", "code_hash": shard_module.ENGINE_CODE_HASH, "configuration_hash": engine.registry.hash, "fidelity_level": "L0", "fidelity_levels": ["L0"]},
        "source_contracts": {
            "target_schema": "toy",
            "case_schema": "toy",
            "evaluation_standards": {"toy": ["toy"]},
            "target_hashes": [engine.target_hash],
        },
        "semantic_vocabulary": {
            "version": "toy-1.0",
            "names": vocabulary_names,
            "content_hash": content_hash(vocabulary_names),
        },
        "label_admission": synthetic_label_admission(),
        "generation": {
            "workers": 1,
            "chunksize": 1,
            "failure_policy": "precomputed_results",
            "input_order_preserved": True,
            "tensor_compression": "none",
        },
        "tensor_contract": {
            "version": "1.4.0",
            "fields": [
                {
                    "name": "split_group_ids",
                    "dtype": "string",
                    "shape": "[B]",
                    "semantic_role": "metadata",
                }
            ],
        },
        "partitions": [{"name": "tensors", "format": "npz", "path_glob": "tensors.npz", "row_count": len(case_list), "content_hash": file_hash(self.output_directory / "tensors.npz")}],
        "quality_summary": {"n_total": len(case_list), "n_valid": len(case_list), "n_refused": 0, "refusal_counts": {}, "unverified_parameter_fraction": 0},
        "randomness": {
            "split_seed": 0,
            "aleatory_seed_field": "analysis.aleatory_seed",
            "epistemic_sample_id_field": "analysis.epistemic_sample_id",
            "parallel_result_order_deterministic": True,
        },
        "split_policy": {
            "method": "grouped",
            "group_keys": ["split_group"],
            "seed": 0,
            "leakage_audit_passed": True,
        },
        "known_limitations": [],
        "prohibited_uses": ["toy software fixture only"],
    }
    _write_json_atomic(self.output_directory / "manifest.json", manifest)
    return manifest


@pytest.fixture
def toy_writer(monkeypatch):
    monkeypatch.setattr(DatasetWriter, "run_cases", _fake_write)
    return _engine()


def test_snapshot_identity_tracks_content_and_is_location_independent(tmp_path):
    source = tmp_path / "source.json"
    source.write_bytes(b"[]")
    case = {"case_id": "toy", "warhead": {"fragment_source": {"fragment_list_ref": str(source)}}}
    with freeze_case_inputs([case]) as frozen:
        copied = next(iter(frozen))
        digest = case_identity_hash(copied)
        frozen_path = Path(copied["warhead"]["fragment_source"]["fragment_list_ref"])
        source.write_bytes(b"[1]")
        assert frozen_path.read_bytes() == b"[]"
        assert case_identity_hash(pickle.loads(pickle.dumps(copied))) == digest
        assert case_identity_hash(next(iter(frozen))) == digest
        assert frozen.metadata["expected_case_count"] == 1
    with freeze_case_inputs([case]) as changed:
        assert case_identity_hash(next(iter(changed))) != digest
    relocated = tmp_path / "elsewhere.json"
    relocated.write_bytes(b"[]")
    case["warhead"]["fragment_source"]["fragment_list_ref"] = str(relocated)
    with freeze_case_inputs([case]) as moved:
        assert case_identity_hash(next(iter(moved))) == digest


def test_relative_snapshot_requires_explicit_base_and_raw_loaded_hash_matches(tmp_path):
    source = tmp_path / "source.json"
    source.write_bytes(b"[1]")
    case = {"case_id": "toy", "warhead": {"fragment_source": {"fragment_list_ref": "source.json"}}}
    with pytest.raises(InputValidationError, match="cases_base_directory"):
        with freeze_case_inputs([case]):
            pass
    with freeze_case_inputs([case], cases_base_directory=tmp_path) as frozen:
        copied = next(iter(frozen))
        loaded = {"content_hash": file_hash(source), "format": "json"}
        assert case_identity_hash(case, loaded_dependency=loaded) == case_identity_hash(copied)


def test_re_freezing_keeps_same_snapshot_and_full_input_identity(tmp_path):
    source = tmp_path / "source.json"
    source.write_bytes(b"[]")
    cases = [{"case_id": "toy", "warhead": {"fragment_source": {"fragment_list_ref": str(source)}}}]
    with freeze_case_inputs(cases) as first:
        source.write_bytes(b"changed")
        with freeze_case_inputs(first) as second:
            assert first.identity == second.identity
            assert next(iter(first)) == next(iter(second))


def test_completed_partition_paths_are_exact_committed_files(tmp_path, toy_writer):
    output = tmp_path / "published"
    manifest = ShardedDatasetWriter(output, shard_size=1).run_cases(toy_writer, _cases())
    checkpoint = json.loads((output / "checkpoint.json").read_text())
    declared = {item["path_glob"] for item in manifest["partitions"]}
    expected = {item["path"] + "/tensors.npz" for item in checkpoint["completed_shards"]}
    assert declared == expected
    assert all(not any(character in item["path_glob"] for character in "*?[]") for item in manifest["partitions"])
    assert sum(item["row_count"] for item in manifest["partitions"]) == 3
    assert all(file_hash(output / item["path_glob"]) == item["content_hash"] for item in manifest["partitions"])
    assert checkpoint["identity"]["input_snapshot"]["expected_case_count"] == 3
    assert manifest["generation"]["environment"]["identity_hash"]


def test_shortened_finite_input_resume_rejected_before_publication(tmp_path, toy_writer, monkeypatch):
    output = tmp_path / "shortened"
    calls = 0

    def interrupted(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("toy interruption before shard creation")
        return _fake_write(self, *args, **kwargs)

    monkeypatch.setattr(DatasetWriter, "run_cases", interrupted)
    with pytest.raises(RuntimeError, match="toy interruption"):
        ShardedDatasetWriter(output, shard_size=1).run_cases(toy_writer, _cases())
    with pytest.raises(InputValidationError, match="configuration"):
        ShardedDatasetWriter(output, shard_size=1).run_cases(toy_writer, _cases(1), resume=True)
    assert not (output / "manifest.json").exists()
    assert (output / ".dataset.incomplete").exists()


def test_rename_before_checkpoint_orphan_is_preserved_and_rejected(tmp_path, toy_writer, monkeypatch):
    output = tmp_path / "orphan"
    atomic = shard_module._write_json_atomic

    def crash(path, value):
        if path.name == "checkpoint.json" and len(value["completed_shards"]) == 2:
            raise RuntimeError("toy crash after rename before checkpoint")
        return atomic(path, value)

    monkeypatch.setattr(shard_module, "_write_json_atomic", crash)
    with pytest.raises(RuntimeError, match="after rename"):
        ShardedDatasetWriter(output, shard_size=1).run_cases(toy_writer, _cases())
    orphan = output / "shards" / "part-000001"
    original_hash = file_hash(orphan / "tensors.npz")
    monkeypatch.setattr(shard_module, "_write_json_atomic", atomic)
    with pytest.raises(InputValidationError, match="orphan or temporary"):
        ShardedDatasetWriter(output, shard_size=1).run_cases(toy_writer, _cases(), resume=True)
    assert file_hash(orphan / "tensors.npz") == original_hash
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("extra", ["part-000010", ".part-000010.tmp", "unrecognized"])
def test_completed_resume_rejects_extra_disk_entries(tmp_path, toy_writer, extra):
    output = tmp_path / "extra"
    ShardedDatasetWriter(output, shard_size=1).run_cases(toy_writer, _cases(1))
    artifact = output / "shards" / extra
    artifact.mkdir()
    (artifact / "evidence.txt").write_text("preserve me")
    with pytest.raises(InputValidationError, match="inventory"):
        ShardedDatasetWriter(output, shard_size=1).run_cases(toy_writer, _cases(1), resume=True)
    assert (artifact / "evidence.txt").read_text() == "preserve me"


def test_single_writer_lock_and_expected_count_prevent_mutation(tmp_path, toy_writer):
    output = tmp_path / "locked"
    with _single_writer_lock(output):
        with pytest.raises(InputValidationError, match="single-writer"):
            ShardedDatasetWriter(output, shard_size=1).run_cases(toy_writer, _cases(1))
    assert not output.exists()
    with pytest.raises(InputValidationError, match="expected_case_count"):
        ShardedDatasetWriter(output, shard_size=1).run_cases(toy_writer, _cases(1), expected_case_count=2)
    assert not output.exists()


def test_environment_change_is_strict_execution_resume_mismatch(tmp_path, toy_writer, monkeypatch):
    output = tmp_path / "environment"
    ShardedDatasetWriter(output, shard_size=1).run_cases(toy_writer, _cases(1))
    identity = deepcopy(capture_environment_identity())
    identity["dependencies"]["numpy"] = "different-test-version"
    identity["identity_hash"] = content_hash(identity)
    monkeypatch.setattr(shard_module, "capture_environment_identity", lambda: identity)
    with pytest.raises(InputValidationError, match="configuration"):
        ShardedDatasetWriter(output, shard_size=1).run_cases(toy_writer, _cases(1), resume=True)


def test_external_same_path_changed_byte_rejects_completed_checkpoint(tmp_path, toy_writer):
    source = tmp_path / "toy.json"
    source.write_bytes(b"[1]")
    case = {"case_id": "toy", "warhead": {"fragment_source": {"fragment_list_ref": str(source)}}}
    output = tmp_path / "external"
    ShardedDatasetWriter(output, shard_size=1).run_cases(toy_writer, [case])
    source.write_bytes(b"[2]")
    with pytest.raises(InputValidationError, match="configuration"):
        ShardedDatasetWriter(output, shard_size=1).run_cases(toy_writer, [case], resume=True)


def test_constructor_has_no_directory_side_effect(tmp_path):
    output = tmp_path / "new"
    ShardedDatasetWriter(output, shard_size=1)
    assert not output.exists()


def test_atomic_replace_retries_a_transient_windows_lock(tmp_path, monkeypatch):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    original = Path.replace
    calls = 0

    def transient(self, destination):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("transient test lock")
        return original(self, destination)

    monkeypatch.setattr(Path, "replace", transient)
    shard_module._replace_with_transient_retry(
        source, target, attempts=3, delay_seconds=0.0
    )
    assert calls == 3
    assert target.is_dir()


@pytest.mark.parametrize("value", [{"warhead": None}, {"warhead": {"fragment_source": []}}])
def test_malformed_dependency_containers_are_input_refusals(value):
    with pytest.raises(InputValidationError, match="JSON object"):
        with freeze_case_inputs([value]):
            pass


def test_frozen_dependency_cannot_be_rebound_or_misreport_loaded_digest(tmp_path):
    source = tmp_path / "source.json"
    source.write_bytes(b"[]")
    case = {"case_id": "toy", "warhead": {"fragment_source": {"fragment_list_ref": str(source)}}}
    with freeze_case_inputs([case]) as frozen:
        copied = next(iter(frozen))
        with pytest.raises(InputValidationError, match="differs from frozen"):
            case_identity_hash(copied, loaded_dependency={"content_hash": "sha256:" + "0" * 64, "format": "json"})
        copied["warhead"]["fragment_source"]["fragment_list_ref"] = str(source)
        with pytest.raises(InputValidationError, match="reference has changed"):
            case_identity_hash(copied)


@pytest.mark.parametrize("observation_count", [1, 2])
def test_actual_toy_writer_shards_schema_loader_and_global_axes(
    tmp_path, synthetic_writer, monkeypatch, observation_count,
):
    from damage_engine.torch_dataset import TorchShardedDamageDataset
    from damage_engine.validation import _validate_schema

    engine, pair = synthetic_writer
    cases = [pair(f"integration-{index}", count=observation_count)[0] for index in range(3)]

    def precomputed_only(self, engine, case_values, **kwargs):
        values = list(case_values)
        results = [pair(case["case_id"], count=observation_count)[1] for case in values]
        return self.write(engine, values, results)

    monkeypatch.setattr(DatasetWriter, "run_cases", precomputed_only)
    manifest = ShardedDatasetWriter(tmp_path, shard_size=2).run_cases(engine, cases)
    _validate_schema(manifest, "dataset_manifest.schema.json")
    axes = {axis["axis_id"]: axis["size"] for axis in manifest["tensor_contract"]["axes"]}
    assert axes["B"] == axes["N"] == 3
    if observation_count == 1:
        assert manifest["quality_summary"]["mc_ci_half_width_quantiles"] == []
    else:
        np.testing.assert_allclose(manifest["quality_summary"]["mc_ci_half_width_quantiles"], [.1, .1, .1])
    archive = TorchShardedDamageDataset(tmp_path)
    assert [archive[index]["case_id"] for index in range(len(archive))] == [case["case_id"] for case in cases]
    assert manifest["capabilities"]["storage_layout"] == "sharded"
    assert manifest["label_admission"]["approved_uses"] == ["software_development"]
    assert manifest["quality_summary"]["purpose_admission_counts"]["research_training"] == 0
