from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from damage_engine.dataset import DatasetWriter
from damage_engine.engine import DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.input_identity import FrozenCaseInputs
from damage_engine.label_spec import LabelSpec
from damage_engine.result_ledger import iter_result_ledger, resolve_result_ledger
from damage_engine.sampling_checkpoint import (
    _ReferenceFormalFrozenInputs,
    initialize_sampling_run,
    iter_reference_formal_cases,
)
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.sharded_dataset import ShardedDatasetWriter
from damage_engine.utils import file_hash
from damage_engine.utils import content_hash


ROOT = Path(__file__).resolve().parents[1]


def _initialize(root: Path, target_data: dict, registry, *, base_case=None):
    engine = DamageEngine(target_data, registry)
    case = base_case or json.loads(
        (ROOT / "examples/case.uav_control.demo.json").read_text(encoding="utf-8")
    )
    spec = SamplingSpec.from_file(
        ROOT / "examples/sampling_spec.uav_control.demo.json"
    )
    labels = LabelSpec.from_file(ROOT / "examples/label_spec.demo.json")
    checkpoint, _ = initialize_sampling_run(
        root,
        engine=engine,
        base_case=case,
        sampling_spec=spec,
        label_spec=labels,
        cases_base_directory=root.parent,
    )
    return engine, checkpoint


def _writer(path: Path) -> ShardedDatasetWriter:
    return ShardedDatasetWriter(path, shard_size=2, split_seed=2026101303)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def _ledger_payloads(shard: Path) -> list[dict]:
    manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
    payloads = list(iter_result_ledger(resolve_result_ledger(shard, manifest)))
    for payload in payloads:
        # Wall-clock provenance is expected to differ across two executions;
        # every content/identity/result field must remain equal.
        payload.get("provenance", {}).pop("created_at", None)
    return payloads


def _read_plan_rows(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return stream.read().splitlines()


def _write_plan_rows(path: Path, rows: list[str]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0
        ) as stream:
            stream.write(("\n".join(rows) + "\n").encode("utf-8"))


def _rebind_file_only(root: Path, relative: str) -> None:
    checkpoint_path = root / "sampling_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    for record in checkpoint["committed_files"]:
        if record["path"] == relative:
            record["content_hash"] = file_hash(root / relative)
            break
    else:  # pragma: no cover - helper contract
        raise AssertionError(relative)
    checkpoint_path.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_compact_and_legacy_freeze_are_semantically_equivalent(
    tmp_path, target_data, registry
):
    root = tmp_path / "sampling"
    engine, checkpoint = _initialize(root, target_data, registry)

    compact_path = root / "formal_dataset"
    compact = _writer(compact_path).run_cases(
        engine,
        iter_reference_formal_cases(root),
        fail_closed=True,
        workers=1,
        expected_case_count=4,
    )
    # A materialized list is intentionally the unchanged generic/legacy path.
    legacy_cases = list(iter_reference_formal_cases(root))
    legacy_path = tmp_path / "legacy_dataset"
    legacy = _writer(legacy_path).run_cases(
        engine,
        legacy_cases,
        fail_closed=True,
        workers=1,
        expected_case_count=4,
    )

    assert compact["dataset_id"] == legacy["dataset_id"]
    assert compact["quality_summary"] == legacy["quality_summary"]
    compact_checkpoint = json.loads(
        (compact_path / "checkpoint.json").read_text(encoding="utf-8")
    )
    legacy_checkpoint = json.loads(
        (legacy_path / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert compact_checkpoint["identity"]["input_snapshot"] == checkpoint[
        "formal_input_identity"
    ]
    assert [
        digest
        for shard in compact_checkpoint["completed_shards"]
        for digest in shard["case_hashes"]
    ] == [
        digest
        for shard in legacy_checkpoint["completed_shards"]
        for digest in shard["case_hashes"]
    ]

    for index in range(2):
        compact_shard = compact_path / "shards" / f"part-{index:06d}"
        legacy_shard = legacy_path / "shards" / f"part-{index:06d}"
        compact_arrays = _load_npz(compact_shard / "tensors.npz")
        legacy_arrays = _load_npz(legacy_shard / "tensors.npz")
        assert compact_arrays.keys() == legacy_arrays.keys()
        assert all(
            np.array_equal(compact_arrays[name], legacy_arrays[name])
            for name in compact_arrays
        )
        assert _ledger_payloads(compact_shard) == _ledger_payloads(legacy_shard)
        shard_manifest = json.loads(
            (compact_shard / "manifest.json").read_text(encoding="utf-8")
        )
        shard_snapshot = shard_manifest["generation"]["input_snapshot"]
        assert shard_snapshot["expected_case_count"] == 2
        assert shard_snapshot["ordered_case_identity_hash"] != checkpoint[
            "formal_input_identity"
        ]["ordered_case_identity_hash"]

    transport = compact["generation"]["input_snapshot"]["transport"]
    assert transport["repeated_contract_documents_in_spool"] is False
    assert transport["ledger_bytes"] == (
        root / "rounds/round-000/formal_cases.jsonl.gz"
    ).stat().st_size


@pytest.mark.parametrize("defect", ["short", "extra", "duplicate"])
def test_compact_preflight_rejects_ledger_shape_before_dataset_write(
    tmp_path, target_data, registry, defect
):
    root = tmp_path / "sampling"
    engine, _ = _initialize(root, target_data, registry)
    relative = "rounds/round-000/formal_cases.jsonl.gz"
    ledger = root / relative
    rows = _read_plan_rows(ledger)
    if defect == "short":
        rows = rows[:-1]
    elif defect == "extra":
        rows.append(rows[-1])
    else:
        rows[-1] = rows[0]
    _write_plan_rows(ledger, rows)
    _rebind_file_only(root, relative)

    output = root / "formal_dataset"
    with pytest.raises(InputValidationError, match="ordered input identity"):
        _writer(output).run_cases(
            engine,
            iter_reference_formal_cases(root),
            workers=1,
            expected_case_count=4,
        )
    assert not (output / "checkpoint.json").exists()
    assert not (output / "shards").exists()


def test_compact_sidecar_tamper_fails_before_dataset_write(
    tmp_path, target_data, registry
):
    root = tmp_path / "sampling"
    engine, _ = _initialize(root, target_data, registry)
    relative = "plan/sampling_spec.json"
    path = root / relative
    document = json.loads(path.read_text(encoding="utf-8"))
    document["sampling_spec_id"] += "-tampered"
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _rebind_file_only(root, relative)

    output = root / "formal_dataset"
    with pytest.raises(InputValidationError, match="identities differ"):
        _writer(output).run_cases(
            engine, iter_reference_formal_cases(root), workers=1
        )
    assert not (output / "checkpoint.json").exists()
    assert not (output / "shards").exists()


@pytest.mark.parametrize("defect", ["status", "rounds", "marker"])
def test_compact_preflight_rejects_invalid_sampling_checkpoint_state(
    tmp_path, target_data, registry, defect
):
    root = tmp_path / "sampling"
    engine, _ = _initialize(root, target_data, registry)
    checkpoint_path = root / "sampling_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if defect == "status":
        checkpoint["status"] = "unknown"
    elif defect == "rounds":
        checkpoint["rounds_committed"] = [0, 1]
    else:
        (root / ".sampling.incomplete").write_text("incomplete\n", encoding="utf-8")
    if defect != "marker":
        checkpoint_path.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    output = root / "formal_dataset"
    with pytest.raises(InputValidationError):
        _writer(output).run_cases(
            engine, iter_reference_formal_cases(root), workers=1
        )
    assert not (output / "checkpoint.json").exists()
    assert not (output / "shards").exists()


def test_compact_interrupted_prefix_resumes_and_completed_resume_is_stable(
    tmp_path, target_data, registry, monkeypatch
):
    root = tmp_path / "sampling"
    engine, _ = _initialize(root, target_data, registry)
    output = root / "formal_dataset"
    original = DatasetWriter._run_frozen_cases
    calls = 0

    def interrupt_second(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("intentional interruption")
        return original(self, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(DatasetWriter, "_run_frozen_cases", interrupt_second)
        with pytest.raises(RuntimeError, match="intentional interruption"):
            _writer(output).run_cases(
                engine, iter_reference_formal_cases(root), workers=1
            )
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert len(checkpoint["completed_shards"]) == 1
    assert (output / "shards/part-000000").is_dir()
    assert not (output / "shards/part-000001").exists()

    resumed = _writer(output).run_cases(
        engine,
        iter_reference_formal_cases(root),
        workers=1,
        resume=True,
        expected_case_count=4,
    )
    completed_hash = file_hash(output / "manifest.json")
    again = _writer(output).run_cases(
        engine,
        iter_reference_formal_cases(root),
        workers=1,
        resume=True,
        expected_case_count=4,
    )
    assert again["dataset_id"] == resumed["dataset_id"]
    assert file_hash(output / "manifest.json") == completed_hash


def test_compact_resume_rejects_tail_tamper_before_new_shard(
    tmp_path, target_data, registry, monkeypatch
):
    root = tmp_path / "sampling"
    engine, _ = _initialize(root, target_data, registry)
    output = root / "formal_dataset"
    original = DatasetWriter._run_frozen_cases
    calls = 0

    def interrupt_second(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("intentional interruption")
        return original(self, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(DatasetWriter, "_run_frozen_cases", interrupt_second)
        with pytest.raises(RuntimeError):
            _writer(output).run_cases(
                engine, iter_reference_formal_cases(root), workers=1
            )

    relative = "rounds/round-000/formal_cases.jsonl.gz"
    ledger = root / relative
    rows = _read_plan_rows(ledger)
    tail = json.loads(rows[-1])
    tail["case_id"] += "-tampered"
    rows[-1] = json.dumps(tail, ensure_ascii=False, separators=(",", ":"))
    _write_plan_rows(ledger, rows)
    _rebind_file_only(root, relative)

    with pytest.raises(InputValidationError, match="ordered input identity"):
        _writer(output).run_cases(
            engine,
            iter_reference_formal_cases(root),
            workers=1,
            resume=True,
        )
    assert (output / "shards/part-000000").is_dir()
    assert not (output / "shards/part-000001").exists()


def test_compact_workers_two_spawn_path(tmp_path, target_data, registry):
    root = tmp_path / "sampling"
    engine, _ = _initialize(root, target_data, registry)
    manifest = _writer(root / "formal_dataset").run_cases(
        engine,
        iter_reference_formal_cases(root),
        fail_closed=True,
        workers=2,
        chunksize=1,
        expected_case_count=4,
    )
    assert manifest["quality_summary"]["n_total"] == 4
    assert manifest["quality_summary"]["n_refused"] == 0


def test_declared_external_fragment_dependency_uses_legacy_freezer(
    tmp_path, target_data, registry, monkeypatch
):
    root = tmp_path / "sampling"
    _initialize(root, target_data, registry)
    dependency = {
        "logical_source_id": "warhead.fragment_source.fragment_list_ref",
        "content_hash": "sha256:" + "1" * 64,
        "format": "json",
        "parser_version": "fragment-list-local-1",
    }
    input_path = root / "plan/input_snapshot.json"
    input_snapshot = json.loads(input_path.read_text(encoding="utf-8"))
    input_snapshot["dependencies"] = [dependency]
    input_path.write_text(
        json.dumps(input_snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checkpoint_path = root / "sampling_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["identity"]["input_dependency_manifest_hash"] = content_hash(
        [dependency]
    )
    checkpoint["identity"]["identity_hash"] = content_hash(
        {
            key: value
            for key, value in checkpoint["identity"].items()
            if key != "identity_hash"
        }
    )
    checkpoint_path.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _rebind_file_only(root, "plan/input_snapshot.json")

    sentinel = FrozenCaseInputs.__new__(FrozenCaseInputs)
    called = {}

    @contextmanager
    def legacy_freezer(cases, *, cases_base_directory=None):
        called["iterable_type"] = type(cases)
        called["base"] = cases_base_directory
        yield sentinel

    monkeypatch.setattr(
        "damage_engine.sampling_checkpoint.freeze_case_inputs", legacy_freezer
    )
    source = iter_reference_formal_cases(root)
    with source._freeze_compact(cases_base_directory=root) as frozen:
        assert frozen is sentinel
    assert called["iterable_type"].__name__ == "generator"
    assert called["base"] == root
