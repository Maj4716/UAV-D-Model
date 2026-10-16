from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from damage_engine.errors import InputValidationError
from damage_engine.input_identity import case_identity_hash
from damage_engine.sampling_checkpoint import (
    _REFERENCE_SPLIT_REGISTRY_PATH,
    _iter_hydrated_formal_cases,
    _verify_committed_sidecars,
    _write_formal_cases_atomic,
)
from damage_engine.split_registry import SplitRegistry
from damage_engine.utils import content_hash, file_hash


def _plan(case_count: int):
    registry = SplitRegistry("storage-regression-splits")
    cases = []
    for index in range(case_count):
        lineage = f"lineage-{index:04d}"
        registry.register(
            lineage_id=lineage,
            split="train",
            role="train_pilot",
            epistemic_group_id=lineage,
        )
        cases.append((index, lineage))
    registry.freeze()
    document = registry.definition_document()
    label_document = {"schema_version": "test", "purpose": "storage regression"}
    label_spec = SimpleNamespace(
        data=label_document,
        hash=content_hash(label_document),
        spec_id="storage-labels",
    )
    sampling_spec = SimpleNamespace()
    values = []
    for index, lineage in cases:
        values.append({
            "case_id": f"case-{index:04d}",
            "metadata": {
                "frozen_split": "train",
                "split_registry_hash": registry.definition_hash,
                "sampling": {
                    "schema_version": "1.0.0",
                    "lineage_id": lineage,
                    "role": "train_pilot",
                    "epistemic_group_id": lineage,
                    "frozen_split": "train",
                    "split_registry_id": registry.registry_id,
                    "split_registry_hash": registry.definition_hash,
                    "split_registry_document": document,
                    "label_spec_id": label_spec.spec_id,
                    "label_spec_hash": label_spec.hash,
                    "label_spec_document": label_spec.data,
                }
            },
        })
    return registry, sampling_spec, label_spec, values


def test_formal_case_transport_externalizes_registry_and_restores_identity(tmp_path):
    root = tmp_path / "run"
    ledger = root / "rounds/round-000/formal_cases.jsonl"
    registry, sampling_spec, label_spec, cases = _plan(12)
    registry.write(root / "plan/split_registry.json")

    _write_formal_cases_atomic(
        ledger, cases, registry_reference=_REFERENCE_SPLIT_REGISTRY_PATH
    )

    raw = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert all(
        "split_registry_document" not in row["metadata"]["sampling"]
        for row in raw
    )
    assert all(
        "label_spec_document" not in row["metadata"]["sampling"]
        for row in raw
    )
    assert {
        row["metadata"]["sampling"]["split_registry_reference"] for row in raw
    } == {_REFERENCE_SPLIT_REGISTRY_PATH}

    hydrated = list(_iter_hydrated_formal_cases(
        ledger, registry, sampling_spec, label_spec,
        expected_reference=_REFERENCE_SPLIT_REGISTRY_PATH,
    ))
    assert [case_identity_hash(case) for case in hydrated] == [
        case_identity_hash(case) for case in cases
    ]
    assert hydrated[0]["metadata"]["sampling"]["split_registry_document"] == (
        registry.definition_document()
    )


def test_externalized_ledger_removes_quadratic_registry_payload(tmp_path):
    root = tmp_path / "run"
    ledger = root / "rounds/round-000/formal_cases.jsonl"
    registry, _, _, cases = _plan(80)
    registry.write(root / "plan/split_registry.json")
    inline_bytes = sum(
        len(json.dumps(case, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        + 1
        for case in cases
    )

    _write_formal_cases_atomic(
        ledger, cases, registry_reference=_REFERENCE_SPLIT_REGISTRY_PATH
    )

    # The exact ratio is intentionally loose; the regression is that the full
    # 80-assignment document occurs once, rather than 80 times.
    externalized_bytes = ledger.stat().st_size + (root / "plan/split_registry.json").stat().st_size
    assert externalized_bytes < inline_bytes * 0.12


def test_externalized_registry_reference_fails_closed_when_tampered(tmp_path):
    root = tmp_path / "run"
    ledger = root / "rounds/round-000/formal_cases.jsonl"
    registry, sampling_spec, label_spec, cases = _plan(2)
    registry.write(root / "plan/split_registry.json")
    _write_formal_cases_atomic(
        ledger, cases, registry_reference=_REFERENCE_SPLIT_REGISTRY_PATH
    )
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    changed = deepcopy(rows[0])
    changed["metadata"]["sampling"]["split_registry_reference"] = "other.json"
    ledger.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in [changed, *rows[1:]]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InputValidationError, match="reference is missing or unexpected"):
        list(_iter_hydrated_formal_cases(
            ledger, registry, sampling_spec, label_spec,
            expected_reference=_REFERENCE_SPLIT_REGISTRY_PATH,
        ))


def test_hydration_sidecar_inventory_hash_fails_closed_on_changed_bytes(tmp_path):
    root = tmp_path / "run"
    paths = [
        "plan/split_registry.json",
        "plan/sampling_spec.json",
        "plan/label_spec.json",
        "rounds/round-000/formal_cases.jsonl",
    ]
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n", encoding="utf-8")
    checkpoint = {
        "committed_files": [
            {"path": relative, "content_hash": file_hash(root / relative)}
            for relative in paths
        ]
    }
    (root / "sampling_checkpoint.json").write_text(
        json.dumps(checkpoint), encoding="utf-8"
    )
    _verify_committed_sidecars(root)

    (root / "plan/sampling_spec.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(InputValidationError, match="sidecar is missing or differs"):
        _verify_committed_sidecars(root)


def test_hydration_rejects_case_local_contract_document_override(tmp_path):
    root = tmp_path / "run"
    ledger = root / "rounds/round-000/formal_cases.jsonl"
    registry, sampling_spec, label_spec, cases = _plan(1)
    registry.write(root / "plan/split_registry.json")
    _write_formal_cases_atomic(
        ledger, cases, registry_reference=_REFERENCE_SPLIT_REGISTRY_PATH
    )
    row = json.loads(ledger.read_text(encoding="utf-8"))
    row["metadata"]["sampling"]["label_spec_document"] = {"locally": "changed"}
    ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(InputValidationError, match="missing, unexpected, or ambiguous"):
        list(_iter_hydrated_formal_cases(
            ledger, registry, sampling_spec, label_spec,
            expected_reference=_REFERENCE_SPLIT_REGISTRY_PATH,
        ))
