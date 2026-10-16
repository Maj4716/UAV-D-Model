"""Artificial loader-role tests only; no DamageEngine execution."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from damage_engine.archive_validation import semantic_vocabulary
from damage_engine.dataset_policy import DatasetUsePolicy
from damage_engine.errors import InputValidationError
from damage_engine.quality import synthetic_label_admission
from damage_engine.torch_dataset import TorchDamageDataset, TorchShardedDamageDataset
from damage_engine.utils import content_hash, file_hash


ROLE_NAMES = np.asarray(
    ["reference_test", "rare_stress_test", "reference_train", "adaptive_train"]
)


def _arrays(prefix: str, roles: tuple[str, ...]) -> dict[str, np.ndarray]:
    count = len(roles)
    return {
        "case_x": np.arange(count, dtype=np.float32).reshape(count, 1),
        "case_ids": np.asarray([f"{prefix}-{index}" for index in range(count)]),
        "case_hashes": np.asarray([f"hash-{prefix}-{index}" for index in range(count)]),
        "split_names": np.asarray(
            ["train" if role in {"reference_train", "adaptive_train"} else "test" for role in roles]
        ),
        "case_feature_names": np.asarray(["artificial-feature"]),
        "case_feature_units": np.asarray(["1"]),
        "sampling_role_index": np.asarray(
            [ROLE_NAMES.tolist().index(role) for role in roles], dtype=np.int64
        ),
        "sampling_role_names": ROLE_NAMES.copy(),
        "mechanism_names": np.asarray([], dtype=str),
        "state_names": np.asarray([], dtype=str),
        "system_output_names": np.asarray([], dtype=str),
        "evaluation_standard_ids": np.asarray([], dtype=str),
    }


def _shape(name: str) -> str:
    return {
        "case_x": "[B,F_case]",
        "case_ids": "[B]",
        "case_hashes": "[B]",
        "split_names": "[B]",
        "case_feature_names": "[F_case]",
        "case_feature_units": "[F_case]",
        "sampling_role_index": "[B]",
        "sampling_role_names": "[R]",
        "mechanism_names": "[M]",
        "state_names": "[S]",
        "system_output_names": "[O]",
        "evaluation_standard_ids": "[E_standard]",
    }[name]


def _write_archive(
    directory: Path, *, prefix: str, roles: tuple[str, ...]
) -> dict:
    arrays = _arrays(prefix, roles)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "tensors.npz", **arrays)
    manifest = {
        "dataset_id": "artificial-" + prefix,
        "source_contracts": {"target_hashes": ["artificial-target"]},
        "label_admission": synthetic_label_admission(),
        "split_policy": {"leakage_audit_passed": True},
        "quality_summary": {"n_valid": len(roles)},
        "partitions": [
            {
                "name": "tensors",
                "format": "npz",
                "path_glob": "tensors.npz",
                "row_count": len(roles),
                "content_hash": file_hash(directory / "tensors.npz"),
            }
        ],
        "tensor_contract": {
            "version": "1.4.0",
            "fields": [
                {
                    "name": name,
                    "dtype": "string" if value.dtype.kind in "US" else str(value.dtype),
                    "shape": _shape(name),
                }
                for name, value in arrays.items()
            ],
            "axes": [
                {
                    "axis_id": "F_case",
                    "size": 1,
                    "names_field": "case_feature_names",
                    "units_field": "case_feature_units",
                },
                {
                    "axis_id": "R",
                    "size": len(ROLE_NAMES),
                    "names_field": "sampling_role_names",
                    "units_field": None,
                },
            ],
        },
        "semantic_vocabulary": semantic_vocabulary(
            {
                name: tuple(map(str, arrays[name]))
                for name in (
                    "mechanism_names",
                    "state_names",
                    "system_output_names",
                    "evaluation_standard_ids",
                )
            }
        ),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


def _write_sharded(root: Path) -> None:
    records, partitions = [], []
    first = None
    for index, role in enumerate(("reference_test", "rare_stress_test")):
        shard = root / "shards" / f"part-{index:06d}"
        manifest = _write_archive(
            shard, prefix=f"shard-{index}", roles=(role, role)
        )
        first = first or manifest
        with np.load(shard / "tensors.npz", allow_pickle=False) as arrays:
            case_ids = arrays["case_ids"].astype(str).tolist()
            case_hashes = arrays["case_hashes"].astype(str).tolist()
        records.append(
            {
                "index": index,
                "path": shard.relative_to(root).as_posix(),
                "dataset_id": manifest["dataset_id"],
                "manifest_hash": file_hash(shard / "manifest.json"),
                "case_ids": case_ids,
                "case_hashes": case_hashes,
            }
        )
        partitions.extend(
            {
                **partition,
                "path_glob": (
                    shard.relative_to(root) / partition["path_glob"]
                ).as_posix(),
            }
            for partition in manifest["partitions"]
        )
    root_manifest = {
        **first,
        "dataset_id": "artificial-sharded",
        "partitions": partitions,
        "quality_summary": {"n_valid": 4},
    }
    (root / "manifest.json").write_text(json.dumps(root_manifest), encoding="utf-8")
    identity = {"shard_size": 2}
    identity["identity_hash"] = content_hash(identity)
    checkpoint = {
        "schema_version": "2.0.0",
        "identity": identity,
        "dataset_id": root_manifest["dataset_id"],
        "status": "complete",
        "completed_shards": records,
        "root_manifest_hash": file_hash(root / "manifest.json"),
    }
    (root / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")


def _policy(
    directory: Path,
    classification: str,
    *,
    model_hash: str | None = None,
    admitted_purpose: str | None = None,
):
    seal = (
        {
            "state": "opened_for_frozen_model" if model_hash else "sealed",
            "frozen_model_hash": model_hash,
        }
        if classification == "sealed_reference_test"
        else {"state": "not_applicable", "frozen_model_hash": None}
    )
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if admitted_purpose is not None:
        approved_uses = manifest["label_admission"]["approved_uses"]
        if admitted_purpose not in approved_uses:
            approved_uses.append(admitted_purpose)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return DatasetUsePolicy(
        {
            "schema_version": "1.0.0",
            "dataset_id": manifest["dataset_id"],
            "dataset_identity_hash": file_hash(directory / "manifest.json"),
            "classification": classification,
            "proxy_scope": "fixed_target",
            "proxy_boundary": "end_to_end",
            "scientific_evidence_status": "not_evaluated",
            "research_admission": {
                "status": "approved" if admitted_purpose else "not_evaluated",
                "purposes": (
                    [admitted_purpose]
                    if admitted_purpose
                    else ["software_development"]
                ),
            },
            "seal": seal,
        }
    )


def test_monolithic_test_split_fails_closed_until_role_is_selected(tmp_path):
    _write_archive(
        tmp_path,
        prefix="mixed",
        roles=("reference_test", "rare_stress_test", "reference_train"),
    )
    with pytest.raises(ValueError, match="mixes Reference Test and Rare/Stress Test"):
        TorchDamageDataset(tmp_path, split="test", fields=("case_x",))
    with pytest.raises(ValueError, match="exactly one"):
        TorchDamageDataset(
            tmp_path,
            split="test",
            sampling_role=("reference_test", "rare_stress_test"),
            fields=("case_x",),
        )

    reference = TorchDamageDataset(
        tmp_path,
        split="test",
        sampling_role="reference_test",
        fields=("case_x",),
    )
    assert len(reference) == 1
    assert reference[0]["sampling_role"] == "reference_test"
    assert reference[0]["stream_role"] == "reference"

    stress = TorchDamageDataset(
        tmp_path, split="test", stream_role="stress", fields=("case_x",)
    )
    assert len(stress) == 1
    assert stress[0]["sampling_role"] == "rare_stress_test"
    assert stress[0]["stream_role"] == "stress"


def test_role_filters_refuse_empty_or_unavailable_views(tmp_path):
    _write_archive(tmp_path, prefix="reference", roles=("reference_test",))
    with pytest.raises(ValueError, match="matched no cases"):
        TorchDamageDataset(
            tmp_path,
            split="test",
            sampling_role="rare_stress_test",
            fields=("case_x",),
        )
    with pytest.raises(ValueError, match="nonempty strings"):
        TorchDamageDataset(
            tmp_path, split="test", sampling_role=[], fields=("case_x",)
        )


def test_manifest_cannot_relabel_a_fixed_sampling_role_to_another_stream(tmp_path):
    manifest = _write_archive(
        tmp_path, prefix="conflicting-stream", roles=("reference_test",)
    )
    manifest["source_contracts"]["sampling_spec_document"] = {
        "stream_roles": [
            {"role": "reference_test", "stream": "stress"},
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting stream"):
        TorchDamageDataset(tmp_path, split="test", fields=("case_x",))


def test_loader_policy_binds_archive_operation_model_and_selected_role(tmp_path):
    _write_archive(
        tmp_path,
        prefix="mixed-policy",
        roles=("reference_test", "rare_stress_test"),
    )
    model_hash = "sha256:" + "a" * 64
    policy = _policy(
        tmp_path,
        "sealed_reference_test",
        model_hash=model_hash,
        admitted_purpose="research_test",
    )
    dataset = TorchDamageDataset(
        tmp_path,
        split="test",
        sampling_role="reference_test",
        fields=("case_x",),
        policy=policy,
        operation="final_evaluation",
        model_hash=model_hash,
    )
    assert len(dataset) == 1

    with pytest.raises(ValueError, match="requires an explicit operation"):
        TorchDamageDataset(
            tmp_path,
            split="test",
            sampling_role="reference_test",
            fields=("case_x",),
            policy=policy,
        )
    with pytest.raises(InputValidationError, match="different frozen model"):
        TorchDamageDataset(
            tmp_path,
            split="test",
            sampling_role="reference_test",
            fields=("case_x",),
            policy=policy,
            operation="final_evaluation",
            model_hash="sha256:" + "b" * 64,
        )
    with pytest.raises(ValueError, match="incompatible"):
        TorchDamageDataset(
            tmp_path,
            split="test",
            sampling_role="rare_stress_test",
            fields=("case_x",),
            policy=policy,
            operation="final_evaluation",
            model_hash=model_hash,
        )


def test_policy_rejects_disallowed_operation_and_unadmitted_research_purpose(tmp_path):
    _write_archive(tmp_path, prefix="stress-policy", roles=("rare_stress_test",))
    policy = _policy(tmp_path, "rare_stress_test")
    with pytest.raises(InputValidationError, match="does not permit"):
        TorchDamageDataset(
            tmp_path,
            split="test",
            sampling_role="rare_stress_test",
            fields=("case_x",),
            policy=policy,
            operation="training",
        )
    with pytest.raises(InputValidationError, match="lacks explicit admission"):
        TorchDamageDataset(
            tmp_path,
            split="test",
            sampling_role="rare_stress_test",
            fields=("case_x",),
            policy=policy,
            operation="stress_evaluation",
        )


def test_sharded_detection_crosses_shard_boundaries_and_survives_pickle(tmp_path):
    _write_sharded(tmp_path)
    with pytest.raises(ValueError, match="mixes Reference Test and Rare/Stress Test"):
        TorchShardedDamageDataset(tmp_path, split="test", fields=("case_x",))

    reference = TorchShardedDamageDataset(
        tmp_path,
        split="test",
        stream_role="reference",
        fields=("case_x",),
        max_cached_shards=1,
    )
    assert len(reference) == 2
    assert {reference[index]["stream_role"] for index in range(len(reference))} == {
        "reference"
    }
    restored = pickle.loads(pickle.dumps(reference))
    assert len(restored) == 2
    assert restored[0]["sampling_role"] == "reference_test"

    stress = TorchShardedDamageDataset(
        tmp_path, split="test", stream_role="stress", fields=("case_x",)
    )
    assert len(stress) == 2
    assert stress.metadata["sampling_role_names"] == tuple(ROLE_NAMES.tolist())
    assert stress[0]["sampling_role"] == "rare_stress_test"
