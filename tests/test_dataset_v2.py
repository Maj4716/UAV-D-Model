from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from damage_engine.dataset_v2 import (
    load_dataset_v2_core,
    validate_dataset_v2,
    write_dataset_v2,
    write_tree_view,
)
from damage_engine.errors import InputValidationError


def _trace(probability: float) -> dict:
    return {
        "schema_version": "1.0.0",
        "semantics": "joint_per_replicate_component_mechanism_states_before_tree_aggregation",
        "replicate_count": 2,
        "records": [
            {
                "replicate_index": 0,
                "component_mechanism_states": {
                    "fragment": {
                        "c1": {"degraded": probability, "failed": probability},
                        "c2": {"degraded": 0.5 * probability, "failed": 0.5 * probability},
                    }
                },
            },
            {
                "replicate_index": 1,
                "component_mechanism_states": {
                    "fragment": {
                        "c1": {"degraded": probability + 0.1, "failed": probability + 0.1},
                        "c2": {
                            "degraded": 0.5 * probability + 0.1,
                            "failed": 0.5 * probability + 0.1,
                        },
                    }
                },
            },
        ],
    }


def _candidate(authority: str = "unreviewed") -> dict:
    return {
        "schema_version": "offline-monotone-tree-candidate@1.0.0",
        "candidate_id": "identity-c1",
        "status": "reviewed" if authority == "approved" else "provisional",
        "semantic_authority": {
            "status": authority,
            "source": "domain-review-001" if authority == "approved" else None,
            "scope": "c1 fragment failed output",
        },
        "inputs": [
            {
                "input_id": "c1_failed",
                "component_id": "c1",
                "mechanism": "fragment",
                "damage_state": "failed",
            }
        ],
        "outputs": [
            {
                "output_id": "c1_failed_view",
                "gate": {"kind": "identity", "inputs": ["c1_failed"]},
            }
        ],
    }


def make_dataset(path: Path) -> Path:
    case_ids = np.asarray(["a", "b", "c", "d"])
    component_ids = np.asarray(["c1", "c2"])
    probabilities = np.asarray(
        [[0.20, 0.10], [0.30, 0.15], [0.60, 0.30], [0.70, 0.35]]
    )
    write_dataset_v2(
        path,
        dataset_id="test-dataset-v2",
        inputs={
            "case_ids": case_ids,
            "x": np.arange(8, dtype=float).reshape(4, 2),
            "feature_names": np.asarray(["x1", "x2"]),
        },
        component_state_traces=[
            {"case_id": case_id, "component_state_trace": _trace(float(probability))}
            for case_id, probability in zip(case_ids.tolist(), [0.2, 0.3, 0.6, 0.7])
        ],
        component_labels={
            "case_ids": case_ids,
            "component_ids": component_ids,
            "p_hit": probabilities,
            "p_hit_std_error": np.full((4, 2), 0.02),
            "p_hit_ci95": np.stack(
                [np.maximum(0.0, probabilities - 0.05), np.minimum(1.0, probabilities + 0.05)],
                axis=-1,
            ),
            "p_degraded": probabilities,
            "p_failed": probabilities * 0.8,
            "label_mask": np.ones((4, 2), dtype=bool),
        },
        sampling_weights={
            "case_ids": case_ids,
            "selection_probability": np.full(4, 0.5),
            "design_weight": np.ones(4),
            "target_measure_weight": np.ones(4),
            "training_weight": np.ones(4),
            "split": np.asarray(["development", "development", "validation", "validation"]),
            "design_seed": np.asarray([11, 11, 12, 12], dtype=np.int64),
        },
        input_contract={"schema_version": "test-inputs@1"},
        label_spec_id="test-labels@1",
        source_code_hash="sha256:" + "1" * 64,
    )
    return path


def test_dataset_v2_core_and_view_are_hash_checked_and_tree_separated(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "dataset")
    manifest_before = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    core_hashes_before = {
        key: value["sha256"] for key, value in manifest_before["core_files"].items()
    }
    report = write_tree_view(dataset, _candidate())
    assert report["valid"] is True
    assert report["view_count"] == 1
    manifest_after = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    assert {
        key: value["sha256"] for key, value in manifest_after["core_files"].items()
    } == core_hashes_before
    assert set(load_dataset_v2_core(dataset)) == {
        "inputs",
        "component_labels",
        "sampling_weights",
        "component_state_traces",
    }
    view_entry = next(iter(manifest_after["views"].values()))
    view_path = dataset / view_entry["path"]
    view_path.write_bytes(view_path.read_bytes() + b"tamper")
    with pytest.raises(InputValidationError, match="view hash mismatch"):
        validate_dataset_v2(dataset)


def test_dataset_v2_rejects_tree_fields_in_core_archives(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "dataset")
    inputs_path = dataset / "core" / "inputs.npz"
    with np.load(inputs_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["tree_outputs"] = np.zeros((4, 1))
    with inputs_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    from damage_engine.utils import file_hash

    manifest["core_files"]["inputs"]["sha256"] = file_hash(inputs_path)
    manifest.pop("manifest_hash")
    from damage_engine.utils import content_hash

    manifest["manifest_hash"] = content_hash(manifest)
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(InputValidationError, match="invalid field set"):
        validate_dataset_v2(dataset)


def test_dataset_v2_checks_state_order_on_per_target_observations(
    tmp_path: Path,
) -> None:
    case_ids = np.asarray(["a"])
    component_ids = np.asarray(["c1"])
    labels = {
        "case_ids": case_ids,
        "component_ids": component_ids,
        "p_hit": np.asarray([[0.0]]),
        "p_hit_std_error": np.asarray([[0.0]]),
        "p_hit_ci95": np.asarray([[[0.0, 0.0]]]),
        "p_hit_mask": np.asarray([[False]]),
        "p_degraded": np.asarray([[0.2]]),
        "p_degraded_mask": np.asarray([[True]]),
        "p_failed": np.asarray([[0.3]]),
        "p_failed_mask": np.asarray([[True]]),
        "label_mask": np.asarray([[False]]),
    }

    with pytest.raises(InputValidationError, match="both state targets are observed"):
        write_dataset_v2(
            tmp_path / "invalid-state-order",
            dataset_id="invalid-state-order",
            inputs={
                "case_ids": case_ids,
                "x": np.asarray([[0.0]]),
                "feature_names": np.asarray(["x"]),
            },
            component_state_traces=[
                {"case_id": "a", "component_state_trace": _trace(0.0)}
            ],
            component_labels=labels,
            sampling_weights={
                "case_ids": case_ids,
                "selection_probability": np.asarray([1.0]),
                "design_weight": np.asarray([1.0]),
                "target_measure_weight": np.asarray([1.0]),
                "training_weight": np.asarray([1.0]),
                "split": np.asarray(["train"]),
                "design_seed": np.asarray([1], dtype=np.int64),
            },
            input_contract={"schema_version": "test-inputs@1"},
            label_spec_id="test-labels@1",
            source_code_hash="sha256:" + "1" * 64,
        )
