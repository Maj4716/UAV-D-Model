from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from damage_engine.dataset_v2 import validate_dataset_v2, write_dataset_v2
from damage_engine.errors import InputValidationError
from damage_engine.numpy_hurdle_baseline import run_numpy_p_hit_learning_curves
from damage_engine.utils import content_hash, file_hash
from tools.evaluate_dataset_v2_model_readiness import evaluate_model_readiness


def _trace(value: float) -> dict:
    return {
        "schema_version": "1.0.0",
        "semantics": (
            "joint_per_replicate_component_mechanism_states_before_tree_aggregation"
        ),
        "replicate_count": 2,
        "records": [
            {
                "replicate_index": index,
                "component_mechanism_states": {
                    "fragment": {
                        "c1": {"degraded": value, "failed": 0.5 * value},
                        "c2": {"degraded": 0.5 * value, "failed": 0.25 * value},
                    }
                },
            }
            for index in range(2)
        ],
    }


def _dataset(path: Path) -> Path:
    count = 36
    case_ids = np.asarray([f"case-{index:03d}" for index in range(count)])
    component_ids = np.asarray(["c1", "c2"])
    x1 = np.tile(np.linspace(-1.0, 1.0, 12), 3)
    x = np.column_stack([x1, x1**2])
    x[30:, 0] += 1000.0
    p_hit = np.column_stack(
        [0.1 + 0.8 * (x1 + 1.0) / 2.0, 0.9 - 0.8 * (x1 + 1.0) / 2.0]
    )
    p_degraded = p_hit.copy()
    p_failed = 0.5 * p_hit
    splits = np.asarray(["train"] * 24 + ["validation"] * 6 + ["test"] * 6)
    p_hit_mask = np.ones((count, 2), dtype=bool)
    p_hit_mask[30:] = False
    degraded_mask = p_hit_mask.copy()
    failed_mask = p_hit_mask.copy()
    p_hit_interval = np.stack([p_hit, p_hit], axis=-1)
    degraded_interval = np.stack([p_degraded, p_degraded], axis=-1)
    failed_interval = np.stack([p_failed, p_failed], axis=-1)
    p_hit_interval[30:] = [0.0, 1.0]
    degraded_interval[30:] = [0.0, 1.0]
    failed_interval[30:] = [0.0, 1.0]
    write_dataset_v2(
        path,
        dataset_id="model-readiness-test",
        inputs={
            "case_ids": case_ids,
            "x": x,
            "feature_names": np.asarray(["x1", "x1_squared"]),
        },
        component_state_traces=[
            {"case_id": case_id, "component_state_trace": _trace(float(value))}
            for case_id, value in zip(
                case_ids.tolist(), np.linspace(0.05, 0.75, count)
            )
        ],
        component_labels={
            "case_ids": case_ids,
            "component_ids": component_ids,
            "p_hit": p_hit,
            "p_hit_std_error": np.full((count, 2), 0.03),
            "p_hit_ci95": np.stack(
                [np.maximum(0.0, p_hit - 0.05), np.minimum(1.0, p_hit + 0.05)],
                axis=-1,
            ),
            "p_hit_identification_interval": p_hit_interval,
            "p_hit_mask": p_hit_mask,
            "p_degraded": p_degraded,
            "p_degraded_identification_interval": degraded_interval,
            "p_degraded_mask": degraded_mask,
            "p_failed": p_failed,
            "p_failed_identification_interval": failed_interval,
            "p_failed_mask": failed_mask,
            "label_mask": p_hit_mask & degraded_mask & failed_mask,
        },
        sampling_weights={
            "case_ids": case_ids,
            "selection_probability": np.full(count, 0.5),
            "design_weight": 1.0 + 0.1 * (np.arange(count) % 3),
            "target_measure_weight": np.ones(count),
            "training_weight": np.ones(count),
            "split": splits,
            "design_seed": np.full(count, 23, dtype=np.int64),
        },
        input_contract={"schema_version": "model-readiness-test-input@1"},
        label_spec_id="model-readiness-test-label@1",
        source_code_hash="sha256:" + "9" * 64,
    )
    (path / "sealed_test.json").write_text(
        json.dumps(
            {
                "schema_version": "dataset-v2-sealed-test@1.0.0",
                "test_open_count": 0,
                "test_labels_present_in_core": False,
                "test_result_json_lines_decoded": 0,
            }
        ),
        encoding="utf-8",
    )
    return path


def _admission(path: Path, dataset: Path) -> Path:
    validation = validate_dataset_v2(dataset, recompute_views=False)
    document = {
        "schema_version": "dataset-v2-task-admission@1.0.0",
        "dataset_identity": {
            "dataset_id": validation["dataset_id"],
            "dataset_manifest_hash": validation["manifest_hash"],
            "audit_report_hash": "sha256:" + "a" * 64,
        },
        "sealed_test": {
            "test_open_count": 0,
            "test_result_json_lines_decoded": 0,
        },
        "component_task_admission": {
            task: {
                "admitted_component_count": 2,
                "admitted_component_ids": ["c1", "c2"],
            }
            for task in ("p_hit", "p_degraded", "p_failed")
        },
    }
    document["artifact_hash"] = content_hash(document)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    dataset = _dataset(tmp_path / "dataset")
    admission = _admission(tmp_path / "task_admission.json", dataset)
    baseline = tmp_path / "baseline"
    run_numpy_p_hit_learning_curves(
        dataset,
        baseline,
        fractions=(0.25, 1.0),
        seeds=(7,),
        epochs=16,
        hidden_size=6,
        task_admission_path=admission,
    )
    return (
        dataset,
        admission,
        baseline / "selected_model.npz",
        baseline / "results.json",
    )


def test_model_readiness_reconstructs_validation_and_preserves_seal(
    tmp_path: Path,
) -> None:
    dataset, admission, model, results = _artifacts(tmp_path)
    seal_hash = file_hash(dataset / "sealed_test.json")
    report = evaluate_model_readiness(dataset, model, results, admission)

    assert report["scope"] == "admitted_p_hit_train_validation_only"
    assert report["admission"]["admitted_component_count"] == 2
    assert report["validation_metrics"]["reconstruction_absolute_error"] <= 1.0e-12
    assert report["validation_metrics"]["skill_evaluable_component_count"] == 2
    assert len(report["per_component"]) == 2
    assert report["learning_curve"]["quarter_fraction_brier_mean"] >= 0.0
    assert report["sealed_test"]["test_predictions_computed"] is False
    assert report["sealed_test"]["test_result_json_lines_decoded"] == 0
    assert file_hash(dataset / "sealed_test.json") == seal_hash
    payload = dict(report)
    report_hash = payload.pop("report_hash")
    assert report_hash == content_hash(payload)


def test_model_readiness_rejects_opened_test(tmp_path: Path) -> None:
    dataset, admission, model, results = _artifacts(tmp_path)
    seal_path = dataset / "sealed_test.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["test_open_count"] = 1
    seal_path.write_text(json.dumps(seal), encoding="utf-8")

    with pytest.raises(InputValidationError, match="already been opened"):
        evaluate_model_readiness(dataset, model, results, admission)


def test_model_readiness_rejects_different_admission(tmp_path: Path) -> None:
    dataset, admission, model, results = _artifacts(tmp_path)
    document = json.loads(admission.read_text(encoding="utf-8"))
    document.pop("artifact_hash")
    for row in document["component_task_admission"].values():
        row["admitted_component_count"] = 1
        row["admitted_component_ids"] = ["c1"]
    document["artifact_hash"] = content_hash(document)
    admission.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(InputValidationError, match="different task-admission"):
        evaluate_model_readiness(dataset, model, results, admission)

