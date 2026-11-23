from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from damage_engine.dataset_v2 import write_dataset_v2
from damage_engine.errors import InputValidationError
from damage_engine.numpy_hurdle_baseline import (
    _initialize_parameters,
    _loss_and_gradients,
    run_numpy_p_hit_learning_curves,
)
from damage_engine.utils import content_hash, file_hash
from tools.run_dataset_v2_numpy_baseline import main as numpy_baseline_main


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


def _dataset(path: Path, *, validation_observable: bool = True) -> Path:
    count = 30
    case_ids = np.asarray([f"case-{index:03d}" for index in range(count)])
    component_ids = np.asarray(["c1", "c2"])
    x1 = np.tile(np.linspace(-1.0, 1.0, 10), 3)
    x = np.column_stack([x1, x1**2])
    # Test covariates are deliberately extreme.  Their values must not affect
    # normalization fitted to the first 18 training cases.
    x[24:, 0] += 1000.0
    p_first = np.where(x1 <= -0.25, 0.0, np.minimum(0.9, 0.2 + 0.6 * x1))
    p_second = np.where(x1 >= 0.25, 0.0, np.minimum(0.9, 0.2 - 0.6 * x1))
    p_hit = np.clip(np.column_stack([p_first, p_second]), 0.0, 1.0)
    p_degraded = np.maximum(p_hit, 0.2 * p_hit)
    p_failed = 0.5 * p_degraded

    splits = np.asarray(["train"] * 18 + ["validation"] * 6 + ["test"] * 6)
    p_hit_mask = np.ones((count, 2), dtype=bool)
    p_hit_mask[3, 0] = False
    p_hit_mask[21, 1] = False
    if not validation_observable:
        p_hit_mask[18:24] = False
    degraded_mask = np.ones_like(p_hit_mask)
    failed_mask = np.ones_like(p_hit_mask)
    p_hit_mask[24:] = False
    degraded_mask[24:] = False
    failed_mask[24:] = False
    label_mask = p_hit_mask & degraded_mask & failed_mask

    point_interval = np.stack([p_hit, p_hit], axis=-1)
    point_interval[3, 0] = [0.00, 0.45]
    point_interval[21, 1] = [0.20, 0.80]
    if not validation_observable:
        point_interval[18:24] = [0.0, 1.0]
    point_interval[24:] = [0.0, 1.0]
    ci95 = np.stack(
        [np.maximum(0.0, p_hit - 0.05), np.minimum(1.0, p_hit + 0.05)],
        axis=-1,
    )
    write_dataset_v2(
        path,
        dataset_id="numpy-p-hit-test",
        inputs={
            "case_ids": case_ids,
            "x": x,
            "feature_names": np.asarray(["x1", "x1_squared"]),
        },
        component_state_traces=[
            {"case_id": case_id, "component_state_trace": _trace(float(value))}
            for case_id, value in zip(case_ids.tolist(), np.linspace(0.05, 0.75, count))
        ],
        component_labels={
            "case_ids": case_ids,
            "component_ids": component_ids,
            "p_hit": p_hit,
            "p_hit_std_error": np.full((count, 2), 0.03),
            "p_hit_ci95": ci95,
            "p_hit_identification_interval": point_interval,
            "p_hit_mask": p_hit_mask,
            "p_degraded": p_degraded,
            "p_degraded_identification_interval": np.stack(
                [p_degraded, p_degraded], axis=-1
            ),
            "p_degraded_mask": degraded_mask,
            "p_failed": p_failed,
            "p_failed_identification_interval": np.stack(
                [p_failed, p_failed], axis=-1
            ),
            "p_failed_mask": failed_mask,
            "label_mask": label_mask,
        },
        sampling_weights={
            "case_ids": case_ids,
            "selection_probability": np.full(count, 0.5),
            "design_weight": 1.0 + 0.1 * (np.arange(count) % 3),
            "target_measure_weight": np.ones(count),
            "training_weight": np.ones(count),
            "split": splits,
            "design_seed": np.full(count, 19, dtype=np.int64),
        },
        input_contract={"schema_version": "numpy-baseline-test-input@1"},
        label_spec_id="numpy-baseline-test-label@1",
        source_code_hash="sha256:" + "7" * 64,
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


def _task_admission(path: Path, dataset: Path) -> Path:
    from damage_engine.dataset_v2 import validate_dataset_v2

    validation = validate_dataset_v2(dataset, recompute_views=False)
    document = {
        "schema_version": "dataset-v2-task-admission@1.0.0",
        "dataset_identity": {
            "dataset_id": validation["dataset_id"],
            "dataset_manifest_hash": validation["manifest_hash"],
            "audit_report_hash": "sha256:" + "c" * 64,
        },
        "selection_contract": {
            "scope": "train_validation_support_only",
            "source_audit_schema_version": "test-audit@1",
        },
        "sealed_test": {
            "test_open_count": 0,
            "test_result_json_lines_decoded": 0,
        },
        "component_task_admission": {
            target: {
                "thresholds": {},
                "admitted_component_count": 1,
                "admitted_component_ids": ["c1"],
                "rejected_component_count": 1,
                "rejected_components": [],
                "component_order_hash": "sha256:" + "d" * 64,
            }
            for target in ("p_hit", "p_degraded", "p_failed")
        },
    }
    document["artifact_hash"] = content_hash(document)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_numpy_loss_gradient_and_vacuous_interval_contract() -> None:
    rng = np.random.default_rng(5)
    parameters = _initialize_parameters(2, 3, 2, rng)
    x = np.asarray([[-0.7, 0.2], [0.4, 0.8], [0.9, -0.3]])
    target = np.asarray([[0.2, 0.8], [0.6, 0.1], [0.9, 0.3]])
    mask = np.asarray([[True, False], [False, True], [True, True]])
    interval = np.asarray(
        [
            [[0.2, 0.2], [0.90, 1.00]],
            [[0.00, 0.10], [0.1, 0.1]],
            [[0.9, 0.9], [0.3, 0.3]],
        ]
    )
    weight = np.asarray([1.0, 2.0, 0.7])
    loss, gradients, diagnostics = _loss_and_gradients(
        parameters,
        x,
        target,
        mask,
        interval,
        weight,
        interval_loss_weight=0.25,
        l2_penalty=0.01,
    )
    assert diagnostics["informative_interval_count"] == 2
    epsilon = 1.0e-6
    for name, parameter in parameters.items():
        for index in np.ndindex(parameter.shape):
            original = float(parameter[index])
            parameter[index] = original + epsilon
            plus = _loss_and_gradients(
                parameters,
                x,
                target,
                mask,
                interval,
                weight,
                interval_loss_weight=0.25,
                l2_penalty=0.01,
            )[0]
            parameter[index] = original - epsilon
            minus = _loss_and_gradients(
                parameters,
                x,
                target,
                mask,
                interval,
                weight,
                interval_loss_weight=0.25,
                l2_penalty=0.01,
            )[0]
            parameter[index] = original
            assert gradients[name][index] == pytest.approx(
                (plus - minus) / (2.0 * epsilon), abs=2.0e-6
            )
    assert np.isfinite(loss)

    vacuous_mask = np.asarray([[True, False], [True, False], [True, False]])
    vacuous_interval = np.zeros((3, 2, 2), dtype=np.float64)
    vacuous_interval[:, 1] = [0.0, 1.0]
    with_interval = _loss_and_gradients(
        parameters,
        x,
        target,
        vacuous_mask,
        vacuous_interval,
        weight,
        interval_loss_weight=10.0,
        l2_penalty=0.0,
    )
    without_interval = _loss_and_gradients(
        parameters,
        x,
        target,
        vacuous_mask,
        vacuous_interval,
        weight,
        interval_loss_weight=0.0,
        l2_penalty=0.0,
    )
    assert with_interval[2]["informative_interval_count"] == 0
    assert with_interval[0] == pytest.approx(without_interval[0])
    for name in parameters:
        np.testing.assert_allclose(with_interval[1][name], without_interval[1][name])


def test_numpy_baseline_is_train_normalized_and_keeps_test_sealed(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "dataset")
    seal_hash = file_hash(dataset / "sealed_test.json")
    output = tmp_path / "numpy-baseline"
    report = run_numpy_p_hit_learning_curves(
        dataset,
        output,
        fractions=(0.5, 1.0),
        seeds=(3, 5),
        epochs=35,
        hidden_size=8,
    )
    assert report["status"] == "completed"
    assert len(report["runs"]) == 4
    assert report["selected_model"]["validation_metrics"]["brier"] is not None
    assert report["learning_curve"]["1.0"]["brier"]["count"] == 2
    assert report["learning_curve"]["1.0"]["brier"]["variance"] >= 0.0
    assert report["test_access_policy"] == {
        "test_identity_count": 6,
        "test_labels_present_in_core": False,
        "labels_used_in_loss": False,
        "labels_used_in_metrics": False,
        "labels_used_in_selection": False,
        "test_result_json_lines_decoded": 0,
        "open_count": 0,
    }
    assert file_hash(dataset / "sealed_test.json") == seal_hash
    assert (output / "results.json").is_file()
    assert (output / "selected_model.npz").is_file()
    with np.load(output / "selected_model.npz", allow_pickle=False) as model:
        expected = np.mean(
            np.column_stack(
                [
                    np.tile(np.linspace(-1.0, 1.0, 10), 3)[:18],
                    np.tile(np.linspace(-1.0, 1.0, 10), 3)[:18] ** 2,
                ]
            ),
            axis=0,
        )
        np.testing.assert_allclose(model["train_mean"], expected)
    assert all(
        row["validation"]["point_observation_count"] > 0
        for row in report["support_by_component"]
    )


def test_numpy_baseline_uses_only_frozen_admitted_p_hit_components(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "dataset")
    admission = _task_admission(tmp_path / "admission.json", dataset)
    report = run_numpy_p_hit_learning_curves(
        dataset,
        tmp_path / "numpy-admitted",
        fractions=(1.0,),
        seeds=(3,),
        epochs=8,
        hidden_size=6,
        task_admission_path=admission,
    )

    assert report["task_admission"]["mode"] == "frozen_component_task_subset"
    assert report["selected_model"]["validation_metrics"][
        "point_observation_count"
    ] == 6
    by_id = {row["component_id"]: row for row in report["support_by_component"]}
    assert by_id["c1"]["admitted_for_p_hit"] is True
    assert by_id["c2"]["admitted_for_p_hit"] is False
    assert by_id["c2"]["validation"]["point_observation_count"] == 0


def test_numpy_baseline_cli_and_empty_validation_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _dataset(tmp_path / "dataset")
    output = tmp_path / "cli-output"
    assert (
        numpy_baseline_main(
            [
                "--dataset",
                str(dataset),
                "--output",
                str(output),
                "--fraction",
                "1.0",
                "--seed",
                "13",
                "--epochs",
                "12",
                "--hidden-size",
                "6",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "completed"
    assert summary["run_count"] == 1
    assert summary["test_access_policy"]["open_count"] == 0

    empty = _dataset(tmp_path / "empty-validation", validation_observable=False)
    with pytest.raises(InputValidationError, match="no point-identified"):
        run_numpy_p_hit_learning_curves(
            empty,
            tmp_path / "should-not-exist",
            fractions=(1.0,),
            seeds=(7,),
            epochs=2,
            hidden_size=4,
        )
