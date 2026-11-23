from __future__ import annotations

import gzip
import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest

from damage_engine.dataset_v2 import validate_dataset_v2, write_dataset_v2
from damage_engine.dataset_v2_sampler import convert_sampler_pilot_to_dataset_v2
from damage_engine.errors import InputValidationError
from damage_engine.hurdle_baseline import (
    run_hurdle_learning_curves,
    run_target_encoder_ablation,
)
from damage_engine.engine import DamageEngine
from damage_engine.utils import content_hash
from tools.run_dataset_v2_hurdle_baseline import main as hurdle_baseline_main
from tools.run_proxy_learning_ablation import (
    ARMS,
    _build_inadmissible_proxy_learning_report,
)


def _trace(value: float) -> dict:
    return {
        "schema_version": "1.0.0",
        "semantics": "joint_per_replicate_component_mechanism_states_before_tree_aggregation",
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


def _extended_dataset(
    path: Path, *, observable_validation_p_hit: bool = True
) -> Path:
    count = 12
    case_ids = np.asarray([f"case-{index}" for index in range(count)])
    components = np.asarray(["c1", "c2"])
    base = np.linspace(0.05, 0.8, count)[:, None]
    p_hit = np.hstack([base, 0.5 * base])
    mask = np.ones((count, 2), dtype=bool)
    interval = np.stack(
        [np.maximum(0.0, p_hit - 0.05), np.minimum(1.0, p_hit + 0.05)],
        axis=-1,
    )
    p_degraded = 0.8 * p_hit
    p_degraded_interval = np.stack(
        [
            np.maximum(0.0, p_degraded - 0.05),
            np.minimum(1.0, p_degraded + 0.05),
        ],
        axis=-1,
    )
    p_failed = 0.4 * p_hit
    p_failed_interval = np.stack(
        [
            np.maximum(0.0, p_failed - 0.05),
            np.minimum(1.0, p_failed + 0.05),
        ],
        axis=-1,
    )
    conditional_mask = p_hit > 0.1
    label_mask = mask.copy()
    if not observable_validation_p_hit:
        label_mask[6:9] = False
    write_dataset_v2(
        path,
        dataset_id="extended-hurdle-test",
        inputs={
            "case_ids": case_ids,
            "x": np.column_stack(
                [np.linspace(-1.0, 1.0, count), np.linspace(1.0, -1.0, count)]
            ),
            "feature_names": np.asarray(["x1", "x2"]),
        },
        component_state_traces=[
            {"case_id": case_id, "component_state_trace": _trace(float(value))}
            for case_id, value in zip(case_ids.tolist(), base[:, 0].tolist(), strict=True)
        ],
        component_labels={
            "case_ids": case_ids,
            "component_ids": components,
            "p_hit": p_hit,
            "p_hit_std_error": np.full((count, 2), 0.03),
            "p_hit_ci95": interval,
            "p_hit_success_count": np.rint(16 * p_hit).astype(np.int64),
            "p_hit_trial_count": np.full((count, 2), 16, dtype=np.int64),
            "p_hit_identification_interval": interval,
            "p_degraded": p_degraded,
            "p_degraded_std_error": np.full((count, 2), 0.02),
            "p_degraded_ci95": p_degraded_interval,
            "p_degraded_identification_interval": p_degraded_interval,
            "p_failed": p_failed,
            "p_failed_std_error": np.full((count, 2), 0.01),
            "p_failed_ci95": p_failed_interval,
            "p_failed_identification_interval": p_failed_interval,
            "conditional_log1p_energy": np.log1p(100.0 * p_hit),
            "conditional_log1p_energy_std_error": np.full((count, 2), 0.05),
            "conditional_log1p_energy_mask": conditional_mask,
            "conditional_log1p_hit_count": np.log1p(10.0 * p_hit),
            "conditional_log1p_hit_count_std_error": np.full((count, 2), 0.03),
            "conditional_log1p_hit_count_mask": conditional_mask,
            "derived_support": (p_hit >= 0.25).astype(np.int8),
            "derived_support_mask": mask,
            "label_mask": label_mask,
        },
        sampling_weights={
            "case_ids": case_ids,
            "selection_probability": np.full(count, 0.5),
            "design_weight": np.ones(count),
            "target_measure_weight": np.ones(count),
            "training_weight": np.ones(count),
            "split": np.asarray(["train"] * 6 + ["validation"] * 3 + ["test"] * 3),
            "design_seed": np.full(count, 11, dtype=np.int64),
        },
        input_contract={"schema_version": "test-input@1"},
        label_spec_id="hurdle-test@1",
        source_code_hash="sha256:" + "3" * 64,
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


def _task_admission(
    path: Path,
    dataset: Path,
    admitted: list[str],
    *,
    schema_version: str = "dataset-v2-task-admission@1.0.0",
    conditional_admitted: list[str] | None = None,
) -> Path:
    validation = validate_dataset_v2(dataset, recompute_views=False)
    document = {
        "schema_version": schema_version,
        "dataset_identity": {
            "dataset_id": validation["dataset_id"],
            "dataset_manifest_hash": validation["manifest_hash"],
            "audit_report_hash": "sha256:" + "a" * 64,
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
                "admitted_component_count": len(admitted),
                "admitted_component_ids": admitted,
                "rejected_component_count": 2 - len(admitted),
                "rejected_components": [],
                "component_order_hash": "sha256:" + "b" * 64,
            }
            for target in ("p_hit", "p_degraded", "p_failed")
        },
    }
    if schema_version == "dataset-v2-task-admission@1.1.0":
        selected = admitted if conditional_admitted is None else conditional_admitted
        for target in (
            "conditional_log1p_energy",
            "conditional_log1p_hit_count",
        ):
            document["component_task_admission"][target] = {
                "thresholds": {},
                "admitted_component_count": len(selected),
                "admitted_component_ids": selected,
                "rejected_component_count": 2 - len(selected),
                "rejected_components": [],
                "component_order_hash": "sha256:" + "b" * 64,
            }
    document["artifact_hash"] = content_hash(document)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_encoder_ablation_cli_records_empty_validation_gate(
    tmp_path: Path,
) -> None:
    dataset = _extended_dataset(
        tmp_path / "dataset", observable_validation_p_hit=False
    )
    output = tmp_path / "encoder-ablation"
    assert hurdle_baseline_main(
        [
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--encoder-ablation",
        ]
    ) == 0

    report = json.loads(
        (output / "ablation_results.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "all_bundles_inadmissible"
    assert report["reason"]["code"] == "no_observable_validation_p_hit"
    assert report["selected_bundle_id"] is None
    assert report["observation_counts"]["validation"][
        "p_hit_observation_count"
    ] == 0
    assert report["observation_counts"]["validation"][
        "conditional_energy_observation_count"
    ] > 0
    assert len(report["candidate_bundles"]) == 4
    assert all(
        candidate["status"] == "inadmissible"
        for candidate in report["candidate_bundles"]
    )
    assert report["test_access_policy"]["open_count"] == 0


def test_proxy_learning_records_all_arms_when_validation_gate_is_empty(
    tmp_path: Path,
) -> None:
    dataset = _extended_dataset(
        tmp_path / "dataset", observable_validation_p_hit=False
    )
    target = tmp_path / "target.json"
    physics = tmp_path / "physics.json"
    target.write_text("{}\n", encoding="utf-8")
    physics.write_text("{}\n", encoding="utf-8")

    report = _build_inadmissible_proxy_learning_report(
        dataset=dataset,
        target=target,
        physics=physics,
        proxy_pool_size=512,
        fractions=(0.25, 0.50, 1.0),
        seeds=(17, 29, 43),
        epochs=120,
        pretrain_epochs=60,
        x1_gate_status="failed",
    )

    assert report is not None
    assert report["status"] == "all_arms_inadmissible"
    assert report["reason"]["code"] == "no_observable_validation_p_hit"
    assert report["selected_arm"] is None
    assert report["arms"] == list(ARMS)
    assert len(report["arm_results"]) == 4
    assert all(row["validation_brier"] is None for row in report["arm_results"])
    assert report["proxy_pretraining_pool"] == {
        "requested_count": 512,
        "status": "skipped_primary_validation_gate",
        "teacher_labels_used": False,
    }
    assert report["test_access_policy"]["test_result_json_lines_decoded"] == 0
    assert report["test_access_policy"]["open_count"] == 0


def test_extended_hurdle_core_and_validation_only_baseline(tmp_path: Path) -> None:
    dataset = _extended_dataset(tmp_path / "dataset")
    assert validate_dataset_v2(dataset)["valid"] is True
    output = tmp_path / "baseline"
    result = run_hurdle_learning_curves(
        dataset,
        output,
        fractions=[1.0],
        seeds=[7],
        epochs=5,
        hidden_size=8,
    )
    assert result["split_counts"]["test_sealed_unread"] == 3
    assert result["test_access_policy"]["open_count"] == 0
    assert result["selected_frozen_model"]["test_labels_evaluated"] is False
    assert result["training_readiness"]["formal_neural_training_ready"] is False
    assert result["target_encoder_methods"] == {
        "energy": "empirical_quantile_normal",
        "hit_count": "box_cox",
    }
    assert result["runs"][0]["target_encoders"]["energy"]["source_split"] == (
        "reference_train"
    )
    assert (output / "frozen_model.pt").is_file()


def test_hurdle_baseline_restricts_loss_selection_and_metrics_to_admitted_tasks(
    tmp_path: Path,
) -> None:
    dataset = _extended_dataset(tmp_path / "dataset")
    admission = _task_admission(tmp_path / "admission.json", dataset, ["c1"])
    result = run_hurdle_learning_curves(
        dataset,
        tmp_path / "baseline",
        fractions=[1.0],
        seeds=[7],
        epochs=3,
        hidden_size=8,
        task_admission_path=admission,
    )

    assert result["task_admission"]["mode"] == "frozen_component_task_subset"
    assert result["task_admission"]["admitted_component_count_by_task"] == {
        "p_hit": 1,
        "p_degraded": 1,
        "p_failed": 1,
        "conditional_log1p_energy": 1,
        "conditional_log1p_hit_count": 1,
    }
    assert result["task_admission"]["component_task_mask_source"][
        "conditional_log1p_energy"
    ] == "inherited_from_p_hit_v1_compatibility"
    metrics = result["runs"][0]["validation_metrics"]
    assert metrics["p_hit"]["observation_count"] == 3
    assert metrics["p_degraded"]["observation_count"] == 3
    assert metrics["p_failed"]["observation_count"] == 3
    assert metrics["conditional_log1p_energy"]["observation_count"] == 3
    assert metrics["conditional_log1p_hit_count"]["observation_count"] == 3
    assert result["training_readiness"]["p_hit_admitted_component_count"] == 1
    assert result["selected_frozen_model"]["task_admission"]["artifact_hash"] == (
        json.loads(admission.read_text(encoding="utf-8"))["artifact_hash"]
    )


def test_hurdle_baseline_uses_v1p1_explicit_conditional_task_masks(
    tmp_path: Path,
) -> None:
    dataset = _extended_dataset(tmp_path / "dataset")
    admission = _task_admission(
        tmp_path / "admission.json",
        dataset,
        ["c1", "c2"],
        schema_version="dataset-v2-task-admission@1.1.0",
        conditional_admitted=["c2"],
    )
    result = run_hurdle_learning_curves(
        dataset,
        tmp_path / "baseline",
        fractions=[1.0],
        seeds=[7],
        epochs=3,
        hidden_size=8,
        task_admission_path=admission,
    )

    metrics = result["runs"][0]["validation_metrics"]
    assert metrics["p_hit"]["observation_count"] == 6
    assert metrics["conditional_log1p_energy"]["observation_count"] == 3
    assert metrics["conditional_log1p_hit_count"]["observation_count"] == 3
    assert result["task_admission"]["component_task_mask_source"][
        "conditional_log1p_energy"
    ] == "explicit"
    support = result["training_readiness"]["conditional_component_support"]
    assert support["conditional_log1p_energy"] == {
        "admitted_component_count": 1,
        "train_observation_count_per_admitted_component": {
            "minimum": 3,
            "median": 3.0,
            "maximum": 3,
        },
        "validation_observation_count_per_admitted_component": {
            "minimum": 3,
            "median": 3.0,
            "maximum": 3,
        },
    }
    gates = result["training_readiness"]["gates"]
    assert gates[
        "conditional_log1p_energy_minimum_20_train_observations_per_component"
    ] is False
    assert gates[
        "conditional_log1p_energy_minimum_5_validation_observations_per_component"
    ] is False


def test_target_encoder_ablation_is_train_only_and_keeps_test_sealed(
    tmp_path: Path,
) -> None:
    dataset = _extended_dataset(tmp_path / "dataset")
    bundles = {
        "identity": {"energy": "identity", "hit_count": "identity"},
        "box": {"energy": "box_cox", "hit_count": "box_cox"},
        "quantile": {
            "energy": "empirical_quantile_normal",
            "hit_count": "empirical_quantile_normal",
        },
    }
    output = tmp_path / "encoder-ablation"
    report = run_target_encoder_ablation(
        dataset,
        output,
        bundles=bundles,
        fractions=[1.0],
        seeds=[7],
        epochs=3,
        hidden_size=8,
    )

    assert report["selection_scope"] == "reference_validation_only"
    assert report["test_access_policy"]["open_count"] == 0
    assert report["selected_bundle_id"] in bundles
    assert {row["bundle_id"] for row in report["candidate_bundles"]} == set(
        bundles
    )
    for name, methods in bundles.items():
        candidate = json.loads(
            (output / name / "results.json").read_text(encoding="utf-8")
        )
        assert candidate["target_encoder_methods"] == methods
        assert candidate["test_access_policy"]["open_count"] == 0
        for contract in candidate["runs"][0]["target_encoders"].values():
            assert contract["fit_scope"] == "train_only"
            assert contract["source_split"] == "reference_train"
            assert contract["training_observation_count"] <= 12


def test_hurdle_baseline_rejects_double_log1p_target_encoding(tmp_path: Path) -> None:
    dataset = _extended_dataset(tmp_path / "dataset")
    with pytest.raises(InputValidationError, match="already defined on a log1p scale"):
        run_hurdle_learning_curves(
            dataset,
            tmp_path / "baseline",
            fractions=[1.0],
            seeds=[7],
            epochs=1,
            hidden_size=8,
            energy_encoder_method="log1p",
        )


def test_sampler_converter_fails_closed_without_component_trace(tmp_path: Path) -> None:
    source = tmp_path / "pilot" / "formal_dataset"
    shard = source / "shards" / "part-000000"
    shard.mkdir(parents=True)
    manifest = {
        "dataset_id": "source",
        "engine": {"code_hash": "sha256:" + "4" * 64},
        "source_contracts": {
            "label_spec_id": "label@1",
            "sampling_spec_document": {"seeds": {"design": 17}},
        },
        "model_input_contract": {
            "tensor_field": "model_x",
            "mask_field": "model_mask",
        },
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (shard / "tensors.npz").open("wb") as stream:
        np.savez_compressed(
            stream,
            case_ids=np.asarray(["case-1"]),
            split_names=np.asarray(["train"]),
            input_frozen_splits=np.asarray(["train"]),
            model_x=np.zeros((1, 2)),
            model_mask=np.ones((1, 2), dtype=bool),
            model_feature_names=np.asarray(["x1", "x2"]),
            component_ids=np.asarray(["c1"]),
            sampling_weight_mask=np.asarray([True]),
            sampling_selection_probability=np.asarray([0.5]),
            sampling_design_weight_raw=np.asarray([1.0]),
        )
    with gzip.open(shard / "results.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"case_id": "case-1"}) + "\n")
    with pytest.raises(InputValidationError, match="capture_component_state_trace=true"):
        convert_sampler_pilot_to_dataset_v2(tmp_path / "pilot", tmp_path / "out")


def test_sampler_converter_packages_real_component_trace(
    tmp_path: Path, target_data, case_data, registry
) -> None:
    case = deepcopy(case_data)
    case["analysis"]["replicates"] = 2
    case["analysis"]["adaptive_ci_half_width"] = None
    case["analysis"]["capture_component_state_trace"] = True
    result = DamageEngine(target_data, registry).run(case).to_dict()
    source = tmp_path / "pilot" / "formal_dataset"
    shard = source / "shards" / "part-000000"
    shard.mkdir(parents=True)
    manifest = {
        "dataset_id": "source-real-trace",
        "engine": {"code_hash": "sha256:" + "5" * 64},
        "source_contracts": {
            "label_spec_id": "label@1",
            "sampling_spec_document": {"seeds": {"design": 19}},
        },
        "model_input_contract": {
            "schema_version": "test-input@1",
            "tensor_field": "model_x",
            "mask_field": "model_mask",
        },
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (shard / "tensors.npz").open("wb") as stream:
        np.savez_compressed(
            stream,
            case_ids=np.asarray([result["case_id"]]),
            split_names=np.asarray(["train"]),
            input_frozen_splits=np.asarray(["train"]),
            model_x=np.zeros((1, 2)),
            model_mask=np.ones((1, 2), dtype=bool),
            model_feature_names=np.asarray(["x1", "x2"]),
            component_ids=np.asarray(result["component_order"]),
            sampling_weight_mask=np.asarray([True]),
            sampling_selection_probability=np.asarray([0.5]),
            sampling_design_weight_raw=np.asarray([1.0]),
        )
    with gzip.open(shard / "results.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(result) + "\n")
    report = convert_sampler_pilot_to_dataset_v2(
        tmp_path / "pilot", tmp_path / "dataset-v2"
    )
    assert report["valid"] is True
    assert report["case_count"] == 1
    assert report["split_counts"]["train"] == 1
