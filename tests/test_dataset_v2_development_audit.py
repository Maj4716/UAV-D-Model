from __future__ import annotations

from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np

from damage_engine.engine import DamageEngine
from tools.audit_dataset_v2_development import (
    _holm_adjust,
    _ks_diagnostics,
    audit_development_dataset,
)
from tools.package_sampler_dataset_v2_development import (
    convert_sampler_pilot_to_strict_development_dataset_v2,
)


def _result(
    case_data: dict,
    *,
    case_id: str,
    seed: int,
    target_data: dict,
    registry,
) -> dict:
    case = deepcopy(case_data)
    case["case_id"] = case_id
    case["analysis"]["aleatory_seed"] = seed
    case["analysis"]["replicates"] = 2
    case["analysis"]["adaptive_ci_half_width"] = None
    case["analysis"]["capture_component_state_trace"] = True
    return DamageEngine(target_data, registry).run(case).to_dict()


def _formal_case(case_data: dict, case_id: str, latent_shift: float) -> dict:
    case = deepcopy(case_data)
    case["case_id"] = case_id
    case.setdefault("metadata", {})["continuous_direction_sampling"] = {
        "latent_names": [
            "direction_region_selector",
            "direction_coordinate_1",
            "direction_coordinate_2",
            "aim_u_quantile",
            "aim_v_quantile",
            "speed_quantile",
            "body_x_roll_quantile",
        ],
        "physical_latent": [
            0.1,
            0.2 + latent_shift,
            0.3 + latent_shift,
            0.4,
            0.5,
            0.6,
            0.7,
        ],
        "region_id": "pos_x",
        "aim_u_m": 0.1 + latent_shift,
        "aim_v_m": 0.2 + latent_shift,
        "speed_mps": 120.0 + latent_shift,
        "body_x_roll_rad": 0.3 + latent_shift,
    }
    return case


def test_development_audit_keeps_invalid_test_result_opaque(
    tmp_path: Path,
    target_data,
    case_data,
    registry,
) -> None:
    train = _result(
        case_data,
        case_id="audit-train",
        seed=11,
        target_data=target_data,
        registry=registry,
    )
    validation = _result(
        case_data,
        case_id="audit-validation",
        seed=22,
        target_data=target_data,
        registry=registry,
    )
    test_case_id = "audit-test-opaque"
    case_ids = np.asarray([train["case_id"], validation["case_id"], test_case_id])
    splits = np.asarray(["train", "validation", "test"])
    component_ids = np.asarray(train["component_order"])

    pilot = tmp_path / "pilot"
    source = pilot / "formal_dataset"
    shard = source / "shards" / "part-000000"
    shard.mkdir(parents=True)
    manifest = {
        "dataset_id": "development-audit-source",
        "engine": {"code_hash": "sha256:" + "7" * 64},
        "source_contracts": {
            "label_spec_id": "hurdle-label@2",
            "sampling_spec_document": {"seeds": {"design": 31}},
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
            case_ids=case_ids,
            split_names=splits,
            input_frozen_splits=splits,
            model_x=np.asarray([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]),
            model_mask=np.ones((3, 2), dtype=bool),
            model_feature_names=np.asarray(["x1", "x2"]),
            component_ids=component_ids,
            sampling_weight_mask=np.ones(3, dtype=bool),
            sampling_selection_probability=np.full(3, 0.5),
            sampling_design_weight_raw=np.ones(3),
        )
    with gzip.open(shard / "results.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(train) + "\n")
        stream.write(json.dumps(validation) + "\n")
        stream.write("{this sealed test payload is deliberately invalid JSON\n")

    round_root = pilot / "rounds" / "round-000"
    round_root.mkdir(parents=True)
    with gzip.open(round_root / "formal_cases.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(_formal_case(case_data, train["case_id"], 0.0)) + "\n")
        stream.write(
            json.dumps(_formal_case(case_data, validation["case_id"], 0.1)) + "\n"
        )
        stream.write(json.dumps(_formal_case(case_data, test_case_id, 0.2)) + "\n")

    dataset = tmp_path / "dataset-v2-development"
    convert_sampler_pilot_to_strict_development_dataset_v2(pilot, dataset)
    report = audit_development_dataset(dataset, pilot, epsilon=0.01)

    assert report["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert report["result_access"]["development_result_json_lines_decoded"] == 2
    assert report["result_access"]["test_result_json_lines_seen"] == 1
    assert report["result_access"]["test_result_json_lines_decoded"] == 0
    assert report["sealed_test"]["test_metrics_computed"] is False
    assert report["by_split"]["train"]["case_count"] == 1
    assert report["by_split"]["validation"]["case_count"] == 1
    assert report["training_readiness"]["formal_neural_training_ready"] is False
    assert report["component_task_admission"]["p_hit"]["support_kind"] == (
        "binary_probability"
    )
    for target in (
        "conditional_log1p_energy",
        "conditional_log1p_hit_count",
    ):
        admission = report["component_task_admission"][target]
        assert admission["support_kind"] == "observed_continuous"
        assert admission["thresholds"] == {
            "train_observed_min": 20,
            "validation_observed_min": 5,
        }
        assert admission["admitted_component_count"] == 0
        assert admission["admitted_component_ids"] == []
        assert admission["rejected_components"]
        assert set(admission["train_observed_count"]) == set(
            admission["validation_observed_count"]
        )
        assert all(
            "train_observed_below_minimum" in row["reasons"]
            and "validation_observed_below_minimum" in row["reasons"]
            for row in admission["rejected_components"]
        )
    assert report["report_hash"].startswith("sha256:")


def test_holm_adjustment_and_ks_diagnostic_are_explicit() -> None:
    records = [
        {"status": "evaluated", "p_value": 0.01},
        {"status": "evaluated", "p_value": 0.03},
        {"status": "evaluated", "p_value": 0.20},
    ]
    _holm_adjust(records, alpha=0.05)
    assert records[0]["p_value_holm"] == 0.03
    assert records[0]["significant_holm_0_05"] is True
    assert records[1]["p_value_holm"] == 0.06
    assert records[1]["significant_holm_0_05"] is False

    metadata = []
    for index in range(8):
        metadata.append(
            {
                "region_id": "pos_x",
                "profile_id": "small_front",
                **{axis: float(index) for axis in (
                    "direction_coordinate_1",
                    "direction_coordinate_2",
                    "aim_u_m",
                    "aim_v_m",
                    "speed_mps",
                    "body_x_roll_rad",
                )},
            }
        )
    report = _ks_diagnostics(
        metadata,
        np.asarray(["train"] * 4 + ["validation"] * 4),
        np.asarray([True, True, False, False, True, True, False, False]),
        alpha=0.05,
    )
    assert report["evaluated_record_count"] > 0
    assert report["multiple_comparison"].startswith("Holm")
    assert "subset" in report["subset_dependence_note"]

