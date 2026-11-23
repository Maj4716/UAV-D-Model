from __future__ import annotations

from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np

from damage_engine.dataset_v2 import (
    load_component_state_traces,
    load_dataset_v2_core,
    validate_dataset_v2,
)
from damage_engine.engine import DamageEngine
from damage_engine.hurdle_baseline import run_hurdle_learning_curves
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


def test_strict_converter_keeps_invalid_test_json_opaque_and_baseline_compatible(
    tmp_path: Path,
    target_data,
    case_data,
    registry,
) -> None:
    train = _result(
        case_data,
        case_id="strict-train",
        seed=101,
        target_data=target_data,
        registry=registry,
    )
    validation = _result(
        case_data,
        case_id="strict-validation",
        seed=202,
        target_data=target_data,
        registry=registry,
    )
    test_case_id = "strict-test-opaque"
    case_ids = np.asarray([train["case_id"], validation["case_id"], test_case_id])
    splits = np.asarray(["train", "validation", "test"])
    component_ids = np.asarray(train["component_order"])

    source = tmp_path / "pilot" / "formal_dataset"
    shard = source / "shards" / "part-000000"
    shard.mkdir(parents=True)
    manifest = {
        "dataset_id": "strict-seal-source",
        "engine": {"code_hash": "sha256:" + "6" * 64},
        "source_contracts": {
            "label_spec_id": "hurdle-label@2",
            "sampling_spec_document": {"seeds": {"design": 23}},
        },
        "model_input_contract": {
            "schema_version": "test-input@1",
            "tensor_field": "model_x",
            "mask_field": "model_mask",
        },
    }
    (source / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
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
            deliberately_unread_test_label_tensor=np.asarray([1.0, 2.0, np.nan]),
        )
    with gzip.open(shard / "results.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(train) + "\n")
        stream.write(json.dumps(validation) + "\n")
        stream.write("{this sealed test line is deliberately invalid JSON\n")

    output = tmp_path / "dataset-v2-development"
    report = convert_sampler_pilot_to_strict_development_dataset_v2(
        tmp_path / "pilot", output
    )

    assert report["valid"] is True
    assert report["split_counts"] == {"test": 1, "train": 1, "validation": 1}
    assert report["result_access"]["development_result_json_lines_decoded"] == 2
    assert report["result_access"]["test_result_json_lines_decoded"] == 0
    assert validate_dataset_v2(output)["valid"] is True

    seal = json.loads((output / "sealed_test.json").read_text(encoding="utf-8"))
    assert seal["test_labels_present_in_core"] is False
    assert seal["test_component_traces_present_in_core"] is False
    assert seal["test_label_masks_all_false"] is True
    assert seal["test_result_json_lines_decoded"] == 0
    assert "deliberately_unread_test_label_tensor" not in seal[
        "source_tensor_fields_read"
    ]

    core = load_dataset_v2_core(output)
    labels = core["component_labels"]
    weights = core["sampling_weights"]
    test_index = int(np.flatnonzero(weights["split"].astype(str) == "test")[0])
    assert not np.any(labels["label_mask"][test_index])
    assert not np.any(labels["p_hit_mask"][test_index])
    assert not np.any(labels["p_degraded_mask"][test_index])
    assert not np.any(labels["p_failed_mask"][test_index])
    assert not np.any(labels["conditional_log1p_energy_mask"][test_index])
    assert not np.any(labels["conditional_log1p_hit_count_mask"][test_index])
    assert not np.any(labels["derived_support_mask"][test_index])
    assert np.all(labels["p_hit_trial_count"][test_index] == 1)
    assert np.all(labels["p_hit_identification_interval"][test_index, :, 0] == 0.0)
    assert np.all(labels["p_hit_identification_interval"][test_index, :, 1] == 1.0)

    traces = load_component_state_traces(output)
    assert traces[test_index]["component_state_trace"]["sealed_test_placeholder"] is True
    assert traces[test_index]["component_state_trace"]["records"][0][
        "component_mechanism_states"
    ] == {}

    learning = run_hurdle_learning_curves(
        output,
        tmp_path / "baseline",
        fractions=[1.0],
        seeds=[7],
        epochs=2,
        hidden_size=8,
    )
    assert learning["split_counts"]["test_sealed_unread"] == 1
    assert learning["test_access_policy"]["open_count"] == 0
    assert learning["selected_frozen_model"]["test_labels_evaluated"] is False

