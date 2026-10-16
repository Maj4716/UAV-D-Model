from __future__ import annotations

from pathlib import Path

import numpy as np

from damage_engine.learning_input_identity import (
    UAV_STATE_FEATURE_NAMES,
    UAV_STATE_FEATURE_SEMANTICS,
    UAV_STATE_FEATURE_SOURCE_PATHS,
    UAV_STATE_FEATURE_TRANSFORMS,
    UAV_STATE_FEATURE_UNITS,
)
from tools.build_dataset_health_report import (
    HTML_V2,
    _feature_report,
    _load_case_level_arrays,
    _model_input_binding,
    _role_report,
)


def _uav_manifest() -> dict:
    return {
        "model_input_contract": {
            "version": "uav-state-input@1.0.0",
            "tensor_field": "uav_state_x",
            "mask_field": "uav_state_feature_mask",
            "fixed_scenario_policy": "single_frozen_scenario_per_archive",
            "fixed_scenario_hash": "sha256:" + "a" * 64,
            "evidence_boundary": "fixed threat frame; software evidence only",
            "excluded_tensor_fields": ["case_physics_x", "solver_context_x"],
            "features": [
                {
                    "name": name,
                    "unit": unit,
                    "semantic": semantic,
                    "source_path": source,
                    "transformation": transform,
                }
                for name, unit, semantic, source, transform in zip(
                    UAV_STATE_FEATURE_NAMES,
                    UAV_STATE_FEATURE_UNITS,
                    UAV_STATE_FEATURE_SEMANTICS,
                    UAV_STATE_FEATURE_SOURCE_PATHS,
                    UAV_STATE_FEATURE_TRANSFORMS,
                    strict=True,
                )
            ],
        }
    }


def _write_uav_shard(root: Path) -> Path:
    shard = root / "part-000000"
    shard.mkdir(parents=True)
    root_half = np.sqrt(0.5)
    uav = np.asarray(
        [
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [2, 0, 0, 2, 0, 0, 0, 0, root_half, root_half],
            [0, 3, 0, 0, 3, 0, 0, root_half, 0, root_half],
            [0, 0, 4, 0, 0, 4, root_half, 0, 0, root_half],
        ],
        dtype=np.float32,
    )
    system_y = np.asarray(
        [[0.1, 0.0], [0.2, 0.1], [0.5, 0.2], [0.9, 0.7]], dtype=np.float32
    )
    intervals = np.stack(
        [np.clip(system_y - 0.05, 0, 1), np.clip(system_y + 0.05, 0, 1)],
        axis=-1,
    )
    aux = np.asarray(
        [[[1.0, 0.0]], [[2.0, 1.0]], [[0.0, 0.0]], [[3.0, 3.0]]],
        dtype=np.float32,
    )
    np.savez_compressed(
        shard / "tensors.npz",
        uav_state_x=uav,
        uav_state_feature_mask=np.ones_like(uav, dtype=bool),
        split_names=np.asarray(["train", "validation", "train", "train"]),
        system_y=system_y,
        system_label_mask=np.ones_like(system_y, dtype=bool),
        system_source_available_mask=np.ones_like(system_y, dtype=bool),
        system_uncertainty_mask=np.ones_like(system_y, dtype=bool),
        system_y_ci95=intervals,
        replicate_counts=np.asarray([4, 8, 32, 32], dtype=np.int64),
        label_task_available_mask=np.ones((4, 1), dtype=bool),
        label_task_names=np.asarray(["system_events"]),
        system_output_names=np.asarray(["mission_degraded", "mission_loss"]),
        sampling_selection_probability=np.ones(4, dtype=np.float64),
        sampling_design_weight_raw=np.ones(4, dtype=np.float64),
        sampling_weight_mask=np.asarray([True, True, False, False]),
        sampling_round=np.zeros(4, dtype=np.int64),
        sampling_role_names=np.asarray(
            ["reference_train", "geometry_stratified_train"]
        ),
        sampling_role_index=np.asarray([0, 0, 1, 1], dtype=np.int64),
        sampling_stratum_names=np.asarray(["p0", "near_cone"]),
        sampling_stratum_index=np.asarray([0, 0, 1, 1], dtype=np.int64),
        component_aux_label_names=np.asarray(
            ["fragment_hits", "fragment_penetrations"]
        ),
        component_aux_y=aux,
        component_aux_label_mask=np.ones_like(aux, dtype=bool),
        armor_path_feature_names=np.asarray(["intersects_path"]),
        component_armor_path_x=np.asarray(
            [[[[0.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]]], dtype=np.float32
        ),
        component_armor_path_mask=np.ones((4, 1, 1, 1), dtype=bool),
    )
    return shard


def test_uav_health_report_prefers_contract_and_builds_10d_to_12d_view(tmp_path):
    shard = _write_uav_shard(tmp_path)
    arrays, static, binding = _load_case_level_arrays([shard], _uav_manifest())
    features, derived, training = _feature_report(arrays, binding)

    assert binding["kind"] == "uav_control"
    assert binding["tensor_field"] == "uav_state_x"
    assert len(features) == 10
    assert training["available"] is True
    assert training["feature_count"] == 12
    assert len(training["feature_names"]) == 12
    assert {item["id"] for item in derived["vector_checks"]} == {
        "relative_position_norm",
        "relative_speed_norm",
        "quaternion_norm",
    }

    roles = {item["role"]: item for item in _role_report(arrays, static)}
    assert roles["reference_train"]["category"] == "reference"
    assert roles["reference_train"]["weights"]["ess"] == 2.0
    assert roles["reference_train"]["events"]["fragment_hit"]["count"] == 2
    targeted = roles["geometry_stratified_train"]
    assert targeted["category"] == "targeted"
    assert targeted["weights"]["eligible_n"] == 0
    assert targeted["system_labels"][0]["ci_half_width"]["p95"] is not None


def test_health_report_keeps_legacy_input_fallback(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        case_physics_x=np.zeros((2, 3), dtype=np.float32),
        case_physics_feature_mask=np.ones((2, 3), dtype=bool),
        case_physics_feature_names=np.asarray(["a", "b", "c"]),
        case_physics_feature_units=np.asarray(["1", "1", "1"]),
    )
    with np.load(path, allow_pickle=False) as tensors:
        binding = _model_input_binding({}, tensors)
    assert binding["kind"] == "legacy"
    assert binding["tensor_field"] == "case_physics_x"
    assert binding["names"] == ["a", "b", "c"]


def test_health_report_template_has_no_dataset_specific_fixed_counts():
    assert "2000 条数据集" not in HTML_V2
    assert "起爆半径固定为 2 m" not in HTML_V2
    assert "实际变化的 7 个特征" not in HTML_V2
    assert "REPORT.model_input.tensor_field" in HTML_V2
    assert "REPORT.roles" in HTML_V2
