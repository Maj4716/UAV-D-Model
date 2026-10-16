from __future__ import annotations

import json
from pathlib import Path

from damage_engine.case_factory import CaseFactory
from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.label_spec import LabelSpec
from damage_engine.learning_input_identity import (
    UAV_STATE_FEATURE_NAMES,
    UAV_TRAINING_FEATURE_NAMES,
)
from damage_engine.normalization import load_training_normalization_artifact
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.sharded_dataset import ShardedDatasetWriter
from damage_engine.utils import content_hash
from tools.validate_uav_control_dataset import main


ROOT = Path(__file__).resolve().parents[1]


def test_real_sharded_uav_archive_passes_strict_acceptance(
    tmp_path, target_data, registry
) -> None:
    engine = DamageEngine(target_data, registry)
    base_case = json.loads(
        (ROOT / "examples" / "case.uav_control.demo.json").read_text(
            encoding="utf-8"
        )
    )
    sampling_spec = SamplingSpec.from_file(
        ROOT / "examples" / "sampling_spec.uav_control.demo.json"
    )
    label_spec = LabelSpec.from_file(ROOT / "examples" / "label_spec.demo.json")
    plan = CaseFactory(
        base_case,
        sampling_spec,
        label_spec,
        target_hash=engine.target_hash,
        damage_tree_definition_hash=content_hash(engine.target_data["damage_tree"]),
        physics_registry_hash=engine.registry.hash,
        engine_code_hash=ENGINE_CODE_HASH,
        physics_registry=engine.registry,
    ).build_reference_plan()

    formal_dataset = tmp_path / "formal_dataset"
    ShardedDatasetWriter(formal_dataset, shard_size=2).run_cases(
        engine,
        plan.cases,
        fail_closed=True,
        workers=1,
    )
    output = tmp_path / "uav-acceptance.json"
    normalizer = tmp_path / "uav-normalization.json"
    exit_code = main(
        [
            str(tmp_path),
            "--output",
            str(output),
            "--normalization-output",
            str(normalizer),
        ]
    )
    assert exit_code == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    artifact = load_training_normalization_artifact(normalizer)
    assert report["status"] == "passed"
    assert report["dataset"]["input_kind"] == "sampling_run"
    assert report["dataset"]["case_count"] == 4
    assert report["dataset"]["split_counts"] == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }
    assert report["model_input_contract"]["raw_feature_names"] == list(
        UAV_STATE_FEATURE_NAMES
    )
    assert report["model_input_contract"]["training_feature_names"] == list(
        UAV_TRAINING_FEATURE_NAMES
    )
    assert report["fixed_scenario"]["unique_hash_count"] == 1
    assert report["leakage_and_duplicates"]["learning_inputs_crossing_splits"] == 0
    assert report["result_ledgers"]["formats"] == ["jsonl_gzip"]
    assert report["normalization"]["fit_split"] == "train"
    assert report["split_traversal"]["validation"]["normalization_operation"] == "apply_only"
    assert report["split_traversal"]["test"]["normalization_operation"] == "apply_only"
    assert report["research_admission"]["admitted"] is False
    assert artifact["normalization_id"] == report["normalization"]["normalization_id"]
    assert [item["name"] for item in artifact["features"]] == list(
        UAV_TRAINING_FEATURE_NAMES
    )
    assert sum(group["n"] for group in report["groups"]) == 4
    assert all("ess" in group["weights"] for group in report["groups"])
    assert all(
        "fragment_hit" in group["events"] for group in report["groups"]
    )
