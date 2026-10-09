from __future__ import annotations

from copy import deepcopy
import pickle
import sys
import types

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from damage_engine.cli import build_parser
from damage_engine.dataset import (
    ARMOR_FEATURE_NAMES,
    ARMOR_LABEL_NAMES,
    ARMOR_LABEL_UNITS,
    AUXILIARY_LABEL_NAMES,
    CASE_FEATURE_NAMES,
    COMPONENT_FEATURE_NAMES,
    DatasetWriter,
    FRECHET_BOUND_NAMES,
)
from damage_engine.engine import DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.schema_resources import load_schema


def _cases(engine, case_data, count=2, *, replicates=2):
    cases = []
    for index in range(count):
        case = deepcopy(case_data)
        case["case_id"] = f"dataset-case-{index}"
        case["target_ref"]["content_hash"] = engine.target_hash
        case["analysis"]["aleatory_seed"] = 100 + index
        case["analysis"]["replicates"] = replicates
        case["analysis"]["adaptive_ci_half_width"] = None
        case["metadata"]["split_group"] = f"group-{index}"
        cases.append(case)
    return cases


def test_dataset_export_contract(tmp_path, target_data, case_data, registry):
    engine = DamageEngine(target_data, registry)
    cases = _cases(engine, case_data)
    manifest = DatasetWriter(tmp_path, split_seed=17).run_cases(engine, cases)
    schema = load_schema("dataset_manifest.schema.json")
    Draft202012Validator(schema).validate(manifest)

    with np.load(tmp_path / "tensors.npz", allow_pickle=False) as tensors:
        assert tensors["case_x"].shape == (2, len(CASE_FEATURE_NAMES))
        assert tensors["component_x"].shape == (2, 1, len(COMPONENT_FEATURE_NAMES))
        assert tensors["armor_x"].shape == (2, 1, len(ARMOR_FEATURE_NAMES))
        assert tensors["armor_y"].shape == (2, 1, len(ARMOR_LABEL_NAMES))
        assert tensors["armor_y_stddev"].shape == (2, 1, len(ARMOR_LABEL_NAMES))
        assert tensors["armor_y_ci95"].shape == (2, 1, len(ARMOR_LABEL_NAMES), 2)
        assert tensors["armor_label_mask"].shape == (2, 1, len(ARMOR_LABEL_NAMES))
        assert tensors["armor_label_mask"].all()
        assert tensors["armor_label_names"].tolist() == ARMOR_LABEL_NAMES
        assert tensors["armor_label_units"].tolist() == ARMOR_LABEL_UNITS
        assert tensors["component_armor_path_x"].shape == (2, 1, 1, 4)
        assert tensors["component_y"].shape == (2, 1, 3, 2)
        assert tensors["component_frechet_y"].shape == (
            2, 1, 2, len(FRECHET_BOUND_NAMES)
        )
        assert tensors["component_frechet_y_stddev"].shape == (
            2, 1, 2, len(FRECHET_BOUND_NAMES)
        )
        assert tensors["component_frechet_y_ci95"].shape == (
            2, 1, 2, len(FRECHET_BOUND_NAMES), 2
        )
        assert tensors["component_frechet_label_mask"].all()
        assert tensors["frechet_bound_names"].tolist() == FRECHET_BOUND_NAMES
        combined = tensors["component_y"][:, :, 2, :]
        np.testing.assert_array_less(
            tensors["component_frechet_y"][:, :, :, 0] - 1.0e-7,
            combined,
        )
        np.testing.assert_array_less(
            combined - 1.0e-7,
            tensors["component_frechet_y"][:, :, :, 1],
        )
        assert tensors["component_aux_y"].shape == (2, 1, len(AUXILIARY_LABEL_NAMES))
        assert tensors["component_aux_y_ci95"].shape == (2, 1, len(AUXILIARY_LABEL_NAMES), 2)
        assert tensors["system_y"].shape == (2, 2)
        assert tensors["system_label_mask"].shape == (2, 2)
        assert tensors["evaluation_standard_ids"].tolist() == ["demo-tree-1"]
        assert tensors["standard_output_mask"].shape == (1, 2)
        assert tensors["standard_output_mask"].tolist() == [[True, True]]
        assert tensors["requested_standard_mask"].tolist() == [[True], [True]]
        assert tensors["split_names"].shape == (2,)
        assert tensors["replicate_counts"].tolist() == [2, 2]
        assert tensors["input_case_valid_mask"].tolist() == [True, True]
        assert tensors["armor_vertices_m"].shape[1] == 3
        assert tensors["tree_edge_index"].shape[0] == 2
        contract_names = {field["name"] for field in manifest["tensor_contract"]["fields"]}
        assert contract_names == set(tensors.files)

    assert manifest["schema_version"] == "1.1.0"
    assert manifest["tensor_contract"]["version"] == "1.3.0"
    assert manifest["split_policy"]["seed"] == 17
    assert manifest["split_policy"]["group_keys"] == [
        "split_group", "condition_cluster", "physical_case_hash"
    ]
    assert manifest["split_policy"]["leakage_audit_passed"] is True
    assert manifest["engine"]["fidelity_levels"] == ["L2"]
    assert manifest["source_contracts"]["evaluation_standards"] == {
        "demo-tree-1": ["mission_degraded", "mission_loss"]
    }
    assert any(partition["name"] == "tensors" for partition in manifest["partitions"])
    assert any(partition["name"] == "armor_labels" for partition in manifest["partitions"])
    assert not (tmp_path / ".dataset.incomplete").exists()
    case_field = next(
        field for field in manifest["tensor_contract"]["fields"] if field["name"] == "case_x"
    )
    assert case_field["unit"] == "mixed; see case_feature_units"

    import pyarrow.parquet as pq

    armor_label_rows = pq.read_table(tmp_path / "armor_labels.parquet").to_pylist()
    assert len(armor_label_rows) == 2 * len(ARMOR_LABEL_NAMES)
    assert {row["metric"] for row in armor_label_rows} == set(ARMOR_LABEL_NAMES)


def test_failure_isolation_and_duplicate_ids(tmp_path, target_data, case_data, registry):
    engine = DamageEngine(target_data, registry)
    valid, invalid = _cases(engine, case_data)
    invalid["target_ref"]["content_hash"] = "sha256:" + "0" * 64
    manifest = DatasetWriter(tmp_path).run_cases(engine, [valid, invalid])
    assert manifest["quality_summary"]["n_valid"] == 1
    assert manifest["quality_summary"]["n_refused"] == 1
    with np.load(tmp_path / "tensors.npz", allow_pickle=False) as tensors:
        assert tensors["case_ids"].tolist() == [valid["case_id"]]
        assert tensors["input_case_valid_mask"].tolist() == [True, False]
        assert tensors["input_to_valid_index"].tolist() == [0, -1]

    duplicate = deepcopy(valid)
    with pytest.raises(InputValidationError, match="case_id"):
        DatasetWriter(tmp_path / "duplicate").run_cases(engine, [valid, duplicate])


def test_automatic_split_groups_monte_carlo_repeats(target_data, case_data, registry):
    engine = DamageEngine(target_data, registry)
    first, second = _cases(engine, case_data)
    for case in (first, second):
        case["metadata"].pop("split_group", None)
        case["metadata"].pop("condition_cluster", None)
    second["encounter"] = deepcopy(first["encounter"])
    second["warhead"] = deepcopy(first["warhead"])
    second["environment"] = deepcopy(first["environment"])
    second["analysis"] = deepcopy(first["analysis"])
    second["analysis"]["aleatory_seed"] = first["analysis"]["aleatory_seed"] + 1
    second["analysis"]["replicates"] = first["analysis"]["replicates"] + 1
    second["analysis"]["evaluation_standard_ids"] = ["a-different-reporting-view"]
    from damage_engine.dataset import _split_name

    first_group = _split_name(first, seed=99)
    second_group = _split_name(second, seed=99)
    assert first_group[0] == second_group[0]
    assert first_group[1] == second_group[1] == "physical_case_hash"
    assert first_group[3] == second_group[3]


def test_multi_standard_masks_and_tree_rows(
    tmp_path, target_data, case_data, registry
):
    target = deepcopy(target_data)
    target["damage_tree"]["standards"] = [
        {"id": "loss-only", "outputs": ["mission_loss"]},
        {"id": "degraded-only", "outputs": ["mission_degraded"]},
        {
            "id": "all-effects",
            "outputs": ["mission_loss", "mission_degraded"],
        },
    ]
    engine = DamageEngine(target, registry)
    cases = _cases(engine, case_data, count=2, replicates=1)
    cases[0]["analysis"]["evaluation_standard_ids"] = ["loss-only", "all-effects"]
    cases[1]["analysis"]["evaluation_standard_ids"] = ["degraded-only"]

    manifest = DatasetWriter(tmp_path).run_cases(engine, cases)

    with np.load(tmp_path / "tensors.npz", allow_pickle=False) as tensors:
        assert tensors["system_y"].shape == (2, 2)
        assert tensors["evaluation_standard_ids"].tolist() == [
            "loss-only", "degraded-only", "all-effects"
        ]
        assert tensors["standard_output_mask"].tolist() == [
            [False, True], [True, False], [True, True]
        ]
        assert tensors["requested_standard_mask"].tolist() == [
            [True, False, True], [False, True, False]
        ]
        assert tensors["system_label_mask"].tolist() == [
            [True, True], [True, False]
        ]

    assert manifest["source_contracts"]["evaluation_standards"] == {
        "loss-only": ["mission_loss"],
        "degraded-only": ["mission_degraded"],
        "all-effects": ["mission_loss", "mission_degraded"],
    }

    import pyarrow.parquet as pq

    rows = pq.read_table(tmp_path / "tree_outputs.parquet").to_pylist()
    assert [
        (row["case_id"], row["evaluation_standard_id"], row["output_node"])
        for row in rows
    ] == [
        ("dataset-case-0", "loss-only", "mission_loss"),
        ("dataset-case-0", "all-effects", "mission_loss"),
        ("dataset-case-0", "all-effects", "mission_degraded"),
        ("dataset-case-1", "degraded-only", "mission_degraded"),
    ]


def test_parallel_and_serial_tensor_order_match(tmp_path, target_data, case_data, registry):
    engine = DamageEngine(target_data, registry)
    cases = _cases(engine, case_data, count=3, replicates=1)
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial_manifest = DatasetWriter(serial_dir).run_cases(engine, cases, workers=1)
    parallel_manifest = DatasetWriter(parallel_dir).run_cases(engine, cases, workers=2, chunksize=1)
    assert serial_manifest["dataset_id"] == parallel_manifest["dataset_id"]
    with (
        np.load(serial_dir / "tensors.npz", allow_pickle=False) as serial,
        np.load(parallel_dir / "tensors.npz", allow_pickle=False) as parallel,
    ):
        assert serial.files == parallel.files
        for name in serial.files:
            np.testing.assert_array_equal(serial[name], parallel[name], err_msg=name)

    import pyarrow.parquet as pq

    serial_labels = pq.read_table(serial_dir / "labels.parquet").to_pydict()
    parallel_labels = pq.read_table(parallel_dir / "labels.parquet").to_pydict()
    assert serial_labels == parallel_labels


def test_torch_adapter_is_pickleable_and_uses_npz_splits(
    tmp_path, target_data, case_data, registry, monkeypatch
):
    engine = DamageEngine(target_data, registry)
    DatasetWriter(tmp_path).run_cases(engine, _cases(engine, case_data, count=1, replicates=1))
    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = np.asarray
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    from damage_engine.torch_dataset import TorchDamageDataset

    dataset = TorchDamageDataset(tmp_path)
    restored = pickle.loads(pickle.dumps(dataset))
    sample = restored[0]
    assert "armor_x" in sample
    assert "armor_y" in sample
    assert "armor_label_mask" in sample
    assert "component_aux_y" in sample
    assert "component_frechet_y" in sample
    assert "component_frechet_label_mask" in sample
    assert "requested_standard_mask" in sample
    assert "standard_output_mask" in sample
    assert sample["requested_standard_mask"].tolist() == [True]
    assert sample["standard_output_mask"].tolist() == [[True, True]]
    assert "tree_edge_index" in sample
    assert restored.metadata["case_feature_names"] == tuple(CASE_FEATURE_NAMES)
    assert restored.metadata["armor_label_names"] == tuple(ARMOR_LABEL_NAMES)
    assert restored.metadata["frechet_bound_names"] == tuple(FRECHET_BOUND_NAMES)
    assert restored.metadata["evaluation_standard_ids"] == ("demo-tree-1",)
    with pytest.raises(ValueError, match="split must"):
        TorchDamageDataset(tmp_path, split="holdout")


def test_armor_label_mask_tracks_enabled_mechanisms(
    tmp_path, target_data, case_data, registry
):
    engine = DamageEngine(target_data, registry)
    case = _cases(engine, case_data, count=1, replicates=1)[0]
    case["analysis"]["mechanisms"] = ["fragment"]
    case["analysis"]["coupling_model"] = "none"
    DatasetWriter(tmp_path).run_cases(engine, [case])

    with np.load(tmp_path / "tensors.npz", allow_pickle=False) as tensors:
        mask = tensors["armor_label_mask"][0, 0]
        assert mask[ARMOR_LABEL_NAMES.index("post_fragment_integrity")]
        assert mask[ARMOR_LABEL_NAMES.index("post_combined_integrity")]
        assert not mask[ARMOR_LABEL_NAMES.index("blast_failure_probability")]
        assert not mask[ARMOR_LABEL_NAMES.index("blast_reflected_peak_pressure_pa")]
        assert not tensors["component_frechet_label_mask"].any()


def test_multi_target_collate_pads_axes_aligns_outputs_and_offsets_graph(
    tmp_path, target_data, case_data, registry, monkeypatch
):
    first_engine = DamageEngine(target_data, registry)
    first_directory = tmp_path / "first"
    DatasetWriter(first_directory).run_cases(
        first_engine,
        _cases(first_engine, case_data, count=1, replicates=1),
    )

    second_target = deepcopy(target_data)
    extra_component = deepcopy(second_target["components"][0])
    extra_component["id"] = "backup-control"
    extra_component["name"] = "Synthetic backup control module"
    extra_component["geometry"]["center_m"] = [0.0, 0.6, 0.0]
    second_target["components"].append(extra_component)
    extra_armor = deepcopy(second_target["armor"][0])
    extra_armor["id"] = "armor-secondary"
    extra_armor["name"] = "Synthetic secondary panel"
    extra_armor["geometry"]["vertices_m"] = [
        [vertex[0], vertex[1] + 2.0, vertex[2]]
        for vertex in extra_armor["geometry"]["vertices_m"]
    ]
    second_target["armor"].append(extra_armor)
    second_target["damage_tree"]["nodes"].append(
        {
            "id": "alternate_effect",
            "node_type": "standard_projection",
            "inputs": ["control_blast_degraded"],
            "component_id": None,
            "mechanism": "not_applicable",
            "damage_state": None,
            "k": None,
            "dependency_model_id": "independent",
            "threshold_source_id": "DEMO-NOT-VALIDATED",
            "description": "Additional target-specific output",
        }
    )
    second_target["damage_tree"]["outputs"].append("alternate_effect")
    second_engine = DamageEngine(second_target, registry)
    second_directory = tmp_path / "second"
    DatasetWriter(second_directory).run_cases(
        second_engine,
        _cases(second_engine, case_data, count=1, replicates=1),
    )

    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = np.asarray
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    from damage_engine.torch_dataset import (
        TorchMultiTargetDataset,
        collate_damage_batch,
    )

    dataset = TorchMultiTargetDataset([first_directory, second_directory])
    first_sample, second_sample = dataset[0], dataset[1]
    first_edge_count = first_sample["tree_edge_index"].shape[1]
    first_node_count = len(first_sample["_topology"]["graph_node_ids"])
    batch = collate_damage_batch([first_sample, second_sample])

    assert batch["component_x"].shape[:2] == (2, 2)
    assert batch["component_frechet_y"].shape[:2] == (2, 2)
    assert batch["component_frechet_y"].shape[2:] == (
        2, len(FRECHET_BOUND_NAMES)
    )
    assert not batch["component_frechet_label_mask"][0, 1].any()
    assert batch["armor_x"].shape[:2] == (2, 2)
    assert batch["armor_y"].shape[:2] == (2, 2)
    assert batch["component_armor_path_x"].shape[:3] == (2, 2, 2)
    assert batch["component_presence_mask"].tolist() == [
        [True, False], [True, True]
    ]
    assert batch["armor_presence_mask"].tolist() == [
        [True, False], [True, True]
    ]
    assert batch["system_output_names"] == (
        "mission_degraded", "mission_loss", "alternate_effect"
    )
    assert batch["system_y"].shape == (2, 3)
    assert batch["system_label_mask"].tolist()[0] == [True, True, False]
    assert np.all(
        batch["tree_edge_index"][:, first_edge_count:] >= first_node_count
    )
    assert batch["graph_node_offsets"].tolist() == [
        0,
        first_node_count,
        first_node_count + len(second_sample["_topology"]["graph_node_ids"]),
    ]


def test_generate_cli_defaults_to_failure_isolation():
    arguments = build_parser().parse_args(
        [
            "generate", "--target", "target.json", "--cases", "cases.jsonl",
            "--physics", "physics.json", "--output-directory", "dataset",
        ]
    )
    assert arguments.fail_fast is False
    assert tuple(arguments.split_ratios) == (0.8, 0.1, 0.1)
