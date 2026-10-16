"""Reaudit fixes on artificial arrays only; no domain execution or training."""
from __future__ import annotations

from copy import deepcopy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from damage_engine.archive_validation import validate_tensor_arrays, semantic_vocabulary
from damage_engine.errors import InputValidationError
from damage_engine.label_spec import LabelSpec, system_output_availability
from damage_engine.quality import synthetic_label_admission
from damage_engine.torch_dataset import TorchDamageDataset, TorchMultiTargetDataset, TorchShardedDamageDataset
from damage_engine.split_registry import SplitRegistry
from damage_engine.utils import content_hash, file_hash


LABELS = {
    "component_y": (2, 1, 1, 1), "component_frechet_y": (2, 1, 1, 2),
    "component_aux_y": (2, 1, 1), "armor_y": (2, 1, 1), "system_y": (2, 2),
}


def spec_document(task_type="multi_label_event_probability", *, outputs=None):
    return {
        "schema_version": "1.0.0", "label_spec_id": "artificial-labels", "revision": "one",
        "status": "provisional", "source": {
            "damage_tree_id": "abstract-event-graph", "damage_tree_revision": "one",
            "damage_tree_definition_hash": content_hash({"abstract_definition": True}),
        }, "tasks": {"events": {"type": task_type, "outputs": outputs or ["event-a", "event-b"], "loss": "declared_test_loss"}},
        "provenance": {"approval_status": "not_approved", "evidence_ref": None},
    }


def artificial_arrays(*, n=2, version="1.5.0", weights=(True, False)):
    arrays = {
        "case_x": np.zeros((2, 1), np.float32), "case_feature_mask": np.ones((2, 1), bool),
        "case_ids": np.asarray(["artificial-0", "artificial-1"]),
        "case_hashes": np.asarray(["case-hash-0", "case-hash-1"]),
        "split_names": np.asarray(["train", "test"]),
        "split_group_ids": np.asarray(["group-0", "group-1"]),
        "leakage_keys": np.asarray(["input-0", "input-1"]),
        "epistemic_sample_ids": np.asarray(["parameter-0", "parameter-1"]),
        "case_feature_names": np.asarray(["abstract-input"]), "case_feature_units": np.asarray(["1"]),
        "component_ids": np.asarray(["abstract-holder"]), "armor_ids": np.asarray(["abstract-panel"]),
        "mechanism_names": np.asarray(["abstract-source"]), "state_names": np.asarray(["abstract-state"]),
        "system_output_names": np.asarray(["event-a", "event-b"]),
        "evaluation_standard_ids": np.asarray(["abstract-standard"]),
        "frechet_bound_names": np.asarray(["lower", "upper"]), "frechet_bound_units": np.asarray(["1", "1"]),
        "requested_standard_mask": np.ones((2, 1), bool), "standard_output_mask": np.ones((1, 2), bool),
        "replicate_counts": np.full(2, n, np.int64), "aleatory_seeds": np.asarray([5, 6], np.int64),
    }
    for prefix in ("component", "armor"):
        arrays[prefix + "_x"] = np.zeros((2, 1, 1), np.float32)
        arrays[prefix + "_feature_mask"] = np.ones((2, 1, 1), bool)
        arrays[prefix + "_feature_names"] = np.asarray(["abstract-" + prefix + "-feature"])
        arrays[prefix + "_feature_units"] = np.asarray(["1"])
    arrays.update(
        armor_label_names=np.asarray(["abstract-response"]), armor_label_units=np.asarray(["1"]),
        component_aux_label_names=np.asarray(["abstract-auxiliary"]), component_aux_label_units=np.asarray(["1"]),
        armor_path_feature_names=np.asarray(["abstract-path"]), armor_path_feature_units=np.asarray(["1"]),
        component_armor_path_x=np.zeros((2, 1, 1, 1), np.float32), component_armor_path_mask=np.ones((2, 1, 1, 1), bool),
    )
    for label, shape in LABELS.items():
        stem = label.removesuffix("_y")
        arrays[label] = np.full(shape, .25, np.float32)
        arrays[label + "_stddev"] = np.full(shape, .1 if n >= 2 else np.nan, np.float32)
        arrays[label + "_ci95"] = np.tile(np.asarray([.1, .4] if n >= 2 else [np.nan, np.nan], np.float32), (*shape, 1))
        arrays[stem + "_label_mask"] = np.ones(shape, bool)
        arrays[stem + "_uncertainty_mask"] = np.full(shape, n >= 2, bool)
        arrays[stem + "_sample_count"] = np.full(shape, n, np.int64)
    if version != "1.4.0":
        arrays.update(
            case_physics_x=np.zeros((2, 1), np.float32), case_physics_feature_mask=np.ones((2, 1), bool),
            case_physics_feature_names=np.asarray(["abstract-input"]), case_physics_feature_units=np.asarray(["1"]),
            solver_context_x=np.zeros((2, 1), np.float32), solver_context_feature_mask=np.ones((2, 1), bool),
            solver_context_feature_names=np.asarray(["abstract-runtime"]), solver_context_feature_units=np.asarray(["1"]),
            sampling_selection_probability=np.asarray([.5, .25], np.float64),
            sampling_design_weight_raw=np.asarray([2., 4.], np.float64), sampling_weight_mask=np.asarray(weights, bool),
            sampling_round=np.zeros(2, np.int64), sampling_stratum_index=np.zeros(2, np.int64), sampling_role_index=np.zeros(2, np.int64),
            sampling_stratum_names=np.asarray(["stratum-0"]), sampling_role_names=np.asarray(["reference_pilot"]),
            sampling_candidate_ids=np.asarray(["candidate-0", "candidate-1"]), sampling_lineage_ids=np.asarray(["lineage-0", "lineage-1"]),
        )
    if version == "1.6.0":
        arrays.update(
            system_source_available_mask=np.ones((2, 2), bool),
            label_task_names=np.asarray(["events"]), label_task_available_mask=np.ones((2, 1), bool),
            sampling_epistemic_group_ids=np.asarray(["epistemic-0", "epistemic-1"]),
            input_case_ids=arrays["case_ids"].copy(), input_case_status=np.asarray(["ok", "ok"]),
            input_case_valid_mask=np.ones(2, bool), input_to_valid_index=np.arange(2, dtype=np.int64),
            input_sampling_candidate_ids=arrays["sampling_candidate_ids"].copy(),
            input_sampling_lineage_ids=arrays["sampling_lineage_ids"].copy(),
            input_sampling_roles=np.asarray(["reference_pilot", "reference_pilot"]),
            input_sampling_epistemic_group_ids=np.asarray(["epistemic-0", "epistemic-1"]),
            input_frozen_splits=arrays["split_names"].copy(),
        )
    return arrays


def artificial_manifest(arrays, *, version="1.5.0"):
    source = {name: content_hash({"abstract": name}) for name in (
        "damage_tree_definition_hash", "label_spec_hash", "sampling_design_hash", "stratification_spec_hash", "split_registry_hash",
    )}
    source["target_hashes"] = [content_hash({"abstract_target": True})]
    manifest = {
        "dataset_id": "artificial-contract-check", "label_admission": synthetic_label_admission(),
        "source_contracts": source, "split_policy": {"leakage_audit_passed": True},
        "quality_summary": {"n_valid": 2},
        "tensor_contract": {"version": version, "fields": [
            {"name": name, "dtype": "string" if value.dtype.kind in "US" else str(value.dtype),
             "shape": "[" + ",".join(map(str, value.shape)) + "]"}
            for name, value in arrays.items()
        ]}, "semantic_vocabulary": semantic_vocabulary({name: arrays[name].tolist() for name in (
            "mechanism_names", "state_names", "system_output_names", "evaluation_standard_ids",
        )}),
    }
    if version == "1.6.0":
        spec = LabelSpec(spec_document())
        source.update(label_spec_document=spec.data, label_spec_hash=spec.hash, label_spec_id=spec.spec_id,
                      damage_tree_definition_hash=spec.data["source"]["damage_tree_definition_hash"])
        registry = SplitRegistry("artificial-split-registry")
        for index in range(2):
            registry.register(lineage_id=f"lineage-{index}", split=["train", "test"][index],
                              role="reference_pilot", epistemic_group_id=f"epistemic-{index}")
        registry.freeze()
        source.update(split_registry_id=registry.registry_id, split_registry_hash=registry.definition_hash)
        manifest["generation"] = {"sampling": {"split_registry_document": registry.to_dict()}}
    return manifest


def write_artificial(directory: Path, arrays=None, manifest=None, *, version="1.5.0"):
    arrays = artificial_arrays(version=version) if arrays is None else arrays
    manifest = artificial_manifest(arrays, version=version) if manifest is None else deepcopy(manifest)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "tensors.npz", **arrays)
    manifest["partitions"] = [{"name": "tensors", "format": "npz", "path_glob": "tensors.npz", "row_count": 2, "content_hash": file_hash(directory / "tensors.npz")}]
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


@pytest.mark.parametrize("version", ["1.4.0", "1.5.0", "1.6.0"])
@pytest.mark.parametrize("label", tuple(LABELS))
@pytest.mark.parametrize("suffix", ["_uncertainty_mask", "_sample_count"])
@pytest.mark.parametrize("fields", [None, ("case_x",), ("system_y",)])
def test_version_inheritance_never_depends_on_requested_fields(tmp_path, version, label, suffix, fields):
    arrays = artificial_arrays(version=version)
    del arrays[label.removesuffix("_y") + suffix]
    # Deliberately rehash and redeclare fields: internal hashes alone cannot
    # excuse deletion of a version's required inherited capabilities.
    write_artificial(tmp_path, arrays, version=version)
    with pytest.raises(ValueError, match="uncertainty|sample_count"):
        TorchDamageDataset(tmp_path, fields=fields, include_static_graph=False)


@pytest.mark.parametrize("version", ["1.4.0", "1.5.0", "1.6.0"])
@pytest.mark.parametrize("fields", [None, ("case_x",)])
def test_all_new_versions_require_the_saved_vocabulary(tmp_path, version, fields):
    arrays = artificial_arrays(version=version)
    manifest = artificial_manifest(arrays, version=version)
    del manifest["semantic_vocabulary"]
    write_artificial(tmp_path, arrays, manifest, version=version)
    with pytest.raises(ValueError, match="saved semantic vocabulary"):
        TorchDamageDataset(tmp_path, fields=fields, include_static_graph=False)


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("version", ["1.4.0", "1.5.0", "1.6.0"])
def test_complete_artificial_archives_accept_both_observation_counts(tmp_path, n, version):
    write_artificial(tmp_path, artificial_arrays(version=version, n=n), version=version)
    dataset = TorchDamageDataset(tmp_path, include_static_graph=False)
    assert len(dataset) == 2
    assert dataset[0]["system_uncertainty_mask"].tolist() == [n >= 2, n >= 2]


@pytest.mark.parametrize("mask", [(True, False), (False, False), (True, True)])
def test_recorded_weights_survive_masks_but_unusable_weights_stay_masked(tmp_path, mask):
    write_artificial(tmp_path, artificial_arrays(weights=mask))
    dataset = TorchDamageDataset(tmp_path, fields=("sampling_design_weight_raw", "sampling_weight_mask"), include_static_graph=False)
    assert [dataset[index]["sampling_design_weight_raw"].item() for index in range(2)] == [2., 4.]
    assert [dataset[index]["sampling_weight_mask"].item() for index in range(2)] == list(mask)


@pytest.mark.parametrize("field,value", [
    ("sampling_selection_probability", 0.), ("sampling_selection_probability", 1.5),
    ("sampling_selection_probability", np.nan), ("sampling_design_weight_raw", 0.),
    ("sampling_design_weight_raw", np.nan), ("sampling_design_weight_raw", 99.),
])
def test_mask_false_does_not_hide_invalid_source_weights(tmp_path, field, value):
    arrays = artificial_arrays(weights=(False, False))
    arrays[field][0] = value
    write_artificial(tmp_path, arrays)
    with pytest.raises(ValueError, match="probabilities|weights"):
        TorchDamageDataset(tmp_path, fields=("case_x",), include_static_graph=False)


def test_label_spec_public_views_are_independent_and_hash_is_bound_to_content():
    original = spec_document()
    spec = LabelSpec(original)
    original["tasks"]["events"]["outputs"][0] = "changed-original"
    view = spec.data
    view["tasks"]["events"]["outputs"][0] = "changed-view"
    spec.tasks.clear()
    assert spec.task("events").outputs == ("event-a", "event-b")
    assert spec.hash == content_hash(spec.data)
    for name in ("data", "tasks", "hash"):
        with pytest.raises(AttributeError):
            setattr(spec, name, {})
    restored = pickle.loads(pickle.dumps(spec))
    assert restored.data == spec.data and restored.hash == spec.hash


@pytest.mark.parametrize("task_type", ["ordinal_cumulative", "mutually_exclusive_class"])
def test_label_tasks_reject_incompatible_actual_values(task_type):
    spec = LabelSpec(spec_document(task_type))
    with pytest.raises(InputValidationError, match="monotonically|sum to one"):
        spec.validate_system_outputs(["event-a", "event-b"], [.8, .9], [True, True])


def test_label_validation_uses_task_order_and_does_not_relabel_raw_events():
    raw = LabelSpec(spec_document())
    assert bool(raw.validate_system_outputs(["event-a", "event-b"], [.8, .9])["events"])
    ordinal = LabelSpec(spec_document("ordinal_cumulative"))
    assert bool(ordinal.validate_system_outputs(["event-b", "event-a"], [.2, .8])["events"])
    exclusive = LabelSpec(spec_document("mutually_exclusive_class"))
    assert bool(exclusive.validate_system_outputs(["event-a", "event-b"], [.4, .6])["events"])
    with pytest.raises(InputValidationError, match="absent"):
        ordinal.validate_system_outputs(["event-a"], [.8])
    assert not bool(ordinal.validate_system_outputs(["event-a", "event-b"], [.8, .2], [True, False])["events"])
    assert not bool(ordinal.validate_system_outputs(["event-a", "event-b"], [np.nan, np.nan], [False, False])["events"])


def availability_nodes():
    return {
        "source-a": {"node_type": "leaf", "component_id": "abstract-holder", "mechanism": "fragment", "damage_state": "abstract-state", "inputs": []},
        "source-b": {"node_type": "leaf", "component_id": "abstract-holder", "mechanism": "blast", "damage_state": "abstract-state", "inputs": []},
        "combined": {"node_type": "leaf", "component_id": "abstract-holder", "mechanism": "combined", "damage_state": "abstract-state", "inputs": []},
        "joint": {"node_type": "or", "inputs": ["source-a", "source-b"]},
        "projection": {"node_type": "standard_projection", "inputs": ["joint"]},
    }


def test_availability_masks_propagate_missing_sources_without_evaluating_events():
    nodes = availability_nodes()
    assert system_output_availability(nodes, ["source-a", "source-b", "combined", "projection"], ["fragment"]) == {
        "source-a": True, "source-b": False, "combined": False, "projection": False,
    }
    assert system_output_availability(nodes, ["projection", "combined"], ["fragment", "blast"]) == {"projection": True, "combined": True}
    availability = {("abstract-holder", "fragment", "abstract-state"): True,
                    ("abstract-holder", "blast", "abstract-state"): False}
    assert system_output_availability(nodes, ["projection"], ["fragment", "blast"], component_availability=availability) == {"projection": False}
    nodes["joint"]["inputs"] = ["source-b", "missing"]
    with pytest.raises(InputValidationError, match="missing dependency"):
        system_output_availability(nodes, ["projection"], ["fragment"])


@pytest.mark.parametrize("task_type", ["ordinal_cumulative", "mutually_exclusive_class"])
def test_public_reader_checks_saved_task_definitions_against_actual_labels(tmp_path, task_type):
    arrays = artificial_arrays(version="1.6.0")
    arrays["system_y"][0] = [.8, .9]
    manifest = artificial_manifest(arrays, version="1.6.0")
    spec = LabelSpec(spec_document(task_type))
    manifest["source_contracts"].update(label_spec_document=spec.data, label_spec_hash=spec.hash)
    write_artificial(tmp_path, arrays, manifest, version="1.6.0")
    with pytest.raises(InputValidationError, match="monotonically|sum to one"):
        TorchDamageDataset(tmp_path, fields=("case_x",), include_static_graph=False)


@pytest.mark.parametrize("task_type,task_usable", [("ordinal_cumulative", False), ("multi_label_event_probability", True)])
def test_source_availability_and_task_masks_preserve_known_individual_outputs(tmp_path, task_type, task_usable):
    arrays = artificial_arrays(version="1.6.0")
    arrays["system_source_available_mask"][0, 1] = False
    arrays["system_label_mask"][0, 1] = False
    arrays["system_uncertainty_mask"][0, 1] = False
    arrays["system_sample_count"][0, 1] = 0
    arrays["label_task_available_mask"][0] = task_usable
    manifest = artificial_manifest(arrays, version="1.6.0")
    spec = LabelSpec(spec_document(task_type))
    manifest["source_contracts"].update(label_spec_document=spec.data, label_spec_hash=spec.hash)
    write_artificial(tmp_path, arrays, manifest, version="1.6.0")
    dataset = TorchDamageDataset(tmp_path, include_static_graph=False)
    assert dataset[0]["system_label_mask"].tolist() == [True, False]
    assert dataset[0]["label_task_available_mask"].tolist() == [task_usable]
    assert dataset[0]["sampling_epistemic_group_id"] == "epistemic-0"


def test_optional_source_availability_is_honored_in_new_base_exports(tmp_path):
    arrays = artificial_arrays(version="1.4.0")
    arrays["system_source_available_mask"] = np.zeros((2, 2), bool)
    arrays["system_label_mask"][:] = False
    arrays["system_uncertainty_mask"][:] = False
    arrays["system_sample_count"][:] = 0
    write_artificial(tmp_path, arrays, version="1.4.0")
    dataset = TorchDamageDataset(tmp_path, include_static_graph=False)
    assert not dataset[0]["system_label_mask"].any()
    assert not dataset[0]["system_source_available_mask"].any()


@pytest.mark.parametrize("missing", ["label_spec_document", "split_registry_document", "system_source_available_mask", "label_task_available_mask"])
def test_v16_subsets_cannot_hide_missing_provenance_or_availability(tmp_path, missing):
    arrays = artificial_arrays(version="1.6.0")
    manifest = artificial_manifest(arrays, version="1.6.0")
    if missing == "label_spec_document":
        del manifest["source_contracts"][missing]
    elif missing == "split_registry_document":
        del manifest["generation"]["sampling"][missing]
    else:
        del arrays[missing]
        manifest = artificial_manifest(arrays, version="1.6.0")
    write_artificial(tmp_path, arrays, manifest, version="1.6.0")
    with pytest.raises((ValueError, InputValidationError), match="document|registry|source_available|task_available"):
        TorchDamageDataset(tmp_path, fields=("case_x",), include_static_graph=False)


def test_refused_input_assignments_and_candidates_remain_in_archive_admission(tmp_path):
    arrays = artificial_arrays(version="1.6.0")
    # A third input was refused and is absent from the valid-case tensor axis.
    for field, value in (
        ("input_case_ids", "refused-case"), ("input_case_status", "refused"),
        ("input_case_valid_mask", False), ("input_to_valid_index", -1),
        ("input_sampling_candidate_ids", "candidate-refused"),
        ("input_sampling_lineage_ids", "lineage-refused"),
        ("input_sampling_roles", "reference_pilot"),
        ("input_sampling_epistemic_group_ids", "epistemic-refused"),
        ("input_frozen_splits", "test"),
    ):
        arrays[field] = np.append(arrays[field], value).astype(arrays[field].dtype if arrays[field].dtype.kind not in "US" else str)
    manifest = artificial_manifest(arrays, version="1.6.0")
    registry = SplitRegistry.from_dict(manifest["generation"]["sampling"]["split_registry_document"])
    # Build a fresh registry containing the refused row before any results.
    extended = SplitRegistry(registry.registry_id)
    for row in registry.to_dict()["assignments"]:
        extended.register(**row)
    extended.register(lineage_id="lineage-refused", split="test", role="reference_pilot", epistemic_group_id="epistemic-refused")
    extended.freeze()
    manifest["source_contracts"]["split_registry_hash"] = extended.definition_hash
    manifest["generation"]["sampling"]["split_registry_document"] = extended.to_dict()
    write_artificial(tmp_path, arrays, manifest, version="1.6.0")
    assert len(TorchDamageDataset(tmp_path, fields=("case_x",), include_static_graph=False)) == 2
    arrays["input_frozen_splits"][-1] = "train"
    write_artificial(tmp_path, arrays, manifest, version="1.6.0")
    with pytest.raises(InputValidationError, match="actual frozen split registry"):
        TorchDamageDataset(tmp_path, fields=("case_x",), include_static_graph=False)
    arrays["input_frozen_splits"][-1] = "test"
    arrays["input_sampling_candidate_ids"][-1] = "candidate-0"
    write_artificial(tmp_path, arrays, manifest, version="1.6.0")
    with pytest.raises(InputValidationError, match="globally unique"):
        TorchDamageDataset(tmp_path, fields=("case_x",), include_static_graph=False)


def test_multisource_same_name_requires_known_compatible_definitions(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    write_artificial(first, version="1.4.0")
    write_artificial(second, version="1.4.0")
    with pytest.raises(ValueError, match="unknown task/event definitions"):
        TorchMultiTargetDataset([first, second], fields=("case_x",))
    legacy = TorchMultiTargetDataset([first, second], fields=("case_x",), allow_legacy_semantic_combination=True)
    assert legacy.audit_overrides == ("allow_legacy_semantic_combination=True",)
    arrays = artificial_arrays(version="1.6.0")
    write_artificial(first, arrays, version="1.6.0")
    write_artificial(second, arrays, version="1.6.0")
    compatible = TorchMultiTargetDataset([first, second], include_static_graph=False)
    assert compatible.collate_fn([compatible[0]])["label_task_names"] == ("events",)
    assert compatible.collate_fn([compatible[0]])["label_task_available_mask"].shape == (1, 1)
    changed = artificial_manifest(arrays, version="1.6.0")
    spec = LabelSpec(spec_document("ordinal_cumulative"))
    changed["source_contracts"].update(label_spec_document=spec.data, label_spec_hash=spec.hash)
    write_artificial(second, arrays, changed, version="1.6.0")
    with pytest.raises(ValueError, match="incompatible.*definitions"):
        TorchMultiTargetDataset([first, second], fields=("case_x",), allow_legacy_semantic_combination=True)


def test_global_sharded_candidate_guard_is_independent_of_case_ids(tmp_path):
    root_partitions, records = [], []
    root_manifest = None
    for index in range(2):
        arrays = artificial_arrays(version="1.5.0")
        # Distinct execution names do not make repeated candidates distinct.
        for name in ("case_ids", "case_hashes", "split_group_ids", "leakage_keys"):
            arrays[name] = np.asarray([f"shard-{index}-{value}" for value in arrays[name]])
        directory = tmp_path / "shards" / f"part-{index:06d}"
        manifest = write_artificial(directory, arrays)
        root_manifest = manifest if root_manifest is None else root_manifest
        records.append({"index": index, "path": directory.relative_to(tmp_path).as_posix(),
                        "dataset_id": manifest["dataset_id"], "manifest_hash": file_hash(directory / "manifest.json"),
                        "case_ids": arrays["case_ids"].astype(str).tolist(),
                        "case_hashes": arrays["case_hashes"].astype(str).tolist()})
        root_partitions.extend([{**partition, "path_glob": (directory.relative_to(tmp_path) / partition["path_glob"]).as_posix()}
                                for partition in manifest["partitions"]])
    root_manifest = {**root_manifest, "dataset_id": "artificial-sharded", "partitions": root_partitions, "quality_summary": {"n_valid": 4}}
    (tmp_path / "manifest.json").write_text(json.dumps(root_manifest), encoding="utf-8")
    checkpoint_identity = {"shard_size": 2}
    checkpoint_identity["identity_hash"] = content_hash(checkpoint_identity)
    (tmp_path / "checkpoint.json").write_text(json.dumps({
        "schema_version": "2.0.0", "identity": checkpoint_identity,
        "dataset_id": root_manifest["dataset_id"], "status": "complete", "completed_shards": records,
        "root_manifest_hash": file_hash(tmp_path / "manifest.json"),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate IDs.*duplicated"):
        TorchShardedDamageDataset(tmp_path, fields=("case_x",), include_static_graph=False)


def test_task_availability_and_spec_identity_cannot_be_redeclared_silently(tmp_path):
    arrays = artificial_arrays(version="1.6.0")
    arrays["label_task_available_mask"][:] = False
    write_artificial(tmp_path, arrays, version="1.6.0")
    with pytest.raises(ValueError, match="actual task availability"):
        TorchDamageDataset(tmp_path, fields=("case_x",), include_static_graph=False)
    arrays["label_task_available_mask"][:] = True
    manifest = artificial_manifest(arrays, version="1.6.0")
    manifest["source_contracts"]["label_spec_document"]["tasks"]["events"]["outputs"].reverse()
    write_artificial(tmp_path, arrays, manifest, version="1.6.0")
    with pytest.raises(ValueError, match="content/identity"):
        TorchDamageDataset(tmp_path, fields=("case_x",), include_static_graph=False)
