"""Generic candidate/array/file regressions; no simulation, effect model or training."""
from copy import deepcopy
import json

import numpy as np
import pytest

from damage_engine.archive_validation import ArchiveIntegrityError
from damage_engine.errors import InputValidationError
from damage_engine.sampling_audit import audit_sampling
from damage_engine.sampling_validation import (input_sampling_rows_from_arrays, load_verified_split_registry,
                                                validate_candidate_set, validate_registry_assignments)
from damage_engine.sharded_dataset import ShardedDatasetWriter
from damage_engine.split_registry import SplitRegistry
from damage_engine.utils import content_hash, file_hash


GATES = {"positivity_min": .1, "min_ess_fraction": .5, "max_weight_cv": 10.}


def toy_rows():
    probabilities = [.5, .25, 1., .5]
    return [{"candidate_id": f"candidate-{index}", "lineage_id": f"lineage-{index}",
             "epistemic_group_id": f"epistemic-{index}", "split": "train", "role": "toy-role",
             "stream": "toy-stream", "selected": index != 1, "weight_mask": index != 3,
             "inclusion_probability": value, "design_weight_raw": 1. / value}
            for index, value in enumerate(probabilities)]


def toy_registry(rows):
    registry = SplitRegistry("toy-registry")
    for row in rows:
        registry.register(**{key: row[key] for key in ("lineage_id", "split", "role", "epistemic_group_id")})
    registry.freeze()
    return registry


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -.01, 1.01])
def test_latent_values_must_be_finite_and_in_unit_interval(invalid):
    latent = np.asarray([[.2], [.1], [.8], [.9]])
    latent[1, 0] = invalid
    with pytest.raises(InputValidationError, match="finite values in"):
        audit_sampling(toy_rows(), latent, quality_gates=GATES)


@pytest.mark.parametrize("defect", ["candidate", "lineage", "epistemic", "boolean", "masked_weight"])
def test_candidate_integrity_is_checked_before_diagnostics(defect):
    rows = toy_rows()
    if defect == "candidate":
        rows[1]["candidate_id"] = rows[0]["candidate_id"]
    elif defect in {"lineage", "epistemic"}:
        key = "lineage_id" if defect == "lineage" else "epistemic_group_id"
        rows[1][key] = rows[0][key]
        rows[1]["split"] = "test"
    elif defect == "boolean":
        rows[1]["selected"] = "false"
    else:
        rows[3]["design_weight_raw"] = 9.
    with pytest.raises(InputValidationError):
        audit_sampling(rows, np.asarray([[.2], [.1], [.8], [.9]]), quality_gates=GATES)


def test_selected_weighted_cdf_matches_hand_calculation_and_ignores_unusable_rows():
    rows = toy_rows()
    latent = np.asarray([[.2], [.1], [.8], [.9]])
    report = audit_sampling(rows, latent, quality_gates=GATES)
    assert report["candidate_frame_unweighted"]["ks_by_axis"] == pytest.approx([.3])
    assert report["selected_weighted"]["ks_by_axis"] == pytest.approx([2 / 3 - .2])
    assert report["effective_sample_size"] == pytest.approx(9 / 5)
    assert report["selected_count"] == 3 and report["weighted_selected_count"] == 2
    assert report["sets"]["candidate_frame"]["split_counts"] == {"train": 4}
    assert report["sets"]["selected"]["split_counts"] == {"train": 3}
    assert report["sets"]["selected_weight_eligible"]["split_counts"] == {"train": 2}
    assert report["sets"]["successful"]["status"] == "not_evaluated"
    assert report["quality_gate_passed"] is True and report["coverage_guaranteed"] is False
    changed = latent.copy()
    changed[1, 0] = .6  # Unselected unit.
    changed[3, 0] = .05  # Selected but its weight is unavailable for a weighted use.
    second = audit_sampling(rows, changed, quality_gates=GATES)
    assert second["selected_weighted"] == report["selected_weighted"]
    assert second["candidate_frame_unweighted"] != report["candidate_frame_unweighted"]


def test_all_masked_weights_preserve_numeric_records_without_a_weighted_estimate():
    rows = toy_rows()
    for row in rows:
        row["weight_mask"] = False
    original = deepcopy(rows)
    report = audit_sampling(rows, np.asarray([[.2], [.1], [.8], [.9]]), quality_gates=GATES)
    assert rows == original
    assert report["weighted_selected_count"] == 0
    assert report["selected_weighted"]["status"] == "not_estimable"
    assert report["selected_weighted"]["ks_by_axis"] == [None]
    assert report["quality_gate_passed"] is False
    json.dumps(report, allow_nan=False)


def test_execution_success_and_admission_are_distinct_nested_sets():
    report = audit_sampling(toy_rows(), np.asarray([[.2], [.1], [.8], [.9]]), quality_gates=GATES,
                            executed_candidate_ids=["candidate-0", "candidate-2", "candidate-3"],
                            successful_candidate_ids=["candidate-0", "candidate-2"],
                            admitted_candidate_ids=["candidate-0"])
    assert [report["sets"][key]["count"] for key in ("candidate_frame", "selected", "executed", "successful", "admitted")] == [4, 3, 3, 2, 1]
    with pytest.raises(InputValidationError, match="nested"):
        audit_sampling(toy_rows(), np.zeros((4, 1)), quality_gates=GATES, executed_candidate_ids=["candidate-1"])


@pytest.mark.parametrize("field", ["split", "role", "epistemic_group_id"])
def test_real_registry_assignment_cannot_be_overridden_by_its_true_hash(field):
    rows = toy_rows()
    registry = toy_registry(rows)
    loaded = load_verified_split_registry(registry.to_dict(), expected_hash=registry.definition_hash,
                                          expected_id=registry.registry_id)
    validate_registry_assignments(loaded, rows)
    changed = deepcopy(rows)
    changed[0][field] = "test" if field == "split" else "contradictory-value"
    changed[0]["split_registry_hash"] = registry.definition_hash
    with pytest.raises(InputValidationError, match="actual frozen"):
        validate_registry_assignments(loaded, changed)


def test_registry_snapshot_content_and_unique_assignments_are_checked(tmp_path):
    registry = toy_registry(toy_rows())
    path = tmp_path / "registry.json"
    registry.write(path)
    assert load_verified_split_registry(path, expected_hash=registry.definition_hash).definition_hash == registry.definition_hash
    changed = registry.to_dict()
    changed["assignments"][0]["role"] = "changed-role"
    with pytest.raises(InputValidationError, match="definition_hash"):
        load_verified_split_registry(changed, expected_hash=registry.definition_hash)
    duplicate = registry.to_dict()
    duplicate["assignments"].append(deepcopy(duplicate["assignments"][0]))
    with pytest.raises(InputValidationError, match="duplicate"):
        load_verified_split_registry(duplicate)
    with pytest.raises(InputValidationError, match="snapshot"):
        load_verified_split_registry(None)


def test_audit_declared_registry_requires_snapshot_and_matching_registry_id():
    rows = toy_rows()
    registry = toy_registry(rows)
    for row in rows:
        row["split_registry_hash"] = registry.definition_hash
        row["split_registry_id"] = registry.registry_id
    with pytest.raises(InputValidationError, match="complete frozen"):
        audit_sampling(rows, np.zeros((4, 1)), quality_gates=GATES)
    report = audit_sampling(rows, np.zeros((4, 1)), quality_gates=GATES, registry_document=registry.to_dict())
    assert report["registry_assignment_status"] == "assignments_verified"
    rows[0]["split_registry_id"] = "different-registry"
    with pytest.raises(InputValidationError, match="different split registry"):
        audit_sampling(rows, np.zeros((4, 1)), quality_gates=GATES, registry_document=registry.to_dict())


def test_canonical_registry_snapshot_replays_without_fabricating_creation_time():
    first = toy_registry(toy_rows())
    second = toy_registry(toy_rows())
    assert first.definition_document() == second.definition_document()
    assert "frozen_at" in first.to_dict()
    loaded = load_verified_split_registry(first.definition_document(), expected_hash=first.definition_hash)
    assert loaded.to_dict() == first.definition_document()
    assert "frozen_at" not in loaded.to_dict()
    validate_registry_assignments(loaded, toy_rows())
    for invalid_time in (None, "bad-time", "2026-10-11T00:00:00"):
        document = first.definition_document()
        document["frozen_at"] = invalid_time
        with pytest.raises(InputValidationError, match="frozen_at"):
            load_verified_split_registry(document)


def sampled_cases(rows):
    registry = toy_registry(rows)
    return [{"case_id": f"case-{index}", "analysis": {"aleatory_seed": index}, "metadata": {"sampling": {
        **row, "split_registry_document": registry.to_dict(), "split_registry_hash": registry.definition_hash,
        "split_registry_id": registry.registry_id}}} for index, row in enumerate(rows)]


def test_sharded_full_input_candidate_uniqueness_precedes_checkpoint_publication(tmp_path):
    rows = toy_rows()[:2]
    rows[1]["candidate_id"] = rows[0]["candidate_id"]
    cases = sampled_cases(rows)
    cases[1]["metadata"]["future_result_status"] = "refused"  # Still in the full input identity set.
    output = tmp_path / "rejected"
    with pytest.raises(InputValidationError, match="globally unique"):
        ShardedDatasetWriter(output, shard_size=1).run_cases(None, cases)
    assert not (output / "checkpoint.json").exists()


def test_sharded_real_assignment_mismatch_precedes_any_execution(tmp_path):
    cases = sampled_cases(toy_rows()[:2])
    cases[1]["metadata"]["sampling"]["split"] = "test"
    output = tmp_path / "rejected"
    with pytest.raises(InputValidationError, match="actual frozen"):
        ShardedDatasetWriter(output, shard_size=1).run_cases(None, cases)
    assert not (output / "checkpoint.json").exists()


def test_sharded_full_input_repeat_across_splits_precedes_teacher_and_checkpoint(tmp_path, monkeypatch):
    rows = toy_rows()[:2]
    rows[1]["split"] = "test"
    cases = sampled_cases(rows)
    cases[0]["analysis"]["epistemic_sample_id"] = "first-name"
    cases[1]["analysis"]["epistemic_sample_id"] = "second-name"
    cases[1]["metadata"]["future_result_status"] = "refused"
    def forbidden_teacher(*args, **kwargs):
        pytest.fail("teacher execution must not precede full-input split-leakage rejection")
    from damage_engine.dataset import DatasetWriter
    monkeypatch.setattr(DatasetWriter, "run_cases", forbidden_teacher)
    output = tmp_path / "rejected"
    with pytest.raises(InputValidationError, match="identical learning inputs cross"):
        ShardedDatasetWriter(output, shard_size=1).run_cases(None, cases)
    assert not (output / "checkpoint.json").exists()


def test_sharded_same_split_input_repeats_are_allowed_in_preflight():
    ShardedDatasetWriter._validate_input_sampling_identity(sampled_cases(toy_rows()[:2]))


def test_checkpoint_history_and_resume_preserve_candidate_execution_mapping(tmp_path):
    writer = ShardedDatasetWriter(tmp_path / "archive", shard_size=1)
    identity = {"shard_size": writer.shard_size}
    identity["identity_hash"] = content_hash(identity)
    checkpoint = writer._new_checkpoint(identity)
    records = [{"index": index, "path": f"shards/part-{index:06d}", "case_ids": [f"case-{index}"],
                "case_hashes": ["sha256:toy"], "manifest_hash": "sha256:toy", "dataset_id": "toy",
                "candidate_ids": ["same-candidate"]} for index in range(2)]
    checkpoint["completed_shards"] = records
    with pytest.raises(InputValidationError, match="across committed") as caught:
        writer._validate_checkpoint_structure(checkpoint)
    assert isinstance(caught.value, ArchiveIntegrityError)
    assert caught.value.context["integrity_kind"] == "checkpoint_contract_invalid"
    assert "across committed" in caught.value.context["cause_message"]
    case = sampled_cases(toy_rows()[:1])[0]
    record = {**records[0], "case_hashes": [content_hash(case)], "candidate_ids": ["different-candidate"]}
    with pytest.raises(InputValidationError, match="resume candidate"):
        writer._validate_completed_resume([case], {0: record}, tmp_path, checkpoint)


def toy_input_arrays(rows):
    return {"input_case_ids": np.asarray([f"case-{index}" for index in range(len(rows))]),
            "input_sampling_candidate_ids": np.asarray([row["candidate_id"] for row in rows]),
            "input_sampling_lineage_ids": np.asarray([row["lineage_id"] for row in rows]),
            "input_sampling_roles": np.asarray([row["role"] for row in rows]),
            "input_sampling_epistemic_group_ids": np.asarray([row["epistemic_group_id"] for row in rows]),
            "input_frozen_splits": np.asarray([row["split"] for row in rows]),
            "input_case_status": np.asarray(["ok", "refused"][:len(rows)])}


def test_full_input_array_guard_includes_refused_rows():
    rows = toy_rows()[:2]
    arrays = toy_input_arrays(rows)
    assert len(input_sampling_rows_from_arrays(arrays)) == 2
    arrays["input_sampling_candidate_ids"][1] = arrays["input_sampling_candidate_ids"][0]
    with pytest.raises(InputValidationError, match="globally unique"):
        validate_candidate_set(input_sampling_rows_from_arrays(arrays))
    del arrays["input_sampling_epistemic_group_ids"]
    with pytest.raises(InputValidationError, match="full-input"):
        input_sampling_rows_from_arrays(arrays)


@pytest.mark.parametrize("defect", ["duplicate_refused", "missing_registry", "wrong_assignment"])
def test_committed_shard_rechecks_all_candidates_and_actual_registry(tmp_path, defect):
    rows = toy_rows()[:2]
    registry = toy_registry(rows)
    if defect == "duplicate_refused":
        rows[1]["candidate_id"] = rows[0]["candidate_id"]
    elif defect == "wrong_assignment":
        rows[1]["split"] = "test"
    arrays = toy_input_arrays(rows)
    np.savez(tmp_path / "tensors.npz", **arrays)
    manifest = {"dataset_id": "toy", "source_contracts": {"split_registry_hash": registry.definition_hash,
                "split_registry_id": registry.registry_id}, "generation": {"sampling": {} if defect == "missing_registry" else {
                    "split_registry_document": registry.to_dict()}},
                "partitions": [{"name": "tensors", "path_glob": "tensors.npz", "content_hash": file_hash(tmp_path / "tensors.npz")}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    record = {"index": 0, "dataset_id": "toy", "manifest_hash": file_hash(tmp_path / "manifest.json"),
              "case_ids": arrays["input_case_ids"].tolist(), "candidate_ids": arrays["input_sampling_candidate_ids"].tolist()}
    with pytest.raises(InputValidationError):
        ShardedDatasetWriter._verify_completed_shard(record, tmp_path)
