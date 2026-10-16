"""Generic scalar plans and artificial precomputed archives; no engine.run."""
from copy import deepcopy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import damage_engine.sampling_checkpoint as sampling_module
from damage_engine.archive_validation import ArchiveIntegrityError
from damage_engine.case_factory import CaseFactory, load_case_template
from damage_engine.dataset import DatasetWriter
from damage_engine.errors import InputValidationError
from damage_engine.input_identity import case_identity_hash, single_writer_lock
from damage_engine.learning_input_identity import learning_input_fingerprint
from damage_engine.sampling_candidate_schema import candidate_table
from damage_engine.sampling_checkpoint import initialize_sampling_run, mark_pilot_complete, sampling_run_transaction, verify_sampling_checkpoint
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.sharded_dataset import ShardedDatasetWriter
from damage_engine.utils import content_hash, file_hash
from test_sampling_checkpoint import _label_spec, _sampling_spec
from test_writer_integrity_review import synthetic_writer


@pytest.fixture
def scalar_plan(monkeypatch, synthetic_writer):
    engine, pair = synthetic_writer
    case, _ = pair()
    case["toy_scalar"] = 0.0
    document = _sampling_spec().data
    document["reference_distribution"]["variables"] = [{"name": "scalar", "path": "/toy_scalar", "distribution": "uniform", "low": 0.0, "high": 1.0}]
    spec = SamplingSpec(document)
    label = _label_spec(engine)
    monkeypatch.setattr(sampling_module, "validate_inputs", lambda *_: None)
    return engine, pair, case, spec, label


def _initialize(root, fixture, **kwargs):
    engine, _, case, spec, label = fixture
    return initialize_sampling_run(root, engine=engine, base_case=case, sampling_spec=spec, label_spec=label, **kwargs)


def _factory(fixture, spec):
    engine, _, case, _, label = fixture
    return CaseFactory(case, spec, label, target_hash=engine.target_hash, damage_tree_definition_hash=content_hash(engine.target_data["damage_tree"]), physics_registry_hash=engine.registry.hash, engine_code_hash=sampling_module.ENGINE_CODE_HASH)


def test_sampling_spec_public_nested_mutation_cannot_change_hash_or_behavior(scalar_plan):
    spec = scalar_plan[3]
    original_hash = spec.hash
    exposed = spec.data
    exposed["reference_distribution"]["variables"][0]["low"] = 99
    variables = spec.variables
    variables[0]["low"] = 42
    assert spec.data["reference_distribution"]["variables"][0]["low"] == 0
    assert spec.hash == content_hash(spec.data) == original_hash
    assert pickle.loads(pickle.dumps(spec)).hash == original_hash
    with pytest.raises(AttributeError):
        spec.hash = "spoofed"
    with pytest.raises(AttributeError):
        spec.data = exposed


@pytest.mark.parametrize("document", [
    {"warhead": None}, {"warhead": []},
    {"warhead": {"fragment_source": []}},
    {"warhead": {"fragment_source": {"fragment_list_ref": None}}},
    {"warhead": {"fragment_source": {"fragment_list_ref": 3}}},
    {"warhead": {"fragment_source": {"fragment_list_ref": " "}}},
])
def test_load_case_template_invalid_nested_inputs_raise_structured_error(tmp_path, document):
    source = tmp_path / "case.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(InputValidationError, match="base case"):
        load_case_template(source)


@pytest.mark.parametrize("variable", [
    {"name": "scalar", "path": "/toy_scalar", "distribution": "choice", "values": [0]},
    {"name": "scalar", "path": "/toy_scalar", "distribution": "choice", "values": [0, 0, 0]},
    {"name": "scalar", "path": "/toy_scalar", "distribution": "integer_uniform", "low": 0, "high": 1},
])
def test_discrete_repetition_cannot_cross_split_via_new_ids(scalar_plan, variable):
    document = scalar_plan[3].data
    document["reference_distribution"]["variables"] = [variable]
    with pytest.raises(InputValidationError, match="duplicate learning input crosses"):
        _factory(scalar_plan, SamplingSpec(document)).build_reference_plan()


def test_learning_fingerprint_ignores_only_execution_identity(scalar_plan):
    case = scalar_plan[2]
    same = deepcopy(case)
    same["case_id"] = "new-id"
    same["metadata"] = {"split_group": "new-group"}
    same["analysis"]["aleatory_seed"] = 123456
    same["analysis"]["epistemic_sample_id"] = "new-index-name"
    same["analysis"].update(replicates=999, adaptive_ci_half_width=0.01,
                            adaptive_physical_ci={"unrelated_budget": 1})
    assert learning_input_fingerprint(case) == learning_input_fingerprint(same)
    same["analysis"]["mechanisms"] = ["fragment"]
    assert learning_input_fingerprint(case) != learning_input_fingerprint(same)


def test_repeat_plan_identity_does_not_include_registry_freeze_time(scalar_plan):
    first = _factory(scalar_plan, scalar_plan[3]).build_reference_plan()
    second = _factory(scalar_plan, scalar_plan[3]).build_reference_plan()
    assert first.cases == second.cases
    assert first.candidate_rows == second.candidate_rows


def test_same_split_repeats_record_relationship(scalar_plan, monkeypatch):
    document = scalar_plan[3].data
    document["budgets"].update(train_pilot=3, validation=1, test=1)
    document["reference_distribution"]["variables"] = [{"name": "scalar", "path": "/toy_scalar", "distribution": "choice", "values": [0, 1, 2]}]
    # Distinct validation/test inputs; only the three train rows repeat.
    monkeypatch.setattr("damage_engine.case_factory.randomized_lhs", lambda *_args, **_kwargs: np.asarray([[0.1], [0.4], [0.8], [0.8], [0.8]]))
    plan = _factory(scalar_plan, SamplingSpec(document)).build_reference_plan()
    assert all(row["duplicate_of_candidate_id"] is None for row in plan.candidate_rows[:3])
    assert all(row["duplicate_of_candidate_id"] == plan.candidate_rows[2]["candidate_id"] for row in plan.candidate_rows[3:])


def test_input_bytes_and_environment_are_outer_resume_identity(tmp_path, scalar_plan, monkeypatch):
    source = tmp_path / "source.json"
    source.write_bytes(b"[0]")
    scalar_plan[2]["warhead"] = {"fragment_source": {"fragment_list_ref": str(source)}}
    root = tmp_path / "sampling"
    checkpoint, plan = _initialize(root, scalar_plan)
    original_digest = checkpoint["identity"]["input_dependency_manifest_hash"]
    snapshots = list((root / "plan/input_snapshots").iterdir())
    assert len(snapshots) == 1 and snapshots[0].read_bytes() == b"[0]"
    source.write_bytes(b"[1]")
    assert Path(plan.cases[0]["warhead"]["fragment_source"]["fragment_list_ref"]).read_bytes() == b"[0]"
    verify_sampling_checkpoint(root)
    with pytest.raises(InputValidationError, match="identity differs"):
        _initialize(root, scalar_plan, resume=True)
    source.write_bytes(b"[0]")
    relocated = tmp_path / "relocated.json"
    relocated.write_bytes(b"[0]")
    scalar_plan[2]["warhead"]["fragment_source"]["fragment_list_ref"] = str(relocated)
    resumed, _ = _initialize(root, scalar_plan, resume=True)
    assert resumed["identity"]["input_dependency_manifest_hash"] == original_digest
    environment = deepcopy(checkpoint["identity"]["environment"])
    environment["dependencies"]["numpy"] = "changed-toy-version"
    monkeypatch.setattr(sampling_module, "capture_environment_identity", lambda: environment)
    with pytest.raises(InputValidationError, match="identity differs"):
        _initialize(root, scalar_plan, resume=True)


@pytest.mark.parametrize("defect", ["status", "duplicate_file", "missing_file", "rounds", "orphan_round"])
def test_sampling_checkpoint_complete_plan_invariants(tmp_path, scalar_plan, defect):
    root = tmp_path / "sampling"
    checkpoint, _ = _initialize(root, scalar_plan)
    if defect == "status":
        checkpoint["status"] = "unknown"
    elif defect == "duplicate_file":
        checkpoint["committed_files"].append(checkpoint["committed_files"][0])
    elif defect == "missing_file":
        checkpoint["committed_files"].pop()
    elif defect == "rounds":
        checkpoint["rounds_committed"] = [0, 0]
    else:
        (root / "rounds/round-001").mkdir()
    (root / "sampling_checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(InputValidationError):
        verify_sampling_checkpoint(root)


def test_second_sampling_writer_is_refused_without_output_mutation(tmp_path, scalar_plan):
    root = tmp_path / "sampling"
    with single_writer_lock(root):
        with pytest.raises(InputValidationError, match="single-writer"):
            _initialize(root, scalar_plan)
        assert not root.exists()
    with sampling_run_transaction(root):
        checkpoint, _ = _initialize(root, scalar_plan)
        assert checkpoint["status"] == "initialized"
    assert verify_sampling_checkpoint(root)["status"] == "initialized"


def test_identity_strings_only_manifest_cannot_mark_archive_complete(tmp_path, scalar_plan):
    root = tmp_path / "sampling"
    checkpoint, _ = _initialize(root, scalar_plan)
    manifest = root / "formal_dataset/manifest.json"
    manifest.parent.mkdir()
    identity = checkpoint["identity"]
    manifest.write_text(json.dumps({"engine": {"code_hash": identity["engine_code_hash"], "configuration_hash": identity["physics_registry_hash"]}, "source_contracts": {"target_hashes": [identity["target_hash"]], "sampling_design_hash": identity["sampling_spec_hash"], "label_spec_hash": identity["label_spec_hash"], "split_registry_hash": checkpoint["split_registry_hash"], "damage_tree_definition_hash": identity["damage_tree_definition_hash"]}}), encoding="utf-8")
    before = (root / "sampling_checkpoint.json").read_bytes()
    with pytest.raises(ArchiveIntegrityError) as caught:
        mark_pilot_complete(root, manifest)
    assert caught.value.context["integrity_kind"] == "inventory_mismatch"
    assert (root / "sampling_checkpoint.json").read_bytes() == before


def _write_artificial_formal(root, fixture, plan, monkeypatch, *, refused_ids=()):
    engine, pair, _, spec, _ = fixture
    def precomputed(self, active_engine, cases, **kwargs):
        cases = list(cases)
        results = []
        for case in cases:
            _, result = pair(case["case_id"], count=spec.formal_replicates)
            result.payload["provenance"].update(case_hash=case_identity_hash(case), aleatory_seed=case["analysis"]["aleatory_seed"], epistemic_sample_id=case["analysis"]["epistemic_sample_id"])
            if case["case_id"] in refused_ids:
                result.payload["status"] = "refused"
                result.payload["refusal"] = {"code": "REF_ARTIFICIAL", "message": "Artificial refusal used to test archive counts"}
                result.payload["provenance"]["replicate_count"] = 0
            results.append(result)
        return self.write(active_engine, cases, results)
    monkeypatch.setattr(DatasetWriter, "run_cases", precomputed)
    return ShardedDatasetWriter(root / "formal_dataset", shard_size=2).run_cases(engine, plan.cases)


def test_actual_artificial_archive_completion_checks_full_inputs_and_keeps_quality_separate(tmp_path, scalar_plan, monkeypatch):
    root = tmp_path / "sampling"
    checkpoint, plan = _initialize(root, scalar_plan)
    _write_artificial_formal(root, scalar_plan, plan, monkeypatch)
    completed = mark_pilot_complete(root, root / "formal_dataset/manifest.json")
    assert completed["status"] == "archive_complete"
    assert completed["archive_summary"]["execution_success_count"] == 3
    assert completed["archive_summary"]["purpose_admission_counts"]["research_training"] == 0
    assert completed["archive_summary"]["scientific_validation_status"] == "not_established_by_archive_completion"
    assert verify_sampling_checkpoint(root)["status"] == "archive_complete"


def test_sampling_checkpoint_entry_preserves_common_archive_tamper_error(
    tmp_path, scalar_plan, monkeypatch,
):
    root = tmp_path / "sampling"
    _, plan = _initialize(root, scalar_plan)
    _write_artificial_formal(root, scalar_plan, plan, monkeypatch)
    tensor_path = root / "formal_dataset/shards/part-000000/tensors.npz"
    with tensor_path.open("ab") as stream:
        stream.write(b"tampered-before-sampling-bind")

    with pytest.raises(ArchiveIntegrityError) as raised:
        mark_pilot_complete(root, root / "formal_dataset/manifest.json")
    assert raised.value.context["integrity_kind"] == "partition_hash_mismatch"
    assert raised.value.context["archive_scope"] == "shard"
    assert raised.value.context["shard_index"] == 0
    assert raised.value.context["path"].endswith("tensors.npz")


def test_archive_completion_counts_refused_executions_without_admitting_them(tmp_path, scalar_plan, monkeypatch):
    root = tmp_path / "sampling"
    _, plan = _initialize(root, scalar_plan)
    _write_artificial_formal(root, scalar_plan, plan, monkeypatch, refused_ids=[plan.cases[0]["case_id"]])
    completed = mark_pilot_complete(root, root / "formal_dataset/manifest.json")
    summary = completed["archive_summary"]
    assert summary["formal_input_count"] == summary["executed_candidate_count"] == 3
    assert summary["execution_success_count"] == 2 and summary["refusal_count"] == 1
    sets = summary["candidate_sets"]
    assert sets["candidate_frame"]["count"] == sets["selected"]["count"] == sets["executed"]["count"] == 3
    assert sets["successful"]["count"] == sets["admitted_by_purpose"]["software_development"]["count"] == 2
    assert sets["refused"]["candidate_ids"] == [plan.candidate_rows[0]["candidate_id"]]
    assert all(sets["admitted_by_purpose"][purpose]["count"] == 0 for purpose in ("research_training", "research_validation", "research_test"))
    assert verify_sampling_checkpoint(root)["archive_summary"] == summary


@pytest.mark.parametrize("defect", ["marker", "orphan", "wrong_inputs", "wrong_candidates", "fake_admission"])
def test_artificial_archive_cannot_complete_with_missing_invariants(tmp_path, scalar_plan, monkeypatch, defect):
    root = tmp_path / "sampling"
    checkpoint, plan = _initialize(root, scalar_plan)
    _write_artificial_formal(root, scalar_plan, plan, monkeypatch)
    formal = root / "formal_dataset"
    if defect == "marker":
        (formal / ".dataset.incomplete").write_text("toy evidence")
    elif defect == "orphan":
        (formal / "shards/part-000002").mkdir()
    elif defect in {"wrong_inputs", "wrong_candidates"}:
        data = json.loads((formal / "checkpoint.json").read_text())
        field = "case_ids" if defect == "wrong_inputs" else "candidate_ids"
        data["completed_shards"][0][field][0] = "wrong-case"
        (formal / "checkpoint.json").write_text(json.dumps(data), encoding="utf-8")
    else:
        manifest = json.loads((formal / "manifest.json").read_text())
        manifest["quality_summary"]["purpose_admission_counts"]["research_training"] = 1
        (formal / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        data = json.loads((formal / "checkpoint.json").read_text())
        data["root_manifest_hash"] = file_hash(formal / "manifest.json")
        (formal / "checkpoint.json").write_text(json.dumps(data), encoding="utf-8")
    before = (root / "sampling_checkpoint.json").read_bytes()
    with pytest.raises(InputValidationError):
        mark_pilot_complete(root, formal / "manifest.json")
    assert (root / "sampling_checkpoint.json").read_bytes() == before


def test_candidate_parquet_keeps_late_optional_columns_and_refuses_unknown(scalar_plan):
    plan = _factory(scalar_plan, scalar_plan[3]).build_reference_plan()
    rows = deepcopy(plan.candidate_rows)
    rows[1]["executed"] = True
    rows[1]["successful"] = False
    rows[1]["refusal"] = {"code": "TOY", "message": "artificial", "context_json": "{}"}
    table = candidate_table(rows)
    assert table.to_pylist()[0]["executed"] is None
    assert table.to_pylist()[1]["executed"] is True
    assert table.to_pylist()[1]["refusal"]["code"] == "TOY"
    rows[1]["unknown_optional"] = "must-not-disappear"
    with pytest.raises(InputValidationError, match="unknown"):
        candidate_table(rows)
