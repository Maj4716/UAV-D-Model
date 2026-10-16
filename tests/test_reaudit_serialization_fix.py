"""Generic artificial records only; no engine execution or physical labels."""
from copy import deepcopy
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from damage_engine.archive_validation import validate_tensor_arrays
from damage_engine.case_factory import CaseFactory
from damage_engine.dataset import DatasetWriter, _write_parquet
from damage_engine.engine import ENGINE_CODE_HASH, EngineResult
from damage_engine.errors import InputValidationError
from damage_engine.label_spec import LabelSpec
from damage_engine.parquet_schema import PARQUET_SCHEMA_VERSION, TABLE_NAMES, table_schema
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.torch_dataset import TorchDamageDataset
from damage_engine.utils import content_hash, statistics_dict
from damage_engine.validation import _validate_schema
from test_writer_integrity_review import synthetic_writer


def label_spec(engine, task_type="multi_label_event_probability"):
    return LabelSpec({"schema_version": "1.0.0", "label_spec_id": "toy-labels", "revision": "1",
        "status": "provisional", "source": {"damage_tree_id": "toy-tree", "damage_tree_revision": engine.damage_tree.version,
        "damage_tree_definition_hash": content_hash(engine.target_data["damage_tree"])},
        "tasks": {"toy-task": {"type": task_type, "outputs": list(engine.damage_tree.outputs),
                              "loss": "binary_cross_entropy" if task_type != "mutually_exclusive_class" else "cross_entropy"}},
        "provenance": {"approval_status": "not_approved", "evidence_ref": None}})


def sampling_spec(counts=(1, 1, 1)):
    return SamplingSpec({"schema_version": "1.0.0", "sampling_spec_id": "toy-design", "revision": "1", "status": "provisional",
        "reference_distribution": {"distribution_id": "toy-p0", "interpretation": "design_distribution",
            "variables": [{"name": "toy", "path": "/toy_scalar", "distribution": "uniform", "low": 0., "high": 1.}], "constraints": []},
        "budgets": dict(zip(("train_pilot", "validation", "test", "rare_stress_test"), (*counts, 0))),
        "seeds": dict(zip(("design", "selection", "split", "screening", "formal"), range(21, 26))),
        "formal_policy": {"replicates": 2, "adaptive_ci_half_width": None, "epistemic_policy": "unique_per_lineage"},
        "split_policy": {"registry_id": "toy-registry", "freeze_before_outcomes": True, "adaptive_train_only": True},
        "quality_gates": {"positivity_min": .1, "min_ess_fraction": .5, "max_weight_cv": 2.}})


def toy_plan(engine, base, spec=None):
    base = deepcopy(base)
    base["toy_scalar"] = .5
    spec = sampling_spec() if spec is None else spec
    return CaseFactory(base, spec, label_spec(engine), target_hash=engine.target_hash,
        damage_tree_definition_hash=content_hash(engine.target_data["damage_tree"]),
        physics_registry_hash=engine.registry.hash, engine_code_hash=ENGINE_CODE_HASH).build_reference_plan()


def toy_result(case, pair, engine):
    _, result = pair(case["case_id"])
    result.payload["provenance"].update(case_hash=content_hash(case), aleatory_seed=case["analysis"]["aleatory_seed"],
        epistemic_sample_id=case["analysis"]["epistemic_sample_id"], target_hash=engine.target_hash)
    return result


def test_parquet_nested_statistics_survive_first_row_and_reverse(tmp_path):
    common = {"case_id": "toy", "component_id": "toy-component", "damage_state": "toy", "available": True}
    base = statistics_dict([.1, .3], probability=True)
    rows = [{**common, "mechanism": "fragment", **base},
            {**common, "mechanism": "combined", **base,
             "frechet_union_lower_bound": statistics_dict([.1, .2], probability=True),
             "frechet_union_upper_bound": statistics_dict([.4, .5], probability=True)}]
    for directory, order in (("forward", rows), ("reverse", list(reversed(rows)))):
        target = tmp_path / directory
        target.mkdir()
        _write_parquet(target / "labels.parquet", order)
        table = pq.read_table(target / "labels.parquet")
        assert table.schema.metadata[b"table_schema_version"].decode() == PARQUET_SCHEMA_VERSION
        actual = next(row for row in table.to_pylist() if row["mechanism"] == "combined")
        for field in ("frechet_union_lower_bound", "frechet_union_upper_bound"):
            assert actual[field] == rows[1][field]
    assert pq.read_schema(tmp_path / "forward/labels.parquet") == pq.read_schema(tmp_path / "reverse/labels.parquet")


@pytest.mark.parametrize("table", TABLE_NAMES)
def test_empty_tables_have_explicit_typed_schemas(tmp_path, table):
    _write_parquet(tmp_path / f"{table}.parquet", [])
    schema = pq.read_schema(tmp_path / f"{table}.parquet")
    assert schema.equals(table_schema(table), check_metadata=True)
    assert not any(pa.types.is_null(field.type) for field in schema)


@pytest.mark.parametrize("nested", [False, True])
def test_parquet_unknown_field_refused_before_write(tmp_path, nested):
    row = {"case_id": "toy", "component_id": "toy-component", "mechanism": "combined",
           "damage_state": "toy", "available": True, **statistics_dict([.1])}
    if nested:
        row["frechet_union_lower_bound"] = {"mean": .1, "future_statistic": 123.}
    else:
        row["future_statistic"] = 123.
    with pytest.raises(InputValidationError, match="unknown"):
        _write_parquet(tmp_path / "labels.parquet", [row])
    assert not (tmp_path / "labels.parquet").exists()


def test_parquet_null_then_estimated_stats_stay_typed(tmp_path):
    common = {"case_id": "toy", "component_id": "toy-component", "mechanism": "fragment",
              "damage_state": "toy", "available": True}
    rows = [{**common, **statistics_dict([.2])}, {**common, **statistics_dict([.1, .3])}]
    _write_parquet(tmp_path / "labels.parquet", rows)
    table = pq.read_table(tmp_path / "labels.parquet")
    assert table.schema.field("stddev").type == pa.float64()
    assert table.schema.field("insufficient_reason").type == pa.string()
    assert table.to_pylist()[0]["stddev"] is None
    assert table.to_pylist()[1]["stddev"] == rows[1]["stddev"]


@pytest.mark.parametrize("mechanisms,available", [(["fragment", "blast"], True), (["fragment"], False), (["blast"], False)])
def test_writer_system_source_mask_and_table_agree(tmp_path, synthetic_writer, mechanisms, available):
    engine, pair = synthetic_writer
    case, result = pair()
    case["analysis"]["mechanisms"] = mechanisms
    result.payload["mechanisms"] = mechanisms
    result.payload["provenance"]["case_hash"] = content_hash(case)
    manifest = DatasetWriter(tmp_path).write(engine, [case], [result])
    with np.load(tmp_path / "tensors.npz", allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    validate_tensor_arrays(arrays, manifest)
    assert arrays["system_source_available_mask"].tolist() == [[available]]
    assert arrays["system_label_mask"].tolist() == [[available]]
    assert arrays["system_sample_count"].tolist() == [[2 if available else 0]]
    assert bool(arrays["system_uncertainty_mask"][0, 0]) is available
    assert pq.read_table(tmp_path / "tree_outputs.parquet").to_pylist()[0]["available"] is available
    if not available:
        assert np.isnan(arrays["system_y_stddev"]).all()


@pytest.mark.parametrize("weight_masks", [[True, False, True], [False, False, False]])
def test_sampling_writer_contract16_strict_reader_roundtrip(tmp_path, synthetic_writer, weight_masks):
    engine, pair = synthetic_writer
    base, _ = pair()
    plan = toy_plan(engine, base)
    for case, flag in zip(plan.cases, weight_masks):
        case["metadata"]["sampling"]["weight_mask"] = flag
    results = [toy_result(case, pair, engine) for case in plan.cases]
    manifest = DatasetWriter(tmp_path).write(engine, plan.cases, results)
    _validate_schema(manifest, "dataset_manifest.schema.json")
    assert manifest["tensor_contract"]["version"] == "1.6.0"
    dataset = TorchDamageDataset(tmp_path)
    assert len(dataset) == 3
    assert dataset[0]["label_task_available_mask"].tolist() == [True]
    assert dataset[0]["system_source_available_mask"].tolist() == [True]
    assert manifest["quality_summary"]["sampling_diagnostics"]["usable_weight_count"] == sum(weight_masks)
    assert manifest["split_policy"]["coverage_guaranteed"] is False
    with np.load(tmp_path / "tensors.npz", allow_pickle=False) as arrays:
        assert arrays["input_sampling_candidate_ids"].shape == (3,)
        assert arrays["sampling_design_weight_raw"].tolist() == [1., 1., 1.]
    table = pq.read_table(tmp_path / "labels.parquet").to_pylist()
    assert all(row["frechet_union_lower_bound"] is not None for row in table if row["mechanism"] == "combined")


def test_writer_rejects_exclusive_semantics_before_complete_archive(tmp_path, synthetic_writer):
    engine, pair = synthetic_writer
    base, _ = pair()
    plan = toy_plan(engine, base)
    spec = label_spec(engine, "mutually_exclusive_class")
    for case in plan.cases:
        case["metadata"]["sampling"].update(label_spec_document=spec.data, label_spec_hash=spec.hash)
    results = [toy_result(case, pair, engine) for case in plan.cases]
    with pytest.raises(InputValidationError, match="sum to one"):
        DatasetWriter(tmp_path).write(engine, plan.cases, results)
    assert not (tmp_path / "manifest.json").exists()


def test_statistics_metadata_does_not_claim_fixed_all_parameters():
    assert statistics_dict([.1, .3])["conditioning"] == "fixed_case_and_epistemic_realization_with_possible_within_replicate_parameter_variation"


def test_writer_rejects_label_document_bound_to_other_tree(tmp_path, synthetic_writer):
    engine, pair = synthetic_writer
    base, _ = pair()
    plan = toy_plan(engine, base)
    document = label_spec(engine).data
    document["source"]["damage_tree_definition_hash"] = content_hash({"other": "toy-tree"})
    spec = LabelSpec(document)
    for case in plan.cases:
        case["metadata"]["sampling"].update(label_spec_document=spec.data, label_spec_hash=spec.hash)
    results = [toy_result(case, pair, engine) for case in plan.cases]
    with pytest.raises(InputValidationError, match="different tree"):
        DatasetWriter(tmp_path).write(engine, plan.cases, results)
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("field", ["case_id", "available", "output_node", "future_statistic"])
def test_statistics_cannot_override_reserved_table_identity_or_availability(tmp_path, synthetic_writer, field):
    engine, pair = synthetic_writer
    case, result = pair()
    result.payload["damage_tree"]["outputs"]["shared-name"][field] = True
    target = tmp_path / "rejected"
    with pytest.raises(InputValidationError, match="unknown or reserved"):
        DatasetWriter(target).write(engine, [case], [result])
    assert not target.exists()


@pytest.mark.parametrize("field,value", [("selection_probability", "not-a-number"), ("design_weight_raw", "1"),
                                        ("formal_seed", "3"), ("screen_seed", -1)])
def test_malformed_sampling_scalars_have_stable_validation_errors(synthetic_writer, field, value):
    from damage_engine.dataset import _sampling_context
    engine, pair = synthetic_writer
    base, _ = pair()
    plan = toy_plan(engine, base)
    plan.cases[0]["metadata"]["sampling"][field] = value
    with pytest.raises(InputValidationError):
        _sampling_context(plan.cases)


def test_all_refused_sampling_archive_retains_input_identity_with_empty_labels(tmp_path, synthetic_writer):
    engine, pair = synthetic_writer
    base, _ = pair()
    plan = toy_plan(engine, base)
    results = []
    for case in plan.cases:
        result = toy_result(case, pair, engine)
        result.payload.update(status="refused", refusal={"code": "REF_TOY", "message": "artificial refusal"})
        result.payload["provenance"]["replicate_count"] = 0
        results.append(result)
    manifest = DatasetWriter(tmp_path).write(engine, plan.cases, results)
    _validate_schema(manifest, "dataset_manifest.schema.json")
    assert manifest["quality_summary"]["n_valid"] == 0
    assert manifest["quality_summary"]["n_refused"] == 3
    assert len(TorchDamageDataset(tmp_path)) == 0
    with np.load(tmp_path / "tensors.npz", allow_pickle=False) as arrays:
        assert arrays["input_sampling_candidate_ids"].shape == (3,)
        assert arrays["label_task_available_mask"].shape == (0, 1)
    assert pq.read_table(tmp_path / "labels.parquet").num_rows == 0


def test_cli_pilot_holds_sampling_lock_across_whole_transaction(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from damage_engine import cli
    root = tmp_path / "toy-run"
    lock = root.parent / ".toy-run.writer.lock"
    base_path = tmp_path / "case-directory" / "base.json"
    spec = SimpleNamespace(budgets={"train_pilot": 1, "validation": 1, "test": 1}, seeds={"split": 5})
    monkeypatch.setattr(cli, "_sampling_inputs", lambda _: (None, {}, spec, None))
    stages = []
    def initialize(*_, **kwargs):
        assert lock.exists()
        assert kwargs["cases_base_directory"] == base_path.parent
        stages.append("initialize")
        return {}, None
    def write(*_, **kwargs):
        assert lock.exists()
        stages.append("write")
        return {"dataset_id": "toy", "tensor_contract": {"version": "1.6.0"},
                "quality_summary": {"n_total": 3, "n_valid": 3, "n_refused": 0}, "split_policy": {"split_counts": {}}}
    def complete(*_):
        assert lock.exists()
        stages.append("complete")
        return {"status": "archive_complete"}
    monkeypatch.setattr(cli, "initialize_sampling_run", initialize)
    monkeypatch.setattr(cli, "_iter_cases", lambda _: iter([]))
    monkeypatch.setattr(cli.ShardedDatasetWriter, "run_cases", write)
    monkeypatch.setattr(cli, "mark_pilot_complete", complete)
    arguments = SimpleNamespace(output_directory=root, base_case=base_path, workers=1, shard_size=2,
                                chunksize=1, resume=False, fail_fast=False, tensor_compression="compressed")
    assert cli.command_sampler_pilot(arguments) == 0
    assert stages == ["initialize", "write", "complete"]
    assert not lock.exists()
