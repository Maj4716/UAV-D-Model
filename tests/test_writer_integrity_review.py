"""Artificial precomputed arrays only: never executes a simulation or trains a model."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

import damage_engine.dataset as dataset_module
from damage_engine.damage_tree import compile_damage_tree
from damage_engine.dataset import DatasetWriter
from damage_engine.engine import ENGINE_CODE_HASH, EngineResult
from damage_engine.errors import InputValidationError
from damage_engine.quality import validate_label_admission
from damage_engine.utils import content_hash, file_hash


def _passing_scientific_validation():
    return {
        "schema_version": "1.0.0",
        "independent_data_sources": [{
            "source_id": "artificial-withheld-validation", "role": "validation",
            "content_hash": "sha256:" + "1" * 64, "used_for_calibration": False,
        }],
        "dimensional_error_metrics": [{
            "metric_id": "artificial-error", "mechanism": "fragment", "unit": "m/s",
            "observed_error": 1.0, "acceptance_limit": 2.0, "passed": True,
        }],
        "interval_coverage": [{
            "metric_id": "artificial-coverage", "mechanism": "blast", "nominal": 0.95,
            "observed": 0.95, "sample_count": 20, "passed": True,
        }],
        "mechanism_stratified_errors": [{
            "stratum_id": "artificial-stratum", "metric_id": "artificial-error",
            "value": 1.0, "acceptance_limit": 2.0, "passed": True,
        }],
        "extrapolation_failure_rules": [{
            "rule_id": "outside-domain", "condition": "outside frozen domain", "action": "refuse",
        }],
        "uncertainty_ledger": {
            "parameter_epistemic": "assessed", "model_form": "assessed",
            "numerical_error": "assessed",
        },
        "approval": {
            "approver_id": "artificial-reviewer", "approved_at": "2026-10-13T00:00:00Z",
            "protocol_version": "artificial-1",
        },
    }


@pytest.fixture
def synthetic_writer(monkeypatch):
    target = {
        "target_id": "artificial-target",
        "components": [{"id": "shared-name", "damage_states": ["intact", "toy-state"]}],
        "damage_tree": {
            "version": "toy-tree", "outputs": ["shared-name"],
            "nodes": [{"id": "shared-name", "node_type": "leaf", "inputs": [],
                       "component_id": "shared-name", "mechanism": "combined",
                       "damage_state": "toy-state", "k": None, "dependency_model_id": None}],
        },
    }
    graph = compile_damage_tree(target)
    component = SimpleNamespace(
        object_id="shared-name", geometry=object(),
        source={"damage_states": ["intact", "toy-state"], "component_type": "toy-type", "name": "artificial"},
        response={"material_id": "toy-material", "fragment_response_model_id": "toy-response",
                  "blast_response": {"model_id": "toy-response"}},
    )
    engine = SimpleNamespace(
        damage_tree=graph, target_data=target, target_hash=content_hash(target),
        compiled_target=SimpleNamespace(components=[component], component_order=("shared-name",), armor=[]),
        registry=SimpleNamespace(hash=content_hash({"registry": "artificial"})),
    )
    # Exercise the writer, not physical feature builders or any teacher computation.
    monkeypatch.setattr(dataset_module, "_case_features", lambda *_: ([0.] * len(dataset_module.CASE_FEATURE_NAMES), [True] * len(dataset_module.CASE_FEATURE_NAMES)))
    monkeypatch.setattr(dataset_module, "_component_features", lambda *_: ([0.] * len(dataset_module.COMPONENT_FEATURE_NAMES), [True] * len(dataset_module.COMPONENT_FEATURE_NAMES)))
    monkeypatch.setattr(dataset_module, "_armor_path_features", lambda *_: (np.zeros((1, 0, 4)), np.zeros((1, 0, 4), dtype=bool)))

    def pair(identifier="toy-a", count=2):
        case = {"case_id": identifier, "analysis": {
            "mechanisms": ["fragment", "blast"], "aleatory_seed": 3,
            "epistemic_sample_id": "toy-index", "evaluation_standard_ids": list(graph.standards),
        }}
        def stats():
            return {"mean": .2, "stddev": .01 if count > 1 else None,
                    "ci95_lower": .1 if count > 1 else None, "ci95_upper": .3 if count > 1 else None,
                    "sample_count": count, "uncertainty_estimable": count > 1}
        states = {mechanism: {"toy-state": stats()} for mechanism in ("fragment", "blast", "combined")}
        for bound in ("lower", "upper"):
            states["combined"]["toy-state"][f"frechet_union_{bound}_bound"] = stats()
        realization = {"shared-name": {}}
        payload = {
            "status": "ok", "case_id": identifier, "target_id": target["target_id"],
            "component_order": ["shared-name"], "replicate_count": count,
            "mechanisms": ["fragment", "blast"], "parameter_realization": realization,
            "armor": {}, "components": {"shared-name": {
                "damage_states": states,
                "fragment_metrics": {key: stats() for key in dataset_module._FRAGMENT_AUXILIARY_KEYS},
                "blast_metrics": {key: stats() for key in dataset_module._BLAST_AUXILIARY_KEYS},
            }},
            "damage_tree": {"outputs": {"shared-name": stats()}},
            "provenance": {"case_hash": content_hash(case), "target_hash": engine.target_hash,
                           "physics_registry_hash": engine.registry.hash, "engine_code_hash": ENGINE_CODE_HASH,
                           "damage_tree_version": graph.version, "aleatory_seed": 3,
                           "epistemic_sample_id": "toy-index", "evaluation_standard_ids": list(graph.standards),
                           "replicate_count": count, "parameter_realization_hash": content_hash(realization)},
        }
        return case, EngineResult(payload)
    return engine, pair


@pytest.mark.parametrize("defect", ["swapped", "case_hash", "target", "registry", "code", "duplicate", "structure", "realization", "probability"])
def test_precomputed_identity_rejected_before_any_output(tmp_path, monkeypatch, synthetic_writer, defect):
    engine, pair = synthetic_writer
    cases, results = zip(pair("toy-a"), pair("toy-b"))
    cases, results = list(cases), list(results)
    if defect == "swapped":
        results.reverse()
    elif defect == "duplicate":
        cases[1] = deepcopy(cases[0])
    elif defect == "structure":
        results[0].payload["components"] = {}
    elif defect == "realization":
        results[0].payload["parameter_realization"]["shared-name"] = {"value": 99}
    elif defect == "probability":
        results[0].payload["damage_tree"]["outputs"]["shared-name"]["mean"] = 1.2
    else:
        key = {"case_hash": "case_hash", "target": "target_hash", "registry": "physics_registry_hash", "code": "engine_code_hash"}[defect]
        results[0].payload["provenance"][key] = "sha256:incorrect"
    output = tmp_path / "rejected"
    writer = DatasetWriter(output)
    import damage_engine.input_identity as input_identity
    def unexpected_snapshot(*_, **__):
        raise AssertionError("invalid precomputed results must fail before any snapshot write")
    monkeypatch.setattr(input_identity, "freeze_case_inputs", unexpected_snapshot)
    with pytest.raises(InputValidationError, match="precomputed"):
        writer.write(engine, cases, results)
    assert not output.exists()


def test_refusal_with_missing_analysis_is_recordable(tmp_path, synthetic_writer):
    engine, _ = synthetic_writer
    case = {"case_id": "artificial-refusal"}
    result = EngineResult({
        "status": "refused", "case_id": case["case_id"],
        "refusal": {"code": "REF_TOY", "message": "Artificial malformed input"},
        "provenance": {"case_hash": content_hash(case), "target_hash": engine.target_hash,
                       "physics_registry_hash": engine.registry.hash, "engine_code_hash": ENGINE_CODE_HASH,
                       "damage_tree_version": engine.damage_tree.version, "aleatory_seed": None,
                       "epistemic_sample_id": None, "evaluation_standard_ids": [], "replicate_count": 0},
    })
    manifest = DatasetWriter(tmp_path).write(engine, [case], [result])
    assert manifest["quality_summary"]["n_refused"] == 1
    assert manifest["quality_summary"]["execution_success_count"] == 0
    with np.load(tmp_path / "tensors.npz", allow_pickle=False) as tensors:
        assert tensors["input_case_valid_mask"].tolist() == [False]
        assert tensors["input_to_valid_index"].tolist() == [-1]
        assert tensors["case_x"].shape[0] == 0


def test_precomputed_count_mismatch_is_stable_validation_error(tmp_path, synthetic_writer):
    engine, pair = synthetic_writer
    case, _ = pair()
    with pytest.raises(InputValidationError, match="equal length"):
        DatasetWriter(tmp_path / "rejected").write(engine, [case], [])
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("invalid", ["warhead_null", "warhead_list", "source_null", "source_list",
                                     "reference_number", "analysis_null", "analysis_list", "metadata_list"])
def test_nonobject_nested_case_fails_before_snapshot_or_output(tmp_path, monkeypatch, synthetic_writer, invalid):
    engine, pair = synthetic_writer
    case, result = pair()
    if invalid.startswith("warhead_"):
        case["warhead"] = None if invalid.endswith("null") else []
    elif invalid.startswith("source_"):
        case["warhead"] = {"fragment_source": None if invalid.endswith("null") else []}
    elif invalid == "reference_number":
        case["warhead"] = {"fragment_source": {"fragment_list_ref": 42}}
    elif invalid.startswith("analysis_"):
        case["analysis"] = None if invalid.endswith("null") else []
    else:
        case["metadata"] = ["malformed"]
    result.payload["provenance"]["case_hash"] = content_hash(case)
    import damage_engine.input_identity as input_identity
    def unexpected_snapshot(*_, **__):
        raise AssertionError("invalid nested input must be rejected without snapshot writes")
    monkeypatch.setattr(input_identity, "freeze_case_inputs", unexpected_snapshot)
    output = tmp_path / "rejected"
    with pytest.raises(InputValidationError, match="precomputed"):
        DatasetWriter(output).write(engine, [case], [result])
    assert not output.exists()


def test_namespace_graph_and_approved_uses(tmp_path, synthetic_writer):
    engine, pair = synthetic_writer
    case, result = pair()
    manifest = DatasetWriter(tmp_path).write(engine, [case], [result])
    with np.load(tmp_path / "tensors.npz", allow_pickle=False) as tensors:
        from damage_engine.archive_validation import validate_tensor_arrays
        validate_tensor_arrays({name: tensors[name] for name in tensors.files}, manifest)
        assert tensors["graph_node_ids"].tolist() == ["shared-name", "shared-name"]
        assert tensors["graph_node_types"].tolist() == ["component", "damage_tree"]
        assert tensors["tree_edge_index"].tolist() == [[0], [1]]
        assert tensors["component_sample_count"].tolist() == [[[[2], [2], [2]]]]
        assert tensors["component_uncertainty_mask"].all()
    import pyarrow.parquet as pq
    edge = pq.read_table(tmp_path / "edges.parquet").to_pylist()[0]
    assert (edge["source_type"], edge["target_type"]) == ("component", "damage_tree")
    assert manifest["quality_summary"]["n_valid_semantics"] == "execution_success_count"
    assert manifest["quality_summary"]["execution_success_count"] == 1
    assert manifest["quality_summary"]["purpose_admission_counts"]["research_training"] == 0
    assert manifest["split_policy"]["independence_guaranteed"] is False
    assert manifest["split_policy"]["split_registry_status"] == "not_supplied"
    axes = {item["axis_id"] for item in manifest["tensor_contract"]["axes"]}
    assert {"E_standard", "E_edge"}.issubset(axes)
    validate_label_admission(manifest)
    with pytest.raises(InputValidationError, match="purpose"):
        validate_label_admission(manifest, "research_training")
    assert "not implemented" not in " ".join(manifest["known_limitations"])
    from damage_engine.validation import _validate_schema
    _validate_schema(manifest, "dataset_manifest.schema.json")


def test_one_observation_mean_remains_usable_uncertainty_masked(tmp_path, synthetic_writer):
    engine, pair = synthetic_writer
    case, result = pair(count=1)
    manifest = DatasetWriter(tmp_path).write(engine, [case], [result])
    assert manifest["quality_summary"]["mc_ci_half_width_quantiles"] == []
    with np.load(tmp_path / "tensors.npz", allow_pickle=False) as tensors:
        from damage_engine.archive_validation import validate_tensor_arrays
        validate_tensor_arrays({name: tensors[name] for name in tensors.files}, manifest)
        assert tensors["component_label_mask"].all()
        assert np.isfinite(tensors["component_y"]).all()
        assert not tensors["component_uncertainty_mask"].any()
        assert np.isnan(tensors["component_y_stddev"]).all()
        assert np.isnan(tensors["component_y_ci95"]).all()
        assert (tensors["component_sample_count"] == 1).all()


@pytest.mark.parametrize("defect", ["missing", "wrong_hash", "expired", "wrong_scope"])
def test_evidence_claim_requires_existing_current_bound_record(tmp_path, synthetic_writer, defect):
    engine, pair = synthetic_writer
    case, result = pair()
    manifest = DatasetWriter(tmp_path / "archive").write(engine, [case], [result])
    admission = manifest["label_admission"]
    admission.update(scientific_evidence_status="independently_reviewed",
                     approved_uses=["research_training"], evaluation_definition_approval_status="approved",
                     source_object_completeness_status="complete")
    binding = {"target_hashes": manifest["source_contracts"]["target_hashes"],
               "parameter_set_hashes": manifest["source_contracts"]["parameter_set_hashes"],
               "engine_code_hash": manifest["engine"]["code_hash"],
               "configuration_hash": manifest["engine"]["configuration_hash"],
               "source_contracts_hash": content_hash(manifest["source_contracts"]),
               "semantic_vocabulary_hash": manifest["semantic_vocabulary"]["content_hash"]}
    evidence = {"status": "approved", "reviewer_id": "artificial-reviewer", "record_version": "1",
                "valid_until": "2000-01-01T00:00:00Z" if defect == "expired" else "2999-01-01T00:00:00Z",
                "approved_uses": ["research_training"], "label_source_type": "synthetic_teacher",
                "object_scope_hash": content_hash(binding) if defect != "wrong_scope" else "sha256:wrong",
                "scientific_validation": _passing_scientific_validation()}
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    admission["evidence_record"] = {"path": "missing.json" if defect == "missing" else path.name,
                                     "content_hash": "sha256:wrong" if defect == "wrong_hash" else file_hash(path)}
    with pytest.raises(InputValidationError, match="evidence"):
        validate_label_admission(manifest, "research_training", evidence_root=tmp_path)


@pytest.mark.parametrize("changed_field", ["registry", "evaluation_standards", "damage_tree_versions"])
def test_evidence_scope_binds_registry_and_complete_source_contracts(tmp_path, synthetic_writer, changed_field):
    engine, pair = synthetic_writer
    case, result = pair()
    manifest = DatasetWriter(tmp_path / "archive").write(engine, [case], [result])
    admission = manifest["label_admission"]
    admission.update(scientific_evidence_status="independently_reviewed",
                     approved_uses=["research_training"], evaluation_definition_approval_status="approved",
                     source_object_completeness_status="complete")
    binding = {"target_hashes": manifest["source_contracts"]["target_hashes"],
               "parameter_set_hashes": manifest["source_contracts"]["parameter_set_hashes"],
               "engine_code_hash": manifest["engine"]["code_hash"],
               "configuration_hash": manifest["engine"]["configuration_hash"],
               "source_contracts_hash": content_hash(manifest["source_contracts"]),
               "semantic_vocabulary_hash": manifest["semantic_vocabulary"]["content_hash"]}
    record = {"status": "approved", "reviewer_id": "artificial-reviewer", "record_version": "1",
              "valid_until": "2999-01-01T00:00:00Z", "approved_uses": ["research_training"],
              "label_source_type": "synthetic_teacher", "object_scope_hash": content_hash(binding),
              "scientific_validation": _passing_scientific_validation()}
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    admission["evidence_record"] = {"path": path.name, "content_hash": file_hash(path)}
    # Establish that refusal is caused by the scope change, not malformed evidence.
    validate_label_admission(manifest, "research_training", evidence_root=tmp_path)
    if changed_field == "registry":
        manifest["engine"]["configuration_hash"] = content_hash({"changed_registry": True})
    elif changed_field == "evaluation_standards":
        manifest["source_contracts"]["evaluation_standards"] = {"changed-definition": ["shared-name"]}
    else:
        manifest["source_contracts"]["damage_tree_versions"] = ["changed-tree-version"]
    with pytest.raises(InputValidationError, match="object/definition scope"):
        validate_label_admission(manifest, "research_training", evidence_root=tmp_path)


def test_realization_key_and_content_digest_are_distinct(target_data, registry):
    from damage_engine.engine import DamageEngine
    # Only provenance serialization; no engine.run, sampling or domain labels.
    engine = DamageEngine(target_data, registry)
    case = {"case_id": "toy", "analysis": {"epistemic_sample_id": "stable-index"}}
    first = engine._provenance(case, replicate_count=0, epistemic_realization={"toy-value": 1})
    second = engine._provenance(case, replicate_count=0, epistemic_realization={"toy-value": 2})
    assert first["epistemic_realization_key"] == second["epistemic_realization_key"]
    assert first["epistemic_realization_hash"] != second["epistemic_realization_hash"]
    assert first["epistemic_realization_hash"] == content_hash({"toy-value": 1})


def test_frozen_input_provenance_preserves_original_location(tmp_path, target_data, registry):
    from damage_engine.engine import DamageEngine
    from damage_engine.input_identity import DEPENDENCY_LOGICAL_ID, FrozenCase, case_identity_hash
    engine = DamageEngine(target_data, registry)
    original = tmp_path / "artificial.json"
    original.write_text('{"artificial_metadata": true}', encoding="utf-8")
    temporary_reference = str(tmp_path / "ephemeral-snapshot.json")
    digest = file_hash(original)
    # Construct the frozen identity manually: this test only checks serialization,
    # and must not depend on parsing, snapshot creation or domain computation.
    snapshot_case = FrozenCase(
        {"case_id": "toy", "warhead": {"fragment_source": {"fragment_list_ref": temporary_reference}},
         "analysis": {"epistemic_sample_id": "toy-index"}},
        input_identity={
            "snapshot_reference": temporary_reference,
            "dependencies": [{"logical_source_id": DEPENDENCY_LOGICAL_ID, "content_hash": digest,
                              "format": "json", "parser_version": "artificial-parser"}],
            "locations": [{"logical_source_id": DEPENDENCY_LOGICAL_ID,
                           "reference": original.name, "resolved_path": str(original)}],
        },
    )
    loaded = {"reference": temporary_reference, "resolved_path": temporary_reference,
              "content_hash": digest, "format": "json", "status": "loaded"}
    provenance = engine._provenance(snapshot_case, replicate_count=0, fragment_list_provenance=loaded)
    assert provenance["case_hash"] == case_identity_hash(snapshot_case, loaded_dependency=loaded)
    assert provenance["fragment_list"]["reference"] == original.name
    assert provenance["fragment_list"]["resolved_path"] == str(original)
    assert provenance["fragment_list"]["snapshot_used"] is True
    assert provenance["fragment_list"]["content_hash"] == loaded["content_hash"] == file_hash(original)
    assert provenance["fragment_list"]["resolved_path"] != temporary_reference
    assert loaded["reference"] == temporary_reference  # The caller's metadata remains unchanged.


def test_single_observation_cannot_report_convergence(monkeypatch, target_data, registry):
    from damage_engine.engine import DamageEngine
    engine = DamageEngine(target_data, registry)
    monkeypatch.setattr(engine, "_probability_series", lambda _: [np.asarray([.2])])
    monkeypatch.setattr(engine, "_physical_metric_series", lambda *_: [("toy", np.asarray([.2]), False)])
    assert engine._maximum_probability_half_width([{}]) is None
    assert not engine._has_converged([{}], probability_target=1.0, physical_tolerances={})
    report = engine._physical_convergence_report([{}], {"summary.total_component_hits": {"absolute_half_width": 1.0}})
    assert report["target_met"] is False
    assert report["worst_series"]["uncertainty_estimable"] is False
    assert report["worst_series"]["ci95_half_width"] is None
