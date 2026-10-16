"""Contract-only integration checks for SamplingSpec 1.1 and manifest 1.5.

The fixtures are artificial and this module never constructs or runs
``DamageEngine``.  It verifies the identity boundary between CaseFactory,
``dataset._sampling_context`` and the published manifest schemas.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from damage_engine.case_factory import CaseFactory
from damage_engine.dataset import _sampling_context
from damage_engine.errors import InputValidationError
from damage_engine.label_spec import LabelSpec
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.utils import content_hash


ROOT = Path(__file__).resolve().parents[1]
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64


def _label_spec() -> LabelSpec:
    return LabelSpec(
        {
            "schema_version": "1.0.0",
            "label_spec_id": "manifest-integration-toy-labels",
            "revision": "1",
            "status": "provisional",
            "source": {
                "damage_tree_id": "toy-tree",
                "damage_tree_revision": "1",
                "damage_tree_definition_hash": SHA_B,
            },
            "tasks": {
                "toy": {
                    "type": "multi_label_event_probability",
                    "outputs": ["toy-event"],
                    "loss": "binary_cross_entropy",
                }
            },
            "provenance": {
                "approval_status": "not_approved",
                "evidence_ref": None,
            },
        }
    )


def _sampling_document_11() -> dict:
    variables = [
        {
            "name": "temperature",
            "path": "/environment/temperature_k",
            "distribution": "uniform",
            "low": 286.0,
            "high": 290.0,
        },
        {
            "name": "pressure",
            "path": "/environment/ambient_pressure_pa",
            "distribution": "uniform",
            "low": 100000.0,
            "high": 101000.0,
        },
    ]
    return {
        "schema_version": "1.1.0",
        "sampling_spec_id": "manifest-integration-toy-p0",
        "revision": "1",
        "status": "provisional",
        "reference_distribution": {
            "distribution_id": "manifest-integration-reference",
            "interpretation": "design_distribution",
            "variables": variables,
            "constraints": [],
        },
        "budgets": {
            "train_pilot": 1,
            "validation": 1,
            "test": 1,
            "rare_stress_test": 0,
        },
        "seeds": {
            "design": 801,
            "selection": 802,
            "split": 803,
            "screening": 804,
            "formal": 805,
        },
        "formal_policy": {
            "replicates": 2,
            "adaptive_ci_half_width": None,
            "epistemic_policy": "unique_per_lineage",
        },
        "split_policy": {
            "registry_id": "manifest-integration-toy-splits",
            "freeze_before_outcomes": True,
            "adaptive_train_only": True,
        },
        "quality_gates": {
            "positivity_min": 0.1,
            "min_ess_fraction": 0.5,
            "max_weight_cv": 2.0,
        },
        "feature_roles": [
            {
                "feature_id": item["name"],
                "path": item["path"],
                "tensor_feature_names": [item["path"].rsplit("/", 1)[-1]],
                "role": "learnable_variable",
                "availability_stage": "raw_input",
                "deployment_available": True,
            }
            for item in variables
        ],
        "variable_blocks": [],
        "geometry_strata": [],
        "stream_roles": [
            {
                "role": "reference_train",
                "stream": "reference",
                "split": "train",
                "train_eligible": True,
            },
            {
                "role": "reference_validation",
                "stream": "reference",
                "split": "validation",
                "train_eligible": False,
            },
            {
                "role": "reference_test",
                "stream": "reference",
                "split": "test",
                "train_eligible": False,
            },
        ],
        "round_budgets": [
            {
                "round": 0,
                "allocations": [
                    {
                        "role": "reference_validation",
                        "stratum_id": "p0",
                        "proposal_id": "manifest-integration-reference",
                        "count": 1,
                    },
                    {
                        "role": "reference_test",
                        "stratum_id": "p0",
                        "proposal_id": "manifest-integration-reference",
                        "count": 1,
                    },
                    {
                        "role": "reference_train",
                        "stratum_id": "p0",
                        "proposal_id": "manifest-integration-reference",
                        "count": 1,
                    },
                ],
            }
        ],
        "replicate_policy": {
            "default_replicates": 2,
            "outcome_adaptive": False,
            "overrides": [],
        },
        "key_projections": [
            {
                "projection_id": "environment-joint-audit",
                "paths": [
                    "/environment/temperature_k",
                    "/environment/ambient_pressure_pa",
                ],
                "purpose": "coverage_audit",
                "pre_label_only": True,
            }
        ],
        "proxy_scope": {
            "scope": "fixed_target",
            "geometry_boundary": "solver_proxy",
            "target_identity_policy": "fixed_target_hash",
            "deployment_features_match_training": True,
            "evidence_status": "unverified",
        },
        "legal_sampling": {
            "invalid_candidate_policy": "fail_closed",
            "max_attempts_per_allocation": 3,
        },
    }


def _sampling_document_10() -> dict:
    document = _sampling_document_11()
    document["schema_version"] = "1.0.0"
    for field in (
        "feature_roles",
        "variable_blocks",
        "geometry_strata",
        "stream_roles",
        "round_budgets",
        "replicate_policy",
        "key_projections",
        "proxy_scope",
        "legal_sampling",
    ):
        document.pop(field)
    return document


def _plan(case_data: dict, document: dict):
    specification = SamplingSpec(document)
    plan = CaseFactory(
        deepcopy(case_data),
        specification,
        _label_spec(),
        target_hash=SHA_A,
        damage_tree_definition_hash=SHA_B,
        physics_registry_hash=SHA_C,
        engine_code_hash=SHA_D,
    ).build_reference_plan()
    return specification, plan


def test_sampling_context_accepts_casefactory_11_and_binds_full_design(case_data) -> None:
    specification, plan = _plan(case_data, _sampling_document_11())

    records = _sampling_context(plan.cases)

    assert records is not None and len(records) == 3
    assert {record["schema_version"] for record in records} == {"1.1.0"}
    assert all(record["sampling_spec_document"] == specification.data for record in records)
    assert all(
        record["training_design_document"] == specification.training_design_document
        and record["training_design_hash"] == specification.training_design_hash
        for record in records
    )


@pytest.mark.parametrize(
    "tamper",
    [
        lambda record: record["sampling_spec_document"].update(revision="tampered"),
        lambda record: record.update(sampling_spec_hash=SHA_A),
        lambda record: record["training_design_document"]["proxy_scope"].update(
            evidence_status="provisional"
        ),
        lambda record: record.update(training_design_hash=SHA_A),
    ],
    ids=[
        "sampling-document",
        "sampling-hash",
        "training-document",
        "training-hash",
    ],
)
def test_sampling_context_rejects_tampered_11_documents_and_hashes(
    case_data, tamper
) -> None:
    _, plan = _plan(case_data, _sampling_document_11())
    cases = deepcopy(plan.cases)
    for case in cases:
        tamper(case["metadata"]["sampling"])

    with pytest.raises(InputValidationError):
        _sampling_context(cases)


def test_sampling_context_preserves_10_casefactory_compatibility(case_data) -> None:
    _, plan = _plan(case_data, _sampling_document_10())

    records = _sampling_context(plan.cases)

    assert records is not None and len(records) == 3
    assert {record["schema_version"] for record in records} == {"1.0.0"}
    assert all("sampling_spec_document" not in record for record in records)
    assert all("training_design_document" not in record for record in records)


def _manifest_15() -> dict:
    sampling = SamplingSpec(_sampling_document_11())
    label = _label_spec()
    return {
        "schema_version": "1.5.0",
        "dataset_id": "artificial-manifest-contract",
        "created_at": "2026-10-11T00:00:00+00:00",
        "engine": {
            "version": "test-only",
            "code_hash": SHA_D,
            "configuration_hash": SHA_C,
            "fidelity_level": "unknown",
        },
        "source_contracts": {
            "target_schema": "target.schema.json@1.0.0",
            "case_schema": "simulation_case.schema.json@1.0.0",
            "evaluation_standards": {"toy-standard": ["toy-event"]},
            "damage_tree_definition_hash": SHA_B,
            "label_spec_id": label.spec_id,
            "label_spec_hash": label.hash,
            "label_spec_document": label.data,
            "sampling_spec_id": sampling.spec_id,
            "sampling_design_hash": sampling.hash,
            "sampling_spec_document": sampling.data,
            "training_design_document": sampling.training_design_document,
            "training_design_hash": sampling.training_design_hash,
            "stratification_spec_id": "none",
            "stratification_spec_hash": content_hash({}),
            "split_registry_id": "artificial-splits",
            "split_registry_hash": SHA_A,
            "label_task_semantics": "test-only",
        },
        "semantic_vocabulary": {
            "version": "1.0.0",
            "names": {
                "mechanism_names": [],
                "state_names": [],
                "system_output_names": ["toy-event"],
                "evaluation_standard_ids": ["toy-standard"],
            },
            "content_hash": SHA_A,
        },
        "label_admission": {
            "label_source_type": "synthetic_teacher",
            "input_integrity_status": "verified",
            "scientific_evidence_status": "not_evaluated",
            "approved_uses": ["software_development"],
            "reason": "artificial contract fixture",
        },
        "capabilities": {
            "storage_layout": "monolithic",
            "bounded_tensor_batch": False,
            "checkpoint_resume": False,
            "sharded_writer_available": True,
            "single_target_topology_per_archive": True,
            "semantic_vocabulary_frozen": True,
            "independent_scientific_validation": False,
        },
        "generation": {
            "workers": 1,
            "chunksize": 1,
            "failure_policy": "precomputed_results",
            "input_order_preserved": True,
            "tensor_compression": "compressed",
            "sampling": {
                "sampling_envelope_version": "1.1.0",
                "rounds": [0],
                "streams": ["reference"],
                "roles": [
                    "reference_train",
                    "reference_validation",
                    "reference_test",
                ],
                "strata": ["p0"],
                "selection_probability_field": "sampling_selection_probability",
                "raw_design_weight_field": "sampling_design_weight_raw",
                "raw_design_weight_semantics": "inverse_inclusion_probability_for_weight_mask_true_reference_cases",
                "split_registry_document": {},
            },
        },
        "randomness": {
            "split_seed": 0,
            "aleatory_seed_field": "analysis.aleatory_seed",
            "epistemic_sample_id_field": "analysis.epistemic_sample_id",
            "parallel_result_order_deterministic": True,
        },
        "partitions": [
            {
                "name": "audits",
                "format": "jsonl",
                "path_glob": "artificial-results.jsonl",
                "row_count": 0,
                "content_hash": SHA_A,
            }
        ],
        "tensor_contract": {
            "version": "1.6.0",
            "fields": [
                {
                    "name": "case_ids",
                    "dtype": "string",
                    "shape": "[B]",
                    "semantic_role": "metadata",
                    "unit": None,
                    "normalization_id": None,
                }
            ],
        },
        "split_policy": {
            "method": "grouped",
            "group_keys": ["split_group"],
            "seed": 0,
            "leakage_audit_passed": True,
            "split_registry_status": "frozen_pre_outcome",
            "split_registry_hash": SHA_A,
            "independence_guaranteed": False,
            "coverage_guaranteed": False,
        },
        "quality_summary": {
            "n_total": 0,
            "n_valid": 0,
            "n_refused": 0,
            "refusal_counts": {},
            "unverified_parameter_fraction": 1.0,
            "execution_success_count": 0,
            "n_valid_semantics": "execution_success_count",
            "purpose_admission_counts": {
                "software_development": 0,
                "research_training": 0,
                "research_validation": 0,
                "research_test": 0,
            },
            "sampling_diagnostics": {
                "positivity_min": None,
                "effective_sample_size": 0.0,
                "ess_fraction": 0.0,
                "weight_cv": None,
                "raw_weight_preserved": True,
                "candidate_ledger_external": True,
            },
        },
        "prohibited_uses": ["test-only artifact"],
    }


def _manifest_validator() -> Draft202012Validator:
    schema = json.loads(
        (ROOT / "damage_engine" / "schemas" / "dataset_manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )
    return Draft202012Validator(schema)


def _errors(document: dict) -> list:
    return list(_manifest_validator().iter_errors(document))


@pytest.mark.parametrize(
    "schema_name",
    ["dataset_manifest.schema.json", "sampling_spec.schema.json"],
)
def test_packaged_and_published_contract_schemas_are_byte_identical(
    schema_name: str,
) -> None:
    packaged = ROOT / "damage_engine" / "schemas" / schema_name
    published = ROOT / "schemas" / schema_name

    assert packaged.read_bytes() == published.read_bytes()


def test_manifest_15_artificial_fixture_satisfies_contract() -> None:
    assert _errors(_manifest_15()) == []


@pytest.mark.parametrize(
    "field",
    ["semantic_vocabulary", "label_admission", "capabilities"],
)
def test_manifest_15_requires_common_post_12_contract_sections(field: str) -> None:
    manifest = _manifest_15()
    manifest.pop(field)

    assert _errors(manifest), f"manifest 1.5 unexpectedly accepted without {field}"


@pytest.mark.parametrize(
    "field",
    [
        "label_spec_document",
        "sampling_spec_document",
        "training_design_document",
        "training_design_hash",
    ],
)
def test_manifest_15_requires_embedded_identity_documents(field: str) -> None:
    manifest = _manifest_15()
    manifest["source_contracts"].pop(field)

    assert _errors(manifest), f"manifest 1.5 unexpectedly accepted without {field}"


@pytest.mark.parametrize(
    "field",
    [
        "split_registry_document",
        "sampling_envelope_version",
        "raw_design_weight_semantics",
    ],
)
def test_manifest_15_requires_extended_sampling_metadata(field: str) -> None:
    manifest = _manifest_15()
    manifest["generation"]["sampling"].pop(field)

    assert _errors(manifest), f"manifest 1.5 unexpectedly accepted without {field}"


def test_manifest_15_requires_sampling_envelope_11() -> None:
    manifest = _manifest_15()
    manifest["generation"]["sampling"]["sampling_envelope_version"] = "1.0.0"

    assert _errors(manifest), "manifest 1.5 unexpectedly accepted a 1.0 sampling envelope"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("tensor_contract", "version"), "1.5.0"),
        (("split_policy", "coverage_guaranteed"), True),
        (("split_policy", "independence_guaranteed"), True),
    ],
)
def test_manifest_15_freezes_tensor_and_evidence_boundaries(path, value) -> None:
    manifest = _manifest_15()
    manifest[path[0]][path[1]] = value

    assert _errors(manifest), f"manifest 1.5 unexpectedly accepted {path}={value!r}"
