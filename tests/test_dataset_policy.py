import json

import pytest

from damage_engine.dataset_policy import DatasetUsePolicy, write_dataset_policy
from damage_engine.errors import InputValidationError
from damage_engine.utils import file_hash


def document(classification="software_interpolation_pilot"):
    seal = {"state": "sealed", "frozen_model_hash": None} if classification == "sealed_reference_test" else {"state": "not_applicable", "frozen_model_hash": None}
    return {
        "schema_version": "1.0.0",
        "dataset_id": "dataset-1",
        "dataset_identity_hash": "sha256:" + "1" * 64,
        "classification": classification,
        "proxy_scope": "fixed_target",
        "proxy_boundary": "solver_surrogate",
        "scientific_evidence_status": "not_evaluated",
        "research_admission": {"status": "not_evaluated", "purposes": ["software_development"]},
        "seal": seal,
    }


def test_development_pilot_cannot_be_relabelled_as_training_or_final_test():
    policy = DatasetUsePolicy(document())
    policy.assert_operation("software_development")
    with pytest.raises(InputValidationError, match="does not permit"):
        policy.assert_operation("training")
    with pytest.raises(InputValidationError, match="does not permit"):
        policy.assert_operation("final_evaluation", model_hash="sha256:" + "2" * 64)


def test_reference_test_requires_one_frozen_model_and_cannot_train():
    policy = DatasetUsePolicy(document("sealed_reference_test"))
    with pytest.raises(InputValidationError, match="frozen model"):
        policy.assert_operation("final_evaluation")
    unadmitted = policy.open_reference_test("sha256:" + "2" * 64)
    with pytest.raises(InputValidationError, match="lacks explicit admission"):
        unadmitted.assert_operation(
            "final_evaluation", model_hash="sha256:" + "2" * 64
        )

    value = document("sealed_reference_test")
    value["research_admission"] = {
        "status": "approved",
        "purposes": ["research_test"],
    }
    opened = DatasetUsePolicy(value).open_reference_test("sha256:" + "2" * 64)
    opened.assert_operation("final_evaluation", model_hash="sha256:" + "2" * 64)
    with pytest.raises(InputValidationError, match="different frozen model"):
        opened.assert_operation("final_evaluation", model_hash="sha256:" + "3" * 64)
    with pytest.raises(InputValidationError, match="another frozen model"):
        opened.open_reference_test("sha256:" + "3" * 64)
    with pytest.raises(InputValidationError, match="does not permit"):
        opened.assert_operation("training")


def test_test_roles_cannot_claim_training_admission():
    value = document("rare_stress_test")
    value["research_admission"] = {"status": "approved", "purposes": ["research_training"]}
    with pytest.raises(InputValidationError, match="cannot be admitted"):
        DatasetUsePolicy(value)


@pytest.mark.parametrize(
    ("classification", "operation", "required_purpose"),
    [
        ("reference_train", "training", "research_training"),
        ("reference_train", "normalizer_fit", "research_training"),
        ("adaptive_train_round", "acquisition_fit", "research_training"),
        (
            "reference_validation",
            "validation_model_selection",
            "research_validation",
        ),
        ("reference_validation", "evaluation", "research_validation"),
        ("sealed_reference_test", "final_evaluation", "research_test"),
        ("rare_stress_test", "stress_evaluation", "research_stress_test"),
    ],
)
def test_research_operations_always_require_matching_admission(
    classification, operation, required_purpose,
):
    model_hash = "sha256:" + "8" * 64 if operation == "final_evaluation" else None

    unadmitted = DatasetUsePolicy(document(classification))
    if operation == "final_evaluation":
        unadmitted = unadmitted.open_reference_test(model_hash)
    with pytest.raises(InputValidationError, match="lacks explicit admission"):
        unadmitted.assert_operation(
            operation,
            model_hash=model_hash,
            require_research_admission=False,
        )

    admitted_document = document(classification)
    admitted_document["research_admission"] = {
        "status": "approved",
        "purposes": [required_purpose],
    }
    admitted = DatasetUsePolicy(admitted_document)
    if operation == "final_evaluation":
        admitted = admitted.open_reference_test(model_hash)
    admitted.assert_operation(
        operation,
        model_hash=model_hash,
        require_research_admission=False,
    )


def test_software_development_and_exploration_do_not_require_research_admission():
    policy = DatasetUsePolicy(document())
    policy.assert_operation("software_development")
    policy.assert_operation("exploratory_modeling")


def test_policy_is_canonical_immutable_and_roundtrips(tmp_path):
    value = document()
    policy = DatasetUsePolicy(value)
    value["classification"] = "reference_train"
    first = policy.data
    first["classification"] = "reference_train"
    assert policy.classification == "software_interpolation_pilot"
    output = tmp_path / "policy.json"
    write_dataset_policy(output, policy)
    restored = DatasetUsePolicy.from_file(output)
    assert restored.hash == policy.hash
    assert json.loads(output.read_text(encoding="utf-8"))["classification"] == policy.classification


def test_policy_requires_complete_sha256_identities():
    value = document()
    value["dataset_identity_hash"] = "sha256:1234"
    with pytest.raises(InputValidationError, match="complete sha256"):
        DatasetUsePolicy(value)

    sealed = DatasetUsePolicy(document("sealed_reference_test"))
    with pytest.raises(InputValidationError, match="sha256 identity"):
        sealed.open_reference_test("sha256:1234")


def test_research_operation_uses_manifest_admission_vocabulary_and_exact_binding(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "dataset_id": "dataset-1",
        "label_admission": {
            "scientific_evidence_status": "validated",
            "approved_uses": ["research_training"],
        },
    }), encoding="utf-8")
    value = document("reference_train")
    value["dataset_identity_hash"] = file_hash(manifest_path)
    value["research_admission"] = {
        "status": "approved",
        "purposes": ["research_training"],
    }
    policy = DatasetUsePolicy(value)

    assert policy.assert_archive_binding(manifest_path)["dataset_id"] == "dataset-1"
    policy.assert_operation("training")
    policy.assert_operation("normalizer_fit")
    with pytest.raises(InputValidationError, match="does not permit"):
        policy.assert_operation("evaluation", require_research_admission=True)

    manifest_path.write_text(json.dumps({
        "dataset_id": "dataset-1",
        "label_admission": {
            "scientific_evidence_status": "validated",
            "approved_uses": ["research_training"],
        },
        "tampered": True,
    }), encoding="utf-8")
    with pytest.raises(InputValidationError, match="different manifest content hash"):
        policy.assert_archive_binding(manifest_path)


def test_policy_admission_cannot_exceed_manifest_approved_uses(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "dataset_id": "dataset-1",
        "label_admission": {"approved_uses": ["archive_audit"]},
    }), encoding="utf-8")
    value = document("reference_train")
    value["dataset_identity_hash"] = file_hash(manifest_path)
    value["research_admission"] = {
        "status": "approved",
        "purposes": ["research_training"],
    }
    with pytest.raises(InputValidationError, match="exceeds"):
        DatasetUsePolicy(value).assert_archive_binding(manifest_path)
