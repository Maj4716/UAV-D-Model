import json

from damage_engine.cli import main
from damage_engine.dataset_policy import DatasetUsePolicy, write_dataset_policy
from damage_engine.utils import file_hash


def write_manifest(path, dataset_id, *, approved_uses=None):
    path.write_text(json.dumps({
        "dataset_id": dataset_id,
        "label_admission": {
            "scientific_evidence_status": "not_evaluated",
            "approved_uses": approved_uses or ["software_development"],
        },
    }), encoding="utf-8")
    return path


def test_cli_validates_development_role_and_refuses_training(tmp_path, capsys):
    manifest = write_manifest(tmp_path / "manifest.json", "development")
    policy = DatasetUsePolicy({
        "schema_version": "1.0.0",
        "dataset_id": "development",
        "dataset_identity_hash": file_hash(manifest),
        "classification": "software_interpolation_pilot",
        "proxy_scope": "fixed_target",
        "proxy_boundary": "solver_surrogate",
        "scientific_evidence_status": "not_evaluated",
        "research_admission": {"status": "not_evaluated", "purposes": ["software_development"]},
        "seal": {"state": "not_applicable", "frozen_model_hash": None},
    })
    path = tmp_path / "policy.json"
    write_dataset_policy(path, policy)
    assert main(["dataset-policy", "validate", "--policy", str(path), "--manifest", str(manifest), "--operation", "software_development"]) == 0
    assert json.loads(capsys.readouterr().out)["classification"] == "software_interpolation_pilot"
    assert main(["dataset-policy", "validate", "--policy", str(path), "--manifest", str(manifest), "--operation", "training"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "refused"


def test_cli_opens_reference_test_for_one_model(tmp_path, capsys):
    manifest = write_manifest(
        tmp_path / "test-manifest.json",
        "test",
        approved_uses=["software_development", "research_test"],
    )
    policy = DatasetUsePolicy({
        "schema_version": "1.0.0",
        "dataset_id": "test",
        "dataset_identity_hash": file_hash(manifest),
        "classification": "sealed_reference_test",
        "proxy_scope": "fixed_target",
        "proxy_boundary": "solver_surrogate",
        "scientific_evidence_status": "not_evaluated",
        "research_admission": {"status": "approved", "purposes": ["research_test"]},
        "seal": {"state": "sealed", "frozen_model_hash": None},
    })
    source, output = tmp_path / "sealed.json", tmp_path / "opened.json"
    write_dataset_policy(source, policy)
    model = "sha256:" + "2" * 64
    assert main(["dataset-policy", "open-test", "--policy", str(source), "--frozen-model-hash", model, "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["frozen_model_hash"] == model
    opened = DatasetUsePolicy.from_file(output)
    opened.assert_operation("final_evaluation", model_hash=model)


def test_cli_research_operation_cannot_omit_admission_check(tmp_path, capsys):
    manifest = write_manifest(tmp_path / "train-manifest.json", "train")
    policy = DatasetUsePolicy({
        "schema_version": "1.0.0",
        "dataset_id": "train",
        "dataset_identity_hash": file_hash(manifest),
        "classification": "reference_train",
        "proxy_scope": "fixed_target",
        "proxy_boundary": "end_to_end",
        "scientific_evidence_status": "not_evaluated",
        "research_admission": {
            "status": "not_evaluated",
            "purposes": ["software_development"],
        },
        "seal": {"state": "not_applicable", "frozen_model_hash": None},
    })
    path = tmp_path / "train-policy.json"
    write_dataset_policy(path, policy)

    assert main([
        "dataset-policy", "validate", "--policy", str(path),
        "--manifest", str(manifest), "--operation", "training",
    ]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert "lacks explicit admission" in payload["error"]["message"]


def test_cli_operation_requires_exact_manifest_binding(tmp_path, capsys):
    policy = DatasetUsePolicy({
        "schema_version": "1.0.0",
        "dataset_id": "development",
        "dataset_identity_hash": "sha256:" + "1" * 64,
        "classification": "software_interpolation_pilot",
        "proxy_scope": "fixed_target",
        "proxy_boundary": "end_to_end",
        "scientific_evidence_status": "not_evaluated",
        "research_admission": {
            "status": "not_evaluated",
            "purposes": ["software_development"],
        },
        "seal": {"state": "not_applicable", "frozen_model_hash": None},
    })
    path = tmp_path / "policy.json"
    write_dataset_policy(path, policy)

    assert main([
        "dataset-policy", "validate", "--policy", str(path),
        "--operation", "software_development",
    ]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert "requires --manifest" in payload["error"]["message"]
