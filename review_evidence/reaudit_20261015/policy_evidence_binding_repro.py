"""Abstract metadata-policy/evidence consistency checks; no training or simulator."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).absolute().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from damage_engine.dataset_policy import DatasetUsePolicy
from damage_engine.errors import InputValidationError
from damage_engine.quality import validate_label_admission
from damage_engine.torch_dataset import TorchDamageDataset
from damage_engine.utils import file_hash
from test_loader_stream_separation import _write_archive


def main():
    temp = Path(r"C:\Users\Administrator\.codex\visualizations\2026\10\09\01a11ff4-76f9-7262-9bf8-10f9bed66e97") / ("policy_toy_20261015_" + uuid.uuid4().hex[:8])
    manifest = _write_archive(temp, prefix="abstract-evidence", roles=("reference_train",))
    manifest["label_admission"]["approved_uses"].append("research_training")
    path = temp / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    policy = DatasetUsePolicy({
        "schema_version": "1.0.0", "dataset_id": manifest["dataset_id"],
        "dataset_identity_hash": file_hash(path), "classification": "reference_train",
        "proxy_scope": "fixed_target", "proxy_boundary": "end_to_end",
        "scientific_evidence_status": "unverified",
        "research_admission": {"status": "approved", "purposes": ["research_training"]},
        "seal": {"state": "not_applicable", "frozen_model_hash": None},
    })
    output = {"scope": "abstract metadata and array API only; no simulator or training",
              "python": sys.executable,
              "manifest_scientific_status": manifest["label_admission"]["scientific_evidence_status"],
              "manifest_has_independent_evidence": manifest["label_admission"].get("evidence_record") is not None,
              "toy_is_minimal_loader_archive_not_complete_manifest_schema": True,
              "toy_directory": str(temp)}
    policy.assert_archive_binding(path)
    policy.assert_operation("training")
    output["policy_operations_accept_declarations_without_evidence"] = True
    try:
        validate_label_admission(manifest, "research_training")
    except InputValidationError as exc:
        output["direct_research_evidence_gate_rejects"] = str(exc)
    for purpose in ("software_development", "research_training"):
        try:
            dataset = TorchDamageDataset(temp, fields=("case_x",), policy=policy,
                                         operation="training", purpose=purpose)
            output["loader_" + purpose] = {"accepted": True, "length": len(dataset),
                                            "effective_purpose": dataset.purpose,
                                            "operation": dataset.operation,
                                            "audit_overrides": list(dataset.audit_overrides)}
        except Exception as exc:
            output["loader_" + purpose] = {"accepted": False, "error_type": type(exc).__name__,
                                            "error": str(exc)}
    Path(__file__).with_suffix(".json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
