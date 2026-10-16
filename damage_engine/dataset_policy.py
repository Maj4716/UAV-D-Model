"""Dataset-role and sealed-evaluation policy independent of physical models.

The policy prevents software/development, reference, adaptive and stress data
from being silently reused for a different experimental role.  It is an access
contract, not evidence that labels or a sampling distribution are scientifically
valid.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .content_contract import is_sha256_identity
from .errors import InputValidationError
from .utils import content_hash, file_hash


DATASET_POLICY_VERSION = "1.0.0"

_CLASS_OPERATIONS = {
    "software_interpolation_pilot": {
        "archive_audit", "software_development", "exploratory_modeling"
    },
    "reference_train": {
        "archive_audit", "training", "normalizer_fit", "acquisition_fit"
    },
    "adaptive_train_round": {
        "archive_audit", "training", "normalizer_fit", "acquisition_fit"
    },
    "reference_validation": {
        "archive_audit", "validation_model_selection", "evaluation"
    },
    "sealed_reference_test": {"archive_audit", "final_evaluation"},
    "rare_stress_test": {"archive_audit", "stress_evaluation"},
}

_TRAIN_CLASSES = {"reference_train", "adaptive_train_round"}
_TEST_CLASSES = {"sealed_reference_test", "rare_stress_test"}
_RESEARCH_PURPOSE_BY_OPERATION = {
    "archive_audit": "archive_audit",
    "software_development": "software_development",
    "exploratory_modeling": "exploratory_modeling",
    "training": "research_training",
    "normalizer_fit": "research_training",
    "acquisition_fit": "research_training",
    "validation_model_selection": "research_validation",
    "evaluation": "research_validation",
    "final_evaluation": "research_test",
    "stress_evaluation": "research_stress_test",
}
_RESEARCH_PURPOSES = frozenset(_RESEARCH_PURPOSE_BY_OPERATION.values())
_RESEARCH_OPERATIONS = frozenset(
    operation
    for operation, purpose in _RESEARCH_PURPOSE_BY_OPERATION.items()
    if purpose.startswith("research_")
)


def _canonical_copy(document: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(document, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise InputValidationError("dataset policy must contain canonical finite JSON") from exc


class DatasetUsePolicy:
    """Immutable role policy with explicit sealed-test access transitions."""

    def __init__(self, document: dict[str, Any]):
        if not isinstance(document, dict):
            raise InputValidationError("dataset policy must be a JSON object")
        data = _canonical_copy(document)
        required = {
            "schema_version", "dataset_id", "dataset_identity_hash", "classification",
            "proxy_scope", "proxy_boundary", "scientific_evidence_status",
            "research_admission", "seal",
        }
        missing = sorted(required - set(data))
        if missing:
            raise InputValidationError("dataset policy is missing required fields", context={"missing": missing})
        if data["schema_version"] != DATASET_POLICY_VERSION:
            raise InputValidationError("unsupported dataset policy version")
        classification = data["classification"]
        if classification not in _CLASS_OPERATIONS:
            raise InputValidationError("dataset policy classification is invalid")
        if not isinstance(data["dataset_id"], str) or not data["dataset_id"]:
            raise InputValidationError("dataset policy dataset_id must be non-empty")
        if not is_sha256_identity(data["dataset_identity_hash"]):
            raise InputValidationError(
                "dataset policy dataset_identity_hash must be a complete sha256 identity"
            )
        if data["proxy_scope"] not in {"fixed_target", "general_target"}:
            raise InputValidationError("proxy_scope must be fixed_target or general_target")
        if data["proxy_boundary"] not in {"solver_surrogate", "end_to_end"}:
            raise InputValidationError("proxy_boundary must be solver_surrogate or end_to_end")
        if data["scientific_evidence_status"] not in {
            "not_evaluated", "unverified", "validated"
        }:
            raise InputValidationError("scientific_evidence_status is invalid")
        admission = data["research_admission"]
        if not isinstance(admission, dict) or set(admission) != {"status", "purposes"}:
            raise InputValidationError("research_admission requires status and purposes")
        if admission["status"] not in {"not_evaluated", "refused", "approved"}:
            raise InputValidationError("research admission status is invalid")
        purposes = admission["purposes"]
        if (not isinstance(purposes, list) or len(purposes) != len(set(purposes))
                or any(not isinstance(value, str) or not value for value in purposes)):
            raise InputValidationError("research admission purposes must be unique strings")
        unknown_purposes = sorted(set(purposes) - _RESEARCH_PURPOSES)
        if unknown_purposes:
            raise InputValidationError(
                "research admission contains unsupported purposes",
                context={"purposes": unknown_purposes},
            )
        seal = data["seal"]
        if not isinstance(seal, dict) or set(seal) != {"state", "frozen_model_hash"}:
            raise InputValidationError("seal requires state and frozen_model_hash")
        if seal["state"] not in {"not_applicable", "sealed", "opened_for_frozen_model"}:
            raise InputValidationError("dataset policy seal state is invalid")
        if classification == "sealed_reference_test":
            if seal["state"] == "not_applicable":
                raise InputValidationError("reference test policy must be sealed or model-bound")
            if seal["state"] == "opened_for_frozen_model" and not is_sha256_identity(
                seal["frozen_model_hash"]
            ):
                raise InputValidationError("opened reference test requires a frozen model hash")
        elif seal != {"state": "not_applicable", "frozen_model_hash": None}:
            raise InputValidationError("only reference test policies may carry a seal")
        if classification in _TRAIN_CLASSES and admission["status"] != "approved":
            if "research_training" in purposes:
                raise InputValidationError("unapproved data cannot list research_training admission")
        if classification in _TEST_CLASSES and any(
            value in purposes for value in ("training", "research_training", "acquisition_fit")
        ):
            raise InputValidationError("test data cannot be admitted for training or acquisition")
        self._json = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_file(cls, path: str | Path) -> "DatasetUsePolicy":
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InputValidationError("unable to load dataset policy") from exc
        return cls(document)

    @property
    def data(self) -> dict[str, Any]:
        return json.loads(self._json)

    @property
    def hash(self) -> str:
        return content_hash(self.data)

    @property
    def classification(self) -> str:
        return str(self.data["classification"])

    def assert_operation(
        self,
        operation: str,
        *,
        model_hash: str | None = None,
        require_research_admission: bool = False,
    ) -> None:
        """Refuse role-incompatible or scientifically unadmitted operations.

        Research training, model selection and evaluation always require the
        matching ``research_*`` admission.  The legacy boolean remains useful
        for callers that deliberately require an approved admission even for
        an otherwise non-research operation; setting it to ``False`` cannot
        bypass a research-operation gate.
        """

        if operation not in _CLASS_OPERATIONS[self.classification]:
            raise InputValidationError(
                "dataset role does not permit the requested operation",
                context={"classification": self.classification, "operation": operation},
            )
        data = self.data
        if operation == "final_evaluation":
            seal = data["seal"]
            if seal["state"] != "opened_for_frozen_model" or not model_hash:
                raise InputValidationError("sealed reference test requires an explicit frozen model binding")
            if seal["frozen_model_hash"] != model_hash:
                raise InputValidationError("reference test is bound to a different frozen model")
        if operation in _RESEARCH_OPERATIONS or require_research_admission:
            admission = data["research_admission"]
            required_purpose = _RESEARCH_PURPOSE_BY_OPERATION[operation]
            if (
                admission["status"] != "approved"
                or required_purpose not in admission["purposes"]
            ):
                raise InputValidationError("dataset lacks explicit admission for this research operation")

    def assert_archive_binding(self, manifest_path: str | Path) -> dict[str, Any]:
        """Compatibility-only binding to manifest bytes and admission metadata.

        This legacy API does not verify partition or shard bytes.  New archive
        consumers must first obtain a ``VerifiedArchive`` and call
        :meth:`assert_verified_archive_binding` instead.
        """

        path = Path(manifest_path)
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InputValidationError("unable to load the dataset manifest for policy binding") from exc
        if not isinstance(manifest, dict):
            raise InputValidationError("dataset manifest for policy binding must be an object")
        return self._assert_manifest_binding(manifest, file_hash(path))

    def _assert_manifest_binding(
        self, manifest: dict[str, Any], manifest_hash: str,
    ) -> dict[str, Any]:
        data = self.data
        if manifest.get("dataset_id") != data["dataset_id"]:
            raise InputValidationError("dataset policy is bound to a different dataset_id")
        if manifest_hash != data["dataset_identity_hash"]:
            raise InputValidationError("dataset policy is bound to a different manifest content hash")
        admission = data["research_admission"]
        manifest_admission = manifest.get("label_admission")
        if admission["status"] == "approved":
            if not isinstance(manifest_admission, dict):
                raise InputValidationError("approved policy requires a manifest label_admission record")
            approved_uses = manifest_admission.get("approved_uses")
            if (
                not isinstance(approved_uses, list)
                or not set(admission["purposes"]).issubset(set(approved_uses))
            ):
                raise InputValidationError(
                    "policy research admission exceeds the manifest approved uses"
                )
        return manifest

    def assert_verified_archive_binding(self, receipt) -> dict[str, Any]:
        """Bind policy to a fully verified monolithic or sharded receipt."""

        from .archive_validation import VerifiedArchive, verify_archive_receipt

        if not isinstance(receipt, VerifiedArchive):
            raise TypeError("receipt must be a VerifiedArchive")
        manifest = receipt.manifest
        if manifest is None or receipt.manifest_hash is None:
            raise InputValidationError(
                "dataset policy cannot bind a legacy archive without a manifest"
            )
        verify_archive_receipt(receipt)
        return self._assert_manifest_binding(manifest, receipt.manifest_hash)

    def open_reference_test(self, model_hash: str) -> "DatasetUsePolicy":
        if self.classification != "sealed_reference_test":
            raise InputValidationError("only a sealed reference test can be opened")
        if not is_sha256_identity(model_hash):
            raise InputValidationError("frozen model hash must be a sha256 identity")
        data = deepcopy(self.data)
        seal = data["seal"]
        if seal["state"] == "opened_for_frozen_model" and seal["frozen_model_hash"] != model_hash:
            raise InputValidationError("reference test is already bound to another frozen model")
        seal.update(state="opened_for_frozen_model", frozen_model_hash=model_hash)
        return DatasetUsePolicy(data)


def write_dataset_policy(path: str | Path, policy: DatasetUsePolicy) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(policy.data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
