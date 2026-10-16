"""Executable software-integrity checks for tensor archives.

Passing these checks confirms internal consistency only. It does not establish
scientific validity of any labels or independence of a research split unit.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from .content_contract import is_sha256_identity, validate_json_schema
from .errors import InputValidationError
from .utils import content_hash, file_hash
from .quality import validate_label_admission
from .label_spec import LabelSpec
from .learning_input_identity import (
    MODEL_INPUT_AXIS_NAME_FIELDS,
    MODEL_INPUT_MASK_PAIRS,
    MODEL_INPUT_SEMANTIC_SHAPES,
    model_input_definition_for_version,
)


SHARDED_CHECKPOINT_SCHEMA_VERSION = "2.0.0"


class ArchiveIntegrityError(InputValidationError, ValueError):
    """Stable fail-closed error for an on-disk archive integrity violation.

    ``InputValidationError`` remains the public compatibility category and
    refusal code; ``ValueError`` preserves the historical loader catch
    category for tensor-contract violations. The additional context keys let
    writers, loaders, policy gates and sampling checkpoints report the same
    machine-readable reason.
    """

    def __init__(
        self,
        message: str,
        *,
        integrity_kind: str,
        archive_scope: str,
        path: str | Path | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        details: dict[str, object] = {
            "integrity_kind": str(integrity_kind),
            "archive_scope": str(archive_scope),
        }
        if path is not None:
            details["path"] = str(path)
        if context:
            details.update(dict(context))
        super().__init__(message, context=details)

    def __reduce__(self):
        """Preserve the stable error contract across spawn/pickle boundaries."""

        return (_restore_archive_integrity_error, (self.message, self.context))


def _restore_archive_integrity_error(
    message: str, context: Mapping[str, object],
) -> ArchiveIntegrityError:
    """Rebuild :class:`ArchiveIntegrityError` without losing typed context."""

    details = dict(context)
    integrity_kind = str(details.pop("integrity_kind"))
    archive_scope = str(details.pop("archive_scope"))
    path = details.pop("path", None)
    return ArchiveIntegrityError(
        message,
        integrity_kind=integrity_kind,
        archive_scope=archive_scope,
        path=path,
        context=details,
    )


@dataclass(frozen=True)
class VerifiedArchive:
    """Immutable receipt for one verified archive snapshot.

    JSON documents are retained as strings so callers cannot mutate the
    verification result through a nested dictionary.  Properties return fresh
    objects for consumers that need the manifest or checkpoint contents.
    """

    directory: Path
    storage_layout: str
    manifest_document: str | None
    manifest_hash: str | None
    dataset_id: str | None
    committed_files: tuple[tuple[str, str], ...]
    checkpoint_hash: str | None = None
    audit_overrides: tuple[str, ...] = ()
    checkpoint_document: str | None = None
    shard_manifest_documents: tuple[str, ...] = ()

    @property
    def manifest(self) -> dict | None:
        return (
            None
            if self.manifest_document is None
            else json.loads(self.manifest_document)
        )

    @property
    def checkpoint(self) -> dict | None:
        return (
            None
            if self.checkpoint_document is None
            else json.loads(self.checkpoint_document)
        )

    @property
    def shard_manifests(self) -> tuple[dict, ...]:
        return tuple(json.loads(value) for value in self.shard_manifest_documents)


def _frozen_json(document: Mapping[str, object] | None) -> str | None:
    if document is None:
        return None
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _archive_error(
    message: str,
    *,
    integrity_kind: str,
    archive_scope: str,
    path: str | Path | None = None,
    context: Mapping[str, object] | None = None,
) -> ArchiveIntegrityError:
    return ArchiveIntegrityError(
        message,
        integrity_kind=integrity_kind,
        archive_scope=archive_scope,
        path=path,
        context=context,
    )


def sharded_checkpoint_contract_error(
    error: InputValidationError | TypeError | ValueError,
    *,
    path: str | Path,
) -> ArchiveIntegrityError:
    """Project a checkpoint validator failure onto the shared archive error.

    Keep the original human-readable reason in both the compatibility message
    and machine-readable context. Older callers matched those reason strings,
    while new callers should branch on ``integrity_kind``.
    """

    cause_message = getattr(error, "message", str(error))
    cause_context = getattr(error, "context", None)
    context = dict(cause_context) if isinstance(cause_context, Mapping) else {}
    context["cause_message"] = cause_message
    return _archive_error(
        f"sharded checkpoint does not satisfy its archive contract: {cause_message}",
        integrity_kind="checkpoint_contract_invalid",
        archive_scope="root",
        path=path,
        context=context,
    )


def validate_sharded_checkpoint(
    checkpoint: object, *, shard_size: int,
) -> dict:
    """Validate the complete structural contract of a sharded checkpoint."""

    normalized_shard_size = int(shard_size)
    if normalized_shard_size < 1:
        raise ValueError("shard_size must be at least one")
    if not isinstance(checkpoint, dict):
        raise InputValidationError("checkpoint root must be a JSON object")
    if checkpoint.get("schema_version") != SHARDED_CHECKPOINT_SCHEMA_VERSION:
        raise InputValidationError(
            "checkpoint schema version is not supported",
            context={"schema_version": checkpoint.get("schema_version")},
        )
    if checkpoint.get("status") not in {"in_progress", "complete"}:
        raise InputValidationError(
            "checkpoint status is invalid",
            context={"status": checkpoint.get("status")},
        )
    if not isinstance(checkpoint.get("identity"), dict):
        raise InputValidationError("checkpoint identity must be an object")
    identity = checkpoint["identity"]
    identity_hash = identity.get("identity_hash")
    if (
        not isinstance(identity_hash, str)
        or identity_hash
        != content_hash(
            {key: value for key, value in identity.items() if key != "identity_hash"}
        )
    ):
        raise InputValidationError("checkpoint execution identity hash is invalid")
    records = checkpoint.get("completed_shards")
    if not isinstance(records, list):
        raise InputValidationError("checkpoint completed_shards must be an array")
    indices: list[int] = []
    checkpoint_candidates: set[str] = set()
    required = {
        "index",
        "path",
        "case_ids",
        "case_hashes",
        "manifest_hash",
        "dataset_id",
    }
    for position, record in enumerate(records):
        if not isinstance(record, dict) or not required.issubset(record):
            raise InputValidationError(
                "checkpoint shard record is incomplete",
                context={"record_position": position},
            )
        index = record.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise InputValidationError(
                "checkpoint shard index must be a non-negative integer",
                context={"record_position": position},
            )
        expected_path = f"shards/part-{index:06d}"
        if record.get("path") != expected_path:
            raise InputValidationError(
                "checkpoint shard path is not canonical",
                context={"index": index, "expected_path": expected_path},
            )
        case_ids = record.get("case_ids")
        case_hashes = record.get("case_hashes")
        if (
            not isinstance(case_ids, list)
            or not isinstance(case_hashes, list)
            or not case_ids
            or len(case_ids) != len(case_hashes)
            or len(case_ids) > normalized_shard_size
        ):
            raise InputValidationError(
                "checkpoint shard case lists are invalid",
                context={"index": index},
            )
        if "candidate_ids" in record:
            candidate_ids = record["candidate_ids"]
            if (
                not isinstance(candidate_ids, list)
                or len(candidate_ids) != len(case_ids)
                or any(
                    not isinstance(item, str) or not item
                    for item in candidate_ids
                )
                or len(set(candidate_ids)) != len(candidate_ids)
            ):
                raise InputValidationError(
                    "checkpoint candidate IDs must uniquely match every input case",
                    context={"index": index},
                )
            if checkpoint_candidates.intersection(candidate_ids):
                raise InputValidationError(
                    "checkpoint candidate IDs repeat across committed shards"
                )
            checkpoint_candidates.update(candidate_ids)
        indices.append(index)
    if indices != list(range(len(records))):
        raise InputValidationError(
            "checkpoint shard indices must be unique, ordered and contiguous",
            context={"indices": indices},
        )
    if checkpoint["status"] == "complete" and (
        not checkpoint.get("dataset_id")
        or not checkpoint.get("root_manifest_hash")
    ):
        raise InputValidationError(
            "completed checkpoint lacks root dataset identity or manifest hash"
        )
    return checkpoint


def validate_committed_shard_inventory(
    shards_directory: Path, completed: Mapping[int, dict],
) -> None:
    """Reject missing, orphaned, symlinked, or temporary shard entries."""

    expected = {f"part-{index:06d}" for index in completed}
    entries = list(shards_directory.iterdir())
    observed = {entry.name for entry in entries}
    invalid = [
        entry.name for entry in entries if not entry.is_dir() or entry.is_symlink()
    ]
    if observed != expected or invalid:
        raise _archive_error(
            "shard directory inventory differs from committed checkpoint; orphan or temporary data requires explicit investigation",
            integrity_kind="inventory_mismatch",
            archive_scope="root",
            path=shards_directory,
            context={
                "missing": sorted(expected - observed),
                "uncommitted": sorted(observed - expected),
                "invalid_entries": sorted(invalid),
            },
        )


def verify_committed_shard(record: dict, final_directory: Path) -> dict:
    """Verify a committed shard's manifest, files, and sampling assignments."""

    manifest_path = final_directory / "manifest.json"
    if (
        not manifest_path.exists()
        or file_hash(manifest_path) != record.get("manifest_hash")
    ):
        raise _archive_error(
            "completed shard manifest is missing or has changed",
            integrity_kind="manifest_hash_mismatch",
            archive_scope="shard",
            path=manifest_path,
            context={
                "shard_index": record.get("index"),
            },
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _archive_error(
            "completed shard manifest cannot be decoded",
            integrity_kind="manifest_unreadable",
            archive_scope="shard",
            path=manifest_path,
            context={"shard_index": record.get("index")},
        ) from exc
    if manifest.get("dataset_id") != record.get("dataset_id"):
        raise _archive_error(
            "completed shard dataset identity has changed",
            integrity_kind="dataset_identity_mismatch",
            archive_scope="shard",
            path=manifest_path,
            context={
                "shard_index": record.get("index"),
                "expected_dataset_id": record.get("dataset_id"),
                "observed_dataset_id": manifest.get("dataset_id"),
            },
        )
    for partition in manifest.get("partitions", []):
        pattern = partition.get("path_glob")
        if not isinstance(pattern, str):
            raise _archive_error(
                "completed shard partition metadata is incomplete",
                integrity_kind="partition_contract_invalid",
                archive_scope="shard",
                path=manifest_path,
                context={
                    "shard_index": record.get("index"),
                    "partition": partition.get("name"),
                },
            )
        relative = Path(pattern)
        if relative.is_absolute() or ".." in relative.parts:
            raise _archive_error(
                "completed shard partition path escapes the shard root",
                integrity_kind="partition_contract_invalid",
                archive_scope="shard",
                path=manifest_path,
                context={
                    "shard_index": record.get("index"),
                    "partition": partition.get("name"),
                    "path_glob": pattern,
                },
            )
        paths = sorted(final_directory.glob(pattern))
        if (
            len(paths) != 1
            or not paths[0].is_file()
            or not paths[0].resolve().is_relative_to(final_directory.resolve())
            or file_hash(paths[0]) != partition.get("content_hash")
        ):
            raise _archive_error(
                "completed shard partition is missing or has changed",
                integrity_kind="partition_hash_mismatch",
                archive_scope="shard",
                path=(paths[0] if len(paths) == 1 else final_directory / pattern),
                context={
                    "shard_index": record.get("index"),
                    "partition": partition.get("name"),
                    "path_glob": pattern,
                },
            )
    if (
        manifest.get("source_contracts", {}).get("split_registry_hash") is not None
        and "candidate_ids" in record
    ):
        from .sampling_validation import (
            input_sampling_rows_from_arrays,
            load_verified_split_registry,
            validate_candidate_set,
            validate_registry_assignments,
        )

        with np.load(final_directory / "tensors.npz", allow_pickle=False) as arrays:
            rows = input_sampling_rows_from_arrays(arrays)
        if [row["candidate_id"] for row in rows] != record.get("candidate_ids"):
            raise _archive_error(
                "committed shard candidates differ from the checkpoint's full-input identities",
                integrity_kind="checkpoint_binding_mismatch",
                archive_scope="shard",
                path=final_directory / "tensors.npz",
                context={"shard_index": record.get("index")},
            )
        if [row["case_id"] for row in rows] != record.get("case_ids"):
            raise _archive_error(
                "committed shard full-input case identities differ from its checkpoint",
                integrity_kind="checkpoint_binding_mismatch",
                archive_scope="shard",
                path=final_directory / "tensors.npz",
                context={"shard_index": record.get("index")},
            )
        validate_candidate_set(rows)
        source = manifest["source_contracts"]
        registry = load_verified_split_registry(
            manifest.get("generation", {})
            .get("sampling", {})
            .get("split_registry_document"),
            expected_hash=source["split_registry_hash"],
            expected_id=source.get("split_registry_id"),
        )
        validate_registry_assignments(registry, rows)
    return manifest


SEMANTIC_NAME_FIELDS = (
    "mechanism_names", "state_names", "system_output_names", "evaluation_standard_ids",
)
# Versions declare inherited capabilities explicitly; an opt-in fields subset
# changes returned samples, never archive admission requirements.
CONTRACT_CAPABILITIES = {
    "1.3.0": frozenset(),
    "1.4.0": frozenset({"semantic_vocabulary", "label_uncertainty"}),
    "1.5.0": frozenset({"semantic_vocabulary", "label_uncertainty", "sampling"}),
    "1.6.0": frozenset({"semantic_vocabulary", "label_uncertainty", "sampling", "task_definitions", "source_availability", "registry_assignments"}),
    "1.7.0": frozenset({"semantic_vocabulary", "label_uncertainty", "sampling", "task_definitions", "source_availability", "registry_assignments", "adaptive_sampling"}),
    "1.8.0": frozenset({"semantic_vocabulary", "label_uncertainty", "sampling", "task_definitions", "source_availability", "registry_assignments", "versioned_model_input"}),
}
_AXIS_NAMES = {
    "C": "component_ids", "A": "armor_ids", "M": "mechanism_names",
    "S": "state_names", "O": "system_output_names", "E_standard": "evaluation_standard_ids",
    "F_case": "case_feature_names", "F_component": "component_feature_names",
    "F_armor": "armor_feature_names", "K": "armor_label_names", "D": "frechet_bound_names",
    "F_path": "armor_path_feature_names", "U": "component_aux_label_names",
    "G": "graph_node_ids",
    "F_case_physics": "case_physics_feature_names",
    "F_solver": "solver_context_feature_names",
    **MODEL_INPUT_AXIS_NAME_FIELDS,
    "Q": "sampling_stratum_names",
    "R": "sampling_role_names",
    "T": "label_task_names",
}
_MASK_PAIRS = {
    "case_x": "case_feature_mask", "component_x": "component_feature_mask",
    "case_physics_x": "case_physics_feature_mask",
    **MODEL_INPUT_MASK_PAIRS,
    "solver_context_x": "solver_context_feature_mask",
    "armor_x": "armor_feature_mask", "component_armor_path_x": "component_armor_path_mask",
    "component_y": "component_label_mask", "component_frechet_y": "component_frechet_label_mask",
    "component_aux_y": "component_aux_label_mask", "armor_y": "armor_label_mask",
    "system_y": "system_label_mask",
}
_SEMANTIC_SHAPES = {
    "case_x": ("B", "F_case"), "case_feature_mask": ("B", "F_case"),
    "case_physics_x": ("B", "F_case_physics"),
    "case_physics_feature_mask": ("B", "F_case_physics"),
    "solver_context_x": ("B", "F_solver"),
    "solver_context_feature_mask": ("B", "F_solver"),
    **MODEL_INPUT_SEMANTIC_SHAPES,
    "component_x": ("B", "C", "F_component"), "armor_x": ("B", "A", "F_armor"),
    "component_armor_path_x": ("B", "C", "A", "F_path"),
    "component_y": ("B", "C", "M", "S"), "component_frechet_y": ("B", "C", "S", "D"),
    "component_aux_y": ("B", "C", "U"), "armor_y": ("B", "A", "K"),
    "system_y": ("B", "O"), "requested_standard_mask": ("B", "E_standard"),
    "standard_output_mask": ("E_standard", "O"),
}


def semantic_vocabulary(names: Mapping[str, object]) -> dict:
    if any(not isinstance(names.get(name, ()), (list, tuple)) for name in SEMANTIC_NAME_FIELDS):
        raise ValueError("semantic vocabulary axes must be lists or tuples of names")
    if any(not isinstance(value, str) for name in SEMANTIC_NAME_FIELDS for value in names.get(name, ())):
        raise ValueError("semantic vocabulary names must be strings without coercion")
    payload = {"version": "1.0.0", "names": {
        name: list(names.get(name, ())) for name in SEMANTIC_NAME_FIELDS
    }}
    for name, values in payload["names"].items():
        if len(values) != len(set(values)) or any(not value for value in values):
            raise ValueError(f"semantic vocabulary {name!r} contains duplicate or empty names")
    return {**payload, "content_hash": content_hash(payload)}


def validate_vocabulary(vocabulary: Mapping) -> dict:
    if vocabulary.get("version") != "1.0.0" or not isinstance(vocabulary.get("names"), dict):
        raise ValueError("semantic vocabulary has an unsupported or missing version/names")
    if set(vocabulary["names"]) != set(SEMANTIC_NAME_FIELDS):
        raise ValueError("semantic vocabulary must declare all four semantic axes")
    expected = semantic_vocabulary(vocabulary["names"])
    if vocabulary.get("content_hash") != expected["content_hash"]:
        raise ValueError("semantic vocabulary content_hash does not match its names")
    return expected


def manifest_label_spec(manifest: Mapping, *, required: bool = False) -> LabelSpec | None:
    """Resolve a content-bound task definition without trusting its hash alone."""
    source = manifest.get("source_contracts", {})
    document = source.get("label_spec_document")
    if document is None and not required:
        return None
    if not isinstance(document, dict):
        raise ValueError("the tensor contract requires source_contracts.label_spec_document")
    spec = LabelSpec(document)
    if spec.hash != source.get("label_spec_hash") or spec.spec_id != source.get("label_spec_id"):
        raise ValueError("saved LabelSpec content/identity does not match source contracts")
    if spec.data["source"]["damage_tree_definition_hash"] != source.get("damage_tree_definition_hash"):
        raise ValueError("saved LabelSpec source does not match the damage-tree definition hash")
    return spec


def load_archive_manifest(
    directory: Path, *, verify_integrity: bool = True,
    allow_split_leakage: bool = False, allow_legacy_manifest: bool = False,
    purpose: str = "software_development", evidence_root: str | Path | None = None,
    require_tensors_partition: bool = True,
    verify_partitions: bool = True,
) -> tuple[dict | None, tuple[str, ...]]:
    """Load and check an archive's admission record; explicit overrides are recorded."""
    if purpose != "software_development" and (not verify_integrity or allow_split_leakage or allow_legacy_manifest):
        raise ValueError("research-purpose loading cannot use forensic integrity, split or legacy overrides")
    overrides = []
    if not verify_integrity:
        overrides.append("verify_integrity=False")
    if allow_split_leakage:
        overrides.append("allow_split_leakage=True")
    path = directory / "manifest.json"
    if not path.exists():
        if not allow_legacy_manifest:
            raise FileNotFoundError(f"required dataset manifest is missing: {path}")
        overrides.append("allow_legacy_manifest=True:missing_manifest")
        return None, tuple(overrides)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _archive_error(
            "dataset manifest cannot be decoded",
            integrity_kind="manifest_unreadable",
            archive_scope="root",
            path=path,
        ) from exc
    if not isinstance(manifest, dict):
        raise _archive_error(
            "dataset manifest root must be an object",
            integrity_kind="manifest_schema_invalid",
            archive_scope="root",
            path=path,
        )
    # Versioned archives opt into the complete executable schema. Historical
    # unversioned manifests retain their established semantic-reader contract;
    # they are never emitted by current writers.
    if "schema_version" in manifest:
        try:
            validate_json_schema(
                manifest,
                "dataset_manifest.schema.json",
                failure_message="dataset manifest does not satisfy its schema",
            )
        except InputValidationError as exc:
            if allow_legacy_manifest and purpose == "software_development":
                overrides.append("allow_legacy_manifest=True:schema_validation_failed")
            else:
                raise _archive_error(
                    "dataset manifest does not satisfy its schema",
                    integrity_kind="manifest_schema_invalid",
                    archive_scope="root",
                    path=path,
                    context=exc.context,
                ) from exc
    if "label_admission" not in manifest and allow_legacy_manifest and purpose == "software_development":
        overrides.append("allow_legacy_manifest=True:missing_label_admission")
    else:
        validate_label_admission(manifest, purpose=purpose, evidence_root=evidence_root)
    if manifest.get("split_policy", {}).get("leakage_audit_passed") is not True and not allow_split_leakage:
        raise RuntimeError("dataset failed the split-leakage audit; allow_split_leakage=True is forensic only")
    contract = manifest.get("tensor_contract")
    if not isinstance(contract, dict) or not isinstance(contract.get("fields"), list) or not contract["fields"]:
        if not allow_legacy_manifest:
            raise ValueError("dataset manifest is missing its executable tensor_contract")
        overrides.append("allow_legacy_manifest=True:missing_tensor_contract")
    elif contract.get("version") not in CONTRACT_CAPABILITIES:
        raise ValueError("unsupported tensor_contract version; migrate the archive explicitly")
    if verify_integrity and verify_partitions:
        partitions = manifest.get("partitions")
        if not isinstance(partitions, list) or not partitions:
            raise _archive_error(
                "dataset manifest is missing partition integrity metadata",
                integrity_kind="partition_contract_invalid",
                archive_scope="root",
                path=path,
            )
        tensor_seen = False
        root = directory.resolve()
        for partition in partitions:
            pattern = partition.get("path_glob")
            expected = partition.get("content_hash")
            if not isinstance(pattern, str) or not isinstance(expected, str):
                raise _archive_error(
                    "dataset partition is missing integrity metadata",
                    integrity_kind="partition_contract_invalid",
                    archive_scope="root",
                    path=path,
                    context={"partition": partition.get("name")},
                )
            relative = Path(pattern)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("dataset partition path escapes the dataset root")
            matches = sorted(directory.glob(pattern))
            if len(matches) != 1 or not matches[0].is_file() or not matches[0].resolve().is_relative_to(root):
                raise _archive_error(
                    f"dataset partition is missing or ambiguous: {pattern}",
                    integrity_kind="partition_inventory_mismatch",
                    archive_scope="root",
                    path=directory / pattern,
                    context={
                        "partition": partition.get("name"),
                        "path_glob": pattern,
                    },
                )
            if file_hash(matches[0]) != expected:
                raise _archive_error(
                    f"dataset partition has changed: {pattern}",
                    integrity_kind="partition_hash_mismatch",
                    archive_scope="root",
                    path=matches[0],
                    context={
                        "partition": partition.get("name"),
                        "path_glob": pattern,
                    },
                )
            tensor_seen |= matches[0].resolve() == (directory / "tensors.npz").resolve()
        if require_tensors_partition and not tensor_seen:
            raise _archive_error(
                "dataset manifest does not bind the tensors.npz partition",
                integrity_kind="partition_contract_invalid",
                archive_scope="root",
                path=path,
                context={"required_partition": "tensors.npz"},
            )
    return manifest, tuple(overrides)


def validate_tensor_arrays(arrays: Mapping[str, np.ndarray], manifest: dict | None, *, allow_split_leakage: bool = False) -> None:
    """Check names, axes, dtype, masks, graph/vertex indices and input mappings."""
    if "case_x" not in arrays or np.asarray(arrays["case_x"]).ndim != 2:
        raise ValueError("case_x must be a two-dimensional valid-case tensor")
    batch = len(arrays["case_x"])
    if "case_ids" not in arrays or arrays["case_ids"].shape != (batch,):
        raise ValueError("case_ids must match the valid case axis")
    if len(set(map(str, arrays["case_ids"]))) != batch:
        raise ValueError("case_ids must be unique on the valid case axis")
    if arrays["case_ids"].dtype.kind not in "US" and batch:
        raise ValueError("case_ids must contain strings without coercion")
    if "split_names" in arrays:
        splits = arrays["split_names"]
        if splits.shape != (batch,) or not set(map(str, splits)).issubset({"train", "validation", "test"}):
            raise ValueError("split_names must contain legal splits on the valid case axis")
        for key in ("split_group_ids", "leakage_keys"):
            if key in arrays:
                if arrays[key].shape != (batch,):
                    raise ValueError(f"{key} must match the valid case axis")
                registry = {}
                for identifier, split in zip(map(str, arrays[key]), map(str, splits)):
                    if identifier in registry and registry[identifier] != split:
                        if not allow_split_leakage:
                            raise ValueError(f"{key} crosses split boundaries")
                    registry[identifier] = split
    elif manifest is not None and manifest.get("tensor_contract", {}).get("version") in CONTRACT_CAPABILITIES:
        raise ValueError("split_names is required by the current tensor contract")
    contract_version = manifest.get("tensor_contract", {}).get("version") if manifest else None
    if contract_version is not None and contract_version not in CONTRACT_CAPABILITIES:
        raise ValueError("unsupported tensor_contract version")
    capabilities = CONTRACT_CAPABILITIES.get(contract_version, frozenset())
    if (
        contract_version == "1.8.0"
        and isinstance(manifest, dict)
        and manifest.get("generation", {}).get("sampling", {}).get("sampling_envelope_version")
        == "1.2.0"
    ):
        capabilities = capabilities | {"adaptive_sampling"}
    if "versioned_model_input" in capabilities:
        model_contract = manifest.get("model_input_contract") if manifest else None
        if not isinstance(model_contract, dict):
            raise ValueError("manifest model_input_contract is missing or inconsistent")
        model_version = model_contract.get("version")
        definition = model_input_definition_for_version(model_version)
        if definition is None:
            raise ValueError("manifest model_input_contract has an unsupported version")
        tensor_field = definition.tensor_field
        mask_field = definition.mask_field
        feature_names = definition.feature_names
        required = {
            tensor_field,
            mask_field,
            *definition.metadata_fields,
        }
        missing = sorted(required - set(arrays))
        if missing:
            raise ValueError(
                f"tensor contract {contract_version} is missing versioned model-input fields: {missing}"
            )
        for field, expected in definition.metadata_fields.items():
            if tuple(map(str, arrays[field].tolist())) != tuple(expected):
                raise ValueError(f"{field} does not match the versioned model-input contract")
        state = np.asarray(arrays[tensor_field])
        state_mask = np.asarray(arrays[mask_field])
        if state.dtype != np.dtype("float32") or state.shape != (batch, len(feature_names)):
            raise ValueError(f"{tensor_field} must be float32 with its declared [B,F] shape")
        if state_mask.dtype != np.dtype(bool) or state_mask.shape != state.shape or not np.all(state_mask):
            raise ValueError(f"{mask_field} must be an all-true bool mask matching {tensor_field}")
        if not np.isfinite(state).all():
            raise ValueError(f"{tensor_field} must contain only finite values")
        source_contracts = manifest.get("source_contracts", {}) if manifest else {}
        if (
            model_contract.get("tensor_field") != tensor_field
            or model_contract.get("mask_field") != mask_field
            or [item.get("name") for item in model_contract.get("features", [])]
            != list(feature_names)
            or model_contract.get("fixed_scenario_hash")
            != source_contracts.get("fixed_scenario_hash")
        ):
            raise ValueError("manifest model_input_contract is missing or inconsistent")
    if "source_availability" in capabilities:
        missing = {"system_source_available_mask", "requested_standard_mask", "standard_output_mask", "system_label_mask"} - set(arrays)
        if missing:
            raise ValueError(f"source availability contract is missing required masks: {sorted(missing)}")
    if "sampling" in capabilities:
        required = {
            "case_physics_x", "case_physics_feature_mask", "solver_context_x",
            "solver_context_feature_mask", "sampling_selection_probability",
            "sampling_design_weight_raw", "sampling_weight_mask", "sampling_round",
            "sampling_stratum_index", "sampling_role_index", "sampling_stratum_names",
            "sampling_role_names", "sampling_candidate_ids", "sampling_lineage_ids",
        }
        missing = sorted(required - set(arrays))
        if missing:
            raise ValueError(f"tensor contract {contract_version} is missing sampling fields: {missing}")
        probability = np.asarray(arrays["sampling_selection_probability"])
        weight = np.asarray(arrays["sampling_design_weight_raw"])
        weight_mask = np.asarray(arrays["sampling_weight_mask"])
        if probability.dtype != np.dtype("float64") or probability.shape != (batch,):
            raise ValueError("sampling_selection_probability must be float64 [B]")
        if weight.dtype != np.dtype("float64") or weight.shape != (batch,):
            raise ValueError("sampling_design_weight_raw must be float64 [B]")
        if weight_mask.dtype != np.dtype(bool) or weight_mask.shape != (batch,):
            raise ValueError("sampling_weight_mask must be bool [B]")
        if np.any(~np.isfinite(probability)) or np.any(probability <= 0.0) or np.any(probability > 1.0):
            raise ValueError("sampling selection probabilities must lie in (0, 1]")
        # False means a recorded weight is unavailable for this use; it does
        # not mean its source value is unknown or silently replaced by NaN.
        if np.any(~np.isfinite(weight) | (weight <= 0.0)):
            raise ValueError("all recorded raw design weights must be finite and positive")
        if "adaptive_sampling" not in capabilities and not np.allclose(
            weight, 1.0 / probability, rtol=1.0e-12, atol=1.0e-12
        ):
            raise ValueError("raw design weights must equal inverse inclusion probabilities")
        for name in ("sampling_round", "sampling_stratum_index", "sampling_role_index"):
            value = np.asarray(arrays[name])
            if value.dtype != np.dtype("int64") or value.shape != (batch,) or np.any(value < 0):
                raise ValueError(f"{name} must be nonnegative int64 [B]")
        if np.any(arrays["sampling_stratum_index"] >= len(arrays["sampling_stratum_names"])):
            raise ValueError("sampling_stratum_index exceeds its vocabulary")
        if np.any(arrays["sampling_role_index"] >= len(arrays["sampling_role_names"])):
            raise ValueError("sampling_role_index exceeds its vocabulary")
        role_names = list(map(str, arrays["sampling_role_names"]))
        if "adaptive_train" in role_names and "split_names" in arrays:
            adaptive_index = role_names.index("adaptive_train")
            if np.any(
                (arrays["sampling_role_index"] == adaptive_index)
                & (np.asarray(arrays["split_names"]).astype(str) != "train")
            ):
                raise ValueError("adaptive_train samples must belong to the train split")
        source = manifest.get("source_contracts", {}) if manifest else {}
        for name in (
            "damage_tree_definition_hash", "label_spec_hash", "sampling_design_hash",
            "stratification_spec_hash", "split_registry_hash",
        ):
            value = source.get(name)
            if not is_sha256_identity(value, acceptance="legacy_prefix_only"):
                raise ValueError(f"tensor contract {contract_version} requires source_contracts.{name}")
        for name in ("sampling_candidate_ids", "sampling_lineage_ids"):
            value = np.asarray(arrays[name])
            if value.shape != (batch,) or value.dtype.kind not in "US" or any(not str(item) for item in value):
                raise ValueError(f"{name} must contain nonempty strings on the valid-case axis")
        if len(set(map(str, arrays["sampling_candidate_ids"]))) != batch:
            raise ValueError("sampling_candidate_ids must be unique on the valid-case axis")
    if "adaptive_sampling" in capabilities:
        numeric_names = (
            "sampling_p0_density", "sampling_adaptive_density",
            "sampling_proposal_density", "sampling_p0_hierarchy_probability",
            "sampling_proposal_hierarchy_probability",
        )
        identity_names = (
            "sampling_acquisition_model_hashes", "sampling_proposal_hashes",
            "sampling_round_lineage_ids", "sampling_selected_case_set_hashes",
        )
        input_count = len(np.asarray(arrays.get("input_case_ids", ())))
        required = {
            *numeric_names,
            *identity_names,
            *("input_" + name for name in numeric_names),
            *("input_" + name for name in identity_names),
            "input_sampling_selection_probability",
            "input_sampling_design_weight_raw",
            "input_sampling_weight_mask",
        }
        missing = sorted(required - set(arrays))
        if missing:
            raise ValueError(f"adaptive tensor contract is missing fields: {missing}")
        for name in numeric_names:
            value = np.asarray(arrays[name])
            input_value = np.asarray(arrays["input_" + name])
            if value.dtype != np.dtype("float64") or value.shape != (batch,):
                raise ValueError(f"{name} must be float64 [B]")
            if input_value.dtype != np.dtype("float64") or input_value.shape != (input_count,):
                raise ValueError(f"input_{name} must be float64 [N]")
            if np.any(~np.isfinite(value)) or np.any(value <= 0.0):
                raise ValueError(f"{name} must contain finite positive values")
            if np.any(~np.isfinite(input_value)) or np.any(input_value <= 0.0):
                raise ValueError(f"input_{name} must contain finite positive values")
        for name in (
            "sampling_p0_hierarchy_probability",
            "sampling_proposal_hierarchy_probability",
        ):
            if np.any(arrays[name] > 1.0) or np.any(arrays["input_" + name] > 1.0):
                raise ValueError(f"{name} must lie in (0, 1]")
        for name in identity_names:
            value = np.asarray(arrays[name])
            input_value = np.asarray(arrays["input_" + name])
            if value.shape != (batch,) or value.dtype.kind not in "US" or any(not str(item) for item in value):
                raise ValueError(f"{name} must contain nonempty strings on [B]")
            if input_value.shape != (input_count,) or input_value.dtype.kind not in "US" or any(not str(item) for item in input_value):
                raise ValueError(f"input_{name} must contain nonempty strings on [N]")
        input_probability = np.asarray(arrays["input_sampling_selection_probability"])
        input_weight = np.asarray(arrays["input_sampling_design_weight_raw"])
        input_mask = np.asarray(arrays["input_sampling_weight_mask"])
        if (
            input_probability.dtype != np.dtype("float64")
            or input_probability.shape != (input_count,)
            or input_weight.dtype != np.dtype("float64")
            or input_weight.shape != (input_count,)
            or input_mask.dtype != np.dtype(bool)
            or input_mask.shape != (input_count,)
            or np.any(~np.isfinite(input_probability))
            or np.any(input_probability <= 0.0)
            or np.any(input_probability > 1.0)
            or np.any(~np.isfinite(input_weight))
            or np.any(input_weight <= 0.0)
            or not np.all(input_mask)
        ):
            raise ValueError("adaptive full-input probability/weight fields are invalid")
        expected = (
            arrays["sampling_p0_density"]
            * arrays["sampling_p0_hierarchy_probability"]
            / (
                arrays["sampling_proposal_density"]
                * arrays["sampling_proposal_hierarchy_probability"]
                * arrays["sampling_selection_probability"]
            )
        )
        expected_input = (
            arrays["input_sampling_p0_density"]
            * arrays["input_sampling_p0_hierarchy_probability"]
            / (
                arrays["input_sampling_proposal_density"]
                * arrays["input_sampling_proposal_hierarchy_probability"]
                * input_probability
            )
        )
        if (
            not np.allclose(arrays["sampling_design_weight_raw"], expected, rtol=1.0e-12, atol=1.0e-12)
            or not np.allclose(input_weight, expected_input, rtol=1.0e-12, atol=1.0e-12)
        ):
            raise ValueError("adaptive raw design weights violate the general proposal formula")
        sampling = manifest.get("generation", {}).get("sampling", {}) if manifest else {}
        fixed_fields = {
            "sampling_acquisition_model_hashes": "acquisition_model_hash",
            "sampling_proposal_hashes": "proposal_hash",
            "sampling_round_lineage_ids": "round_lineage_id",
            "sampling_selected_case_set_hashes": "selected_case_set_hash",
        }
        if (
            sampling.get("sampling_envelope_version") != "1.2.0"
            or sampling.get("raw_design_weight_semantics")
            != "general_two_stage_reference_joint_design"
            or sampling.get("design_weight_formula")
            != "p0_density*p0_hierarchy_probability/(proposal_density*proposal_hierarchy_probability*inclusion_probability)"
        ):
            raise ValueError("adaptive manifest sampling semantics are invalid")
        for array_name, manifest_name in fixed_fields.items():
            expected_identity = sampling.get(manifest_name)
            if any(str(value) != expected_identity for value in arrays[array_name]) or any(
                str(value) != expected_identity for value in arrays["input_" + array_name]
            ):
                raise ValueError(f"{array_name} differs from the manifest identity")
    if "registry_assignments" in capabilities:
        from .sampling_validation import (
            input_sampling_rows_from_arrays, load_verified_split_registry,
            validate_candidate_set, validate_registry_assignments,
        )
        sampling = manifest.get("generation", {}).get("sampling", {})
        source = manifest.get("source_contracts", {})
        registry = load_verified_split_registry(
            sampling.get("split_registry_document"),
            expected_hash=source.get("split_registry_hash"), expected_id=source.get("split_registry_id"),
        )
        input_rows = input_sampling_rows_from_arrays(arrays)
        validate_candidate_set(input_rows)
        validate_registry_assignments(registry, input_rows)
        groups = arrays.get("sampling_epistemic_group_ids")
        if groups is None or groups.shape != (batch,) or groups.dtype.kind not in "US" or any(not str(item) for item in groups):
            raise ValueError("sampling_epistemic_group_ids must contain nonempty strings on the valid-case axis")
        valid_rows = [{
            "lineage_id": str(arrays["sampling_lineage_ids"][index]),
            "split": str(arrays["split_names"][index]),
            "role": str(arrays["sampling_role_names"][arrays["sampling_role_index"][index]]),
            "epistemic_group_id": str(groups[index]),
        } for index in range(batch)]
        validate_registry_assignments(registry, valid_rows)
    dimensions = {"B": batch}
    if "input_case_ids" in arrays:
        dimensions["N"] = len(arrays["input_case_ids"])
    for axis, field in _AXIS_NAMES.items():
        if field not in arrays:
            continue
        values = arrays[field]
        if values.ndim != 1:
            raise ValueError(f"{field} must be a one-dimensional name axis")
        if values.size and values.dtype.kind not in "US":
            raise ValueError(f"{field} must contain strings without coercion")
        if field != "graph_node_ids" and (len(set(map(str, values))) != len(values) or any(not str(v) for v in values)):
            raise ValueError(f"{field} contains duplicate or empty names")
        dimensions[axis] = len(values)
    for name, value in arrays.items():
        if name.endswith("_units"):
            names_field = name.removesuffix("_units") + "_names"
            if names_field in arrays and value.shape != arrays[names_field].shape:
                raise ValueError(f"{name} does not match its names axis")
            if value.size and (value.dtype.kind not in "US" or any(not str(unit) for unit in value)):
                raise ValueError(f"{name} must contain nonempty string units")
        if name.endswith("_mask") or name.endswith("mask"):
            if value.dtype != np.dtype(bool):
                raise ValueError(f"{name} must have bool dtype")
    if "tree_edge_index" in arrays:
        edge = arrays["tree_edge_index"]
        if edge.ndim != 2 or edge.shape[0] != 2 or edge.dtype.kind not in "iu":
            raise ValueError("tree_edge_index must be an integer [2,E_edge] array")
        dimensions["E_edge"] = edge.shape[1]
    for field, axis in (("armor_vertices_m", "V"), ("component_vertices_m", "Vc")):
        if field in arrays:
            dimensions[axis] = len(arrays[field])
    for name, axes in _SEMANTIC_SHAPES.items():
        if name in arrays and all(axis in dimensions for axis in axes):
            if arrays[name].shape != tuple(dimensions[axis] for axis in axes):
                raise ValueError(f"{name} does not match its semantic name axes")
    if {"requested_standard_mask", "standard_output_mask", "system_label_mask"}.issubset(arrays):
        expected_mask = np.any(
            arrays["requested_standard_mask"][:, :, None] & arrays["standard_output_mask"][None, :, :],
            axis=1,
        )
        if "source_availability" in capabilities or "system_source_available_mask" in arrays:
            source_available = arrays.get("system_source_available_mask")
            if source_available is None or source_available.dtype != np.dtype(bool) or source_available.shape != expected_mask.shape:
                raise ValueError("system_source_available_mask must be bool [B,O]")
            expected_mask &= source_available
        if not np.array_equal(expected_mask, arrays["system_label_mask"]):
            raise ValueError("system_label_mask does not match the requested standard/output availability")
    if manifest is not None:
        valid_count = manifest.get("quality_summary", {}).get("n_valid")
        if type(valid_count) is not int or valid_count != batch:
            raise ValueError("quality_summary.n_valid does not match the valid case axis")
        fields = manifest.get("tensor_contract", {}).get("fields", [])
        seen = set()
        for specification in fields:
            name = specification.get("name")
            if not isinstance(name, str) or name in seen or name not in arrays:
                raise ValueError(f"tensor_contract field is duplicate, missing or invalid: {name!r}")
            seen.add(name)
            value = arrays[name]
            dtype = specification.get("dtype")
            if dtype == "string":
                matches = value.dtype.kind in "US" or value.size == 0
            else:
                matches = dtype in {"float32", "float64", "int16", "int32", "int64", "bool"} and value.dtype == np.dtype(dtype)
            if not matches:
                raise ValueError(f"{name} dtype does not match tensor_contract: {dtype}")
            shape = specification.get("shape", "")
            if not isinstance(shape, str) or not re.fullmatch(r"\[[A-Za-z0-9_+,]*\]", shape):
                raise ValueError(f"{name} has an invalid shape contract")
            axes = shape[1:-1].split(",") if shape != "[]" else []
            if len(axes) != value.ndim:
                raise ValueError(f"{name} rank does not match tensor_contract")
            for actual, token in zip(value.shape, axes):
                # Explicit migration interpretation of the historical E alias.
                if token == "E":
                    token = "E_edge" if name.startswith("tree_edge_") else "E_standard"
                if token.isdigit():
                    expected = int(token)
                else:
                    axis, _, increment = token.partition("+")
                    if axis not in dimensions:
                        dimensions[axis] = actual - int(increment or 0)
                    expected = dimensions[axis] + int(increment or 0)
                if actual != expected:
                    raise ValueError(f"{name} shape does not match axis {token}")
        if fields and seen != set(arrays):
            raise ValueError("tensor archive contains fields absent from tensor_contract")
        declared_axes = set()
        for axis in manifest.get("tensor_contract", {}).get("axes", []):
            identifier = axis.get("axis_id")
            size = axis.get("size")
            if not isinstance(identifier, str) or identifier in declared_axes or type(size) is not int or size < 0:
                raise ValueError("tensor_contract axes must have unique IDs and nonnegative integer sizes")
            declared_axes.add(identifier)
            if identifier in dimensions and size != dimensions[identifier]:
                raise ValueError(f"axis {identifier} declared size does not match its tensor dimension")
            names = axis.get("names_field")
            units = axis.get("units_field")
            if names is not None and (names not in arrays or len(arrays[names]) != axis.get("size")):
                raise ValueError(f"axis {axis.get('axis_id')} names/size do not match")
            if units is not None and (units not in arrays or len(arrays[units]) != axis.get("size")):
                raise ValueError(f"axis {axis.get('axis_id')} units/size do not match")
        vocabulary = manifest.get("semantic_vocabulary")
        if vocabulary is not None:
            frozen = validate_vocabulary(vocabulary)
            for name, values in frozen["names"].items():
                if tuple(map(str, arrays.get(name, ()))) != tuple(values):
                    raise ValueError(f"semantic vocabulary {name} does not match tensor archive")
        elif "semantic_vocabulary" in capabilities:
            raise ValueError(f"tensor_contract {contract_version} requires a saved semantic vocabulary")
        if "label_uncertainty" in capabilities:
            for name in ("component_y", "component_frechet_y", "component_aux_y", "armor_y", "system_y"):
                if name in arrays:
                    stem = name.removesuffix("_y")
                    if not {stem + "_uncertainty_mask", stem + "_sample_count"}.issubset(arrays):
                        raise ValueError(f"{name} requires explicit uncertainty mask/sample_count")
        if "task_definitions" in capabilities:
            spec = manifest_label_spec(manifest, required=True)
            for name in ("system_output_names", "system_y", "system_label_mask", "label_task_names", "label_task_available_mask"):
                if name not in arrays:
                    raise ValueError(f"saved task definitions require tensor field {name}")
            task_masks = spec.validate_system_outputs(
                arrays["system_output_names"].tolist(), arrays["system_y"], arrays["system_label_mask"],
            )
            task_names = arrays["label_task_names"].tolist()
            actual_masks = arrays["label_task_available_mask"]
            if set(task_names) != set(task_masks) or actual_masks.shape != (batch, len(task_names)) or actual_masks.dtype != np.dtype(bool):
                raise ValueError("label_task_available_mask must match the complete declared task vocabulary")
            expected_masks = np.stack([task_masks[name] for name in task_names], axis=-1)
            if not np.array_equal(actual_masks, expected_masks):
                raise ValueError("label_task_available_mask disagrees with actual task availability")
    counts = arrays.get("replicate_counts")
    if counts is not None and (counts.shape != (batch,) or counts.dtype.kind not in "iu" or np.any(counts < 1)):
        raise ValueError("replicate_counts must be positive integers on the valid case axis")
    for field, mask_name in _MASK_PAIRS.items():
        if field not in arrays or mask_name not in arrays:
            continue
        value, mask = arrays[field], arrays[mask_name]
        if mask.shape != value.shape:
            raise ValueError(f"{mask_name} shape does not match {field}")
        if not np.isfinite(value[mask]).all():
            raise ValueError(f"{field} has nonfinite values selected by its mask")
        if field in {"component_y", "component_frechet_y", "system_y"} and np.any((value[mask] < 0) | (value[mask] > 1)):
            raise ValueError(f"{field} probability labels must lie within [0,1]")
        stem = field.removesuffix("_y")
        uncertainty_mask = arrays.get(stem + "_uncertainty_mask")
        sample_counts = arrays.get(stem + "_sample_count")
        if uncertainty_mask is not None or sample_counts is not None:
            if uncertainty_mask is None or sample_counts is None:
                raise ValueError(f"{field} uncertainty mask/count must appear together")
            if uncertainty_mask.shape != value.shape or sample_counts.shape != value.shape or sample_counts.dtype.kind not in "iu" or np.any(sample_counts < 0):
                raise ValueError(f"{field} uncertainty mask/count shape or dtype is invalid")
            if np.any(uncertainty_mask & ~mask) or np.any(uncertainty_mask & (sample_counts < 2)) or np.any(~mask & (sample_counts != 0)):
                raise ValueError(f"{field} uncertainty mask/count does not match label availability")
            if counts is not None:
                maximum = np.broadcast_to(counts.reshape((batch,) + (1,) * (value.ndim - 1)), value.shape)
                if np.any(sample_counts > maximum) or np.any(mask & (sample_counts < 1)):
                    raise ValueError(f"{field} sample_count exceeds replicates or masks a missing observation")
            if np.any(mask & (sample_counts >= 2) & ~uncertainty_mask):
                raise ValueError(f"{field} estimable uncertainty is silently disabled")
        for suffix in ("_stddev", "_ci95"):
            name = field + suffix
            if name not in arrays:
                continue
            uncertainty = arrays[name]
            expected_shape = value.shape + ((2,) if suffix == "_ci95" else ())
            if uncertainty.shape != expected_shape:
                raise ValueError(f"{name} does not match its label shape")
            selected = np.broadcast_to(mask[..., None], uncertainty.shape) if suffix == "_ci95" else mask
            estimable = uncertainty_mask if uncertainty_mask is not None else (
                np.ones(value.shape, dtype=bool) if counts is None else np.broadcast_to((counts >= 2).reshape((batch,) + (1,) * (value.ndim - 1)), value.shape)
            )
            required = selected & (estimable[..., None] if suffix == "_ci95" else estimable)
            if not np.isfinite(uncertainty[required]).all() or np.isinf(uncertainty[selected]).any():
                raise ValueError(f"{name} has invalid estimable uncertainty")
            if uncertainty_mask is not None:
                unavailable = selected & (~estimable[..., None] if suffix == "_ci95" else ~estimable)
                if not np.isnan(uncertainty[unavailable]).all():
                    raise ValueError(f"{name} must mark unestimable uncertainty with NaN")
            if suffix == "_stddev" and np.any(uncertainty[selected & np.isfinite(uncertainty)] < 0):
                raise ValueError(f"{name} contains a negative standard deviation")
            if suffix == "_ci95":
                both_finite = np.isfinite(uncertainty).all(axis=-1)
                if np.any(mask & both_finite & (uncertainty[..., 0] > uncertainty[..., 1])):
                    raise ValueError(f"{name} has reversed interval bounds")
                if np.any(mask & both_finite & ((value < uncertainty[..., 0]) | (value > uncertainty[..., 1]))):
                    raise ValueError(f"{name} does not contain its mean")
                if np.any(mask & (np.isfinite(uncertainty).any(axis=-1) != both_finite)):
                    raise ValueError(f"{name} has only one finite interval endpoint")
    for prefix, axis in (("armor", "A"), ("component", "C")):
        vertex_name, offset_name = prefix + "_vertices_m", prefix + "_vertex_offsets"
        if vertex_name in arrays or offset_name in arrays:
            if vertex_name not in arrays or offset_name not in arrays:
                raise ValueError(f"{prefix} vertices and offsets must appear together")
            vertices, offsets = arrays[vertex_name], arrays[offset_name]
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                raise ValueError(f"{vertex_name} must contain finite [V,3] vertices")
            if offsets.dtype.kind not in "iu" or offsets.shape != (dimensions.get(axis, 0) + 1,) or offsets[0] != 0 or offsets[-1] != len(vertices) or np.any(np.diff(offsets.astype(np.int64)) < 0):
                raise ValueError(f"{offset_name} must be monotone and end at the vertex count")
    if "tree_edge_index" in arrays:
        edge, types = arrays["tree_edge_index"], arrays.get("tree_edge_type")
        node_ids, node_types = arrays.get("graph_node_ids"), arrays.get("graph_node_types")
        if node_ids is None or node_types is None or node_ids.shape != node_types.shape:
            raise ValueError("graph node IDs and types must share their axis")
        if len(set(zip(map(str, node_types), map(str, node_ids)))) != len(node_ids):
            raise ValueError("graph nodes must be unique within their namespace")
        if not set(map(str, node_types)).issubset({"component", "damage_tree"}):
            raise ValueError("graph contains an unsupported node type")
        if "component_ids" in arrays and not np.array_equal(
            node_ids[node_types.astype(str) == "component"].astype(str), arrays["component_ids"].astype(str),
        ):
            raise ValueError("graph component nodes do not match component_ids")
        if edge.size and (edge.min() < 0 or edge.max() >= len(node_ids)):
            raise ValueError("tree_edge_index endpoint lies outside the graph node axis")
        if types is None or types.shape != (edge.shape[1],) or types.dtype.kind not in "iu" or not set(map(int, types)).issubset({0, 1}):
            raise ValueError("tree_edge_type does not match graph edges")
        for index, kind in enumerate(types):
            source_type, target_type = map(str, node_types[edge[:, index]])
            expected = ("component", "damage_tree") if kind == 0 else ("damage_tree", "damage_tree")
            if (source_type, target_type) != expected:
                raise ValueError("graph edge endpoints do not match the declared edge type")
    mapping_fields = {"input_case_ids", "input_case_status", "input_case_valid_mask", "input_to_valid_index"}
    if mapping_fields & set(arrays):
        if not mapping_fields.issubset(arrays):
            raise ValueError("input-axis mapping fields must appear together")
        length = len(arrays["input_case_ids"])
        if any(arrays[name].shape != (length,) for name in mapping_fields):
            raise ValueError("input-axis mapping fields have inconsistent lengths")
        mask, mapping = arrays["input_case_valid_mask"], arrays["input_to_valid_index"]
        if mapping.dtype.kind not in "iu" or not np.array_equal(mask, arrays["input_case_status"].astype(str) == "ok"):
            raise ValueError("input status/mask/index mapping is inconsistent")
        if np.any(mapping[~mask] != -1) or not np.array_equal(np.sort(mapping[mask]), np.arange(batch)):
            raise ValueError("input_to_valid_index must bijectively map accepted inputs")
        if not np.array_equal(arrays["input_case_ids"][mask].astype(str), arrays["case_ids"][mapping[mask]].astype(str)):
            raise ValueError("input and valid case IDs disagree")
        if "registry_assignments" in capabilities:
            valid_roles = arrays["sampling_role_names"][arrays["sampling_role_index"]]
            for input_field, valid_values in (
                ("input_sampling_candidate_ids", arrays["sampling_candidate_ids"]),
                ("input_sampling_lineage_ids", arrays["sampling_lineage_ids"]),
                ("input_sampling_epistemic_group_ids", arrays["sampling_epistemic_group_ids"]),
                ("input_sampling_roles", valid_roles), ("input_frozen_splits", arrays["split_names"]),
            ):
                if not np.array_equal(arrays[input_field][mask].astype(str), valid_values[mapping[mask]].astype(str)):
                    raise ValueError(f"{input_field} disagrees with the accepted-case sampling identity")


def _load_npz_arrays(
    path: Path,
    *,
    archive_scope: str,
    shard_index: int | None = None,
) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {name: np.asarray(loaded[name]) for name in loaded.files}
    except (OSError, ValueError, KeyError) as exc:
        context = {} if shard_index is None else {"shard_index": shard_index}
        raise _archive_error(
            "tensor partition cannot be reopened",
            integrity_kind="partition_unreadable",
            archive_scope=archive_scope,
            path=path,
            context=context,
        ) from exc


def _validate_archive_tensor_contract(
    arrays: Mapping[str, np.ndarray],
    manifest: dict | None,
    *,
    path: Path,
    archive_scope: str,
    shard_index: int | None = None,
    allow_split_leakage: bool = False,
) -> None:
    try:
        validate_tensor_arrays(
            arrays,
            manifest,
            allow_split_leakage=allow_split_leakage,
        )
    except ArchiveIntegrityError:
        raise
    except (InputValidationError, KeyError, TypeError, ValueError) as exc:
        context: dict[str, object] = {"cause_message": str(exc)}
        if shard_index is not None:
            context["shard_index"] = shard_index
        raise _archive_error(
            f"tensor partition does not satisfy its archive contract: {exc}",
            integrity_kind="tensor_contract_invalid",
            archive_scope=archive_scope,
            path=path,
            context=context,
        ) from exc


def _receipt_files(manifest: Mapping[str, object] | None) -> tuple[tuple[str, str], ...]:
    if manifest is None:
        return ()
    values: list[tuple[str, str]] = []
    for partition in manifest.get("partitions", []):
        if not isinstance(partition, Mapping):
            continue
        path = partition.get("path_glob")
        digest = partition.get("content_hash")
        if isinstance(path, str) and isinstance(digest, str):
            values.append((path, digest))
    return tuple(sorted(values))


def verify_monolithic_archive(
    directory: str | Path,
    *,
    verify_integrity: bool = True,
    allow_split_leakage: bool = False,
    allow_legacy_manifest: bool = False,
    purpose: str = "software_development",
    evidence_root: str | Path | None = None,
    validate_tensors: bool = True,
    allow_incomplete_marker: bool = False,
) -> VerifiedArchive:
    """Reopen one monolithic archive and return its immutable receipt."""

    root = Path(directory)
    marker = root / ".dataset.incomplete"
    if marker.exists() and not allow_incomplete_marker:
        raise _archive_error(
            "dataset generation is incomplete; rerun generation before use",
            integrity_kind="incomplete_publication",
            archive_scope="root",
            path=marker,
        )
    manifest, overrides = load_archive_manifest(
        root,
        verify_integrity=verify_integrity,
        allow_split_leakage=allow_split_leakage,
        allow_legacy_manifest=allow_legacy_manifest,
        purpose=purpose,
        evidence_root=evidence_root,
    )
    tensors_path = root / "tensors.npz"
    if not tensors_path.is_file():
        raise _archive_error(
            "tensor archive is missing",
            integrity_kind="partition_inventory_mismatch",
            archive_scope="root",
            path=tensors_path,
            context={"partition": "tensors"},
        )
    if validate_tensors:
        arrays = _load_npz_arrays(tensors_path, archive_scope="root")
        _validate_archive_tensor_contract(
            arrays,
            manifest,
            path=tensors_path,
            archive_scope="root",
            allow_split_leakage=allow_split_leakage,
        )
    manifest_path = root / "manifest.json"
    manifest_digest = (
        file_hash(manifest_path) if manifest is not None and manifest_path.is_file() else None
    )
    committed = _receipt_files(manifest)
    if manifest is None and tensors_path.is_file():
        committed = (("tensors.npz", file_hash(tensors_path)),)
    return VerifiedArchive(
        directory=root,
        storage_layout="monolithic",
        manifest_document=_frozen_json(manifest),
        manifest_hash=manifest_digest,
        dataset_id=(manifest or {}).get("dataset_id"),
        committed_files=committed,
        audit_overrides=overrides,
    )


def verify_sharded_archive(
    directory: str | Path,
    *,
    verify_integrity: bool = True,
    allow_split_leakage: bool = False,
    allow_legacy_manifest: bool = False,
    purpose: str = "software_development",
    evidence_root: str | Path | None = None,
    validate_tensors: bool = True,
    allow_incomplete_marker: bool = False,
) -> VerifiedArchive:
    """Reopen a complete sharded archive through one shared integrity seam."""

    root = Path(directory)
    marker = root / ".dataset.incomplete"
    if marker.exists() and not allow_incomplete_marker:
        raise _archive_error(
            "dataset generation is incomplete; rerun or resume generation before use",
            integrity_kind="incomplete_publication",
            archive_scope="root",
            path=marker,
        )
    checkpoint_path = root / "checkpoint.json"
    manifest_path = root / "manifest.json"
    for required_path, name in (
        (checkpoint_path, "checkpoint"),
        (manifest_path, "manifest"),
    ):
        if not required_path.is_file():
            raise _archive_error(
                f"sharded dataset {name} is missing",
                integrity_kind="inventory_mismatch",
                archive_scope="root",
                path=required_path,
            )
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _archive_error(
            "sharded checkpoint cannot be decoded",
            integrity_kind="checkpoint_contract_invalid",
            archive_scope="root",
            path=checkpoint_path,
        ) from exc
    shard_size = (
        checkpoint.get("identity", {}).get("shard_size", 0)
        if isinstance(checkpoint, dict)
        else 0
    )
    try:
        validate_sharded_checkpoint(checkpoint, shard_size=shard_size)
    except ArchiveIntegrityError:
        raise
    except (InputValidationError, TypeError, ValueError) as exc:
        raise sharded_checkpoint_contract_error(exc, path=checkpoint_path) from exc
    if checkpoint.get("status") != "complete":
        raise _archive_error(
            "sharded dataset checkpoint is not complete",
            integrity_kind="incomplete_publication",
            archive_scope="root",
            path=checkpoint_path,
            context={"status": checkpoint.get("status")},
        )
    if verify_integrity:
        expected_root_hash = checkpoint.get("root_manifest_hash")
        if (
            not isinstance(expected_root_hash, str)
            or file_hash(manifest_path) != expected_root_hash
        ):
            raise _archive_error(
                "root manifest is missing its checkpointed hash or has changed",
                integrity_kind="manifest_hash_mismatch",
                archive_scope="root",
                path=manifest_path,
            )
    manifest, overrides = load_archive_manifest(
        root,
        verify_integrity=verify_integrity,
        allow_split_leakage=allow_split_leakage,
        allow_legacy_manifest=allow_legacy_manifest,
        purpose=purpose,
        evidence_root=evidence_root,
        require_tensors_partition=False,
        verify_partitions=False,
    )
    if manifest is None:
        raise _archive_error(
            "a sharded archive requires a root manifest",
            integrity_kind="manifest_unreadable",
            archive_scope="root",
            path=manifest_path,
        )
    if checkpoint.get("dataset_id") != manifest.get("dataset_id"):
        raise _archive_error(
            "checkpoint and root manifest dataset identities do not match",
            integrity_kind="dataset_identity_mismatch",
            archive_scope="root",
            path=manifest_path,
            context={
                "expected_dataset_id": checkpoint.get("dataset_id"),
                "observed_dataset_id": manifest.get("dataset_id"),
            },
        )
    records = checkpoint["completed_shards"]
    completed = {int(record["index"]): record for record in records}
    shards_directory = root / "shards"
    try:
        validate_committed_shard_inventory(shards_directory, completed)
    except FileNotFoundError as exc:
        raise _archive_error(
            "committed shard directory is missing",
            integrity_kind="inventory_mismatch",
            archive_scope="root",
            path=shards_directory,
        ) from exc

    actual_partitions: dict[str, dict] = {}
    child_documents: list[str] = []
    for record in records:
        shard_index = int(record["index"])
        shard = root / record["path"]
        shard_marker = shard / ".dataset.incomplete"
        if shard_marker.exists():
            raise _archive_error(
                "committed shard generation is incomplete",
                integrity_kind="incomplete_publication",
                archive_scope="shard",
                path=shard_marker,
                context={"shard_index": shard_index},
            )
        if verify_integrity:
            verify_committed_shard(record, shard)
        try:
            child, _ = load_archive_manifest(
                shard,
                verify_integrity=verify_integrity,
                allow_split_leakage=allow_split_leakage,
                allow_legacy_manifest=False,
                purpose=purpose,
                evidence_root=evidence_root,
            )
        except ArchiveIntegrityError as exc:
            details = dict(exc.context)
            details["archive_scope"] = "shard"
            details["shard_index"] = shard_index
            raise ArchiveIntegrityError(
                exc.message,
                integrity_kind=str(details.pop("integrity_kind")),
                archive_scope=str(details.pop("archive_scope")),
                path=details.pop("path", shard),
                context=details,
            ) from exc
        assert child is not None
        if child.get("dataset_id") != record.get("dataset_id"):
            raise _archive_error(
                "checkpoint and shard manifest dataset identities do not match",
                integrity_kind="dataset_identity_mismatch",
                archive_scope="shard",
                path=shard / "manifest.json",
                context={
                    "shard_index": shard_index,
                    "expected_dataset_id": record.get("dataset_id"),
                    "observed_dataset_id": child.get("dataset_id"),
                },
            )
        child_documents.append(_frozen_json(child) or "{}")
        for partition in child.get("partitions", []):
            relative = (Path(record["path"]) / str(partition["path_glob"])).as_posix()
            if relative in actual_partitions:
                raise _archive_error(
                    "committed shard partitions contain duplicate paths",
                    integrity_kind="partition_contract_invalid",
                    archive_scope="root",
                    path=root / relative,
                    context={"shard_index": shard_index},
                )
            actual_partitions[relative] = partition
        if validate_tensors:
            tensors_path = shard / "tensors.npz"
            arrays = _load_npz_arrays(
                tensors_path,
                archive_scope="shard",
                shard_index=shard_index,
            )
            _validate_archive_tensor_contract(
                arrays,
                child,
                path=tensors_path,
                archive_scope="shard",
                shard_index=shard_index,
                allow_split_leakage=allow_split_leakage,
            )

    declared_partitions: dict[str, dict] = {}
    for partition in manifest.get("partitions", []):
        path = partition.get("path_glob")
        if (
            not isinstance(path, str)
            or not path
            or any(character in path for character in "*?[")
            or path in declared_partitions
        ):
            raise _archive_error(
                "root partitions must enumerate unique exact committed file paths",
                integrity_kind="partition_contract_invalid",
                archive_scope="root",
                path=manifest_path,
            )
        declared_partitions[path] = partition
    if set(declared_partitions) != set(actual_partitions):
        raise _archive_error(
            "root partition file set does not match committed shard manifests",
            integrity_kind="partition_inventory_mismatch",
            archive_scope="root",
            path=manifest_path,
            context={
                "missing": sorted(set(actual_partitions) - set(declared_partitions)),
                "uncommitted": sorted(set(declared_partitions) - set(actual_partitions)),
            },
        )
    for path, actual in actual_partitions.items():
        declared = declared_partitions[path]
        if (
            declared.get("content_hash") != actual.get("content_hash")
            or declared.get("row_count") != actual.get("row_count")
        ):
            raise _archive_error(
                "root partition metadata differs from its committed shard manifest",
                integrity_kind="partition_binding_mismatch",
                archive_scope="root",
                path=root / path,
                context={"partition": declared.get("name")},
            )
    return VerifiedArchive(
        directory=root,
        storage_layout="sharded",
        manifest_document=_frozen_json(manifest),
        manifest_hash=file_hash(manifest_path),
        dataset_id=manifest.get("dataset_id"),
        committed_files=_receipt_files(manifest),
        checkpoint_hash=file_hash(checkpoint_path),
        audit_overrides=overrides,
        checkpoint_document=_frozen_json(checkpoint),
        shard_manifest_documents=tuple(child_documents),
    )


def verify_archive_receipt(receipt: VerifiedArchive) -> VerifiedArchive:
    """Fail closed if bytes or inventory changed after a verification receipt."""

    if not isinstance(receipt, VerifiedArchive):
        raise TypeError("receipt must be a VerifiedArchive")
    root = receipt.directory.resolve()
    marker = root / ".dataset.incomplete"
    if marker.exists():
        raise _archive_error(
            "dataset changed to an incomplete publication after verification",
            integrity_kind="incomplete_publication",
            archive_scope="root",
            path=marker,
        )
    manifest_path = root / "manifest.json"
    if (
        receipt.manifest_hash is None
        or not manifest_path.is_file()
        or file_hash(manifest_path) != receipt.manifest_hash
    ):
        raise _archive_error(
            "dataset manifest changed after archive verification",
            integrity_kind="manifest_hash_mismatch",
            archive_scope="root",
            path=manifest_path,
        )
    if receipt.storage_layout == "sharded":
        checkpoint_path = root / "checkpoint.json"
        if (
            receipt.checkpoint_hash is None
            or not checkpoint_path.is_file()
            or file_hash(checkpoint_path) != receipt.checkpoint_hash
        ):
            raise _archive_error(
                "dataset checkpoint changed after archive verification",
                integrity_kind="checkpoint_hash_mismatch",
                archive_scope="root",
                path=checkpoint_path,
            )
        checkpoint = receipt.checkpoint
        if checkpoint is None:
            raise _archive_error(
                "verified sharded receipt lacks its checkpoint snapshot",
                integrity_kind="checkpoint_contract_invalid",
                archive_scope="root",
                path=checkpoint_path,
            )
        completed = {
            int(record["index"]): record
            for record in checkpoint["completed_shards"]
        }
        try:
            validate_committed_shard_inventory(root / "shards", completed)
        except FileNotFoundError as exc:
            raise _archive_error(
                "committed shard directory changed after archive verification",
                integrity_kind="inventory_mismatch",
                archive_scope="root",
                path=root / "shards",
            ) from exc
        for shard_index, record in completed.items():
            shard = root / record["path"]
            shard_marker = shard / ".dataset.incomplete"
            if shard_marker.exists():
                raise _archive_error(
                    "committed shard changed to an incomplete publication after verification",
                    integrity_kind="incomplete_publication",
                    archive_scope="shard",
                    path=shard_marker,
                    context={"shard_index": shard_index},
                )
            shard_manifest = shard / "manifest.json"
            if (
                not shard_manifest.is_file()
                or file_hash(shard_manifest) != record.get("manifest_hash")
            ):
                raise _archive_error(
                    "committed shard manifest changed after archive verification",
                    integrity_kind="manifest_hash_mismatch",
                    archive_scope="shard",
                    path=shard_manifest,
                    context={"shard_index": shard_index},
                )
    elif receipt.storage_layout != "monolithic":
        raise ValueError("receipt storage_layout is not supported")

    for relative_text, expected_hash in receipt.committed_files:
        relative = Path(relative_text)
        path = (root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_relative_to(root)
            or not path.is_file()
            or file_hash(path) != expected_hash
        ):
            parts = relative.parts
            shard_index = None
            if len(parts) >= 2 and parts[0] == "shards" and re.fullmatch(r"part-\d{6}", parts[1]):
                shard_index = int(parts[1].removeprefix("part-"))
            context: dict[str, object] = {"partition_path": relative.as_posix()}
            if shard_index is not None:
                context["shard_index"] = shard_index
            raise _archive_error(
                "dataset partition changed after archive verification",
                integrity_kind="partition_hash_mismatch",
                archive_scope="shard" if shard_index is not None else "root",
                path=path,
                context=context,
            )
    return receipt
