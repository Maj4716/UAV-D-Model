from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
from itertools import islice
import json
import math
import os
from pathlib import Path
import time
from typing import Callable, Iterable, Iterator

import numpy as np

from .archive_validation import (
    ArchiveIntegrityError,
    SHARDED_CHECKPOINT_SCHEMA_VERSION,
    sharded_checkpoint_contract_error,
    validate_committed_shard_inventory,
    validate_sharded_checkpoint,
    verify_committed_shard,
    verify_sharded_archive,
)
from .content_contract import validate_json_schema
from .dataset import DatasetWriter, dataset_identity_hash
from .engine import ENGINE_CODE_HASH, DamageEngine
from .errors import InputValidationError
from .estimands import StreamingPositiveWeightMoments
from .input_identity import (
    CapturedCaseInputs,
    CapturedCaseSource,
    FrozenCaseInputs,
    capture_environment_identity,
    case_identity_hash,
    freeze_case_inputs,
)
from .input_identity import single_writer_lock as _single_writer_lock
from .learning_input_identity import learning_input_fingerprint
from .quality import synthetic_label_admission
from .result_ledger import iter_result_ledger, resolve_result_ledger
from .sampling_validation import input_sampling_rows_from_arrays, load_verified_split_registry
from .sampling_validation import validate_candidate_set, validate_registry_assignments
from .utils import content_hash, file_hash


CHECKPOINT_SCHEMA_VERSION = SHARDED_CHECKPOINT_SCHEMA_VERSION


def _replace_with_transient_retry(
    source: Path,
    target: Path,
    *,
    attempts: int = 20,
    delay_seconds: float = 0.05,
) -> None:
    """Keep atomic replace semantics while tolerating brief Windows locks.

    Antivirus and indexer handles can make an otherwise valid same-volume
    replace fail momentarily with ``PermissionError``.  Retry only that error
    for a bounded interval; persistent permission failures still fail closed.
    """
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(delay_seconds)


def _chunks(values: Iterable[dict], size: int) -> Iterator[list[dict]]:
    iterator = iter(values)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _replace_with_transient_retry(temporary, path)


class ShardedDatasetWriter:
    """Bounded-memory dataset generation with deterministic resume.

    Each shard is a complete DatasetWriter product, so existing Parquet/NPZ
    consumers remain usable. The root manifest lists exact committed files;
    ``checkpoint.json`` records the exact ordered case hashes
    used for resume validation.
    """

    def __init__(
        self,
        output_directory: str | Path,
        *,
        shard_size: int,
        split_seed: int = 0,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        tensor_compression: str = "compressed",
        result_compression: str = "auto",
    ):
        if int(shard_size) < 1:
            raise ValueError("shard_size must be at least one")
        self.output_directory = Path(output_directory)
        self.shard_size = int(shard_size)
        self.split_seed = int(split_seed)
        self.split_ratios = tuple(float(value) for value in split_ratios)
        self.tensor_compression = tensor_compression
        self.result_compression = result_compression
        # Validate without touching output data before acquiring the lock.
        if self.split_seed < 0:
            raise ValueError("split_seed must be non-negative")
        if len(self.split_ratios) != 3 or any(value < 0.0 for value in self.split_ratios):
            raise ValueError("split_ratios must contain three non-negative values")
        if not math.isclose(sum(self.split_ratios), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("split_ratios must sum to one")
        if tensor_compression not in {"compressed", "none"}:
            raise ValueError("tensor_compression must be 'compressed' or 'none'")
        if result_compression not in {"auto", "none", "gzip"}:
            raise ValueError("result_compression must be 'auto', 'none', or 'gzip'")

    def _identity(
        self,
        engine: DamageEngine,
        *,
        fail_closed: bool,
        workers: int,
        chunksize: int,
        input_identity: dict | None = None,
    ) -> dict:
        """Checkpoint identity, including execution settings used by every shard.

        This identity is deliberately stricter than the logical dataset id: it
        prevents a resumed run from publishing misleading mixed execution
        provenance, while the final dataset id remains independent of storage
        layout and worker count.
        """
        value = {
            "target_hash": engine.target_hash,
            "physics_registry_hash": engine.registry.hash,
            "engine_code_hash": ENGINE_CODE_HASH,
            "split_seed": self.split_seed,
            "split_ratios": list(self.split_ratios),
            "tensor_compression": self.tensor_compression,
            "result_compression": self.result_compression,
            "shard_size": self.shard_size,
            "failure_policy": "fail_fast" if fail_closed else "isolate_and_record",
            "workers": int(workers),
            "chunksize": int(chunksize),
            "environment": capture_environment_identity(),
            "input_snapshot": input_identity,
        }
        value["identity_hash"] = content_hash(value)
        return value

    def _new_checkpoint(self, identity: dict) -> dict:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "status": "in_progress",
            "identity": identity,
            "completed_shards": [],
        }

    def _validate_checkpoint_structure(self, checkpoint: object) -> dict:
        try:
            return validate_sharded_checkpoint(checkpoint, shard_size=self.shard_size)
        except ArchiveIntegrityError:
            raise
        except (InputValidationError, TypeError, ValueError) as exc:
            raise sharded_checkpoint_contract_error(
                exc,
                path=self.output_directory / "checkpoint.json",
            ) from exc

    @staticmethod
    def _inventory_committed_directories(shards_directory: Path, completed: dict[int, dict]) -> None:
        return validate_committed_shard_inventory(shards_directory, completed)

    def _load_checkpoint(self, identity: dict, *, resume: bool) -> dict:
        checkpoint_path = self.output_directory / "checkpoint.json"
        if resume:
            if not checkpoint_path.exists():
                raise InputValidationError(
                    "resume requested but checkpoint.json does not exist",
                    context={"output_directory": str(self.output_directory)},
                )
            checkpoint = self._validate_checkpoint_structure(
                json.loads(checkpoint_path.read_text(encoding="utf-8"))
            )
            if checkpoint.get("identity") != identity:
                raise InputValidationError(
                    "resume configuration does not match the checkpoint identity",
                    context={
                        "expected_identity_hash": checkpoint.get("identity", {}).get("identity_hash"),
                        "observed_identity_hash": identity["identity_hash"],
                    },
                )
            return checkpoint
        if self.output_directory.exists() and any(self.output_directory.iterdir()):
            raise InputValidationError(
                "sharded output directory must be empty unless --resume is used",
                context={"output_directory": str(self.output_directory)},
            )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = self._new_checkpoint(identity)
        _write_json_atomic(checkpoint_path, checkpoint)
        return checkpoint

    @staticmethod
    def _verify_completed_shard(record: dict, final_directory: Path) -> dict:
        return verify_committed_shard(record, final_directory)

    @staticmethod
    def _candidate_ids(case_chunk: list[dict]) -> list[str] | None:
        records = [(case.get("metadata") or {}).get("sampling") for case in case_chunk]
        if all(record is None for record in records):
            return None
        if any(not isinstance(record, dict) for record in records):
            raise InputValidationError("sampling-aware archives require sampling identity on every input case")
        return [record["candidate_id"] for record in records]

    @staticmethod
    def _validate_input_sampling_identity(cases: Iterable[dict]) -> None:
        """Scan the frozen full input, including future refused executions."""
        def rows():
            mode = None
            registry = None
            registry_snapshot_hash = None
            registry_snapshot_object = None
            input_splits: dict[str, str] = {}
            for case in cases:
                metadata = case.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise InputValidationError("case metadata must be an object")
                record = metadata.get("sampling")
                is_sampling = record is not None
                if mode is not None and mode != is_sampling:
                    raise InputValidationError("cannot mix sampled and unsampled full-input cases across shards")
                mode = is_sampling
                if not is_sampling:
                    continue
                if not isinstance(record, dict):
                    raise InputValidationError("sampling identity must be an object")
                snapshot = record.get("split_registry_document")
                # Compact sampling hydration deliberately shares one verified
                # immutable sidecar object.  Avoid serializing its O(N)
                # assignment array once for every case in this read-only scan.
                snapshot_hash = (
                    registry_snapshot_hash
                    if snapshot is registry_snapshot_object
                    else content_hash(snapshot)
                )
                if registry is None or snapshot_hash != registry_snapshot_hash:
                    observed = load_verified_split_registry(snapshot, expected_hash=record.get("split_registry_hash"),
                                                            expected_id=record.get("split_registry_id"))
                    if registry is not None and observed.definition_hash != registry.definition_hash:
                        raise InputValidationError("sampling archive mixes different actual frozen split registries")
                    registry, registry_snapshot_hash = observed, snapshot_hash
                    registry_snapshot_object = snapshot
                elif (record.get("split_registry_hash") != registry.definition_hash
                      or record.get("split_registry_id") != registry.registry_id):
                    raise InputValidationError("sampling archive declares a registry different from its snapshot")
                validate_registry_assignments(registry, [record])
                split = record.get("split", record.get("frozen_split"))
                fingerprint = learning_input_fingerprint(case)
                if fingerprint in input_splits and input_splits[fingerprint] != split:
                    raise InputValidationError("identical learning inputs cross frozen split boundaries",
                                               context={"learning_input_hash": fingerprint,
                                                        "first_split": input_splits[fingerprint], "split": split})
                input_splits[fingerprint] = split
                yield record
        with closing(rows()) as records:
            validate_candidate_set(records)

    def _validate_completed_resume(
        self,
        cases: Iterable[dict],
        completed: dict[int, dict],
        shards_directory: Path,
        checkpoint: dict,
        *,
        case_hash: Callable[[dict], str] = case_identity_hash,
    ) -> dict:
        seen_case_ids: set[str] = set()
        observed_shards = 0
        for shard_index, case_chunk in enumerate(_chunks(cases, self.shard_size)):
            observed_shards += 1
            if shard_index not in completed:
                raise InputValidationError(
                    "a completed sharded dataset is immutable and cannot be extended by resume",
                    context={"unexpected_shard_index": shard_index},
                )
            case_ids = [str(case.get("case_id") or "") for case in case_chunk]
            if any(not value for value in case_ids) or any(
                value in seen_case_ids for value in case_ids
            ) or len(case_ids) != len(set(case_ids)):
                raise InputValidationError(
                    "dataset case_id values must be globally unique across shards"
                )
            seen_case_ids.update(case_ids)
            case_hashes = [case_hash(case) for case in case_chunk]
            record = completed[shard_index]
            if record.get("candidate_ids") != self._candidate_ids(case_chunk):
                raise InputValidationError("resume candidate identities differ from the committed checkpoint")
            if record.get("case_hashes") != case_hashes:
                raise InputValidationError(
                    "resume input cases do not match the completed shard",
                    context={"shard_index": shard_index},
                )
            self._verify_completed_shard(
                record, shards_directory / f"part-{shard_index:06d}"
            )
        if observed_shards != len(completed):
            raise InputValidationError(
                "resume input ended before all checkpointed shards were matched",
                context={"expected_shards": len(completed), "observed_shards": observed_shards},
            )
        manifest_path = self.output_directory / "manifest.json"
        if not manifest_path.exists():
            raise InputValidationError("completed dataset is missing its root manifest")
        expected_manifest_hash = checkpoint.get("root_manifest_hash")
        if not expected_manifest_hash or file_hash(manifest_path) != expected_manifest_hash:
            raise InputValidationError(
                "completed dataset root manifest is missing its checkpointed hash or has changed",
                context={"path": str(manifest_path)},
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_dataset_id = checkpoint.get("dataset_id")
        if manifest.get("dataset_id") != checkpoint_dataset_id:
            raise InputValidationError(
                "completed dataset root identity does not match its checkpoint",
                context={
                    "manifest_dataset_id": manifest.get("dataset_id"),
                    "checkpoint_dataset_id": checkpoint_dataset_id,
                },
            )
        return manifest

    def run_cases(
        self,
        engine: DamageEngine,
        cases: Iterable[dict],
        *,
        fail_closed: bool = False,
        workers: int = 1,
        chunksize: int | None = None,
        resume: bool = False,
        cases_base_directory: str | Path | None = None,
        expected_case_count: int | None = None,
    ) -> dict:
        # All external inputs and the finite input stream are captured before
        # the first checkpoint or shard can be published.  Only the exact
        # captured-input sources may use their declared transport freezer;
        # ordinary iterables and every legacy/adaptive path retain the generic
        # freezer unchanged.
        compact_source = isinstance(cases, CapturedCaseSource)
        with _single_writer_lock(self.output_directory):
            freezer = (
                cases.freeze_captured_inputs(
                    cases_base_directory=cases_base_directory
                )
                if compact_source
                else freeze_case_inputs(
                    cases, cases_base_directory=cases_base_directory
                )
            )
            with freezer as frozen:
                if compact_source and not isinstance(frozen, CapturedCaseInputs):
                    raise InputValidationError(
                        "captured case source returned an incompatible input view"
                    )
                if len(frozen) == 0:
                    raise ValueError("cases must contain at least one case")
                if expected_case_count is not None and (
                    isinstance(expected_case_count, bool)
                    or not isinstance(expected_case_count, int)
                    or expected_case_count < 1
                    or len(frozen) != expected_case_count
                ):
                    raise InputValidationError("finite input count differs from expected_case_count", context={"expected": expected_case_count, "observed": len(frozen)})
                return self._run_frozen_cases(
                    engine,
                    frozen,
                    fail_closed=fail_closed,
                    workers=workers,
                    chunksize=chunksize,
                    resume=resume,
                    compact_transport=compact_source,
                )

    def _run_frozen_cases(
        self,
        engine: DamageEngine,
        cases: FrozenCaseInputs | CapturedCaseInputs,
        *,
        fail_closed: bool,
        workers: int,
        chunksize: int | None,
        resume: bool,
        compact_transport: bool = False,
    ) -> dict:
        if workers < 1:
            raise ValueError("workers must be at least one")
        if chunksize is not None and chunksize < 1:
            raise ValueError("chunksize must be at least one")
        if compact_transport:
            if not isinstance(cases, CapturedCaseInputs):
                raise InputValidationError(
                    "compact sampling transport requires captured finite inputs"
                )
            hash_case = cases.case_hash
        else:
            hash_case = case_identity_hash
        self._validate_input_sampling_identity(cases)
        effective_chunksize = int(
            chunksize or max(1, self.shard_size // max(1, workers * 4))
        )
        identity = self._identity(
            engine,
            fail_closed=fail_closed,
            workers=workers,
            chunksize=effective_chunksize,
            input_identity=cases.identity,
        )
        checkpoint = self._load_checkpoint(identity, resume=resume)
        checkpoint_path = self.output_directory / "checkpoint.json"
        shards_directory = self.output_directory / "shards"
        shards_directory.mkdir(parents=True, exist_ok=True)
        completed = {
            int(item["index"]): item for item in checkpoint.get("completed_shards", [])
        }
        self._inventory_committed_directories(shards_directory, completed)
        if checkpoint.get("status") == "complete":
            verify_sharded_archive(
                self.output_directory,
                validate_tensors=False,
                allow_split_leakage=True,
                allow_incomplete_marker=True,
            )
            manifest = self._validate_completed_resume(
                cases,
                completed,
                shards_directory,
                checkpoint,
                case_hash=hash_case,
            )
            # A crash can occur after publishing the complete checkpoint but
            # before removing the marker.  Only clear it after the complete
            # dataset and every committed partition have been revalidated.
            (self.output_directory / ".dataset.incomplete").unlink(missing_ok=True)
            return manifest

        incomplete_path = self.output_directory / ".dataset.incomplete"
        incomplete_path.write_text(
            json.dumps({"identity_hash": identity["identity_hash"]}, ensure_ascii=False),
            encoding="utf-8",
        )
        seen_case_ids: set[str] = set()
        observed_case_hashes: list[str] = []
        observed_shard_count = 0

        for shard_index, case_chunk in enumerate(_chunks(cases, self.shard_size)):
            observed_shard_count += 1
            case_ids = [str(case.get("case_id") or "") for case in case_chunk]
            if any(not value for value in case_ids):
                raise InputValidationError(
                    "sharded dataset cases require non-empty case_id values",
                    context={"shard_index": shard_index},
                )
            duplicate_ids = sorted(
                value for value in case_ids if value in seen_case_ids
            )
            if len(case_ids) != len(set(case_ids)):
                duplicate_ids.extend(
                    value for value in case_ids if case_ids.count(value) > 1
                )
            if duplicate_ids:
                raise InputValidationError(
                    "dataset case_id values must be globally unique across shards",
                    context={"duplicates": sorted(set(duplicate_ids))},
                )
            seen_case_ids.update(case_ids)
            case_hashes = [hash_case(case) for case in case_chunk]
            candidate_ids = self._candidate_ids(case_chunk)
            observed_case_hashes.extend(case_hashes)
            shard_name = f"part-{shard_index:06d}"
            final_directory = shards_directory / shard_name

            if shard_index in completed:
                record = completed[shard_index]
                if record.get("candidate_ids") != candidate_ids:
                    raise InputValidationError("resume candidate identities differ from the committed checkpoint")
                if record.get("case_hashes") != case_hashes:
                    raise InputValidationError(
                        "resume input cases do not match the completed shard",
                        context={"shard_index": shard_index},
                    )
                self._verify_completed_shard(record, final_directory)
                continue

            temporary_directory = shards_directory / f".{shard_name}.tmp"
            if temporary_directory.exists() or final_directory.exists():
                raise InputValidationError("uncommitted shard data exists and will not be deleted automatically", context={"shard_index": shard_index})
            writer = DatasetWriter(
                temporary_directory,
                split_seed=self.split_seed,
                split_ratios=self.split_ratios,
                tensor_compression=self.tensor_compression,
                result_compression=self.result_compression,
            )
            if compact_transport:
                # The complete transport and ordered identity were already
                # captured and checked before checkpoint creation.  Re-freezing
                # hydrated cases here would re-expand the full registry into a
                # shard-local TEMP spool.  Pass the materialized FrozenCase
                # list to the private captured-input path instead, with the
                # exact chunk identity a legacy per-shard freeze would create.
                shard_manifest = writer._run_frozen_cases(
                    engine,
                    case_chunk,
                    fail_closed=fail_closed,
                    workers=workers,
                    chunksize=effective_chunksize,
                    input_snapshot=cases.chunk_input_metadata(case_hashes),
                    _captured_inputs=True,
                    _case_hash=hash_case,
                )
            else:
                shard_manifest = writer.run_cases(
                    engine,
                    case_chunk,
                    fail_closed=fail_closed,
                    workers=workers,
                    chunksize=effective_chunksize,
                )
            _replace_with_transient_retry(temporary_directory, final_directory)
            manifest_path = final_directory / "manifest.json"
            record = {
                "index": shard_index,
                "path": f"shards/{shard_name}",
                "case_ids": case_ids,
                "case_hashes": case_hashes,
                "manifest_hash": file_hash(manifest_path),
                "dataset_id": shard_manifest["dataset_id"],
                **({"candidate_ids": candidate_ids} if candidate_ids is not None else {}),
            }
            checkpoint["completed_shards"].append(record)
            completed[shard_index] = record
            _write_json_atomic(checkpoint_path, checkpoint)

        if observed_shard_count == 0:
            raise ValueError("cases must contain at least one case")
        unexpected = sorted(index for index in completed if index >= observed_shard_count)
        if unexpected:
            raise InputValidationError(
                "resume input ended before all checkpointed shards were matched",
                context={"unexpected_shards": unexpected},
            )

        manifests = [
            json.loads(
                (shards_directory / f"part-{index:06d}" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            for index in range(observed_shard_count)
        ]
        self._inventory_committed_directories(shards_directory, completed)
        root_manifest = self._aggregate_manifest(
            manifests,
            identity=identity,
            ordered_case_hashes=observed_case_hashes,
            fail_closed=fail_closed,
            workers=workers,
            chunksize=effective_chunksize,
            input_metadata=cases.metadata,
        )
        validate_json_schema(root_manifest, "dataset_manifest.schema.json")
        root_manifest_path = self.output_directory / "manifest.json"
        _write_json_atomic(root_manifest_path, root_manifest)
        checkpoint["status"] = "complete"
        checkpoint["dataset_id"] = root_manifest["dataset_id"]
        checkpoint["root_manifest_hash"] = file_hash(root_manifest_path)
        _write_json_atomic(checkpoint_path, checkpoint)
        # Reopen the committed root/checkpoint/shard inventory before exposing
        # a complete archive.  Child DatasetWriter instances already performed
        # full tensor-contract reopening before their atomic directory rename.
        verify_sharded_archive(
            self.output_directory,
            validate_tensors=False,
            allow_split_leakage=True,
            allow_incomplete_marker=True,
        )
        incomplete_path.unlink(missing_ok=True)
        return root_manifest

    def _aggregate_manifest(
        self,
        manifests: list[dict],
        *,
        identity: dict,
        ordered_case_hashes: list[str],
        fail_closed: bool,
        workers: int,
        chunksize: int | None,
        input_metadata: dict | None = None,
    ) -> dict:
        first = manifests[0]
        contract = deepcopy(first["tensor_contract"])

        def comparable_contract(value: dict) -> dict:
            compared = deepcopy(value)
            for axis in compared.get("axes", []):
                if axis.get("axis_id") in {"B", "N"}:
                    axis["size"] = None
            return compared

        for index, manifest in enumerate(manifests[1:], start=1):
            if comparable_contract(manifest["tensor_contract"]) != comparable_contract(contract):
                raise InputValidationError(
                    "tensor contracts differ between generated shards",
                    context={"shard_index": index},
                )
            if manifest.get("semantic_vocabulary") != first.get("semantic_vocabulary"):
                raise InputValidationError("semantic vocabularies differ between generated shards", context={"shard_index": index})
            if manifest.get("model_input_contract") != first.get("model_input_contract"):
                raise InputValidationError(
                    "model-input contracts differ between generated shards",
                    context={"shard_index": index},
                )
            if manifest.get("generation", {}).get("result_compression", "none") != first.get("generation", {}).get("result_compression", "none"):
                raise InputValidationError(
                    "raw-result compression differs between generated shards",
                    context={"shard_index": index},
                )
            first_sampling = first.get("generation", {}).get("sampling", {})
            current_sampling = manifest.get("generation", {}).get("sampling", {})
            if first_sampling.get("sampling_envelope_version") == "1.2.0":
                fixed_adaptive_fields = (
                    "sampling_envelope_version", "raw_design_weight_semantics",
                    "design_weight_formula", "proposal_hash", "adaptive_proposal_hash",
                    "acquisition_model_hash", "round_lineage_id",
                    "parent_round_lineage_id", "selected_case_set_hash",
                    "adaptive_factor_fields", "split_registry_document",
                )
                if any(
                    current_sampling.get(field) != first_sampling.get(field)
                    for field in fixed_adaptive_fields
                ):
                    raise InputValidationError(
                        "adaptive sampling identities differ between generated shards",
                        context={"shard_index": index},
                    )

        target_schemas = {
            manifest["source_contracts"]["target_schema"] for manifest in manifests
        }
        case_schemas = {
            manifest["source_contracts"]["case_schema"] for manifest in manifests
        }
        if len(target_schemas) != 1 or len(case_schemas) != 1:
            raise InputValidationError("source schema versions differ between shards")
        evaluation_standards: dict[str, set[str]] = defaultdict(set)
        for manifest in manifests:
            for standard_id, output_ids in manifest["source_contracts"].get(
                "evaluation_standards", {}
            ).items():
                evaluation_standards[str(standard_id)].update(map(str, output_ids))
        source_contracts = {
            "target_schema": next(iter(target_schemas)),
            "case_schema": next(iter(case_schemas)),
            "damage_tree_versions": sorted(
                {
                    str(version)
                    for manifest in manifests
                    for version in manifest["source_contracts"].get(
                        "damage_tree_versions", []
                    )
                }
            ),
            "evaluation_standards": {
                standard_id: sorted(output_ids)
                for standard_id, output_ids in sorted(evaluation_standards.items())
            },
            "parameter_set_hashes": sorted(
                {
                    str(value)
                    for manifest in manifests
                    for value in manifest["source_contracts"].get(
                        "parameter_set_hashes", []
                    )
                }
            ),
            "parameter_set_ids": sorted(
                {
                    str(value)
                    for manifest in manifests
                    for value in manifest["source_contracts"].get(
                        "parameter_set_ids", []
                    )
                }
            ),
            "target_hashes": sorted(
                {
                    str(value)
                    for manifest in manifests
                    for value in manifest["source_contracts"].get(
                        "target_hashes", []
                    )
                }
            ),
        }
        optional_source_fields = (
            "damage_tree_definition_hash", "label_spec_id", "label_spec_hash",
            "sampling_spec_id", "sampling_design_hash", "stratification_spec_id",
            "stratification_spec_hash", "split_registry_id", "split_registry_hash",
            "label_task_semantics", "label_spec_document", "sampling_spec_document",
            "training_design_document", "training_design_hash",
            "fixed_scenario_hash", "fixed_scenario_identity_version",
            "learning_input_identity_version",
        )
        for field in optional_source_fields:
            values = [
                manifest["source_contracts"].get(field) for manifest in manifests
            ]
            if all(value is None for value in values):
                continue
            if any(value is None for value in values) or any(value != values[0] for value in values[1:]):
                raise InputValidationError(
                    "sampling/label source contracts differ between generated shards",
                    context={"field": field},
                )
            source_contracts[field] = deepcopy(values[0])

        partitions = []
        for shard_index, manifest in enumerate(manifests):
            for partition in manifest["partitions"]:
                relative = Path(str(partition["path_glob"]))
                if relative.is_absolute() or ".." in relative.parts or any(character in str(relative) for character in "*?[]"):
                    raise InputValidationError("shard partitions must declare exact relative file paths")
                partitions.append({**partition, "path_glob": f"shards/part-{shard_index:06d}/{relative.as_posix()}"})

        refusal_counts: dict[str, int] = defaultdict(int)
        # Audit globally rather than trusting each shard's local result.  Read
        # the compact split arrays from NPZ so importing the core CLI does not
        # make the optional PyArrow dependency mandatory.
        split_rows: list[dict] = []
        sampling_weight_moments = StreamingPositiveWeightMoments()
        sampling_probability_min: float | None = None
        aggregate_sampling_rows: list[dict] = []
        sampled_registry = None
        if source_contracts.get("split_registry_hash") is not None:
            sampled_registry = load_verified_split_registry(
                first.get("generation", {}).get("sampling", {}).get("split_registry_document"),
                expected_hash=source_contracts["split_registry_hash"], expected_id=source_contracts.get("split_registry_id"),
            )
        for shard_index in range(len(manifests)):
            tensors_path = (
                self.output_directory
                / "shards"
                / f"part-{shard_index:06d}"
                / "tensors.npz"
            )
            with np.load(tensors_path, allow_pickle=False) as arrays:
                if sampled_registry is not None:
                    current_registry = load_verified_split_registry(
                        manifests[shard_index].get("generation", {}).get("sampling", {}).get("split_registry_document"),
                        expected_hash=source_contracts["split_registry_hash"], expected_id=source_contracts.get("split_registry_id"),
                    )
                    input_rows = input_sampling_rows_from_arrays(arrays)
                    validate_registry_assignments(current_registry, input_rows)
                    aggregate_sampling_rows.extend(input_rows)
                split_rows.extend(
                    {
                        "group_id": str(group_id),
                        "leakage_key": str(leakage_key),
                        "split": str(split),
                    }
                    for group_id, leakage_key, split in zip(
                        arrays["split_group_ids"],
                        arrays["leakage_keys"],
                        arrays["split_names"],
                    )
                )
                if "sampling_design_weight_raw" in arrays:
                    weights = np.asarray(arrays["sampling_design_weight_raw"], dtype=np.float64)
                    mask = np.asarray(arrays["sampling_weight_mask"], dtype=bool)
                    probabilities = np.asarray(arrays["sampling_selection_probability"], dtype=np.float64)
                    active = weights[mask]
                    sampling_weight_moments.update(
                        active, field="successful eligible design weights"
                    )
                    if active.size:
                        observed_min = float(np.min(probabilities[mask]))
                        sampling_probability_min = observed_min if sampling_probability_min is None else min(sampling_probability_min, observed_min)
        if sampled_registry is not None:
            validate_candidate_set(aggregate_sampling_rows)
        split_counts = {
            name: sum(str(row.get("split")) == name for row in split_rows)
            for name in ("train", "validation", "test")
        }
        for manifest in manifests:
            for code, count in manifest["quality_summary"]["refusal_counts"].items():
                refusal_counts[code] += int(count)
        total = sum(int(item["quality_summary"]["n_total"]) for item in manifests)
        if sampled_registry is not None and len(aggregate_sampling_rows) != total:
            raise InputValidationError("full-input sampling identities do not match the total committed execution count")
        valid = sum(int(item["quality_summary"]["n_valid"]) for item in manifests)
        refused = sum(int(item["quality_summary"]["n_refused"]) for item in manifests)
        if identity.get("input_snapshot") is not None and total != identity["input_snapshot"]["expected_case_count"]:
            raise InputValidationError("committed row count differs from frozen input count")
        for axis in contract.get("axes", []):
            if axis.get("axis_id") == "B":
                axis["size"] = valid
            elif axis.get("axis_id") == "N":
                axis["size"] = total
        unverified_fraction = (
            sum(
                float(item["quality_summary"]["unverified_parameter_fraction"])
                * int(item["quality_summary"]["n_total"])
                for item in manifests
            )
            / max(1, total)
        )
        fidelity_levels = sorted(
            {
                level
                for manifest in manifests
                for level in manifest["engine"].get("fidelity_levels", [])
            }
        )
        engine = dict(first["engine"])
        engine["fidelity_levels"] = fidelity_levels
        engine["fidelity_level"] = (
            fidelity_levels[0]
            if len(fidelity_levels) == 1
            else ("mixed" if fidelity_levels else "unknown")
        )
        known_limitations = list(
            dict.fromkeys(
                limitation
                for manifest in manifests
                for limitation in manifest.get("known_limitations", [])
                if "materializes cases, results" not in limitation
            )
        )
        known_limitations.append(
            "Static topology arrays are repeated in each shard; TorchShardedDamageDataset opens shards lazily but does not deduplicate those files on disk."
        )
        known_limitations.extend([
            "The writer fully spools finite inputs before execution; an open-ended live stream requires a separate explicit ingestion protocol.",
            "Global case IDs, case hashes and split rows grow linearly with input count; rewriting the cumulative checkpoint after each shard has quadratic cumulative metadata traffic at fixed shard size. No large-scale throughput or RSS guarantee is made.",
            "Atomic same-filesystem replacement and checkpoint file fsync are used, but shard payload fsync and Windows directory durability are not guaranteed; this is not a tested power-loss persistence guarantee.",
            "A process crash can leave an exclusive writer lock and temporary or orphan shard data. Recovery fails closed until an operator explicitly archives those artifacts after checking the owner and committed inventory.",
        ])
        # Preserve the monolithic quality statistic without retaining every
        # interval in RAM.  The temporary memmap is bounded by the number of
        # scalar damage-tree outputs and is removed before publication.
        first_tensors = self.output_directory / "shards" / "part-000000" / "tensors.npz"
        with np.load(first_tensors, allow_pickle=False) as arrays:
            output_count = int(len(arrays["system_output_names"]))
        width_count = valid * output_count
        if width_count:
            width_path = self.output_directory / ".ci-half-widths.tmp"
            widths = np.memmap(width_path, dtype=np.float64, mode="w+", shape=(width_count,))
            position = 0
            observed_outputs = 0
            try:
                for shard_index in range(len(manifests)):
                    shard_directory = (
                        self.output_directory
                        / "shards"
                        / f"part-{shard_index:06d}"
                    )
                    results_path = resolve_result_ledger(
                        shard_directory, manifests[shard_index]
                    )
                    for payload in iter_result_ledger(results_path):
                        if payload.get("status") != "ok":
                            continue
                        for stats in payload.get("damage_tree", {}).get(
                            "outputs", {}
                        ).values():
                            observed_outputs += 1
                            lower, upper = stats.get("ci95_lower"), stats.get("ci95_upper")
                            if stats.get("uncertainty_estimable") is False:
                                continue
                            if lower is None or upper is None or not np.isfinite(float(lower)) or not np.isfinite(float(upper)):
                                raise InputValidationError("an estimable interval is missing or nonfinite in committed shard results")
                            widths[position] = 0.5 * (
                                float(upper) - float(lower)
                            )
                            position += 1
                if observed_outputs != width_count:
                    raise InputValidationError(
                        "damage-tree interval count differs between shard results and tensor contract",
                        context={"expected": width_count, "observed": observed_outputs},
                    )
                widths.flush()
                ci_quantiles = [
                    float(value)
                    for value in np.quantile(
                        widths[:position], [0.5, 0.9, 0.99], overwrite_input=True
                    )
                ] if position else []
            finally:
                del widths
                width_path.unlink(missing_ok=True)
        else:
            ci_quantiles = []
        return {
            "schema_version": first["schema_version"],
            "dataset_id": "damage-dataset-"
            + dataset_identity_hash(
                target_hash=identity["target_hash"],
                physics_registry_hash=identity["physics_registry_hash"],
                engine_code_hash=identity["engine_code_hash"],
                ordered_case_hashes=ordered_case_hashes,
                split_seed=identity["split_seed"],
                split_ratios=identity["split_ratios"],
                tensor_contract_version=contract["version"],
                damage_tree_definition_hash=source_contracts.get("damage_tree_definition_hash"),
                label_spec_hash=source_contracts.get("label_spec_hash"),
                sampling_design_hash=source_contracts.get("sampling_design_hash"),
                stratification_spec_hash=source_contracts.get("stratification_spec_hash"),
                split_registry_hash=source_contracts.get("split_registry_hash"),
            ).split(":", 1)[1][:24],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "engine": engine,
            "source_contracts": source_contracts,
            "semantic_vocabulary": first.get("semantic_vocabulary"),
            **(
                {"model_input_contract": deepcopy(first["model_input_contract"])}
                if first.get("model_input_contract") is not None
                else {}
            ),
            "label_admission": synthetic_label_admission(),
            "capabilities": {
                "storage_layout": "sharded",
                "bounded_tensor_batch": True,
                "checkpoint_resume": True,
                "sharded_writer_available": True,
                "single_target_topology_per_archive": True,
                "semantic_vocabulary_frozen": first.get("semantic_vocabulary") is not None,
                "independent_scientific_validation": False,
            },
            "generation": {
                "workers": int(workers),
                "chunksize": int(chunksize),
                "failure_policy": "fail_fast" if fail_closed else "isolate_and_record",
                "input_order_preserved": True,
                "tensor_compression": self.tensor_compression,
                "result_compression": first.get("generation", {}).get(
                    "result_compression", "none"
                ),
                "environment": identity["environment"],
                "input_snapshot": input_metadata,
                "storage_guarantees": {
                    "single_writer": "exclusive_creation_lock",
                    "checkpoint_file_fsync": True,
                    "power_loss_durability_verified": False,
                    "orphan_recovery": "fail_closed_preserve_unknown_data",
                    "published_files": "exact_committed_partition_paths",
                },
                "resource_policy": {
                    "finite_input_spool": True,
                    "global_metadata_memory": "linear_in_case_count",
                    "cumulative_checkpoint_io": "quadratic_in_shard_count_at_fixed_shard_size",
                    "large_scale_benchmark_verified": False,
                },
                **(
                    {"sampling": first.get("generation", {}).get("sampling")}
                    if first.get("generation", {}).get("sampling") is not None
                    else {}
                ),
            },
            "randomness": first["randomness"],
            "partitions": partitions,
            "tensor_contract": contract,
            "split_policy": {
                **first["split_policy"],
                "leakage_audit_passed": DatasetWriter._check_split_leakage(split_rows),
                "split_counts": split_counts,
                "coverage_guaranteed": False,
            },
            "quality_summary": {
                "n_total": total,
                "n_valid": valid,
                "n_valid_semantics": "execution_success_count",
                "execution_success_count": valid,
                "purpose_admission_counts": {
                    "software_development": valid,
                    "research_training": 0,
                    "research_validation": 0,
                    "research_test": 0,
                },
                "n_refused": refused,
                "refusal_counts": dict(refusal_counts),
                "unverified_parameter_fraction": unverified_fraction,
                "mc_ci_half_width_quantiles": ci_quantiles,
                "coverage_report_id": None,
                **(
                    {
                        "sampling_diagnostics": {
                            "set": "successful_weight_eligible",
                            "weighted_successful_count": sampling_weight_moments.count,
                            "coverage_status": "not_validated_from_successful_execution_subset",
                            "positivity_min": sampling_probability_min,
                            "effective_sample_size": sampling_weight_moments.effective_sample_size,
                            "ess_fraction": (
                                sampling_weight_moments.effective_sample_size
                                / sampling_weight_moments.count
                                if sampling_weight_moments.count
                                else 0.0
                            ),
                            "weight_cv": sampling_weight_moments.coefficient_of_variation,
                            "raw_weight_preserved": True,
                            "candidate_ledger_external": True,
                        }
                    }
                    if source_contracts.get("split_registry_hash") is not None
                    else {}
                ),
            },
            "known_limitations": known_limitations,
            "prohibited_uses": list(first["prohibited_uses"]),
        }

