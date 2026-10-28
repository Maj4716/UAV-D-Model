"""Content-bound sampling plans and archive completion; no scientific admission."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import numpy as np

from .case_factory import CaseFactory, ReferencePlan
from .content_contract import is_sha256_identity
from .engine import ENGINE_CODE_HASH, DamageEngine
from .errors import InputValidationError
from .input_identity import (
    CapturedCaseInputs,
    CapturedCaseSource,
    FrozenCase,
    _canonical_subtree_cache,
    capture_environment_identity,
    case_identity_hash,
    freeze_case_inputs,
    input_identity_from_case_hashes,
    single_writer_lock,
)
from .label_spec import LabelSpec
from .learning_input_identity import learning_input_fingerprint
from .sampling_audit import audit_sampling
from .sampling_spec import SamplingSpec
from .split_registry import SplitRegistry
from .result_ledger import load_result_ledger
from .utils import content_hash, file_hash
from .validation import validate_inputs

if TYPE_CHECKING:
    from .archive_validation import VerifiedArchive

SAMPLING_CHECKPOINT_VERSION = "2.1.0"
SUPPORTED_SAMPLING_CHECKPOINT_VERSIONS = frozenset({"2.0.0", "2.1.0"})
MULTIROUND_CHECKPOINT_VERSION = "multiround-sampling-checkpoint@1.0.0"
MULTIROUND_CHECKPOINT_FILE = "multiround_sampling_checkpoint.json"
_ACTIVE_TRANSACTIONS = ContextVar("sampling_transactions", default=frozenset())
_PLAN_FILES_COMMON = {
    "plan/sampling_spec.json", "plan/label_spec.json", "plan/base_case.json",
    "plan/input_snapshot.json", "plan/formal_input_identity.json",
    "plan/split_registry.json", "plan/frozen_hashes.json",
    "rounds/round-000/candidates.parquet",
    "rounds/round-000/round_manifest.json",
    "audits/sampling_quality.json",
}
_REFERENCE_FORMAL_CASES_LEDGER = "rounds/round-000/formal_cases.jsonl.gz"
_LEGACY_REFERENCE_CANDIDATES_LEDGER = "rounds/round-000/candidates.jsonl"
_LEGACY_REFERENCE_FORMAL_CASES_LEDGER = "rounds/round-000/formal_cases.jsonl"
_PLAN_FILES = _PLAN_FILES_COMMON | {
    _REFERENCE_FORMAL_CASES_LEDGER,
}

# The formal-case JSONL ledger is a transport artifact, not the canonical home of the
# split assignment registry.  Older plans embedded the complete N-assignment
# registry in every one of the N cases, which made the ledger grow O(N^2).
# New plans keep exactly one immutable snapshot and place this relative
# reference in each transport row.  Readers hydrate the original in-memory
# case before identity checks or engine execution, so scientific/audit
# identities remain unchanged.
_SPLIT_REGISTRY_REFERENCE_FIELD = "split_registry_reference"
_CONTRACT_DOCUMENTS_REFERENCE_FIELD = "contract_document_references"
_REFERENCE_SPLIT_REGISTRY_PATH = "../../plan/split_registry.json"
_ADAPTIVE_SPLIT_REGISTRY_PATH = "adaptive_split_registry.json"
_LABEL_SPEC_REFERENCE = "../../plan/label_spec.json"
_SAMPLING_SPEC_REFERENCE = "../../plan/sampling_spec.json"
_TRAINING_DESIGN_REFERENCE = "derived:sampling_spec.training_design_document"


@contextmanager
def sampling_run_transaction(output_directory: str | Path):
    """Hold the sampling lock across initialization, execution, and completion."""
    root = str(Path(output_directory).resolve())
    active = _ACTIVE_TRANSACTIONS.get()
    if root in active:
        yield
        return
    with single_writer_lock(Path(output_directory)):
        token = _ACTIVE_TRANSACTIONS.set(active | {root})
        try:
            yield
        finally:
            _ACTIVE_TRANSACTIONS.reset(token)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.name.endswith(".jsonl.gz"):
        with temporary.open("wb") as raw:
            # A blank filename and zero timestamp make gzip bytes reproducible
            # across directories and runs with identical records.
            with gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0
            ) as stream:
                for row in rows:
                    stream.write((json.dumps(
                        row,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ) + "\n").encode("utf-8"))
            raw.flush()
            os.fsync(raw.fileno())
    else:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                # JSONL is machine-facing and content-bound by file hash. Compact
                # separators reduce process-file size without losing information.
                stream.write(json.dumps(
                    row, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                ) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    temporary.replace(path)


def _externalize_split_registry(case: dict, reference: str) -> dict:
    """Return a transport row with repeated immutable documents removed."""

    row = dict(case)
    metadata = dict(row.get("metadata", {}))
    sampling = dict(metadata.get("sampling", {}))
    document = sampling.pop("split_registry_document", None)
    if not isinstance(document, dict):
        raise InputValidationError(
            "formal case lacks the complete split registry snapshot before externalization",
            context={"case_id": case.get("case_id")},
        )
    if _SPLIT_REGISTRY_REFERENCE_FIELD in sampling:
        raise InputValidationError(
            "formal case cannot contain both embedded and external split registry declarations",
            context={"case_id": case.get("case_id")},
        )
    sampling[_SPLIT_REGISTRY_REFERENCE_FIELD] = reference
    references: dict[str, str] = {}
    for field, location in (
        ("label_spec_document", _LABEL_SPEC_REFERENCE),
        ("sampling_spec_document", _SAMPLING_SPEC_REFERENCE),
        ("training_design_document", _TRAINING_DESIGN_REFERENCE),
    ):
        value = sampling.pop(field, None)
        if value is not None:
            if not isinstance(value, dict):
                raise InputValidationError(
                    "formal case contract document must be an object before externalization",
                    context={"case_id": case.get("case_id"), "field": field},
                )
            references[field] = location
    if "label_spec_document" not in references:
        raise InputValidationError(
            "formal case lacks the label specification document before externalization",
            context={"case_id": case.get("case_id")},
        )
    if _CONTRACT_DOCUMENTS_REFERENCE_FIELD in sampling:
        raise InputValidationError(
            "formal case already declares external contract documents",
            context={"case_id": case.get("case_id")},
        )
    sampling[_CONTRACT_DOCUMENTS_REFERENCE_FIELD] = references
    metadata["sampling"] = sampling
    row["metadata"] = metadata
    return row


def _write_formal_cases_atomic(
    path: Path, cases: Iterable[dict], *, registry_reference: str,
) -> None:
    _write_jsonl_atomic(
        path,
        (_externalize_split_registry(case, registry_reference) for case in cases),
    )


def _write_candidates_parquet(path: Path, rows: list[dict]) -> None:
    import pyarrow.parquet as pq
    from .sampling_candidate_schema import candidate_table
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(candidate_table(rows), temporary, compression="zstd")
    temporary.replace(path)


def _load_verified_candidates_parquet(path: Path) -> list[dict]:
    """Load the sole canonical reference-candidate ledger fail closed."""

    import pyarrow.parquet as pq
    from .sampling_candidate_schema import candidate_table_schema

    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise InputValidationError(
            "unable to read reference candidate Parquet ledger",
            context={"path": str(path)},
        ) from exc
    if not table.schema.equals(candidate_table_schema(), check_metadata=True):
        raise InputValidationError("candidate Parquet schema is invalid")
    return table.to_pylist()


def _write_adaptive_candidates_parquet(path: Path, rows: list[dict]) -> None:
    import pyarrow.parquet as pq
    from .sampling_candidate_schema import adaptive_candidate_table

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(adaptive_candidate_table(rows), temporary, compression="zstd")
    temporary.replace(path)


def _load_verified_adaptive_candidates_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    from .sampling_candidate_schema import adaptive_candidate_table_schema

    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise InputValidationError(
            "unable to read adaptive candidate Parquet ledger",
            context={"path": str(path)},
        ) from exc
    if not table.schema.equals(adaptive_candidate_table_schema(), check_metadata=True):
        raise InputValidationError("adaptive candidate Parquet schema is invalid")
    return table.to_pylist()


def _identity(engine, base_case, sampling_spec, label_spec, damage_tree_definition_hash, *, input_snapshot=None) -> dict:
    value = {
        "target_hash": engine.target_hash, "physics_registry_hash": engine.registry.hash,
        "engine_code_hash": ENGINE_CODE_HASH,
        "damage_tree_definition_hash": damage_tree_definition_hash,
        "base_case_hash": case_identity_hash(base_case),
        "sampling_spec_hash": sampling_spec.hash, "label_spec_hash": label_spec.hash,
        "seeds": sampling_spec.seeds, "environment": capture_environment_identity(),
        "input_dependency_manifest_hash": (input_snapshot or {}).get("dependency_manifest_hash"),
    }
    return {**value, "identity_hash": content_hash(value)}


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("unable to load sampling checkpoint artifact", context={"path": str(path)}) from exc
    if not isinstance(value, dict):
        raise InputValidationError("sampling checkpoint artifact must be a JSON object")
    return value


def _iter_rows(path: Path, *, allow_empty: bool = False) -> Iterator[dict]:
    """Parse a JSONL ledger incrementally and report the exact bad line."""

    count = 0
    try:
        opener = gzip.open if path.name.endswith(".jsonl.gz") else open
        with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise InputValidationError(
                        "sampling ledger is not readable UTF-8 JSONL",
                        context={
                            "path": str(path), "line": line_number,
                            "column": exc.colno,
                        },
                    ) from exc
                if not isinstance(row, dict):
                    raise InputValidationError(
                        "sampling ledger must contain object records",
                        context={"path": str(path), "line": line_number},
                    )
                count += 1
                yield row
    except InputValidationError:
        raise
    except (OSError, UnicodeError, EOFError) as exc:
        raise InputValidationError(
            "sampling ledger is not readable UTF-8 JSONL",
            context={"path": str(path)},
        ) from exc
    if count == 0 and not allow_empty:
        raise InputValidationError(
            "sampling ledger must contain nonempty object records",
            context={"path": str(path)},
        )


def _load_rows(path: Path) -> list[dict]:
    return list(_iter_rows(path))


def _load_rows_allow_empty(path: Path) -> list[dict]:
    return list(_iter_rows(path, allow_empty=True))


def _resolve_reference_ledger(
    root: Path, *, compressed_relative: str, legacy_relative: str,
    require_exists: bool = True,
) -> Path:
    """Resolve one committed reference ledger with legacy plain-JSONL support."""

    compressed = root / compressed_relative
    legacy = root / legacy_relative
    existing = [path for path in (compressed, legacy) if path.is_file()]
    if not existing and not require_exists:
        # Callers that immediately create or consume the ledger may need the
        # canonical location before a mocked/test initializer has materialized
        # it. Ambiguous dual ledgers remain an error in every mode.
        return compressed
    if len(existing) != 1:
        raise InputValidationError(
            "reference sampling ledger is missing or ambiguous",
            context={
                "compressed_path": str(compressed),
                "legacy_path": str(legacy),
            },
        )
    return existing[0]


def reference_formal_cases_path(
    output_directory: str | Path, *, require_exists: bool = True,
) -> Path:
    return _resolve_reference_ledger(
        Path(output_directory),
        compressed_relative=_REFERENCE_FORMAL_CASES_LEDGER,
        legacy_relative=_LEGACY_REFERENCE_FORMAL_CASES_LEDGER,
        require_exists=require_exists,
    )


def _load_reference_candidate_rows_compat(root: Path) -> list[dict]:
    """Load reference candidates for old and compact multiround archives."""

    parquet_path = root / "rounds/round-000/candidates.parquet"
    legacy_path = root / _LEGACY_REFERENCE_CANDIDATES_LEDGER
    if parquet_path.is_file():
        rows = _load_verified_candidates_parquet(parquet_path)
        if legacy_path.is_file():
            legacy_rows = _load_rows(legacy_path)
            if len(legacy_rows) != len(rows) or any(
                any(stored.get(key) != value for key, value in row.items())
                or any(value is not None for key, value in stored.items() if key not in row)
                for row, stored in zip(legacy_rows, rows)
            ):
                raise InputValidationError(
                    "legacy candidate JSONL differs from the canonical Parquet ledger"
                )
        return rows
    if legacy_path.is_file():
        return _load_rows(legacy_path)
    raise InputValidationError(
        "reference candidate ledger is missing",
        context={"parquet_path": str(parquet_path), "legacy_path": str(legacy_path)},
    )


def _reference_plan_files(root: Path) -> set[str]:
    files = _PLAN_FILES_COMMON | {
        reference_formal_cases_path(root).relative_to(root).as_posix(),
    }
    if (root / _LEGACY_REFERENCE_CANDIDATES_LEDGER).is_file():
        files.add(_LEGACY_REFERENCE_CANDIDATES_LEDGER)
    return files


def _hydrate_split_registry(
    case: dict, registry: SplitRegistry, *, expected_reference: str,
    registry_document: dict | None = None,
) -> dict:
    """Hydrate one transport case and verify its immutable registry reference."""

    metadata = case.get("metadata")
    sampling = metadata.get("sampling") if isinstance(metadata, dict) else None
    if not isinstance(sampling, dict):
        raise InputValidationError(
            "formal case lacks a sampling envelope",
            context={"case_id": case.get("case_id")},
        )
    embedded = sampling.get("split_registry_document")
    reference = sampling.get(_SPLIT_REGISTRY_REFERENCE_FIELD)
    if embedded is not None:
        # Legacy checkpoint compatibility.  Reject an ambiguous dual source and
        # verify the historical inline snapshot against the canonical registry.
        if reference is not None:
            raise InputValidationError(
                "formal case declares both embedded and external split registries",
                context={"case_id": case.get("case_id")},
            )
        from .sampling_validation import load_verified_split_registry
        load_verified_split_registry(
            embedded,
            expected_hash=registry.definition_hash,
            expected_id=registry.registry_id,
        )
        return case
    if reference != expected_reference:
        raise InputValidationError(
            "formal case split registry reference is missing or unexpected",
            context={
                "case_id": case.get("case_id"),
                "expected_reference": expected_reference,
                "actual_reference": reference,
            },
        )
    if (
        sampling.get("split_registry_hash") != registry.definition_hash
        or sampling.get("split_registry_id") != registry.registry_id
        or metadata.get("split_registry_hash") != registry.definition_hash
        or metadata.get("frozen_split") != sampling.get("frozen_split")
    ):
        raise InputValidationError(
            "formal case split registry reference differs from its declared identity",
            context={"case_id": case.get("case_id")},
        )
    lineage_id = sampling.get("lineage_id")
    assignment = registry.assignment(lineage_id).to_dict()
    actual_assignment = {
        "lineage_id": lineage_id,
        "split": sampling.get("frozen_split"),
        "role": sampling.get("role"),
        "epistemic_group_id": sampling.get("epistemic_group_id"),
    }
    if actual_assignment != assignment:
        raise InputValidationError(
            "formal case contradicts its external frozen split registry",
            context={"case_id": case.get("case_id")},
        )
    hydrated = dict(case)
    hydrated_metadata = dict(metadata)
    hydrated_sampling = dict(sampling)
    hydrated_sampling.pop(_SPLIT_REGISTRY_REFERENCE_FIELD)
    hydrated_sampling["split_registry_document"] = (
        registry_document
        if registry_document is not None
        else registry.definition_document()
    )
    hydrated_metadata["sampling"] = hydrated_sampling
    hydrated["metadata"] = hydrated_metadata
    return hydrated


def _hydrate_contract_documents(
    case: dict, sampling_spec: SamplingSpec, label_spec: LabelSpec,
    *, training_design_document: dict | None = None,
) -> dict:
    metadata = case.get("metadata")
    sampling = metadata.get("sampling") if isinstance(metadata, dict) else None
    if not isinstance(sampling, dict):
        raise InputValidationError("formal case lacks a sampling envelope")
    references = sampling.get(_CONTRACT_DOCUMENTS_REFERENCE_FIELD)
    if references is None:
        # Legacy checkpoints retain complete per-case documents.
        return case
    if not isinstance(references, dict):
        raise InputValidationError("formal case contract document references are malformed")
    expected = {"label_spec_document": _LABEL_SPEC_REFERENCE}
    if sampling.get("schema_version") in {
        "1.1.0",
        "1.2.0",
        "1.3.0",
        "1.4.0",
    }:
        expected.update({
            "sampling_spec_document": _SAMPLING_SPEC_REFERENCE,
            "training_design_document": _TRAINING_DESIGN_REFERENCE,
        })
    if references != expected or any(field in sampling for field in expected):
        raise InputValidationError(
            "formal case contract document references are missing, unexpected, or ambiguous",
            context={"case_id": case.get("case_id")},
        )
    if (
        sampling.get("label_spec_hash") != label_spec.hash
        or sampling.get("label_spec_id") != label_spec.spec_id
    ):
        raise InputValidationError(
            "formal case label specification reference differs from its declared identity",
            context={"case_id": case.get("case_id")},
        )
    hydrated = dict(case)
    hydrated_metadata = dict(metadata)
    hydrated_sampling = dict(sampling)
    hydrated_sampling.pop(_CONTRACT_DOCUMENTS_REFERENCE_FIELD)
    hydrated_sampling["label_spec_document"] = label_spec.data
    if "sampling_spec_document" in expected:
        if (
            sampling.get("sampling_spec_hash") != sampling_spec.hash
            or sampling.get("sampling_spec_id") != sampling_spec.spec_id
            or sampling.get("training_design_hash") != sampling_spec.training_design_hash
        ):
            raise InputValidationError(
                "formal case sampling specification reference differs from its declared identity",
                context={"case_id": case.get("case_id")},
            )
        hydrated_sampling["sampling_spec_document"] = sampling_spec.data
        hydrated_sampling["training_design_document"] = (
            training_design_document
            if training_design_document is not None
            else sampling_spec.training_design_document
        )
    hydrated_metadata["sampling"] = hydrated_sampling
    hydrated["metadata"] = hydrated_metadata
    return hydrated


def _iter_hydrated_formal_cases(
    path: Path, registry: SplitRegistry, sampling_spec: SamplingSpec,
    label_spec: LabelSpec, *, expected_reference: str,
) -> Iterator[dict]:
    # Reuse immutable, already content-verified sidecar documents.  Besides
    # bounding RAM, object sharing lets identity verification serialize the
    # O(N) registry once instead of once per case.
    registry_document = registry.definition_document()
    training_design_document = getattr(
        sampling_spec, "training_design_document", None
    )
    for case in _iter_rows(path):
        hydrated = _hydrate_split_registry(
            case,
            registry,
            expected_reference=expected_reference,
            registry_document=registry_document,
        )
        yield _hydrate_contract_documents(
            hydrated,
            sampling_spec,
            label_spec,
            training_design_document=training_design_document,
        )


class _ReferenceFormalFrozenInputs(CapturedCaseInputs):
    """Repeatable frozen view over one compact reference transport ledger.

    The copied ledger never contains the registry/specification documents.
    Each pass reopens the copied JSONL file and hydrates against content-checked
    sidecar copies.  This is intentionally an internal type: only the sampling
    checkpoint can construct it and only :class:`ShardedDatasetWriter` opts in
    to its pre-captured shard path.
    """

    def __init__(
        self,
        ledger: Path,
        registry: SplitRegistry,
        sampling_spec: SamplingSpec,
        label_spec: LabelSpec,
        *,
        expected_identity: dict,
        source_root: Path,
    ) -> None:
        self.spool_path = ledger
        self.identity = deepcopy(expected_identity)
        self._registry = registry
        self._sampling_spec = sampling_spec
        self._label_spec = label_spec
        self._registry_document = registry.definition_document()
        self._training_design_document = sampling_spec.training_design_document
        self._canonical_subtrees = _canonical_subtree_cache(
            (
                self._registry_document,
                sampling_spec.data,
                self._training_design_document,
                label_spec.data,
            )
        )
        self.metadata = {
            **deepcopy(expected_identity),
            "dependencies": [],
            "locations": [],
            "cases_base_directory": str(source_root),
            "freeze_policy": (
                "finite_sampling_transport_spooled_compact_then_strictly_hydrated"
            ),
            "transport": {
                "schema_version": "sampling-compact-freeze@1.0.0",
                "ledger_bytes": ledger.stat().st_size,
                "repeated_contract_documents_in_spool": False,
            },
        }

    def __len__(self) -> int:
        return int(self.identity["expected_case_count"])

    def __iter__(self) -> Iterator[FrozenCase]:
        for row in _iter_rows(self.spool_path):
            hydrated = _hydrate_split_registry(
                row,
                self._registry,
                expected_reference=_REFERENCE_SPLIT_REGISTRY_PATH,
                registry_document=self._registry_document,
            )
            hydrated = _hydrate_contract_documents(
                hydrated,
                self._sampling_spec,
                self._label_spec,
                training_design_document=self._training_design_document,
            )
            # External fragment inputs deliberately use the legacy freezer.
            # Encountering one here means the committed input snapshot and the
            # transport rows disagree, which must fail before publication.
            source = hydrated.get("warhead", {}).get("fragment_source", {})
            if isinstance(source, dict) and source.get("fragment_list_ref") is not None:
                raise InputValidationError(
                    "compact sampling freeze encountered an undeclared external input"
                )
            yield FrozenCase(
                hydrated,
                input_identity={"dependencies": [], "locations": []},
            )

    def case_hash(self, case: dict) -> str:
        return case_identity_hash(
            case, _canonical_subtrees=self._canonical_subtrees
        )

    def chunk_input_metadata(self, case_hashes: Iterable[str]) -> dict:
        identity = input_identity_from_case_hashes(case_hashes)
        return {
            **identity,
            "dependencies": [],
            "locations": [],
            "cases_base_directory": str(self.spool_path.parent),
            "freeze_policy": self.metadata["freeze_policy"],
            "transport": deepcopy(self.metadata["transport"]),
        }

    def validate_complete_identity(self) -> None:
        hashes: list[str] = []
        for case in self:
            hashes.append(self.case_hash(case))
        observed = input_identity_from_case_hashes(hashes)
        if observed != self.identity:
            raise InputValidationError(
                "compact formal case transport differs from the frozen ordered input identity",
                context={
                    "expected_count": self.identity.get("expected_case_count"),
                    "observed_count": observed.get("expected_case_count"),
                    "expected_ordered_hash": self.identity.get(
                        "ordered_case_identity_hash"
                    ),
                    "observed_ordered_hash": observed.get(
                        "ordered_case_identity_hash"
                    ),
                },
            )


class _ReferenceFormalCaseSource(CapturedCaseSource):
    """Internal, replayable source for a committed reference sampling run."""

    def __init__(self, output_directory: str | Path) -> None:
        self.root = Path(output_directory).resolve()

    def __iter__(self) -> Iterator[dict]:
        from .sampling_validation import load_verified_split_registry

        _verify_committed_sidecars(self.root)
        registry = load_verified_split_registry(self.root / "plan/split_registry.json")
        sampling_spec = SamplingSpec.from_file(self.root / "plan/sampling_spec.json")
        label_spec = LabelSpec.from_file(self.root / "plan/label_spec.json")
        yield from _iter_hydrated_formal_cases(
            reference_formal_cases_path(self.root),
            registry,
            sampling_spec,
            label_spec,
            expected_reference=_REFERENCE_SPLIT_REGISTRY_PATH,
        )

    def freeze_captured_inputs(
        self, *, cases_base_directory: str | Path | None = None,
    ) -> Any:
        """Expose the compact capture through the neutral input seam."""

        return self._freeze_compact(
            cases_base_directory=cases_base_directory
        )

    @contextmanager
    def _freeze_compact(
        self, *, cases_base_directory: str | Path | None = None,
    ) -> Iterator[Any]:
        """Capture compact transport bytes, or conservatively use legacy freeze."""

        checkpoint = _load_json(self.root / "sampling_checkpoint.json")
        if (self.root / ".sampling.incomplete").exists():
            raise InputValidationError(
                "sampling run has an incomplete marker and requires operator review"
            )
        identity = checkpoint.get("identity")
        if (
            checkpoint.get("schema_version")
            not in SUPPORTED_SAMPLING_CHECKPOINT_VERSIONS
            or checkpoint.get("status") not in {"initialized", "archive_complete"}
            or checkpoint.get("rounds_committed") != [0]
            or not isinstance(identity, dict)
            or identity.get("identity_hash")
            != content_hash(
                {key: value for key, value in identity.items() if key != "identity_hash"}
            )
        ):
            raise InputValidationError(
                "compact sampling freeze requires a valid committed reference checkpoint"
            )
        _verify_committed_sidecars(self.root)
        input_snapshot = _load_json(self.root / "plan/input_snapshot.json")
        dependencies = input_snapshot.get("dependencies")
        if not isinstance(dependencies, list):
            raise InputValidationError("sampling input dependency manifest is malformed")
        if content_hash(dependencies) != identity.get(
            "input_dependency_manifest_hash"
        ):
            raise InputValidationError(
                "sampling input dependency manifest differs from checkpoint identity"
            )
        if dependencies:
            # External input snapshot semantics are security-sensitive and are
            # already correct in the generic freezer.  Keep that exact path
            # until a separately versioned compact dependency transport exists.
            with freeze_case_inputs(
                (case for case in self),
                cases_base_directory=cases_base_directory,
            ) as frozen:
                yield frozen
            return

        records = checkpoint.get("committed_files")
        if (
            not isinstance(records, list)
            or not records
            or any(
                not isinstance(record, dict)
                or set(record) != {"path", "content_hash"}
                for record in records
            )
        ):
            raise InputValidationError("sampling checkpoint has no committed file inventory")
        inventory: dict[str, str] = {}
        for record in records:
            relative = record["path"]
            if relative in inventory:
                raise InputValidationError(
                    "sampling committed inventory contains duplicate paths"
                )
            inventory[relative] = record["content_hash"]

        formal_relative = reference_formal_cases_path(self.root).relative_to(
            self.root
        ).as_posix()
        required = (
            "plan/split_registry.json",
            "plan/sampling_spec.json",
            "plan/label_spec.json",
            "plan/formal_input_identity.json",
            "plan/input_snapshot.json",
            formal_relative,
        )
        with tempfile.TemporaryDirectory(
            prefix="sampling-compact-input-snapshot-"
        ) as temporary:
            captured_root = Path(temporary)
            for relative in required:
                expected_hash = inventory.get(relative)
                if not is_sha256_identity(
                    expected_hash, acceptance="legacy_python_hex"
                ):
                    raise InputValidationError(
                        "compact sampling freeze input is absent from committed inventory",
                        context={"path": relative},
                    )
                source = _checked_path(self.root, relative)
                target = captured_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copyfile(source, target)
                except OSError as exc:
                    raise InputValidationError(
                        "compact sampling freeze could not capture a committed sidecar",
                        context={"path": relative},
                    ) from exc
                if (
                    file_hash(source) != expected_hash
                    or file_hash(target) != expected_hash
                ):
                    raise InputValidationError(
                        "compact sampling freeze sidecar changed during capture",
                        context={"path": relative},
                    )
                target.chmod(0o444)

            formal_identity = _load_json(
                captured_root / "plan/formal_input_identity.json"
            )
            identity_without_hash = {
                key: value
                for key, value in formal_identity.items()
                if key != "identity_hash"
            }
            if formal_identity.get("identity_hash") != content_hash(
                identity_without_hash
            ):
                raise InputValidationError(
                    "formal input snapshot identity hash is invalid"
                )
            if checkpoint.get("formal_input_identity") != formal_identity:
                raise InputValidationError(
                    "checkpoint formal input identity differs from committed plan"
                )

            from .sampling_validation import load_verified_split_registry

            registry = load_verified_split_registry(
                captured_root / "plan/split_registry.json"
            )
            sampling_spec = SamplingSpec.from_file(
                captured_root / "plan/sampling_spec.json"
            )
            label_spec = LabelSpec.from_file(
                captured_root / "plan/label_spec.json"
            )
            if (
                registry.definition_hash != checkpoint.get("split_registry_hash")
                or sampling_spec.hash != identity.get("sampling_spec_hash")
                or label_spec.hash != identity.get("label_spec_hash")
            ):
                raise InputValidationError(
                    "compact sampling freeze sidecar identities differ from checkpoint"
                )
            frozen = _ReferenceFormalFrozenInputs(
                captured_root / formal_relative,
                registry,
                sampling_spec,
                label_spec,
                expected_identity=formal_identity,
                source_root=self.root,
            )
            # This complete pass happens before the writer can create a shard or
            # checkpoint.  It catches short/extra rows and any case-local
            # transport tampering while the copied bytes are stable.
            frozen.validate_complete_identity()
            yield frozen


def iter_reference_formal_cases(output_directory: str | Path) -> Iterable[dict]:
    """Return a replayable engine-ready view of the reference formal cases."""

    return _ReferenceFormalCaseSource(output_directory)


def iter_adaptive_formal_cases(
    output_directory: str | Path, round_index: int,
) -> Iterator[dict]:
    """Stream engine-ready selected cases for one adaptive Train round."""

    from .sampling_validation import load_verified_split_registry

    directory = Path(output_directory) / "rounds" / f"round-{round_index:03d}"
    registry = load_verified_split_registry(directory / "adaptive_split_registry.json")
    sampling_spec = SamplingSpec.from_file(Path(output_directory) / "plan/sampling_spec.json")
    label_spec = LabelSpec.from_file(Path(output_directory) / "plan/label_spec.json")
    yield from _iter_hydrated_formal_cases(
        directory / "selected_formal_cases.jsonl",
        registry,
        sampling_spec,
        label_spec,
        expected_reference=_ADAPTIVE_SPLIT_REGISTRY_PATH,
    )


def _checked_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or any(char in relative for char in "*?[]"):
        raise InputValidationError("sampling inventory path must be an exact nonempty string")
    value = Path(relative)
    path = (root / value).resolve()
    if value.is_absolute() or ".." in value.parts or not path.is_relative_to(root.resolve()):
        raise InputValidationError("sampling checkpoint file escapes the run directory")
    return path


def _verify_inventory(root: Path, files: Any, expected_paths: set[str] | None = None) -> None:
    if not isinstance(files, list) or not files:
        raise InputValidationError("sampling checkpoint has no committed file inventory")
    paths = set()
    for record in files:
        if not isinstance(record, dict) or set(record) != {"path", "content_hash"}:
            raise InputValidationError("sampling checkpoint file inventory is malformed")
        path = _checked_path(root, record["path"])
        canonical = path.relative_to(root.resolve()).as_posix()
        if canonical in paths:
            raise InputValidationError("sampling committed inventory contains duplicate paths")
        paths.add(canonical)
        if not path.is_file() or file_hash(path) != record["content_hash"]:
            raise InputValidationError("sampling checkpoint committed file is missing or changed", context={"path": canonical})
    if expected_paths is not None and paths != expected_paths:
        raise InputValidationError("sampling committed inventory differs from the required frozen plan", context={"missing": sorted(expected_paths - paths), "extra": sorted(paths - expected_paths)})


def _verify_committed_sidecars(root: Path) -> None:
    """Bind every hydration input to bytes in the committed checkpoint."""

    checkpoint = _load_json(root / "sampling_checkpoint.json")
    records = checkpoint.get("committed_files")
    if (
        not isinstance(records, list)
        or not records
        or any(
            not isinstance(record, dict)
            or set(record) != {"path", "content_hash"}
            for record in records
        )
    ):
        raise InputValidationError("sampling checkpoint has no committed sidecar inventory")
    paths = [record["path"] for record in records]
    if len(paths) != len(set(paths)):
        raise InputValidationError("sampling checkpoint sidecar inventory has duplicate paths")
    inventory = {record["path"]: record["content_hash"] for record in records}
    formal_relative = reference_formal_cases_path(root).relative_to(root).as_posix()
    for relative in (
        "plan/split_registry.json",
        "plan/sampling_spec.json",
        "plan/label_spec.json",
        formal_relative,
    ):
        expected = inventory.get(relative)
        path = _checked_path(root, relative)
        if (
            not is_sha256_identity(expected, acceptance="legacy_python_hex")
            or not path.is_file()
            or file_hash(path) != expected
        ):
            raise InputValidationError(
                "formal-case hydration sidecar is missing or differs from the committed checkpoint",
                context={"path": relative},
            )


def _sequence_identity(cases: list[dict]) -> tuple[list[str], str]:
    shared_documents: list[Any] = []
    if cases:
        sampling = cases[0].get("metadata", {}).get("sampling", {})
        if isinstance(sampling, dict):
            for name in (
                "split_registry_document",
                "sampling_spec_document",
                "training_design_document",
                "label_spec_document",
                "role_names",
                "stratum_names",
            ):
                value = sampling.get(name)
                if isinstance(value, (dict, list, tuple)):
                    shared_documents.append(value)
    canonical_subtrees = _canonical_subtree_cache(shared_documents)
    values = [
        case_identity_hash(case, _canonical_subtrees=canonical_subtrees)
        for case in cases
    ]
    sequence = hashlib.sha256()
    for value in values:
        sequence.update(value.encode("ascii") + b"\n")
    return values, "sha256:" + sequence.hexdigest()


def _load_existing_plan(root: Path, sampling_spec=None, label_spec=None) -> ReferencePlan:
    from .sampling_validation import load_verified_split_registry
    registry = load_verified_split_registry(root / "plan" / "split_registry.json")
    sampling_spec = sampling_spec or SamplingSpec.from_file(root / "plan/sampling_spec.json")
    label_spec = label_spec or LabelSpec.from_file(root / "plan/label_spec.json")
    cases = list(_iter_hydrated_formal_cases(
        reference_formal_cases_path(root),
        registry,
        sampling_spec,
        label_spec,
        expected_reference=_REFERENCE_SPLIT_REGISTRY_PATH,
    ))
    rows = _load_verified_candidates_parquet(
        root / "rounds/round-000/candidates.parquet"
    )
    try:
        latent = np.asarray([row["latent"] for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise InputValidationError("candidate ledger latent design is malformed") from exc
    return ReferencePlan(cases=cases, candidate_rows=rows, latent_design=latent, split_registry=registry)


def _verify_contact_conditioned_plan(
    plan: ReferencePlan, sampling_spec: SamplingSpec
) -> None:
    """Recheck persisted 1.3 refusals and per-case S-CONTACT bindings."""

    if not sampling_spec.has_contact_conditioning:
        return
    conditioning = sampling_spec.contact_conditioning
    if conditioning is None:
        raise InputValidationError(
            "contact-conditioned checkpoint lacks its SamplingSpec declaration"
        )
    latent = np.asarray(plan.latent_design, dtype=np.float64)
    if (
        latent.ndim != 2
        or latent.shape
        != (len(plan.candidate_rows), sampling_spec.latent_dimension)
        or not np.isfinite(latent).all()
    ):
        raise InputValidationError(
            "contact-conditioned candidate ledger and latent design are not aligned"
        )

    selected_ids = {
        row["candidate_id"]
        for row in plan.candidate_rows
        if row.get("selected") is True
    }
    formal_ids = {
        case.get("metadata", {}).get("sampling", {}).get("candidate_id")
        for case in plan.cases
    }
    if selected_ids != formal_ids:
        raise InputValidationError(
            "contact-conditioned formal cases differ from selected candidates"
        )

    for row in plan.candidate_rows:
        if row.get("selected") is True:
            if row.get("refusal") is not None:
                raise InputValidationError(
                    "selected contact candidate cannot carry a refusal record",
                    context={"candidate_id": row.get("candidate_id")},
                )
            if any(
                row.get(name) is not None
                for name in ("executed", "successful", "admitted")
            ):
                raise InputValidationError(
                    "selected pre-execution contact candidate must preserve "
                    "null execution state",
                    context={"candidate_id": row.get("candidate_id")},
                )
            continue
        if row.get("weight_mask") is not False or row.get("formal_count") != 0:
            raise InputValidationError(
                "refused contact candidate must be unselected, unweighted and unexecuted",
                context={"candidate_id": row.get("candidate_id")},
            )
        if any(row.get(name) is not False for name in ("executed", "successful", "admitted")):
            raise InputValidationError(
                "refused contact candidate execution flags are missing or invalid",
                context={"candidate_id": row.get("candidate_id")},
            )
        refusal = row.get("refusal")
        if (
            not isinstance(refusal, dict)
            or set(refusal) != {"code", "message", "context_json"}
            or not isinstance(refusal.get("code"), str)
            or not refusal["code"]
            or not isinstance(refusal.get("message"), str)
            or not refusal["message"]
            or not isinstance(refusal.get("context_json"), str)
        ):
            raise InputValidationError(
                "refused contact candidate lacks a complete refusal record",
                context={"candidate_id": row.get("candidate_id")},
            )
        try:
            refusal_context = json.loads(refusal["context_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise InputValidationError(
                "refused contact candidate context is not canonical JSON",
                context={"candidate_id": row.get("candidate_id")},
            ) from exc
        if (
            not isinstance(refusal_context, dict)
            or refusal_context.get("contact_contract_id")
            != conditioning["contract_id"]
            or refusal_context.get("contact_contract_hash")
            != conditioning["contract_hash"]
            or refusal_context.get("selection_declaration_hash")
            != conditioning["selection_declaration_hash"]
            or refusal_context.get("formal_teacher_executed") is not False
        ):
            raise InputValidationError(
                "refused contact candidate context is not bound to S-CONTACT",
                context={"candidate_id": row.get("candidate_id")},
            )
        source_context = refusal_context.get("refusal_context")
        if (
            isinstance(source_context, dict)
            and source_context.get("reason_code") is not None
            and source_context.get("reason_code") != refusal["code"]
        ):
            raise InputValidationError(
                "refused contact candidate reason code is inconsistent",
                context={"candidate_id": row.get("candidate_id")},
            )
        expected_no_formal_hash = content_hash(
            {
                "candidate_id": row["candidate_id"],
                "formal_case_status": "not_created",
                "refusal": refusal,
            }
        )
        if row.get("formal_case_hash") != expected_no_formal_hash:
            raise InputValidationError(
                "refused contact candidate claims an invalid formal-case sentinel",
                context={"candidate_id": row.get("candidate_id")},
            )

    for case in plan.cases:
        sampling = case.get("metadata", {}).get("sampling", {})
        binding = sampling.get("contact_conditioning")
        if not isinstance(binding, dict):
            raise InputValidationError(
                "formal contact case lacks its eligibility binding",
                context={"case_id": case.get("case_id")},
            )
        eligibility = binding.get("eligibility_record")
        expected_binding = {
            "contract_id": conditioning["contract_id"],
            "contract_hash": conditioning["contract_hash"],
            "mode": conditioning["mode"],
            "estimand": conditioning["estimand"],
            "selection_declaration_hash": conditioning[
                "selection_declaration_hash"
            ],
        }
        observed_binding = {
            key: binding.get(key) for key in expected_binding
        }
        if observed_binding != expected_binding or not isinstance(
            eligibility, dict
        ):
            raise InputValidationError(
                "formal contact case has an inconsistent S-CONTACT identity",
                context={"case_id": case.get("case_id")},
            )
        checks = eligibility.get("checks")
        if (
            eligibility.get("contract_id") != conditioning["contract_id"]
            or eligibility.get("contract_hash") != conditioning["contract_hash"]
            or eligibility.get("contact_mode") != conditioning["mode"]
            or (eligibility.get("estimand") or {}).get("id")
            != conditioning["estimand"]
            or (eligibility.get("selection") or {}).get("content_hash")
            != conditioning["selection_declaration_hash"]
            or eligibility.get("status") != "eligible_contact_conditioned"
            or eligibility.get("formal_label_allowed") is not True
            or not isinstance(checks, dict)
            or not checks
            or not all(value is True for value in checks.values())
        ):
            raise InputValidationError(
                "formal contact case eligibility record is incomplete or inconsistent",
                context={"case_id": case.get("case_id")},
            )


def _verify_plan_semantics(root: Path, checkpoint: dict) -> ReferencePlan:
    from .sampling_validation import validate_registry_assignments, validate_candidate_set
    from .sampling_candidate_schema import candidate_table, candidate_table_schema
    from .learning_input_identity import (
        learning_input_fingerprint,
        loitering_impact_contact_pose_learning_input_fingerprint,
        loitering_impact_learning_input_fingerprint,
    )
    import pyarrow.parquet as pq
    sampling_spec = SamplingSpec.from_file(root / "plan" / "sampling_spec.json")
    label_spec = LabelSpec.from_file(root / "plan" / "label_spec.json")
    identity = checkpoint["identity"]
    if sampling_spec.hash != identity["sampling_spec_hash"] or label_spec.hash != identity["label_spec_hash"]:
        raise InputValidationError("persisted specifications differ from sampling identity")
    if case_identity_hash(_load_json(root / "plan" / "base_case.json")) != identity["base_case_hash"]:
        raise InputValidationError("persisted base case differs from frozen input identity")
    frozen = _load_json(root / "plan" / "frozen_hashes.json")
    if any(frozen.get(name) != value for name, value in identity.items()) or frozen.get("frozen_before_outcomes") is not True:
        raise InputValidationError("frozen plan identities do not match checkpoint")
    plan = _load_existing_plan(root)
    _verify_contact_conditioned_plan(plan, sampling_spec)
    if plan.split_registry.definition_hash != checkpoint.get("split_registry_hash") or frozen.get("split_registry_hash") != plan.split_registry.definition_hash:
        raise InputValidationError("frozen split registry does not match checkpoint")
    validate_registry_assignments(plan.split_registry, plan.candidate_rows)
    validate_candidate_set(plan.candidate_rows)
    candidate_table(plan.candidate_rows)  # Include unknown/null keys and scalar types.
    table = pq.read_table(root / "rounds/round-000/candidates.parquet")
    if not table.schema.equals(candidate_table_schema(), check_metadata=True):
        raise InputValidationError("candidate Parquet schema differs from the versioned ledger contract")
    parquet_rows = table.to_pylist()
    if parquet_rows != plan.candidate_rows:
        raise InputValidationError(
            "canonical candidate Parquet ledger changed while verifying the plan"
        )
    legacy_candidate_path = root / _LEGACY_REFERENCE_CANDIDATES_LEDGER
    if legacy_candidate_path.is_file():
        legacy_rows = _load_rows(legacy_candidate_path)
        if len(legacy_rows) != len(plan.candidate_rows) or any(
            any(stored.get(key) != value for key, value in row.items())
            or any(value is not None for key, value in stored.items() if key not in row)
            for row, stored in zip(legacy_rows, plan.candidate_rows)
        ):
            raise InputValidationError(
                "legacy candidate JSONL differs from the canonical Parquet ledger"
            )
    selected = [row for row in plan.candidate_rows if row.get("selected") is True]
    if len(selected) != len(plan.cases):
        raise InputValidationError("selected candidates and formal input count differ")
    round_manifest = _load_json(root / "rounds/round-000/round_manifest.json")
    if (round_manifest.get("round") != 0 or round_manifest.get("labels_observed_at_freeze") is not False
        or round_manifest.get("candidate_count") != len(plan.candidate_rows)
        or round_manifest.get("selected_count") != len(plan.cases)
        or round_manifest.get("sampling_spec_hash") != sampling_spec.hash
        or round_manifest.get("label_spec_hash") != label_spec.hash
        or round_manifest.get("split_registry_hash") != plan.split_registry.definition_hash
        or round_manifest.get("candidate_ledger_hash") != file_hash(root / "rounds/round-000/candidates.parquet")
        or round_manifest.get("formal_cases_hash") != file_hash(reference_formal_cases_path(root))):
        raise InputValidationError("round manifest is not bound to its complete frozen plan")
    if checkpoint.get("schema_version") == "2.1.0" and (
        round_manifest.get("schema_version") != "2.1.0"
        or round_manifest.get("candidate_ledger_format") != "parquet_zstd"
        or round_manifest.get("formal_cases_format") != "jsonl_gzip"
    ):
        raise InputValidationError(
            "round manifest storage formats differ from checkpoint schema 2.1.0"
        )
    if sampling_spec.has_contact_conditioning:
        conditioning = sampling_spec.contact_conditioning
        refused_ids = [
            row["candidate_id"]
            for row in plan.candidate_rows
            if row.get("selected") is False
        ]
        expected_contact_manifest = {
            "contract_hash": conditioning["contract_hash"],
            "selection_declaration_hash": conditioning[
                "selection_declaration_hash"
            ],
            "attempted_count": len(plan.candidate_rows),
            "accepted_count": len(plan.cases),
            "input_refused_count": len(refused_ids),
            "input_refused_candidate_ids_hash": content_hash(refused_ids),
        }
        if round_manifest.get("contact_conditioning") != expected_contact_manifest:
            raise InputValidationError(
                "round manifest does not preserve the complete contact-refusal ledger identity"
            )
    candidate_ids = [row.get("candidate_id") for row in plan.candidate_rows]
    case_ids = [case.get("case_id") for case in plan.cases]
    if any(not isinstance(value, str) or not value for value in candidate_ids + case_ids) or len(set(candidate_ids)) != len(candidate_ids) or len(set(case_ids)) != len(case_ids):
        raise InputValidationError("plan contains duplicate or invalid candidate/case identities")
    hashes, sequence = _sequence_identity(plan.cases)
    formal_identity = _load_json(root / "plan" / "formal_input_identity.json")
    if formal_identity.get("expected_case_count") != len(plan.cases) or formal_identity.get("ordered_case_identity_hash") != sequence:
        raise InputValidationError("formal input identities do not match frozen plan")
    if formal_identity.get("identity_hash") != content_hash({key: value for key, value in formal_identity.items() if key != "identity_hash"}):
        raise InputValidationError("formal input snapshot identity hash is invalid")
    if checkpoint.get("formal_input_identity") != formal_identity:
        raise InputValidationError("checkpoint formal input identity differs from committed plan")
    actual_rows = []
    learning_inputs = {}
    for case, row, digest in zip(plan.cases, selected, hashes):
        envelope = case.get("metadata", {}).get("sampling", {})
        if row.get("formal_case_hash") != digest or envelope.get("candidate_id") != row["candidate_id"] or envelope.get("lineage_id") != row["lineage_id"]:
            raise InputValidationError("formal cases and selected candidate ledger are not aligned")
        if sampling_spec.is_contact_pose_model_input:
            learning_hash = (
                loitering_impact_contact_pose_learning_input_fingerprint(case)
            )
        elif sampling_spec.is_loitering_impact_model_input:
            learning_hash = loitering_impact_learning_input_fingerprint(case)
        else:
            learning_hash = learning_input_fingerprint(case)
        previous = learning_inputs.get(learning_hash)
        if row.get("learning_input_hash") != learning_hash or (previous is not None and previous[0] != envelope.get("frozen_split")):
            raise InputValidationError("frozen plan learning input fingerprints are inconsistent or cross splits")
        if row.get("duplicate_of_candidate_id") != (previous[1] if previous is not None else None):
            raise InputValidationError("frozen plan duplicate relationships are inconsistent")
        learning_inputs.setdefault(learning_hash, (envelope.get("frozen_split"), row["candidate_id"]))
        actual_rows.append({"lineage_id": envelope.get("lineage_id"), "split": envelope.get("frozen_split"), "role": envelope.get("role"), "epistemic_group_id": envelope.get("epistemic_group_id")})
    validate_registry_assignments(plan.split_registry, actual_rows)
    return plan


def verify_sampling_checkpoint(output_directory: str | Path, expected_identity: dict | None = None) -> dict:
    root = Path(output_directory)
    checkpoint = _load_json(root / "sampling_checkpoint.json")
    if checkpoint.get("schema_version") not in SUPPORTED_SAMPLING_CHECKPOINT_VERSIONS:
        raise InputValidationError("unsupported sampling checkpoint version; legacy pilots require explicit migration")
    if checkpoint.get("schema_version") == "2.1.0" and checkpoint.get(
        "storage_layout"
    ) != {
        "reference_candidate_ledger": "parquet_zstd_only",
        "formal_case_ledger": "jsonl_gzip",
        "machine_json": "compact",
    }:
        raise InputValidationError("sampling checkpoint storage layout is missing or unsupported")
    if checkpoint.get("status") not in {"initialized", "archive_complete"}:
        raise InputValidationError("sampling checkpoint status is invalid")
    if checkpoint.get("rounds_committed") != [0]:
        raise InputValidationError("sampling checkpoint committed rounds are invalid for a reference pilot")
    if {entry.name for entry in (root / "rounds").iterdir()} != {"round-000"}:
        if (root / MULTIROUND_CHECKPOINT_FILE).is_file():
            raise InputValidationError(
                "reference round is attached to explicit multiround state; use verify_multiround_sampling_checkpoint"
            )
        raise InputValidationError("disk sampling rounds differ from committed rounds; preserve and investigate orphan data")
    if (root / ".sampling.incomplete").exists():
        raise InputValidationError("sampling run has an incomplete marker and requires operator review")
    identity = checkpoint.get("identity")
    if not isinstance(identity, dict) or identity.get("identity_hash") != content_hash({key: value for key, value in identity.items() if key != "identity_hash"}):
        raise InputValidationError("sampling checkpoint identity hash is invalid")
    if expected_identity is not None and identity != expected_identity:
        raise InputValidationError("sampling checkpoint identity differs from the requested plan", context={"existing": identity.get("identity_hash"), "requested": expected_identity.get("identity_hash")})
    snapshot = _load_json(root / "plan" / "input_snapshot.json")
    dependencies = snapshot.get("dependencies")
    if not isinstance(dependencies, list) or content_hash(dependencies) != identity.get("input_dependency_manifest_hash"):
        raise InputValidationError("input dependency manifest differs from sampling identity")
    snapshots = snapshot.get("snapshot_files", [])
    if not isinstance(snapshots, list) or any(not isinstance(record, dict) or set(record) != {"path", "content_hash"} for record in snapshots):
        raise InputValidationError("input snapshot file inventory is malformed")
    if snapshots:
        _verify_inventory(root, snapshots)
    if (root / "plan/input_snapshots").exists() and {entry.relative_to(root).as_posix() for entry in (root / "plan/input_snapshots").iterdir()} != {item["path"] for item in snapshots}:
        raise InputValidationError("disk input snapshots differ from the committed snapshot set")
    _verify_inventory(
        root,
        checkpoint.get("committed_files"),
        _reference_plan_files(root) | {item["path"] for item in snapshots},
    )
    plan = _verify_plan_semantics(root, checkpoint)
    formal = checkpoint.get("formal_dataset")
    if checkpoint["status"] == "initialized":
        if formal is not None or checkpoint.get("archive_summary") is not None:
            raise InputValidationError("initialized state must not claim an archived formal dataset")
    else:
        if not isinstance(formal, dict) or not {"manifest_path", "manifest_hash", "committed_files"}.issubset(formal):
            raise InputValidationError("archive_complete state lacks complete formal inventory")
        _verify_inventory(root, formal["committed_files"])
        path = _checked_path(root, formal["manifest_path"])
        if file_hash(path) != formal["manifest_hash"]:
            raise InputValidationError("formal dataset manifest is missing or changed")
        summary, records = _verify_formal_archive(root, checkpoint, plan, path)
        if summary != checkpoint.get("archive_summary") or records != formal["committed_files"]:
            raise InputValidationError("archive completion record differs from current committed dataset")
    return checkpoint


def initialize_sampling_run(output_directory: str | Path, *, engine: DamageEngine, base_case: dict, sampling_spec: SamplingSpec, label_spec: LabelSpec, resume: bool = False, cases_base_directory: str | Path | None = None) -> tuple[dict, ReferencePlan]:
    root = Path(output_directory)
    with sampling_run_transaction(root):
        with freeze_case_inputs([base_case], cases_base_directory=cases_base_directory) as base_inputs:
            frozen_base = next(iter(base_inputs))
            tree_hash = content_hash(engine.target_data["damage_tree"])
            if label_spec.data["source"]["damage_tree_definition_hash"] != tree_hash:
                raise InputValidationError("label specification is bound to a different damage-tree definition")
            label_spec.validate_output_vocabulary(engine.damage_tree.outputs)
            identity = _identity(engine, frozen_base, sampling_spec, label_spec, tree_hash, input_snapshot=base_inputs.identity)
            if root.exists() and any(root.iterdir()):
                if not resume:
                    raise InputValidationError("sampling output directory is not empty; use --resume only for an identical committed plan", context={"path": str(root)})
                checkpoint = verify_sampling_checkpoint(root, identity)
                return checkpoint, _load_existing_plan(root)
            factory = CaseFactory(frozen_base, sampling_spec, label_spec, target_hash=engine.target_hash, damage_tree_definition_hash=tree_hash, physics_registry_hash=engine.registry.hash, engine_code_hash=ENGINE_CODE_HASH, physics_registry=engine.registry, compiled_target=engine.compiled_target)
            plan = factory.build_reference_plan()
            # Do all input checks against the same captured bytes before touching
            # the output directory. Validation does not execute the engine.
            for case in plan.cases:
                if (case.get("encounter") or {}).get("detonation_trigger") == "first_armor_contact":
                    validate_inputs(
                        engine.target_data,
                        case,
                        engine.registry,
                        compiled_target=engine.compiled_target,
                    )
                else:
                    validate_inputs(engine.target_data, case, engine.registry)
            root.mkdir(parents=True, exist_ok=True)
            marker = root / ".sampling.incomplete"
            marker.write_text("sampling initialization in progress\n", encoding="utf-8")
            snapshot_metadata = deepcopy(base_inputs.metadata)
            snapshot_metadata["snapshot_files"] = []
            if frozen_base.input_identity["dependencies"]:
                dependency = frozen_base.input_identity["dependencies"][0]
                relative = "plan/input_snapshots/" + dependency["content_hash"].split(":")[1] + "." + dependency["format"]
                persisted = root / relative
                persisted.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(frozen_base.input_identity["snapshot_reference"], persisted)
                if file_hash(persisted) != dependency["content_hash"]:
                    raise InputValidationError("persisted input snapshot differs from captured bytes")
                persisted.chmod(0o444)
                snapshot_metadata["snapshot_files"].append({"path": relative, "content_hash": file_hash(persisted)})
                for case in [frozen_base] + plan.cases:
                    case["warhead"]["fragment_source"]["fragment_list_ref"] = str(persisted.resolve())
                    case.input_identity["snapshot_reference"] = str(persisted.resolve())
            selected_rows = [
                row for row in plan.candidate_rows if row.get("selected") is True
            ]
            if len(selected_rows) != len(plan.cases) or any(
                row.get("candidate_id")
                != case.get("metadata", {}).get("sampling", {}).get("candidate_id")
                or not is_sha256_identity(
                    row.get("formal_case_hash"), acceptance="legacy_python_hex"
                )
                for case, row in zip(plan.cases, selected_rows)
            ):
                raise InputValidationError(
                    "factory formal case hashes are missing or misaligned"
                )
            formal_identity = input_identity_from_case_hashes(
                (row["formal_case_hash"] for row in selected_rows),
                dependencies=snapshot_metadata["dependencies"],
            )
            _write_json_atomic(root / "plan/sampling_spec.json", sampling_spec.data)
            _write_json_atomic(root / "plan/label_spec.json", label_spec.data)
            _write_json_atomic(root / "plan/base_case.json", dict(frozen_base))
            _write_json_atomic(root / "plan/input_snapshot.json", snapshot_metadata)
            _write_json_atomic(root / "plan/formal_input_identity.json", formal_identity)
            plan.split_registry.write(root / "plan/split_registry.json")
            _write_json_atomic(root / "plan/frozen_hashes.json", {**identity, "split_registry_hash": plan.split_registry.definition_hash, "p0_hash": content_hash(sampling_spec.data["reference_distribution"]), "frozen_before_outcomes": True})
            _write_candidates_parquet(root / "rounds/round-000/candidates.parquet", plan.candidate_rows)
            _write_formal_cases_atomic(
                root / _REFERENCE_FORMAL_CASES_LEDGER,
                plan.cases,
                registry_reference=_REFERENCE_SPLIT_REGISTRY_PATH,
            )
            contact_input_refused_candidate_ids = (
                [
                    row["candidate_id"]
                    for row in plan.candidate_rows
                    if row.get("selected") is False
                ]
                if sampling_spec.has_contact_conditioning
                else None
            )
            audit = audit_sampling(
                plan.candidate_rows,
                plan.latent_design,
                quality_gates=sampling_spec.data["quality_gates"],
                registry_document=plan.split_registry.to_dict(),
            )
            _write_json_atomic(root / "audits/sampling_quality.json", audit)
            round_manifest = {"schema_version": "2.1.0", "round": 0, "phase": "reference_pilot_frozen", "candidate_count": len(plan.candidate_rows), "selected_count": len(plan.cases), "labels_observed_at_freeze": False, "sampling_spec_hash": sampling_spec.hash, "label_spec_hash": label_spec.hash, "split_registry_hash": plan.split_registry.definition_hash, "candidate_ledger_hash": file_hash(root / "rounds/round-000/candidates.parquet"), "candidate_ledger_format": "parquet_zstd", "formal_cases_hash": file_hash(root / _REFERENCE_FORMAL_CASES_LEDGER), "formal_cases_format": "jsonl_gzip"}
            if sampling_spec.has_contact_conditioning:
                conditioning = sampling_spec.contact_conditioning
                round_manifest["contact_conditioning"] = {
                    "contract_hash": conditioning["contract_hash"],
                    "selection_declaration_hash": conditioning[
                        "selection_declaration_hash"
                    ],
                    "attempted_count": len(plan.candidate_rows),
                    "accepted_count": len(plan.cases),
                    "input_refused_count": len(
                        contact_input_refused_candidate_ids
                    ),
                    "input_refused_candidate_ids_hash": content_hash(
                        contact_input_refused_candidate_ids
                    ),
                }
            _write_json_atomic(root / "rounds/round-000/round_manifest.json", round_manifest)
            files = _PLAN_FILES | {item["path"] for item in snapshot_metadata["snapshot_files"]}
            checkpoint = {"schema_version": SAMPLING_CHECKPOINT_VERSION, "storage_layout": {"reference_candidate_ledger": "parquet_zstd_only", "formal_case_ledger": "jsonl_gzip", "machine_json": "compact"}, "status": "initialized", "created_at": datetime.now(timezone.utc).isoformat(), "identity": identity, "split_registry_hash": plan.split_registry.definition_hash, "rounds_committed": [0], "formal_input_identity": formal_identity, "formal_dataset": None, "archive_summary": None, "committed_files": [{"path": relative, "content_hash": file_hash(root / relative)} for relative in sorted(files)]}
            _write_json_atomic(root / "sampling_checkpoint.json", checkpoint)
            marker.unlink()
            return checkpoint, plan


def _archive_receipt_inventory(
    root: Path, archive_receipt: "VerifiedArchive"
) -> list[dict]:
    """Render the legacy completion inventory from one verified receipt.

    The inventory shape and ordering are part of the sampling-checkpoint byte
    contract.  Hashes for manifests and partitions come from the immutable
    receipt/checkpoint that already bound those files; only ``checkpoint.json``
    itself needs a direct digest because it is not a root-manifest partition.
    """

    checkpoint = archive_receipt.checkpoint
    if (
        checkpoint is None
        or archive_receipt.manifest_hash is None
        or archive_receipt.checkpoint_hash is None
    ):
        raise RuntimeError("sharded archive receipt is incomplete")
    archive = archive_receipt.directory
    file_hashes: dict[Path, str] = {
        archive / "manifest.json": archive_receipt.manifest_hash,
        archive / "checkpoint.json": archive_receipt.checkpoint_hash,
    }
    for record in checkpoint["completed_shards"]:
        file_hashes[archive / record["path"] / "manifest.json"] = record[
            "manifest_hash"
        ]
    for relative, digest in archive_receipt.committed_files:
        file_hashes[archive / relative] = digest
    return [
        {
            "path": path.resolve().relative_to(root.resolve()).as_posix(),
            "content_hash": file_hashes[path],
        }
        for path in sorted(file_hashes)
    ]


def _verify_formal_archive(root: Path, checkpoint: dict, plan: ReferencePlan, manifest_path: Path):
    from .archive_validation import verify_sharded_archive
    from .sampling_validation import validate_registry_assignments, input_sampling_rows_from_arrays
    from .quality import validate_label_admission

    directory = manifest_path.parent
    if manifest_path.name != "manifest.json":
        raise InputValidationError("formal dataset path does not name its published manifest")
    archive_receipt = verify_sharded_archive(directory, validate_tensors=True)
    manifest = archive_receipt.manifest
    data_checkpoint = archive_receipt.checkpoint
    assert manifest is not None and data_checkpoint is not None
    expected = checkpoint["identity"]
    source, engine = manifest.get("source_contracts", {}), manifest.get("engine", {})
    checks = {
        "target_hash": source.get("target_hashes") == [expected["target_hash"]],
        "physics_registry_hash": engine.get("configuration_hash") == expected["physics_registry_hash"],
        "engine_code_hash": engine.get("code_hash") == expected["engine_code_hash"],
        "damage_tree_definition_hash": source.get("damage_tree_definition_hash") == expected["damage_tree_definition_hash"],
        "sampling_spec_hash": source.get("sampling_design_hash") == expected["sampling_spec_hash"],
        "label_spec_hash": source.get("label_spec_hash") == expected["label_spec_hash"],
        "split_registry_hash": source.get("split_registry_hash") == checkpoint["split_registry_hash"],
    }
    mismatches = [key for key, matches in checks.items() if not matches]
    if mismatches:
        raise InputValidationError("formal dataset identity differs from the frozen sampling plan", context={"mismatched_fields": mismatches})
    data_identity = data_checkpoint["identity"]
    formal_identity = checkpoint["formal_input_identity"]
    if data_identity.get("input_snapshot") != formal_identity or data_identity.get("environment") != expected["environment"]:
        raise InputValidationError("formal execution input/environment differs from the frozen sampling plan")
    if any(manifest.get("generation", {}).get("input_snapshot", {}).get(key) != value for key, value in formal_identity.items()):
        raise InputValidationError("formal manifest snapshot differs from frozen ordered inputs")
    if manifest.get("generation", {}).get("environment") != expected["environment"]:
        raise InputValidationError("formal manifest execution environment differs from sampling plan")
    expected_hashes, _ = _sequence_identity(plan.cases)
    expected_ids = [case["case_id"] for case in plan.cases]
    observed_ids = [identifier for record in data_checkpoint["completed_shards"] for identifier in record["case_ids"]]
    observed_hashes = [digest for record in data_checkpoint["completed_shards"] for digest in record["case_hashes"]]
    expected_candidates = [case["metadata"]["sampling"]["candidate_id"] for case in plan.cases]
    observed_candidates = [identifier for record in data_checkpoint["completed_shards"] for identifier in record.get("candidate_ids", [])]
    if observed_ids != expected_ids or observed_hashes != expected_hashes or observed_candidates != expected_candidates:
        raise InputValidationError("formal committed input cases do not match selected plan cases")
    cases_by_id = {case["case_id"]: case for case in plan.cases}
    hashes_by_id = dict(zip(expected_ids, expected_hashes))
    success = 0
    refusal = 0
    successful_candidates = set()
    executed_candidates = set()
    refused_candidates = set()
    purposes = ("software_development", "research_training", "research_validation", "research_test")
    admitted_candidates = {purpose: set() for purpose in purposes}
    for record, child in zip(
        data_checkpoint["completed_shards"],
        archive_receipt.shard_manifests,
        strict=True,
    ):
        shard = directory / record["path"]
        with np.load(shard / "tensors.npz", allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        input_rows = input_sampling_rows_from_arrays(arrays)
        validate_registry_assignments(plan.split_registry, input_rows)
        executed_candidates.update(row["candidate_id"] for row in input_rows)
        for row in input_rows:
            expected_case = cases_by_id.get(row["case_id"])
            envelope = expected_case.get("metadata", {}).get("sampling", {}) if expected_case else {}
            if any(row.get(key) != envelope.get("frozen_split" if key == "split" else key) for key in ("candidate_id", "lineage_id", "role", "epistemic_group_id", "split")):
                raise InputValidationError("formal full-input candidate/registry assignments differ from frozen selected cases")
        rows = []
        for index, case_id in enumerate(arrays["case_ids"].astype(str)):
            case = cases_by_id.get(case_id)
            envelope = case.get("metadata", {}).get("sampling", {}) if case else {}
            candidate = str(arrays["sampling_candidate_ids"][index])
            if candidate != envelope.get("candidate_id") or str(arrays["case_hashes"][index]) != hashes_by_id.get(case_id) or candidate in successful_candidates:
                raise InputValidationError("formal candidate IDs/hashes differ from frozen selected cases")
            successful_candidates.add(candidate)
            rows.append({"lineage_id": str(arrays["sampling_lineage_ids"][index]), "split": str(arrays["split_names"][index]), "role": str(arrays["sampling_role_names"][arrays["sampling_role_index"][index]]), "epistemic_group_id": envelope.get("epistemic_group_id")})
        validate_registry_assignments(plan.split_registry, rows)
        results = load_result_ledger(shard, child)
        if [item.get("case_id") for item in results] != record["case_ids"] or [item.get("provenance", {}).get("case_hash") for item in results] != record["case_hashes"]:
            raise InputValidationError("formal result records differ from committed selected inputs")
        statuses = [item.get("status") for item in results]
        if any(status not in {"ok", "refused"} for status in statuses) or statuses != arrays["input_case_status"].astype(str).tolist():
            raise InputValidationError("formal result statuses differ from tensor input mapping")
        success += statuses.count("ok")
        refusal += statuses.count("refused")
        refused_candidates.update(row["candidate_id"] for row, status in zip(input_rows, statuses) if status == "refused")
        shard_successful = {row["candidate_id"] for row, status in zip(input_rows, statuses) if status == "ok"}
        child_counts = child["quality_summary"].get("purpose_admission_counts", {})
        if set(child_counts) != set(purposes):
            raise InputValidationError("formal shard purpose counts are incomplete")
        for purpose in purposes:
            count = child_counts[purpose]
            if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= len(shard_successful):
                raise InputValidationError("formal shard admission count exceeds successful inputs")
            expected_count = len(shard_successful) if purpose in child["label_admission"]["approved_uses"] else 0
            if count != expected_count:
                raise InputValidationError("formal shard admission counts contradict approved uses")
            if count:
                validate_label_admission(child, purpose, evidence_root=shard)
                admitted_candidates[purpose].update(shard_successful)
    quality = manifest["quality_summary"]
    if quality.get("n_total") != len(plan.cases) or quality.get("n_valid") != success or quality.get("n_refused") != refusal or success != len(successful_candidates):
        raise InputValidationError("formal archive counts differ from complete input/results sets")
    purpose_counts = quality.get("purpose_admission_counts", {})
    if set(purpose_counts) != set(purposes) or any(purpose_counts[purpose] != len(admitted_candidates[purpose]) for purpose in purposes):
        raise InputValidationError("formal root purpose counts differ from actual shard admission sets")
    for purpose in purposes:
        count = purpose_counts[purpose]
        if count != (success if purpose in manifest["label_admission"]["approved_uses"] else 0):
            raise InputValidationError("formal root admission counts contradict approved uses")
        if count:
            validate_label_admission(manifest, purpose, evidence_root=directory)
    audit = _load_json(root / "audits/sampling_quality.json")
    def described_set(values):
        return {"status": "verified_archive_set", "count": len(values), "candidate_ids": sorted(values)}
    candidate_sets = {"candidate_frame": described_set({row["candidate_id"] for row in plan.candidate_rows}), "selected": described_set(set(expected_candidates)), "executed": described_set(executed_candidates), "successful": described_set(successful_candidates), "refused": described_set(refused_candidates), "admitted_by_purpose": {purpose: described_set(values) for purpose, values in admitted_candidates.items()}}
    summary = {"dataset_id": manifest["dataset_id"], "planned_candidate_count": len(plan.candidate_rows), "selected_candidate_count": len(plan.cases), "formal_input_count": len(plan.cases), "executed_candidate_count": len(executed_candidates), "execution_success_count": success, "refusal_count": refusal, "successful_candidate_count": len(successful_candidates), "purpose_admission_counts": purpose_counts, "candidate_sets": candidate_sets, "planned_design_quality_passed": audit.get("quality_gate_passed") is True, "realized_sampling_quality_status": "not_evaluated", "scientific_validation_status": "not_established_by_archive_completion"}
    inventory = _archive_receipt_inventory(root, archive_receipt)
    return summary, inventory


def mark_pilot_complete(output_directory: str | Path, formal_manifest_path: str | Path) -> dict:
    root = Path(output_directory)
    with sampling_run_transaction(root):
        checkpoint = verify_sampling_checkpoint(root)
        path = Path(formal_manifest_path).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise InputValidationError("formal dataset manifest must be an existing file inside the sampling run")
        plan = _load_existing_plan(root)
        summary, inventory = _verify_formal_archive(root, checkpoint, plan, path)
        checkpoint["status"] = "archive_complete"
        checkpoint["completed_at"] = datetime.now(timezone.utc).isoformat()
        checkpoint["archive_summary"] = summary
        checkpoint["formal_dataset"] = {"manifest_path": path.relative_to(root.resolve()).as_posix(), "manifest_hash": file_hash(path), "committed_files": inventory}
        _write_json_atomic(root / "sampling_checkpoint.json", checkpoint)
        return checkpoint


def _multiround_checkpoint_core(checkpoint: dict) -> dict:
    return {key: value for key, value in checkpoint.items() if key != "checkpoint_hash"}


def _write_multiround_checkpoint(root: Path, checkpoint: dict) -> dict:
    value = deepcopy(checkpoint)
    value["checkpoint_hash"] = content_hash(_multiround_checkpoint_core(value))
    _write_json_atomic(root / MULTIROUND_CHECKPOINT_FILE, value)
    return value


def _verify_reference_anchor(root: Path, state: dict) -> tuple[dict, ReferencePlan]:
    """Verify the immutable v2 reference run without assuming it is the only round."""
    base_path = root / "sampling_checkpoint.json"
    if not base_path.is_file() or file_hash(base_path) != state.get("reference_checkpoint_hash"):
        raise InputValidationError("multiround reference checkpoint is missing or changed")
    base = _load_json(base_path)
    if base.get("schema_version") not in SUPPORTED_SAMPLING_CHECKPOINT_VERSIONS:
        raise InputValidationError("multiround state requires an explicitly verified v2 reference checkpoint")
    if base.get("status") != "archive_complete" or base.get("rounds_committed") != [0]:
        raise InputValidationError("adaptive rounds require a complete immutable reference round")
    identity = base.get("identity")
    if (
        not isinstance(identity, dict)
        or identity.get("identity_hash")
        != content_hash({key: value for key, value in identity.items() if key != "identity_hash"})
        or identity.get("identity_hash") != state.get("run_identity_hash")
    ):
        raise InputValidationError("multiround run identity differs from the reference checkpoint")
    snapshot = _load_json(root / "plan/input_snapshot.json")
    snapshots = snapshot.get("snapshot_files", [])
    if not isinstance(snapshots, list):
        raise InputValidationError("reference input snapshot inventory is malformed")
    _verify_inventory(
        root,
        base.get("committed_files"),
        _reference_plan_files(root)
        | {record.get("path") for record in snapshots if isinstance(record, dict)},
    )
    plan = _verify_plan_semantics(root, base)
    formal = base.get("formal_dataset")
    if not isinstance(formal, dict) or not {"manifest_path", "manifest_hash", "committed_files"}.issubset(formal):
        raise InputValidationError("reference round has no complete formal archive identity")
    _verify_inventory(root, formal["committed_files"])
    manifest_path = _checked_path(root, formal["manifest_path"])
    if file_hash(manifest_path) != formal["manifest_hash"]:
        raise InputValidationError("reference formal manifest is missing or changed")
    summary, records = _verify_formal_archive(root, base, plan, manifest_path)
    if summary != base.get("archive_summary") or records != formal["committed_files"]:
        raise InputValidationError("reference archive no longer matches its completion record")
    return base, plan


def _verify_adaptive_proposal(
    proposal: Any, *, expected_round: int | None = None
) -> tuple[dict, list[dict], list[dict]]:
    """Validate a frozen proposal without executing an engine or fitting a model."""
    from .adaptive_sampling import (
        defensive_mixture_density,
        exact_design_weights,
        inclusion_probabilities,
        select_bernoulli,
        validate_train_only_acquisition,
    )

    if not isinstance(proposal, dict) or set(proposal) != {
        "manifest", "rows", "candidate_cases"
    }:
        raise InputValidationError(
            "adaptive proposal must contain exactly manifest, rows and candidate_cases"
        )
    manifest = proposal["manifest"]
    rows = proposal["rows"]
    candidate_cases = proposal["candidate_cases"]
    if (
        not isinstance(manifest, dict)
        or not isinstance(rows, list)
        or not rows
        or not isinstance(candidate_cases, list)
    ):
        raise InputValidationError("adaptive proposal manifest and candidate ledger are incomplete")
    if manifest.get("schema_version") != "adaptive-proposal@1.1.0":
        raise InputValidationError("unsupported adaptive proposal schema version")
    proposal_hash = manifest.get("proposal_hash")
    if proposal_hash != content_hash({key: value for key, value in manifest.items() if key != "proposal_hash"}):
        raise InputValidationError("adaptive proposal hash is invalid")
    round_index = manifest.get("round")
    if (
        not isinstance(round_index, int)
        or isinstance(round_index, bool)
        or round_index < 1
        or (expected_round is not None and round_index != expected_round)
    ):
        raise InputValidationError("adaptive proposal round is invalid")
    if manifest.get("candidate_ledger_hash") != content_hash(rows):
        raise InputValidationError("adaptive candidate ledger differs from its proposal identity")
    if manifest.get("candidate_count") != len(rows):
        raise InputValidationError("adaptive proposal candidate count is inconsistent")
    if manifest.get("stream") != "adaptive_train" or manifest.get("role") != "adaptive_train" or manifest.get("split") != "train":
        raise InputValidationError("reference and stress flows must remain isolated from adaptive Train")
    lineage = manifest.get("round_lineage_id")
    parent = manifest.get("parent_round_lineage_id")
    if not isinstance(lineage, str) or not lineage or not isinstance(parent, str) or not parent:
        raise InputValidationError("adaptive proposal round lineage is incomplete")
    model_hash = manifest.get("acquisition_model_hash")
    if not is_sha256_identity(model_hash, acceptance="legacy_python_hex"):
        raise InputValidationError("adaptive proposal acquisition model hash is invalid")
    for name in ("p0_hash", "adaptive_proposal_hash"):
        value = manifest.get(name)
        if not is_sha256_identity(value, acceptance="legacy_python_hex"):
            raise InputValidationError(f"adaptive proposal {name} is invalid")
    history = manifest.get("acquisition_history")
    if (
        not isinstance(history, dict)
        or history.get("policy") != "observed_successful_formal_train_only"
        or not isinstance(history.get("record_count"), int)
        or history.get("record_count", 0) < 1
        or not isinstance(history.get("history_hash"), str)
    ):
        raise InputValidationError("adaptive proposal lacks a Train-only acquisition history identity")
    if history != validate_train_only_acquisition(history.get("records", [])):
        raise InputValidationError("adaptive proposal acquisition history is not a complete Train-only snapshot")
    random_identity = manifest.get("selection_random_identity")
    if not isinstance(random_identity, dict):
        raise InputValidationError("adaptive proposal lacks a selection random identity")
    random_core = {key: value for key, value in random_identity.items() if key != "identity_hash"}
    if random_identity.get("identity_hash") != content_hash(random_core):
        raise InputValidationError("adaptive proposal selection random identity is invalid")
    required_random = {
        "algorithm", "seed", "candidate_order_hash", "inclusion_probability_hash", "random_draw_hash"
    }
    if set(random_core) != required_random:
        raise InputValidationError("adaptive proposal selection random identity fields are incomplete")
    if (
        random_identity.get("algorithm") != "numpy.default_rng-pcg64-random-v1"
        or not isinstance(random_identity.get("seed"), int)
        or isinstance(random_identity.get("seed"), bool)
        or random_identity.get("seed") < 0
    ):
        raise InputValidationError("adaptive proposal selection algorithm or seed is invalid")
    candidate_ids = [row.get("candidate_id") if isinstance(row, dict) else None for row in rows]
    if (
        any(not isinstance(value, str) or not value for value in candidate_ids)
        or len(set(candidate_ids)) != len(candidate_ids)
        or random_identity["candidate_order_hash"] != content_hash(candidate_ids)
    ):
        raise InputValidationError("adaptive proposal candidate order identity is invalid")
    binding_status = manifest.get("candidate_binding_status")
    if binding_status == "content_bound":
        if len(candidate_cases) != len(rows):
            raise InputValidationError(
                "adaptive candidate case ledger is not aligned with the proposal rows"
            )
        case_hashes: list[str] = []
        learning_hashes: list[str] = []
        case_ids: set[str] = set()
        for index, (row, case) in enumerate(zip(rows, candidate_cases)):
            if not isinstance(case, dict):
                raise InputValidationError("adaptive candidate case ledger contains a non-object")
            case_id = case.get("case_id")
            if not isinstance(case_id, str) or not case_id or case_id in case_ids:
                raise InputValidationError("adaptive candidate case IDs must be unique nonempty strings")
            metadata = case.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise InputValidationError("adaptive candidate case metadata must be an object")
            if isinstance(metadata, dict) and {
                "sampling", "frozen_split", "split_registry_hash"
            }.intersection(metadata):
                raise InputValidationError(
                    "adaptive candidate case ledger contains reserved sampling metadata",
                    context={"candidate_id": candidate_ids[index]},
                )
            analysis = case.get("analysis")
            formal_seed = analysis.get("aleatory_seed") if isinstance(analysis, dict) else None
            if (
                not isinstance(formal_seed, int)
                or isinstance(formal_seed, bool)
                or formal_seed < 0
            ):
                raise InputValidationError(
                    "adaptive candidate case lacks a valid formal seed",
                    context={"candidate_id": candidate_ids[index]},
                )
            case_hash = case_identity_hash(case)
            learning_hash = learning_input_fingerprint(case)
            if (
                row.get("candidate_case_hash") != case_hash
                or row.get("learning_input_hash") != learning_hash
            ):
                raise InputValidationError(
                    "adaptive candidate case content differs from its frozen row identity",
                    context={"candidate_id": candidate_ids[index]},
                )
            case_ids.add(case_id)
            case_hashes.append(case_hash)
            learning_hashes.append(learning_hash)
        if len(set(learning_hashes)) != len(learning_hashes):
            raise InputValidationError(
                "adaptive candidate case ledger contains duplicate learning inputs"
            )
        if (
            manifest.get("candidate_case_order_hash") != content_hash(case_hashes)
            or manifest.get("candidate_case_ledger_hash") != content_hash(case_hashes)
            or manifest.get("learning_input_order_hash") != content_hash(learning_hashes)
        ):
            raise InputValidationError("adaptive candidate case ledger identity is invalid")
        selected_bindings = [
            {
                "candidate_id": row["candidate_id"],
                "candidate_case_hash": row["candidate_case_hash"],
                "learning_input_hash": row["learning_input_hash"],
                "inclusion_probability": row["inclusion_probability"],
                "design_weight_raw": row["design_weight_raw"],
            }
            for row in rows
            if row["selected"]
        ]
        if (
            not selected_bindings
            or manifest.get("selected_case_set_hash") != content_hash(selected_bindings)
        ):
            raise InputValidationError("adaptive selected case-set identity is invalid")
    elif binding_status == "unbound_abstract":
        if candidate_cases or any(
            row.get("candidate_case_hash") is not None
            or row.get("learning_input_hash") is not None
            for row in rows
        ):
            raise InputValidationError("unbound abstract proposals cannot carry candidate cases")
        if any(
            manifest.get(name) is not None
            for name in (
                "candidate_case_order_hash", "candidate_case_ledger_hash",
                "learning_input_order_hash", "selected_case_set_hash",
            )
        ):
            raise InputValidationError("unbound abstract proposal case identities must be null")
    else:
        raise InputValidationError("adaptive proposal candidate binding status is invalid")
    try:
        rho = np.asarray([row["inclusion_probability"] for row in rows], dtype=np.float64)
        draws = np.asarray([row["random_draw"] for row in rows], dtype=np.float64)
        replay_rho = inclusion_probabilities(
            [row["score_total"] for row in rows],
            expected_count=manifest["expected_selected_count"],
            rho_min=manifest["rho_min"],
        )
        replay_selected, replay_draws = select_bernoulli(
            replay_rho, seed=random_identity["seed"]
        )
        expected_proposal_density = defensive_mixture_density(
            [row["p0_density"] for row in rows],
            [row["adaptive_density"] for row in rows],
            reference_fraction=manifest["defensive_reference_fraction"],
        )
        observed_proposal_density = np.asarray(
            [row["proposal_density"] for row in rows], dtype=np.float64
        )
        expected_weights = exact_design_weights(
            [row["p0_density"] for row in rows],
            [row["proposal_density"] for row in rows],
            rho,
            p0_hierarchy_probability=[row["p0_hierarchy_probability"] for row in rows],
            proposal_hierarchy_probability=[row["proposal_hierarchy_probability"] for row in rows],
        )
        observed_weights = np.asarray([row["design_weight_raw"] for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise InputValidationError("adaptive proposal probability fields are malformed") from exc
    if (
        not np.all(np.isfinite(draws))
        or np.any(draws < 0.0)
        or np.any(draws >= 1.0)
        or random_identity["inclusion_probability_hash"] != content_hash(rho.tolist())
        or random_identity["random_draw_hash"] != content_hash(draws.tolist())
        or not np.array_equal(rho, replay_rho)
        or not np.array_equal(draws, replay_draws)
        or not np.allclose(observed_proposal_density, expected_proposal_density, rtol=1.0e-12, atol=1.0e-12)
        or not np.allclose(observed_weights, expected_weights, rtol=1.0e-12, atol=1.0e-12)
    ):
        raise InputValidationError("adaptive proposal random draws or exact design weights are invalid")
    selected = []
    for row, probability, draw in zip(rows, rho, draws):
        if (
            row.get("round") != round_index
            or row.get("round_lineage_id") != lineage
            or row.get("parent_round_lineage_id") != parent
            or row.get("split") != "train"
            or row.get("role") != "adaptive_train"
            or row.get("stream") != "adaptive_train"
            or row.get("acquisition_model_hash") != model_hash
            or row.get("selection_seed") != random_identity["seed"]
            or type(row.get("selected")) is not bool
            or type(row.get("weight_mask")) is not bool
            or row.get("selected") != bool(draw < probability)
            or row.get("selected") != bool(replay_selected[len(selected)])
            or row.get("p0_id") != manifest.get("p0_id")
            or row.get("p0_hash") != manifest.get("p0_hash")
        ):
            raise InputValidationError("adaptive proposal candidate identity or selection replay is inconsistent")
        for score_name in ("score_rarity", "score_uncertainty", "score_coverage", "score_total"):
            score = row.get(score_name)
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)) or score < 0:
                raise InputValidationError("adaptive proposal score components must be finite nonnegative values")
        selected.append(row["selected"])
    if manifest.get("realized_selected_count") != sum(selected):
        raise InputValidationError("adaptive proposal selected count is inconsistent")
    if (
        manifest.get("candidate_order_hash") != content_hash(candidate_ids)
        or manifest.get("score_policy")
        != "externally_supplied_components_and_selection_score_no_domain_rule_inferred"
        or manifest.get("design_weight_formula")
        != "p0_density*p0_hierarchy_probability/(proposal_density*proposal_hierarchy_probability*inclusion_probability)"
    ):
        raise InputValidationError("adaptive proposal audit policy identities are inconsistent")
    return manifest, rows, candidate_cases


_ADAPTIVE_DESIGN_WEIGHT_FORMULA = (
    "p0_density*p0_hierarchy_probability/"
    "(proposal_density*proposal_hierarchy_probability*inclusion_probability)"
)


def _build_selected_adaptive_cases(
    root: Path,
    manifest: dict,
    rows: list[dict],
    candidate_cases: list[dict],
    reference_plan: ReferencePlan,
) -> tuple[list[dict], list[dict], SplitRegistry]:
    """Attach the frozen adaptive envelope without changing candidate physics."""

    sampling_spec = SamplingSpec.from_file(root / "plan/sampling_spec.json")
    label_spec = LabelSpec.from_file(root / "plan/label_spec.json")
    if sampling_spec.data.get("schema_version") not in {"1.1.0", "1.2.0"}:
        raise InputValidationError(
            "adaptive executable rounds require SamplingSpec 1.1.0 or 1.2.0"
        )
    if not reference_plan.cases:
        raise InputValidationError("adaptive execution requires a frozen reference plan")
    reference_envelope = reference_plan.cases[0].get("metadata", {}).get("sampling", {})
    if reference_envelope.get("schema_version") != "1.1.0":
        raise InputValidationError(
            "adaptive execution requires a reference envelope with the 1.1 design contract"
        )
    rows_by_id = {row["candidate_id"]: row for row in rows}
    case_by_id = dict(zip((row["candidate_id"] for row in rows), candidate_cases))
    selected_rows = [row for row in rows if row["selected"]]
    registry = SplitRegistry(
        "adaptive-round-"
        + str(manifest["round"])
        + "-"
        + manifest["proposal_hash"].split(":", 1)[1][:16]
    )
    for row in selected_rows:
        registry.register(
            lineage_id=row["lineage_id"],
            split="train",
            role="adaptive_train",
            epistemic_group_id=row["epistemic_group_id"],
        )
    registry_hash = registry.freeze()
    # Reuse one immutable snapshot while cases are in memory.  Reconstructing
    # the complete assignment array inside the selected-case loop is O(N^2).
    registry_document = registry.definition_document()
    role_names = list(reference_envelope["role_names"])
    if "adaptive_train" not in role_names:
        role_names.append("adaptive_train")
    stratum_names = list(dict.fromkeys(row["stratum"] for row in selected_rows))
    selected_cases: list[dict] = []
    selected_index: list[dict] = []
    for row in selected_rows:
        candidate = case_by_id[row["candidate_id"]]
        if rows_by_id[row["candidate_id"]] is not row:
            raise InputValidationError("adaptive candidate row identity is ambiguous")
        case = deepcopy(candidate)
        metadata = case.setdefault("metadata", {})
        metadata["split_group"] = row["lineage_id"]
        case["analysis"]["epistemic_sample_id"] = row["epistemic_group_id"]
        formal_seed = case["analysis"]["aleatory_seed"]
        envelope = {
            "schema_version": "1.2.0",
            "candidate_id": row["candidate_id"],
            "lineage_id": row["lineage_id"],
            "design_group_id": row["lineage_id"],
            "round": row["round"],
            "stream": "adaptive_train",
            "role": "adaptive_train",
            "stratum": row["stratum"],
            "role_names": role_names,
            "stratum_names": stratum_names,
            "selection_probability": row["inclusion_probability"],
            "design_weight_raw": row["design_weight_raw"],
            "weight_mask": True,
            "sampling_spec_id": sampling_spec.spec_id,
            "sampling_spec_hash": sampling_spec.hash,
            "sampling_spec_document": sampling_spec.data,
            "training_design_document": sampling_spec.training_design_document,
            "training_design_hash": sampling_spec.training_design_hash,
            "p0_id": manifest["p0_id"],
            "p0_hash": manifest["p0_hash"],
            "label_spec_id": label_spec.spec_id,
            "label_spec_hash": label_spec.hash,
            "label_spec_document": label_spec.data,
            "stratification_spec_id": reference_envelope["stratification_spec_id"],
            "stratification_spec_hash": reference_envelope["stratification_spec_hash"],
            "split_registry_id": registry.registry_id,
            "split_registry_hash": registry_hash,
            "split_registry_document": registry_document,
            "frozen_split": "train",
            "epistemic_group_id": row["epistemic_group_id"],
            "screen_seed": manifest["selection_random_identity"]["seed"],
            "formal_seed": formal_seed,
            "damage_tree_definition_hash": reference_envelope[
                "damage_tree_definition_hash"
            ],
            "target_hash": reference_envelope["target_hash"],
            "physics_registry_hash": reference_envelope["physics_registry_hash"],
            "engine_code_hash": reference_envelope["engine_code_hash"],
            "p0_density": row["p0_density"],
            "adaptive_density": row["adaptive_density"],
            "proposal_density": row["proposal_density"],
            "p0_hierarchy_probability": row["p0_hierarchy_probability"],
            "proposal_hierarchy_probability": row[
                "proposal_hierarchy_probability"
            ],
            "acquisition_model_hash": manifest["acquisition_model_hash"],
            "adaptive_proposal_hash": manifest["adaptive_proposal_hash"],
            "proposal_hash": manifest["proposal_hash"],
            "round_lineage_id": manifest["round_lineage_id"],
            "parent_round_lineage_id": manifest["parent_round_lineage_id"],
            "candidate_case_hash": row["candidate_case_hash"],
            "learning_input_hash": row["learning_input_hash"],
            "selected_case_set_hash": manifest["selected_case_set_hash"],
            "design_weight_formula": _ADAPTIVE_DESIGN_WEIGHT_FORMULA,
        }
        metadata["sampling"] = envelope
        metadata["frozen_split"] = "train"
        metadata["split_registry_hash"] = registry_hash
        formal_hash = case_identity_hash(case)
        selected_cases.append(case)
        selected_index.append(
            {
                "candidate_id": row["candidate_id"],
                "case_id": case["case_id"],
                "candidate_case_hash": row["candidate_case_hash"],
                "learning_input_hash": row["learning_input_hash"],
                "formal_case_hash": formal_hash,
                "proposal_hash": manifest["proposal_hash"],
                "acquisition_model_hash": manifest["acquisition_model_hash"],
                "round": manifest["round"],
                "round_lineage_id": manifest["round_lineage_id"],
                "selected_case_set_hash": manifest["selected_case_set_hash"],
            }
        )
    return selected_cases, selected_index, registry


def _verify_selected_adaptive_cases(
    directory: Path, manifest: dict, rows: list[dict]
) -> tuple[list[dict], list[dict]]:
    candidate_cases = _load_rows(directory / "candidate_cases.jsonl")
    selected_cases = _load_rows(directory / "selected_formal_cases.jsonl")
    selected_index = _load_rows(directory / "selected_case_index.jsonl")
    selected_rows = [row for row in rows if row["selected"]]
    if len(candidate_cases) != len(rows) or len(selected_cases) != len(selected_rows):
        raise InputValidationError("adaptive case ledgers have inconsistent row counts")
    for row, case in zip(rows, candidate_cases):
        if (
            case_identity_hash(case) != row["candidate_case_hash"]
            or learning_input_fingerprint(case) != row["learning_input_hash"]
        ):
            raise InputValidationError("adaptive candidate case ledger changed after proposal freeze")
    if len(selected_index) != len(selected_cases):
        raise InputValidationError("adaptive selected case index is incomplete")
    for row, case, index in zip(selected_rows, selected_cases, selected_index):
        envelope = case.get("metadata", {}).get("sampling", {})
        if (
            index.get("candidate_id") != row["candidate_id"]
            or index.get("case_id") != case.get("case_id")
            or index.get("formal_case_hash") != case_identity_hash(case)
            or index.get("candidate_case_hash") != row["candidate_case_hash"]
            or index.get("learning_input_hash") != learning_input_fingerprint(case)
            or index.get("proposal_hash") != manifest["proposal_hash"]
            or index.get("round_lineage_id") != manifest["round_lineage_id"]
            or envelope.get("schema_version") != "1.2.0"
            or envelope.get("candidate_id") != row["candidate_id"]
            or envelope.get("selected_case_set_hash")
            != manifest["selected_case_set_hash"]
        ):
            raise InputValidationError("adaptive selected formal case binding is invalid")
    return selected_cases, selected_index


def _verify_adaptive_formal_archive_files(
    root: Path,
    round_index: int,
    proposal_manifest: dict,
    proposal_rows: list[dict],
) -> tuple[dict, list[dict], list[dict]]:
    """Verify a sharded adaptive archive and derive outcome identities from bytes."""

    from .archive_validation import verify_sharded_archive
    from .sampling_validation import (
        input_sampling_rows_from_arrays,
        validate_registry_assignments,
    )

    directory = root / f"rounds/round-{round_index:03d}"
    selected_cases, selected_index = _verify_selected_adaptive_cases(
        directory, proposal_manifest, proposal_rows
    )
    selected_by_case = {case["case_id"]: case for case in selected_cases}
    index_by_case = {item["case_id"]: item for item in selected_index}
    adaptive_registry = SplitRegistry.from_file(
        directory / "adaptive_split_registry.json"
    )
    archive = directory / "formal_dataset"
    archive_receipt = verify_sharded_archive(archive, validate_tensors=True)
    archive_manifest = archive_receipt.manifest
    data_checkpoint = archive_receipt.checkpoint
    assert archive_manifest is not None and data_checkpoint is not None
    base = _load_json(root / "sampling_checkpoint.json")
    expected = base["identity"]
    source = archive_manifest.get("source_contracts", {})
    engine = archive_manifest.get("engine", {})
    sampling = archive_manifest.get("generation", {}).get("sampling", {})
    expected_sampling = {
        "sampling_envelope_version": "1.2.0",
        "proposal_hash": proposal_manifest["proposal_hash"],
        "adaptive_proposal_hash": proposal_manifest["adaptive_proposal_hash"],
        "acquisition_model_hash": proposal_manifest["acquisition_model_hash"],
        "round_lineage_id": proposal_manifest["round_lineage_id"],
        "parent_round_lineage_id": proposal_manifest["parent_round_lineage_id"],
        "selected_case_set_hash": proposal_manifest["selected_case_set_hash"],
        "raw_design_weight_semantics": "general_two_stage_reference_joint_design",
        "design_weight_formula": _ADAPTIVE_DESIGN_WEIGHT_FORMULA,
    }
    mismatches = []
    checks = {
        "target_hash": source.get("target_hashes") == [expected["target_hash"]],
        "physics_registry_hash": engine.get("configuration_hash")
        == expected["physics_registry_hash"],
        "engine_code_hash": engine.get("code_hash") == expected["engine_code_hash"],
        "damage_tree_definition_hash": source.get("damage_tree_definition_hash")
        == expected["damage_tree_definition_hash"],
        "sampling_spec_hash": source.get("sampling_design_hash")
        == expected["sampling_spec_hash"],
        "label_spec_hash": source.get("label_spec_hash")
        == expected["label_spec_hash"],
        "split_registry_hash": source.get("split_registry_hash")
        == adaptive_registry.definition_hash,
        **{
            f"sampling.{name}": sampling.get(name) == value
            for name, value in expected_sampling.items()
        },
    }
    mismatches.extend(name for name, matches in checks.items() if not matches)
    if mismatches:
        raise InputValidationError(
            "adaptive formal archive differs from the frozen proposal",
            context={"mismatched_fields": mismatches},
        )
    data_identity = data_checkpoint["identity"]
    input_snapshot = data_identity.get("input_snapshot", {})
    expected_hashes, expected_sequence_hash = _sequence_identity(selected_cases)
    if (
        input_snapshot.get("expected_case_count") != len(selected_cases)
        or input_snapshot.get("ordered_case_identity_hash") != expected_sequence_hash
    ):
        raise InputValidationError(
            "adaptive formal checkpoint input snapshot differs from selected cases"
        )
    expected_ids = [case["case_id"] for case in selected_cases]
    expected_candidates = [
        case["metadata"]["sampling"]["candidate_id"] for case in selected_cases
    ]
    observed_ids = [
        identifier
        for record in data_checkpoint["completed_shards"]
        for identifier in record["case_ids"]
    ]
    observed_hashes = [
        digest
        for record in data_checkpoint["completed_shards"]
        for digest in record["case_hashes"]
    ]
    observed_candidates = [
        identifier
        for record in data_checkpoint["completed_shards"]
        for identifier in record.get("candidate_ids", [])
    ]
    if (
        observed_ids != expected_ids
        or observed_hashes != expected_hashes
        or observed_candidates != expected_candidates
    ):
        raise InputValidationError(
            "adaptive archive inputs differ from the frozen selected-case ledger"
        )
    outcomes: list[dict] = []
    for record, child in zip(
        data_checkpoint["completed_shards"],
        archive_receipt.shard_manifests,
        strict=True,
    ):
        shard = archive / record["path"]
        with np.load(shard / "tensors.npz", allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        input_rows = input_sampling_rows_from_arrays(arrays)
        validate_registry_assignments(adaptive_registry, input_rows)
        results = load_result_ledger(shard, child)
        if len(results) != len(input_rows):
            raise InputValidationError("adaptive shard results and input identities differ")
        statuses = arrays["input_case_status"].astype(str).tolist()
        for result, input_row, status in zip(results, input_rows, statuses):
            case_id = input_row["case_id"]
            case = selected_by_case.get(case_id)
            index = index_by_case.get(case_id)
            if (
                case is None
                or index is None
                or result.get("case_id") != case_id
                or result.get("provenance", {}).get("case_hash")
                != index["formal_case_hash"]
                or input_row["candidate_id"] != index["candidate_id"]
                or status != result.get("status")
                or status not in {"ok", "refused"}
            ):
                raise InputValidationError(
                    "adaptive result record is not bound to its selected formal case"
                )
            outcomes.append(
                {
                    "candidate_id": index["candidate_id"],
                    "round": round_index,
                    "round_lineage_id": proposal_manifest["round_lineage_id"],
                    "split": "train",
                    "role": "adaptive_train",
                    "stream": "adaptive_train",
                    "formal_status": status,
                    "label_observed": status == "ok",
                    "formal_result_hash": content_hash(result),
                    "formal_case_hash": index["formal_case_hash"],
                    "dataset_id": archive_manifest["dataset_id"],
                    "archive_manifest_hash": archive_receipt.manifest_hash,
                }
            )
    order = {candidate: index for index, candidate in enumerate(expected_candidates)}
    outcomes.sort(key=lambda row: order[row["candidate_id"]])
    _verify_adaptive_outcomes(outcomes, proposal_rows, archive_bound=True)
    quality = archive_manifest.get("quality_summary", {})
    success = sum(row["formal_status"] == "ok" for row in outcomes)
    refusal = len(outcomes) - success
    if (
        quality.get("n_total") != len(selected_cases)
        or quality.get("n_valid") != success
        or quality.get("n_refused") != refusal
    ):
        raise InputValidationError("adaptive archive quality counts are inconsistent")
    summary = {
        "dataset_id": archive_manifest["dataset_id"],
        "manifest_hash": archive_receipt.manifest_hash,
        "selected_case_count": len(selected_cases),
        "success_count": success,
        "refusal_count": refusal,
        "outcome_ledger_hash": content_hash(outcomes),
        "proposal_hash": proposal_manifest["proposal_hash"],
        "selected_case_set_hash": proposal_manifest["selected_case_set_hash"],
        "scientific_validation_status": "not_established_by_archive_completion",
    }
    inventory = _archive_receipt_inventory(root, archive_receipt)
    return summary, inventory, outcomes


def _verify_adaptive_outcomes(
    rows: list[dict], proposal_rows: list[dict], *, archive_bound: bool = False
) -> dict:
    selected_ids = [row["candidate_id"] for row in proposal_rows if row["selected"]]
    observed_ids = [row.get("candidate_id") if isinstance(row, dict) else None for row in rows]
    if (
        any(not isinstance(value, str) or not value for value in observed_ids)
        or len(set(observed_ids)) != len(observed_ids)
        or set(observed_ids) != set(selected_ids)
        or len(observed_ids) != len(selected_ids)
    ):
        raise InputValidationError("adaptive outcome ledger must cover each selected candidate exactly once")
    proposal_by_id = {row["candidate_id"]: row for row in proposal_rows}
    success = 0
    refusal = 0
    for row in rows:
        proposal = proposal_by_id[row["candidate_id"]]
        status = row.get("formal_status")
        if (
            row.get("split") != "train"
            or row.get("role") != "adaptive_train"
            or row.get("stream") != "adaptive_train"
            or status not in {"ok", "refused"}
            or not is_sha256_identity(
                row.get("formal_result_hash"), acceptance="legacy_python_hex"
            )
            or row.get("round") != proposal.get("round")
            or row.get("round_lineage_id") != proposal.get("round_lineage_id")
        ):
            raise InputValidationError("adaptive outcomes differ from the frozen Train proposal")
        if archive_bound and (
            not is_sha256_identity(
                row.get("formal_case_hash"), acceptance="legacy_python_hex"
            )
            or not isinstance(row.get("dataset_id"), str)
            or not row["dataset_id"]
            or not is_sha256_identity(
                row.get("archive_manifest_hash"), acceptance="legacy_python_hex"
            )
        ):
            raise InputValidationError(
                "adaptive outcomes lack a verified formal archive binding"
            )
        if status == "ok" and row.get("label_observed") is not True:
            raise InputValidationError("successful adaptive outcomes must declare observed formal labels")
        if status == "refused" and row.get("label_observed") is not False:
            raise InputValidationError("refused adaptive outcomes must not declare observed labels")
        success += status == "ok"
        refusal += status == "refused"
    return {
        "outcome_count": len(rows),
        "success_count": success,
        "refusal_count": refusal,
        "outcome_ledger_hash": content_hash(rows),
    }


def _reference_acquisition_history(root: Path, base: dict, plan: ReferencePlan) -> list[dict]:
    """Recover successful Train result identities from the verified reference archive."""
    formal = base["formal_dataset"]
    formal_directory = _checked_path(root, formal["manifest_path"]).parent
    data_checkpoint = _load_json(formal_directory / "checkpoint.json")
    candidate_by_id = {row["candidate_id"]: row for row in plan.candidate_rows}
    records: list[dict] = []
    for shard_record in data_checkpoint.get("completed_shards", []):
        shard = formal_directory / shard_record["path"]
        results = load_result_ledger(shard, _load_json(shard / "manifest.json"))
        try:
            with np.load(shard / "tensors.npz", allow_pickle=False) as loaded:
                candidate_ids = loaded["input_sampling_candidate_ids"].astype(str).tolist()
                splits = loaded["input_frozen_splits"].astype(str).tolist()
                statuses = loaded["input_case_status"].astype(str).tolist()
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise InputValidationError("reference archive cannot supply acquisition history identities") from exc
        if not (len(results) == len(candidate_ids) == len(splits) == len(statuses)):
            raise InputValidationError("reference result and candidate axes are inconsistent")
        for result, candidate_id, split, status in zip(results, candidate_ids, splits, statuses):
            candidate = candidate_by_id.get(candidate_id)
            if status != "ok" or split != "train":
                continue
            if candidate is None or result.get("status") != "ok":
                raise InputValidationError("reference acquisition history differs from the frozen candidate plan")
            records.append({
                "candidate_id": candidate_id,
                "split": "train",
                "role": candidate["role"],
                "stream": candidate["stream"],
                "formal_status": "ok",
                "label_observed": True,
                "formal_result_hash": content_hash(result),
            })
    return records


def acquisition_history_records(output_directory: str | Path) -> list[dict]:
    """Return content-bound successful Train results eligible for acquisition."""
    root = Path(output_directory)
    state = verify_multiround_sampling_checkpoint(root)
    base, plan = _verify_reference_anchor(root, state)
    records = _reference_acquisition_history(root, base, plan)
    for entry in state["round_chain"][1:]:
        if (
            entry.get("phase") != "outcomes_committed"
            or entry.get("outcome_source") != "verified_formal_archive"
        ):
            continue
        directory = root / f"rounds/round-{entry['round']:03d}"
        rows = _load_rows_allow_empty(directory / "formal_outcomes.jsonl")
        records.extend({
            "candidate_id": row["candidate_id"],
            "split": "train",
            "role": "adaptive_train",
            "stream": "adaptive_train",
            "formal_status": "ok",
            "label_observed": True,
            "formal_result_hash": row["formal_result_hash"],
        } for row in rows if row["formal_status"] == "ok")
    # Reuse the public gate so this accessor can never leak Validation/Test or
    # stress records if an upstream archive changes shape.
    from .adaptive_sampling import validate_train_only_acquisition
    identity = validate_train_only_acquisition(records)
    return identity["records"]


def initialize_multiround_sampling_state(
    output_directory: str | Path, *, resume: bool = False
) -> dict:
    """Attach an auditable multiround sidecar to a complete v2 reference run.

    The immutable v2 round-0 artifacts are preserved byte-for-byte.  This is
    the explicit compatibility path; older checkpoints are never silently
    interpreted as multiround state.
    """
    root = Path(output_directory)
    with sampling_run_transaction(root):
        state_path = root / MULTIROUND_CHECKPOINT_FILE
        if state_path.exists():
            if not resume:
                raise InputValidationError("multiround checkpoint already exists; use resume for verification")
            return verify_multiround_sampling_checkpoint(root)
        base = verify_sampling_checkpoint(root)
        if base.get("status") != "archive_complete":
            raise InputValidationError("adaptive rounds require a completed reference archive")
        reference_manifest = root / "rounds/round-000/round_manifest.json"
        reference_manifest_hash = file_hash(reference_manifest)
        reference_lineage_id = "reference-round-" + reference_manifest_hash.split(":", 1)[1][:24]
        now = datetime.now(timezone.utc).isoformat()
        state = {
            "schema_version": MULTIROUND_CHECKPOINT_VERSION,
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "run_identity_hash": base["identity"]["identity_hash"],
            "reference_checkpoint_hash": file_hash(root / "sampling_checkpoint.json"),
            "reference_round_manifest_hash": reference_manifest_hash,
            "reference_round_lineage_id": reference_lineage_id,
            "flow_isolation": {
                "reference_round": 0,
                "adaptive_stream": "adaptive_train",
                "adaptive_split": "train",
                "stress_included": False,
            },
            "rounds_committed": [0],
            "round_chain": [
                {
                    "round": 0,
                    "round_lineage_id": reference_lineage_id,
                    "parent_round_lineage_id": None,
                    "stream": "reference",
                    "round_manifest_path": "rounds/round-000/round_manifest.json",
                    "round_manifest_hash": reference_manifest_hash,
                    "proposal_hash": None,
                    "acquisition_model_hash": None,
                    "selection_random_identity_hash": None,
                    "phase": "reference_complete",
                    "outcome_ledger_file_hash": None,
                }
            ],
        }
        return _write_multiround_checkpoint(root, state)


def verify_multiround_sampling_checkpoint(
    output_directory: str | Path, *, expected_run_identity_hash: str | None = None
) -> dict:
    root = Path(output_directory)
    state = _load_json(root / MULTIROUND_CHECKPOINT_FILE)
    if state.get("schema_version") != MULTIROUND_CHECKPOINT_VERSION:
        raise InputValidationError("unsupported multiround checkpoint; explicit migration is required")
    if state.get("checkpoint_hash") != content_hash(_multiround_checkpoint_core(state)):
        raise InputValidationError("multiround checkpoint hash is invalid")
    if state.get("status") not in {"active", "finalized"}:
        raise InputValidationError("multiround checkpoint status is invalid")
    if expected_run_identity_hash is not None and state.get("run_identity_hash") != expected_run_identity_hash:
        raise InputValidationError("multiround checkpoint differs from the requested run identity")
    _verify_reference_anchor(root, state)
    rounds = state.get("rounds_committed")
    chain = state.get("round_chain")
    if (
        not isinstance(rounds, list)
        or not rounds
        or rounds != list(range(len(rounds)))
        or not isinstance(chain, list)
        or [entry.get("round") if isinstance(entry, dict) else None for entry in chain] != rounds
    ):
        raise InputValidationError("multiround checkpoint round chain is not contiguous")
    expected_directories = {f"round-{index:03d}" for index in rounds}
    rounds_path = root / "rounds"
    actual_directories = {entry.name for entry in rounds_path.iterdir() if entry.is_dir()}
    if actual_directories != expected_directories:
        raise InputValidationError("disk sampling rounds differ from committed multiround state")
    reference = chain[0]
    if (
        reference.get("round_lineage_id") != state.get("reference_round_lineage_id")
        or reference.get("parent_round_lineage_id") is not None
        or reference.get("stream") != "reference"
        or reference.get("round_manifest_hash") != state.get("reference_round_manifest_hash")
        or reference.get("phase") != "reference_complete"
        or file_hash(_checked_path(root, reference.get("round_manifest_path")))
        != state.get("reference_round_manifest_hash")
    ):
        raise InputValidationError("multiround reference lineage anchor is invalid")
    seen_candidates = {
        row.get("candidate_id")
        for row in _load_reference_candidate_rows_compat(root)
    }
    seen_lineages = {state["reference_round_lineage_id"]}
    previous_lineage = state["reference_round_lineage_id"]
    for index, entry in enumerate(chain[1:], start=1):
        if not isinstance(entry, dict) or entry.get("stream") != "adaptive_train":
            raise InputValidationError("non-reference rounds must remain isolated adaptive Train rounds")
        directory = root / f"rounds/round-{index:03d}"
        if (directory / ".round.incomplete").exists() or (directory / ".outcomes.incomplete").exists():
            raise InputValidationError("adaptive round has an incomplete marker and requires operator review")
        proposal_manifest = _load_json(directory / "proposal.json")
        proposal = {
            "manifest": proposal_manifest,
            "rows": _load_rows(directory / "adaptive_candidates.jsonl"),
            "candidate_cases": (
                _load_rows(directory / "candidate_cases.jsonl")
                if proposal_manifest.get("candidate_binding_status") == "content_bound"
                else []
            ),
        }
        manifest, rows, candidate_cases = _verify_adaptive_proposal(
            proposal, expected_round=index
        )
        parquet_path = directory / "adaptive_candidates.parquet"
        parquet_rows = _load_verified_adaptive_candidates_parquet(parquet_path)
        if parquet_rows != rows:
            raise InputValidationError(
                "adaptive candidate JSONL and Parquet ledgers differ"
            )
        round_manifest = _load_json(directory / "round_manifest.json")
        required = {
            "schema_version": (
                "adaptive-round-manifest@1.1.0"
                if manifest["candidate_binding_status"] == "content_bound"
                else "adaptive-round-manifest@1.0.0"
            ),
            "round": index,
            "round_lineage_id": manifest["round_lineage_id"],
            "parent_round_lineage_id": previous_lineage,
            "stream": "adaptive_train",
            "proposal_hash": manifest["proposal_hash"],
            "acquisition_model_hash": manifest["acquisition_model_hash"],
            "selection_random_identity_hash": manifest["selection_random_identity"]["identity_hash"],
            "proposal_file_hash": file_hash(directory / "proposal.json"),
            "candidate_ledger_file_hash": file_hash(directory / "adaptive_candidates.jsonl"),
            "candidate_parquet_file_hash": file_hash(parquet_path),
            "candidate_count": len(rows),
            "selected_count": sum(row["selected"] for row in rows),
            "candidate_binding_status": manifest["candidate_binding_status"],
            "phase": entry.get("phase"),
        }
        if manifest["candidate_binding_status"] == "content_bound":
            selected_cases, selected_index = _verify_selected_adaptive_cases(
                directory, manifest, rows
            )
            registry_path = directory / "adaptive_split_registry.json"
            adaptive_registry = SplitRegistry.from_file(registry_path)
            required.update(
                {
                    "candidate_cases_file_hash": file_hash(
                        directory / "candidate_cases.jsonl"
                    ),
                    "selected_formal_cases_file_hash": file_hash(
                        directory / "selected_formal_cases.jsonl"
                    ),
                    "selected_case_index_file_hash": file_hash(
                        directory / "selected_case_index.jsonl"
                    ),
                    "selected_case_index_hash": content_hash(selected_index),
                    "selected_case_set_hash": manifest["selected_case_set_hash"],
                    "adaptive_split_registry_file_hash": file_hash(registry_path),
                    "adaptive_split_registry_hash": adaptive_registry.definition_hash,
                }
            )
        if any(round_manifest.get(key) != value for key, value in required.items()):
            raise InputValidationError("adaptive round manifest differs from its committed files")
        phase = entry.get("phase")
        outcome_path = directory / "formal_outcomes.jsonl"
        archive_summary = None
        archive_outcomes = None
        if (
            manifest["candidate_binding_status"] == "content_bound"
            and phase in {"archive_complete", "outcomes_committed"}
        ):
            archive_summary, archive_inventory, archive_outcomes = (
                _verify_adaptive_formal_archive_files(root, index, manifest, rows)
            )
            formal_archive = entry.get("formal_archive")
            if (
                not isinstance(formal_archive, dict)
                or formal_archive.get("summary") != archive_summary
                or formal_archive.get("committed_files") != archive_inventory
                or round_manifest.get("formal_archive") != formal_archive
            ):
                raise InputValidationError(
                    "adaptive formal archive binding differs from committed files"
                )
        if phase == "proposal_frozen":
            if outcome_path.exists() or entry.get("outcome_ledger_file_hash") is not None:
                raise InputValidationError("proposal-only adaptive round cannot claim formal outcomes")
        elif phase == "archive_complete":
            if outcome_path.exists() or entry.get("outcome_ledger_file_hash") is not None:
                raise InputValidationError(
                    "archive-complete adaptive round cannot claim committed outcomes"
                )
        elif phase == "outcomes_committed":
            if not outcome_path.is_file() or file_hash(outcome_path) != entry.get("outcome_ledger_file_hash"):
                raise InputValidationError("adaptive outcome ledger is missing or changed")
            outcomes = _load_rows_allow_empty(outcome_path)
            outcome_summary = _verify_adaptive_outcomes(
                outcomes,
                rows,
                archive_bound=manifest["candidate_binding_status"] == "content_bound",
            )
            if archive_outcomes is not None and outcomes != archive_outcomes:
                raise InputValidationError(
                    "adaptive outcome ledger differs from the verified formal archive"
                )
            if (
                round_manifest.get("outcome_ledger_file_hash") != file_hash(outcome_path)
                or round_manifest.get("outcome_ledger_hash") != outcome_summary["outcome_ledger_hash"]
                or round_manifest.get("outcome_count") != outcome_summary["outcome_count"]
                or round_manifest.get("success_count") != outcome_summary["success_count"]
                or round_manifest.get("refusal_count") != outcome_summary["refusal_count"]
            ):
                raise InputValidationError("adaptive outcome summary differs from the committed ledger")
            expected_source = (
                "verified_formal_archive"
                if manifest["candidate_binding_status"] == "content_bound"
                else "unverified_abstract_compatibility"
            )
            if (
                round_manifest.get("outcome_source") != expected_source
                or entry.get("outcome_source") != expected_source
            ):
                raise InputValidationError("adaptive outcome source identity is invalid")
        else:
            raise InputValidationError("adaptive round phase is invalid")
        if (
            entry.get("round_lineage_id") != manifest["round_lineage_id"]
            or entry.get("parent_round_lineage_id") != previous_lineage
            or entry.get("round_manifest_path") != f"rounds/round-{index:03d}/round_manifest.json"
            or entry.get("round_manifest_hash") != file_hash(directory / "round_manifest.json")
            or entry.get("proposal_hash") != manifest["proposal_hash"]
            or entry.get("acquisition_model_hash") != manifest["acquisition_model_hash"]
            or entry.get("selection_random_identity_hash")
            != manifest["selection_random_identity"]["identity_hash"]
            or entry.get("candidate_binding_status")
            != manifest["candidate_binding_status"]
        ):
            raise InputValidationError("multiround lineage entry differs from its adaptive round")
        candidate_ids = {row["candidate_id"] for row in rows}
        if seen_candidates.intersection(candidate_ids):
            raise InputValidationError("candidate identities must be globally unique across rounds")
        if manifest["round_lineage_id"] in seen_lineages:
            raise InputValidationError("round lineage identities must be globally unique")
        seen_candidates.update(candidate_ids)
        seen_lineages.add(manifest["round_lineage_id"])
        previous_lineage = manifest["round_lineage_id"]
    return state


def commit_adaptive_sampling_round(output_directory: str | Path, proposal: dict) -> dict:
    """Atomically publish one frozen adaptive-Train proposal and advance state."""
    root = Path(output_directory)
    with sampling_run_transaction(root):
        state = verify_multiround_sampling_checkpoint(root)
        if state["status"] != "active":
            raise InputValidationError("finalized multiround sampling state cannot accept another round")
        if len(state["round_chain"]) > 1 and state["round_chain"][-1].get("phase") != "outcomes_committed":
            raise InputValidationError("the preceding adaptive round must commit formal outcomes before proposing another")
        next_round = len(state["rounds_committed"])
        manifest, rows, candidate_cases = _verify_adaptive_proposal(
            proposal, expected_round=next_round
        )
        previous_lineage = state["round_chain"][-1]["round_lineage_id"]
        if manifest["parent_round_lineage_id"] != previous_lineage:
            raise InputValidationError("adaptive round parent lineage differs from the latest committed round")
        _, reference_plan = _verify_reference_anchor(root, state)
        if reference_plan is not None:
            reference_p0 = {
                (row.get("p0_id"), row.get("p0_hash"))
                for row in reference_plan.candidate_rows
            }
            if reference_p0 != {(manifest["p0_id"], manifest["p0_hash"])}:
                raise InputValidationError(
                    "adaptive proposal p0 identity differs from the frozen reference design"
                )
        available_history = {row["candidate_id"]: row for row in acquisition_history_records(root)}
        declared_history = manifest["acquisition_history"]["records"]
        if any(available_history.get(row["candidate_id"]) != row for row in declared_history):
            raise InputValidationError("adaptive proposal acquisition history is not a verified prior Train result subset")
        existing_candidates = {
            row.get("candidate_id")
            for row in _load_reference_candidate_rows_compat(root)
        }
        for index in state["rounds_committed"][1:]:
            existing_candidates.update(
                row.get("candidate_id")
                for row in _load_rows(root / f"rounds/round-{index:03d}/adaptive_candidates.jsonl")
            )
        if existing_candidates.intersection(row["candidate_id"] for row in rows):
            raise InputValidationError("candidate identities must be globally unique across rounds")
        if any(entry["round_lineage_id"] == manifest["round_lineage_id"] for entry in state["round_chain"]):
            raise InputValidationError("round lineage identities must be globally unique")
        selected_cases: list[dict] = []
        selected_index: list[dict] = []
        adaptive_registry = None
        if manifest["candidate_binding_status"] == "content_bound":
            if reference_plan is None:
                raise InputValidationError(
                    "content-bound adaptive proposals require a verified reference plan"
                )
            selected_cases, selected_index, adaptive_registry = _build_selected_adaptive_cases(
                root, manifest, rows, candidate_cases, reference_plan
            )
        final_directory = root / f"rounds/round-{next_round:03d}"
        if final_directory.exists():
            raise InputValidationError("adaptive round path already exists; preserve it for operator review")
        final_directory.mkdir(parents=False)
        marker = final_directory / ".round.incomplete"
        marker.write_text("adaptive round publication in progress\n", encoding="utf-8")
        try:
            _write_jsonl_atomic(final_directory / "adaptive_candidates.jsonl", rows)
            _write_adaptive_candidates_parquet(
                final_directory / "adaptive_candidates.parquet", rows
            )
            _write_json_atomic(final_directory / "proposal.json", manifest)
            if manifest["candidate_binding_status"] == "content_bound":
                _write_jsonl_atomic(
                    final_directory / "candidate_cases.jsonl", candidate_cases
                )
                assert adaptive_registry is not None
                adaptive_registry.write(final_directory / "adaptive_split_registry.json")
                _write_jsonl_atomic(
                    final_directory / "selected_formal_cases.jsonl",
                    selected_cases,
                )
                _write_jsonl_atomic(
                    final_directory / "selected_case_index.jsonl", selected_index
                )
            round_manifest = {
                "schema_version": (
                    "adaptive-round-manifest@1.1.0"
                    if manifest["candidate_binding_status"] == "content_bound"
                    else "adaptive-round-manifest@1.0.0"
                ),
                "round": next_round,
                "round_lineage_id": manifest["round_lineage_id"],
                "parent_round_lineage_id": previous_lineage,
                "stream": "adaptive_train",
                "proposal_hash": manifest["proposal_hash"],
                "acquisition_model_hash": manifest["acquisition_model_hash"],
                "selection_random_identity_hash": manifest["selection_random_identity"]["identity_hash"],
                "proposal_file_hash": file_hash(final_directory / "proposal.json"),
                "candidate_ledger_file_hash": file_hash(final_directory / "adaptive_candidates.jsonl"),
                "candidate_parquet_file_hash": file_hash(final_directory / "adaptive_candidates.parquet"),
                "candidate_count": len(rows),
                "selected_count": sum(row["selected"] for row in rows),
                "candidate_binding_status": manifest["candidate_binding_status"],
                "phase": "proposal_frozen",
            }
            if manifest["candidate_binding_status"] == "content_bound":
                round_manifest.update(
                    {
                        "candidate_cases_file_hash": file_hash(
                            final_directory / "candidate_cases.jsonl"
                        ),
                        "selected_formal_cases_file_hash": file_hash(
                            final_directory / "selected_formal_cases.jsonl"
                        ),
                        "selected_case_index_file_hash": file_hash(
                            final_directory / "selected_case_index.jsonl"
                        ),
                        "selected_case_index_hash": content_hash(selected_index),
                        "selected_case_set_hash": manifest["selected_case_set_hash"],
                        "adaptive_split_registry_file_hash": file_hash(
                            final_directory / "adaptive_split_registry.json"
                        ),
                        "adaptive_split_registry_hash": adaptive_registry.definition_hash,
                    }
                )
            _write_json_atomic(final_directory / "round_manifest.json", round_manifest)
            marker.unlink()
        except Exception:
            # The incomplete marker or orphan final directory is deliberate
            # fail-closed evidence. Resume verification will reject it.
            raise
        entry = {
            "round": next_round,
            "round_lineage_id": manifest["round_lineage_id"],
            "parent_round_lineage_id": previous_lineage,
            "stream": "adaptive_train",
            "round_manifest_path": f"rounds/round-{next_round:03d}/round_manifest.json",
            "round_manifest_hash": file_hash(final_directory / "round_manifest.json"),
            "proposal_hash": manifest["proposal_hash"],
            "acquisition_model_hash": manifest["acquisition_model_hash"],
            "selection_random_identity_hash": manifest["selection_random_identity"]["identity_hash"],
            "candidate_binding_status": manifest["candidate_binding_status"],
            "phase": "proposal_frozen",
            "outcome_ledger_file_hash": None,
            "formal_archive": None,
        }
        changed = deepcopy(state)
        changed.pop("checkpoint_hash", None)
        changed["rounds_committed"].append(next_round)
        changed["round_chain"].append(entry)
        changed["updated_at"] = datetime.now(timezone.utc).isoformat()
        committed = _write_multiround_checkpoint(root, changed)
        verify_multiround_sampling_checkpoint(root)
        return committed


def bind_adaptive_round_archive(
    output_directory: str | Path,
    round_index: int,
    formal_manifest_path: str | Path,
) -> dict:
    """Bind a complete ShardedDatasetWriter product to one frozen round."""

    root = Path(output_directory)
    with sampling_run_transaction(root):
        return _bind_adaptive_round_archive_locked(
            root, round_index, formal_manifest_path
        )


def _bind_adaptive_round_archive_locked(
    root: Path,
    round_index: int,
    formal_manifest_path: str | Path,
) -> dict:
    """Bind an archive while the caller holds the sampling writer lock."""

    state = verify_multiround_sampling_checkpoint(root)
    if state["status"] != "active":
        raise InputValidationError("finalized multiround state cannot bind an archive")
    if (
        not isinstance(round_index, int)
        or isinstance(round_index, bool)
        or round_index < 1
        or round_index != state["rounds_committed"][-1]
    ):
        raise InputValidationError("adaptive archive must bind the latest committed round")
    entry = state["round_chain"][round_index]
    if entry.get("phase") != "proposal_frozen":
        raise InputValidationError("adaptive round is not awaiting formal execution")
    if entry.get("candidate_binding_status") != "content_bound":
        raise InputValidationError(
            "an unbound abstract proposal cannot accept a formal archive"
        )
    directory = root / f"rounds/round-{round_index:03d}"
    expected_manifest = (directory / "formal_dataset/manifest.json").resolve()
    supplied_manifest = Path(formal_manifest_path).resolve()
    if supplied_manifest != expected_manifest:
        raise InputValidationError(
            "adaptive archive manifest must use the frozen round formal_dataset path"
        )
    proposal_manifest = _load_json(directory / "proposal.json")
    proposal_rows = _load_rows(directory / "adaptive_candidates.jsonl")
    summary, inventory, _ = _verify_adaptive_formal_archive_files(
        root, round_index, proposal_manifest, proposal_rows
    )
    formal_archive = {
        "manifest_path": expected_manifest.relative_to(root.resolve()).as_posix(),
        "summary": summary,
        "committed_files": inventory,
    }
    round_manifest_path = directory / "round_manifest.json"
    round_manifest = _load_json(round_manifest_path)
    round_manifest["phase"] = "archive_complete"
    round_manifest["formal_archive"] = formal_archive
    _write_json_atomic(round_manifest_path, round_manifest)
    changed = deepcopy(state)
    changed.pop("checkpoint_hash", None)
    changed_entry = changed["round_chain"][round_index]
    changed_entry["phase"] = "archive_complete"
    changed_entry["formal_archive"] = formal_archive
    changed_entry["round_manifest_hash"] = file_hash(round_manifest_path)
    changed["updated_at"] = datetime.now(timezone.utc).isoformat()
    committed = _write_multiround_checkpoint(root, changed)
    verify_multiround_sampling_checkpoint(root)
    return committed


def commit_adaptive_round_outcomes(
    output_directory: str | Path,
    round_index: int,
    outcome_rows: list[dict] | None = None,
) -> dict:
    """Bind every selected candidate to a formal success/refusal identity.

    This ledger contains hashes and status only, not label values.  Successful
    Train hashes become the only eligible history for a later acquisition
    model; refused candidates remain auditable and are never relabelled.
    """
    root = Path(output_directory)
    with sampling_run_transaction(root):
        state = verify_multiround_sampling_checkpoint(root)
        if state["status"] != "active":
            raise InputValidationError("finalized multiround sampling state cannot accept outcomes")
        if (
            not isinstance(round_index, int)
            or isinstance(round_index, bool)
            or round_index < 1
            or round_index != state["rounds_committed"][-1]
        ):
            raise InputValidationError("outcomes must update the latest committed adaptive round")
        entry = state["round_chain"][round_index]
        if entry.get("phase") not in {"proposal_frozen", "archive_complete"}:
            raise InputValidationError("adaptive round outcomes are already committed")
        directory = root / f"rounds/round-{round_index:03d}"
        proposal_rows = _load_rows(directory / "adaptive_candidates.jsonl")
        proposal_manifest = _load_json(directory / "proposal.json")
        archive_bound = entry.get("candidate_binding_status") == "content_bound"
        if archive_bound:
            if entry.get("phase") != "archive_complete":
                raise InputValidationError(
                    "content-bound adaptive outcomes require a verified formal archive"
                )
            archive_summary, inventory, derived_rows = _verify_adaptive_formal_archive_files(
                root, round_index, proposal_manifest, proposal_rows
            )
            formal_archive = entry.get("formal_archive")
            if (
                not isinstance(formal_archive, dict)
                or formal_archive.get("summary") != archive_summary
                or formal_archive.get("committed_files") != inventory
            ):
                raise InputValidationError(
                    "adaptive formal archive changed after it was bound"
                )
            if outcome_rows is not None and outcome_rows != derived_rows:
                raise InputValidationError(
                    "supplied adaptive outcomes differ from the verified formal archive"
                )
            outcome_rows = derived_rows
        elif outcome_rows is None:
            raise InputValidationError(
                "unbound abstract outcomes require an explicit non-CLI audit ledger"
            )
        if not isinstance(outcome_rows, list):
            raise InputValidationError("adaptive outcomes must be a list")
        summary = _verify_adaptive_outcomes(
            outcome_rows, proposal_rows, archive_bound=archive_bound
        )
        outcome_path = directory / "formal_outcomes.jsonl"
        if outcome_path.exists():
            raise InputValidationError("adaptive outcome path already exists; preserve it for operator review")
        marker = directory / ".outcomes.incomplete"
        marker.write_text("adaptive outcome publication in progress\n", encoding="utf-8")
        try:
            _write_jsonl_atomic(outcome_path, outcome_rows)
            round_manifest_path = directory / "round_manifest.json"
            round_manifest = _load_json(round_manifest_path)
            round_manifest.update({
                "phase": "outcomes_committed",
                "outcome_source": (
                    "verified_formal_archive"
                    if archive_bound
                    else "unverified_abstract_compatibility"
                ),
                "outcome_ledger_file_hash": file_hash(outcome_path),
                **summary,
            })
            _write_json_atomic(round_manifest_path, round_manifest)
            marker.unlink()
        except Exception:
            raise
        changed = deepcopy(state)
        changed.pop("checkpoint_hash", None)
        changed_entry = changed["round_chain"][round_index]
        changed_entry["phase"] = "outcomes_committed"
        changed_entry["outcome_source"] = (
            "verified_formal_archive"
            if archive_bound
            else "unverified_abstract_compatibility"
        )
        changed_entry["outcome_ledger_file_hash"] = file_hash(outcome_path)
        changed_entry["round_manifest_hash"] = file_hash(round_manifest_path)
        changed["updated_at"] = datetime.now(timezone.utc).isoformat()
        committed = _write_multiround_checkpoint(root, changed)
        verify_multiround_sampling_checkpoint(root)
        return committed


def finalize_multiround_sampling(output_directory: str | Path) -> dict:
    root = Path(output_directory)
    with sampling_run_transaction(root):
        state = verify_multiround_sampling_checkpoint(root)
        if state["status"] == "finalized":
            return state
        if any(
            entry.get("phase") != "outcomes_committed"
            or entry.get("outcome_source") != "verified_formal_archive"
            for entry in state["round_chain"][1:]
        ):
            raise InputValidationError(
                "all adaptive rounds must commit outcomes derived from verified formal archives before finalization"
            )
        changed = deepcopy(state)
        changed.pop("checkpoint_hash", None)
        changed["status"] = "finalized"
        changed["updated_at"] = datetime.now(timezone.utc).isoformat()
        changed["finalized_at"] = changed["updated_at"]
        result = _write_multiround_checkpoint(root, changed)
        verify_multiround_sampling_checkpoint(root)
        return result
