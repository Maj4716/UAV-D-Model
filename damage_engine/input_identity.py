"""Generic finite-input snapshots and execution identity.

These helpers identify bytes and execution software. They do not assert the
validity of the contents or of any scientific model that consumes them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator, Mapping

import numpy as np

from .content_contract import is_sha256_identity
from .errors import InputValidationError
from .utils import canonical_json_bytes, content_hash, file_hash


INPUT_IDENTITY_VERSION = "1.0.0"
DEPENDENCY_LOGICAL_ID = "warhead.fragment_source.fragment_list_ref"
PARSER_VERSION = "fragment-list-local-1"


@contextmanager
def single_writer_lock(output_directory: Path):
    """Refuse competing writers and retain an ownership record after hard exit."""
    resolved = output_directory.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lock_path = resolved.parent / ("." + resolved.name + ".writer.lock")
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise InputValidationError(
            "single-writer lock already exists; investigate its owner before explicit recovery",
            context={"lock_path": str(lock_path)},
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "output_directory": str(resolved), "created_at": datetime.now(timezone.utc).isoformat()}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        lock_path.unlink(missing_ok=True)


class FrozenCase(dict):
    """A schema-compatible dictionary carrying immutable input identity."""

    def __init__(self, value: dict, *, input_identity: dict):
        super().__init__(value)
        self.input_identity = input_identity


# Canonical JSON is part of the persisted identity contract.  Large sampling
# plans intentionally share a few immutable documents (most importantly the
# complete split registry) between every materialized case.  Calling
# ``json.dumps`` independently for every case serializes those same bytes N
# times and turns an N-case plan into O(N^2) hashing work.  The helpers below
# preserve the exact ``canonical_json_bytes`` byte stream while allowing only
# explicitly supplied, identity-checked object instances to reuse frozen
# bytes.  They are private because caching a mutable object beyond one tightly
# scoped freeze/build operation would be unsafe.
_CanonicalSubtreeCache = Mapping[int, tuple[Any, bytes]]


def _canonical_subtree_cache(values: Iterable[Any]) -> dict[int, tuple[Any, bytes]]:
    """Freeze canonical bytes for shared JSON subtrees in the current scope."""

    cache: dict[int, tuple[Any, bytes]] = {}
    for value in values:
        cache[id(value)] = (value, canonical_json_bytes(value))
    return cache


def _write_canonical_json(
    value: Any,
    emit: Callable[[bytes], None],
    cached_subtrees: _CanonicalSubtreeCache,
) -> None:
    """Emit exactly the bytes produced by :func:`canonical_json_bytes`.

    Sampling/case documents use string object keys.  For any other mapping-key
    type we deliberately fall back to the reference encoder so its errors and
    coercion rules remain authoritative.
    """

    placeholders: dict[int, tuple[Any, str, bytes, int]] = {}

    def replace_shared(item: Any) -> Any:
        cached = cached_subtrees.get(id(item))
        if cached is not None and cached[0] is item:
            entry = placeholders.get(id(item))
            if entry is None:
                placeholder = (
                    f"__damage_engine_frozen_json_{len(placeholders)}_{id(item):x}__"
                )
                entry = (item, placeholder, cached[1], 0)
            placeholders[id(item)] = (*entry[:3], entry[3] + 1)
            return entry[1]
        if isinstance(item, dict):
            return {key: replace_shared(child) for key, child in item.items()}
        if isinstance(item, list):
            return [replace_shared(child) for child in item]
        if isinstance(item, tuple):
            return tuple(replace_shared(child) for child in item)
        return item

    transformed = replace_shared(value)
    encoded = canonical_json_bytes(transformed)
    spans: list[tuple[int, int, bytes]] = []
    for _item, placeholder, replacement, expected_count in placeholders.values():
        token = canonical_json_bytes(placeholder)
        start = 0
        observed_count = 0
        while True:
            position = encoded.find(token, start)
            if position < 0:
                break
            spans.append((position, position + len(token), replacement))
            observed_count += 1
            start = position + len(token)
        if observed_count != expected_count:
            # An original string collided with an internal placeholder. This is
            # vanishingly unlikely, but identity code must fail closed rather
            # than silently rewrite user data.
            raise InputValidationError(
                "canonical JSON cache placeholder collided with case content"
            )
    spans.sort(key=lambda span: span[0])
    cursor = 0
    for start, end, replacement in spans:
        if start < cursor:
            raise InputValidationError("canonical JSON cache spans overlap")
        emit(encoded[cursor:start])
        emit(replacement)
        cursor = end
    emit(encoded[cursor:])


def _canonical_json_bytes_with_cached_subtrees(
    value: Any, cached_subtrees: _CanonicalSubtreeCache,
) -> bytes:
    """Testing aid for byte-for-byte comparison with canonical_json_bytes."""

    chunks: list[bytes] = []
    _write_canonical_json(value, chunks.append, cached_subtrees)
    return b"".join(chunks)


def _content_hash_with_cached_subtrees(
    value: Any, cached_subtrees: _CanonicalSubtreeCache,
) -> str:
    digest = hashlib.sha256()
    _write_canonical_json(value, digest.update, cached_subtrees)
    return "sha256:" + digest.hexdigest()


def _reference(case: dict) -> str | None:
    warhead = case.get("warhead", {})
    if not isinstance(warhead, dict):
        raise InputValidationError("warhead must be a JSON object")
    source = warhead.get("fragment_source", {})
    if not isinstance(source, dict):
        raise InputValidationError("fragment_source must be a JSON object")
    return source.get("fragment_list_ref")


def _dependency_identity(digest: str, suffix: str) -> dict:
    return {
        "logical_source_id": DEPENDENCY_LOGICAL_ID,
        "content_hash": digest,
        "format": suffix.lower().lstrip("."),
        "parser_version": PARSER_VERSION,
    }


def semantic_case_value(case: dict, *, loaded_dependency: dict | None = None) -> dict:
    """Normalize dependency locations to a content address for logical identity."""
    value = deepcopy(dict(case))
    reference = _reference(case)
    if reference is None:
        return value
    if isinstance(case, FrozenCase):
        if not case.input_identity.get("dependencies") or reference != case.input_identity.get("snapshot_reference"):
            raise InputValidationError("frozen external input reference has changed")
        dependency = case.input_identity["dependencies"][0]
        observed_hash = loaded_dependency.get("content_hash") if loaded_dependency is not None else file_hash(reference)
        if observed_hash != dependency["content_hash"]:
            raise InputValidationError("loaded external input differs from frozen snapshot")
    elif loaded_dependency is not None and loaded_dependency.get("content_hash"):
        dependency = _dependency_identity(
            loaded_dependency["content_hash"], "." + loaded_dependency["format"]
        )
    else:
        if not isinstance(reference, str) or not reference.strip():
            raise InputValidationError("external input reference must be a non-empty path")
        try:
            path = Path(reference).expanduser().resolve(strict=True)
            dependency = _dependency_identity(file_hash(path), path.suffix)
        except (OSError, RuntimeError) as exc:
            raise InputValidationError("external input could not be identified") from exc
    value["warhead"]["fragment_source"]["fragment_list_ref"] = (
        "input-content:" + dependency["content_hash"]
    )
    return value


def _semantic_case_value_for_hash(
    case: dict, *, loaded_dependency: dict | None = None,
) -> tuple[dict, dict | None]:
    """Return the logical case using copy-on-write for the reference path.

    ``semantic_case_value`` remains the public detached-copy API.  Identity
    hashing only needs a read-only view, so cloning the complete (and possibly
    registry-heavy) tree is unnecessary.  The three dictionaries on the
    external-reference path are copied before substitution; every other child
    stays shared and therefore remains eligible for scoped canonical caching.
    """

    reference = _reference(case)
    if reference is None:
        return case, None
    if isinstance(case, FrozenCase):
        if (
            not case.input_identity.get("dependencies")
            or reference != case.input_identity.get("snapshot_reference")
        ):
            raise InputValidationError("frozen external input reference has changed")
        dependency = case.input_identity["dependencies"][0]
        observed_hash = (
            loaded_dependency.get("content_hash")
            if loaded_dependency is not None
            else file_hash(reference)
        )
        if observed_hash != dependency["content_hash"]:
            raise InputValidationError("loaded external input differs from frozen snapshot")
    elif loaded_dependency is not None and loaded_dependency.get("content_hash"):
        dependency = _dependency_identity(
            loaded_dependency["content_hash"], "." + loaded_dependency["format"]
        )
    else:
        if not isinstance(reference, str) or not reference.strip():
            raise InputValidationError("external input reference must be a non-empty path")
        try:
            path = Path(reference).expanduser().resolve(strict=True)
            dependency = _dependency_identity(file_hash(path), path.suffix)
        except (OSError, RuntimeError) as exc:
            raise InputValidationError("external input could not be identified") from exc
    value = dict(case)
    warhead = dict(value["warhead"])
    source = dict(warhead["fragment_source"])
    source["fragment_list_ref"] = "input-content:" + dependency["content_hash"]
    warhead["fragment_source"] = source
    value["warhead"] = warhead
    return value, dependency


def case_identity_hash(
    case: dict,
    *,
    loaded_dependency: dict | None = None,
    _canonical_subtrees: _CanonicalSubtreeCache | None = None,
) -> str:
    # Recompute from current case fields even for FrozenCase instances; cached
    # bytes apply only to explicitly frozen shared child objects, never to the
    # mutable case root.
    value, dependency = _semantic_case_value_for_hash(
        case, loaded_dependency=loaded_dependency
    )
    dependencies = (
        case.input_identity["dependencies"]
        if isinstance(case, FrozenCase)
        else ([dependency] if dependency is not None else [])
    )
    identity_value: Any = (
        {"case": value, "dependencies": dependencies}
        if dependencies
        else value
    )
    if _canonical_subtrees is not None:
        return _content_hash_with_cached_subtrees(
            identity_value, _canonical_subtrees
        )
    return content_hash(identity_value)


def input_identity_from_case_hashes(
    case_hashes: Iterable[str], *, dependencies: Iterable[dict] = (),
) -> dict:
    """Build the exact finite-input identity from already verified case hashes.

    This is deliberately narrower than :func:`freeze_case_inputs`: it does not
    capture files or materialize cases.  Callers must have computed every hash
    from the final formal case and must supply the dependency identities from a
    completed input snapshot.  Sampling initialization uses it only after the
    factory has filled and aligned ``formal_case_hash`` for every selected row.
    """

    sequence = hashlib.sha256()
    count = 0
    for digest in case_hashes:
        if not is_sha256_identity(digest, acceptance="legacy_python_hex"):
            raise InputValidationError(
                "precomputed case identity is not a SHA-256 digest"
            )
        sequence.update(digest.encode("ascii") + b"\n")
        count += 1
    dependency_by_hash: dict[str, dict] = {}
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise InputValidationError("input dependency identity must be a JSON object")
        dependency_by_hash[content_hash(dependency)] = dependency
    dependency_list = [dependency_by_hash[key] for key in sorted(dependency_by_hash)]
    identity = {
        "version": INPUT_IDENTITY_VERSION,
        "expected_case_count": count,
        "ordered_case_identity_hash": "sha256:" + sequence.hexdigest(),
        "dependency_manifest_hash": content_hash(dependency_list),
        "location_policy": "content_and_format_sensitive_location_independent",
    }
    identity["identity_hash"] = content_hash(identity)
    return identity


class FrozenCaseInputs:
    def __init__(self, spool_path: Path, identity: dict, metadata: dict):
        self.spool_path = spool_path
        self.identity = identity
        self.metadata = metadata

    def __len__(self) -> int:
        return int(self.identity["expected_case_count"])

    def __iter__(self) -> Iterator[FrozenCase]:
        with self.spool_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                yield FrozenCase(record["case"], input_identity=record["input_identity"])


class CapturedCaseInputs(ABC):
    """Neutral contract for a pre-captured, replayable finite input stream.

    Implementations own their capture representation and hashing accelerator,
    while dataset writers consume only this stable contract.  The logical
    identity and per-chunk metadata remain identical to the generic freezer's
    values; this interface does not define a new archive or hash format.
    """

    identity: dict
    metadata: dict

    @abstractmethod
    def __len__(self) -> int:
        """Return the exact number of captured cases."""

    @abstractmethod
    def __iter__(self) -> Iterator[FrozenCase]:
        """Replay captured cases in their committed order."""

    @abstractmethod
    def case_hash(self, case: dict) -> str:
        """Return the canonical full-input identity for one replayed case."""

    @abstractmethod
    def chunk_input_metadata(self, case_hashes: Iterable[str]) -> dict:
        """Build the existing per-shard input snapshot metadata."""


class CapturedCaseSource(ABC):
    """Source capable of producing a stable captured-input context."""

    @abstractmethod
    def freeze_captured_inputs(
        self, *, cases_base_directory: str | Path | None = None,
    ) -> Any:
        """Return a context manager yielding captured finite inputs."""


@contextmanager
def freeze_case_inputs(
    cases: Iterable[dict], *, cases_base_directory: str | Path | None = None,
) -> Iterator[FrozenCaseInputs]:
    """Freeze a finite input stream before any dataset publication.

    Relative references require an explicit base directory. The logical
    identity is content/format sensitive and location independent. Locations
    and the base directory are retained separately as execution provenance.
    Every case is spooled on disk; RAM use does not grow with the case count.
    Snapshot copies live for the whole context, including worker execution.
    """
    base = Path(cases_base_directory).expanduser().resolve() if cases_base_directory is not None else None
    with tempfile.TemporaryDirectory(prefix="dataset-input-snapshot-") as temporary:
        root = Path(temporary)
        spool = root / "cases.jsonl"
        dependency_cache: dict[str, tuple[Path, dict]] = {}
        dependencies: dict[str, dict] = {}
        locations: dict[str, dict] = {}
        sequence = hashlib.sha256()
        count = 0
        with spool.open("w", encoding="utf-8", newline="\n") as stream:
            for case in cases:
                if not isinstance(case, dict):
                    raise InputValidationError("dataset inputs must be JSON objects")
                value = deepcopy(dict(case))
                reference = _reference(case)
                case_dependencies: list[dict] = []
                if isinstance(case, FrozenCase):
                    info = deepcopy(case.input_identity)
                    case_dependencies = info["dependencies"]
                    if reference is not None:
                        if file_hash(reference) != case_dependencies[0]["content_hash"]:
                            raise InputValidationError("frozen external input snapshot has changed")
                    for dependency in case_dependencies:
                        dependencies[content_hash(dependency)] = dependency
                    for location in info.get("locations", []):
                        locations[content_hash(location)] = location
                else:
                    info = {"dependencies": [], "locations": []}
                    if reference is not None:
                        if not isinstance(reference, str) or not reference.strip():
                            raise InputValidationError("external input reference must be a non-empty path")
                        requested = Path(reference).expanduser()
                        if not requested.is_absolute():
                            if base is None:
                                raise InputValidationError("relative external input references require cases_base_directory")
                            requested = base / requested
                        try:
                            source_path = requested.resolve(strict=True)
                            if not source_path.is_file():
                                raise OSError("not a regular file")
                        except (OSError, RuntimeError) as exc:
                            raise InputValidationError("external input reference does not resolve to a regular file", context={"reference": reference}) from exc
                        if source_path.suffix.lower() not in {".csv", ".json", ".jsonl"}:
                            raise InputValidationError("external input format is unsupported", context={"reference": reference})
                        key = str(source_path)
                        if key not in dependency_cache:
                            target = root / (f"source-{len(dependency_cache):06d}" + source_path.suffix.lower())
                            digest = hashlib.sha256()
                            try:
                                with source_path.open("rb") as source, target.open("xb") as destination:
                                    for block in iter(lambda: source.read(1024 * 1024), b""):
                                        digest.update(block)
                                        destination.write(block)
                            except OSError as exc:
                                raise InputValidationError("external input snapshot could not be captured", context={"reference": reference}) from exc
                            dependency = _dependency_identity("sha256:" + digest.hexdigest(), source_path.suffix)
                            # Check that the bytes captured were a coherent version
                            # of the source rather than an observed concurrent edit.
                            if file_hash(source_path) != dependency["content_hash"]:
                                raise InputValidationError("external input changed during snapshot capture")
                            target.chmod(0o444)
                            dependency_cache[key] = (target, dependency)
                        target, dependency = dependency_cache[key]
                        location = {"logical_source_id": DEPENDENCY_LOGICAL_ID, "reference": reference, "resolved_path": str(source_path), "content_hash": dependency["content_hash"]}
                        info = {"dependencies": [dependency], "locations": [location]}
                        case_dependencies = [dependency]
                        dependencies[content_hash(dependency)] = dependency
                        locations[content_hash(location)] = location
                        value["warhead"]["fragment_source"]["fragment_list_ref"] = str(target)
                        info["snapshot_reference"] = str(target)
                frozen = FrozenCase(value, input_identity=info)
                digest = case_identity_hash(frozen)
                sequence.update(digest.encode("ascii") + b"\n")
                stream.write(json.dumps({"case": value, "input_identity": info}, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
        dependency_list = [dependencies[key] for key in sorted(dependencies)]
        identity = {
            "version": INPUT_IDENTITY_VERSION,
            "expected_case_count": count,
            "ordered_case_identity_hash": "sha256:" + sequence.hexdigest(),
            "dependency_manifest_hash": content_hash(dependency_list),
            "location_policy": "content_and_format_sensitive_location_independent",
        }
        identity["identity_hash"] = content_hash(identity)
        metadata = {
            **identity,
            "dependencies": dependency_list,
            "locations": [locations[key] for key in sorted(locations)],
            "cases_base_directory": str(base) if base is not None else None,
            "freeze_policy": "finite_stream_fully_spooled_before_execution",
        }
        yield FrozenCaseInputs(spool, identity, metadata)


def capture_environment_identity() -> dict:
    """Record execution software separately from logical dataset identity."""
    versions = {}
    for name in ("numpy", "jsonschema", "pyarrow", "torch", "damage-dataset-engine"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    value = {
        "version": "1.0.0",
        "python": {"implementation": platform.python_implementation(), "version": platform.python_version(), "build": sys.version, "executable": str(Path(sys.executable).resolve())},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "architecture": platform.architecture()[0]},
        "dependencies": versions,
        "numeric_thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "numpy_build_hash": content_hash(json.loads(json.dumps(getattr(np.__config__, "CONFIG", {}), default=str))),
        "rng_identity": {"bit_generator": "numpy.random.PCG64", "seed_sequence": "numpy.random.SeedSequence", "numpy_version": np.__version__},
        "accelerator_runtime_probe": "not_performed; package version only",
    }
    value["identity_hash"] = content_hash(value)
    return value
