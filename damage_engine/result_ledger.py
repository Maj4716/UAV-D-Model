"""Deterministic raw-result ledgers with gzip/plain backward compatibility."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator

from .errors import InputValidationError


RESULT_LEDGER_FORMATS = frozenset({"jsonl", "jsonl_gzip"})


def write_result_ledger(
    directory: str | Path,
    rows: Iterable[dict],
    *,
    compression: str,
) -> tuple[Path, str]:
    """Write a deterministic compact JSONL ledger and return path/format."""

    root = Path(directory)
    if compression not in {"none", "gzip"}:
        raise ValueError("result ledger compression must be 'none' or 'gzip'")
    path = root / ("results.jsonl.gz" if compression == "gzip" else "results.jsonl")

    def encoded_rows() -> Iterator[bytes]:
        for row in rows:
            yield (
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")

    if compression == "gzip":
        # filename="" and mtime=0 remove host path/time from the header, so
        # identical result records produce identical committed bytes.
        with path.open("wb") as raw:
            with gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0
            ) as stream:
                for value in encoded_rows():
                    stream.write(value)
    else:
        with path.open("wb") as stream:
            for value in encoded_rows():
                stream.write(value)
    return path, ("jsonl_gzip" if compression == "gzip" else "jsonl")


def result_partition(manifest: dict) -> dict:
    matches = [
        item
        for item in manifest.get("partitions", [])
        if item.get("name") == "audits"
    ]
    if len(matches) != 1 or matches[0].get("format") not in RESULT_LEDGER_FORMATS:
        raise InputValidationError(
            "dataset manifest must declare exactly one supported raw-result ledger"
        )
    return matches[0]


def resolve_result_ledger(
    directory: str | Path, manifest: dict | None = None,
) -> Path:
    """Resolve one ledger from its manifest or fail closed on ambiguity."""

    root = Path(directory).resolve()
    if manifest is not None:
        partition = result_partition(manifest)
        relative = Path(str(partition.get("path_glob", "")))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or any(character in str(relative) for character in "*?[]")
        ):
            raise InputValidationError("raw-result ledger path must be one exact relative path")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise InputValidationError("raw-result ledger declared by manifest is missing")
        expected_suffix = ".jsonl.gz" if partition["format"] == "jsonl_gzip" else ".jsonl"
        if not path.name.endswith(expected_suffix):
            raise InputValidationError("raw-result ledger format and filename disagree")
        return path
    candidates = [
        path for path in (root / "results.jsonl.gz", root / "results.jsonl")
        if path.is_file()
    ]
    if len(candidates) != 1:
        raise InputValidationError(
            "raw-result ledger is missing or ambiguous",
            context={"directory": str(root)},
        )
    return candidates[0]


def iter_result_ledger(path: str | Path) -> Iterator[dict]:
    value = Path(path)
    opener = gzip.open if value.name.endswith(".jsonl.gz") else open
    try:
        with opener(value, "rt", encoding="utf-8-sig", newline="") as stream:
            count = 0
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise InputValidationError(
                        "raw-result ledger is invalid JSONL",
                        context={"path": str(value), "line": line_number, "column": exc.colno},
                    ) from exc
                if not isinstance(row, dict):
                    raise InputValidationError(
                        "raw-result ledger records must be objects",
                        context={"path": str(value), "line": line_number},
                    )
                count += 1
                yield row
    except InputValidationError:
        raise
    except (OSError, UnicodeError, EOFError) as exc:
        raise InputValidationError(
            "raw-result ledger is unreadable or corrupt",
            context={"path": str(value)},
        ) from exc
    if count == 0:
        raise InputValidationError(
            "raw-result ledger must contain at least one object record",
            context={"path": str(value)},
        )


def load_result_ledger(
    directory: str | Path, manifest: dict | None = None,
) -> list[dict]:
    return list(iter_result_ledger(resolve_result_ledger(directory, manifest)))
