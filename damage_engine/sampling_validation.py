"""Shared, outcome-independent sampling identity and assignment guards."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .errors import InputValidationError
from .split_registry import LEGAL_SPLITS, SplitRegistry


def load_verified_split_registry(
    source: dict | str | Path | SplitRegistry, *,
    expected_hash: str | None = None, expected_id: str | None = None,
) -> SplitRegistry:
    if isinstance(source, SplitRegistry):
        registry = SplitRegistry.from_dict(source.to_dict())
    elif isinstance(source, dict):
        registry = SplitRegistry.from_dict(source)
    elif isinstance(source, (str, Path)):
        registry = SplitRegistry.from_file(source)
    else:
        raise InputValidationError("a complete frozen split registry snapshot is required")
    if expected_hash is not None and registry.definition_hash != expected_hash:
        raise InputValidationError("split registry snapshot differs from the declared definition hash")
    if expected_id is not None and registry.registry_id != expected_id:
        raise InputValidationError("split registry snapshot differs from the declared registry ID")
    return registry


def _assignment_row(row: dict) -> dict:
    if not isinstance(row, dict):
        raise InputValidationError("sampling assignment rows must be objects")
    split = row.get("split", row.get("frozen_split"))
    if "split" in row and "frozen_split" in row and row["split"] != row["frozen_split"]:
        raise InputValidationError("sampling split and frozen_split declarations disagree")
    assignment = {"lineage_id": row.get("lineage_id"), "split": split,
                  "role": row.get("role"), "epistemic_group_id": row.get("epistemic_group_id")}
    if (any(not isinstance(value, str) or not value for value in assignment.values())
            or split not in LEGAL_SPLITS):
        raise InputValidationError("sampling assignments require named lineage, role, epistemic group and legal split")
    return assignment


def validate_registry_assignments(registry: SplitRegistry, rows: Iterable[dict]) -> None:
    """Compare every declared assignment, including refused/masked input rows."""
    for row in rows:
        actual = _assignment_row(row)
        expected = registry.assignment(actual["lineage_id"]).to_dict()
        if actual != expected:
            raise InputValidationError("sampling assignment contradicts the actual frozen split registry",
                                       context={"declared": actual, "expected": expected})


def validate_candidate_set(rows: Iterable[dict]) -> None:
    """The default archive contract is one candidate to one execution."""
    seen: set[str] = set()
    groups: dict[tuple[str, str], str] = {}
    for row in rows:
        assignment = _assignment_row(row)
        identifier = row.get("candidate_id")
        if not isinstance(identifier, str) or not identifier:
            raise InputValidationError("sampling candidate_id must be a nonempty string")
        if identifier in seen:
            raise InputValidationError("sampling candidate_id must be globally unique for single-execution archives",
                                       context={"candidate_id": identifier})
        seen.add(identifier)
        for field in ("lineage_id", "epistemic_group_id"):
            key = (field, assignment[field])
            if key in groups and groups[key] != assignment["split"]:
                raise InputValidationError("sampling lineage or epistemic group crosses split boundaries",
                                           context={"field": field, "group": assignment[field]})
            groups[key] = assignment["split"]


def input_sampling_rows_from_arrays(arrays: Mapping[str, np.ndarray]) -> list[dict]:
    fields = {"case_id": "input_case_ids", "candidate_id": "input_sampling_candidate_ids",
              "lineage_id": "input_sampling_lineage_ids", "role": "input_sampling_roles",
              "epistemic_group_id": "input_sampling_epistemic_group_ids", "split": "input_frozen_splits"}
    values = {}
    length = None
    for key, field in fields.items():
        if field not in arrays:
            raise InputValidationError(f"sampling archive is missing required full-input identity field {field}")
        value = np.asarray(arrays[field])
        if value.ndim != 1 or value.dtype.kind not in "US" or any(not str(item) for item in value):
            raise InputValidationError(f"{field} must be a one-dimensional nonempty-string identity axis")
        length = len(value) if length is None else length
        if len(value) != length:
            raise InputValidationError("sampling full-input identity axes have inconsistent lengths")
        values[key] = value.tolist()
    return [dict(zip(values, row)) for row in zip(*values.values())]
