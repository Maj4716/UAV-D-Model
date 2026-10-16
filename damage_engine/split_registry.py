"""Outcome-blind, immutable split assignment registry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

from .errors import InputValidationError
from .utils import content_hash


SPLIT_REGISTRY_SCHEMA_VERSION = "1.0.0"
LEGAL_SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True)
class SplitAssignment:
    lineage_id: str
    split: str
    role: str
    epistemic_group_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "lineage_id": self.lineage_id,
            "split": self.split,
            "role": self.role,
            "epistemic_group_id": self.epistemic_group_id,
        }


class SplitRegistry:
    """Build then freeze physical-group split assignments before outcomes exist."""

    def __init__(self, registry_id: str):
        if not isinstance(registry_id, str) or not registry_id:
            raise InputValidationError("split registry_id must be a nonempty string")
        self.registry_id = registry_id
        self._assignments: dict[str, SplitAssignment] = {}
        self._epistemic_splits: dict[str, str] = {}
        self._frozen = False
        self._definition_hash: str | None = None
        self._frozen_at: str | None = None

    def register(
        self,
        *,
        lineage_id: str,
        split: str,
        role: str,
        epistemic_group_id: str,
    ) -> None:
        if self._frozen:
            raise InputValidationError("split registry is frozen and cannot accept new assignments")
        if split not in LEGAL_SPLITS:
            raise InputValidationError("split assignment is not train/validation/test")
        if not all(isinstance(value, str) and value for value in (lineage_id, role, epistemic_group_id)):
            raise InputValidationError("split lineage, role and epistemic group must be nonempty strings")
        if role == "adaptive_train" and split != "train":
            raise InputValidationError("adaptive_train lineages must be assigned to train")
        assignment = SplitAssignment(lineage_id, split, role, epistemic_group_id)
        previous = self._assignments.get(lineage_id)
        if previous is not None and previous != assignment:
            raise InputValidationError(
                "lineage has conflicting split assignments",
                context={"lineage_id": lineage_id},
            )
        epistemic_split = self._epistemic_splits.get(epistemic_group_id)
        if epistemic_split is not None and epistemic_split != split:
            raise InputValidationError(
                "epistemic group crosses split boundaries",
                context={"epistemic_group_id": epistemic_group_id},
            )
        self._assignments[lineage_id] = assignment
        self._epistemic_splits[epistemic_group_id] = split

    def register_many(self, assignments: Iterable[SplitAssignment]) -> None:
        for assignment in assignments:
            self.register(**assignment.to_dict())

    def _definition(self) -> dict:
        return {
            "schema_version": SPLIT_REGISTRY_SCHEMA_VERSION,
            "registry_id": self.registry_id,
            "assignment_phase": "pre_outcome",
            "assignments": [
                self._assignments[key].to_dict() for key in sorted(self._assignments)
            ],
        }

    def freeze(self) -> str:
        if not self._assignments:
            raise InputValidationError("split registry must contain at least one assignment")
        definition = self._definition()
        self._definition_hash = content_hash(definition)
        self._frozen_at = datetime.now(timezone.utc).isoformat()
        self._frozen = True
        return self._definition_hash

    @property
    def definition_hash(self) -> str:
        if not self._frozen or self._definition_hash is None:
            raise InputValidationError("split registry must be frozen before its hash is used")
        return self._definition_hash

    def assignment(self, lineage_id: str) -> SplitAssignment:
        if not self._frozen:
            raise InputValidationError("split registry must be frozen before assignments are consumed")
        try:
            return self._assignments[lineage_id]
        except KeyError as exc:
            raise InputValidationError(
                "lineage is absent from the frozen split registry",
                context={"lineage_id": lineage_id},
            ) from exc

    def definition_document(self) -> dict:
        """Canonical complete assignments for reproducible execution identity."""
        if not self._frozen:
            raise InputValidationError("split registry must be frozen before serialization")
        return {
            **self._definition(),
            "definition_hash": self.definition_hash,
        }

    def to_dict(self) -> dict:
        document = self.definition_document()
        if self._frozen_at is not None:
            document["frozen_at"] = self._frozen_at
        return document

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    @classmethod
    def from_file(cls, path: str | Path) -> "SplitRegistry":
        value = Path(path)
        try:
            document = json.loads(value.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InputValidationError("unable to load split registry", context={"path": str(value)}) from exc
        return cls.from_dict(document)

    @classmethod
    def from_dict(cls, document: dict) -> "SplitRegistry":
        """Parse and re-hash the complete frozen assignment snapshot."""
        required = {
            "schema_version", "registry_id", "assignment_phase", "assignments",
            "definition_hash",
        }
        if (not isinstance(document, dict) or not required.issubset(document)
                or not set(document).issubset(required | {"frozen_at"})):
            raise InputValidationError("split registry has missing or unknown fields")
        if document["schema_version"] != SPLIT_REGISTRY_SCHEMA_VERSION:
            raise InputValidationError("unsupported split registry schema version")
        if document["assignment_phase"] != "pre_outcome":
            raise InputValidationError("split registry was not frozen before outcomes")
        registry = cls(document["registry_id"])
        if not isinstance(document["assignments"], list):
            raise InputValidationError("split registry assignments must be an array")
        seen_lineages: set[str] = set()
        for item in document["assignments"]:
            if not isinstance(item, dict) or set(item) != {
                "lineage_id", "split", "role", "epistemic_group_id"
            }:
                raise InputValidationError("split registry assignment has invalid fields")
            if not isinstance(item["lineage_id"], str) or not item["lineage_id"]:
                raise InputValidationError("split registry lineage IDs must be nonempty strings")
            if item["lineage_id"] in seen_lineages:
                raise InputValidationError("split registry contains duplicate lineage assignments")
            seen_lineages.add(item["lineage_id"])
            registry.register(**item)
        if not registry._assignments:
            raise InputValidationError("split registry must contain at least one assignment")
        registry._definition_hash = content_hash(registry._definition())
        registry._frozen = True
        if registry.definition_hash != document["definition_hash"]:
            raise InputValidationError("split registry definition_hash does not match its assignments")
        if "frozen_at" in document:
            try:
                frozen_at = datetime.fromisoformat(document["frozen_at"].replace("Z", "+00:00"))
            except (TypeError, AttributeError, ValueError) as exc:
                raise InputValidationError("split registry frozen_at must be an ISO timestamp") from exc
            if frozen_at.tzinfo is None:
                raise InputValidationError("split registry frozen_at must include a timezone")
            registry._frozen_at = document["frozen_at"]
        return registry
