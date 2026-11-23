"""Hash-bound C1-C6 credibility dossiers with explicit external-evidence gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import InputValidationError
from .utils import content_hash, file_hash


CREDIBILITY_CATEGORIES = ("C1", "C2", "C3", "C4", "C5", "C6")
_STATUSES = {"pass", "fail", "incomplete", "external_required", "not_applicable"}


def expert_assertion_template(count: int = 10) -> list[dict[str, Any]]:
    if count < 10:
        raise InputValidationError("idea2 C4 requires at least ten expert assertions")
    return [
        {
            "assertion_id": f"expert-pattern-{index + 1:02d}",
            "claim": None,
            "expected_direction_or_ordering": None,
            "applicable_conditions": None,
            "authority_name_or_role": None,
            "source_reference": None,
            "confidence": None,
            "status": "awaiting_domain_authority",
        }
        for index in range(count)
    ]


def validate_expert_assertions(assertions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if isinstance(assertions, (str, bytes)) or not isinstance(assertions, Sequence):
        raise InputValidationError("expert assertions must be a sequence")
    complete = []
    identifiers = set()
    required = {
        "assertion_id",
        "claim",
        "expected_direction_or_ordering",
        "applicable_conditions",
        "authority_name_or_role",
        "source_reference",
        "confidence",
    }
    for record in assertions:
        if not isinstance(record, Mapping):
            raise InputValidationError("expert assertion records must be objects")
        identifier = record.get("assertion_id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise InputValidationError("expert assertion IDs must be unique and nonempty")
        identifiers.add(identifier)
        if required <= set(record) and all(
            record.get(name) is not None and record.get(name) != ""
            for name in required
        ):
            complete.append(identifier)
    return {
        "declared_count": len(assertions),
        "complete_count": len(complete),
        "complete_assertion_ids": complete,
        "passes_C4_minimum": len(complete) >= 10,
    }


def build_credibility_dossier(
    plan: Mapping[str, Any],
    *,
    workspace_root: Path,
) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise InputValidationError("credibility plan must be an object")
    categories = plan.get("categories")
    if not isinstance(categories, Sequence) or isinstance(categories, (str, bytes)):
        raise InputValidationError("credibility plan requires category records")
    by_id = {}
    for record in categories:
        if not isinstance(record, Mapping):
            raise InputValidationError("credibility category records must be objects")
        identifier = str(record.get("category_id"))
        status = str(record.get("status"))
        if identifier not in CREDIBILITY_CATEGORIES or identifier in by_id or status not in _STATUSES:
            raise InputValidationError("credibility category ID or status is invalid")
        evidence = []
        for relative in record.get("evidence_paths", []):
            path = (workspace_root / str(relative)).resolve()
            try:
                path.relative_to(workspace_root.resolve())
            except ValueError as exc:
                raise InputValidationError("credibility evidence must stay inside the workspace") from exc
            evidence.append(
                {
                    "path": str(relative).replace("\\", "/"),
                    "exists": path.is_file(),
                    "sha256": file_hash(path) if path.is_file() else None,
                }
            )
        if status == "pass" and (not evidence or not all(item["exists"] for item in evidence)):
            raise InputValidationError("passing credibility categories require existing evidence")
        by_id[identifier] = {
            "category_id": identifier,
            "title": str(record.get("title", identifier)),
            "status": status,
            "acceptance_rule": str(record.get("acceptance_rule", "")),
            "evidence": evidence,
            "open_items": list(map(str, record.get("open_items", []))),
        }
    if set(by_id) != set(CREDIBILITY_CATEGORIES):
        raise InputValidationError("credibility plan must cover C1-C6 exactly")
    assertions = plan.get("expert_assertions", [])
    assertion_report = validate_expert_assertions(assertions)
    if by_id["C4"]["status"] == "pass" and not assertion_report["passes_C4_minimum"]:
        raise InputValidationError("C4 cannot pass without ten complete expert assertions")
    document = {
        "schema_version": "credibility-dossier@1.0.0",
        "dossier_id": str(plan.get("dossier_id", "idea2-credibility-dossier")),
        "model_scope": str(plan.get("model_scope", "")),
        "categories": [by_id[name] for name in CREDIBILITY_CATEGORIES],
        "expert_assertion_report": assertion_report,
        "expert_assertions": list(assertions),
        "software_consistency_complete": all(by_id[name]["status"] == "pass" for name in ("C1", "C2")),
        "research_admission": all(by_id[name]["status"] == "pass" for name in CREDIBILITY_CATEGORIES),
        "physical_validation_claim_allowed": False,
    }
    return {**document, "dossier_hash": content_hash(document)}


__all__ = [
    "CREDIBILITY_CATEGORIES",
    "build_credibility_dossier",
    "expert_assertion_template",
    "validate_expert_assertions",
]
