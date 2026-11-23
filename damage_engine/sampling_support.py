"""Outcome-blind support diagnostics for staged contact sampling.

The module separates three questions that cannot be inferred from the same
evidence: geometric reachability, overlap of the declared proposal with the
reachable set, and finite candidate budget.  Candidate counts alone therefore
never prove that a stratum is structurally unreachable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .errors import InputValidationError


SUPPORT_AUDIT_SCHEMA_VERSION = "sampling-support-audit@1.0.0"
SUPPORT_EXPANSION_SCHEMA_VERSION = "outcome-blind-support-expansion@1.0.0"


@dataclass(frozen=True)
class CandidateSupportObservation:
    """One pre-teacher candidate-ledger row for a declared proposal."""

    candidate_id: str
    design_seed: int
    stratum: str
    proposal_id: str
    s_contact_admitted: bool
    proxy_domain_complete: bool | None
    inclusion_probability: float
    refusal_code: str | None = None


@dataclass(frozen=True)
class SupportReachabilityEvidence:
    """Independent evidence used to classify, not merely observe, a gap."""

    stratum: str
    geometry_reachable: bool | None
    proposal_overlaps_reachable_region: bool | None
    method_id: str
    evidence_hash: str


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"{field} must be a nonempty string")
    return value


def _nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InputValidationError(f"{field} must be a nonnegative integer")
    return int(value)


def _positive_integer(value: Any, *, field: str) -> int:
    result = _nonnegative_integer(value, field=field)
    if result < 1:
        raise InputValidationError(f"{field} must be positive")
    return result


def _finite_probability(value: Any, *, field: str, allow_zero: bool) -> float:
    if isinstance(value, bool):
        raise InputValidationError(f"{field} must be a finite probability")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            f"{field} must be a finite probability"
        ) from exc
    lower_ok = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not lower_ok or result > 1.0:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise InputValidationError(f"{field} must lie in {interval}")
    return result


def _validate_observation(row: CandidateSupportObservation) -> None:
    if not isinstance(row, CandidateSupportObservation):
        raise InputValidationError(
            "support rows must be CandidateSupportObservation instances"
        )
    _nonempty_string(row.candidate_id, field="candidate_id")
    _nonnegative_integer(row.design_seed, field="design_seed")
    _nonempty_string(row.stratum, field="stratum")
    _nonempty_string(row.proposal_id, field="proposal_id")
    if not isinstance(row.s_contact_admitted, bool):
        raise InputValidationError("s_contact_admitted must be boolean")
    if row.s_contact_admitted:
        if not isinstance(row.proxy_domain_complete, bool):
            raise InputValidationError(
                "admitted candidates require boolean proxy_domain_complete"
            )
        if row.refusal_code is not None:
            raise InputValidationError(
                "admitted candidates cannot carry an S-CONTACT refusal_code"
            )
    else:
        if row.proxy_domain_complete is not None:
            raise InputValidationError(
                "rejected candidates require proxy_domain_complete=null"
            )
        _nonempty_string(row.refusal_code, field="refusal_code")
    _finite_probability(
        row.inclusion_probability,
        field="inclusion_probability",
        allow_zero=True,
    )


def _validate_evidence(item: SupportReachabilityEvidence) -> None:
    if not isinstance(item, SupportReachabilityEvidence):
        raise InputValidationError(
            "reachability evidence must use SupportReachabilityEvidence"
        )
    _nonempty_string(item.stratum, field="evidence.stratum")
    _nonempty_string(item.method_id, field="evidence.method_id")
    _nonempty_string(item.evidence_hash, field="evidence.evidence_hash")
    if item.geometry_reachable not in {True, False, None}:
        raise InputValidationError("geometry_reachable must be boolean or null")
    if item.proposal_overlaps_reachable_region not in {True, False, None}:
        raise InputValidationError(
            "proposal_overlaps_reachable_region must be boolean or null"
        )
    if (
        item.geometry_reachable is False
        and item.proposal_overlaps_reachable_region is True
    ):
        raise InputValidationError(
            "an unreachable stratum cannot have a reachable-proposal overlap"
        )


def _wilson_interval(successes: int, trials: int) -> list[float] | None:
    if trials == 0:
        return None
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _gap_classification(
    *,
    count_checks_pass: bool,
    positivity_pass: bool,
    evidence: SupportReachabilityEvidence | None,
) -> str:
    if count_checks_pass and positivity_pass:
        return "adequate"
    if count_checks_pass and not positivity_pass:
        return "selection_positivity_failure"
    if evidence is None:
        return "indeterminate_support_gap"
    if evidence.geometry_reachable is False:
        return "structurally_unreachable"
    if (
        evidence.geometry_reachable is True
        and evidence.proposal_overlaps_reachable_region is False
    ):
        return "proposal_range_error"
    if (
        evidence.geometry_reachable is True
        and evidence.proposal_overlaps_reachable_region is True
    ):
        return "budget_insufficient"
    return "indeterminate_support_gap"


def audit_sampling_support(
    rows: Sequence[CandidateSupportObservation],
    *,
    required_strata: Sequence[str],
    minimum_attempt_count: int,
    minimum_contact_count: int,
    minimum_proxy_complete_count: int,
    minimum_inclusion_probability: float,
    reachability_evidence: Sequence[SupportReachabilityEvidence] = (),
) -> dict[str, Any]:
    """Audit finite-frame support without consulting teacher outcomes."""

    required = tuple(
        _nonempty_string(value, field="required_strata")
        for value in required_strata
    )
    if not required or len(set(required)) != len(required):
        raise InputValidationError(
            "required_strata must be a nonempty sequence of unique names"
        )
    minimum_attempt = _positive_integer(
        minimum_attempt_count, field="minimum_attempt_count"
    )
    minimum_contact = _positive_integer(
        minimum_contact_count, field="minimum_contact_count"
    )
    minimum_proxy = _positive_integer(
        minimum_proxy_complete_count,
        field="minimum_proxy_complete_count",
    )
    if minimum_contact > minimum_attempt or minimum_proxy > minimum_attempt:
        raise InputValidationError(
            "support count thresholds cannot exceed minimum_attempt_count"
        )
    positivity_min = _finite_probability(
        minimum_inclusion_probability,
        field="minimum_inclusion_probability",
        allow_zero=False,
    )

    evidence_by_stratum: dict[str, SupportReachabilityEvidence] = {}
    for item in reachability_evidence:
        _validate_evidence(item)
        if item.stratum not in required:
            raise InputValidationError(
                "reachability evidence references an undeclared stratum"
            )
        if item.stratum in evidence_by_stratum:
            raise InputValidationError(
                "reachability evidence must be unique per stratum"
            )
        evidence_by_stratum[item.stratum] = item

    values = tuple(rows)
    seen: set[tuple[int, str]] = set()
    grouped: dict[str, list[CandidateSupportObservation]] = {
        stratum: [] for stratum in required
    }
    for row in values:
        _validate_observation(row)
        if row.stratum not in grouped:
            raise InputValidationError(
                "support observation references an undeclared stratum"
            )
        key = (int(row.design_seed), row.candidate_id)
        if key in seen:
            raise InputValidationError(
                "duplicate design_seed and candidate_id in support ledger"
            )
        seen.add(key)
        grouped[row.stratum].append(row)

    cell_reports: dict[str, dict[str, Any]] = {}
    for stratum in required:
        members = grouped[stratum]
        attempts = len(members)
        admitted = sum(row.s_contact_admitted for row in members)
        proxy_complete = sum(
            bool(row.proxy_domain_complete)
            for row in members
            if row.s_contact_admitted
        )
        positive_probabilities = [
            float(row.inclusion_probability)
            for row in members
            if row.inclusion_probability > 0.0
        ]
        zero_inclusion = attempts - len(positive_probabilities)
        checks = {
            "minimum_attempt_count": attempts >= minimum_attempt,
            "minimum_contact_count": admitted >= minimum_contact,
            "minimum_proxy_complete_count": proxy_complete >= minimum_proxy,
            "positive_support_for_every_candidate": zero_inclusion == 0,
            "minimum_inclusion_probability": bool(positive_probabilities)
            and min(positive_probabilities) >= positivity_min,
        }
        count_checks_pass = all(
            checks[name]
            for name in (
                "minimum_attempt_count",
                "minimum_contact_count",
                "minimum_proxy_complete_count",
            )
        )
        positivity_pass = (
            checks["positive_support_for_every_candidate"]
            and checks["minimum_inclusion_probability"]
        )
        evidence = evidence_by_stratum.get(stratum)
        cell_reports[stratum] = {
            "attempt_count": attempts,
            "s_contact_admitted_count": admitted,
            "proxy_complete_count": proxy_complete,
            "contact_rate": None if attempts == 0 else admitted / attempts,
            "contact_rate_wilson_95": _wilson_interval(admitted, attempts),
            "proxy_complete_rate_per_attempt": (
                None if attempts == 0 else proxy_complete / attempts
            ),
            "proxy_complete_rate_given_contact": (
                None if admitted == 0 else proxy_complete / admitted
            ),
            "proxy_complete_rate_given_contact_wilson_95": _wilson_interval(
                proxy_complete, admitted
            ),
            "positive_inclusion_count": len(positive_probabilities),
            "zero_inclusion_count": zero_inclusion,
            "minimum_positive_inclusion_probability": (
                None if not positive_probabilities else min(positive_probabilities)
            ),
            "proposal_ids": sorted({row.proposal_id for row in members}),
            "refusal_code_counts": {
                code: sum(row.refusal_code == code for row in members)
                for code in sorted(
                    {
                        row.refusal_code
                        for row in members
                        if row.refusal_code is not None
                    }
                )
            },
            "checks": checks,
            "gap_classification": _gap_classification(
                count_checks_pass=count_checks_pass,
                positivity_pass=positivity_pass,
                evidence=evidence,
            ),
            "reachability_evidence": (
                None
                if evidence is None
                else {
                    "geometry_reachable": evidence.geometry_reachable,
                    "proposal_overlaps_reachable_region": (
                        evidence.proposal_overlaps_reachable_region
                    ),
                    "method_id": evidence.method_id,
                    "evidence_hash": evidence.evidence_hash,
                }
            ),
        }

    return {
        "schema_version": SUPPORT_AUDIT_SCHEMA_VERSION,
        "stage": "pre_teacher_support_audit",
        "uses_teacher_outcomes": False,
        "contract": {
            "required_strata": list(required),
            "minimum_attempt_count": minimum_attempt,
            "minimum_contact_count": minimum_contact,
            "minimum_proxy_complete_count": minimum_proxy,
            "minimum_inclusion_probability": positivity_min,
        },
        "candidate_count": len(values),
        "cells": cell_reports,
        "overall_pass": all(
            all(cell["checks"].values()) for cell in cell_reports.values()
        ),
    }


def plan_outcome_blind_expansion(
    support_report: Mapping[str, Any],
    *,
    maximum_additional_attempts_per_stratum: int,
    safety_factor: float = 1.25,
) -> dict[str, Any]:
    """Translate proven budget gaps into deterministic extra-attempt budgets."""

    if not isinstance(support_report, Mapping) or support_report.get(
        "schema_version"
    ) != SUPPORT_AUDIT_SCHEMA_VERSION:
        raise InputValidationError("support_report has an unsupported schema")
    if support_report.get("uses_teacher_outcomes") is not False:
        raise InputValidationError(
            "support expansion requires an outcome-blind support report"
        )
    maximum = _positive_integer(
        maximum_additional_attempts_per_stratum,
        field="maximum_additional_attempts_per_stratum",
    )
    if isinstance(safety_factor, bool):
        raise InputValidationError("safety_factor must be finite and at least one")
    try:
        safety = float(safety_factor)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            "safety_factor must be finite and at least one"
        ) from exc
    if not math.isfinite(safety) or safety < 1.0:
        raise InputValidationError("safety_factor must be finite and at least one")

    contract = support_report.get("contract")
    cells = support_report.get("cells")
    if not isinstance(contract, Mapping) or not isinstance(cells, Mapping):
        raise InputValidationError("support_report is missing contract or cells")
    minimum_contact = _positive_integer(
        contract.get("minimum_contact_count"),
        field="contract.minimum_contact_count",
    )
    minimum_proxy = _positive_integer(
        contract.get("minimum_proxy_complete_count"),
        field="contract.minimum_proxy_complete_count",
    )

    allocations: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for stratum in sorted(cells):
        cell = cells[stratum]
        if not isinstance(cell, Mapping):
            raise InputValidationError("support cell must be an object")
        classification = cell.get("gap_classification")
        if classification != "budget_insufficient":
            if classification != "adequate":
                skipped.append(
                    {
                        "stratum": str(stratum),
                        "reason": str(classification),
                    }
                )
            continue
        attempts = _nonnegative_integer(
            cell.get("attempt_count"), field="cell.attempt_count"
        )
        admitted = _nonnegative_integer(
            cell.get("s_contact_admitted_count"),
            field="cell.s_contact_admitted_count",
        )
        proxy_complete = _nonnegative_integer(
            cell.get("proxy_complete_count"),
            field="cell.proxy_complete_count",
        )
        # Jeffreys smoothing prevents division by zero while keeping the
        # resulting allocation explicitly provisional and capped.
        contact_rate = (admitted + 0.5) / (attempts + 1.0)
        proxy_rate = (proxy_complete + 0.5) / (attempts + 1.0)
        contact_need = max(0, minimum_contact - admitted)
        proxy_need = max(0, minimum_proxy - proxy_complete)
        uncapped = int(
            math.ceil(
                safety
                * max(
                    contact_need / contact_rate,
                    proxy_need / proxy_rate,
                )
            )
        )
        additional = min(maximum, max(1, uncapped))
        allocations.append(
            {
                "stratum": str(stratum),
                "source_proposal_ids": list(cell.get("proposal_ids", [])),
                "additional_attempts": additional,
                "uncapped_additional_attempts": uncapped,
                "capped": additional < uncapped,
                "smoothed_contact_rate": contact_rate,
                "smoothed_proxy_complete_rate_per_attempt": proxy_rate,
                "safety_factor": safety,
                "uses_teacher_outcomes": False,
            }
        )

    return {
        "schema_version": SUPPORT_EXPANSION_SCHEMA_VERSION,
        "stage": "outcome_blind_candidate_pool_expansion_plan",
        "source_support_schema_version": SUPPORT_AUDIT_SCHEMA_VERSION,
        "uses_teacher_outcomes": False,
        "maximum_additional_attempts_per_stratum": maximum,
        "allocations": allocations,
        "skipped_gaps": skipped,
    }
