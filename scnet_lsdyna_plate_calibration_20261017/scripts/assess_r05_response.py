"""Fail-closed cohort gates for the preregistered R05 P0 response design.

The assessor consumes one identity-bound response payload. It deliberately does
not accept a bare list of case records: every case in one assessment must share a
single artifact, solver-run, and parser identity. The real D3PART reader remains
a separate parser-acceptance boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


EXPECTED_PACKAGE_REVISION = "R05_capture_ledger_preregistered"
EXPECTED_DESIGN_CANONICAL_SHA256 = (
    "sha256:85b35f1a23ac0b822d0ce65fd9232327c9e566e5cfdddb080814e01414ebdf97"
)
EXPECTED_DESIGN_SHA256 = (
    "sha256:b1d2027d57fc6d9cd1ee3f932ea044fd16a112344d8f2b6f508071f45191a06f"
)
EXPECTED_DECK_SHA256 = (
    "sha256:6fbd753a113d0214e7ba4ff5136dc526d151e6f2c52c94e70399b29c4b9d3d0c"
)
EXPECTED_CASE_MAP_SHA256 = (
    "sha256:b05b59249e49c3e9d15904f475c2f9478bdff1638fb797d1b552298490aa5295"
)
EXPECTED_PACKAGE_MANIFEST_SHA256 = (
    "sha256:e178b6c28832519aaa2befdc969dcd93d2141ea4073ad740b52b17968d92ba44"
)
EXPECTED_PAYLOAD_SCHEMA = "hf-p0-r05-case-results@1.0.0"
EXPECTED_LEDGER_SCHEMA = "hf-p0-capture-ledger-result@1.1.0"

FROZEN_CASES = (
    ("p0_v0600_h0p75", 600.0, 0.75),
    ("p0_v0800_h0p75", 800.0, 0.75),
    ("p0_v1000_h0p75", 1000.0, 0.75),
    ("p0_v1200_h0p75", 1200.0, 0.75),
    ("p0_v2000_h0p75", 2000.0, 0.75),
    ("p0_v2000_h1p00", 2000.0, 1.0),
    ("p0_v2000_h0p50", 2000.0, 0.5),
)
FROZEN_SPEED_CASE_IDS = (
    "p0_v0600_h0p75",
    "p0_v0800_h0p75",
    "p0_v1000_h0p75",
    "p0_v1200_h0p75",
    "p0_v2000_h0p75",
)
FROZEN_MESH_CASE_IDS = (
    "p0_v2000_h1p00",
    "p0_v2000_h0p75",
    "p0_v2000_h0p50",
)
FROZEN_CASE_BY_ID = {
    case_id: {"impact_speed_mps": speed, "mesh_mm": mesh}
    for case_id, speed, mesh in FROZEN_CASES
}

FROZEN_CAPTURE_LIMITS = {
    "initial_discrete_mass_relative_error_max": 0.005,
    "mass_accounting_relative_error_max": 0.000001,
    "temporal_ambiguity_mass_fraction_max": 0.01,
    "deleted_upstream_mass_fraction_max": 0.05,
    "tail_increment_qM_max": 0.01,
    "tail_increment_qP_max": 0.01,
    "tail_increment_qE_max": 0.01,
    "tail_qV_relative_range_max": 0.05,
}
FROZEN_CLASSIFICATION_LIMITS = {
    "confirmed_perforation_qM_min": 0.02,
    "confirmed_perforation_qP_min": 0.01,
    "confirmed_nonperforation_qM_max": 0.005,
    "confirmed_nonperforation_qP_max": 0.005,
    "confirmed_nonperforation_worst_case_crossing_mass_max": 0.005,
}
FROZEN_SPEED_LIMITS = {
    "minimum_confirmed_perforation_points": 1,
    "minimum_confirmed_nonperforation_points": 1,
    "maximum_nearest_bracket_width_mps": 200.0,
    "forbid_perforation_to_nonperforation_reversal": True,
    "confirmed_perforation_qV_downward_jump_mps_max": 100.0,
}
FROZEN_MESH_LIMITS = {
    "coarse_to_medium_absolute_difference_max": 0.10,
    "medium_to_fine_absolute_difference_max": 0.05,
}

KNOWN_CLASSIFICATIONS = {
    "confirmed_perforation",
    "confirmed_nonperforation",
    "transition_or_indeterminate",
    "unavailable",
}
LEDGER_STATES = {
    "alive_crossed",
    "alive_upstream",
    "deleted_upstream",
    "deleted_after_crossing",
    "temporal_ambiguity",
}
TAIL_FIELDS = {"increment_qM", "increment_qP", "increment_qE", "qV_relative_range"}
SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _nonnegative_finite(value: Any) -> bool:
    return _finite_number(value) and float(value) >= 0.0


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_json(item, f"{path}[{index}]")


def strict_json_loads(text: str) -> Any:
    """Parse JSON while rejecting duplicate keys and every non-finite number."""

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON token is forbidden: {token}")

    value = json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    # A syntactically valid exponent such as 1e999 becomes infinity in Python.
    _reject_nonfinite_json(value)
    return value


def _fail(reason: str) -> dict:
    return {"status": "fail", "reasons": [reason]}


def _gate(reasons: list[str], **details: Any) -> dict:
    return {"status": "pass" if not reasons else "fail", "reasons": reasons, **details}


def _validate_exact_object(
    value: Any,
    *,
    name: str,
    expected_keys: set[str],
    reasons: list[str],
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        reasons.append(f"{name} must be an object")
        return None
    keys = set(value)
    missing = sorted(expected_keys - keys)
    extra = sorted(keys - expected_keys)
    if missing:
        reasons.append(f"{name} missing fields: {', '.join(missing)}")
    if extra:
        reasons.append(f"{name} has unexpected fields: {', '.join(extra)}")
    return value


def _valid_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_identity_payload(payload: Any) -> tuple[dict, list[dict]]:
    reasons: list[str] = []
    if not isinstance(payload, dict):
        return _gate(["response payload must be an object"]), []

    required_top = {
        "schema_version",
        "artifact_identity",
        "run_identity",
        "parser_identity",
        "cases",
    }
    extra_top = sorted(set(payload) - required_top)
    missing_top = sorted(required_top - set(payload))
    if missing_top:
        reasons.append("response payload missing fields: " + ", ".join(missing_top))
    if extra_top:
        reasons.append("response payload has unexpected fields: " + ", ".join(extra_top))
    if payload.get("schema_version") != EXPECTED_PAYLOAD_SCHEMA:
        reasons.append("response payload schema_version is not the frozen schema")

    artifact = _validate_exact_object(
        payload.get("artifact_identity"),
        name="artifact_identity",
        expected_keys={
            "package_revision",
            "design_sha256",
            "design_canonical_sha256",
            "deck_sha256",
            "case_map_sha256",
            "package_manifest_sha256",
        },
        reasons=reasons,
    )
    if artifact is not None:
        expected_artifact_values = {
            "package_revision": EXPECTED_PACKAGE_REVISION,
            "design_sha256": EXPECTED_DESIGN_SHA256,
            "design_canonical_sha256": EXPECTED_DESIGN_CANONICAL_SHA256,
            "deck_sha256": EXPECTED_DECK_SHA256,
            "case_map_sha256": EXPECTED_CASE_MAP_SHA256,
            "package_manifest_sha256": EXPECTED_PACKAGE_MANIFEST_SHA256,
        }
        for field, expected in expected_artifact_values.items():
            if artifact.get(field) != expected:
                reasons.append(f"artifact_identity.{field} does not match the frozen artifact")
    run = _validate_exact_object(
        payload.get("run_identity"),
        name="run_identity",
        expected_keys={"run_id", "solver_build", "returned_artifact_sha256"},
        reasons=reasons,
    )
    if run is not None:
        for field in ("run_id", "solver_build"):
            if not _valid_nonempty_string(run.get(field)):
                reasons.append(f"run_identity.{field} must be a non-empty string")
        result_hash = run.get("returned_artifact_sha256")
        if not isinstance(result_hash, str) or not SHA256_RE.fullmatch(result_hash):
            reasons.append("run_identity.returned_artifact_sha256 is not a lowercase sha256 digest")

    parser = _validate_exact_object(
        payload.get("parser_identity"),
        name="parser_identity",
        expected_keys={"name", "version", "source_sha256", "acceptance_status"},
        reasons=reasons,
    )
    if parser is not None:
        for field in ("name", "version"):
            if not _valid_nonempty_string(parser.get(field)):
                reasons.append(f"parser_identity.{field} must be a non-empty string")
        source_hash = parser.get("source_sha256")
        if not isinstance(source_hash, str) or not SHA256_RE.fullmatch(source_hash):
            reasons.append("parser_identity.source_sha256 is not a lowercase sha256 digest")
        if parser.get("acceptance_status") != "pass":
            reasons.append("parser_identity.acceptance_status must be pass")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        reasons.append("response payload cases must be a list")
        raw_cases = []
    return _gate(reasons), raw_cases


def _recomputed_unavailable_reasons(record: dict, reasons: list[str]) -> list[str] | None:
    scalar_fields = (
        "initial_discrete_mass_relative_error",
        "mass_accounting_relative_error",
        "missing_reference_mass_fraction",
        "unresolved_crossing_mass_fraction",
        "worst_case_crossing_mass_fraction",
    )
    for field in scalar_fields:
        if not _nonnegative_finite(record.get(field)):
            reasons.append(f"{field} must be a finite nonnegative number")

    states = record.get("mass_fraction_by_ledger_state")
    if not isinstance(states, dict):
        reasons.append("mass_fraction_by_ledger_state must be an object")
        states = None
    elif set(states) != LEDGER_STATES:
        reasons.append("mass_fraction_by_ledger_state must contain exactly the frozen ledger states")
    if states is not None:
        for state in LEDGER_STATES:
            if not _nonnegative_finite(states.get(state)):
                reasons.append(f"mass_fraction_by_ledger_state.{state} must be finite and nonnegative")

    tail = record.get("tail_window")
    if not isinstance(tail, dict):
        reasons.append("tail_window must be an object")
        tail = None
    elif set(tail) != TAIL_FIELDS:
        reasons.append("tail_window must contain exactly the frozen diagnostic fields")
    if tail is not None:
        for field in TAIL_FIELDS:
            if not _nonnegative_finite(tail.get(field)):
                reasons.append(f"tail_window.{field} must be finite and nonnegative")

    if (
        states is None
        or tail is None
        or any(not _nonnegative_finite(record.get(field)) for field in scalar_fields)
        or any(not _nonnegative_finite(states.get(state)) for state in LEDGER_STATES)
        or any(not _nonnegative_finite(tail.get(field)) for field in TAIL_FIELDS)
    ):
        return None

    observed_fraction = sum(float(states[state]) for state in LEDGER_STATES)
    expected_initial_error = abs(observed_fraction - 1.0)
    expected_missing = max(0.0, 1.0 - observed_fraction)
    expected_unresolved = (
        float(states["temporal_ambiguity"])
        + float(states["deleted_upstream"])
        + expected_missing
    )
    if not _close(record["initial_discrete_mass_relative_error"], expected_initial_error):
        reasons.append("initial_discrete_mass_relative_error is inconsistent with ledger fractions")
    if not _close(record["missing_reference_mass_fraction"], expected_missing):
        reasons.append("missing_reference_mass_fraction is inconsistent with ledger fractions")
    if not _close(record["unresolved_crossing_mass_fraction"], expected_unresolved):
        reasons.append("unresolved_crossing_mass_fraction is inconsistent with ledger fractions")

    unavailable: list[str] = []
    if float(record["initial_discrete_mass_relative_error"]) > FROZEN_CAPTURE_LIMITS[
        "initial_discrete_mass_relative_error_max"
    ]:
        unavailable.append("initial_discrete_mass_relative_error")
    if float(record["mass_accounting_relative_error"]) > FROZEN_CAPTURE_LIMITS[
        "mass_accounting_relative_error_max"
    ]:
        unavailable.append("mass_accounting_error")
    if float(states["temporal_ambiguity"]) > FROZEN_CAPTURE_LIMITS[
        "temporal_ambiguity_mass_fraction_max"
    ]:
        unavailable.append("temporal_ambiguity_mass")
    if float(states["deleted_upstream"]) > FROZEN_CAPTURE_LIMITS[
        "deleted_upstream_mass_fraction_max"
    ]:
        unavailable.append("deleted_upstream_mass")
    for suffix in ("qM", "qP", "qE"):
        if float(tail[f"increment_{suffix}"]) > FROZEN_CAPTURE_LIMITS[
            f"tail_increment_{suffix}_max"
        ]:
            unavailable.append(f"unstable_tail_{suffix}")
    if float(tail["qV_relative_range"]) > FROZEN_CAPTURE_LIMITS[
        "tail_qV_relative_range_max"
    ]:
        unavailable.append("unstable_tail_qV")
    return unavailable


def _recompute_classification(qM: float, qP: float, unresolved: float) -> str:
    if (
        qM >= FROZEN_CLASSIFICATION_LIMITS["confirmed_perforation_qM_min"]
        and qP >= FROZEN_CLASSIFICATION_LIMITS["confirmed_perforation_qP_min"]
    ):
        return "confirmed_perforation"
    if (
        qM <= FROZEN_CLASSIFICATION_LIMITS["confirmed_nonperforation_qM_max"]
        and qP <= FROZEN_CLASSIFICATION_LIMITS["confirmed_nonperforation_qP_max"]
        and qM + unresolved
        <= FROZEN_CLASSIFICATION_LIMITS[
            "confirmed_nonperforation_worst_case_crossing_mass_max"
        ]
    ):
        return "confirmed_nonperforation"
    return "transition_or_indeterminate"


def _validate_case_record(record: dict, expected: dict) -> tuple[list[str], dict, str | None]:
    reasons: list[str] = []
    required_fields = {
        "ledger_schema_version",
        "case_id",
        "impact_speed_mps",
        "mesh_mm",
        "primary_qoi_available",
        "unavailable_reasons",
        "run_integrity_gate",
        "classification",
        "qM",
        "qP",
        "qE",
        "qV_over_v0",
        "initial_discrete_mass_relative_error",
        "mass_accounting_relative_error",
        "mass_fraction_by_ledger_state",
        "missing_reference_mass_fraction",
        "unresolved_crossing_mass_fraction",
        "worst_case_crossing_mass_fraction",
        "tail_window",
    }
    missing = sorted(required_fields - set(record))
    if missing:
        reasons.append("missing required fields: " + ", ".join(missing))
    forbidden_local_identities = sorted(
        {"artifact_identity", "run_identity", "parser_identity"} & set(record)
    )
    if forbidden_local_identities:
        reasons.append(
            "case-local identities are forbidden; use the single top-level binding: "
            + ", ".join(forbidden_local_identities)
        )

    if record.get("ledger_schema_version") != EXPECTED_LEDGER_SCHEMA:
        reasons.append("ledger_schema_version is not the frozen ledger-result schema")
    for field in ("impact_speed_mps", "mesh_mm"):
        value = record.get(field)
        if not _finite_number(value) or float(value) != float(expected[field]):
            reasons.append(f"{field} does not match the frozen case design")
    if not isinstance(record.get("primary_qoi_available"), bool):
        reasons.append("primary_qoi_available must be boolean")
    if record.get("run_integrity_gate") not in {"pass", "fail"}:
        reasons.append("run_integrity_gate must be pass or fail")
    if record.get("classification") not in KNOWN_CLASSIFICATIONS:
        reasons.append("classification is unknown")

    reported_unavailable = record.get("unavailable_reasons")
    if not isinstance(reported_unavailable, list) or any(
        not _valid_nonempty_string(item) for item in reported_unavailable
    ):
        reasons.append("unavailable_reasons must be a list of non-empty strings")
        reported_unavailable = None
    elif len(reported_unavailable) != len(set(reported_unavailable)):
        reasons.append("unavailable_reasons must not contain duplicates")

    recomputed_unavailable = _recomputed_unavailable_reasons(record, reasons)
    recomputed_available = recomputed_unavailable == [] if recomputed_unavailable is not None else False
    if isinstance(record.get("primary_qoi_available"), bool) and (
        record["primary_qoi_available"] != recomputed_available
    ):
        reasons.append("primary_qoi_available does not match the frozen recomputed availability gate")
    if reported_unavailable is not None and recomputed_unavailable is not None and (
        reported_unavailable != recomputed_unavailable
    ):
        reasons.append("unavailable_reasons do not match the frozen recomputed reasons")

    qM = record.get("qM")
    qP = record.get("qP")
    qE = record.get("qE")
    qV = record.get("qV_over_v0")
    recomputed_classification: str | None = None
    if recomputed_available:
        for field, value in (("qM", qM), ("qP", qP), ("qE", qE)):
            if not _nonnegative_finite(value):
                reasons.append(f"{field} must be finite and nonnegative when primary QoI are available")
        if all(_nonnegative_finite(value) for value in (qM, qP, qE)):
            states = record.get("mass_fraction_by_ledger_state")
            unresolved = record.get("unresolved_crossing_mass_fraction")
            worst = record.get("worst_case_crossing_mass_fraction")
            if isinstance(states, dict) and all(
                _nonnegative_finite(states.get(state)) for state in LEDGER_STATES
            ):
                expected_qM = float(states["alive_crossed"]) + float(
                    states["deleted_after_crossing"]
                )
                if not _close(qM, expected_qM):
                    reasons.append("qM is inconsistent with crossing ledger fractions")
            if _nonnegative_finite(unresolved) and _nonnegative_finite(worst) and not _close(
                worst, float(qM) + float(unresolved)
            ):
                reasons.append("worst_case_crossing_mass_fraction is inconsistent with qM")

            if float(qM) >= FROZEN_CLASSIFICATION_LIMITS["confirmed_perforation_qM_min"]:
                if not _nonnegative_finite(qV):
                    reasons.append("qV_over_v0 must be finite when qM reaches the frozen threshold")
                elif not _close(float(qV), float(qP) / float(qM)):
                    reasons.append("qV_over_v0 is inconsistent with qP/qM")
            elif qV is not None:
                reasons.append("qV_over_v0 must be null below the frozen qM threshold")

            if _nonnegative_finite(unresolved):
                recomputed_classification = _recompute_classification(
                    float(qM), float(qP), float(unresolved)
                )
    else:
        if any(value is not None for value in (qM, qP, qE, qV)):
            reasons.append("primary QoI must all be null when the recomputed availability gate fails")
        recomputed_classification = "unavailable"

    if recomputed_classification is not None and record.get("classification") != recomputed_classification:
        reasons.append("reported classification does not match recomputed classification")

    normalized = dict(record)
    if recomputed_classification is not None:
        normalized["classification"] = recomputed_classification
    normalized["primary_qoi_available"] = recomputed_available
    return reasons, normalized, recomputed_classification


def _case_metric(record: dict, name: str) -> float | None:
    if name == "deleted_upstream_mass_fraction":
        states = record.get("mass_fraction_by_ledger_state")
        value = states.get("deleted_upstream") if isinstance(states, dict) else None
    else:
        value = record.get(name)
    return float(value) if _finite_number(value) else None


def _assess_r05_response(design: Any, response_payload: Any) -> dict:
    design_reasons: list[str] = []
    try:
        design_digest = _canonical_sha256(design)
    except (TypeError, ValueError, OverflowError):
        design_digest = None
    if not isinstance(design, dict) or design_digest != EXPECTED_DESIGN_CANONICAL_SHA256:
        design_reasons.append("design does not match the exact frozen R05 canonical contract")
    design_gate = _gate(
        design_reasons,
        expected_canonical_sha256=EXPECTED_DESIGN_CANONICAL_SHA256,
        observed_canonical_sha256=design_digest,
    )

    identity_binding_gate, raw_cases = _validate_identity_payload(response_payload)
    records: dict[str, dict] = {}
    identity_reasons: list[str] = []
    contract_reasons: list[str] = []
    case_contract_status: dict[str, str] = {}
    recomputed_classifications: dict[str, str | None] = {}

    for index, raw_record in enumerate(raw_cases):
        if not isinstance(raw_record, dict):
            contract_reasons.append(f"case record at index {index} must be an object")
            continue
        case_id = raw_record.get("case_id")
        if not isinstance(case_id, str) or case_id not in FROZEN_CASE_BY_ID:
            identity_reasons.append(f"unexpected or missing case_id at index {index}: {case_id!r}")
            continue
        if case_id in records:
            identity_reasons.append(f"duplicate case result: {case_id}")
            continue
        case_reasons, normalized, recomputed = _validate_case_record(
            raw_record, FROZEN_CASE_BY_ID[case_id]
        )
        records[case_id] = normalized
        recomputed_classifications[case_id] = recomputed
        case_contract_status[case_id] = "pass" if not case_reasons else "fail"
        contract_reasons.extend(f"{case_id}: {reason}" for reason in case_reasons)

    missing = sorted(set(FROZEN_CASE_BY_ID) - set(records))
    if missing:
        identity_reasons.append("missing case results: " + ", ".join(missing))
    case_identity_gate = _gate(identity_reasons)
    per_case_contract_gate = _gate(
        contract_reasons,
        case_status=case_contract_status,
        recomputed_classifications=recomputed_classifications,
    )

    unavailable_case_ids = [
        case_id
        for case_id in FROZEN_CASE_BY_ID
        if case_id not in records
        or case_contract_status.get(case_id) != "pass"
        or records[case_id].get("primary_qoi_available") is not True
    ]
    availability_gate = _gate(
        ["one or more cases lack a valid available primary QoI set"] if unavailable_case_ids else [],
        unavailable_case_ids=unavailable_case_ids,
    )
    failed_run_ids = [
        case_id
        for case_id in FROZEN_CASE_BY_ID
        if case_id not in records
        or case_contract_status.get(case_id) != "pass"
        or records[case_id].get("run_integrity_gate") != "pass"
    ]
    run_integrity_gate = _gate(
        ["one or more cases lack a valid passing run-integrity result"] if failed_run_ids else [],
        failed_or_missing_case_ids=failed_run_ids,
    )

    if any(
        case_id not in records or case_contract_status.get(case_id) != "pass"
        for case_id in FROZEN_SPEED_CASE_IDS
    ):
        speed_gate = _fail("one or more frozen speed-scan cases are missing or violate the per-case contract")
    else:
        speed_records = sorted(
            (records[case_id] for case_id in FROZEN_SPEED_CASE_IDS),
            key=lambda item: float(item["impact_speed_mps"]),
        )
        perforated = [r for r in speed_records if r["classification"] == "confirmed_perforation"]
        nonperforated = [
            r for r in speed_records if r["classification"] == "confirmed_nonperforation"
        ]
        # Only adjacent samples in the complete sorted scan form a valid bracket.
        adjacent_brackets = [
            (float(lower["impact_speed_mps"]), float(upper["impact_speed_mps"]))
            for lower, upper in zip(speed_records, speed_records[1:])
            if lower["classification"] == "confirmed_nonperforation"
            and upper["classification"] == "confirmed_perforation"
        ]
        nearest_bracket = min(
            adjacent_brackets,
            key=lambda pair: pair[1] - pair[0],
            default=None,
        )
        reversal_pairs = [
            (float(a["impact_speed_mps"]), float(b["impact_speed_mps"]))
            for index, a in enumerate(speed_records)
            for b in speed_records[index + 1 :]
            if a["classification"] == "confirmed_perforation"
            and b["classification"] == "confirmed_nonperforation"
        ]
        adjacent_qv_checks: list[dict] = []
        qv_failure = False
        for lower, upper in zip(speed_records, speed_records[1:]):
            if (
                lower["classification"] == "confirmed_perforation"
                and upper["classification"] == "confirmed_perforation"
            ):
                lower_ratio = _case_metric(lower, "qV_over_v0")
                upper_ratio = _case_metric(upper, "qV_over_v0")
                if lower_ratio is None or upper_ratio is None:
                    downward_jump = None
                    pair_pass = False
                else:
                    lower_qv = lower_ratio * float(lower["impact_speed_mps"])
                    upper_qv = upper_ratio * float(upper["impact_speed_mps"])
                    downward_jump = max(0.0, lower_qv - upper_qv)
                    pair_pass = downward_jump <= FROZEN_SPEED_LIMITS[
                        "confirmed_perforation_qV_downward_jump_mps_max"
                    ]
                qv_failure = qv_failure or not pair_pass
                adjacent_qv_checks.append(
                    {
                        "lower_case_id": lower["case_id"],
                        "upper_case_id": upper["case_id"],
                        "downward_jump_mps": downward_jump,
                        "status": "pass" if pair_pass else "fail",
                    }
                )
        reasons: list[str] = []
        if len(perforated) < FROZEN_SPEED_LIMITS["minimum_confirmed_perforation_points"]:
            reasons.append("insufficient confirmed-perforation points")
        if len(nonperforated) < FROZEN_SPEED_LIMITS["minimum_confirmed_nonperforation_points"]:
            reasons.append("insufficient confirmed-nonperforation points")
        if nearest_bracket is None:
            reasons.append("no adjacent confirmed nonperforation-to-perforation bracket")
        elif nearest_bracket[1] - nearest_bracket[0] > FROZEN_SPEED_LIMITS[
            "maximum_nearest_bracket_width_mps"
        ]:
            reasons.append("nearest adjacent confirmed bracket exceeds the frozen width")
        if FROZEN_SPEED_LIMITS["forbid_perforation_to_nonperforation_reversal"] and reversal_pairs:
            reasons.append("confirmed perforation-to-nonperforation reversal exists")
        if qv_failure:
            reasons.append(
                "an adjacent confirmed-perforation qV pair exceeds or cannot evaluate the frozen jump limit"
            )
        speed_gate = _gate(
            reasons,
            confirmed_perforation_case_ids=[r["case_id"] for r in perforated],
            confirmed_nonperforation_case_ids=[r["case_id"] for r in nonperforated],
            adjacent_brackets_mps=[list(pair) for pair in adjacent_brackets],
            nearest_bracket_mps=list(nearest_bracket) if nearest_bracket else None,
            nearest_bracket_width_mps=(
                nearest_bracket[1] - nearest_bracket[0] if nearest_bracket else None
            ),
            reversal_pairs_mps=[list(pair) for pair in reversal_pairs],
            adjacent_confirmed_perforation_qV_checks=adjacent_qv_checks,
        )

    if any(
        case_id not in records or case_contract_status.get(case_id) != "pass"
        for case_id in FROZEN_MESH_CASE_IDS
    ):
        mesh_gate = _fail("one or more frozen mesh-scan cases are missing or violate the per-case contract")
    else:
        mesh_records = sorted(
            (records[case_id] for case_id in FROZEN_MESH_CASE_IDS),
            key=lambda item: float(item["mesh_mm"]),
            reverse=True,
        )
        mesh_reasons: list[str] = []
        if any(record.get("primary_qoi_available") is not True for record in mesh_records):
            mesh_reasons.append("one or more mesh-scan primary QoI sets are unavailable")
        classifications = [record["classification"] for record in mesh_records]
        if "unavailable" in classifications:
            mesh_reasons.append("mesh-scan classification unavailable is forbidden")
        if len(set(classifications)) != 1:
            mesh_reasons.append("mesh-scan classifications do not match")
        metric_checks: dict[str, dict] = {}
        for metric in ("qM", "qP", "qE", "qV_over_v0", "deleted_upstream_mass_fraction"):
            values = [_case_metric(record, metric) for record in mesh_records]
            if any(value is None for value in values):
                coarse_medium = None
                medium_fine = None
                metric_pass = False
            else:
                coarse_medium = abs(values[0] - values[1])
                medium_fine = abs(values[1] - values[2])
                metric_pass = (
                    coarse_medium
                    <= FROZEN_MESH_LIMITS["coarse_to_medium_absolute_difference_max"]
                    and medium_fine
                    <= FROZEN_MESH_LIMITS["medium_to_fine_absolute_difference_max"]
                )
            if not metric_pass:
                mesh_reasons.append(f"{metric} fails or cannot evaluate the frozen mesh limits")
            metric_checks[metric] = {
                "values_coarse_medium_fine": values,
                "coarse_to_medium_absolute_difference": coarse_medium,
                "medium_to_fine_absolute_difference": medium_fine,
                "status": "pass" if metric_pass else "fail",
            }
        mesh_gate = _gate(
            mesh_reasons,
            ordered_case_ids_coarse_to_fine=[r["case_id"] for r in mesh_records],
            classifications=classifications,
            metric_checks=metric_checks,
        )

    gates = {
        "frozen_design_contract": design_gate,
        "top_level_identity_binding": identity_binding_gate,
        "case_identity": case_identity_gate,
        "per_case_contract": per_case_contract_gate,
        "all_case_capture_ledger_availability": availability_gate,
        "all_case_run_integrity": run_integrity_gate,
        "speed_scan": speed_gate,
        "mesh_scan_2000_mps": mesh_gate,
    }
    all_pass = all(gate["status"] == "pass" for gate in gates.values())
    bound_identity = None
    if identity_binding_gate["status"] == "pass" and isinstance(response_payload, dict):
        bound_identity = {
            name: dict(response_payload[name])
            for name in ("artifact_identity", "run_identity", "parser_identity")
        }
    return {
        "schema_version": "hf-p0-r05-response-assessment@1.1.0",
        "package_revision": EXPECTED_PACKAGE_REVISION,
        "bound_identity": bound_identity,
        "overall_status": "pass" if all_pass else "fail_closed",
        "gates": gates,
        "p0_response_calibration_admission": "allowed" if all_pass else "denied",
        "p1_submission_admission": "eligible_for_separate_review" if all_pass else "denied",
        "claim_boundary": (
            "Deterministic numerical qualification only; this assessment does not establish "
            "experimental V50, material calibration, or physical validity."
        ),
    }


def assess_r05_response(design: Any, response_payload: Any) -> dict:
    """Apply frozen R05 gates and convert every malformed input into denial."""
    try:
        return _assess_r05_response(design, response_payload)
    except Exception as exc:  # The admission boundary must never fail open on malformed input.
        reason = f"malformed input caused fail-closed validation: {type(exc).__name__}: {exc}"
        gates = {
            "frozen_design_contract": _fail(reason),
            "top_level_identity_binding": _fail(reason),
            "case_identity": _fail(reason),
            "per_case_contract": _fail(reason),
            "all_case_capture_ledger_availability": _fail(reason),
            "all_case_run_integrity": _fail(reason),
            "speed_scan": _fail(reason),
            "mesh_scan_2000_mps": _fail(reason),
        }
        return {
            "schema_version": "hf-p0-r05-response-assessment@1.1.0",
            "package_revision": EXPECTED_PACKAGE_REVISION,
            "bound_identity": None,
            "overall_status": "fail_closed",
            "gates": gates,
            "p0_response_calibration_admission": "denied",
            "p1_submission_admission": "denied",
            "claim_boundary": (
                "Deterministic numerical qualification only; malformed inputs cannot establish "
                "experimental V50, material calibration, or physical validity."
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--case-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        design = strict_json_loads(args.design.read_text(encoding="utf-8"))
        payload = strict_json_loads(args.case_results.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"strict JSON input rejected: {exc}")
    result = assess_r05_response(design, payload)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
