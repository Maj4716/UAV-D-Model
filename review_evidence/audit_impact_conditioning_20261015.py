"""Read-only audit of the frozen 5,000-case impact-conditioning contract.

This script checks persisted formal cases against the current impact geometry
implementation.  It intentionally does not execute fragment/blast simulation or
rewrite any formal dataset artifact.
"""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import damage_engine  # noqa: E402
from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine  # noqa: E402
from damage_engine import impact as impact_module  # noqa: E402
from damage_engine.impact import (  # noqa: E402
    IMPACT_MODEL_ID,
    IMPACT_TRIGGER,
    is_impact_case,
    validate_impact_detonation,
)
from damage_engine.registry import PhysicsRegistry  # noqa: E402
from damage_engine.utils import quaternion_matrix_xyzw  # noqa: E402


DATASET_ROOT = (
    REPOSITORY_ROOT
    / "raw_data"
    / "dataset_5000_loitering_impact_fragment_only_v1"
)
FORMAL_CASES = DATASET_ROOT / "rounds" / "round-000" / "formal_cases.jsonl.gz"
ROUND_MANIFEST = DATASET_ROOT / "rounds" / "round-000" / "round_manifest.json"
DATASET_MANIFEST = DATASET_ROOT / "formal_dataset" / "manifest.json"
FROZEN_HASHES = DATASET_ROOT / "plan" / "frozen_hashes.json"
TARGET = REPOSITORY_ROOT / "raw_data" / "target.rebuilt.fragment_only.v1.json"
PHYSICS_REGISTRY = (
    REPOSITORY_ROOT / "examples" / "physics.loitering_impact.json"
)
OUTPUT = REPOSITORY_ROOT / "review_evidence" / "impact_conditioning_audit_20261015.json"
AUDIT_SCRIPT = Path(__file__).resolve()

AUDIT_DATE = "2026-10-15"
MAX_FAILURE_EXAMPLES = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def _finite_vector(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vector.shape != (3,) or not np.isfinite(vector).all():
        return None
    return vector


def _update_extrema(
    extrema: dict[str, dict[str, float | None]], name: str, value: float
) -> None:
    if not math.isfinite(value):
        return
    record = extrema.setdefault(name, {"minimum": None, "maximum": None})
    current_minimum = record["minimum"]
    current_maximum = record["maximum"]
    if current_minimum is None or value < current_minimum:
        record["minimum"] = float(value)
    if current_maximum is None or value > current_maximum:
        record["maximum"] = float(value)


def main() -> None:
    input_hash_before = _sha256(FORMAL_CASES)
    round_manifest = _load_json(ROUND_MANIFEST)
    dataset_manifest = _load_json(DATASET_MANIFEST)
    frozen_hashes = _load_json(FROZEN_HASHES)
    target_data = _load_json(TARGET)
    registry = PhysicsRegistry.from_file(PHYSICS_REGISTRY)
    engine = DamageEngine(target_data, registry)

    alignment_minimum = float(impact_module._ALIGNMENT_COSINE_MINIMUM)
    position_tolerance_m = float(impact_module._POSITION_TOLERANCE_M)
    armor_ids = {item.object_id for item in engine.compiled_target.armor}

    criterion_names = (
        "json_record_parsed",
        "first_armor_contact_trigger",
        "velocity_finite_nonzero",
        "body_plus_x_aligned_with_velocity",
        "origin_outside_target_armor",
        "forward_ray_has_armor_intersection",
        "computed_first_collision_nonnegative",
        "persisted_collision_distance_present_finite_nonnegative",
        "persisted_collision_id_present_and_in_target",
        "persisted_collision_id_matches_computed_first_contact",
        "current_validate_impact_detonation_pass",
        "all_requested_conditions_pass",
    )
    passed = Counter({name: 0 for name in criterion_names})
    failure_reasons: Counter[str] = Counter()
    failure_examples: list[dict[str, Any]] = []
    trigger_counts: Counter[str] = Counter()
    collision_id_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    velocity_direction_counts: Counter[str] = Counter()
    body_orientation_counts: Counter[str] = Counter()
    extrema: dict[str, dict[str, float | None]] = {}
    seen_case_ids: set[str] = set()
    duplicate_case_ids: set[str] = set()
    total_lines = 0

    with gzip.open(FORMAL_CASES, "rt", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            total_lines += 1
            try:
                case = json.loads(raw_line)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                failure_reasons["json_record_parsed"] += 1
                if len(failure_examples) < MAX_FAILURE_EXAMPLES:
                    failure_examples.append(
                        {
                            "line_number": line_number,
                            "case_id": None,
                            "failed_criteria": ["json_record_parsed"],
                            "exception_type": type(error).__name__,
                            "exception_message": str(error),
                        }
                    )
                continue

            case_id = str(case.get("case_id", ""))
            if case_id in seen_case_ids:
                duplicate_case_ids.add(case_id)
            seen_case_ids.add(case_id)
            encounter = case.get("encounter") or {}
            trigger = str(encounter.get("detonation_trigger", "<missing>"))
            trigger_counts[trigger] += 1
            profile_counts[str((case.get("munition") or {}).get("profile_id", "<missing>"))] += 1
            split_counts[str((case.get("metadata") or {}).get("frozen_split", "<missing>"))] += 1

            case_checks = {name: False for name in criterion_names}
            case_checks["json_record_parsed"] = True
            validator_error: Exception | None = None

            case_checks["first_armor_contact_trigger"] = is_impact_case(case)

            velocity = _finite_vector(encounter.get("velocity_target_mps"))
            speed: float | None = None
            direction: np.ndarray | None = None
            if velocity is not None:
                speed = float(np.linalg.norm(velocity))
                if math.isfinite(speed) and speed > impact_module.EPS:
                    direction = velocity / speed
                    case_checks["velocity_finite_nonzero"] = True
                    _update_extrema(extrema, "speed_mps", speed)
                    direction_key = json.dumps(
                        [round(float(value), 12) for value in direction.tolist()],
                        separators=(",", ":"),
                    )
                    velocity_direction_counts[direction_key] += 1

            if direction is not None:
                try:
                    raw_orientation = np.asarray(
                        encounter.get("body_to_target_orientation_xyzw"),
                        dtype=np.float64,
                    )
                    if raw_orientation.shape == (4,) and np.isfinite(
                        raw_orientation
                    ).all():
                        orientation_norm = float(np.linalg.norm(raw_orientation))
                        if orientation_norm > impact_module.EPS:
                            canonical_orientation = raw_orientation / orientation_norm
                            first_nonzero = next(
                                (
                                    value
                                    for value in canonical_orientation
                                    if abs(float(value)) > 1e-15
                                ),
                                1.0,
                            )
                            if first_nonzero < 0.0:
                                canonical_orientation = -canonical_orientation
                            orientation_key = json.dumps(
                                [
                                    round(float(value), 12)
                                    for value in canonical_orientation.tolist()
                                ],
                                separators=(",", ":"),
                            )
                            body_orientation_counts[orientation_key] += 1
                    rotation = quaternion_matrix_xyzw(
                        encounter.get("body_to_target_orientation_xyzw")
                    )
                    body_forward = rotation @ np.asarray([1.0, 0.0, 0.0])
                    alignment = float(np.dot(body_forward, direction))
                    _update_extrema(extrema, "alignment_cosine", alignment)
                    if -1.0 <= alignment <= 1.0:
                        angular_error_deg = math.degrees(math.acos(np.clip(alignment, -1.0, 1.0)))
                        _update_extrema(
                            extrema, "body_velocity_angular_error_deg", angular_error_deg
                        )
                    case_checks["body_plus_x_aligned_with_velocity"] = (
                        math.isfinite(alignment) and alignment >= alignment_minimum
                    )
                except (KeyError, TypeError, ValueError):
                    pass

            origin = _finite_vector(encounter.get("trajectory_origin_m"))
            detonation_position = _finite_vector(
                encounter.get("detonation_position_m")
            )
            if detonation_position is not None:
                for axis, value in zip("xyz", detonation_position, strict=True):
                    _update_extrema(
                        extrema,
                        f"detonation_position_{axis}_m",
                        float(value),
                    )
            first_hit = None
            if origin is not None:
                for axis, value in zip("xyz", origin, strict=True):
                    _update_extrema(
                        extrema,
                        f"trajectory_origin_{axis}_m",
                        float(value),
                    )
                containing_armor = [
                    item.object_id
                    for item in engine.compiled_target.armor
                    if item.geometry.contains(origin)
                ]
                case_checks["origin_outside_target_armor"] = not containing_armor
                if direction is not None:
                    armor_hits = [
                        hit
                        for hit in engine.compiled_target.ray_hits(origin, direction)
                        if hit.role == "armor"
                    ]
                    if armor_hits:
                        first_hit = min(
                            armor_hits, key=lambda item: (item.entry_t, item.object_id)
                        )
                        case_checks["forward_ray_has_armor_intersection"] = True
                        entry_t = float(first_hit.entry_t)
                        _update_extrema(extrema, "computed_first_collision_distance_m", entry_t)
                        case_checks["computed_first_collision_nonnegative"] = (
                            entry_t >= -position_tolerance_m
                        )

            persisted_distance = encounter.get("collision_distance_m")
            try:
                persisted_distance_float = float(persisted_distance)
            except (TypeError, ValueError):
                persisted_distance_float = math.nan
            if (
                math.isfinite(persisted_distance_float)
                and persisted_distance_float >= -position_tolerance_m
            ):
                case_checks[
                    "persisted_collision_distance_present_finite_nonnegative"
                ] = True
                _update_extrema(
                    extrema, "persisted_collision_distance_m", persisted_distance_float
                )

            persisted_collision_id = encounter.get("collision_armor_id")
            if (
                isinstance(persisted_collision_id, str)
                and persisted_collision_id
                and persisted_collision_id in armor_ids
            ):
                case_checks["persisted_collision_id_present_and_in_target"] = True
                collision_id_counts[persisted_collision_id] += 1
            if (
                first_hit is not None
                and persisted_collision_id == first_hit.object_id
            ):
                case_checks[
                    "persisted_collision_id_matches_computed_first_contact"
                ] = True

            if case_checks["first_armor_contact_trigger"]:
                try:
                    validate_impact_detonation(case, engine.compiled_target)
                except Exception as error:  # retain exact current-source refusal type
                    validator_error = error
                else:
                    case_checks["current_validate_impact_detonation_pass"] = True

            requested_conditions = (
                "first_armor_contact_trigger",
                "velocity_finite_nonzero",
                "body_plus_x_aligned_with_velocity",
                "origin_outside_target_armor",
                "forward_ray_has_armor_intersection",
                "computed_first_collision_nonnegative",
                "persisted_collision_distance_present_finite_nonnegative",
                "persisted_collision_id_present_and_in_target",
                "persisted_collision_id_matches_computed_first_contact",
                "current_validate_impact_detonation_pass",
            )
            case_checks["all_requested_conditions_pass"] = all(
                case_checks[name] for name in requested_conditions
            )

            failed_criteria = [
                name for name in criterion_names if not case_checks[name]
            ]
            for name, condition_passed in case_checks.items():
                if condition_passed:
                    passed[name] += 1
                else:
                    failure_reasons[name] += 1
            if failed_criteria and len(failure_examples) < MAX_FAILURE_EXAMPLES:
                example: dict[str, Any] = {
                    "line_number": line_number,
                    "case_id": case_id or None,
                    "failed_criteria": failed_criteria,
                }
                if validator_error is not None:
                    example["exception_type"] = type(validator_error).__name__
                    example["exception_message"] = str(validator_error)
                failure_examples.append(example)

    input_hash_after = _sha256(FORMAL_CASES)
    criterion_counts = {
        name: {
            "passed": int(passed[name]),
            "failed": int(total_lines - passed[name]),
        }
        for name in criterion_names
    }
    all_passed = (
        total_lines == int(round_manifest["selected_count"])
        and passed["all_requested_conditions_pass"] == total_lines
        and not duplicate_case_ids
        and input_hash_before == input_hash_after
    )

    audit = {
        "schema_version": "1.0.0",
        "audit_id": "impact-conditioning-formal-5000-current-source-20261015",
        "audit_date": AUDIT_DATE,
        "scope": "read_only_existing_formal_archive_diagnostic",
        "result": "pass" if all_passed else "fail",
        "audit_script": {
            "path": _relative(AUDIT_SCRIPT),
            "sha256": _sha256(AUDIT_SCRIPT),
        },
        "formal_input": {
            "path": _relative(FORMAL_CASES),
            "file_size_bytes": FORMAL_CASES.stat().st_size,
            "sha256_before": input_hash_before,
            "sha256_after": input_hash_after,
            "unchanged_during_audit": input_hash_before == input_hash_after,
            "round_manifest_declared_sha256": round_manifest["formal_cases_hash"],
            "matches_round_manifest": (
                input_hash_before == round_manifest["formal_cases_hash"]
            ),
            "round_manifest_candidate_count": int(round_manifest["candidate_count"]),
            "round_manifest_selected_count": int(round_manifest["selected_count"]),
        },
        "execution_environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
        "archive_identity": {
            "archive_engine_version": dataset_manifest["engine"]["version"],
            "archive_engine_code_hash": dataset_manifest["engine"]["code_hash"],
            "frozen_target_hash": frozen_hashes["target_hash"],
            "frozen_physics_registry_hash": frozen_hashes["physics_registry_hash"],
        },
        "current_source_identity": {
            "damage_engine_version": damage_engine.__version__,
            "engine_code_hash": ENGINE_CODE_HASH,
            "engine_code_matches_archive": (
                ENGINE_CODE_HASH == dataset_manifest["engine"]["code_hash"]
            ),
            "target_path": _relative(TARGET),
            "target_file_sha256": _sha256(TARGET),
            "normalized_target_hash": engine.target_hash,
            "normalized_target_matches_frozen": (
                engine.target_hash == frozen_hashes["target_hash"]
            ),
            "physics_registry_path": _relative(PHYSICS_REGISTRY),
            "physics_registry_hash": registry.hash,
            "physics_registry_matches_frozen": (
                registry.hash == frozen_hashes["physics_registry_hash"]
            ),
            "physics_registry_validation_status": registry.validation_status,
            "impact_trigger": IMPACT_TRIGGER,
            "impact_model_id": IMPACT_MODEL_ID,
            "alignment_cosine_minimum": alignment_minimum,
            "position_tolerance_m": position_tolerance_m,
        },
        "population": {
            "nonblank_record_count": total_lines,
            "parsed_record_count": int(passed["json_record_parsed"]),
            "unique_case_id_count": len(seen_case_ids),
            "duplicate_case_id_count": len(duplicate_case_ids),
            "duplicate_case_ids": sorted(duplicate_case_ids),
            "trigger_counts": dict(sorted(trigger_counts.items())),
            "munition_profile_counts": dict(sorted(profile_counts.items())),
            "frozen_split_counts": dict(sorted(split_counts.items())),
            "persisted_collision_armor_id_counts": dict(
                sorted(collision_id_counts.items())
            ),
        },
        "criterion_counts": criterion_counts,
        "numeric_ranges": extrema,
        "input_bias_diagnostics": {
            "normalized_velocity_direction_unique_count": len(
                velocity_direction_counts
            ),
            "normalized_velocity_direction_counts": dict(
                sorted(velocity_direction_counts.items())
            ),
            "body_orientation_unique_count_sign_canonicalized": len(
                body_orientation_counts
            ),
            "body_orientation_counts_sign_canonicalized": dict(
                sorted(body_orientation_counts.items())
            ),
            "single_velocity_direction": len(velocity_direction_counts) == 1,
            "single_body_orientation": len(body_orientation_counts) == 1,
            "interpretation": (
                "Passing the contact contract does not establish directional or attitude coverage. "
                "A single normalized direction or body orientation is a frozen-design bias and must "
                "either remain an explicit scope restriction or be varied in a new contact-conditioned "
                "design with the corresponding variables exposed to the learner."
            ),
        },
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "failure_examples": failure_examples,
        "contract_interpretation": {
            "admitted_population": (
                "Cases conditioned on a valid target-facing, first-armor-contact impact "
                "under the frozen design distribution."
            ),
            "supported_claim": (
                "The persisted formal encounters are compatible with the current "
                "geometric impact checks when every requested criterion passes."
            ),
            "unsupported_claims": [
                "Unconditional probability of hitting the vehicle.",
                "Guidance, navigation, or end-to-end mission success.",
                "Physical validity of target geometry, source strength, penetration, or damage response.",
                "Equivalence of all archived simulation outputs to a rerun with the current engine source.",
            ],
        },
        "evidence_boundary": (
            "This read-only audit includes all already-observed train, validation, and test cases. "
            "It validates stored encounter geometry against the current source and the current "
            "normalized target; it does not rerun the archived engine or inspect rejected proposal "
            "attempts. Therefore it is diagnostic evidence of an impact-conditioned data contract, "
            "not confirmatory model-performance evidence or real-world physical validation. A current "
            "engine-code mismatch with the archive is reported explicitly and prevents claiming full "
            "current-source/archive execution equivalence."
        ),
    }
    OUTPUT.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
