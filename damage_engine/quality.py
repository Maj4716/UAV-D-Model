"""Software identity and evidence gates, independent of simulation formulae."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .errors import InputValidationError
from .scientific_validation import validate_scientific_validation_record
from .utils import content_hash, file_hash


def validate_precomputed_results(
    engine: Any, cases: list[dict], payloads: list[dict], *,
    case_hash: Callable[[dict], str], engine_code_hash: str,
    fragment_metrics: list[str], blast_metrics: list[str], armor_metrics: list[str],
) -> None:
    """Reject mismatched or malformed supervision before opening output files."""
    if len(cases) != len(payloads):
        raise InputValidationError("cases and results must have equal length")
    if any(not isinstance(case, dict) for case in cases):
        raise InputValidationError("precomputed cases must be objects")
    identifiers = [case.get("case_id") for case in cases]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise InputValidationError("precomputed case_id values must be nonempty strings")
    if len(set(identifiers)) != len(identifiers):
        raise InputValidationError("precomputed case_id values must be unique")
    components = list(engine.compiled_target.component_order)
    armor = [item.object_id for item in engine.compiled_target.armor]

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise InputValidationError("precomputed result " + message)

    def stats(value: Any, *, count: int, path: str, probability: bool = False) -> None:
        require(isinstance(value, dict), f"statistics missing at {path}")
        allowed = {"mean", "stddev", "ci95_lower", "ci95_upper", "sample_count",
                   "uncertainty_estimable", "insufficient_reason", "estimand", "stddev_estimand",
                   "conditioning", "random_level", "interval_type", "interval_method",
                   "probability_bounds_clipped", "coverage_status",
                   "frechet_union_lower_bound", "frechet_union_upper_bound"}
        require(not (set(value) - allowed), f"statistics contain unknown or reserved row keys at {path}")
        require(all(name in value for name in ("mean", "stddev", "ci95_lower", "ci95_upper")),
                f"statistics incomplete at {path}")
        mean = value["mean"]
        require(isinstance(mean, (int, float)) and not isinstance(mean, bool) and math.isfinite(mean),
                f"mean is not finite at {path}")
        require(not probability or 0 <= mean <= 1, f"probability mean is outside [0,1] at {path}")
        if "sample_count" in value:
            require(value["sample_count"] == count, f"sample_count mismatch at {path}")
        uncertainty = [value[name] for name in ("stddev", "ci95_lower", "ci95_upper")]
        if count < 2:
            require(all(item is None for item in uncertainty),
                    f"single-observation uncertainty must be null at {path}")
            require(value.get("uncertainty_estimable") is False,
                    f"single-observation uncertainty must be masked at {path}")
        else:
            require(all(isinstance(item, (int, float)) and not isinstance(item, bool)
                        and math.isfinite(item) for item in uncertainty),
                    f"uncertainty is not finite at {path}")
            require(uncertainty[0] >= 0 and uncertainty[1] <= mean <= uncertainty[2],
                    f"invalid uncertainty bounds at {path}")
            require(value.get("uncertainty_estimable", True) is True,
                    f"uncertainty is unexpectedly masked at {path}")

    for position, (case, payload) in enumerate(zip(cases, payloads)):
        try:
            require(isinstance(payload, dict), f"payload {position} is not an object")
            require(payload.get("case_id") == case["case_id"], f"case_id mismatch at {position}")
            require(payload.get("status") in {"ok", "refused"}, f"status is invalid at {position}")
            provenance = payload.get("provenance")
            require(isinstance(provenance, dict), f"provenance missing at {position}")
            expected = {"case_hash": case_hash(case), "target_hash": engine.target_hash,
                        "physics_registry_hash": engine.registry.hash,
                        "engine_code_hash": engine_code_hash,
                        "damage_tree_version": engine.damage_tree.version}
            for field, value in expected.items():
                require(provenance.get(field) == value, f"{field} mismatch at {position}")
            analysis = case.get("analysis", {})
            require(isinstance(analysis, dict), f"case analysis must be an object at {position}")
            require(isinstance(case.get("metadata", {}), dict), f"case metadata must be an object at {position}")
            for field in ("aleatory_seed", "epistemic_sample_id", "evaluation_standard_ids"):
                expected_value = analysis.get(field, [] if field == "evaluation_standard_ids" else None)
                require(provenance.get(field) == expected_value, f"{field} mismatch at {position}")
            # Serialize now, so nonfinite JSON values fail before even mkdir.
            json.dumps(payload, allow_nan=False)
            if payload["status"] == "refused":
                require(isinstance(payload.get("refusal"), dict) and
                        isinstance(payload["refusal"].get("code"), str), f"refusal is invalid at {position}")
                require(provenance.get("replicate_count") == 0, f"refusal replicate_count mismatch at {position}")
                continue
            require(payload.get("target_id") == engine.target_data["target_id"], f"target_id mismatch at {position}")
            require(payload.get("component_order") == components, f"component_order mismatch at {position}")
            count = payload.get("replicate_count")
            require(isinstance(count, int) and not isinstance(count, bool) and count > 0,
                    f"replicate_count invalid at {position}")
            require(provenance.get("replicate_count") == count, f"replicate_count mismatch at {position}")
            require(set(payload["components"]) == set(components), f"component identities mismatch at {position}")
            require(set(payload["armor"]) == set(armor), f"armor identities mismatch at {position}")
            require(set(payload["parameter_realization"]) == set(components + armor),
                    f"parameter_realization identities mismatch at {position}")
            require(provenance.get("parameter_realization_hash") == content_hash(payload["parameter_realization"]),
                    f"parameter_realization_hash mismatch at {position}")
            require(set(payload.get("mechanisms", [])) == set(analysis.get("mechanisms", [])),
                    f"mechanisms mismatch at {position}")
            for identifier, values in payload["parameter_realization"].items():
                require(isinstance(values, dict), f"parameter_realization invalid for {identifier}")
            requested = analysis.get("evaluation_standard_ids", [])
            require(set(requested).issubset(engine.damage_tree.standards), f"unknown evaluation standard at {position}")
            require(set(payload["damage_tree"]["outputs"]) == set(engine.damage_tree.outputs),
                    f"output identities mismatch at {position}")
            for item in engine.compiled_target.components:
                result = payload["components"][item.object_id]
                for mechanism in ("fragment", "blast", "combined"):
                    for state in item.source["damage_states"]:
                        if state == "intact":
                            continue
                        value = result["damage_states"][mechanism][state]
                        stats(value, count=count, path=f"{item.object_id}/{mechanism}/{state}", probability=True)
                        if mechanism == "combined":
                            for bound in ("lower", "upper"):
                                stats(value[f"frechet_union_{bound}_bound"], count=count,
                                      path=f"{item.object_id}/{state}/{bound}", probability=True)
                for section, keys in (("fragment_metrics", fragment_metrics), ("blast_metrics", blast_metrics)):
                    for key in keys:
                        stats(result[section][key], count=count, path=f"{item.object_id}/{section}/{key}")
            for identifier in armor:
                result = payload["armor"][identifier]
                for key in ("post_fragment_integrity", "post_combined_integrity"):
                    stats(result[key], count=count, path=f"{identifier}/{key}", probability=True)
                for key in armor_metrics:
                    stats(result["blast_metrics"][key], count=count, path=f"{identifier}/{key}",
                          probability=key.endswith("probability") or key.endswith("integrity"))
            for identifier, value in payload["damage_tree"]["outputs"].items():
                stats(value, count=count, path=f"output/{identifier}", probability=True)
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, InputValidationError):
                raise
            raise InputValidationError(f"precomputed result structure invalid at {position}: {error}") from error


def synthetic_label_admission() -> dict:
    return {
        "label_source_type": "synthetic_teacher",
        "input_integrity_status": "verified",
        "scientific_evidence_status": "not_evaluated",
        "approved_uses": ["software_development"],
        "validation_record_status": "missing",
        "coverage_report_status": "not_performed",
        "evaluation_definition_approval_status": "not_independently_verified",
        "source_object_completeness_status": "not_independently_verified",
        "reason": "Execution and archive integrity do not establish independent scientific validation.",
        "evidence_record": None,
    }


def validate_label_admission(manifest: dict, purpose: str = "software_development", *,
                             evidence_root: str | Path | None = None) -> None:
    """Check declared purpose and content-bound review evidence; never relabel synthetic truth.

    Reviewer authentication and scientific adequacy remain governance requirements.
    This gate verifies the record's existence, content, expiry and object scope.
    """
    admission = manifest.get("label_admission") or {}
    if admission.get("label_source_type") != "synthetic_teacher":
        raise InputValidationError("label source must explicitly identify synthetic_teacher")
    if admission.get("input_integrity_status") != "verified" or purpose not in admission.get("approved_uses", []):
        raise InputValidationError(f"label admission does not approve purpose {purpose!r}")
    if purpose == "software_development":
        return
    if admission.get("scientific_evidence_status") != "independently_reviewed" or evidence_root is None:
        raise InputValidationError("independent evidence is required for research-purpose admission")
    if (admission.get("evaluation_definition_approval_status") != "approved"
            or admission.get("source_object_completeness_status") != "complete"):
        raise InputValidationError("independent definition approval and object completeness are required")
    reference = admission.get("evidence_record") or {}
    try:
        root = Path(evidence_root).resolve()
        path = (root / reference["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != reference["content_hash"]:
            raise ValueError("evidence file is missing, outside root or hash mismatched")
        record = json.loads(path.read_text(encoding="utf-8"))
        expires = datetime.fromisoformat(record["valid_until"].replace("Z", "+00:00"))
        if expires.tzinfo is None or expires <= datetime.now(timezone.utc):
            raise ValueError("evidence record has expired or lacks timezone")
        if not record.get("reviewer_id") or not record.get("record_version") or record.get("status") != "approved":
            raise ValueError("evidence approval metadata is incomplete")
        if purpose not in record.get("approved_uses", []) or record.get("label_source_type") != "synthetic_teacher":
            raise ValueError("evidence purpose or label-source scope mismatch")
        binding = {
            "target_hashes": manifest["source_contracts"]["target_hashes"],
            "parameter_set_hashes": manifest["source_contracts"]["parameter_set_hashes"],
            "engine_code_hash": manifest["engine"]["code_hash"],
            "configuration_hash": manifest["engine"]["configuration_hash"],
            "source_contracts_hash": content_hash(manifest["source_contracts"]),
            "semantic_vocabulary_hash": manifest["semantic_vocabulary"]["content_hash"],
        }
        if record.get("object_scope_hash") != content_hash(binding):
            raise ValueError("evidence is outside the current object/definition scope")
        validate_scientific_validation_record(record.get("scientific_validation"))
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise InputValidationError(f"independent evidence rejected: {error}") from error
