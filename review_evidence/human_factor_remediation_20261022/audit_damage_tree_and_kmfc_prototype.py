"""Build a reproducible audit of the legacy tree and KMFC software prototype.

The script reuses the previously hash-bound 5000-case diagnostic audits.  It
does not use that observed archive to fit scales or thresholds.  It compiles
the unfitted candidate and exercises only synthetic sentinel inputs to prove
software properties.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from damage_engine.errors import InputValidationError
from damage_engine.kmfc_ordinal_teacher import (
    KMFC_CATEGORY_ORDER,
    aggregate_kmfc_ordinal_teacher,
    compile_kmfc_ordinal_teacher,
    evaluate_kmfc_ordinal_teacher,
)


EVIDENCE_ROOT = Path(__file__).resolve().parent
LEGACY_AUDIT = REPO_ROOT / "review_evidence/kmfc_ordinal_feasibility_audit_20261015.json"
SUPPORT_AUDIT = REPO_ROOT / "review_evidence/kmfc_v0_component_support_audit_20261015.json"
FRAGMENT_SOURCE = REPO_ROOT / "damage_engine/fragments.py"
PROTOTYPE_SOURCE = REPO_ROOT / "damage_engine/kmfc_ordinal_teacher.py"
PROTOTYPE_CONFIG = EVIDENCE_ROOT / "kmfc_project_specific_software_prototype_v0_unfitted.json"
TARGETED_TEST_LOG = EVIDENCE_ROOT / "kmfc_targeted_tests.log"
OUTPUT = EVIDENCE_ROOT / "damage_tree_validity_and_software_prototype_audit.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _source_line(path: Path, needle: str) -> int:
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if needle in line:
            return line_number
    raise RuntimeError(f"source marker not found in {path}: {needle}")


def _synthetic_property_checks(config: dict[str, Any]) -> dict[str, Any]:
    compiled = compile_kmfc_ordinal_teacher(config)
    high_low_evidence: dict[str, np.ndarray] = {}
    for category in compiled.categories:
        width = len(category.evidence_channels)
        high_low_evidence[category.category_id] = np.vstack(
            [np.zeros(width, dtype=np.float64), np.ones(width, dtype=np.float64)]
        )
    result = evaluate_kmfc_ordinal_teacher(high_low_evidence, compiled)
    aggregate = aggregate_kmfc_ordinal_teacher(result)

    zero_anchor: dict[str, Any] = {}
    nested: dict[str, Any] = {}
    inner_gate: dict[str, Any] = {}
    for category_id in KMFC_CATEGORY_ORDER:
        item = result.category(category_id)
        score = item.projection.severity_score
        tau1 = item.category.projection.tau1
        available = item.projection.availability
        zero_anchor[category_id] = {
            "zero_severity": float(score[0]),
            "zero_level_ge_1": int(item.projection.level_ge_1[0]),
            "zero_level_ge_2": int(item.projection.level_ge_2[0]),
            "passed": bool(
                score[0] == 0.0
                and item.projection.level_ge_1[0] == 0.0
                and item.projection.level_ge_2[0] == 0.0
            ),
        }
        nested[category_id] = {
            "available_replicates": int(np.count_nonzero(available)),
            "violations": int(
                np.count_nonzero(
                    item.projection.level_ge_2[available]
                    > item.projection.level_ge_1[available]
                )
            ),
        }
        mean_score = float(np.mean(score[available]))
        inner_gate[category_id] = {
            "per_replicate_q_ge_1": aggregate["categories"][category_id]["q1"],
            "mean_severity": mean_score,
            "forbidden_mean_then_gate_indicator": int(mean_score >= tau1),
            "noncommuting_example_observed": bool(
                aggregate["categories"][category_id]["q1"]
                != float(mean_score >= tau1)
            ),
        }

    unavailable_evidence: dict[str, np.ndarray] = {}
    unavailable_masks: dict[str, np.ndarray] = {}
    for category in compiled.categories:
        width = len(category.evidence_channels)
        values = np.ones((2, width), dtype=np.float64)
        values[1, 0] = np.nan
        mask = np.ones((2, width), dtype=bool)
        mask[1, 0] = False
        unavailable_evidence[category.category_id] = values
        unavailable_masks[category.category_id] = mask
    unavailable = evaluate_kmfc_ordinal_teacher(
        unavailable_evidence,
        compiled,
        evidence_availability_by_category=unavailable_masks,
    )
    missingness: dict[str, Any] = {}
    for category_id in KMFC_CATEGORY_ORDER:
        item = unavailable.category(category_id)
        missingness[category_id] = {
            "availability": item.projection.availability.tolist(),
            "exclusive_level": [
                int(value) if np.isfinite(value) else None
                for value in item.exclusive_level
            ],
            "missing_is_not_level_zero": bool(
                not item.projection.availability[1]
                and np.isnan(item.exclusive_level[1])
            ),
        }

    confused = deepcopy(config)
    confused["categories"]["K"]["evidence_channels"][0][
        "quantity"
    ] = "fragment_deposited_energy_j"
    loss_as_residual_refused = False
    refusal_message = None
    try:
        compile_kmfc_ordinal_teacher(confused)
    except InputValidationError as exc:
        loss_as_residual_refused = True
        refusal_message = exc.message

    return {
        "configuration_content_hash": compiled.hash,
        "zero_anchor": zero_anchor,
        "ordinal_nesting": nested,
        "per_replicate_gate_before_aggregation": inner_gate,
        "missingness": missingness,
        "penetration_energy_loss_as_teacher_evidence": {
            "refused": loss_as_residual_refused,
            "message": refusal_message,
        },
        "all_software_properties_passed": bool(
            all(record["passed"] for record in zero_anchor.values())
            and all(record["violations"] == 0 for record in nested.values())
            and all(
                record["noncommuting_example_observed"]
                for record in inner_gate.values()
            )
            and all(
                record["missing_is_not_level_zero"]
                for record in missingness.values()
            )
            and loss_as_residual_refused
        ),
    }


def main() -> None:
    legacy = _read_json(LEGACY_AUDIT)
    support = _read_json(SUPPORT_AUDIT)
    candidate = _read_json(PROTOTYPE_CONFIG)
    synthetic = _synthetic_property_checks(candidate)

    support_by_output = {
        item["output"]: item for item in legacy["output_support"]
    }
    direct_by_category = {
        item["category"]: item for item in legacy["direct_ordinalization"]
    }
    legacy_category = {
        "effect-k1": "K",
        "effect-k2": "K",
        "effect-m1": "M",
        "effect-m2": "M",
        "effect-f1": "F",
        "effect-c1": "C",
        "effect-c2": "C",
    }
    legacy_mapping = []
    for output in legacy["legacy_output_contract"]["outputs"]:
        category_id = legacy_category[output]
        components = sorted(
            {
                leaf["component_id"]
                for leaf in legacy["leaf_events_by_output"][output]
            }
        )
        branch_matches = {}
        for branch, branch_components in support["category_support"][category_id][
            "branch_component_ids"
        ].items():
            overlap = sorted(set(components) & set(branch_components))
            if overlap:
                branch_matches[branch] = overlap
        legacy_mapping.append(
            {
                "legacy_output": output,
                "legacy_category": category_id,
                "legacy_leaf_count": len(legacy["leaf_events_by_output"][output]),
                "legacy_components": components,
                "candidate_branch_overlap": branch_matches,
                "available_count": support_by_output[output]["available_count"],
                "nonzero_count": support_by_output[output]["nonzero_count"],
                "safe_as_new_ordinal_level": False,
                "reason": (
                    "legacy event retained only as lineage evidence; new levels must be "
                    "derived from one category severity and parented hard gates"
                ),
            }
        )

    deposited_line = _source_line(
        FRAGMENT_SOURCE,
        "deposited = 0.5 * fragment.mass_kg * max(0.0, speed**2 - residual**2)",
    )
    residual_line = _source_line(
        FRAGMENT_SOURCE,
        "residual_energy = 0.5 * fragment.mass_kg * residual**2",
    )
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "audit_date": "2026-10-22",
        "scope": "legacy_tree_validity_and_project_specific_software_prototype",
        "verdict": {
            "legacy_tree_direct_kmfc_ordinal_relabel": "rejected",
            "legacy_tree_physical_validity": "not_established",
            "software_prototype": (
                "software_properties_passed_not_admitted"
                if synthetic["all_software_properties_passed"]
                else "software_property_failure"
            ),
            "physical_calibration": "not_performed",
            "research_training_admission": False,
            "legacy_tree_mutated": False,
        },
        "sources": {
            "audit_script": {
                "path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "legacy_ordinal_audit": {
                "path": LEGACY_AUDIT.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(LEGACY_AUDIT),
                "archive_dataset_id": legacy["sources"]["dataset_id"],
                "case_count": legacy["sources"]["case_count"],
                "boundary": "observed diagnostic archive; excluded from parameter fitting",
            },
            "candidate_support_audit": {
                "path": SUPPORT_AUDIT.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(SUPPORT_AUDIT),
                "boundary": support["evidence_boundary"],
            },
            "fragment_implementation": {
                "path": FRAGMENT_SOURCE.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(FRAGMENT_SOURCE),
            },
            "prototype_implementation": {
                "path": PROTOTYPE_SOURCE.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(PROTOTYPE_SOURCE),
            },
            "unfitted_prototype_configuration": {
                "path": PROTOTYPE_CONFIG.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(PROTOTYPE_CONFIG),
                "content_hash": synthetic["configuration_content_hash"],
                "parameter_selection_policy": candidate[
                    "parameter_selection_policy"
                ],
                "parameter_status": candidate["parameter_status"],
            },
            "targeted_test_log": {
                "path": TARGETED_TEST_LOG.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(TARGETED_TEST_LOG),
                "result": "51 passed",
                "environment": "pytorch312-lmdt; Python 3.12.13; torch 2.8.0+cu128",
            },
        },
        "legacy_seven_output_contract": {
            "outputs": legacy["legacy_output_contract"]["outputs"],
            "missing_new_output": legacy["legacy_output_contract"][
                "missing_required_outputs"
            ],
            "safe_in_place_relabel": legacy["legacy_output_contract"][
                "safe_in_place_relabel"
            ],
            "support": {
                name: {
                    "available_count": record["available_count"],
                    "nonzero_count": record["nonzero_count"],
                    "soft_mass": record["soft_mass"],
                }
                for name, record in support_by_output.items()
            },
            "direct_ordinalization": {
                category: {
                    "status": record["direct_ordinalization_status"],
                    "reason": record["reason"],
                    "strict_violation_count": record.get("empirical", {}).get(
                        "strict_violation_count"
                    ),
                }
                for category, record in direct_by_category.items()
            },
            "legacy_to_candidate_lineage": legacy_mapping,
        },
        "candidate_kmfc_branch_inventory": {
            category: {
                "branches": support["category_support"][category][
                    "branch_component_ids"
                ],
                "diagnostic_positive_support": {
                    "any_positive_case_count": support["category_support"][category][
                        "any_positive_case_count"
                    ],
                    "branch_positive_case_counts": support["category_support"][category][
                        "branch_positive_case_counts"
                    ],
                    "at_least_two_active_branches_count": support[
                        "category_support"
                    ][category]["at_least_two_active_branches_count"],
                },
                "admission": support["conclusion"].get(
                    f"{category}_fixed_threshold_calibration_from_current_archive",
                    support["conclusion"].get("M_structure_prototyping"),
                ),
            }
            for category in KMFC_CATEGORY_ORDER
        },
        "energy_semantics": {
            "penetration_energy_loss": {
                "archive_name": "deposited_energy_j / fragment_deposited_energy_j",
                "formula": "0.5*m*(pre_penetration_speed^2-post_penetration_speed^2)",
                "implementation_line": deposited_line,
                "role": "penetration interaction loss diagnostic",
            },
            "component_response_driver": {
                "canonical_name": "component_residual_kinetic_energy_j",
                "formula": "0.5*m*post_penetration_residual_speed^2",
                "implementation_line": residual_line,
                "role": "current component degraded/failed response driver",
            },
            "interchangeable": False,
            "current_archive_calibration_limit": (
                "the cited 5000-case support audit stores case-level replicate-mean "
                "penetration energy loss, not per-replicate residual kinetic energy; it "
                "cannot calibrate a residual-energy teacher"
            ),
        },
        "synthetic_software_property_evidence": synthetic,
        "parameter_boundary": {
            "scales_and_thresholds_in_candidate": "executable sentinel values only",
            "allowed_fit_partition": "future independent tree_design_fit only",
            "forbidden_selection_partitions": [
                "current observed archive validation",
                "current observed archive test",
                "future model_validation",
                "future sealed_reference_test",
            ],
            "required_before_dataset_regeneration": [
                "approve and hash project-specific codebook and lower-level branch definitions",
                "generate independent result-blind tree_design_fit and tree_design_holdout",
                "save per-replicate evidence and availability",
                "fit scales and ordered thresholds on tree_design_fit only",
                "pass zero-anchor, monotonicity, nesting, stability and support gates",
                "freeze new tree and label-spec versions before model data generation",
            ],
        },
        "limitations": [
            "No external or high-fidelity physical calibration was used.",
            "Synthetic sentinels prove implementation properties, not armored-vehicle damage validity.",
            "The unfitted configuration is not connected to the legacy engine output path.",
            "The observed 5000-case archive is diagnostic and was not used to select any prototype parameter.",
            "K/F/C remain unsupported for threshold fitting in the observed archive; M support is only sufficient for future independent prototyping.",
        ],
    }
    if not synthetic["all_software_properties_passed"]:
        raise RuntimeError("KMFC synthetic software-property checks failed")
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(OUTPUT.relative_to(REPO_ROOT).as_posix())


if __name__ == "__main__":
    main()

