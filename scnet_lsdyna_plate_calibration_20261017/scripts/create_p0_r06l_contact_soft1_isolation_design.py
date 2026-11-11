from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from generate_p0_r06_diagnostics import strict_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_DESIGN = ROOT / "design" / "p0_r06k_fragment_negative_volume_isolation_design.json"
DEFAULT_PARENT_CONTRACT = ROOT / "design" / "p0_r06k_fragment_negative_volume_output_contract.json"
DEFAULT_AUDIT = (
    ROOT
    / "analysis"
    / "72764387_r06k_0800_fragment_psfail_plate_20261031"
    / "r06k_fragment_psfail_return_audit.json"
)
DEFAULT_DESIGN_OUTPUT = ROOT / "design" / "p0_r06l_contact_soft1_isolation_design.json"
DEFAULT_CONTRACT_OUTPUT = ROOT / "design" / "p0_r06l_contact_soft1_output_contract.json"

REVISION = "R06_contact_soft1_isolation_20261031"
DESIGN_STATUS = "frozen_after_r06k_fragment_deletion_suppressed_but_contact_energy_timestep_unstable_before_soft1_outcome"
JOB_ID = "r06l_0800_contact_soft1"
EXPERIMENT_ID = "r06l_contact_soft_isolation_800"


def _require_parent_audit(audit: dict) -> None:
    decision = audit["decision"]
    if (
        audit["source_evidence"]["platform_job_id"] != "72764387"
        or audit["source_evidence"]["job_id"] != "r06k_0800_fragment_psfail_plate"
        or decision["solver_parser_runtime"] != "pass"
        or decision["input_identity"] != "pass"
        or decision["resource_identity"] != "pass"
        or decision["measurement_channels"] != "pass"
        or decision["fragment_negative_volume_hypothesis"] != "supported"
        or decision["fragment_deletion_suppression"] != "pass"
        or decision["numerical_stability"] != "fail_contact_energy_and_timestep_collapse"
        or decision["release_r06l_contact_soft1_isolation"] is not True
    ):
        raise ValueError("Job 72764387 audit does not authorize the R06l SOFT=1 isolation")


def build_contract(parent: dict, audit: dict) -> dict:
    _require_parent_audit(audit)
    contract = deepcopy(parent)
    contract["schema_version"] = "hf-p0-r06l-contact-soft1-output-contract@1.0.0"
    contract["contract_id"] = "p0_r06l_contact_soft1_output_contract"
    contract["status"] = DESIGN_STATUS
    boundary = contract["version_boundary"]
    boundary["applies_to_package_revision"] = REVISION
    boundary["effective_scope"] = "prospective_r06l_contact_soft1_single_factor_diagnostic"
    boundary["parent_fragment_psfail"] = {
        "package_revision": audit["source_evidence"]["package_revision"],
        "job_id": audit["source_evidence"]["job_id"],
        "platform_job_id": audit["source_evidence"]["platform_job_id"],
        "deck_sha256": audit["source_evidence"]["deck_sha256"],
        "fragment_negative_volume_hypothesis": audit["decision"]["fragment_negative_volume_hypothesis"],
        "fragment_deletion_suppression": audit["decision"]["fragment_deletion_suppression"],
        "numerical_stability": audit["decision"]["numerical_stability"],
        "retroactive_gate_rewrite": False,
    }
    contract.pop("fragment_negative_volume_isolation", None)
    parent_stability = audit["formal_fragment_negative_volume_comparison_to_job_72751725"]["performance_and_numerical_stability"]
    contract["contact_soft_isolation"] = {
        "factor": "CONTACT_ERODING_SINGLE_SURFACE Optional Card A field 1 SOFT",
        "control_job_id": "r06k_0800_fragment_psfail_plate",
        "control_platform_job_id": "72764387",
        "control_level": 2,
        "treatment_job_id": JOB_ID,
        "treatment_level": 1,
        "fixed_controls": {
            "psfail_part_set_id": 910001,
            "psfail_part_ids": [1002],
            "fragment_part_id": 2002,
            "fragment_mat015_erod": 1,
            "plate_mat015_erod": 0,
            "tssfac": 0.05,
            "dt2ms": 0.0,
            "post_explosion_initial_state": "unchanged",
        },
        "single_factor_rule": (
            "The R06l deck may differ from returned R06k only in TITLE and CONTACT_ERODING_SINGLE_SURFACE "
            "Optional Card A field 1 SOFT, 2 to 1."
        ),
        "rationale": (
            "R06k retained all heavily damaged fragment solids under segment-based SOFT=2 and exposed extreme "
            "negative sliding energy plus a late explicit-timestep collapse. SOFT=1 is a documented contact-stability "
            "diagnostic, not a physical calibration."
        ),
        "sources": [
            "Ansys LS-DYNA instability guidance: inspect penetrations/contact energy and consider automatic contact with SOFT=1",
            "LS-DYNA Keyword User's Manual Volume I: SOFT selects the contact formulation",
        ],
        "diagnostic_gates": {
            "parser_and_resource_identity": "required",
            "normal_termination": "required_for_response_review",
            "no_mass_scaling": True,
            "fragment_deleted_element_count": {"required": 0},
            "fragment_negative_volume_element_count": {"required": 0},
            "hvar6_invalid_alive_count_max": 0,
            "total_energy_ratio": {"minimum": 0.95, "maximum": 1.05, "classification": "candidate_engineering_gate"},
            "absolute_final_sliding_energy_fraction_of_initial": {"maximum": 0.05, "classification": "candidate_engineering_gate"},
            "cycle_count": {
                "control": parent_stability["treatment_cycle_count"],
                "treatment_maximum": int(parent_stability["treatment_cycle_count"] // 10),
                "meaning": "at_least_tenfold_reduction_from_r06k_contact_instability",
                "classification": "diagnostic_efficiency_gate_not_physical_standard",
            },
            "minimum_time_step_ms": {
                "control": parent_stability["minimum_time_step_ms"],
                "treatment_minimum": parent_stability["minimum_time_step_ms"] * 10.0,
                "meaning": "at_least_tenfold_improvement_from_r06k_late_step_collapse",
                "classification": "diagnostic_efficiency_gate_not_physical_standard",
            },
            "direct_contact_force_semantics": "pending_sign_side_unit_and_closure_validation",
        },
        "interpretation_rules": {
            "all_numerical_gates_pass": (
                "supports SOFT=1 as a numerically preferable candidate for a later controlled convergence and physical-validation program; does not admit calibration"
            ),
            "fragment_deletion_or_negative_volume_reappears": "rejects the candidate numerical mechanism",
            "energy_or_timestep_gate_fails": (
                "stop contact-parameter tuning and redesign failed-fragment model form, with adaptive solid-to-SPH as the next documented candidate"
            ),
            "primary_qoi_policy": (
                "never promote this contact-algorithm diagnostic response to a physical label, calibration target or reference dataset"
            ),
        },
    }
    contract["admission"] = {
        "contact_soft1_isolation_submission": "allowed_after_static_release_verification",
        "production_contact_or_erosion_policy": "pending_external_target_solver_diagnostic",
        "further_contact_parameter_tuning_if_failed": "denied_model_form_redesign_required",
        "r06f_2000_control": "denied",
        "treatments": "denied",
        "p0_response_calibration": "denied",
        "p1_submission": "denied",
        "engine_fitting": "denied",
        "damage_tree_fitting": "denied",
        "reference_dataset_generation": "denied",
        "physical_validity": "not_established_requires_independent_experimental_evidence",
    }
    contract["claim_boundary"] = [
        "R06k isolated the ERODING-contact negative-volume deletion path by removing all 35 fragment deletions.",
        "R06k is numerically inadmissible because of a 66.8-fold cycle increase, explicit-timestep collapse and extreme negative sliding energy.",
        "R06l changes only contact SOFT from 2 to 1 while retaining PSFAIL=910001 and all other model controls.",
        "The model continues to start from the same post-explosion fragment state.",
        "Any R06l result is numerical-mechanism evidence only and cannot authorize engine fitting, damage-tree fitting or dataset admission.",
    ]
    return contract


def build_design(parent: dict, contract: dict, audit: dict) -> dict:
    _require_parent_audit(audit)
    if contract["version_boundary"]["applies_to_package_revision"] != REVISION:
        raise ValueError("contact SOFT contract is not bound to the new revision")
    design = deepcopy(parent)
    design["schema_version"] = "hf-p0-r06-diagnostic-design@1.9.0"
    design["package_revision"] = REVISION
    design["design_status"] = DESIGN_STATUS
    design["title"] = "SCNet P0 R06l 800 m/s contact SOFT=1 isolation"
    design["created_date"] = "2026-10-31"
    design["purpose"] = (
        "Test whether changing only the eroding single-surface contact formulation from SOFT=2 to SOFT=1 removes "
        "the severe sliding-energy and timestep instability exposed when R06k retained failed fragment solids."
    )
    design["provenance"] = {
        "parent_revision": audit["source_evidence"]["package_revision"],
        "parent_job_id": audit["source_evidence"]["job_id"],
        "parent_platform_job_id": audit["source_evidence"]["platform_job_id"],
        "parent_deck_sha256": audit["source_evidence"]["deck_sha256"],
        "parent_result": audit["decision"]["classification"],
        "parent_return_audit": "analysis/72764387_r06k_0800_fragment_psfail_plate_20261031/r06k_fragment_psfail_return_audit.json",
        "successor_contract": "design/p0_r06l_contact_soft1_output_contract.json",
        "identity_rule": (
            "Mesh, IDs, post-explosion initial state, both MAT_015 definitions, EOS, contact cards except SOFT, "
            "PSFAIL=910001, section, boundaries, speed, termination, timestep, outputs and resources remain identical "
            "to Job 72764387. Only Optional Card A SOFT changes 2 to 1."
        ),
        "factor_scope": "CONTACT_ERODING_SINGLE_SURFACE Optional Card A field 1 SOFT only, 2 to 1",
    }
    design["frozen_solver_controls"]["control_timestep"]["tssfac"] = 0.05
    design["shared_measurement_upgrade"]["status"] = "validated_on_job_72764387_and_frozen_for_contact_soft1_isolation"
    design["explosion_instant_start_policy"] = deepcopy(contract["explosion_instant_start_policy"])
    design["paired_experiments"] = [
        {
            "experiment_id": EXPERIMENT_ID,
            "question": "Does SOFT=1 remove the R06k contact-energy and timestep instability without restoring fragment deletion?",
            "case_identity": {
                "r05_case_id": "p0_v0800_h0p75",
                "r05_original_index": 2,
                "impact_speed_mps": 800.0,
                "mesh_mm": 0.75,
                "plate_part_id": 1002,
                "fragment_part_id": 2002,
            },
            "factor": "contact_optional_card_a_soft",
            "control_level": 2,
            "treatment_level": 1,
            "external_control": {
                "job_id": "r06k_0800_fragment_psfail_plate",
                "platform_job_id": "72764387",
                "deck_sha256": audit["source_evidence"]["deck_sha256"],
                "resource_identity": "one_node_32_ranks",
                "tssfac": 0.05,
                "fragment_mat015_erod": 1,
                "psfail_part_set_id": 910001,
                "contact_soft": 2,
                "fragment_deleted_element_count": audit["diagnostic_response"]["fragment_deleted_element_count"],
                "negative_volume_fragment_element_count": audit["execution"]["negative_volume_topology"]["fragment_unique_element_count"],
                "cycle_count": audit["execution"]["cycle_count"],
                "minimum_time_step_ms": audit["execution"]["minimum_time_step_ms"],
                "final_sliding_energy_fraction_of_initial": audit["returned_outputs"]["glstat"]["final_sliding_energy_fraction_of_initial"],
            },
            "jobs": [
                {
                    "job_id": JOB_ID,
                    "directory": "job_01_v0800_contact_soft1",
                    "role": "contact_soft1_numerical_stability_diagnostic",
                    "termination_time_ms": 0.04,
                    "plate_only_psfail": True,
                    "tssfac": 0.05,
                    "fragment_mat015_erod": 1,
                    "contact_soft": 1,
                    "factor_level": 1,
                    "platform_job_name": "uav_hf_r06l_0800_soft1",
                    "parent_r06k_job_id": "r06k_0800_fragment_psfail_plate",
                    "parent_platform_job_id": "72764387",
                }
            ],
            "preregistered_evaluation": deepcopy(contract["contact_soft_isolation"]),
        }
    ]
    design["paired_comparison_tolerances"] = deepcopy(contract["contact_soft_isolation"]["diagnostic_gates"])
    design["submission_sequence"] = {
        "wave_0": [JOB_ID],
        "wave_0_gate": (
            "Target parser/resource identity, normal completion, no mass scaling, zero fragment deletion/negative volume, "
            "energy/contact-energy gates and timestep/cycle improvement are all reviewed before any successor."
        ),
        "hold_after_wave_0": "Do not release 2000 m/s, treatments, fitting or labels. Audit the single SOFT factor first.",
        "resource_identity": (
            "Use exactly one node, 32 ranks, LS-DYNA R15 MPP double precision, and the same memory settings as Job 72764387."
        ),
    }
    design["deferred_not_generated"] = [
        {"candidate": "adaptive solid-to-SPH failed-fragment redesign", "reason": "Generate only if the SOFT=1 diagnostic remains unstable."},
        {"candidate": "additional contact parameter sweeps", "reason": "Denied after a failed SOFT=1 result; model-form redesign takes precedence."},
        {"candidate": "2000 m/s and all treatments", "reason": "Hold until numerical stability, convergence and physical validation are resolved."},
    ]
    design["admission"] = deepcopy(contract["admission"])
    design["claim_boundary"] = deepcopy(contract["claim_boundary"])
    return design


def _write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen design artifact: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the R06l contact SOFT=1 diagnostic")
    parser.add_argument("--parent-design", type=Path, default=DEFAULT_PARENT_DESIGN)
    parser.add_argument("--parent-contract", type=Path, default=DEFAULT_PARENT_CONTRACT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--design-output", type=Path, default=DEFAULT_DESIGN_OUTPUT)
    parser.add_argument("--contract-output", type=Path, default=DEFAULT_CONTRACT_OUTPUT)
    args = parser.parse_args()
    audit = strict_json(args.audit.resolve())
    contract = build_contract(strict_json(args.parent_contract.resolve()), audit)
    design = build_design(strict_json(args.parent_design.resolve()), contract, audit)
    _write_new(args.contract_output.resolve(), contract)
    _write_new(args.design_output.resolve(), design)
    print(json.dumps({"package_revision": REVISION, "job_id": JOB_ID, "design": str(args.design_output.resolve()), "contract": str(args.contract_output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
