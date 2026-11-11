from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from generate_p0_r06_diagnostics import strict_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_DESIGN = ROOT / "design" / "p0_r06j_fragment_erosion_isolation_design.json"
DEFAULT_PARENT_CONTRACT = ROOT / "design" / "p0_r06j_fragment_erosion_output_contract.json"
DEFAULT_AUDIT = (
    ROOT
    / "analysis"
    / "72751725_r06j_0800_fragment_erod1_20261030"
    / "r06j_fragment_erod1_return_audit.json"
)
DEFAULT_DESIGN_OUTPUT = ROOT / "design" / "p0_r06k_fragment_negative_volume_isolation_design.json"
DEFAULT_CONTRACT_OUTPUT = ROOT / "design" / "p0_r06k_fragment_negative_volume_output_contract.json"

REVISION = "R06_fragment_negative_volume_isolation_20261030"
DESIGN_STATUS = "frozen_after_r06j_partial_erod_causality_before_fragment_negative_volume_psfail_outcome"
JOB_ID = "r06k_0800_fragment_psfail_plate"
EXPERIMENT_ID = "r06k_fragment_negative_volume_psfail_isolation_800"


def _require_parent_audit(audit: dict) -> None:
    decision = audit["decision"]
    if (
        audit["source_evidence"]["platform_job_id"] != "72751725"
        or audit["source_evidence"]["job_id"] != "r06j_0800_fragment_erod1"
        or decision["solver_parser_runtime"] != "pass"
        or decision["input_identity"] != "pass"
        or decision["resource_identity"] != "pass"
        or decision["measurement_channels"] != "pass"
        or decision["fragment_failure_erosion_hypothesis"] != "partial_support"
        or decision["remaining_deletion_mechanism"] != "eroding_contact_automatic_negative_volume_failure"
        or decision["response_availability"] != "fail"
        or decision["release_r06k_fragment_negative_volume_psfail_isolation"] is not True
    ):
        raise ValueError("Job 72751725 audit does not authorize the R06k PSFAIL isolation")


def build_contract(parent: dict, audit: dict) -> dict:
    _require_parent_audit(audit)
    contract = deepcopy(parent)
    contract["schema_version"] = "hf-p0-r06k-fragment-negative-volume-output-contract@1.0.0"
    contract["contract_id"] = "p0_r06k_fragment_negative_volume_output_contract"
    contract["status"] = DESIGN_STATUS
    boundary = contract["version_boundary"]
    boundary["applies_to_package_revision"] = REVISION
    boundary["effective_scope"] = "prospective_r06k_fragment_psfail_plate_only"
    boundary["parent_fragment_erod1"] = {
        "package_revision": audit["source_evidence"]["package_revision"],
        "job_id": audit["source_evidence"]["job_id"],
        "platform_job_id": audit["source_evidence"]["platform_job_id"],
        "deck_sha256": audit["source_evidence"]["deck_sha256"],
        "fragment_failure_erosion_hypothesis": audit["decision"]["fragment_failure_erosion_hypothesis"],
        "remaining_deletion_mechanism": audit["decision"]["remaining_deletion_mechanism"],
        "retroactive_gate_rewrite": False,
    }
    contract["fragment_negative_volume_isolation"] = {
        "factor": "CONTROL_SOLID Card 1 field 5 PSFAIL",
        "control_job_id": "r06j_0800_fragment_erod1",
        "control_platform_job_id": "72751725",
        "control_level": 0,
        "treatment_job_id": JOB_ID,
        "treatment_level": 910001,
        "treatment_part_set": {"set_id": 910001, "part_ids": [1002], "meaning": "plate_only"},
        "fragment_part_id": 2002,
        "fragment_mat015_erod": 1,
        "plate_mat015_erod": 0,
        "tssfac": 0.05,
        "dt2ms": 0.0,
        "single_factor_rule": (
            "The R06k deck may differ from returned R06j only in TITLE and CONTROL_SOLID Card 1 field 5 "
            "PSFAIL, 0 to plate-only part set 910001."
        ),
        "solver_semantics": {
            "eroding_contact_side_effect": (
                "ERODING contact automatically activates a negative-volume failure criterion for solid elements"
            ),
            "psfail_scope": (
                "A nonzero PSFAIL part-set ID restricts that criterion to solids in the named part set"
            ),
            "treatment_effect": (
                "PSFAIL=910001 keeps the criterion on plate part 1002 and excludes fragment part 2002"
            ),
            "source": "Ansys LS-DYNA Explicit Technical Guide, ERODING contact and PSFAIL guidance",
        },
        "diagnostic_gates": {
            "parser_and_resource_identity": "required",
            "normal_termination": "report_but_not_required_for_mechanism_evidence",
            "no_mass_scaling": True,
            "fragment_negative_volume_messages": "compare_to_35_in_job_72751725",
            "fragment_deleted_element_count": "compare_to_35_in_job_72751725",
            "fragment_hvar6_failure_evidence": "required_if_outputs_are_readable",
            "timestep_collapse_or_solver_abort": "retain_as_diagnostic_evidence",
            "energy_and_contact_outputs": "mechanism_evidence_only",
        },
        "interpretation_rules": {
            "fragment_deletion_removed_but_distortion_or_timestep_failure_remains": (
                "supports negative-volume erosion as the remaining deletion path and demonstrates that simple deletion suppression is not a production remedy"
            ),
            "fragment_deletion_persists": (
                "refutes exclusive attribution and requires another deletion-path audit"
            ),
            "normal_completion_without_fragment_deletion": (
                "still does not establish physical validity because the treatment suppresses a numerical failure criterion"
            ),
            "primary_qoi_policy": (
                "never promote PSFAIL-isolation response values to physical labels, calibration targets or reference data"
            ),
        },
    }
    contract["admission"] = {
        "fragment_psfail_isolation_submission": "allowed_after_static_release_verification",
        "production_contact_or_erosion_policy": "pending_external_target_solver_diagnostic",
        "further_timestep_halving": "denied",
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
        "R06j partially suppressed fragment deletion but left exactly 35 negative-volume-driven fragment deletions.",
        "R06k changes only PSFAIL from 0 to plate-only set 910001, excluding fragment part 2002 from the ERODING-contact negative-volume criterion.",
        "Fragment MAT_015 EROD remains 1 and the contact formulation remains ERODING_SINGLE_SURFACE.",
        "The model continues to start from the same post-explosion fragment state.",
        "Any R06k result is mechanism evidence only and cannot authorize engine fitting, damage-tree fitting or dataset admission.",
    ]
    return contract


def build_design(parent: dict, contract: dict, audit: dict) -> dict:
    _require_parent_audit(audit)
    if contract["version_boundary"]["applies_to_package_revision"] != REVISION:
        raise ValueError("fragment negative-volume contract is not bound to the new revision")
    design = deepcopy(parent)
    design["schema_version"] = "hf-p0-r06-diagnostic-design@1.8.0"
    design["package_revision"] = REVISION
    design["design_status"] = DESIGN_STATUS
    design["title"] = "SCNet P0 R06k 800 m/s fragment negative-volume PSFAIL isolation"
    design["created_date"] = "2026-10-30"
    design["purpose"] = (
        "Test whether the 35 remaining R06j fragment deletions are entirely caused by the negative-volume "
        "failure criterion automatically activated by ERODING contact."
    )
    design["provenance"] = {
        "parent_revision": audit["source_evidence"]["package_revision"],
        "parent_job_id": audit["source_evidence"]["job_id"],
        "parent_platform_job_id": audit["source_evidence"]["platform_job_id"],
        "parent_deck_sha256": audit["source_evidence"]["deck_sha256"],
        "parent_result": audit["decision"]["classification"],
        "parent_return_audit": "analysis/72751725_r06j_0800_fragment_erod1_20261030/r06j_fragment_erod1_return_audit.json",
        "successor_contract": "design/p0_r06k_fragment_negative_volume_output_contract.json",
        "identity_rule": (
            "Mesh, IDs, post-explosion initial state, both MAT_015 definitions including fragment EROD=1, EOS, "
            "contact, section, boundaries, speed, termination, timestep, outputs and resources remain identical "
            "to Job 72751725. Only CONTROL_SOLID PSFAIL changes 0 to 910001."
        ),
        "factor_scope": "CONTROL_SOLID Card 1 field 5 PSFAIL only, 0 to plate-only set 910001",
    }
    design["frozen_solver_controls"]["control_timestep"]["tssfac"] = 0.05
    design["shared_measurement_upgrade"]["status"] = (
        "validated_on_job_72751725_and_frozen_for_fragment_negative_volume_psfail_isolation"
    )
    design["explosion_instant_start_policy"] = deepcopy(contract["explosion_instant_start_policy"])
    design["paired_experiments"] = [
        {
            "experiment_id": EXPERIMENT_ID,
            "question": (
                "Does excluding fragment part 2002 from the ERODING-contact negative-volume failure criterion remove the 35 remaining deletions?"
            ),
            "case_identity": {
                "r05_case_id": "p0_v0800_h0p75",
                "r05_original_index": 2,
                "impact_speed_mps": 800.0,
                "mesh_mm": 0.75,
                "plate_part_id": 1002,
                "fragment_part_id": 2002,
            },
            "factor": "control_solid_psfail_part_set",
            "control_level": 0,
            "treatment_level": 910001,
            "external_control": {
                "job_id": "r06j_0800_fragment_erod1",
                "platform_job_id": "72751725",
                "deck_sha256": audit["source_evidence"]["deck_sha256"],
                "resource_identity": "one_node_32_ranks",
                "tssfac": 0.05,
                "fragment_mat015_erod": 1,
                "psfail_part_set_id": 0,
                "fragment_negative_volume_element_count": audit["execution"]["negative_volume_topology"]["fragment_unique_element_count"],
                "fragment_deleted_element_count": 686 - audit["diagnostic_response"]["final_alive_fragment_count"],
                "deleted_upstream_mass_fraction": audit["diagnostic_response"]["deleted_upstream_mass_fraction"],
            },
            "jobs": [
                {
                    "job_id": JOB_ID,
                    "directory": "job_01_v0800_fragment_psfail_plate",
                    "role": "fragment_negative_volume_failure_exclusion_diagnostic",
                    "termination_time_ms": 0.04,
                    "plate_only_psfail": True,
                    "tssfac": 0.05,
                    "fragment_mat015_erod": 1,
                    "factor_level": 910001,
                    "platform_job_name": "uav_hf_r06k_0800_fpsfail",
                    "parent_r06j_job_id": "r06j_0800_fragment_erod1",
                    "parent_platform_job_id": "72751725",
                }
            ],
            "preregistered_evaluation": deepcopy(contract["fragment_negative_volume_isolation"]),
        }
    ]
    design["paired_comparison_tolerances"] = {
        "control_fragment_negative_volume_count": audit["execution"]["negative_volume_topology"]["fragment_unique_element_count"],
        "control_fragment_deleted_element_count": 686 - audit["diagnostic_response"]["final_alive_fragment_count"],
        "expected_treatment_fragment_deleted_element_count": 0,
        "expected_treatment_fragment_negative_volume_erosion_message_count": 0,
        "hvar6_invalid_alive_count_max": 0,
        "added_mass_fraction_max": 0.0,
        "missing_or_nonfinite_policy": "fail_closed",
    }
    design["submission_sequence"] = {
        "wave_0": [JOB_ID],
        "wave_0_gate": (
            "Target parser/resource identity and no mass scaling are required. Normal completion is not assumed: "
            "timestep collapse, severe distortion or solver termination must be preserved as diagnostic evidence."
        ),
        "hold_after_wave_0": (
            "Do not release 2000 m/s, treatments, fitting or labels. Audit the negative-volume mechanism first."
        ),
        "resource_identity": (
            "Use exactly one node, 32 ranks, LS-DYNA R15 MPP double precision, and the same memory settings as Job 72751725."
        ),
    }
    design["deferred_not_generated"] = [
        {"candidate": "CONTACT keyword/formulation change", "reason": "Deferred until the PSFAIL-only mechanism isolation is audited."},
        {"candidate": "fragment remeshing or SPH", "reason": "Model-form redesign is premature before the deletion path is isolated."},
        {"candidate": "2000 m/s and all treatments", "reason": "Hold until numerical deletion and response availability are resolved."},
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
    parser = argparse.ArgumentParser(description="Freeze the R06k fragment negative-volume PSFAIL diagnostic")
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
    print(
        json.dumps(
            {
                "package_revision": REVISION,
                "job_id": JOB_ID,
                "design": str(args.design_output.resolve()),
                "contract": str(args.contract_output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
