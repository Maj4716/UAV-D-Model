from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from generate_p0_r06_diagnostics import strict_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_DESIGN = ROOT / "design" / "p0_r06i_terminal_timestep_refinement_design.json"
DEFAULT_PARENT_CONTRACT = ROOT / "design" / "p0_r06i_terminal_timestep_output_contract.json"
DEFAULT_AUDIT = (
    ROOT
    / "analysis"
    / "72740022_r06i_0800_tssfac005_20261030"
    / "r06i_tssfac005_return_audit.json"
)
DEFAULT_DESIGN_OUTPUT = ROOT / "design" / "p0_r06j_fragment_erosion_isolation_design.json"
DEFAULT_CONTRACT_OUTPUT = ROOT / "design" / "p0_r06j_fragment_erosion_output_contract.json"

REVISION = "R06_fragment_erosion_isolation_20261030"
DESIGN_STATUS = "frozen_after_r06i_terminal_timestep_stable_but_response_unavailable_before_fragment_erod1_outcome"
JOB_ID = "r06j_0800_fragment_erod1"
EXPERIMENT_ID = "r06j_fragment_erosion_isolation_800"


def _require_parent_audit(audit: dict) -> None:
    decision = audit["decision"]
    if (
        audit["source_evidence"]["platform_job_id"] != "72740022"
        or audit["source_evidence"]["job_id"] != "r06i_0800_tssfac005"
        or decision["solver_parser_runtime"] != "pass"
        or decision["input_identity"] != "pass"
        or decision["resource_identity"] != "pass"
        or decision["measurement_channels"] != "pass"
        or decision["timestep_terminal_consistency"] != "pass"
        or decision["response_availability"] != "fail"
        or decision["release_r06j_fragment_erod1_diagnostic"] is not True
        or decision["release_any_further_timestep_halving"] is not False
    ):
        raise ValueError("Job 72740022 audit does not authorize the fragment EROD diagnostic")


def build_contract(parent: dict, audit: dict) -> dict:
    _require_parent_audit(audit)
    contract = deepcopy(parent)
    contract["schema_version"] = "hf-p0-r06j-fragment-erosion-output-contract@1.0.0"
    contract["contract_id"] = "p0_r06j_fragment_erosion_output_contract"
    contract["status"] = DESIGN_STATUS
    boundary = contract["version_boundary"]
    boundary["applies_to_package_revision"] = REVISION
    boundary["effective_scope"] = "prospective_r06j_fragment_erod1_only"
    boundary["parent_tssfac005"] = {
        "package_revision": audit["source_evidence"]["package_revision"],
        "job_id": audit["source_evidence"]["job_id"],
        "platform_job_id": audit["source_evidence"]["platform_job_id"],
        "deck_sha256": audit["source_evidence"]["deck_sha256"],
        "timestep_terminal_consistency": audit["decision"]["timestep_terminal_consistency"],
        "response_availability": audit["decision"]["response_availability"],
        "retroactive_gate_rewrite": False,
    }
    contract["fragment_erosion_isolation"] = {
        "factor": "fragment MAT_015 Card 4 field 3 EROD",
        "control_job_id": "r06i_0800_tssfac005",
        "control_platform_job_id": "72740022",
        "control_level": 0,
        "treatment_job_id": JOB_ID,
        "treatment_level": 1,
        "plate_mat015_erod": 0,
        "tssfac": 0.05,
        "dt2ms": 0.0,
        "single_factor_rule": (
            "The R06j deck may differ from the returned R06i deck only in TITLE and fragment material "
            "MAT_015 Card 4 field 3 EROD, 0 to 1. Plate EROD remains 0."
        ),
        "solver_semantics": {
            "erod_0": "element erosion is allowed after failure (default)",
            "erod_nonzero": (
                "the failed element is not eroded; its deviatoric stresses are set to zero after failure"
            ),
            "source": "ANSYS LS-DYNA Materials Manual R14, MAT_015 Card 4 EROD",
        },
        "diagnostic_gates": {
            "normal_termination_and_32_ranks": True,
            "no_mass_scaling": True,
            "d3part_full_d3plot_hvar6_crosscheck": "required",
            "fragment_deleted_upstream_mass_fraction": "compare_to_job_72740022",
            "fragment_hvar6_failure_evidence": "must_remain_available_and_finite",
            "negative_volume_topology": "report_fail_closed_if_solver_aborts",
            "energy_and_contact_outputs": "mechanism_evidence_only",
        },
        "interpretation_rules": {
            "deletion_suppressed_with_failure_damage_present": (
                "supports fragment failure erosion as the cause of the upstream deletion mechanism only"
            ),
            "negative_volume_or_solver_failure": (
                "supports numerical dependence on erosion but does not validate the material or contact model"
            ),
            "primary_qoi_policy": (
                "never promote EROD=1 response values to physical labels, calibration targets or reference data"
            ),
        },
    }
    contract["admission"] = {
        "fragment_erod1_diagnostic_submission": "allowed_after_static_release_verification",
        "production_material_policy": "pending_external_target_solver_diagnostic",
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
        "R06i established local timestep consistency but not response availability.",
        "R06j changes only the fragment material EROD flag from 0 to 1; plate erosion remains enabled with EROD=0.",
        "EROD=1 intentionally changes the post-failure numerical treatment and is diagnostic, not a physical treatment.",
        "The model continues to start from the same post-explosion fragment state.",
        "Neither static PASS nor a normal R06j termination authorizes engine fitting, damage-tree fitting or dataset admission.",
    ]
    return contract


def build_design(parent: dict, contract: dict, audit: dict) -> dict:
    _require_parent_audit(audit)
    if contract["version_boundary"]["applies_to_package_revision"] != REVISION:
        raise ValueError("fragment erosion output contract is not bound to the new revision")
    design = deepcopy(parent)
    design["schema_version"] = "hf-p0-r06-diagnostic-design@1.7.0"
    design["package_revision"] = REVISION
    design["design_status"] = DESIGN_STATUS
    design["title"] = "SCNet P0 R06j 800 m/s fragment Johnson-Cook erosion isolation"
    design["created_date"] = "2026-10-30"
    design["purpose"] = (
        "Isolate whether the unavailable R06i response is caused by deletion of failed fragment elements, "
        "while keeping the terminal TSSFAC=0.05 policy and every other model field frozen."
    )
    design["provenance"] = {
        "parent_revision": audit["source_evidence"]["package_revision"],
        "parent_job_id": audit["source_evidence"]["job_id"],
        "parent_platform_job_id": audit["source_evidence"]["platform_job_id"],
        "parent_deck_sha256": audit["source_evidence"]["deck_sha256"],
        "parent_result": audit["decision"]["classification"],
        "parent_return_audit": "analysis/72740022_r06i_0800_tssfac005_20261030/r06i_tssfac005_return_audit.json",
        "successor_contract": "design/p0_r06j_fragment_erosion_output_contract.json",
        "identity_rule": (
            "Mesh, coordinates, connectivity, IDs, PIDs, post-explosion initial state, material strength and "
            "failure coefficients, EOS, contact, section, boundaries, speed, termination, timestep, output cards "
            "and resources remain identical to Job 72740022. Only fragment MAT_015 Card 4 EROD changes 0 to 1."
        ),
        "factor_scope": "fragment MAT_015 Card 4 field 3 EROD only, 0 to 1",
    }
    design["frozen_solver_controls"]["control_timestep"]["tssfac"] = 0.05
    design["shared_measurement_upgrade"]["status"] = (
        "validated_on_job_72740022_and_frozen_for_fragment_erod1_isolation"
    )
    design["explosion_instant_start_policy"] = deepcopy(contract["explosion_instant_start_policy"])
    design["paired_experiments"] = [
        {
            "experiment_id": EXPERIMENT_ID,
            "question": (
                "Does suppressing fragment element erosion remove the upstream deleted-mass mechanism while "
                "Johnson-Cook damage/failure evidence remains observable?"
            ),
            "case_identity": {
                "r05_case_id": "p0_v0800_h0p75",
                "r05_original_index": 2,
                "impact_speed_mps": 800.0,
                "mesh_mm": 0.75,
                "plate_part_id": 1002,
                "fragment_part_id": 2002,
            },
            "factor": "fragment_mat015_erod_flag",
            "control_level": 0,
            "treatment_level": 1,
            "external_control": {
                "job_id": "r06i_0800_tssfac005",
                "platform_job_id": "72740022",
                "deck_sha256": audit["source_evidence"]["deck_sha256"],
                "resource_identity": "one_node_32_ranks",
                "tssfac": 0.05,
                "fragment_mat015_erod": 0,
                "primary_qoi_available": audit["diagnostic_response"]["primary_qoi_available"],
                "unavailable_reasons": audit["diagnostic_response"]["unavailable_reasons"],
                "deleted_upstream_mass_fraction": audit["diagnostic_response"]["deleted_upstream_mass_fraction"],
                "crossing_event_count": audit["diagnostic_response"]["crossing_event_count"],
            },
            "jobs": [
                {
                    "job_id": JOB_ID,
                    "directory": "job_01_v0800_fragment_erod1",
                    "role": "fragment_failure_erosion_suppression_diagnostic",
                    "termination_time_ms": 0.04,
                    "plate_only_psfail": False,
                    "tssfac": 0.05,
                    "fragment_mat015_erod": 1,
                    "factor_level": 1,
                    "platform_job_name": "uav_hf_r06j_0800_ferod1",
                    "parent_r06i_job_id": "r06i_0800_tssfac005",
                    "parent_platform_job_id": "72740022",
                }
            ],
            "preregistered_evaluation": deepcopy(contract["fragment_erosion_isolation"]),
        }
    ]
    design["paired_comparison_tolerances"] = {
        "fragment_deleted_upstream_mass_fraction_control": audit["diagnostic_response"]["deleted_upstream_mass_fraction"],
        "suppression_direction": "treatment_less_than_control",
        "zero_deletion_is_expected_but_not_required_for_mechanism_inference": True,
        "hvar6_invalid_alive_count_max": 0,
        "energy_ratio_range": [0.95, 1.05],
        "hourglass_to_peak_internal_energy_max": 0.05,
        "added_mass_fraction_max": 0.0,
        "missing_or_nonfinite_policy": "fail_closed",
    }
    design["submission_sequence"] = {
        "wave_0": [JOB_ID],
        "wave_0_gate": (
            "Zero KEY errors, 1 node/32 ranks, no mass scaling, readable D3PART/full-D3PLOT HVAR6/NCFORC, "
            "and a formal comparison with Job 72740022. Solver failure/negative volume is retained as diagnostic evidence."
        ),
        "hold_after_wave_0": (
            "Do not release 2000 m/s, treatments, fitting or labels. Audit the single-factor erosion mechanism first."
        ),
        "resource_identity": (
            "Use exactly one node, 32 ranks, LS-DYNA R15 MPP double precision, and the same memory settings as Job 72740022."
        ),
    }
    design["deferred_not_generated"] = [
        {"candidate": "TSSFAC=0.025", "reason": "Denied by the terminal timestep stop rule."},
        {"candidate": "plate EROD change", "reason": "Outside the isolated fragment-only factor."},
        {"candidate": "2000 m/s and all treatments", "reason": "Hold until the deletion mechanism is resolved."},
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
    parser = argparse.ArgumentParser(description="Freeze the R06j fragment EROD=1 isolation diagnostic")
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
