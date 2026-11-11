from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from generate_p0_r06_diagnostics import strict_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_DESIGN = ROOT / "design" / "p0_r06h_timestep_confirmation_design.json"
DEFAULT_PARENT_CONTRACT = ROOT / "design" / "p0_r06h_timestep_output_contract.json"
DEFAULT_AUDIT = (
    ROOT
    / "analysis"
    / "72701744_r06h_0800_tssfac01_20261030"
    / "r06h_tssfac01_return_audit.json"
)
DEFAULT_DESIGN_OUTPUT = ROOT / "design" / "p0_r06i_terminal_timestep_refinement_design.json"
DEFAULT_CONTRACT_OUTPUT = ROOT / "design" / "p0_r06i_terminal_timestep_output_contract.json"

REVISION = "R06_timestep_terminal_refinement_20261030"
DESIGN_STATUS = "frozen_after_r06h_tssfac01_confirmation_fail_before_tssfac005_outcome"
JOB_ID = "r06i_0800_tssfac005"
EXPERIMENT_ID = "r06i_terminal_timestep_refinement_800"


def _require_parent_audit(audit: dict) -> None:
    decision = audit["decision"]
    if (
        audit["source_evidence"]["platform_job_id"] != "72701744"
        or audit["source_evidence"]["job_id"] != "r06h_0800_tssfac01"
        or decision["solver_parser_runtime"] != "pass"
        or decision["input_identity"] != "pass"
        or decision["resource_identity"] != "pass"
        or decision["measurement_channels"] != "pass"
        or decision["timestep_confirmation_gate"] != "fail"
        or decision["failure_scope"] != "deleted_upstream_divergence_and_response_unavailability"
        or decision["release_r06i_0800_tssfac005"] is not True
        or decision["release_any_further_timestep_halving_after_r06i"] is not False
    ):
        raise ValueError("Job 72701744 audit does not authorize the terminal TSSFAC=0.05 refinement")


def build_contract(parent: dict, audit: dict) -> dict:
    _require_parent_audit(audit)
    contract = deepcopy(parent)
    contract["schema_version"] = "hf-p0-r06i-terminal-timestep-output-contract@1.0.0"
    contract["contract_id"] = "p0_r06i_terminal_timestep_output_contract"
    contract["status"] = DESIGN_STATUS
    boundary = contract["version_boundary"]
    boundary["applies_to_package_revision"] = REVISION
    boundary["effective_scope"] = "prospective_r06i_tssfac005_only"
    boundary.pop("parent_tssfac02", None)
    boundary["parent_tssfac01"] = {
        "package_revision": audit["source_evidence"]["package_revision"],
        "job_id": audit["source_evidence"]["job_id"],
        "platform_job_id": audit["source_evidence"]["platform_job_id"],
        "deck_sha256": audit["source_evidence"]["deck_sha256"],
        "solver_parser_runtime": audit["decision"]["solver_parser_runtime"],
        "timestep_confirmation_gate": audit["decision"]["timestep_confirmation_gate"],
        "failure_scope": audit["decision"]["failure_scope"],
        "retroactive_gate_rewrite": False,
    }
    contract.pop("timestep_confirmation", None)
    contract["timestep_terminal_refinement"] = {
        "factor": "CONTROL_TIMESTEP.TSSFAC",
        "control_job_id": "r06h_0800_tssfac01",
        "control_platform_job_id": "72701744",
        "control_level": 0.1,
        "treatment_job_id": JOB_ID,
        "treatment_level": 0.05,
        "dt2ms": 0.0,
        "erode": 0,
        "single_factor_rule": (
            "The terminal refinement deck may differ from the returned TSSFAC=0.1 deck only "
            "in TITLE and CONTROL_TIMESTEP field 2 TSSFAC, 0.1 to 0.05."
        ),
        "comparison_tolerances": {
            "qM_qP_qE_absolute": 0.01,
            "qV_relative_when_both_available": 0.05,
            "qV_both_null_policy": "not_applicable_only_when_both_runs_have_zero_confirmed_crossing_mass",
            "deleted_upstream_mass_fraction_absolute": 0.01,
            "deleted_upstream_mass_fraction_availability_max": 0.05,
            "energy_ratio_range": [0.95, 1.05],
            "hourglass_to_peak_internal_energy_max": 0.05,
            "added_mass_fraction_max": 0.0,
            "missing_or_nonfinite_policy": "fail_closed_except_preregistered_both_null_qV_case",
        },
        "terminal_stop_rule": {
            "no_automatic_tssfac_below": 0.05,
            "if_all_gates_and_response_availability_pass": (
                "Select TSSFAC=0.05 as the local production candidate, then audit whether the 2000 m/s control may be released."
            ),
            "if_diagnostic_values_agree_but_response_unavailable": (
                "Stop timestep halving and pivot to fragment deletion, material failure and contact diagnostics."
            ),
            "if_any_comparison_gate_fails": (
                "Stop timestep halving and classify the current control as numerically/model-form unstable."
            ),
        },
    }
    contract["explosion_instant_start_policy"] = deepcopy(parent["explosion_instant_start_policy"])
    contract["admission"] = {
        "terminal_timestep_refinement_submission": "allowed_after_static_release_verification",
        "production_timestep_strategy": "pending_external_target_solver_run",
        "further_timestep_halving": "denied_after_r06i_without_new_explicit_diagnostic_design",
        "r06f_2000_control": "denied_until_r06i_audit",
        "treatments": "denied_until_control_strategy_and_response_availability_are_resolved",
        "p0_response_calibration": "denied",
        "p1_submission": "denied",
        "engine_fitting": "denied",
        "damage_tree_fitting": "denied",
        "reference_dataset_generation": "denied",
        "physical_validity": "not_established_requires_independent_experimental_evidence",
    }
    contract["claim_boundary"] = [
        "Job 72701744 solved successfully but failed the 0.2-versus-0.1 confirmation because deleted-upstream mass diverged and the primary response became unavailable.",
        "This package changes only TSSFAC from 0.1 to 0.05 and keeps DT2MS=0.",
        "This is the final automatic timestep halving; no TSSFAC=0.025 job is authorized by this contract.",
        "The current fragment-plate model already begins from a post-explosion fragment state, so no pre-explosion phase is added.",
        "Static PASS and normal termination do not establish physical validity, P0/P1 admission, engine or damage-tree fitting, or dataset-label admission.",
    ]
    return contract


def build_design(parent: dict, contract: dict, audit: dict) -> dict:
    _require_parent_audit(audit)
    if contract["version_boundary"]["applies_to_package_revision"] != REVISION:
        raise ValueError("terminal refinement output contract is not bound to the new revision")
    design = deepcopy(parent)
    design["schema_version"] = "hf-p0-r06-diagnostic-design@1.6.0"
    design["package_revision"] = REVISION
    design["design_status"] = DESIGN_STATUS
    design["title"] = "SCNet P0 R06i 800 m/s no-mass-scaling terminal TSSFAC refinement"
    design["created_date"] = "2026-10-30"
    design["purpose"] = (
        "Run one final factor-two refinement to TSSFAC=0.05 to determine whether the increasing "
        "fragment deletion at TSSFAC=0.1 approaches a stable fine-timestep limit or remains divergent."
    )
    design["provenance"] = {
        "parent_revision": audit["source_evidence"]["package_revision"],
        "parent_job_id": audit["source_evidence"]["job_id"],
        "parent_platform_job_id": audit["source_evidence"]["platform_job_id"],
        "parent_deck_sha256": audit["source_evidence"]["deck_sha256"],
        "parent_result": audit["decision"]["classification"],
        "parent_return_audit": "analysis/72701744_r06h_0800_tssfac01_20261030/r06h_tssfac01_return_audit.json",
        "successor_contract": "design/p0_r06i_terminal_timestep_output_contract.json",
        "identity_rule": (
            "Mesh, coordinates, connectivity, IDs, PIDs, initial state, material, EOS, contact, "
            "section, boundary conditions, speed, termination, output cards, resource contract and "
            "every CONTROL_TIMESTEP field except TSSFAC remain identical to returned Job 72701744."
        ),
        "factor_scope": "CONTROL_TIMESTEP field 2 TSSFAC only, 0.1 to 0.05",
    }
    design["frozen_solver_controls"]["control_timestep"]["tssfac"] = 0.1
    design["shared_measurement_upgrade"]["status"] = (
        "validated_on_job_72701744_and_frozen_for_tssfac005_terminal_refinement"
    )
    design["explosion_instant_start_policy"] = deepcopy(contract["explosion_instant_start_policy"])
    design["paired_experiments"] = [
        {
            "experiment_id": EXPERIMENT_ID,
            "question": (
                "Does the TSSFAC=0.1 deletion divergence stabilize under one terminal refinement to 0.05?"
            ),
            "case_identity": {
                "r05_case_id": "p0_v0800_h0p75",
                "r05_original_index": 2,
                "impact_speed_mps": 800.0,
                "mesh_mm": 0.75,
                "plate_part_id": 1002,
                "fragment_part_id": 2002,
            },
            "factor": "control_timestep_tssfac",
            "control_level": 0.1,
            "treatment_level": 0.05,
            "external_control": {
                "job_id": "r06h_0800_tssfac01",
                "platform_job_id": "72701744",
                "deck_sha256": audit["source_evidence"]["deck_sha256"],
                "resource_identity": "one_node_32_ranks",
                "diagnostic_values": audit["diagnostic_response"]["diagnostic_values_before_availability_gate"],
                "primary_qoi_available": audit["diagnostic_response"]["primary_qoi_available"],
                "unavailable_reasons": audit["diagnostic_response"]["unavailable_reasons"],
                "deleted_upstream_mass_fraction": audit["diagnostic_response"]["deleted_upstream_mass_fraction"],
                "crossing_event_count": audit["diagnostic_response"]["crossing_event_count"],
            },
            "jobs": [
                {
                    "job_id": JOB_ID,
                    "directory": "job_01_v0800_tssfac005",
                    "role": "no_mass_scaling_terminal_timestep_refinement",
                    "termination_time_ms": 0.04,
                    "plate_only_psfail": False,
                    "tssfac": 0.05,
                    "factor_level": 0.05,
                    "platform_job_name": "uav_hf_r06i_0800_ts005",
                    "parent_r06h_job_id": "r06h_0800_tssfac01",
                    "parent_platform_job_id": "72701744",
                }
            ],
            "preregistered_evaluation": {
                "primary_comparison": (
                    f"{JOB_ID} versus platform Job 72701744 using no-crossing-aware convergence, deletion and response-availability gates"
                ),
                "support_condition": (
                    "Target R15.0.2 parses and terminates normally at one node/32 ranks; required outputs pass; "
                    "qM/qP/qE absolute differences are each <=0.01; deleted-upstream mass-fraction difference "
                    "is <=0.01 and the candidate value is <=0.05; qV follows the frozen both-null rule; energy "
                    "and no-mass-scaling gates pass."
                ),
                "refute_condition": (
                    "Parser/runtime/resource failure, missing output, mass scaling, nonfinite evidence, response "
                    "unavailability, deletion disagreement, or asymmetric qV availability."
                ),
                "secondary_diagnostics": (
                    "Energy, negative-volume topology, alive masks, HVAR6 and NCFORC are mechanism evidence and "
                    "do not replace the primary gates."
                ),
                "terminal_stop_rule": deepcopy(
                    contract["timestep_terminal_refinement"]["terminal_stop_rule"]
                ),
                "claim_boundary": (
                    "Agreement would support a local timestep policy for this single 800 m/s case only, not "
                    "material truth or dataset admission."
                ),
            },
        }
    ]
    design["paired_comparison_tolerances"] = deepcopy(
        contract["timestep_terminal_refinement"]["comparison_tolerances"]
    )
    design["submission_sequence"] = {
        "wave_0": [JOB_ID],
        "wave_0_gate": (
            "Zero KEY errors, 1 node/32 ranks, normal termination, no mass scaling, run-integrity PASS, "
            "readable D3PART/full-D3PLOT HVAR6/NCFORC, and the preregistered comparison against Job 72701744."
        ),
        "hold_after_wave_0": (
            "Do not submit TSSFAC=0.025, 2000 m/s or any treatment. Audit R06i and then either select a "
            "stable policy or pivot to deletion/material/contact diagnostics."
        ),
        "resource_identity": (
            "Use exactly one node, 32 ranks, LS-DYNA R15 MPP double precision, and the same memory settings as Job 72701744."
        ),
    }
    design["deferred_not_generated"] = [
        {"candidate": "TSSFAC=0.025", "reason": "Not authorized; R06i is the terminal automatic halving."},
        {
            "candidate": "2000 m/s control and all treatments",
            "reason": "Hold until R06i is audited and response availability is resolved.",
        },
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the R06i terminal TSSFAC=0.05 refinement")
    parser.add_argument("--parent-design", type=Path, default=DEFAULT_PARENT_DESIGN)
    parser.add_argument("--parent-contract", type=Path, default=DEFAULT_PARENT_CONTRACT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--design-output", type=Path, default=DEFAULT_DESIGN_OUTPUT)
    parser.add_argument("--contract-output", type=Path, default=DEFAULT_CONTRACT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
