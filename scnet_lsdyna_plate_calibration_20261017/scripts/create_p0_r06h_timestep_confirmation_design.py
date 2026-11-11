from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from generate_p0_r06_diagnostics import strict_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_DESIGN = ROOT / "design" / "p0_r06g_timestep_refinement_design.json"
DEFAULT_PARENT_CONTRACT = ROOT / "design" / "p0_r06g_timestep_output_contract.json"
DEFAULT_AUDIT = (
    ROOT
    / "analysis"
    / "72639935_r06g_0800_tssfac02_20261029"
    / "r06g_tssfac02_return_audit.json"
)
DEFAULT_DESIGN_OUTPUT = ROOT / "design" / "p0_r06h_timestep_confirmation_design.json"
DEFAULT_CONTRACT_OUTPUT = ROOT / "design" / "p0_r06h_timestep_output_contract.json"

REVISION = "R06_timestep_confirmation_20261029"
DESIGN_STATUS = "frozen_after_r06g_tssfac02_local_gate_pass_before_tssfac01_outcome"
JOB_ID = "r06h_0800_tssfac01"
EXPERIMENT_ID = "r06h_timestep_confirmation_800"


def _require_parent_audit(audit: dict) -> None:
    decision = audit["decision"]
    if (
        audit["source_evidence"]["platform_job_id"] != "72639935"
        or audit["source_evidence"]["job_id"] != "r06g_0800_tssfac02"
        or decision["solver_parser_runtime"] != "pass"
        or decision["input_identity"] != "pass"
        or decision["resource_identity"] != "pass"
        or decision["measurement_channels"] != "pass"
        or decision["timestep_refinement_gate"] != "pass"
        or decision["release_r06h_0800_tssfac01"] is not True
    ):
        raise ValueError("Job 72639935 audit does not authorize the TSSFAC=0.1 confirmation")


def build_contract(parent: dict, audit: dict) -> dict:
    _require_parent_audit(audit)
    contract = deepcopy(parent)
    contract["schema_version"] = "hf-p0-r06h-timestep-output-contract@1.0.0"
    contract["contract_id"] = "p0_r06h_timestep_output_contract"
    contract["status"] = DESIGN_STATUS
    boundary = contract["version_boundary"]
    boundary["applies_to_package_revision"] = REVISION
    boundary["effective_scope"] = "prospective_r06h_tssfac01_only"
    boundary.pop("parent_tssfac04", None)
    boundary["parent_tssfac02"] = {
        "package_revision": audit["source_evidence"]["package_revision"],
        "job_id": audit["source_evidence"]["job_id"],
        "platform_job_id": audit["source_evidence"]["platform_job_id"],
        "deck_sha256": audit["source_evidence"]["deck_sha256"],
        "solver_parser_runtime": audit["decision"]["solver_parser_runtime"],
        "timestep_refinement_gate": audit["decision"]["timestep_refinement_gate"],
        "retroactive_gate_rewrite": False,
    }
    contract.pop("timestep_refinement", None)
    contract["timestep_confirmation"] = {
        "factor": "CONTROL_TIMESTEP.TSSFAC",
        "control_job_id": "r06g_0800_tssfac02",
        "control_platform_job_id": "72639935",
        "control_level": 0.2,
        "treatment_job_id": JOB_ID,
        "treatment_level": 0.1,
        "dt2ms": 0.0,
        "erode": 0,
        "single_factor_rule": (
            "The confirmation deck may differ from the returned TSSFAC=0.2 deck only "
            "in TITLE and CONTROL_TIMESTEP field 2 TSSFAC, 0.2 to 0.1."
        ),
        "comparison_tolerances": {
            "qM_qP_qE_absolute": 0.01,
            "qV_relative_when_both_available": 0.05,
            "qV_both_null_policy": "not_applicable_only_when_both_runs_have_zero_confirmed_crossing_mass",
            "deleted_upstream_mass_fraction_absolute": 0.01,
            "energy_ratio_range": [0.95, 1.05],
            "hourglass_to_peak_internal_energy_max": 0.05,
            "added_mass_fraction_max": 0.0,
            "missing_or_nonfinite_policy": "fail_closed_except_preregistered_both_null_qV_case",
        },
    }
    contract["explosion_instant_start_policy"] = {
        "current_model_status": "already_starts_from_post_explosion_fragment_initial_state",
        "pre_explosion_detonation_or_flight_simulated": False,
        "r06h_rule": "initial geometry, velocity and all other state remain byte-equivalent to Job 72639935",
        "future_full_warhead_rule": (
            "A full-warhead high-fidelity model may start at explosion time only after the mapped "
            "mass, momentum, energy, fragment-state and boundary-state transfer is frozen and audited."
        ),
    }
    contract["admission"] = {
        "timestep_confirmation_submission": "allowed_after_static_release_verification",
        "production_timestep_strategy": "pending_external_target_solver_run",
        "r06f_2000_control": "denied_until_confirmation_audit",
        "treatments": "denied_until_control_strategy_is_converged",
        "p0_response_calibration": "denied",
        "p1_submission": "denied",
        "engine_fitting": "denied",
        "damage_tree_fitting": "denied",
        "reference_dataset_generation": "denied",
        "physical_validity": "not_established_requires_independent_experimental_evidence",
    }
    contract["claim_boundary"] = [
        "Job 72639935 passed the frozen 0.4-versus-0.2 local timestep-refinement gate for one 800 m/s case.",
        "This package changes only TSSFAC from 0.2 to 0.1 and keeps DT2MS=0.",
        "The current fragment-plate model already begins from a post-explosion fragment state, so no pre-explosion phase is added.",
        "A null qV is not zero and is only a convergence non-applicable case when both runs have zero confirmed crossing mass.",
        "Static PASS and normal termination do not establish physical validity, P0/P1 admission, engine or damage-tree fitting, or dataset-label admission.",
    ]
    return contract


def build_design(parent: dict, contract: dict, audit: dict) -> dict:
    _require_parent_audit(audit)
    if contract["version_boundary"]["applies_to_package_revision"] != REVISION:
        raise ValueError("confirmation output contract is not bound to the new revision")
    design = deepcopy(parent)
    design["schema_version"] = "hf-p0-r06-diagnostic-design@1.5.0"
    design["package_revision"] = REVISION
    design["design_status"] = DESIGN_STATUS
    design["title"] = "SCNet P0 R06h 800 m/s no-mass-scaling TSSFAC confirmation"
    design["created_date"] = "2026-10-29"
    design["purpose"] = (
        "Confirm that the 0.4-to-0.2 local agreement lies on a stable fine-timestep plateau "
        "by one additional factor-two refinement to TSSFAC=0.1 while DT2MS remains zero."
    )
    design["provenance"] = {
        "parent_revision": audit["source_evidence"]["package_revision"],
        "parent_job_id": audit["source_evidence"]["job_id"],
        "parent_platform_job_id": audit["source_evidence"]["platform_job_id"],
        "parent_deck_sha256": audit["source_evidence"]["deck_sha256"],
        "parent_result": audit["decision"]["classification"],
        "parent_return_audit": "analysis/72639935_r06g_0800_tssfac02_20261029/r06g_tssfac02_return_audit.json",
        "successor_contract": "design/p0_r06h_timestep_output_contract.json",
        "identity_rule": (
            "Mesh, coordinates, connectivity, IDs, PIDs, initial state, material, EOS, contact, "
            "section, boundary conditions, speed, termination, output cards, resource contract and "
            "every CONTROL_TIMESTEP field except TSSFAC remain identical to returned Job 72639935."
        ),
        "factor_scope": "CONTROL_TIMESTEP field 2 TSSFAC only, 0.2 to 0.1",
    }
    design["frozen_solver_controls"]["control_timestep"]["tssfac"] = 0.2
    design["shared_measurement_upgrade"]["status"] = (
        "validated_on_job_72639935_and_frozen_for_tssfac01_confirmation"
    )
    design["explosion_instant_start_policy"] = deepcopy(contract["explosion_instant_start_policy"])
    design["paired_experiments"] = [
        {
            "experiment_id": EXPERIMENT_ID,
            "question": (
                "Does a factor-two TSSFAC reduction from 0.2 to 0.1 preserve the 800 m/s "
                "no-confirmed-crossing diagnostic response?"
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
            "control_level": 0.2,
            "treatment_level": 0.1,
            "external_control": {
                "job_id": "r06g_0800_tssfac02",
                "platform_job_id": "72639935",
                "deck_sha256": audit["source_evidence"]["deck_sha256"],
                "resource_identity": "one_node_32_ranks",
                "diagnostic_values": audit["diagnostic_response"]["diagnostic_values_before_availability_gate"],
                "deleted_upstream_mass_fraction": audit["diagnostic_response"]["deleted_upstream_mass_fraction"],
                "crossing_event_count": audit["diagnostic_response"]["crossing_event_count"],
            },
            "jobs": [
                {
                    "job_id": JOB_ID,
                    "directory": "job_01_v0800_tssfac01",
                    "role": "no_mass_scaling_timestep_confirmation",
                    "termination_time_ms": 0.04,
                    "plate_only_psfail": False,
                    "tssfac": 0.1,
                    "factor_level": 0.1,
                    "platform_job_name": "uav_hf_r06h_0800_ts01",
                    "parent_r06g_job_id": "r06g_0800_tssfac02",
                    "parent_platform_job_id": "72639935",
                }
            ],
            "preregistered_evaluation": {
                "primary_comparison": (
                    f"{JOB_ID} versus platform Job 72639935 using no-crossing-aware convergence gates"
                ),
                "support_condition": (
                    "Target R15.0.2 parses and terminates normally at one node/32 ranks; required "
                    "outputs pass; qM/qP/qE absolute differences are each <=0.01; deleted-upstream "
                    "mass-fraction difference is <=0.01; and qV is either within 5% when available "
                    "in both runs or not applicable because both runs have zero confirmed crossing mass."
                ),
                "refute_condition": (
                    "Parser/runtime/resource failure, missing required output, mass scaling, nonfinite "
                    "evidence, response/deleted-mass disagreement, or asymmetric qV availability."
                ),
                "secondary_diagnostics": (
                    "Energy, negative-volume topology, alive masks, HVAR6 and NCFORC are reported as "
                    "mechanism evidence and do not replace the primary gates."
                ),
                "claim_boundary": (
                    "Agreement would support selecting a local timestep policy for this single 800 m/s "
                    "case only, not material truth or dataset admission."
                ),
            },
        }
    ]
    design["paired_comparison_tolerances"] = deepcopy(
        contract["timestep_confirmation"]["comparison_tolerances"]
    )
    design["submission_sequence"] = {
        "wave_0": [JOB_ID],
        "wave_0_gate": (
            "Zero KEY errors, 1 node/32 ranks, normal termination, no mass scaling, run-integrity PASS, "
            "readable D3PART/full-D3PLOT HVAR6/NCFORC, and the preregistered no-crossing-aware comparison "
            "against Job 72639935."
        ),
        "hold_after_wave_0": (
            "Do not submit 2000 m/s or any treatment until the returned TSSFAC=0.1 confirmation is "
            "audited and the production timestep strategy is explicitly selected."
        ),
        "resource_identity": (
            "Use exactly one node, 32 ranks, LS-DYNA R15 MPP double precision, and the same memory "
            "settings as Job 72639935."
        ),
    }
    design["deferred_not_generated"] = [
        {
            "candidate": "2000 m/s control and all treatments",
            "reason": "Hold until the TSSFAC=0.1 confirmation diagnostic is audited.",
        }
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
    parser = argparse.ArgumentParser(description="Freeze the R06h TSSFAC=0.1 confirmation")
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
