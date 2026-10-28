from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from generate_p0_r06_diagnostics import strict_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_DESIGN = ROOT / "design" / "p0_r06f_measurement_repair_design.json"
DEFAULT_PARENT_CONTRACT = ROOT / "design" / "p0_r06f_real_output_contract.json"
DEFAULT_AUDIT = (
    ROOT
    / "analysis"
    / "72617857_r06f_0800_ctrl_20261028"
    / "r06f_return_audit.json"
)
DEFAULT_DESIGN_OUTPUT = ROOT / "design" / "p0_r06f_timestep_sensitivity_design.json"
DEFAULT_CONTRACT_OUTPUT = ROOT / "design" / "p0_r06f_timestep_output_contract.json"

REVISION = "R06_timestep_sensitivity_20261028"
DESIGN_STATUS = "frozen_after_r06f_recanary_pass_before_timestep_solver_outcome"
JOB_ID = "r06f_0800_tssfac04"
EXPERIMENT_ID = "r06f_timestep_sensitivity_800"


def _require_parent_audit(audit: dict) -> None:
    decision = audit["decision"]
    if (
        audit["source_evidence"]["platform_job_id"] != "72617857"
        or audit["source_evidence"]["job_id"] != "r06f_0800_ctrl"
        or decision["resource_identity"] != "pass"
        or decision["measurement_channel_repair"] != "pass"
        or decision["frozen_wave0_repeatability"] != "pass"
        or decision["release_r06f_0800_tssfac04"] is not True
    ):
        raise ValueError("Job 72617857 audit does not authorize the TSSFAC=0.4 diagnostic")


def build_contract(parent: dict, audit: dict) -> dict:
    _require_parent_audit(audit)
    contract = deepcopy(parent)
    contract["schema_version"] = "hf-p0-r06f-timestep-output-contract@1.0.0"
    contract["contract_id"] = "p0_r06f_timestep_output_contract"
    contract["status"] = DESIGN_STATUS
    boundary = contract["version_boundary"]
    boundary["applies_to_package_revision"] = REVISION
    boundary["effective_scope"] = "prospective_r06f_tssfac04_only"
    boundary["historical_r06f_control"] = {
        "package_revision": audit["source_evidence"]["package_revision"],
        "job_id": audit["source_evidence"]["job_id"],
        "platform_job_id": audit["source_evidence"]["platform_job_id"],
        "deck_sha256": audit["source_evidence"]["deck_sha256"],
        "resource_identity": audit["decision"]["resource_identity"],
        "frozen_wave0_repeatability": audit["decision"]["frozen_wave0_repeatability"],
        "retroactive_gate_rewrite": False,
    }
    contract["timestep_sensitivity"] = {
        "factor": "CONTROL_TIMESTEP.TSSFAC",
        "control_job_id": "r06f_0800_ctrl",
        "control_platform_job_id": "72617857",
        "control_level": 0.8,
        "treatment_job_id": JOB_ID,
        "treatment_level": 0.4,
        "dt2ms": 0.0,
        "erode": 0,
        "single_factor_rule": (
            "The treatment deck may differ from the released Job 72617857 control deck "
            "only in TITLE and CONTROL_TIMESTEP field 2 TSSFAC, 0.8 to 0.4."
        ),
        "comparison_tolerances": {
            "qM_qP_qE_absolute": 0.01,
            "qV_relative": 0.05,
            "energy_ratio_range": [0.95, 1.05],
            "hourglass_to_peak_internal_energy_max": 0.05,
            "added_mass_fraction_max": 0.0,
        },
    }
    contract["admission"] = {
        "timestep_diagnostic_submission": "allowed_after_static_release_verification",
        "timestep_solver_acceptance": "pending_external_target_solver_run",
        "r06f_2000_control": "denied_until_timestep_gate_passes",
        "treatments": "denied_until_both_controls_pass",
        "p0_response_calibration": "denied",
        "p1_submission": "denied",
        "engine_fitting": "denied",
        "reference_dataset_generation": "denied",
        "physical_validity": "not_established_requires_independent_experimental_evidence",
    }
    contract["claim_boundary"] = [
        "The resource-matched R06f recanary passed; its formal primary QoI remains unavailable.",
        "This package changes only TSSFAC from 0.8 to 0.4 and keeps DT2MS=0.",
        "INTFOR is not requested; missing evidence is never a zero response.",
        "Static PASS and normal termination do not establish numerical robustness, physical validity, P0/P1 admission, engine fitting, or dataset-label admission.",
    ]
    return contract


def build_design(parent: dict, contract: dict, audit: dict) -> dict:
    _require_parent_audit(audit)
    if contract["version_boundary"]["applies_to_package_revision"] != REVISION:
        raise ValueError("timestep output contract is not bound to the new revision")
    design = deepcopy(parent)
    design["schema_version"] = "hf-p0-r06-diagnostic-design@1.3.0"
    design["package_revision"] = REVISION
    design["design_status"] = DESIGN_STATUS
    design["title"] = "SCNet P0 R06f 800 m/s no-mass-scaling TSSFAC sensitivity"
    design["created_date"] = "2026-10-28"
    design["purpose"] = (
        "Test whether the frozen 800 m/s, 0.75 mm R06f response is numerically "
        "sensitive to halving TSSFAC from 0.8 to 0.4 while DT2MS remains zero."
    )
    design["provenance"] = {
        "parent_revision": audit["source_evidence"]["package_revision"],
        "parent_job_id": audit["source_evidence"]["job_id"],
        "parent_platform_job_id": audit["source_evidence"]["platform_job_id"],
        "parent_deck_sha256": audit["source_evidence"]["deck_sha256"],
        "parent_result": audit["decision"]["classification"],
        "parent_return_audit": "analysis/72617857_r06f_0800_ctrl_20261028/r06f_return_audit.json",
        "successor_contract": "design/p0_r06f_timestep_output_contract.json",
        "identity_rule": (
            "Mesh, coordinates, connectivity, IDs, PIDs, material, EOS, contact, "
            "section, boundary conditions, speed, termination, output cards, resource "
            "contract and every CONTROL_TIMESTEP field except TSSFAC remain identical "
            "to the released Job 72617857 control deck."
        ),
        "factor_scope": "CONTROL_TIMESTEP field 2 TSSFAC only, 0.8 to 0.4",
    }
    design["shared_measurement_upgrade"]["status"] = (
        "validated_on_resource_matched_job_72617857_and_frozen_for_timestep_diagnostic"
    )
    design["paired_experiments"] = [
        {
            "experiment_id": EXPERIMENT_ID,
            "question": (
                "Does halving TSSFAC from 0.8 to 0.4 materially change the frozen "
                "800 m/s response without mass scaling?"
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
            "control_level": 0.8,
            "treatment_level": 0.4,
            "external_control": {
                "job_id": "r06f_0800_ctrl",
                "platform_job_id": "72617857",
                "deck_sha256": audit["source_evidence"]["deck_sha256"],
                "resource_identity": "one_node_32_ranks",
            },
            "jobs": [
                {
                    "job_id": JOB_ID,
                    "directory": "job_01_v0800_tssfac04",
                    "role": "no_mass_scaling_timestep_sensitivity",
                    "termination_time_ms": 0.04,
                    "plate_only_psfail": False,
                    "tssfac": 0.4,
                    "factor_level": 0.4,
                    "platform_job_name": "uav_hf_r06f_0800_ts04",
                    "parent_r06f_job_id": "r06f_0800_ctrl",
                    "parent_platform_job_id": "72617857",
                }
            ],
            "preregistered_evaluation": {
                "primary_comparison": (
                    f"{JOB_ID} versus platform Job 72617857 using the frozen diagnostic q tolerances"
                ),
                "support_condition": (
                    "Target R15.0.2 parses and terminates normally; one-node/32-rank "
                    "resource identity and required outputs pass; qM/qP/qE absolute "
                    "differences are each <=0.01 and qV relative difference is <=0.05."
                ),
                "refute_condition": (
                    "Parser/runtime/resource failure, missing required output, mass scaling, "
                    "unexplained invalid data, or any frozen numerical tolerance is exceeded."
                ),
                "secondary_diagnostics": (
                    "Energy, negative-volume topology, alive masks, HVAR6 and NCFORC are "
                    "reported as mechanism evidence and do not replace the primary q gates."
                ),
                "claim_boundary": (
                    "A passed timestep comparison is numerical qualification only and does "
                    "not recover the unavailable formal primary QoI or establish physical validity."
                ),
            },
        }
    ]
    design["submission_sequence"] = {
        "wave_0": [JOB_ID],
        "wave_0_gate": (
            "Zero KEY errors, 1 node/32 ranks, normal termination, no mass scaling, "
            "run-integrity PASS, readable D3PART/full-D3PLOT HVAR6/NCFORC, and frozen "
            "q comparison tolerances against Job 72617857."
        ),
        "hold_after_wave_0": (
            "Do not submit the 2000 m/s control or any treatment until the returned "
            "TSSFAC=0.4 diagnostic is audited."
        ),
        "resource_identity": (
            "Use exactly one node, 32 ranks, LS-DYNA R15 MPP double precision, and "
            "the same memory settings as Job 72617857."
        ),
    }
    design["deferred_not_generated"] = [
        {
            "candidate": "r06f_2000_ctrl and both treatments",
            "reason": "Hold until the TSSFAC=0.4 numerical comparison passes.",
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
    parser = argparse.ArgumentParser(description="Freeze the R06f TSSFAC=0.4 diagnostic")
    parser.add_argument("--parent-design", type=Path, default=DEFAULT_PARENT_DESIGN)
    parser.add_argument("--parent-contract", type=Path, default=DEFAULT_PARENT_CONTRACT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--design-output", type=Path, default=DEFAULT_DESIGN_OUTPUT)
    parser.add_argument("--contract-output", type=Path, default=DEFAULT_CONTRACT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parent_design = strict_json(args.parent_design.resolve())
    parent_contract = strict_json(args.parent_contract.resolve())
    audit = strict_json(args.audit.resolve())
    contract = build_contract(parent_contract, audit)
    design = build_design(parent_design, contract, audit)
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
