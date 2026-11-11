from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_r06g_tssfac02_return_audit import strict_json


ROOT = Path(__file__).resolve().parents[1]
REVISION = "R06_speed_upper_anchor_2000_20261102"
JOB_ID = "r06r_2000_speed_upper_anchor"
EXPERIMENT_ID = "r06r_numerical_speed_upper_anchor_2000"
PARENT_JOB_ID = "r06q_0800_d3plot_0p00025_confirmation"
PARENT_PLATFORM_JOB_ID = "72936268"
PARENT_DECK_SHA256 = "sha256:a6229cda9778adb13b4c174346a4356deea3e62b175fda9cf8c7dbf12be60002"
PARENT_SPEED_MPS = 800.0
CANDIDATE_SPEED_MPS = 2000.0
PROBLEM_TIME_MS = 0.5
PARENT_MAXIMUM_CYCLE_COUNT = 829930
CANDIDATE_MAXIMUM_CYCLE_COUNT = 2_500_000
PARENT_BINARY_INTERVAL_MS = 0.00025
CANDIDATE_BINARY_INTERVAL_MS = 0.0001
ASCII_INTERVAL_MS = 0.00025
MAXIMUM_INTERSTATE_DOWNWARD_STEP_MM = 0.25
DEFAULT_AUDIT = ROOT / "analysis" / "72936268_r06q_0800_d3plot_0p00025_confirmation_20261102" / "r06q_d3plot_confirmation_return_audit.json"
DEFAULT_DESIGN = ROOT / "design" / "p0_r06r_2000_speed_upper_anchor_design.json"
DEFAULT_CONTRACT = ROOT / "design" / "p0_r06r_2000_speed_upper_anchor_output_contract.json"


def _require_parent(audit: dict) -> None:
    source = audit["source_evidence"]
    decision = audit["decision"]
    response = audit["response_assessment"]
    if (
        source["job_id"] != PARENT_JOB_ID
        or source["platform_job_id"] != PARENT_PLATFORM_JOB_ID
        or source["deck_sha256"] != PARENT_DECK_SHA256
        or decision["release_r06r_2000_speed_upper_anchor"] is not True
        or decision["release_broad_speed_scan"] is not False
        or decision["engine_fitting"] != "denied"
        or response["classification"] != "confirmed_nonperforation"
        or response["overall_response"] != "pass"
    ):
        raise ValueError("R06q audit does not authorize the R06r upper anchor")


def build_contract(audit: dict) -> dict:
    _require_parent(audit)
    observed_step = audit["measurement_confirmation"]["r06q_maximum_downward_step_mm"]
    projected_step = (
        observed_step
        * CANDIDATE_SPEED_MPS / PARENT_SPEED_MPS
        * CANDIDATE_BINARY_INTERVAL_MS / PARENT_BINARY_INTERVAL_MS
    )
    if projected_step >= MAXIMUM_INTERSTATE_DOWNWARD_STEP_MM:
        raise ValueError("candidate binary cadence does not prospectively clear the frozen step gate")
    return {
        "schema_version": "hf-p0-r06r-speed-upper-anchor-contract@1.0.0",
        "contract_id": "p0_r06r_2000_speed_upper_anchor_output_contract",
        "status": "frozen_after_r06q_admission_before_r06r_outcome",
        "version_boundary": {
            "applies_to_package_revision": REVISION,
            "effective_scope": "one_prospective_numerical_speed_upper_anchor",
            "retroactive_gate_rewrite": False,
        },
        "evidence_chain": {
            "parent_job_id": PARENT_JOB_ID,
            "parent_platform_job_id": PARENT_PLATFORM_JOB_ID,
            "parent_deck_sha256": PARENT_DECK_SHA256,
            "parent_classification": audit["decision"]["classification"],
            "parent_numerical_response": audit["response_assessment"]["classification"],
            "parent_cycle_count": audit["execution"]["cycle_count"],
            "parent_solver_elapsed_seconds": audit["execution"]["solver_elapsed_seconds"],
        },
        "experimental_question": (
            "Does the admitted R06q model produce an unambiguous numerical perforation response at 2000 m/s, "
            "thereby establishing an upper endpoint for a later numerical speed bracket?"
        ),
        "factor_model": {
            "physical_factor": {
                "name": "fragment_initial_speed_mps",
                "parent_level": PARENT_SPEED_MPS,
                "candidate_level": CANDIDATE_SPEED_MPS,
                "single_physical_factor_change": True,
            },
            "measurement_controls": {
                "d3plot_interval_ms": {"parent": PARENT_BINARY_INTERVAL_MS, "candidate": CANDIDATE_BINARY_INTERVAL_MS},
                "d3part_interval_ms": {"parent": PARENT_BINARY_INTERVAL_MS, "candidate": CANDIDATE_BINARY_INTERVAL_MS},
                "reason": "retain <=0.25 mm full-history carrier displacement resolution at the higher speed",
                "physical_model_semantics_unchanged": True,
            },
            "execution_safety_control": {
                "maximum_cycle_count": {"parent": PARENT_MAXIMUM_CYCLE_COUNT, "candidate": CANDIDATE_MAXIMUM_CYCLE_COUNT},
                "reason": "avoid premature truncation if the higher-speed response reduces the stable timestep",
                "solution_effect_if_not_reached": "none",
            },
            "permitted_deck_differences": [
                "TITLE_identifier_line",
                "CONTROL_TERMINATION_endcyc_829930_to_2500000",
                "DATABASE_BINARY_D3PLOT_dt_0p00025_to_0p0001",
                "DATABASE_BINARY_D3PART_dt_0p00025_to_0p0001",
                "INITIAL_VELOCITY_GENERATION_part_2002_vz_minus800_to_minus2000",
            ],
        },
        "prospective_resolution": {
            "parent_observed_maximum_downward_step_mm": observed_step,
            "speed_ratio": CANDIDATE_SPEED_MPS / PARENT_SPEED_MPS,
            "cadence_ratio": CANDIDATE_BINARY_INTERVAL_MS / PARENT_BINARY_INTERVAL_MS,
            "projected_maximum_downward_step_mm": projected_step,
            "required_maximum_downward_step_mm": MAXIMUM_INTERSTATE_DOWNWARD_STEP_MM,
            "projection_status": "pass",
            "claim_boundary": "linear speed/cadence scaling is a planning estimate; observed R06r data must pass",
        },
        "explosion_instant_start_policy": {
            "required": True,
            "implementation": "retain the R06q post-explosion initial geometry and apply the preregistered 2000 m/s fragment velocity",
            "pre_impact_flight_or_detonation_simulation": "not_included",
        },
        "frozen_physical_controls": {
            "mesh_mm": 0.75,
            "termination_time_ms": PROBLEM_TIME_MS,
            "tssfac": 0.05,
            "dt2ms": 0.0,
            "plate_mat015_erod": 1,
            "fragment_mat015_erod": 1,
            "plate_part_id": 1002,
            "fragment_solid_part_id": 2002,
            "adaptive_sph_part_id": 2003,
            "particles_per_failed_hexahedron": 8,
            "solid_contact_soft": 2,
            "dedicated_sph_to_solid_contact_soft": 1,
            "control_solid_psfail": 0,
            "ascii_database_interval_ms": ASCII_INTERVAL_MS,
            "nodes": 1,
            "ranks": 32,
            "solver_family": "LS-DYNA R15 MPP double precision",
        },
        "run_planning": {
            "requested_walltime": "12:00:00",
            "maximum_cycle_count": CANDIDATE_MAXIMUM_CYCLE_COUNT,
            "projected_d3plot_state_count": 5002,
            "projected_d3part_state_count": 5002,
            "projected_d3plot_bytes": 30_127_702_311,
            "projected_d3part_bytes": 758_871_579,
            "projected_total_output_bytes": 32_804_692_460,
            "projected_total_output_gb_decimal": 32.805,
            "projected_total_output_gib": 30.55,
            "storage_claim_boundary": "R06q binary families scaled by state count; actual size can differ",
        },
        "preregistered_gates": {
            "input_deck_parent_hash": PARENT_DECK_SHA256,
            "deck_difference": "exactly_five_registered_lines",
            "resource_identity": {"nodes": 1, "ranks": 32, "required": True},
            "normal_termination_at_problem_time_ms": {"required": PROBLEM_TIME_MS, "tolerance_ms": 1e-5},
            "cycle_guard": {"maximum": CANDIDATE_MAXIMUM_CYCLE_COUNT},
            "no_mass_scaling": True,
            "maximum_full_history_downward_carrier_step_mm": {"maximum": MAXIMUM_INTERSTATE_DOWNWARD_STEP_MM},
            "capture_plane_z_mm": -12.0,
            "combined_solid_sph_crossing_ledger": "required",
            "activation_below_plane_count": {"required": 0},
            "deletion_interval_plane_ambiguity_count": {"required": 0},
            "unresolved_crossing_mass_fraction": {"maximum": 0.005},
            "plate_deletion_count": {"required": 0},
            "target_plate_element_39948_alive_at_terminal_state": True,
            "statewise_active_sph_per_converted_fragment_solid": {"required": 8},
            "combined_fragment_solid_plus_active_sph_mass_relative_error": {"maximum_absolute": 0.01},
            "total_energy_ratio": {"minimum": 0.95, "maximum": 1.05},
            "absolute_final_sliding_energy_fraction_of_initial": {"maximum": 0.05},
            "sph_deletion_count": {"required": 0},
            "d3part_full_d3plot_crosscheck": "required",
        },
        "response_classification_rules": {
            "confirmed_perforation": "unambiguous downward capture-plane crossing with qM>0.005 and all engineering/ledger gates passing",
            "confirmed_nonperforation": "qM<=0.005, qP<=0.005, worst_case_crossing_mass<=0.005, stable terminal separation and all gates passing",
            "unavailable": "any temporal, transfer, activation, deletion-interval, mass-closure or capture-ledger ambiguity",
        },
        "sequential_decision": {
            "if_confirmed_perforation": "establish numerical bracket [800,2000] m/s and release one 1200 m/s midpoint case",
            "if_confirmed_nonperforation": "do not increase speed automatically; review velocity range and model-form adequacy",
            "if_unavailable_or_engineering_fail": "repair or diagnose the failed gate without fitting",
        },
        "admission": {
            "r06r_submission": "allowed_after_static_release_verification",
            "broad_speed_scan": "held",
            "p0_response_calibration": "denied",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "reference_dataset_generation": "denied",
            "physical_validity": "not_established_requires_mesh_replication_and_independent_experimental_evidence",
        },
        "claim_boundary": [
            "R06r is one deterministic numerical upper-anchor experiment, not a fitted ballistic limit.",
            "The binary-cadence and cycle-guard edits are registered measurement/execution controls, not extra physical factors.",
            "A successful R06r can establish only a numerical bracket endpoint and release one midpoint experiment.",
        ],
    }


def build_design(contract: dict) -> dict:
    if contract["version_boundary"]["applies_to_package_revision"] != REVISION:
        raise ValueError("R06r contract revision mismatch")
    return {
        "schema_version": "hf-p0-r06r-speed-upper-anchor-design@1.0.0",
        "package_revision": REVISION,
        "status": "frozen_before_r06r_outcome",
        "objective": "establish a 2000 m/s numerical upper anchor after admission of the 800 m/s nonperforation point",
        "parent": {
            "job_id": PARENT_JOB_ID,
            "platform_job_id": PARENT_PLATFORM_JOB_ID,
            "deck_sha256": PARENT_DECK_SHA256,
            "audit_classification": contract["evidence_chain"]["parent_classification"],
        },
        "jobs": [{
            "job_id": JOB_ID,
            "experiment_id": EXPERIMENT_ID,
            "role": "numerical_speed_upper_anchor",
            "impact_speed_mps": CANDIDATE_SPEED_MPS,
            "termination_time_ms": PROBLEM_TIME_MS,
            "maximum_cycle_count": CANDIDATE_MAXIMUM_CYCLE_COUNT,
            "d3plot_interval_ms": CANDIDATE_BINARY_INTERVAL_MS,
            "d3part_interval_ms": CANDIDATE_BINARY_INTERVAL_MS,
            "ascii_interval_ms": ASCII_INTERVAL_MS,
            "nodes": 1,
            "ranks": 32,
        }],
        "factor_model": contract["factor_model"],
        "frozen_physical_controls": contract["frozen_physical_controls"],
        "run_planning": contract["run_planning"],
        "acceptance_contract": "p0_r06r_2000_speed_upper_anchor_output_contract.json",
        "release_scope": "one R06r high-fidelity job only",
        "prohibited": [
            "changing material, contact, adaptive-SPH, timestep scale, horizon, mesh or resources",
            "relaxing the 0.25 mm or crossing-ledger gates after observing R06r",
            "using R06r directly for engine fitting, damage-tree fitting or a reference label",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the R06r 2000 m/s numerical upper anchor")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    if args.design.exists() or args.contract.exists():
        raise FileExistsError("R06r design or contract already exists")
    audit = strict_json(args.audit)
    contract = build_contract(audit)
    design = build_design(contract)
    args.design.parent.mkdir(parents=True, exist_ok=True)
    args.contract.write_text(json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    args.design.write_text(json.dumps(design, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "frozen", "design": str(args.design.resolve()), "contract": str(args.contract.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
