from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_r06g_tssfac02_return_audit import strict_json


ROOT = Path(__file__).resolve().parents[1]
REVISION = "R06_d3plot_0p00025_measurement_confirmation_20261102"
JOB_ID = "r06q_0800_d3plot_0p00025_confirmation"
EXPERIMENT_ID = "r06q_terminal_response_measurement_confirmation_800"
PARENT_JOB_ID = "r06p_0800_terminal_horizon_0p50"
PARENT_PLATFORM_JOB_ID = "72915268"
PARENT_DECK_SHA256 = "sha256:d471f09e9a726c9e4695565abc120e9300bfd68ca2b272bf23852322f8dc8629"
PROBLEM_TIME_MS = 0.5
MAXIMUM_CYCLE_COUNT = 829930
PARENT_D3PLOT_INTERVAL_MS = 0.0005
CANDIDATE_D3PLOT_INTERVAL_MS = 0.00025
D3PART_INTERVAL_MS = 0.00025
ASCII_INTERVAL_MS = 0.00025
MAXIMUM_INTERSTATE_DOWNWARD_STEP_MM = 0.25
DEFAULT_AUDIT = ROOT / "analysis" / "72915268_r06p_0800_terminal_horizon_0p50_20261102" / "r06p_terminal_horizon_return_audit.json"
DEFAULT_DESIGN = ROOT / "design" / "p0_r06q_d3plot_0p00025_confirmation_design.json"
DEFAULT_CONTRACT = ROOT / "design" / "p0_r06q_d3plot_0p00025_confirmation_output_contract.json"


def _require_parent(audit: dict) -> None:
    source = audit["source_evidence"]
    decision = audit["decision"]
    if (
        source["job_id"] != PARENT_JOB_ID
        or source["platform_job_id"] != PARENT_PLATFORM_JOB_ID
        or source["deck_sha256"] != PARENT_DECK_SHA256
        or decision["release_r06q_d3plot_0p00025_measurement_confirmation"] is not True
        or decision["release_speed_scan"] is not False
        or decision["p0_response_calibration"] != "denied"
    ):
        raise ValueError("R06p audit does not authorize the R06q measurement confirmation")


def build_contract(audit: dict) -> dict:
    _require_parent(audit)
    response = audit["response_assessment"]
    observed_step = response["temporal_resolution"]["observed_maximum_downward_step_mm"]
    projected_step = observed_step * CANDIDATE_D3PLOT_INTERVAL_MS / PARENT_D3PLOT_INTERVAL_MS
    if projected_step >= MAXIMUM_INTERSTATE_DOWNWARD_STEP_MM:
        raise ValueError("halved D3PLOT cadence does not prospectively clear the frozen step gate")
    return {
        "schema_version": "hf-p0-r06q-d3plot-measurement-confirmation-contract@1.0.0",
        "contract_id": "p0_r06q_d3plot_0p00025_confirmation_output_contract",
        "status": "frozen_after_r06p_output_resolution_failure_before_r06q_outcome",
        "version_boundary": {
            "applies_to_package_revision": REVISION,
            "effective_scope": "prospective_exact_model_repeat_with_finer_d3plot_measurement",
            "retroactive_gate_rewrite": False,
        },
        "evidence_chain": {
            "parent_job_id": PARENT_JOB_ID,
            "parent_platform_job_id": PARENT_PLATFORM_JOB_ID,
            "parent_deck_sha256": PARENT_DECK_SHA256,
            "parent_classification": audit["decision"]["classification"],
            "parent_cycle_count": audit["execution"]["cycle_count"],
            "parent_solver_elapsed_seconds": audit["execution"]["solver_elapsed_seconds"],
            "parent_primary_qoi_available": response["primary_qoi_available"],
            "parent_primary_qoi_unavailable_reasons": response["unavailable_reasons"],
            "parent_zero_crossing_diagnostic": response["zero_crossing_diagnostic"],
            "parent_terminal_separation": response["terminal_separation"],
        },
        "experimental_question": (
            "Does an exact R06p model repeat with D3PLOT interval 0.00025 ms satisfy the prospectively frozen <=0.25 mm "
            "full-history downward-step gate and therefore admit the 800 m/s zero-crossing/nonperforation response?"
        ),
        "single_change": {
            "layer": "measurement_only",
            "factor": "d3plot_output_interval_ms",
            "parent_level": PARENT_D3PLOT_INTERVAL_MS,
            "candidate_level": CANDIDATE_D3PLOT_INTERVAL_MS,
            "permitted_deck_differences": [
                "TITLE_identifier_line",
                "DATABASE_BINARY_D3PLOT_dt_0p0005_to_0p00025",
            ],
            "physical_model_semantics_unchanged": True,
        },
        "r06p_failure_record": {
            "registered_maximum_downward_step_mm": response["temporal_resolution"]["registered_maximum_downward_step_mm"],
            "observed_maximum_downward_step_mm": observed_step,
            "maximum_step_event": response["temporal_resolution"]["maximum_step_event"],
            "root_cause": response["temporal_resolution"]["root_cause"],
            "correction_policy": "preserve the failed R06p result and repeat prospectively; do not reinterpret its primary QoI post hoc",
        },
        "prospective_resolution": {
            "candidate_d3plot_interval_ms": CANDIDATE_D3PLOT_INTERVAL_MS,
            "projected_maximum_downward_step_mm": projected_step,
            "required_maximum_downward_step_mm": MAXIMUM_INTERSTATE_DOWNWARD_STEP_MM,
            "projection_status": "pass",
            "claim_boundary": "linear cadence scaling is a planning estimate; R06q must pass on observed full-history data",
        },
        "explosion_instant_start_policy": {
            "required": True,
            "implementation": "reuse_exact_r06p_post_explosion_initial_state",
            "pre_impact_flight_or_detonation_simulation": "not_included",
        },
        "frozen_controls": {
            "impact_speed_mps": 800.0,
            "mesh_mm": 0.75,
            "termination_time_ms": PROBLEM_TIME_MS,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
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
            "d3part_interval_ms": D3PART_INTERVAL_MS,
            "ascii_database_interval_ms": ASCII_INTERVAL_MS,
            "nodes": 1,
            "ranks": 32,
            "solver_family": "LS-DYNA R15 MPP double precision",
        },
        "run_planning": {
            "expected_cycle_count_from_exact_model_parent": audit["execution"]["cycle_count"],
            "cycle_guard": MAXIMUM_CYCLE_COUNT,
            "observed_parent_cycle_guard_remaining": audit["execution"]["cycle_guard_remaining"],
            "requested_walltime": "12:00:00",
            "projected_d3plot_state_count": 2002,
            "projected_d3plot_bytes": 12058308682,
            "projected_total_output_bytes": 14280157940,
            "storage_claim_boundary": "R06p family sizes scaled by state count; actual segmentation and size can differ",
        },
        "preregistered_gates": {
            "input_deck_parent_hash": PARENT_DECK_SHA256,
            "deck_difference": "TITLE_plus_D3PLOT_interval_only",
            "resource_identity": {"nodes": 1, "ranks": 32, "required": True},
            "normal_termination_at_problem_time_ms": {"required": PROBLEM_TIME_MS, "tolerance_ms": 1e-5},
            "cycle_guard": {"maximum": MAXIMUM_CYCLE_COUNT},
            "no_mass_scaling": True,
            "maximum_full_history_downward_carrier_step_mm": {"maximum": MAXIMUM_INTERSTATE_DOWNWARD_STEP_MM},
            "capture_plane_z_mm": -12.0,
            "solid_crossing_event_count": {"required": 0},
            "sph_crossing_event_count": {"required": 0},
            "activation_below_plane_count": {"required": 0},
            "deletion_interval_plane_ambiguity_count": {"required": 0},
            "terminal_tail_maximum_downward_carrier_count": {"required": 0},
            "plate_deletion_count": {"required": 0},
            "target_plate_element_39948_alive_at_terminal_state": True,
            "statewise_active_sph_per_converted_fragment_solid": {"required": 8},
            "combined_fragment_solid_plus_active_sph_mass_relative_error": {"maximum_absolute": 0.01},
            "total_energy_ratio": {"minimum": 0.95, "maximum": 1.05},
            "absolute_final_sliding_energy_fraction_of_initial": {"maximum": 0.05},
            "sph_deletion_count": {"required": 0},
            "d3part_full_d3plot_crosscheck": "required",
        },
        "interpretation_rules": {
            "all_gates_pass_and_zero_crossing": "admit_single_run_numerical_confirmed_nonperforation_at_800_mps_then_release_one_speed_scan_case",
            "output_resolution_fails": "hold_primary_response_and_review_finer_measurement_without_changing_the_model",
            "engineering_or_resource_gate_fails": "reject_repeat_and_do_not_release_speed_scan",
            "crossing_occurs": "build_combined_solid_sph_crossing_ledger_before_any_classification",
        },
        "admission": {
            "r06q_submission": "allowed_after_static_release_verification",
            "speed_scan": "held_until_r06q_return_passes",
            "p0_response_calibration": "denied",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "reference_dataset_generation": "denied",
            "physical_validity": "not_established_requires_replication_mesh_velocity_and_independent_experimental_evidence",
        },
        "claim_boundary": [
            "R06q is a measurement-resolution confirmation of one deterministic numerical case.",
            "It does not change the physical model or repair R06p by reinterpretation.",
            "Passing R06q could release only the next numerical speed-control case, not fitting or physical validity.",
        ],
    }


def build_design(contract: dict) -> dict:
    if contract["version_boundary"]["applies_to_package_revision"] != REVISION:
        raise ValueError("R06q contract revision mismatch")
    return {
        "schema_version": "hf-p0-r06q-d3plot-measurement-confirmation-design@1.0.0",
        "package_revision": REVISION,
        "status": "frozen_before_r06q_outcome",
        "objective": "confirm the 800 m/s terminal response with prospectively adequate full-history D3PLOT temporal resolution",
        "parent": {
            "job_id": PARENT_JOB_ID,
            "platform_job_id": PARENT_PLATFORM_JOB_ID,
            "deck_sha256": PARENT_DECK_SHA256,
            "audit_classification": contract["evidence_chain"]["parent_classification"],
        },
        "jobs": [{
            "job_id": JOB_ID,
            "experiment_id": EXPERIMENT_ID,
            "role": "terminal_response_measurement_resolution_confirmation",
            "impact_speed_mps": 800.0,
            "termination_time_ms": PROBLEM_TIME_MS,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "d3plot_interval_ms": CANDIDATE_D3PLOT_INTERVAL_MS,
            "d3part_interval_ms": D3PART_INTERVAL_MS,
            "ascii_interval_ms": ASCII_INTERVAL_MS,
            "nodes": 1,
            "ranks": 32,
        }],
        "single_change": contract["single_change"],
        "frozen_controls": contract["frozen_controls"],
        "run_planning": contract["run_planning"],
        "acceptance_contract": "p0_r06q_d3plot_0p00025_confirmation_output_contract.json",
        "release_scope": "one R06q high-fidelity job only",
        "prohibited": [
            "changing speed, material, contact, adaptive-SPH, timestep, horizon, mesh or resources",
            "relaxing the 0.25 mm gate after observing R06q",
            "using R06p or R06q directly for engine fitting, damage-tree fitting or a reference label",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the R06q D3PLOT 0.00025 ms confirmation")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    if args.design.exists() or args.contract.exists():
        raise FileExistsError("R06q design or contract already exists")
    audit = strict_json(args.audit)
    contract = build_contract(audit)
    design = build_design(contract)
    args.design.parent.mkdir(parents=True, exist_ok=True)
    args.contract.write_text(json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    args.design.write_text(json.dumps(design, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "frozen", "design": str(args.design.resolve()), "contract": str(args.contract.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
