from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_r06g_tssfac02_return_audit import strict_json


ROOT = Path(__file__).resolve().parents[1]
REVISION = "R06_speed_stability_diagnostic_1200_20261103"
JOB_ID = "r06s_1200_speed_stability_diagnostic"
EXPERIMENT_ID = "r06s_numerical_speed_stability_diagnostic_1200"
PARENT_JOB_ID = "r06r_2000_speed_upper_anchor"
PARENT_PLATFORM_JOB_ID = "72963617"
PARENT_DECK_SHA256 = "sha256:511cbde9d33b1422981bc5716b0c9ae16a3fcd14f091e1dec003d053f3808e48"
PARENT_SPEED_MPS = 2000.0
REFERENCE_SPEED_MPS = 800.0
CANDIDATE_SPEED_MPS = 1200.0
PROBLEM_TIME_MS = 0.5
MAXIMUM_CYCLE_COUNT = 2_500_000
PARENT_BINARY_INTERVAL_MS = 0.0001
CANDIDATE_BINARY_INTERVAL_MS = 0.00015
ASCII_INTERVAL_MS = 0.00025
MAXIMUM_INTERSTATE_DOWNWARD_STEP_MM = 0.25
DEFAULT_AUDIT = ROOT / "analysis" / "72963617_r06r_2000_speed_upper_anchor_stage1_20261103" / "r06r_stage1_return_audit.json"
DEFAULT_DESIGN = ROOT / "design" / "p0_r06s_1200_speed_stability_design.json"
DEFAULT_CONTRACT = ROOT / "design" / "p0_r06s_1200_speed_stability_output_contract.json"


def _require_parent(audit: dict) -> None:
    source = audit["source_evidence"]
    decision = audit["stage1_decision"]
    if (
        source["job_id"] != PARENT_JOB_ID
        or source["platform_job_id"] != PARENT_PLATFORM_JOB_ID
        or source["deck_sha256"] != PARENT_DECK_SHA256
        or decision["release_r06s_1200_speed_stability_diagnostic"] is not True
        or decision["stop_heavy_download"] is not True
        or decision["engine_fitting"] != "denied"
    ):
        raise ValueError("R06r stage-1 audit does not authorize the R06s stability diagnostic")


def build_contract(audit: dict) -> dict:
    _require_parent(audit)
    reference_step = 0.22859651895222832
    projected_step = (
        reference_step
        * CANDIDATE_SPEED_MPS / REFERENCE_SPEED_MPS
        * CANDIDATE_BINARY_INTERVAL_MS / 0.00025
    )
    if projected_step >= MAXIMUM_INTERSTATE_DOWNWARD_STEP_MM:
        raise ValueError("R06s binary cadence does not prospectively clear the frozen displacement gate")
    failure = audit["negative_volume_attribution"]
    engineering = audit["global_engineering"]
    return {
        "schema_version": "hf-p0-r06s-speed-stability-diagnostic-contract@1.0.0",
        "contract_id": "p0_r06s_1200_speed_stability_output_contract",
        "status": "frozen_after_r06r_stage1_failure_before_r06s_outcome",
        "version_boundary": {
            "applies_to_package_revision": REVISION,
            "effective_scope": "one_prospective_mid_speed_numerical_stability_diagnostic",
            "retroactive_gate_rewrite": False,
        },
        "evidence_chain": {
            "parent_job_id": PARENT_JOB_ID,
            "parent_platform_job_id": PARENT_PLATFORM_JOB_ID,
            "parent_deck_sha256": PARENT_DECK_SHA256,
            "parent_classification": audit["stage1_decision"]["classification"],
            "parent_cycle_count": audit["execution"]["cycle_count"],
            "parent_solver_elapsed_seconds": audit["execution"]["solver_elapsed_seconds"],
            "parent_plate_negative_volume_count": failure["plate_unique_count"],
            "parent_target_plate_negative_volume_event": failure["target_plate_negative_volume_event"],
            "parent_absolute_final_sliding_fraction": engineering["absolute_final_sliding_energy_fraction"],
        },
        "experimental_question": (
            "At 1200 m/s, does the exact R06r physical model avoid the 2000 m/s plate negative-volume and sliding-energy failures "
            "while preserving the admitted measurement and resource contracts?"
        ),
        "factor_model": {
            "physical_factor": {
                "name": "fragment_initial_speed_mps",
                "parent_level": PARENT_SPEED_MPS,
                "candidate_level": CANDIDATE_SPEED_MPS,
                "reference_admitted_level": REFERENCE_SPEED_MPS,
                "single_physical_factor_change": True,
                "role": "stability_diagnostic_not_midpoint_calibration",
            },
            "measurement_controls": {
                "d3plot_interval_ms": {"parent": PARENT_BINARY_INTERVAL_MS, "candidate": CANDIDATE_BINARY_INTERVAL_MS},
                "d3part_interval_ms": {"parent": PARENT_BINARY_INTERVAL_MS, "candidate": CANDIDATE_BINARY_INTERVAL_MS},
                "reason": "use the coarsest prospective cadence that remains below the frozen 0.25 mm carrier-step gate",
                "physical_model_semantics_unchanged": True,
            },
            "execution_safety_control": {
                "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
                "changed_from_parent": False,
            },
            "permitted_deck_differences": [
                "TITLE_identifier_line",
                "DATABASE_BINARY_D3PLOT_dt_0p0001_to_0p00015",
                "DATABASE_BINARY_D3PART_dt_0p0001_to_0p00015",
                "INITIAL_VELOCITY_GENERATION_part_2002_vz_minus2000_to_minus1200",
            ],
        },
        "prospective_resolution": {
            "r06q_observed_maximum_downward_step_mm": reference_step,
            "reference_speed_mps": REFERENCE_SPEED_MPS,
            "candidate_speed_mps": CANDIDATE_SPEED_MPS,
            "candidate_binary_interval_ms": CANDIDATE_BINARY_INTERVAL_MS,
            "projected_maximum_downward_step_mm": projected_step,
            "required_maximum_downward_step_mm": MAXIMUM_INTERSTATE_DOWNWARD_STEP_MM,
            "projection_status": "pass",
            "claim_boundary": "linear speed/cadence scaling is planning only; observed R06s output must pass",
        },
        "explosion_instant_start_policy": {
            "required": True,
            "implementation": "retain the R06r post-explosion initial geometry and apply the preregistered 1200 m/s fragment velocity",
            "pre_impact_flight_or_detonation_simulation": "not_included",
        },
        "frozen_controls": {
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
            "ascii_database_interval_ms": ASCII_INTERVAL_MS,
            "nodes": 1,
            "ranks": 32,
            "solver_family": "LS-DYNA R15 MPP double precision",
        },
        "staged_return_policy": {
            "stage1_download": [
                "submitted deck", "manifest.json", "job_contract.json", "Slurm script", "platform log",
                "std.out", "std.err", "glstat", "all mes0000 through mes0031",
            ],
            "stage1_expected_size_mib": 26,
            "if_stage1_fails": "stop_without_downloading_d3plot_d3part_or_binout",
            "if_stage1_passes": "download_complete_d3plot_family_then_complete_d3part_and_binout_families",
        },
        "run_planning": {
            "requested_walltime": "12:00:00",
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "projected_d3plot_state_count": 3335,
            "projected_d3part_state_count": 3335,
            "projected_total_output_bytes": 22_510_000_000,
            "projected_total_output_gb_decimal": 22.51,
            "storage_claim_boundary": "R06r binary families scaled by state count; actual segmentation and size can differ",
        },
        "preregistered_stage1_gates": {
            "input_deck_parent_hash": PARENT_DECK_SHA256,
            "deck_difference": "exactly_four_registered_lines",
            "resource_identity": {"nodes": 1, "ranks": 32, "required": True},
            "normal_termination_at_problem_time_ms": {"required": PROBLEM_TIME_MS, "tolerance_ms": 1e-5},
            "cycle_guard": {"maximum": MAXIMUM_CYCLE_COUNT},
            "no_mass_scaling": True,
            "plate_negative_volume_element_count": {"required": 0},
            "target_plate_element_39948_negative_volume_event": False,
            "total_energy_ratio": {"minimum": 0.95, "maximum": 1.05},
            "absolute_final_sliding_energy_fraction_of_initial": {"maximum": 0.05},
            "maximum_absolute_hourglass_over_internal": {"maximum": 0.05},
        },
        "binary_gates_if_stage1_passes": {
            "maximum_full_history_downward_carrier_step_mm": {"maximum": MAXIMUM_INTERSTATE_DOWNWARD_STEP_MM},
            "combined_solid_sph_crossing_ledger": "required",
            "activation_below_plane_count": {"required": 0},
            "deletion_interval_plane_ambiguity_count": {"required": 0},
            "unresolved_crossing_mass_fraction": {"maximum": 0.005},
            "target_plate_element_39948_alive_at_terminal_state": True,
            "statewise_active_sph_per_converted_fragment_solid": {"required": 8},
            "combined_fragment_solid_plus_active_sph_mass_relative_error": {"maximum_absolute": 0.01},
            "sph_deletion_count": {"required": 0},
            "d3part_full_d3plot_crosscheck": "required",
        },
        "sequential_decision": {
            "if_stage1_fails": "stop_speed_scan_and_redesign_plate_or_contact_model_form",
            "if_stage1_passes_and_binary_response_is_confirmed_perforation": "establish_numerical_bracket_800_to_1200_then_release_one_bracket_refinement",
            "if_stage1_passes_and_binary_response_is_confirmed_nonperforation": "do_not_reuse_failed_2000_anchor_review_model_form_before_any_higher_speed",
            "if_binary_response_unavailable": "repair_measurement_or_transfer_ambiguity_without_fitting",
        },
        "admission": {
            "r06s_submission": "allowed_after_static_release_verification",
            "speed_bracket": "held_until_r06s_stage1_and_binary_gates_pass",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "reference_dataset_generation": "denied",
            "physical_validity": "not_established",
        },
        "claim_boundary": [
            "R06s is a stability diagnostic prompted by a failed 2000 m/s run, not a post-hoc midpoint calibration point.",
            "Passing Stage 1 only authorizes downloading binary response evidence.",
            "No fitting or physical label is admitted before all frozen gates pass.",
        ],
    }


def build_design(contract: dict) -> dict:
    if contract["version_boundary"]["applies_to_package_revision"] != REVISION:
        raise ValueError("R06s contract revision mismatch")
    return {
        "schema_version": "hf-p0-r06s-speed-stability-diagnostic-design@1.0.0",
        "package_revision": REVISION,
        "status": "frozen_before_r06s_outcome",
        "objective": "test whether the R06r engineering failures are confined to the 2000 m/s extreme-load point",
        "parent": {
            "job_id": PARENT_JOB_ID,
            "platform_job_id": PARENT_PLATFORM_JOB_ID,
            "deck_sha256": PARENT_DECK_SHA256,
            "audit_classification": contract["evidence_chain"]["parent_classification"],
        },
        "jobs": [{
            "job_id": JOB_ID,
            "experiment_id": EXPERIMENT_ID,
            "role": "numerical_speed_stability_diagnostic",
            "impact_speed_mps": CANDIDATE_SPEED_MPS,
            "termination_time_ms": PROBLEM_TIME_MS,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "d3plot_interval_ms": CANDIDATE_BINARY_INTERVAL_MS,
            "d3part_interval_ms": CANDIDATE_BINARY_INTERVAL_MS,
            "ascii_interval_ms": ASCII_INTERVAL_MS,
            "nodes": 1,
            "ranks": 32,
        }],
        "factor_model": contract["factor_model"],
        "frozen_controls": contract["frozen_controls"],
        "staged_return_policy": contract["staged_return_policy"],
        "run_planning": contract["run_planning"],
        "acceptance_contract": "p0_r06s_1200_speed_stability_output_contract.json",
        "release_scope": "one R06s high-fidelity stability-diagnostic job only",
        "prohibited": [
            "changing material, contact, adaptive-SPH, timestep scale, horizon, mesh or resources",
            "relaxing plate-negative-volume or 5-percent sliding-energy gates after observing R06s",
            "using R06r or R06s directly for engine fitting, damage-tree fitting or a reference label",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the R06s 1200 m/s stability diagnostic")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    if args.design.exists() or args.contract.exists():
        raise FileExistsError("R06s design or contract already exists")
    audit = strict_json(args.audit)
    contract = build_contract(audit)
    design = build_design(contract)
    args.design.parent.mkdir(parents=True, exist_ok=True)
    args.contract.write_text(json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    args.design.write_text(json.dumps(design, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "frozen", "design": str(args.design.resolve()), "contract": str(args.contract.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
