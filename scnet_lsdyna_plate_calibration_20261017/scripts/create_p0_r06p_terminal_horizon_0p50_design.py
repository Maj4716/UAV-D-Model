from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from build_r06g_tssfac02_return_audit import strict_json


ROOT = Path(__file__).resolve().parents[1]
REVISION = "R06_terminal_horizon_0p50_20261102"
JOB_ID = "r06p_0800_terminal_horizon_0p50"
EXPERIMENT_ID = "r06p_terminal_response_horizon_0p50_800"
PARENT_JOB_ID = "r06o_0800_plate_erod1_isolation"
PARENT_PLATFORM_JOB_ID = "72894870"
PARENT_DECK_SHA256 = "sha256:d71d2d11091a813f35ca98656c31fb7ad4f8174b024828a62e0f874daab3524f"
PARENT_HORIZON_MS = 0.04
CANDIDATE_HORIZON_MS = 0.5
MAXIMUM_CYCLE_COUNT = 829930
D3PLOT_INTERVAL_MS = 0.0005
D3PART_INTERVAL_MS = 0.00025
ASCII_INTERVAL_MS = 0.00025
CAPTURE_PLANE_Z_MM = -12.0
OUTPUT_SPATIAL_DISPLACEMENT_LIMIT_MM = 0.25
DEFAULT_AUDIT = ROOT / "analysis" / "72894870_r06o_0800_plate_erod1_isolation_20261102" / "r06o_plate_erod1_return_audit.json"
DEFAULT_DESIGN = ROOT / "design" / "p0_r06p_terminal_horizon_0p50_design.json"
DEFAULT_CONTRACT = ROOT / "design" / "p0_r06p_terminal_horizon_0p50_output_contract.json"


def _require_parent(audit: dict) -> None:
    source = audit["source_evidence"]
    decision = audit["decision"]
    if (
        source["job_id"] != PARENT_JOB_ID
        or source["platform_job_id"] != PARENT_PLATFORM_JOB_ID
        or source["deck_sha256"] != PARENT_DECK_SHA256
        or decision["release_r06p_terminal_horizon_0p50"] is not True
        or decision["numerical_engineering_confirmation"] != "pass_plate_erod1_isolation"
        or decision["research_admission"] != "hold"
    ):
        raise ValueError("R06o audit does not authorize the 0.50 ms terminal-horizon experiment")


def build_contract(audit: dict) -> dict:
    _require_parent(audit)
    execution = audit["execution"]
    tail = audit["response_availability"]["terminal_tail_diagnostic"]
    maximum_downward_speed = abs(tail["downward_active_sph_vz_mm_per_ms"]["minimum"])
    d3plot_max_displacement = maximum_downward_speed * D3PLOT_INTERVAL_MS
    expected_cycles = math.ceil(execution["cycle_count"] * CANDIDATE_HORIZON_MS / PARENT_HORIZON_MS)
    expected_guard_buffer = MAXIMUM_CYCLE_COUNT - expected_cycles
    projected_output = {
        "source_return_total_bytes": audit["source_evidence"]["total_bytes"],
        "source_d3plot_bytes": 2422915072,
        "source_d3part_bytes": 121729024,
        "source_binout_bytes": 671868235,
        "projected_d3plot_bytes": 6033179072,
        "projected_d3part_bytes": 303715433,
        "projected_binout_bytes": 1678412407,
        "projected_total_bytes_conservative": 8605902837,
        "projection_basis": "R06o observed family sizes scaled by requested state-count ratios; nonbinary remainder conservatively multiplied by 2.5",
        "storage_claim_boundary": "planning estimate only; adaptive output segmentation and later state content can change actual size",
    }
    return {
        "schema_version": "hf-p0-r06p-terminal-horizon-output-contract@1.0.0",
        "contract_id": "p0_r06p_terminal_horizon_0p50_output_contract",
        "status": "frozen_after_r06o_engineering_pass_before_r06p_outcome",
        "version_boundary": {
            "applies_to_package_revision": REVISION,
            "effective_scope": "prospective_single_factor_terminal_horizon_extension",
            "retroactive_gate_rewrite": False,
        },
        "evidence_chain": {
            "parent_job_id": PARENT_JOB_ID,
            "parent_platform_job_id": PARENT_PLATFORM_JOB_ID,
            "parent_deck_sha256": PARENT_DECK_SHA256,
            "parent_classification": audit["decision"]["classification"],
            "parent_cycle_count": execution["cycle_count"],
            "parent_minimum_time_step_ms": audit["engineering_gates"]["efficiency"]["minimum_time_step_ms"],
            "parent_plate_deleted_element_count": audit["engineering_gates"]["plate_retention"]["deleted_plate_element_count"],
            "parent_target_plate_element_terminal_alive": audit["engineering_gates"]["plate_retention"]["target_terminal_alive"],
            "parent_primary_qoi_available": audit["response_availability"]["primary_qoi_available"],
        },
        "experimental_question": (
            "Does the frozen resource-correct R06o engineering model reach a sufficiently late terminal state to observe and ledger "
            "solid/adaptive-SPH mass crossing z=-12 mm without violating cycle, plate-retention, mass or energy gates?"
        ),
        "single_factor": {
            "factor": "terminal_response_horizon_ms",
            "parent_level": PARENT_HORIZON_MS,
            "candidate_level": CANDIDATE_HORIZON_MS,
            "permitted_model_difference": "CONTROL_TERMINATION_endtim_0p04_to_0p50",
            "measurement_only_changes": {
                "d3plot_interval_ms": {"parent": 0.0001, "candidate": D3PLOT_INTERVAL_MS},
                "d3part_interval_ms": {"parent": 0.00005, "candidate": D3PART_INTERVAL_MS},
                "ascii_database_interval_ms": {"parent": 0.00005, "candidate": ASCII_INTERVAL_MS},
                "ascii_database_cards": ["GLSTAT", "MATSUM", "SLEOUT", "NODOUT", "NCFORC"],
            },
            "permitted_deck_differences": [
                "TITLE_identifier_line",
                "CONTROL_TERMINATION_endtim_0p04_to_0p50",
                "DATABASE_BINARY_D3PLOT_dt_0p0001_to_0p0005",
                "DATABASE_BINARY_D3PART_dt_0p00005_to_0p00025",
                "DATABASE_GLSTAT_dt_0p00005_to_0p00025",
                "DATABASE_MATSUM_dt_0p00005_to_0p00025",
                "DATABASE_SLEOUT_dt_0p00005_to_0p00025",
                "DATABASE_NODOUT_dt_0p00005_to_0p00025",
                "DATABASE_NCFORC_dt_0p00005_to_0p00025",
            ],
            "model_semantics_unchanged": True,
        },
        "horizon_selection": {
            "selected_ms": CANDIDATE_HORIZON_MS,
            "earliest_linear_projected_sph_crossing_ms": audit["response_availability"]["terminal_tail_diagnostic"]["linear_projected_sph_crossing_time_ms"]["minimum"],
            "median_linear_projected_sph_crossing_ms": audit["response_availability"]["terminal_tail_diagnostic"]["linear_projected_sph_crossing_time_ms"]["p50"],
            "earliest_linear_projected_solid_crossing_ms": audit["response_availability"]["terminal_tail_diagnostic"]["linear_projected_solid_crossing_time_ms"]["minimum"],
            "claim_boundary": "constant terminal velocity is used only for prospective horizon selection; it is not a trajectory or perforation result",
        },
        "explosion_instant_start_policy": {
            "required": True,
            "implementation": "reuse_exact_r06o_post_explosion_initial_state",
            "pre_impact_flight_or_detonation_simulation": "not_included",
        },
        "frozen_controls": {
            "impact_speed_mps": 800.0,
            "mesh_mm": 0.75,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "tssfac": 0.05,
            "dt2ms": 0.0,
            "plate_part_id": 1002,
            "fragment_solid_part_id": 2002,
            "adaptive_sph_part_id": 2003,
            "plate_mat015_erod": 1,
            "fragment_mat015_erod": 1,
            "particles_per_failed_hexahedron": 8,
            "solid_contact_soft": 2,
            "dedicated_sph_to_solid_contact_soft": 1,
            "control_solid_psfail": 0,
            "nodes": 1,
            "ranks": 32,
            "solver_family": "LS-DYNA R15 MPP double precision",
        },
        "run_planning": {
            "linear_cycle_projection": expected_cycles,
            "cycle_guard": MAXIMUM_CYCLE_COUNT,
            "projected_cycle_guard_buffer": expected_guard_buffer,
            "projected_cycle_guard_buffer_fraction": expected_guard_buffer / MAXIMUM_CYCLE_COUNT,
            "cycle_projection_claim_boundary": "diagnostic only; later distortion/contact can change timestep and may trigger the guard",
            "observed_parent_solver_seconds": execution["solver_elapsed_seconds"],
            "linear_solver_seconds_projection": execution["solver_elapsed_seconds"] * CANDIDATE_HORIZON_MS / PARENT_HORIZON_MS,
            "requested_walltime": "12:00:00",
            "projected_output": projected_output,
        },
        "output_resolution_gate": {
            "observed_maximum_downward_speed_mm_per_ms": maximum_downward_speed,
            "d3plot_interval_ms": D3PLOT_INTERVAL_MS,
            "maximum_projected_interstate_displacement_mm": d3plot_max_displacement,
            "maximum_allowed_interstate_displacement_mm": OUTPUT_SPATIAL_DISPLACEMENT_LIMIT_MM,
            "status": "pass" if d3plot_max_displacement <= OUTPUT_SPATIAL_DISPLACEMENT_LIMIT_MM else "fail",
            "interpretation": "prospective temporal resolution check at the maximum downward speed observed at R06o terminal state",
        },
        "preregistered_gates": {
            "input_deck_parent_hash": PARENT_DECK_SHA256,
            "deck_difference": "TITLE_plus_terminal_horizon_plus_measurement_only_output_cadence",
            "resource_identity": {"nodes": 1, "ranks": 32, "required": True},
            "normal_termination_at_problem_time_ms": {"required": CANDIDATE_HORIZON_MS, "tolerance_ms": 1e-5},
            "cycle_guard": {"maximum": MAXIMUM_CYCLE_COUNT},
            "no_mass_scaling": True,
            "plate_deletion_count": {"required": 0},
            "target_plate_element_39948_alive_at_terminal_state": True,
            "statewise_active_sph_per_converted_fragment_solid": {"required": 8},
            "combined_fragment_solid_plus_active_sph_mass_relative_error": {"maximum_absolute": 0.01},
            "total_energy_ratio": {"minimum": 0.95, "maximum": 1.05},
            "absolute_final_sliding_energy_fraction_of_initial": {"maximum": 0.05},
            "sph_deletion_count": {"required": 0},
            "finite_active_sph_coordinates_velocity_and_mass": True,
            "d3part_full_d3plot_crosscheck": "required",
            "output_interstate_displacement_mm": {"maximum": OUTPUT_SPATIAL_DISPLACEMENT_LIMIT_MM},
        },
        "response_rules": {
            "capture_plane_z_mm": CAPTURE_PLANE_Z_MM,
            "crossing_observed": "build a de-duplicated solid-plus-active-SPH crossing ledger with reference mass conservation",
            "no_crossing_but_downward_tail_exhausted": "admit a terminal no-perforation outcome only after stable-state review",
            "no_crossing_and_downward_tail_remains": "hold response and design a later horizon without fitting",
            "cycle_guard_before_0p50ms": "failed horizon run; return outputs without restart or parameter tuning",
        },
        "admission": {
            "r06p_submission": "allowed_after_static_release_verification",
            "p0_response_calibration": "held_until_r06p_complete_response_ledger",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "reference_dataset_generation": "denied",
            "physical_validity": "not_established_requires_replication_mesh_velocity_and_independent_experimental_evidence",
        },
        "claim_boundary": [
            "R06p is a terminal-response acquisition run, not calibration or physical validation.",
            "Output cadence changes are measurement-only and are preregistered; all model semantics remain frozen.",
            "Even a complete capture ledger remains subject to later repeatability, discretization, velocity-control and external physical-validity gates.",
        ],
    }


def build_design(contract: dict) -> dict:
    if contract["version_boundary"]["applies_to_package_revision"] != REVISION:
        raise ValueError("contract revision mismatch")
    return {
        "schema_version": "hf-p0-r06p-terminal-horizon-design@1.0.0",
        "package_revision": REVISION,
        "status": "frozen_before_r06p_outcome",
        "objective": "acquire the first qualified late-time solid/adaptive-SPH capture-plane response without changing model semantics",
        "parent": {
            "job_id": PARENT_JOB_ID,
            "platform_job_id": PARENT_PLATFORM_JOB_ID,
            "deck_sha256": PARENT_DECK_SHA256,
            "audit_classification": contract["evidence_chain"]["parent_classification"],
        },
        "jobs": [{
            "job_id": JOB_ID,
            "experiment_id": EXPERIMENT_ID,
            "role": "terminal_response_horizon_0p50_canary",
            "impact_speed_mps": 800.0,
            "termination_time_ms": CANDIDATE_HORIZON_MS,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "d3plot_interval_ms": D3PLOT_INTERVAL_MS,
            "d3part_interval_ms": D3PART_INTERVAL_MS,
            "ascii_interval_ms": ASCII_INTERVAL_MS,
            "nodes": 1,
            "ranks": 32,
        }],
        "single_factor": contract["single_factor"],
        "frozen_controls": contract["frozen_controls"],
        "run_planning": contract["run_planning"],
        "acceptance_contract": "p0_r06p_terminal_horizon_0p50_output_contract.json",
        "release_scope": "one R06p high-fidelity job only",
        "prohibited": [
            "changing material, contact, adaptive-SPH, timestep, mesh or resource settings",
            "restarting or tuning after a cycle-guard or numerical failure",
            "treating a completed run as physical validation, engine fitting or damage-tree fitting",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the R06p 0.50 ms terminal-horizon design")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    if args.design.exists() or args.contract.exists():
        raise FileExistsError("R06p design or contract already exists")
    audit = strict_json(args.audit)
    contract = build_contract(audit)
    design = build_design(contract)
    args.design.parent.mkdir(parents=True, exist_ok=True)
    args.contract.write_text(json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    args.design.write_text(json.dumps(design, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "frozen", "design": str(args.design.resolve()), "contract": str(args.contract.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
