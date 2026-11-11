from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_r06g_tssfac02_return_audit import strict_json


ROOT = Path(__file__).resolve().parents[1]
REVISION = "R06_plate_adaptive_sph_canary_20261104"
JOB_ID = "r06u_1200_plate_adaptive_sph_canary"
EXPERIMENT_ID = "r06u_plate_adaptive_sph_capability_1200"
BRANCH_PARENT_JOB_ID = "r06s_1200_speed_stability_diagnostic"
BRANCH_PARENT_PLATFORM_JOB_ID = "73059442"
BRANCH_PARENT_DECK_SHA256 = "sha256:eac511d8579d9cad92f08c41b5241129801df46b297140a3f601acacd5b0dd0e"
REJECTED_JOB_ID = "r06t_1200_fragment_only_psfail_diagnostic"
REJECTED_PLATFORM_JOB_ID = "73087868"
SPEED_MPS = 1200.0
PROBLEM_TIME_MS = 0.15
MAXIMUM_CYCLE_COUNT = 2_500_000
BINARY_INTERVAL_MS = 0.00015
ASCII_INTERVAL_MS = 0.00025
MINIMUM_TIME_STEP_FLOOR_MS = 3.0e-8
DEFAULT_R06S_AUDIT = ROOT / "analysis" / "73059442_r06s_1200_speed_stability_stage1_20261103" / "r06s_stage1_return_audit.json"
DEFAULT_R06T_AUDIT = ROOT / "analysis" / "73087868_r06t_fragment_only_psfail_failure_20261104" / "r06t_failure_audit.json"
DEFAULT_DESIGN = ROOT / "design" / "p0_r06u_plate_adaptive_sph_canary_design.json"
DEFAULT_CONTRACT = ROOT / "design" / "p0_r06u_plate_adaptive_sph_canary_output_contract.json"


def _require_evidence(r06s: dict, r06t: dict) -> None:
    if (
        r06s["source_evidence"]["job_id"] != BRANCH_PARENT_JOB_ID
        or r06s["source_evidence"]["platform_job_id"] != BRANCH_PARENT_PLATFORM_JOB_ID
        or r06s["source_evidence"]["deck_sha256"] != BRANCH_PARENT_DECK_SHA256
        or r06s["execution"]["normal_termination_at_required_time"] is not True
        or r06s["negative_volume_attribution"]["plate_unique_count"] != 70
        or r06s["global_engineering"]["sliding_energy_gate"] != "pass"
    ):
        raise ValueError("R06s is not the expected completed branch parent")
    if (
        r06t["source_evidence"]["job_id"] != REJECTED_JOB_ID
        or r06t["source_evidence"]["platform_job_id"] != REJECTED_PLATFORM_JOB_ID
        or r06t["terminal_failure"]["element_id"] != 23830
        or r06t["terminal_failure"]["part_id"] != 1002
        or r06t["decision"]["fragment_only_psfail_hypothesis"] != "rejected_as_numerically_unstable"
        or r06t["decision"]["release_r06u_plate_adaptive_sph_capability_canary"] is not True
    ):
        raise ValueError("R06t does not authorize the R06u plate adaptive-SPH canary")


def build_contract(r06s: dict, r06t: dict) -> dict:
    _require_evidence(r06s, r06t)
    return {
        "schema_version": "hf-p0-r06u-plate-adaptive-sph-contract@1.0.0",
        "contract_id": "p0_r06u_plate_adaptive_sph_canary_output_contract",
        "status": "frozen_after_r06t_failure_before_r06u_outcome",
        "version_boundary": {
            "applies_to_package_revision": REVISION,
            "effective_scope": "one_prospective_plate-adaptive-SPH capability canary",
            "retroactive_gate_rewrite": False,
        },
        "evidence_chain": {
            "completed_branch_parent": {
                "job_id": BRANCH_PARENT_JOB_ID,
                "platform_job_id": BRANCH_PARENT_PLATFORM_JOB_ID,
                "deck_sha256": BRANCH_PARENT_DECK_SHA256,
                "plate_negative_volume_count": 70,
                "minimum_time_step_ms": r06s["global_engineering"]["minimum_time_step_ms"],
            },
            "rejected_psfail_scope_run": {
                "job_id": REJECTED_JOB_ID,
                "platform_job_id": REJECTED_PLATFORM_JOB_ID,
                "classification": r06t["decision"]["classification"],
                "plate_negative_volume_count_before_failure": r06t["negative_volume_attribution_before_failure"]["plate_unique_count"],
                "complex_sound_speed_element": r06t["terminal_failure"]["element_id"],
                "problem_time_ms": r06t["terminal_failure"]["problem_time_ms"],
                "terminal_time_step_ms": r06t["terminal_failure"]["time_step_ms"],
            },
        },
        "experimental_question": (
            "Can failed plate solids be replaced by same-material SPH particles and pass beyond the R06t 0.10872 ms "
            "collapse point without time-step, energy or mass failure?"
        ),
        "explosion_instant_start_policy": {
            "required": True,
            "implementation": "reuse the R06s post-explosion geometry and 1200 m/s fragment velocity",
            "pre_impact_flight_or_detonation_simulation": "not_included",
        },
        "model_form_bundle": {
            "branch_rule": "branch_from_completed_R06s_not_failed_R06t",
            "control_solid_psfail": 0,
            "plate_adaptive_solid_to_sph": {
                "solid_part_id": 1002,
                "sph_part_id": 1003,
                "sph_section_id": 2,
                "material_id": 1,
                "eos_id": 1,
                "nq": 2,
                "particles_per_failed_hexahedron": 8,
                "icpl": 0,
                "iopt": 0,
                "interpretation": "local conversion occurs only when a plate solid fails",
            },
            "plate_sph_to_solid_contact": {
                "keyword": "CONTACT_AUTOMATIC_NODES_TO_SURFACE",
                "slave_part_id": 1003,
                "master_part_set_id": 920001,
                "sstyp": 3,
                "mstyp": 2,
                "soft": 1,
                "sofscl": 0.1,
            },
            "retained_fragment_adaptive_sph": {"solid_part_id": 2002, "sph_part_id": 2003, "unchanged": True},
            "sph_sph_interpart_contact": "not_added_in_capability_canary_terminal_physical_response_remains_denied",
            "registered_deck_changes": [
                "TITLE_and_identity_comments",
                "CONTROL_TERMINATION_endtim_0p50_to_0p15",
                "add_plate_SPH_part_1003_same_material_and_EOS_as_plate",
                "add_plate_SPH_to_solid_contact",
                "add_plate_adaptive_solid_to_SPH_definition",
            ],
        },
        "frozen_controls": {
            "impact_speed_mps": SPEED_MPS,
            "mesh_mm": 0.75,
            "termination_time_ms": PROBLEM_TIME_MS,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "tssfac": 0.05,
            "dt2ms": 0.0,
            "plate_mat015_erod": 1,
            "fragment_mat015_erod": 1,
            "solid_eroding_contact_soft": 2,
            "fragment_sph_to_solid_contact_soft": 1,
            "d3plot_interval_ms": BINARY_INTERVAL_MS,
            "d3part_interval_ms": BINARY_INTERVAL_MS,
            "ascii_database_interval_ms": ASCII_INTERVAL_MS,
            "nodes": 1,
            "ranks": 32,
            "solver_family": "LS-DYNA R15 MPP double precision",
        },
        "staged_return_policy": {
            "stage1_download": [
                "submitted deck", "manifest.json", "job_contract.json", "Slurm script", "platform log",
                "std.out", "std.err", "glstat", "matsum", "sleout", "nodout", "ncforc",
                "sldsph_frag if present", "all mes0000 through mes0031",
            ],
            "stage1_expected_size_mib": 35,
            "if_stage1_fails": "stop_without_downloading_d3plot_d3part_or_binout",
            "if_stage1_passes": "download complete d3plot then d3part and binout families for conversion and mass ledgers",
        },
        "run_planning": {
            "requested_walltime": "12:00:00",
            "termination_time_ms": PROBLEM_TIME_MS,
            "r06t_failure_time_ms": r06t["terminal_failure"]["problem_time_ms"],
            "required_margin_beyond_failure_time_ms": PROBLEM_TIME_MS - r06t["terminal_failure"]["problem_time_ms"],
            "projected_total_output_gb_decimal": 6.753,
            "claim_boundary": "linear horizon scaling from R06s output volume; actual R06u size can differ",
        },
        "preregistered_stage1_gates": {
            "input_deck_parent_hash": BRANCH_PARENT_DECK_SHA256,
            "resource_identity": {"nodes": 1, "ranks": 32, "required": True},
            "normal_termination_at_problem_time_ms": {"required": PROBLEM_TIME_MS, "tolerance_ms": 1e-5},
            "cycle_guard": {"maximum": MAXIMUM_CYCLE_COUNT},
            "no_mass_scaling": True,
            "no_complex_sound_speed_or_other_solver_error": True,
            "minimum_time_step_ms": {"minimum": MINIMUM_TIME_STEP_FLOOR_MS},
            "total_energy_ratio": {"minimum": 0.95, "maximum": 1.05},
            "absolute_final_sliding_energy_fraction_of_initial": {"maximum": 0.05},
            "maximum_absolute_hourglass_over_internal": {"maximum": 0.05},
            "plate_sph_part_1003_positive_mass_after_plate_failure": True,
            "combined_plate_solid_plus_sph_mass_relative_change": {"maximum_absolute": 0.01},
            "combined_fragment_solid_plus_sph_mass_relative_change": {"maximum_absolute": 0.01},
        },
        "sequential_decision": {
            "if_stage1_passes": "download binary families and verify plate conversion, mass and contact ledgers before a full 0.50 ms confirmation",
            "if_parser_or_runtime_fails_before_conversion": "repair only the plate adaptive-SPH bundle and repeat the same 0.15 ms canary",
            "if_time_step_energy_or_mass_gate_fails": "reject this plate adaptive-SPH formulation and stop model-form expansion pending expert review",
            "additional_contact_parameter_tuning": "not_authorized",
        },
        "admission": {
            "r06u_submission": "allowed_after_static_release verification",
            "full_0p50ms_confirmation": "held_until_r06u_stage1_and_binary_gates_pass",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "reference_dataset_generation": "denied",
            "physical_validity": "not_established",
        },
        "sources": [
            "https://ansyshelp.ansys.com/public/Views/Secured/corp/v252/en/lsdyna_ug/lsdyna_ug_adaptive_solid_sph.html",
            "https://ansyshelp.ansys.com/public/Views/Secured/Doc_Assets/Release/v242/LS-DYNA_Manual_Volume_I_R14.pdf",
        ],
        "claim_boundary": [
            "R06u is a short-horizon model-form capability canary, not a calibration point.",
            "Adaptive conversion can preserve failed material numerically but does not establish a physically valid fracture model.",
            "Passing R06u does not authorize a terminal response until the full 0.50 ms confirmation and binary ledgers pass.",
        ],
    }


def build_design(contract: dict) -> dict:
    return {
        "schema_version": "hf-p0-r06u-plate-adaptive-sph-design@1.0.0",
        "package_revision": REVISION,
        "status": "frozen_before_r06u_outcome",
        "objective": "test a local same-material SPH replacement for failed plate solids beyond the R06t collapse time",
        "branch_parent": {
            "job_id": BRANCH_PARENT_JOB_ID,
            "platform_job_id": BRANCH_PARENT_PLATFORM_JOB_ID,
            "deck_sha256": BRANCH_PARENT_DECK_SHA256,
        },
        "rejected_predecessor": {"job_id": REJECTED_JOB_ID, "platform_job_id": REJECTED_PLATFORM_JOB_ID},
        "jobs": [{
            "job_id": JOB_ID,
            "experiment_id": EXPERIMENT_ID,
            "role": "plate_adaptive_sph_parser_runtime_mass_energy_capability_canary",
            "impact_speed_mps": SPEED_MPS,
            "termination_time_ms": PROBLEM_TIME_MS,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "d3plot_interval_ms": BINARY_INTERVAL_MS,
            "d3part_interval_ms": BINARY_INTERVAL_MS,
            "ascii_interval_ms": ASCII_INTERVAL_MS,
            "nodes": 1,
            "ranks": 32,
        }],
        "model_form_bundle": contract["model_form_bundle"],
        "frozen_controls": contract["frozen_controls"],
        "staged_return_policy": contract["staged_return_policy"],
        "acceptance_contract": "p0_r06u_plate_adaptive_sph_canary_output_contract.json",
        "release_scope": "one R06u 0.15 ms high-fidelity capability canary only",
        "prohibited": [
            "submitting the failed R06t PSFAIL scope again",
            "changing speed, material constants, mesh, timestep scale, output cadence or resources",
            "running the full 0.50 ms model before this canary passes",
            "using R06u directly for engine fitting, damage-tree fitting or a reference label",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the R06u plate adaptive-SPH canary")
    parser.add_argument("--r06s-audit", type=Path, default=DEFAULT_R06S_AUDIT)
    parser.add_argument("--r06t-audit", type=Path, default=DEFAULT_R06T_AUDIT)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    if args.design.exists() or args.contract.exists():
        raise FileExistsError("R06u design or contract already exists")
    r06s = strict_json(args.r06s_audit)
    r06t = strict_json(args.r06t_audit)
    contract = build_contract(r06s, r06t)
    design = build_design(contract)
    args.design.parent.mkdir(parents=True, exist_ok=True)
    args.contract.write_text(json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    args.design.write_text(json.dumps(design, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "frozen", "job_id": JOB_ID, "design": str(args.design.resolve()), "contract": str(args.contract.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
