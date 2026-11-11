from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_r06g_tssfac02_return_audit import strict_json


ROOT = Path(__file__).resolve().parents[1]
REVISION = "R06_plate_negative_volume_psfail_scope_20261103"
JOB_ID = "r06t_1200_fragment_only_psfail_diagnostic"
EXPERIMENT_ID = "r06t_plate_negative_volume_psfail_scope_1200"
PARENT_JOB_ID = "r06s_1200_speed_stability_diagnostic"
PARENT_PLATFORM_JOB_ID = "73059442"
PARENT_DECK_SHA256 = "sha256:eac511d8579d9cad92f08c41b5241129801df46b297140a3f601acacd5b0dd0e"
SPEED_MPS = 1200.0
PROBLEM_TIME_MS = 0.5
MAXIMUM_CYCLE_COUNT = 2_500_000
BINARY_INTERVAL_MS = 0.00015
ASCII_INTERVAL_MS = 0.00025
PARENT_MINIMUM_TIME_STEP_MS = 3.29177e-7
MINIMUM_TIME_STEP_FLOOR_MS = PARENT_MINIMUM_TIME_STEP_MS / 10.0
DEFAULT_AUDIT = ROOT / "analysis" / "73059442_r06s_1200_speed_stability_stage1_20261103" / "r06s_stage1_return_audit.json"
DEFAULT_DESIGN = ROOT / "design" / "p0_r06t_fragment_only_psfail_design.json"
DEFAULT_CONTRACT = ROOT / "design" / "p0_r06t_fragment_only_psfail_output_contract.json"


def _require_parent(audit: dict) -> None:
    source = audit["source_evidence"]
    decision = audit["stage1_decision"]
    engineering = audit["global_engineering"]
    failure = audit["negative_volume_attribution"]
    if (
        source["job_id"] != PARENT_JOB_ID
        or source["platform_job_id"] != PARENT_PLATFORM_JOB_ID
        or source["deck_sha256"] != PARENT_DECK_SHA256
        or decision["release_r06t_fragment_only_psfail_model_form_diagnostic"] is not True
        or decision["stop_heavy_download"] is not True
        or engineering["sliding_energy_gate"] != "pass"
        or failure["plate_unique_count"] != 70
        or decision["engine_fitting"] != "denied"
    ):
        raise ValueError("R06s Stage-1 audit does not authorize the R06t mechanism diagnostic")


def build_contract(audit: dict) -> dict:
    _require_parent(audit)
    failure = audit["negative_volume_attribution"]
    return {
        "schema_version": "hf-p0-r06t-fragment-only-psfail-contract@1.0.0",
        "contract_id": "p0_r06t_fragment_only_psfail_output_contract",
        "status": "frozen_after_r06s_stage1_failure_before_r06t_outcome",
        "version_boundary": {
            "applies_to_package_revision": REVISION,
            "effective_scope": "one_prospective_plate-negative-volume-mechanism diagnostic",
            "retroactive_gate_rewrite": False,
        },
        "evidence_chain": {
            "parent_job_id": PARENT_JOB_ID,
            "parent_platform_job_id": PARENT_PLATFORM_JOB_ID,
            "parent_deck_sha256": PARENT_DECK_SHA256,
            "parent_classification": audit["stage1_decision"]["classification"],
            "parent_plate_negative_volume_count": failure["plate_unique_count"],
            "parent_fragment_negative_volume_count": failure["fragment_unique_count"],
            "parent_target_plate_negative_volume_event": failure["target_plate_negative_volume_event"],
            "parent_minimum_time_step_ms": audit["global_engineering"]["minimum_time_step_ms"],
        },
        "experimental_question": (
            "At the unchanged 1200 m/s R06s state, are the 70 plate negative-volume messages caused by applying "
            "the ERODING-contact automatic negative-volume failure criterion to every solid part?"
        ),
        "factor_model": {
            "single_model_form_factor": {
                "keyword": "CONTROL_SOLID",
                "field": "PSFAIL",
                "parent_level": 0,
                "candidate_level": 900001,
                "candidate_set_semantics": "existing fragment-only part set containing part 2002",
                "plate_part_1002_excluded": True,
                "role": "mechanism_isolation_not_calibration",
            },
            "permitted_deck_differences": ["TITLE_identifier_line", "CONTROL_SOLID_PSFAIL_0_to_fragment_only_set_900001"],
        },
        "explosion_instant_start_policy": {
            "required": True,
            "implementation": "retain the R06s post-explosion geometry and 1200 m/s fragment initial velocity",
            "pre_impact_flight_or_detonation_simulation": "not_included",
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
            "plate_part_id": 1002,
            "fragment_solid_part_id": 2002,
            "adaptive_sph_part_id": 2003,
            "adaptive_solid_to_sph": "2002_to_2003_8_particles_per_failed_hex",
            "solid_contact_soft": 2,
            "dedicated_sph_to_solid_contact_soft": 1,
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
                "std.out", "std.err", "glstat", "all mes0000 through mes0031",
            ],
            "stage1_expected_size_mib": 26,
            "if_stage1_fails": "stop_without_downloading_d3plot_d3part_or_binout",
            "if_stage1_passes": "download_complete_d3plot_family_then_complete_d3part_and_binout_families",
        },
        "run_planning": {
            "requested_walltime": "12:00:00",
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "projected_total_output_gb_decimal": 22.51,
            "storage_claim_boundary": "same cadence and horizon as R06s; actual segmentation and size can differ",
        },
        "preregistered_stage1_gates": {
            "input_deck_parent_hash": PARENT_DECK_SHA256,
            "deck_difference": "exactly_two_registered_lines",
            "resource_identity": {"nodes": 1, "ranks": 32, "required": True},
            "normal_termination_at_problem_time_ms": {"required": PROBLEM_TIME_MS, "tolerance_ms": 1e-5},
            "cycle_guard": {"maximum": MAXIMUM_CYCLE_COUNT},
            "no_mass_scaling": True,
            "plate_negative_volume_element_count": {"required": 0},
            "target_plate_element_39948_negative_volume_event": False,
            "total_energy_ratio": {"minimum": 0.95, "maximum": 1.05},
            "absolute_final_sliding_energy_fraction_of_initial": {"maximum": 0.05},
            "maximum_absolute_hourglass_over_internal": {"maximum": 0.05},
            "minimum_time_step_ms": {"minimum": MINIMUM_TIME_STEP_FLOOR_MS, "basis": "one_tenth_of_R06s_observed_minimum"},
        },
        "sequential_decision": {
            "if_stage1_passes": "download_binary_families_and_test_response_measurement_without_fitting",
            "if_plate_messages_disappear_but_timestep_or_energy_gate_fails": "reject_psfail_suppression_and_design_localized_plate_failure_representation",
            "if_plate_messages_persist": "reject_psfail_scope_hypothesis_and_design_localized_plate_failure_representation",
            "further_contact_parameter_tuning": "not_authorized",
        },
        "admission": {
            "r06t_submission": "allowed_after_static_release_verification",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "reference_dataset_generation": "denied",
            "physical_validity": "not_established",
        },
        "claim_boundary": [
            "R06t is a controlled model-form mechanism diagnostic, not a calibration point.",
            "Passing Stage 1 only authorizes downloading binary response evidence.",
            "PSFAIL scope may suppress messages without establishing a physically valid plate-failure model.",
        ],
    }


def build_design(contract: dict) -> dict:
    if contract["version_boundary"]["applies_to_package_revision"] != REVISION:
        raise ValueError("R06t contract revision mismatch")
    return {
        "schema_version": "hf-p0-r06t-fragment-only-psfail-design@1.0.0",
        "package_revision": REVISION,
        "status": "frozen_before_r06t_outcome",
        "objective": "isolate whether plate negative-volume events arise from the global PSFAIL scope",
        "parent": {
            "job_id": PARENT_JOB_ID,
            "platform_job_id": PARENT_PLATFORM_JOB_ID,
            "deck_sha256": PARENT_DECK_SHA256,
            "audit_classification": contract["evidence_chain"]["parent_classification"],
        },
        "jobs": [{
            "job_id": JOB_ID,
            "experiment_id": EXPERIMENT_ID,
            "role": "plate_negative_volume_psfail_scope_model_form_diagnostic",
            "impact_speed_mps": SPEED_MPS,
            "termination_time_ms": PROBLEM_TIME_MS,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "d3plot_interval_ms": BINARY_INTERVAL_MS,
            "d3part_interval_ms": BINARY_INTERVAL_MS,
            "ascii_interval_ms": ASCII_INTERVAL_MS,
            "nodes": 1,
            "ranks": 32,
        }],
        "factor_model": contract["factor_model"],
        "frozen_controls": contract["frozen_controls"],
        "staged_return_policy": contract["staged_return_policy"],
        "acceptance_contract": "p0_r06t_fragment_only_psfail_output_contract.json",
        "release_scope": "one R06t high-fidelity model-form mechanism diagnostic only",
        "prohibited": [
            "changing speed, material, contact, adaptive-SPH, timestep, output cadence, horizon, mesh or resources",
            "relaxing energy, timestep or plate-negative-volume gates after observing R06t",
            "using R06t directly for engine fitting, damage-tree fitting or a reference label",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the R06t fragment-only PSFAIL diagnostic")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    if args.design.exists() or args.contract.exists():
        raise FileExistsError("R06t design or contract already exists")
    contract = build_contract(strict_json(args.audit))
    design = build_design(contract)
    args.design.parent.mkdir(parents=True, exist_ok=True)
    args.contract.write_text(json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    args.design.write_text(json.dumps(design, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "frozen", "design": str(args.design.resolve()), "contract": str(args.contract.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
