from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_r06g_tssfac02_return_audit import strict_json


ROOT = Path(__file__).resolve().parents[1]
REVISION = "R06_plate_material_erod1_isolation_20261102"
JOB_ID = "r06o_0800_plate_erod1_isolation"
EXPERIMENT_ID = "r06o_plate_material_failure_erosion_isolation_800"
PARENT_JOB_ID = "r06n_0800_adaptive_sph_32rank_confirmation"
PARENT_PLATFORM_JOB_ID = "72864757"
PARENT_DECK_SHA256 = "sha256:edc5f3b80174e52c81eb96a16e0e5abe89f1587b8a850f87f7048bd40f1ac4b7"
MAXIMUM_CYCLE_COUNT = 829930
TARGET_PLATE_ELEMENT_ID = 39948
DEFAULT_AUDIT = (
    ROOT
    / "analysis"
    / "72864757_r06n_0800_adaptive_sph_32rank_confirmation_20261102"
    / "r06n_adaptive_sph_32rank_return_audit.json"
)
DEFAULT_DESIGN = ROOT / "design" / "p0_r06o_plate_erod1_isolation_design.json"
DEFAULT_CONTRACT = ROOT / "design" / "p0_r06o_plate_erod1_output_contract.json"


def _require_parent(audit: dict) -> None:
    source = audit["source_evidence"]
    decision = audit["decision"]
    if (
        source["job_id"] != PARENT_JOB_ID
        or source["platform_job_id"] != PARENT_PLATFORM_JOB_ID
        or source["deck_sha256"] != PARENT_DECK_SHA256
        or decision["release_r06o_plate_mat_erod1_isolation"] is not True
        or decision["resource_identity"] != "pass"
    ):
        raise ValueError("R06n audit does not authorize the plate EROD=1 isolation")


def build_contract(audit: dict) -> dict:
    _require_parent(audit)
    return {
        "schema_version": "hf-p0-r06o-plate-erod1-output-contract@1.0.0",
        "contract_id": "p0_r06o_plate_erod1_output_contract",
        "status": "frozen_after_r06n_plate_material_failure_before_r06o_outcome",
        "version_boundary": {
            "applies_to_package_revision": REVISION,
            "effective_scope": "prospective_plate_material_failure_erosion_isolation",
            "retroactive_gate_rewrite": False,
        },
        "evidence_chain": {
            "parent_job_id": PARENT_JOB_ID,
            "parent_platform_job_id": PARENT_PLATFORM_JOB_ID,
            "parent_deck_sha256": PARENT_DECK_SHA256,
            "parent_classification": audit["decision"]["classification"],
            "parent_plate_deleted_element_ids": audit["engineering_gates"]["plate_deletion"]["element_ids"],
            "parent_plate_failure_diagnosis": audit["plate_failure_diagnosis"],
            "parent_cycle_count": audit["execution"]["cycle_count"],
        },
        "experimental_question": (
            "Does suppressing plate MAT_JOHNSON_COOK failure erosion retain the failed plate element without causing "
            "timestep collapse, mass-energy failure or loss of fragment adaptive-SPH capability?"
        ),
        "single_factor": {
            "factor": "plate_mat_johnson_cook_erod",
            "parent_level": 0,
            "candidate_level": 1,
            "plate_material_id": 1,
            "fragment_material_id": 2,
            "fragment_erod_remains": 1,
            "permitted_deck_differences": ["TITLE_identifier_line", "plate_MAT_JOHNSON_COOK_EROD_field_0_to_1"],
        },
        "explosion_instant_start_policy": {
            "required": True,
            "implementation": "reuse_exact_r06n_post_explosion_initial_state",
            "pre_impact_flight_or_detonation_simulation": "not_included",
        },
        "frozen_controls": {
            "impact_speed_mps": 800.0,
            "mesh_mm": 0.75,
            "termination_time_ms": 0.04,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "tssfac": 0.05,
            "dt2ms": 0.0,
            "plate_part_id": 1002,
            "fragment_solid_part_id": 2002,
            "adaptive_sph_part_id": 2003,
            "particles_per_failed_hexahedron": 8,
            "solid_contact_soft": 2,
            "dedicated_sph_to_solid_contact_soft": 1,
            "control_solid_psfail": 0,
            "nodes": 1,
            "ranks": 32,
            "solver_family": "LS-DYNA R15 MPP double precision",
        },
        "preregistered_gates": {
            "input_deck_parent_hash": PARENT_DECK_SHA256,
            "deck_difference": "TITLE_plus_exactly_one_plate_EROD_field",
            "resource_identity": {"nodes": 1, "ranks": 32, "required": True},
            "normal_termination_at_problem_time_ms": {"required": 0.04, "tolerance_ms": 1e-6},
            "cycle_guard": {"maximum": MAXIMUM_CYCLE_COUNT},
            "efficiency_diagnostic_cycle_ratio_to_r06n": {"maximum": 5.0, "baseline_cycle_count": audit["execution"]["cycle_count"]},
            "minimum_time_step_ratio_to_r06n": {"minimum": 0.1, "classification": "timestep_collapse_guard"},
            "no_mass_scaling": True,
            "plate_deletion_count": {"required": 0},
            "target_plate_element_39948_alive_at_terminal_state": True,
            "target_plate_element_hvar6_available_while_alive": True,
            "statewise_active_sph_per_converted_fragment_solid": {"required": 8},
            "combined_fragment_solid_plus_active_sph_mass_relative_error": {"maximum_absolute": 0.01},
            "total_energy_ratio": {"minimum": 0.95, "maximum": 1.05},
            "absolute_final_sliding_energy_fraction_of_initial": {"maximum": 0.05},
            "sph_deletion_count": {"required": 0},
            "finite_active_sph_coordinates_velocity_and_mass": True,
            "d3part_full_d3plot_crosscheck": "required",
        },
        "interpretation_rules": {
            "plate_retained_and_all_gates_pass": "release_only_a_single_factor_terminal_horizon_extension_at_32_ranks",
            "plate_retained_but_timestep_or_energy_gate_fails": "reject_simple_plate_erosion_suppression_and_hold_for_model_form_review",
            "plate_still_deleted": "EROD_field_does_not_control_the_observed_failure_path_repeat_not_authorized",
            "resource_identity_fail": "repeat_the_same_isolation_without_model_changes",
            "response_at_0p04ms": "diagnostic_only_not_calibration",
        },
        "admission": {
            "r06o_submission": "allowed_after_static_release_verification",
            "terminal_horizon_extension": "held_until_r06o_return_passes",
            "r06f_2000_control": "denied",
            "treatments": "denied",
            "p0_response_calibration": "denied",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "reference_dataset_generation": "denied",
            "physical_validity": "not_established_requires_independent_experimental_evidence",
        },
        "claim_boundary": [
            "R06o is a single-field material-failure-erosion isolation, not a physically validated erosion law.",
            "Retaining a failed element can produce severe distortion or timestep collapse; normal completion and all gates remain required.",
            "The 0.04 ms response horizon remains unresolved and is intentionally unchanged.",
            "Passing R06o would establish only a resource-correct engineering baseline for a later horizon experiment.",
        ],
    }


def build_design(contract: dict) -> dict:
    return {
        "schema_version": "hf-p0-r06-diagnostic-design@2.0.0",
        "package_revision": REVISION,
        "design_status": contract["status"],
        "title": "SCNet P0 R06o 800 m/s plate MAT EROD=1 isolation",
        "created_date": "2026-11-02",
        "purpose": contract["experimental_question"],
        "provenance": {
            "branch_parent_job_id": PARENT_JOB_ID,
            "branch_parent_platform_job_id": PARENT_PLATFORM_JOB_ID,
            "branch_parent_deck_sha256": PARENT_DECK_SHA256,
            "branch_parent_return_audit": "analysis/72864757_r06n_0800_adaptive_sph_32rank_confirmation_20261102/r06n_adaptive_sph_32rank_return_audit.json",
            "successor_contract": "design/p0_r06o_plate_erod1_output_contract.json",
        },
        "single_factor": contract["single_factor"],
        "explosion_instant_start_policy": contract["explosion_instant_start_policy"],
        "frozen_controls": contract["frozen_controls"],
        "target_plate_element_id": TARGET_PLATE_ELEMENT_ID,
        "jobs": [
            {
                "job_id": JOB_ID,
                "experiment_id": EXPERIMENT_ID,
                "directory": "job_01_v0800_plate_erod1_isolation",
                "role": "plate_material_failure_erosion_isolation_canary",
                "platform_job_name": "uav_hf_r06o_0800_plate_erod1",
                "termination_time_ms": 0.04,
                "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
                "required_nodes": 1,
                "required_ranks": 32,
            }
        ],
        "preregistered_gates": contract["preregistered_gates"],
        "interpretation_rules": contract["interpretation_rules"],
        "admission": contract["admission"],
        "claim_boundary": contract["claim_boundary"],
    }


def _write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the R06o plate EROD=1 isolation")
    parser.add_argument("--r06n-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--design-output", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--contract-output", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    audit = strict_json(args.r06n_audit.resolve())
    contract = build_contract(audit)
    design = build_design(contract)
    _write_new(args.contract_output.resolve(), contract)
    _write_new(args.design_output.resolve(), design)
    print(json.dumps({"package_revision": REVISION, "job_id": JOB_ID}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
