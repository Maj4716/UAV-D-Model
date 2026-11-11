from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_r06g_tssfac02_return_audit import strict_json


ROOT = Path(__file__).resolve().parents[1]
REVISION = "R06_adaptive_sph_32rank_confirmation_20261101"
JOB_ID = "r06n_0800_adaptive_sph_32rank_confirmation"
EXPERIMENT_ID = "r06n_adaptive_sph_resource_confirmation_800"
PARENT_JOB_ID = "r06m_0800_adaptive_sph_canary"
PARENT_PLATFORM_JOB_ID = "72861050"
PARENT_DECK_SHA256 = "sha256:634436d875c42592b5801de9bf27fce6255691fd27f17d28bd201b80e4cf1107"
MAXIMUM_CYCLE_COUNT = 829930
DEFAULT_AUDIT = (
    ROOT
    / "analysis"
    / "72861050_r06m_0800_adaptive_sph_canary_20261101"
    / "r06m_adaptive_sph_return_audit.json"
)
DEFAULT_DESIGN = ROOT / "design" / "p0_r06n_adaptive_sph_32rank_confirmation_design.json"
DEFAULT_CONTRACT = ROOT / "design" / "p0_r06n_adaptive_sph_32rank_output_contract.json"


def _require_parent(audit: dict) -> None:
    decision = audit["decision"]
    source = audit["source_evidence"]
    if (
        source["job_id"] != PARENT_JOB_ID
        or source["platform_job_id"] != PARENT_PLATFORM_JOB_ID
        or source["deck_sha256"] != PARENT_DECK_SHA256
        or decision["release_r06n_exact_32rank_confirmation"] is not True
        or decision["resource_identity"] != "fail_observed_64_ranks_expected_32"
    ):
        raise ValueError("R06m audit does not authorize the exact 32-rank confirmation")


def build_contract(audit: dict) -> dict:
    _require_parent(audit)
    baseline_cycle_count = audit["execution"]["cycle_count"]
    baseline_converted = audit["engineering_gates"]["conversion"]["final_converted_solid_count"]
    return {
        "schema_version": "hf-p0-r06n-adaptive-sph-32rank-output-contract@1.0.0",
        "contract_id": "p0_r06n_adaptive_sph_32rank_output_contract",
        "status": "frozen_after_r06m_capability_pass_and_resource_identity_fail_before_r06n_outcome",
        "version_boundary": {
            "applies_to_package_revision": REVISION,
            "effective_scope": "prospective_exact_32rank_resource_confirmation",
            "retroactive_gate_rewrite": False,
        },
        "evidence_chain": {
            "parent_job_id": PARENT_JOB_ID,
            "parent_platform_job_id": PARENT_PLATFORM_JOB_ID,
            "parent_deck_sha256": PARENT_DECK_SHA256,
            "parent_classification": audit["decision"]["classification"],
            "parent_normal_completion": audit["execution"]["normal_termination_at_required_time"],
            "parent_observed_resources": audit["execution"]["observed_resources"],
            "parent_cycle_count": baseline_cycle_count,
            "parent_final_converted_solid_count": baseline_converted,
        },
        "experimental_question": (
            "Does the unchanged adaptive-SPH model reproduce its engineering capability when the frozen one-node, "
            "32-rank resource identity is actually honored?"
        ),
        "single_factor": {
            "factor": "actual_mpp_rank_count",
            "parent_observed_level": 64,
            "candidate_required_level": 32,
            "deck_semantics": "unchanged",
            "permitted_deck_difference": "TITLE_identifier_line_only",
        },
        "explosion_instant_start_policy": {
            "required": True,
            "implementation": "reuse_exact_r06m_post_explosion_deck_state",
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
            "nodes": 1,
            "ranks": 32,
            "solver_family": "LS-DYNA R15 MPP double precision",
        },
        "preregistered_gates": {
            "input_deck_parent_hash": PARENT_DECK_SHA256,
            "deck_difference": "TITLE_identifier_line_only",
            "resource_identity": {"nodes": 1, "ranks": 32, "required": True},
            "normal_termination_at_problem_time_ms": {"required": 0.04, "tolerance_ms": 1e-6},
            "cycle_guard": {"maximum": MAXIMUM_CYCLE_COUNT},
            "cycle_count_ratio_to_r06m": {"minimum": 0.8, "maximum": 1.25, "classification": "engineering_repeatability_diagnostic"},
            "no_mass_scaling": True,
            "statewise_active_sph_per_converted_solid": {"required": 8},
            "combined_alive_solid_plus_active_sph_mass_relative_error": {"maximum_absolute": 0.01},
            "conversion_count_difference_from_r06m": {"maximum_absolute": 4, "baseline": baseline_converted},
            "total_energy_ratio": {"minimum": 0.95, "maximum": 1.05},
            "absolute_final_sliding_energy_fraction_of_initial": {"maximum": 0.05},
            "plate_deletion_count": {"required": 0},
            "sph_deletion_count": {"required": 0},
            "finite_active_sph_coordinates_velocity_and_mass": True,
            "d3part_full_d3plot_crosscheck": "required",
        },
        "interpretation_rules": {
            "resource_identity_fail": "repeat_the_same_exact_confirmation_without_model_changes",
            "solver_or_adaptive_gate_fail": "hold_and_diagnose_do_not_extend_horizon",
            "all_gates_pass": "release_only_a_single_factor_terminal_horizon_extension_at_32_ranks",
            "response_at_0p04ms": "diagnostic_only_even_if_no_capture_plane_crossing",
        },
        "admission": {
            "r06n_submission": "allowed_after_static_release_verification",
            "terminal_horizon_extension": "held_until_r06n_return_passes",
            "r06f_2000_control": "denied",
            "treatments": "denied",
            "p0_response_calibration": "denied",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "reference_dataset_generation": "denied",
            "physical_validity": "not_established_requires_independent_experimental_evidence",
        },
        "claim_boundary": [
            "R06n changes actual compute resources from the observed 64 ranks to the preregistered 32 ranks; model semantics are unchanged.",
            "The 0.04 ms response horizon remains unresolved and is intentionally not changed in the same run.",
            "Passing R06n would establish resource-correct engineering repeatability, not physical validity or calibration.",
            "MATSUM part 2002 and 2003 columns must not be naively summed; D3PLOT reference-mass accounting remains authoritative.",
        ],
    }


def build_design(contract: dict) -> dict:
    return {
        "schema_version": "hf-p0-r06-diagnostic-design@2.0.0",
        "package_revision": REVISION,
        "design_status": contract["status"],
        "title": "SCNet P0 R06n 800 m/s adaptive-SPH exact 32-rank confirmation",
        "created_date": "2026-11-01",
        "purpose": contract["experimental_question"],
        "provenance": {
            "branch_parent_job_id": PARENT_JOB_ID,
            "branch_parent_platform_job_id": PARENT_PLATFORM_JOB_ID,
            "branch_parent_deck_sha256": PARENT_DECK_SHA256,
            "branch_parent_return_audit": "analysis/72861050_r06m_0800_adaptive_sph_canary_20261101/r06m_adaptive_sph_return_audit.json",
            "successor_contract": "design/p0_r06n_adaptive_sph_32rank_output_contract.json",
        },
        "single_factor": contract["single_factor"],
        "explosion_instant_start_policy": contract["explosion_instant_start_policy"],
        "frozen_controls": contract["frozen_controls"],
        "jobs": [
            {
                "job_id": JOB_ID,
                "experiment_id": EXPERIMENT_ID,
                "directory": "job_01_v0800_adaptive_sph_32rank_confirmation",
                "role": "resource_identity_and_engineering_repeatability_confirmation",
                "platform_job_name": "uav_hf_r06n_0800_adaptive_sph_32rank",
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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the R06n exact 32-rank adaptive-SPH confirmation")
    parser.add_argument("--r06m-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--design-output", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--contract-output", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    audit = strict_json(args.r06m_audit.resolve())
    contract = build_contract(audit)
    design = build_design(contract)
    _write_new(args.contract_output.resolve(), contract)
    _write_new(args.design_output.resolve(), design)
    print(json.dumps({"package_revision": REVISION, "job_id": JOB_ID}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
