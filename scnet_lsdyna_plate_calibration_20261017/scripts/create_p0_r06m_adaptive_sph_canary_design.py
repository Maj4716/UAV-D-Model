from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_p0_r06_diagnostics import strict_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_R06J_AUDIT = (
    ROOT
    / "analysis"
    / "72751725_r06j_0800_fragment_erod1_20261030"
    / "r06j_fragment_erod1_return_audit.json"
)
DEFAULT_R06L_AUDIT = (
    ROOT
    / "analysis"
    / "72782972_r06l_0800_contact_soft1_20261101"
    / "r06l_contact_soft1_return_audit.json"
)
DEFAULT_DESIGN_OUTPUT = ROOT / "design" / "p0_r06m_adaptive_sph_canary_design.json"
DEFAULT_CONTRACT_OUTPUT = ROOT / "design" / "p0_r06m_adaptive_sph_output_contract.json"

REVISION = "R06_adaptive_solid_to_sph_canary_20261101"
DESIGN_STATUS = "frozen_after_r06l_soft1_rejected_before_adaptive_sph_capability_outcome"
JOB_ID = "r06m_0800_adaptive_sph_canary"
EXPERIMENT_ID = "r06m_adaptive_solid_to_sph_canary_800"
R06J_DECK_SHA256 = "sha256:8ec1613324159b3ebc00217871ad29321564423b0a7f73a267207e4952bada77"
MAXIMUM_CYCLE_COUNT = 829_930


def _require_evidence(r06j: dict, r06l: dict) -> None:
    if (
        r06j["source_evidence"]["platform_job_id"] != "72751725"
        or r06j["source_evidence"]["deck_sha256"] != R06J_DECK_SHA256
        or r06j["execution"]["normal_termination"] is not True
        or r06j["decision"]["remaining_deletion_mechanism"]
        != "eroding_contact_automatic_negative_volume_failure"
        or r06j["decision"]["release_r06k_fragment_negative_volume_psfail_isolation"] is not True
    ):
        raise ValueError("Job 72751725 is not the expected completed R06j branch point")
    if (
        r06l["source_evidence"]["platform_job_id"] != "72782972"
        or r06l["decision"]["contact_soft1_hypothesis"] != "rejected"
        or r06l["decision"]["classification"]
        != "valid_negative_result_soft1_rejected_model_form_pivot_required"
        or r06l["decision"]["release_r06m_adaptive_solid_to_sph_canary"] is not True
        or r06l["decision"]["release_additional_contact_parameter_tuning"] is not False
    ):
        raise ValueError("Job 72782972 does not authorize the adaptive solid-to-SPH canary")


def build_contract(r06j: dict, r06l: dict) -> dict:
    _require_evidence(r06j, r06l)
    return {
        "schema_version": "hf-p0-r06m-adaptive-sph-output-contract@1.0.0",
        "contract_id": "p0_r06m_adaptive_sph_output_contract",
        "status": DESIGN_STATUS,
        "version_boundary": {
            "applies_to_package_revision": REVISION,
            "effective_scope": "prospective_r06m_adaptive_solid_to_sph_capability_canary",
            "retroactive_gate_rewrite": False,
        },
        "evidence_chain": {
            "completed_branch_point": {
                "job_id": "r06j_0800_fragment_erod1",
                "platform_job_id": "72751725",
                "deck_sha256": R06J_DECK_SHA256,
                "normal_termination": True,
                "cycle_count": r06j["execution"]["cycle_count"],
                "fragment_deleted_element_count": 686
                - r06j["diagnostic_response"]["final_alive_fragment_count"],
                "negative_volume_fragment_element_count": r06j["execution"]["negative_volume_topology"][
                    "fragment_unique_element_count"
                ],
            },
            "rejected_contact_branch": {
                "job_id": "r06l_0800_contact_soft1",
                "platform_job_id": "72782972",
                "termination_kind": r06l["execution"]["termination_kind"],
                "elapsed_hms": r06l["execution"]["elapsed_hms"],
                "progress_fraction": r06l["execution"]["progress_fraction"],
                "cycle_count_at_cancel": r06l["execution"]["latest_cycle"],
                "minimum_time_step_ms": r06l["execution"]["minimum_glstat_time_step_ms"],
                "final_partial_sliding_energy_fraction_of_initial": r06l["returned_outputs"]["glstat"][
                    "final_sliding_energy_fraction_of_initial"
                ],
                "classification": r06l["decision"]["classification"],
            },
        },
        "explosion_instant_start_policy": {
            "required": True,
            "implementation": "reuse_the_same_post_explosion_fragment_state_and_initial_velocity_as_r06j",
            "pre_impact_flight_or_detonation_simulation": "not_included",
        },
        "adaptive_solid_to_sph_canary": {
            "purpose": (
                "Verify parser/runtime/output and mass-energy behavior when failed fragment solids are converted "
                "to SPH particles instead of being silently removed. This is a model-form capability canary."
            ),
            "branch_rule": "branch_from_completed_r06j_not_from_failed_r06l",
            "frozen_controls": {
                "case_id": "p0_v0800_h0p75",
                "impact_speed_mps": 800.0,
                "termination_time_ms": 0.04,
                "tssfac": 0.05,
                "dt2ms": 0.0,
                "plate_part_id": 1002,
                "fragment_solid_part_id": 2002,
                "fragment_mat015_erod": 1,
                "plate_mat015_erod": 0,
                "control_solid_psfail": 0,
                "solid_eroding_contact_soft": 2,
                "nodes": 1,
                "ranks": 32,
            },
            "model_form_bundle": {
                "define_adaptive_solid_to_sph": {
                    "ipid": 2002,
                    "itype": 0,
                    "nq": 2,
                    "particles_per_failed_hexahedron": 8,
                    "ipsph": 2003,
                    "issph": 2,
                    "icpl": 0,
                    "iopt": 0,
                },
                "sph_part": {"part_id": 2003, "section_id": 2, "material_id": 2, "eos_id": 2},
                "section_sph": {
                    "section_id": 2,
                    "cslh": 1.2,
                    "hmin": 0.2,
                    "hmax": 2.0,
                    "sphini": 0,
                    "death": 1.0e20,
                    "start": 0.0,
                    "iavg": 0,
                },
                "solid_contact_scope": {
                    "keyword": "CONTACT_ERODING_SINGLE_SURFACE",
                    "part_set_id": 920001,
                    "part_ids": [1002, 2002],
                    "soft": 2,
                },
                "sph_solid_contact": {
                    "keyword": "CONTACT_AUTOMATIC_NODES_TO_SURFACE",
                    "slave_part_id": 2003,
                    "master_part_set_id": 920001,
                    "sstyp": 3,
                    "mstyp": 2,
                    "soft": 1,
                    "sofscl": 0.1,
                    "interpretation": "dedicated_asymmetric_sph_to_solid_contact_not_the_rejected_global_r06l_soft1_change",
                },
                "runaway_guard": {
                    "control_termination_endcyc": MAXIMUM_CYCLE_COUNT,
                    "basis": "ten_times_the_completed_r06j_cycle_count",
                    "problem_time_must_still_reach_ms": 0.04,
                },
            },
            "measurement_plan": {
                "solid_fragment_history": "retain existing D3PART part set 900001 containing part 2002",
                "solver_created_sph": "inspect full D3PLOT POINT elements and MATSUM part 2003",
                "d3part_sph_claim": "not_assumed_not_preregistered",
                "combined_mass": "solid part 2002 plus SPH part 2003",
                "contact_force": "NCFORC retained but sign, side and unit semantics remain pending",
            },
            "gates": {
                "parser_runtime_start": "required",
                "terminal_problem_time_ms": {"required": 0.04, "tolerance_ms": 1.0e-6},
                "runaway_cycle_guard": {"maximum": MAXIMUM_CYCLE_COUNT, "classification": "engineering_safety_gate"},
                "no_mass_scaling": True,
                "adaptive_sph_part_2003_appears_after_fragment_failure": "required",
                "combined_fragment_solid_plus_sph_mass_relative_change": {
                    "maximum_absolute": 0.01,
                    "classification": "candidate_engineering_diagnostic_not_physical_standard",
                },
                "total_energy_ratio": {
                    "minimum": 0.95,
                    "maximum": 1.05,
                    "classification": "candidate_engineering_gate",
                },
                "absolute_final_sliding_energy_fraction_of_initial": {
                    "maximum": 0.05,
                    "classification": "candidate_engineering_gate",
                },
                "plate_deletion_count": {"required": 0},
                "finite_sph_coordinates_velocity_and_mass": "required",
            },
            "interpretation_rules": {
                "parser_or_runtime_failure": "repair_only_the_adaptive_sph_bundle_and_repeat_the_same_canary",
                "cycle_guard_before_problem_time": "reject_as_numerically_impractical_without_waiting_for_a_multi_day_run",
                "sph_missing_after_fragment_failure": "capability_gate_fail",
                "mass_or_energy_gate_fail": "model_form_candidate_not_admissible",
                "all_gates_pass": (
                    "release_only_a_same_case_repeat_or_tightly_controlled_numerical_confirmation; do not release calibration"
                ),
            },
        },
        "admission": {
            "adaptive_sph_canary_submission": "allowed_after_static_release_verification",
            "additional_contact_parameter_tuning": "denied",
            "r06f_2000_control": "denied",
            "treatments": "denied",
            "p0_response_calibration": "denied",
            "p1_submission": "denied",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "reference_dataset_generation": "denied",
            "physical_validity": "not_established_requires_independent_experimental_evidence",
        },
        "claim_boundary": [
            "R06l rejects global SOFT=1 tuning but does not provide a terminal response.",
            "R06m is a bundled model-form capability canary, not a single-factor physical experiment.",
            "SOFT=1 is used only on the dedicated asymmetric SPH-to-solid contact; the solid contact remains SOFT=2.",
            "A normal solver exit is insufficient unless problem time reaches 0.04 ms and all mass-energy-output gates are reviewed.",
            "No R06m response may be used to fit the simulation engine, damage tree, or reference dataset.",
        ],
        "sources": [
            "https://ansyshelp.ansys.com/public/Views/Secured/corp/v252/en/lsdyna_ug/lsdyna_ug_adaptive_solid_sph.html",
            "https://ansyshelp.ansys.com/public/Views/Secured/Doc_Assets/Release/v242/LS-DYNA_Manual_Volume_I_R14.pdf",
            "https://ansyshelp.ansys.com/public/Views/Secured/Doc_Assets/Release/v252/LS-DYNA_Manual_Volume_II_R14.pdf",
            "https://lsdyna.ansys.com/bird/",
        ],
    }


def build_design(contract: dict, r06j: dict, r06l: dict) -> dict:
    _require_evidence(r06j, r06l)
    return {
        "schema_version": "hf-p0-r06-diagnostic-design@2.0.0",
        "package_revision": REVISION,
        "design_status": DESIGN_STATUS,
        "title": "SCNet P0 R06m 800 m/s adaptive solid-to-SPH capability canary",
        "created_date": "2026-11-01",
        "purpose": contract["adaptive_solid_to_sph_canary"]["purpose"],
        "provenance": {
            "branch_parent_job_id": "r06j_0800_fragment_erod1",
            "branch_parent_platform_job_id": "72751725",
            "branch_parent_deck_sha256": R06J_DECK_SHA256,
            "branch_parent_return_audit": (
                "analysis/72751725_r06j_0800_fragment_erod1_20261030/r06j_fragment_erod1_return_audit.json"
            ),
            "rejected_predecessor_job_id": "r06l_0800_contact_soft1",
            "rejected_predecessor_platform_job_id": "72782972",
            "rejected_predecessor_return_audit": (
                "analysis/72782972_r06l_0800_contact_soft1_20261101/r06l_contact_soft1_return_audit.json"
            ),
            "successor_contract": "design/p0_r06m_adaptive_sph_output_contract.json",
        },
        "explosion_instant_start_policy": contract["explosion_instant_start_policy"],
        "model_form_bundle": contract["adaptive_solid_to_sph_canary"]["model_form_bundle"],
        "measurement_plan": contract["adaptive_solid_to_sph_canary"]["measurement_plan"],
        "paired_experiments": [
            {
                "experiment_id": EXPERIMENT_ID,
                "question": "Can failed fragment solids be converted to readable SPH debris without deletion-driven mass loss or a new numerical runaway?",
                "factor": "bundled_adaptive_solid_to_sph_model_form_capability",
                "case_identity": {
                    "r05_case_id": "p0_v0800_h0p75",
                    "r05_original_index": 2,
                    "impact_speed_mps": 800.0,
                    "mesh_mm": 0.75,
                    "plate_part_id": 1002,
                    "fragment_part_id": 2002,
                    "sph_part_id": 2003,
                },
                "external_control": {
                    "job_id": "r06j_0800_fragment_erod1",
                    "platform_job_id": "72751725",
                    "deck_sha256": R06J_DECK_SHA256,
                    "cycle_count": r06j["execution"]["cycle_count"],
                    "fragment_deleted_element_count": 686
                    - r06j["diagnostic_response"]["final_alive_fragment_count"],
                },
                "jobs": [
                    {
                        "job_id": JOB_ID,
                        "directory": "job_01_v0800_adaptive_sph_canary",
                        "role": "adaptive_solid_to_sph_parser_runtime_mass_energy_capability_canary",
                        "platform_job_name": "uav_hf_r06m_0800_adaptive_sph",
                        "termination_time_ms": 0.04,
                        "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
                        "factor_level": "adaptive_sph_bundle_v1",
                    }
                ],
                "preregistered_evaluation": contract["adaptive_solid_to_sph_canary"],
            }
        ],
        "submission_sequence": {
            "wave_0": [JOB_ID],
            "wave_0_gate": "Audit parser/runtime, full problem time, cycle cap, SPH appearance, combined mass, energy and plate integrity.",
            "hold_after_wave_0": "Do not release 2000 m/s, treatments, fitting or labels.",
            "resource_identity": "Use one node and 32 MPP ranks with LS-DYNA R15 MPP double precision.",
        },
        "admission": contract["admission"],
        "claim_boundary": contract["claim_boundary"],
    }


def _write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen design artifact: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the R06m adaptive solid-to-SPH capability canary")
    parser.add_argument("--r06j-audit", type=Path, default=DEFAULT_R06J_AUDIT)
    parser.add_argument("--r06l-audit", type=Path, default=DEFAULT_R06L_AUDIT)
    parser.add_argument("--design-output", type=Path, default=DEFAULT_DESIGN_OUTPUT)
    parser.add_argument("--contract-output", type=Path, default=DEFAULT_CONTRACT_OUTPUT)
    args = parser.parse_args()
    r06j = strict_json(args.r06j_audit.resolve())
    r06l = strict_json(args.r06l_audit.resolve())
    contract = build_contract(r06j, r06l)
    design = build_design(contract, r06j, r06l)
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
