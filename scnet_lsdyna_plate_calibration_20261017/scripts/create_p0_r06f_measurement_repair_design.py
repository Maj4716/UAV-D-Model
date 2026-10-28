from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT / "design" / "p0_r06_keyword_parser_repair_design.json"
DEFAULT_R05 = ROOT / "design" / "p0_cases_r05.json"
DEFAULT_DESIGN = ROOT / "design" / "p0_r06f_measurement_repair_design.json"
DEFAULT_CONTRACT = ROOT / "design" / "p0_r06f_real_output_contract.json"


def strict_json(path: Path) -> dict:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=no_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value in {path}: {value}")
        ),
    )


def build_contract(r05: dict) -> dict:
    gates = r05["qualification_gates"]
    return {
        "schema_version": "hf-p0-r06f-real-output-contract@1.0.0",
        "contract_id": "p0_r06f_real_output_contract",
        "status": "frozen_after_r06e_wave0_capability_audit_before_any_r06f_solver_outcome",
        "version_boundary": {
            "applies_to_revision_family": "R06",
            "applies_to_package_revision": "R06_measurement_contract_repair_20261027",
            "effective_scope": "prospective_r06f_only",
            "historical_r06e": {
                "package_revision": "R06_keyword_parser_repair_20261025",
                "job_id": "r06e_0800_ctrl",
                "platform_job_id": "72397333",
                "evidence_policy": "read_only_immutable",
                "retroactive_gate_rewrite": False,
                "frozen_outcome": "parser_runtime_and_run_integrity_pass_but_wave0_fail_hold_under_original_intfor_gate",
            },
        },
        "measurement_sources": {
            "capture_ledger": {
                "primary_artifact": "d3part*",
                "uses": [
                    "fragment position",
                    "fragment velocity",
                    "alive/deleted state",
                    "capture-plane first-crossing ledger",
                ],
                "required": True,
            },
            "johnson_cook_damage": {
                "primary_artifact": "full d3plot*",
                "array": "element_solid_history_variables",
                "history_variable_number_one_based": 6,
                "array_index_zero_based": 5,
                "material": "*MAT_015/*MAT_JOHNSON_COOK",
                "required": True,
                "alive_mask_required": True,
                "post_deletion_zero_policy": "never_interpret_as_damage_recovery",
                "sentinel_policy": "absolute value at or above 1e29 is unavailable with a reason, never zero",
            },
            "d3part_history_variables": {
                "request": "*DATABASE_EXTENT_D3PART with NEIPH=6",
                "status": "experimental_redundant_channel_not_required_for_wave0_pass",
                "acceptance": "identify channels by returned semantics and cross-checks; never infer the damage channel from array position alone",
            },
            "direct_contact": {
                "intfor": {
                    "request_in_r06f": False,
                    "status": "unavailable_by_solver_for_eroding_contact",
                    "solver_evidence": "Warning 20340 (STR+340): binary interface force data is unavailable for eroding contact types - interface #1",
                    "missing_is_zero": False,
                },
                "ncforc": {
                    "request_in_r06f": True,
                    "status": "best_effort_nonblocking_not_proven_structurally_unavailable",
                    "evidence_boundary": "Job 72397333 returned no NCFORC artifact, but Warning 20340 applies specifically to binary INTFOR and does not prove NCFORC is unsupported",
                    "missing_is_zero": False,
                },
                "unexpected_file_policy": "preserve_and_report_if_the_solver_returns_one",
                "sleout_boundary": "interface energy only; never substitute for pressure, gap, penetration, or nodal interface force",
            },
        },
        "d3part_keyword": {
            "binary_keyword": "*DATABASE_BINARY_D3PART",
            "binary_card": [0.00005, 0, 0, 0, 900001],
            "extent_keyword": "*DATABASE_EXTENT_D3PART",
            "extent_cards": [
                [6, 0, 3, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            "card_1_fields": [
                "NEIPH",
                "NEIPS",
                "MAXINT",
                "STRFLG",
                "SIGFLG",
                "EPSFLG",
                "RLTFLG",
                "ENGFLG",
            ],
            "intent": "override D3PART content with six solid history variables while retaining documented defaults for all other fields",
        },
        "capture_ledger_gates": {
            "capture_ledger": deepcopy(gates["capture_ledger"]),
            "numerical_perforation_classification": deepcopy(
                gates["numerical_perforation_classification"]
            ),
        },
        "recanary_comparison": {
            "reference_platform_job_id": "72397333",
            "reference_job_id": "r06e_0800_ctrl",
            "reference_diagnostic_values": {
                "qM": 0.873177835,
                "qP": 0.561307455,
                "qE": 0.360875291,
                "qV_over_v0": 0.642832917,
                "deleted_upstream_mass_fraction": 0.107871719,
                "unique_failure_event_count": 107,
            },
            "tolerances": {
                "qM_absolute": 0.01,
                "qP_absolute": 0.01,
                "qE_absolute": 0.01,
                "qV_relative": 0.05,
            },
            "missing_or_unavailable_policy": "fail_closed",
            "scope": "numerical repeatability diagnostic only; not physical validation or response admission",
        },
        "return_evidence": {
            "required_families": [
                "submitted .k",
                "solver o= output or d3hsp",
                "mes*",
                "glstat",
                "matsum",
                "sleout",
                "nodout",
                "binout*",
                "d3plot*",
                "d3part*",
                "solver stdout/stderr",
                "Slurm script and resource record",
            ],
            "optional_preserve_if_present": ["intfor*", "ncforc*"],
            "missing_required_policy": "fail_closed",
            "missing_optional_policy": "null_with_reason_never_zero",
        },
        "admission": {
            "r06f_wave0_submission": "allowed_after_static_release_verification",
            "r06f_wave0_solver_acceptance": "pending_external_target_solver_run",
            "timestep_diagnostic": "hold_until_r06f_wave0_return_is_audited",
            "r06f_2000_control": "denied_until_recanary_and_timestep_gates_pass",
            "treatments": "denied_until_both_controls_pass",
            "p0_response_calibration": "denied",
            "p1_submission": "denied",
            "engine_fitting": "denied",
            "reference_dataset_generation": "denied",
            "physical_validity": "not_established_requires_independent_experimental_evidence",
        },
        "claim_boundary": [
            "This prospective contract does not rewrite the frozen R06e failure/hold decision.",
            "INTFOR and NCFORC absence is not a zero contact response.",
            "A readable HVAR6 is a solver measurement channel, not evidence that the material law is physically calibrated.",
            "Static PASS and normal termination do not establish numerical repeatability, P0 admission, or dataset-label admission.",
        ],
    }


def build_design(parent: dict, contract: dict) -> dict:
    design = deepcopy(parent)
    design["schema_version"] = "hf-p0-r06-diagnostic-design@1.2.0"
    design["package_revision"] = "R06_measurement_contract_repair_20261027"
    design["design_status"] = (
        "frozen_after_r06e_wave0_solver_capability_audit_before_any_r06f_solver_outcome"
    )
    design["title"] = "SCNet P0 R06f 800 m/s measurement-contract repair recanary"
    design["created_date"] = "2026-10-27"
    design["purpose"] = (
        "Re-run only the unchanged 800 m/s, 0.75 mm isolated control after prospectively "
        "repairing the output contract: remove unsupported direct-contact requests, explicitly "
        "request six D3PART solid history variables, and bind MAT_015 damage to full-D3PLOT HVAR6."
    )
    design["provenance"] = {
        "parent_revision": "R06_keyword_parser_repair_20261025",
        "parent_job_id": "r06e_0800_ctrl",
        "parent_platform_job_id": "72397333",
        "parent_return_directory": "D:/Desktop/UAV新/06",
        "parent_evidence_policy": "read_only_immutable",
        "parent_deck_sha256": "sha256:47c5fcacbec9ba3edac10d06611b6ba602f58a135b27b62d968290334a763a70",
        "parent_result": "parser_runtime_and_run_integrity_pass_but_frozen_intfor_gate_fail_hold",
        "successor_contract": "design/p0_r06f_real_output_contract.json",
        "identity_rule": (
            "Mesh, coordinates, connectivity, node/element IDs, PIDs, material, EOS, contact, "
            "section, timestep, boundary conditions, impact speed, and termination time remain "
            "identical to r06e_0800_ctrl; only database-output cards and prospective contracts change."
        ),
        "repair_scope": "measurement_and_postprocessing_only",
    }

    measurement = design["shared_measurement_upgrade"]
    measurement["status"] = (
        "prospective_measurement_contract_repair_not_a_physics_or_solver_control_factor"
    )
    measurement.pop("intfor_interval_ms", None)
    measurement.pop("intfor", None)
    measurement["d3part_extent_cards"] = deepcopy(contract["d3part_keyword"]["extent_cards"])
    measurement["direct_contact_output_policy"] = {
        "request_binary_intfor": False,
        "request_ascii_ncforc": True,
        "intfor_status": "unavailable_by_solver_for_eroding_contact_warning_20340",
        "ncforc_status": "best_effort_nonblocking",
        "missing_is_zero": False,
        "unexpected_file_policy": "preserve_and_report",
    }
    measurement["damage_source"] = deepcopy(
        contract["measurement_sources"]["johnson_cook_damage"]
    )
    measurement["required_return_families"] = deepcopy(
        contract["return_evidence"]["required_families"]
    )
    measurement["optional_return_families"] = deepcopy(
        contract["return_evidence"]["optional_preserve_if_present"]
    )
    measurement["database_extent_d3part_policy"] = (
        "Explicit NEIPH=6 request is a redundant-channel experiment; full D3PLOT HVAR6 remains authoritative."
    )

    experiment = deepcopy(design["paired_experiments"][0])
    experiment["experiment_id"] = "r06f_recanary_800_measurement_contract"
    experiment["question"] = (
        "Does an unchanged isolated 800 m/s control reproduce Job 72397333 after only the "
        "prospective output contract changes?"
    )
    experiment["factor"] = "measurement_contract_revision_only"
    experiment["control_level"] = "r06e_measurement_contract"
    experiment["treatment_level"] = "r06f_measurement_contract"
    job = deepcopy(experiment["jobs"][0])
    job.update(
        {
            "job_id": "r06f_0800_ctrl",
            "directory": "job_01_v0800_control",
            "role": "measurement_repair_recanary_control",
            "parent_r06e_job_id": "r06e_0800_ctrl",
            "platform_job_name": "uav_hf_r06f_0800_c",
        }
    )
    job.pop("parent_r06d_job_id", None)
    experiment["jobs"] = [job]
    experiment["preregistered_evaluation"] = {
        "primary_comparison": "r06f_0800_ctrl versus platform Job 72397333 using the frozen diagnostic q tolerances",
        "support_condition": (
            "Target R15.0.2 parses and terminates normally; required artifacts are readable; "
            "qM/qP/qE absolute differences are each <=0.01 and qV relative difference is <=0.05."
        ),
        "refute_condition": (
            "Parser/runtime failure, missing required output, unexplained non-finite/sentinel data, "
            "or any frozen numerical comparison tolerance is exceeded."
        ),
        "damage_channel_gate": (
            "Full D3PLOT HVAR6 must be readable with alive masking. D3PART HVAR6 is informative "
            "only and does not block Wave 0 if the authoritative D3PLOT channel is valid."
        ),
        "claim_boundary": "A repeatable recanary does not recover the unavailable primary QoI or establish physical validity.",
    }
    design["paired_experiments"] = [experiment]
    design["submission_sequence"] = {
        "wave_0": ["r06f_0800_ctrl"],
        "wave_0_gate": (
            "Zero KEY errors, normal termination, run-integrity PASS, readable D3PART capture "
            "ledger, readable full-D3PLOT HVAR6 with alive masking, and frozen repeatability tolerances."
        ),
        "hold_after_wave_0": (
            "Do not submit the TSSFAC=0.4 diagnostic, 2000 m/s control, or any treatment until "
            "the returned R06f recanary is audited."
        ),
        "resource_identity": (
            "Use exactly one node, 32 ranks, LS-DYNA R15 MPP double precision, and the same memory settings as Job 72397333."
        ),
    }
    design["deferred_not_generated"] = [
        {
            "candidate": "r06f_0800_tssfac04",
            "reason": "Generate only after the R06f recanary return is audited; it is the next single-factor numerical diagnostic.",
        },
        {
            "candidate": "r06f_2000_ctrl and both treatments",
            "reason": "Hold until both the recanary and the TSSFAC=0.4 comparison pass.",
        },
    ]
    design["admission"] = deepcopy(contract["admission"])
    design["claim_boundary"] = deepcopy(contract["claim_boundary"]) + [
        "Only one R06f 800 m/s control job is generated and exposed in this release.",
        "The targeted canary remains outside the reference response distribution and training labels.",
    ]
    return design


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the prospective R06f measurement repair")
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--r05-design", type=Path, default=DEFAULT_R05)
    parser.add_argument("--design-output", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--contract-output", type=Path, default=DEFAULT_CONTRACT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parent = strict_json(args.parent.resolve())
    r05 = strict_json(args.r05_design.resolve())
    contract = build_contract(r05)
    design = build_design(parent, contract)
    write_new(args.contract_output.resolve(), contract)
    write_new(args.design_output.resolve(), design)
    print(
        json.dumps(
            {
                "package_revision": design["package_revision"],
                "job_id": design["submission_sequence"]["wave_0"][0],
                "design": str(args.design_output.resolve()),
                "contract": str(args.contract_output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
