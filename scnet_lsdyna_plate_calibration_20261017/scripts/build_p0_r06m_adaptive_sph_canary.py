from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from create_p0_r06m_adaptive_sph_canary_design import (
    DESIGN_STATUS,
    EXPERIMENT_ID,
    JOB_ID,
    MAXIMUM_CYCLE_COUNT,
    REVISION,
    R06J_DECK_SHA256,
)
from generate_p0_r06_diagnostics import strict_json
from validate_p0_r06_diagnostics import comparable_blocks, grouped_blocks, sha256, split_fields


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_DECK = (
    ROOT
    / "analysis"
    / "72751725_r06j_0800_fragment_erod1_20261030"
    / "immutable_parent_r06j_0800_fragment_erod1.k"
)
DEFAULT_DESIGN = ROOT / "design" / "p0_r06m_adaptive_sph_canary_design.json"
DEFAULT_CONTRACT = ROOT / "design" / "p0_r06m_adaptive_sph_output_contract.json"
DEFAULT_MATERIAL = ROOT / "materials" / "jc4340_reference.json"
DEFAULT_OUTPUT = ROOT / "manual_upload_20261101_r06m_adaptive_sph_canary"
JOB_DIR = "job_01_v0800_adaptive_sph_canary"
HEAD_DECK = "r06m_0800_adaptive_sph_canary.k"


BASELINE_TITLE = "SCNet P0 R06_fragment_erosion_isolation_20261030 r06j_0800_fragment_erod1"
NEW_TITLE = "SCNet P0 R06_adaptive_solid_to_sph_canary_20261101 r06m_0800_adaptive_sph_canary"

BASELINE_TERMINATION = """*CONTROL_TERMINATION
4.000000000e-02,0,0.000000000e+00,0.000000000e+00,0.000000000e+00"""
NEW_TERMINATION = f"""*CONTROL_TERMINATION
4.000000000e-02,{MAXIMUM_CYCLE_COUNT},0.000000000e+00,0.000000000e+00,0.000000000e+00"""

BASELINE_CONTACT = """*CONTACT_ERODING_SINGLE_SURFACE
0,0,0,0,0,0,1,1
0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,2.000000000e+01,0,0.000000000e+00,0.000000000e+00
0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00
1,1,1
2,0.000000000e+00,0,0,3,5,0,0"""
NEW_CONTACT = """*CONTACT_ERODING_SINGLE_SURFACE
920001,0,2,0,0,0,1,1
0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,2.000000000e+01,0,0.000000000e+00,0.000000000e+00
0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00
1,1,1
2,0.000000000e+00,0,0,3,5,0,0
*CONTACT_AUTOMATIC_NODES_TO_SURFACE
2003,920001,3,2,0,0,1,1
0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0,0.000000000e+00,0.000000000e+00
0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00
1,1.000000000e-01,0,0,0,0,0,0"""

BASELINE_SECTION = """*SECTION_SOLID_TITLE
Fully integrated S/R solid for numerical qualification
1,-1"""
NEW_SECTION = """*SECTION_SOLID_TITLE
Fully integrated S/R solid for numerical qualification
1,-1
*SECTION_SPH
2,1.200000000e+00,2.000000000e-01,2.000000000e+00,0,1.000000000e+20,0.000000000e+00,0"""

BASELINE_FRAGMENT_PART = """*PART
fragment__p0_v0800_h0p75
2002,1,2,2,0,0,0,0"""
NEW_FRAGMENT_PART = """*PART
fragment__p0_v0800_h0p75
2002,1,2,2,0,0,0,0
*PART
adaptive_sph_debris__p0_v0800_h0p75
2003,2,2,2,0,0,0,0"""

ADAPTIVE_TAIL = """*DEFINE_ADAPTIVE_SOLID_TO_SPH
2002,0,2,2003,2,0,0,
*SET_PART_LIST_TITLE
R06m solids for eroding contact and SPH master surface
920001,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,MECH,,
1002,2002
*END"""


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise AssertionError(f"{label}: expected one exact replacement target, found {count}")
    return text.replace(old, new, 1)


def transform_deck(baseline_text: str) -> str:
    candidate = baseline_text
    candidate = _replace_once(candidate, BASELINE_TITLE, NEW_TITLE, "TITLE")
    candidate = _replace_once(candidate, BASELINE_TERMINATION, NEW_TERMINATION, "CONTROL_TERMINATION")
    candidate = _replace_once(candidate, BASELINE_CONTACT, NEW_CONTACT, "contact bundle")
    candidate = _replace_once(candidate, BASELINE_SECTION, NEW_SECTION, "SECTION_SPH")
    candidate = _replace_once(candidate, BASELINE_FRAGMENT_PART, NEW_FRAGMENT_PART, "SPH PART")
    candidate = _replace_once(candidate, "*END", ADAPTIVE_TAIL, "adaptive definition tail")
    candidate = _replace_once(
        candidate,
        "$ Pair=r06j_fragment_erosion_isolation_800 factor=fragment_mat015_erod_flag role=fragment_failure_erosion_suppression_diagnostic",
        "$ Pair=r06m_adaptive_solid_to_sph_canary_800 factor=adaptive_sph_bundle role=parser_runtime_mass_energy_capability_canary",
        "diagnostic title metadata",
    )
    return candidate


def _fields(line: str) -> list[str]:
    return split_fields(line)


def validate_transformed_deck(baseline_text: str, candidate_text: str) -> dict:
    candidate_text.encode("ascii")
    baseline = grouped_blocks(baseline_text)
    candidate = grouped_blocks(candidate_text)
    ignored = {
        "*TITLE",
        "*CONTROL_TERMINATION",
        "*CONTACT_ERODING_SINGLE_SURFACE",
        "*CONTACT_AUTOMATIC_NODES_TO_SURFACE",
        "*SECTION_SPH",
        "*PART",
        "*DEFINE_ADAPTIVE_SOLID_TO_SPH",
        "*SET_PART_LIST_TITLE",
    }
    if comparable_blocks(candidate_text, ignored) != comparable_blocks(baseline_text, ignored):
        raise AssertionError("candidate changed a keyword outside the preregistered model-form bundle")
    if candidate["*TITLE"] != [[NEW_TITLE]]:
        raise AssertionError("candidate TITLE differs")

    termination = _fields(candidate["*CONTROL_TERMINATION"][0][0])
    baseline_termination = _fields(baseline["*CONTROL_TERMINATION"][0][0])
    if float(termination[0]) != 0.04 or int(termination[1]) != MAXIMUM_CYCLE_COUNT:
        raise AssertionError("runaway cycle guard differs")
    if termination[2:] != baseline_termination[2:]:
        raise AssertionError("CONTROL_TERMINATION changed beyond ENDCYC")

    if candidate["*PART"][:2] != baseline["*PART"]:
        raise AssertionError("existing plate/fragment PART cards changed")
    if len(candidate["*PART"]) != 3 or candidate["*PART"][2] != [
        "adaptive_sph_debris__p0_v0800_h0p75",
        "2003,2,2,2,0,0,0,0",
    ]:
        raise AssertionError("SPH PART 2003 differs")

    if candidate["*SET_PART_LIST_TITLE"][:2] != baseline["*SET_PART_LIST_TITLE"]:
        raise AssertionError("existing D3PART/PSFAIL sets changed")
    if len(candidate["*SET_PART_LIST_TITLE"]) != 3:
        raise AssertionError("solid contact part set count differs")
    new_set = candidate["*SET_PART_LIST_TITLE"][2]
    if _fields(new_set[1])[0] != "920001" or [int(value) for value in _fields(new_set[2])] != [1002, 2002]:
        raise AssertionError("solid contact part set 920001 differs")

    solid_contact = candidate["*CONTACT_ERODING_SINGLE_SURFACE"][0]
    baseline_solid_contact = baseline["*CONTACT_ERODING_SINGLE_SURFACE"][0]
    if solid_contact[1:] != baseline_solid_contact[1:]:
        raise AssertionError("solid contact changed beyond Card 1 scope")
    solid_card_1 = _fields(solid_contact[0])
    if [int(solid_card_1[index]) for index in (0, 1, 2, 3)] != [920001, 0, 2, 0]:
        raise AssertionError("solid eroding contact scope differs")
    if int(_fields(solid_contact[4])[0]) != 2:
        raise AssertionError("solid eroding contact must retain SOFT=2")

    sph_contact = candidate["*CONTACT_AUTOMATIC_NODES_TO_SURFACE"][0]
    if len(sph_contact) != 4:
        raise AssertionError("SPH node-to-surface contact card count differs")
    if [int(value) for value in _fields(sph_contact[0])[:4]] != [2003, 920001, 3, 2]:
        raise AssertionError("SPH contact slave/master identity differs")
    sph_optional_a = _fields(sph_contact[3])
    if int(sph_optional_a[0]) != 1 or float(sph_optional_a[1]) != 0.1:
        raise AssertionError("dedicated SPH contact SOFT/SOFSCL differs")

    adaptive = _fields(candidate["*DEFINE_ADAPTIVE_SOLID_TO_SPH"][0][0])
    if [int(value) for value in adaptive[:7]] != [2002, 0, 2, 2003, 2, 0, 0]:
        raise AssertionError("adaptive solid-to-SPH definition differs")
    section_sph = _fields(candidate["*SECTION_SPH"][0][0])
    expected_section = [2.0, 1.2, 0.2, 2.0, 0.0, 1.0e20, 0.0, 0.0]
    if [float(value) for value in section_sph] != expected_section:
        raise AssertionError("SECTION_SPH differs")

    for keyword in (
        "*CONTROL_TIMESTEP",
        "*CONTROL_SOLID",
        "*CONTROL_CONTACT",
        "*MAT_JOHNSON_COOK",
        "*EOS_GRUNEISEN",
        "*NODE",
        "*ELEMENT_SOLID",
        "*INITIAL_VELOCITY_GENERATION",
        "*DATABASE_BINARY_D3PLOT",
        "*DATABASE_BINARY_D3PART",
        "*DATABASE_MATSUM",
        "*DATABASE_NCFORC",
    ):
        if candidate[keyword] != baseline[keyword]:
            raise AssertionError(f"immutable keyword changed: {keyword}")
    timestep = _fields(candidate["*CONTROL_TIMESTEP"][0][0])
    if float(timestep[1]) != 0.05 or float(timestep[3]) != 0.0:
        raise AssertionError("TSSFAC/DT2MS differs")
    control_solid = _fields(candidate["*CONTROL_SOLID"][0][0])
    if int(control_solid[4]) != 0:
        raise AssertionError("PSFAIL must remain 0 on the R06j branch")

    node_lines = candidate["*NODE"][0]
    element_lines = candidate["*ELEMENT_SOLID"][0]
    report = {
        "schema_version": "hf-p0-r06m-static-validation@1.0.0",
        "overall_status": "pass",
        "scope": "static_keyword_and_package_validation_not_target_solver_execution",
        "checks": {
            "baseline_identity": {"status": "pass", "sha256": R06J_DECK_SHA256},
            "ascii_deck": {"status": "pass"},
            "mesh_identity": {
                "status": "pass",
                "node_count": len(node_lines),
                "element_count": len(element_lines),
                "node_block_sha256": "sha256:" + hashlib.sha256("\n".join(node_lines).encode("ascii")).hexdigest(),
                "element_block_sha256": "sha256:"
                + hashlib.sha256("\n".join(element_lines).encode("ascii")).hexdigest(),
            },
            "post_explosion_state_identity": {"status": "pass", "impact_speed_mps": 800.0},
            "timestep_identity": {"status": "pass", "tssfac": 0.05, "dt2ms": 0.0},
            "runaway_guard": {
                "status": "pass",
                "termination_time_ms": 0.04,
                "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
                "completion_requires_problem_time_ms": 0.04,
            },
            "adaptive_solid_to_sph": {
                "status": "pass",
                "solid_part_id": 2002,
                "sph_part_id": 2003,
                "nq": 2,
                "particles_per_failed_hexahedron": 8,
                "sph_section_id": 2,
                "material_id": 2,
                "eos_id": 2,
            },
            "contact_partition": {
                "status": "pass",
                "solid_contact": {"part_set_id": 920001, "part_ids": [1002, 2002], "soft": 2},
                "sph_to_solid_contact": {
                    "slave_part_id": 2003,
                    "master_part_set_id": 920001,
                    "sstyp": 3,
                    "mstyp": 2,
                    "soft": 1,
                    "sofscl": 0.1,
                },
            },
            "measurement_boundary": {
                "status": "pass",
                "d3part_set_900001_remains_solid_part_2002_only": True,
                "sph_part_2003_expected_in_full_d3plot_and_matsum": True,
                "d3part_sph_support_not_assumed": True,
            },
            "resource_identity": {"status": "pass", "nodes": 1, "ranks": 32},
        },
        "limitations": [
            "Static PASS does not prove target-solver parsing or adaptive conversion.",
            "Normal termination is insufficient if problem time does not reach 0.04 ms.",
            "The first adaptive-SPH response is not a calibration target or reference label.",
        ],
    }
    return report


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _copy_text_artifact(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)


def build_batch(
    *,
    baseline_deck: Path,
    design_path: Path,
    contract_path: Path,
    material_path: Path,
    output_dir: Path,
) -> dict:
    baseline_deck = baseline_deck.resolve()
    design_path = design_path.resolve()
    contract_path = contract_path.resolve()
    material_path = material_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen batch: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(f"batch staging directory already exists: {staging}")
    if sha256(baseline_deck) != R06J_DECK_SHA256:
        raise AssertionError("immutable R06j parent deck hash differs")
    design = strict_json(design_path)
    contract = strict_json(contract_path)
    if (
        design["package_revision"] != REVISION
        or design["design_status"] != DESIGN_STATUS
        or contract["version_boundary"]["applies_to_package_revision"] != REVISION
        or design["paired_experiments"][0]["jobs"][0]["job_id"] != JOB_ID
    ):
        raise AssertionError("R06m design/contract identity differs")

    baseline_text = baseline_deck.read_text(encoding="ascii")
    candidate_text = transform_deck(baseline_text)
    static_validation = validate_transformed_deck(baseline_text, candidate_text)

    staging.mkdir(parents=True)
    job = staging / JOB_DIR
    job.mkdir()
    deck_path = job / HEAD_DECK
    deck_path.write_text(candidate_text, encoding="ascii", newline="\n")
    design_name = design_path.name
    contract_name = contract_path.name
    material_name = material_path.name
    _copy_text_artifact(design_path, job / design_name)
    _copy_text_artifact(contract_path, job / contract_name)
    _copy_text_artifact(material_path, job / material_name)

    case_map_path = job / "p0_case_part_map.csv"
    with case_map_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "job_id",
                "experiment_id",
                "role",
                "case_id",
                "source_r05_index",
                "plate_part_id",
                "fragment_solid_part_id",
                "adaptive_sph_part_id",
                "fragment_center_history_node_id",
                "impact_speed_mps",
                "mesh_mm",
                "plate_elements",
                "fragment_solid_elements",
                "termination_time_ms",
                "maximum_cycle_count",
                "primary_measurement_source",
            ]
        )
        writer.writerow(
            [
                JOB_ID,
                EXPERIMENT_ID,
                "adaptive_solid_to_sph_parser_runtime_mass_energy_capability_canary",
                "p0_v0800_h0p75",
                2,
                1002,
                2002,
                2003,
                47039,
                800.0,
                0.75,
                20216,
                686,
                0.04,
                MAXIMUM_CYCLE_COUNT,
                "full_d3plot_point_elements_plus_matsum_part_2003",
            ]
        )

    job_contract = {
        "schema_version": "hf-p0-r06-job-contract@1.1.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "experiment_id": EXPERIMENT_ID,
        "role": "adaptive_solid_to_sph_parser_runtime_mass_energy_capability_canary",
        "factor": "bundled_adaptive_solid_to_sph_model_form_capability",
        "factor_level": "adaptive_sph_bundle_v1",
        "case": {
            "case_id": "p0_v0800_h0p75",
            "source_r05_index": 2,
            "impact_speed_mps": 800.0,
            "mesh_mm": 0.75,
            "plate_part_id": 1002,
            "fragment_solid_part_id": 2002,
            "adaptive_sph_part_id": 2003,
            "plate_node_count": 22815,
            "fragment_node_count": 960,
            "plate_element_count": 20216,
            "fragment_element_count": 686,
            "fragment_center_history_node_id": 47039,
            "fragment_initial_reference_mass_kg": 0.002,
            "termination_time_ms": 0.04,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "fragment_mat015_erod": 1,
            "control_solid_psfail": 0,
        },
        "head_deck": HEAD_DECK,
        "resource_contract": {
            "software_platform_label": "ls-dyna_R15_mmp_d",
            "solver_family": "LS-DYNA R15 MPP double precision",
            "nodes": 1,
            "ranks": 32,
            "walltime": "12:00:00",
            "head_memory_mb": 512,
            "child_memory_mb": 256,
            "command_line_parameters": "",
        },
        "required_return_families": [
            "submitted .k",
            "solver o= output or d3hsp",
            "mes*",
            "glstat",
            "matsum",
            "sleout",
            "nodout",
            "ncforc",
            "binout*",
            "d3plot*",
            "d3part*",
            "solver stdout/stderr",
            "Slurm script and resource record",
        ],
        "completion_rule": {
            "required_problem_time_ms": 0.04,
            "cycle_guard": MAXIMUM_CYCLE_COUNT,
            "cycle_guard_exit_before_required_problem_time": "failed_canary_not_completed_response",
        },
        "status": "prepared_not_submitted_not_solver_validated",
    }
    _write_json(job / "job_contract.json", job_contract)
    _write_json(job / "static_validation.json", static_validation)

    limitations = [
        "This package is statically validated but has not been parsed or run by the target solver.",
        "R06m branches from completed R06j, not from the cancelled R06l SOFT=1 deck.",
        "The adaptive-SPH bundle is a model-form capability test, not a single-factor physical calibration.",
        "The cycle guard prevents another multi-day numerical runaway; an early guard exit is a failed canary.",
        "D3PART remains solid part 2002 only; solver-created SPH must be assessed in full D3PLOT and MATSUM.",
        "No response is admitted for engine fitting, damage-tree fitting or reference labels.",
    ]
    manifest = {
        "schema_version": "hf-p0-r06m-job-manifest@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "generated_not_solver_validated",
        "evidence_class": "high_fidelity_synthetic_model_form_capability_candidate",
        "deck": {
            "path": HEAD_DECK,
            "size_bytes": deck_path.stat().st_size,
            "sha256": sha256(deck_path),
            "node_count": 23775,
            "element_count": 20902,
            "new_solver_created_sph_expected": True,
        },
        "immutable_parent": {
            "job_id": "r06j_0800_fragment_erod1",
            "platform_job_id": "72751725",
            "deck_sha256": R06J_DECK_SHA256,
        },
        "design": {"path": design_name, "sha256": sha256(job / design_name)},
        "material": {"path": material_name, "sha256": sha256(job / material_name)},
        "successor_contract": {"path": contract_name, "sha256": sha256(job / contract_name)},
        "case_map": {"path": case_map_path.name, "sha256": sha256(case_map_path)},
        "job_contract": {"path": "job_contract.json", "sha256": sha256(job / "job_contract.json")},
        "limitations": limitations,
    }
    _write_json(job / "manifest.json", manifest)
    readme = f"""R06m adaptive solid-to-SPH capability canary

Head deck: {HEAD_DECK}
Run: one node, 32 MPP ranks, ls-dyna_R15_mmp_d, no extra command-line parameters.
The deck starts from the same post-explosion state as R06j.

This model-form canary converts failed fragment solids (part 2002) into SPH debris (part 2003),
uses eight particles per failed hexahedron, retains SOFT=2 for solid contact, and uses a dedicated
SOFT=1 asymmetric node-to-surface contact only for SPH part 2003 against solid set 920001.

Safety guard: the run must reach problem time 0.04 ms before cycle {MAXIMUM_CYCLE_COUNT}.
If the cycle limit is reached first, treat the canary as failed and return all partial outputs.

This is not physical validation and not a calibration response. Return the complete job directory.
"""
    (job / "UPLOAD_README.txt").write_text(readme, encoding="utf-8", newline="\n")

    expected_job_files = {
        HEAD_DECK,
        design_name,
        contract_name,
        material_name,
        "p0_case_part_map.csv",
        "job_contract.json",
        "static_validation.json",
        "manifest.json",
        "UPLOAD_README.txt",
    }
    if {path.name for path in job.iterdir()} != expected_job_files:
        raise AssertionError("R06m job files differ from the exact whitelist")
    batch_manifest = {
        "schema_version": "hf-p0-r06m-batch-manifest@1.0.0",
        "package_revision": REVISION,
        "job_count": 1,
        "jobs": [{"job_id": JOB_ID, "directory": JOB_DIR, "head_deck": HEAD_DECK}],
        "design": {"path": design_name, "sha256": sha256(design_path)},
        "contract": {"path": contract_name, "sha256": sha256(contract_path)},
        "static_validation": "pass",
    }
    _write_json(staging / "batch_manifest.json", batch_manifest)
    _write_json(
        staging / "R06M_SUBMISSION_READINESS.json",
        {
            "schema_version": "hf-p0-r06m-submission-readiness@1.0.0",
            "overall_status": "pass",
            "scope": "static_only_not_solver_validated",
            "job_id": JOB_ID,
            "head_deck": HEAD_DECK,
            "deck_sha256": manifest["deck"]["sha256"],
            "direct_folder_only": True,
            "zip_count": 0,
        },
    )
    finalize_staging_directory(staging, output_dir)
    return {"batch_manifest": batch_manifest, "job_manifest": manifest, "static_validation": static_validation}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R06m adaptive solid-to-SPH canary batch")
    parser.add_argument("--baseline-deck", type=Path, default=DEFAULT_BASELINE_DECK)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--material", type=Path, default=DEFAULT_MATERIAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_batch(
        baseline_deck=args.baseline_deck,
        design_path=args.design,
        contract_path=args.contract,
        material_path=args.material,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": result["static_validation"]["overall_status"],
                "output": str(args.output_dir.resolve()),
                "job_id": JOB_ID,
                "head_deck": HEAD_DECK,
                "deck_sha256": result["job_manifest"]["deck"]["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
