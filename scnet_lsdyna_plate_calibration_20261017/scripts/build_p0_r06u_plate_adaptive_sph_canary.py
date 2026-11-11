from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from build_r06g_tssfac02_return_audit import sha256, strict_json
from create_p0_r06u_plate_adaptive_sph_canary_design import (
    ASCII_INTERVAL_MS,
    BINARY_INTERVAL_MS,
    BRANCH_PARENT_DECK_SHA256,
    BRANCH_PARENT_JOB_ID,
    BRANCH_PARENT_PLATFORM_JOB_ID,
    EXPERIMENT_ID,
    JOB_ID,
    MAXIMUM_CYCLE_COUNT,
    PROBLEM_TIME_MS,
    REJECTED_JOB_ID,
    REJECTED_PLATFORM_JOB_ID,
    REVISION,
    SPEED_MPS,
)
from validate_p0_r06_diagnostics import comparable_blocks, grouped_blocks, split_fields


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_DECK = ROOT.parent / "uav_hf_r06s_1200_speed_stability_diagnostic" / "r06s_1200_speed_stability_diagnostic.k"
DEFAULT_DESIGN = ROOT / "design" / "p0_r06u_plate_adaptive_sph_canary_design.json"
DEFAULT_CONTRACT = ROOT / "design" / "p0_r06u_plate_adaptive_sph_canary_output_contract.json"
DEFAULT_MATERIAL = ROOT / "materials" / "jc4340_reference.json"
DEFAULT_CASE_MAP = ROOT / "manual_upload_20261103_r06s_1200_speed_stability_diagnostic" / "job_01_v1200_speed_stability_diagnostic" / "p0_case_part_map.csv"
DEFAULT_OUTPUT = ROOT / "manual_upload_20261104_r06u_plate_adaptive_sph_canary"
JOB_DIR = "job_01_v1200_plate_adaptive_sph_canary"
HEAD_DECK = "r06u_1200_plate_adaptive_sph_canary.k"

PARENT_TITLE = "SCNet P0 R06s 1200mps stability diagnostic 20261103"
NEW_TITLE = "SCNet P0 R06u plate adaptive-SPH canary 20261104"
PARENT_PAIR_COMMENT = "$ Pair=r06m_adaptive_solid_to_sph_canary_800 factor=adaptive_sph_bundle role=parser_runtime_mass_energy_capability_canary"
NEW_PAIR_COMMENT = "$ Pair=r06u_plate_adaptive_sph_canary_1200 factor=plate_adaptive_sph_bundle role=parser_runtime_mass_energy_capability_canary"
PARENT_CASE_COMMENT = "$ CASE p0_v0800_h0p75 source_index=2 speed=800.0 mesh=0.75"
NEW_CASE_COMMENT = "$ CASE p0_v1200_h0p75 source_index=2 speed=1200.0 mesh=0.75"
PARENT_TERMINATION = """*CONTROL_TERMINATION
5.000000000e-01,2500000,0.000000000e+00,0.000000000e+00,0.000000000e+00"""
NEW_TERMINATION = """*CONTROL_TERMINATION
1.500000000e-01,2500000,0.000000000e+00,0.000000000e+00,0.000000000e+00"""
PARENT_SPH_CONTACT = """*CONTACT_AUTOMATIC_NODES_TO_SURFACE
2003,920001,3,2,0,0,1,1
0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0,0.000000000e+00,0.000000000e+00
0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00
1,1.000000000e-01,0,0,0,0,0,0"""
NEW_SPH_CONTACTS = PARENT_SPH_CONTACT + """
*CONTACT_AUTOMATIC_NODES_TO_SURFACE
1003,920001,3,2,0,0,1,1
0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0,0.000000000e+00,0.000000000e+00
0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00
1,1.000000000e-01,0,0,0,0,0,0"""
PARENT_SPH_PART = """*PART
adaptive_sph_debris__p0_v0800_h0p75
2003,2,2,2,0,0,0,0"""
NEW_SPH_PARTS = PARENT_SPH_PART + """
*PART
adaptive_plate_sph_debris__p0_v1200_h0p75
1003,2,1,1,0,0,0,0"""
PARENT_ADAPTIVE = """*DEFINE_ADAPTIVE_SOLID_TO_SPH
2002,0,2,2003,2,0,0,"""
NEW_ADAPTIVE = PARENT_ADAPTIVE + """
*DEFINE_ADAPTIVE_SOLID_TO_SPH
1002,0,2,1003,2,0,0,"""


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise AssertionError(f"{label}: expected one exact replacement target, found {count}")
    return text.replace(old, new, 1)


def transform_deck(parent_text: str) -> str:
    candidate = parent_text
    candidate = _replace_once(candidate, PARENT_TITLE, NEW_TITLE, "TITLE")
    candidate = _replace_once(candidate, PARENT_PAIR_COMMENT, NEW_PAIR_COMMENT, "pair comment")
    candidate = _replace_once(candidate, PARENT_CASE_COMMENT, NEW_CASE_COMMENT, "case comment")
    candidate = _replace_once(candidate, PARENT_TERMINATION, NEW_TERMINATION, "CONTROL_TERMINATION")
    candidate = _replace_once(candidate, PARENT_SPH_CONTACT, NEW_SPH_CONTACTS, "plate SPH contact")
    candidate = _replace_once(candidate, PARENT_SPH_PART, NEW_SPH_PARTS, "plate SPH part")
    candidate = _replace_once(candidate, PARENT_ADAPTIVE, NEW_ADAPTIVE, "plate adaptive definition")
    return candidate


def _fields(line: str) -> list[str]:
    return split_fields(line)


def validate_transformed_deck(parent_text: str, candidate_text: str) -> dict:
    candidate_text.encode("ascii")
    parent = grouped_blocks(parent_text)
    candidate = grouped_blocks(candidate_text)
    ignored = {"*TITLE", "*CONTROL_TERMINATION", "*CONTACT_AUTOMATIC_NODES_TO_SURFACE", "*PART", "*DEFINE_ADAPTIVE_SOLID_TO_SPH"}
    if comparable_blocks(candidate_text, ignored) != comparable_blocks(parent_text, ignored):
        raise AssertionError("candidate changed a keyword outside the registered canary bundle")
    if candidate["*TITLE"] != [[NEW_TITLE]]:
        raise AssertionError("candidate TITLE differs")
    if NEW_PAIR_COMMENT not in candidate_text or NEW_CASE_COMMENT not in candidate_text:
        raise AssertionError("candidate identity comments differ")

    termination = _fields(candidate["*CONTROL_TERMINATION"][0][0])
    parent_termination = _fields(parent["*CONTROL_TERMINATION"][0][0])
    if float(termination[0]) != PROBLEM_TIME_MS or int(termination[1]) != MAXIMUM_CYCLE_COUNT:
        raise AssertionError("R06u horizon or cycle guard differs")
    if termination[1:] != parent_termination[1:]:
        raise AssertionError("CONTROL_TERMINATION changed beyond ENDTIM")

    if candidate["*PART"][:3] != parent["*PART"]:
        raise AssertionError("existing plate, fragment or fragment-SPH part changed")
    if len(candidate["*PART"]) != 4 or candidate["*PART"][3] != [
        "adaptive_plate_sph_debris__p0_v1200_h0p75",
        "1003,2,1,1,0,0,0,0",
    ]:
        raise AssertionError("plate SPH part 1003 differs")

    if candidate["*CONTACT_AUTOMATIC_NODES_TO_SURFACE"][0] != parent["*CONTACT_AUTOMATIC_NODES_TO_SURFACE"][0]:
        raise AssertionError("existing fragment-SPH contact changed")
    if len(candidate["*CONTACT_AUTOMATIC_NODES_TO_SURFACE"]) != 2:
        raise AssertionError("plate SPH contact count differs")
    plate_contact = candidate["*CONTACT_AUTOMATIC_NODES_TO_SURFACE"][1]
    if [int(value) for value in _fields(plate_contact[0])[:4]] != [1003, 920001, 3, 2]:
        raise AssertionError("plate SPH contact identity differs")
    optional_a = _fields(plate_contact[3])
    if int(optional_a[0]) != 1 or float(optional_a[1]) != 0.1:
        raise AssertionError("plate SPH contact SOFT/SOFSCL differs")

    if candidate["*DEFINE_ADAPTIVE_SOLID_TO_SPH"][0] != parent["*DEFINE_ADAPTIVE_SOLID_TO_SPH"][0]:
        raise AssertionError("existing fragment adaptive definition changed")
    if len(candidate["*DEFINE_ADAPTIVE_SOLID_TO_SPH"]) != 2:
        raise AssertionError("plate adaptive definition count differs")
    adaptive = _fields(candidate["*DEFINE_ADAPTIVE_SOLID_TO_SPH"][1][0])
    if [int(value) for value in adaptive[:7]] != [1002, 0, 2, 1003, 2, 0, 0]:
        raise AssertionError("plate adaptive definition differs")

    for keyword in (
        "*CONTROL_TIMESTEP", "*CONTROL_SOLID", "*CONTROL_CONTACT", "*CONTROL_ENERGY",
        "*CONTACT_ERODING_SINGLE_SURFACE", "*SECTION_SOLID_TITLE", "*SECTION_SPH",
        "*MAT_JOHNSON_COOK", "*EOS_GRUNEISEN", "*NODE", "*ELEMENT_SOLID",
        "*INITIAL_VELOCITY_GENERATION", "*DATABASE_BINARY_D3PLOT", "*DATABASE_BINARY_D3PART",
        "*DATABASE_GLSTAT", "*DATABASE_MATSUM", "*DATABASE_SLEOUT", "*DATABASE_NODOUT", "*DATABASE_NCFORC",
        "*SET_PART_LIST_TITLE",
    ):
        if candidate[keyword] != parent[keyword]:
            raise AssertionError(f"immutable keyword changed: {keyword}")
    control_solid = _fields(candidate["*CONTROL_SOLID"][0][0])
    if int(control_solid[4]) != 0:
        raise AssertionError("R06u must restore the completed R06s PSFAIL=0 branch")
    timestep = _fields(candidate["*CONTROL_TIMESTEP"][0][0])
    if float(timestep[1]) != 0.05 or float(timestep[3]) != 0.0:
        raise AssertionError("TSSFAC/DT2MS differs")

    node_lines = candidate["*NODE"][0]
    element_lines = candidate["*ELEMENT_SOLID"][0]
    return {
        "schema_version": "hf-p0-r06u-static-validation@1.0.0",
        "overall_status": "pass",
        "scope": "static_keyword_and_package_validation_not_target_solver_execution",
        "parent_deck_sha256": BRANCH_PARENT_DECK_SHA256,
        "registered_changes": [
            "TITLE_and_identity_comments",
            "CONTROL_TERMINATION_endtim_0p50_to_0p15",
            "add_plate_SPH_part_1003_same_material_and_EOS_as_plate",
            "add_plate_SPH_to_solid_contact",
            "add_plate_adaptive_solid_to_SPH_definition",
        ],
        "all_other_keyword_blocks_identical": True,
        "mesh_identity": {
            "node_count": len(node_lines),
            "element_count": len(element_lines),
            "node_block_sha256": "sha256:" + hashlib.sha256("\n".join(node_lines).encode("ascii")).hexdigest(),
            "element_block_sha256": "sha256:" + hashlib.sha256("\n".join(element_lines).encode("ascii")).hexdigest(),
        },
        "impact_speed_mps": SPEED_MPS,
        "termination_time_ms": PROBLEM_TIME_MS,
        "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
        "control_solid_psfail": 0,
        "plate_adaptive_sph": {"solid_part_id": 1002, "sph_part_id": 1003, "particles_per_failed_hexahedron": 8, "material_id": 1, "eos_id": 1},
        "fragment_adaptive_sph_unchanged": {"solid_part_id": 2002, "sph_part_id": 2003},
        "d3plot_interval_ms": BINARY_INTERVAL_MS,
        "d3part_interval_ms": BINARY_INTERVAL_MS,
        "ascii_interval_ms": ASCII_INTERVAL_MS,
        "resource_identity_is_external_to_deck": True,
        "limitations": [
            "Static PASS does not prove target-solver parsing, adaptive plate conversion or stability.",
            "No SPH-SPH interpart contact is added in this capability canary; terminal physical response remains denied.",
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _write_case_map(source_path: Path, output_path: Path) -> None:
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if fieldnames is None or len(rows) != 1:
        raise ValueError("R06u case-map source must contain exactly one identified case")
    row = dict(rows[0])
    updates = {
        "job_id": JOB_ID,
        "experiment_id": EXPERIMENT_ID,
        "role": "plate_adaptive_sph_parser_runtime_mass_energy_capability_canary",
        "case_id": "p0_v1200_h0p75_plate_adaptive_sph",
        "impact_speed_mps": "1200.0",
        "termination_time_ms": "0.15",
        "maximum_cycle_count": str(MAXIMUM_CYCLE_COUNT),
        "primary_measurement_source": "staged_ascii_mass_energy_gate_then_full_d3plot_plate_sph_conversion_ledger",
    }
    row.update({key: value for key, value in updates.items() if key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def build_batch(*, parent_deck: Path, design_path: Path, contract_path: Path, material_path: Path, case_map_path: Path, output_dir: Path) -> dict:
    parent_deck = parent_deck.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if sha256(parent_deck) != BRANCH_PARENT_DECK_SHA256:
        raise ValueError("immutable R06s branch-parent deck hash differs")
    design = strict_json(design_path.resolve())
    contract = strict_json(contract_path.resolve())
    if design["package_revision"] != REVISION or design["jobs"][0]["job_id"] != JOB_ID:
        raise ValueError("R06u design identity differs")
    if contract["version_boundary"]["applies_to_package_revision"] != REVISION:
        raise ValueError("R06u contract identity differs")

    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    job = staging / JOB_DIR
    job.mkdir()
    parent_text = parent_deck.read_text(encoding="ascii")
    candidate_text = transform_deck(parent_text)
    deck_path = job / HEAD_DECK
    deck_path.write_text(candidate_text, encoding="ascii", newline="")
    static = validate_transformed_deck(parent_text, candidate_text)
    static["candidate_deck_sha256"] = sha256(deck_path)
    _write_json(job / "static_validation.json", static)

    copied = {
        "p0_r06u_plate_adaptive_sph_canary_design.json": design_path.resolve(),
        "p0_r06u_plate_adaptive_sph_canary_output_contract.json": contract_path.resolve(),
        "jc4340_reference.json": material_path.resolve(),
    }
    for name, source in copied.items():
        shutil.copy2(source, job / name)
    _write_case_map(case_map_path.resolve(), job / "p0_case_part_map.csv")
    job_contract = {
        "schema_version": "hf-p0-r06-job-contract@1.1.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "experiment_id": EXPERIMENT_ID,
        "role": "plate_adaptive_sph_parser_runtime_mass_energy_capability_canary",
        "factor": "plate_adaptive_solid_to_sph_model_form_bundle",
        "factor_level": "part_1002_to_same_material_sph_part_1003",
        "factor_layer": "model_form_capability_canary_not_physical_calibration",
        "case": {
            "case_id": "p0_v1200_h0p75_plate_adaptive_sph",
            "impact_speed_mps": SPEED_MPS,
            "mesh_mm": 0.75,
            "plate_solid_part_id": 1002,
            "plate_sph_part_id": 1003,
            "fragment_solid_part_id": 2002,
            "fragment_sph_part_id": 2003,
            "termination_time_ms": PROBLEM_TIME_MS,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "control_solid_psfail": 0,
        },
        "output_intervals_ms": {"d3plot": BINARY_INTERVAL_MS, "d3part": BINARY_INTERVAL_MS, "ascii_databases": ASCII_INTERVAL_MS},
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
            "resource_identity_is_a_hard_gate": True,
        },
        "required_return_families": [
            "stage1: submitted .k, manifest, job_contract, all mes*, glstat/matsum/sleout/nodout/ncforc, sldsph_frag if present, stdout/stderr, Slurm and resource log",
            "stage2 only after stage1 pass: all d3plot*",
            "stage3 only after stage2 pass: all d3part* and binout*",
        ],
        "status": "prepared_not_submitted_not_solver_validated",
    }
    _write_json(job / "job_contract.json", job_contract)
    manifest = {
        "schema_version": "hf-p0-r06u-job-manifest@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "generated_not_solver_validated",
        "evidence_class": "high_fidelity_synthetic_plate_adaptive_sph_capability_candidate",
        "deck": {"path": HEAD_DECK, "size_bytes": deck_path.stat().st_size, "sha256": sha256(deck_path), "registered_changes": static["registered_changes"]},
        "immutable_branch_parent": {"job_id": BRANCH_PARENT_JOB_ID, "platform_job_id": BRANCH_PARENT_PLATFORM_JOB_ID, "deck_sha256": BRANCH_PARENT_DECK_SHA256},
        "rejected_predecessor": {"job_id": REJECTED_JOB_ID, "platform_job_id": REJECTED_PLATFORM_JOB_ID, "classification": "plate_material_state_collapse"},
        "design": {"path": "p0_r06u_plate_adaptive_sph_canary_design.json", "sha256": sha256(job / "p0_r06u_plate_adaptive_sph_canary_design.json")},
        "successor_contract": {"path": "p0_r06u_plate_adaptive_sph_canary_output_contract.json", "sha256": sha256(job / "p0_r06u_plate_adaptive_sph_canary_output_contract.json")},
        "material": {"path": "jc4340_reference.json", "sha256": sha256(job / "jc4340_reference.json")},
        "case_map": {"path": "p0_case_part_map.csv", "sha256": sha256(job / "p0_case_part_map.csv")},
        "job_contract": {"path": "job_contract.json", "sha256": sha256(job / "job_contract.json")},
        "limitations": [
            "Static validation does not prove target-platform parsing, plate conversion or numerical stability.",
            "R06u stops at 0.15 ms and cannot provide the terminal 0.50 ms response.",
            "No SPH-SPH interpart contact is added; terminal physical response remains denied.",
        ],
    }
    _write_json(job / "manifest.json", manifest)
    (job / "UPLOAD_README.txt").write_text(
        f"""R06u plate adaptive-SPH capability canary\n\nSelect {HEAD_DECK} as the head deck.\nUse 1 node, exactly 32 MPP ranks, ls-dyna_R15_mmp_d, head memory 512 MB, child memory 256 MB and no extra command-line parameters.\nThis branches from completed R06s, restores PSFAIL=0, and adds same-material plate solid-to-SPH conversion part 1002 -> part 1003.\nThe canary stops at {PROBLEM_TIME_MS:.2f} ms, beyond R06t's 0.10872 ms failure point. Keep ENDCYC={MAXIMUM_CYCLE_COUNT}, speed 1200 m/s, output cadence, materials, mesh and timestep controls frozen.\nAfter completion, download only the Stage-1 small-file set first. Do not download d3plot/d3part/binout until Stage 1 passes.\nThis is a model-form capability canary, not physical validation or calibration.\n""",
        encoding="utf-8",
        newline="\n",
    )
    job_files = sorted(path for path in job.iterdir() if path.is_file())
    _write_json(staging / "batch_manifest.json", {
        "schema_version": "hf-p0-r06u-batch-manifest@1.0.0",
        "package_revision": REVISION,
        "job_count": 1,
        "jobs": [{"job_id": JOB_ID, "directory": JOB_DIR, "head_deck": HEAD_DECK, "head_deck_sha256": manifest["deck"]["sha256"]}],
        "files": [{"path": path.relative_to(staging).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in job_files],
    })
    _write_json(staging / "R06U_SUBMISSION_READINESS.json", {
        "schema_version": "hf-p0-r06u-submission-readiness@1.0.0",
        "overall_status": "pass",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "required_resources": {"nodes": 1, "ranks": 32},
        "archive_policy": "no_archive_direct_folder_delivery",
        "solver_status": "not_submitted_not_solver_validated",
    })
    finalize_staging_directory(staging, output_dir)
    return {"job_id": JOB_ID, "head_deck": HEAD_DECK, "head_deck_sha256": manifest["deck"]["sha256"], "output": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R06u plate adaptive-SPH canary")
    parser.add_argument("--parent-deck", type=Path, default=DEFAULT_PARENT_DECK)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--material", type=Path, default=DEFAULT_MATERIAL)
    parser.add_argument("--case-map", type=Path, default=DEFAULT_CASE_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_batch(parent_deck=args.parent_deck, design_path=args.design, contract_path=args.contract, material_path=args.material, case_map_path=args.case_map, output_dir=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
