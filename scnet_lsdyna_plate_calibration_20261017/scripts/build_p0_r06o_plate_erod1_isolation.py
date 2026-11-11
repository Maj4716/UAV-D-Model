from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from build_r06g_tssfac02_return_audit import sha256, strict_json
from create_p0_r06o_plate_erod1_isolation_design import (
    EXPERIMENT_ID,
    JOB_ID,
    MAXIMUM_CYCLE_COUNT,
    PARENT_DECK_SHA256,
    PARENT_JOB_ID,
    PARENT_PLATFORM_JOB_ID,
    REVISION,
    TARGET_PLATE_ELEMENT_ID,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_DECK = (
    ROOT
    / "analysis"
    / "72864757_r06n_0800_adaptive_sph_32rank_confirmation_20261102"
    / "immutable_parent_r06n_0800_adaptive_sph_32rank_confirmation.k"
)
DEFAULT_DESIGN = ROOT / "design" / "p0_r06o_plate_erod1_isolation_design.json"
DEFAULT_CONTRACT = ROOT / "design" / "p0_r06o_plate_erod1_output_contract.json"
DEFAULT_MATERIAL = ROOT / "materials" / "jc4340_reference.json"
DEFAULT_CASE_MAP = (
    ROOT
    / "manual_upload_20261101_r06n_adaptive_sph_32rank_confirmation"
    / "job_01_v0800_adaptive_sph_32rank_confirmation"
    / "p0_case_part_map.csv"
)
DEFAULT_OUTPUT = ROOT / "manual_upload_20261102_r06o_plate_erod1_isolation"
JOB_DIR = "job_01_v0800_plate_erod1_isolation"
HEAD_DECK = "r06o_0800_plate_erod1_isolation.k"
PARENT_TITLE = "SCNet P0 R06n adaptive SPH exact 32rank confirmation 20261101"
NEW_TITLE = "SCNet P0 R06o plate MAT EROD1 isolation 20261102"
PLATE_EROD0_LINE = "6.100000000e-01,0.000000000e+00,0.000000000e+00,1.000000000e-06,0"
PLATE_EROD1_LINE = "6.100000000e-01,0.000000000e+00,1,1.000000000e-06,0"


def transform_deck(parent_text: str) -> str:
    lines = parent_text.splitlines(keepends=True)
    title_indexes = [i for i, line in enumerate(lines) if line.strip().upper() == "*TITLE"]
    if title_indexes != [1] or lines[2].rstrip("\r\n") != PARENT_TITLE:
        raise ValueError("parent TITLE layout differs")
    title_newline = lines[2][len(PARENT_TITLE):]
    lines[2] = NEW_TITLE + title_newline
    erod_indexes = [i for i, line in enumerate(lines) if line.rstrip("\r\n") == PLATE_EROD0_LINE]
    if len(erod_indexes) != 1:
        raise ValueError(f"expected exactly one plate EROD=0 material line, found {erod_indexes}")
    erod_index = erod_indexes[0]
    erod_newline = lines[erod_index][len(PLATE_EROD0_LINE):]
    lines[erod_index] = PLATE_EROD1_LINE + erod_newline
    return "".join(lines)


def validate_transformed_deck(parent_text: str, candidate_text: str) -> dict:
    parent_lines = parent_text.splitlines()
    candidate_lines = candidate_text.splitlines()
    if len(parent_lines) != len(candidate_lines):
        raise ValueError("candidate line count differs")
    differences = [i for i, pair in enumerate(zip(parent_lines, candidate_lines), start=1) if pair[0] != pair[1]]
    plate_line_number = parent_lines.index(PLATE_EROD0_LINE) + 1
    expected_differences = [3, plate_line_number]
    if differences != expected_differences:
        raise ValueError(f"candidate differs outside TITLE and plate EROD: {differences[:20]}")
    if candidate_lines[2] != NEW_TITLE or candidate_lines[plate_line_number - 1] != PLATE_EROD1_LINE:
        raise ValueError("candidate TITLE or plate EROD line differs")
    restored = list(candidate_lines)
    restored[2] = PARENT_TITLE
    restored[plate_line_number - 1] = PLATE_EROD0_LINE
    if restored != parent_lines:
        raise ValueError("restoring the two allowed lines did not recover exact parent deck")
    if candidate_text.count(PLATE_EROD1_LINE) != 2:
        raise ValueError("plate and fragment materials must both carry EROD=1 after transformation")
    required_tokens = [
        "*DEFINE_ADAPTIVE_SOLID_TO_SPH",
        "2002,0,2,2003,2,0,0,",
        "*CONTACT_ERODING_SINGLE_SURFACE",
        "*CONTACT_AUTOMATIC_NODES_TO_SURFACE",
        "4.000000000e-02,829930",
        "0.000000000e+00,5.000000000e-02,0,0.000000000e+00",
    ]
    missing = [token for token in required_tokens if token not in candidate_text]
    if missing:
        raise ValueError(f"candidate is missing frozen tokens: {missing}")
    return {
        "schema_version": "hf-p0-r06o-static-validation@1.0.0",
        "overall_status": "pass",
        "parent_deck_sha256": PARENT_DECK_SHA256,
        "permitted_differences": ["TITLE_identifier_line", "plate_MAT_JOHNSON_COOK_EROD_field_0_to_1"],
        "differing_line_numbers": differences,
        "parent_title": PARENT_TITLE,
        "candidate_title": NEW_TITLE,
        "plate_erod_parent": 0,
        "plate_erod_candidate": 1,
        "fragment_erod_candidate": 1,
        "all_other_deck_lines_byte_identical": True,
        "termination_time_ms": 0.04,
        "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
        "tssfac": 0.05,
        "dt2ms": 0.0,
        "resource_identity_is_external_to_deck": True,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _write_case_map(source_path: Path, output_path: Path) -> None:
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if fieldnames is None or len(rows) != 1:
        raise ValueError("R06o case-map source must contain exactly one identified case")
    row = dict(rows[0])
    row.update(
        {
            "job_id": JOB_ID,
            "experiment_id": EXPERIMENT_ID,
            "role": "plate_material_failure_erosion_isolation_canary",
        }
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def build_batch(
    *,
    parent_deck: Path,
    design_path: Path,
    contract_path: Path,
    material_path: Path,
    case_map_path: Path,
    output_dir: Path,
) -> dict:
    parent_deck = parent_deck.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if sha256(parent_deck) != PARENT_DECK_SHA256:
        raise ValueError("immutable R06n parent deck hash differs")
    design = strict_json(design_path.resolve())
    contract = strict_json(contract_path.resolve())
    if design["package_revision"] != REVISION or design["jobs"][0]["job_id"] != JOB_ID:
        raise ValueError("R06o design identity differs")
    if contract["version_boundary"]["applies_to_package_revision"] != REVISION:
        raise ValueError("R06o contract identity differs")

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
    static_validation = validate_transformed_deck(parent_text, candidate_text)
    static_validation["candidate_deck_sha256"] = sha256(deck_path)
    _write_json(job / "static_validation.json", static_validation)

    copied = {
        "p0_r06o_plate_erod1_isolation_design.json": design_path.resolve(),
        "p0_r06o_plate_erod1_output_contract.json": contract_path.resolve(),
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
        "role": "plate_material_failure_erosion_isolation_canary",
        "factor": "plate_mat_johnson_cook_erod",
        "factor_level": "plate_erod_1",
        "case": {
            "case_id": "p0_v0800_h0p75",
            "impact_speed_mps": 800.0,
            "mesh_mm": 0.75,
            "plate_part_id": 1002,
            "fragment_solid_part_id": 2002,
            "adaptive_sph_part_id": 2003,
            "target_plate_element_id": TARGET_PLATE_ELEMENT_ID,
            "termination_time_ms": 0.04,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "plate_mat015_erod": 1,
            "fragment_mat015_erod": 1,
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
            "resource_identity_is_a_hard_gate": True,
        },
        "required_return_families": [
            "submitted .k", "d3hsp/o= and mes*", "glstat", "matsum", "sleout", "nodout", "ncforc",
            "binout*", "d3plot*", "d3part*", "solver stdout/stderr", "Slurm script and resource record",
        ],
        "completion_rule": {
            "required_problem_time_ms": 0.04,
            "cycle_guard": MAXIMUM_CYCLE_COUNT,
            "required_actual_nodes": 1,
            "required_actual_ranks": 32,
        },
        "status": "prepared_not_submitted_not_solver_validated",
    }
    _write_json(job / "job_contract.json", job_contract)
    manifest = {
        "schema_version": "hf-p0-r06o-job-manifest@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "generated_not_solver_validated",
        "evidence_class": "high_fidelity_synthetic_plate_failure_erosion_isolation_candidate",
        "deck": {
            "path": HEAD_DECK,
            "size_bytes": deck_path.stat().st_size,
            "sha256": sha256(deck_path),
            "differences_from_parent": static_validation["permitted_differences"],
        },
        "immutable_parent": {
            "job_id": PARENT_JOB_ID,
            "platform_job_id": PARENT_PLATFORM_JOB_ID,
            "deck_sha256": PARENT_DECK_SHA256,
            "classification": "resource_correct_but_plate_material_failure_confirmation_rejected",
        },
        "design": {"path": list(copied)[0], "sha256": sha256(job / list(copied)[0])},
        "successor_contract": {"path": list(copied)[1], "sha256": sha256(job / list(copied)[1])},
        "material": {"path": "jc4340_reference.json", "sha256": sha256(job / "jc4340_reference.json")},
        "case_map": {"path": "p0_case_part_map.csv", "sha256": sha256(job / "p0_case_part_map.csv")},
        "job_contract": {"path": "job_contract.json", "sha256": sha256(job / "job_contract.json")},
        "limitations": [
            "Static validation proves only the two permitted deck-line differences; target-platform execution is untested.",
            "Plate EROD=1 is a numerical mechanism isolation, not a physically validated plate-failure law.",
            "Retaining a failed plate element may cause timestep collapse; all completion and energy gates remain required.",
            "The 0.04 ms horizon remains unresolved; no response is admitted for engine or damage-tree fitting.",
        ],
    }
    _write_json(job / "manifest.json", manifest)
    readme = f"""R06o plate MAT EROD=1 isolation\n\nSelect {HEAD_DECK} as the head deck.\nRequired resources: 1 node, exactly 32 MPP ranks, ls-dyna_R15_mmp_d.\nThe deck changes only the TITLE and plate MAT_JOHNSON_COOK EROD field 0->1 from returned R06n.\nAll fragment adaptive-SPH, contact, timestep, output and 0.04 ms controls remain unchanged.\nThis is a numerical isolation canary. Return the complete output even if timestep collapse or cycle-guard exit occurs.\nIt is not physical, engine or damage-tree calibration.\n"""
    (job / "UPLOAD_README.txt").write_text(readme, encoding="utf-8", newline="\n")

    job_files = sorted(path for path in job.iterdir() if path.is_file())
    batch_manifest = {
        "schema_version": "hf-p0-r06o-batch-manifest@1.0.0",
        "package_revision": REVISION,
        "job_count": 1,
        "jobs": [{"job_id": JOB_ID, "directory": JOB_DIR, "head_deck": HEAD_DECK, "head_deck_sha256": manifest["deck"]["sha256"]}],
        "files": [{"path": path.relative_to(staging).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in job_files],
    }
    _write_json(staging / "batch_manifest.json", batch_manifest)
    readiness = {
        "schema_version": "hf-p0-r06o-submission-readiness@1.0.0",
        "overall_status": "pass",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "allowed_deck_differences": static_validation["permitted_differences"],
        "required_resources": {"nodes": 1, "ranks": 32},
        "archive_policy": "no_archive_direct_folder_delivery",
        "solver_status": "not_submitted_not_solver_validated",
    }
    _write_json(staging / "R06O_SUBMISSION_READINESS.json", readiness)
    finalize_staging_directory(staging, output_dir)
    return {"job_id": JOB_ID, "head_deck": HEAD_DECK, "head_deck_sha256": manifest["deck"]["sha256"], "output": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R06o plate EROD=1 isolation batch")
    parser.add_argument("--parent-deck", type=Path, default=DEFAULT_PARENT_DECK)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--material", type=Path, default=DEFAULT_MATERIAL)
    parser.add_argument("--case-map", type=Path, default=DEFAULT_CASE_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_batch(
        parent_deck=args.parent_deck,
        design_path=args.design,
        contract_path=args.contract,
        material_path=args.material,
        case_map_path=args.case_map,
        output_dir=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
