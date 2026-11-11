from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from build_r06g_tssfac02_return_audit import sha256, strict_json
from create_p0_r06p_terminal_horizon_0p50_design import (
    ASCII_INTERVAL_MS,
    CANDIDATE_HORIZON_MS,
    D3PART_INTERVAL_MS,
    D3PLOT_INTERVAL_MS,
    EXPERIMENT_ID,
    JOB_ID,
    MAXIMUM_CYCLE_COUNT,
    PARENT_DECK_SHA256,
    PARENT_JOB_ID,
    PARENT_PLATFORM_JOB_ID,
    REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_DECK = ROOT / "analysis" / "72894870_r06o_0800_plate_erod1_isolation_20261102" / "immutable_parent_r06o_0800_plate_erod1_isolation.k"
DEFAULT_DESIGN = ROOT / "design" / "p0_r06p_terminal_horizon_0p50_design.json"
DEFAULT_CONTRACT = ROOT / "design" / "p0_r06p_terminal_horizon_0p50_output_contract.json"
DEFAULT_MATERIAL = ROOT / "materials" / "jc4340_reference.json"
DEFAULT_CASE_MAP = ROOT / "manual_upload_20261102_r06o_plate_erod1_isolation" / "job_01_v0800_plate_erod1_isolation" / "p0_case_part_map.csv"
DEFAULT_OUTPUT = ROOT / "manual_upload_20261102_r06p_terminal_horizon_0p50"
JOB_DIR = "job_01_v0800_terminal_horizon_0p50"
HEAD_DECK = "r06p_0800_terminal_horizon_0p50.k"
PARENT_TITLE = "SCNet P0 R06o plate MAT EROD1 isolation 20261102"
NEW_TITLE = "SCNet P0 R06p terminal horizon 0p50ms 20261102"
PARENT_TERMINATION = "4.000000000e-02,829930,0.000000000e+00,0.000000000e+00,0.000000000e+00"
CANDIDATE_TERMINATION = "5.000000000e-01,829930,0.000000000e+00,0.000000000e+00,0.000000000e+00"
PARENT_D3PLOT = "1.000000000e-04"
CANDIDATE_D3PLOT = "5.000000000e-04"
PARENT_FINE_INTERVAL = "5.000000000e-05"
CANDIDATE_FINE_INTERVAL = "2.500000000e-04"
PARENT_D3PART = "5.000000000e-05,0,0,0,900001"
CANDIDATE_D3PART = "2.500000000e-04,0,0,0,900001"
EXPECTED_DIFFERING_LINES = [3, 8, 19, 24, 29, 31, 33, 35, 37]


def transform_deck(parent_text: str) -> str:
    lines = parent_text.splitlines(keepends=True)
    replacements = {
        3: (PARENT_TITLE, NEW_TITLE),
        8: (PARENT_TERMINATION, CANDIDATE_TERMINATION),
        19: (PARENT_D3PLOT, CANDIDATE_D3PLOT),
        24: (PARENT_D3PART, CANDIDATE_D3PART),
        29: (PARENT_FINE_INTERVAL, CANDIDATE_FINE_INTERVAL),
        31: (PARENT_FINE_INTERVAL, CANDIDATE_FINE_INTERVAL),
        33: (PARENT_FINE_INTERVAL, CANDIDATE_FINE_INTERVAL),
        35: (PARENT_FINE_INTERVAL, CANDIDATE_FINE_INTERVAL),
        37: (PARENT_FINE_INTERVAL, CANDIDATE_FINE_INTERVAL),
    }
    for line_number, (old, new) in replacements.items():
        index = line_number - 1
        actual = lines[index].rstrip("\r\n")
        if actual != old:
            raise ValueError(f"parent line {line_number} differs: {actual!r}")
        newline = lines[index][len(old):]
        lines[index] = new + newline
    return "".join(lines)


def validate_transformed_deck(parent_text: str, candidate_text: str) -> dict:
    parent_lines = parent_text.splitlines()
    candidate_lines = candidate_text.splitlines()
    if len(parent_lines) != len(candidate_lines):
        raise ValueError("candidate line count differs")
    differences = [i for i, pair in enumerate(zip(parent_lines, candidate_lines), start=1) if pair[0] != pair[1]]
    if differences != EXPECTED_DIFFERING_LINES:
        raise ValueError(f"candidate differs outside registered horizon/output lines: {differences[:30]}")
    restored = list(candidate_lines)
    for line_number in EXPECTED_DIFFERING_LINES:
        restored[line_number - 1] = parent_lines[line_number - 1]
    if restored != parent_lines:
        raise ValueError("restoring registered lines did not recover the exact R06o parent")
    required_tokens = [
        "*DEFINE_ADAPTIVE_SOLID_TO_SPH",
        "2002,0,2,2003,2,0,0,",
        "*CONTACT_ERODING_SINGLE_SURFACE",
        "*CONTACT_AUTOMATIC_NODES_TO_SURFACE",
        "6.100000000e-01,0.000000000e+00,1,1.000000000e-06,0",
        CANDIDATE_TERMINATION,
        CANDIDATE_D3PLOT,
        CANDIDATE_D3PART,
    ]
    missing = [token for token in required_tokens if token not in candidate_text]
    if missing:
        raise ValueError(f"candidate is missing frozen tokens: {missing}")
    if candidate_text.count("6.100000000e-01,0.000000000e+00,1,1.000000000e-06,0") != 2:
        raise ValueError("both plate and fragment must retain EROD=1")
    if any(candidate_lines[line_number - 1] != CANDIDATE_FINE_INTERVAL for line_number in (29, 31, 33, 35, 37)):
        raise ValueError("all five registered ASCII output cards must use the new interval")
    return {
        "schema_version": "hf-p0-r06p-static-validation@1.0.0",
        "overall_status": "pass",
        "parent_deck_sha256": PARENT_DECK_SHA256,
        "permitted_differences": [
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
        "differing_line_numbers": differences,
        "all_other_deck_lines_byte_identical": True,
        "model_semantics_unchanged": True,
        "termination_time_ms": CANDIDATE_HORIZON_MS,
        "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
        "d3plot_interval_ms": D3PLOT_INTERVAL_MS,
        "d3part_interval_ms": D3PART_INTERVAL_MS,
        "ascii_interval_ms": ASCII_INTERVAL_MS,
        "plate_erod": 1,
        "fragment_erod": 1,
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
        raise ValueError("R06p case-map source must contain exactly one identified case")
    row = dict(rows[0])
    row.update({
        "job_id": JOB_ID,
        "experiment_id": EXPERIMENT_ID,
        "role": "terminal_response_horizon_0p50_canary",
        "termination_time_ms": str(CANDIDATE_HORIZON_MS),
    })
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
        raise ValueError("immutable R06o parent deck hash differs")
    design = strict_json(design_path.resolve())
    contract = strict_json(contract_path.resolve())
    if design["package_revision"] != REVISION or design["jobs"][0]["job_id"] != JOB_ID:
        raise ValueError("R06p design identity differs")
    if contract["version_boundary"]["applies_to_package_revision"] != REVISION:
        raise ValueError("R06p contract identity differs")
    if contract["output_resolution_gate"]["status"] != "pass":
        raise ValueError("R06p output temporal-resolution gate did not pass")

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
        "p0_r06p_terminal_horizon_0p50_design.json": design_path.resolve(),
        "p0_r06p_terminal_horizon_0p50_output_contract.json": contract_path.resolve(),
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
        "role": "terminal_response_horizon_0p50_canary",
        "factor": "terminal_response_horizon_ms",
        "factor_level": "0p50_ms",
        "case": {
            "case_id": "p0_v0800_h0p75",
            "impact_speed_mps": 800.0,
            "mesh_mm": 0.75,
            "plate_part_id": 1002,
            "fragment_solid_part_id": 2002,
            "adaptive_sph_part_id": 2003,
            "capture_plane_z_mm": -12.0,
            "termination_time_ms": CANDIDATE_HORIZON_MS,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "plate_mat015_erod": 1,
            "fragment_mat015_erod": 1,
        },
        "output_intervals_ms": {"d3plot": D3PLOT_INTERVAL_MS, "d3part": D3PART_INTERVAL_MS, "ascii_databases": ASCII_INTERVAL_MS},
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
            "required_problem_time_ms": CANDIDATE_HORIZON_MS,
            "cycle_guard": MAXIMUM_CYCLE_COUNT,
            "required_actual_nodes": 1,
            "required_actual_ranks": 32,
        },
        "status": "prepared_not_submitted_not_solver_validated",
    }
    _write_json(job / "job_contract.json", job_contract)
    manifest = {
        "schema_version": "hf-p0-r06p-job-manifest@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "generated_not_solver_validated",
        "evidence_class": "high_fidelity_synthetic_terminal_response_horizon_candidate",
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
            "classification": "successful_plate_erod1_engineering_isolation_response_horizon_still_unqualified",
        },
        "design": {"path": "p0_r06p_terminal_horizon_0p50_design.json", "sha256": sha256(job / "p0_r06p_terminal_horizon_0p50_design.json")},
        "successor_contract": {"path": "p0_r06p_terminal_horizon_0p50_output_contract.json", "sha256": sha256(job / "p0_r06p_terminal_horizon_0p50_output_contract.json")},
        "material": {"path": "jc4340_reference.json", "sha256": sha256(job / "jc4340_reference.json")},
        "case_map": {"path": "p0_case_part_map.csv", "sha256": sha256(job / "p0_case_part_map.csv")},
        "job_contract": {"path": "job_contract.json", "sha256": sha256(job / "job_contract.json")},
        "limitations": [
            "Static validation proves only the registered horizon and measurement-cadence differences; target-platform execution is untested.",
            "The 0.50 ms selection uses terminal linear projection only and is not a physical trajectory prediction.",
            "The cycle guard may stop this run before 0.50 ms if later distortion reduces the timestep.",
            "A complete response remains diagnostic and is not yet engine fitting, damage-tree fitting or physical validation.",
        ],
    }
    _write_json(job / "manifest.json", manifest)
    readme = f"""R06p terminal response horizon 0.50 ms\n\nSelect {HEAD_DECK} as the head deck.\nRequired resources: 1 node, exactly 32 MPP ranks, ls-dyna_R15_mmp_d.\nThe model is the exact returned R06o engineering baseline. Model semantics are frozen; only termination time and measurement-only output cadence change.\nTermination is 0.50 ms with {MAXIMUM_CYCLE_COUNT} cycle guard. If the guard is hit, do not restart or tune; return the complete output.\nThis is response acquisition, not physical, engine or damage-tree calibration.\n"""
    (job / "UPLOAD_README.txt").write_text(readme, encoding="utf-8", newline="\n")

    job_files = sorted(path for path in job.iterdir() if path.is_file())
    batch_manifest = {
        "schema_version": "hf-p0-r06p-batch-manifest@1.0.0",
        "package_revision": REVISION,
        "job_count": 1,
        "jobs": [{"job_id": JOB_ID, "directory": JOB_DIR, "head_deck": HEAD_DECK, "head_deck_sha256": manifest["deck"]["sha256"]}],
        "files": [{"path": path.relative_to(staging).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in job_files],
    }
    _write_json(staging / "batch_manifest.json", batch_manifest)
    readiness = {
        "schema_version": "hf-p0-r06p-submission-readiness@1.0.0",
        "overall_status": "pass",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "allowed_deck_differences": static_validation["permitted_differences"],
        "required_resources": {"nodes": 1, "ranks": 32},
        "archive_policy": "no_archive_direct_folder_delivery",
        "solver_status": "not_submitted_not_solver_validated",
    }
    _write_json(staging / "R06P_SUBMISSION_READINESS.json", readiness)
    finalize_staging_directory(staging, output_dir)
    return {"job_id": JOB_ID, "head_deck": HEAD_DECK, "head_deck_sha256": manifest["deck"]["sha256"], "output": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R06p 0.50 ms terminal-horizon batch")
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
