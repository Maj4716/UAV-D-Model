from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from build_r06g_tssfac02_return_audit import sha256, strict_json
from create_p0_r06s_1200_speed_stability_design import (
    ASCII_INTERVAL_MS,
    CANDIDATE_BINARY_INTERVAL_MS,
    CANDIDATE_SPEED_MPS,
    EXPERIMENT_ID,
    JOB_ID,
    MAXIMUM_CYCLE_COUNT,
    PARENT_DECK_SHA256,
    PARENT_JOB_ID,
    PARENT_PLATFORM_JOB_ID,
    PROBLEM_TIME_MS,
    REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_DECK = ROOT.parent / "uav_hf_r06r_2000_speed_upper_anchor" / "r06r_2000_speed_upper_anchor.k"
DEFAULT_DESIGN = ROOT / "design" / "p0_r06s_1200_speed_stability_design.json"
DEFAULT_CONTRACT = ROOT / "design" / "p0_r06s_1200_speed_stability_output_contract.json"
DEFAULT_MATERIAL = ROOT / "materials" / "jc4340_reference.json"
DEFAULT_CASE_MAP = ROOT / "manual_upload_20261102_r06r_2000_speed_upper_anchor" / "job_01_v2000_speed_upper_anchor" / "p0_case_part_map.csv"
DEFAULT_OUTPUT = ROOT / "manual_upload_20261103_r06s_1200_speed_stability_diagnostic"
JOB_DIR = "job_01_v1200_speed_stability_diagnostic"
HEAD_DECK = "r06s_1200_speed_stability_diagnostic.k"
PARENT_TITLE = "SCNet P0 R06r 2000mps upper anchor 20261102"
NEW_TITLE = "SCNet P0 R06s 1200mps stability diagnostic 20261103"
PARENT_D3PLOT = "1.000000000e-04"
CANDIDATE_D3PLOT = "1.500000000e-04"
PARENT_D3PART = "1.000000000e-04,0,0,0,900001"
CANDIDATE_D3PART = "1.500000000e-04,0,0,0,900001"
PARENT_VELOCITY = "2002,2,0.000000000e+00,0.000000000e+00,0.000000000e+00,-2.000000000e+03,0,0"
CANDIDATE_VELOCITY = "2002,2,0.000000000e+00,0.000000000e+00,0.000000000e+00,-1.200000000e+03,0,0"
EXPECTED_DIFFERING_LINES = [3, 19, 24, 45102]


def transform_deck(parent_text: str) -> str:
    lines = parent_text.splitlines(keepends=True)
    replacements = {
        3: (PARENT_TITLE, NEW_TITLE),
        19: (PARENT_D3PLOT, CANDIDATE_D3PLOT),
        24: (PARENT_D3PART, CANDIDATE_D3PART),
        45102: (PARENT_VELOCITY, CANDIDATE_VELOCITY),
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
        raise ValueError(f"candidate differs outside the four registered lines: {differences[:30]}")
    restored = list(candidate_lines)
    for line_number in EXPECTED_DIFFERING_LINES:
        restored[line_number - 1] = parent_lines[line_number - 1]
    if restored != parent_lines:
        raise ValueError("restoring the registered lines did not recover exact R06r")
    required_tokens = [
        "5.000000000e-01,2500000,0.000000000e+00,0.000000000e+00,0.000000000e+00",
        "0.000000000e+00,5.000000000e-02,0,0.000000000e+00",
        "*DEFINE_ADAPTIVE_SOLID_TO_SPH",
        "2002,0,2,2003,2,0,0,",
        "*CONTACT_ERODING_SINGLE_SURFACE",
        "*CONTACT_AUTOMATIC_NODES_TO_SURFACE",
        "6.100000000e-01,0.000000000e+00,1,1.000000000e-06,0",
        CANDIDATE_D3PLOT,
        CANDIDATE_D3PART,
        CANDIDATE_VELOCITY,
    ]
    missing = [token for token in required_tokens if token not in candidate_text]
    if missing:
        raise ValueError(f"candidate is missing frozen tokens: {missing}")
    if candidate_text.count("6.100000000e-01,0.000000000e+00,1,1.000000000e-06,0") != 2:
        raise ValueError("plate and fragment must both retain EROD=1")
    if candidate_lines[18] != CANDIDATE_D3PLOT or candidate_lines[23] != CANDIDATE_D3PART:
        raise ValueError("R06s binary output intervals differ from 0.00015 ms")
    if any(candidate_lines[line_number - 1] != "2.500000000e-04" for line_number in (29, 31, 33, 35, 37)):
        raise ValueError("five ASCII output intervals must remain 0.00025 ms")
    return {
        "schema_version": "hf-p0-r06s-static-validation@1.0.0",
        "overall_status": "pass",
        "parent_deck_sha256": PARENT_DECK_SHA256,
        "permitted_differences": [
            "TITLE_identifier_line",
            "DATABASE_BINARY_D3PLOT_dt_0p0001_to_0p00015",
            "DATABASE_BINARY_D3PART_dt_0p0001_to_0p00015",
            "INITIAL_VELOCITY_GENERATION_part_2002_vz_minus2000_to_minus1200",
        ],
        "differing_line_numbers": differences,
        "all_other_deck_lines_byte_identical": True,
        "single_physical_factor": "fragment_initial_speed_mps_2000_to_1200",
        "measurement_control_changes": ["d3plot_interval_ms", "d3part_interval_ms"],
        "termination_time_ms": PROBLEM_TIME_MS,
        "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
        "impact_speed_mps": CANDIDATE_SPEED_MPS,
        "d3plot_interval_ms": CANDIDATE_BINARY_INTERVAL_MS,
        "d3part_interval_ms": CANDIDATE_BINARY_INTERVAL_MS,
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
        raise ValueError("R06s case-map source must contain exactly one identified case")
    row = dict(rows[0])
    updates = {
        "job_id": JOB_ID,
        "experiment_id": EXPERIMENT_ID,
        "role": "numerical_speed_stability_diagnostic",
        "case_id": "p0_v1200_h0p75",
        "impact_speed_mps": "1200.0",
        "termination_time_ms": "0.5",
        "maximum_cycle_count": str(MAXIMUM_CYCLE_COUNT),
        "primary_measurement_source": "staged_ascii_gate_then_combined_solid_sph_crossing_ledger",
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
    if sha256(parent_deck) != PARENT_DECK_SHA256:
        raise ValueError("immutable R06r parent deck hash differs")
    design = strict_json(design_path.resolve())
    contract = strict_json(contract_path.resolve())
    if design["package_revision"] != REVISION or design["jobs"][0]["job_id"] != JOB_ID:
        raise ValueError("R06s design identity differs")
    if contract["version_boundary"]["applies_to_package_revision"] != REVISION:
        raise ValueError("R06s contract identity differs")

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
        "p0_r06s_1200_speed_stability_design.json": design_path.resolve(),
        "p0_r06s_1200_speed_stability_output_contract.json": contract_path.resolve(),
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
        "role": "numerical_speed_stability_diagnostic",
        "factor": "fragment_initial_speed_mps",
        "factor_level": "1200_mps",
        "factor_layer": "physical_single_factor_diagnostic",
        "case": {
            "case_id": "p0_v1200_h0p75",
            "impact_speed_mps": CANDIDATE_SPEED_MPS,
            "mesh_mm": 0.75,
            "plate_part_id": 1002,
            "fragment_solid_part_id": 2002,
            "adaptive_sph_part_id": 2003,
            "capture_plane_z_mm": -12.0,
            "termination_time_ms": PROBLEM_TIME_MS,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "plate_mat015_erod": 1,
            "fragment_mat015_erod": 1,
        },
        "output_intervals_ms": {"d3plot": CANDIDATE_BINARY_INTERVAL_MS, "d3part": CANDIDATE_BINARY_INTERVAL_MS, "ascii_databases": ASCII_INTERVAL_MS},
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
            "stage1: submitted .k, manifest, job_contract, all mes*, glstat, stdout/stderr, Slurm and resource log",
            "stage2 only after stage1 pass: all d3plot*",
            "stage3 only after stage2 pass: all d3part* and binout*",
        ],
        "completion_rule": {
            "required_problem_time_ms": PROBLEM_TIME_MS,
            "cycle_guard": MAXIMUM_CYCLE_COUNT,
            "required_actual_nodes": 1,
            "required_actual_ranks": 32,
        },
        "status": "prepared_not_submitted_not_solver_validated",
    }
    _write_json(job / "job_contract.json", job_contract)
    manifest = {
        "schema_version": "hf-p0-r06s-job-manifest@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "generated_not_solver_validated",
        "evidence_class": "high_fidelity_synthetic_numerical_speed_stability_diagnostic_candidate",
        "deck": {"path": HEAD_DECK, "size_bytes": deck_path.stat().st_size, "sha256": sha256(deck_path), "differences_from_parent": static_validation["permitted_differences"]},
        "immutable_parent": {
            "job_id": PARENT_JOB_ID,
            "platform_job_id": PARENT_PLATFORM_JOB_ID,
            "deck_sha256": PARENT_DECK_SHA256,
            "classification": "completed_but_engineering_failed_plate_negative_volume_and_sliding_energy",
        },
        "design": {"path": "p0_r06s_1200_speed_stability_design.json", "sha256": sha256(job / "p0_r06s_1200_speed_stability_design.json")},
        "successor_contract": {"path": "p0_r06s_1200_speed_stability_output_contract.json", "sha256": sha256(job / "p0_r06s_1200_speed_stability_output_contract.json")},
        "material": {"path": "jc4340_reference.json", "sha256": sha256(job / "jc4340_reference.json")},
        "case_map": {"path": "p0_case_part_map.csv", "sha256": sha256(job / "p0_case_part_map.csv")},
        "job_contract": {"path": "job_contract.json", "sha256": sha256(job / "job_contract.json")},
        "limitations": [
            "Static validation proves only the four registered deck-line changes; target-platform execution is untested.",
            "R06s is a stability diagnostic after a failed 2000 m/s run, not an admitted midpoint response.",
            "Binary result families must not be downloaded unless the staged ASCII gate passes.",
        ],
    }
    _write_json(job / "manifest.json", manifest)
    (job / "UPLOAD_README.txt").write_text(
        f"""R06s 1200 m/s numerical stability diagnostic\n\nSelect {HEAD_DECK} as the head deck.\nRequired resources: 1 node, exactly 32 MPP ranks, ls-dyna_R15_mmp_d.\nSingle physical factor: part 2002 initial speed changes from 2000 to 1200 m/s.\nD3PLOT/D3PART are measurement controls at 0.00015 ms; ASCII stays 0.00025 ms.\nKeep 0.50 ms, ENDCYC={MAXIMUM_CYCLE_COUNT}, TSSFAC=0.05, DT2MS=0 and all physical controls unchanged.\nAfter completion, download only the Stage-1 small-file set first. Do not download d3plot/d3part/binout until Stage 1 passes.\nThis is a numerical stability diagnostic, not physical, engine or damage-tree calibration.\n""",
        encoding="utf-8",
        newline="\n",
    )

    job_files = sorted(path for path in job.iterdir() if path.is_file())
    _write_json(staging / "batch_manifest.json", {
        "schema_version": "hf-p0-r06s-batch-manifest@1.0.0",
        "package_revision": REVISION,
        "job_count": 1,
        "jobs": [{"job_id": JOB_ID, "directory": JOB_DIR, "head_deck": HEAD_DECK, "head_deck_sha256": manifest["deck"]["sha256"]}],
        "files": [{"path": path.relative_to(staging).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in job_files],
    })
    _write_json(staging / "R06S_SUBMISSION_READINESS.json", {
        "schema_version": "hf-p0-r06s-submission-readiness@1.0.0",
        "overall_status": "pass",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "allowed_deck_differences": static_validation["permitted_differences"],
        "required_resources": {"nodes": 1, "ranks": 32},
        "archive_policy": "no_archive_direct_folder_delivery",
        "solver_status": "not_submitted_not_solver_validated",
    })
    finalize_staging_directory(staging, output_dir)
    return {"job_id": JOB_ID, "head_deck": HEAD_DECK, "head_deck_sha256": manifest["deck"]["sha256"], "output": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R06s 1200 m/s stability-diagnostic batch")
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
