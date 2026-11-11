from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from build_r06g_tssfac02_return_audit import sha256, strict_json
from create_p0_r06t_fragment_only_psfail_design import (
    ASCII_INTERVAL_MS,
    BINARY_INTERVAL_MS,
    EXPERIMENT_ID,
    JOB_ID,
    MAXIMUM_CYCLE_COUNT,
    PARENT_DECK_SHA256,
    PARENT_JOB_ID,
    PARENT_PLATFORM_JOB_ID,
    PROBLEM_TIME_MS,
    REVISION,
    SPEED_MPS,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_DECK = ROOT.parent / "uav_hf_r06s_1200_speed_stability_diagnostic" / "r06s_1200_speed_stability_diagnostic.k"
DEFAULT_DESIGN = ROOT / "design" / "p0_r06t_fragment_only_psfail_design.json"
DEFAULT_CONTRACT = ROOT / "design" / "p0_r06t_fragment_only_psfail_output_contract.json"
DEFAULT_MATERIAL = ROOT / "materials" / "jc4340_reference.json"
DEFAULT_CASE_MAP = ROOT / "manual_upload_20261103_r06s_1200_speed_stability_diagnostic" / "job_01_v1200_speed_stability_diagnostic" / "p0_case_part_map.csv"
DEFAULT_OUTPUT = ROOT / "manual_upload_20261103_r06t_fragment_only_psfail_diagnostic"
JOB_DIR = "job_01_v1200_fragment_only_psfail_diagnostic"
HEAD_DECK = "r06t_1200_fragment_only_psfail_diagnostic.k"
PARENT_TITLE = "SCNet P0 R06s 1200mps stability diagnostic 20261103"
NEW_TITLE = "SCNet P0 R06t fragment-only PSFAIL diagnostic 20261103"
PARENT_CONTROL_SOLID = "0,0,0,0,0,0.000000000e+00,0,0"
CANDIDATE_CONTROL_SOLID = "0,0,0,0,900001,0.000000000e+00,0,0"
EXPECTED_DIFFERING_LINES = [3, 12]


def transform_deck(parent_text: str) -> str:
    lines = parent_text.splitlines(keepends=True)
    replacements = {3: (PARENT_TITLE, NEW_TITLE), 12: (PARENT_CONTROL_SOLID, CANDIDATE_CONTROL_SOLID)}
    for line_number, (old, new) in replacements.items():
        index = line_number - 1
        actual = lines[index].rstrip("\r\n")
        if actual != old:
            raise ValueError(f"parent line {line_number} differs: {actual!r}")
        lines[index] = new + lines[index][len(old):]
    return "".join(lines)


def validate_transformed_deck(parent_text: str, candidate_text: str) -> dict:
    parent_lines = parent_text.splitlines()
    candidate_lines = candidate_text.splitlines()
    if len(parent_lines) != len(candidate_lines):
        raise ValueError("candidate line count differs")
    differences = [i for i, pair in enumerate(zip(parent_lines, candidate_lines), start=1) if pair[0] != pair[1]]
    if differences != EXPECTED_DIFFERING_LINES:
        raise ValueError(f"candidate differs outside the two registered lines: {differences[:30]}")
    restored = list(candidate_lines)
    for line_number in EXPECTED_DIFFERING_LINES:
        restored[line_number - 1] = parent_lines[line_number - 1]
    if restored != parent_lines:
        raise ValueError("restoring the registered lines did not recover exact R06s")
    required_tokens = [
        CANDIDATE_CONTROL_SOLID,
        "R06 current fragment only for high-rate D3PART\n900001,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,MECH,,\n2002",
        "R06 current plate only for PSFAIL treatment\n910001,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,MECH,,\n1002",
        "2002,0,2,2003,2,0,0,",
        "1.500000000e-04,0,0,0,900001",
        "2002,2,0.000000000e+00,0.000000000e+00,0.000000000e+00,-1.200000000e+03,0,0",
    ]
    normalized = candidate_text.replace("\r\n", "\n")
    missing = [token for token in required_tokens if token not in normalized]
    if missing:
        raise ValueError(f"candidate is missing frozen or set-semantic tokens: {missing}")
    if candidate_lines[11] != CANDIDATE_CONTROL_SOLID:
        raise ValueError("R06t PSFAIL must be fragment-only set 900001")
    if any(candidate_lines[line_number - 1] != "2.500000000e-04" for line_number in (29, 31, 33, 35, 37)):
        raise ValueError("five ASCII output intervals must remain 0.00025 ms")
    return {
        "schema_version": "hf-p0-r06t-static-validation@1.0.0",
        "overall_status": "pass",
        "parent_deck_sha256": PARENT_DECK_SHA256,
        "permitted_differences": ["TITLE_identifier_line", "CONTROL_SOLID_PSFAIL_0_to_fragment_only_set_900001"],
        "differing_line_numbers": differences,
        "all_other_deck_lines_byte_identical": True,
        "single_model_form_factor": "CONTROL_SOLID_PSFAIL_0_to_fragment_only_set_900001",
        "psfail_set_id": 900001,
        "psfail_set_part_ids": [2002],
        "plate_part_1002_excluded_from_psfail_set": True,
        "termination_time_ms": PROBLEM_TIME_MS,
        "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
        "impact_speed_mps": SPEED_MPS,
        "d3plot_interval_ms": BINARY_INTERVAL_MS,
        "d3part_interval_ms": BINARY_INTERVAL_MS,
        "ascii_interval_ms": ASCII_INTERVAL_MS,
        "adaptive_solid_to_sph": "part_2002_to_part_2003_unchanged",
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
        raise ValueError("R06t case-map source must contain exactly one identified case")
    row = dict(rows[0])
    updates = {
        "job_id": JOB_ID,
        "experiment_id": EXPERIMENT_ID,
        "role": "plate_negative_volume_psfail_scope_model_form_diagnostic",
        "case_id": "p0_v1200_h0p75_psfail_fragment_only",
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
        raise ValueError("immutable R06s parent deck hash differs")
    design = strict_json(design_path.resolve())
    contract = strict_json(contract_path.resolve())
    if design["package_revision"] != REVISION or design["jobs"][0]["job_id"] != JOB_ID:
        raise ValueError("R06t design identity differs")
    if contract["version_boundary"]["applies_to_package_revision"] != REVISION:
        raise ValueError("R06t contract identity differs")

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
        "p0_r06t_fragment_only_psfail_design.json": design_path.resolve(),
        "p0_r06t_fragment_only_psfail_output_contract.json": contract_path.resolve(),
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
        "role": "plate_negative_volume_psfail_scope_model_form_diagnostic",
        "factor": "CONTROL_SOLID.PSFAIL",
        "factor_level": "fragment_only_set_900001",
        "factor_layer": "single_model_form_mechanism_diagnostic",
        "case": {
            "case_id": "p0_v1200_h0p75_psfail_fragment_only",
            "impact_speed_mps": SPEED_MPS,
            "mesh_mm": 0.75,
            "plate_part_id": 1002,
            "fragment_solid_part_id": 2002,
            "adaptive_sph_part_id": 2003,
            "termination_time_ms": PROBLEM_TIME_MS,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "control_solid_psfail": 900001,
            "psfail_set_part_ids": [2002],
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
            "stage1: submitted .k, manifest, job_contract, all mes*, glstat, stdout/stderr, Slurm and resource log",
            "stage2 only after stage1 pass: all d3plot*",
            "stage3 only after stage2 pass: all d3part* and binout*",
        ],
        "status": "prepared_not_submitted_not_solver_validated",
    }
    _write_json(job / "job_contract.json", job_contract)
    manifest = {
        "schema_version": "hf-p0-r06t-job-manifest@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "generated_not_solver_validated",
        "evidence_class": "high_fidelity_synthetic_model_form_mechanism_diagnostic_candidate",
        "deck": {"path": HEAD_DECK, "size_bytes": deck_path.stat().st_size, "sha256": sha256(deck_path), "differences_from_parent": static["permitted_differences"]},
        "immutable_parent": {
            "job_id": PARENT_JOB_ID,
            "platform_job_id": PARENT_PLATFORM_JOB_ID,
            "deck_sha256": PARENT_DECK_SHA256,
            "classification": "completed_energy_pass_but_plate_negative_volume_persists",
        },
        "design": {"path": "p0_r06t_fragment_only_psfail_design.json", "sha256": sha256(job / "p0_r06t_fragment_only_psfail_design.json")},
        "successor_contract": {"path": "p0_r06t_fragment_only_psfail_output_contract.json", "sha256": sha256(job / "p0_r06t_fragment_only_psfail_output_contract.json")},
        "material": {"path": "jc4340_reference.json", "sha256": sha256(job / "jc4340_reference.json")},
        "case_map": {"path": "p0_case_part_map.csv", "sha256": sha256(job / "p0_case_part_map.csv")},
        "job_contract": {"path": "job_contract.json", "sha256": sha256(job / "job_contract.json")},
        "limitations": [
            "Static validation proves only the two registered deck-line changes; target-platform execution is untested.",
            "R06t isolates PSFAIL scope and cannot by itself establish a physically valid plate-failure model.",
            "Binary result families must not be downloaded unless the staged ASCII gate passes.",
        ],
    }
    _write_json(job / "manifest.json", manifest)
    (job / "UPLOAD_README.txt").write_text(
        f"""R06t fragment-only PSFAIL model-form diagnostic\n\nSelect {HEAD_DECK} as the head deck.\nRequired resources: 1 node, exactly 32 MPP ranks, ls-dyna_R15_mmp_d.\nOnly model-form factor: CONTROL_SOLID.PSFAIL changes from 0 to existing fragment-only set 900001 (part 2002).\nKeep speed 1200 m/s, 0.50 ms, ENDCYC={MAXIMUM_CYCLE_COUNT}, D3PLOT/D3PART 0.00015 ms, ASCII 0.00025 ms, TSSFAC=0.05, DT2MS=0 and every other control unchanged.\nAfter completion, download only the Stage-1 small-file set first. Do not download d3plot/d3part/binout until Stage 1 passes.\nThis is a model-form mechanism diagnostic, not physical, engine or damage-tree calibration.\n""",
        encoding="utf-8",
        newline="\n",
    )
    job_files = sorted(path for path in job.iterdir() if path.is_file())
    _write_json(staging / "batch_manifest.json", {
        "schema_version": "hf-p0-r06t-batch-manifest@1.0.0",
        "package_revision": REVISION,
        "job_count": 1,
        "jobs": [{"job_id": JOB_ID, "directory": JOB_DIR, "head_deck": HEAD_DECK, "head_deck_sha256": manifest["deck"]["sha256"]}],
        "files": [{"path": path.relative_to(staging).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in job_files],
    })
    _write_json(staging / "R06T_SUBMISSION_READINESS.json", {
        "schema_version": "hf-p0-r06t-submission-readiness@1.0.0",
        "overall_status": "pass",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "allowed_deck_differences": static["permitted_differences"],
        "required_resources": {"nodes": 1, "ranks": 32},
        "archive_policy": "no_archive_direct_folder_delivery",
        "solver_status": "not_submitted_not_solver_validated",
    })
    finalize_staging_directory(staging, output_dir)
    return {"job_id": JOB_ID, "head_deck": HEAD_DECK, "head_deck_sha256": manifest["deck"]["sha256"], "output": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R06t fragment-only PSFAIL diagnostic batch")
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
