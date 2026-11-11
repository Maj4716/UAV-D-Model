from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from build_r06g_tssfac02_return_audit import sha256, strict_json
from create_p0_r06n_adaptive_sph_32rank_confirmation_design import (
    EXPERIMENT_ID,
    JOB_ID,
    MAXIMUM_CYCLE_COUNT,
    PARENT_DECK_SHA256,
    PARENT_JOB_ID,
    PARENT_PLATFORM_JOB_ID,
    REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_DECK = (
    ROOT
    / "analysis"
    / "72861050_r06m_0800_adaptive_sph_canary_20261101"
    / "immutable_parent_r06m_0800_adaptive_sph_canary.k"
)
DEFAULT_DESIGN = ROOT / "design" / "p0_r06n_adaptive_sph_32rank_confirmation_design.json"
DEFAULT_CONTRACT = ROOT / "design" / "p0_r06n_adaptive_sph_32rank_output_contract.json"
DEFAULT_MATERIAL = ROOT / "materials" / "jc4340_reference.json"
DEFAULT_CASE_MAP = (
    ROOT
    / "manual_upload_20261101_r06m_adaptive_sph_canary"
    / "job_01_v0800_adaptive_sph_canary"
    / "p0_case_part_map.csv"
)
DEFAULT_OUTPUT = ROOT / "manual_upload_20261101_r06n_adaptive_sph_32rank_confirmation"
JOB_DIR = "job_01_v0800_adaptive_sph_32rank_confirmation"
HEAD_DECK = "r06n_0800_adaptive_sph_32rank_confirmation.k"
PARENT_TITLE = "SCNet P0 R06_adaptive_solid_to_sph_canary_20261101 r06m_0800_adaptive_sph_canary"
NEW_TITLE = "SCNet P0 R06n adaptive SPH exact 32rank confirmation 20261101"


def transform_deck(parent_text: str) -> str:
    lines = parent_text.splitlines(keepends=True)
    title_keyword_indexes = [i for i, line in enumerate(lines) if line.strip().upper() == "*TITLE"]
    if title_keyword_indexes != [1] or len(lines) < 3:
        raise ValueError("parent deck has an unexpected TITLE layout")
    original = lines[2].rstrip("\r\n")
    if original != PARENT_TITLE:
        raise ValueError(f"parent title differs: {original!r}")
    newline = lines[2][len(original):]
    lines[2] = NEW_TITLE + newline
    return "".join(lines)


def validate_transformed_deck(parent_text: str, candidate_text: str) -> dict:
    parent_lines = parent_text.splitlines()
    candidate_lines = candidate_text.splitlines()
    if len(parent_lines) != len(candidate_lines):
        raise ValueError("candidate line count differs from parent")
    differences = [i for i, pair in enumerate(zip(parent_lines, candidate_lines), start=1) if pair[0] != pair[1]]
    if differences != [3]:
        raise ValueError(f"candidate must differ only at TITLE line 3, got {differences[:20]}")
    if parent_lines[2] != PARENT_TITLE or candidate_lines[2] != NEW_TITLE:
        raise ValueError("TITLE identity differs from the frozen transformation")
    restored = list(candidate_lines)
    restored[2] = PARENT_TITLE
    if restored != parent_lines:
        raise ValueError("restoring TITLE did not recover exact parent deck")
    required_tokens = [
        "*DEFINE_ADAPTIVE_SOLID_TO_SPH",
        "*SECTION_SPH",
        "*CONTACT_ERODING_SINGLE_SURFACE",
        "*CONTACT_AUTOMATIC_NODES_TO_SURFACE",
        "4.000000000e-02,829930",
        "0.000000000e+00,5.000000000e-02,0,0.000000000e+00",
    ]
    missing = [token for token in required_tokens if token not in candidate_text]
    if missing:
        raise ValueError(f"candidate is missing frozen model tokens: {missing}")
    return {
        "schema_version": "hf-p0-r06n-static-validation@1.0.0",
        "overall_status": "pass",
        "parent_deck_sha256": PARENT_DECK_SHA256,
        "permitted_difference": "TITLE_identifier_line_only",
        "differing_line_numbers": differences,
        "parent_title": PARENT_TITLE,
        "candidate_title": NEW_TITLE,
        "solver_model_output_cards_byte_identical": True,
        "termination_time_ms": 0.04,
        "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
        "tssfac": 0.05,
        "dt2ms": 0.0,
        "adaptive_sph_part_id": 2003,
        "particles_per_failed_hexahedron": 8,
        "resource_identity_is_external_to_deck": True,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


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
        raise ValueError("immutable R06m parent deck hash differs")
    design = strict_json(design_path.resolve())
    contract = strict_json(contract_path.resolve())
    if design["package_revision"] != REVISION or design["jobs"][0]["job_id"] != JOB_ID:
        raise ValueError("R06n design identity differs")
    if contract["version_boundary"]["applies_to_package_revision"] != REVISION:
        raise ValueError("R06n contract identity differs")

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
        "p0_r06n_adaptive_sph_32rank_confirmation_design.json": design_path.resolve(),
        "p0_r06n_adaptive_sph_32rank_output_contract.json": contract_path.resolve(),
        "jc4340_reference.json": material_path.resolve(),
        "p0_case_part_map.csv": case_map_path.resolve(),
    }
    for name, source in copied.items():
        shutil.copy2(source, job / name)

    job_contract = {
        "schema_version": "hf-p0-r06-job-contract@1.1.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "experiment_id": EXPERIMENT_ID,
        "role": "resource_identity_and_engineering_repeatability_confirmation",
        "factor": "actual_mpp_rank_count",
        "factor_level": "one_node_exactly_32_ranks",
        "case": {
            "case_id": "p0_v0800_h0p75",
            "impact_speed_mps": 800.0,
            "mesh_mm": 0.75,
            "plate_part_id": 1002,
            "fragment_solid_part_id": 2002,
            "adaptive_sph_part_id": 2003,
            "termination_time_ms": 0.04,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
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
        "schema_version": "hf-p0-r06n-job-manifest@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "generated_not_solver_validated",
        "evidence_class": "high_fidelity_synthetic_resource_confirmation_candidate",
        "deck": {
            "path": HEAD_DECK,
            "size_bytes": deck_path.stat().st_size,
            "sha256": sha256(deck_path),
            "difference_from_parent": "TITLE_identifier_line_only",
        },
        "immutable_parent": {
            "job_id": PARENT_JOB_ID,
            "platform_job_id": PARENT_PLATFORM_JOB_ID,
            "deck_sha256": PARENT_DECK_SHA256,
            "observed_nodes": 1,
            "observed_ranks": 64,
            "resource_contract_status": "failed_expected_32_ranks",
        },
        "design": {"path": list(copied)[0], "sha256": sha256(job / list(copied)[0])},
        "successor_contract": {"path": list(copied)[1], "sha256": sha256(job / list(copied)[1])},
        "material": {"path": "jc4340_reference.json", "sha256": sha256(job / "jc4340_reference.json")},
        "case_map": {"path": "p0_case_part_map.csv", "sha256": sha256(job / "p0_case_part_map.csv")},
        "job_contract": {"path": "job_contract.json", "sha256": sha256(job / "job_contract.json")},
        "limitations": [
            "Static validation proves a TITLE-only deck change; it does not prove target-platform execution.",
            "The platform must actually allocate one node and exactly 32 MPP ranks.",
            "The 0.04 ms horizon remains intentionally unchanged and does not yet qualify the capture-plane response.",
            "No response is admitted for engine fitting, damage-tree fitting or reference labels.",
        ],
    }
    _write_json(job / "manifest.json", manifest)
    readme = f"""R06n exact 32-rank adaptive-SPH confirmation\n\nSelect {HEAD_DECK} as the head deck.\nRequired resources: 1 node, exactly 32 MPP ranks, ls-dyna_R15_mmp_d.\nDo not edit the deck. The only deck difference from returned R06m is the TITLE identifier line.\nStart state is already the post-explosion state; no pre-detonation flight is simulated.\nReturn the complete output folder even if the job stops early.\nThis is engineering confirmation, not physical or damage-tree calibration.\n"""
    (job / "UPLOAD_README.txt").write_text(readme, encoding="utf-8", newline="\n")

    job_files = sorted(path for path in job.iterdir() if path.is_file())
    batch_manifest = {
        "schema_version": "hf-p0-r06n-batch-manifest@1.0.0",
        "package_revision": REVISION,
        "job_count": 1,
        "jobs": [{"job_id": JOB_ID, "directory": JOB_DIR, "head_deck": HEAD_DECK, "head_deck_sha256": manifest["deck"]["sha256"]}],
        "files": [{"path": path.relative_to(staging).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in job_files],
    }
    _write_json(staging / "batch_manifest.json", batch_manifest)
    readiness = {
        "schema_version": "hf-p0-r06n-submission-readiness@1.0.0",
        "overall_status": "pass",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "title_only_deck_change": True,
        "required_resources": {"nodes": 1, "ranks": 32},
        "archive_policy": "no_archive_direct_folder_delivery",
        "solver_status": "not_submitted_not_solver_validated",
    }
    _write_json(staging / "R06N_SUBMISSION_READINESS.json", readiness)
    finalize_staging_directory(staging, output_dir)
    return {"job_id": JOB_ID, "head_deck": HEAD_DECK, "head_deck_sha256": manifest["deck"]["sha256"], "output": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R06n exact 32-rank confirmation batch")
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
