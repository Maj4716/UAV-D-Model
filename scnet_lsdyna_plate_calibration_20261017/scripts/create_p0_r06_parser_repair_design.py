from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from generate_p0_r06_diagnostics import strict_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT / "design" / "p0_r06_diagnostic_design.json"
DEFAULT_OUTPUT = ROOT / "design" / "p0_r06_keyword_parser_repair_design.json"
EXPECTED_PARENT_SHA256 = "9443fc7397c78db6d9fb7fd0b2b55ac185972f52fae5c90ea3641df8598da402"

JOB_RENAMES = {
    "r06d_0800_ctrl": ("r06e_0800_ctrl", "uav_hf_r06e_0800_c"),
    "r06d_0800_tend008": ("r06e_0800_tend008", "uav_hf_r06e_0800_t"),
    "r06d_2000_ctrl": ("r06e_2000_ctrl", "uav_hf_r06e_2000_c"),
    "r06d_2000_psfail_plate": (
        "r06e_2000_psfail_plate",
        "uav_hf_r06e_2000_p",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_design(parent_path: Path, output_path: Path) -> dict:
    parent_path = parent_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite parser-repair design: {output_path}")
    actual_parent_hash = sha256(parent_path)
    if actual_parent_hash != EXPECTED_PARENT_SHA256:
        raise ValueError(
            "frozen R06d parent design identity changed: "
            f"{actual_parent_hash} != {EXPECTED_PARENT_SHA256}"
        )

    design = copy.deepcopy(strict_json(parent_path))
    design["schema_version"] = "hf-p0-r06-diagnostic-design@1.1.0"
    design["package_revision"] = "R06_keyword_parser_repair_20261025"
    design["design_status"] = (
        "frozen_after_r06d_wave1_keyword_failure_before_any_repaired_solver_outcome"
    )
    design["title"] = (
        "SCNet P0 R06 keyword-parser repair and paired single-factor diagnostics "
        "at 800 and 2000 m/s, 0.75 mm"
    )
    design["created_date"] = "2026-10-25"
    design["purpose"] = (
        "Repair the R06d R15.0.2 database-card serialization without changing "
        "the frozen meshes, materials, contacts, controls, pair factors, or QoI; "
        "then run the original two paired diagnostics behind a one-job parser gate."
    )
    design["provenance"]["parent_r06d_design"] = str(
        parent_path.relative_to(ROOT).as_posix()
    )
    design["provenance"]["parent_r06d_design_sha256"] = (
        "sha256:" + actual_parent_hash
    )
    design["provenance"]["failed_controls"] = [
        {
            "job_id": "r06d_0800_ctrl",
            "platform_job_id": "72269229",
            "return_zip_sha256": "sha256:fe6718e5081580a71db144409f43f9d1fd866343cc244026655c0ec2922ff93f",
            "deck_sha256": "sha256:37f5f2ef4e6d0276e28d05aaf15c4114a4ae26e7655c11ba3ff1b2a8f8725daa",
        },
        {
            "job_id": "r06d_2000_ctrl",
            "platform_job_id": "72269336",
            "return_zip_sha256": "sha256:cd382fb6e569293b8a7e05462185c3817d22dd604b49da884c5a1e72800be72f",
            "deck_sha256": "sha256:3b61883283b0a68ede37ee40069af5e0ad85ce6fef57a43f572e56573d177ae2",
        },
    ]
    design["provenance"]["failure_class"] = (
        "target_R15_keyword_reader_before_time_integration"
    )
    design["provenance"]["repair_scope"] = (
        "shared_database_output_serialization_only"
    )

    measurement = design["shared_measurement_upgrade"]
    measurement["status"] = (
        "identical_parser_repair_within_and_across_pairs_not_a_physics_factor"
    )
    measurement["r15_parser_contract"] = {
        "target_solver": "LS-DYNA R15.0.2",
        "database_extent_binary_cards": [
            [6, 0, 3, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 2, 1],
        ],
        "intfor_file_timing_card_tail": [0, 0, 0],
        "intfor_file_ioopt_card": [0],
    }
    measurement["database_extent_binary_policy"] = (
        "Set NEIPH=6 and explicitly preserve all other R15.0.2 documented "
        "Card 1/Card 2 defaults; Card 2 is mandatory."
    )
    measurement["intfor"]["timing_card_field_count"] = 4
    measurement["intfor"]["repair_note"] = (
        "R06d wrote five timing-card fields and triggered KEY+457; R06e writes "
        "the four fields accepted by the returned R15.0.2 parser and retains "
        "IOOPT=0 on the next card."
    )

    for experiment in design["paired_experiments"]:
        for job in experiment["jobs"]:
            old_job_id = job["job_id"]
            new_job_id, platform_job_name = JOB_RENAMES[old_job_id]
            job["parent_r06d_job_id"] = old_job_id
            job["job_id"] = new_job_id
            job["platform_job_name"] = platform_job_name

    design["submission_sequence"] = {
        "wave_0": ["r06e_0800_ctrl"],
        "wave_0_gate": (
            "Target R15.0.2 reports zero KEY errors and zero excess-field "
            "warnings for the repaired cards, advances beyond cycle 0, "
            "terminates normally, and returns readable d3part and intfor families."
        ),
        "wave_1": ["r06e_2000_ctrl"],
        "wave_1_gate": (
            "Both repaired controls pass parser, normal-termination, run-integrity, "
            "and required-output readability gates before any treatment is submitted."
        ),
        "wave_2": ["r06e_0800_tend008", "r06e_2000_psfail_plate"],
        "wave_2_gate": (
            "Interpret each treatment only against its hash-bound repaired control; "
            "do not pool the two pairs as four calibration points."
        ),
        "resource_identity": (
            "Use exactly one node, 32 ranks, LS-DYNA R15 MPP double precision, "
            "and identical memory settings for all four jobs."
        ),
    }
    design["admission"]["r06_diagnostic_submission"] = (
        "allowed_only_in_frozen_wave_order_after_static_release_verifier_passes"
    )
    design["claim_boundary"].extend(
        [
            "The R06d failures establish a parser-contract defect only; they do not evaluate numerical or physical response.",
            "R06e changes only shared database-output serialization and explicit documented defaults; the paired physics factors remain unchanged.",
            "A local static PASS cannot promote Wave 0 without returned target-R15 parser and runtime evidence.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(design, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return design


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the non-overwriting R06e parser-repair design"
    )
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    design = build_design(args.parent, args.output)
    print(
        json.dumps(
            {
                "package_revision": design["package_revision"],
                "wave_0": design["submission_sequence"]["wave_0"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
