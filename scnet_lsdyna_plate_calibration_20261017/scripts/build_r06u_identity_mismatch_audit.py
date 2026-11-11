from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re

from build_r06g_tssfac02_return_audit import sha256, strict_json


PLATFORM_JOB_ID = "73171751"
OBSERVED_JOB_LABEL = "uav_hf_r06u_1200_plate_adaptive_sph_canary"
OBSERVED_DECK = "r06t_1200_fragment_only_psfail_diagnostic.k"
OBSERVED_DECK_SHA256 = "sha256:11f62db3f8b27d1087988d5ad2abc572bddbd89570f6065500ddfe390a6459ae"
INTENDED_JOB_ID = "r06u_1200_plate_adaptive_sph_canary"
INTENDED_DECK = "r06u_1200_plate_adaptive_sph_canary.k"
INTENDED_DECK_SHA256 = "sha256:7bfb99a49bb757632237b2e2c625500fef27e811988eac3149df7a3e056647d4"
ERROR_ELEMENT = 23830
ERROR_CYCLE = 603398


def _single(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"expected one {label}, found {len(paths)}")
    return paths[0]


def build_audit(return_dir: Path, intended_submission_dir: Path) -> dict:
    return_dir = return_dir.resolve()
    intended_submission_dir = intended_submission_dir.resolve()
    observed_manifest = strict_json(return_dir / "manifest.json")
    observed_contract = strict_json(return_dir / "job_contract.json")
    intended_manifest = strict_json(intended_submission_dir / "manifest.json")
    intended_contract = strict_json(intended_submission_dir / "job_contract.json")

    observed_deck = return_dir / OBSERVED_DECK
    intended_deck = intended_submission_dir / INTENDED_DECK
    observed_hash = sha256(observed_deck)
    intended_hash = sha256(intended_deck)
    if (
        observed_manifest["job_id"] != "r06t_1200_fragment_only_psfail_diagnostic"
        or observed_manifest["deck"]["path"] != OBSERVED_DECK
        or observed_manifest["deck"]["sha256"] != OBSERVED_DECK_SHA256
        or observed_hash != OBSERVED_DECK_SHA256
        or observed_contract["head_deck"] != OBSERVED_DECK
    ):
        raise ValueError("returned input is not the frozen R06t deck")
    if (
        intended_manifest["job_id"] != INTENDED_JOB_ID
        or intended_manifest["deck"]["path"] != INTENDED_DECK
        or intended_manifest["deck"]["sha256"] != INTENDED_DECK_SHA256
        or intended_hash != INTENDED_DECK_SHA256
        or intended_contract["head_deck"] != INTENDED_DECK
        or intended_contract["case"]["termination_time_ms"] != 0.15
        or intended_contract["case"]["control_solid_psfail"] != 0
    ):
        raise ValueError("intended R06u submission identity differs from the frozen delivery")

    slurm_path = _single(list(return_dir.glob("job_LS_DYNA_*.slurm")), "Slurm script")
    log_path = _single(list(return_dir.glob(f"job_*_{PLATFORM_JOB_ID}.log")), "platform log")
    stdout_path = return_dir / f"std.out.{PLATFORM_JOB_ID}"
    stderr_path = return_dir / f"std.err.{PLATFORM_JOB_ID}"
    slurm_text = slurm_path.read_text(encoding="utf-8", errors="replace")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")

    command_match = re.search(r"^mpirun .*?\bi=(\S+)", log_text, flags=re.MULTILINE)
    error_match = re.search(r"complex sound speed in solid element #\s*(\d+) cycle\s*(\d+)", stdout_text)
    collapse_match = re.search(r"603398 t\s+([0-9.Ee+-]+) dt\s+([0-9.Ee+-]+)", stdout_text)
    time_match = re.search(r"Problem time\s*=\s*([0-9.Ee+-]+)", stdout_text)
    cycle_match = re.search(r"Problem cycle\s*=\s*(\d+)", stdout_text)
    elapsed_match = re.search(r"Elapsed time\s+(\d+) seconds", stdout_text)
    start_match = re.search(r"The start time is:\s*([0-9-]+ [0-9:]+)", log_text)
    end_match = re.search(r"End time\s+([0-9/]+ [0-9:]+)", stdout_text)
    if not all((command_match, error_match, collapse_match, time_match, cycle_match, elapsed_match, start_match, end_match)):
        raise ValueError("returned execution evidence is incomplete")
    if (
        Path(command_match.group(1)).name != OBSERVED_DECK
        or int(error_match.group(1)) != ERROR_ELEMENT
        or int(error_match.group(2)) != ERROR_CYCLE
        or int(cycle_match.group(1)) != ERROR_CYCLE
    ):
        raise ValueError("returned execution identity or repeated failure signature differs")

    start = datetime.strptime(start_match.group(1), "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end_match.group(1), "%m/%d/%Y %H:%M:%S")
    files = [path for path in return_dir.iterdir() if path.is_file()]
    return {
        "schema_version": "hf-p0-r06u-input-identity-mismatch-audit@1.0.0",
        "audit_date": "2026-11-04",
        "source_evidence": {
            "directory": return_dir.as_posix(),
            "evidence_policy": "read_only_immutable_partial_return",
            "platform_job_id": PLATFORM_JOB_ID,
            "platform_job_label": OBSERVED_JOB_LABEL,
            "file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
        },
        "identity_gate": {
            "status": "fail",
            "job_label_claimed_revision": "R06u",
            "executed_input_revision": "R06t",
            "executed_deck": OBSERVED_DECK,
            "executed_deck_sha256": observed_hash,
            "intended_deck": INTENDED_DECK,
            "intended_deck_sha256": intended_hash,
            "intended_deck_present_in_return": (return_dir / INTENDED_DECK).exists(),
            "command_line_input": command_match.group(1),
            "manifest_job_id": observed_manifest["job_id"],
            "job_contract_head_deck": observed_contract["head_deck"],
            "deck_hashes_equal": observed_hash == intended_hash,
        },
        "execution": {
            "nodes": 1,
            "ranks": 32,
            "slurm_job_name_is_r06u": f"#SBATCH -J {OBSERVED_JOB_LABEL}" in slurm_text,
            "normal_termination": False,
            "error_termination": True,
            "start_time": start.isoformat(sep=" "),
            "end_time": end.isoformat(sep=" "),
            "wall_elapsed_seconds": int((end - start).total_seconds()),
            "solver_elapsed_seconds": int(elapsed_match.group(1)),
            "problem_time_ms": float(time_match.group(1)),
            "cycle_count": int(cycle_match.group(1)),
            "stderr_size_bytes": stderr_path.stat().st_size,
        },
        "repeated_r06t_failure_signature": {
            "matches_platform_job_73087868": True,
            "error_id": 40510,
            "message": "complex sound speed in solid element",
            "element_id": int(error_match.group(1)),
            "part_id": 1002,
            "cycle": int(error_match.group(2)),
            "problem_time_ms": float(collapse_match.group(1)),
            "time_step_ms": float(collapse_match.group(2)),
        },
        "decision": {
            "classification": "job_label_r06u_but_exact_frozen_r06t_input_was_resubmitted",
            "r06u_was_executed": False,
            "r06u_numerical_result_available": False,
            "change_r06u_model": False,
            "release_exact_r06u_identity_guarded_resubmission": True,
            "download_heavy_binary_families": False,
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "dataset_reference_promotion": "denied",
        },
        "interpretation": [
            "The R06u-labelled job executed the exact frozen R06t deck, not the delivered R06u deck.",
            "The repeated R06t failure signature is a deterministic duplicate and is not evidence about R06u.",
            "The correct next action is an unchanged R06u resubmission in a fresh folder and fresh platform work directory.",
            "No material, contact, erosion, adaptive-SPH or output-control parameter may be changed for this retry.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an R06u-labelled run that actually executed R06t")
    parser.add_argument("--return-dir", type=Path, required=True)
    parser.add_argument("--intended-submission-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.resolve().is_relative_to(args.return_dir.resolve()):
        raise ValueError("audit output must remain outside the immutable return directory")
    payload = build_audit(args.return_dir, args.intended_submission_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "classification": payload["decision"]["classification"],
        "r06u_was_executed": payload["decision"]["r06u_was_executed"],
        "release_resubmission": payload["decision"]["release_exact_r06u_identity_guarded_resubmission"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
