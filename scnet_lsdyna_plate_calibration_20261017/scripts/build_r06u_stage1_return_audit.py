from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re

from build_r06g_tssfac02_return_audit import _negative_volume_topology, sha256, strict_json
from parse_lsdyna_ascii import parse_control_timestep, parse_termination


EXPECTED_PLATFORM_JOB_ID = "73179287"
EXPECTED_JOB_ID = "r06u_1200_plate_adaptive_sph_canary"
EXPECTED_REVISION = "R06_plate_adaptive_sph_canary_20261104"
EXPECTED_DECK = "r06u_1200_plate_adaptive_sph_canary.k"
EXPECTED_DECK_SHA256 = "sha256:7bfb99a49bb757632237b2e2c625500fef27e811988eac3149df7a3e056647d4"
EXPECTED_RANKS = 32
REQUIRED_TIME_MS = 0.15
R06T_FAILURE_TIME_MS = 0.10872
R06T_FAILURE_ELEMENT = 23830
MINIMUM_TIME_STEP_GATE_MS = 3e-8

REQUIRED_SMALL_DATABASES = {
    "glstat": ("glstat", "glstat.glstat"),
    "matsum": ("matsum", "matsum.matsum"),
    "sleout": ("sleout", "sleout.sleout"),
    "nodout": ("nodout", "nodout.nodout"),
    "ncforc": ("ncforc", "ncforc.ncforc"),
}


def _single(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"expected one {label}, found {len(paths)}")
    return paths[0]


def _database_inventory(return_dir: Path) -> dict[str, dict]:
    inventory: dict[str, dict] = {}
    for family, candidates in REQUIRED_SMALL_DATABASES.items():
        found = [return_dir / name for name in candidates if (return_dir / name).is_file()]
        inventory[family] = {
            "available": bool(found),
            "path": found[0].name if found else None,
            "size_bytes": found[0].stat().st_size if found else None,
            "accepted_filenames": list(candidates),
        }
    optional = [path for name in ("sldsph_frag", "sldsph_frag.sldsph_frag") if (path := return_dir / name).is_file()]
    inventory["sldsph_frag_optional"] = {
        "available": bool(optional),
        "path": optional[0].name if optional else None,
        "size_bytes": optional[0].stat().st_size if optional else None,
        "accepted_filenames": ["sldsph_frag", "sldsph_frag.sldsph_frag"],
    }
    return inventory


def build_audit(return_dir: Path) -> dict:
    return_dir = return_dir.resolve()
    manifest = strict_json(return_dir / "manifest.json")
    contract = strict_json(return_dir / "job_contract.json")
    deck_path = return_dir / EXPECTED_DECK
    deck_hash = sha256(deck_path)
    if (
        manifest["job_id"] != EXPECTED_JOB_ID
        or manifest["package_revision"] != EXPECTED_REVISION
        or manifest["deck"]["path"] != EXPECTED_DECK
        or manifest["deck"]["sha256"] != EXPECTED_DECK_SHA256
        or deck_hash != EXPECTED_DECK_SHA256
        or contract["head_deck"] != EXPECTED_DECK
        or contract["case"]["termination_time_ms"] != REQUIRED_TIME_MS
        or contract["case"]["control_solid_psfail"] != 0
    ):
        raise ValueError("returned R06u input identity differs from the frozen canary")
    if contract["resource_contract"]["nodes"] != 1 or contract["resource_contract"]["ranks"] != EXPECTED_RANKS:
        raise ValueError("embedded R06u resource contract differs")
    timestep_control = parse_control_timestep(deck_path)
    if timestep_control["dt2ms"] != 0.0:
        raise ValueError("R06u deck unexpectedly enables mass scaling")

    slurm_path = _single(list(return_dir.glob("job_LS_DYNA_*.slurm")), "Slurm script")
    log_path = _single(list(return_dir.glob(f"job_*_{EXPECTED_PLATFORM_JOB_ID}.log")), "platform log")
    stdout_path = return_dir / f"std.out.{EXPECTED_PLATFORM_JOB_ID}"
    stderr_path = return_dir / f"std.err.{EXPECTED_PLATFORM_JOB_ID}"
    slurm_text = slurm_path.read_text(encoding="utf-8", errors="replace")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    resource_pass = all((
        "#SBATCH -N 1" in slurm_text,
        "#SBATCH --ntasks-per-node=32" in slurm_text,
        "The total cores is: 32" in log_text,
        "mpirun -np 32" in log_text,
        bool(re.search(r"MPP execution with\s+32 procs", stdout_text)),
    ))
    command_match = re.search(r"^mpirun .*?\bi=(\S+)", log_text, flags=re.MULTILINE)
    if command_match is None or Path(command_match.group(1)).name != EXPECTED_DECK:
        raise ValueError("platform command did not execute the frozen R06u deck")

    start_match = re.search(r"The start time is:\s*([0-9-]+ [0-9:]+)", log_text)
    end_match = re.search(r"The end time is:\s*([0-9-]+ [0-9:]+)", log_text)
    cycle_matches = re.findall(r"Problem cycle\s*=\s*(\d+)", stdout_text)
    time_matches = re.findall(r"Problem time\s*=\s*([0-9.Ee+-]+)", stdout_text)
    elapsed_match = re.search(r"Elapsed time\s+(\d+) seconds", stdout_text)
    if not all((start_match, end_match, cycle_matches, time_matches, elapsed_match)):
        raise ValueError("R06u completed-run timing evidence is incomplete")
    start = datetime.fromisoformat(start_match.group(1))
    end = datetime.fromisoformat(end_match.group(1))
    cycle_count = int(cycle_matches[-1])
    final_time_ms = float(time_matches[-1])

    termination = parse_termination(return_dir)
    topology = _negative_volume_topology(return_dir, deck_path)
    if topology["unique_element_count"] != termination["negative_volume_unique_elements"]:
        raise ValueError("negative-volume topology disagrees with the independent parser")
    plate = [item for item in topology["elements"] if item["part_kind"] == "plate"]
    fragment = [item for item in topology["elements"] if item["part_kind"] == "fragment"]
    target = [item for item in plate if item["element_id"] == R06T_FAILURE_ELEMENT]
    if len(target) != 1:
        raise ValueError("R06t target plate element is absent or ambiguous in the R06u ledger")

    progress_steps = [float(value) for value in re.findall(r"\bdt\s+([0-9.Ee+-]+)", stdout_text)]
    if not progress_steps:
        raise ValueError("stdout contains no progress time-step samples")
    complex_sound_speed = bool(re.search(r"complex sound speed", stdout_text, flags=re.IGNORECASE))
    completion_pass = bool(
        termination["normal_termination"]
        and not termination["error_termination"]
        and termination["rank_log_count"] == EXPECTED_RANKS
        and termination["normal_termination_rank_count"] == EXPECTED_RANKS
        and stderr_path.stat().st_size == 0
        and abs(final_time_ms - REQUIRED_TIME_MS) <= 1e-5
        and cycle_count < contract["case"]["maximum_cycle_count"]
        and not complex_sound_speed
    )

    databases = _database_inventory(return_dir)
    missing = [family for family in REQUIRED_SMALL_DATABASES if not databases[family]["available"]]
    files = [path for path in return_dir.iterdir() if path.is_file()]
    glstat_available = databases["glstat"]["available"]
    stage1_complete = not missing
    return {
        "schema_version": "hf-p0-r06u-stage1-partial-return-audit@1.0.0",
        "audit_date": "2026-11-04",
        "source_evidence": {
            "directory": return_dir.as_posix(),
            "evidence_policy": "read_only_immutable_partial_return",
            "download_stage": "stage1_identity_rank_logs_and_stdout_but_required_ascii_databases_incomplete",
            "file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
            "platform_job_id": EXPECTED_PLATFORM_JOB_ID,
            "job_id": EXPECTED_JOB_ID,
            "package_revision": EXPECTED_REVISION,
            "deck_sha256": deck_hash,
            "solver_build": "LS-DYNA R15.0.2 MPP double precision",
        },
        "execution": {
            "input_identity": "pass",
            "resource_identity": "pass" if resource_pass else "fail",
            "resource_contract": {"nodes": 1, "ranks": 32, "walltime": contract["resource_contract"]["walltime"]},
            "observed_resources": {"nodes": 1, "ranks": 32, "slurm_walltime": re.search(r"#SBATCH --time\s+(\S+)", slurm_text).group(1)},
            "normal_termination_at_required_time": completion_pass,
            "start_time": start.isoformat(sep=" "),
            "end_time": end.isoformat(sep=" "),
            "portal_elapsed_seconds": int((end - start).total_seconds()),
            "solver_elapsed_seconds": int(elapsed_match.group(1)),
            "cycle_count": cycle_count,
            "cycle_guard": contract["case"]["maximum_cycle_count"],
            "final_problem_time_ms": final_time_ms,
            "margin_beyond_r06t_failure_time_ms": final_time_ms - R06T_FAILURE_TIME_MS,
            "rank_log_count": termination["rank_log_count"],
            "normal_termination_rank_count": termination["normal_termination_rank_count"],
            "stderr_size_bytes": stderr_path.stat().st_size,
            "numbered_error_count": sum(item["occurrences"] for item in termination["error_fatal_counts"]),
            "complex_sound_speed_present": complex_sound_speed,
            "mass_scaling_configured": timestep_control["configured_mass_scaling"],
            "progress_time_step_sample_minimum_ms": min(progress_steps),
            "progress_time_step_sample_minimum_gate_ms": MINIMUM_TIME_STEP_GATE_MS,
            "progress_time_step_sample_gate": "provisional_pass_not_formal_glstat_minimum",
        },
        "adaptive_sph_runtime_evidence": {
            "plate_negative_volume_unique_element_count": len(plate),
            "fragment_negative_volume_unique_element_count": len(fragment),
            "plate_first_event_time_ms": min(item["first_time_ms"] for item in plate),
            "plate_last_event_time_ms": max(item["last_time_ms"] for item in plate),
            "fragment_first_event_time_ms": min(item["first_time_ms"] for item in fragment),
            "fragment_last_event_time_ms": max(item["last_time_ms"] for item in fragment),
            "r06t_target_element": target[0],
            "continued_from_target_failure_to_terminal_time_ms": final_time_ms - target[0]["first_time_ms"],
            "sph_memory_allocation_warning_count": len(re.findall(r"Warning 41125", stdout_text)),
            "interpretation": "parser/runtime capability evidence only; active-particle and mass closure require the staged databases and binary ledgers",
        },
        "stage1_small_database_inventory": databases,
        "missing_required_small_databases": missing,
        "preliminary_gates": {
            "input_identity": "pass",
            "resource_identity": "pass" if resource_pass else "fail",
            "solver_parser_runtime": "pass" if completion_pass else "fail",
            "target_element_survived_as_computation_beyond_r06t_collapse": "pass" if final_time_ms > R06T_FAILURE_TIME_MS else "fail",
            "formal_minimum_time_step": "hold_missing_glstat" if not glstat_available else "pending_parse",
            "energy": "hold_missing_glstat" if not glstat_available else "pending_parse",
            "plate_sph_positive_mass": "hold_missing_matsum" if not databases["matsum"]["available"] else "pending_parse",
            "contact_and_node_ledgers": "hold_missing_sleout_nodout_ncforc" if any(name in missing for name in ("sleout", "nodout", "ncforc")) else "pending_parse",
        },
        "decision": {
            "classification": "r06u_completed_runtime_capability_but_stage1_hold_missing_required_small_ascii_databases",
            "stage1_complete": stage1_complete,
            "release_heavy_binary_download": False,
            "release_next_compute_submission": False,
            "change_model": False,
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "dataset_reference_promotion": "denied",
        },
        "next_single_action": {
            "action": "download_only_the_missing_small_ascii_databases_from_job_73179287_into_the_same_local_directory",
            "required": missing,
            "optional_if_present": ["sldsph_frag"],
            "do_not_download_yet": ["d3plot*", "d3part*", "binout*"],
            "then_send": return_dir.as_posix(),
            "stop_condition": "do_not_build_or_submit_R06v_until_stage1_energy_timestep_mass_and_contact_gates_are_reviewed",
        },
        "claim_boundary": (
            "R06u establishes correct input execution, 32-rank normal completion and progress beyond the R06t collapse point. "
            "The missing GLSTAT/MATSUM/SLEOUT/NODOUT/NCFORC files prevent the preregistered Stage-1 engineering verdict; "
            "no binary download, 0.50 ms confirmation, physical validity, calibration or label admission is released."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the partial Stage-1 R06u return audit")
    parser.add_argument("--return-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.resolve().is_relative_to(args.return_dir.resolve()):
        raise ValueError("audit output must remain outside the immutable return directory")
    payload = build_audit(args.return_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "classification": payload["decision"]["classification"],
        "normal_completion": payload["execution"]["normal_termination_at_required_time"],
        "missing_required_small_databases": payload["missing_required_small_databases"],
        "release_next_compute": payload["decision"]["release_next_compute_submission"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
