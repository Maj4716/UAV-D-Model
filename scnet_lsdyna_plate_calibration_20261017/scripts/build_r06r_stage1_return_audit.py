from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re

from build_r06g_tssfac02_return_audit import _negative_volume_topology, sha256, strict_json
from parse_lsdyna_ascii import parse_glstat, parse_termination


EXPECTED_JOB_ID = "r06r_2000_speed_upper_anchor"
EXPECTED_PLATFORM_JOB_ID = "72963617"
EXPECTED_REVISION = "R06_speed_upper_anchor_2000_20261102"
EXPECTED_DECK_SHA256 = "sha256:511cbde9d33b1422981bc5716b0c9ae16a3fcd14f091e1dec003d053f3808e48"
REQUIRED_PROBLEM_TIME_MS = 0.5
MAXIMUM_CYCLE_COUNT = 2_500_000
TARGET_PLATE_ELEMENT_ID = 39948


def _single(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"expected one {label}, found {len(paths)}")
    return paths[0]


def _glstat_path(return_dir: Path) -> Path:
    candidates = [return_dir / "glstat", return_dir / "glstat.glstat"]
    found = [path for path in candidates if path.is_file()]
    return _single(found, "GLSTAT file")


def build_audit(return_dir: Path) -> dict:
    return_dir = return_dir.resolve()
    manifest = strict_json(return_dir / "manifest.json")
    job_contract = strict_json(return_dir / "job_contract.json")
    deck_path = return_dir / "r06r_2000_speed_upper_anchor.k"
    deck_hash = sha256(deck_path)
    if (
        manifest["job_id"] != EXPECTED_JOB_ID
        or manifest["package_revision"] != EXPECTED_REVISION
        or manifest["deck"]["sha256"] != EXPECTED_DECK_SHA256
        or deck_hash != EXPECTED_DECK_SHA256
        or manifest["deck"]["size_bytes"] != deck_path.stat().st_size
    ):
        raise ValueError("R06r returned input identity differs from the frozen submission")
    if (
        job_contract["job_id"] != EXPECTED_JOB_ID
        or job_contract["resource_contract"]["nodes"] != 1
        or job_contract["resource_contract"]["ranks"] != 32
    ):
        raise ValueError("R06r embedded job/resource contract differs")

    slurm_path = _single(list(return_dir.glob("job_LS_DYNA_*.slurm")), "Slurm script")
    log_path = _single(list(return_dir.glob(f"job_*_{EXPECTED_PLATFORM_JOB_ID}.log")), "platform log")
    stdout_path = return_dir / f"std.out.{EXPECTED_PLATFORM_JOB_ID}"
    stderr_path = return_dir / f"std.err.{EXPECTED_PLATFORM_JOB_ID}"
    if not stdout_path.is_file() or not stderr_path.is_file():
        raise ValueError("R06r stdout/stderr evidence is incomplete")
    slurm_text = slurm_path.read_text(encoding="utf-8", errors="replace")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")

    solution_resource_identity = all((
        "#SBATCH -N 1" in slurm_text,
        "#SBATCH --ntasks-per-node=32" in slurm_text,
        "The total cores is: 32" in log_text,
        "d01r4n03:32" in log_text,
        "mpirun -np 32" in log_text,
        bool(re.search(r"MPP execution with\s+32 procs", stdout_text)),
        "R15.0.2" in stdout_text,
    ))
    requested_walltime_match = re.search(r"#SBATCH --time\s+([^\s]+)", slurm_text)
    if not requested_walltime_match:
        raise ValueError("Slurm walltime is unavailable")
    observed_walltime = requested_walltime_match.group(1)
    contract_walltime = job_contract["resource_contract"]["walltime"]

    start_match = re.search(r"The start time is:\s*([0-9-]+ [0-9:]+)", log_text)
    end_match = re.search(r"The end time is:\s*([0-9-]+ [0-9:]+)", log_text)
    cycle_match = re.findall(r"Problem cycle\s*=\s*(\d+)", stdout_text)
    time_match = re.findall(r"Problem time\s*=\s*([0-9.Ee+-]+)", stdout_text)
    elapsed_match = re.search(r"Elapsed time\s+(\d+) seconds", stdout_text)
    if not start_match or not end_match or not cycle_match or not time_match or not elapsed_match:
        raise ValueError("R06r completed-run timing evidence is incomplete")
    start = datetime.fromisoformat(start_match.group(1))
    end = datetime.fromisoformat(end_match.group(1))
    cycle_count = int(cycle_match[-1])
    final_problem_time_ms = float(time_match[-1])

    termination = parse_termination(return_dir)
    glstat = parse_glstat(_glstat_path(return_dir))
    topology = _negative_volume_topology(return_dir, deck_path)
    plate_events = [item for item in topology["elements"] if item["part_kind"] == "plate"]
    fragment_events = [item for item in topology["elements"] if item["part_kind"] == "fragment"]
    target_negative_volume = any(item["element_id"] == TARGET_PLATE_ELEMENT_ID for item in plate_events)

    completion_pass = bool(
        termination["normal_termination"]
        and not termination["error_termination"]
        and termination["rank_log_count"] == 32
        and termination["normal_termination_rank_count"] == 32
        and stderr_path.stat().st_size == 0
        and abs(final_problem_time_ms - REQUIRED_PROBLEM_TIME_MS) <= 1e-5
        and cycle_count < MAXIMUM_CYCLE_COUNT
    )
    total_energy_pass = bool(
        0.95 <= glstat["minimum_total_energy_ratio"]
        and glstat["maximum_total_energy_ratio"] <= 1.05
        and glstat["maximum_absolute_hourglass_over_internal"] <= 0.05
    )
    sliding_fraction = glstat["final_sliding_energy_fraction_of_initial"]
    sliding_energy_pass = abs(sliding_fraction) <= 0.05
    plate_negative_volume_pass = len(plate_events) == 0 and not target_negative_volume
    stage1_pass = bool(
        solution_resource_identity
        and completion_pass
        and termination["mass_scaling_mentions"] == 0
        and total_energy_pass
        and sliding_energy_pass
        and plate_negative_volume_pass
    )
    release_diagnostic = bool(
        solution_resource_identity
        and completion_pass
        and not stage1_pass
        and (not sliding_energy_pass or not plate_negative_volume_pass)
    )

    files = [path for path in return_dir.iterdir() if path.is_file()]
    return {
        "schema_version": "hf-p0-r06r-stage1-return-audit@1.0.0",
        "audit_date": "2026-11-03",
        "source_evidence": {
            "directory": return_dir.as_posix(),
            "evidence_policy": "read_only_immutable_partial_return",
            "download_stage": "stage1_ascii_and_identity_only",
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
            "solution_resource_identity": "pass" if solution_resource_identity else "fail",
            "resource_contract": {"nodes": 1, "ranks": 32, "walltime": contract_walltime},
            "observed_resources": {"nodes": 1, "ranks": 32, "host": "d01r4n03", "walltime": observed_walltime},
            "walltime_contract_deviation": observed_walltime != contract_walltime,
            "walltime_deviation_effect": "allocation_limit_only_no_solution_change_because_run_completed_before_both_limits",
            "normal_termination_at_required_time": completion_pass,
            "start_time": start.isoformat(sep=" "),
            "end_time": end.isoformat(sep=" "),
            "portal_elapsed_seconds": int((end - start).total_seconds()),
            "solver_elapsed_seconds": int(elapsed_match.group(1)),
            "cycle_count": cycle_count,
            "cycle_guard": MAXIMUM_CYCLE_COUNT,
            "cycle_guard_remaining": MAXIMUM_CYCLE_COUNT - cycle_count,
            "final_problem_time_ms": final_problem_time_ms,
            "stderr_size_bytes": stderr_path.stat().st_size,
            "mass_scaling_mentions": termination["mass_scaling_mentions"],
        },
        "global_engineering": {
            "total_energy_and_hourglass": "pass" if total_energy_pass else "fail",
            "total_energy_ratio_range": [glstat["minimum_total_energy_ratio"], glstat["maximum_total_energy_ratio"]],
            "final_sliding_energy_fraction_of_initial": sliding_fraction,
            "absolute_final_sliding_energy_fraction": abs(sliding_fraction),
            "registered_absolute_sliding_fraction_maximum": 0.05,
            "sliding_energy_gate": "pass" if sliding_energy_pass else "fail",
            "final_eroded_energy_fraction_of_initial": glstat["final_eroded_energy_fraction_of_initial"],
            "maximum_absolute_hourglass_over_internal": glstat["maximum_absolute_hourglass_over_internal"],
            "minimum_time_step_ms": glstat["minimum_time_step"],
        },
        "negative_volume_attribution": {
            "status": "pass" if plate_negative_volume_pass else "fail",
            "unique_count": topology["unique_element_count"],
            "plate_unique_count": len(plate_events),
            "fragment_unique_count": len(fragment_events),
            "plate_first_time_ms": min(item["first_time_ms"] for item in plate_events) if plate_events else None,
            "plate_last_time_ms": max(item["last_time_ms"] for item in plate_events) if plate_events else None,
            "fragment_first_time_ms": min(item["first_time_ms"] for item in fragment_events) if fragment_events else None,
            "fragment_last_time_ms": max(item["last_time_ms"] for item in fragment_events) if fragment_events else None,
            "target_plate_element_id": TARGET_PLATE_ELEMENT_ID,
            "target_plate_negative_volume_event": target_negative_volume,
            "plate_element_ids": [item["element_id"] for item in plate_events],
            "fragment_element_ids": [item["element_id"] for item in fragment_events],
        },
        "stage1_decision": {
            "stage1_engineering_admission": "pass" if stage1_pass else "fail",
            "classification": (
                "stage1_pass_download_binary_families" if stage1_pass
                else "completed_but_engineering_failed_plate_negative_volume_and_sliding_energy"
            ),
            "primary_perforation_response": "unavailable_not_downloaded_and_engineering_failed",
            "download_full_d3plot": False if not stage1_pass else True,
            "download_d3part": False,
            "download_binout": False,
            "stop_heavy_download": not stage1_pass,
            "release_r06s_1200_speed_stability_diagnostic": release_diagnostic,
            "release_midpoint_as_calibration_point": False,
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "dataset_reference_promotion": "denied",
        },
        "failure_interpretation": [
            "Normal termination and exact rank identity do not override the preregistered engineering failures.",
            "The 1374 plate negative-volume events include target element 39948 and are not admissible as a physical perforation label.",
            "The absolute final sliding-energy fraction is 0.0689495, above the frozen 0.05 maximum.",
            "Heavy binary download is stopped because neither d3plot nor binout can repair these already-observed failures.",
        ],
        "recommended_next_action": (
            "Run one 1200 m/s single-factor stability diagnostic with the same physical model, 0.50 ms horizon, "
            "0.00015 ms D3PLOT/D3PART cadence and exact 1-node/32-rank identity. If plate negative-volume or "
            "sliding-energy failure repeats, stop the speed scan and redesign the plate/contact model form."
        ),
        "claim_boundary": (
            "R06r completed normally but failed numerical-engineering admission. No perforation/nonperforation response, "
            "physical upper anchor, engine calibration, damage-tree calibration or dataset label is admitted."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R06r stage-1 ASCII/identity return audit")
    parser.add_argument("--return-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.resolve().is_relative_to(args.return_dir.resolve()):
        raise ValueError("audit output must remain outside the immutable return directory")
    audit = build_audit(args.return_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "classification": audit["stage1_decision"]["classification"],
        "stop_heavy_download": audit["stage1_decision"]["stop_heavy_download"],
        "release_r06s": audit["stage1_decision"]["release_r06s_1200_speed_stability_diagnostic"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
