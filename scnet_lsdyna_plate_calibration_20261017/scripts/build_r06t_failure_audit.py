from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re

from build_r06g_tssfac02_return_audit import _negative_volume_topology, sha256, strict_json
from parse_lsdyna_ascii import parse_termination


EXPECTED_JOB_ID = "r06t_1200_fragment_only_psfail_diagnostic"
EXPECTED_PLATFORM_JOB_ID = "73087868"
EXPECTED_REVISION = "R06_plate_negative_volume_psfail_scope_20261103"
EXPECTED_DECK_SHA256 = "sha256:11f62db3f8b27d1087988d5ad2abc572bddbd89570f6065500ddfe390a6459ae"
EXPECTED_ERROR_ELEMENT = 23830
EXPECTED_ERROR_CYCLE = 603398


def _single(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"expected one {label}, found {len(paths)}")
    return paths[0]


def build_audit(return_dir: Path) -> dict:
    return_dir = return_dir.resolve()
    manifest = strict_json(return_dir / "manifest.json")
    job_contract = strict_json(return_dir / "job_contract.json")
    deck_path = return_dir / "r06t_1200_fragment_only_psfail_diagnostic.k"
    deck_hash = sha256(deck_path)
    if (
        manifest["job_id"] != EXPECTED_JOB_ID
        or manifest["package_revision"] != EXPECTED_REVISION
        or manifest["deck"]["sha256"] != EXPECTED_DECK_SHA256
        or deck_hash != EXPECTED_DECK_SHA256
        or manifest["deck"]["size_bytes"] != deck_path.stat().st_size
    ):
        raise ValueError("R06t returned input identity differs from the frozen submission")
    if job_contract["resource_contract"]["nodes"] != 1 or job_contract["resource_contract"]["ranks"] != 32:
        raise ValueError("R06t embedded resource contract differs")

    slurm_path = _single(list(return_dir.glob("job_LS_DYNA_*.slurm")), "Slurm script")
    log_path = _single(list(return_dir.glob(f"job_*_{EXPECTED_PLATFORM_JOB_ID}.log")), "platform log")
    stdout_path = return_dir / f"std.out.{EXPECTED_PLATFORM_JOB_ID}"
    stderr_path = return_dir / f"std.err.{EXPECTED_PLATFORM_JOB_ID}"
    slurm_text = slurm_path.read_text(encoding="utf-8", errors="replace")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    host_match = re.search(r"\n([A-Za-z0-9_-]+):32\s*\n", log_text)
    solution_resource_identity = all((
        "#SBATCH -N 1" in slurm_text,
        "#SBATCH --ntasks-per-node=32" in slurm_text,
        "The total cores is: 32" in log_text,
        host_match is not None,
        "mpirun -np 32" in log_text,
        bool(re.search(r"MPP execution with\s+32 procs", stdout_text)),
        "R15.0.2" in stdout_text,
    ))
    walltime_match = re.search(r"#SBATCH --time\s+([^\s]+)", slurm_text)
    start_match = re.search(r"Start time\s+([0-9/]+ [0-9:]+)", stdout_text)
    end_match = re.search(r"End time\s+([0-9/]+ [0-9:]+)", stdout_text)
    elapsed_match = re.search(r"Elapsed time\s+(\d+) seconds", stdout_text)
    time_match = re.search(r"Problem time\s*=\s*([0-9.Ee+-]+)", stdout_text)
    cycle_match = re.search(r"Problem cycle\s*=\s*(\d+)", stdout_text)
    error_match = re.search(r"complex sound speed in solid element #\s*(\d+) cycle\s*(\d+)", stdout_text)
    collapse_match = re.search(r"603398 t\s+([0-9.Ee+-]+) dt\s+([0-9.Ee+-]+)", stdout_text)
    if not all((walltime_match, start_match, end_match, elapsed_match, time_match, cycle_match, error_match, collapse_match)):
        raise ValueError("R06t failure timing or error evidence is incomplete")
    start = datetime.strptime(start_match.group(1), "%m/%d/%Y %H:%M:%S")
    end = datetime.strptime(end_match.group(1), "%m/%d/%Y %H:%M:%S")
    error_element = int(error_match.group(1))
    error_cycle = int(error_match.group(2))
    if error_element != EXPECTED_ERROR_ELEMENT or error_cycle != EXPECTED_ERROR_CYCLE:
        raise ValueError("R06t terminal error identity differs")

    termination = parse_termination(return_dir)
    topology = _negative_volume_topology(return_dir, deck_path)
    plate_events = [item for item in topology["elements"] if item["part_kind"] == "plate"]
    fragment_events = [item for item in topology["elements"] if item["part_kind"] == "fragment"]
    if error_element not in {item["element_id"] for item in _plate_elements(deck_path)}:
        raise ValueError("R06t complex-sound-speed element is not in plate part 1002")
    if not termination["error_termination"] or termination["normal_termination"]:
        raise ValueError("R06t termination classification differs")

    files = [path for path in return_dir.iterdir() if path.is_file()]
    return {
        "schema_version": "hf-p0-r06t-failure-audit@1.0.0",
        "audit_date": "2026-11-04",
        "source_evidence": {
            "directory": return_dir.as_posix(),
            "evidence_policy": "read_only_immutable_partial_return",
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
            "resource_contract": {"nodes": 1, "ranks": 32, "walltime": job_contract["resource_contract"]["walltime"]},
            "observed_resources": {"nodes": 1, "ranks": 32, "host": host_match.group(1) if host_match else None, "walltime": walltime_match.group(1)},
            "normal_termination": False,
            "error_termination": True,
            "start_time": start.isoformat(sep=" "),
            "end_time": end.isoformat(sep=" "),
            "solver_elapsed_seconds": int(elapsed_match.group(1)),
            "wall_elapsed_seconds": int((end - start).total_seconds()),
            "problem_time_ms": float(time_match.group(1)),
            "cycle_count": int(cycle_match.group(1)),
            "terminal_time_step_ms": float(collapse_match.group(2)),
            "stderr_size_bytes": stderr_path.stat().st_size,
            "mass_scaling_mentions": termination["mass_scaling_mentions"],
            "glstat_available": any(path.is_file() for path in (return_dir / "glstat", return_dir / "glstat.glstat")),
        },
        "terminal_failure": {
            "error_id": 40510,
            "error_code": "SOL+510",
            "message": "complex sound speed in solid element",
            "element_id": error_element,
            "part_id": 1002,
            "part_kind": "plate",
            "cycle": error_cycle,
            "problem_time_ms": float(collapse_match.group(1)),
            "time_step_ms": float(collapse_match.group(2)),
        },
        "negative_volume_attribution_before_failure": {
            "unique_count": topology["unique_element_count"],
            "plate_unique_count": len(plate_events),
            "fragment_unique_count": len(fragment_events),
            "plate_negative_volume_suppression_observed": len(plate_events) == 0,
            "fragment_first_time_ms": min(item["first_time_ms"] for item in fragment_events),
            "fragment_last_time_ms": max(item["last_time_ms"] for item in fragment_events),
            "fragment_element_ids": [item["element_id"] for item in fragment_events],
        },
        "decision": {
            "classification": "psfail_scope_suppresses_plate_deletion_but_causes_plate_material_state_collapse",
            "fragment_only_psfail_hypothesis": "rejected_as_numerically_unstable",
            "download_heavy_binary_families": False,
            "release_r06u_plate_adaptive_sph_capability_canary": True,
            "branch_parent": "r06s_1200_speed_stability_diagnostic",
            "restore_control_solid_psfail": 0,
            "further_contact_parameter_tuning": "denied",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "dataset_reference_promotion": "denied",
        },
        "interpretation": [
            "R06t proves that fragment-only PSFAIL removes plate negative-volume deletion messages before failure.",
            "The same plate failure region remains physically unresolved and element 23830 reaches an invalid constitutive/EOS state instead of being replaced.",
            "The next model-form canary must replace failed plate solids with same-material SPH particles; it must not continue PSFAIL or contact-parameter tuning.",
            "No terminal response, physical validity, calibration response or dataset label is available from R06t.",
        ],
    }


def _plate_elements(deck_path: Path) -> list[dict]:
    elements: list[dict] = []
    in_elements = False
    for raw in deck_path.read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if line.upper().startswith("*ELEMENT_SOLID"):
            in_elements = True
            continue
        if in_elements and line.startswith("*"):
            break
        if in_elements and line and not line.startswith("$"):
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 2 and int(fields[1]) == 1002:
                elements.append({"element_id": int(fields[0]), "part_id": int(fields[1])})
    return elements


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R06t failed-run audit")
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
        "classification": audit["decision"]["classification"],
        "release_r06u": audit["decision"]["release_r06u_plate_adaptive_sph_capability_canary"],
        "heavy_download": audit["decision"]["download_heavy_binary_families"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
