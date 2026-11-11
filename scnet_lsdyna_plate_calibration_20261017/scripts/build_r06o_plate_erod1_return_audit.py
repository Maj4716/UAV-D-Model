from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re

from build_r06g_tssfac02_return_audit import sha256, strict_json


EXPECTED_JOB_ID = "r06o_0800_plate_erod1_isolation"
EXPECTED_PLATFORM_JOB_ID = "72894870"
EXPECTED_REVISION = "R06_plate_material_erod1_isolation_20261102"
EXPECTED_DECK_SHA256 = "sha256:d71d2d11091a813f35ca98656c31fb7ad4f8174b024828a62e0f874daab3524f"
TARGET_PLATE_ELEMENT_ID = 39948
NEXT_HORIZON_MS = 0.5
MAXIMUM_CYCLE_COUNT = 829930


def _single(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"expected one {label}, found {len(paths)}")
    return paths[0]


def build_audit(*, return_dir: Path, analysis_dir: Path) -> dict:
    return_dir = return_dir.resolve()
    analysis_dir = analysis_dir.resolve()
    manifest = strict_json(return_dir / "manifest.json")
    contract = strict_json(return_dir / "p0_r06o_plate_erod1_output_contract.json")
    job_contract = strict_json(return_dir / "job_contract.json")
    ascii_metrics = strict_json(analysis_dir / "ascii_metrics.json")
    validation = strict_json(analysis_dir / "adaptive_sph_validation.json")
    comparison = strict_json(analysis_dir / "r06n_r06o_plate_erod_horizon_comparison.json")
    deck_path = return_dir / "r06o_0800_plate_erod1_isolation.k"

    if manifest["job_id"] != EXPECTED_JOB_ID or manifest["package_revision"] != EXPECTED_REVISION:
        raise ValueError("returned identity differs from frozen R06o")
    deck_hash = sha256(deck_path)
    if (
        deck_hash != EXPECTED_DECK_SHA256
        or manifest["deck"]["path"] != deck_path.name
        or manifest["deck"]["sha256"] != deck_hash
        or manifest["deck"]["size_bytes"] != deck_path.stat().st_size
    ):
        raise ValueError("returned R06o deck identity mismatch")
    if job_contract["resource_contract"]["nodes"] != 1 or job_contract["resource_contract"]["ranks"] != 32:
        raise ValueError("embedded R06o resource contract differs")

    slurm_path = _single(list(return_dir.glob("job_LS_DYNA_*.slurm")), "Slurm script")
    log_path = _single(list(return_dir.glob(f"job_*_{EXPECTED_PLATFORM_JOB_ID}.log")), "platform log")
    stdout_path = return_dir / f"std.out.{EXPECTED_PLATFORM_JOB_ID}"
    stderr_path = return_dir / f"std.err.{EXPECTED_PLATFORM_JOB_ID}"
    if not stdout_path.is_file() or not stderr_path.is_file():
        raise ValueError("solver stdout/stderr evidence is incomplete")
    slurm_text = slurm_path.read_text(encoding="utf-8", errors="replace")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    resource_pass = all(
        (
            "#SBATCH -N 1" in slurm_text,
            "#SBATCH --ntasks-per-node=32" in slurm_text,
            "The total cores is: 32" in log_text,
            "mpirun -np 32" in log_text,
            bool(re.search(r"MPP execution with\s+32 procs", stdout_text)),
        )
    )

    start_match = re.search(r"The start time is:\s*([0-9-]+ [0-9:]+)", log_text)
    end_match = re.search(r"The end time is:\s*([0-9-]+ [0-9:]+)", log_text)
    cycle_match = re.findall(r"Problem cycle\s*=\s*(\d+)", stdout_text)
    time_match = re.findall(r"Problem time\s*=\s*([0-9.Ee+-]+)", stdout_text)
    elapsed_match = re.search(r"Elapsed time\s+(\d+) seconds", stdout_text)
    if not start_match or not end_match or not cycle_match or not time_match or not elapsed_match:
        raise ValueError("completed-run timing evidence is incomplete")
    start = datetime.fromisoformat(start_match.group(1))
    end = datetime.fromisoformat(end_match.group(1))
    cycle_count = int(cycle_match[-1])
    final_problem_time_ms = float(time_match[-1])

    termination = ascii_metrics["termination"]
    glstat = ascii_metrics["global_statistics"]
    adaptive = validation["adaptive_sph"]
    conversion = adaptive["conversion"]
    mass = adaptive["mass_accounting"]
    capture = adaptive["capture_plane"]
    crosscheck = validation["d3part_full_d3plot_crosscheck"]
    ncforc = validation["ncforc"]
    paired = comparison["paired_diagnostics"]
    target = comparison["r06o"]["run"]["target_plate_element"]
    tail = comparison["r06o"]["run"]["terminal_tail_diagnostic"]

    completion_pass = bool(
        termination["normal_termination"]
        and not termination["error_termination"]
        and termination["rank_log_count"] == 32
        and termination["normal_termination_rank_count"] == 32
        and stderr_path.stat().st_size == 0
        and abs(final_problem_time_ms - 0.04) <= 1e-6
        and cycle_count < MAXIMUM_CYCLE_COUNT
    )
    energy_gate_pass = bool(
        0.95 <= glstat["minimum_total_energy_ratio"]
        and glstat["maximum_total_energy_ratio"] <= 1.05
        and abs(glstat["final_sliding_energy_fraction_of_initial"]) <= 0.05
        and glstat["maximum_absolute_hourglass_over_internal"] <= 0.05
    )
    mass_and_sph_pass = bool(
        conversion["statewise_exact_activation_correspondence"] is True
        and conversion["final_active_sph_count"] == 8 * conversion["final_converted_solid_count"]
        and conversion["sph_deleted_particle_count"] == 0
        and mass["gate"] == "pass"
        and adaptive["active_sph"]["all_positions_finite"] is True
        and adaptive["active_sph"]["all_velocities_finite"] is True
    )
    plate_retention_pass = bool(
        conversion["plate_deleted_element_count"] == 0
        and conversion["maximum_plate_deleted_element_count"] == 0
        and conversion["final_deleted_plate_element_ids"] == []
        and target["element_id"] == TARGET_PLATE_ELEMENT_ID
        and target["terminal_alive"] is True
        and target["final_hvar6"] > 0.0
    )
    efficiency_pass = bool(
        paired["cycle_count_ratio_r06o_over_r06n"] <= contract["preregistered_gates"]["efficiency_diagnostic_cycle_ratio_to_r06n"]["maximum"]
        and paired["minimum_time_step_ratio_r06o_over_r06n"] >= contract["preregistered_gates"]["minimum_time_step_ratio_to_r06n"]["minimum"]
    )
    measurement_pass = bool(
        crosscheck["status"] == "pass"
        and crosscheck["maximum_absolute_difference_all_7_channels"] == 0.0
        and crosscheck["alive_mask_mismatch_count"] == 0
        and ncforc["status"] == "available_ragged_pending_semantic_validation"
        and ncforc["all_coordinates_forces_pressure_finite"] is True
    )
    negative_volume_attribution_pass = bool(
        comparison["r06o"]["negative_volume"]["unique_element_count"] == termination["negative_volume_unique_elements"]
        and comparison["r06o"]["negative_volume"]["plate_unique_element_count"] == 0
        and comparison["r06o"]["negative_volume"]["fragment_unique_element_count"] == termination["negative_volume_unique_elements"]
    )
    engineering_isolation_pass = bool(
        resource_pass
        and completion_pass
        and energy_gate_pass
        and mass_and_sph_pass
        and plate_retention_pass
        and efficiency_pass
        and measurement_pass
        and negative_volume_attribution_pass
        and termination["mass_scaling_mentions"] == 0
    )
    release_horizon = bool(
        engineering_isolation_pass
        and capture["primary_qoi_available"] is False
        and paired["recommended_next_horizon_ms"] == NEXT_HORIZON_MS
    )

    files = [path for path in return_dir.iterdir() if path.is_file()]
    return {
        "schema_version": "hf-p0-r06o-plate-erod1-return-audit@1.0.0",
        "audit_date": "2026-11-02",
        "source_evidence": {
            "directory": return_dir.as_posix(),
            "evidence_policy": "read_only_immutable",
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
            "resource_contract": {"nodes": 1, "ranks": 32},
            "observed_resources": {"nodes": 1, "ranks": 32},
            "normal_termination_at_required_time": completion_pass,
            "start_time": start.isoformat(sep=" "),
            "end_time": end.isoformat(sep=" "),
            "portal_elapsed_seconds": int((end - start).total_seconds()),
            "solver_elapsed_seconds": int(elapsed_match.group(1)),
            "cycle_count": cycle_count,
            "cycle_guard": MAXIMUM_CYCLE_COUNT,
            "final_problem_time_ms": final_problem_time_ms,
            "stderr_size_bytes": stderr_path.stat().st_size,
            "mass_scaling_mentions": termination["mass_scaling_mentions"],
            "warning_counts": termination["warning_counts"],
            "negative_volume_unique_element_count": termination["negative_volume_unique_elements"],
        },
        "engineering_gates": {
            "resource_identity": "pass" if resource_pass else "fail",
            "solver_completion": "pass" if completion_pass else "fail",
            "fragment_sph_mass_and_finiteness": "pass" if mass_and_sph_pass else "fail",
            "energy": {
                "status": "pass" if energy_gate_pass else "fail",
                "observed_total_energy_ratio_range": [glstat["minimum_total_energy_ratio"], glstat["maximum_total_energy_ratio"]],
                "observed_final_sliding_fraction": glstat["final_sliding_energy_fraction_of_initial"],
                "observed_eroded_energy_fraction": glstat["final_eroded_energy_fraction_of_initial"],
            },
            "plate_retention": {
                "status": "pass" if plate_retention_pass else "fail",
                "deleted_plate_element_count": conversion["plate_deleted_element_count"],
                "target_element_id": TARGET_PLATE_ELEMENT_ID,
                "target_terminal_alive": target["terminal_alive"],
                "target_final_hvar6": target["final_hvar6"],
            },
            "efficiency": {
                "status": "pass" if efficiency_pass else "fail",
                "minimum_time_step_ms": glstat["minimum_time_step"],
                "cycle_count_ratio_r06o_over_r06n": paired["cycle_count_ratio_r06o_over_r06n"],
                "minimum_time_step_ratio_r06o_over_r06n": paired["minimum_time_step_ratio_r06o_over_r06n"],
            },
            "negative_volume_attribution": {
                "status": "pass" if negative_volume_attribution_pass else "fail",
                "total_unique": comparison["r06o"]["negative_volume"]["unique_element_count"],
                "fragment_unique": comparison["r06o"]["negative_volume"]["fragment_unique_element_count"],
                "plate_unique": comparison["r06o"]["negative_volume"]["plate_unique_element_count"],
            },
            "measurement_crosscheck": "pass" if measurement_pass else "fail",
            "matsum_combined_source_sph_energy": "denied_duplicate_columns",
        },
        "response_availability": {
            "status": "terminal_horizon_unqualified",
            "primary_qoi_available": False,
            "capture_plane": capture,
            "terminal_tail_diagnostic": tail,
            "recommended_next_horizon_ms": paired["recommended_next_horizon_ms"],
            "reason": "the 0.04 ms terminal state retains substantial downward-moving solid/SPH mass before any z=-12 mm crossing",
            "projection_boundary": "terminal-position/velocity linear projections select the next horizon only and are not physical results",
        },
        "decision": {
            "software_input_identity": "pass",
            "solver_parser_runtime": "pass",
            "resource_identity": "pass",
            "numerical_engineering_confirmation": "pass_plate_erod1_isolation" if engineering_isolation_pass else "fail",
            "physical_validity": "not_established",
            "research_admission": "hold",
            "classification": "successful_plate_erod1_engineering_isolation_response_horizon_still_unqualified",
            "release_r06p_terminal_horizon_0p50": release_horizon,
            "release_2000_control": False,
            "release_treatments": False,
            "p0_response_calibration": "denied",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "dataset_reference_promotion": "denied",
        },
        "recommended_next_sequence": [
            "Branch from the exact returned R06o deck and extend only the terminal horizon from 0.04 to 0.50 ms.",
            "Relax output cadence prospectively to cap storage while preserving sub-0.025 mm capture-plane temporal displacement at the observed maximum downward speed.",
            "Keep plate/fragment materials, adaptive-SPH, contact, TSSFAC=0.05, DT2MS=0, cycle guard and exact one-node/32-rank resources frozen.",
            "If the cycle guard is reached before 0.50 ms, return the failed run without restart or parameter tuning.",
            "Do not fit the engine or damage tree until a complete response ledger and later replication/mesh/velocity/physical-validity gates pass.",
        ],
        "claim_boundary": (
            "R06o establishes a resource-correct engineering baseline for plate EROD=1 at 0.04 ms: the target plate element is retained without timestep collapse. "
            "It does not establish a terminal response, physical validity, engine calibration, damage-tree calibration or a reference label."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R06o plate EROD=1 return audit")
    parser.add_argument("--return-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.resolve().is_relative_to(args.return_dir.resolve()):
        raise ValueError("audit output must remain outside the immutable return directory")
    audit = build_audit(return_dir=args.return_dir, analysis_dir=args.analysis_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": audit["decision"]["classification"], "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
