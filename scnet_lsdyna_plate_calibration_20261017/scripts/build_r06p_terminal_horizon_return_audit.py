from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re

from build_r06g_tssfac02_return_audit import _negative_volume_topology, sha256, strict_json


EXPECTED_JOB_ID = "r06p_0800_terminal_horizon_0p50"
EXPECTED_PLATFORM_JOB_ID = "72915268"
EXPECTED_REVISION = "R06_terminal_horizon_0p50_20261102"
EXPECTED_DECK_SHA256 = "sha256:d471f09e9a726c9e4695565abc120e9300bfd68ca2b272bf23852322f8dc8629"
TARGET_PLATE_ELEMENT_ID = 39948
MAXIMUM_CYCLE_COUNT = 829930
REQUIRED_PROBLEM_TIME_MS = 0.5


def _single(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"expected one {label}, found {len(paths)}")
    return paths[0]


def build_audit(*, return_dir: Path, analysis_dir: Path) -> dict:
    return_dir = return_dir.resolve()
    analysis_dir = analysis_dir.resolve()
    manifest = strict_json(return_dir / "manifest.json")
    contract = strict_json(return_dir / "p0_r06p_terminal_horizon_0p50_output_contract.json")
    job_contract = strict_json(return_dir / "job_contract.json")
    ascii_metrics = strict_json(analysis_dir / "ascii_metrics.json")
    validation = strict_json(analysis_dir / "adaptive_sph_validation.json")
    ledger = strict_json(analysis_dir / "terminal_response_ledger.json")
    deck_path = return_dir / "r06p_0800_terminal_horizon_0p50.k"

    if manifest["job_id"] != EXPECTED_JOB_ID or manifest["package_revision"] != EXPECTED_REVISION:
        raise ValueError("returned identity differs from frozen R06p")
    deck_hash = sha256(deck_path)
    if (
        deck_hash != EXPECTED_DECK_SHA256
        or manifest["deck"]["path"] != deck_path.name
        or manifest["deck"]["sha256"] != deck_hash
        or manifest["deck"]["size_bytes"] != deck_path.stat().st_size
    ):
        raise ValueError("returned R06p deck identity mismatch")
    if job_contract["resource_contract"]["nodes"] != 1 or job_contract["resource_contract"]["ranks"] != 32:
        raise ValueError("embedded R06p resource contract differs")

    slurm_path = _single(list(return_dir.glob("job_LS_DYNA_*.slurm")), "Slurm script")
    log_path = _single(list(return_dir.glob(f"job_*_{EXPECTED_PLATFORM_JOB_ID}.log")), "platform log")
    stdout_path = return_dir / f"std.out.{EXPECTED_PLATFORM_JOB_ID}"
    stderr_path = return_dir / f"std.err.{EXPECTED_PLATFORM_JOB_ID}"
    if not stdout_path.is_file() or not stderr_path.is_file():
        raise ValueError("solver stdout/stderr evidence is incomplete")
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
    crosscheck = validation["d3part_full_d3plot_crosscheck"]
    ncforc = validation["ncforc"]
    capture = ledger["capture_observations"]
    tail = ledger["terminal_tail"]
    primary = ledger["primary_response"]
    plate = ledger["plate_retention"]
    topology = _negative_volume_topology(return_dir, deck_path)
    converted_ids = set(ledger["solid_sph_transfer"]["final_converted_solid_element_ids"])
    negative_ids = {item["element_id"] for item in topology["elements"]}

    completion_pass = bool(
        termination["normal_termination"]
        and not termination["error_termination"]
        and termination["rank_log_count"] == 32
        and termination["normal_termination_rank_count"] == 32
        and stderr_path.stat().st_size == 0
        and abs(final_problem_time_ms - REQUIRED_PROBLEM_TIME_MS) <= 1e-5
        and cycle_count < MAXIMUM_CYCLE_COUNT
    )
    energy_pass = bool(
        0.95 <= glstat["minimum_total_energy_ratio"]
        and glstat["maximum_total_energy_ratio"] <= 1.05
        and abs(glstat["final_sliding_energy_fraction_of_initial"]) <= 0.05
        and glstat["maximum_absolute_hourglass_over_internal"] <= 0.05
    )
    mass_sph_pass = bool(
        adaptive["status"] == "pass_adaptive_sph_capability_and_mass_gate"
        and conversion["statewise_exact_activation_correspondence"] is True
        and conversion["final_active_sph_count"] == 8 * conversion["final_converted_solid_count"]
        and conversion["sph_deleted_particle_count"] == 0
        and mass["gate"] == "pass"
        and ledger["solid_sph_transfer"]["gate"] == "pass"
    )
    plate_pass = bool(
        conversion["plate_deleted_element_count"] == 0
        and plate["maximum_deleted_plate_element_count"] == 0
        and plate["target_element_id"] == TARGET_PLATE_ELEMENT_ID
        and plate["target_terminal_alive"] is True
    )
    measurement_pass = bool(
        crosscheck["status"] == "pass"
        and crosscheck["maximum_absolute_difference_all_7_channels"] == 0.0
        and crosscheck["alive_mask_mismatch_count"] == 0
        and ncforc["all_coordinates_forces_pressure_finite"] is True
    )
    negative_volume_attribution_pass = bool(
        topology["unique_element_count"] == 45
        and topology["plate_unique_element_count"] == 0
        and topology["fragment_unique_element_count"] == 45
        and negative_ids == converted_ids
    )
    terminal_separation_pass = tail["gate"] == "pass"
    zero_crossing_diagnostic_pass = bool(
        capture["solid_crossing_event_count"] == 0
        and capture["sph_crossing_event_count"] == 0
        and capture["newly_active_sph_below_plane_count"] == 0
        and capture["solid_deletion_interval_plane_ambiguity_count"] == 0
        and capture["active_or_alive_endpoint_at_or_below_plane_count"] == 0
    )
    registered_maximum_step = contract["preregistered_gates"]["output_interstate_displacement_mm"]["maximum"]
    observed_maximum_step = capture["maximum_observed_consecutive_downward_carrier_z_step_mm"]
    temporal_resolution_pass = bool(
        capture["temporal_resolution_gate"] == "pass"
        and observed_maximum_step <= registered_maximum_step
    )
    engineering_pass = bool(
        resource_pass
        and completion_pass
        and energy_pass
        and mass_sph_pass
        and plate_pass
        and measurement_pass
        and negative_volume_attribution_pass
        and termination["mass_scaling_mentions"] == 0
    )
    only_response_blocker_is_temporal_resolution = bool(
        engineering_pass
        and terminal_separation_pass
        and zero_crossing_diagnostic_pass
        and not temporal_resolution_pass
        and primary["available"] is False
        and primary["unavailable_reasons"] == ["temporal_resolution"]
    )

    files = [path for path in return_dir.iterdir() if path.is_file()]
    return {
        "schema_version": "hf-p0-r06p-terminal-horizon-return-audit@1.0.0",
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
            "cycle_guard_remaining": MAXIMUM_CYCLE_COUNT - cycle_count,
            "final_problem_time_ms": final_problem_time_ms,
            "stderr_size_bytes": stderr_path.stat().st_size,
            "mass_scaling_mentions": termination["mass_scaling_mentions"],
            "warning_counts": termination["warning_counts"],
        },
        "engineering_gates": {
            "resource_identity": "pass" if resource_pass else "fail",
            "solver_completion": "pass" if completion_pass else "fail",
            "fragment_sph_mass_and_finiteness": "pass" if mass_sph_pass else "fail",
            "plate_retention": "pass" if plate_pass else "fail",
            "energy": {
                "status": "pass" if energy_pass else "fail",
                "total_energy_ratio_range": [glstat["minimum_total_energy_ratio"], glstat["maximum_total_energy_ratio"]],
                "final_sliding_energy_fraction": glstat["final_sliding_energy_fraction_of_initial"],
                "final_eroded_energy_fraction": glstat["final_eroded_energy_fraction_of_initial"],
            },
            "negative_volume_attribution": {
                "status": "pass" if negative_volume_attribution_pass else "fail",
                "unique_count": topology["unique_element_count"],
                "fragment_count": topology["fragment_unique_element_count"],
                "plate_count": topology["plate_unique_element_count"],
                "exactly_matches_converted_fragment_set": negative_ids == converted_ids,
            },
            "measurement_crosscheck": "pass" if measurement_pass else "fail",
            "matsum_combined_source_sph_energy": "denied_duplicate_columns",
            "overall_engineering": "pass" if engineering_pass else "fail",
        },
        "response_assessment": {
            "primary_qoi_available": primary["available"],
            "classification": primary["classification"],
            "unavailable_reasons": primary["unavailable_reasons"],
            "zero_crossing_diagnostic": "pass" if zero_crossing_diagnostic_pass else "fail",
            "terminal_separation": "pass" if terminal_separation_pass else "fail",
            "temporal_resolution": {
                "status": "pass" if temporal_resolution_pass else "fail",
                "registered_maximum_downward_step_mm": registered_maximum_step,
                "observed_maximum_downward_step_mm": observed_maximum_step,
                "maximum_step_event": capture["maximum_downward_step_event"],
                "minimum_capture_plane_clearance_mm": capture["minimum_observed_capture_plane_clearance_mm"],
                "root_cause": (
                    "R06p planned cadence used the 0.04 ms terminal downward SPH speed rather than the full-history 800 m/s initial motion; "
                    "the actual early solid step exceeds the frozen 0.25 mm limit"
                ),
            },
            "diagnostic_nonperforation_evidence": {
                "crossing_event_count": capture["solid_crossing_event_count"] + capture["sph_crossing_event_count"],
                "minimum_carrier_z_mm": capture["minimum_observed_carrier_z_mm"],
                "terminal_tail_minimum_z_mm": tail["minimum_carrier_z_mm"],
                "terminal_tail_minimum_vz_mm_per_ms": tail["minimum_carrier_vz_mm_per_ms"],
                "terminal_tail_maximum_downward_count": tail["maximum_downward_carrier_count"],
                "interpretation": "strong rebound/nonperforation diagnostic only; frozen primary response remains unavailable",
            },
        },
        "decision": {
            "software_input_identity": "pass",
            "solver_parser_runtime": "pass",
            "resource_identity": "pass",
            "numerical_engineering_confirmation": "pass",
            "physical_validity": "not_established",
            "research_admission": "hold",
            "classification": "successful_long_horizon_run_response_held_by_preregistered_output_resolution_gate",
            "release_r06q_d3plot_0p00025_measurement_confirmation": only_response_blocker_is_temporal_resolution,
            "release_speed_scan": False,
            "release_2000_control": False,
            "release_treatments": False,
            "p0_response_calibration": "denied",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "dataset_reference_promotion": "denied",
        },
        "recommended_next_sequence": [
            "Repeat the exact R06p physical model, horizon and 32-rank resource identity.",
            "Change only the D3PLOT output interval from 0.0005 to 0.00025 ms; keep D3PART and ASCII intervals at 0.00025 ms.",
            "Require maximum full-history downward carrier step <=0.25 mm, exact zero-crossing/ambiguity counts, terminal separation and all engineering gates.",
            "Do not release the speed scan until the frozen 800 m/s primary response is admitted.",
        ],
        "claim_boundary": (
            "R06p is a complete and engineering-stable 0.50 ms run with strong diagnostic evidence of rebound and zero capture-plane crossing. "
            "Its primary QoI remains unavailable because the prospectively frozen output-resolution gate failed. This is not physical validation, "
            "engine calibration, damage-tree calibration or a reference label."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R06p terminal-horizon return audit")
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
    print(json.dumps({"classification": audit["decision"]["classification"], "release_r06q": audit["decision"]["release_r06q_d3plot_0p00025_measurement_confirmation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
