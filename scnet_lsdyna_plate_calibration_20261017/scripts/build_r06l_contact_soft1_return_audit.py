from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re

from analyze_r05_failure_modes import parse_mes_failure_events
from build_r06g_tssfac02_return_audit import _negative_volume_topology, sha256, strict_json
from validate_p0_r06_diagnostics import parse_mesh


EXPECTED_JOB_ID = "r06l_0800_contact_soft1"
EXPECTED_PLATFORM_JOB_ID = "72782972"
EXPECTED_REVISION = "R06_contact_soft1_isolation_20261031"
EXPECTED_DECK_SHA256 = "sha256:73ba78b3ce25dc0b56b8bc2327b7cc0f5374ab35fef3978aa4ba0fdc62606576"


def build_audit(*, return_dir: Path, analysis_dir: Path, baseline_audit_path: Path) -> dict:
    return_dir = return_dir.resolve()
    analysis_dir = analysis_dir.resolve()
    baseline = strict_json(baseline_audit_path.resolve())
    independent = strict_json(analysis_dir / "independent_measurement_validation.json")
    ascii_metrics = strict_json(analysis_dir / "ascii_metrics.json")
    manifest = strict_json(return_dir / "manifest.json")
    deck_path = return_dir / "r06l_0800_contact_soft1.k"
    if manifest["job_id"] != EXPECTED_JOB_ID or manifest["package_revision"] != EXPECTED_REVISION:
        raise ValueError("returned identity differs from frozen R06l")
    deck_hash = sha256(deck_path)
    if (
        deck_hash != EXPECTED_DECK_SHA256
        or manifest["deck"]["path"] != deck_path.name
        or manifest["deck"]["sha256"] != deck_hash
        or manifest["deck"]["size_bytes"] != deck_path.stat().st_size
    ):
        raise ValueError("returned R06l deck identity mismatch")

    slurm_paths = list(return_dir.glob("job_LS_DYNA_*.slurm"))
    log_paths = list(return_dir.glob(f"job_*_{EXPECTED_PLATFORM_JOB_ID}.log"))
    stdout_path = return_dir / f"std.out.{EXPECTED_PLATFORM_JOB_ID}"
    stderr_path = return_dir / f"std.err.{EXPECTED_PLATFORM_JOB_ID}"
    if len(slurm_paths) != 1 or len(log_paths) != 1 or not stdout_path.is_file() or not stderr_path.is_file():
        raise ValueError("scheduler/resource evidence is incomplete or ambiguous")
    slurm_text = slurm_paths[0].read_text(encoding="utf-8", errors="replace")
    log_text = log_paths[0].read_text(encoding="utf-8", errors="replace")
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
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
    cancel_match = re.search(r"CANCELLED AT ([0-9T:-]+)", stderr_text)
    if not start_match or not cancel_match:
        raise ValueError("cancelled wall-clock interval is unavailable")
    start_time = datetime.fromisoformat(start_match.group(1))
    cancel_time = datetime.fromisoformat(cancel_match.group(1))
    elapsed_seconds = int((cancel_time - start_time).total_seconds())
    progress_rows = re.findall(r"(?m)^\s*(\d+)\s+t\s+([0-9.Ee+-]+)\s+dt\s+([0-9.Ee+-]+)", stdout_text)
    if not progress_rows:
        raise ValueError("solver progress evidence is unavailable")
    latest_cycle = int(progress_rows[-1][0])
    latest_time = float(progress_rows[-1][1])
    latest_dt = float(progress_rows[-1][2])
    termination_time = 0.04

    termination = ascii_metrics["termination"]
    global_stats = ascii_metrics["global_statistics"]
    crosscheck = independent["d3part_full_d3plot_crosscheck"]
    ncforc = independent["ncforc"]
    measurement_pass = (
        crosscheck["status"] == "pass"
        and crosscheck["hvar6_alive_status"] == "available"
        and crosscheck["hvar6_invalid_alive_count"] == 0
        and ncforc["status"] == "available_ragged_pending_semantic_validation"
        and ncforc["all_coordinates_forces_pressure_finite"] is True
    )
    topology = _negative_volume_topology(return_dir, deck_path)
    if topology["unique_element_count"] != termination["negative_volume_unique_elements"]:
        raise ValueError("negative-volume topology disagrees with ASCII metrics")
    failure_events = parse_mes_failure_events(return_dir)
    _, elements = parse_mesh(deck_path.read_text(encoding="ascii"))
    failure_counts: Counter[tuple[str, int | None]] = Counter()
    for event in failure_events["unique_events"]:
        part_id = elements.get(event["element_id"], (None, None))[0]
        failure_counts[(event["failure_type"], part_id)] += 1
    compact_failure_counts = {
        f"{failure_type}:part_{part_id}": count
        for (failure_type, part_id), count in sorted(failure_counts.items())
    }

    cycle_gate_max = 554704
    minimum_dt_gate = 1.49828e-08
    energy_gate_pass = (
        global_stats["minimum_total_energy_ratio"] is not None
        and global_stats["maximum_total_energy_ratio"] is not None
        and 0.95 <= global_stats["minimum_total_energy_ratio"]
        and global_stats["maximum_total_energy_ratio"] <= 1.05
    )
    sliding_fraction = global_stats["final_sliding_energy_fraction_of_initial"]
    sliding_gate_pass = sliding_fraction is not None and abs(sliding_fraction) <= 0.05
    efficiency_gate_pass = latest_cycle <= cycle_gate_max and global_stats["minimum_time_step"] >= minimum_dt_gate
    fragment_retained_but_fully_failed = (
        crosscheck["final_alive_fragment_count"] == 686
        and crosscheck["final_fragment_hvar6"]["minimum"] == 1.0
        and crosscheck["final_fragment_hvar6"]["maximum"] == 1.0
    )
    cancelled = "CANCELLED AT" in stderr_text
    release_adaptive_sph = (
        resource_pass
        and termination["mass_scaling_mentions"] == 0
        and cancelled
        and measurement_pass
        and fragment_retained_but_fully_failed
        and not energy_gate_pass
        and not sliding_gate_pass
        and not efficiency_gate_pass
    )

    files = [path for path in return_dir.iterdir() if path.is_file()]
    d3plots = [path for path in return_dir.glob("d3plot*") if path.is_file()]
    d3parts = [path for path in return_dir.glob("d3part*") if path.is_file()]
    binouts = [path for path in return_dir.glob("binout*") if path.is_file()]
    baseline_stability = baseline["formal_fragment_negative_volume_comparison_to_job_72751725"]["performance_and_numerical_stability"]
    return {
        "schema_version": "hf-p0-r06l-contact-soft1-cancelled-return-audit@1.0.0",
        "audit_date": "2026-11-01",
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
            "termination_kind": "user_or_scheduler_cancelled" if cancelled else "unknown_incomplete",
            "normal_termination": termination["normal_termination"],
            "error_termination": termination["error_termination"],
            "start_time": start_time.isoformat(sep=" "),
            "cancel_time": cancel_time.isoformat(sep=" "),
            "elapsed_seconds": elapsed_seconds,
            "elapsed_hms": f"{elapsed_seconds // 3600:02d}:{(elapsed_seconds % 3600) // 60:02d}:{elapsed_seconds % 60:02d}",
            "latest_cycle": latest_cycle,
            "latest_problem_time_ms": latest_time,
            "termination_time_ms": termination_time,
            "progress_fraction": latest_time / termination_time,
            "latest_reported_time_step_ms": latest_dt,
            "minimum_glstat_time_step_ms": global_stats["minimum_time_step"],
            "mass_scaling_mentions": termination["mass_scaling_mentions"],
            "negative_volume_unique_element_count": termination["negative_volume_unique_elements"],
            "negative_volume_topology": topology,
            "failure_message_ledger": {
                "rank_file_count": failure_events["rank_file_count"],
                "raw_observation_count": failure_events["raw_observation_count"],
                "unique_event_count": failure_events["unique_event_count"],
                "unique_element_count": failure_events["unique_element_count"],
                "cross_type_overlap_count": failure_events["cross_type_element_overlap_count"],
                "counts_by_type_and_part": compact_failure_counts,
            },
        },
        "returned_outputs": {
            "d3plot": {"file_count": len(d3plots), "total_bytes": sum(p.stat().st_size for p in d3plots), "state_count": crosscheck["full_d3plot_state_count"], "readability": "pass_partial"},
            "d3part": {"file_count": len(d3parts), "total_bytes": sum(p.stat().st_size for p in d3parts), "state_count": crosscheck["d3part_state_count"], "readability": "pass_partial"},
            "binout": {"file_count": len(binouts), "total_bytes": sum(p.stat().st_size for p in binouts), "readability": "pass_partial", "logical_databases": ncforc["logical_databases"]},
            "glstat": global_stats,
        },
        "measurement_contract": {
            "status": "pass_partial_readability_only" if measurement_pass else "fail",
            "terminal_response_available": False,
            "d3part_full_d3plot_crosscheck": crosscheck,
            "ncforc": ncforc,
        },
        "formal_contact_soft1_comparison_to_job_72764387": {
            "status": "rejected_soft1_worsened_contact_timestep_energy_and_plate_failure",
            "single_factor": "CONTACT_ERODING_SINGLE_SURFACE Optional Card A SOFT, 2 to 1",
            "control": {
                "job_id": "r06k_0800_fragment_psfail_plate",
                "platform_job_id": "72764387",
                "completed": True,
                "cycle_count": baseline_stability["treatment_cycle_count"],
                "minimum_time_step_ms": baseline_stability["minimum_time_step_ms"],
                "final_sliding_energy_fraction_of_initial": baseline["returned_outputs"]["glstat"]["final_sliding_energy_fraction_of_initial"],
                "fragment_deleted_element_count": 0,
                "plate_deleted_element_count": 0,
            },
            "treatment": {
                "job_id": EXPECTED_JOB_ID,
                "platform_job_id": EXPECTED_PLATFORM_JOB_ID,
                "completed": False,
                "progress_fraction": latest_time / termination_time,
                "cycle_count_at_cancel": latest_cycle,
                "minimum_time_step_ms": global_stats["minimum_time_step"],
                "final_partial_sliding_energy_fraction_of_initial": sliding_fraction,
                "fragment_deleted_element_count_at_last_d3part": 686 - crosscheck["final_alive_fragment_count"],
                "plate_deleted_element_count_at_last_d3plot": 20216 - crosscheck["final_alive_plate_count"],
                "fragment_hvar6_at_last_state": crosscheck["final_fragment_hvar6"],
            },
            "preregistered_gates": {
                "normal_completion": "fail",
                "fragment_deleted_element_count_zero": "pass_partial_only_not_terminal",
                "fragment_negative_volume_element_count_zero": "pass_partial_only_not_terminal",
                "energy_ratio": {"status": "pass" if energy_gate_pass else "fail", "allowed": [0.95, 1.05], "observed": [global_stats["minimum_total_energy_ratio"], global_stats["maximum_total_energy_ratio"]]},
                "absolute_sliding_energy_fraction": {"status": "pass" if sliding_gate_pass else "fail", "maximum": 0.05, "observed": sliding_fraction},
                "cycle_count": {"status": "pass" if latest_cycle <= cycle_gate_max else "fail", "maximum": cycle_gate_max, "observed_at_cancel": latest_cycle},
                "minimum_time_step": {"status": "pass" if global_stats["minimum_time_step"] >= minimum_dt_gate else "fail", "minimum_ms": minimum_dt_gate, "observed_ms": global_stats["minimum_time_step"]},
            },
            "interpretation": (
                "SOFT=1 did not stabilize the retained failed fragment. By the last readable partial state every fragment solid had HVAR6=1, "
                "the timestep collapsed by another order of magnitude, contact energy became more nonphysical, and thousands of plate solids "
                "were no longer alive. Continuing the run could not recover the preregistered efficiency or energy gates."
            ),
        },
        "decision": {
            "solver_parser_runtime": "pass_until_cancelled",
            "input_identity": "pass",
            "resource_identity": "pass" if resource_pass else "fail",
            "measurement_channels": "pass_partial_readability_only" if measurement_pass else "fail",
            "contact_soft1_hypothesis": "rejected",
            "numerical_stability": "fail_severe_timestep_contact_energy_and_plate_deletion",
            "terminal_response_availability": "fail_cancelled_before_termination",
            "classification": "valid_negative_result_soft1_rejected_model_form_pivot_required",
            "release_r06m_adaptive_solid_to_sph_canary": release_adaptive_sph,
            "release_additional_contact_parameter_tuning": False,
            "release_2000_control": False,
            "release_treatments": False,
            "p0_response_calibration": "denied",
            "p1_submission": "denied",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "dataset_reference_promotion": "denied",
        },
        "recommended_next_sequence": [
            "Return to the completed R06j SOFT=2/PSFAIL=0/fragment-EROD=1 baseline rather than retaining failed solids.",
            "Add one adaptive solid-to-SPH capability canary so failed fragment solids are converted instead of silently losing mass.",
            "Keep the same post-explosion state, 800 m/s case, TSSFAC=0.05, DT2MS=0 and 32-rank resource identity.",
            "Treat the first adaptive-SPH run as parser/runtime/mass-energy capability evidence only; do not use its response for calibration.",
            "Hold 2000 m/s, treatments, engine fitting, damage-tree fitting and reference labels.",
        ],
        "claim_boundary": (
            "This cancelled run is sufficient to reject SOFT=1 as the planned numerical stabilization because irreversible preregistered "
            "energy, timestep and efficiency gates failed. Its partial response is not a terminal QoI and has no physical or calibration admission."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cancelled R06l SOFT=1 return audit")
    parser.add_argument("--return-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.resolve().is_relative_to(args.return_dir.resolve()):
        raise ValueError("audit output must remain outside immutable return directory")
    audit = build_audit(return_dir=args.return_dir, analysis_dir=args.analysis_dir, baseline_audit_path=args.baseline_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": audit["decision"]["classification"], "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
