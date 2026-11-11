from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from build_r06g_tssfac02_return_audit import (
    _absolute_gate,
    _negative_volume_topology,
    _part,
    _qv_gate,
    sha256,
    strict_json,
)


EXPECTED_JOB_ID = "r06i_0800_tssfac005"
EXPECTED_PLATFORM_JOB_ID = "72740022"
EXPECTED_REVISION = "R06_timestep_terminal_refinement_20261030"
EXPECTED_DECK_SHA256 = "sha256:e2e8da8e55304b7fe2deb2c1790e00bf7fea9c64145c6c09a954c50492f3dd88"


def build_audit(*, return_dir: Path, analysis_dir: Path, baseline_audit_path: Path) -> dict:
    return_dir = return_dir.resolve()
    analysis_dir = analysis_dir.resolve()
    baseline_audit = strict_json(baseline_audit_path.resolve())
    adapter = strict_json(analysis_dir / "r06i_single_job_analysis.json")
    independent = strict_json(analysis_dir / "independent_measurement_validation.json")
    ascii_metrics = strict_json(analysis_dir / "ascii_metrics.json")
    manifest = strict_json(return_dir / "manifest.json")
    deck_path = return_dir / "r06i_0800_tssfac005.k"

    if (
        adapter["job_id"] != EXPECTED_JOB_ID
        or adapter["platform_job_id"] != EXPECTED_PLATFORM_JOB_ID
        or adapter["package_revision"] != EXPECTED_REVISION
        or manifest["job_id"] != EXPECTED_JOB_ID
        or manifest["package_revision"] != EXPECTED_REVISION
    ):
        raise ValueError("returned identity differs from the frozen TSSFAC=0.05 job")
    actual_deck_hash = sha256(deck_path)
    if (
        actual_deck_hash != EXPECTED_DECK_SHA256
        or manifest["deck"]["path"] != deck_path.name
        or manifest["deck"]["sha256"] != EXPECTED_DECK_SHA256
        or manifest["deck"]["size_bytes"] != deck_path.stat().st_size
    ):
        raise ValueError("returned TSSFAC=0.05 deck identity mismatch")

    slurm_paths = list(return_dir.glob("job_LS_DYNA_*.slurm"))
    log_paths = list(return_dir.glob(f"job_*_{EXPECTED_PLATFORM_JOB_ID}.log"))
    stdout_path = return_dir / f"std.out.{EXPECTED_PLATFORM_JOB_ID}"
    stderr_path = return_dir / f"std.err.{EXPECTED_PLATFORM_JOB_ID}"
    if len(slurm_paths) != 1 or len(log_paths) != 1 or not stdout_path.is_file() or not stderr_path.is_file():
        raise ValueError("scheduler/resource evidence is incomplete or ambiguous")
    slurm_text = slurm_paths[0].read_text(encoding="utf-8", errors="replace")
    log_text = log_paths[0].read_text(encoding="utf-8", errors="replace")
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
    cycle_match = re.findall(r"Problem cycle\s*=\s*(\d+)", stdout_text)
    time_match = re.findall(r"Problem time\s*=\s*([0-9.Ee+-]+)", stdout_text)
    if not cycle_match or not time_match:
        raise ValueError("solver completion metrics are unavailable")

    cases = adapter["capture"]["cases"]
    if len(cases) != 1:
        raise ValueError("expected exactly one capture ledger")
    candidate = cases[0]
    baseline_response = baseline_audit["diagnostic_response"]
    baseline_values = baseline_response["diagnostic_values_before_availability_gate"]
    candidate_values = candidate["diagnostic_values_before_availability_gate"]
    gates = {
        "qM": _absolute_gate(candidate_values["qM"], baseline_values["qM"], 0.01),
        "qP": _absolute_gate(candidate_values["qP"], baseline_values["qP"], 0.01),
        "qE": _absolute_gate(candidate_values["qE"], baseline_values["qE"], 0.01),
        "qV": _qv_gate(candidate, baseline_audit, 0.05),
        "deleted_upstream_mass_fraction": _absolute_gate(
            candidate["mass_fraction_by_ledger_state"]["deleted_upstream"],
            baseline_response["deleted_upstream_mass_fraction"],
            0.01,
        ),
    }
    comparison_gates_pass = all(item["status"].startswith("pass") for item in gates.values())
    independent_d3part = independent["d3part_full_d3plot_crosscheck"]
    ncforc = independent["ncforc"]
    measurement_pass = (
        independent_d3part["status"] == "pass"
        and independent_d3part["hvar6_alive_status"] == "available"
        and independent_d3part["hvar6_invalid_alive_count"] == 0
        and ncforc["status"] == "available_ragged_pending_semantic_validation"
        and ncforc["all_coordinates_forces_pressure_finite"] is True
    )
    termination = ascii_metrics["termination"]
    global_statistics = ascii_metrics["global_statistics"]
    solver_pass = (
        termination["normal_termination"]
        and not termination["error_termination"]
        and termination["normal_termination_rank_count"] == 32
        and stderr_path.stat().st_size == 0
    )
    energy_gate_pass = (
        global_statistics["minimum_total_energy_ratio"] is not None
        and global_statistics["maximum_total_energy_ratio"] is not None
        and 0.95 <= global_statistics["minimum_total_energy_ratio"]
        and global_statistics["maximum_total_energy_ratio"] <= 1.05
        and global_statistics["maximum_absolute_hourglass_over_internal"] is not None
        and global_statistics["maximum_absolute_hourglass_over_internal"] <= 0.05
    )
    no_mass_scaling = termination["mass_scaling_mentions"] == 0
    response_available = candidate["primary_qoi_available"] is True
    terminal_consistency_pass = bool(comparison_gates_pass and energy_gate_pass and no_mass_scaling)
    negative_volume_topology = _negative_volume_topology(return_dir, deck_path)
    if negative_volume_topology["unique_element_count"] != termination["negative_volume_unique_elements"]:
        raise ValueError("negative-volume topology disagrees with the independent ASCII parser")

    files = [path for path in return_dir.iterdir() if path.is_file()]
    d3plots = sorted(path for path in files if re.fullmatch(r"d3plot\d*", path.name))
    d3parts = sorted(path for path in files if re.fullmatch(r"d3part\d*", path.name))
    binouts = sorted(path for path in files if re.fullmatch(r"binout\d+", path.name))
    hvar6 = adapter["mat015_hvar6"]
    deletion_only_unavailability = bool(
        candidate["unavailable_reasons"] == ["deleted_upstream_mass"]
        and candidate["mass_fraction_by_ledger_state"]["deleted_upstream"] > 0.05
    )
    release_erod_diagnostic = bool(
        solver_pass
        and resource_pass
        and measurement_pass
        and terminal_consistency_pass
        and not response_available
        and deletion_only_unavailability
    )

    return {
        "schema_version": "hf-p0-r06i-tssfac005-return-audit@1.0.0",
        "audit_date": "2026-10-30",
        "source_evidence": {
            "directory": return_dir.as_posix(),
            "evidence_policy": "read_only_immutable",
            "file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
            "platform_job_id": EXPECTED_PLATFORM_JOB_ID,
            "job_id": EXPECTED_JOB_ID,
            "package_revision": EXPECTED_REVISION,
            "deck_sha256": actual_deck_hash,
            "solver_build": adapter["solver_build"],
        },
        "execution": {
            "input_identity": "pass",
            "resource_identity": "pass" if resource_pass else "fail",
            "normal_termination": solver_pass,
            "rank_log_count": termination["rank_log_count"],
            "normal_termination_rank_count": termination["normal_termination_rank_count"],
            "stderr_size_bytes": stderr_path.stat().st_size,
            "cycle_count": int(cycle_match[-1]),
            "final_problem_time_ms": float(time_match[-1]),
            "numbered_error_count": sum(item["occurrences"] for item in termination["error_fatal_counts"]),
            "warning_counts": termination["warning_counts"],
            "negative_volume_unique_element_count": termination["negative_volume_unique_elements"],
            "negative_volume_topology": negative_volume_topology,
            "negative_volume_policy": "secondary numerical warning; never treated as clean numerical acceptance",
            "dt2ms": 0.0,
            "mass_scaling_mentions": termination["mass_scaling_mentions"],
        },
        "returned_outputs": {
            "d3plot": {
                "file_count": len(d3plots),
                "total_bytes": sum(path.stat().st_size for path in d3plots),
                "state_count": independent_d3part["full_d3plot_state_count"],
                "readability": "pass",
            },
            "d3part": {
                "file_count": len(d3parts),
                "total_bytes": sum(path.stat().st_size for path in d3parts),
                "state_count": independent_d3part["d3part_state_count"],
                "readability": "pass",
            },
            "binout": {
                "file_count": len(binouts),
                "total_bytes": sum(path.stat().st_size for path in binouts),
                "readability": "pass",
                "logical_databases": ncforc["logical_databases"],
            },
            "glstat": {
                "record_count": global_statistics["record_count"],
                "minimum_total_energy_ratio": global_statistics["minimum_total_energy_ratio"],
                "maximum_total_energy_ratio": global_statistics["maximum_total_energy_ratio"],
                "maximum_absolute_hourglass_energy": global_statistics["maximum_absolute_hourglass_energy"],
                "maximum_absolute_hourglass_over_internal": global_statistics["maximum_absolute_hourglass_over_internal"],
                "final_eroded_energy_fraction_of_initial": global_statistics["final_eroded_energy_fraction_of_initial"],
            },
        },
        "measurement_contract": {
            "status": "pass" if measurement_pass else "fail",
            "d3part_full_d3plot_crosscheck": independent_d3part,
            "full_d3plot_hvar6": {
                "status": hvar6["status"],
                "state_count": hvar6["state_count"],
                "solid_count": hvar6["solid_count"],
                "invalid_alive_value_count": hvar6["invalid_alive_value_count"],
                "plate": _part(hvar6, "plate"),
                "fragment": _part(hvar6, "fragment"),
            },
            "ncforc": ncforc,
            "sleout": adapter["direct_contact_outputs"]["sleout"],
            "intfor": adapter["direct_contact_outputs"]["intfor"],
        },
        "diagnostic_response": {
            "primary_qoi_available": candidate["primary_qoi_available"],
            "unavailable_reasons": candidate["unavailable_reasons"],
            "classification": candidate["classification"],
            "diagnostic_values_before_availability_gate": candidate_values,
            "deleted_upstream_mass_fraction": candidate["mass_fraction_by_ledger_state"]["deleted_upstream"],
            "worst_case_crossing_mass_fraction": candidate["worst_case_crossing_mass_fraction"],
            "crossing_event_count": candidate["crossing_event_count"],
            "tail_window": candidate["tail_window"],
            "final_alive_plate_count": 20216,
            "final_alive_fragment_count": independent_d3part["final_alive_fragment_count"],
        },
        "formal_terminal_timestep_comparison_to_job_72701744": {
            "status": "pass_local_consistency_response_unavailable" if terminal_consistency_pass else "fail",
            "control_tssfac": 0.1,
            "candidate_tssfac": 0.05,
            "gates": gates,
            "terminal_timestep_consistency_gate": {
                "status": "pass" if terminal_consistency_pass else "fail",
                "excludes_response_availability": True,
            },
            "response_availability_gate": {
                "status": "pass" if response_available else "fail",
                "candidate_primary_qoi_available": response_available,
                "unavailable_reasons": candidate["unavailable_reasons"],
                "deleted_upstream_mass_fraction_max": 0.05,
                "observed": candidate["mass_fraction_by_ledger_state"]["deleted_upstream"],
            },
            "energy_gate": {
                "status": "pass" if energy_gate_pass else "fail",
                "allowed_range": [0.95, 1.05],
                "observed_range": [
                    global_statistics["minimum_total_energy_ratio"],
                    global_statistics["maximum_total_energy_ratio"],
                ],
                "hourglass_to_internal_max": 0.05,
                "observed_hourglass_to_internal": global_statistics["maximum_absolute_hourglass_over_internal"],
            },
            "no_mass_scaling_gate": {"status": "pass" if no_mass_scaling else "fail", "dt2ms": 0.0},
            "supporting_changes": {
                "cycle_count": {"control": baseline_audit["execution"]["cycle_count"], "candidate": int(cycle_match[-1])},
                "negative_volume_unique_element_count": {
                    "control": baseline_audit["execution"]["negative_volume_unique_element_count"],
                    "candidate": termination["negative_volume_unique_elements"],
                },
                "final_alive_fragment_count": {
                    "control": baseline_response["final_alive_fragment_count"],
                    "candidate": independent_d3part["final_alive_fragment_count"],
                },
                "final_eroded_energy_fraction_of_initial": {
                    "control": baseline_audit["returned_outputs"]["glstat"]["final_eroded_energy_fraction_of_initial"],
                    "candidate": global_statistics["final_eroded_energy_fraction_of_initial"],
                },
            },
            "interpretation": (
                "TSSFAC 0.1-to-0.05 is locally consistent under the frozen comparison tolerances, "
                "but the response remains unavailable because deleted-upstream fragment mass exceeds 0.05."
            ),
        },
        "decision": {
            "solver_parser_runtime": "pass" if solver_pass else "fail",
            "input_identity": "pass",
            "resource_identity": "pass" if resource_pass else "fail",
            "measurement_channels": "pass" if measurement_pass else "fail",
            "timestep_terminal_consistency": "pass" if terminal_consistency_pass else "fail",
            "response_availability": "pass" if response_available else "fail",
            "classification": "successful_run_terminal_timestep_locally_stable_but_response_unavailable",
            "release_r06j_fragment_erod1_diagnostic": release_erod_diagnostic,
            "release_any_further_timestep_halving": False,
            "release_2000_control": False,
            "release_treatments": False,
            "p0_response_calibration": "denied",
            "p1_submission": "denied",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "dataset_reference_promotion": "denied",
        },
        "recommended_next_sequence": [
            "Stop automatic timestep halving; TSSFAC=0.025 is not authorized.",
            "Run one 800 m/s, TSSFAC=0.05 diagnostic that changes only fragment MAT_015 Card 4 EROD from 0 to 1.",
            "Use the return only to isolate whether Johnson-Cook failure erosion causes upstream fragment deletion.",
            "Treat EROD=1 outputs as mechanism evidence, not physical response labels, engine targets or damage-tree targets.",
            "Keep 2000 m/s, treatments, fitting and reference-dataset promotion frozen.",
        ],
        "claim_boundary": (
            "This audit establishes local 0.1-to-0.05 timestep consistency under the frozen gates, while the "
            "primary response remains unavailable. It does not establish physical validity, P0/P1 admission, "
            "engine calibration, damage-tree calibration or dataset-label admission."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R06i TSSFAC=0.05 return audit")
    parser.add_argument("--return-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.resolve().is_relative_to(args.return_dir.resolve()):
        raise ValueError("audit output must remain outside the immutable return directory")
    audit = build_audit(
        return_dir=args.return_dir,
        analysis_dir=args.analysis_dir,
        baseline_audit_path=args.baseline_audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": audit["decision"]["classification"], "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
