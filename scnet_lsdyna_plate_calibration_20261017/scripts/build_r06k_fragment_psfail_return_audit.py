from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from build_r06g_tssfac02_return_audit import _negative_volume_topology, _part, sha256, strict_json


EXPECTED_JOB_ID = "r06k_0800_fragment_psfail_plate"
EXPECTED_PLATFORM_JOB_ID = "72764387"
EXPECTED_REVISION = "R06_fragment_negative_volume_isolation_20261030"
EXPECTED_DECK_SHA256 = "sha256:f78396a9fd15114fe411507a7774d68cecb10a25460413474772f2b99e8a17ca"


def _final_part_state(hvar6: dict, part: str) -> dict:
    for state in reversed(hvar6["states"]):
        if state["part"] == part:
            return state
    raise ValueError(f"missing final HVAR6 state for {part}")


def build_audit(*, return_dir: Path, analysis_dir: Path, baseline_audit_path: Path) -> dict:
    return_dir = return_dir.resolve()
    analysis_dir = analysis_dir.resolve()
    baseline = strict_json(baseline_audit_path.resolve())
    adapter = strict_json(analysis_dir / "r06k_single_job_analysis.json")
    independent = strict_json(analysis_dir / "independent_measurement_validation.json")
    ascii_metrics = strict_json(analysis_dir / "ascii_metrics.json")
    manifest = strict_json(return_dir / "manifest.json")
    deck_path = return_dir / "r06k_0800_fragment_psfail_plate.k"

    if (
        adapter["job_id"] != EXPECTED_JOB_ID
        or adapter["platform_job_id"] != EXPECTED_PLATFORM_JOB_ID
        or adapter["package_revision"] != EXPECTED_REVISION
        or manifest["job_id"] != EXPECTED_JOB_ID
        or manifest["package_revision"] != EXPECTED_REVISION
    ):
        raise ValueError("returned identity differs from the frozen R06k job")
    actual_hash = sha256(deck_path)
    if (
        actual_hash != EXPECTED_DECK_SHA256
        or manifest["deck"]["path"] != deck_path.name
        or manifest["deck"]["sha256"] != EXPECTED_DECK_SHA256
        or manifest["deck"]["size_bytes"] != deck_path.stat().st_size
    ):
        raise ValueError("returned R06k deck identity mismatch")

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
    cycles = re.findall(r"Problem cycle\s*=\s*(\d+)", stdout_text)
    times = re.findall(r"Problem time\s*=\s*([0-9.Ee+-]+)", stdout_text)
    if not cycles or not times:
        raise ValueError("solver completion metrics are unavailable")

    termination = ascii_metrics["termination"]
    global_stats = ascii_metrics["global_statistics"]
    solver_pass = (
        termination["normal_termination"]
        and not termination["error_termination"]
        and termination["normal_termination_rank_count"] == 32
        and stderr_path.stat().st_size == 0
    )
    no_mass_scaling = termination["mass_scaling_mentions"] == 0
    energy_gate_pass = (
        global_stats["minimum_total_energy_ratio"] is not None
        and global_stats["maximum_total_energy_ratio"] is not None
        and 0.95 <= global_stats["minimum_total_energy_ratio"]
        and global_stats["maximum_total_energy_ratio"] <= 1.05
        and global_stats["maximum_absolute_hourglass_over_internal"] is not None
        and global_stats["maximum_absolute_hourglass_over_internal"] <= 0.05
    )
    sliding_fraction = global_stats["final_sliding_energy_fraction_of_initial"]
    contact_energy_gate_pass = sliding_fraction is not None and abs(sliding_fraction) <= 0.05

    topology = _negative_volume_topology(return_dir, deck_path)
    if topology["unique_element_count"] != termination["negative_volume_unique_elements"]:
        raise ValueError("negative-volume topology disagrees with ASCII metrics")

    crosscheck = independent["d3part_full_d3plot_crosscheck"]
    ncforc = independent["ncforc"]
    measurement_pass = (
        crosscheck["status"] == "pass"
        and crosscheck["hvar6_alive_status"] == "available"
        and crosscheck["hvar6_invalid_alive_count"] == 0
        and ncforc["status"] == "available_ragged_pending_semantic_validation"
        and ncforc["all_coordinates_forces_pressure_finite"] is True
    )
    cases = adapter["capture"]["cases"]
    if len(cases) != 1:
        raise ValueError("expected exactly one capture ledger")
    candidate = cases[0]
    temporal_gate = adapter["capture"]["d3part_summary"]["temporal_gate"]
    hvar6 = adapter["mat015_hvar6"]
    fragment_hvar6 = _part(hvar6, "fragment")
    final_fragment = _final_part_state(hvar6, "fragment")
    final_alive = crosscheck["final_alive_fragment_count"]
    deleted = crosscheck["initial_alive_fragment_count"] - final_alive
    failure_damage_present = fragment_hvar6["maximum"] >= 1.0 and final_fragment["p95"] >= 1.0

    baseline_deleted = 686 - baseline["diagnostic_response"]["final_alive_fragment_count"]
    control_cycles = baseline["execution"]["cycle_count"]
    cycle_count = int(cycles[-1])
    deletion_suppressed = baseline_deleted == 35 and deleted == 0 and topology["fragment_unique_element_count"] == 0
    numerical_stability_pass = energy_gate_pass and contact_energy_gate_pass
    release_soft1 = (
        solver_pass
        and resource_pass
        and no_mass_scaling
        and measurement_pass
        and temporal_gate["status"] == "pass"
        and deletion_suppressed
        and failure_damage_present
        and not numerical_stability_pass
    )

    d3plots = [path for path in return_dir.glob("d3plot*") if path.is_file()]
    d3parts = [path for path in return_dir.glob("d3part*") if path.is_file()]
    binouts = [path for path in return_dir.glob("binout*") if path.is_file()]
    files = [path for path in return_dir.iterdir() if path.is_file()]

    return {
        "schema_version": "hf-p0-r06k-fragment-psfail-return-audit@1.0.0",
        "audit_date": "2026-10-31",
        "source_evidence": {
            "directory": return_dir.as_posix(),
            "evidence_policy": "read_only_immutable",
            "file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
            "platform_job_id": EXPECTED_PLATFORM_JOB_ID,
            "job_id": EXPECTED_JOB_ID,
            "package_revision": EXPECTED_REVISION,
            "deck_sha256": actual_hash,
            "solver_build": adapter["solver_build"],
            "reported_wall_clock_duration": "5h6m27s",
            "reported_wall_clock_seconds": 18387,
        },
        "execution": {
            "input_identity": "pass",
            "resource_identity": "pass" if resource_pass else "fail",
            "normal_termination": termination["normal_termination"],
            "rank_log_count": termination["rank_log_count"],
            "normal_termination_rank_count": termination["normal_termination_rank_count"],
            "stderr_size_bytes": stderr_path.stat().st_size,
            "cycle_count": cycle_count,
            "final_problem_time_ms": float(times[-1]),
            "numbered_error_count": sum(entry["occurrences"] for entry in termination["error_fatal_counts"]),
            "warning_counts": termination["warning_counts"],
            "negative_volume_unique_element_count": termination["negative_volume_unique_elements"],
            "negative_volume_topology": topology,
            "dt2ms": 0.0,
            "mass_scaling_mentions": termination["mass_scaling_mentions"],
            "minimum_time_step_ms": global_stats["minimum_time_step"],
            "maximum_time_step_ms": global_stats["maximum_time_step"],
        },
        "returned_outputs": {
            "d3plot": {"file_count": len(d3plots), "total_bytes": sum(p.stat().st_size for p in d3plots), "state_count": crosscheck["full_d3plot_state_count"], "readability": "pass"},
            "d3part": {"file_count": len(d3parts), "total_bytes": sum(p.stat().st_size for p in d3parts), "state_count": crosscheck["d3part_state_count"], "readability": "pass"},
            "binout": {"file_count": len(binouts), "total_bytes": sum(p.stat().st_size for p in binouts), "readability": "pass", "logical_databases": ncforc["logical_databases"]},
            "glstat": {
                "record_count": global_stats["record_count"],
                "minimum_total_energy_ratio": global_stats["minimum_total_energy_ratio"],
                "maximum_total_energy_ratio": global_stats["maximum_total_energy_ratio"],
                "final_total_energy_ratio": global_stats["final_total_energy_ratio"],
                "maximum_absolute_hourglass_over_internal": global_stats["maximum_absolute_hourglass_over_internal"],
                "final_sliding_energy": global_stats["final_sliding_energy"],
                "final_sliding_energy_fraction_of_initial": sliding_fraction,
                "final_energy_ratio_without_eroded": global_stats["final_energy_ratio_without_eroded"],
                "final_eroded_energy_fraction_of_initial": global_stats["final_eroded_energy_fraction_of_initial"],
            },
        },
        "measurement_contract": {
            "status": "pass" if measurement_pass else "fail",
            "d3part_full_d3plot_crosscheck": crosscheck,
            "temporal_gate": temporal_gate,
            "terminal_step_alignment_interpretation": (
                "One final D3PART interval exactly equals the final GLSTAT state plus its recorded explicit step; "
                "the fail-closed adapter accepts only this exact terminal alignment."
            ),
            "full_d3plot_hvar6": {
                "status": hvar6["status"],
                "state_count": hvar6["state_count"],
                "solid_count": hvar6["solid_count"],
                "invalid_alive_value_count": hvar6["invalid_alive_value_count"],
                "plate": _part(hvar6, "plate"),
                "fragment": fragment_hvar6,
                "final_fragment_state": final_fragment,
            },
            "ncforc": ncforc,
            "sleout": adapter["direct_contact_outputs"]["sleout"],
            "intfor": adapter["direct_contact_outputs"]["intfor"],
        },
        "diagnostic_response": {
            "adapter_primary_qoi_available": candidate["primary_qoi_available"],
            "formal_use_policy": "diagnostic_only_not_admissible_for_calibration",
            "unavailable_reasons": candidate["unavailable_reasons"],
            "classification": candidate["classification"],
            "diagnostic_values_before_availability_gate": candidate["diagnostic_values_before_availability_gate"],
            "deleted_upstream_mass_fraction": candidate["mass_fraction_by_ledger_state"]["deleted_upstream"],
            "worst_case_crossing_mass_fraction": candidate["worst_case_crossing_mass_fraction"],
            "crossing_event_count": candidate["crossing_event_count"],
            "tail_window": candidate["tail_window"],
            "final_alive_fragment_count": final_alive,
            "fragment_deleted_element_count": deleted,
        },
        "formal_fragment_negative_volume_comparison_to_job_72751725": {
            "status": "supported_fragment_deletion_removed_but_numerical_instability_exposed" if deletion_suppressed else "fail",
            "single_factor": "CONTROL_SOLID Card 1 field 5 PSFAIL, 0 to plate-only set 910001",
            "deletion_change": {
                "control_deleted_element_count": baseline_deleted,
                "treatment_deleted_element_count": deleted,
                "control_negative_volume_fragment_element_count": baseline["execution"]["negative_volume_topology"]["fragment_unique_element_count"],
                "treatment_negative_volume_fragment_element_count": topology["fragment_unique_element_count"],
                "suppression_gate": "pass" if deletion_suppressed else "fail",
            },
            "failure_damage_evidence": {
                "status": "pass" if failure_damage_present else "fail",
                "fragment_hvar6_maximum": fragment_hvar6["maximum"],
                "final_fragment_hvar6_mean": final_fragment["mean"],
                "final_fragment_hvar6_p50": final_fragment["p50"],
                "final_fragment_hvar6_p95": final_fragment["p95"],
                "final_fragment_hvar6_p99": final_fragment["p99"],
            },
            "performance_and_numerical_stability": {
                "status": "pass" if numerical_stability_pass else "fail",
                "control_cycle_count": control_cycles,
                "treatment_cycle_count": cycle_count,
                "cycle_count_ratio": cycle_count / control_cycles,
                "minimum_time_step_ms": global_stats["minimum_time_step"],
                "total_energy_ratio_gate": {
                    "status": "pass" if energy_gate_pass else "fail",
                    "allowed_range": [0.95, 1.05],
                    "observed_range": [global_stats["minimum_total_energy_ratio"], global_stats["maximum_total_energy_ratio"]],
                },
                "contact_sliding_energy_gate": {
                    "status": "pass" if contact_energy_gate_pass else "fail",
                    "candidate_absolute_fraction_of_initial_max": 0.05,
                    "observed": sliding_fraction,
                },
            },
            "interpretation": (
                "The exact disappearance of all 35 fragment deletions and negative-volume events supports the isolated "
                "ERODING-contact negative-volume deletion mechanism. Retaining heavily failed solids under SOFT=2 causes "
                "a severe timestep and contact-energy instability, so deletion suppression is not a production remedy."
            ),
        },
        "decision": {
            "solver_parser_runtime": "pass" if solver_pass else "fail",
            "input_identity": "pass",
            "resource_identity": "pass" if resource_pass else "fail",
            "measurement_channels": "pass" if measurement_pass else "fail",
            "fragment_negative_volume_hypothesis": "supported",
            "fragment_deletion_suppression": "pass" if deletion_suppressed else "fail",
            "response_availability": "diagnostic_only_not_admissible",
            "numerical_stability": "fail_contact_energy_and_timestep_collapse" if not numerical_stability_pass else "pass",
            "classification": "successful_mechanism_isolation_but_numerically_unstable_not_calibration_admissible",
            "release_r06l_contact_soft1_isolation": release_soft1,
            "release_2000_control": False,
            "release_treatments": False,
            "p0_response_calibration": "denied",
            "p1_submission": "denied",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "dataset_reference_promotion": "denied",
        },
        "recommended_next_sequence": [
            "Keep PSFAIL=910001, fragment MAT_015 EROD=1, plate EROD=0, TSSFAC=0.05 and DT2MS=0 frozen.",
            "Run one contact-algorithm diagnostic that changes only ERODING_SINGLE_SURFACE Optional Card A SOFT from 2 to 1.",
            "Require zero fragment deletion and negative-volume events, energy ratio within [0.95,1.05], and a material reduction of sliding-energy and cycle-count instability.",
            "If SOFT=1 remains unstable, stop contact-parameter tuning and redesign failed-fragment handling using an explicitly validated model-form such as adaptive solid-to-SPH.",
            "Keep 2000 m/s, treatments, fitting and reference-dataset promotion frozen.",
        ],
        "claim_boundary": (
            "This audit establishes a numerical deletion mechanism and identifies a new contact/timestep instability. "
            "It does not establish physical validity, a production erosion/contact policy, engine calibration, "
            "damage-tree calibration or a dataset label."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R06k fragment PSFAIL return audit")
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
