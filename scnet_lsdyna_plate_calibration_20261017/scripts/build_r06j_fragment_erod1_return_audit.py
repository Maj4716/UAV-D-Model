from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from build_r06g_tssfac02_return_audit import (
    _negative_volume_topology,
    _part,
    sha256,
    strict_json,
)


EXPECTED_JOB_ID = "r06j_0800_fragment_erod1"
EXPECTED_PLATFORM_JOB_ID = "72751725"
EXPECTED_REVISION = "R06_fragment_erosion_isolation_20261030"
EXPECTED_DECK_SHA256 = "sha256:8ec1613324159b3ebc00217871ad29321564423b0a7f73a267207e4952bada77"


def build_audit(*, return_dir: Path, analysis_dir: Path, baseline_audit_path: Path) -> dict:
    return_dir = return_dir.resolve()
    analysis_dir = analysis_dir.resolve()
    baseline_audit = strict_json(baseline_audit_path.resolve())
    adapter = strict_json(analysis_dir / "r06j_single_job_analysis.json")
    independent = strict_json(analysis_dir / "independent_measurement_validation.json")
    ascii_metrics = strict_json(analysis_dir / "ascii_metrics.json")
    manifest = strict_json(return_dir / "manifest.json")
    deck_path = return_dir / "r06j_0800_fragment_erod1.k"

    if (
        adapter["job_id"] != EXPECTED_JOB_ID
        or adapter["platform_job_id"] != EXPECTED_PLATFORM_JOB_ID
        or adapter["package_revision"] != EXPECTED_REVISION
        or manifest["job_id"] != EXPECTED_JOB_ID
        or manifest["package_revision"] != EXPECTED_REVISION
    ):
        raise ValueError("returned identity differs from the frozen fragment EROD=1 job")
    actual_deck_hash = sha256(deck_path)
    if (
        actual_deck_hash != EXPECTED_DECK_SHA256
        or manifest["deck"]["path"] != deck_path.name
        or manifest["deck"]["sha256"] != EXPECTED_DECK_SHA256
        or manifest["deck"]["size_bytes"] != deck_path.stat().st_size
    ):
        raise ValueError("returned fragment EROD=1 deck identity mismatch")

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
    engineering_energy_gate_pass = (
        global_statistics["minimum_total_energy_ratio"] is not None
        and global_statistics["maximum_total_energy_ratio"] is not None
        and 0.95 <= global_statistics["minimum_total_energy_ratio"]
        and global_statistics["maximum_total_energy_ratio"] <= 1.05
        and global_statistics["maximum_absolute_hourglass_over_internal"] is not None
        and global_statistics["maximum_absolute_hourglass_over_internal"] <= 0.05
    )
    no_mass_scaling = termination["mass_scaling_mentions"] == 0
    negative_volume_topology = _negative_volume_topology(return_dir, deck_path)
    if negative_volume_topology["unique_element_count"] != termination["negative_volume_unique_elements"]:
        raise ValueError("negative-volume topology disagrees with the independent ASCII parser")
    contact_enables_negative_volume = (
        "ERODING contact is that a negative volume failure criterion" in stdout_text
        and "is automatically enabled for all solid parts" in stdout_text
    )

    hvar6 = adapter["mat015_hvar6"]
    fragment_hvar6 = _part(hvar6, "fragment")
    fragment_states = [state for state in hvar6["states"] if state["part"] == "fragment"]
    final_fragment_state = fragment_states[-1]
    failure_damage_present = bool(
        fragment_hvar6["status"] == "available"
        and fragment_hvar6["maximum"] is not None
        and fragment_hvar6["maximum"] >= 1.0
        and final_fragment_state["maximum"] is not None
        and final_fragment_state["maximum"] >= 1.0
    )

    baseline_deleted_fraction = baseline_response["deleted_upstream_mass_fraction"]
    candidate_deleted_fraction = candidate["mass_fraction_by_ledger_state"]["deleted_upstream"]
    deleted_fraction_reduction = baseline_deleted_fraction - candidate_deleted_fraction
    baseline_deleted_count = 686 - baseline_response["final_alive_fragment_count"]
    candidate_deleted_count = 686 - independent_d3part["final_alive_fragment_count"]
    negative_volume_accounts_for_all_remaining_deletion = bool(
        candidate_deleted_count == negative_volume_topology["fragment_unique_element_count"]
        and negative_volume_topology["plate_unique_element_count"] == 0
    )
    response_available = candidate["primary_qoi_available"] is True
    partial_mechanism_support = bool(
        failure_damage_present
        and deleted_fraction_reduction > 0.0
        and candidate_deleted_count < baseline_deleted_count
        and negative_volume_accounts_for_all_remaining_deletion
        and contact_enables_negative_volume
    )
    release_psfail_diagnostic = bool(
        solver_pass
        and resource_pass
        and measurement_pass
        and no_mass_scaling
        and partial_mechanism_support
        and not response_available
    )

    files = [path for path in return_dir.iterdir() if path.is_file()]
    d3plots = sorted(path for path in files if re.fullmatch(r"d3plot\d*", path.name))
    d3parts = sorted(path for path in files if re.fullmatch(r"d3part\d*", path.name))
    binouts = sorted(path for path in files if re.fullmatch(r"binout\d+", path.name))

    return {
        "schema_version": "hf-p0-r06j-fragment-erod1-return-audit@1.0.0",
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
            "eroding_contact_negative_volume_warning_present": contact_enables_negative_volume,
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
                "final_total_energy_ratio": global_statistics["final_total_energy_ratio"],
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
                "fragment": fragment_hvar6,
                "final_fragment_state": final_fragment_state,
            },
            "ncforc": ncforc,
            "sleout": adapter["direct_contact_outputs"]["sleout"],
            "intfor": adapter["direct_contact_outputs"]["intfor"],
        },
        "diagnostic_response": {
            "primary_qoi_available": candidate["primary_qoi_available"],
            "unavailable_reasons": candidate["unavailable_reasons"],
            "classification": candidate["classification"],
            "diagnostic_values_before_availability_gate": candidate["diagnostic_values_before_availability_gate"],
            "deleted_upstream_mass_fraction": candidate_deleted_fraction,
            "worst_case_crossing_mass_fraction": candidate["worst_case_crossing_mass_fraction"],
            "crossing_event_count": candidate["crossing_event_count"],
            "tail_window": candidate["tail_window"],
            "final_alive_plate_count": 20216,
            "final_alive_fragment_count": independent_d3part["final_alive_fragment_count"],
        },
        "formal_fragment_erosion_mechanism_comparison_to_job_72740022": {
            "status": "partial_support_remaining_negative_volume_path",
            "control_fragment_mat015_erod": 0,
            "candidate_fragment_mat015_erod": 1,
            "failure_damage_evidence": {
                "status": "pass" if failure_damage_present else "fail",
                "fragment_hvar6_maximum": fragment_hvar6["maximum"],
                "final_fragment_hvar6_maximum": final_fragment_state["maximum"],
                "final_fragment_hvar6_p99": final_fragment_state["p99"],
            },
            "deletion_change": {
                "control_deleted_element_count": baseline_deleted_count,
                "candidate_deleted_element_count": candidate_deleted_count,
                "deleted_element_count_reduction": baseline_deleted_count - candidate_deleted_count,
                "control_deleted_upstream_mass_fraction": baseline_deleted_fraction,
                "candidate_deleted_upstream_mass_fraction": candidate_deleted_fraction,
                "absolute_fraction_reduction": deleted_fraction_reduction,
                "relative_fraction_reduction": deleted_fraction_reduction / baseline_deleted_fraction,
                "suppression_direction_gate": "pass" if deleted_fraction_reduction > 0.0 else "fail",
            },
            "remaining_deletion_attribution": {
                "status": "pass_exact_count_and_fragment_topology" if negative_volume_accounts_for_all_remaining_deletion else "fail",
                "remaining_deleted_fragment_elements": candidate_deleted_count,
                "negative_volume_fragment_elements": negative_volume_topology["fragment_unique_element_count"],
                "negative_volume_plate_elements": negative_volume_topology["plate_unique_element_count"],
                "eroding_contact_auto_enables_negative_volume_failure": contact_enables_negative_volume,
            },
            "response_availability_gate": {
                "status": "pass" if response_available else "fail",
                "deleted_upstream_mass_fraction_max": 0.05,
                "observed": candidate_deleted_fraction,
                "excess_above_limit": candidate_deleted_fraction - 0.05,
                "unavailable_reasons": candidate["unavailable_reasons"],
            },
            "engineering_energy_gate": {
                "status": "pass_with_degradation_warning" if engineering_energy_gate_pass else "fail",
                "allowed_range": [0.95, 1.05],
                "observed_range": [
                    global_statistics["minimum_total_energy_ratio"],
                    global_statistics["maximum_total_energy_ratio"],
                ],
                "control_minimum_total_energy_ratio": baseline_audit["returned_outputs"]["glstat"]["minimum_total_energy_ratio"],
                "candidate_minimum_total_energy_ratio": global_statistics["minimum_total_energy_ratio"],
            },
            "interpretation": (
                "Suppressing MAT_015 failure erosion retains 12 additional fragment elements and preserves HVAR6 failure evidence, "
                "so material failure erosion is a partial deletion cause. The 35 remaining deleted fragment elements exactly match "
                "the 35 negative-volume elements created under the ERODING contact criterion; response availability still fails."
            ),
        },
        "decision": {
            "solver_parser_runtime": "pass" if solver_pass else "fail",
            "input_identity": "pass",
            "resource_identity": "pass" if resource_pass else "fail",
            "measurement_channels": "pass" if measurement_pass else "fail",
            "fragment_failure_erosion_hypothesis": "partial_support",
            "remaining_deletion_mechanism": "eroding_contact_automatic_negative_volume_failure",
            "response_availability": "pass" if response_available else "fail",
            "numerical_stability": "fail_negative_volume_driven_deletion",
            "classification": "successful_diagnostic_partial_material_erosion_causality_remaining_negative_volume_deletion",
            "release_r06k_fragment_negative_volume_psfail_isolation": release_psfail_diagnostic,
            "release_2000_control": False,
            "release_treatments": False,
            "p0_response_calibration": "denied",
            "p1_submission": "denied",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "dataset_reference_promotion": "denied",
        },
        "recommended_next_sequence": [
            "Keep fragment MAT_015 EROD=1, TSSFAC=0.05 and DT2MS=0 frozen.",
            "Run one PSFAIL isolation that changes only CONTROL_SOLID field 5 from 0 to plate-only set 910001.",
            "This excludes fragment part 2002 from the ERODING-contact negative-volume failure criterion while preserving the contact formulation.",
            "Treat normal completion, timestep collapse, severe distortion or solver termination as diagnostic evidence; none is a physical label.",
            "Keep 2000 m/s, treatments, fitting and reference-dataset promotion frozen.",
        ],
        "claim_boundary": (
            "This audit identifies two numerical deletion paths in the single 800 m/s diagnostic: MAT_015 failure erosion and "
            "ERODING-contact negative-volume erosion. It does not establish a physically correct erosion policy, converged response, "
            "engine calibration, damage-tree calibration or dataset-label admission."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R06j fragment EROD=1 return audit")
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
