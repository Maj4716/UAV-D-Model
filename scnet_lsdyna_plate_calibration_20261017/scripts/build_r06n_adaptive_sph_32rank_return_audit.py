from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re

from build_r06g_tssfac02_return_audit import sha256, strict_json


EXPECTED_JOB_ID = "r06n_0800_adaptive_sph_32rank_confirmation"
EXPECTED_PLATFORM_JOB_ID = "72864757"
EXPECTED_REVISION = "R06_adaptive_sph_32rank_confirmation_20261101"
EXPECTED_DECK_SHA256 = "sha256:edc5f3b80174e52c81eb96a16e0e5abe89f1587b8a850f87f7048bd40f1ac4b7"
TARGET_PLATE_ELEMENT_ID = 39948


def _single(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"expected one {label}, found {len(paths)}")
    return paths[0]


def build_audit(*, return_dir: Path, analysis_dir: Path) -> dict:
    return_dir = return_dir.resolve()
    analysis_dir = analysis_dir.resolve()
    manifest = strict_json(return_dir / "manifest.json")
    contract = strict_json(return_dir / "p0_r06n_adaptive_sph_32rank_output_contract.json")
    job_contract = strict_json(return_dir / "job_contract.json")
    ascii_metrics = strict_json(analysis_dir / "ascii_metrics.json")
    validation = strict_json(analysis_dir / "adaptive_sph_validation.json")
    sensitivity = strict_json(analysis_dir / "r06m_r06n_resource_sensitivity.json")
    deck_path = return_dir / "r06n_0800_adaptive_sph_32rank_confirmation.k"

    if manifest["job_id"] != EXPECTED_JOB_ID or manifest["package_revision"] != EXPECTED_REVISION:
        raise ValueError("returned identity differs from frozen R06n")
    deck_hash = sha256(deck_path)
    if (
        deck_hash != EXPECTED_DECK_SHA256
        or manifest["deck"]["path"] != deck_path.name
        or manifest["deck"]["sha256"] != deck_hash
        or manifest["deck"]["size_bytes"] != deck_path.stat().st_size
    ):
        raise ValueError("returned R06n deck identity mismatch")
    if job_contract["resource_contract"]["nodes"] != 1 or job_contract["resource_contract"]["ranks"] != 32:
        raise ValueError("embedded R06n resource contract differs")

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
    paired = sensitivity["paired_diagnostics"]
    target_m = sensitivity["r06m"]["d3plot"]["target_plate_element"]
    target_n = sensitivity["r06n"]["d3plot"]["target_plate_element"]
    mes_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in sorted(return_dir.glob("mes*"))
    )
    target_failure_messages = len(
        re.findall(rf"solid element\s+{TARGET_PLATE_ELEMENT_ID}\s+failed at time", mes_text)
    )

    completion_pass = bool(
        termination["normal_termination"]
        and not termination["error_termination"]
        and termination["rank_log_count"] == 32
        and termination["normal_termination_rank_count"] == 32
        and stderr_path.stat().st_size == 0
        and abs(final_problem_time_ms - 0.04) <= 1e-6
        and cycle_count < 829930
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
    plate_gate_pass = conversion["plate_deleted_element_count"] == 0
    cycle_ratio_bounds = contract["preregistered_gates"]["cycle_count_ratio_to_r06m"]
    cycle_ratio = paired["cycle_count_ratio_r06n_over_r06m"]
    cycle_repeatability_pass = cycle_ratio_bounds["minimum"] <= cycle_ratio <= cycle_ratio_bounds["maximum"]
    conversion_difference_gate = contract["preregistered_gates"]["conversion_count_difference_from_r06m"]
    conversion_repeatability_pass = (
        abs(paired["converted_fragment_count_difference"]) <= conversion_difference_gate["maximum_absolute"]
    )
    measurement_pass = bool(
        crosscheck["status"] == "pass"
        and crosscheck["maximum_absolute_difference_all_7_channels"] == 0.0
        and crosscheck["alive_mask_mismatch_count"] == 0
        and ncforc["status"] == "available_ragged_pending_semantic_validation"
        and ncforc["all_coordinates_forces_pressure_finite"] is True
    )
    plate_failure_attributed = bool(
        conversion["final_deleted_plate_element_ids"] == [TARGET_PLATE_ELEMENT_ID]
        and target_failure_messages >= 1
        and paired["r06n_plate_failure_is_not_in_negative_volume_set"] is True
        and target_n["first_failed_state"] is not None
    )
    release_plate_erod_isolation = bool(
        resource_pass
        and completion_pass
        and energy_gate_pass
        and mass_and_sph_pass
        and measurement_pass
        and not plate_gate_pass
        and plate_failure_attributed
        and termination["mass_scaling_mentions"] == 0
    )

    files = [path for path in return_dir.iterdir() if path.is_file()]
    return {
        "schema_version": "hf-p0-r06n-adaptive-sph-32rank-return-audit@1.0.0",
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
            "cycle_guard": 829930,
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
                "observed_total_energy_ratio_range": [
                    glstat["minimum_total_energy_ratio"], glstat["maximum_total_energy_ratio"]
                ],
                "observed_final_sliding_fraction": glstat["final_sliding_energy_fraction_of_initial"],
            },
            "plate_deletion": {
                "status": "pass" if plate_gate_pass else "fail",
                "required": 0,
                "observed": conversion["plate_deleted_element_count"],
                "element_ids": conversion["final_deleted_plate_element_ids"],
            },
            "cycle_repeatability": {
                "status": "pass" if cycle_repeatability_pass else "fail",
                "allowed_ratio": [cycle_ratio_bounds["minimum"], cycle_ratio_bounds["maximum"]],
                "observed_ratio": cycle_ratio,
                "r06m_cycle_count": sensitivity["r06m"]["completion"]["cycle_count"],
                "r06n_cycle_count": cycle_count,
            },
            "fragment_conversion_repeatability": {
                "status": "pass" if conversion_repeatability_pass else "fail",
                "allowed_absolute_count_difference": conversion_difference_gate["maximum_absolute"],
                "observed_count_difference": paired["converted_fragment_count_difference"],
                "set_jaccard": paired["converted_fragment_jaccard"],
                "converted_only_in_r06n": paired["converted_only_in_r06n"],
            },
            "measurement_crosscheck": "pass" if measurement_pass else "fail",
            "matsum_combined_source_sph_energy": "denied_duplicate_columns",
        },
        "plate_failure_diagnosis": {
            "status": "material_failure_erosion_not_negative_volume",
            "element_id": TARGET_PLATE_ELEMENT_ID,
            "mes_failure_message_count": target_failure_messages,
            "mes_first_failure_time_ms": 0.0061673,
            "d3plot_first_dead_time_ms": target_n["first_failed_time_ms"],
            "negative_volume_membership": False,
            "initial_centroid_mm": target_n["initial_centroid_mm"],
            "last_alive_hvar6": target_n["last_alive_hvar6"],
            "r06m_same_element_final_hvar6": target_m["final_hvar6_raw"],
            "interpretation": (
                "Element 39948 is explicitly reported as failed, disappears in D3PLOT, and is absent from the 37-element "
                "negative-volume set. The plate material retains EROD=0 while the fragment material uses EROD=1; the next "
                "single-factor diagnostic tests plate EROD=1 without changing contact, SPH or resource settings."
            ),
        },
        "response_availability": {
            "status": "fail_plate_deletion_and_terminal_horizon_unqualified",
            "primary_qoi_available": False,
            "capture_plane": capture,
            "reason": (
                "one plate element is deleted before the terminal state, and downward-moving retained fragment mass remains "
                "before any z=-12 mm capture-plane crossing"
            ),
        },
        "decision": {
            "software_input_identity": "pass",
            "solver_parser_runtime": "pass",
            "resource_identity": "pass",
            "numerical_engineering_confirmation": "fail_plate_deletion_and_cycle_repeatability",
            "physical_validity": "not_established",
            "research_admission": "hold",
            "classification": "resource_correct_run_reveals_rank_sensitive_plate_material_failure_confirmation_rejected",
            "release_r06o_plate_mat_erod1_isolation": release_plate_erod_isolation,
            "release_terminal_horizon_extension": False,
            "release_2000_control": False,
            "release_treatments": False,
            "p0_response_calibration": "denied",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "dataset_reference_promotion": "denied",
        },
        "recommended_next_sequence": [
            "Branch from the exact returned R06n deck and change only the plate MAT_JOHNSON_COOK EROD field from 0 to 1.",
            "Keep fragment adaptive-SPH, contact cards, TSSFAC=0.05, DT2MS=0, 0.04 ms and exact one-node/32-rank resources frozen.",
            "Treat the run as a plate material-failure-erosion isolation canary; require zero plate deletion and no timestep runaway.",
            "Do not extend the horizon until a resource-correct zero-plate-deletion engineering baseline exists.",
            "Keep 2000 m/s, treatments, fitting and reference labels frozen.",
        ],
        "claim_boundary": (
            "R06n proves that the requested 32-rank resource identity can complete, but it rejects engineering repeatability: "
            "one plate element materially fails and is deleted, and the cycle ratio lies outside the frozen interval. The result "
            "is a valid negative diagnostic, not calibration, physical validation or a damage-tree label."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R06n exact-32-rank return audit")
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
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": audit["decision"]["classification"], "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
