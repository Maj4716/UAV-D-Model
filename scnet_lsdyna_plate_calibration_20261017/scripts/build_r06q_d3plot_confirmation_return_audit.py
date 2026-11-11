from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re

from build_r06g_tssfac02_return_audit import _negative_volume_topology, sha256, strict_json


EXPECTED_JOB_ID = "r06q_0800_d3plot_0p00025_confirmation"
EXPECTED_PLATFORM_JOB_ID = "72936268"
EXPECTED_REVISION = "R06_d3plot_0p00025_measurement_confirmation_20261102"
EXPECTED_DECK_SHA256 = "sha256:a6229cda9778adb13b4c174346a4356deea3e62b175fda9cf8c7dbf12be60002"
TARGET_PLATE_ELEMENT_ID = 39948
MAXIMUM_CYCLE_COUNT = 829930
REQUIRED_PROBLEM_TIME_MS = 0.5
MAXIMUM_DOWNWARD_STEP_MM = 0.25


def _single(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"expected one {label}, found {len(paths)}")
    return paths[0]


def _same_float(left: object, right: object, *, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tolerance


def build_audit(*, return_dir: Path, analysis_dir: Path, parent_analysis_dir: Path) -> dict:
    return_dir = return_dir.resolve()
    analysis_dir = analysis_dir.resolve()
    parent_analysis_dir = parent_analysis_dir.resolve()
    manifest = strict_json(return_dir / "manifest.json")
    contract = strict_json(return_dir / "p0_r06q_d3plot_0p00025_confirmation_output_contract.json")
    job_contract = strict_json(return_dir / "job_contract.json")
    ascii_metrics = strict_json(analysis_dir / "ascii_metrics.json")
    validation = strict_json(analysis_dir / "adaptive_sph_validation.json")
    ledger = strict_json(analysis_dir / "terminal_response_ledger.json")
    parent_ascii = strict_json(parent_analysis_dir / "ascii_metrics.json")
    parent_ledger = strict_json(parent_analysis_dir / "terminal_response_ledger.json")
    deck_path = return_dir / "r06q_0800_d3plot_0p00025_confirmation.k"

    if manifest["job_id"] != EXPECTED_JOB_ID or manifest["package_revision"] != EXPECTED_REVISION:
        raise ValueError("returned identity differs from frozen R06q")
    deck_hash = sha256(deck_path)
    if (
        deck_hash != EXPECTED_DECK_SHA256
        or manifest["deck"]["path"] != deck_path.name
        or manifest["deck"]["sha256"] != deck_hash
        or manifest["deck"]["size_bytes"] != deck_path.stat().st_size
    ):
        raise ValueError("returned R06q deck identity mismatch")
    if job_contract["resource_contract"]["nodes"] != 1 or job_contract["resource_contract"]["ranks"] != 32:
        raise ValueError("embedded R06q resource contract differs")

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
    zero_crossing_pass = bool(
        capture["solid_crossing_event_count"] == 0
        and capture["sph_crossing_event_count"] == 0
        and capture["newly_active_sph_below_plane_count"] == 0
        and capture["solid_deletion_interval_plane_ambiguity_count"] == 0
        and capture["active_or_alive_endpoint_at_or_below_plane_count"] == 0
    )
    temporal_resolution_pass = bool(
        capture["temporal_resolution_gate"] == "pass"
        and capture["maximum_observed_consecutive_downward_carrier_z_step_mm"] <= MAXIMUM_DOWNWARD_STEP_MM
    )
    response_pass = bool(
        primary["available"] is True
        and primary["unavailable_reasons"] == []
        and primary["classification"] == "confirmed_nonperforation"
        and primary["qM"] == 0.0
        and primary["qP"] == 0.0
        and primary["qE"] == 0.0
        and primary["qV_over_v0"] is None
        and zero_crossing_pass
        and tail["gate"] == "pass"
        and temporal_resolution_pass
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

    parent_glstat = parent_ascii["global_statistics"]
    comparison_keys = [
        "final_kinetic_energy", "final_internal_energy", "final_hourglass_energy",
        "final_sliding_energy", "final_eroded_kinetic_energy", "final_total_energy_ratio",
    ]
    ascii_key_metrics_equal = all(_same_float(glstat[key], parent_glstat[key]) for key in comparison_keys)
    numerical_repeat_pass = bool(
        cycle_count == 775413
        and parent_ascii["termination"]["negative_volume_unique_elements"] == termination["negative_volume_unique_elements"]
        and ascii_key_metrics_equal
        and parent_ledger["solid_sph_transfer"]["final_converted_solid_element_ids"]
        == ledger["solid_sph_transfer"]["final_converted_solid_element_ids"]
        and _same_float(parent_ledger["plate_retention"]["target_final_hvar6"], plate["target_final_hvar6"])
        and parent_ledger["primary_response"]["classification"] == "unavailable"
        and parent_ledger["primary_response"]["unavailable_reasons"] == ["temporal_resolution"]
    )
    release_speed_anchor = bool(engineering_pass and response_pass and numerical_repeat_pass)

    files = [path for path in return_dir.iterdir() if path.is_file()]
    return {
        "schema_version": "hf-p0-r06q-d3plot-confirmation-return-audit@1.0.0",
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
        "measurement_confirmation": {
            "parent_r06p_d3plot_interval_ms": 0.0005,
            "r06q_d3plot_interval_ms": 0.00025,
            "parent_state_count": parent_ledger["source"]["state_count"],
            "r06q_state_count": ledger["source"]["state_count"],
            "parent_maximum_downward_step_mm": parent_ledger["capture_observations"]["maximum_observed_consecutive_downward_carrier_z_step_mm"],
            "r06q_maximum_downward_step_mm": capture["maximum_observed_consecutive_downward_carrier_z_step_mm"],
            "required_maximum_downward_step_mm": MAXIMUM_DOWNWARD_STEP_MM,
            "temporal_resolution": "pass" if temporal_resolution_pass else "fail",
            "cycle_count_identical_to_parent": cycle_count == 775413,
            "ascii_key_metrics_identical_to_parent": ascii_key_metrics_equal,
            "converted_fragment_set_identical_to_parent": parent_ledger["solid_sph_transfer"]["final_converted_solid_element_ids"] == ledger["solid_sph_transfer"]["final_converted_solid_element_ids"],
            "target_plate_response_identical_to_parent": _same_float(parent_ledger["plate_retention"]["target_final_hvar6"], plate["target_final_hvar6"]),
            "numerical_repeat": "pass" if numerical_repeat_pass else "fail",
        },
        "response_assessment": {
            "primary_qoi_available": primary["available"],
            "classification": primary["classification"],
            "qM": primary["qM"],
            "qP": primary["qP"],
            "qE": primary["qE"],
            "qV_over_v0": primary["qV_over_v0"],
            "zero_crossing_ambiguity_gate": "pass" if zero_crossing_pass else "fail",
            "terminal_separation": tail["gate"],
            "minimum_capture_plane_clearance_mm": capture["minimum_observed_capture_plane_clearance_mm"],
            "maximum_downward_step_event": capture["maximum_downward_step_event"],
            "overall_response": "pass" if response_pass else "fail",
        },
        "decision": {
            "software_input_identity": "pass",
            "solver_parser_runtime": "pass",
            "resource_identity": "pass",
            "numerical_engineering_confirmation": "pass" if engineering_pass else "fail",
            "single_point_numerical_response_admission": "pass" if response_pass else "fail",
            "physical_validity": "not_established",
            "research_admission": "single_deterministic_numerical_point_only" if release_speed_anchor else "hold",
            "classification": (
                "successful_measurement_confirmation_800mps_numerical_nonperforation_admitted"
                if release_speed_anchor else "r06q_confirmation_failed_or_incomplete"
            ),
            "release_r06r_2000_speed_upper_anchor": release_speed_anchor,
            "release_broad_speed_scan": False,
            "release_treatments": False,
            "p0_response_calibration": "denied",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "dataset_reference_promotion": "denied",
        },
        "recommended_next_sequence": [
            "Run one 2000 m/s numerical upper anchor from the explosion instant using the admitted R06q model.",
            "Refine D3PLOT and D3PART to 0.0001 ms to preserve the <=0.25 mm full-history measurement gate at the higher speed.",
            "Raise the cycle guard only as a non-triggered execution safety limit; keep the 0.50 ms horizon and physical controls frozen.",
            "If 2000 m/s perforates, release a midpoint bracket case; if it does not, stop escalation and review model-form/range adequacy.",
        ],
        "claim_boundary": (
            "R06q admits one deterministic numerical nonperforation point at 800 m/s and 0.75 mm mesh. "
            "It does not establish experimental V50, material calibration, mesh convergence, physical validity, engine fitting, "
            "damage-tree fitting or a reference dataset label."
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build the R06q D3PLOT-confirmation return audit")
    parser.add_argument("--return-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument(
        "--parent-analysis-dir",
        type=Path,
        default=root / "analysis" / "72915268_r06p_0800_terminal_horizon_0p50_20261102",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.resolve().is_relative_to(args.return_dir.resolve()):
        raise ValueError("audit output must remain outside the immutable return directory")
    audit = build_audit(
        return_dir=args.return_dir,
        analysis_dir=args.analysis_dir,
        parent_analysis_dir=args.parent_analysis_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "classification": audit["decision"]["classification"],
        "release_r06r": audit["decision"]["release_r06r_2000_speed_upper_anchor"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
