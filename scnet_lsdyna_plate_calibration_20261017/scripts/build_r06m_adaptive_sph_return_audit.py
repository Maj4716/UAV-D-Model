from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re

from build_r06g_tssfac02_return_audit import sha256, strict_json


EXPECTED_JOB_ID = "r06m_0800_adaptive_sph_canary"
EXPECTED_PLATFORM_JOB_ID = "72861050"
EXPECTED_REVISION = "R06_adaptive_solid_to_sph_canary_20261101"
EXPECTED_DECK_SHA256 = "sha256:634436d875c42592b5801de9bf27fce6255691fd27f17d28bd201b80e4cf1107"
EXPECTED_NODES = 1
EXPECTED_RANKS = 32
OBSERVED_RANKS = 64


def _single(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"expected one {label}, found {len(paths)}")
    return paths[0]


def build_audit(*, return_dir: Path, analysis_dir: Path) -> dict:
    return_dir = return_dir.resolve()
    analysis_dir = analysis_dir.resolve()
    manifest = strict_json(return_dir / "manifest.json")
    job_contract = strict_json(return_dir / "job_contract.json")
    ascii_metrics = strict_json(analysis_dir / "ascii_metrics.json")
    validation = strict_json(analysis_dir / "adaptive_sph_validation.json")
    deck_path = return_dir / "r06m_0800_adaptive_sph_canary.k"

    if manifest["job_id"] != EXPECTED_JOB_ID or manifest["package_revision"] != EXPECTED_REVISION:
        raise ValueError("returned identity differs from frozen R06m")
    deck_hash = sha256(deck_path)
    if (
        deck_hash != EXPECTED_DECK_SHA256
        or manifest["deck"]["path"] != deck_path.name
        or manifest["deck"]["sha256"] != deck_hash
        or manifest["deck"]["size_bytes"] != deck_path.stat().st_size
    ):
        raise ValueError("returned R06m deck identity mismatch")
    expected_resource = job_contract["resource_contract"]
    if expected_resource["nodes"] != EXPECTED_NODES or expected_resource["ranks"] != EXPECTED_RANKS:
        raise ValueError("embedded R06m resource contract differs")

    slurm_path = _single(list(return_dir.glob("job_LS_DYNA_*.slurm")), "Slurm script")
    log_path = _single(
        list(return_dir.glob(f"job_*_{EXPECTED_PLATFORM_JOB_ID}.log")), "platform job log"
    )
    stdout_path = return_dir / f"std.out.{EXPECTED_PLATFORM_JOB_ID}"
    stderr_path = return_dir / f"std.err.{EXPECTED_PLATFORM_JOB_ID}"
    if not stdout_path.is_file() or not stderr_path.is_file():
        raise ValueError("solver stdout/stderr evidence is incomplete")
    slurm_text = slurm_path.read_text(encoding="utf-8", errors="replace")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    observed_nodes = 1 if "#SBATCH -N 1" in slurm_text else None
    rank_matches = [
        re.search(r"#SBATCH --ntasks-per-node=(\d+)", slurm_text),
        re.search(r"The total cores is:\s*(\d+)", log_text),
        re.search(r"mpirun -np\s+(\d+)", log_text),
        re.search(r"MPP execution with\s+(\d+) procs", stdout_text),
    ]
    if any(match is None for match in rank_matches):
        raise ValueError("resource evidence is incomplete")
    observed_rank_values = [int(match.group(1)) for match in rank_matches if match is not None]
    if len(set(observed_rank_values)) != 1:
        raise ValueError(f"resource evidence disagrees: {observed_rank_values}")
    observed_ranks = observed_rank_values[0]
    if observed_ranks != OBSERVED_RANKS:
        raise ValueError(f"R06m expected observed mismatch evidence at 64 ranks, got {observed_ranks}")
    resource_identity_pass = observed_nodes == EXPECTED_NODES and observed_ranks == EXPECTED_RANKS

    start_match = re.search(r"The start time is:\s*([0-9-]+ [0-9:]+)", log_text)
    end_match = re.search(r"The end time is:\s*([0-9-]+ [0-9:]+)", log_text)
    cycle_match = re.findall(r"Problem cycle\s*=\s*(\d+)", stdout_text)
    time_match = re.findall(r"Problem time\s*=\s*([0-9.Ee+-]+)", stdout_text)
    elapsed_match = re.search(r"Elapsed time\s+(\d+) seconds", stdout_text)
    if not start_match or not end_match or not cycle_match or not time_match or not elapsed_match:
        raise ValueError("completed-run timing evidence is incomplete")
    start = datetime.fromisoformat(start_match.group(1))
    end = datetime.fromisoformat(end_match.group(1))
    portal_elapsed_seconds = int((end - start).total_seconds())
    solver_elapsed_seconds = int(elapsed_match.group(1))
    cycle_count = int(cycle_match[-1])
    final_problem_time_ms = float(time_match[-1])

    termination = ascii_metrics["termination"]
    glstat = ascii_metrics["global_statistics"]
    adaptive = validation["adaptive_sph"]
    conversion = adaptive["conversion"]
    mass = adaptive["mass_accounting"]
    capture = adaptive["capture_plane"]
    matsum = adaptive["matsum"]
    crosscheck = validation["d3part_full_d3plot_crosscheck"]
    ncforc = validation["ncforc"]

    solver_completion_pass = bool(
        termination["normal_termination"]
        and not termination["error_termination"]
        and termination["rank_log_count"] == OBSERVED_RANKS
        and termination["normal_termination_rank_count"] == OBSERVED_RANKS
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
    adaptive_gate_pass = bool(
        adaptive["status"] == "pass_adaptive_sph_capability_and_mass_gate"
        and conversion["statewise_exact_activation_correspondence"] is True
        and conversion["particles_per_converted_solid"] == 8
        and conversion["final_active_sph_count"] == 8 * conversion["final_converted_solid_count"]
        and conversion["plate_deleted_element_count"] == 0
        and conversion["sph_deleted_particle_count"] == 0
        and mass["gate"] == "pass"
        and adaptive["active_sph"]["all_positions_finite"] is True
        and adaptive["active_sph"]["all_velocities_finite"] is True
    )
    measurement_pass = bool(
        crosscheck["status"] == "pass"
        and crosscheck["maximum_absolute_difference_all_7_channels"] == 0.0
        and crosscheck["alive_mask_mismatch_count"] == 0
        and ncforc["status"] == "available_ragged_pending_semantic_validation"
        and ncforc["all_coordinates_forces_pressure_finite"] is True
    )
    release_confirmation = bool(
        solver_completion_pass
        and energy_gate_pass
        and adaptive_gate_pass
        and measurement_pass
        and termination["mass_scaling_mentions"] == 0
        and observed_nodes == EXPECTED_NODES
        and observed_ranks == OBSERVED_RANKS
        and not resource_identity_pass
    )

    files = [path for path in return_dir.iterdir() if path.is_file()]
    d3plots = [path for path in files if re.fullmatch(r"d3plot\d*", path.name)]
    d3parts = [path for path in files if re.fullmatch(r"d3part\d*", path.name)]
    binouts = [path for path in files if re.fullmatch(r"binout\d+", path.name)]
    return {
        "schema_version": "hf-p0-r06m-adaptive-sph-return-audit@1.0.0",
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
            "resource_identity": "fail_observed_64_ranks_expected_32",
            "resource_contract": {"nodes": EXPECTED_NODES, "ranks": EXPECTED_RANKS},
            "observed_resources": {"nodes": observed_nodes, "ranks": observed_ranks},
            "normal_termination_at_required_time": solver_completion_pass,
            "start_time": start.isoformat(sep=" "),
            "end_time": end.isoformat(sep=" "),
            "portal_elapsed_seconds": portal_elapsed_seconds,
            "solver_elapsed_seconds": solver_elapsed_seconds,
            "cycle_count": cycle_count,
            "cycle_guard": 829930,
            "final_problem_time_ms": final_problem_time_ms,
            "rank_log_count": termination["rank_log_count"],
            "normal_termination_rank_count": termination["normal_termination_rank_count"],
            "stderr_size_bytes": stderr_path.stat().st_size,
            "mass_scaling_mentions": termination["mass_scaling_mentions"],
            "warning_counts": termination["warning_counts"],
            "negative_volume_unique_element_count": termination["negative_volume_unique_elements"],
        },
        "returned_outputs": {
            "d3plot": {"file_count": len(d3plots), "state_count": adaptive["d3plot"]["state_count"], "readability": "pass"},
            "d3part": {"file_count": len(d3parts), "state_count": crosscheck["d3part_state_count"], "readability": "pass"},
            "binout": {"file_count": len(binouts), "logical_databases": ncforc["logical_databases"], "readability": "pass"},
        },
        "engineering_gates": {
            "adaptive_sph_capability": "pass" if adaptive_gate_pass else "fail",
            "mass_accounting": mass,
            "energy": {
                "status": "pass" if energy_gate_pass else "fail",
                "total_energy_ratio_allowed": [0.95, 1.05],
                "observed_minimum_total_energy_ratio": glstat["minimum_total_energy_ratio"],
                "observed_maximum_total_energy_ratio": glstat["maximum_total_energy_ratio"],
                "maximum_absolute_sliding_fraction": 0.05,
                "observed_final_sliding_fraction": glstat["final_sliding_energy_fraction_of_initial"],
                "observed_final_eroded_fraction": glstat["final_eroded_energy_fraction_of_initial"],
            },
            "conversion": conversion,
            "measurement_crosscheck": "pass" if measurement_pass else "fail",
            "matsum_policy": matsum,
        },
        "response_availability": {
            "status": "hold_terminal_horizon_not_qualified",
            "primary_qoi_available": capture["primary_qoi_available"],
            "capture_plane": capture,
            "interpretation": (
                "No z=-12 mm crossing is observed by 0.04 ms, but all retained solid fragments and 62 active SPH "
                "particles still move downward. This is not confirmed nonperforation."
            ),
        },
        "decision": {
            "software_input_identity": "pass",
            "solver_parser_runtime": "pass",
            "numerical_engineering_gates": "pass",
            "resource_identity": "fail_observed_64_ranks_expected_32",
            "physical_validity": "not_established",
            "research_admission": "hold",
            "classification": "successful_adaptive_sph_capability_but_resource_identity_mismatch_and_response_horizon_unresolved",
            "release_r06n_exact_32rank_confirmation": release_confirmation,
            "release_terminal_horizon_extension": False,
            "release_2000_control": False,
            "release_treatments": False,
            "p0_response_calibration": "denied",
            "engine_fitting": "denied",
            "damage_tree_fitting": "denied",
            "dataset_reference_promotion": "denied",
        },
        "recommended_next_sequence": [
            "Repeat the exact R06m deck at one node and exactly 32 MPP ranks; change only the TITLE identifier.",
            "Keep the 0.04 ms horizon so the resource correction is the only experimental factor.",
            "Require normal completion, exact resource identity, 8:1 conversion, mass closure, energy/contact gates and zero plate deletion.",
            "If the 32-rank confirmation passes, extend only the terminal horizon in the subsequent run to resolve the capture-plane QoI.",
            "Continue holding 2000 m/s, treatments, engine fitting, damage-tree fitting and reference labels.",
        ],
        "claim_boundary": (
            "R06m establishes adaptive solid-to-SPH parser/runtime and engineering mass-energy capability at the observed "
            "64-rank execution. It violates the frozen 32-rank resource identity and ends before the response horizon is "
            "qualified, so it is not calibration, physical validation, a damage-tree label or a reference-dataset sample."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the completed R06m adaptive-SPH return audit")
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
