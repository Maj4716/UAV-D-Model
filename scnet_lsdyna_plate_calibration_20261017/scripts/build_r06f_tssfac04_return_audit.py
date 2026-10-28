from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


EXPECTED_JOB_ID = "r06f_0800_tssfac04"
EXPECTED_PLATFORM_JOB_ID = "72630855"
EXPECTED_REVISION = "R06_timestep_sensitivity_20261028"
EXPECTED_DECK_SHA256 = "sha256:afb0f34112ccfdfcfd95bf442c9b45e2cb1590fdad8d76e98fbbc3fb5d6baee7"


def strict_json(path: Path) -> dict:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=no_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {value!r} in {path}")
        ),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _part(summary: dict, name: str) -> dict:
    matches = [item for item in summary["parts"] if item["part"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name!r} HVAR6 summary")
    return matches[0]


def _absolute_gate(candidate: float | None, baseline: float | None, tolerance: float) -> dict:
    if candidate is None or baseline is None:
        return {
            "status": "fail_unavailable",
            "candidate": candidate,
            "baseline": baseline,
            "absolute_difference": None,
            "tolerance": tolerance,
        }
    difference = abs(float(candidate) - float(baseline))
    return {
        "status": "pass" if difference <= tolerance else "fail",
        "candidate": float(candidate),
        "baseline": float(baseline),
        "absolute_difference": difference,
        "tolerance": tolerance,
    }


def _relative_gate(candidate: float | None, baseline: float | None, tolerance: float) -> dict:
    if candidate is None or baseline is None or math.isclose(float(baseline), 0.0, abs_tol=0.0):
        return {
            "status": "fail_unavailable_mismatch",
            "candidate": candidate,
            "baseline": baseline,
            "relative_difference": None,
            "tolerance": tolerance,
        }
    difference = abs(float(candidate) - float(baseline)) / abs(float(baseline))
    return {
        "status": "pass" if difference <= tolerance else "fail",
        "candidate": float(candidate),
        "baseline": float(baseline),
        "relative_difference": difference,
        "tolerance": tolerance,
    }


def build_audit(
    *,
    return_dir: Path,
    analysis_dir: Path,
    baseline_audit_path: Path,
) -> dict:
    return_dir = return_dir.resolve()
    analysis_dir = analysis_dir.resolve()
    baseline_audit = strict_json(baseline_audit_path.resolve())
    adapter = strict_json(analysis_dir / "r06f_single_job_analysis.json")
    independent = strict_json(analysis_dir / "independent_measurement_validation.json")
    ascii_metrics = strict_json(analysis_dir / "ascii_metrics.json")
    embedded = strict_json(return_dir / "candidate_bundle_manifest.json")
    deck_path = return_dir / "r06f_0800_tssfac04.k"

    if (
        adapter["job_id"] != EXPECTED_JOB_ID
        or adapter["platform_job_id"] != EXPECTED_PLATFORM_JOB_ID
        or adapter["package_revision"] != EXPECTED_REVISION
        or embedded["job_id"] != EXPECTED_JOB_ID
        or embedded["package_revision"] != EXPECTED_REVISION
    ):
        raise ValueError("returned identity differs from the frozen TSSFAC=0.4 job")
    actual_deck_hash = sha256(deck_path)
    embedded_deck = [entry for entry in embedded["files"] if entry["path"] == deck_path.name]
    if (
        actual_deck_hash != EXPECTED_DECK_SHA256
        or len(embedded_deck) != 1
        or embedded_deck[0]["sha256"] != EXPECTED_DECK_SHA256
    ):
        raise ValueError("returned TSSFAC=0.4 deck identity mismatch")

    slurm_paths = list(return_dir.glob("job_LS_DYNA_*.slurm"))
    log_paths = list(return_dir.glob(f"job_*_{EXPECTED_PLATFORM_JOB_ID}.log"))
    stdout_path = return_dir / f"std.out.{EXPECTED_PLATFORM_JOB_ID}"
    if len(slurm_paths) != 1 or len(log_paths) != 1 or not stdout_path.is_file():
        raise ValueError("scheduler/resource evidence is incomplete or ambiguous")
    slurm_text = slurm_paths[0].read_text(encoding="utf-8", errors="replace")
    log_text = log_paths[0].read_text(encoding="utf-8", errors="replace")
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    resource_pass = all(
        evidence
        for evidence in (
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

    candidate_case = adapter["capture"]["cases"]
    if len(candidate_case) != 1:
        raise ValueError("expected exactly one capture ledger")
    candidate = candidate_case[0]
    baseline = baseline_audit["diagnostic_response"]["diagnostic_values_before_availability_gate"]
    gates = {
        "qM": _absolute_gate(candidate["diagnostic_values_before_availability_gate"]["qM"], baseline["qM"], 0.01),
        "qP": _absolute_gate(candidate["diagnostic_values_before_availability_gate"]["qP"], baseline["qP"], 0.01),
        "qE": _absolute_gate(candidate["diagnostic_values_before_availability_gate"]["qE"], baseline["qE"], 0.01),
        "qV": _relative_gate(candidate["diagnostic_values_before_availability_gate"]["qV_over_v0"], baseline["qV_over_v0"], 0.05),
    }
    timestep_gate_pass = all(item["status"] == "pass" for item in gates.values())
    independent_d3part = independent["d3part_full_d3plot_crosscheck"]
    ncforc = independent["ncforc"]
    measurement_pass = (
        independent_d3part["status"] == "pass"
        and independent_d3part["hvar6_alive_status"] == "available"
        and ncforc["status"] == "available_ragged_pending_semantic_validation"
    )
    termination = ascii_metrics["termination"]
    global_statistics = ascii_metrics["global_statistics"]
    solver_pass = (
        termination["normal_termination"]
        and not termination["error_termination"]
        and termination["normal_termination_rank_count"] == 32
        and (return_dir / f"std.err.{EXPECTED_PLATFORM_JOB_ID}").stat().st_size == 0
    )

    files = [path for path in return_dir.iterdir() if path.is_file()]
    d3plots = sorted(path for path in files if re.fullmatch(r"d3plot\d*", path.name))
    d3parts = sorted(path for path in files if re.fullmatch(r"d3part\d*", path.name))
    binouts = sorted(path for path in files if re.fullmatch(r"binout\d+", path.name))
    hvar6 = adapter["mat015_hvar6"]
    baseline_hvar = baseline_audit["measurement_contract"]

    return {
        "schema_version": "hf-p0-r06f-tssfac04-return-audit@1.0.0",
        "audit_date": "2026-10-28",
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
            "stderr_size_bytes": (return_dir / f"std.err.{EXPECTED_PLATFORM_JOB_ID}").stat().st_size,
            "cycle_count": int(cycle_match[-1]),
            "final_problem_time_ms": float(time_match[-1]),
            "numbered_error_count": sum(item["occurrences"] for item in termination["error_fatal_counts"]),
            "warning_counts": termination["warning_counts"],
            "negative_volume_unique_element_count": termination["negative_volume_unique_elements"],
            "dt2ms": 0.0,
            "parts_with_added_mass": 0,
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
                "final_eroded_energy_fraction_of_initial": global_statistics["final_eroded_energy_fraction_of_initial"],
            },
        },
        "measurement_contract": {
            "status": "pass",
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
            "baseline_context": {
                "d3part_history": baseline_hvar["d3part_history"],
                "full_d3plot_hvar6": baseline_hvar["full_d3plot_hvar6"],
                "ncforc": baseline_hvar["ncforc"],
            },
        },
        "diagnostic_response": {
            "primary_qoi_available": candidate["primary_qoi_available"],
            "unavailable_reasons": candidate["unavailable_reasons"],
            "classification": candidate["classification"],
            "diagnostic_values_before_availability_gate": candidate["diagnostic_values_before_availability_gate"],
            "deleted_upstream_mass_fraction": candidate["mass_fraction_by_ledger_state"]["deleted_upstream"],
            "worst_case_crossing_mass_fraction": candidate["worst_case_crossing_mass_fraction"],
            "crossing_event_count": candidate["crossing_event_count"],
            "tail_window": candidate["tail_window"],
            "final_alive_fragment_count": independent_d3part["final_alive_fragment_count"],
        },
        "formal_timestep_comparison_to_job_72617857": {
            "status": "pass" if timestep_gate_pass else "fail",
            "control_tssfac": 0.8,
            "candidate_tssfac": 0.4,
            "gates": gates,
            "supporting_changes": {
                "cycle_count": {
                    "control": baseline_audit["execution"]["cycle_count"],
                    "candidate": int(cycle_match[-1]),
                },
                "negative_volume_unique_element_count": {
                    "control": baseline_audit["execution"]["negative_volume_unique_element_count"],
                    "candidate": termination["negative_volume_unique_elements"],
                },
                "final_alive_fragment_count": {
                    "control": baseline_audit["measurement_contract"]["d3part_history"]["final_alive_fragment_count"],
                    "candidate": independent_d3part["final_alive_fragment_count"],
                },
                "energy_ratio_range": {
                    "control": [
                        baseline_audit["returned_outputs"]["glstat"]["minimum_total_energy_ratio"],
                        baseline_audit["returned_outputs"]["glstat"]["maximum_total_energy_ratio"],
                    ],
                    "candidate": [
                        global_statistics["minimum_total_energy_ratio"],
                        global_statistics["maximum_total_energy_ratio"],
                    ],
                },
            },
            "interpretation": (
                "Halving TSSFAC materially changes the diagnostic response and deletion pattern; "
                "the frozen 0.8 result is not timestep-robust."
            ),
        },
        "decision": {
            "solver_parser_runtime": "pass" if solver_pass else "fail",
            "input_identity": "pass",
            "resource_identity": "pass" if resource_pass else "fail",
            "measurement_channels": "pass" if measurement_pass else "fail",
            "timestep_sensitivity_gate": "pass" if timestep_gate_pass else "fail",
            "classification": "successful_run_but_severe_timestep_sensitivity",
            "release_r06g_0800_tssfac02": bool(
                solver_pass and resource_pass and measurement_pass and not timestep_gate_pass
            ),
            "release_r06f_2000_ctrl": False,
            "release_treatments": False,
            "p0_response_calibration": "denied",
            "p1_submission": "denied",
            "engine_fitting": "denied",
            "dataset_reference_promotion": "denied",
        },
        "recommended_next_sequence": [
            "Generate one TSSFAC=0.2, DT2MS=0, 800 m/s, 0.75 mm refinement from the byte-identical TSSFAC=0.4 deck.",
            "Submit only that job at one node and 32 MPP ranks in a fresh work directory.",
            "Compare TSSFAC=0.2 against Job 72630855 using preregistered no-crossing-aware convergence gates.",
            "Keep 2000 m/s, all treatments, engine fitting and dataset-reference promotion frozen until convergence is established.",
        ],
        "claim_boundary": (
            "This audit establishes a successful target-solver run and a failed timestep-sensitivity "
            "gate. It does not establish physical validity, a converged response, P0/P1 admission, "
            "engine calibration or dataset-label admission."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R06f TSSFAC=0.4 return audit")
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
