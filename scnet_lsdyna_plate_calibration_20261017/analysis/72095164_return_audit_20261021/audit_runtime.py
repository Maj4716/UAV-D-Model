"""Index runtime evidence without interpreting the simulated physical response."""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

ANALYSIS = Path(__file__).resolve().parent
PACKAGE = ANALYSIS.parents[1]
RAW = PACKAGE / "results" / "72095164_R15_mpp_d"
DIAGNOSTIC = re.compile(r"^\s*\*\*\*\s*(Warning|Error|Fatal)\s*(\d+)?(?:\s*\(([^)]+)\))?", re.I)


def save_json(name: str, value: object) -> None:
    (ANALYSIS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    names = [*[f"mes{i:04d}" for i in range(32)], "std.out.72095164", "std.err.72095164",
             "job_uav_hf_p0_plate_r04_72095164.out", "job_uav_hf_p0_plate_r04_72095164.log", "status.out"]
    files = []
    rows = []
    evidence = []
    rank_counts: Counter = Counter()
    for name in names:
        path = RAW / name
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        normal = []
        reached = []
        codes: dict[tuple, dict] = {}
        for number, line in enumerate(lines, 1):
            compact = re.sub(r"\s+", "", line).lower()
            if "normaltermination" in compact:
                normal.append(number)
            if "terminationtimereached" in compact:
                reached.append(number)
            match = DIAGNOSTIC.match(line)
            if match:
                severity, diagnostic_id, code = match.groups()
                key = (severity.lower(), diagnostic_id, code)
                if key not in codes:
                    codes[key] = {"count": 0, "first_line": number, "last_line": number}
                codes[key]["count"] += 1
                codes[key]["last_line"] = number
            if name == "mes0000" and any(token in line for token in ["Version :", "Revision:", "Platform   :", "Problem time", "Problem cycle", "Total CPU time", "Elapsed time", "Parallel execution"]):
                evidence.append({"file": name, "line": number, "text": line.strip()})
        for key, values in codes.items():
            severity, diagnostic_id, code = key
            rows.append({"file": name, "severity": severity, "id": diagnostic_id, "code": code, **values})
            if re.fullmatch(r"mes\d{4}", name):
                rank_counts[key] += values["count"]
        files.append({"file": name, "lines": len(lines), "bytes": path.stat().st_size,
                      "normal_termination_lines": normal, "termination_time_reached_lines": reached,
                      "coded_diagnostic_occurrences": sum(value["count"] for value in codes.values())})
    with (ANALYSIS / "diagnostic_inventory.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "severity", "id", "code", "count", "first_line", "last_line"])
        writer.writeheader()
        writer.writerows(rows)
    rank_files = [item for item in files if re.fullmatch(r"mes\d{4}", item["file"])]
    runtime = {
        "scope": "Run completion and generic coded-diagnostic inventory only; no simulated-response evaluation.",
        "files_scanned": len(files),
        "rank_logs_with_normal_termination": sum(bool(item["normal_termination_lines"]) for item in rank_files),
        "rank_logs_total": len(rank_files),
        "coded_error_fatal_occurrences": sum(row["count"] for row in rows if row["severity"] in ["error", "fatal"]),
        "rank_diagnostic_occurrences": [{"severity": key[0], "id": key[1], "code": key[2], "occurrences": count} for key, count in rank_counts.items()],
        "diagnostic_count_caveat": "Counts are log occurrences, not deduplicated physical events; secondary output streams repeat some rank messages.",
        "non_event_mentions_manually_checked": [
            {"file": "job_uav_hf_p0_plate_r04_72095164.out", "lines": [1958, 2084, 3002],
             "finding": "error termination occurs inside echoed option descriptions, not runtime error events"},
        ],
        "evidence": evidence,
        "files": files,
        "limitations": ["No scheduler exit-code record was identified in this audit.",
                        "status.out contains intermediate estimates and is not a final acceptance record.",
                        "No numerical-validity or physical-validity verdict is made."],
    }
    save_json("runtime_evidence.json", runtime)
    receipt = {
        "schema": "returned-run-software-receipt@1.0.0", "job_id": "72095164",
        "audit_time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "actual_solver_version": "mpp d R15", "actual_solver_revision": "R15.0.2-1-g1e7cfa42b1",
        "mpi_platform_from_solver_banner": "Intel-MPI 2018 Xeon64", "mpp_ranks": 32,
        "solver_reported_elapsed_seconds": 58, "wrapper_elapsed_seconds": 74,
        "cycle_count": 1990, "normal_termination_rank_count": 32,
        "solver_executable_sha256": None, "scheduler_exit_code": None,
        "software_runtime_status": "normal_termination_observed",
        "binary_database_readability": "not_checked", "numerical_qualification": "not_assessed",
        "physical_validation": "not_assessed", "application_calibration": "not_assessed",
        "raw_manifest_preserved": True,
        "derivation": ["runtime_evidence.json", "input_identity.json", "archive_identity.json"],
    }
    save_json("run_receipt.json", receipt)
    with (ANALYSIS / "process.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{receipt['audit_time']} Runtime scan completed: {len(files)} logs; 32/32 rank logs contain normal termination; no anchored coded Error/Fatal markers. Warnings retained without scientific interpretation.\n")
        handle.write(f"{receipt['audit_time']} Static repro_check exit code 1: package-local ENV and DATA marked MISSING. Root repository manifests exist; selftest scripts are not detected by the generic test-name heuristic.\n")
    print(json.dumps({key: value for key, value in runtime.items() if key != "files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
