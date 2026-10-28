from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT = Path(__file__).resolve().parents[1]
VENDOR_DIR = PROJECT / ".vendor" / "lasso-python-2.0.4"
if str(PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT / "scripts"))

from extract_r05_d3part import d3part_family_identity  # noqa: E402
from parse_lsdyna_ascii import (  # noqa: E402
    parse_control_timestep,
    parse_glstat_series,
    parse_rank_diagnostics,
    parse_termination,
)


ADDED_MASS_PARTS = re.compile(
    r"^\s*number of parts with added mass.*?(\d+)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
DIAGNOSTIC = re.compile(
    r"^\s*\*\*\*\s*(Error|Fatal)\b",
    flags=re.IGNORECASE | re.MULTILINE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def strict_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def parse_added_mass_parts(text: str) -> int:
    matches = [int(match.group(1)) for match in ADDED_MASS_PARTS.finditer(text)]
    if not matches:
        raise ValueError("solver output does not expose an explicit added-mass part count")
    if len(set(matches)) != 1:
        raise ValueError(f"solver output reports inconsistent added-mass part counts: {matches}")
    return matches[0]


def _natural_key(value: str) -> list[int | str]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value)]


def _matching_names(result_dir: Path, pattern: re.Pattern[str]) -> list[str]:
    return sorted(
        (path.name for path in result_dir.iterdir() if path.is_file() and pattern.fullmatch(path.name)),
        key=_natural_key,
    )


def inventory_return_files(result_dir: Path) -> dict[str, list[str]]:
    root = Path(result_dir).resolve()
    return {
        "solver_output": _matching_names(root, re.compile(r"job_.*_\d+\.out", re.IGNORECASE)),
        "rank_messages": _matching_names(root, re.compile(r"mes\d{4}", re.IGNORECASE)),
        "glstat": _matching_names(root, re.compile(r"glstat", re.IGNORECASE)),
        "binout": _matching_names(root, re.compile(r"binout\d*", re.IGNORECASE)),
        "d3plot": _matching_names(root, re.compile(r"d3plot\d*", re.IGNORECASE)),
        "d3part": _matching_names(root, re.compile(r"d3part\d*", re.IGNORECASE)),
        "solver_stdout": _matching_names(root, re.compile(r"std\.out\..+", re.IGNORECASE)),
        "solver_stderr": _matching_names(root, re.compile(r"std\.err\..+", re.IGNORECASE)),
        "slurm_script": _matching_names(root, re.compile(r".+\.slurm", re.IGNORECASE)),
        "resource_record": sorted(
            set(
                _matching_names(root, re.compile(r"job_.+\.ma", re.IGNORECASE))
                + _matching_names(root, re.compile(r"job_.+\.log", re.IGNORECASE))
            ),
            key=_natural_key,
        ),
    }


def required_return_equivalences(
    *,
    inventory: Mapping[str, list[str]],
    binout_directories: set[str],
    expected_rank_count: int,
) -> dict:
    logical = {name.lower() for name in binout_directories}

    def direct(key: str, *, exact_count: int | None = None) -> dict:
        sources = list(inventory.get(key, []))
        satisfied = len(sources) > 0 if exact_count is None else len(sources) == exact_count
        return {"satisfied": satisfied, "mode": "standalone_file_family", "sources": sources}

    solver_output = list(inventory.get("solver_output", []))
    items = {
        "d3hsp": {
            "satisfied": len(solver_output) == 1,
            "mode": "solver_o_output",
            "sources": solver_output,
            "rule": "the explicit LS-DYNA o= text output is the accepted d3hsp-equivalent stream",
        },
        "mes*": direct("rank_messages", exact_count=expected_rank_count),
        "glstat": direct("glstat"),
        "matsum": {
            "satisfied": "matsum" in logical,
            "mode": "binout_logical_database",
            "sources": list(inventory.get("binout", [])),
        },
        "sleout": {
            "satisfied": "sleout" in logical,
            "mode": "binout_logical_database",
            "sources": list(inventory.get("binout", [])),
        },
        "nodout": {
            "satisfied": "nodout" in logical,
            "mode": "binout_logical_database",
            "sources": list(inventory.get("binout", [])),
        },
        "binout*": direct("binout"),
        "d3plot*": direct("d3plot"),
        "d3part*": direct("d3part"),
        "solver stdout/stderr": {
            "satisfied": bool(inventory.get("solver_stdout")) and bool(inventory.get("solver_stderr")),
            "mode": "paired_scheduler_streams",
            "sources": list(inventory.get("solver_stdout", []))
            + list(inventory.get("solver_stderr", [])),
        },
        "Slurm script and resource record": {
            "satisfied": bool(inventory.get("slurm_script"))
            and bool(inventory.get("resource_record")),
            "mode": "scheduler_submission_and_resource_records",
            "sources": list(inventory.get("slurm_script", []))
            + list(inventory.get("resource_record", [])),
        },
    }
    return {
        "status": "pass" if all(item["satisfied"] for item in items.values()) else "fail",
        "items": items,
        "claim_boundary": (
            "These equivalence rules establish returned-evidence completeness only. "
            "They do not establish numerical response availability or physical validity."
        ),
    }


def build_artifact_manifest(
    *,
    result_dir: Path,
    paths: Iterable[Path],
    run_id: str,
    solver_build: str,
) -> dict:
    root = Path(result_dir).resolve()
    unique: dict[str, Path] = {}
    for value in paths:
        path = Path(value).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"artifact is outside result directory: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        unique[relative] = path
    files = [
        {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for relative, path in sorted(unique.items())
    ]
    return {
        "schema_version": "hf-p0-r05-run-artifact-manifest@1.0.0",
        "run_id": str(run_id),
        "solver_build": str(solver_build),
        "source_directory": root.as_posix(),
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def evaluate_run_integrity(
    observed: Mapping[str, Any],
    gates: Mapping[str, Any],
    *,
    expected_rank_count: int,
) -> dict:
    if gates.get("missing_or_nonfinite_policy") != "unavailable and fail-closed":
        raise ValueError("run-integrity missing/nonfinite policy is not fail-closed")

    rank_ok = (
        observed.get("rank_log_count") == expected_rank_count
        and observed.get("normal_termination_rank_count") == expected_rank_count
        and observed.get("normal_termination") is True
    )
    fatal_limit = int(gates["fatal_or_error_count_max"])
    fatal_count = observed.get("fatal_or_error_count")
    fatal_ok = (
        isinstance(fatal_count, int)
        and fatal_count <= fatal_limit
        and observed.get("error_termination") is False
        and observed.get("nonfinite_token_count") == 0
    )

    ratio_min = observed.get("total_energy_ratio_minimum")
    ratio_max = observed.get("total_energy_ratio_maximum")
    energy_ok = (
        _finite(ratio_min)
        and _finite(ratio_max)
        and float(ratio_min) >= float(gates["total_energy_ratio_min"])
        and float(ratio_max) <= float(gates["total_energy_ratio_max"])
    )
    hourglass = observed.get("hourglass_to_peak_internal_energy")
    hourglass_ok = _finite(hourglass) and abs(float(hourglass)) <= float(
        gates["absolute_hourglass_to_peak_internal_energy_max"]
    )
    added_mass_parts = observed.get("added_mass_parts")
    dt2ms = observed.get("dt2ms")
    added_mass_fraction = 0.0 if added_mass_parts == 0 else None
    mass_ok = (
        added_mass_parts == 0
        and _finite(dt2ms)
        and math.isclose(float(dt2ms), 0.0, rel_tol=0.0, abs_tol=0.0)
        and added_mass_fraction <= float(gates["added_mass_fraction_max"])
    )
    return_ok = observed.get("return_contract_status") == "pass"

    assessed = {
        "all_ranks_normal_termination": {
            "status": "pass" if rank_ok else "fail",
            "observed": {
                "rank_log_count": observed.get("rank_log_count"),
                "normal_termination_rank_count": observed.get(
                    "normal_termination_rank_count"
                ),
                "normal_termination": observed.get("normal_termination"),
            },
            "required_rank_count": expected_rank_count,
        },
        "fatal_error_and_nonfinite": {
            "status": "pass" if fatal_ok else "fail",
            "observed": {
                "fatal_or_error_count": fatal_count,
                "error_termination": observed.get("error_termination"),
                "nonfinite_token_count": observed.get("nonfinite_token_count"),
            },
            "fatal_or_error_count_max": fatal_limit,
        },
        "total_energy_ratio": {
            "status": "pass" if energy_ok else "fail",
            "observed_minimum": ratio_min,
            "observed_maximum": ratio_max,
            "limits": [
                float(gates["total_energy_ratio_min"]),
                float(gates["total_energy_ratio_max"]),
            ],
        },
        "hourglass_to_peak_internal_energy": {
            "status": "pass" if hourglass_ok else "fail",
            "observed": hourglass,
            "maximum": float(gates["absolute_hourglass_to_peak_internal_energy_max"]),
        },
        "added_mass": {
            "status": "pass" if mass_ok else "fail",
            "observed_added_mass_parts": added_mass_parts,
            "observed_added_mass_fraction": added_mass_fraction,
            "observed_dt2ms": dt2ms,
            "maximum_added_mass_fraction": float(gates["added_mass_fraction_max"]),
        },
        "returned_evidence_contract": {
            "status": "pass" if return_ok else "fail",
            "observed": observed.get("return_contract_status"),
        },
    }
    return {
        "status": "pass" if all(item["status"] == "pass" for item in assessed.values()) else "fail",
        "gates": assessed,
    }


def _binout_directories(result_dir: Path) -> set[str]:
    if str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))
    try:
        from lasso.dyna import Binout
    except ImportError as exc:
        raise RuntimeError("project-local lasso-python reader is unavailable") from exc
    binout = Binout(str(result_dir / "binout*"))
    return {str(value) for value in binout.read()}


def _line_evidence(text: str, pattern: re.Pattern[str]) -> dict | None:
    for line_number, line in enumerate(text.splitlines(), 1):
        if pattern.search(line):
            return {"line": line_number, "text": line.strip()}
    return None


def _selected_artifact_paths(
    result_dir: Path, inventory: Mapping[str, list[str]]
) -> list[Path]:
    names: set[str] = set()
    for values in inventory.values():
        names.update(values)
    for name in (
        "p0_velocity_mesh_calibration.k",
        "p0_cases_r05.json",
        "p0_case_part_map.csv",
        "jc4340_reference.json",
        "manifest.json",
        "static_validation.json",
        "UPLOAD_README.txt",
        "p1_preregistered_cases.csv",
    ):
        if (result_dir / name).is_file():
            names.add(name)
    return [result_dir / name for name in sorted(names, key=_natural_key)]


def build_run_integrity(
    *,
    result_dir: Path,
    output_dir: Path,
    run_id: str,
    solver_build: str,
    expected_rank_count: int,
) -> dict:
    root = Path(result_dir).resolve()
    destination = Path(output_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    outputs = {
        "manifest": destination / "run_artifact_manifest.json",
        "audit": destination / "run_integrity_audit.json",
        "evidence": destination / "run_integrity_evidence.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite run-integrity outputs: {existing}")

    design = json.loads((root / "p0_cases_r05.json").read_text(encoding="utf-8"))
    run_gates = design["qualification_gates"]["run_integrity"]
    inventory = inventory_return_files(root)
    if len(inventory["solver_output"]) != 1:
        raise ValueError(
            f"exactly one solver o= output is required, found {inventory['solver_output']}"
        )
    solver_output_path = root / inventory["solver_output"][0]
    solver_text = solver_output_path.read_text(encoding="utf-8", errors="replace")
    added_mass_parts = parse_added_mass_parts(solver_text)
    added_mass_line = _line_evidence(solver_text, ADDED_MASS_PARTS)

    stdout_text = "\n".join(
        (root / name).read_text(encoding="utf-8", errors="replace")
        for name in inventory["solver_stdout"]
    )
    command_bound = solver_output_path.name in stdout_text and " o=" in stdout_text
    if not command_bound:
        raise ValueError("scheduler stdout does not bind the solver o= output path")

    binout_directories = _binout_directories(root)
    return_contract = required_return_equivalences(
        inventory=inventory,
        binout_directories=binout_directories,
        expected_rank_count=expected_rank_count,
    )
    rank = parse_rank_diagnostics(root)
    termination = parse_termination(root)
    glstat_records = parse_glstat_series(root / "glstat")
    ratios = [
        float(row["total_energy_ratio"])
        for row in glstat_records
        if row.get("total_energy_ratio") is not None
    ]
    internal = [
        float(row["internal_energy"])
        for row in glstat_records
        if row.get("internal_energy") is not None and float(row["internal_energy"]) > 0.0
    ]
    hourglass = [
        abs(float(row["hourglass_energy"]))
        for row in glstat_records
        if row.get("hourglass_energy") is not None
    ]
    peak_internal = max(internal, default=None)
    hg_ratio = (
        max(hourglass, default=0.0) / peak_internal
        if peak_internal is not None and peak_internal > 0.0
        else None
    )
    timestep = parse_control_timestep(root / "p0_velocity_mesh_calibration.k")
    diagnostic_count = len(DIAGNOSTIC.findall(solver_text + "\n" + stdout_text)) + sum(
        item["occurrences"] for item in rank["error_fatal_counts"]
    )
    required_numbers = [*ratios, *(internal or []), *(hourglass or []), timestep["dt2ms"]]
    nonfinite_count = sum(not math.isfinite(float(value)) for value in required_numbers)
    observed = {
        "rank_log_count": rank["rank_log_count"],
        "normal_termination_rank_count": rank["normal_termination_rank_count"],
        "normal_termination": termination["normal_termination"],
        "error_termination": termination["error_termination"],
        "fatal_or_error_count": diagnostic_count,
        "nonfinite_token_count": nonfinite_count,
        "total_energy_ratio_minimum": min(ratios, default=None),
        "total_energy_ratio_maximum": max(ratios, default=None),
        "hourglass_to_peak_internal_energy": hg_ratio,
        "added_mass_parts": added_mass_parts,
        "dt2ms": timestep["dt2ms"],
        "return_contract_status": return_contract["status"],
    }
    assessment = evaluate_run_integrity(
        observed, run_gates, expected_rank_count=expected_rank_count
    )

    destination.mkdir(parents=True, exist_ok=True)
    manifest = build_artifact_manifest(
        result_dir=root,
        paths=_selected_artifact_paths(root, inventory),
        run_id=run_id,
        solver_build=solver_build,
    )
    outputs["manifest"].write_text(strict_json(manifest), encoding="utf-8", newline="\n")
    manifest_hash = sha256_file(outputs["manifest"])
    d3part_identity = d3part_family_identity(
        root / "d3part", run_id=run_id, solver_build=solver_build
    )
    evidence = {
        "schema_version": "hf-p0-r05-run-integrity-evidence@1.0.0",
        "evidence_id": f"job-{run_id}-hash-bound-run-integrity",
        "status": assessment["status"],
        "run_id": str(run_id),
        "solver_build": str(solver_build),
        "d3part_family_sha256": d3part_identity["d3part_family_sha256"],
        "returned_artifact_sha256": manifest_hash,
    }
    audit = {
        "schema_version": "hf-p0-r05-run-integrity-audit@1.0.0",
        "run_identity": {
            "run_id": str(run_id),
            "solver_build": str(solver_build),
            "solver_output": solver_output_path.name,
            "solver_output_command_bound": command_bound,
            "d3part_family_sha256": d3part_identity["d3part_family_sha256"],
        },
        "artifact_manifest": {
            "path": outputs["manifest"].name,
            "sha256": manifest_hash,
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
        },
        "returned_evidence_contract": return_contract,
        "binout_logical_directories": sorted(binout_directories),
        "observed": observed,
        "assessment": assessment,
        "source_evidence": {
            "added_mass_parts": {
                "path": solver_output_path.name,
                **(added_mass_line or {}),
            },
            "control_timestep": {
                "path": "p0_velocity_mesh_calibration.k",
                **timestep,
            },
            "rank_messages": inventory["rank_messages"],
            "glstat": inventory["glstat"],
            "binout": inventory["binout"],
        },
        "nonblocking_keyword_compatibility_finding": {
            "status_for_next_candidate": "fail",
            "solver_output_occurrences": len(
                re.findall(r"Warning\s+10457\s+\(KEY\+457\)", solver_text)
            ),
            "scheduler_stdout_occurrences": len(
                re.findall(r"Warning\s+10457\s+\(KEY\+457\)", stdout_text)
            ),
            "unique_parser_event_count": max(
                len(re.findall(r"Warning\s+10457\s+\(KEY\+457\)", solver_text)),
                len(re.findall(r"Warning\s+10457\s+\(KEY\+457\)", stdout_text)),
            ),
            "decision": (
                "Run integrity is assessed separately from the R15 D3PART-card compatibility "
                "finding. The historical R05 input remains immutable; R06 must use five fields."
            ),
        },
        "bridge_compatible_evidence": evidence,
        "claim_boundary": (
            "A passing run-integrity result establishes execution and returned-evidence integrity "
            "only. D3PART integration, case QoI availability, numerical response qualification, "
            "and physical validity remain separate fail-closed gates."
        ),
    }
    outputs["audit"].write_text(strict_json(audit), encoding="utf-8", newline="\n")
    outputs["evidence"].write_text(strict_json(evidence), encoding="utf-8", newline="\n")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hash-bound R05 run-integrity evidence without promoting P0 QoI"
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--solver-build", required=True)
    parser.add_argument("--expected-ranks", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_run_integrity(
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        solver_build=args.solver_build,
        expected_rank_count=args.expected_ranks,
    )
    print(strict_json(audit))


if __name__ == "__main__":
    main()
