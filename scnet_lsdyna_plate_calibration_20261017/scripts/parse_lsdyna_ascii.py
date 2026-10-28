from __future__ import annotations

from collections import Counter
import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
DIAGNOSTIC = re.compile(
    r"^\s*\*\*\*\s*(Warning|Error|Fatal)\s*(\d+)?(?:\s*\(([^)]+)\))?",
    flags=re.IGNORECASE,
)
NEGATIVE_VOLUME = re.compile(
    rf"negative volume in solid element #\s*(\d+)\s+cycle\s*(\d+)\s+time\s*({NUMBER})",
    flags=re.IGNORECASE,
)


def number(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def labeled_values(text: str, labels: Iterable[str]) -> list[float]:
    """Read LS-DYNA label/value lines with dots, whitespace, '=' or ':'."""
    matches: list[tuple[int, float]] = []
    for label in labels:
        pattern = re.compile(
            rf"^\s*{re.escape(label)}(?:\s*\.+\s*|\s+)(?:=|:)?\s*({NUMBER})\s*$",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        matches.extend((match.start(), number(match.group(1))) for match in pattern.finditer(text))
    return [value for _, value in sorted(matches)]


def last_labeled_value(text: str, labels: list[str]) -> float | None:
    matches = labeled_values(text, labels)
    return matches[-1] if matches else None


GLSTAT_FIELDS: dict[str, list[str]] = {
    "time": ["time"],
    "time_step": ["time step"],
    "kinetic_energy": ["kinetic energy"],
    "internal_energy": ["internal energy"],
    "hourglass_energy": ["hourglass energy"],
    "sliding_energy": ["sliding interface energy", "sliding energy"],
    "external_work": ["external work"],
    "eroded_kinetic_energy": ["eroded kinetic energy"],
    "eroded_internal_energy": ["eroded internal energy"],
    "eroded_hourglass_energy": ["eroded hourglass energy"],
    "total_energy": ["total energy"],
    "total_energy_ratio": ["total energy / initial energy", "energy ratio"],
    "energy_ratio_without_eroded": ["energy ratio w/o eroded energy"],
    "added_mass": ["added mass"],
}


def parse_glstat_series(path: Path) -> list[dict[str, float | None]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    arrays = {name: labeled_values(text, labels) for name, labels in GLSTAT_FIELDS.items()}
    record_count = max((len(values) for values in arrays.values()), default=0)
    records: list[dict[str, float | None]] = []
    for index in range(record_count):
        records.append(
            {
                name: values[index] if index < len(values) else None
                for name, values in arrays.items()
            }
        )
    return records


def summarize_glstat(records: list[dict[str, float | None]]) -> dict:
    if not records:
        return {"record_count": 0}

    def values(field: str) -> list[float]:
        return [float(row[field]) for row in records if row.get(field) is not None]

    def final(field: str) -> float | None:
        candidates = values(field)
        return candidates[-1] if candidates else None

    ratios = values("total_energy_ratio")
    steps = values("time_step")
    hourglass = values("hourglass_energy")
    total = values("total_energy")
    initial_total = total[0] if total else None
    hg_ratios: list[float] = []
    for row in records:
        hg = row.get("hourglass_energy")
        internal = row.get("internal_energy")
        if hg is not None and internal is not None and abs(internal) > 1e-12:
            hg_ratios.append(abs(hg / internal))

    eroded_final = sum(
        final(field) or 0.0
        for field in ("eroded_kinetic_energy", "eroded_internal_energy", "eroded_hourglass_energy")
    )
    return {
        "record_count": len(records),
        "final_time": final("time"),
        "minimum_time_step": min(steps) if steps else None,
        "maximum_time_step": max(steps) if steps else None,
        "final_total_energy_ratio": final("total_energy_ratio"),
        "minimum_total_energy_ratio": min(ratios) if ratios else None,
        "maximum_total_energy_ratio": max(ratios) if ratios else None,
        "maximum_total_energy_ratio_absolute_deviation": max((abs(value - 1.0) for value in ratios), default=None),
        "final_energy_ratio_without_eroded": final("energy_ratio_without_eroded"),
        "initial_total_energy": initial_total,
        "final_total_energy": final("total_energy"),
        "final_kinetic_energy": final("kinetic_energy"),
        "final_internal_energy": final("internal_energy"),
        "final_hourglass_energy": final("hourglass_energy"),
        "maximum_absolute_hourglass_energy": max((abs(value) for value in hourglass), default=None),
        "maximum_absolute_hourglass_over_internal": max(hg_ratios, default=None),
        "final_sliding_energy": final("sliding_energy"),
        "final_eroded_kinetic_energy": final("eroded_kinetic_energy"),
        "final_eroded_internal_energy": final("eroded_internal_energy"),
        "final_eroded_hourglass_energy": final("eroded_hourglass_energy"),
        "final_eroded_energy_sum": eroded_final,
        "final_eroded_energy_fraction_of_initial": (
            eroded_final / initial_total if initial_total not in (None, 0.0) else None
        ),
        "final_sliding_energy_fraction_of_initial": (
            (final("sliding_energy") or 0.0) / initial_total
            if initial_total not in (None, 0.0)
            else None
        ),
        "final_added_mass": final("added_mass"),
    }


def parse_glstat(path: Path) -> dict:
    """Backward-compatible final-value view plus robust R15 summary fields."""
    records = parse_glstat_series(path)
    summary = summarize_glstat(records)
    return {
        "final_total_energy_ratio": summary.get("final_total_energy_ratio"),
        "final_kinetic_energy": summary.get("final_kinetic_energy"),
        "final_internal_energy": summary.get("final_internal_energy"),
        "final_hourglass_energy": summary.get("final_hourglass_energy"),
        "final_sliding_energy": summary.get("final_sliding_energy"),
        "final_added_mass": summary.get("final_added_mass"),
        **summary,
    }


def rank_log_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and re.fullmatch(r"mes\d{4}", path.name)
    )


def parse_rank_diagnostics(directory: Path) -> dict:
    warnings: Counter[tuple[str | None, str | None]] = Counter()
    errors: Counter[tuple[str, str | None, str | None]] = Counter()
    negative_events: list[dict] = []
    normal_rank_count = 0
    files = rank_log_paths(directory)
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        compact = re.sub(r"\s+", "", text).lower()
        normal_rank_count += int("normaltermination" in compact)
        for line_number, line in enumerate(text.splitlines(), 1):
            diagnostic = DIAGNOSTIC.match(line)
            if diagnostic:
                severity, diagnostic_id, code = diagnostic.groups()
                if severity.lower() == "warning":
                    warnings[(diagnostic_id, code)] += 1
                else:
                    errors[(severity.lower(), diagnostic_id, code)] += 1
            event = NEGATIVE_VOLUME.search(line)
            if event:
                negative_events.append(
                    {
                        "file": path.name,
                        "line": line_number,
                        "element_id": int(event.group(1)),
                        "cycle": int(event.group(2)),
                        "time": number(event.group(3)),
                    }
                )
    unique_event_keys = {
        (row["element_id"], row["cycle"], row["time"]) for row in negative_events
    }
    return {
        "rank_log_count": len(files),
        "normal_termination_rank_count": normal_rank_count,
        "warning_counts": [
            {"id": key[0], "code": key[1], "occurrences": count}
            for key, count in sorted(warnings.items(), key=lambda item: (item[0][0] or "", item[0][1] or ""))
        ],
        "error_fatal_counts": [
            {"severity": key[0], "id": key[1], "code": key[2], "occurrences": count}
            for key, count in sorted(errors.items())
        ],
        "negative_volume_events": negative_events,
        "negative_volume_event_count": len(negative_events),
        "negative_volume_unique_event_count": len(unique_event_keys),
        "negative_volume_unique_element_count": len({row["element_id"] for row in negative_events}),
    }


def parse_termination(directory: Path) -> dict:
    diagnostics = parse_rank_diagnostics(directory)
    ancillary_texts = []
    for pattern in ("d3hsp", "messag", "std.out.*"):
        for path in sorted(directory.glob(pattern)):
            ancillary_texts.append(path.read_text(encoding="utf-8", errors="replace"))
    ancillary = "\n".join(ancillary_texts)
    compact = re.sub(r"\s+", "", ancillary).lower()
    normal = diagnostics["normal_termination_rank_count"] > 0 or "normaltermination" in compact
    error_count = sum(item["occurrences"] for item in diagnostics["error_fatal_counts"])
    return {
        "normal_termination": normal,
        "error_termination": error_count > 0 or "fatalerror" in compact or "errortermination" in compact,
        "negative_volume_mentions": diagnostics["negative_volume_event_count"],
        "negative_volume_unique_elements": diagnostics["negative_volume_unique_element_count"],
        "mass_scaling_mentions": 0,
        "source_files_found": diagnostics["rank_log_count"] + len(ancillary_texts),
        "rank_log_count": diagnostics["rank_log_count"],
        "normal_termination_rank_count": diagnostics["normal_termination_rank_count"],
        "warning_counts": diagnostics["warning_counts"],
        "error_fatal_counts": diagnostics["error_fatal_counts"],
    }


def parse_control_timestep(deck_path: Path) -> dict:
    lines = deck_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.strip().upper() != "*CONTROL_TIMESTEP":
            continue
        for candidate in lines[index + 1 :]:
            stripped = candidate.strip()
            if not stripped or stripped.startswith("$"):
                continue
            if stripped.startswith("*"):
                break
            fields = [field.strip() for field in stripped.split(",")]
            if len(fields) < 5:
                raise ValueError("*CONTROL_TIMESTEP card has fewer than five fields")
            dt2ms = number(fields[4])
            return {
                "card_line": index + 2,
                "dt2ms": dt2ms,
                "configured_mass_scaling": not math.isclose(dt2ms, 0.0, abs_tol=0.0),
            }
    raise ValueError("*CONTROL_TIMESTEP not found")


def map_elements_to_parts(deck_path: Path, element_ids: set[int]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    in_element_solid = False
    with deck_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.upper() == "*ELEMENT_SOLID":
                in_element_solid = True
                continue
            if stripped.startswith("*"):
                in_element_solid = False
                continue
            if not in_element_solid or not stripped or stripped.startswith("$"):
                continue
            fields = [field.strip() for field in stripped.split(",")]
            if len(fields) < 2:
                continue
            element_id = int(fields[0])
            if element_id in element_ids:
                mapping[element_id] = int(fields[1])
    return mapping


def load_cases(case_map_path: Path) -> list[dict[str, str]]:
    with case_map_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bind_negative_volume_events(
    events: list[dict], deck_path: Path, case_map_path: Path
) -> tuple[list[dict], list[dict]]:
    element_part = map_elements_to_parts(deck_path, {row["element_id"] for row in events})
    cases = load_cases(case_map_path)
    part_context: dict[int, tuple[str, str, str]] = {}
    for case in cases:
        part_context[int(case["plate_part_id"])] = (case["case_id"], case["role"], "plate")
        part_context[int(case["fragment_part_id"])] = (case["case_id"], case["role"], "fragment")

    bound: list[dict] = []
    for event in events:
        part_id = element_part.get(event["element_id"])
        case_id, role, part_kind = part_context.get(part_id, (None, None, "unmapped"))
        bound.append(
            {
                **event,
                "part_id": part_id,
                "case_id": case_id,
                "case_role": role,
                "part_kind": part_kind,
            }
        )

    summary: list[dict] = []
    for case in cases:
        case_events = [row for row in bound if row["case_id"] == case["case_id"]]
        plate_events = [row for row in case_events if row["part_kind"] == "plate"]
        fragment_events = [row for row in case_events if row["part_kind"] == "fragment"]
        summary.append(
            {
                "case_id": case["case_id"],
                "role": case["role"],
                "impact_speed_mps": float(case["impact_speed_mps"]),
                "mesh_mm": float(case["mesh_mm"]),
                "plate_part_id": int(case["plate_part_id"]),
                "fragment_part_id": int(case["fragment_part_id"]),
                "plate_negative_volume_events": len(plate_events),
                "fragment_negative_volume_events": len(fragment_events),
                "total_negative_volume_events": len(case_events),
                "first_event_time": min((row["time"] for row in case_events), default=None),
                "last_event_time": max((row["time"] for row in case_events), default=None),
            }
        )
    return bound, summary


def assess(termination: dict, glstat: dict) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not termination["normal_termination"] or termination["error_termination"]:
        reasons.append("solver did not demonstrate clean normal termination")
    if termination["negative_volume_mentions"]:
        reasons.append("negative-volume events require case-level review")
    ratio = glstat.get("final_total_energy_ratio")
    if ratio is None:
        reasons.append("energy ratio unavailable")
    elif not 0.95 <= ratio <= 1.05:
        reasons.append(f"final energy ratio {ratio:.6g} outside preregistered candidate band [0.95, 1.05]")
    hourglass_ratio = glstat.get("maximum_absolute_hourglass_over_internal")
    if hourglass_ratio is None:
        reasons.append("hourglass/internal-energy ratio unavailable")
    elif hourglass_ratio > 0.05:
        reasons.append("hourglass/internal-energy magnitude exceeds candidate 5% gate")
    return ("pass" if not reasons else "review_or_fail"), reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--case-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = load_cases(args.case_map)
    termination = parse_termination(args.result_dir)
    glstat_path = args.result_dir / "glstat"
    glstat = parse_glstat(glstat_path) if glstat_path.exists() else {}
    status, reasons = assess(termination, glstat)
    result = {
        "schema_version": "hf-p0-postprocess@1.1.0",
        "numerical_qa_status": status,
        "termination": termination,
        "global_statistics": glstat,
        "case_identity": cases,
        "limitations": [
            "Global ASCII QA is necessary but not sufficient for case-level perforation acceptance.",
            "Residual velocity and perforation state still require BINOUT/NODOUT and d3plot review.",
            "Candidate 5% gates are preregistered engineering checks, not universal LS-DYNA standards.",
        ],
        "review_reasons": reasons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
