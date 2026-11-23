"""Audit proxy non-reach with formal Monte Carlo teacher component hits."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.engine import DamageEngine  # noqa: E402
from damage_engine.exposure_audit import (  # noqa: E402
    build_exposure_teacher_audit,
    select_teacher_audit_ids,
    teacher_component_hit_probabilities,
)
from damage_engine.registry import PhysicsRegistry  # noqa: E402


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_cases(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        document = _load_json(path)
        values = document if isinstance(document, list) else document.get("cases", [document])
    output: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(values):
        case = row.get("case", row)
        identifier = str(row.get("candidate_id", case.get("case_id", f"case-{index}")))
        if identifier in output:
            raise ValueError(f"duplicate case candidate_id {identifier}")
        output[identifier] = case
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposure-field", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("configs/targets/target.fragment_only.v1.json"),
    )
    parser.add_argument(
        "--physics",
        type=Path,
        default=Path("configs/physics/loitering_impact.json"),
    )
    parser.add_argument("--audit-size", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026112003)
    parser.add_argument("--teacher-positive-threshold", type=float, default=0.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.replicates < 1:
        raise ValueError("--replicates must be positive")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    field = _load_json(args.exposure_field)
    rows = field.get("candidate_rows")
    if not isinstance(rows, list):
        raise ValueError("exposure field must include candidate_rows")
    audit_ids = select_teacher_audit_ids(
        rows, sample_size=min(args.audit_size, len(rows)), seed=args.seed
    )
    cases = _load_cases(args.cases)
    missing = sorted(set(audit_ids) - set(cases))
    if missing:
        raise ValueError(f"audit cases are missing: {missing}")
    engine = DamageEngine(
        _load_json(args.target), PhysicsRegistry(_load_json(args.physics))
    )
    teacher_rows = []
    for offset, identifier in enumerate(audit_ids):
        case = deepcopy(cases[identifier])
        case.setdefault("analysis", {})["replicates"] = args.replicates
        case["analysis"]["aleatory_seed"] = args.seed + offset
        case["analysis"]["capture_fragment_audit"] = False
        case["analysis"]["capture_replicate_trace"] = False
        result = engine.run(case, fail_closed=True)
        teacher_rows.append(
            {
                "candidate_id": identifier,
                "component_hit_probability": teacher_component_hit_probabilities(result),
            }
        )
    report = build_exposure_teacher_audit(
        rows,
        teacher_rows,
        teacher_positive_threshold=args.teacher_positive_threshold,
        confidence=args.confidence,
    )
    report["audit_candidate_ids"] = audit_ids
    report["teacher_replicates_per_candidate"] = args.replicates
    report["teacher_rows"] = teacher_rows
    report["source_exposure_field"] = str(args.exposure_field)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "audited": len(audit_ids)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
