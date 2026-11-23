"""Build a finite-frame component exposure field with FastSamplingProxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.engine import DamageEngine  # noqa: E402
from damage_engine.exposure_field import (  # noqa: E402
    build_exposure_field,
    exposure_row_from_proxy,
)
from damage_engine.fast_sampling_proxy import FastSamplingProxy  # noqa: E402
from damage_engine.registry import PhysicsRegistry  # noqa: E402
from damage_engine.utils import content_hash  # noqa: E402


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_case_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        document = _load_json(path)
        if isinstance(document, list):
            values = document
        elif isinstance(document, dict) and isinstance(document.get("cases"), list):
            values = document["cases"]
        else:
            values = [document]
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"case row {index} must be an object")
        case = value.get("case", value)
        if not isinstance(case, dict):
            raise ValueError(f"case row {index} lacks a case object")
        rows.append(
            {
                "candidate_id": str(value.get("candidate_id", case.get("case_id", f"case-{index}"))),
                "cell_id": value.get("cell_id"),
                "measure_weight": float(value.get("measure_weight", 1.0)),
                "case": case,
            }
        )
    return rows


def _infer_cell(case: dict[str, Any]) -> str:
    metadata = case.get("metadata", {})
    manifold = metadata.get("continuous_direction_sampling", {}) if isinstance(metadata, dict) else {}
    if not isinstance(manifold, dict) or not manifold.get("region_id"):
        manifold = metadata.get("contact_manifold_sampling", {}) if isinstance(metadata, dict) else {}
    approach = (
        manifold.get("region_id", manifold.get("approach_id", "unclassified_direction"))
        if isinstance(manifold, dict)
        else "unclassified_direction"
    )
    profile = case.get("munition", {}).get("profile_id", "unclassified_profile")
    return f"{approach}::{profile}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--max-cases", type=int, default=32)
    parser.add_argument("--azimuths-per-ring", type=int, default=4)
    parser.add_argument("--source-seed", type=int, default=2026112001)
    parser.add_argument("--reach-probability-cutoff", type=float, default=1.0e-5)
    parser.add_argument("--component-support-threshold", type=float, default=0.05)
    parser.add_argument("--layer-quadrature-samples", type=int, default=16)
    parser.add_argument("--reach-tolerance", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.max_cases < 1:
        raise ValueError("--max-cases must be positive")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    target = _load_json(args.target)
    physics = _load_json(args.physics)
    engine = DamageEngine(target, PhysicsRegistry(physics))
    proxy = FastSamplingProxy(
        engine,
        source_seed=args.source_seed,
        azimuths_per_ring=args.azimuths_per_ring,
        reach_probability_cutoff=args.reach_probability_cutoff,
        component_support_threshold=args.component_support_threshold,
        layer_quadrature_samples=args.layer_quadrature_samples,
    )
    source_rows = _load_case_rows(args.cases)[: args.max_cases]
    rows = []
    for source in source_rows:
        result = proxy.evaluate(source["case"])
        rows.append(
            exposure_row_from_proxy(
                result,
                candidate_id=source["candidate_id"],
                cell_id=str(source["cell_id"] or _infer_cell(source["case"])),
                measure_weight=source["measure_weight"],
            )
        )
    report = build_exposure_field(
        rows, reach_tolerance=args.reach_tolerance, include_candidate_rows=True
    )
    report["provenance"] = {
        "target_hash": engine.target_hash,
        "physics_registry_hash": content_hash(physics),
        "cases_source": str(args.cases),
        "cases_source_hash": content_hash(
            {"rows": source_rows}
        ),
        "proxy_parameters": {
            "source_seed": args.source_seed,
            "azimuths_per_ring": args.azimuths_per_ring,
            "reach_probability_cutoff": args.reach_probability_cutoff,
            "component_support_threshold": args.component_support_threshold,
            "layer_quadrature_samples": args.layer_quadrature_samples,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "candidate_count": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
