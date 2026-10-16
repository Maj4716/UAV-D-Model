"""Project formal-plan storage without executing any physics labels.

The estimator reads one representative formal case and the immutable plan
sidecars.  It never materializes the requested 5k/50k/300k case counts; every
projection is constant-memory and reports the bounded serialized row size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from damage_engine.sampling_checkpoint import (
    _REFERENCE_SPLIT_REGISTRY_PATH,
    _externalize_split_registry,
    _iter_rows,
    reference_formal_cases_path,
)


def _compact_size(value) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        .encode("utf-8")
    )


def estimate(root: Path, counts: list[int]) -> dict:
    ledger = reference_formal_cases_path(root)
    first = next(_iter_rows(ledger))
    registry_path = root / "plan/split_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    assignments = registry.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("source run has no split registry assignments")
    actual_count = len(assignments)

    sampling = first.get("metadata", {}).get("sampling", {})
    external = (
        first
        if sampling.get("split_registry_reference") is not None
        else _externalize_split_registry(first, _REFERENCE_SPLIT_REGISTRY_PATH)
    )
    external_row_bytes = _compact_size(external) + 1
    inline_row_bytes = external_row_bytes + _compact_size(registry)
    registry_document_bytes = _compact_size(
        {key: value for key, value in registry.items() if key != "frozen_at"}
    )
    inline_base_without_registry = inline_row_bytes - registry_document_bytes

    empty_registry = dict(registry)
    empty_registry["assignments"] = []
    registry_fixed_compact = _compact_size(empty_registry)
    assignment_compact_average = (
        _compact_size(registry) - registry_fixed_compact
    ) / actual_count
    # SplitRegistry.write uses pretty JSON. Its observed bytes/assignment give
    # a conservative linear projection for the one external sidecar.
    registry_fixed_pretty = len(
        (json.dumps(empty_registry, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        .encode("utf-8")
    )
    assignment_pretty_average = (
        registry_path.stat().st_size - registry_fixed_pretty
    ) / actual_count
    fixed_plan_bytes = sum(
        (root / "plan" / name).stat().st_size
        for name in ("sampling_spec.json", "label_spec.json")
    )

    projections = []
    for count in counts:
        registry_compact = registry_fixed_compact + assignment_compact_average * count
        registry_pretty = registry_fixed_pretty + assignment_pretty_average * count
        new_bytes = round(external_row_bytes * count + registry_pretty + fixed_plan_bytes)
        legacy_bytes = round(
            count * (inline_base_without_registry + registry_compact)
            + registry_pretty
            + fixed_plan_bytes
        )
        projections.append({
            "case_count": count,
            "externalized_estimated_bytes": new_bytes,
            "externalized_estimated_mib": new_bytes / 1024**2,
            "legacy_inline_estimated_bytes": legacy_bytes,
            "legacy_inline_estimated_gib": legacy_bytes / 1024**3,
            "estimated_reduction_percent": 100.0 * (1.0 - new_bytes / legacy_bytes),
        })
    return {
        "schema_version": "sampling-plan-storage-estimate@1.0.0",
        "source_run": str(root.resolve()),
        "source_assignment_count": actual_count,
        "source_formal_cases_actual_bytes": ledger.stat().st_size,
        "bounded_externalized_row_bytes": external_row_bytes,
        "method": "one representative compact transport row plus one linearly projected immutable registry sidecar; no physics labels executed and requested case counts are not materialized",
        "scope": "round-000 formal_cases plus split/sampling/label plan sidecars; excludes candidate ledgers and formal result archives",
        "projections": projections,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument(
        "--counts", type=int, nargs="+", default=[5_000, 50_000, 300_000]
    )
    arguments = parser.parse_args()
    if any(value < 1 for value in arguments.counts):
        parser.error("all counts must be positive")
    print(json.dumps(
        estimate(arguments.source_run, arguments.counts),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
