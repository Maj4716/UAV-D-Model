#!/usr/bin/env python3
"""Benchmark exact cached plan identities against the reference JSON encoder.

This is a software-performance benchmark, not physical/scientific validation.
The baseline disables only the shared canonical-subtree cache; case generation,
sampling seeds, validation, identity schemas and digest algorithms are identical.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.case_factory import CaseFactory
from damage_engine.input_identity import input_identity_from_case_hashes
from damage_engine.label_spec import LabelSpec
from damage_engine.sampling_spec import SamplingSpec


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _scaled_spec(path: Path, case_count: int) -> SamplingSpec:
    if case_count < 4:
        raise ValueError("case_count must be at least four")
    document = _load(path)
    train_count = case_count - 2
    document["budgets"].update(
        {"train_pilot": train_count, "validation": 1, "test": 1}
    )
    document["legal_sampling"]["max_attempts_per_allocation"] = max(
        int(document["legal_sampling"]["max_attempts_per_allocation"]),
        train_count,
    )
    allocation_counts = {
        "reference_validation": 1,
        "reference_test": 1,
        "reference_train": train_count,
    }
    for round_document in document["round_budgets"]:
        round_document["allocations"] = [
            allocation
            for allocation in round_document["allocations"]
            if allocation["role"] in allocation_counts
        ]
        for allocation in round_document["allocations"]:
            allocation["count"] = allocation_counts[allocation["role"]]
    return SamplingSpec(document)


def _factory(
    base_case: dict, spec: SamplingSpec, label: LabelSpec,
) -> CaseFactory:
    return CaseFactory(
        deepcopy(base_case),
        spec,
        label,
        target_hash="sha256:" + "1" * 64,
        damage_tree_definition_hash="sha256:" + "2" * 64,
        physics_registry_hash="sha256:" + "3" * 64,
        engine_code_hash="sha256:" + "4" * 64,
    )


def _time_build(
    base_case: dict, spec: SamplingSpec, label: LabelSpec, *, cached: bool,
):
    started = perf_counter()
    if cached:
        plan = _factory(base_case, spec, label).build_reference_plan()
    else:
        # This reproduces the old exact-encoder path without maintaining a
        # second implementation of CaseFactory.
        with patch(
            "damage_engine.case_factory._canonical_subtree_cache",
            lambda _values: None,
        ):
            plan = _factory(base_case, spec, label).build_reference_plan()
    return plan, perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument(
        "--case", type=Path, default=ROOT / "examples" / "case.demo.json"
    )
    parser.add_argument(
        "--sampling-spec",
        type=Path,
        default=ROOT / "examples" / "sampling_spec.uav_control.demo.json",
    )
    parser.add_argument(
        "--label-spec",
        type=Path,
        default=ROOT / "examples" / "label_spec.demo.json",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    base_case = _load(arguments.case)
    spec = _scaled_spec(arguments.sampling_spec, arguments.cases)
    label = LabelSpec.from_file(arguments.label_spec)
    baseline, baseline_seconds = _time_build(
        base_case, spec, label, cached=False
    )
    optimized, optimized_seconds = _time_build(
        base_case, spec, label, cached=True
    )
    baseline_hashes = [row["formal_case_hash"] for row in baseline.candidate_rows]
    optimized_hashes = [row["formal_case_hash"] for row in optimized.candidate_rows]
    baseline_identity = input_identity_from_case_hashes(baseline_hashes)
    optimized_identity = input_identity_from_case_hashes(optimized_hashes)
    if baseline_hashes != optimized_hashes or baseline_identity != optimized_identity:
        raise RuntimeError("optimized identities differ from the reference encoder")

    result = {
        "schema_version": "plan-identity-performance@1.0.0",
        "scope": "software_performance_only",
        "case_count": arguments.cases,
        "sampling_spec_schema": spec.data["schema_version"],
        "baseline": {
            "method": "reference_canonical_json_no_shared_subtree_cache",
            "seconds": baseline_seconds,
        },
        "optimized": {
            "method": "byte_exact_shared_canonical_subtree_cache",
            "seconds": optimized_seconds,
        },
        "speedup": baseline_seconds / optimized_seconds,
        "wall_time_reduction_fraction": 1.0 - optimized_seconds / baseline_seconds,
        "formal_case_hashes_identical": True,
        "formal_input_identity_identical": True,
        "ordered_case_identity_hash": optimized_identity[
            "ordered_case_identity_hash"
        ],
        "evidence_boundary": (
            "Measures deterministic plan construction and identity hashing only; "
            "does not validate physics, model accuracy, or full archive RSS."
        ),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
