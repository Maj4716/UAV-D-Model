"""Recompute damage-tree views from lightweight per-replicate state traces."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, TextIO


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.damage_tree import compile_damage_tree
from damage_engine.utils import content_hash


TRACE_SCHEMA_VERSION = "1.0.0"
TRACE_SEMANTICS = (
    "joint_per_replicate_component_mechanism_states_before_tree_aggregation"
)


def _open_text(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return path.open("r", encoding="utf-8-sig")


def iter_result_documents(path: Path) -> Iterable[dict[str, Any]]:
    """Yield engine result objects from JSON, JSONL, or gzip-wrapped variants."""

    with _open_text(path) as stream:
        if path.name.lower().endswith((".jsonl", ".jsonl.gz")):
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path}:{line_number} must contain a JSON object"
                    )
                yield value
            return
        value = json.load(stream)
    if isinstance(value, dict):
        yield value
        return
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        yield from value
        return
    raise ValueError(f"{path} must contain one result object or an array of objects")


def _mean_outputs(
    replicate_outputs: list[dict[str, float]],
    output_names: tuple[str, ...],
) -> dict[str, float]:
    count = len(replicate_outputs)
    return {
        name: float(math.fsum(row[name] for row in replicate_outputs) / count)
        for name in output_names
    }


def _online_output_means(
    result: Mapping[str, Any],
    output_names: tuple[str, ...],
) -> dict[str, float] | None:
    try:
        records = result["damage_tree"]["outputs"]
        if not all(name in records for name in output_names):
            return None
        return {name: float(records[name]["mean"]) for name in output_names}
    except (KeyError, TypeError, ValueError):
        return None


def recompute_tree_outputs(
    result: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate a target tree per replicate and only then average its outputs."""

    trace = result.get("component_state_trace")
    if not isinstance(trace, Mapping):
        raise ValueError("engine result is missing component_state_trace")
    if trace.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError(
            "unsupported component-state trace schema version: "
            f"{trace.get('schema_version')!r}"
        )
    if trace.get("semantics") != TRACE_SEMANTICS:
        raise ValueError("component-state trace has incompatible aggregation semantics")
    records = trace.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("component-state trace must contain replicate records")
    if trace.get("replicate_count") != len(records):
        raise ValueError("component-state trace replicate count is inconsistent")

    target_document = dict(target)
    tree = compile_damage_tree(target_document)
    tree_definition_hash = content_hash(target_document["damage_tree"])
    evaluated_records: list[dict[str, Any]] = []
    outputs_for_mean: list[dict[str, float]] = []
    for expected_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("component-state trace records must be objects")
        replicate_index = record.get("replicate_index")
        if replicate_index != expected_index:
            raise ValueError(
                "component-state trace replicate indices must be contiguous and ordered"
            )
        states = record.get("component_mechanism_states")
        if not isinstance(states, dict):
            raise ValueError(
                f"replicate {replicate_index} is missing component mechanism states"
            )
        evaluation = tree.evaluate(states)
        outputs = {
            name: float(evaluation["outputs"][name]) for name in tree.outputs
        }
        outputs_for_mean.append(outputs)
        evaluated_records.append(
            {
                "replicate_index": replicate_index,
                "parameter_realization_hash": record.get(
                    "parameter_realization_hash"
                ),
                "penetration_affected_replicate": float(
                    record.get("domain_censoring", {}).get(
                        "penetration_affected_replicate", 0.0
                    )
                ),
                "outputs": outputs,
            }
        )

    output_means = _mean_outputs(outputs_for_mean, tree.outputs)
    online_means = _online_output_means(result, tree.outputs)
    comparison = None
    if online_means is not None:
        absolute_errors = {
            name: abs(output_means[name] - online_means[name])
            for name in tree.outputs
        }
        comparison = {
            "online_output_means": online_means,
            "absolute_errors": absolute_errors,
            "maximum_absolute_error": max(absolute_errors.values(), default=0.0),
        }

    source_identity = trace.get("source_identity", {})
    return {
        "schema_version": "1.0.0",
        "semantics": "damage_tree_outputs_evaluated_per_replicate_then_averaged",
        "case_id": str(result.get("case_id", source_identity.get("case_id", ""))),
        "target_id": str(
            result.get("target_id", source_identity.get("target_id", ""))
        ),
        "source_component_state_trace": {
            "schema_version": trace["schema_version"],
            "source_damage_tree_revision": source_identity.get(
                "damage_tree_revision"
            ),
            "source_damage_tree_definition_hash": source_identity.get(
                "damage_tree_definition_hash"
            ),
        },
        "recomputed_tree": {
            "revision": tree.version,
            "definition_hash": tree_definition_hash,
            "output_names": list(tree.outputs),
        },
        "replicate_count": len(evaluated_records),
        "replicate_outputs": evaluated_records,
        "output_means": output_means,
        "online_same_output_comparison": comparison,
    }


def build_report(
    input_paths: Iterable[Path],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    cases = [
        recompute_tree_outputs(result, target)
        for path in input_paths
        for result in iter_result_documents(path)
    ]
    if not cases:
        raise ValueError("no engine result documents were found")
    tree_definition_hash = content_hash(target["damage_tree"])
    return {
        "schema_version": "1.0.0",
        "tool": "recompute_tree_outputs",
        "tree_definition_hash": tree_definition_hash,
        "case_count": len(cases),
        "cases": cases,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate a target damage tree against lightweight component-state "
            "traces stored in engine result JSON/JSONL files."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    target = json.loads(args.target.read_text(encoding="utf-8-sig"))
    report = build_report(args.inputs, target)
    serialized = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
