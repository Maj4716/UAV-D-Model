"""Build dataset-v2 cores, add offline tree views, or emit identity fallbacks."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Any, Iterable, TextIO

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.dataset_v2 import write_dataset_v2, write_tree_view
from damage_engine.offline_tree_candidates import build_identity_component_fallback


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _open_text(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return path.open("r", encoding="utf-8-sig")


def _documents(path: Path) -> Iterable[dict[str, Any]]:
    with _open_text(path) as stream:
        if path.name.lower().endswith((".jsonl", ".jsonl.gz")):
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("trace JSONL rows must be objects")
                    yield value
            return
        value = json.load(stream)
    if isinstance(value, dict):
        yield value
    elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
        yield from value
    else:
        raise ValueError("trace input must contain an object or array of objects")


def _trace_wrappers(path: Path) -> list[dict[str, Any]]:
    wrappers = []
    for value in _documents(path):
        if "component_state_trace" not in value:
            raise ValueError("trace input row must contain component_state_trace")
        wrappers.append(
            {
                "case_id": str(value.get("case_id", "")),
                "component_state_trace": value["component_state_trace"],
            }
        )
    return wrappers


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack = subparsers.add_parser("pack", help="create a new immutable dataset-v2 core")
    pack.add_argument("--inputs", required=True, type=Path)
    pack.add_argument("--component-labels", required=True, type=Path)
    pack.add_argument("--sampling-weights", required=True, type=Path)
    pack.add_argument("--component-state-traces", required=True, type=Path)
    pack.add_argument("--input-contract", required=True, type=Path)
    pack.add_argument("--output", required=True, type=Path)
    pack.add_argument("--dataset-id", required=True)
    pack.add_argument("--label-spec-id", required=True)
    pack.add_argument("--source-code-hash", required=True)
    pack.add_argument("--status", default="provisional")
    pack.add_argument("--provisional-reason")

    view = subparsers.add_parser("add-tree-view", help="add a content-addressed derived view")
    view.add_argument("--dataset", required=True, type=Path)
    view.add_argument("--candidate", required=True, type=Path)

    fallback = subparsers.add_parser(
        "identity-fallback", help="emit provisional component-state pass-through candidate"
    )
    fallback.add_argument("--components", required=True, type=Path)
    fallback.add_argument("--output", required=True, type=Path)
    fallback.add_argument("--mechanism", action="append", default=[])
    fallback.add_argument("--damage-state", action="append", default=[])
    fallback.add_argument("--candidate-id", default="identity-component-fallback-v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "pack":
        report = write_dataset_v2(
            args.output,
            dataset_id=args.dataset_id,
            inputs=_load_npz(args.inputs),
            component_state_traces=_trace_wrappers(args.component_state_traces),
            component_labels=_load_npz(args.component_labels),
            sampling_weights=_load_npz(args.sampling_weights),
            input_contract=_json(args.input_contract),
            label_spec_id=args.label_spec_id,
            source_code_hash=args.source_code_hash,
            status=args.status,
            provisional_reason=args.provisional_reason,
        )
    elif args.command == "add-tree-view":
        report = write_tree_view(args.dataset, _json(args.candidate))
    else:
        components = _json(args.components)
        if isinstance(components, dict):
            components = components.get("component_ids")
        if not isinstance(components, list):
            raise ValueError("components file must be a JSON list or contain component_ids")
        report = build_identity_component_fallback(
            components,
            mechanisms=args.mechanism or ["fragment"],
            damage_states=args.damage_state or ["degraded", "failed"],
            candidate_id=args.candidate_id,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
