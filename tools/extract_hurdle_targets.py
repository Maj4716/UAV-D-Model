"""Convert DamageEngine result JSON/JSONL into versioned hurdle targets."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Any, Iterable, TextIO


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.hurdle_targets import (  # noqa: E402
    DEFAULT_SUPPORT_PROBABILITY_THRESHOLD,
    extract_component_hurdle_targets,
)


def _open_text(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return path.open("r", encoding="utf-8-sig")


def _results(path: Path) -> Iterable[dict[str, Any]]:
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
    elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
        yield from value
    else:
        raise ValueError(f"{path} must contain a result object or an array")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract R-independent component hurdle targets from engine results."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--support-probability-threshold",
        type=float,
        default=DEFAULT_SUPPORT_PROBABILITY_THRESHOLD,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases = [
        extract_component_hurdle_targets(
            result,
            support_probability_threshold=args.support_probability_threshold,
        )
        for path in args.inputs
        for result in _results(path)
    ]
    if not cases:
        raise ValueError("no engine results were found")
    report = {
        "schema_version": "1.0.0",
        "tool": "extract_hurdle_targets",
        "case_count": len(cases),
        "cases": cases,
    }
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
