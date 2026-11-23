#!/usr/bin/env python
"""Package a completed sampler pilot as an immutable tree-independent dataset-v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.dataset_v2_sampler import (  # noqa: E402
    convert_sampler_pilot_to_dataset_v2,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tree-candidate", type=Path)
    parser.add_argument("--status", default="provisional")
    parser.add_argument(
        "--provisional-reason",
        default=(
            "software-development dataset; physical parameters and semantic "
            "authority remain unverified"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    candidate = None
    if args.tree_candidate is not None:
        candidate = json.loads(args.tree_candidate.read_text(encoding="utf-8-sig"))
    report = convert_sampler_pilot_to_dataset_v2(
        args.pilot,
        args.output,
        tree_candidate=candidate,
        status=args.status,
        provisional_reason=args.provisional_reason,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
