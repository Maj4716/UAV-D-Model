"""Select an offline tree on one design seed and validate on another."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.tree_candidate_selection import select_and_validate_tree_candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--candidate", required=True, action="append", type=Path)
    parser.add_argument("--development-seed", required=True, type=int)
    parser.add_argument("--validation-seed", required=True, type=int)
    parser.add_argument("--preregistration-id", required=True)
    parser.add_argument("--gates", type=Path)
    parser.add_argument(
        "--weight-field",
        choices=["design_weight", "target_measure_weight"],
        default="design_weight",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    candidates = [
        json.loads(path.read_text(encoding="utf-8-sig")) for path in args.candidate
    ]
    gates = (
        None
        if args.gates is None
        else json.loads(args.gates.read_text(encoding="utf-8-sig"))
    )
    report = select_and_validate_tree_candidate(
        args.dataset,
        candidates,
        development_seed=args.development_seed,
        validation_seed=args.validation_seed,
        preregistration_id=args.preregistration_id,
        gates=gates,
        weight_field=args.weight_field,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
