"""Select a fixed-size Train-only exposure-balanced enrichment arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.exposure_sampling import select_exposure_balanced_arm  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposure-field", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026112002)
    parser.add_argument("--rho-min", type=float, default=0.08)
    parser.add_argument("--smoothing", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    field = json.loads(args.exposure_field.read_text(encoding="utf-8-sig"))
    rows = field.get("candidate_rows")
    if not isinstance(rows, list):
        raise ValueError("exposure field must include candidate_rows")
    result = select_exposure_balanced_arm(
        rows,
        sample_size=args.sample_size,
        seed=args.seed,
        rho_min=args.rho_min,
        smoothing=args.smoothing,
        iterations=args.iterations,
    )
    result["source_exposure_field"] = str(args.exposure_field)
    result["source_exposure_schema_version"] = field.get("schema_version")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "selected_count": args.sample_size}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
