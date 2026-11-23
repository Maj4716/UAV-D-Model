#!/usr/bin/env python
"""Run the pure-NumPy dataset-v2 p_hit neural diagnostic baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.numpy_hurdle_baseline import (  # noqa: E402
    run_numpy_p_hit_learning_curves,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--fraction",
        action="append",
        type=float,
        help="training fraction; repeat for a curve (default: 0.25, 0.50, 1.0)",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="optimization/subsampling seed; repeat for replication",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument(
        "--task-admission",
        type=Path,
        help="frozen dataset-v2 component-task admission JSON",
    )
    parser.add_argument("--interval-loss-weight", type=float, default=0.25)
    parser.add_argument("--l2-penalty", type=float, default=1.0e-4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_numpy_p_hit_learning_curves(
        args.dataset,
        args.output,
        fractions=args.fraction or (0.25, 0.50, 1.0),
        seeds=args.seed or (17, 29, 43),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        interval_loss_weight=args.interval_loss_weight,
        l2_penalty=args.l2_penalty,
        task_admission_path=args.task_admission,
    )
    selected = result["selected_model"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "dataset_id": result["dataset_id"],
                "run_count": len(result["runs"]),
                "selected_seed": selected["seed"],
                "selected_epoch": selected["best_epoch"],
                "selected_validation_brier": selected["validation_metrics"][
                    "brier"
                ],
                "test_access_policy": result["test_access_policy"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
