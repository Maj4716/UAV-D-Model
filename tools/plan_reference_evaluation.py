"""Command-line wrapper for outcome-independent reference-set planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from damage_engine.evaluation_planning import (
    plan_reference_evaluation,
    plan_reference_soft_event_mass,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prevalence", required=True, type=float)
    parser.add_argument("--target-events", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--half-widths", nargs="+", type=float, default=[0.01, 0.005])
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--effective-sample-fraction", type=float, default=1.0)
    parser.add_argument("--soft-event-mean", type=float)
    parser.add_argument(
        "--target-soft-event-mass", nargs="+", type=float, default=[50.0, 100.0]
    )
    parser.add_argument("--output")
    arguments = parser.parse_args()
    report = plan_reference_evaluation(
        arguments.prevalence,
        target_event_counts=arguments.target_events,
        absolute_half_widths=arguments.half_widths,
        confidence=arguments.confidence,
        effective_sample_fraction=arguments.effective_sample_fraction,
    )
    if arguments.soft_event_mean is not None:
        report["soft_event_mass_planning"] = plan_reference_soft_event_mass(
            arguments.soft_event_mean,
            target_soft_event_masses=arguments.target_soft_event_mass,
            effective_sample_fraction=arguments.effective_sample_fraction,
        )
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if arguments.output:
        destination = Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
