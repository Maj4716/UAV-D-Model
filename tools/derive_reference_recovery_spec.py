"""Derive a smaller or larger outcome-blind Reference run from a frozen design.

Only budgets, replicate count, identities, revision, and preregistered seeds are
changed.  Geometry, target measure, feature contract, and quality gates remain
identical to the source document.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from damage_engine.errors import InputValidationError
from damage_engine.sampling_spec import SamplingSpec


def derive_reference_recovery_spec(
    source: dict[str, Any],
    *,
    spec_id: str,
    split_registry_id: str,
    train: int,
    validation: int,
    test: int,
    replicates: int,
    seed_base: int,
    revision: str,
    train_replicates: int | None = None,
) -> dict[str, Any]:
    if min(train, validation, test, replicates) < 1:
        raise InputValidationError(
            "reference recovery budgets and replicate count must be positive"
        )
    if seed_base < 0:
        raise InputValidationError("reference recovery seed base must be nonnegative")
    if train_replicates is not None and train_replicates < 1:
        raise InputValidationError("reference recovery train replicates must be positive")
    document = deepcopy(source)
    source_largest_allocation = max(
        int(allocation["count"])
        for round_item in source["round_budgets"]
        for allocation in round_item["allocations"]
    )
    source_attempt_factor = max(
        1,
        math.ceil(
            int(source["legal_sampling"]["max_attempts_per_allocation"])
            / source_largest_allocation
        ),
    )
    document["sampling_spec_id"] = str(spec_id)
    document["revision"] = str(revision)
    document["budgets"] = {
        "rare_stress_test": 0,
        "test": int(test),
        "train_pilot": int(train),
        "validation": int(validation),
    }
    document["formal_policy"]["replicates"] = int(replicates)
    document["replicate_policy"]["default_replicates"] = int(replicates)
    document["replicate_policy"]["overrides"] = (
        [
            {
                "role": "reference_train",
                "replicates": int(train_replicates),
            }
        ]
        if train_replicates is not None and train_replicates != replicates
        else []
    )
    counts = {
        "reference_validation": int(validation),
        "reference_test": int(test),
        "reference_train": int(train),
    }
    allocations = document["round_budgets"][0]["allocations"]
    for allocation in allocations:
        allocation["count"] = counts[str(allocation["role"])]
    largest_allocation = max(counts.values())
    document["legal_sampling"]["max_attempts_per_allocation"] = max(
        int(source["legal_sampling"]["max_attempts_per_allocation"]),
        source_attempt_factor * largest_allocation,
    )
    document["seeds"] = {
        "design": int(seed_base),
        "selection": int(seed_base + 1),
        "split": int(seed_base + 2),
        "screening": int(seed_base + 3),
        "formal": int(seed_base + 4),
    }
    document["split_policy"]["registry_id"] = str(split_registry_id)
    SamplingSpec(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spec-id", required=True)
    parser.add_argument("--split-registry-id", required=True)
    parser.add_argument("--train", required=True, type=int)
    parser.add_argument("--validation", required=True, type=int)
    parser.add_argument("--test", required=True, type=int)
    parser.add_argument("--replicates", required=True, type=int)
    parser.add_argument(
        "--train-replicates",
        type=int,
        help=(
            "Optional preregistered reference-train override; validation and "
            "test retain --replicates."
        ),
    )
    parser.add_argument("--seed-base", required=True, type=int)
    parser.add_argument("--revision", required=True)
    arguments = parser.parse_args()

    source_path = Path(arguments.base)
    output_path = Path(arguments.output)
    if output_path.exists():
        raise InputValidationError(
            "reference recovery output already exists",
            context={"path": str(output_path)},
        )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    document = derive_reference_recovery_spec(
        source,
        spec_id=arguments.spec_id,
        split_registry_id=arguments.split_registry_id,
        train=arguments.train,
        validation=arguments.validation,
        test=arguments.test,
        replicates=arguments.replicates,
        train_replicates=arguments.train_replicates,
        seed_base=arguments.seed_base,
        revision=arguments.revision,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sampling_spec_id": document["sampling_spec_id"],
                "budgets": document["budgets"],
                "replicates": document["formal_policy"]["replicates"],
                "replicate_overrides": document["replicate_policy"]["overrides"],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
