"""Check whether the frozen Train archive can fill the proposed R-study strata."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.sampling_checkpoint import iter_reference_formal_cases
from damage_engine.utils import file_hash


DATASET = ROOT / "raw_data" / "dataset_5000_loitering_impact_fragment_only_v1"
FORMAL = DATASET / "formal_dataset"
OUTPUT = Path(__file__).resolve().parent / "replicate_convergence_selection_readiness.json"
TASKS = ("effect-m2", "effect-m1")
BANDS = ("low", "transition", "high")
MINIMUM_PER_STRATUM = 4


def _train_scores() -> dict[str, float]:
    scores: dict[str, float] = {}
    for shard in sorted((FORMAL / "shards").glob("part-*")):
        with np.load(shard / "tensors.npz", allow_pickle=False) as archive:
            names = tuple(map(str, archive["system_output_names"].tolist()))
            indices = [names.index(name) for name in TASKS]
            role_names = tuple(map(str, archive["sampling_role_names"].tolist()))
            for case_id, split, role_index, values in zip(
                archive["case_ids"].astype(str),
                archive["split_names"].astype(str),
                archive["sampling_role_index"],
                archive["system_y"][:, indices],
            ):
                role = role_names[int(role_index)]
                if split == "train" and role == "reference_train":
                    scores[case_id] = float(np.max(values))
    return scores


def _band(score: float) -> str:
    if score <= 0.0:
        return "low"
    if score >= 1.0:
        return "high"
    return "transition"


def main() -> int:
    scores = _train_scores()
    counts: Counter[tuple[str, str, str]] = Counter()
    candidates: dict[tuple[str, str, str], list[str]] = {}
    profiles: set[str] = set()
    armors: set[str] = set()
    for case in iter_reference_formal_cases(DATASET):
        case_id = str(case["case_id"])
        if case_id not in scores:
            continue
        profile = str(case["munition"]["profile_id"])
        armor = str(case["encounter"]["collision_armor_id"])
        key = (profile, armor, _band(scores[case_id]))
        profiles.add(profile)
        armors.add(armor)
        counts[key] += 1
        candidates.setdefault(key, []).append(case_id)
    required = [
        (profile, armor, band)
        for profile in sorted(profiles)
        for armor in sorted(armors)
        for band in BANDS
    ]
    shortfalls = {
        "|".join(key): {
            "available": counts[key],
            "required": MINIMUM_PER_STRATUM,
            "shortfall": max(0, MINIMUM_PER_STRATUM - counts[key]),
        }
        for key in required
        if counts[key] < MINIMUM_PER_STRATUM
    }
    ready = not shortfalls
    selected_case_ids = (
        [
            case_id
            for key in required
            for case_id in sorted(candidates[key])[:MINIMUM_PER_STRATUM]
        ]
        if ready
        else []
    )
    report = {
        "schema_version": "1.0.0",
        "status": "ready" if ready else "quota_unreachable",
        "selection_source_split": "train",
        "selection_source_role": "reference_train",
        "uses_validation_or_test_outcomes": False,
        "outcome_informed": True,
        "source_label_replicates": 2,
        "response_score": {
            "definition": "max(effect-m2, effect-m1)",
            "tasks": list(TASKS),
            "bands": {
                "low": "score == 0",
                "transition": "0 < score < 1",
                "high": "score == 1",
            },
            "boundary": (
                "Existing R=2 Train labels are used only to plan a new convergence "
                "diagnostic. They are not treated as converged truth."
            ),
        },
        "required_design": {
            "profiles": sorted(profiles),
            "collision_armor_ids": sorted(armors),
            "response_bands": list(BANDS),
            "minimum_per_crossed_stratum": MINIMUM_PER_STRATUM,
            "required_case_count": len(required) * MINIMUM_PER_STRATUM,
        },
        "available_counts": {
            "|".join(key): counts[key] for key in required
        },
        "shortfalls": shortfalls,
        "selected_case_ids": selected_case_ids,
        "bindings": {
            "formal_manifest_hash": file_hash(FORMAL / "manifest.json"),
            "sampling_checkpoint_hash": file_hash(DATASET / "sampling_checkpoint.json"),
        },
        "decision": (
            "Do not execute or claim the formal 72-case R=4/8/16/32/64/128 "
            "study until a new Train-only reachability design can fill every frozen stratum."
            if not ready
            else "Freeze these ordered case IDs before producing any new replicate outcomes."
        ),
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "shortfalls": shortfalls}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
