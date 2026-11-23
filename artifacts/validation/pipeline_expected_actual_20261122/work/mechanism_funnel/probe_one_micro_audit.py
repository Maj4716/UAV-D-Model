"""Probe one archived train case with opt-in fragment audit enabled."""

from __future__ import annotations

from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.engine import DamageEngine
from damage_engine.registry import PhysicsRegistry


DATASET = ROOT / "artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122"


def main() -> int:
    shard = None
    for candidate_shard in sorted((DATASET / "formal_dataset/shards").glob("part-*")):
        with np.load(candidate_shard / "tensors.npz", allow_pickle=False) as arrays:
            candidate_splits = arrays["split_names"].astype(str)
        if np.any(candidate_splits == "train"):
            shard = candidate_shard
            break
    if shard is None:
        raise RuntimeError("no train shard found")
    with np.load(shard / "tensors.npz", allow_pickle=False) as arrays:
        case_ids = arrays["case_ids"].astype(str)
        splits = arrays["split_names"].astype(str)
    train_index = next(index for index, split in enumerate(splits) if split == "train")
    case_id = str(case_ids[train_index])
    archived_result = None
    with gzip.open(shard / "results.jsonl.gz", "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index == train_index:
                archived_result = json.loads(line)
                break
    if archived_result is None:
        raise RuntimeError("archived result not found")

    case = None
    with gzip.open(
        DATASET / "rounds/round-000/formal_cases.jsonl.gz", "rt", encoding="utf-8"
    ) as stream:
        for line in stream:
            candidate = json.loads(line)
            if candidate["case_id"] == case_id:
                case = candidate
                break
    if case is None:
        raise RuntimeError("formal case not found")

    target = json.loads(
        (ROOT / "configs/targets/target.fragment_only.v1.json").read_text(
            encoding="utf-8-sig"
        )
    )
    engine = DamageEngine(
        target,
        PhysicsRegistry.from_file(
            ROOT / "configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json"
        ),
    )
    audited_case = deepcopy(case)
    audited_case["analysis"]["capture_fragment_audit"] = True
    rerun = engine.run(audited_case, fail_closed=True).to_dict()

    summary_error = max(
        abs(float(rerun["summary"][name]["mean"]) - float(value["mean"]))
        for name, value in archived_result["summary"].items()
    )
    component_error = 0.0
    for component_id, archived_component in archived_result["components"].items():
        rerun_component = rerun["components"][component_id]
        for name, value in archived_component["fragment_metrics"].items():
            archived_mean = value["mean"]
            rerun_mean = rerun_component["fragment_metrics"][name]["mean"]
            if archived_mean is None or rerun_mean is None:
                if archived_mean != rerun_mean:
                    raise RuntimeError(f"metric missingness mismatch for {component_id}:{name}")
                continue
            component_error = max(
                component_error,
                abs(float(rerun_mean) - float(archived_mean)),
            )
        for state in ("degraded", "failed"):
            component_error = max(
                component_error,
                abs(
                    float(rerun_component["damage_states"]["fragment"][state]["mean"])
                    - float(archived_component["damage_states"]["fragment"][state]["mean"])
                ),
            )
    tree_error = max(
        abs(
            float(rerun["damage_tree"]["outputs"][name]["mean"])
            - float(value["mean"])
        )
        for name, value in archived_result["damage_tree"]["outputs"].items()
    )

    type_counts: dict[str, int] = {}
    first_by_type: dict[str, dict] = {}
    event_count = 0
    for wrapper in rerun["fragment_audit"]:
        for event in wrapper["events"]:
            event_count += 1
            record_type = str(event.get("record_type", "unknown"))
            type_counts[record_type] = type_counts.get(record_type, 0) + 1
            first_by_type.setdefault(record_type, event)

    print(
        json.dumps(
            {
                "case_id": case_id,
                "profile": case["munition"]["profile_id"],
                "region": case["metadata"]["contact_manifold_sampling"]["region_id"],
                "replicates": case["analysis"]["replicates"],
                "summary_max_abs_error": summary_error,
                "component_max_abs_error": component_error,
                "tree_max_abs_error": tree_error,
                "audit_wrapper_count": len(rerun["fragment_audit"]),
                "event_count": event_count,
                "record_type_counts": type_counts,
                "event_keys_by_type": {
                    name: sorted(event) for name, event in first_by_type.items()
                },
                "event_samples": first_by_type,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
