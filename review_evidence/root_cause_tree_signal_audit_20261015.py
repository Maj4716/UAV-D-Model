import json
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import damage_engine

from damage_engine.utils import content_hash


ROOT = Path("raw_data/dataset_5000_loitering_impact_fragment_only_v1/formal_dataset/shards")
TARGET = Path("raw_data/target.rebuilt.fragment_only.v1.json")
OUTPUT = Path("review_evidence/root_cause_tree_signal_audit_20261015.json")
FROZEN_HASHES = Path("raw_data/dataset_5000_loitering_impact_fragment_only_v1/plan/frozen_hashes.json")
MANIFEST = Path("raw_data/dataset_5000_loitering_impact_fragment_only_v1/formal_dataset/manifest.json")


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    target = json.loads(TARGET.read_text(encoding="utf-8"))
    frozen_hashes = json.loads(FROZEN_HASHES.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tree_hash = content_hash(target["damage_tree"])
    if tree_hash != frozen_hashes["damage_tree_definition_hash"]:
        raise RuntimeError("current damage-tree definition does not match the frozen archive")
    nodes = {node["id"]: node for node in target["damage_tree"]["nodes"]}
    outputs = target["damage_tree"]["outputs"]

    labels = pd.concat(
        [
            pq.read_table(
                path,
                columns=["case_id", "component_id", "mechanism", "damage_state", "available", "mean"],
            ).to_pandas()
            for path in ROOT.glob("part-*/labels.parquet")
        ],
        ignore_index=True,
    )
    trees = pd.concat(
        [
            pq.read_table(path, columns=["case_id", "output_node", "available", "mean"]).to_pandas()
            for path in ROOT.glob("part-*/tree_outputs.parquet")
        ],
        ignore_index=True,
    )
    aux = pd.concat(
        [
            pq.read_table(
                path,
                columns=["case_id", "component_id", "mechanism", "metric", "available", "mean"],
            ).to_pandas()
            for path in ROOT.glob("part-*/component_aux.parquet")
        ],
        ignore_index=True,
    )

    leaf_rows = []
    for output in outputs:
        for leaf_id in nodes[output]["inputs"]:
            leaf = nodes[leaf_id]
            leaf_rows.append(
                {
                    "task": output,
                    "component_id": leaf["component_id"],
                    "mechanism": leaf["mechanism"],
                    "damage_state": leaf["damage_state"],
                }
            )
    leaf_map = pd.DataFrame(leaf_rows)
    selected = labels.merge(leaf_map, on=["component_id", "mechanism", "damage_state"], how="inner")
    leaf_signal = selected.groupby(["task", "case_id"], as_index=False)["mean"].max()
    task_components = leaf_map[["task", "component_id", "mechanism"]].drop_duplicates()
    selected_aux = aux.merge(task_components, on=["component_id", "mechanism"], how="inner")
    aux_signal = (
        selected_aux.groupby(["task", "case_id", "metric"], as_index=False)["mean"]
        .max()
        .pivot(index=["task", "case_id"], columns="metric", values="mean")
        .reset_index()
    )

    records = []
    for output in outputs:
        node = nodes[output]
        tree = trees.loc[trees["output_node"] == output, ["case_id", "available", "mean"]].copy()
        leaf = leaf_signal.loc[leaf_signal["task"] == output, ["case_id", "mean"]].rename(
            columns={"mean": "leaf_max"}
        )
        joined = tree.merge(leaf, on="case_id", how="left")
        task_aux = aux_signal.loc[aux_signal["task"] == output].drop(columns="task")
        joined = joined.merge(task_aux, on="case_id", how="left")
        joined["leaf_max"] = joined["leaf_max"].fillna(0.0)
        joined["mean"] = joined["mean"].fillna(0.0)
        records.append(
            {
                "task": output,
                "gate": node["node_type"],
                "dependency": node.get("dependency_model_id"),
                "leaf_count": len(node["inputs"]),
                "tree_rows": len(joined),
                "available_tree_rows": int(joined["available"].fillna(False).sum()),
                "leaf_positive_cases": int((joined["leaf_max"] > 0.0).sum()),
                "tree_positive_cases": int((joined["mean"] > 0.0).sum()),
                "leaf_positive_tree_zero": int(
                    ((joined["leaf_max"] > 0.0) & (joined["mean"] <= 0.0)).sum()
                ),
                "leaf_zero_tree_positive": int(
                    ((joined["leaf_max"] <= 0.0) & (joined["mean"] > 0.0)).sum()
                ),
                "cases_with_any_fragment_hit": int(
                    (joined.get("fragment_hits", pd.Series(0.0, index=joined.index)).fillna(0.0) > 0.0).sum()
                ),
                "cases_with_any_fragment_penetration": int(
                    (
                        joined.get("fragment_penetrations", pd.Series(0.0, index=joined.index)).fillna(0.0)
                        > 0.0
                    ).sum()
                ),
                "cases_with_any_deposited_energy": int(
                    (
                        joined.get("fragment_deposited_energy_j", pd.Series(0.0, index=joined.index)).fillna(0.0)
                        > 0.0
                    ).sum()
                ),
                "tree_max": float(joined["mean"].max()),
            }
        )

    audit = {
        "schema_version": "1.0.0",
        "scope": "existing_full_archive_diagnostic_only",
        "target_definition": {"path": TARGET.as_posix(), "sha256": sha256(TARGET)},
        "damage_tree_definition_hash": tree_hash,
        "archive_frozen_damage_tree_definition_hash": frozen_hashes["damage_tree_definition_hash"],
        "damage_tree_definition_matches_archive": (
            tree_hash == frozen_hashes["damage_tree_definition_hash"]
        ),
        "archive_engine_version": manifest["engine"]["version"],
        "current_package_version": damage_engine.__version__,
        "case_count": int(trees["case_id"].nunique()),
        "tasks": records,
        "evidence_boundary": (
            "This audit localizes the first nonzero stage in the existing synthetic archive. "
            "It includes the already-observed test split, does not validate real-world source strength, "
            "geometry, penetration, response thresholds, damage-code semantics, or current-source equivalence, "
            "and must not be used as a confirmatory model-performance result."
        ),
    }
    payload = json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    OUTPUT.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
