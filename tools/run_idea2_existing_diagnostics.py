#!/usr/bin/env python
"""Run the zero-engine-cost E4/E6 and partial E2/E7 diagnostics."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from damage_engine.mechanism_diagnostics import (
    log_tail_regression,
    tree_distribution_metrics,
)
from damage_engine.utils import content_hash, file_hash


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_membership(root: Path) -> tuple[set[str], dict[str, float]]:
    members: set[str] = set()
    weights: dict[str, float] = {}
    with gzip.open(root / "selection_observations.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("strategy") == "reference_uniform":
                identifier = str(row["candidate_id"])
                members.add(identifier)
                weights[identifier] = float(row.get("design_weight_raw", 1.0))
    return members, weights


def _case_mechanism_inputs(root: Path) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    with gzip.open(root / "cases.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            case = row["case"]
            source = case["warhead"]["fragment_source"]
            mass = float(source["mass_distribution"]["parameters"]["value_kg"])
            constant = float(source["gurney_constant_mps"])
            charge = float(case["warhead"]["charge_mass_kg"])
            casing = float(case["warhead"]["casing_mass_kg"])
            speed = constant / np.sqrt(casing / charge + 0.5)
            encounter = case["encounter"]
            distance = float(encounter.get("collision_distance_m", 1.0))
            output[str(row["candidate_id"])] = {
                "range_m": max(distance, 1.0e-9),
                "mass_velocity2": mass * speed * speed,
                "fragment_count": float(source["total_count"]),
            }
    return output


def run(root: Path, target_path: Path) -> dict[str, Any]:
    tensors_path = root / "tensors.npz"
    arrays = np.load(tensors_path, allow_pickle=False)
    reference, weight_map = _reference_membership(root)
    mechanism = _case_mechanism_inputs(root)
    candidate = arrays["candidate_id"].astype(str)
    index = np.asarray([identifier in reference for identifier in candidate], dtype=bool)
    complete = arrays["domain_complete"].astype(bool)
    selected = index & complete
    weights = np.asarray([weight_map.get(identifier, 1.0) for identifier in candidate[selected]])

    hits = arrays["component_hits"][selected].astype(np.float64)
    energy = arrays["component_energy_j"][selected].astype(np.float64)
    ranges = np.asarray([mechanism[identifier]["range_m"] for identifier in candidate[selected]])
    mv2 = np.asarray([mechanism[identifier]["mass_velocity2"] for identifier in candidate[selected]])
    positive = (hits > 0.0) & (energy > 0.0)
    repeated_range = np.broadcast_to(ranges[:, None], hits.shape)
    repeated_mv2 = np.broadcast_to(mv2[:, None], hits.shape)
    e4 = log_tail_regression(
        energy[positive],
        hits[positive],
        repeated_range[positive],
        repeated_mv2[positive],
        hill_k_values=[20, 50, 100, 200, 500],
    )

    tree = arrays["tree_outputs"][selected].astype(np.float64)
    tree_names = [f"tree_output_{index}" for index in range(tree.shape[1])]
    e6 = {
        name: tree_distribution_metrics(tree[:, index], weight=weights)
        for index, name in enumerate(tree_names)
    }
    target = _json(target_path)
    response_ids = {
        str(item["id"]): str(item["response"]["fragment_response_model_id"])
        for item in target["components"]
    }
    component_ids = [str(item["id"]) for item in target["components"]]
    component_match = {}
    for column, identifier in enumerate(component_ids):
        values = energy[:, column]
        positive_values = values[values > 0.0]
        component_match[identifier] = {
            "response_model_id": response_ids[identifier],
            "reachable_case_fraction": float(np.average(hits[:, column] > 0.0, weights=weights)),
            "positive_energy_count": int(positive_values.size),
            "log_energy_quantiles": (
                np.quantile(np.log(positive_values), [0.05, 0.5, 0.95]).tolist()
                if positive_values.size
                else None
            ),
            "status": "partial_E7_missing_per_impact_v50_and_component_E50_audit",
        }

    fragment_count = np.asarray([mechanism[identifier]["fragment_count"] for identifier in candidate[selected]])
    total_hits = np.sum(hits, axis=1)
    total_penetrations = np.sum(arrays["component_penetrations"][selected], axis=1)
    e2_partial = {
        "mean_fragments_emitted_per_replicate": float(np.average(fragment_count, weights=weights)),
        "mean_component_hits_per_replicate": float(np.average(total_hits, weights=weights)),
        "mean_component_penetrations_per_replicate": float(np.average(total_penetrations, weights=weights)),
        "case_component_positive_fraction": float(np.mean(hits > 0.0)),
        "status": "partial_missing_fragment_audit_armor_intersection_and_domain_stages",
    }
    report = {
        "schema_version": "idea2-existing-diagnostics@1.0.0",
        "dataset_root": str(root),
        "dataset_manifest_hash": file_hash(root / "dataset_manifest.json"),
        "tensors_hash": file_hash(tensors_path),
        "reference_unique_count": int(np.count_nonzero(index)),
        "reference_complete_count": int(np.count_nonzero(selected)),
        "E4_tail_regression": e4,
        "E6_tree_distribution": e6,
        "E2_partial_funnel": e2_partial,
        "E7_partial_component_matching": component_match,
        "limitations": [
            "Archived development tensors do not contain fragment-audit armor stages or per-impact V50.",
            "This diagnostic does not substitute for the preregistered 500-case E2/E7 run.",
        ],
    }
    return {**report, "report_hash": content_hash(report)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/development_dataset"))
    parser.add_argument("--target", type=Path, default=Path("configs/targets/target.fragment_only.v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.dataset, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report_hash": report["report_hash"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
