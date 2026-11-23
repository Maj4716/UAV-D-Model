#!/usr/bin/env python
"""Run the definition-complete E4/E6 diagnostics on archived development data.

The report uses only Reference cases whose teacher domain is complete.  E4 fits
the preregistered mechanism regression with approach and munition-profile fixed
effects.  E6 reports current-tree distributions by approach/profile and a
discrete information-compression diagnostic.  Archived tensors do not contain
the exact per-replicate tree-leaf vectors, so the default information ratio is
explicitly marked as a component-support proxy.  Optional NPZ candidate views
can supply exact ``leaf_states`` and alternative ``tree_outputs`` later.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.distribution_metrics import hill_tail_stability  # noqa: E402
from damage_engine.mechanism_diagnostics import tree_distribution_metrics  # noqa: E402
from damage_engine.utils import content_hash, file_hash  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_membership(root: Path) -> tuple[set[str], dict[str, float]]:
    members: set[str] = set()
    weights: dict[str, float] = {}
    with gzip.open(root / "selection_observations.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("strategy") != "reference_uniform":
                continue
            identifier = str(row["candidate_id"])
            members.add(identifier)
            weights[identifier] = float(row.get("design_weight_raw", 1.0))
    return members, weights


def _case_metadata(root: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with gzip.open(root / "cases.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            case = row["case"]
            source = case["warhead"]["fragment_source"]
            mass = float(source["mass_distribution"]["parameters"]["value_kg"])
            constant = float(source["gurney_constant_mps"])
            charge = float(case["warhead"]["charge_mass_kg"])
            casing = float(case["warhead"]["casing_mass_kg"])
            speed = constant / math.sqrt(casing / charge + 0.5)
            sampling = case.get("metadata", {}).get("idea1_sampling", {})
            contact = case.get("metadata", {}).get("contact_manifold_sampling", {})
            profile = str(
                sampling.get("profile_id")
                or case.get("munition", {}).get("profile_id")
                or "unknown_profile"
            )
            approach = str(
                sampling.get("approach_id")
                or contact.get("approach_id")
                or "unknown_approach"
            )
            output[str(row["candidate_id"])] = {
                "mass_velocity2": mass * speed * speed,
                "fragment_count": int(source["total_count"]),
                "detonation_position_m": np.asarray(
                    case["encounter"]["detonation_position_m"], dtype=np.float64
                ),
                "profile_id": profile,
                "approach_id": approach,
            }
    return output


def _component_centers_m(target: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    identifiers: list[str] = []
    centers: list[np.ndarray] = []
    for component in target["components"]:
        identifiers.append(str(component["id"]))
        geometry = component["geometry"]
        if "center_cm" not in geometry:
            raise ValueError(f"component {component['id']} lacks center_cm")
        centers.append(np.asarray(geometry["center_cm"], dtype=np.float64) / 100.0)
    return identifiers, np.asarray(centers, dtype=np.float64)


def _weighted_r2(
    target: np.ndarray,
    design: np.ndarray,
    weight: np.ndarray,
) -> tuple[float, np.ndarray, int, float]:
    y = np.asarray(target, dtype=np.float64)
    x = np.asarray(design, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    if (
        y.ndim != 1
        or x.ndim != 2
        or w.ndim != 1
        or x.shape[0] != y.size
        or w.size != y.size
        or y.size < 3
        or not np.isfinite(y).all()
        or not np.isfinite(x).all()
        or not np.isfinite(w).all()
        or np.any(w <= 0.0)
    ):
        raise ValueError("weighted regression inputs must be finite, aligned and positive-weighted")
    normalized = w / np.sum(w)
    root_weight = np.sqrt(normalized)
    coefficient, _, rank, _ = np.linalg.lstsq(
        x * root_weight[:, None], y * root_weight, rcond=None
    )
    fitted = x @ coefficient
    mean = float(np.sum(normalized * y))
    total = float(np.sum(normalized * (y - mean) ** 2))
    residual = float(np.sum(normalized * (y - fitted) ** 2))
    r2 = 0.0 if total <= 0.0 else 1.0 - residual / total
    return float(r2), coefficient, int(rank), residual


def _categorical_columns(
    values: Sequence[str], *, prefix: str
) -> tuple[np.ndarray, list[str], str, list[str]]:
    labels = np.asarray(values, dtype=str)
    categories = sorted(set(labels.tolist()))
    if not categories:
        raise ValueError(f"{prefix} categories are empty")
    reference = categories[0]
    retained = categories[1:]
    if not retained:
        return np.empty((labels.size, 0), dtype=np.float64), [], reference, categories
    matrix = np.column_stack([(labels == category).astype(np.float64) for category in retained])
    return matrix, [f"{prefix}:{category}" for category in retained], reference, categories


def mechanism_regression_with_drop_one(
    *,
    energy_j: Any,
    hits: Any,
    range_m: Any,
    mass_velocity2: Any,
    approach_id: Sequence[str],
    profile_id: Sequence[str],
    weight: Any,
) -> dict[str, Any]:
    energy = np.asarray(energy_j, dtype=np.float64)
    hit = np.asarray(hits, dtype=np.float64)
    distance = np.asarray(range_m, dtype=np.float64)
    mv2 = np.asarray(mass_velocity2, dtype=np.float64)
    weights = np.asarray(weight, dtype=np.float64)
    n = energy.size
    if any(value.shape != (n,) for value in (hit, distance, mv2, weights)):
        raise ValueError("mechanism-regression vectors must align")
    if len(approach_id) != n or len(profile_id) != n:
        raise ValueError("fixed-effect labels must align with observations")
    selected = (
        np.isfinite(energy)
        & np.isfinite(hit)
        & np.isfinite(distance)
        & np.isfinite(mv2)
        & np.isfinite(weights)
        & (energy > 0.0)
        & (hit > 0.0)
        & (distance > 0.0)
        & (mv2 > 0.0)
        & (weights > 0.0)
    )
    if int(np.count_nonzero(selected)) < 20:
        raise ValueError("mechanism regression requires at least twenty positive observations")
    energy = energy[selected]
    hit = hit[selected]
    distance = distance[selected]
    mv2 = mv2[selected]
    weights = weights[selected]
    approaches = np.asarray(approach_id, dtype=str)[selected]
    profiles = np.asarray(profile_id, dtype=str)[selected]
    approach_matrix, approach_names, approach_reference, approach_categories = _categorical_columns(
        approaches, prefix="approach"
    )
    profile_matrix, profile_names, profile_reference, profile_categories = _categorical_columns(
        profiles, prefix="profile"
    )
    columns: list[np.ndarray] = [
        np.ones(energy.size, dtype=np.float64),
        np.log(hit),
        -2.0 * np.log(distance),
        np.log(mv2),
    ]
    names = [
        "intercept",
        "log_hits",
        "log_inverse_range_squared",
        "log_mass_velocity_squared",
    ]
    groups: dict[str, list[int]] = {
        "log_hits": [1],
        "log_inverse_range_squared": [2],
        "log_mass_velocity_squared": [3],
    }
    if approach_matrix.shape[1]:
        start = len(columns)
        columns.extend(approach_matrix[:, index] for index in range(approach_matrix.shape[1]))
        names.extend(approach_names)
        groups["approach_fixed_effects"] = list(range(start, start + approach_matrix.shape[1]))
    if profile_matrix.shape[1]:
        start = len(columns)
        columns.extend(profile_matrix[:, index] for index in range(profile_matrix.shape[1]))
        names.extend(profile_names)
        groups["profile_fixed_effects"] = list(range(start, start + profile_matrix.shape[1]))
    design = np.column_stack(columns)
    target = np.log(energy)
    full_r2, coefficient, rank, residual = _weighted_r2(target, design, weights)
    normalized_weight = weights / np.sum(weights)
    weighted_design = design * np.sqrt(normalized_weight)[:, None]
    singular_values = np.linalg.svd(weighted_design, compute_uv=False)
    smallest = float(singular_values[-1])
    largest = float(singular_values[0])
    condition_number = math.inf if smallest <= 0.0 else largest / smallest
    increments: dict[str, Any] = {}
    all_indices = np.arange(design.shape[1])
    for group, removed in groups.items():
        keep = np.asarray([index for index in all_indices if index not in removed], dtype=np.int64)
        reduced_r2, _, reduced_rank, _ = _weighted_r2(target, design[:, keep], weights)
        increments[group] = {
            "reduced_r2": reduced_r2,
            "drop_one_incremental_r2": max(0.0, full_r2 - reduced_r2),
            "raw_r2_difference": full_r2 - reduced_r2,
            "reduced_rank": reduced_rank,
        }
    dof = max(1, energy.size - rank)
    return {
        "positive_observation_count": int(energy.size),
        "full_model_r2": full_r2,
        "rank": rank,
        "coefficient_count": int(design.shape[1]),
        "full_column_rank": bool(rank == design.shape[1]),
        "rank_deficiency": int(design.shape[1] - rank),
        "weighted_design_condition_number": (
            None if not math.isfinite(condition_number) else float(condition_number)
        ),
        "identifiability": (
            "all declared coefficients identifiable in this design"
            if rank == design.shape[1]
            else (
                "rank deficient: at least one mechanism term is aliased with fixed effects; "
                "use drop-one predictive R2 and collect within-profile mechanism variation "
                "before interpreting aliased coefficients"
            )
        ),
        "coefficients": {name: float(value) for name, value in zip(names, coefficient)},
        "weighted_residual_stddev": float(math.sqrt(residual * energy.size / dof)),
        "drop_one_groups": increments,
        "fixed_effects": {
            "approach_reference": approach_reference,
            "approach_categories": approach_categories,
            "profile_reference": profile_reference,
            "profile_categories": profile_categories,
        },
        "weight_semantics": "Reference inverse-inclusion design weight repeated over positive component observations",
    }


def _valid_hill_k(count: int, requested: Iterable[int]) -> list[int]:
    valid = sorted({int(value) for value in requested if 2 <= int(value) < count})
    if valid:
        return valid
    return [max(2, min(count - 1, count // 5))]


def _weighted_joint_entropy(matrix: Any, weight: Any) -> dict[str, Any]:
    values = np.asarray(matrix)
    weights = np.asarray(weight, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != weights.size or values.shape[0] < 1:
        raise ValueError("joint-entropy matrix and weights must align")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("joint-entropy weights must be finite and positive")
    contiguous = np.ascontiguousarray(values)
    _, inverse = np.unique(contiguous, axis=0, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)
    probability = mass[mass > 0.0] / np.sum(mass)
    entropy = float(-np.sum(probability * np.log2(probability)))
    return {
        "entropy_bits": entropy,
        "unique_pattern_count": int(probability.size),
        "effective_pattern_count": float(2.0**entropy),
    }


def _discretize_tree_outputs(values: np.ndarray) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("tree outputs must be a finite matrix")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("tree outputs must lie in [0, 1]")
    result = np.ones(probability.shape, dtype=np.uint8)
    result[probability <= 1.0e-6] = 0
    result[probability >= 0.99] = 2
    return result


def tree_information_retention(
    *,
    leaf_states: Any,
    tree_outputs: Any,
    weight: Any,
    leaf_semantics: str,
    exact_leaf_vector: bool,
) -> dict[str, Any]:
    leaves = np.asarray(leaf_states)
    outputs = np.asarray(tree_outputs, dtype=np.float64)
    weights = np.asarray(weight, dtype=np.float64)
    if leaves.ndim != 2 or outputs.ndim != 2 or leaves.shape[0] != outputs.shape[0]:
        raise ValueError("leaf and tree matrices must share their case axis")
    leaf_discrete = leaves.astype(np.int64, copy=False)
    tree_discrete = _discretize_tree_outputs(outputs)
    leaf_entropy = _weighted_joint_entropy(leaf_discrete, weights)
    tree_entropy = _weighted_joint_entropy(tree_discrete, weights)
    denominator = float(leaf_entropy["entropy_bits"])
    ratio = None if denominator <= 0.0 else float(tree_entropy["entropy_bits"] / denominator)
    per_output: list[dict[str, Any]] = []
    for index in range(tree_discrete.shape[1]):
        item = _weighted_joint_entropy(tree_discrete[:, index : index + 1], weights)
        per_output.append(
            {
                "output_index": index,
                **item,
                "entropy_over_leaf_vector_entropy": (
                    None if denominator <= 0.0 else float(item["entropy_bits"] / denominator)
                ),
            }
        )
    return {
        "leaf_semantics": leaf_semantics,
        "exact_leaf_vector": bool(exact_leaf_vector),
        "leaf_vector": leaf_entropy,
        "tree_output_vector": tree_entropy,
        "joint_entropy_ratio": ratio,
        "per_output": per_output,
        "tree_discretization": {
            "zero": "p <= 1e-6",
            "intermediate": "1e-6 < p < 0.99",
            "saturated": "p >= 0.99",
        },
        "interpretation": "descriptive compression ratio; not mutual information and not a causal attribution",
    }


def _tree_distribution_report(
    tree: np.ndarray,
    *,
    names: Sequence[str],
    weight: np.ndarray,
    approach: np.ndarray,
    profile: np.ndarray,
) -> dict[str, Any]:
    if tree.ndim != 2 or tree.shape[1] != len(names):
        raise ValueError("tree matrix and names must align")

    def metrics(index: np.ndarray) -> dict[str, Any]:
        return {
            name: tree_distribution_metrics(tree[index, column], weight=weight[index])
            for column, name in enumerate(names)
        }

    all_rows = np.ones(tree.shape[0], dtype=bool)
    strata: dict[str, Any] = {}
    cells = sorted({f"{a}|{p}" for a, p in zip(approach.tolist(), profile.tolist())})
    for cell in cells:
        selected = np.asarray(
            [f"{a}|{p}" == cell for a, p in zip(approach.tolist(), profile.tolist())],
            dtype=bool,
        )
        strata[cell] = {
            "case_count": int(np.count_nonzero(selected)),
            "metrics": metrics(selected),
        }
    return {
        "case_count": int(tree.shape[0]),
        "stratum_count": len(strata),
        "overall": metrics(all_rows),
        "by_approach_profile": strata,
    }


def _load_candidate_view(
    path: Path,
    *,
    selected_ids: np.ndarray,
    fallback_leaf_states: np.ndarray,
    weight: np.ndarray,
    approach: np.ndarray,
    profile: np.ndarray,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"candidate_id", "tree_outputs", "tree_output_ids"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"candidate view {path} lacks {missing}")
        identifiers = archive["candidate_id"].astype(str)
        if identifiers.size != np.unique(identifiers).size:
            raise ValueError(f"candidate view {path} contains duplicate candidate IDs")
        lookup = {identifier: index for index, identifier in enumerate(identifiers.tolist())}
        if any(identifier not in lookup for identifier in selected_ids.tolist()):
            raise ValueError(f"candidate view {path} is missing selected cases")
        order = np.asarray([lookup[identifier] for identifier in selected_ids.tolist()], dtype=np.int64)
        tree = archive["tree_outputs"][order].astype(np.float64)
        names = archive["tree_output_ids"].astype(str).tolist()
        if "leaf_states" in archive.files:
            leaves = archive["leaf_states"][order]
            exact = bool(int(np.asarray(archive.get("leaf_states_exact", np.asarray([1]))).ravel()[0]))
            semantics = str(
                np.asarray(
                    archive.get("leaf_state_semantics", np.asarray(["candidate_view_leaf_states"]))
                ).ravel()[0]
            )
        else:
            leaves = fallback_leaf_states
            exact = False
            semantics = "component_support_binary_proxy; candidate view supplied no leaf_states"
    return {
        "path": str(path),
        "file_hash": file_hash(path),
        "tree_distribution": _tree_distribution_report(
            tree,
            names=names,
            weight=weight,
            approach=approach,
            profile=profile,
        ),
        "information_retention": tree_information_retention(
            leaf_states=leaves,
            tree_outputs=tree,
            weight=weight,
            leaf_semantics=semantics,
            exact_leaf_vector=exact,
        ),
    }


def run(
    dataset_root: Path,
    target_path: Path,
    *,
    candidate_views: Sequence[Path] = (),
) -> dict[str, Any]:
    target = _json(target_path)
    component_ids, component_centers = _component_centers_m(target)
    feature_schema = _json(dataset_root / "feature_schema.json")
    tensors_path = dataset_root / "tensors.npz"
    with np.load(tensors_path, allow_pickle=False) as arrays:
        reference, weight_map = _reference_membership(dataset_root)
        metadata = _case_metadata(dataset_root)
        candidate_id = arrays["candidate_id"].astype(str)
        selected = np.asarray(
            [identifier in reference for identifier in candidate_id], dtype=bool
        ) & arrays["domain_complete"].astype(bool)
        selected_ids = candidate_id[selected]
        weight = np.asarray(
            [weight_map.get(identifier, 1.0) for identifier in selected_ids],
            dtype=np.float64,
        )
        hits = arrays["component_hits"][selected].astype(np.float64)
        energy = arrays["component_energy_j"][selected].astype(np.float64)
        tree = arrays["tree_outputs"][selected].astype(np.float64)
        component_support = arrays["component_support"][selected].astype(np.float64)
    if hits.shape[1] != len(component_ids):
        raise ValueError("target component order does not match tensor width")
    case_meta = [metadata[identifier] for identifier in selected_ids]
    approach = np.asarray([item["approach_id"] for item in case_meta], dtype=str)
    profile = np.asarray([item["profile_id"] for item in case_meta], dtype=str)
    mv2_case = np.asarray([item["mass_velocity2"] for item in case_meta], dtype=np.float64)
    n_case = np.asarray([item["fragment_count"] for item in case_meta], dtype=np.float64)
    detonation = np.asarray(
        [item["detonation_position_m"] for item in case_meta], dtype=np.float64
    )
    distance = np.linalg.norm(
        detonation[:, None, :] - component_centers[None, :, :], axis=2
    )
    positive = (hits > 0.0) & (energy > 0.0)
    repeated_mv2 = np.broadcast_to(mv2_case[:, None], hits.shape)
    repeated_weight = np.broadcast_to(weight[:, None], hits.shape)
    repeated_approach = np.broadcast_to(approach[:, None], hits.shape)
    repeated_profile = np.broadcast_to(profile[:, None], hits.shape)
    repeated_n = np.broadcast_to(n_case[:, None], hits.shape)
    regression = mechanism_regression_with_drop_one(
        energy_j=energy[positive],
        hits=hits[positive],
        range_m=distance[positive],
        mass_velocity2=repeated_mv2[positive],
        approach_id=repeated_approach[positive].tolist(),
        profile_id=repeated_profile[positive].tolist(),
        weight=repeated_weight[positive],
    )
    hill_by_profile: dict[str, Any] = {}
    for profile_id in sorted(set(profile.tolist())):
        profile_mask = positive & (repeated_profile == profile_id)
        values = energy[profile_mask]
        hit_values = hits[profile_mask]
        fragment_counts = repeated_n[profile_mask]
        k_values = _valid_hill_k(values.size, (20, 50, 100, 200, 500))
        hill_by_profile[profile_id] = {
            "positive_observation_count": int(values.size),
            "hill": hill_tail_stability(values, k_values=k_values),
            "fragment_count_truncation": {
                "max_hits_over_fragment_count": float(np.max(hit_values / fragment_counts)),
                "p99_hits_over_fragment_count": float(
                    np.quantile(hit_values / fragment_counts, 0.99)
                ),
                "fraction_at_or_above_90_percent_fragment_count": float(
                    np.mean(hit_values >= 0.90 * fragment_counts)
                ),
                "scope": "component hit-count ceiling diagnostic; energy can also be bounded by response and penetration",
            },
        }
    tree_names = list(map(str, feature_schema["tree_output_ids"]))
    if len(tree_names) != tree.shape[1]:
        raise ValueError("tree output names do not match tensor width")
    support_proxy = np.asarray(component_support > 0.5, dtype=np.uint8)
    current_tree = {
        "tree_definition": str(target["damage_tree"]["version"]),
        "tree_output_ids": tree_names,
        "tree_distribution": _tree_distribution_report(
            tree,
            names=tree_names,
            weight=weight,
            approach=approach,
            profile=profile,
        ),
        "information_retention": tree_information_retention(
            leaf_states=support_proxy,
            tree_outputs=tree,
            weight=weight,
            leaf_semantics=(
                "67-component binary support proxy from archived tensors; exact per-replicate, "
                "state-specific current-tree leaf vectors were not archived"
            ),
            exact_leaf_vector=False,
        ),
    }
    alternatives = [
        _load_candidate_view(
            path,
            selected_ids=selected_ids,
            fallback_leaf_states=support_proxy,
            weight=weight,
            approach=approach,
            profile=profile,
        )
        for path in candidate_views
    ]
    report = {
        "schema_version": "idea2-stratified-diagnostics@1.0.0",
        "dataset_root": str(dataset_root),
        "dataset_manifest_hash": file_hash(dataset_root / "dataset_manifest.json"),
        "tensors_hash": file_hash(tensors_path),
        "target_hash": file_hash(target_path),
        "reference_complete_case_count": int(selected_ids.size),
        "component_count": len(component_ids),
        "E4_mechanism_regression": {
            "formula": (
                "ln(E) ~ ln(hits) + ln(1/r^2) + ln(m*V0^2) + "
                "approach fixed effects + munition-profile fixed effects"
            ),
            "range_definition": "Euclidean distance from detonation position to component center",
            "regression": regression,
            "hill_by_profile": hill_by_profile,
        },
        "E6_tree_compression": {
            "current_tree": current_tree,
            "candidate_views": alternatives,
            "candidate_view_contract": {
                "required_npz_arrays": ["candidate_id", "tree_outputs", "tree_output_ids"],
                "optional_npz_arrays": [
                    "leaf_states",
                    "leaf_states_exact",
                    "leaf_state_semantics",
                ],
                "exact_information_requirement": (
                    "supply state-specific per-case leaf_states derived from per-replicate trace; "
                    "the archived development tensors support only the marked proxy ratio"
                ),
            },
        },
        "evidence_boundary": {
            "role": "zero-engine-cost mechanism diagnostic on the archived development teacher",
            "physical_validation": False,
            "test_split_access": "dataset has no sealed test split; only Reference complete development cases are used",
        },
    }
    return {**report, "report_hash": content_hash(report)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/development_dataset"),
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("configs/targets/target.fragment_only.v1.json"),
    )
    parser.add_argument("--candidate-view", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run(args.dataset, args.target, candidate_views=args.candidate_view)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(args.output), "report_hash": report["report_hash"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
