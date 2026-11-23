from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from damage_engine.errors import DamageTreeError
from damage_engine.offline_tree_candidates import (
    IDENTITY_FALLBACK_STATUS,
    build_identity_component_fallback,
    candidate_definition_hash,
    compile_offline_tree_candidate,
    probability_k_of_n,
    two_additive_choquet,
    validate_two_additive_capacity,
    weighted_noisy_or,
)
from damage_engine.utils import file_hash


def test_monotone_gates_are_bounded_and_coordinatewise_nondecreasing() -> None:
    rng = np.random.default_rng(20261120)
    for _ in range(200):
        lower = rng.uniform(0.0, 1.0, size=3)
        upper = lower + rng.uniform(0.0, 1.0 - lower)
        weights = [0.2, 0.8, 1.0]
        assert 0.0 <= weighted_noisy_or(lower, weights) <= weighted_noisy_or(upper, weights) <= 1.0
        assert 0.0 <= probability_k_of_n(lower, 2) <= probability_k_of_n(upper, 2) <= 1.0
        singleton = {"a": 0.3, "b": 0.3, "c": 0.2}
        pairs = {("a", "b"): 0.1, ("b", "c"): 0.1}
        lower_value = two_additive_choquet(dict(zip("abc", lower)), singleton, pairs)
        upper_value = two_additive_choquet(dict(zip("abc", upper)), singleton, pairs)
        assert 0.0 <= lower_value <= upper_value <= 1.0


def test_weighted_noisy_or_uses_fatality_weights_and_unit_weights_recover_or() -> None:
    assert weighted_noisy_or([0.2, 0.3], [0.5, 0.25]) == pytest.approx(
        1.0 - (1.0 - 0.5 * 0.2) * (1.0 - 0.25 * 0.3)
    )
    assert weighted_noisy_or([0.2, 0.3], [1.0, 1.0]) == pytest.approx(
        1.0 - (1.0 - 0.2) * (1.0 - 0.3)
    )
    with pytest.raises(DamageTreeError, match=r"\[0,1\]"):
        weighted_noisy_or([0.2], [1.01])


def test_two_additive_capacity_accepts_redundancy_but_rejects_nonmonotone_mass() -> None:
    validate_two_additive_capacity(
        ["a", "b"], {"a": 0.6, "b": 0.6}, {("a", "b"): -0.2}
    )
    with pytest.raises(DamageTreeError, match="not monotone"):
        validate_two_additive_capacity(
            ["a", "b"], {"a": 0.1, "b": 1.1}, {("a", "b"): -0.2}
        )


def _candidate() -> dict:
    return {
        "schema_version": "offline-monotone-tree-candidate@1.0.0",
        "candidate_id": "candidate-a",
        "status": "unreviewed",
        "semantic_authority": {
            "status": "unreviewed",
            "source": None,
            "scope": "software test only",
        },
        "inputs": [
            {
                "input_id": "a",
                "component_id": "c1",
                "mechanism": "fragment",
                "damage_state": "failed",
            },
            {
                "input_id": "b",
                "component_id": "c2",
                "mechanism": "fragment",
                "damage_state": "failed",
            },
        ],
        "outputs": [
            {
                "output_id": "y",
                "gate": {
                    "kind": "weighted_noisy_or",
                    "inputs": ["a", "b"],
                    "weights": [1.0, 0.5],
                },
            }
        ],
    }


def test_candidate_hash_is_key_order_stable_and_declared_hash_is_checked() -> None:
    candidate = _candidate()
    reordered = {key: candidate[key] for key in reversed(candidate)}
    assert candidate_definition_hash(candidate) == candidate_definition_hash(reordered)
    compiled = compile_offline_tree_candidate(candidate)
    assert compiled.definition_hash.startswith("sha256:")
    bad = copy.deepcopy(candidate)
    bad["definition_hash"] = "sha256:" + "0" * 64
    with pytest.raises(DamageTreeError, match="does not match"):
        compile_offline_tree_candidate(bad)


def test_identity_fallback_is_explicitly_provisional_and_pass_through_only() -> None:
    document = build_identity_component_fallback(["23", "1"])
    compiled = compile_offline_tree_candidate(document)
    assert compiled.status == IDENTITY_FALLBACK_STATUS
    assert compiled.authority_status == "not_provided"
    assert len(compiled.inputs) == 4
    assert all(output["gate"]["kind"] == "identity" for output in compiled.outputs)
    assert all(output["output_id"].startswith("identity::") for output in compiled.outputs)


def test_approved_authority_requires_a_review_source() -> None:
    document = _candidate()
    document["semantic_authority"]["status"] = "approved"
    with pytest.raises(DamageTreeError, match="review source"):
        compile_offline_tree_candidate(document)


def test_checked_in_fallback_matches_the_23_currently_unreferenced_components() -> None:
    target_path = Path("configs/targets/target.fragment_only.v1.json")
    target = json.loads(target_path.read_text(encoding="utf-8"))
    referenced: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("component_id"), str):
                referenced.add(value["component_id"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(target["damage_tree"])
    all_components = {item["id"] for item in target["components"]}
    unreferenced = all_components - referenced
    assert len(unreferenced) == 23
    fallback = json.loads(
        Path(
            "artifacts/validation/tree/identity_unreferenced_23."
            "target_fragment_only_v1.provisional.json"
        ).read_text(encoding="utf-8")
    )
    compiled = compile_offline_tree_candidate(fallback)
    assert compiled.status == IDENTITY_FALLBACK_STATUS
    assert compiled.authority_status == "not_provided"
    assert {item["component_id"] for item in compiled.inputs} == unreferenced
    assert len(compiled.inputs) == 46
    assert fallback["provenance"]["target_file_hash"] == file_hash(target_path)
