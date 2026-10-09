from __future__ import annotations

from copy import deepcopy

import pytest

from damage_engine.damage_tree import (
    compile_damage_tree,
    probability_and,
    probability_k_of_n,
    probability_or,
)
from damage_engine.errors import DamageTreeError


def test_k_of_n_probability():
    assert abs(probability_k_of_n([0.5, 0.5, 0.5], 2) - 0.5) < 1.0e-12


def test_independent_gate_truth_probabilities():
    assert probability_or([0.2, 0.4]) == pytest.approx(0.52)
    assert probability_and([0.2, 0.4]) == pytest.approx(0.08)
    assert probability_k_of_n([0.2, 0.4], 1) == pytest.approx(0.52)
    assert probability_k_of_n([0.2, 0.4], 2) == pytest.approx(0.08)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_gate_probabilities_reject_out_of_range_or_nonfinite_values(value):
    with pytest.raises(DamageTreeError):
        probability_or([value])


@pytest.mark.parametrize("k", [0, 3, 1.5, True])
def test_k_of_n_rejects_invalid_threshold(k):
    with pytest.raises(DamageTreeError):
        probability_k_of_n([0.2, 0.4], k)


def test_damage_tree_outputs(target_data):
    tree = compile_damage_tree(target_data)
    states = {
        "fragment": {"flight-control": {"degraded": 0.2, "failed": 0.1}},
        "blast": {"flight-control": {"degraded": 0.4, "failed": 0.2}},
        "combined": {"flight-control": {"degraded": 0.52, "failed": 0.28}},
    }
    result = tree.evaluate(states)
    assert result["outputs"]["mission_loss"] == 0.28
    assert result["outputs"]["mission_degraded"] == 0.4
    assert result["dependency_models"]["mission_degraded"] == "fully_dependent"


def test_fully_dependent_k_of_n_uses_comonotonic_semantics(target_data):
    target = deepcopy(target_data)
    target["damage_tree"] = {
        "version": "dependent-k",
        "nodes": [
            {
                "id": "degraded",
                "node_type": "leaf",
                "inputs": [],
                "component_id": "flight-control",
                "mechanism": "combined",
                "damage_state": "degraded",
                "k": None,
                "dependency_model_id": None,
            },
            {
                "id": "failed",
                "node_type": "leaf",
                "inputs": [],
                "component_id": "flight-control",
                "mechanism": "combined",
                "damage_state": "failed",
                "k": None,
                "dependency_model_id": None,
            },
            {
                "id": "both",
                "node_type": "k_of_n",
                "inputs": ["degraded", "failed"],
                "k": 2,
                "dependency_model_id": "fully_dependent",
            },
        ],
        "outputs": ["both"],
    }
    tree = compile_damage_tree(target)
    states = {
        "combined": {"flight-control": {"degraded": 0.6, "failed": 0.2}},
    }
    assert tree.evaluate(states)["outputs"]["both"] == 0.2


def test_independent_gate_rejects_overlapping_component_events(target_data):
    target = deepcopy(target_data)
    target["damage_tree"]["nodes"][2]["dependency_model_id"] = "independent"
    with pytest.raises(DamageTreeError, match="overlapping component events") as error:
        compile_damage_tree(target)
    assert error.value.context["node_id"] == "mission_degraded"


def test_duplicate_input_reference_is_rejected(target_data):
    target = deepcopy(target_data)
    target["damage_tree"]["nodes"][2]["inputs"] = ["control_failed", "control_failed"]
    with pytest.raises(DamageTreeError, match="duplicate references"):
        compile_damage_tree(target)


def test_missing_dependency_declaration_is_rejected(target_data):
    target = deepcopy(target_data)
    target["damage_tree"]["nodes"][2]["dependency_model_id"] = None
    with pytest.raises(DamageTreeError, match="explicit supported dependency"):
        compile_damage_tree(target)


def test_leaf_probability_is_not_silently_clipped(target_data):
    tree = compile_damage_tree(target_data)
    states = {
        "fragment": {"flight-control": {"degraded": 0.2, "failed": 0.1}},
        "blast": {"flight-control": {"degraded": 1.2, "failed": 0.2}},
        "combined": {"flight-control": {"degraded": 0.52, "failed": 0.28}},
    }
    with pytest.raises(DamageTreeError, match="leaf probability is invalid") as error:
        tree.evaluate(states)
    assert error.value.context["node_id"] == "control_blast_degraded"


def test_damage_tree_cycle_is_rejected(target_data):
    target = deepcopy(target_data)
    target["damage_tree"] = {
        "version": "cycle",
        "nodes": [
            {"id": "a", "node_type": "not", "inputs": ["b"]},
            {"id": "b", "node_type": "not", "inputs": ["a"]},
        ],
        "outputs": ["a"],
    }
    with pytest.raises(DamageTreeError) as error:
        compile_damage_tree(target)
    assert error.value.context["cycle_nodes"] == ["a", "b"]


def test_damage_tree_compiles_named_evaluation_standard_views(target_data):
    target = deepcopy(target_data)
    target["damage_tree"]["standards"] = [
        {"id": "all-effects", "outputs": ["mission_loss", "mission_degraded"]},
        {"id": "catastrophic-only", "outputs": ["mission_loss"]},
    ]
    tree = compile_damage_tree(target)
    assert tree.standards == {
        "all-effects": ("mission_loss", "mission_degraded"),
        "catastrophic-only": ("mission_loss",),
    }


def test_damage_tree_rejects_standard_view_of_undeclared_output(target_data):
    target = deepcopy(target_data)
    target["damage_tree"]["standards"] = [
        {"id": "bad-view", "outputs": ["control_failed"]}
    ]
    with pytest.raises(DamageTreeError, match="only declared damage-tree outputs"):
        compile_damage_tree(target)
