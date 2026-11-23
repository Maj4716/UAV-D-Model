from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "run_idea2_stratified_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("idea2_stratified_diagnostics", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mechanism_regression_reports_drop_one_fixed_effect_contributions() -> None:
    generator = np.random.default_rng(17)
    count = 600
    hits = np.exp(generator.normal(1.0, 0.35, count))
    distance = np.exp(generator.normal(0.0, 0.20, count))
    mv2 = np.exp(generator.normal(4.0, 0.25, count))
    approach = np.where(np.arange(count) % 2 == 0, "a", "b")
    profile = np.asarray(["small", "medium", "large"])[np.arange(count) % 3]
    log_energy = (
        0.7 * np.log(hits)
        - 2.0 * np.log(distance)
        + 0.4 * np.log(mv2)
        + 0.8 * (approach == "b")
        + 0.5 * (profile == "large")
        + generator.normal(0.0, 0.02, count)
    )
    report = MODULE.mechanism_regression_with_drop_one(
        energy_j=np.exp(log_energy),
        hits=hits,
        range_m=distance,
        mass_velocity2=mv2,
        approach_id=approach.tolist(),
        profile_id=profile.tolist(),
        weight=np.ones(count),
    )
    assert report["full_model_r2"] > 0.99
    assert report["drop_one_groups"]["approach_fixed_effects"]["drop_one_incremental_r2"] > 0.01
    assert report["drop_one_groups"]["profile_fixed_effects"]["drop_one_incremental_r2"] > 0.001
    assert report["coefficients"]["log_inverse_range_squared"] == pytest.approx(1.0, abs=0.02)


def test_tree_information_retention_uses_joint_discrete_entropy() -> None:
    leaves = np.asarray(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8
    )
    outputs = np.asarray([[0.0], [0.0], [1.0], [1.0]], dtype=np.float64)
    report = MODULE.tree_information_retention(
        leaf_states=leaves,
        tree_outputs=outputs,
        weight=np.ones(4),
        leaf_semantics="synthetic exact leaves",
        exact_leaf_vector=True,
    )
    assert report["leaf_vector"]["entropy_bits"] == pytest.approx(2.0)
    assert report["tree_output_vector"]["entropy_bits"] == pytest.approx(1.0)
    assert report["joint_entropy_ratio"] == pytest.approx(0.5)
    assert report["exact_leaf_vector"] is True


def test_tree_distribution_report_is_stratified_by_approach_and_profile() -> None:
    tree = np.asarray([[0.0], [0.5], [1.0], [0.2]], dtype=np.float64)
    report = MODULE._tree_distribution_report(
        tree,
        names=["effect"],
        weight=np.ones(4),
        approach=np.asarray(["a", "a", "b", "b"]),
        profile=np.asarray(["small", "small", "large", "large"]),
    )
    assert sorted(report["by_approach_profile"]) == ["a|small", "b|large"]
    assert report["by_approach_profile"]["a|small"]["case_count"] == 2
    assert report["overall"]["effect"]["zero_fraction"] == pytest.approx(0.25)
