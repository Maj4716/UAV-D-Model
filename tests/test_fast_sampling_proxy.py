from __future__ import annotations

import json
from pathlib import Path

import pytest

from damage_engine.engine import DamageEngine
from damage_engine.fast_sampling_proxy import FastSamplingProxy
from damage_engine.registry import PhysicsRegistry
from tools.evaluate_fast_sampling_proxy import (
    _cell_stratified_arm_difference_bootstrap,
    _random_arm_accuracy_report,
    _selection_design_ledger,
    _stratified_bootstrap_interval,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _engine_and_case() -> tuple[DamageEngine, dict]:
    target = _load(ROOT / "configs/targets/target.fragment_only.v1.json")
    registry = PhysicsRegistry.from_file(ROOT / "configs/physics/loitering_impact.json")
    case = _load(ROOT / "configs/cases/loitering_impact.fragment_only.v1.json")
    case["analysis"]["capture_fragment_audit"] = False
    case["analysis"]["capture_replicate_trace"] = False
    return DamageEngine(target, registry), case


def test_fast_sampling_proxy_is_deterministic_and_weight_normalized():
    engine, case = _engine_and_case()
    proxy = FastSamplingProxy(engine, azimuths_per_ring=4)

    first = proxy.evaluate(case)
    second = proxy.evaluate(case)

    assert first == second
    assert first["source_fragment_count"] == 60
    assert first["representative_ray_count"] == 24
    assert first["represented_fragment_count"] == pytest.approx(60.0)
    assert first["weight_normalization_error"] == pytest.approx(0.0)


def test_fast_sampling_proxy_outputs_finite_nonnegative_sampling_metrics():
    engine, case = _engine_and_case()
    result = FastSamplingProxy(engine, azimuths_per_ring=4).evaluate(case)

    for name in (
        "expected_armor_hits",
        "expected_armor_penetrations",
        "expected_total_component_hits",
        "expected_total_component_penetrations",
        "expected_total_component_absorbed_energy_j",
        "expected_total_component_damage_energy_j",
        "sampling_score",
    ):
        assert result[name] >= 0.0
    assert set(result["component_metrics"]) == set(engine.compiled_target.component_order)
    assert all(0.0 <= value <= 1.0 for value in result["tree_outputs"].values())


def test_fast_sampling_proxy_ray_resolution_preserves_total_fragment_weight():
    engine, case = _engine_and_case()
    coarse = FastSamplingProxy(engine, azimuths_per_ring=2).evaluate(case)
    fine = FastSamplingProxy(engine, azimuths_per_ring=10).evaluate(case)

    assert coarse["representative_ray_count"] == 12
    assert fine["representative_ray_count"] == 60
    assert coarse["represented_fragment_count"] == pytest.approx(60.0)
    assert fine["represented_fragment_count"] == pytest.approx(60.0)


@pytest.mark.parametrize("value", [0, -1])
def test_fast_sampling_proxy_rejects_nonpositive_azimuth_resolution(value: int):
    engine, _ = _engine_and_case()
    with pytest.raises(ValueError, match="azimuths_per_ring"):
        FastSamplingProxy(engine, azimuths_per_ring=value)


def test_selection_ledger_records_exact_union_inclusion_probabilities_and_weights():
    rows = [
        {"candidate_id": f"a-{index}", "approach_profile_cell": "a"}
        for index in range(5)
    ] + [
        {"candidate_id": f"b-{index}", "approach_profile_cell": "b"}
        for index in range(8)
    ]
    greedy = ["a-0", "a-1", "b-0", "b-1"]
    random = ["a-1", "a-3", "b-2", "b-7"]

    design = _selection_design_ledger(
        rows,
        greedy_ids=greedy,
        random_ids=random,
        per_cell=2,
    )
    by_id = {row["candidate_id"]: row for row in design["candidate_ledger"]}

    assert by_id["a-0"]["union_inclusion_probability"] == 1.0
    assert by_id["a-2"]["union_inclusion_probability"] == pytest.approx(2 / 5)
    assert by_id["b-3"]["union_inclusion_probability"] == pytest.approx(2 / 8)
    assert by_id["b-2"]["design_weight_raw"] == pytest.approx(4.0)
    assert by_id["a-1"]["selection_role"] == "greedy_and_random"
    assert by_id["a-2"]["selected"] is False
    assert design["random_exploration_per_cell"] == 2
    assert design["positivity_min"] == pytest.approx(0.25)
    assert 0.0 < design["selected_design_ess_fraction"] <= 1.0


def test_stratified_bootstrap_preserves_each_cell_sample_count():
    strata = ["a", "a", "b", "b", "b"]
    observed_compositions: list[tuple[int, int]] = []

    def metric(indices):
        composition = (
            sum(strata[int(index)] == "a" for index in indices),
            sum(strata[int(index)] == "b" for index in indices),
        )
        observed_compositions.append(composition)
        return float(sum(int(index) for index in indices))

    interval = _stratified_bootstrap_interval(
        strata,
        metric,
        seed=13,
        draws=50,
    )

    assert interval is not None
    assert observed_compositions
    assert set(observed_compositions) == {(2, 3)}


def test_arm_difference_is_paired_within_cell_before_cell_bootstrap():
    teacher_by_id = {
        "a-overlap": {
            "status": "ok",
            "approach_profile_cell": "a",
            "any_component_support": 1,
            "total_component_hits": 10.0,
            "total_component_deposited_energy_j": 100.0,
        },
        "a-greedy": {
            "status": "ok",
            "approach_profile_cell": "a",
            "any_component_support": 1,
            "total_component_hits": 10.0,
            "total_component_deposited_energy_j": 100.0,
        },
        "a-random": {
            "status": "ok",
            "approach_profile_cell": "a",
            "any_component_support": 0,
            "total_component_hits": 0.0,
            "total_component_deposited_energy_j": 0.0,
        },
        "b-overlap": {
            "status": "ok",
            "approach_profile_cell": "b",
            "any_component_support": 1,
            "total_component_hits": 4.0,
            "total_component_deposited_energy_j": 40.0,
        },
        "b-greedy": {
            "status": "ok",
            "approach_profile_cell": "b",
            "any_component_support": 0,
            "total_component_hits": 0.0,
            "total_component_deposited_energy_j": 0.0,
        },
        "b-random": {
            "status": "ok",
            "approach_profile_cell": "b",
            "any_component_support": 1,
            "total_component_hits": 2.0,
            "total_component_deposited_energy_j": 20.0,
        },
    }

    result = _cell_stratified_arm_difference_bootstrap(
        ["a-overlap", "a-greedy", "b-overlap", "b-greedy"],
        ["a-overlap", "a-random", "b-overlap", "b-random"],
        teacher_by_id,
        draws=200,
        seed=17,
    )

    hits = result["mean_total_component_hits_difference"]
    assert result["estimand"] == "equal_approach_profile_cell_mean_difference"
    assert result["bootstrap_unit"] == "approach_profile_cell"
    assert result["cell_count"] == 2
    assert hits["cell_differences"] == {"a": 5.0, "b": -1.0}
    assert hits["point"] == pytest.approx(2.0)
    assert hits["ci95"][0] <= -1.0
    assert hits["ci95"][1] >= 5.0


def test_accuracy_gate_consumes_only_stratified_random_arm():
    teacher_rows = []
    proxy_by_id = {}
    for candidate_id, cell, support, score in (
        ("random-a", "a", 0, 0.1),
        ("random-b", "b", 1, 0.9),
        ("greedy-only", "a", 1, 0.0),
    ):
        teacher_rows.append(
            {
                "candidate_id": candidate_id,
                "status": "ok",
                "approach_profile_cell": cell,
                "any_component_support": support,
                "total_component_hits": float(support),
                "total_component_deposited_energy_j": float(support),
                "components": {
                    "component": {"positive_support": support}
                },
            }
        )
        proxy_by_id[candidate_id] = {
            "expected_total_component_hits": score,
            "expected_total_component_absorbed_energy_j": score,
            "component_support_threshold": 0.5,
            "component_metrics": {"component": {"expected_hits": score}},
        }

    report = _random_arm_accuracy_report(
        proxy_by_id,
        teacher_rows,
        ["random-a", "random-b"],
        bootstrap_draws=20,
        bootstrap_seed=7,
    )

    assert report["evaluation_arm"] == "stratified_random_only"
    assert report["evaluated_candidate_ids"] == ["random-a", "random-b"]
    assert report["case_count"] == 2
    assert report["case_any_hit_auc"] == 1.0
