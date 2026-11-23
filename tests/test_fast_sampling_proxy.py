from __future__ import annotations

import json
import math
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


def test_fast_sampling_proxy_exposure_scope_skips_damage_tree(monkeypatch):
    engine, case = _engine_and_case()
    proxy = FastSamplingProxy(engine, azimuths_per_ring=2)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("damage tree must not run for exposure-only output")

    monkeypatch.setattr(type(engine.damage_tree), "evaluate", fail_if_called)

    exposure = proxy.evaluate(case, output_scope="exposure")

    assert exposure["output_scope"] == "exposure"
    assert exposure["component_state_probabilities"] is None
    assert exposure["tree_outputs"] is None
    assert exposure["not_computed"] == [
        "component_state_probabilities",
        "tree_outputs",
    ]
    with pytest.raises(
        AssertionError, match="damage tree must not run for exposure-only output"
    ):
        proxy.evaluate(case, output_scope="full")


def test_fast_sampling_proxy_exposure_metrics_match_full_scope():
    engine, case = _engine_and_case()
    proxy = FastSamplingProxy(engine, azimuths_per_ring=2)

    default = proxy.evaluate(case)
    full = proxy.evaluate(case, output_scope="full")
    exposure = proxy.evaluate(case, output_scope="exposure")

    assert full == default
    for name in (
        "expected_armor_hits",
        "expected_armor_penetrations",
        "expected_total_component_hits",
        "expected_total_component_penetrations",
        "expected_total_component_absorbed_energy_j",
        "expected_total_component_damage_energy_j",
        "terminal_packet_weight",
        "out_of_domain_packet_weight",
        "proxy_domain_status",
        "proxy_domain_coverage",
        "supported_component_ids",
        "supported_component_count",
        "component_metrics",
    ):
        assert exposure[name] == full[name]
    expected_exposure_score = sum(
        math.log1p(item["expected_hits"])
        for item in exposure["component_metrics"].values()
    )
    expected_exposure_score += 0.25 * math.log1p(
        exposure["expected_total_component_damage_energy_j"]
    )
    assert exposure["sampling_score"] == pytest.approx(expected_exposure_score)
    assert exposure["sampling_score"] <= full["sampling_score"]


def test_fast_sampling_proxy_rejects_unknown_output_scope():
    engine, case = _engine_and_case()
    proxy = FastSamplingProxy(engine, azimuths_per_ring=2)

    with pytest.raises(ValueError, match="output_scope"):
        proxy.evaluate(case, output_scope="labels")


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


# --- numerical options added in proxy 0.4.0 -------------------------------


_EXPOSURE_KEYS = (
    "expected_armor_hits",
    "expected_armor_penetrations",
    "expected_total_component_hits",
    "expected_total_component_penetrations",
    "expected_total_component_absorbed_energy_j",
    "expected_total_component_damage_energy_j",
    "terminal_packet_weight",
    "out_of_domain_packet_weight",
    "component_metrics",
    "component_hit_probabilities",
)


def test_fast_sampling_proxy_new_options_default_to_historical_behavior():
    engine, case = _engine_and_case()
    default = FastSamplingProxy(engine, azimuths_per_ring=4)
    explicit = FastSamplingProxy(
        engine,
        azimuths_per_ring=4,
        source_realizations=1,
        damage_accumulation="independent_hits",
    )

    assert default.source_realizations == 1
    assert default.damage_accumulation == "independent_hits"
    result = default.evaluate(case)
    assert result == explicit.evaluate(case)
    assert result["source_realizations"] == 1
    assert result["damage_accumulation"] == "independent_hits"
    assert "component_damage_cluster_counts" not in result


@pytest.mark.parametrize("value", [0, -2, 1.5])
def test_fast_sampling_proxy_rejects_invalid_source_realizations(value):
    engine, _ = _engine_and_case()
    with pytest.raises(ValueError, match="source_realizations"):
        FastSamplingProxy(engine, source_realizations=value)


def test_fast_sampling_proxy_rejects_unknown_damage_accumulation():
    engine, _ = _engine_and_case()
    with pytest.raises(ValueError, match="damage_accumulation"):
        FastSamplingProxy(engine, damage_accumulation="spatial_cumulative_damage_energy")


def test_source_realizations_average_independent_dispersion_draws(monkeypatch):
    import damage_engine.fast_sampling_proxy as module

    engine, case = _engine_and_case()
    first = FastSamplingProxy(engine, azimuths_per_ring=1_000).evaluate(case)
    original_seed = module._stable_seed

    def second_realization_seed(**kwargs):
        assert kwargs["realization"] == 0
        return original_seed(**{**kwargs, "realization": 1})

    monkeypatch.setattr(module, "_stable_seed", second_realization_seed)
    second = FastSamplingProxy(engine, azimuths_per_ring=1_000).evaluate(case)
    monkeypatch.setattr(module, "_stable_seed", original_seed)
    averaged = FastSamplingProxy(
        engine, azimuths_per_ring=1_000, source_realizations=2
    ).evaluate(case)

    assert first["component_metrics"] != second["component_metrics"]
    assert averaged["source_realizations"] == 2
    assert averaged["represented_fragment_count"] == pytest.approx(60.0)
    assert averaged["weight_normalization_error"] == pytest.approx(0.0)
    for name in (
        "expected_total_component_hits",
        "expected_armor_hits",
        "out_of_domain_packet_weight",
    ):
        assert averaged[name] == pytest.approx(0.5 * (first[name] + second[name]))
    for component_id, metrics in averaged["component_metrics"].items():
        for name, value in metrics.items():
            assert value == pytest.approx(
                0.5
                * (
                    first["component_metrics"][component_id][name]
                    + second["component_metrics"][component_id][name]
                )
            )
        assert averaged["component_hit_probabilities"][component_id] == pytest.approx(
            0.5
            * (
                first["component_hit_probabilities"][component_id]
                + second["component_hit_probabilities"][component_id]
            )
        )
        for state, value in averaged["component_state_probabilities"][
            component_id
        ].items():
            assert value == pytest.approx(
                0.5
                * (
                    first["component_state_probabilities"][component_id][state]
                    + second["component_state_probabilities"][component_id][state]
                )
            )


def test_component_hit_probability_is_arrival_union_bounded_by_poisson_form():
    engine, case = _engine_and_case()
    for azimuths in (2, 1_000):
        result = FastSamplingProxy(engine, azimuths_per_ring=azimuths).evaluate(case)
        for component_id, metrics in result["component_metrics"].items():
            expected_hits = metrics["expected_hits"]
            hit_probability = result["component_hit_probabilities"][component_id]
            assert 0.0 <= hit_probability <= 1.0
            assert (hit_probability == 0.0) == (expected_hits == 0.0)
            # (1-r)^w <= exp(-w r): the union is never below the Poisson form.
            assert hit_probability >= -math.expm1(-expected_hits) - 1.0e-12
            assert hit_probability <= min(1.0, expected_hits) + 1.0e-12


def test_out_of_domain_mass_fraction_replaces_any_packet_flag(monkeypatch):
    import damage_engine.fast_sampling_proxy as module
    from damage_engine.errors import OutOfDomainError

    engine, case = _engine_and_case()
    clean = FastSamplingProxy(engine, azimuths_per_ring=1_000).evaluate(case)
    assert clean["out_of_domain_mass_fraction"] == pytest.approx(
        clean["proxy_domain_coverage"]["out_of_domain_packet_fraction"]
    )
    assert clean["proxy_domain_coverage"]["comparison_quantity"] == (
        "out_of_domain_mass_fraction"
    )

    def refuse_armor(impact, response, registry, **kwargs):
        if response.get("fragment_response_model_id") == "demo_armor_response":
            raise OutOfDomainError("test-only armor refusal")
        return original(impact, response, registry, **kwargs)

    original = module._representative_expected_path
    monkeypatch.setattr(module, "_representative_expected_path", refuse_armor)
    censored = FastSamplingProxy(engine, azimuths_per_ring=1_000).evaluate(case)
    coverage = censored["proxy_domain_coverage"]

    assert censored["proxy_domain_status"] == "partial_proxy_domain_censored"
    assert 0.0 < censored["out_of_domain_mass_fraction"] <= 1.0
    # Fixed-mass fragments: mass and packet-weight fractions coincide.
    assert censored["out_of_domain_mass_fraction"] == pytest.approx(
        coverage["out_of_domain_packet_fraction"]
    )
    assert coverage["out_of_domain_mass_fraction"] == censored[
        "out_of_domain_mass_fraction"
    ]
    assert coverage["out_of_domain_packet_weight_by_role"]["armor"] == pytest.approx(
        censored["out_of_domain_packet_weight"]
    )
    assert coverage["out_of_domain_packet_weight_by_role"]["component"] == 0.0


def test_complete_link_clusters_bound_cluster_diameter():
    import numpy as np

    from damage_engine.fast_sampling_proxy import _complete_link_clusters

    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.06, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    # 0-1 merge (0.02 m); 2 would chain through 1 (0.04 m) but its complete-link
    # distance to {0, 1} is 0.06 m > 0.05 m, so it stays separate.
    assert _complete_link_clusters(points, 0.05) == [[0, 1], [2], [3]]
    assert _complete_link_clusters(points, 0.07) == [[0, 1, 2], [3]]
    assert _complete_link_clusters(points[:1], 0.05) == [[0]]
    assert _complete_link_clusters(points[:0], 0.05) == []


def test_cluster_energy_distribution_enumerates_member_branches():
    import itertools

    import numpy as np

    from damage_engine.fast_sampling_proxy import (
        _ComponentImpact,
        _cluster_energy_distribution,
        _expected_state_probability,
    )

    members = [
        _ComponentImpact(np.zeros(3), 0.5, 1.0, 0.25, 900.0, 400.0),
        _ComponentImpact(np.zeros(3), 1.0, 1.0, 0.8, 1300.0, 200.0),
        _ComponentImpact(np.zeros(3), 0.3, 1.0, 1.0, 700.0, 50.0),
    ]
    energies, probabilities = _cluster_energy_distribution(members)
    assert probabilities.sum() == pytest.approx(1.0)

    def logistic(energy: float) -> float:
        if energy <= 0.0:
            return 0.0
        return 0.85 / (1.0 + (energy / 1900.0) ** -3.5)

    brute_force = 0.0
    branches = []
    for member in members:
        r, p = member.reach_probability, member.perforation_probability
        branches.append(
            [
                (1.0 - r, 0.0),
                (r * p, member.perforated_damage_energy_j),
                (r * (1.0 - p), member.stopped_damage_energy_j),
            ]
        )
    for combination in itertools.product(*branches):
        probability = math.prod(item[0] for item in combination)
        brute_force += probability * logistic(sum(item[1] for item in combination))
    assert _expected_state_probability(
        energies,
        probabilities,
        energy_50_j=1900.0,
        log_slope=3.5,
        vulnerable_ratio=0.85,
    ) == pytest.approx(brute_force, rel=1.0e-12)


def test_deterministic_clusters_reproduce_teacher_spatial_accumulator():
    import numpy as np

    from damage_engine.component_damage import ComponentDamageAccumulator
    from damage_engine.fast_sampling_proxy import (
        _ComponentImpact,
        _cluster_energy_distribution,
        _complete_link_clusters,
        _expected_state_probability,
    )

    registry = PhysicsRegistry.from_file(ROOT / "configs/physics/loitering_impact.json")
    model = registry.fragment_response("demo_component_response")
    states = ("degraded", "failed")
    length = float(model["damage_correlation_length_m"])
    rng = np.random.default_rng(20261123)
    locations = rng.normal(0.0, 0.03, size=(9, 3))
    energies = rng.uniform(100.0, 1500.0, size=9)

    teacher = ComponentDamageAccumulator(
        ordered_states=states,
        state_models=model["damage_states"],
        vulnerable_ratio=0.85,
        accumulation_model="spatial_cumulative_damage_energy",
        spatial_correlation_length_m=length,
    )
    for energy, location in zip(energies, locations):
        teacher.record_impact(float(energy), location_m=location)
    expected = teacher.finalize()

    impacts = [
        _ComponentImpact(location, 1.0, 1.0, 1.0, float(energy), 0.0)
        for energy, location in zip(energies, locations)
    ]
    clusters = _complete_link_clusters(locations, length)
    assert len(clusters) == expected.spatial_cluster_count
    for state in states:
        survival = 1.0
        for cluster in clusters:
            distribution = _cluster_energy_distribution([impacts[i] for i in cluster])
            survival *= 1.0 - _expected_state_probability(
                *distribution,
                energy_50_j=float(model["damage_states"][state]["damage_energy_50_j"]),
                log_slope=float(model["damage_states"][state]["log_slope"]),
                vulnerable_ratio=0.85,
            )
        assert 1.0 - survival == pytest.approx(expected.raw_probabilities[state], abs=1e-12)


def test_registry_accumulation_changes_only_state_outputs():
    engine, case = _engine_and_case()
    for azimuths in (2, 1_000):
        historical = FastSamplingProxy(engine, azimuths_per_ring=azimuths).evaluate(case)
        registry = FastSamplingProxy(
            engine, azimuths_per_ring=azimuths, damage_accumulation="registry"
        ).evaluate(case)
        repeat = FastSamplingProxy(
            engine, azimuths_per_ring=azimuths, damage_accumulation="registry"
        ).evaluate(case)

        assert registry == repeat
        assert registry["damage_accumulation"] == "registry"
        for name in _EXPOSURE_KEYS:
            assert registry[name] == historical[name]
        counts = registry["component_damage_cluster_counts"]
        assert set(counts) == set(engine.compiled_target.component_order)
        for component_id, states in registry["component_state_probabilities"].items():
            assert 0.0 <= states["failed"] <= states["degraded"] <= 1.0
            if registry["component_metrics"][component_id]["expected_hits"] == 0.0:
                assert counts[component_id] == 0
                assert states == {"degraded": 0.0, "failed": 0.0}
        assert all(0.0 <= value <= 1.0 for value in registry["tree_outputs"].values())


def test_registry_accumulation_matches_historical_for_independent_hit_models():
    target = _load(ROOT / "configs/targets/target.fragment_only.v1.json")
    registry_data = _load(ROOT / "configs/physics/loitering_impact.json")
    registry_data["fragment_response_models"]["demo_component_response"][
        "damage_accumulation_model"
    ] = "independent_hits"
    case = _load(ROOT / "configs/cases/loitering_impact.fragment_only.v1.json")
    case["analysis"]["capture_fragment_audit"] = False
    case["analysis"]["capture_replicate_trace"] = False
    from damage_engine.utils import content_hash

    engine = DamageEngine(target, PhysicsRegistry(registry_data))
    case["target_ref"]["content_hash"] = content_hash(target)
    historical = FastSamplingProxy(engine, azimuths_per_ring=4).evaluate(case)
    registry = FastSamplingProxy(
        engine, azimuths_per_ring=4, damage_accumulation="registry"
    ).evaluate(case)

    for component_id, states in historical["component_state_probabilities"].items():
        for state, value in states.items():
            assert registry["component_state_probabilities"][component_id][
                state
            ] == pytest.approx(value, abs=1.0e-12)


def test_rejects_unknown_domain_censoring():
    engine, _ = _engine_and_case()
    with pytest.raises(ValueError, match="domain_censoring"):
        FastSamplingProxy(engine, domain_censoring="teacher")


def _high_obliquity_impact(angle_deg: float, speed_mps: float):
    import numpy as np

    from damage_engine.penetration import FragmentImpact

    angle = math.radians(angle_deg)
    return FragmentImpact(
        speed_mps=speed_mps,
        mass_kg=0.006,
        direction=np.asarray([math.sin(angle), 0.0, -math.cos(angle)]),
        surface_normal=np.asarray([0.0, 0.0, 1.0]),
        equivalent_thickness_m=0.01,
    )


@pytest.mark.parametrize("speed_mps", [300.0, 1200.0, 2500.0])
def test_branch_probability_matches_teacher_sampling_of_high_obliquity_bound(speed_mps):
    import numpy as np

    from damage_engine.errors import OutOfDomainError
    from damage_engine.fast_sampling_proxy import _representative_expected_path
    from damage_engine.layered_penetration import resolve_penetration_path

    registry = PhysicsRegistry.from_file(
        ROOT / "configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json"
    )
    response = {
        "material_id": "demo_steel",
        "fragment_response_model_id": "demo_armor_response",
        "layer_stack_id": None,
    }
    impact = _high_obliquity_impact(85.0, speed_mps)  # validated to 80 deg, bound to 89.9 deg

    expectation = _representative_expected_path(
        impact,
        response,
        registry,
        layer_quadrature_samples=16,
        domain_censoring="branch_probability",
    )
    # The bound certifies non-perforation or refuses; it never perforates.
    assert expectation.penetration_probability == 0.0
    assert expectation.representative_residual_speed_mps == 0.0
    assert 0.0 < expectation.out_of_domain_probability < 1.0

    # Teacher semantics: one uniform ballistic-limit quantile per contact.
    rng = np.random.default_rng(20261123)
    draws, refused, stopped_damage = 4000, 0, []
    for _ in range(draws):
        try:
            outcome = resolve_penetration_path(impact, response, registry, rng)
        except OutOfDomainError:
            refused += 1
            continue
        assert not outcome.perforated
        stopped_damage.append(outcome.damage_energy_j)
    p = expectation.out_of_domain_probability
    assert refused / draws == pytest.approx(p, abs=4.0 * math.sqrt(p * (1 - p) / draws) + 1e-3)
    assert expectation.stopped_damage_energy_j == pytest.approx(stopped_damage[0])
    assert expectation.expected_damage_energy_j == pytest.approx(
        (1.0 - p) * expectation.stopped_damage_energy_j
    )

    # The historical expectation refuses the whole contact.
    with pytest.raises(OutOfDomainError):
        _representative_expected_path(
            impact, response, registry, layer_quadrature_samples=16
        )


def test_branch_probability_refuses_contacts_beyond_every_bound():
    from damage_engine.fast_sampling_proxy import _representative_expected_path

    registry = PhysicsRegistry.from_file(
        ROOT / "configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json"
    )
    response = {
        "material_id": "demo_steel",
        "fragment_response_model_id": "demo_armor_response",
        "layer_stack_id": None,
    }
    expectation = _representative_expected_path(
        _high_obliquity_impact(89.95, 900.0),
        response,
        registry,
        layer_quadrature_samples=16,
        domain_censoring="branch_probability",
    )
    assert expectation.out_of_domain_probability == 1.0
    assert expectation.penetration_probability == 0.0
    assert expectation.expected_absorbed_energy_j == 0.0
    assert expectation.expected_damage_energy_j == 0.0


def test_branch_probability_equals_historical_inside_validated_domain():
    from damage_engine.fast_sampling_proxy import _representative_expected_path

    registry = PhysicsRegistry.from_file(
        ROOT / "configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json"
    )
    for material_id, model_id in (
        ("demo_steel", "demo_armor_response"),
        ("demo_electronics", "demo_component_response"),
    ):
        response = {
            "material_id": material_id,
            "fragment_response_model_id": model_id,
            "layer_stack_id": None,
        }
        for angle, speed in ((10.0, 400.0), (45.0, 1200.0), (70.0, 2500.0)):
            impact = _high_obliquity_impact(angle, speed)
            historical = _representative_expected_path(
                impact, response, registry, layer_quadrature_samples=16
            )
            branch = _representative_expected_path(
                impact,
                response,
                registry,
                layer_quadrature_samples=16,
                domain_censoring="branch_probability",
            )
            assert branch.out_of_domain_probability == 0.0
            for name in (
                "penetration_probability",
                "incidence_cosine",
                "v50_mps",
                "representative_residual_speed_mps",
                "expected_absorbed_energy_j",
                "expected_damage_energy_j",
                "perforated_damage_energy_j",
                "stopped_damage_energy_j",
            ):
                assert getattr(branch, name) == pytest.approx(
                    getattr(historical, name), rel=1e-12, abs=1e-12
                )


def test_branch_probability_counts_refused_component_contacts(monkeypatch):
    import damage_engine.fast_sampling_proxy as module
    from damage_engine.errors import OutOfDomainError

    engine, case = _engine_and_case()
    original = module.resolve_penetration_path

    def refuse_components(impact, response, registry, rng):
        if response.get("fragment_response_model_id") == "demo_component_response":
            raise OutOfDomainError("test-only component refusal")
        return original(impact, response, registry, rng)

    monkeypatch.setattr(module, "resolve_penetration_path", refuse_components)
    historical = FastSamplingProxy(engine, azimuths_per_ring=1_000).evaluate(case)
    branch = FastSamplingProxy(
        engine,
        azimuths_per_ring=1_000,
        domain_censoring="branch_probability",
        damage_accumulation="registry",
    ).evaluate(case)

    assert branch["domain_censoring"] == "branch_probability"
    # The teacher counts an unresolved component contact as a hit but assigns
    # no energy; the historical proxy drops the contact entirely.
    assert historical["expected_total_component_hits"] == 0.0
    assert branch["expected_total_component_hits"] > 0.0
    assert branch["expected_total_component_penetrations"] == 0.0
    assert branch["expected_total_component_damage_energy_j"] == 0.0
    assert branch["proxy_domain_coverage"]["out_of_domain_packet_weight_by_role"][
        "component"
    ] == pytest.approx(branch["expected_total_component_hits"])
    for states in branch["component_state_probabilities"].values():
        assert states == {"degraded": 0.0, "failed": 0.0}


def test_branch_probability_matches_historical_without_refusals():
    from damage_engine.utils import content_hash

    target = _load(ROOT / "configs/targets/target.fragment_only.v1.json")
    registry_data = _load(ROOT / "configs/physics/loitering_impact.json")
    for model in registry_data["fragment_response_models"].values():
        if "ballistic_limit" in model:
            model["ballistic_limit"]["maximum_obliquity_deg"] = 89.9999
    case = _load(ROOT / "configs/cases/loitering_impact.fragment_only.v1.json")
    case["analysis"]["capture_fragment_audit"] = False
    case["analysis"]["capture_replicate_trace"] = False
    case["target_ref"]["content_hash"] = content_hash(target)
    engine = DamageEngine(target, PhysicsRegistry(registry_data))
    historical = FastSamplingProxy(engine, azimuths_per_ring=1_000).evaluate(case)
    assert historical["out_of_domain_packet_weight"] == 0.0
    assert historical["expected_total_component_hits"] > 0.0
    branch = FastSamplingProxy(
        engine, azimuths_per_ring=1_000, domain_censoring="branch_probability"
    ).evaluate(case)
    for name in _EXPOSURE_KEYS:
        if name == "component_metrics":
            for component_id, metrics in historical[name].items():
                assert branch[name][component_id] == pytest.approx(
                    metrics, rel=1e-12, abs=1e-12
                )
        else:
            assert branch[name] == pytest.approx(
                historical[name], rel=1e-12, abs=1e-12
            )
    for component_id, states in historical["component_state_probabilities"].items():
        assert branch["component_state_probabilities"][component_id] == pytest.approx(
            states, rel=1e-12, abs=1e-12
        )
    assert branch["tree_outputs"] == pytest.approx(historical["tree_outputs"], rel=1e-12)


def test_recommended_options_construct_converged_teacher_predictor():
    from damage_engine.fast_sampling_proxy import (
        ONE_RAY_PER_FRAGMENT,
        TEACHER_PREDICTOR_OPTIONS,
    )

    engine, case = _engine_and_case()
    proxy = FastSamplingProxy(engine, **TEACHER_PREDICTOR_OPTIONS)
    assert proxy.azimuths_per_ring == ONE_RAY_PER_FRAGMENT
    assert proxy.damage_accumulation == "registry"
    assert proxy.domain_censoring == "branch_probability"
    assert proxy.source_realizations > 1
    result = proxy.evaluate(case)
    assert result["representative_ray_count"] == result["source_fragment_count"]
    with pytest.raises(TypeError):
        TEACHER_PREDICTOR_OPTIONS["azimuths_per_ring"] = 4  # type: ignore[index]
