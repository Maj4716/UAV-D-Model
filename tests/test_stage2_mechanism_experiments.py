from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from damage_engine.engine import DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.fragments import _sample_direction_noise
from damage_engine.registry import PhysicsRegistry
from damage_engine.stage2_mechanism import (
    analyze_e3_factorial,
    build_c2_report,
    build_e3_variant,
    build_e5_variant,
    e3_factorial_design,
    e5_noise_design,
    extract_stage2_case_outcome,
    prepare_stage2_case,
)
from damage_engine.utils import load_json
from tools.run_stage2_mechanism_experiments import main as stage2_main


ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = ROOT / "configs" / "targets" / "target.fragment_only.v1.json"
PHYSICS_PATH = ROOT / "configs" / "physics" / "loitering_impact.json"
TAYLOR_SHAPIRO_PATH = (
    ROOT / "configs" / "physics" / "loitering_impact.taylor_shapiro.v2.json"
)
CASE_PATH = (
    ROOT / "configs" / "cases" / "loitering_impact.fragment_only.v1.json"
)


def test_stage2_designs_match_documented_sizes_and_noise_shape_extension() -> None:
    e3 = e3_factorial_design()
    assert len(e3) == 16
    assert len({row["variant_id"] for row in e3}) == 16
    assert {
        row["factors"]["F_direction_model"] for row in e3
    } == {"legacy_hollow", "solid_cone"}

    assert len(e5_noise_design()) == 9
    extended = e5_noise_design(("normal", "uniform", "laplace"))
    assert len(extended) == 27
    assert {
        row["factors"]["direction_noise_distribution"] for row in extended
    } == {"normal", "uniform", "laplace"}


@pytest.mark.parametrize("distribution", ["normal", "uniform", "laplace"])
def test_direction_noise_shapes_share_the_declared_standard_deviation(
    distribution: str,
) -> None:
    values = np.asarray(
        _sample_direction_noise(
            np.random.default_rng(20261120),
            0.2,
            size=100_000,
            distribution=distribution,
        )
    )
    assert np.mean(values) == pytest.approx(0.0, abs=0.003)
    assert np.std(values) == pytest.approx(0.2, abs=0.003)


def test_e3_variant_builds_transparent_armor_scaled_thresholds_and_solid_cones() -> None:
    target = load_json(TARGET_PATH)
    registry = PhysicsRegistry.from_file(PHYSICS_PATH)
    factors = {
        "A_armor_mm": 1.0,
        "T_component_threshold_factor": 0.1,
        "F_direction_model": "solid_cone",
        "W_gurney_factor": 1.3,
    }
    variant_target, variant_registry = build_e3_variant(
        target,
        registry.data,
        factors,
    )

    assert all(
        armor["response"]["equivalent_thickness"]["value"] == 1.0
        for armor in variant_target["armor"]
    )
    component = variant_registry.data["fragment_response_models"][
        "demo_component_response"
    ]
    assert component["damage_states"]["degraded"]["damage_energy_50_j"] == 35.0
    for source_id in (
        "loitering_small_front_taylor_source",
        "loitering_medium_front_taylor_source",
        "loitering_large_front_taylor_source",
    ):
        direction = variant_registry.data["fragment_source_models"][source_id][
            "direction"
        ]
        assert direction["kind"] == "cone"
        assert direction["axis_mode"] == "impact_inward"
        assert direction["half_angle_deg"] > 20.0


def test_e3_and_e5_refuse_taylor_shapiro_as_an_implicit_factor_level() -> None:
    target = load_json(TARGET_PATH)
    shapiro = PhysicsRegistry.from_file(TAYLOR_SHAPIRO_PATH)
    with pytest.raises(InputValidationError, match="legacy sources"):
        build_e3_variant(target, shapiro.data, e3_factorial_design()[0]["factors"])
    with pytest.raises(InputValidationError, match="legacy fragment source"):
        build_e5_variant(shapiro.data, e5_noise_design()[0]["factors"])


def test_e5_variant_binds_sigma_tau_and_equal_variance_distribution() -> None:
    registry = PhysicsRegistry.from_file(PHYSICS_PATH)
    factors = {
        "direction_noise_distribution": "laplace",
        "sigma_deg": 5.0,
        "tau": 0.16,
    }
    variant = build_e5_variant(registry.data, factors)
    for source_id in (
        "loitering_small_front_taylor_source",
        "loitering_medium_front_taylor_source",
        "loitering_large_front_taylor_source",
    ):
        direction = variant.data["fragment_source_models"][source_id]["direction"]
        assert direction["spread_sigma_deg"] == 5.0
        assert direction["spread_distribution"] == "laplace"
    assert all(
        model["ballistic_limit"]["transition_width_ratio"] == 0.16
        for model in variant.data["fragment_response_models"].values()
    )


def test_one_e3_corner_executes_with_crn_metadata_and_offline_tree_trace() -> None:
    target = load_json(TARGET_PATH)
    registry = PhysicsRegistry.from_file(PHYSICS_PATH)
    factors = {
        "A_armor_mm": 1.0,
        "T_component_threshold_factor": 0.1,
        "F_direction_model": "solid_cone",
        "W_gurney_factor": 1.3,
    }
    variant_target, variant_registry = build_e3_variant(
        target,
        registry.data,
        factors,
    )
    engine = DamageEngine(variant_target, variant_registry)
    base_case = load_json(CASE_PATH)
    original_seed = base_case["analysis"]["aleatory_seed"]
    case = prepare_stage2_case(
        base_case,
        target=variant_target,
        target_hash=engine.target_hash,
        registry=variant_registry,
        variant_id="test-e3-corner",
        replicates=1,
        gurney_factor=1.3,
        direction_level="solid_cone",
        factors=factors,
    )
    assert case["analysis"]["aleatory_seed"] == original_seed
    assert case["warhead"]["fragment_source"]["direction_model_id"] == "cone"
    assert case["warhead"]["fragment_source"]["gurney_constant_mps"] == pytest.approx(
        2440.0 * 1.3
    )
    result = engine.run(case).to_dict()
    outcome = extract_stage2_case_outcome(
        result,
        variant_target,
        base_case_id=base_case["case_id"],
    )
    assert outcome["base_case_id"] == base_case["case_id"]
    assert outcome["offline_tree_max_abs_error"] <= 1.0e-12
    assert len(outcome["identity_tree_case_values"]) > len(
        outcome["current_tree_case_values"]
    )


def _synthetic_outcome(base_case_id: str, effect: float) -> dict:
    probability = float(np.clip(0.25 + effect, 0.0, 1.0))
    return {
        "base_case_id": base_case_id,
        "status": "ok",
        "component_zero_rate": float(np.clip(0.8 - effect, 0.0, 1.0)),
        "any_component_support": float(effect > 0.0),
        "positive_component_energy_j": [10.0 + effect, 20.0 + effect],
        "domain_complete": 1.0,
        "affected_replicate_fraction": 0.0,
        "current_tree_case_values": [probability, probability],
        "identity_tree_case_values": [probability, probability, probability],
        "offline_tree_max_abs_error": 0.0,
    }


def test_e3_factorial_analysis_uses_paired_case_bootstrap_and_tree_factor() -> None:
    records = []
    for row in e3_factorial_design():
        armor_effect = (
            0.1 if row["factors"]["A_armor_mm"] == 1.0 else 0.0
        )
        records.append(
            {
                **row,
                "outcomes": [
                    _synthetic_outcome("case-a", armor_effect),
                    _synthetic_outcome("case-b", armor_effect + 0.01),
                ],
            }
        )
    report = analyze_e3_factorial(
        records,
        bootstrap_replicates=4,
        bootstrap_seed=17,
    )

    assert report["common_random_numbers"] is True
    assert len(report["physical_variants"]) == 16
    assert "A" in report["effects"]
    assert "A:Tr" in report["effects"]
    assert report["effects"]["A"]["component_zero_rate"][
        "effect_high_minus_low"
    ] < 0.0
    assert report["effects"]["A"]["component_zero_rate"][
        "bootstrap_draw_count"
    ] == 4


def test_c2_reports_three_profile_golds_and_v50_half_probability() -> None:
    report = build_c2_report(
        PhysicsRegistry.from_file(PHYSICS_PATH),
        direction_repeats=2,
        v50_trials_per_speed=64,
        seed=29,
    )
    assert len(report["profile_reports"]) == 3
    assert all(
        item["maximum_gurney_speed_abs_error_mps"] <= 1.0e-12
        for item in report["profile_reports"]
    )
    assert all(
        math.isfinite(item["angle_gold"]["expected_mean_angle_deg"])
        for item in report["profile_reports"]
    )
    for response in report["fragment_response_reports"].values():
        assert response["analytic_probability_at_v50"] == pytest.approx(0.5)
        assert response["transition_width_ratio_tau"] > 0.0
        assert response["ten_to_ninety_speed_ratio"] > 1.0


def test_stage2_cli_dry_runs_bind_default_design_without_teacher_execution(
    tmp_path: Path,
) -> None:
    e3_output = tmp_path / "e3"
    assert stage2_main(
        [
            "e3",
            "--cases",
            str(CASE_PATH),
            "--max-cases",
            "1",
            "--replicates",
            "1",
            "--bootstrap-replicates",
            "0",
            "--dry-run",
            "--output",
            str(e3_output),
        ]
    ) == 0
    e3_report = json.loads((e3_output / "report.json").read_text(encoding="utf-8"))
    assert e3_report["physical_variant_count"] == 16
    assert e3_report["planned_teacher_case_runs"] == 16

    e5_output = tmp_path / "e5"
    assert stage2_main(
        [
            "e5",
            "--cases",
            str(CASE_PATH),
            "--max-cases",
            "1",
            "--replicates",
            "1",
            "--distributions",
            "normal",
            "uniform",
            "laplace",
            "--dry-run",
            "--output",
            str(e5_output),
        ]
    ) == 0
    e5_report = json.loads((e5_output / "report.json").read_text(encoding="utf-8"))
    assert e5_report["variant_count"] == 27
    assert e5_report["planned_teacher_case_runs"] == 27
