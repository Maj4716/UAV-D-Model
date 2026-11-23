import json
from pathlib import Path

import pytest

from damage_engine.engine import DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.sensitivity_execution import (
    C3_PARAMETER_IDS,
    bind_legacy_case_to_c3_variant,
    build_c3_variant,
    parameter_point,
)


ROOT = Path(__file__).resolve().parents[1]


def _documents():
    target = json.loads(
        (ROOT / "configs/targets/target.fragment_only.v1.json").read_text(
            encoding="utf-8"
        )
    )
    registry = json.loads(
        (ROOT / "configs/physics/loitering_impact.json").read_text(
            encoding="utf-8"
        )
    )
    case = json.loads(
        (ROOT / "configs/cases/loitering_impact.fragment_only.v1.json").read_text(
            encoding="utf-8"
        )
    )
    return target, registry, case


def _point():
    return {
        "armor_thickness_scale": 0.5,
        "component_threshold_scale": 0.25,
        "v50_coefficient_scale": 1.1,
        "gurney_speed_scale": 1.2,
        "direction_spread_sigma_deg": 3.0,
    }


def test_c3_variant_maps_all_five_factors_without_mutating_baseline():
    target, registry, _ = _documents()
    original_armor = target["armor"][0]["response"]["equivalent_thickness"]["value"]
    original_component = registry["fragment_response_models"][
        "demo_component_response"
    ]
    original_threshold = original_component["damage_states"]["degraded"][
        "damage_energy_50_j"
    ]
    original_v50 = original_component["ballistic_limit"]["coefficient"]

    variant_target, variant_registry, variant_id = build_c3_variant(
        target, registry, _point()
    )

    assert variant_target["armor"][0]["response"]["equivalent_thickness"][
        "value"
    ] == pytest.approx(original_armor * 0.5)
    response = variant_registry.data["fragment_response_models"][
        "demo_component_response"
    ]
    assert response["damage_states"]["degraded"][
        "damage_energy_50_j"
    ] == pytest.approx(original_threshold * 0.25)
    assert response["ballistic_limit"]["coefficient"] == pytest.approx(
        original_v50 * 1.1
    )
    assert all(
        source["direction"]["spread_sigma_deg"] == pytest.approx(3.0)
        for source in variant_registry.data["fragment_source_models"].values()
    )
    assert variant_registry.data["provenance"]["idea2_c3_sensitivity"][
        "writeback_allowed"
    ] is False
    assert len(variant_id) == 16
    assert target["armor"][0]["response"]["equivalent_thickness"][
        "value"
    ] == original_armor
    assert original_component["damage_states"]["degraded"][
        "damage_energy_50_j"
    ] == original_threshold


def test_c3_case_binding_preserves_common_random_number_seed_and_validates():
    target, registry, case = _documents()
    target_variant, registry_variant, variant_id = build_c3_variant(
        target, registry, _point()
    )
    engine = DamageEngine(target_variant, registry_variant)
    seed = case["analysis"]["aleatory_seed"]

    bound = bind_legacy_case_to_c3_variant(
        case,
        engine=engine,
        point=_point(),
        variant_id=variant_id,
        replicates=2,
    )

    assert bound["analysis"]["aleatory_seed"] == seed
    assert bound["analysis"]["replicates"] == 2
    assert bound["warhead"]["fragment_source"][
        "gurney_constant_mps"
    ] == pytest.approx(case["warhead"]["fragment_source"]["gurney_constant_mps"] * 1.2)
    assert bound["target_ref"]["content_hash"] == engine.target_hash
    assert engine.prepare_case(bound)["case_id"] == bound["case_id"]


def test_c3_parameter_contract_fails_closed_on_order_or_field_drift():
    with pytest.raises(InputValidationError, match="parameter order"):
        parameter_point(tuple(reversed(C3_PARAMETER_IDS)), [1, 1, 1, 1, 1])
    target, registry, _ = _documents()
    bad = _point()
    bad["unknown"] = 1.0
    with pytest.raises(InputValidationError, match="missing or unknown"):
        build_c3_variant(target, registry, bad)
