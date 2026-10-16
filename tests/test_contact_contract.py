from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from damage_engine.contact_contract import (
    S_CONTACT_CONTRACT_DOCUMENT,
    S_CONTACT_CONTRACT_HASH,
    S_CONTACT_CONTRACT_ID,
    S_CONTACT_ESTIMAND,
    audit_s_contact,
    assert_contact_conditioned_estimand,
)
from damage_engine.engine import DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.impact import IMPACT_VALIDATION_HASH, resolve_impact_detonation
from damage_engine.registry import PhysicsRegistry
from damage_engine.utils import load_json
from damage_engine.utils import content_hash


ROOT = Path(__file__).resolve().parents[1]


def _engine() -> DamageEngine:
    return DamageEngine(
        load_json(ROOT / "raw_data" / "target.rebuilt.json"),
        PhysicsRegistry.from_file(
            ROOT / "examples" / "physics.loitering_impact.json"
        ),
    )


def _case(engine: DamageEngine) -> dict:
    raw = load_json(ROOT / "raw_data" / "case.loitering_impact.v1.json")
    return resolve_impact_detonation(raw, engine.compiled_target)


def _selection(*paths: str) -> dict:
    return {
        "schema_version": "1.0.0",
        "selection_rule_id": "frozen-surface-direction-design-v1",
        "selection_stage": "pre_outcome",
        "outcome_blind": True,
        "dependency_paths": list(
            paths
            or (
                "/munition/profile_id",
                "/encounter/trajectory_origin_m",
                "/encounter/velocity_target_mps",
                "/encounter/body_to_target_orientation_xyzw",
            )
        ),
    }


def _audit(engine: DamageEngine, case: dict, selection: dict | None = None) -> dict:
    return audit_s_contact(
        case,
        engine.compiled_target,
        data_role="model_train_reference",
        estimand=S_CONTACT_ESTIMAND,
        selection_declaration=selection or _selection(),
    )


def _reason(error: pytest.ExceptionInfo[InputValidationError]) -> str:
    return str(error.value.context["reason_code"])


def test_valid_case_returns_explicit_contact_conditioned_record_without_mutation() -> None:
    engine = _engine()
    case = _case(engine)
    original = deepcopy(case)

    record = _audit(engine, case)

    assert case == original
    assert record["contract_id"] == S_CONTACT_CONTRACT_ID
    assert record["status"] == "eligible_contact_conditioned"
    assert record["estimand"]["scope"] == "contact_conditioned_only"
    assert all(record["checks"].values())
    assert record["first_contact"]["collision_distance_m"] > 0.0
    assert record["selection"]["content_hash"].startswith("sha256:")


def test_contact_contract_hash_covers_outcome_markers_and_impact_validation() -> None:
    assert S_CONTACT_CONTRACT_DOCUMENT["outcome_dependency_markers"]
    assert S_CONTACT_CONTRACT_DOCUMENT["allowed_pre_outcome_dependency_paths"]
    assert (
        S_CONTACT_CONTRACT_DOCUMENT["impact_validation_hash"]
        == IMPACT_VALIDATION_HASH
    )
    assert S_CONTACT_CONTRACT_HASH == content_hash(S_CONTACT_CONTRACT_DOCUMENT)


def test_miss_is_refused_as_no_forward_armor_intersection() -> None:
    engine = _engine()
    case = _case(engine)
    case["encounter"]["trajectory_origin_m"] = [-10.0, 20.0, 20.0]

    with pytest.raises(InputValidationError) as error:
        _audit(engine, case)

    assert _reason(error) == "no_forward_armor_intersection"


def test_reverse_trajectory_is_distinguished_from_a_geometric_miss() -> None:
    engine = _engine()
    case = _case(engine)
    case["encounter"]["velocity_target_mps"] = [-120.0, 0.0, 0.0]
    # 180 degrees around +Z: body +X now follows the reversed velocity.
    case["encounter"]["body_to_target_orientation_xyzw"] = [0.0, 0.0, 1.0, 0.0]

    with pytest.raises(InputValidationError) as error:
        _audit(engine, case)

    assert _reason(error) == "armor_intersection_only_behind_origin"


def test_zero_velocity_is_refused() -> None:
    engine = _engine()
    case = _case(engine)
    case["encounter"]["velocity_target_mps"] = [0.0, 0.0, 0.0]

    with pytest.raises(InputValidationError) as error:
        _audit(engine, case)

    assert _reason(error) == "velocity_zero"


def test_origin_inside_armor_is_refused() -> None:
    engine = _engine()
    case = _case(engine)
    case["encounter"]["trajectory_origin_m"] = [0.0, 0.0, 1.2]

    with pytest.raises(InputValidationError) as error:
        _audit(engine, case)

    assert _reason(error) == "trajectory_origin_inside_armor"


def test_misaligned_body_axis_is_refused() -> None:
    engine = _engine()
    case = _case(engine)
    case["encounter"]["body_to_target_orientation_xyzw"] = [0.0, 0.0, 1.0, 0.0]

    with pytest.raises(InputValidationError) as error:
        _audit(engine, case)

    assert _reason(error) == "orientation_velocity_misaligned"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("detonation_time_s", 999.0),
        ("collision_distance_m", 999.0),
        ("collision_armor_id", "armor-does-not-exist"),
    ),
)
def test_stale_persisted_first_contact_fields_are_refused(field: str, value) -> None:
    engine = _engine()
    case = _case(engine)
    case["encounter"][field] = value

    with pytest.raises(InputValidationError) as error:
        _audit(engine, case)

    assert _reason(error) == "persisted_first_contact_mismatch"


@pytest.mark.parametrize(
    "path",
    (
        "/results/damage_tree/outputs/effect-k1",
        "/labels/kmfc-k-ge1",
        "/component_aux_y/fragment_deposited_energy_j",
        "/metadata/sampling/score_rarity",
        "/kmfc/q1",
        "/predictions/criterion_probability",
    ),
)
def test_outcome_aware_selection_dependencies_are_refused(path: str) -> None:
    engine = _engine()
    case = _case(engine)

    with pytest.raises(InputValidationError) as error:
        _audit(engine, case, _selection(path))

    assert _reason(error) == "selection_outcome_dependency"


def test_selection_must_explicitly_run_pre_outcome_and_be_outcome_blind() -> None:
    engine = _engine()
    case = _case(engine)
    declaration = _selection()
    declaration["outcome_blind"] = False

    with pytest.raises(InputValidationError) as error:
        _audit(engine, case, declaration)

    assert _reason(error) == "selection_not_outcome_blind"


def test_invalid_geometry_diagnostic_cannot_receive_a_formal_label() -> None:
    engine = _engine()
    case = _case(engine)

    with pytest.raises(InputValidationError) as error:
        audit_s_contact(
            case,
            engine.compiled_target,
            data_role="invalid_geometry_diagnostic",
            estimand=S_CONTACT_ESTIMAND,
            selection_declaration=_selection(),
        )

    assert _reason(error) == "diagnostic_role_has_no_formal_label"


@pytest.mark.parametrize(
    "estimand",
    (
        "hit_probability",
        "guidance_success_probability",
        "mission_success_probability",
        "unconditional_damage_probability",
    ),
)
def test_contact_conditioned_contract_refuses_unconditional_estimands(
    estimand: str,
) -> None:
    with pytest.raises(InputValidationError) as error:
        assert_contact_conditioned_estimand(estimand)

    assert _reason(error) == "unconditional_estimand_forbidden"
