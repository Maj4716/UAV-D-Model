from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from damage_engine.case_factory import CaseFactory
from damage_engine.contact_contract import (
    S_CONTACT_CONTRACT_HASH,
    S_CONTACT_CONTRACT_ID,
    S_CONTACT_ESTIMAND,
    S_CONTACT_MODE,
    validate_outcome_blind_selection,
)
from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.label_spec import LabelSpec
from damage_engine.registry import PhysicsRegistry
from damage_engine.sampling_checkpoint import (
    initialize_sampling_run,
    verify_sampling_checkpoint,
)
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.utils import content_hash, load_json


ROOT = Path(__file__).resolve().parents[1]


def _engine() -> DamageEngine:
    return DamageEngine(
        load_json(ROOT / "raw_data" / "target.rebuilt.json"),
        PhysicsRegistry.from_file(
            ROOT / "examples" / "physics.loitering_impact.json"
        ),
    )


def _base_case(engine: DamageEngine) -> dict:
    case = load_json(ROOT / "raw_data" / "case.loitering_impact.v1.json")
    case["target_ref"]["content_hash"] = engine.target_hash
    return case


def _label_spec() -> LabelSpec:
    return LabelSpec.from_file(
        ROOT / "raw_data" / "label_spec.loitering_impact.v1.json"
    )


def _selection(*paths: str) -> dict:
    return {
        "schema_version": "1.0.0",
        "selection_rule_id": "s-contact-input-admission-v1",
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


def _spec_document(
    *, y_bounds: tuple[float, float] = (-3.0, 2.0), max_attempts: int = 1000
) -> dict:
    document = load_json(
        ROOT / "raw_data" / "sampling_spec.loitering_impact.smoke.v1.json"
    )
    document["schema_version"] = "1.3.0"
    document["model_input_contract"] = "loitering_impact_state_only@1.0.0"
    # Collision fields remain audit-only and are not learner features.
    document["feature_roles"] = document["feature_roles"][:4]
    document["reference_distribution"]["variables"][1].update(
        {"low": y_bounds[0], "high": y_bounds[1]}
    )
    document["legal_sampling"] = {
        "invalid_candidate_policy": "reject_and_resample",
        "max_attempts_per_allocation": max_attempts,
    }
    document["contact_conditioning"] = {
        "contract_id": S_CONTACT_CONTRACT_ID,
        "mode": S_CONTACT_MODE,
        "estimand": S_CONTACT_ESTIMAND,
        "selection_declaration": _selection(),
    }
    return document


def _factory(
    engine: DamageEngine,
    case: dict,
    spec: SamplingSpec,
    *,
    engine_code_hash: str = ENGINE_CODE_HASH,
) -> CaseFactory:
    return CaseFactory(
        case,
        spec,
        _label_spec(),
        target_hash=engine.target_hash,
        damage_tree_definition_hash=content_hash(
            engine.target_data["damage_tree"]
        ),
        physics_registry_hash=engine.registry.hash,
        engine_code_hash=engine_code_hash,
        physics_registry=engine.registry,
        compiled_target=engine.compiled_target,
    )


def test_sampling_spec_13_binds_contact_and_selection_hashes() -> None:
    baseline = SamplingSpec(_spec_document())

    conditioning = baseline.data["contact_conditioning"]
    assert baseline.is_loitering_impact_model_input
    assert baseline.has_contact_conditioning
    assert conditioning["contract_hash"] == S_CONTACT_CONTRACT_HASH
    assert conditioning["selection_declaration_hash"].startswith("sha256:")
    assert (
        baseline.training_design_document["contract_mode"]
        == "loitering_impact_contact_conditioned_1.3"
    )

    changed = _spec_document()
    changed["contact_conditioning"]["selection_declaration"][
        "dependency_paths"
    ].append("/environment/temperature_k")
    assert SamplingSpec(changed).hash != baseline.hash


@pytest.mark.parametrize(
    "outcome_path",
    (
        "/results/damage_tree/outputs/effect-k1",
        "/outcomes/final_kmfc",
        "/kmfc/q1",
        "/predictions/criterion_probability",
    ),
)
def test_sampling_spec_13_rejects_outcome_aware_or_non_loitering_declarations(
    outcome_path: str,
) -> None:
    outcome_aware = _spec_document()
    outcome_aware["contact_conditioning"]["selection_declaration"] = _selection(
        outcome_path
    )
    with pytest.raises(InputValidationError) as error:
        SamplingSpec(outcome_aware)
    assert error.value.context["reason_code"] == "selection_outcome_dependency"

    uav = _spec_document()
    uav["model_input_contract"] = "uav_state_only@1.0.0"
    with pytest.raises(InputValidationError, match="schema validation"):
        SamplingSpec(uav)


@pytest.mark.parametrize("invalid_path", ({}, "/case/~2bad", "/case/~"))
def test_selection_declaration_invalid_item_fails_closed_without_type_error(
    invalid_path: object,
) -> None:
    declaration = _selection()
    declaration["dependency_paths"] = [invalid_path]

    with pytest.raises(InputValidationError) as error:
        validate_outcome_blind_selection(declaration)

    assert error.value.context["reason_code"] == "selection_declaration_invalid"


def test_reject_and_resample_ledgers_misses_and_replays_deterministically() -> None:
    engine = _engine()
    case = _base_case(engine)
    spec = SamplingSpec(_spec_document(y_bounds=(-20.0, 2.0)))

    fixed_engine_hash = "sha256:" + "5" * 64
    first = _factory(
        engine, case, spec, engine_code_hash=fixed_engine_hash
    ).build_reference_plan()
    replay = _factory(
        engine, case, spec, engine_code_hash=fixed_engine_hash
    ).build_reference_plan()

    assert len(first.cases) == 12
    assert len(first.candidate_rows) > len(first.cases)
    assert first.latent_design.shape == (
        len(first.candidate_rows),
        spec.latent_dimension,
    )
    assert first.candidate_rows == replay.candidate_rows
    np.testing.assert_array_equal(first.latent_design, replay.latent_design)
    assert content_hash(
        {
            "cases": first.cases,
            "candidate_rows": first.candidate_rows,
            "latent_design": first.latent_design.tolist(),
            "split_registry": first.split_registry.definition_document(),
        }
    ) == "sha256:20c4db256cf63fa351cadf95deb68070806482b42db6d00978a6436106c23340"

    rejected = [row for row in first.candidate_rows if not row["selected"]]
    assert rejected
    assert {row["refusal"]["code"] for row in rejected} == {
        "no_forward_armor_intersection"
    }
    assert all(row["weight_mask"] is False for row in rejected)
    assert all(row["formal_count"] == 0 for row in rejected)
    assert all(row["executed"] is False for row in rejected)
    assert all(row["successful"] is False for row in rejected)
    assert all(row["admitted"] is False for row in rejected)
    for row in rejected:
        context = json.loads(row["refusal"]["context_json"])
        assert context["formal_teacher_executed"] is False
        assert context["contact_contract_hash"] == S_CONTACT_CONTRACT_HASH

    selected_ids = {
        row["candidate_id"] for row in first.candidate_rows if row["selected"]
    }
    selected_rows = [row for row in first.candidate_rows if row["selected"]]
    assert all(
        row.get(name) is None
        for row in selected_rows
        for name in ("executed", "successful", "admitted")
    )
    assert selected_ids == {
        case["metadata"]["sampling"]["candidate_id"]
        for case in first.cases
    }
    for formal_case in first.cases:
        binding = formal_case["metadata"]["sampling"]["contact_conditioning"]
        assert binding["contract_hash"] == S_CONTACT_CONTRACT_HASH
        assert (
            binding["selection_declaration_hash"]
            == spec.contact_selection_hash
        )
        assert binding["eligibility_record"]["status"] == (
            "eligible_contact_conditioned"
        )


@pytest.mark.parametrize(
    ("geometry", "reason_code"),
    (
        ("miss", "no_forward_armor_intersection"),
        ("reverse", "armor_intersection_only_behind_origin"),
        ("inside", "trajectory_origin_inside_armor"),
    ),
)
def test_contact_refusals_reach_attempt_limit_with_stable_codes(
    geometry: str, reason_code: str
) -> None:
    engine = _engine()
    case = _base_case(engine)
    document = _spec_document(max_attempts=6)
    if geometry == "miss":
        document["reference_distribution"]["variables"][1].update(
            {"low": 20.0, "high": 21.0}
        )
        document["reference_distribution"]["variables"][2].update(
            {"low": 20.0, "high": 21.0}
        )
    elif geometry == "reverse":
        document["reference_distribution"]["variables"][3].update(
            {"low": -160.0, "high": -80.0}
        )
        case["encounter"]["body_to_target_orientation_xyzw"] = [
            0.0,
            0.0,
            1.0,
            0.0,
        ]
    else:
        case["encounter"]["trajectory_origin_m"][0] = 0.0
        document["reference_distribution"]["variables"][1].update(
            {"low": -0.2, "high": 0.2}
        )
        document["reference_distribution"]["variables"][2].update(
            {"low": 1.1, "high": 1.3}
        )

    with pytest.raises(InputValidationError, match="attempt limit") as error:
        _factory(engine, case, SamplingSpec(document)).build_reference_plan()

    assert error.value.context["accepted"] == 0
    assert error.value.context["attempts"] == 6
    assert error.value.context["refusal_code_counts"] == {reason_code: 6}


@pytest.mark.parametrize("y_bounds", ((-3.0, 2.0), (-20.0, 2.0)))
def test_checkpoint_persists_and_reverifies_complete_contact_attempt_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    y_bounds: tuple[float, float],
) -> None:
    engine = _engine()
    case = _base_case(engine)
    spec = SamplingSpec(_spec_document(y_bounds=y_bounds))
    label_spec = _label_spec()

    def teacher_must_not_run(*_args, **_kwargs):
        raise AssertionError("checkpoint initialization executed the formal teacher")

    monkeypatch.setattr(engine, "run", teacher_must_not_run)
    root = tmp_path / "contact-checkpoint"
    checkpoint, plan = initialize_sampling_run(
        root,
        engine=engine,
        base_case=case,
        sampling_spec=spec,
        label_spec=label_spec,
    )
    assert verify_sampling_checkpoint(root) == checkpoint

    resumed_checkpoint, resumed = initialize_sampling_run(
        root,
        engine=engine,
        base_case=deepcopy(case),
        sampling_spec=SamplingSpec(spec.data),
        label_spec=LabelSpec(label_spec.data),
        resume=True,
    )
    assert resumed_checkpoint == checkpoint
    assert [row["candidate_id"] for row in resumed.candidate_rows] == [
        row["candidate_id"] for row in plan.candidate_rows
    ]
    assert [row["refusal"] for row in resumed.candidate_rows] == [
        row.get("refusal") for row in plan.candidate_rows
    ]
    np.testing.assert_array_equal(resumed.latent_design, plan.latent_design)

    rejected = [row for row in resumed.candidate_rows if not row["selected"]]
    selected = [row for row in resumed.candidate_rows if row["selected"]]
    assert all(
        row[name] is None
        for row in selected
        for name in ("executed", "successful", "admitted")
    )
    if y_bounds == (-3.0, 2.0):
        assert rejected == []
    else:
        assert rejected
    round_manifest = load_json(
        root / "rounds" / "round-000" / "round_manifest.json"
    )
    assert round_manifest["candidate_count"] == len(resumed.candidate_rows)
    contact_manifest = round_manifest["contact_conditioning"]
    assert contact_manifest["attempted_count"] == len(resumed.candidate_rows)
    assert contact_manifest["accepted_count"] == len(resumed.cases)
    assert contact_manifest["input_refused_count"] == len(rejected)
    assert contact_manifest["attempted_count"] == (
        contact_manifest["accepted_count"]
        + contact_manifest["input_refused_count"]
    )
    assert contact_manifest["contract_hash"] == S_CONTACT_CONTRACT_HASH
