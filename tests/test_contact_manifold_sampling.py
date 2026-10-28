from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from damage_engine.case_factory import CaseFactory
from damage_engine.contact_manifold_sampling import (
    CONTACT_MANIFOLD_IMPLEMENTATION_HASH,
    apply_contact_manifold,
)
from damage_engine.dataset import DatasetWriter
from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.label_spec import LabelSpec
from damage_engine.learning_input_identity import (
    LOITERING_IMPACT_CONTACT_POSE_MODEL_INPUT_DEFINITION,
    LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_NAMES,
    loitering_impact_contact_pose_fixed_scenario_fingerprint,
    loitering_impact_contact_pose_learning_input_fingerprint,
    loitering_impact_contact_pose_training_vector,
)
from damage_engine.registry import PhysicsRegistry
from damage_engine.sampling_checkpoint import (
    initialize_sampling_run,
    verify_sampling_checkpoint,
)
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.torch_dataset import TorchDamageDataset
from damage_engine.training_view import TrainingViewDataset, fit_training_view_artifact
from damage_engine.utils import content_hash, load_json


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = (
    ROOT
    / "raw_data"
    / "sampling_spec.loitering_impact.contact_pose.smoke.v1.json"
)


def _engine() -> DamageEngine:
    return DamageEngine(
        load_json(ROOT / "raw_data" / "target.rebuilt.json"),
        PhysicsRegistry.from_file(
            ROOT / "examples" / "physics.loitering_impact.json"
        ),
    )


def _base_case(engine: DamageEngine | None = None) -> dict:
    case = load_json(ROOT / "raw_data" / "case.loitering_impact.v1.json")
    if engine is not None:
        case["target_ref"]["content_hash"] = engine.target_hash
    return case


def _label_spec() -> LabelSpec:
    return LabelSpec.from_file(
        ROOT / "raw_data" / "label_spec.loitering_impact.v1.json"
    )


def _factory(
    engine: DamageEngine,
    case: dict,
    spec: SamplingSpec,
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
        engine_code_hash=ENGINE_CODE_HASH,
        physics_registry=engine.registry,
        compiled_target=engine.compiled_target,
    )


def _rotation_from_xyzw(quaternion: list[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    )


def test_sampling_spec_14_is_closed_hashed_and_six_dimensional() -> None:
    spec = SamplingSpec.from_file(SPEC_PATH)

    assert spec.schema_version == "1.4.0"
    assert spec.independent_latent_dimension == 1
    assert spec.contact_manifold_latent_dimension == 6
    assert spec.latent_dimension == 7
    assert spec.data["reference_distribution"]["interpretation"] == (
        "design_distribution"
    )
    assert (
        spec.data["contact_manifold"]["implementation_hash"]
        == CONTACT_MANIFOLD_IMPLEMENTATION_HASH
    )
    assert (
        spec.training_design_document["contract_mode"]
        == "loitering_impact_contact_pose_1.4"
    )

    changed = load_json(SPEC_PATH)
    changed["contact_manifold"]["approaches"][0]["selection_weight"] = 2.0
    assert SamplingSpec(changed).hash != spec.hash

    for location in ("manifold", "approach"):
        invalid = load_json(SPEC_PATH)
        target = invalid["contact_manifold"]
        if location == "approach":
            target = target["approaches"][0]
        target["undeclared_extension"] = True
        with pytest.raises(InputValidationError, match="schema validation"):
            SamplingSpec(invalid)

    nonorthogonal = load_json(SPEC_PATH)
    nonorthogonal["contact_manifold"]["approaches"][0][
        "tangent_v_target_xyz"
    ] = [0.0, 1.0, 0.0]
    with pytest.raises(InputValidationError, match="mutually orthogonal"):
        SamplingSpec(nonorthogonal)

    downgraded = load_json(SPEC_PATH)
    downgraded["schema_version"] = "1.2.0"
    downgraded.pop("contact_conditioning")
    downgraded.pop("contact_manifold")
    with pytest.raises(InputValidationError, match="schema validation"):
        SamplingSpec(downgraded)


def test_joint_target_facing_pose_construction_and_rotation6d_identity() -> None:
    spec = SamplingSpec.from_file(SPEC_PATH)
    manifold = spec.contact_manifold
    assert manifold is not None
    observed: set[str] = set()

    for index in range(6):
        case = _base_case()
        selector = (index + 0.5) / 6.0
        record = apply_contact_manifold(
            case,
            manifold,
            [selector, 0.5, 0.5, 0.5, 0.5, 0.25],
        )
        observed.add(record["approach_id"])
        velocity = np.asarray(case["encounter"]["velocity_target_mps"])
        direction = velocity / np.linalg.norm(velocity)
        rotation = _rotation_from_xyzw(
            case["encounter"]["body_to_target_orientation_xyzw"]
        )
        origin = np.asarray(case["encounter"]["trajectory_origin_m"])
        aim = np.asarray(record["aim_point_target_m"])

        np.testing.assert_allclose(rotation[:, 0], direction, atol=1.0e-12)
        np.testing.assert_allclose(
            (aim - origin) / np.linalg.norm(aim - origin),
            direction,
            atol=1.0e-12,
        )
        assert record["approach_weight_semantics"] == (
            "declared_design_mass_not_hit_probability_or_operational_prevalence"
        )
        assert "approach_selection_probability" not in record
    assert len(observed) == 6

    case = _base_case()
    apply_contact_manifold(case, manifold, [0.1, 0.5, 0.5, 0.5, 0.5, 0.2])
    other_roll = _base_case()
    apply_contact_manifold(
        other_roll, manifold, [0.1, 0.5, 0.5, 0.5, 0.5, 0.8]
    )
    np.testing.assert_array_equal(
        case["encounter"]["velocity_target_mps"],
        other_roll["encounter"]["velocity_target_mps"],
    )
    assert loitering_impact_contact_pose_learning_input_fingerprint(
        case
    ) != loitering_impact_contact_pose_learning_input_fingerprint(other_roll)

    opposite_quaternion = deepcopy(case)
    opposite_quaternion["encounter"]["body_to_target_orientation_xyzw"] = [
        -value
        for value in case["encounter"]["body_to_target_orientation_xyzw"]
    ]
    assert loitering_impact_contact_pose_training_vector(
        opposite_quaternion
    ) == loitering_impact_contact_pose_training_vector(case)
    assert loitering_impact_contact_pose_learning_input_fingerprint(
        opposite_quaternion
    ) == loitering_impact_contact_pose_learning_input_fingerprint(case)
    assert loitering_impact_contact_pose_fixed_scenario_fingerprint(
        other_roll
    ) == loitering_impact_contact_pose_fixed_scenario_fingerprint(case)


def test_reference_plan_admits_only_s_contact_and_records_all_pose_latents() -> None:
    engine = _engine()
    spec = SamplingSpec.from_file(SPEC_PATH)
    plan = _factory(engine, _base_case(engine), spec).build_reference_plan()

    assert len(plan.cases) == 18
    assert plan.latent_design.shape == (len(plan.candidate_rows), 7)
    assert all(row["selected"] for row in plan.candidate_rows)
    assert all(len(row["latent"]) == 7 for row in plan.candidate_rows)
    approaches = Counter(
        case["metadata"]["contact_manifold_sampling"]["approach_id"]
        for case in plan.cases
    )
    assert len(approaches) == 6

    attitudes: set[tuple[float, ...]] = set()
    for case in plan.cases:
        record = case["metadata"]["contact_manifold_sampling"]
        binding = case["metadata"]["sampling"]["contact_conditioning"]
        assert record["status"] == "admitted_s_contact"
        assert record["s_contact_contract_hash"] == binding["contract_hash"]
        assert binding["eligibility_record"]["status"] == (
            "eligible_contact_conditioned"
        )
        velocity = np.asarray(case["encounter"]["velocity_target_mps"])
        rotation = _rotation_from_xyzw(
            case["encounter"]["body_to_target_orientation_xyzw"]
        )
        np.testing.assert_allclose(
            rotation[:, 0], velocity / np.linalg.norm(velocity), atol=1.0e-12
        )
        attitudes.add(tuple(rotation[:, 1].round(12)))
    assert len(attitudes) > 6


def test_noncontact_candidates_are_refusal_ledger_only_and_teacher_never_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine()
    case = _base_case(engine)
    document = load_json(SPEC_PATH)
    bad = document["contact_manifold"]["approaches"][0]
    bad["aim_u_range_m"] = [20.0, 21.0]
    bad["aim_v_range_m"] = [20.0, 21.0]
    spec = SamplingSpec(document)

    def teacher_must_not_run(*_args, **_kwargs):
        raise AssertionError("checkpoint initialization executed the teacher")

    monkeypatch.setattr(engine, "run", teacher_must_not_run)
    root = tmp_path / "contact-pose-checkpoint"
    checkpoint, plan = initialize_sampling_run(
        root,
        engine=engine,
        base_case=case,
        sampling_spec=spec,
        label_spec=_label_spec(),
    )
    assert verify_sampling_checkpoint(root) == checkpoint

    rejected = [row for row in plan.candidate_rows if not row["selected"]]
    assert rejected
    assert {row["refusal"]["code"] for row in rejected} == {
        "no_forward_armor_intersection"
    }
    assert all(
        row["executed"] is False
        and row["successful"] is False
        and row["admitted"] is False
        and row["formal_count"] == 0
        and len(row["latent"]) == 7
        for row in rejected
    )
    rejected_ids = {row["candidate_id"] for row in rejected}
    formal_ids = {
        item["metadata"]["sampling"]["candidate_id"] for item in plan.cases
    }
    assert rejected_ids.isdisjoint(formal_ids)
    assert (root / "rounds" / "round-000" / "candidates.parquet").is_file()


def test_contact_pose_archive_exposes_exact_15d_learner_input(
    tmp_path: Path,
) -> None:
    engine = _engine()
    spec = SamplingSpec.from_file(SPEC_PATH)
    plan = _factory(engine, _base_case(engine), spec).build_reference_plan()
    output = tmp_path / "contact-pose-archive"
    manifest = DatasetWriter(output).run_cases(engine, plan.cases)

    definition = LOITERING_IMPACT_CONTACT_POSE_MODEL_INPUT_DEFINITION
    assert manifest["model_input_contract"]["tensor_field"] == definition.tensor_field
    assert manifest["model_input_contract"]["features"][-1]["name"] == (
        LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_NAMES[-1]
    )
    with np.load(output / "tensors.npz", allow_pickle=False) as tensors:
        assert tensors[definition.tensor_field].shape == (18, 15)
        assert tensors[f"{definition.metadata_prefix}_feature_names"].tolist() == list(
            LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_NAMES
        )

    dataset = TorchDamageDataset(output)
    sample = dataset[0]
    assert tuple(sample[definition.tensor_field].shape) == (15,)
    assert not any(
        name in sample
        for name in ("case_x", "case_physics_x", "solver_context_x")
    )
    train = TorchDamageDataset(output, split="train")
    artifact = fit_training_view_artifact(train, proxy_boundary="end_to_end")
    view = TrainingViewDataset(train, artifact, proxy_boundary="end_to_end")
    assert view.feature_names == LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_NAMES
    assert tuple(view[0]["training_x"].shape) == (15,)

