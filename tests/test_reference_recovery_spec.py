from __future__ import annotations

import json
from pathlib import Path

from damage_engine.sampling_spec import SamplingSpec
from tools.derive_reference_recovery_spec import derive_reference_recovery_spec


ROOT = Path(__file__).resolve().parents[1]


def test_recovery_spec_changes_only_budget_identity_replicates_and_seeds() -> None:
    source = json.loads(
        (
            ROOT
            / "configs"
            / "sampling"
            / "p0_sphere_entry_continuous_direction_reference.v1.json"
        ).read_text(encoding="utf-8")
    )

    derived = derive_reference_recovery_spec(
        source,
        spec_id="recovery-pilot-v1",
        split_registry_id="recovery-splits-v1",
        train=30,
        validation=15,
        test=15,
        replicates=8,
        seed_base=2026112201,
        revision="2026-11-22",
    )

    SamplingSpec(derived)
    assert derived["budgets"] == {
        "rare_stress_test": 0,
        "test": 15,
        "train_pilot": 30,
        "validation": 15,
    }
    assert derived["formal_policy"]["replicates"] == 8
    assert derived["replicate_policy"]["default_replicates"] == 8
    assert sum(
        allocation["count"]
        for allocation in derived["round_budgets"][0]["allocations"]
    ) == 60
    assert derived["contact_manifold"] == source["contact_manifold"]
    assert derived["reference_distribution"] == source["reference_distribution"]
    assert derived["feature_roles"] == source["feature_roles"]


def test_recovery_spec_can_preregister_lower_train_replicates() -> None:
    source = json.loads(
        (
            ROOT
            / "configs"
            / "sampling"
            / "p0_sphere_entry_continuous_direction_reference.v1.json"
        ).read_text(encoding="utf-8")
    )
    derived = derive_reference_recovery_spec(
        source,
        spec_id="role-budget-v1",
        split_registry_id="role-budget-splits-v1",
        train=2000,
        validation=300,
        test=300,
        replicates=32,
        train_replicates=8,
        seed_base=2026112301,
        revision="2026-11-22",
    )

    spec = SamplingSpec(derived)
    assert spec.replicates_for("reference_train", "p0") == 8
    assert spec.replicates_for("reference_validation", "p0") == 32
    assert spec.replicates_for("reference_test", "p0") == 32
    assert derived["legal_sampling"]["max_attempts_per_allocation"] >= 2000
