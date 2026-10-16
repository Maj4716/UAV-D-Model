"""Artificial-distribution tests for auditable adaptive rounds; no engine or training."""
from __future__ import annotations

import json

import numpy as np
import pytest

import damage_engine.sampling_checkpoint as checkpoint_module
from damage_engine.adaptive_sampling import (
    defensive_mixture_density,
    exact_design_weights,
    freeze_adaptive_proposal,
    validate_train_only_acquisition,
)
from damage_engine.errors import InputValidationError
from damage_engine.sampling_audit import audit_sampling
from damage_engine.sampling_checkpoint import (
    acquisition_history_records,
    commit_adaptive_round_outcomes,
    commit_adaptive_sampling_round,
    finalize_multiround_sampling,
    initialize_multiround_sampling_state,
    mark_pilot_complete,
    verify_sampling_checkpoint,
    verify_multiround_sampling_checkpoint,
)
from damage_engine.utils import content_hash
from test_reaudit_sampling_fix import _initialize, _write_artificial_formal, scalar_plan
from test_writer_integrity_review import synthetic_writer


GATES = {"positivity_min": 0.05, "min_ess_fraction": 0.1, "max_weight_cv": 20.0}


def _hash(value) -> str:
    return content_hash(value)


def _history(identifier: str = "train-formal-0") -> list[dict]:
    return [
        {
            "candidate_id": identifier,
            "split": "train",
            "role": "train_pilot",
            "stream": "reference",
            "formal_status": "ok",
            "label_observed": True,
            "formal_result_hash": _hash({"result": identifier}),
        }
    ]


def _proposal(
    round_index: int,
    parent: str,
    *,
    prefix: str = "adaptive",
    history: list[dict] | None = None,
    p0_id: str = "toy-p0",
    p0_hash: str | None = None,
) -> dict:
    return freeze_adaptive_proposal(
        [f"{prefix}-{round_index}-{index}" for index in range(4)],
        round_index=round_index,
        round_lineage_id=f"round-lineage-{round_index}",
        parent_round_lineage_id=parent,
        p0_id=p0_id,
        p0_hash=_hash("toy-p0") if p0_hash is None else p0_hash,
        adaptive_proposal_hash=_hash({"proposal": round_index}),
        acquisition_model_hash=_hash({"model": round_index}),
        acquisition_history=_history(f"history-{round_index}") if history is None else history,
        p0_density=[0.25, 0.25, 0.25, 0.25],
        adaptive_density=[0.10, 0.20, 0.30, 0.40],
        p0_hierarchy_probability=[0.5, 0.5, 0.5, 0.5],
        proposal_hierarchy_probability=[0.25, 0.50, 0.75, 1.0],
        rarity_scores=[0.1, 0.2, 0.3, 0.4],
        uncertainty_scores=[0.4, 0.3, 0.2, 0.1],
        coverage_scores=[0.2, 0.2, 0.8, 0.8],
        selection_scores=[0.1, 0.5, 1.0, 2.0],
        strata=["common", "common", "rare", "rare"],
        expected_count=2.5,
        rho_min=0.1,
        reference_fraction=0.2,
        selection_seed=700 + round_index,
    )


def _outcomes(proposal: dict, *, refuse_last: bool = False) -> list[dict]:
    selected = [row for row in proposal["rows"] if row["selected"]]
    rows = []
    for index, candidate in enumerate(selected):
        refused = refuse_last and index == len(selected) - 1
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "round": candidate["round"],
            "round_lineage_id": candidate["round_lineage_id"],
            "split": "train",
            "role": "adaptive_train",
            "stream": "adaptive_train",
            "formal_status": "refused" if refused else "ok",
            "label_observed": not refused,
            "formal_result_hash": _hash({"formal": candidate["candidate_id"], "refused": refused}),
        })
    return rows


def test_defensive_mixture_and_general_design_weight_formula_are_exact() -> None:
    p0 = np.asarray([0.2, 0.8])
    adaptive = np.asarray([0.8, 0.2])
    proposal = defensive_mixture_density(p0, adaptive, reference_fraction=0.25)
    np.testing.assert_allclose(proposal, [0.65, 0.35])

    rho = np.asarray([0.5, 0.25])
    weights = exact_design_weights(
        p0,
        proposal,
        rho,
        p0_hierarchy_probability=[0.4, 0.6],
        proposal_hierarchy_probability=[0.2, 0.8],
    )
    np.testing.assert_allclose(
        weights,
        (p0 * np.asarray([0.4, 0.6]))
        / (proposal * np.asarray([0.2, 0.8]) * rho),
    )
    np.testing.assert_allclose(
        exact_design_weights(p0, p0, rho, p0_hierarchy_probability=[0.4, 0.6],
                             proposal_hierarchy_probability=[0.4, 0.6]),
        1.0 / rho,
    )


def test_general_weights_recover_known_hierarchical_p0_expectation_in_design_expectation() -> None:
    # Two hierarchy groups, each with two conditional states.  Both p0 and q
    # are normalized joint designs, and rho is a second-stage Bernoulli gate.
    p0_conditional = np.asarray([0.8, 0.2, 0.3, 0.7])
    q_conditional = np.asarray([0.3, 0.7, 0.6, 0.4])
    p0_hierarchy = np.asarray([0.6, 0.6, 0.4, 0.4])
    q_hierarchy = np.asarray([0.25, 0.25, 0.75, 0.75])
    rho = np.asarray([0.4, 0.8, 0.5, 1.0])
    response = np.asarray([0.0, 1.0, 4.0, 9.0])
    weights = exact_design_weights(
        p0_conditional,
        q_conditional,
        rho,
        p0_hierarchy_probability=p0_hierarchy,
        proposal_hierarchy_probability=q_hierarchy,
    )
    sampling_mass = q_conditional * q_hierarchy * rho
    target_mass = p0_conditional * p0_hierarchy
    assert np.sum(sampling_mass * weights) == pytest.approx(1.0)
    assert np.sum(sampling_mass * weights * response) == pytest.approx(
        np.sum(target_mass * response)
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split", "validation", "Train formal history only"),
        ("role", "rare_stress_test", "sealed from acquisition"),
        ("stream", "stress", "sealed from acquisition"),
        ("formal_status", "refused", "observed successful"),
        ("label_observed", False, "observed successful"),
    ],
)
def test_acquisition_history_gate_is_train_formal_only(field, value, message) -> None:
    history = _history()
    history[0][field] = value
    with pytest.raises(InputValidationError, match=message):
        validate_train_only_acquisition(history)


def test_frozen_proposal_preserves_scores_random_identity_and_general_weights() -> None:
    proposal = _proposal(1, "reference-lineage")
    manifest, rows = proposal["manifest"], proposal["rows"]

    assert manifest["score_policy"].startswith("externally_supplied")
    assert manifest["candidate_ledger_hash"] == content_hash(rows)
    assert manifest["proposal_hash"] == content_hash(
        {key: value for key, value in manifest.items() if key != "proposal_hash"}
    )
    assert [row["score_rarity"] for row in rows] == [0.1, 0.2, 0.3, 0.4]
    assert [row["score_uncertainty"] for row in rows] == [0.4, 0.3, 0.2, 0.1]
    assert [row["score_coverage"] for row in rows] == [0.2, 0.2, 0.8, 0.8]
    for row in rows:
        expected = (
            row["p0_density"] * row["p0_hierarchy_probability"]
            / (
                row["proposal_density"]
                * row["proposal_hierarchy_probability"]
                * row["inclusion_probability"]
            )
        )
        assert row["design_weight_raw"] == pytest.approx(expected)
        assert row["selected"] is (row["random_draw"] < row["inclusion_probability"])
        assert row["split"] == "train" and row["role"] == row["stream"] == "adaptive_train"


def _audit_rows() -> tuple[list[dict], np.ndarray]:
    probabilities = np.asarray([0.5, 0.25, 1.0, 0.5])
    p0 = np.asarray([0.2, 0.3, 0.2, 0.3])
    proposal = np.asarray([0.4, 0.2, 0.25, 0.15])
    p0_hierarchy = np.asarray([0.5, 0.5, 0.25, 0.75])
    proposal_hierarchy = np.asarray([0.25, 0.5, 0.5, 0.75])
    weights = exact_design_weights(
        p0, proposal, probabilities,
        p0_hierarchy_probability=p0_hierarchy,
        proposal_hierarchy_probability=proposal_hierarchy,
    )
    rows = []
    for index in range(4):
        rows.append(
            {
                "candidate_id": f"audit-{index}",
                "lineage_id": f"lineage-{index}",
                "epistemic_group_id": f"epistemic-{index}",
                "split": "train",
                "role": "adaptive_train",
                "stream": "adaptive_train",
                "round": 1 if index < 2 else 2,
                "stratum": "common" if index in {0, 2} else "rare",
                "selected": True,
                "weight_mask": True,
                "inclusion_probability": float(probabilities[index]),
                "p0_density": float(p0[index]),
                "proposal_density": float(proposal[index]),
                "p0_hierarchy_probability": float(p0_hierarchy[index]),
                "proposal_hierarchy_probability": float(proposal_hierarchy[index]),
                "design_weight_raw": float(weights[index]),
                "refusal": {"code": "TOY_REFUSAL"} if index == 3 else None,
            }
        )
    latent = np.asarray([[0.1, 0.2], [0.2, 0.3], [0.8, 0.7], [0.9, 0.8]])
    return rows, latent


def test_audit_reports_weight_tail_soft_ess_round_drift_and_refusal_set() -> None:
    rows, latent = _audit_rows()
    report = audit_sampling(
        rows,
        latent,
        quality_gates=GATES,
        executed_candidate_ids=[row["candidate_id"] for row in rows],
        successful_candidate_ids=["audit-0", "audit-1", "audit-2"],
        admitted_candidate_ids=["audit-0", "audit-2"],
        refused_candidate_ids=["audit-3"],
        soft_strata={
            "transition": [0.1, 0.8, 0.5, 1.0],
            "background": [0.9, 0.2, 0.5, 0.0],
        },
    )

    weights = np.asarray([row["design_weight_raw"] for row in rows])
    assert report["design_weight"]["maximum_normalized"] == pytest.approx(np.max(weights / weights.sum()))
    assert report["design_weight"]["p90"] == pytest.approx(np.quantile(weights, 0.9))
    assert report["design_weight"]["p99"] == pytest.approx(np.quantile(weights, 0.99))
    assert set(report["hard_stratum_effective_sample_size"]) == {"common", "rare"}
    assert report["soft_stratum_effective_sample_size"]["transition"]["soft_effective_sample_size"] > 0
    assert report["cross_round_drift"]["rounds"][1]["candidate_mean_l2_from_previous_round"] > 0
    assert report["sets"]["refused"]["candidate_ids"] == ["audit-3"]
    assert report["sets"]["refused"]["refusal_code_counts"] == {"TOY_REFUSAL": 1}
    assert "proposal conditional density" in report["design_weight"]["weight_meaning"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split", "validation", "isolated in Train"),
        ("stream", "rare_stress", "isolated from Train"),
    ],
)
def test_audit_rejects_adaptive_or_stress_flow_leakage(field, value, message) -> None:
    rows, latent = _audit_rows()
    rows[0][field] = value
    if field == "stream":
        rows[0]["role"] = "rare_stress_test"
    with pytest.raises(InputValidationError, match=message):
        audit_sampling(rows, latent, quality_gates=GATES)


def _fake_reference_root(tmp_path, monkeypatch):
    root = tmp_path / "sampling-run"
    round0 = root / "rounds" / "round-000"
    round0.mkdir(parents=True)
    (round0 / "candidates.jsonl").write_text(
        json.dumps({"candidate_id": "reference-0"}) + "\n", encoding="utf-8"
    )
    (round0 / "round_manifest.json").write_text(
        json.dumps({"round": 0, "phase": "reference"}) + "\n", encoding="utf-8"
    )
    identity_hash = _hash({"run": "toy"})
    base = {"status": "archive_complete", "identity": {"identity_hash": identity_hash}}
    base_path = root / "sampling_checkpoint.json"
    base_path.write_text(json.dumps(base) + "\n", encoding="utf-8")
    original = base_path.read_bytes()
    monkeypatch.setattr(checkpoint_module, "verify_sampling_checkpoint", lambda *_args, **_kwargs: base)
    state = initialize_multiround_sampling_state(root)
    assert base_path.read_bytes() == original
    monkeypatch.setattr(checkpoint_module, "_verify_reference_anchor", lambda *_args, **_kwargs: (base, None))
    monkeypatch.setattr(checkpoint_module, "acquisition_history_records", lambda *_args, **_kwargs: _history("history-1"))
    return root, state


def test_multiround_commit_resume_lineage_and_finalize_are_fail_closed(tmp_path, monkeypatch) -> None:
    root, state = _fake_reference_root(tmp_path, monkeypatch)
    proposal = _proposal(1, state["reference_round_lineage_id"])
    committed = commit_adaptive_sampling_round(root, proposal)

    assert committed["rounds_committed"] == [0, 1]
    assert committed["round_chain"][1]["parent_round_lineage_id"] == state["reference_round_lineage_id"]
    parquet_path = root / "rounds/round-001/adaptive_candidates.parquet"
    assert parquet_path.is_file()
    import pyarrow.parquet as pq

    parquet = pq.read_table(parquet_path)
    assert parquet.schema.metadata[b"table_schema_version"] == b"adaptive-sampling-candidates@1.1.0"
    assert {
        "p0_density", "adaptive_density", "proposal_density",
        "p0_hierarchy_probability", "proposal_hierarchy_probability",
        "inclusion_probability", "design_weight_raw",
    }.issubset(parquet.column_names)
    assert verify_multiround_sampling_checkpoint(root) == committed
    assert initialize_multiround_sampling_state(root, resume=True) == committed

    with pytest.raises(InputValidationError, match="preceding adaptive round"):
        commit_adaptive_sampling_round(root, _proposal(2, "round-lineage-1"))

    committed = commit_adaptive_round_outcomes(root, 1, _outcomes(proposal, refuse_last=True))
    assert committed["round_chain"][1]["phase"] == "outcomes_committed"
    monkeypatch.setattr(
        checkpoint_module, "acquisition_history_records", lambda *_args, **_kwargs: _history("history-2")
    )
    second = _proposal(2, "round-lineage-1")
    committed = commit_adaptive_sampling_round(root, second)
    assert committed["rounds_committed"] == [0, 1, 2]
    committed = commit_adaptive_round_outcomes(root, 2, _outcomes(second))
    assert committed["round_chain"][2]["phase"] == "outcomes_committed"
    with pytest.raises(InputValidationError, match="verified formal archives"):
        finalize_multiround_sampling(root)


def test_actual_artificial_reference_archive_supplies_only_verified_train_history(
    tmp_path, scalar_plan, monkeypatch
) -> None:
    root = tmp_path / "actual-reference"
    _, plan = _initialize(root, scalar_plan)
    _write_artificial_formal(root, scalar_plan, plan, monkeypatch)
    mark_pilot_complete(root, root / "formal_dataset/manifest.json")

    state = initialize_multiround_sampling_state(root)
    history = acquisition_history_records(root)
    assert len(history) == 1
    assert history[0]["split"] == "train"
    assert history[0]["role"] == "train_pilot"
    assert history[0]["formal_status"] == "ok"

    proposal = _proposal(
        1,
        state["reference_round_lineage_id"],
        history=history,
        p0_id=plan.candidate_rows[0]["p0_id"],
        p0_hash=plan.candidate_rows[0]["p0_hash"],
    )
    committed = commit_adaptive_sampling_round(root, proposal)
    assert verify_multiround_sampling_checkpoint(root) == committed
    with pytest.raises(InputValidationError, match="explicit multiround state"):
        verify_sampling_checkpoint(root)


def test_actual_reference_rejects_a_different_p0_identity(
    tmp_path, scalar_plan, monkeypatch
) -> None:
    root = tmp_path / "p0-mismatch"
    _, plan = _initialize(root, scalar_plan)
    _write_artificial_formal(root, scalar_plan, plan, monkeypatch)
    mark_pilot_complete(root, root / "formal_dataset/manifest.json")
    state = initialize_multiround_sampling_state(root)
    history = acquisition_history_records(root)
    proposal = _proposal(1, state["reference_round_lineage_id"], history=history)
    with pytest.raises(InputValidationError, match="p0 identity"):
        commit_adaptive_sampling_round(root, proposal)


def test_multiround_rejects_duplicate_candidate_tampering_and_orphan_round(tmp_path, monkeypatch) -> None:
    root, state = _fake_reference_root(tmp_path, monkeypatch)
    duplicate = _proposal(1, state["reference_round_lineage_id"])
    duplicate["rows"][0]["candidate_id"] = "reference-0"
    duplicate["manifest"]["candidate_ledger_hash"] = content_hash(duplicate["rows"])
    duplicate["manifest"]["selection_random_identity"]["candidate_order_hash"] = content_hash(
        [row["candidate_id"] for row in duplicate["rows"]]
    )
    duplicate["manifest"]["candidate_order_hash"] = content_hash(
        [row["candidate_id"] for row in duplicate["rows"]]
    )
    random_identity = duplicate["manifest"]["selection_random_identity"]
    random_identity["identity_hash"] = content_hash(
        {key: value for key, value in random_identity.items() if key != "identity_hash"}
    )
    duplicate["manifest"]["proposal_hash"] = content_hash(
        {key: value for key, value in duplicate["manifest"].items() if key != "proposal_hash"}
    )
    with pytest.raises(InputValidationError, match="globally unique"):
        commit_adaptive_sampling_round(root, duplicate)

    committed = commit_adaptive_sampling_round(root, _proposal(1, state["reference_round_lineage_id"]))
    proposal_path = root / "rounds/round-001/proposal.json"
    tampered = json.loads(proposal_path.read_text(encoding="utf-8"))
    tampered["acquisition_model_hash"] = _hash("tampered-model")
    proposal_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(InputValidationError):
        verify_multiround_sampling_checkpoint(root)
    # Restore the byte-exact proposal, then show that an uncommitted directory
    # is preserved as evidence and rejected rather than silently resumed.
    checkpoint_module._write_json_atomic(
        proposal_path, _proposal(1, state["reference_round_lineage_id"])["manifest"]
    )
    assert verify_multiround_sampling_checkpoint(root)["checkpoint_hash"] == committed["checkpoint_hash"]
    (root / "rounds/round-002").mkdir()
    with pytest.raises(InputValidationError, match="disk sampling rounds"):
        verify_multiround_sampling_checkpoint(root)
def test_multiround_rejects_cross_format_candidate_drift(tmp_path, monkeypatch) -> None:
    root, state = _fake_reference_root(tmp_path, monkeypatch)
    commit_adaptive_sampling_round(
        root, _proposal(1, state["reference_round_lineage_id"])
    )
    import pyarrow.parquet as pq
    from damage_engine.sampling_candidate_schema import adaptive_candidate_table

    path = root / "rounds/round-001/adaptive_candidates.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["score_rarity"] += 1.0
    pq.write_table(adaptive_candidate_table(rows), path)
    with pytest.raises(InputValidationError, match="JSONL and Parquet"):
        verify_multiround_sampling_checkpoint(root)


def test_failed_state_publish_leaves_detectable_orphan_round(tmp_path, monkeypatch) -> None:
    root, state = _fake_reference_root(tmp_path, monkeypatch)
    original_writer = checkpoint_module._write_multiround_checkpoint

    def fail_state_publish(*_args, **_kwargs):
        raise OSError("artificial state publication failure")

    monkeypatch.setattr(checkpoint_module, "_write_multiround_checkpoint", fail_state_publish)
    with pytest.raises(OSError, match="artificial"):
        commit_adaptive_sampling_round(root, _proposal(1, state["reference_round_lineage_id"]))
    monkeypatch.setattr(checkpoint_module, "_write_multiround_checkpoint", original_writer)
    with pytest.raises(InputValidationError, match="disk sampling rounds"):
        verify_multiround_sampling_checkpoint(root)
