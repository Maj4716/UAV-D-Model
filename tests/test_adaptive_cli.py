"""Artificial CLI regression tests for the multiround sampling sidecar.

The tests exercise command orchestration only.  They never instantiate or run
``DamageEngine`` and use opaque candidate identifiers and numeric score rows.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from damage_engine import cli
from damage_engine.utils import content_hash


@pytest.fixture(autouse=True)
def _forbid_domain_execution(monkeypatch):
    def refuse(*_args, **_kwargs):
        pytest.fail("adaptive CLI tests must not construct or run DamageEngine")

    monkeypatch.setattr(cli, "DamageEngine", refuse)
    monkeypatch.setattr(cli, "_engine", refuse)


def _proposal_input() -> dict:
    return {
        "schema_version": "adaptive-proposal-input@1.1.0",
        "round_lineage_id": "round-lineage-001",
        "adaptive_proposal_hash": "a" * 64,
        "acquisition_model_hash": "b" * 64,
        "candidate_ids": ["candidate-1", "candidate-2"],
        "candidate_cases": [
            {"case_id": "formal-candidate-1", "analysis": {"aleatory_seed": 1}},
            {"case_id": "formal-candidate-2", "analysis": {"aleatory_seed": 2}},
        ],
        "p0_density": [0.4, 0.6],
        "adaptive_density": [0.7, 0.3],
        "p0_hierarchy_probability": [0.5, 0.5],
        "proposal_hierarchy_probability": [0.6, 0.4],
        "score_rarity": [0.1, 0.2],
        "score_uncertainty": [0.2, 0.1],
        "score_coverage": [0.3, 0.4],
        "score_total": [0.6, 0.7],
        "strata": ["stratum-a", "stratum-b"],
        "expected_count": 1,
        "rho_min": 0.05,
        "reference_fraction": 0.2,
        "selection_seed": 17,
        "acquisition_history_candidate_ids": ["train-observed-2"],
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("argv", "handler"),
    [
        (["sampler", "adaptive-init", "--output-directory", "out"], cli.command_sampler_adaptive_init),
        (
            ["sampler", "adaptive-propose", "--output-directory", "out", "--proposal-input", "proposal.json"],
            cli.command_sampler_adaptive_propose,
        ),
        (
            ["sampler", "adaptive-run-selected", "--output-directory", "out", "--round", "1", "--target", "target.json", "--physics", "physics.json"],
            cli.command_sampler_adaptive_run_selected,
        ),
        (
            ["sampler", "adaptive-update", "--output-directory", "out", "--round", "1"],
            cli.command_sampler_adaptive_update,
        ),
        (["sampler", "adaptive-audit", "--output-directory", "out"], cli.command_sampler_adaptive_audit),
        (["sampler", "adaptive-finalize", "--output-directory", "out"], cli.command_sampler_adaptive_finalize),
    ],
)
def test_parser_exposes_all_adaptive_commands(argv, handler):
    arguments = cli.build_parser().parse_args(argv)
    assert arguments.handler is handler


def test_adaptive_init_maps_resume_and_prints_checkpoint(monkeypatch, capsys, tmp_path):
    observed = {}

    def initialize(path, *, resume):
        observed.update(path=path, resume=resume)
        return {
            "status": "active",
            "checkpoint_hash": "checkpoint-init",
            "reference_round_lineage_id": "reference-round",
            "rounds_committed": [0],
        }

    monkeypatch.setattr(cli, "initialize_multiround_sampling_state", initialize)
    assert cli.main([
        "sampler", "adaptive-init", "--output-directory", str(tmp_path), "--resume"
    ]) == 0
    assert observed == {"path": str(tmp_path), "resume": True}
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "status": "active",
        "checkpoint_hash": "checkpoint-init",
        "reference_round_lineage_id": "reference-round",
        "rounds_committed": [0],
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document.pop("score_total"),
        lambda document: document.__setitem__("unknown_field", "must-refuse"),
        lambda document: document.__setitem__("p0_id", "user-override-must-refuse"),
        lambda document: document.__setitem__("p0_hash", "c" * 64),
    ],
    ids=("missing", "unknown", "p0-id-override", "p0-hash-override"),
)
def test_proposal_input_exact_fields_fail_closed_before_checkpoint(
    mutation, monkeypatch, capsys, tmp_path
):
    proposal_path = tmp_path / "proposal.json"
    document = _proposal_input()
    mutation(document)
    _write_json(proposal_path, document)
    monkeypatch.setattr(
        cli,
        "verify_multiround_sampling_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("invalid envelope must be refused before checkpoint access"),
    )

    assert cli.main([
        "sampler", "adaptive-propose",
        "--output-directory", str(tmp_path),
        "--proposal-input", str(proposal_path),
    ]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "refused"
    assert payload["error"]["code"] == "REF_INPUT_INVALID"
    assert payload["error"]["message"] == "adaptive proposal input has missing or unknown fields"


def test_proposal_derives_p0_identity_from_frozen_sampling_spec_and_passes_verified_history(
    monkeypatch, capsys, tmp_path
):
    proposal_path = tmp_path / "proposal.json"
    document = _proposal_input()
    _write_json(proposal_path, document)
    reference_distribution = {
        "id": "reference-distribution-body",
        "opaque_parameters": {"first": 0.25, "second": 0.75},
    }
    frozen_spec = SimpleNamespace(
        distribution_id="frozen-p0-id",
        data={"reference_distribution": reference_distribution},
    )
    observed = {}

    monkeypatch.setattr(
        cli,
        "verify_multiround_sampling_checkpoint",
        lambda path: {
            "status": "active",
            "rounds_committed": [0],
            "round_chain": [{"round_lineage_id": "reference-round"}],
        },
    )

    def load_frozen_spec(path):
        observed["sampling_spec_path"] = path
        return frozen_spec

    monkeypatch.setattr(cli.SamplingSpec, "from_file", load_frozen_spec)
    verified_history = [
        {"candidate_id": "train-observed-1", "split": "Train", "status": "ok"},
        {"candidate_id": "train-observed-2", "split": "Train", "status": "ok"},
    ]
    monkeypatch.setattr(
        cli, "acquisition_history_records", lambda path: list(verified_history)
    )

    def freeze(candidate_ids, **kwargs):
        observed["candidate_ids"] = candidate_ids
        observed["freeze"] = kwargs
        return {
            "manifest": {
                "proposal_hash": "proposal-frozen",
                "candidate_count": 2,
                "realized_selected_count": 1,
            },
            "rows": [],
        }

    monkeypatch.setattr(cli, "freeze_adaptive_proposal", freeze)
    monkeypatch.setattr(
        cli,
        "commit_adaptive_sampling_round",
        lambda path, proposal: {"checkpoint_hash": "checkpoint-after-proposal"},
    )

    assert cli.main([
        "sampler", "adaptive-propose",
        "--output-directory", str(tmp_path),
        "--proposal-input", str(proposal_path),
    ]) == 0
    output = json.loads(capsys.readouterr().out)

    assert observed["sampling_spec_path"] == tmp_path / "plan" / "sampling_spec.json"
    assert observed["candidate_ids"] == document["candidate_ids"]
    assert observed["freeze"]["candidate_cases"] == document["candidate_cases"]
    assert observed["freeze"]["p0_id"] == "frozen-p0-id"
    assert observed["freeze"]["p0_hash"] == content_hash(reference_distribution)
    assert observed["freeze"]["p0_density"] == document["p0_density"]
    assert observed["freeze"]["parent_round_lineage_id"] == "reference-round"
    assert observed["freeze"]["acquisition_history"] == [verified_history[1]]
    assert output["status"] == "proposal_frozen"
    assert output["round"] == 1
    assert output["checkpoint_hash"] == "checkpoint-after-proposal"


def test_proposal_refuses_history_not_returned_by_verified_train_history_api(
    monkeypatch, capsys, tmp_path
):
    proposal_path = tmp_path / "proposal.json"
    document = _proposal_input()
    document["acquisition_history_candidate_ids"] = ["validation-or-unverified"]
    _write_json(proposal_path, document)
    monkeypatch.setattr(
        cli,
        "verify_multiround_sampling_checkpoint",
        lambda _path: {
            "status": "active",
            "rounds_committed": [0],
            "round_chain": [{"round_lineage_id": "reference-round"}],
        },
    )
    monkeypatch.setattr(
        cli.SamplingSpec,
        "from_file",
        lambda _path: SimpleNamespace(
            distribution_id="frozen-p0-id",
            data={"reference_distribution": {"id": "frozen"}},
        ),
    )
    monkeypatch.setattr(
        cli,
        "acquisition_history_records",
        lambda _path: [
            {"candidate_id": "train-observed-2", "split": "Train", "status": "ok"}
        ],
    )
    monkeypatch.setattr(
        cli,
        "freeze_adaptive_proposal",
        lambda *_args, **_kwargs: pytest.fail("unavailable history must not reach proposal freezing"),
    )

    assert cli.main([
        "sampler", "adaptive-propose",
        "--output-directory", str(tmp_path),
        "--proposal-input", str(proposal_path),
    ]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["message"] == (
        "proposal acquisition history references an unavailable Train result"
    )


def test_adaptive_update_derives_outcomes_from_bound_archive(monkeypatch, capsys, tmp_path):
    outcomes_path = tmp_path / "rounds" / "round-003" / "formal_outcomes.jsonl"
    outcomes_path.parent.mkdir(parents=True)
    outcomes = [
        {"candidate_id": "candidate-1", "status": "ok", "result_hash": "1" * 64},
        {"candidate_id": "candidate-2", "status": "refused", "refusal_hash": "2" * 64},
    ]
    outcomes_path.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in outcomes),
        encoding="utf-8",
    )
    observed = {}

    def commit(path, round_index, loaded):
        observed.update(path=path, round_index=round_index, outcomes=loaded)
        return {"checkpoint_hash": "checkpoint-after-outcomes"}

    monkeypatch.setattr(cli, "commit_adaptive_round_outcomes", commit)
    assert cli.main([
        "sampler", "adaptive-update",
        "--output-directory", str(tmp_path),
        "--round", "3",
    ]) == 0
    assert observed == {
        "path": str(tmp_path),
        "round_index": 3,
        "outcomes": None,
    }
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "status": "outcomes_committed",
        "round": 3,
        "checkpoint_hash": "checkpoint-after-outcomes",
        "outcome_count": 2,
    }


def test_adaptive_audit_declares_density_score_scope_and_projection_exclusion(
    monkeypatch, capsys, tmp_path
):
    round_directory = tmp_path / "rounds" / "round-001"
    row = {
        "candidate_id": "candidate-1",
        "p0_density": 4.0,
        "adaptive_density": 7.0,
        "proposal_density": 5.5,
        "p0_hierarchy_probability": 0.5,
        "proposal_hierarchy_probability": 0.6,
        "score_rarity": 0.1,
        "score_uncertainty": 0.2,
        "score_coverage": 0.3,
        "score_total": 6.0,
    }
    _write_json(round_directory / "adaptive_candidates.jsonl", row)
    monkeypatch.setattr(
        cli,
        "verify_multiround_sampling_checkpoint",
        lambda _path: {
            "rounds_committed": [0, 1],
            "checkpoint_hash": "verified-multiround-checkpoint",
        },
    )
    quality_gates = {"soft_ess_fraction_min": 0.2}
    monkeypatch.setattr(
        cli.SamplingSpec,
        "from_file",
        lambda _path: SimpleNamespace(data={"quality_gates": quality_gates}),
    )
    observed = {}

    def audit(rows, design, *, quality_gates):
        observed.update(rows=rows, design=design, quality_gates=quality_gates)
        return {"quality_gate_passed": True, "scope": "artificial-audit-stub"}

    monkeypatch.setattr(cli, "audit_sampling", audit)
    report_path = tmp_path / "evidence" / "adaptive-audit.json"
    assert cli.main([
        "sampler", "adaptive-audit",
        "--output-directory", str(tmp_path),
        "--output", str(report_path),
    ]) == 0
    output = json.loads(capsys.readouterr().out)

    assert observed["rows"] == [row]
    assert observed["quality_gates"] is quality_gates
    assert observed["design"] is None
    assert output["audit_feature_names"] == []
    assert output["proposal_metric_names"] == [
        "p0_density", "adaptive_density", "proposal_density",
        "p0_hierarchy_probability", "proposal_hierarchy_probability",
        "score_rarity", "score_uncertainty", "score_coverage", "score_total",
    ]
    assert "raw-input key projections" in output["audit_scope"]
    assert output["multiround_checkpoint_hash"] == "verified-multiround-checkpoint"
    assert json.loads(report_path.read_text(encoding="utf-8")) == output


def test_adaptive_finalize_maps_checkpoint_state_without_domain_execution(
    monkeypatch, capsys, tmp_path
):
    observed = {}

    def finalize(path):
        observed["path"] = path
        return {
            "status": "finalized",
            "checkpoint_hash": "checkpoint-final",
            "rounds_committed": [0, 1, 2],
            "finalized_at": "2026-10-11T12:00:00Z",
        }

    monkeypatch.setattr(cli, "finalize_multiround_sampling", finalize)
    assert cli.main([
        "sampler", "adaptive-finalize", "--output-directory", str(tmp_path)
    ]) == 0
    assert observed["path"] == str(tmp_path)
    assert json.loads(capsys.readouterr().out) == {
        "status": "finalized",
        "checkpoint_hash": "checkpoint-final",
        "rounds_committed": [0, 1, 2],
        "finalized_at": "2026-10-11T12:00:00Z",
    }
