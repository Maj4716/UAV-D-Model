"""Artificial end-to-end checks for the adaptive execution contract.

All result payloads are precomputed toy values.  The acceptance guard forbids
``DamageEngine.run`` and the physical fragment/blast entry points.
"""
from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from damage_engine import cli
from damage_engine.adaptive_sampling import freeze_adaptive_proposal
from damage_engine.archive_validation import load_archive_manifest, validate_tensor_arrays
from damage_engine.engine import DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.result_ledger import resolve_result_ledger
from damage_engine.input_identity import case_identity_hash
from damage_engine.learning_input_identity import learning_input_fingerprint
from damage_engine.sampling_checkpoint import (
    acquisition_history_records,
    commit_adaptive_sampling_round,
    initialize_multiround_sampling_state,
    mark_pilot_complete,
    verify_multiround_sampling_checkpoint,
)
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.torch_dataset import TorchShardedDamageDataset
from damage_engine.utils import content_hash
from test_distribution_manifest_integration import _sampling_document_11
from test_reaudit_sampling_fix import (
    _initialize,
    _write_artificial_formal,
    scalar_plan,
)
from test_writer_integrity_review import synthetic_writer  # noqa: F401


@pytest.fixture(autouse=True)
def _forbid_damage_engine_run(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("adaptive execution acceptance must not call DamageEngine.run")

    monkeypatch.setattr(DamageEngine, "run", forbidden)


def _candidate_cases(base_case: dict) -> list[dict]:
    cases = []
    for index, scalar in enumerate((0.125, 0.875), 1):
        case = deepcopy(base_case)
        case["case_id"] = f"adaptive-formal-{index}"
        case["toy_scalar"] = scalar
        case["toy_scalar_aux"] = 1.0 - scalar
        case["analysis"]["aleatory_seed"] = 9800 + index
        case["analysis"]["epistemic_sample_id"] = f"candidate-{index}"
        metadata = case.setdefault("metadata", {})
        for name in ("sampling", "frozen_split", "split_registry_hash"):
            metadata.pop(name, None)
        cases.append(case)
    return cases


def _prepare_content_bound_round(tmp_path, scalar_plan, monkeypatch):
    root = tmp_path / "adaptive-run"
    engine, pair, original_base, _, label = scalar_plan
    base_case = deepcopy(original_base)
    base_case["toy_scalar_aux"] = 0.5
    document = _sampling_document_11()
    variables = [
        {
            "name": "toy_scalar",
            "path": "/toy_scalar",
            "distribution": "uniform",
            "low": 0.0,
            "high": 1.0,
        },
        {
            "name": "toy_scalar_aux",
            "path": "/toy_scalar_aux",
            "distribution": "uniform",
            "low": 0.0,
            "high": 1.0,
        },
    ]
    document["reference_distribution"]["variables"] = variables
    document["feature_roles"] = [
        {
            "feature_id": item["name"],
            "path": item["path"],
            "tensor_feature_names": [item["name"]],
            "role": "learnable_variable",
            "availability_stage": "raw_input",
            "deployment_available": True,
        }
        for item in variables
    ]
    document["key_projections"][0]["paths"] = [
        "/toy_scalar", "/toy_scalar_aux"
    ]
    extended_fixture = (engine, pair, base_case, SamplingSpec(document), label)
    _, reference_plan = _initialize(root, extended_fixture)
    _write_artificial_formal(root, extended_fixture, reference_plan, monkeypatch)
    mark_pilot_complete(root, root / "formal_dataset" / "manifest.json")
    state = initialize_multiround_sampling_state(root)
    history = acquisition_history_records(root)
    candidates = _candidate_cases(base_case)
    candidate_ids = [f"adaptive-candidate-{index}" for index in (1, 2)]
    proposal = freeze_adaptive_proposal(
        candidate_ids,
        candidate_cases=candidates,
        round_index=1,
        round_lineage_id="adaptive-round-lineage-001",
        parent_round_lineage_id=state["reference_round_lineage_id"],
        p0_id=reference_plan.candidate_rows[0]["p0_id"],
        p0_hash=reference_plan.candidate_rows[0]["p0_hash"],
        adaptive_proposal_hash=content_hash({"proposal": "artificial-v1"}),
        acquisition_model_hash=content_hash({"model": "artificial-v1"}),
        acquisition_history=history,
        p0_density=[0.4, 0.6],
        adaptive_density=[0.8, 0.2],
        p0_hierarchy_probability=[0.25, 0.75],
        proposal_hierarchy_probability=[0.5, 0.5],
        rarity_scores=[0.2, 0.9],
        uncertainty_scores=[0.7, 0.1],
        coverage_scores=[0.3, 0.8],
        selection_scores=[0.5, 1.0],
        strata=["common", "rare"],
        expected_count=2.0,
        rho_min=0.1,
        reference_fraction=0.25,
        selection_seed=20261011,
    )
    commit_adaptive_sampling_round(root, proposal)
    return root, proposal, candidates, engine


def test_candidate_cases_are_frozen_into_selected_formal_ledgers(
    tmp_path, scalar_plan, monkeypatch
):
    root, proposal, candidates, _ = _prepare_content_bound_round(
        tmp_path, scalar_plan, monkeypatch
    )
    directory = root / "rounds" / "round-001"
    frozen_candidates = [
        json.loads(line)
        for line in (directory / "candidate_cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    selected_cases = [
        json.loads(line)
        for line in (directory / "selected_formal_cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    selected_index = [
        json.loads(line)
        for line in (directory / "selected_case_index.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert frozen_candidates == candidates
    assert len(selected_cases) == len(selected_index) == 2
    assert proposal["manifest"]["candidate_binding_status"] == "content_bound"
    assert proposal["manifest"]["selected_case_set_hash"].startswith("sha256:")
    for candidate, row, formal, index in zip(
        candidates, proposal["rows"], selected_cases, selected_index
    ):
        assert row["candidate_case_hash"] == case_identity_hash(candidate)
        assert row["learning_input_hash"] == learning_input_fingerprint(candidate)
        assert index["formal_case_hash"] == case_identity_hash(formal)
        assert formal["metadata"]["sampling"]["candidate_case_hash"] == row["candidate_case_hash"]
        assert formal["metadata"]["sampling"]["proposal_hash"] == proposal["manifest"]["proposal_hash"]

    # A byte-level case mutation is rejected before any execution can start.
    selected_cases[0]["toy_scalar"] = 99.0
    (directory / "selected_formal_cases.jsonl").write_text(
        "".join(json.dumps(case, allow_nan=False) + "\n" for case in selected_cases),
        encoding="utf-8",
    )
    with pytest.raises(InputValidationError, match="selected formal case binding"):
        verify_multiround_sampling_checkpoint(root)


def test_selected_execution_archive_update_and_tensor17_loader_are_content_bound(
    tmp_path, scalar_plan, monkeypatch
):
    root, proposal, _, artificial_engine = _prepare_content_bound_round(
        tmp_path, scalar_plan, monkeypatch
    )
    monkeypatch.setattr(cli, "_engine", lambda _arguments: artificial_engine)
    arguments = SimpleNamespace(
        output_directory=str(root),
        round=1,
        target="unused-artificial-target.json",
        physics="unused-artificial-physics.json",
        workers=1,
        chunksize=None,
        shard_size=1,
        tensor_compression="compressed",
        fail_fast=True,
        resume=False,
    )
    assert cli.command_sampler_adaptive_run_selected(arguments) == 0
    state = verify_multiround_sampling_checkpoint(root)
    assert state["round_chain"][1]["phase"] == "archive_complete"
    formal = root / "rounds" / "round-001" / "formal_dataset"
    manifest = json.loads((formal / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.6.0"
    assert manifest["tensor_contract"]["version"] == "1.7.0"
    sampling = manifest["generation"]["sampling"]
    assert sampling["sampling_envelope_version"] == "1.2.0"
    assert sampling["proposal_hash"] == proposal["manifest"]["proposal_hash"]
    assert sampling["acquisition_model_hash"] == proposal["manifest"]["acquisition_model_hash"]
    assert sampling["selected_case_set_hash"] == proposal["manifest"]["selected_case_set_hash"]

    for shard in sorted((formal / "shards").iterdir()):
        child, _ = load_archive_manifest(shard)
        with np.load(shard / "tensors.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        validate_tensor_arrays(arrays, child)
        expected_weight = (
            arrays["sampling_p0_density"]
            * arrays["sampling_p0_hierarchy_probability"]
            / (
                arrays["sampling_proposal_density"]
                * arrays["sampling_proposal_hierarchy_probability"]
                * arrays["sampling_selection_probability"]
            )
        )
        np.testing.assert_allclose(arrays["sampling_design_weight_raw"], expected_weight)
        assert arrays["sampling_proposal_hashes"].astype(str).tolist() == [
            proposal["manifest"]["proposal_hash"]
        ]

    dataset = TorchShardedDamageDataset(
        formal,
        split="train",
        sampling_role="adaptive_train",
        include_static_graph=False,
    )
    assert len(dataset) == 2
    sample = dataset[0]
    assert "sampling_p0_density" in sample
    assert "sampling_proposal_hierarchy_probability" in sample
    assert sample["sampling_proposal_hash"] == proposal["manifest"]["proposal_hash"]
    assert sample["sampling_round_lineage_id"] == proposal["manifest"]["round_lineage_id"]

    assert cli.command_sampler_adaptive_update(
        SimpleNamespace(output_directory=str(root), round=1)
    ) == 0
    committed = verify_multiround_sampling_checkpoint(root)
    assert committed["round_chain"][1]["phase"] == "outcomes_committed"
    assert committed["round_chain"][1]["outcome_source"] == "verified_formal_archive"
    outcomes = [
        json.loads(line)
        for line in (root / "rounds" / "round-001" / "formal_outcomes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["candidate_id"] for row in outcomes} == {
        row["candidate_id"] for row in proposal["rows"]
    }
    assert all(row["formal_case_hash"].startswith("sha256:") for row in outcomes)
    assert all(row["archive_manifest_hash"] == state["round_chain"][1]["formal_archive"]["summary"]["manifest_hash"] for row in outcomes)

    # The update is archive-derived; changing one result byte is detected.
    result_shard = formal / "shards" / "part-000000"
    result_path = resolve_result_ledger(
        result_shard,
        json.loads((result_shard / "manifest.json").read_text(encoding="utf-8")),
    )
    with result_path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(InputValidationError):
        verify_multiround_sampling_checkpoint(root)
