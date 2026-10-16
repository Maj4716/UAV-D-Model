from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import warnings

from .dataset import DatasetWriter
from .engine import ENGINE_CODE_HASH, DamageEngine
from .errors import DamageEngineError, InputValidationError
from .input_identity import freeze_case_inputs
from .normalization import normalize_target
from .raw_import import import_raw_target
from .registry import PhysicsRegistry
from .sharded_dataset import ShardedDatasetWriter
from .utils import content_hash, load_json
from .validation import validate_inputs
from .case_factory import CaseFactory, load_case_template
from .dataset_policy import DatasetUsePolicy, write_dataset_policy
from .label_spec import LabelSpec
from .sampling_checkpoint import (
    acquisition_history_records,
    bind_adaptive_round_archive,
    commit_adaptive_round_outcomes,
    commit_adaptive_sampling_round,
    finalize_multiround_sampling,
    initialize_multiround_sampling_state,
    initialize_sampling_run,
    iter_adaptive_formal_cases,
    iter_reference_formal_cases,
    mark_pilot_complete,
    reference_formal_cases_path,
    verify_multiround_sampling_checkpoint,
    verify_sampling_checkpoint,
    sampling_run_transaction,
)
from .sampling_spec import SamplingSpec
from .adaptive_sampling import freeze_adaptive_proposal
from .sampling_audit import audit_sampling


def _iter_cases(path: str | Path):
    value = Path(path).resolve()
    count = 0
    if value.suffix.lower() == ".jsonl":
        with value.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                try:
                    line = raw_line.decode("utf-8-sig" if line_number == 1 else "utf-8")
                except UnicodeDecodeError as exc:
                    raise InputValidationError("case input must use UTF-8", context={"path": str(value), "line": line_number}) from exc
                if line.strip():
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise InputValidationError(
                            "invalid case JSONL",
                            context={"path": str(value), "line": line_number, "column": exc.colno},
                        ) from exc
                    if not isinstance(item, dict):
                        raise InputValidationError(
                            "case record must be an object",
                            context={"path": str(value), "line": line_number},
                        )
                    count += 1
                    yield item
    else:
        try:
            data = json.loads(value.read_text(encoding="utf-8-sig"))
        except UnicodeDecodeError as exc:
            raise InputValidationError("case input must use UTF-8", context={"path": str(value)}) from exc
        except json.JSONDecodeError as exc:
            raise InputValidationError(
                "invalid case JSON",
                context={"path": str(value), "line": exc.lineno, "column": exc.colno},
            ) from exc
        for index, item in enumerate(data if isinstance(data, list) else [data]):
            if not isinstance(item, dict):
                raise InputValidationError(
                    "case record must be an object",
                    context={"path": str(value), "record_index": index},
                )
            count += 1
            yield item
    if count == 0:
        raise InputValidationError("case input must be non-empty", context={"path": str(value)})


def _load_cases(path: str | Path) -> list[dict]:
    return list(_iter_cases(path))


def _load_single_case(path: str | Path) -> dict:
    cases = _load_cases(path)
    if len(cases) != 1:
        raise InputValidationError(
            "single-case command requires exactly one case",
            context={"path": str(Path(path).resolve()), "case_count": len(cases)},
        )
    return cases[0]


def _engine(arguments) -> DamageEngine:
    target = load_json(arguments.target)
    registry = PhysicsRegistry.from_file(arguments.physics)
    return DamageEngine(target, registry)


def command_hash(arguments) -> int:
    print(content_hash(normalize_target(load_json(arguments.target))))
    return 0


def command_validate(arguments) -> int:
    cases = _load_cases(arguments.case)
    engine = _engine(arguments)
    with freeze_case_inputs(cases, cases_base_directory=Path(arguments.case).resolve().parent) as frozen:
        for case in frozen:
            validate_inputs(engine.target_data, case, engine.registry)
    print(json.dumps({"status": "ok", "validated_cases": len(cases)}, ensure_ascii=False))
    return 0


def command_simulate(arguments) -> int:
    case = _load_single_case(arguments.case)
    engine = _engine(arguments)
    with freeze_case_inputs([case], cases_base_directory=Path(arguments.case).resolve().parent) as frozen:
        result = engine.run(next(iter(frozen)), fail_closed=not arguments.record_refusal)
    if arguments.output:
        result.write_json(arguments.output)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.status == "ok" else 2


def command_generate(arguments) -> int:
    for name in ("workers", "chunksize", "shard_size"):
        value = getattr(arguments, name)
        if value is not None and value < 1:
            raise InputValidationError(f"{name} must be at least one")
    if arguments.split_seed < 0:
        raise InputValidationError("split_seed must be non-negative")
    ratios = arguments.split_ratios
    if (len(ratios) != 3 or any(not math.isfinite(value) or value < 0 for value in ratios)
            or not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1.0e-12)):
        raise InputValidationError("split_ratios must be three finite non-negative values summing to one")
    if arguments.resume and arguments.shard_size is None:
        raise InputValidationError("--resume requires --shard-size")
    if getattr(arguments, "record_refusal", False):
        warnings.warn(
            "generate --record-refusal is a deprecated compatibility no-op; "
            "refusals are recorded by default and the flag is scheduled for "
            "removal in the next breaking release",
            FutureWarning,
            stacklevel=2,
        )
    engine = _engine(arguments)
    if arguments.shard_size is not None:
        manifest = ShardedDatasetWriter(
            arguments.output_directory,
            shard_size=arguments.shard_size,
            split_seed=arguments.split_seed,
            split_ratios=tuple(arguments.split_ratios),
            tensor_compression=arguments.tensor_compression,
        ).run_cases(
            engine,
            _iter_cases(arguments.cases),
            fail_closed=arguments.fail_fast,
            workers=arguments.workers,
            chunksize=arguments.chunksize,
            resume=arguments.resume,
            cases_base_directory=Path(arguments.cases).resolve().parent,
        )
    else:
        manifest = DatasetWriter(
            arguments.output_directory,
            split_seed=arguments.split_seed,
            split_ratios=tuple(arguments.split_ratios),
            tensor_compression=arguments.tensor_compression,
        ).run_cases(
            engine,
            _load_cases(arguments.cases),
            fail_closed=arguments.fail_fast,
            workers=arguments.workers,
            chunksize=arguments.chunksize,
            cases_base_directory=Path(arguments.cases).resolve().parent,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def command_import_raw(arguments) -> int:
    config = load_json(arguments.config)
    target, diagnostics = import_raw_target(
        arguments.components,
        arguments.armor,
        config,
        skip_incomplete_components=arguments.skip_incomplete_components,
    )
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(target, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if arguments.diagnostics:
        diagnostics_path = Path(arguments.diagnostics)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def _sampling_inputs(arguments):
    engine = _engine(arguments)
    base_case = load_case_template(arguments.base_case)
    sampling_spec = SamplingSpec.from_file(arguments.sampling_spec)
    label_spec = LabelSpec.from_file(arguments.label_spec)
    return engine, base_case, sampling_spec, label_spec


def command_sampler_validate_plan(arguments) -> int:
    engine, base_case, sampling_spec, label_spec = _sampling_inputs(arguments)
    tree_hash = content_hash(engine.target_data["damage_tree"])
    if label_spec.data["source"]["damage_tree_definition_hash"] != tree_hash:
        raise InputValidationError("label specification is bound to a different damage-tree definition")
    label_spec.validate_output_vocabulary(engine.damage_tree.outputs)
    plan = CaseFactory(
        base_case,
        sampling_spec,
        label_spec,
        target_hash=engine.target_hash,
        damage_tree_definition_hash=tree_hash,
        physics_registry_hash=engine.registry.hash,
        engine_code_hash=ENGINE_CODE_HASH,
        physics_registry=engine.registry,
        compiled_target=engine.compiled_target,
    ).build_reference_plan()
    for case in plan.cases:
        if (case.get("encounter") or {}).get("detonation_trigger") == "first_armor_contact":
            validate_inputs(
                engine.target_data,
                case,
                engine.registry,
                compiled_target=engine.compiled_target,
            )
        else:
            validate_inputs(engine.target_data, case, engine.registry)
    print(json.dumps({
        "status": "ok",
        "sampling_spec_hash": sampling_spec.hash,
        "label_spec_hash": label_spec.hash,
        "split_registry_hash": plan.split_registry.definition_hash,
        "candidate_count": len(plan.candidate_rows),
        "labels_executed": False,
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def command_sampler_init(arguments) -> int:
    engine, base_case, sampling_spec, label_spec = _sampling_inputs(arguments)
    checkpoint, plan = initialize_sampling_run(
        arguments.output_directory,
        engine=engine,
        base_case=base_case,
        sampling_spec=sampling_spec,
        label_spec=label_spec,
        resume=arguments.resume,
        cases_base_directory=Path(arguments.base_case).resolve().parent,
    )
    print(json.dumps({
        "status": checkpoint["status"],
        "sampling_checkpoint": str(Path(arguments.output_directory) / "sampling_checkpoint.json"),
        "formal_cases": str(reference_formal_cases_path(arguments.output_directory)),
        "candidate_count": len(plan.candidate_rows),
        "split_registry_hash": plan.split_registry.definition_hash,
        "labels_executed": False,
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def command_sampler_pilot(arguments) -> int:
    with sampling_run_transaction(arguments.output_directory):
        return _command_sampler_pilot_locked(arguments)


def _command_sampler_pilot_locked(arguments) -> int:
    if arguments.workers < 1 or arguments.shard_size < 1:
        raise InputValidationError("workers and shard_size must be at least one")
    if arguments.chunksize is not None and arguments.chunksize < 1:
        raise InputValidationError("chunksize must be at least one")
    engine, base_case, sampling_spec, label_spec = _sampling_inputs(arguments)
    checkpoint, plan = initialize_sampling_run(
        arguments.output_directory,
        engine=engine,
        base_case=base_case,
        sampling_spec=sampling_spec,
        label_spec=label_spec,
        resume=arguments.resume,
        cases_base_directory=Path(arguments.base_case).resolve().parent,
    )
    root = Path(arguments.output_directory)
    formal_cases_path = reference_formal_cases_path(root, require_exists=False)
    formal_dataset = root / "formal_dataset"
    dataset_checkpoint = formal_dataset / "checkpoint.json"
    dataset_resume = arguments.resume and dataset_checkpoint.is_file()
    if arguments.resume and formal_dataset.exists() and any(formal_dataset.iterdir()) and not dataset_resume:
        raise InputValidationError(
            "formal dataset directory is nonempty but has no resumable checkpoint",
            context={"path": str(formal_dataset)},
        )
    budgets = sampling_spec.budgets
    total = budgets["train_pilot"] + budgets["validation"] + budgets["test"]
    ratios = (
        budgets["train_pilot"] / total,
        budgets["validation"] / total,
        budgets["test"] / total,
    )
    manifest = ShardedDatasetWriter(
        formal_dataset,
        shard_size=arguments.shard_size,
        split_seed=sampling_spec.seeds["split"],
        split_ratios=ratios,
        tensor_compression=arguments.tensor_compression,
    ).run_cases(
        engine,
        iter_reference_formal_cases(root),
        fail_closed=arguments.fail_fast,
        workers=arguments.workers,
        chunksize=arguments.chunksize,
        resume=dataset_resume,
        cases_base_directory=formal_cases_path.parent,
        expected_case_count=total,
    )
    checkpoint = mark_pilot_complete(root, formal_dataset / "manifest.json")
    print(json.dumps({
        "status": checkpoint["status"],
        "dataset_id": manifest["dataset_id"],
        "tensor_contract_version": manifest["tensor_contract"]["version"],
        "n_total": manifest["quality_summary"]["n_total"],
        "n_valid": manifest["quality_summary"]["n_valid"],
        "n_refused": manifest["quality_summary"]["n_refused"],
        "split_counts": manifest["split_policy"]["split_counts"],
        "formal_manifest": str(formal_dataset / "manifest.json"),
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def command_sampler_audit(arguments) -> int:
    checkpoint = verify_sampling_checkpoint(arguments.output_directory)
    root = Path(arguments.output_directory)
    audit_path = root / "audits" / "sampling_quality.json"
    audit = load_json(audit_path)

    def aggregate_only(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key == "candidate_ids":
                    result["candidate_ids_omitted_count"] = (
                        len(item) if isinstance(item, list) else None
                    )
                else:
                    result[key] = aggregate_only(item)
            return result
        if isinstance(value, list):
            return [aggregate_only(item) for item in value]
        return value

    print(json.dumps({
        "status": checkpoint["status"],
        "identity_hash": checkpoint["identity"]["identity_hash"],
        "sampling_quality": aggregate_only(audit),
        "ledgers": {
            "full_sampling_audit": str(audit_path),
            "candidate_parquet": str(root / "rounds/round-000/candidates.parquet"),
            "formal_cases": str(reference_formal_cases_path(root)),
            "split_registry": str(root / "plan/split_registry.json"),
        },
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


_ADAPTIVE_PROPOSAL_INPUT_FIELDS = {
    "schema_version", "round_lineage_id", "adaptive_proposal_hash",
    "acquisition_model_hash", "candidate_ids", "p0_density",
    "adaptive_density", "p0_hierarchy_probability",
    "proposal_hierarchy_probability", "score_rarity", "score_uncertainty",
    "score_coverage", "score_total", "strata", "expected_count", "rho_min",
    "reference_fraction", "selection_seed", "acquisition_history_candidate_ids",
    "candidate_cases",
}


def command_sampler_adaptive_init(arguments) -> int:
    state = initialize_multiround_sampling_state(
        arguments.output_directory, resume=arguments.resume
    )
    print(json.dumps({
        "status": state["status"],
        "checkpoint_hash": state["checkpoint_hash"],
        "reference_round_lineage_id": state["reference_round_lineage_id"],
        "rounds_committed": state["rounds_committed"],
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def command_sampler_adaptive_propose(arguments) -> int:
    document = load_json(arguments.proposal_input)
    if set(document) != _ADAPTIVE_PROPOSAL_INPUT_FIELDS:
        raise InputValidationError(
            "adaptive proposal input has missing or unknown fields",
            context={
                "missing": sorted(_ADAPTIVE_PROPOSAL_INPUT_FIELDS - set(document)),
                "unknown": sorted(set(document) - _ADAPTIVE_PROPOSAL_INPUT_FIELDS),
            },
        )
    if document["schema_version"] != "adaptive-proposal-input@1.1.0":
        raise InputValidationError("unsupported adaptive proposal input version")
    state = verify_multiround_sampling_checkpoint(arguments.output_directory)
    if state["status"] != "active":
        raise InputValidationError("finalized multiround sampling state cannot accept a proposal")
    round_index = len(state["rounds_committed"])
    parent = state["round_chain"][-1]["round_lineage_id"]
    sampling_spec = SamplingSpec.from_file(
        Path(arguments.output_directory) / "plan" / "sampling_spec.json"
    )
    history = acquisition_history_records(arguments.output_directory)
    declared_ids = document["acquisition_history_candidate_ids"]
    if not isinstance(declared_ids, list) or len(declared_ids) != len(set(declared_ids)):
        raise InputValidationError(
            "acquisition_history_candidate_ids must be a unique list"
        )
    available = {row["candidate_id"]: row for row in history}
    if any(identifier not in available for identifier in declared_ids):
        raise InputValidationError(
            "proposal acquisition history references an unavailable Train result"
        )
    proposal = freeze_adaptive_proposal(
        document["candidate_ids"],
        candidate_cases=document["candidate_cases"],
        round_index=round_index,
        round_lineage_id=document["round_lineage_id"],
        parent_round_lineage_id=parent,
        p0_id=sampling_spec.distribution_id,
        p0_hash=content_hash(sampling_spec.data["reference_distribution"]),
        adaptive_proposal_hash=document["adaptive_proposal_hash"],
        acquisition_model_hash=document["acquisition_model_hash"],
        acquisition_history=[available[identifier] for identifier in declared_ids],
        p0_density=document["p0_density"],
        adaptive_density=document["adaptive_density"],
        p0_hierarchy_probability=document["p0_hierarchy_probability"],
        proposal_hierarchy_probability=document[
            "proposal_hierarchy_probability"
        ],
        rarity_scores=document["score_rarity"],
        uncertainty_scores=document["score_uncertainty"],
        coverage_scores=document["score_coverage"],
        selection_scores=document["score_total"],
        strata=document["strata"],
        expected_count=document["expected_count"],
        rho_min=document["rho_min"],
        reference_fraction=document["reference_fraction"],
        selection_seed=document["selection_seed"],
    )
    committed = commit_adaptive_sampling_round(arguments.output_directory, proposal)
    directory = Path(arguments.output_directory) / "rounds" / f"round-{round_index:03d}"
    print(json.dumps({
        "status": "proposal_frozen",
        "round": round_index,
        "proposal_hash": proposal["manifest"]["proposal_hash"],
        "candidate_count": proposal["manifest"]["candidate_count"],
        "selected_count": proposal["manifest"]["realized_selected_count"],
        "checkpoint_hash": committed["checkpoint_hash"],
        "candidate_jsonl": str(directory / "adaptive_candidates.jsonl"),
        "candidate_parquet": str(directory / "adaptive_candidates.parquet"),
        "candidate_cases": str(directory / "candidate_cases.jsonl"),
        "selected_formal_cases": str(directory / "selected_formal_cases.jsonl"),
        "selected_case_index": str(directory / "selected_case_index.jsonl"),
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def command_sampler_adaptive_run_selected(arguments) -> int:
    if arguments.workers < 1 or arguments.shard_size < 1:
        raise InputValidationError("workers and shard_size must be at least one")
    if arguments.chunksize is not None and arguments.chunksize < 1:
        raise InputValidationError("chunksize must be at least one")
    root = Path(arguments.output_directory)
    with sampling_run_transaction(root):
        state = verify_multiround_sampling_checkpoint(root)
        if state["status"] != "active":
            raise InputValidationError("finalized multiround state cannot execute a round")
        if (
            arguments.round < 1
            or arguments.round != state["rounds_committed"][-1]
        ):
            raise InputValidationError("adaptive execution must target the latest committed round")
        entry = state["round_chain"][arguments.round]
        if entry.get("phase") != "proposal_frozen":
            raise InputValidationError("adaptive round is not awaiting formal execution")
        if entry.get("candidate_binding_status") != "content_bound":
            raise InputValidationError("unbound abstract proposal cannot be executed")
        base = load_json(root / "sampling_checkpoint.json")
        engine = _engine(arguments)
        identity = base["identity"]
        mismatches = []
        if engine.target_hash != identity["target_hash"]:
            mismatches.append("target_hash")
        if engine.registry.hash != identity["physics_registry_hash"]:
            mismatches.append("physics_registry_hash")
        if ENGINE_CODE_HASH != identity["engine_code_hash"]:
            mismatches.append("engine_code_hash")
        if mismatches:
            raise InputValidationError(
                "adaptive execution engine differs from the frozen reference run",
                context={"mismatched_fields": mismatches},
            )
        directory = root / "rounds" / f"round-{arguments.round:03d}"
        selected_path = directory / "selected_formal_cases.jsonl"
        proposal_manifest = load_json(directory / "proposal.json")
        selected_count = int(proposal_manifest["realized_selected_count"])
        formal_dataset = directory / "formal_dataset"
        dataset_resume = arguments.resume and (formal_dataset / "checkpoint.json").is_file()
        if (
            arguments.resume
            and formal_dataset.exists()
            and any(formal_dataset.iterdir())
            and not dataset_resume
        ):
            raise InputValidationError(
                "adaptive formal dataset is nonempty but has no resumable checkpoint"
            )
        manifest = ShardedDatasetWriter(
            formal_dataset,
            shard_size=arguments.shard_size,
            split_seed=proposal_manifest["selection_random_identity"]["seed"],
            split_ratios=(1.0, 0.0, 0.0),
            tensor_compression=arguments.tensor_compression,
        ).run_cases(
            engine,
            iter_adaptive_formal_cases(root, arguments.round),
            fail_closed=arguments.fail_fast,
            workers=arguments.workers,
            chunksize=arguments.chunksize,
            resume=dataset_resume,
            cases_base_directory=selected_path.parent,
            expected_case_count=selected_count,
        )
        committed = bind_adaptive_round_archive(
            root, arguments.round, formal_dataset / "manifest.json"
        )
    print(json.dumps({
        "status": "archive_complete",
        "round": arguments.round,
        "dataset_id": manifest["dataset_id"],
        "manifest": str(formal_dataset / "manifest.json"),
        "selected_count": selected_count,
        "checkpoint_hash": committed["checkpoint_hash"],
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def command_sampler_adaptive_update(arguments) -> int:
    state = commit_adaptive_round_outcomes(
        arguments.output_directory, arguments.round, None
    )
    outcomes = _load_cases(
        Path(arguments.output_directory)
        / "rounds"
        / f"round-{arguments.round:03d}"
        / "formal_outcomes.jsonl"
    )
    print(json.dumps({
        "status": "outcomes_committed",
        "round": arguments.round,
        "checkpoint_hash": state["checkpoint_hash"],
        "outcome_count": len(outcomes),
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def command_sampler_adaptive_audit(arguments) -> int:
    state = verify_multiround_sampling_checkpoint(arguments.output_directory)
    rows: list[dict] = []
    root = Path(arguments.output_directory)
    for round_index in state["rounds_committed"][1:]:
        rows.extend(_load_cases(
            root / "rounds" / f"round-{round_index:03d}" / "adaptive_candidates.jsonl"
        ))
    if not rows:
        raise InputValidationError("no adaptive round is available to audit")
    proposal_metric_names = [
        "p0_density", "adaptive_density", "proposal_density",
        "p0_hierarchy_probability", "proposal_hierarchy_probability",
        "score_rarity", "score_uncertainty", "score_coverage", "score_total",
    ]
    for row in rows:
        for name in proposal_metric_names:
            try:
                value = float(row[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise InputValidationError(
                    "adaptive audit proposal metrics are incomplete or nonnumeric",
                    context={"candidate_id": row.get("candidate_id"), "field": name},
                ) from exc
            if not math.isfinite(value):
                raise InputValidationError(
                    "adaptive audit proposal metrics must be finite",
                    context={"candidate_id": row.get("candidate_id"), "field": name},
                )
    sampling_spec = SamplingSpec.from_file(root / "plan" / "sampling_spec.json")
    report = audit_sampling(
        rows, None, quality_gates=sampling_spec.data["quality_gates"]
    )
    report["audit_feature_names"] = []
    report["proposal_metric_names"] = proposal_metric_names
    report["audit_scope"] = (
        "adaptive proposal identity, exact design weights, inclusion probabilities, "
        "hierarchy probabilities, ESS and declared stream isolation; raw-input key "
        "projections and cross-round input-space coverage require a separately frozen "
        "pre-label projection ledger"
    )
    report["multiround_checkpoint_hash"] = state["checkpoint_hash"]
    if arguments.output:
        destination = Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def command_sampler_adaptive_finalize(arguments) -> int:
    state = finalize_multiround_sampling(arguments.output_directory)
    print(json.dumps({
        "status": state["status"],
        "checkpoint_hash": state["checkpoint_hash"],
        "rounds_committed": state["rounds_committed"],
        "finalized_at": state.get("finalized_at"),
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def command_dataset_policy_validate(arguments) -> int:
    policy = DatasetUsePolicy.from_file(arguments.policy)
    manifest = None
    if arguments.manifest:
        manifest = policy.assert_archive_binding(arguments.manifest)
    elif arguments.operation:
        raise InputValidationError(
            "an operation check requires --manifest to bind the policy to the exact archive"
        )
    if arguments.operation:
        policy.assert_operation(
            arguments.operation,
            model_hash=arguments.model_hash,
            require_research_admission=arguments.require_research_admission,
        )
    print(json.dumps({
        "status": "ok",
        "policy_hash": policy.hash,
        "dataset_id": policy.data["dataset_id"],
        "classification": policy.classification,
        "operation": arguments.operation,
        "manifest_bound": manifest is not None,
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def command_dataset_policy_open_test(arguments) -> int:
    policy = DatasetUsePolicy.from_file(arguments.policy)
    opened = policy.open_reference_test(arguments.frozen_model_hash)
    write_dataset_policy(arguments.output, opened)
    print(json.dumps({
        "status": "opened_for_frozen_model",
        "policy_hash": opened.hash,
        "dataset_id": opened.data["dataset_id"],
        "frozen_model_hash": arguments.frozen_model_hash,
        "output": str(Path(arguments.output)),
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="damage-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hash_parser = subparsers.add_parser("hash-target", help="print canonical target content hash")
    hash_parser.add_argument("--target", required=True)
    hash_parser.set_defaults(handler=command_hash)

    import_parser = subparsers.add_parser(
        "import-raw",
        help="convert the observed List.csv/armor.csv export into a target JSON",
    )
    import_parser.add_argument("--components", required=True, help="path to List.csv")
    import_parser.add_argument("--armor", required=True, help="path to armor.csv")
    import_parser.add_argument("--config", required=True, help="explicit import profile JSON")
    import_parser.add_argument("--output", required=True, help="target JSON to write")
    import_parser.add_argument("--diagnostics", help="optional machine-readable import report")
    import_policy = import_parser.add_mutually_exclusive_group()
    import_policy.add_argument(
        "--skip-incomplete-components",
        action="store_true",
        default=None,
        help="isolate rows without complete geometry and record them in diagnostics",
    )
    import_policy.add_argument(
        "--strict-components",
        action="store_false",
        dest="skip_incomplete_components",
        help="reject the import on the first incomplete component",
    )
    import_parser.set_defaults(handler=command_import_raw, skip_incomplete_components=None)

    validate_parser = subparsers.add_parser("validate", help="validate target, case, and registry")
    validate_parser.add_argument("--target", required=True)
    validate_parser.add_argument("--case", required=True)
    validate_parser.add_argument("--physics", required=True)
    validate_parser.set_defaults(handler=command_validate)

    simulate_parser = subparsers.add_parser("simulate", help="run one case")
    simulate_parser.add_argument("--target", required=True)
    simulate_parser.add_argument("--case", required=True)
    simulate_parser.add_argument("--physics", required=True)
    simulate_parser.add_argument("--output")
    simulate_parser.add_argument("--record-refusal", action="store_true")
    simulate_parser.set_defaults(handler=command_simulate)

    generate_parser = subparsers.add_parser("generate", help="run cases and write an ML dataset")
    generate_parser.add_argument("--target", required=True)
    generate_parser.add_argument("--cases", required=True)
    generate_parser.add_argument("--physics", required=True)
    generate_parser.add_argument("--output-directory", required=True)
    failure_group = generate_parser.add_mutually_exclusive_group()
    failure_group.add_argument(
        "--fail-fast",
        action="store_true",
        help="abort the batch on the first refused or failed case (default: isolate and record)",
    )
    failure_group.add_argument(
        "--record-refusal",
        action="store_true",
        help=(
            "deprecated compatibility no-op (since 0.7); refusal recording is "
            "the default; scheduled for removal in the next breaking release"
        ),
    )
    generate_parser.add_argument("--workers", type=int, default=1)
    generate_parser.add_argument(
        "--chunksize",
        type=int,
        help="number of cases sent to each worker per dispatch",
    )
    generate_parser.add_argument("--split-seed", type=int, default=0)
    generate_parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    generate_parser.add_argument(
        "--tensor-compression",
        choices=("compressed", "none"),
        default="compressed",
        help="use 'none' for faster generation/loading at the cost of disk space",
    )
    generate_parser.add_argument(
        "--shard-size",
        type=int,
        help="bound memory by writing independent shards with at most this many cases",
    )
    generate_parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a sharded generation after validating its checkpoint identity",
    )
    generate_parser.set_defaults(handler=command_generate)

    sampler_parser = subparsers.add_parser(
        "sampler", help="build and execute an auditable reference/pilot sampling plan"
    )
    sampler_subparsers = sampler_parser.add_subparsers(dest="sampler_command", required=True)

    def add_sampling_inputs(command_parser):
        command_parser.add_argument("--target", required=True)
        command_parser.add_argument("--physics", required=True)
        command_parser.add_argument("--base-case", required=True)
        command_parser.add_argument("--sampling-spec", required=True)
        command_parser.add_argument("--label-spec", required=True)

    validate_plan_parser = sampler_subparsers.add_parser(
        "validate-plan", help="validate p0, label semantics and all frozen pilot cases without running labels"
    )
    add_sampling_inputs(validate_plan_parser)
    validate_plan_parser.set_defaults(handler=command_sampler_validate_plan)

    sampler_init_parser = sampler_subparsers.add_parser(
        "init", help="freeze reference validation/test and train-pilot candidate ledgers"
    )
    add_sampling_inputs(sampler_init_parser)
    sampler_init_parser.add_argument("--output-directory", required=True)
    sampler_init_parser.add_argument("--resume", action="store_true")
    sampler_init_parser.set_defaults(handler=command_sampler_init)

    pilot_parser = sampler_subparsers.add_parser(
        "pilot", help="freeze a p0 plan and execute its formal cases into a sharded ML dataset"
    )
    add_sampling_inputs(pilot_parser)
    pilot_parser.add_argument("--output-directory", required=True)
    pilot_parser.add_argument("--workers", type=int, default=1)
    pilot_parser.add_argument("--chunksize", type=int)
    pilot_parser.add_argument("--shard-size", type=int, default=32)
    pilot_parser.add_argument("--tensor-compression", choices=("compressed", "none"), default="compressed")
    pilot_parser.add_argument("--fail-fast", action="store_true")
    pilot_parser.add_argument("--resume", action="store_true")
    pilot_parser.set_defaults(handler=command_sampler_pilot)

    audit_parser = sampler_subparsers.add_parser(
        "audit", help="verify a sampling checkpoint and print its pre-label sampling diagnostics"
    )
    audit_parser.add_argument("--output-directory", required=True)
    audit_parser.set_defaults(handler=command_sampler_audit)

    adaptive_init = sampler_subparsers.add_parser(
        "adaptive-init",
        help="attach a fail-closed multiround sidecar to a complete reference archive",
    )
    adaptive_init.add_argument("--output-directory", required=True)
    adaptive_init.add_argument("--resume", action="store_true")
    adaptive_init.set_defaults(handler=command_sampler_adaptive_init)

    adaptive_propose = sampler_subparsers.add_parser(
        "adaptive-propose",
        help="freeze and commit one Train-only general-proposal candidate round",
    )
    adaptive_propose.add_argument("--output-directory", required=True)
    adaptive_propose.add_argument(
        "--proposal-input",
        required=True,
        help="adaptive-proposal-input@1.1.0 JSON with bound candidate cases, external OOF scores and model hash",
    )
    adaptive_propose.set_defaults(handler=command_sampler_adaptive_propose)

    adaptive_run = sampler_subparsers.add_parser(
        "adaptive-run-selected",
        help="execute only the content-bound selected formal-case ledger into a sharded archive",
    )
    adaptive_run.add_argument("--output-directory", required=True)
    adaptive_run.add_argument("--round", type=int, required=True)
    adaptive_run.add_argument("--target", required=True)
    adaptive_run.add_argument("--physics", required=True)
    adaptive_run.add_argument("--workers", type=int, default=1)
    adaptive_run.add_argument("--chunksize", type=int)
    adaptive_run.add_argument("--shard-size", type=int, default=32)
    adaptive_run.add_argument(
        "--tensor-compression", choices=("compressed", "none"), default="compressed"
    )
    adaptive_run.add_argument("--fail-fast", action="store_true")
    adaptive_run.add_argument("--resume", action="store_true")
    adaptive_run.set_defaults(handler=command_sampler_adaptive_run_selected)

    adaptive_update = sampler_subparsers.add_parser(
        "adaptive-update",
        help="derive and commit selected-candidate outcomes from the verified formal archive",
    )
    adaptive_update.add_argument("--output-directory", required=True)
    adaptive_update.add_argument("--round", type=int, required=True)
    adaptive_update.set_defaults(handler=command_sampler_adaptive_update)

    adaptive_audit = sampler_subparsers.add_parser(
        "adaptive-audit",
        help="verify all adaptive rounds and audit proposal weights, tails, ESS and drift",
    )
    adaptive_audit.add_argument("--output-directory", required=True)
    adaptive_audit.add_argument("--output")
    adaptive_audit.set_defaults(handler=command_sampler_adaptive_audit)

    adaptive_finalize = sampler_subparsers.add_parser(
        "adaptive-finalize",
        help="finalize a multiround run after every selected outcome is committed",
    )
    adaptive_finalize.add_argument("--output-directory", required=True)
    adaptive_finalize.set_defaults(handler=command_sampler_adaptive_finalize)

    policy_parser = subparsers.add_parser(
        "dataset-policy", help="validate dataset use roles and sealed-test access"
    )
    policy_subparsers = policy_parser.add_subparsers(dest="policy_command", required=True)
    policy_validate = policy_subparsers.add_parser(
        "validate", help="validate a dataset role policy and an optional requested operation"
    )
    policy_validate.add_argument("--policy", required=True)
    policy_validate.add_argument(
        "--operation",
        choices=(
            "archive_audit", "software_development", "exploratory_modeling",
            "training", "normalizer_fit", "acquisition_fit",
            "validation_model_selection", "evaluation", "final_evaluation",
            "stress_evaluation",
        ),
    )
    policy_validate.add_argument("--model-hash")
    policy_validate.add_argument(
        "--manifest",
        help="exact dataset manifest.json whose ID and file hash must match the policy",
    )
    policy_validate.add_argument("--require-research-admission", action="store_true")
    policy_validate.set_defaults(handler=command_dataset_policy_validate)
    policy_open = policy_subparsers.add_parser(
        "open-test", help="bind a sealed reference test to one already-frozen model"
    )
    policy_open.add_argument("--policy", required=True)
    policy_open.add_argument("--frozen-model-hash", required=True)
    policy_open.add_argument("--output", required=True)
    policy_open.set_defaults(handler=command_dataset_policy_open_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except DamageEngineError as exc:
        print(json.dumps({"status": "refused", "error": exc.to_dict()}, ensure_ascii=False))
        return 2
    except OSError as exc:
        print(json.dumps({"status": "refused", "error": {"code": "REF_INPUT_IO", "message": str(exc)}}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
