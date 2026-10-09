from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import DatasetWriter
from .engine import DamageEngine
from .normalization import normalize_target
from .raw_import import import_raw_target
from .registry import PhysicsRegistry
from .sharded_dataset import ShardedDatasetWriter
from .utils import content_hash, load_json
from .validation import validate_inputs


def _iter_cases(path: str | Path):
    value = Path(path)
    if value.suffix.lower() == ".jsonl":
        with value.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        return
    data = load_json(value)
    if isinstance(data, list):
        yield from data
    else:
        yield data


def _load_cases(path: str | Path) -> list[dict]:
    return list(_iter_cases(path))


def _engine(arguments) -> DamageEngine:
    target = load_json(arguments.target)
    registry = PhysicsRegistry.from_file(arguments.physics)
    return DamageEngine(target, registry)


def command_hash(arguments) -> int:
    print(content_hash(normalize_target(load_json(arguments.target))))
    return 0


def command_validate(arguments) -> int:
    engine = _engine(arguments)
    cases = _load_cases(arguments.case)
    for case in cases:
        validate_inputs(engine.target_data, case, engine.registry)
    print(json.dumps({"status": "ok", "validated_cases": len(cases)}, ensure_ascii=False))
    return 0


def command_simulate(arguments) -> int:
    engine = _engine(arguments)
    case = _load_cases(arguments.case)[0]
    result = engine.run(case, fail_closed=not arguments.record_refusal)
    if arguments.output:
        result.write_json(arguments.output)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.status == "ok" else 2


def command_generate(arguments) -> int:
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
        )
    else:
        if arguments.resume:
            raise ValueError("--resume requires --shard-size")
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
        help="deprecated compatibility flag; refusal recording is now the default",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
