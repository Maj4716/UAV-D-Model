from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.engine import DamageEngine, ENGINE_VERSION
from damage_engine.registry import PhysicsRegistry
from damage_engine.utils import content_hash, load_json
from damage_engine.validation import clear_validation_caches, validation_cache_info


SCIENTIFIC_KEYS = (
    "case_id",
    "replicate_count",
    "parameter_realization",
    "armor",
    "components",
    "damage_tree",
    "summary",
)


def _cases(base: dict, count: int, seed_base: int) -> list[dict]:
    cases = []
    for index in range(count):
        case = deepcopy(base)
        case["case_id"] = f"engine-hotpath-{index:06d}"
        case["analysis"]["aleatory_seed"] = seed_base + index
        cases.append(case)
    return cases


def _scientific_digest(results: list[dict]) -> str:
    payload = [
        {key: result[key] for key in SCIENTIFIC_KEYS}
        for result in results
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic DamageEngine hot-path benchmark with a scientific-output digest."
    )
    parser.add_argument("--target", type=Path, default=ROOT / "examples/target.demo.json")
    parser.add_argument(
        "--case", type=Path, default=ROOT / "raw_data/case.5000.software_demo.v2.json"
    )
    parser.add_argument("--registry", type=Path, default=ROOT / "examples/physics.demo.json")
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=11000)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.cases < 1 or arguments.rounds < 1:
        parser.error("--cases and --rounds must be positive")

    target = load_json(arguments.target)
    base_case = load_json(arguments.case)
    registry = PhysicsRegistry.from_file(arguments.registry)
    engine = DamageEngine(target, registry)
    cases = _cases(base_case, arguments.cases, arguments.seed_base)

    clear_validation_caches()
    engine.run(cases[0])
    elapsed: list[float] = []
    digests: list[str] = []
    for _ in range(arguments.rounds):
        started = time.perf_counter()
        results = [engine.run(case).to_dict() for case in cases]
        elapsed.append(time.perf_counter() - started)
        digests.append(_scientific_digest(results))
    if len(set(digests)) != 1:
        raise RuntimeError("scientific outputs changed between benchmark rounds")

    median_seconds = statistics.median(elapsed)
    report = {
        "benchmark_schema_version": "1.0.0",
        "engine_version": ENGINE_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": {
            "target": str(arguments.target.resolve()),
            "target_hash": content_hash(target),
            "case": str(arguments.case.resolve()),
            "case_template_hash": content_hash(base_case),
            "registry": str(arguments.registry.resolve()),
            "registry_hash": registry.hash,
            "case_count": arguments.cases,
            "round_count": arguments.rounds,
            "seed_base": arguments.seed_base,
        },
        "elapsed_seconds": elapsed,
        "median_seconds": median_seconds,
        "median_cases_per_second": arguments.cases / median_seconds,
        "scientific_output_hash": digests[0],
        "scientific_output_scope": list(SCIENTIFIC_KEYS),
        "validation_cache": validation_cache_info(),
        "evidence_boundary": (
            "software hot-path timing and deterministic simulator-output equivalence only; "
            "not evidence of physical validity"
        ),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
