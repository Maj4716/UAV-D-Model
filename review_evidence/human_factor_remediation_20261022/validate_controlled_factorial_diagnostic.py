"""Fail-closed integrity checks for controlled_factorial_diagnostic outputs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUTPUT_DIR / "controlled_factorial_diagnostic.json"
LEDGER_PATH = OUTPUT_DIR / "candidate_ledger.jsonl"
REPORT_PATH = OUTPUT_DIR / "controlled_factorial_diagnostic.md"
SOURCE_PATH = OUTPUT_DIR / "controlled_factorial_diagnostic.py"
VALIDATION_PATH = OUTPUT_DIR / "controlled_factorial_validation.json"

sys.path.insert(0, str(ROOT))

from damage_engine.engine import ENGINE_CODE_HASH  # noqa: E402


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def strict_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON token: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def main() -> None:
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, observed: Any = None) -> None:
        checks[name] = {"pass": bool(passed), "observed": observed}

    required = [RESULT_PATH, LEDGER_PATH, REPORT_PATH, SOURCE_PATH]
    record(
        "required_files_exist",
        all(path.is_file() for path in required),
        [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()],
    )
    if not checks["required_files_exist"]["pass"]:
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "overall_pass": False,
            "checks": checks,
        }
        VALIDATION_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise SystemExit(1)

    result = strict_json(RESULT_PATH)
    ledger = [strict_json_line for strict_json_line in (
        json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line
    )]
    rows = result["case_results"]
    design = result["design"]
    status_counts = Counter(item["status"] for item in ledger)

    record(
        "ledger_length_matches_attempted",
        len(ledger) == int(design["attempted_case_condition_count"]),
        {"ledger": len(ledger), "attempted": design["attempted_case_condition_count"]},
    )
    record(
        "accepted_rows_match_ledger",
        len(rows) == int(status_counts.get("accepted", 0))
        == int(design["accepted_case_condition_count"]),
        {"rows": len(rows), "statuses": dict(status_counts)},
    )
    denominator_errors = [
        item["attempt_index"]
        for item in ledger
        if bool(item["included_in_label_denominator"])
        != (item["status"] == "accepted")
    ]
    record(
        "label_denominator_is_accepted_only",
        not denominator_errors,
        denominator_errors[:20],
    )

    contract = result["contract_checks"]
    required_contract_flags = (
        "all_accepted_use_first_contact_model",
        "all_accepted_target_facing",
        "all_accepted_body_forward_aligned",
        "all_accepted_have_eight_replicates",
        "accepted_candidate_sets_equal_across_axis_conditions",
    )
    record(
        "contact_and_pairing_contracts_pass",
        all(bool(contract[name]) for name in required_contract_flags),
        {name: contract[name] for name in required_contract_flags},
    )
    record(
        "six_directions_eighteen_orientations",
        int(contract["unique_velocity_direction_count"]) == 6
        and int(contract["unique_orientation_count"]) == 18,
        {
            "directions": contract["unique_velocity_direction_count"],
            "orientations": contract["unique_orientation_count"],
        },
    )

    expected_seed_count = int(design.get("expected_unique_aleatory_seed_count", 162))
    seed_groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        seed_groups.setdefault(int(row["aleatory_seed"]), []).append(row)
    seed_pairing_errors = []
    expected_axes = {"archived_minus_body_x", "counterfactual_plus_body_x"}
    for seed, group in sorted(seed_groups.items()):
        candidate_keys = {row["candidate_key"] for row in group}
        axes = {row["axis_condition"] for row in group}
        if len(group) != 2 or len(candidate_keys) != 1 or axes != expected_axes:
            seed_pairing_errors.append(
                {
                    "aleatory_seed": seed,
                    "row_count": len(group),
                    "candidate_keys": sorted(candidate_keys),
                    "axis_conditions": sorted(axes),
                }
            )
    record(
        "unique_aleatory_seed_count_is_162",
        len(seed_groups) == expected_seed_count == 162,
        {"observed": len(seed_groups), "expected": expected_seed_count},
    )
    record(
        "each_seed_pairs_one_candidate_across_two_axes",
        not seed_pairing_errors,
        seed_pairing_errors[:20],
    )
    record(
        "reported_seed_contract_matches_recomputation",
        int(contract.get("unique_aleatory_seed_count", -1)) == len(seed_groups)
        and int(contract.get("expected_unique_aleatory_seed_count", -1)) == 162
        and bool(
            contract.get(
                "each_seed_maps_to_one_candidate_key_and_two_axis_conditions", False
            )
        )
        == (not seed_pairing_errors),
        {
            "reported_unique": contract.get("unique_aleatory_seed_count"),
            "recomputed_unique": len(seed_groups),
            "reported_expected": contract.get("expected_unique_aleatory_seed_count"),
            "reported_pairing": contract.get(
                "each_seed_maps_to_one_candidate_key_and_two_axis_conditions"
            ),
        },
    )
    candidate_seed_groups: dict[str, set[int]] = {}
    for row in rows:
        candidate_seed_groups.setdefault(row["candidate_key"], set()).add(
            int(row["aleatory_seed"])
        )
    candidate_seed_errors = {
        key: sorted(values)
        for key, values in candidate_seed_groups.items()
        if len(values) != 1
    }
    record(
        "each_candidate_key_has_one_axis_shared_seed",
        len(candidate_seed_groups) == 162 and not candidate_seed_errors,
        {
            "candidate_count": len(candidate_seed_groups),
            "violations": candidate_seed_errors,
        },
    )
    epistemic_ids = {row["epistemic_sample_id"] for row in rows}
    record(
        "single_epistemic_realization_is_explicit",
        len(epistemic_ids) == 1
        and int(design.get("epistemic_realization_count", -1)) == 1
        and bool(design.get("epistemic_limitation")),
        {
            "unique_count": len(epistemic_ids),
            "values": sorted(epistemic_ids),
            "declared_count": design.get("epistemic_realization_count"),
        },
    )

    prefix_errors: list[str] = []
    monotonic_errors: list[str] = []
    for row in rows:
        previous = None
        for level in (1, 2, 4, 8):
            item = row["prefix"].get(str(level))
            if item is None or int(item["replicate_count"]) != level:
                prefix_errors.append(f"{row['axis_condition']}|{row['candidate_key']}|R{level}")
                continue
            current = {
                **{
                    f"category:{name}": bool(item["category_ever_positive"][name])
                    for name in ("K", "M", "F", "C")
                },
                "tree:any": bool(item["tree_any_positive"]),
            }
            if previous is not None:
                for name, was_positive in previous.items():
                    if was_positive and not current[name]:
                        monotonic_errors.append(
                            f"{row['axis_condition']}|{row['candidate_key']}|R{level}|{name}"
                        )
            previous = current
    record("prefix_replicate_counts_exact", not prefix_errors, prefix_errors[:20])
    record("nested_ever_positive_is_monotone", not monotonic_errors, monotonic_errors[:20])

    reported_flip_errors: list[str] = []
    for axis, comparisons in result["nested_r_discovery"].items():
        for comparison, item in comparisons.items():
            for category, counts in item["category"].items():
                if int(counts.get("1->0", 0)) != 0:
                    reported_flip_errors.append(
                        f"{axis}|{comparison}|{category}|{counts['1->0']}"
                    )
            if int(item["tree_any_positive"].get("1->0", 0)) != 0:
                reported_flip_errors.append(
                    f"{axis}|{comparison}|tree_any|"
                    f"{item['tree_any_positive']['1->0']}"
                )
    record(
        "reported_nested_transitions_have_no_one_to_zero",
        not reported_flip_errors,
        reported_flip_errors,
    )

    aggregate_errors: list[str] = []
    for level in (1, 2, 4, 8):
        key = str(level)
        for axis in ("archived_minus_body_x", "counterfactual_plus_body_x"):
            subset = [row for row in rows if row["axis_condition"] == axis]
            summary = result["aggregates"]["by_axis_and_r"][key][axis]
            if int(summary["accepted_case_count"]) != len(subset):
                aggregate_errors.append(f"R{level}|{axis}|n")
            for category in ("K", "M", "F", "C"):
                observed = sum(
                    bool(row["prefix"][key]["category_ever_positive"][category])
                    for row in subset
                )
                if observed != int(summary["category_ever_positive_count"][category]):
                    aggregate_errors.append(f"R{level}|{axis}|{category}")
            observed_tree = sum(
                bool(row["prefix"][key]["tree_any_positive"]) for row in subset
            )
            if observed_tree != int(summary["tree_any_positive_count"]):
                aggregate_errors.append(f"R{level}|{axis}|tree_any")
    record("aggregate_counts_recompute", not aggregate_errors, aggregate_errors)

    identity = result["software_identity"]
    current_files = {
        "script_sha256": file_hash(SOURCE_PATH),
        "legacy_factorial_script_sha256": file_hash(
            ROOT / identity["legacy_factorial_script_path"]
        ),
        "target_sha256": file_hash(ROOT / identity["target_path"]),
        "registry_sha256": file_hash(ROOT / identity["registry_path"]),
        "base_case_sha256": file_hash(ROOT / identity["base_case_path"]),
    }
    drift = {
        name: {"recorded": identity[name], "current": current}
        for name, current in current_files.items()
        if identity.get(name) != current
    }
    if identity.get("engine_code_hash") != ENGINE_CODE_HASH:
        drift["engine_code_hash"] = {
            "recorded": identity.get("engine_code_hash"),
            "current": ENGINE_CODE_HASH,
        }
    record("recorded_identity_matches_current_files", not drift, drift)

    overall = all(item["pass"] for item in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": overall,
        "source_result": RESULT_PATH.relative_to(ROOT).as_posix(),
        "source_result_sha256": file_hash(RESULT_PATH),
        "checks": checks,
        "claim_boundary": (
            "PASS establishes internal bookkeeping and identity consistency only; it does not "
            "establish physical validity, operational representativeness, or training admission."
        ),
    }
    VALIDATION_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
