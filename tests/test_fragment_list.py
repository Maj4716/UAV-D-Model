from __future__ import annotations

import csv
import json
from copy import deepcopy

import numpy as np
import pytest

from damage_engine.engine import DamageEngine
from damage_engine.errors import InputValidationError, OutOfDomainError
from damage_engine.fragments import generate_fragments, load_fragment_list
from damage_engine.validation import validate_inputs


def _records() -> list[dict]:
    return [
        {
            "id": "joint-a",
            "mass_kg": 0.005,
            "direction_body_xyz": [1.0, 0.0, 0.0],
            "initial_velocity_mps": 1000.0,
            "multiplicity": 2,
        },
        {
            "id": "joint-b",
            "mass_kg": 0.006,
            "direction_body_xyz": [0.0, 1.0, 0.0],
            "initial_velocity_mps": 1100.0,
        },
    ]


def _case_with_list(case_data: dict, path, *, total_count: int = 3) -> dict:
    case = deepcopy(case_data)
    source = case["warhead"]["fragment_source"]
    source["fragment_list_ref"] = str(path)
    source["total_count"] = total_count
    return case


@pytest.mark.parametrize("format_name", ["csv", "json", "jsonl"])
def test_fragment_list_formats_preserve_joint_records_and_multiplicity(
    tmp_path,
    case_data,
    registry,
    format_name,
):
    records = _records()
    path = tmp_path / f"fragments.{format_name}"
    if format_name == "csv":
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "id",
                    "mass_kg",
                    "direction_body_x",
                    "direction_body_y",
                    "direction_body_z",
                    "initial_velocity_mps",
                    "multiplicity",
                ],
            )
            writer.writeheader()
            for record in records:
                x, y, z = record["direction_body_xyz"]
                writer.writerow(
                    {
                        "id": record["id"],
                        "mass_kg": record["mass_kg"],
                        "direction_body_x": x,
                        "direction_body_y": y,
                        "direction_body_z": z,
                        "initial_velocity_mps": record["initial_velocity_mps"],
                        "multiplicity": record.get("multiplicity", ""),
                    }
                )
    elif format_name == "json":
        path.write_text(json.dumps({"fragments": records}), encoding="utf-8")
    else:
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

    case = _case_with_list(case_data, path)
    loaded = load_fragment_list(case["warhead"]["fragment_source"])
    assert loaded.expanded_count == 3
    assert loaded.total_mass_kg == pytest.approx(0.016)
    assert loaded.provenance["content_hash"].startswith("sha256:")
    assert loaded.provenance["format"] == format_name

    fragments = generate_fragments(case, registry, np.random.default_rng(123))
    assert [fragment.mass_kg for fragment in fragments] == [0.005, 0.005, 0.006]
    assert [fragment.initial_speed_mps for fragment in fragments] == [
        1000.0,
        1000.0,
        1100.0,
    ]
    assert [fragment.source_fragment_id for fragment in fragments] == [
        "joint-a",
        "joint-a",
        "joint-b",
    ]
    assert [fragment.source_multiplicity_index for fragment in fragments] == [1, 2, 1]
    assert np.allclose(fragments[0].direction, [1.0, 0.0, 0.0])
    assert np.allclose(fragments[2].direction, [0.0, 1.0, 0.0])


def test_fragment_list_rejects_duplicate_ids_and_count_mismatch(tmp_path, case_data):
    duplicate_path = tmp_path / "duplicate.json"
    records = _records()
    records[1]["id"] = records[0]["id"]
    duplicate_path.write_text(json.dumps(records), encoding="utf-8")
    duplicate_case = _case_with_list(case_data, duplicate_path)
    with pytest.raises(InputValidationError, match="ids must be unique"):
        load_fragment_list(duplicate_case["warhead"]["fragment_source"])

    mismatch_path = tmp_path / "mismatch.json"
    mismatch_path.write_text(json.dumps(_records()), encoding="utf-8")
    mismatch_case = _case_with_list(case_data, mismatch_path, total_count=4)
    with pytest.raises(InputValidationError, match="expanded count"):
        load_fragment_list(mismatch_case["warhead"]["fragment_source"])


def test_fragment_list_mass_budget_is_guarded_during_validation(
    tmp_path,
    target_data,
    case_data,
    registry,
):
    path = tmp_path / "over-budget.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "heavy",
                    "mass_kg": 1.1,
                    "direction_body_xyz": [1.0, 0.0, 0.0],
                    "initial_velocity_mps": 1000.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    case = _case_with_list(case_data, path, total_count=1)
    case["warhead"]["fragment_source"]["mass_distribution"]["valid_range_kg"] = [
        0.001,
        2.0,
    ]
    with pytest.raises(OutOfDomainError, match="mass budget"):
        validate_inputs(target_data, case, registry)


def test_fragment_list_hash_reaches_provenance_and_combined_frechet_bounds(
    tmp_path,
    target_data,
    case_data,
    registry,
):
    path = tmp_path / "fragments.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "traceable",
                "mass_kg": 0.005,
                "direction_body_xyz": [1.0, 0.0, 0.0],
                "initial_velocity_mps": 1000.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    case = _case_with_list(case_data, path, total_count=1)
    case["analysis"]["replicates"] = 1
    case["analysis"]["adaptive_ci_half_width"] = None
    result = DamageEngine(target_data, registry).run(case).to_dict()

    fragment_list = result["provenance"]["fragment_list"]
    assert fragment_list["reference"] == str(path)
    assert fragment_list["content_hash"].startswith("sha256:")
    assert fragment_list["expanded_fragment_count"] == 1
    assert result["combined_dependency_bounds"] == "frechet_union"
    for state in ("degraded", "failed"):
        combined = result["components"]["flight-control"]["damage_states"][
            "combined"
        ][state]
        lower = combined["frechet_union_lower_bound"]
        upper = combined["frechet_union_upper_bound"]
        assert lower["mean"] <= combined["mean"] <= upper["mean"]


def test_fragment_list_does_not_accept_component_inventory_csv(tmp_path, case_data):
    path = tmp_path / "List.csv"
    path.write_text("序号,等效结构名称\n1,发动机\n", encoding="utf-8")
    case = _case_with_list(case_data, path, total_count=1)
    with pytest.raises(InputValidationError, match="unsupported columns"):
        load_fragment_list(case["warhead"]["fragment_source"])
