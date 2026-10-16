from __future__ import annotations

from copy import deepcopy
import random

import pytest

from damage_engine.case_factory import CaseFactory
from damage_engine.input_identity import (
    _canonical_json_bytes_with_cached_subtrees,
    _canonical_subtree_cache,
    _content_hash_with_cached_subtrees,
    case_identity_hash,
    freeze_case_inputs,
    input_identity_from_case_hashes,
    semantic_case_value,
)
from damage_engine.label_spec import LabelSpec
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.utils import canonical_json_bytes, content_hash


def test_cached_canonical_writer_is_byte_identical_for_nested_json() -> None:
    rng = random.Random(20261013)
    registry = {
        "schema_version": "1.0.0",
        "registry_id": "中文-registry",
        "assignments": [
            {
                "lineage_id": f"lineage-{index:04d}",
                "split": rng.choice(["train", "validation", "test"]),
                "score": rng.uniform(-1.0e6, 1.0e6),
                "active": bool(index % 2),
                "note": None if index % 3 else "quote=\" slash=\\ newline=\n",
            }
            for index in range(128)
        ],
    }
    shared_spec = {
        "variables": ["position", "velocity", "attitude"],
        "limits": {"low": -30.0, "high": 30.0},
    }
    value = {
        "case_id": "formal-测试",
        "metadata": {
            "sampling": {
                "split_registry_document": registry,
                "sampling_spec_document": shared_spec,
            }
        },
        "values": [0, -0.0, 1.25, True, False, None, "飞行器"],
    }
    cache = _canonical_subtree_cache((registry, shared_spec))

    assert _canonical_json_bytes_with_cached_subtrees(value, cache) == canonical_json_bytes(value)
    assert _content_hash_with_cached_subtrees(value, cache) == content_hash(value)


@pytest.mark.parametrize(
    "spec_path",
    [
        "examples/sampling_spec.demo.json",
        "examples/sampling_spec.uav_control.demo.json",
    ],
)
def test_factory_cached_formal_hashes_match_reference_encoder(
    spec_path, case_data
) -> None:
    spec = SamplingSpec.from_file(spec_path)
    label = LabelSpec.from_file("examples/label_spec.demo.json")
    plan = CaseFactory(
        deepcopy(case_data),
        spec,
        label,
        target_hash="sha256:" + "1" * 64,
        damage_tree_definition_hash="sha256:" + "2" * 64,
        physics_registry_hash="sha256:" + "3" * 64,
        engine_code_hash="sha256:" + "4" * 64,
    ).build_reference_plan()

    assert [row["formal_case_hash"] for row in plan.candidate_rows] == [
        case_identity_hash(case) for case in plan.cases
    ]


def test_precomputed_ordered_identity_equals_full_freeze(case_data) -> None:
    cases = []
    shared_registry = {
        "schema_version": "1.0.0",
        "registry_id": "identity-registry",
        "assignment_phase": "pre_outcome",
        "assignments": [],
        "definition_hash": "sha256:" + "5" * 64,
    }
    for index in range(7):
        case = deepcopy(case_data)
        case["case_id"] = f"formal-{index}"
        case.setdefault("metadata", {})["sampling"] = {
            "candidate_id": f"candidate-{index}",
            "split_registry_document": shared_registry,
        }
        cases.append(case)

    with freeze_case_inputs(cases) as frozen:
        reference_identity = frozen.identity
    hashes = [case_identity_hash(case) for case in cases]
    assert input_identity_from_case_hashes(hashes) == reference_identity


def test_external_fragment_identity_keeps_content_semantics_and_safety(
    tmp_path, case_data
) -> None:
    source = tmp_path / "fragments.json"
    source.write_text('[{"mass_kg":0.005,"velocity_mps":[1,2,3]}]\n', encoding="utf-8")
    case = deepcopy(case_data)
    case["warhead"]["fragment_source"]["fragment_list_ref"] = source.name

    with freeze_case_inputs([case], cases_base_directory=tmp_path) as frozen_inputs:
        frozen_cases = list(frozen_inputs)
        frozen = frozen_cases[0]
        dependency = frozen.input_identity["dependencies"][0]
        expected = content_hash(
            {
                "case": semantic_case_value(frozen, loaded_dependency=dependency),
                "dependencies": [dependency],
            }
        )
        actual = case_identity_hash(frozen, loaded_dependency=dependency)
        derived = input_identity_from_case_hashes(
            [actual], dependencies=frozen_inputs.metadata["dependencies"]
        )

        assert actual == expected
        assert derived == frozen_inputs.identity
        with pytest.raises(Exception, match="differs from frozen snapshot"):
            case_identity_hash(
                frozen,
                loaded_dependency={**dependency, "content_hash": "sha256:" + "0" * 64},
            )


def test_invalid_precomputed_hash_is_rejected() -> None:
    with pytest.raises(Exception, match="not a SHA-256"):
        input_identity_from_case_hashes(["sha256:not-a-digest"])
