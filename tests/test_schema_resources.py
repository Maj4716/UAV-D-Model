from __future__ import annotations

import pytest

from damage_engine.schema_resources import (
    load_schema,
    schema_bytes,
    schema_names,
    schema_tree_hash,
)


EXPECTED_SCHEMAS = (
    "dataset_manifest.schema.json",
    "label_spec.schema.json",
    "physics_registry.schema.json",
    "sampling_spec.schema.json",
    "simulation_case.schema.json",
    "target.schema.json",
)


def test_all_runtime_schemas_are_package_resources():
    assert schema_names() == EXPECTED_SCHEMAS
    for name in EXPECTED_SCHEMAS:
        schema = load_schema(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema_bytes(name)


def test_schema_tree_hash_is_deterministic():
    first = schema_tree_hash()
    second = schema_tree_hash()
    assert first == second
    assert first.startswith("sha256:")
    assert len(first) == len("sha256:") + 64


@pytest.mark.parametrize(
    "name",
    ["../target.schema.json", "schemas/target.schema.json", "target.json", ""],
)
def test_schema_resource_name_rejects_nonlocal_paths(name):
    with pytest.raises(ValueError, match="package-local"):
        load_schema(name)


def test_missing_bundled_schema_is_explicit():
    with pytest.raises(FileNotFoundError, match="bundled schema"):
        load_schema("missing.schema.json")
