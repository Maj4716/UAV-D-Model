from __future__ import annotations

import hashlib
import json
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Any


SCHEMA_PACKAGE = "damage_engine.schemas"
_SCHEMA_SUFFIX = ".schema.json"


def _resource_root() -> Traversable:
    """Return the package-backed schema directory.

    ``importlib.resources`` keeps runtime validation independent of the source
    checkout layout and also works after wheel installation.
    """

    return resources.files(SCHEMA_PACKAGE)


def _validated_name(name: str) -> str:
    value = str(name)
    if (
        not value.endswith(_SCHEMA_SUFFIX)
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(
            "schema name must be a package-local '*.schema.json' filename"
        )
    return value


def schema_names() -> tuple[str, ...]:
    """List the versioned JSON Schema resources in deterministic order."""

    return tuple(
        sorted(
            item.name
            for item in _resource_root().iterdir()
            if item.is_file() and item.name.endswith(_SCHEMA_SUFFIX)
        )
    )


def schema_bytes(name: str) -> bytes:
    """Read one bundled schema as bytes, rejecting path traversal."""

    filename = _validated_name(name)
    resource = _resource_root().joinpath(filename)
    if not resource.is_file():
        raise FileNotFoundError(f"bundled schema resource not found: {filename}")
    return resource.read_bytes()


def load_schema(name: str) -> dict[str, Any]:
    """Load one bundled JSON Schema object."""

    value = json.loads(schema_bytes(name).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"bundled schema must contain a JSON object: {name}")
    return value


def schema_tree_hash() -> str:
    """Hash the names and bytes of every bundled JSON Schema resource."""

    digest = hashlib.sha256()
    for name in schema_names():
        encoded_name = name.encode("utf-8")
        data = schema_bytes(name)
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return "sha256:" + digest.hexdigest()
