"""Lexical content identities and structural JSON document contracts.

This module deliberately does not compute hashes or perform domain-semantic
validation.  It owns the small, stable boundary shared by archive readers,
writers, and contract documents: SHA-256 identity syntax and deterministic
JSON Schema diagnostics.
"""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Literal, TypeGuard

from jsonschema import Draft202012Validator

from .errors import InputValidationError
from .schema_resources import load_schema


Sha256Acceptance = Literal[
    "canonical_lowercase",
    "legacy_lowercase_python_hex",
    "legacy_python_hex",
    "legacy_prefix_length",
    "legacy_prefix_only",
]
SchemaCompatibility = Literal[
    "canonical",
    "legacy_messages",
    "legacy_records",
]

_CANONICAL_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def is_sha256_identity(
    value: object,
    *,
    acceptance: Sha256Acceptance = "canonical_lowercase",
) -> TypeGuard[str]:
    """Return whether *value* matches the selected SHA-256 lexical contract.

    Legacy modes intentionally reproduce historical readers exactly.  They are
    migration aids, not acceptable defaults for newly written contracts.  The
    function never normalizes the supplied identity because doing so would
    silently change a content binding.
    """

    if not isinstance(value, str):
        return False
    if acceptance == "canonical_lowercase":
        return _CANONICAL_SHA256.fullmatch(value) is not None
    if acceptance == "legacy_prefix_only":
        return value.startswith("sha256:")
    if acceptance == "legacy_prefix_length":
        return value.startswith("sha256:") and len(value) == 71
    if acceptance in {"legacy_lowercase_python_hex", "legacy_python_hex"}:
        if not value.startswith("sha256:") or len(value) != 71:
            return False
        try:
            # Preserve the exact historical ``int(payload, 16)`` acceptance
            # set, including uppercase, signs, whitespace and a 0x prefix.
            int(value[7:], 16)
        except ValueError:
            return False
        if acceptance == "legacy_lowercase_python_hex":
            return value[7:] == value[7:].lower()
        return True
    raise ValueError(f"unknown SHA-256 acceptance mode: {acceptance}")


def require_sha256_identity(
    value: object,
    *,
    field: str,
    acceptance: Sha256Acceptance = "canonical_lowercase",
    failure_message: str | None = None,
) -> str:
    """Return an accepted identity or raise the stable input error category."""

    if not is_sha256_identity(value, acceptance=acceptance):
        message = failure_message
        if message is None:
            qualifier = (
                "complete lowercase "
                if acceptance == "canonical_lowercase"
                else ""
            )
            message = f"{field} must be a {qualifier}sha256 identity"
        raise InputValidationError(message)
    return value


@lru_cache(maxsize=None)
def _compiled_schema_validator(schema_name: str) -> Draft202012Validator:
    return Draft202012Validator(load_schema(schema_name))


def _error_sort_key(error) -> tuple:
    path = tuple((type(value).__name__, str(value)) for value in error.absolute_path)
    schema_path = tuple(
        (type(value).__name__, str(value)) for value in error.absolute_schema_path
    )
    return path, schema_path, str(error.validator), error.message


def _error_path(error, *, root: str) -> str:
    return "/".join(str(value) for value in error.absolute_path) or root


def validate_json_schema(
    instance: object,
    schema_name: str,
    *,
    failure_message: str | None = None,
    max_errors: int = 20,
    compatibility: SchemaCompatibility = "canonical",
) -> None:
    """Validate one document and raise deterministic, machine-readable errors.

    ``schema_errors`` always contains the canonical record representation.
    During migration, ``errors`` can additionally retain either historical
    message strings or records without the validator field.
    """

    if not isinstance(max_errors, int) or isinstance(max_errors, bool) or max_errors < 1:
        raise ValueError("max_errors must be a positive integer")
    if compatibility not in {"canonical", "legacy_messages", "legacy_records"}:
        raise ValueError(f"unknown schema compatibility mode: {compatibility}")

    errors = sorted(
        _compiled_schema_validator(schema_name).iter_errors(instance),
        key=_error_sort_key,
    )
    if not errors:
        return

    selected = errors[:max_errors]
    records = [
        {
            "path": _error_path(error, root="$"),
            "message": error.message,
            "validator": error.validator,
        }
        for error in selected
    ]
    if compatibility == "legacy_messages":
        legacy_errors: list[object] = [
            f"{_error_path(error, root='<root>')}: {error.message}"
            for error in selected
        ]
    elif compatibility == "legacy_records":
        legacy_errors = [
            {"path": record["path"], "message": record["message"]}
            for record in records
        ]
    else:
        legacy_errors = records

    raise InputValidationError(
        failure_message or f"{schema_name} validation failed",
        context={
            "schema_name": schema_name,
            "schema_errors": records,
            "error_count": len(errors),
            "errors_truncated": len(errors) > max_errors,
            "errors": legacy_errors,
        },
    )


# Cache diagnostics are attributes of the schema-validation entry point rather
# than additional domain operations.  ``validation_cache_info`` and
# ``clear_validation_caches`` use these to preserve their existing API.
validate_json_schema.cache_info = _compiled_schema_validator.cache_info  # type: ignore[attr-defined]
validate_json_schema.cache_clear = _compiled_schema_validator.cache_clear  # type: ignore[attr-defined]


__all__ = [
    "is_sha256_identity",
    "require_sha256_identity",
    "validate_json_schema",
]
