from __future__ import annotations

import pytest

from damage_engine.content_contract import (
    is_sha256_identity,
    require_sha256_identity,
    validate_json_schema,
)
from damage_engine.errors import InputValidationError


_LOWERCASE = "sha256:" + "a" * 64
_UPPERCASE = "sha256:" + "A" * 64
_NON_HEX = "sha256:" + "z" * 64
_SIGNED_HEX = "sha256:" + "+" + "a" * 63
_SPACE_PREFIXED_HEX = "sha256:" + " " + "a" * 63
_PYTHON_PREFIXED_HEX = "sha256:" + "0x" + "a" * 62
_SHORT = "sha256:abc"
_EMPTY_DIGEST = "sha256:"


@pytest.mark.parametrize(
    (
        "value",
        "canonical",
        "legacy_lower_hex",
        "legacy_hex",
        "legacy_width",
        "legacy_prefix",
    ),
    [
        (_LOWERCASE, True, True, True, True, True),
        (_UPPERCASE, False, False, True, True, True),
        (_NON_HEX, False, False, False, True, True),
        (_SIGNED_HEX, False, True, True, True, True),
        (_SPACE_PREFIXED_HEX, False, True, True, True, True),
        (_PYTHON_PREFIXED_HEX, False, True, True, True, True),
        (_SHORT, False, False, False, False, True),
        (_EMPTY_DIGEST, False, False, False, False, True),
        ("sha512:" + "a" * 64, False, False, False, False, False),
        (b"sha256:" + b"a" * 64, False, False, False, False, False),
        (None, False, False, False, False, False),
    ],
)
def test_sha256_acceptance_profiles_lock_existing_contracts(
    value: object,
    canonical: bool,
    legacy_lower_hex: bool,
    legacy_hex: bool,
    legacy_width: bool,
    legacy_prefix: bool,
) -> None:
    assert is_sha256_identity(value) is canonical
    assert (
        is_sha256_identity(
            value, acceptance="legacy_lowercase_python_hex"
        )
        is legacy_lower_hex
    )
    assert (
        is_sha256_identity(value, acceptance="legacy_python_hex") is legacy_hex
    )
    assert (
        is_sha256_identity(value, acceptance="legacy_prefix_length")
        is legacy_width
    )
    assert (
        is_sha256_identity(value, acceptance="legacy_prefix_only")
        is legacy_prefix
    )


def test_sha256_requirement_preserves_value_message_and_error_category() -> None:
    assert (
        require_sha256_identity(
            _UPPERCASE,
            field="legacy_hash",
            acceptance="legacy_python_hex",
        )
        == _UPPERCASE
    )
    with pytest.raises(InputValidationError, match="^original contract message$"):
        require_sha256_identity(
            _UPPERCASE,
            field="strict_hash",
            failure_message="original contract message",
        )


def test_system_learning_hash_predicate_remains_a_legacy_compatible_shim() -> None:
    from damage_engine.system_learning import is_sha256_identity as legacy_predicate

    assert legacy_predicate(_SIGNED_HEX)
    assert legacy_predicate(_SPACE_PREFIXED_HEX)
    assert legacy_predicate(_PYTHON_PREFIXED_HEX)
    assert not legacy_predicate(_UPPERCASE)


def test_schema_errors_have_canonical_records_and_legacy_message_projection() -> None:
    with pytest.raises(InputValidationError, match="^original schema message$") as error:
        validate_json_schema(
            {},
            "label_spec.schema.json",
            failure_message="original schema message",
            max_errors=2,
            compatibility="legacy_messages",
        )

    context = error.value.context
    assert context["schema_name"] == "label_spec.schema.json"
    assert context["error_count"] > 2
    assert context["errors_truncated"] is True
    assert len(context["schema_errors"]) == 2
    assert all(set(record) == {"path", "message", "validator"} for record in context["schema_errors"])
    assert all(message.startswith("<root>: ") for message in context["errors"])


def test_schema_legacy_record_projection_and_cache_diagnostics() -> None:
    validate_json_schema.cache_clear()
    for _ in range(2):
        with pytest.raises(InputValidationError) as error:
            validate_json_schema(
                {},
                "physics_registry.schema.json",
                compatibility="legacy_records",
            )
        assert all(
            set(record) == {"path", "message"}
            for record in error.value.context["errors"]
        )
        assert all(
            set(record) == {"path", "message", "validator"}
            for record in error.value.context["schema_errors"]
        )
    cache = validate_json_schema.cache_info()
    assert cache.misses == 1
    assert cache.hits == 1


def test_validation_private_schema_entry_remains_a_compatibility_shim() -> None:
    from damage_engine.validation import _validate_schema

    with pytest.raises(
        InputValidationError,
        match=r"^label_spec\.schema\.json validation failed$",
    ) as error:
        _validate_schema({}, "label_spec.schema.json")
    assert error.value.context["errors"] == error.value.context["schema_errors"]
