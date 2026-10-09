from __future__ import annotations


class DamageEngineError(RuntimeError):
    """Base error with a stable, machine-readable refusal code."""

    code = "ENGINE_ERROR"

    def __init__(self, message: str, *, context: dict | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }


class InputValidationError(DamageEngineError):
    code = "REF_INPUT_INVALID"


class OutOfDomainError(DamageEngineError):
    code = "REF_OUT_OF_DOMAIN"


class MissingModelError(DamageEngineError):
    code = "REF_MODEL_MISSING"


class UnverifiedModelError(DamageEngineError):
    code = "REF_MODEL_UNVERIFIED"


class NumericalError(DamageEngineError):
    code = "REF_NUMERICAL"


class DamageTreeError(DamageEngineError):
    code = "REF_DAMAGE_TREE"

