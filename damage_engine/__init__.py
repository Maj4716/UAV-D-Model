"""Traceable fragment/blast damage simulator for ML dataset production."""

from .engine import DamageEngine, EngineResult
from .errors import DamageEngineError, InputValidationError, OutOfDomainError
from .normalization import normalize_target
from .registry import PhysicsRegistry

__all__ = [
    "DamageEngine",
    "EngineResult",
    "PhysicsRegistry",
    "DamageEngineError",
    "InputValidationError",
    "OutOfDomainError",
    "normalize_target",
]

__version__ = "0.4.0"
SCHEMA_VERSION = "1.0.0"
