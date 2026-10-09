from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np

from .errors import InputValidationError, MissingModelError, UnverifiedModelError
from .schema_resources import load_schema
from .utils import content_hash, load_json


def _require_finite_numbers(value: object, *, path: str = "$.") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_numbers(child, path=f"{path}{key}.")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_numbers(child, path=f"{path}{index}.")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise InputValidationError(
                "physics registry contains a non-finite numeric value",
                context={"path": path[:-1]},
            )


def _require_positive(value: object, *, name: str, model_id: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise InputValidationError(
            f"{name} must be finite and positive",
            context={"model_id": model_id, "value": number},
        )
    return number


@dataclass(frozen=True)
class PhysicsRegistry:
    data: dict

    @classmethod
    def from_file(cls, path: str | Path) -> "PhysicsRegistry":
        return cls(load_json(path))

    def __post_init__(self):
        schema = load_schema("physics_registry.schema.json")
        errors = sorted(
            Draft202012Validator(schema).iter_errors(self.data),
            key=lambda error: list(error.absolute_path),
        )
        if errors:
            details = []
            for error in errors[:20]:
                path = "/".join(str(value) for value in error.absolute_path) or "$"
                details.append({"path": path, "message": error.message})
            raise InputValidationError(
                "physics registry schema validation failed",
                context={"errors": details, "error_count": len(errors)},
            )
        _require_finite_numbers(self.data)
        required = {
            "schema_version",
            "config_id",
            "validation_status",
            "fragment_response_models",
            "fragment_source_models",
            "blast_field_models",
            "blast_response_models",
            "transmission_models",
        }
        missing = sorted(required - set(self.data))
        if missing:
            raise InputValidationError(
                "physics registry is missing required sections",
                context={"missing": missing},
            )
        for model_id, model in self.data["fragment_source_models"].items():
            initial = model["initial_velocity"]
            if initial["kind"] == "fixed" and "velocity_mps" not in initial:
                raise InputValidationError(
                    "fixed fragment source requires velocity_mps",
                    context={"model_id": model_id},
                )
            if initial["kind"] == "fixed":
                _require_positive(
                    initial["velocity_mps"],
                    name="fixed fragment velocity",
                    model_id=model_id,
                )
            direction = model["direction"]
            if direction["kind"] == "cone" and "half_angle_deg" not in direction:
                raise InputValidationError(
                    "cone fragment source requires half_angle_deg",
                    context={"model_id": model_id},
                )
            axis = np.asarray(direction.get("axis_body", [1.0, 0.0, 0.0]), dtype=float)
            if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 1.0e-12:
                raise InputValidationError(
                    "fragment source axis_body must be a finite non-zero 3-vector",
                    context={"model_id": model_id},
                )
            valid_speed = model.get("valid_speed_mps")
            if valid_speed is not None:
                lower, upper = map(float, valid_speed)
                if lower <= 0 or lower >= upper:
                    raise InputValidationError(
                        "fragment valid_speed_mps must be positive and strictly ordered",
                        context={"model_id": model_id, "valid_speed_mps": valid_speed},
                    )
        for model_id, model in self.data["blast_field_models"].items():
            table = model["scaled_distance_table"]
            scaled_distance = np.asarray([row["z_m_kg13"] for row in table], dtype=float)
            if np.any(np.diff(scaled_distance) <= 0):
                raise InputValidationError(
                    "blast scaled-distance tables must be strictly increasing",
                    context={"model_id": model_id},
                )
            peak = np.asarray([row["peak_pressure_pa"] for row in table], dtype=float)
            impulse = np.asarray(
                [row["scaled_impulse_pa_s_kgm13"] for row in table], dtype=float
            )
            duration = np.asarray(
                [row["scaled_duration_s_kgm13"] for row in table], dtype=float
            )
            if np.any(np.diff(peak) > 0) or np.any(np.diff(impulse) > 0):
                raise InputValidationError(
                    "blast peak pressure and impulse must not increase with scaled distance",
                    context={"model_id": model_id},
                )
            if np.any(np.diff(duration) < 0):
                raise InputValidationError(
                    "blast positive duration must not decrease with scaled distance",
                    context={"model_id": model_id},
                )
            reflection = model.get("reflection") or {}
            minimum = float(reflection.get("minimum_factor", 1.0))
            maximum = float(reflection.get("maximum_factor", 2.0))
            pressure_scale = float(reflection.get("pressure_scale_pa", 1.0))
            angular_exponent = float(reflection.get("angular_exponent", 1.0))
            if minimum <= 0 or maximum < minimum or pressure_scale <= 0 or angular_exponent < 0:
                raise InputValidationError(
                    "blast reflection parameters are physically inconsistent",
                    context={"model_id": model_id},
                )
            for exposure, amplification in (model.get("compartment_amplification") or {}).items():
                _require_positive(
                    amplification,
                    name=f"compartment amplification for {exposure}",
                    model_id=model_id,
                )
        for model_id, model in self.data["transmission_models"].items():
            _require_positive(
                model["peak_half_thickness_m"],
                name="peak half-thickness",
                model_id=model_id,
            )
            _require_positive(
                model["impulse_half_thickness_m"],
                name="impulse half-thickness",
                model_id=model_id,
            )

    @property
    def hash(self) -> str:
        return content_hash(self.data)

    @property
    def validation_status(self) -> str:
        return str(self.data["validation_status"])

    def require_validated(self) -> None:
        if self.validation_status != "validated":
            raise UnverifiedModelError(
                "L3 execution requires a validated physics registry",
                context={"validation_status": self.validation_status},
            )

    def fragment_response(self, model_id: str) -> dict:
        return self._get("fragment_response_models", model_id)

    def fragment_source(self, model_id: str) -> dict:
        return self._get("fragment_source_models", model_id)

    def blast_field(self, model_id: str) -> dict:
        return self._get("blast_field_models", model_id)

    def blast_response(self, model_id: str) -> dict:
        return self._get("blast_response_models", model_id)

    def transmission(self, model_id: str) -> dict:
        return self._get("transmission_models", model_id)

    def _get(self, section: str, model_id: str) -> dict:
        try:
            model = self.data[section][model_id]
        except KeyError as exc:
            raise MissingModelError(
                f"model {model_id!r} is not registered in {section}",
                context={"section": section, "model_id": model_id},
            ) from exc
        return model
