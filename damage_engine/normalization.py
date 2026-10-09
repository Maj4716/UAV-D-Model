from __future__ import annotations

from copy import deepcopy
from typing import Any

from .errors import InputValidationError


_CM_TO_M = 0.01


def _scale_vector(
    value: Any, *, field: str, scale: float = _CM_TO_M
) -> list[float]:
    """Validate a JSON three-vector and apply a deterministic unit scale."""

    if not isinstance(value, list) or len(value) != 3:
        raise InputValidationError(
            f"{field} must contain exactly three numeric values"
        )
    try:
        return [float(item) * scale for item in value]
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{field} must contain numeric values") from exc


def _coerce_quaternion(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise InputValidationError(
            f"{field} must contain exactly four numeric values"
        )
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{field} must contain numeric values") from exc


def _normalize_dimensionless_geometry(geometry: dict[str, Any]) -> None:
    if "orientation_xyzw" in geometry:
        geometry["orientation_xyzw"] = _coerce_quaternion(
            geometry["orientation_xyzw"], field="orientation_xyzw"
        )
    if "extrusion_direction" in geometry:
        geometry["extrusion_direction"] = _scale_vector(
            geometry["extrusion_direction"],
            field="extrusion_direction",
            scale=1.0,
        )


def _normalize_geometry(geometry: dict[str, Any], *, frame_unit: str) -> None:
    shape = geometry.get("shape")
    if frame_unit == "m":
        expected = {
            "box": ("center_m", "size_lwh_m"),
            "cylinder": ("center_m", "radius_m", "height_m"),
            "extruded_polygon": ("vertices_m", "extrusion_distance_m"),
        }.get(shape)
        forbidden = {
            "center_cm",
            "size_lwh_cm",
            "radius_cm",
            "height_cm",
            "vertices_cm",
            "extrusion_distance_cm",
        }
        if expected is None:
            return
        if any(name in geometry for name in forbidden):
            raise InputValidationError(
                "centimetre geometry fields require coordinate_frame.length_unit='cm'",
                context={"shape": shape},
            )
        missing = [name for name in expected if name not in geometry]
        if missing:
            raise InputValidationError(
                "geometry is missing SI fields",
                context={"shape": shape, "missing": missing},
            )
        if shape == "box":
            geometry["center_m"] = _scale_vector(
                geometry["center_m"], field="center_m", scale=1.0
            )
            geometry["size_lwh_m"] = _scale_vector(
                geometry["size_lwh_m"], field="size_lwh_m", scale=1.0
            )
        elif shape == "cylinder":
            geometry["center_m"] = _scale_vector(
                geometry["center_m"], field="center_m", scale=1.0
            )
            try:
                geometry["radius_m"] = float(geometry["radius_m"])
                geometry["height_m"] = float(geometry["height_m"])
            except (TypeError, ValueError) as exc:
                raise InputValidationError(
                    "radius_m and height_m must be numeric"
                ) from exc
        elif shape == "extruded_polygon":
            vertices = geometry["vertices_m"]
            if not isinstance(vertices, list):
                raise InputValidationError("vertices_m must be an array")
            geometry["vertices_m"] = [
                _scale_vector(vertex, field="vertices_m", scale=1.0)
                for vertex in vertices
            ]
            try:
                geometry["extrusion_distance_m"] = float(
                    geometry["extrusion_distance_m"]
                )
            except (TypeError, ValueError) as exc:
                raise InputValidationError(
                    "extrusion_distance_m must be numeric"
                ) from exc
        _normalize_dimensionless_geometry(geometry)
        return

    if frame_unit != "cm":
        raise InputValidationError(
            "coordinate_frame.length_unit must be 'm' or 'cm'",
            context={"length_unit": frame_unit},
        )

    metre_fields = {
        "center_m",
        "size_lwh_m",
        "radius_m",
        "height_m",
        "vertices_m",
        "extrusion_distance_m",
    }
    if any(name in geometry for name in metre_fields):
        raise InputValidationError(
            "metre and centimetre geometry fields cannot be mixed",
            context={"shape": shape},
        )

    if shape == "box":
        if "center_cm" not in geometry or "size_lwh_cm" not in geometry:
            raise InputValidationError(
                "box geometry in centimetres requires center_cm and size_lwh_cm"
            )
        geometry["center_m"] = _scale_vector(
            geometry.pop("center_cm"), field="center_cm"
        )
        geometry["size_lwh_m"] = _scale_vector(
            geometry.pop("size_lwh_cm"), field="size_lwh_cm"
        )
    elif shape == "cylinder":
        required = ("center_cm", "radius_cm", "height_cm")
        missing = [name for name in required if name not in geometry]
        if missing:
            raise InputValidationError(
                "cylinder geometry in centimetres is incomplete",
                context={"missing": missing},
            )
        geometry["center_m"] = _scale_vector(
            geometry.pop("center_cm"), field="center_cm"
        )
        try:
            geometry["radius_m"] = float(geometry.pop("radius_cm")) * _CM_TO_M
            geometry["height_m"] = float(geometry.pop("height_cm")) * _CM_TO_M
        except (TypeError, ValueError) as exc:
            raise InputValidationError(
                "radius_cm and height_cm must be numeric"
            ) from exc
    elif shape == "extruded_polygon":
        if "vertices_cm" not in geometry or "extrusion_distance_cm" not in geometry:
            raise InputValidationError(
                "extruded polygon in centimetres requires vertices_cm and extrusion_distance_cm"
            )
        vertices = geometry.pop("vertices_cm")
        if not isinstance(vertices, list):
            raise InputValidationError("vertices_cm must be an array")
        geometry["vertices_m"] = [
            _scale_vector(vertex, field="vertices_cm") for vertex in vertices
        ]
        try:
            geometry["extrusion_distance_m"] = (
                float(geometry.pop("extrusion_distance_cm")) * _CM_TO_M
            )
        except (TypeError, ValueError) as exc:
            raise InputValidationError(
                "extrusion_distance_cm must be numeric"
            ) from exc
    _normalize_dimensionless_geometry(geometry)


def normalize_target(target: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical SI representation of a target JSON object.

    Target files may use explicit ``*_m`` geometry fields or explicit ``*_cm``
    fields.  The two forms may not be mixed.  Response parameters retain their
    own unit metadata and are converted later by the parameter layer.
    """

    if not isinstance(target, dict):
        raise InputValidationError("target must be a JSON object")
    normalized = deepcopy(target)
    frame = normalized.get("coordinate_frame")
    if not isinstance(frame, dict):
        return normalized
    frame_unit = frame.get("length_unit")
    if frame_unit not in {"m", "cm"}:
        raise InputValidationError(
            "coordinate_frame.length_unit must be 'm' or 'cm'",
            context={"length_unit": frame_unit},
        )
    for collection_name in ("armor", "components"):
        collection = normalized.get(collection_name)
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict) or not isinstance(item.get("geometry"), dict):
                continue
            _normalize_geometry(item["geometry"], frame_unit=frame_unit)
    frame["length_unit"] = "m"
    return normalized
