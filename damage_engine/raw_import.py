from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .errors import InputValidationError
from .utils import content_hash, file_hash


_LIST_MINIMUM_COLUMNS = 22
_DAMAGE_SLOT_OFFSETS = (13, 16, 19)
_ID_SANITIZER = re.compile(r"[^A-Za-z0-9_-]+")


def _read_csv_rows(path: str | Path) -> list[list[str]]:
    value = Path(path)
    try:
        with value.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.reader(stream))
    except UnicodeDecodeError as exc:
        raise InputValidationError(
            "raw CSV must be UTF-8 (an optional BOM is accepted)",
            context={"path": str(value)},
        ) from exc


def _require_config(config: dict, path: str) -> Any:
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise InputValidationError(
                "raw import configuration is missing a required field",
                context={"field": path},
            )
        current = current[key]
    return current


def _slug(value: str, *, fallback: str) -> str:
    cleaned = _ID_SANITIZER.sub("-", value.strip()).strip("-").lower()
    return cleaned or fallback


def _parse_float(value: str, *, field: str, row_number: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(
            "raw component field must be numeric",
            context={"row": row_number, "field": field, "value": value},
        ) from exc
    if not math.isfinite(number):
        raise InputValidationError(
            "raw component field must be finite",
            context={"row": row_number, "field": field, "value": value},
        )
    return number


def _parse_ratio(value: str, *, row_number: int) -> float:
    text = value.strip()
    if text.endswith("%"):
        ratio = _parse_float(text[:-1], field="易损面积比例", row_number=row_number) / 100.0
    else:
        ratio = _parse_float(text, field="易损面积比例", row_number=row_number)
        if ratio > 1.0:
            ratio /= 100.0
    if not 0.0 <= ratio <= 1.0:
        raise InputValidationError(
            "raw vulnerable-area ratio must be within [0, 1] or [0%, 100%]",
            context={"row": row_number, "value": value},
        )
    return ratio


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _euler_xyz_to_quaternion(value: str, *, row_number: int) -> list[float]:
    if not value.strip():
        return [0.0, 0.0, 0.0, 1.0]
    try:
        angles = json.loads(value)
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            "Rotation must be a JSON array [x_deg, y_deg, z_deg]",
            context={"row": row_number, "value": value},
        ) from exc
    if not isinstance(angles, list) or len(angles) != 3:
        raise InputValidationError(
            "Rotation must contain exactly three Euler angles",
            context={"row": row_number, "value": value},
        )
    numeric = np.asarray(angles, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise InputValidationError(
            "Rotation Euler angles must be finite",
            context={"row": row_number, "value": value},
        )
    total = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    for axis, angle_degrees in zip("xyz", numeric):
        half = math.radians(float(angle_degrees)) / 2.0
        sine, cosine = math.sin(half), math.cos(half)
        axis_quaternion = {
            "x": np.asarray([sine, 0.0, 0.0, cosine]),
            "y": np.asarray([0.0, sine, 0.0, cosine]),
            "z": np.asarray([0.0, 0.0, sine, cosine]),
        }[axis]
        # Extrinsic XYZ rotations for column vectors: q = qz * qy * qx.
        total = _quaternion_multiply(axis_quaternion, total)
    total /= np.linalg.norm(total)
    if total[3] < 0.0:
        total = -total
    return [float(item) for item in total]


def _parameter(
    value: float,
    unit: str,
    *,
    source_id: str,
    validation_status: str = "unverified",
    valid_range: list[float] | tuple[float, float] | None = None,
) -> dict:
    limits = list(valid_range) if valid_range is not None else [float(value), float(value)]
    return {
        "value": float(value),
        "unit": unit,
        "source_id": source_id,
        "calibration_id": None,
        "validation_status": validation_status,
        "valid_range": [float(limits[0]), float(limits[1])],
        "uncertainty": {"kind": "none", "parameters": {}, "epistemic": True},
    }


def _parameter_from_config(specification: dict, *, fallback_source_id: str) -> dict:
    if not isinstance(specification, dict):
        raise InputValidationError("raw response parameter configuration must be an object")
    missing = [field for field in ("value", "unit") if field not in specification]
    if missing:
        raise InputValidationError(
            "raw response parameter configuration is incomplete",
            context={"missing": missing},
        )
    return _parameter(
        float(specification["value"]),
        str(specification["unit"]),
        source_id=str(specification.get("source_id", fallback_source_id)),
        validation_status=str(specification.get("validation_status", "unverified")),
        valid_range=specification.get("valid_range"),
    )


def _blast_response_from_config(
    defaults: dict,
    *,
    source_id: str,
    raw_peak_threshold: float | None = None,
    raw_peak_unit: str = "MPa",
) -> dict:
    response = {
        "model_id": str(_require_config(defaults, "blast_response_model_id")),
        "pressure_impulse_surface_id": defaults.get("pressure_impulse_surface_id"),
    }
    configured = {
        "peak_overpressure_threshold": defaults.get("peak_overpressure_threshold"),
        "impulse_threshold": defaults.get("impulse_threshold"),
        "duration_threshold": defaults.get("duration_threshold"),
    }
    if raw_peak_threshold is not None:
        configured["peak_overpressure_threshold"] = {
            "value": raw_peak_threshold,
            "unit": raw_peak_unit,
            "source_id": source_id,
            "validation_status": "unverified",
            "valid_range": [raw_peak_threshold, raw_peak_threshold],
        }
    for field, specification in configured.items():
        if specification is not None:
            response[field] = _parameter_from_config(
                specification,
                fallback_source_id=source_id,
            )
    return response


def _parse_damage_slots(
    row: list[str],
    *,
    row_number: int,
    reject_incomplete: bool,
) -> list[dict]:
    output = []
    for slot_index, offset in enumerate(_DAMAGE_SLOT_OFFSETS, start=1):
        damage_type, gate, threshold = (row[offset + index].strip() for index in range(3))
        if not any((damage_type, gate, threshold)):
            continue
        if not damage_type or not gate:
            raise InputValidationError(
                "a raw damage-tree slot must contain both damage type and gate",
                context={"row": row_number, "slot": slot_index, "values": row[offset : offset + 3]},
            )
        if gate.upper() != "OR":
            raise InputValidationError(
                "the current raw importer supports the observed OR gate only",
                context={"row": row_number, "slot": slot_index, "gate": gate},
            )
        numeric_threshold = None
        if threshold:
            numeric_threshold = _parse_float(
                threshold,
                field=f"阈值{slot_index}",
                row_number=row_number,
            )
        elif reject_incomplete:
            raise InputValidationError(
                "a raw damage-tree slot is missing its threshold",
                context={"row": row_number, "slot": slot_index, "damage_type": damage_type},
            )
        output.append(
            {
                "damage_type": damage_type,
                "gate": gate.upper(),
                "threshold": numeric_threshold,
                "slot": slot_index,
                "apply_to_provisional_tree": numeric_threshold is not None,
            }
        )
    return output


def _component_from_row(
    row: list[str],
    *,
    csv_row_number: int,
    config: dict,
) -> tuple[dict, list[dict]]:
    if len(row) < _LIST_MINIMUM_COLUMNS:
        raise InputValidationError(
            "raw component row has too few columns",
            context={"row": csv_row_number, "observed": len(row), "required": _LIST_MINIMUM_COLUMNS},
        )
    serial = row[0].strip()
    name = row[1].strip()
    shape = row[2].strip()
    if not serial or not name:
        raise InputValidationError(
            "raw component row requires 序号 and 等效结构名称",
            context={"row": csv_row_number},
        )
    missing_coordinates = [
        field
        for field, value in zip(("X坐标/cm", "Y坐标/cm", "Z坐标/cm"), row[6:9])
        if not value.strip()
    ]
    if missing_coordinates:
        raise InputValidationError(
            "raw component is missing its 3D position",
            context={"row": csv_row_number, "serial": serial, "name": name, "missing": missing_coordinates},
        )
    center = [
        _parse_float(row[index], field=field, row_number=csv_row_number)
        for index, field in zip((6, 7, 8), ("X坐标/cm", "Y坐标/cm", "Z坐标/cm"))
    ]
    orientation = _euler_xyz_to_quaternion(row[9], row_number=csv_row_number)
    length_or_radius = _parse_float(row[3], field="L(R)/cm", row_number=csv_row_number)
    height = _parse_float(row[4], field="H/cm", row_number=csv_row_number)
    if shape == "长方体":
        width = _parse_float(row[5], field="W/cm", row_number=csv_row_number)
        geometry = {
            "shape": "box",
            "center_cm": center,
            # Source order is L, H, W; the engine contract is L, W, H.
            "size_lwh_cm": [length_or_radius, width, height],
            "orientation_xyzw": orientation,
        }
    elif shape == "圆柱体":
        geometry = {
            "shape": "cylinder",
            "center_cm": center,
            "radius_cm": length_or_radius,
            "height_cm": height,
            "orientation_xyzw": orientation,
        }
    else:
        raise InputValidationError(
            "unsupported raw component shape",
            context={"row": csv_row_number, "shape": shape},
        )

    thickness = _parse_float(row[10], field="等效厚度/mm", row_number=csv_row_number)
    vulnerable_ratio = _parse_ratio(row[11], row_number=csv_row_number)
    peak_threshold = (
        _parse_float(row[12], field="超压阈值", row_number=csv_row_number)
        if row[12].strip()
        else None
    )
    defaults = _require_config(config, "component_defaults")
    source_id = f"raw_data/List.csv:row:{csv_row_number}"
    damage_slots = _parse_damage_slots(
        row,
        row_number=csv_row_number,
        reject_incomplete=bool(
            config.get("import_policy", {}).get("reject_incomplete_damage_rules", False)
        ),
    )
    component_id = f"component-{int(serial):03d}" if serial.isdigit() else f"component-{_slug(serial, fallback=str(csv_row_number))}"
    component = {
        "id": component_id,
        "name": name,
        "component_type": str(defaults.get("component_type", "raw_equivalent_structure")),
        "functional_tags": [
            f"damage-code:{item['damage_type']}" for item in damage_slots
        ] or ["damage-code:unassigned"],
        "geometry": geometry,
        "response": {
            "material_id": str(_require_config(defaults, "material_id")),
            "layer_stack_id": defaults.get("layer_stack_id"),
            "equivalent_thickness": _parameter(
                thickness,
                "mm",
                source_id=source_id,
            ),
            "vulnerable_area_ratio": _parameter(
                vulnerable_ratio,
                "1",
                source_id=source_id,
                valid_range=[0.0, 1.0],
            ),
            "fragment_response_model_id": str(
                _require_config(defaults, "fragment_response_model_id")
            ),
            "blast_response": _blast_response_from_config(
                defaults,
                source_id=source_id,
                raw_peak_threshold=peak_threshold,
                raw_peak_unit=str(config.get("raw_overpressure_unit", "MPa")),
            ),
        },
        "compartment_id": defaults.get("compartment_id"),
        "exposure_class": str(defaults.get("exposure_class", "internal")),
        "damage_states": list(defaults.get("damage_states", ["intact", "degraded", "failed"])),
        "provenance": {
            "source_id": source_id,
            "status": str(defaults.get("provenance_status", "unverified")),
            "reviewer": None,
            "reviewed_at": None,
        },
    }
    for item in damage_slots:
        item.update(
            {
                "component_id": component_id,
                "component_name": name,
                "source_id": source_id,
            }
        )
    return component, damage_slots


def _match_cross_sections(
    first: list[dict],
    second: list[dict],
    *,
    tolerance_cm: float,
    object_name: str,
) -> float:
    unmatched = [np.asarray([item["y"], item["z"]], dtype=np.float64) for item in second]
    maximum_error = 0.0
    for item in first:
        point = np.asarray([item["y"], item["z"]], dtype=np.float64)
        distances = [float(np.linalg.norm(point - candidate)) for candidate in unmatched]
        if not distances:
            raise InputValidationError(
                "armor cross-sections contain different vertex counts",
                context={"object": object_name},
            )
        index = int(np.argmin(distances))
        maximum_error = max(maximum_error, distances[index])
        unmatched.pop(index)
    if unmatched or maximum_error > tolerance_cm:
        raise InputValidationError(
            "armor cross-sections are not related by a pure X extrusion",
            context={
                "object": object_name,
                "maximum_yz_error_cm": maximum_error,
                "tolerance_cm": tolerance_cm,
            },
        )
    return maximum_error


def _armor_from_rows(
    name: str,
    rows: list[dict],
    *,
    armor_index: int,
    config: dict,
) -> tuple[dict, dict]:
    tolerance_cm = float(config.get("armor_cross_section_tolerance_cm", 1.0e-3))
    plane_groups: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        plane_groups[float(row["plane_label_x"])].append(row)
    if len(plane_groups) != 2:
        raise InputValidationError(
            "each armor object must contain exactly two X cross-sections",
            context={"object": name, "planes": sorted(plane_groups)},
        )
    ordered_planes = sorted(plane_groups.items())
    first = sorted(ordered_planes[0][1], key=lambda item: item["order"])
    second = sorted(ordered_planes[1][1], key=lambda item: item["order"])
    if len(first) < 3 or len(first) != len(second):
        raise InputValidationError(
            "armor cross-sections must contain matching polygons with at least three vertices",
            context={"object": name, "counts": [len(first), len(second)]},
        )
    matching_error = _match_cross_sections(
        first,
        second,
        tolerance_cm=tolerance_cm,
        object_name=name,
    )
    first_x = float(np.mean([item["x"] for item in first]))
    second_x = float(np.mean([item["x"] for item in second]))
    plane_residual = max(
        max(abs(item["x"] - first_x) for item in first),
        max(abs(item["x"] - second_x) for item in second),
    )
    distance = second_x - first_x
    if distance <= 0.0:
        raise InputValidationError(
            "armor extrusion distance must be positive",
            context={"object": name, "distance_cm": distance},
        )
    defaults = _require_config(config, "armor_defaults")
    source_id = f"raw_data/armor.csv:object:{name}"
    response = {
        "material_id": str(_require_config(defaults, "material_id")),
        "layer_stack_id": defaults.get("layer_stack_id"),
        "equivalent_thickness": _parameter_from_config(
            _require_config(defaults, "equivalent_thickness"),
            fallback_source_id=source_id,
        ),
        "vulnerable_area_ratio": _parameter_from_config(
            _require_config(defaults, "vulnerable_area_ratio"),
            fallback_source_id=source_id,
        ),
        "fragment_response_model_id": str(
            _require_config(defaults, "fragment_response_model_id")
        ),
        "blast_response": _blast_response_from_config(defaults, source_id=source_id),
    }
    armor = {
        "id": f"armor-{armor_index:03d}",
        "name": name,
        "role": "armor",
        "geometry": {
            "shape": "extruded_polygon",
            # Project noisy exported X values onto the declared base plane.
            "vertices_cm": [[first_x, item["y"], item["z"]] for item in first],
            "extrusion_direction": [1.0, 0.0, 0.0],
            "extrusion_distance_cm": distance,
        },
        "response": response,
        "boundary_condition": str(defaults.get("boundary_condition", "raw-unspecified")),
        "provenance": {
            "source_id": source_id,
            "status": str(defaults.get("provenance_status", "unverified")),
            "reviewer": None,
            "reviewed_at": None,
        },
    }
    diagnostics = {
        "armor_id": armor["id"],
        "name": name,
        "vertex_count": len(first),
        "extrusion_distance_cm": distance,
        "maximum_cross_section_yz_error_cm": matching_error,
        "maximum_plane_projection_residual_cm": plane_residual,
    }
    return armor, diagnostics


def _damage_tree(
    associations: list[dict],
    *,
    config: dict,
) -> tuple[dict, list[dict]]:
    defaults = _require_config(config, "damage_tree")
    version = str(_require_config(defaults, "version"))
    standard_id = str(defaults.get("standard_id", version))
    dependency_model_id = str(defaults.get("dependency_model_id", "fully_dependent"))
    damage_state = str(defaults.get("damage_state", "failed"))
    threshold_state_map = {
        str(key): str(value)
        for key, value in defaults.get("threshold_state_map", {}).items()
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    retained_records = []
    for item in associations:
        retained = dict(item)
        threshold = retained.get("threshold")
        threshold_key = None if threshold is None else format(float(threshold), "g")
        retained["resolved_damage_state"] = (
            threshold_state_map.get(threshold_key, damage_state)
            if threshold_key is not None
            else None
        )
        retained_records.append(retained)
        if not retained.get("apply_to_provisional_tree", False):
            continue
        damage_type = str(retained["damage_type"])
        if damage_type not in grouped:
            order.append(damage_type)
        grouped[damage_type].append(retained)
    if not order:
        raise InputValidationError("raw component table does not define any damage-tree association")

    nodes: list[dict] = []
    output_ids: list[str] = []
    leaf_by_event: dict[tuple[str, str], str] = {}
    for damage_type in order:
        for item in grouped[damage_type]:
            event = (str(item["component_id"]), str(item["resolved_damage_state"]))
            if event in leaf_by_event:
                continue
            leaf_id = (
                f"leaf-{_slug(event[0], fallback='component')}-"
                f"{_slug(event[1], fallback='state')}"
            )
            leaf_by_event[event] = leaf_id
            threshold_text = str(item["threshold"])
            nodes.append(
                {
                    "id": leaf_id,
                    "node_type": "leaf",
                    "inputs": [],
                    "component_id": event[0],
                    "mechanism": "combined",
                    "damage_state": event[1],
                    "k": None,
                    "dependency_model_id": None,
                    "threshold_source_id": item["source_id"],
                    "description": (
                        f"Imported component event for {item['component_name']}; "
                        f"raw threshold={threshold_text} mapped to state={event[1]} by the "
                        "explicit import profile."
                    ),
                }
            )
    for damage_type in order:
        group = grouped[damage_type]
        leaf_ids = list(
            dict.fromkeys(
                leaf_by_event[(str(item["component_id"]), str(item["resolved_damage_state"]))]
                for item in group
            )
        )
        output_id = f"effect-{_slug(damage_type, fallback='unknown')}"
        output_ids.append(output_id)
        if len(leaf_ids) == 1:
            nodes.append(
                {
                    "id": output_id,
                    "node_type": "standard_projection",
                    "inputs": leaf_ids,
                    "component_id": None,
                    "mechanism": "not_applicable",
                    "damage_state": None,
                    "k": None,
                    "dependency_model_id": "independent",
                    "threshold_source_id": group[0]["source_id"],
                    "description": f"Imported raw damage code {damage_type}",
                }
            )
        else:
            nodes.append(
                {
                    "id": output_id,
                    "node_type": "or",
                    "inputs": leaf_ids,
                    "component_id": None,
                    "mechanism": "not_applicable",
                    "damage_state": None,
                    "k": None,
                    "dependency_model_id": dependency_model_id,
                    "threshold_source_id": "raw_data/List.csv",
                    "description": (
                        f"Imported OR aggregation for raw damage code {damage_type}; "
                        f"probability dependence model={dependency_model_id}."
                    ),
                }
            )
    return (
        {
            "version": version,
            "nodes": nodes,
            "outputs": output_ids,
            "standards": [
                {
                    "id": standard_id,
                    "name": str(defaults.get("standard_name", "Raw CSV damage codes")),
                    "description": str(
                        defaults.get(
                            "standard_description",
                            "Imported from List.csv; raw numeric threshold semantics require confirmation.",
                        )
                    ),
                    "outputs": output_ids,
                }
            ],
        },
        retained_records,
    )


def import_raw_target(
    list_csv: str | Path,
    armor_csv: str | Path,
    config: dict,
    *,
    skip_incomplete_components: bool | None = None,
) -> tuple[dict, dict]:
    """Convert the observed ``List.csv``/``armor.csv`` export into target schema 1.0.

    The converter is deliberately positional for the three damage-tree slots,
    because the supplied component file repeats the ``毁伤类型2/逻辑门2/阈值2``
    header for the third slot.  Missing physics are supplied only through an
    explicit configuration object and remain marked unverified.
    """

    list_path = Path(list_csv)
    armor_path = Path(armor_csv)
    list_rows = _read_csv_rows(list_path)
    armor_rows = _read_csv_rows(armor_path)
    if not list_rows or len(list_rows[0]) < _LIST_MINIMUM_COLUMNS:
        raise InputValidationError("List.csv has an unsupported header")
    expected_armor_header = ["对象", "平面X", "顺序", "顶点索引", "X坐标", "Y坐标", "Z坐标"]
    if not armor_rows or armor_rows[0] != expected_armor_header:
        raise InputValidationError(
            "armor.csv has an unsupported header",
            context={"expected": expected_armor_header, "observed": armor_rows[0] if armor_rows else []},
        )
    if skip_incomplete_components is None:
        skip_incomplete_components = bool(
            config.get("import_policy", {}).get("skip_incomplete_components", False)
        )

    components = []
    associations = []
    skipped_components = []
    for csv_row_number, row in enumerate(list_rows[1:], start=2):
        if not any(value.strip() for value in row):
            continue
        try:
            component, damage_slots = _component_from_row(
                row,
                csv_row_number=csv_row_number,
                config=config,
            )
        except InputValidationError as exc:
            if not skip_incomplete_components:
                raise
            skipped_components.append(
                {
                    "csv_row": csv_row_number,
                    "serial": row[0].strip() if row else "",
                    "name": row[1].strip() if len(row) > 1 else "",
                    "reason": exc.message,
                    "context": exc.context,
                }
            )
            continue
        components.append(component)
        associations.extend(damage_slots)

    parsed_armor_rows: dict[str, list[dict]] = defaultdict(list)
    armor_order: list[str] = []
    for csv_row_number, row in enumerate(armor_rows[1:], start=2):
        if not any(value.strip() for value in row):
            continue
        if len(row) != len(expected_armor_header):
            raise InputValidationError(
                "armor row has an unexpected column count",
                context={"row": csv_row_number, "observed": len(row)},
            )
        name = row[0].strip()
        if name not in parsed_armor_rows:
            armor_order.append(name)
        values = [
            _parse_float(row[index], field=expected_armor_header[index], row_number=csv_row_number)
            for index in (1, 4, 5, 6)
        ]
        try:
            order = int(row[2])
            vertex_index = int(row[3])
        except ValueError as exc:
            raise InputValidationError(
                "armor order and vertex index must be integers",
                context={"row": csv_row_number, "values": row[2:4]},
            ) from exc
        parsed_armor_rows[name].append(
            {
                "plane_label_x": values[0],
                "order": order,
                "vertex_index": vertex_index,
                "x": values[1],
                "y": values[2],
                "z": values[3],
                "csv_row": csv_row_number,
            }
        )

    armor = []
    armor_diagnostics = []
    for armor_index, name in enumerate(armor_order, start=1):
        item, item_diagnostics = _armor_from_rows(
            name,
            parsed_armor_rows[name],
            armor_index=armor_index,
            config=config,
        )
        armor.append(item)
        armor_diagnostics.append(item_diagnostics)

    damage_tree, retained_records = _damage_tree(associations, config=config)
    coordinate_frame = dict(_require_config(config, "coordinate_frame"))
    coordinate_frame["length_unit"] = "cm"
    target = {
        "schema_version": "1.0.0",
        "target_id": str(_require_config(config, "target_id")),
        "target_revision": str(_require_config(config, "target_revision")),
        "coordinate_frame": coordinate_frame,
        "armor": armor,
        "components": components,
        "damage_tree": damage_tree,
        "metadata": {
            "raw_import": {
                "list_csv_sha256": file_hash(list_path),
                "armor_csv_sha256": file_hash(armor_path),
                "component_rows": len(list_rows) - 1,
                "components_imported": len(components),
                "components_skipped": len(skipped_components),
                "armor_objects_imported": len(armor),
                "rotation_convention": "extrinsic XYZ degrees, converted to xyzw quaternion",
                "raw_overpressure_unit": str(config.get("raw_overpressure_unit", "MPa")),
                "damage_threshold_semantics": (
                    "profile_state_mapping_for_complete_rules; incomplete_rules_metadata_only"
                ),
                "physical_response_status": "unverified_configuration",
                "import_config_hash": content_hash(config),
                "damage_tree_status": "provisional_requires_codebook_confirmation",
                "damage_threshold_state_map": dict(
                    _require_config(config, "damage_tree").get("threshold_state_map", {})
                ),
            },
            "raw_damage_tree_records": retained_records,
            "skipped_components": skipped_components,
        },
    }
    diagnostics = {
        "status": "ok_with_warnings" if skipped_components else "ok",
        "list_csv": str(list_path),
        "armor_csv": str(armor_path),
        "component_rows": len(list_rows) - 1,
        "components_imported": len(components),
        "components_skipped": skipped_components,
        "armor_rows": len(armor_rows) - 1,
        "armor_objects": armor_diagnostics,
        "damage_association_count": len(retained_records),
        "damage_associations_applied": sum(
            bool(item.get("apply_to_provisional_tree")) for item in retained_records
        ),
        "damage_associations_metadata_only": sum(
            not bool(item.get("apply_to_provisional_tree")) for item in retained_records
        ),
        "damage_codes": list(
            dict.fromkeys(str(item["damage_type"]) for item in retained_records)
        ),
        "warnings": [
            "Armor response parameters come from the explicit import config, not armor.csv.",
            "Complete raw damage thresholds use the explicit profile state mapping; incomplete rules remain metadata-only.",
            "All imported response/provenance values remain unverified until approved physical evidence is supplied.",
        ],
    }
    if skipped_components:
        diagnostics["warnings"].append(
            "Components with incomplete geometry were skipped; fill their coordinates and rerun in strict mode."
        )
    return target, diagnostics

