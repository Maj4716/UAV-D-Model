from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from damage_engine.engine import DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.raw_import import _euler_xyz_to_quaternion, import_raw_target
from damage_engine.registry import PhysicsRegistry
from damage_engine.utils import load_json
from damage_engine.validation import validate_inputs


ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    return (
        ROOT / "raw_data" / "List.csv",
        ROOT / "raw_data" / "armor.csv",
        load_json(ROOT / "raw_data" / "import_config.json"),
    )


def test_raw_import_reconstructs_supplied_geometry_and_retains_issues():
    list_path, armor_path, config = _inputs()
    target, diagnostics = import_raw_target(list_path, armor_path, config)

    assert len(target["components"]) == 67
    assert len(target["armor"]) == 2
    assert diagnostics["component_rows"] == 68
    assert diagnostics["components_skipped"] == [
        {
            "csv_row": 58,
            "serial": "57",
            "name": "无线电台",
            "reason": "raw component is missing its 3D position",
            "context": {
                "row": 58,
                "serial": "57",
                "name": "无线电台",
                "missing": ["X坐标/cm", "Y坐标/cm", "Z坐标/cm"],
            },
        }
    ]
    assert diagnostics["damage_association_count"] == 57
    assert diagnostics["damage_associations_applied"] == 56
    assert diagnostics["damage_associations_metadata_only"] == 1
    assert diagnostics["damage_codes"] == ["M2", "K1", "M1", "K2", "F1", "C1", "C2"]

    first = target["components"][0]
    assert first["geometry"]["shape"] == "box"
    assert first["geometry"]["size_lwh_cm"] == [56.619, 45.209, 145.586]
    wheel = next(item for item in target["components"] if item["id"] == "component-004")
    assert wheel["geometry"]["shape"] == "cylinder"
    assert wheel["geometry"]["radius_cm"] == 32.0
    assert wheel["geometry"]["height_cm"] == 34.4

    np.testing.assert_allclose(
        [item["geometry"]["extrusion_distance_cm"] for item in target["armor"]],
        [326.05299949645996, 237.4959945678711],
        rtol=0.0,
        atol=1.0e-12,
    )
    for item in target["armor"]:
        vertices = np.asarray(item["geometry"]["vertices_cm"])
        assert np.max(np.abs(vertices[:, 0] - vertices[0, 0])) == 0.0

    rules = target["metadata"]["raw_damage_tree_records"]
    driver = [item for item in rules if item["component_id"] == "component-058"]
    assert [(item["damage_type"], item["slot"], item["threshold"]) for item in driver] == [
        ("M2", 1, 1.0),
        ("C1", 2, 0.2),
        ("C2", 3, 0.6),
    ]
    incomplete = next(item for item in rules if item["component_id"] == "component-050")
    assert incomplete["damage_type"] == "F1"
    assert incomplete["threshold"] is None
    assert incomplete["apply_to_provisional_tree"] is False


def test_raw_import_is_deterministic_and_validates_against_registry():
    list_path, armor_path, config = _inputs()
    first, _ = import_raw_target(list_path, armor_path, config)
    second, _ = import_raw_target(list_path, armor_path, deepcopy(config))
    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(
        second, ensure_ascii=False, sort_keys=True
    )

    registry = PhysicsRegistry.from_file(ROOT / "examples" / "physics.demo.json")
    engine = DamageEngine(first, registry)
    case = load_json(ROOT / "examples" / "case.demo.json")
    case["target_ref"] = {
        "target_id": first["target_id"],
        "target_revision": first["target_revision"],
        "content_hash": engine.target_hash,
    }
    case["analysis"]["evaluation_standard_ids"] = ["raw-damage-codes-provisional"]
    validate_inputs(engine.target_data, case, registry)
    assert len(engine.compiled_target.components) == 67
    assert len(engine.damage_tree.outputs) == 7


def test_raw_import_strict_mode_rejects_missing_component_position():
    list_path, armor_path, config = _inputs()
    with pytest.raises(InputValidationError, match="missing its 3D position"):
        import_raw_target(
            list_path,
            armor_path,
            config,
            skip_incomplete_components=False,
        )


def test_raw_import_requires_explicit_armor_response_profile():
    list_path, armor_path, config = _inputs()
    config = deepcopy(config)
    del config["armor_defaults"]["equivalent_thickness"]
    with pytest.raises(InputValidationError, match="missing a required field"):
        import_raw_target(list_path, armor_path, config)


def test_raw_euler_conversion_uses_xyzw_and_identity_for_blank():
    assert _euler_xyz_to_quaternion("", row_number=1) == [0.0, 0.0, 0.0, 1.0]
    np.testing.assert_allclose(
        _euler_xyz_to_quaternion("[90,0,0]", row_number=1),
        [2**-0.5, 0.0, 0.0, 2**-0.5],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        _euler_xyz_to_quaternion("[0,90,0]", row_number=1),
        [0.0, 2**-0.5, 0.0, 2**-0.5],
        atol=1.0e-12,
    )

