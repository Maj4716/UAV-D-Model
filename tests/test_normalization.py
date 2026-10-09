from __future__ import annotations

from copy import deepcopy
import json

import pytest

from damage_engine.engine import DamageEngine
from damage_engine.cli import main
from damage_engine.errors import InputValidationError
from damage_engine.normalization import normalize_target
from damage_engine.utils import content_hash
from damage_engine.validation import validate_inputs


def _as_centimetres(target: dict) -> dict:
    converted = deepcopy(target)
    converted["coordinate_frame"]["length_unit"] = "cm"
    for item in [*converted["armor"], *converted["components"]]:
        geometry = item["geometry"]
        shape = geometry["shape"]
        if shape == "box":
            geometry["center_cm"] = [value * 100.0 for value in geometry.pop("center_m")]
            geometry["size_lwh_cm"] = [
                value * 100.0 for value in geometry.pop("size_lwh_m")
            ]
        elif shape == "cylinder":
            geometry["center_cm"] = [value * 100.0 for value in geometry.pop("center_m")]
            geometry["radius_cm"] = geometry.pop("radius_m") * 100.0
            geometry["height_cm"] = geometry.pop("height_m") * 100.0
        elif shape == "extruded_polygon":
            geometry["vertices_cm"] = [
                [value * 100.0 for value in vertex]
                for vertex in geometry.pop("vertices_m")
            ]
            geometry["extrusion_distance_cm"] = (
                geometry.pop("extrusion_distance_m") * 100.0
            )
    return converted


def test_centimetre_target_normalizes_to_canonical_si(target_data):
    assert normalize_target(_as_centimetres(target_data)) == target_data


def test_normalization_does_not_mutate_input(target_data):
    source = _as_centimetres(target_data)
    snapshot = deepcopy(source)
    normalize_target(source)
    assert source == snapshot


def test_mixed_geometry_units_are_rejected(target_data):
    source = _as_centimetres(target_data)
    source["components"][0]["geometry"]["center_m"] = [0.0, 0.0, 0.0]
    with pytest.raises(InputValidationError, match="cannot be mixed"):
        normalize_target(source)


def test_engine_executes_centimetre_target_with_canonical_hash(
    target_data, case_data, registry
):
    centimetre_target = _as_centimetres(target_data)
    engine = DamageEngine(centimetre_target, registry)
    case = deepcopy(case_data)
    case["target_ref"]["content_hash"] = content_hash(target_data)
    case["analysis"]["replicates"] = 1
    case["analysis"]["adaptive_ci_half_width"] = None

    result = engine.run(case).to_dict()
    assert result["status"] == "ok"
    assert engine.target_hash == content_hash(target_data)
    assert result["provenance"]["input_target_hash"] == content_hash(centimetre_target)


def test_direct_validation_uses_same_canonical_centimetre_hash(
    target_data, case_data, registry
):
    centimetre_target = _as_centimetres(target_data)
    case = deepcopy(case_data)
    case["target_ref"]["content_hash"] = content_hash(target_data)
    validate_inputs(centimetre_target, case, registry)


def test_cli_hash_target_is_identical_for_metre_and_centimetre_files(
    target_data, tmp_path, capsys
):
    metre_path = tmp_path / "target-m.json"
    centimetre_path = tmp_path / "target-cm.json"
    metre_path.write_text(json.dumps(target_data), encoding="utf-8")
    centimetre_path.write_text(
        json.dumps(_as_centimetres(target_data)), encoding="utf-8"
    )

    assert main(["hash-target", "--target", str(metre_path)]) == 0
    metre_hash = capsys.readouterr().out.strip()
    assert main(["hash-target", "--target", str(centimetre_path)]) == 0
    centimetre_hash = capsys.readouterr().out.strip()
    assert metre_hash == centimetre_hash == content_hash(target_data)
