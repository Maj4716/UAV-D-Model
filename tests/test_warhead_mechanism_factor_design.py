import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "run_warhead_mechanism_factor_design.py"
SPEC = importlib.util.spec_from_file_location("warhead_factor_design", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

CONFIG = ROOT / "configs" / "uncertainty" / "warhead_mechanism_factor_design.v1.json"
PHYSICS = ROOT / "configs" / "physics" / "loitering_impact.taylor_shapiro.v2.json"
CASES = (
    ROOT
    / "artifacts"
    / "datasets"
    / "idea2_taylor_shapiro_continuous_r32_20261120_v3"
    / "rounds"
    / "round-000"
    / "formal_cases.jsonl.gz"
)


def _first_non_test_case():
    with gzip.open(CASES, "rt", encoding="utf-8") as stream:
        for line in stream:
            case = json.loads(line)
            role = case.get("metadata", {}).get("sampling", {}).get("role")
            if role in {"reference_train", "reference_validation"}:
                return case
    raise AssertionError("no non-test case")


def test_screening_design_is_reproducible_and_identifiable():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    first = MODULE.generate_screening_design(config)
    second = MODULE.generate_screening_design(config)

    assert first["samples"].shape == (32, 5)
    assert np.array_equal(first["samples"], second["samples"])
    quality = MODULE.design_matrix_quality(first["samples"], MODULE.FACTOR_ORDER)
    assert quality["maximum_vif"] < 2.0
    assert quality["maximum_absolute_off_diagonal_correlation"] < 0.35


def test_mapping_is_detached_and_preserves_mechanism_identities():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    registry = json.loads(PHYSICS.read_text(encoding="utf-8"))
    case = _first_non_test_case()
    original = json.loads(json.dumps(case))
    factors = {
        "V0_scale": 1.05,
        "fragment_mass_scale": 0.95,
        "fragment_count_scale": 1.10,
        "taylor_bound_scale": 0.90,
        "spread_sigma_deg": 3.0,
    }

    mapped = MODULE.map_mechanism_point(case, registry, factors, config)
    report = mapped["mapping"]

    assert case == original
    assert report["admitted"] is True
    raw = report["mapped_raw_parameters"]
    mechanism = report["mapped_mechanism"]
    assert mechanism["fragment_count"] % raw["ring_count"] == 0
    assert raw["Taylor_q"] == pytest.approx(
        np.sin(mechanism["taylor_bound_rad"]), abs=1.0e-12
    )
    assert mapped["case"]["warhead"]["fragment_source"]["gurney_constant_mps"] > 0.0


def test_mapping_rejection_ledger_records_infeasible_mass_and_packing():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    registry = json.loads(PHYSICS.read_text(encoding="utf-8"))
    case = _first_non_test_case()
    mapped = MODULE.map_mechanism_point(
        case,
        registry,
        {
            "V0_scale": 1.0,
            "fragment_mass_scale": 10.0,
            "fragment_count_scale": 10.0,
            "taylor_bound_scale": 1.0,
            "spread_sigma_deg": 2.0,
        },
        config,
    )

    assert mapped["mapping"]["admitted"] is False
    assert "fragment_mass_budget_exceeded" in mapped["mapping"]["rejection_reasons"]


def test_saltelli_extension_has_A_B_and_five_mixed_blocks():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    design = MODULE.generate_saltelli_extension(config, base_size=4)

    assert design["samples"].shape == (28, 5)
    assert design["block_order"] == [
        "A",
        "B",
        "AB_V0_scale",
        "AB_fragment_mass_scale",
        "AB_fragment_count_scale",
        "AB_taylor_bound_scale",
        "AB_spread_sigma_deg",
    ]


def test_temporary_profile_transaction_is_explicit_and_restored():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    registry = json.loads(PHYSICS.read_text(encoding="utf-8"))
    case = _first_non_test_case()
    mapped = MODULE.map_mechanism_point(
        case,
        registry,
        {
            "V0_scale": 1.05,
            "fragment_mass_scale": 0.95,
            "fragment_count_scale": 1.10,
            "taylor_bound_scale": 0.90,
            "spread_sigma_deg": 3.0,
        },
        config,
    )
    profile_id = case["munition"]["profile_id"]
    original = MODULE.munition_profile_module._PROFILES[profile_id]

    with MODULE._temporary_profile_transaction(mapped["case"], mapped["mapping"]):
        active = MODULE.munition_profile_module._PROFILES[profile_id]
        assert active is not original
        assert mapped["case"]["metadata"]["warhead_mechanism_factor_variant"][
            "dataset_admission"
        ] is False

    assert MODULE.munition_profile_module._PROFILES[profile_id] is original
