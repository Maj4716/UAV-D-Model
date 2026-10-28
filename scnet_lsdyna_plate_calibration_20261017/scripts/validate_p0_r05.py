from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from validate_package import (
    INTEGER_ITEM,
    NUMERIC_ITEM,
    fail,
    keyword_blocks,
    require_types,
    split_fields,
    validate_keyword_input_format,
    validate_keyword_schema_contracts,
    validate_required_card_layouts,
)


PROJECT = Path(__file__).resolve().parents[1]
SUCCESSOR_REAL_OUTPUT_CONTRACT = PROJECT / "design" / "p0_r06_real_output_contract.json"
EXPECTED_REVISION = "R05_capture_ledger_preregistered"
EXPECTED_DESIGN_STATUS = "frozen_before_r05_solver_outcomes_after_r04_diagnostic"
EXPECTED_QUALIFICATION_GATES = {
    "gate_status": "preregistered_candidate_engineering_thresholds_not_universal_lsdyna_standards",
    "run_integrity": {
        "all_ranks_normal_termination": True,
        "fatal_or_error_count_max": 0,
        "total_energy_ratio_min": 0.95,
        "total_energy_ratio_max": 1.05,
        "absolute_hourglass_to_peak_internal_energy_max": 0.05,
        "added_mass_fraction_max": 0.0,
        "evaluation_window": "all finite records from the first record with a finite nonzero total-energy denominator through normal termination, inclusive; report any excluded initialization records",
        "total_energy_ratio_definition": "GLSTAT total_energy(t) divided by total_energy at the first finite nonzero baseline record; every finite record in the evaluation window must lie within the limits",
        "hourglass_ratio_definition": "maximum over the evaluation window of abs(GLSTAT hourglass_energy(t)) divided by the maximum positive GLSTAT internal_energy over the same window",
        "added_mass_fraction_definition": "maximum nonnegative solver-reported added mass divided by the initial physical model mass",
        "missing_or_nonfinite_policy": "unavailable and fail-closed",
    },
    "capture_ledger": {
        "initial_discrete_mass_relative_error_max": 0.005,
        "mass_accounting_relative_error_max": 0.000001,
        "temporal_ambiguity_mass_fraction_max": 0.01,
        "deleted_upstream_mass_fraction_max": 0.05,
        "tail_window_fraction": 0.10,
        "tail_increment_qM_max": 0.01,
        "tail_increment_qP_max": 0.01,
        "tail_increment_qE_max": 0.01,
        "tail_qV_relative_range_max": 0.05,
        "failure_policy": "mark primary QoI unavailable; never replace unavailable with zero",
    },
    "numerical_perforation_classification": {
        "confirmed_perforation_qM_min": 0.02,
        "confirmed_perforation_qP_min": 0.01,
        "confirmed_nonperforation_qM_max": 0.005,
        "confirmed_nonperforation_qP_max": 0.005,
        "confirmed_nonperforation_worst_case_crossing_mass_max": 0.005,
        "confirmed_nonperforation_worst_case_crossing_mass_definition": "qM + temporal_ambiguity_mass_fraction + deleted_upstream_mass_fraction + max(expected_reference_mass-observed_reference_mass,0)/expected_reference_mass",
        "otherwise": "transition_or_indeterminate",
    },
    "speed_scan": {
        "minimum_confirmed_perforation_points": 1,
        "minimum_confirmed_nonperforation_points": 1,
        "maximum_nearest_bracket_width_mps": 200.0,
        "forbid_perforation_to_nonperforation_reversal": True,
        "confirmed_perforation_qV_downward_jump_mps_max": 100.0,
        "qV_mps_definition": "qV_over_v0 multiplied by that case's impact_speed_mps",
        "qV_downward_jump_comparison": "compare only adjacent cases in the complete speed-sorted scan when both cases are confirmed_perforation; pairs containing transition, indeterminate, unavailable, or confirmed_nonperforation are not silently bridged",
        "reporting_boundary": "deterministic numerical transition bracket only; do not report experimental V50 or confidence intervals",
    },
    "mesh_scan_2000_mps": {
        "all_primary_qois_available": True,
        "classification_must_match": True,
        "metrics": ["qM", "qP", "qE", "qV_over_v0", "deleted_upstream_mass_fraction"],
        "coarse_to_medium_absolute_difference_max": 0.10,
        "medium_to_fine_absolute_difference_max": 0.05,
    },
    "admission": {
        "p0_response_calibration": "allow only if every applicable gate passes",
        "p1_submission": "remain denied until P0 response calibration is allowed",
        "physical_validation": "requires independent experimental evidence; cannot be established by R05",
    },
}
EXPECTED_MEASUREMENT_CONTRACT = {
    "schema_version": "hf-p0-capture-ledger@1.0.0",
    "full_model_d3plot_interval_ms": 0.0005,
    "fragment_d3part_interval_ms": 0.00005,
    "ascii_interval_ms": 0.00005,
    "fragment_part_set_id": 900001,
    "capture_plane": {
        "axis": "z",
        "coordinate_mm": -12.0,
        "undeformed_plate_back_mm": -10.0,
        "offset_behind_plate_mm": 2.0,
        "travel_direction": "negative",
        "element_position": "8-node arithmetic centroid; element contributions are aggregated with initial reference-mass weights",
        "crossing_velocity": "linear interpolation between bracketing d3part states",
        "crossing_policy": "first negative-z crossing only",
    },
    "reference_mass_basis": "initial fragment solid-element reference volumes times density; the D3PART quarter-model ledger is compared against the per-case expected quarter mass before any symmetry scaling",
    "expected_reference_mass_source": "p0_case_part_map.csv fragment_initial_reference_mass_kg, identity-bound by case_id and fragment_part_id",
    "runtime_completeness_policy": "sum of D3PART fragment element reference masses must match the identity-bound expected quarter mass within initial_discrete_mass_relative_error_max; otherwise all primary QoI are unavailable",
    "ledger_states": [
        "alive_crossed",
        "alive_upstream",
        "deleted_upstream",
        "deleted_after_crossing",
        "temporal_ambiguity",
    ],
    "primary_qois": [
        {
            "name": "qM",
            "definition": "first-crossing fragment reference mass divided by the identity-bound expected initial fragment quarter mass",
        },
        {
            "name": "qP",
            "definition": "outward-positive scalar first-crossing momentum sum(mass*max(-vz,0)) divided by the positive initial incident momentum magnitude expected_reference_mass_kg*v0",
        },
        {
            "name": "qE",
            "definition": "first-crossing element-centroid translational kinetic energy divided by 0.5*expected_reference_mass_kg*v0^2; it excludes unresolved intra-element velocity-gradient energy",
        },
        {
            "name": "qV_over_v0",
            "definition": "reference-mass-weighted outward-positive first-crossing speed divided by impact speed; available only when qM is at least the numerical-perforation mass threshold; qV_mps is derived exactly as qV_over_v0*impact_speed_mps",
        },
    ],
    "diagnostic_only": [
        "fragment_center_history_node",
        "surviving-part terminal average velocity",
        "MATSUM part averages",
    ],
    "required_return_files": [
        "d3hsp",
        "mes*",
        "glstat",
        "matsum",
        "sleout",
        "nodout",
        "binout*",
        "d3plot*",
        "d3part*",
        "solver stdout/stderr",
        "Slurm script and resource record",
    ],
}
EXPECTED_DESIGN_PHYSICS = {
    "unit_system": "kg-mm-ms-kN-GPa-K",
    "provenance_change_scope": "case design and measurement contract only; material, EOS, contact, section, geometry, and timestep controls remain unchanged",
    "symmetry": {
        "type": "quarter_model",
        "full_model_scale_factor": 4.0,
        "validity": "normal incidence, face-on cube, centered impact only",
    },
    "solver_controls": {
        "control_contact_card1": {
            "slsfac": 0.0,
            "rwpnal": 0.0,
            "islchk": 2,
            "shlthk": 0,
            "penopt": 1,
            "thkchg": 0,
            "orien": 1,
            "enmass": 0,
        },
        "control_contact_card2": {
            "usrstr": 0,
            "usrfrc": 0,
            "nsbcs": 0,
            "interm": 0,
            "xpene": 0.0,
            "ssthk": 0,
            "ecdt": 0,
            "tiedprj": 0,
        },
        "eroding_contact_card4": {"isym": 1, "erosop": 1, "iadj": 1},
        "rationale": "Retain the solver-read R04 control cards so R05 isolates the lower-speed design and measurement-contract repair. No material or erosion parameter is tuned against R04 outcomes.",
    },
    "common": {
        "fragment_mass_g": 8.0,
        "fragment_density_kg_m3": 7830.0,
        "plate_thickness_mm": 10.0,
        "obliquity_deg_from_normal": 0.0,
        "fragment_orientation": "face_on",
        "material_set": "jc4340_reference_central",
        "plate_half_span_mm": 60.0,
        "refined_half_span_mm": 15.0,
        "outer_target_spacing_mm": 2.5,
        "initial_gap_mm": 1.0,
        "termination_time_ms": 0.04,
    },
}
EXPECTED_MATERIAL_BASELINE = {
    "schema_version": "hf-material-register@1.0.0",
    "material_id": "jc4340_reference_central",
    "physical_validity": "high_fidelity_synthetic_reference_candidate",
    "unit_system": "kg-mm-ms-kN-GPa-K",
    "strength_model": {
        "lsdyna_keyword": "*MAT_JOHNSON_COOK",
        "density_kg_mm3": 7.83e-06,
        "youngs_modulus_gpa": 200.0,
        "poissons_ratio": 0.29,
        "shear_modulus_gpa": 77.0,
        "A_gpa": 0.792,
        "B_gpa": 0.510,
        "n": 0.26,
        "C": 0.014,
        "m": 1.03,
        "melt_temperature_k": 1793.0,
        "reference_temperature_k": 300.0,
        "reference_strain_rate_per_ms": 0.001,
        "specific_heat_j_kgk": 477.0,
        "plastic_strain_iteration": 1,
    },
    "failure_model": {
        "D1": 0.05,
        "D2": 3.44,
        "D3": -2.12,
        "D4": 0.002,
        "D5": 0.61,
        "erosion_enabled": True,
        "minimum_failure_strain": 1e-06,
    },
    "equation_of_state": {
        "lsdyna_keyword": "*EOS_GRUNEISEN",
        "C_mm_ms": 4568.0,
        "S1": 1.49,
        "S2": 0.0,
        "S3": 0.0,
        "gamma0": 2.17,
        "a": 0.0,
        "initial_internal_energy": 0.0,
        "initial_relative_volume": 1.0,
        "status": "generic-steel EOS sensitivity assumption; not independently validated for this heat treatment",
    },
}


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def successor_contract_boundary(validated_package_revision: str) -> dict:
    try:
        contract = json.loads(SUCCESSOR_REAL_OUTPUT_CONTRACT.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read the successor real-output contract: {exc}")
    boundary = contract.get("version_boundary", {})
    historical = boundary.get("historical_r05", {})
    if contract.get("schema_version") != "hf-p0-r06-real-output-contract@1.0.0":
        fail("successor real-output contract schema differs from the frozen version")
    if (
        boundary.get("applies_to_revision_family") != "R06"
        or boundary.get("effective_scope") != "future_successor_candidates_only"
        or boundary.get("r06_deck_status") != "not_generated"
        or boundary.get("r06_submission_status") != "not_submitted"
    ):
        fail("successor real-output contract has an invalid version boundary")
    if (
        historical.get("package_revision") != EXPECTED_REVISION
        or historical.get("job_id") != "72252362"
        or historical.get("evidence_policy") != "read_only_immutable"
        or historical.get("retroactive_admission") is not False
    ):
        fail("successor real-output contract does not preserve historical R05 evidence")
    return {
        "live_contract_scope": "future_r06_successor_candidate_only",
        "validated_package_revision": str(validated_package_revision),
        "successor_contract": {
            "path": "design/p0_r06_real_output_contract.json",
            "schema_version": contract["schema_version"],
            "sha256": sha256(SUCCESSOR_REAL_OUTPUT_CONTRACT),
            "status": contract.get("status"),
        },
        "r06_deck_status": boundary["r06_deck_status"],
        "r06_submission_status": boundary["r06_submission_status"],
        "historical_r05": {
            "package_revision": historical["package_revision"],
            "job_id": historical["job_id"],
            "evidence_status": historical["evidence_policy"],
            "retroactive_admission": historical["retroactive_admission"],
        },
    }


def resolve_inside(package_dir: Path, relative_path: str) -> Path:
    candidate = (package_dir / relative_path).resolve()
    try:
        candidate.relative_to(package_dir.resolve())
    except ValueError as exc:
        raise AssertionError(f"manifest path escapes package directory: {relative_path}") from exc
    return candidate


def require_frozen_mapping(actual: object, expected: object, label: str) -> None:
    """Require an exact, recursively named pre-solver contract."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            fail(f"{label} changed: expected an object")
        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            fail(f"{label} fields changed: missing={missing}, extra={extra}")
        for key, expected_value in expected.items():
            require_frozen_mapping(actual[key], expected_value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            fail(f"{label} changed: expected {expected!r}, found {actual!r}")
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            require_frozen_mapping(actual_value, expected_value, f"{label}[{index}]")
        return
    if type(actual) is not type(expected) or actual != expected:
        fail(f"{label} changed: expected {expected!r}, found {actual!r}")


def validate_frozen_design_and_material_baseline(design: dict, material: dict) -> dict:
    design_physics = {
        "unit_system": design.get("unit_system"),
        "provenance_change_scope": design.get("provenance", {}).get("change_scope"),
        "symmetry": design.get("symmetry"),
        "solver_controls": design.get("solver_controls"),
        "common": design.get("common"),
    }
    require_frozen_mapping(
        design_physics,
        EXPECTED_DESIGN_PHYSICS,
        "frozen R04 design physics baseline",
    )
    material_physics = {
        key: material.get(key)
        for key in (
            "schema_version",
            "material_id",
            "physical_validity",
            "unit_system",
            "strength_model",
            "failure_model",
            "equation_of_state",
        )
    }
    require_frozen_mapping(
        material_physics,
        EXPECTED_MATERIAL_BASELINE,
        "frozen R04 material baseline",
    )
    return {
        "status": "pass",
        "material_id": material["material_id"],
        "scope": ["material", "EOS", "contact", "section", "geometry", "timestep"],
    }


def validate_frozen_physics_deck(deck_text: str) -> dict:
    grouped: dict[str, list[list[str]]] = defaultdict(list)
    for keyword, data in keyword_blocks(deck_text):
        grouped[keyword].append(data)

    def require_numeric_cards(keyword: str, expected_cards: list[list[float]]) -> None:
        blocks = grouped.get(keyword, [])
        if len(blocks) != 1 or len(blocks[0]) != len(expected_cards):
            fail(
                f"frozen R04 physics deck {keyword} card count changed: "
                f"expected {len(expected_cards)}, found "
                f"{len(blocks[0]) if len(blocks) == 1 else len(blocks)}"
            )
        for card_index, (line, expected) in enumerate(
            zip(blocks[0], expected_cards), start=1
        ):
            fields = split_fields(line)
            if len(fields) != len(expected):
                fail(
                    f"frozen R04 physics deck {keyword} Card {card_index} field count changed"
                )
            try:
                actual = [float(value) for value in fields]
            except ValueError as exc:
                raise AssertionError(
                    f"frozen R04 physics deck {keyword} Card {card_index} is nonnumeric"
                ) from exc
            if actual != expected:
                fail(
                    f"frozen R04 physics deck {keyword} Card {card_index} changed: "
                    f"expected={expected}, found={actual}"
                )

    require_numeric_cards("*CONTROL_TERMINATION", [[0.04, 0.0, 0.0, 0.0, 0.0]])
    require_numeric_cards(
        "*CONTROL_TIMESTEP", [[0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    require_numeric_cards("*CONTROL_ENERGY", [[2.0, 2.0, 2.0, 2.0]])
    require_numeric_cards(
        "*CONTROL_CONTACT",
        [
            [0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    )
    require_numeric_cards(
        "*CONTACT_ERODING_SINGLE_SURFACE",
        [
            [0.0] * 8,
            [0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0],
            [0.0] * 8,
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0, 0.0, 3.0, 5.0, 0.0, 0.0],
        ],
    )

    section = grouped.get("*SECTION_SOLID_TITLE", [])
    if len(section) != 1 or len(section[0]) != 2:
        fail("frozen R04 physics deck solid-section block changed")
    if section[0][0] != "Fully integrated S/R solid for numerical qualification":
        fail("frozen R04 physics deck solid-section title changed")
    try:
        section_card = [float(value) for value in split_fields(section[0][1])]
    except ValueError as exc:
        raise AssertionError("frozen R04 physics deck solid-section card is nonnumeric") from exc
    if section_card != [1.0, -1.0]:
        fail("frozen R04 physics deck solid-section ELFORM changed")

    material_cards = grouped.get("*MAT_JOHNSON_COOK", [])
    expected_material_cards = []
    for material_id in (1.0, 2.0):
        expected_material_cards.append(
            [
                [material_id, 7.83e-06, 77.0, 200.0, 0.29, 0.0, 0.0, 0.0],
                [0.792, 0.510, 0.26, 0.014, 1.03, 1793.0, 300.0, 0.001],
                [477.0, 0.0, 0.0, 1.0, 0.05, 3.44, -2.12, 0.002],
                [0.61, 0.0, 0.0, 1e-06, 0.0],
            ]
        )
    if len(material_cards) != len(expected_material_cards):
        fail("frozen R04 physics deck Johnson-Cook material count changed")
    for block_index, (block, expected_cards) in enumerate(
        zip(material_cards, expected_material_cards), start=1
    ):
        if len(block) != len(expected_cards):
            fail(f"frozen R04 physics deck MAT015 block {block_index} card count changed")
        for card_index, (line, expected) in enumerate(zip(block, expected_cards), start=1):
            try:
                actual = [float(value) for value in split_fields(line)]
            except ValueError as exc:
                raise AssertionError(
                    f"frozen R04 physics deck MAT015 block {block_index} Card {card_index} is nonnumeric"
                ) from exc
            if actual != expected:
                fail(
                    f"frozen R04 physics deck MAT015 block {block_index} Card {card_index} changed"
                )

    eos_cards = grouped.get("*EOS_GRUNEISEN", [])
    expected_eos_cards = [
        [[eos_id, 4568.0, 1.49, 0.0, 0.0, 2.17, 0.0, 0.0], [1.0]]
        for eos_id in (1.0, 2.0)
    ]
    if len(eos_cards) != len(expected_eos_cards):
        fail("frozen R04 physics deck Gruneisen EOS count changed")
    for block_index, (block, expected_cards) in enumerate(
        zip(eos_cards, expected_eos_cards), start=1
    ):
        if len(block) != len(expected_cards):
            fail(f"frozen R04 physics deck EOS block {block_index} card count changed")
        for card_index, (line, expected) in enumerate(zip(block, expected_cards), start=1):
            try:
                actual = [float(value) for value in split_fields(line)]
            except ValueError as exc:
                raise AssertionError(
                    f"frozen R04 physics deck EOS block {block_index} Card {card_index} is nonnumeric"
                ) from exc
            if actual != expected:
                fail(
                    f"frozen R04 physics deck EOS block {block_index} Card {card_index} changed"
                )

    node_set_roles: dict[int, str] = {}
    for data in grouped.get("*SET_NODE_LIST_TITLE", []):
        if len(data) < 3:
            fail("frozen R04 physics deck node-set block is incomplete")
        sid = int(split_fields(data[1])[0])
        node_set_roles[sid] = data[0]
    boundary_cards: dict[int, list[float]] = {}
    for data in grouped.get("*BOUNDARY_SPC_SET", []):
        if len(data) != 1:
            fail("frozen R04 physics deck boundary block is incomplete")
        values = [float(value) for value in split_fields(data[0])]
        boundary_cards[int(values[0])] = values
    if set(boundary_cards) != set(node_set_roles):
        fail("frozen R04 physics deck boundary/node-set identities changed")
    for sid, title in node_set_roles.items():
        if title.startswith("symmetry_x__"):
            expected = [float(sid), 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif title.startswith("symmetry_y__"):
            expected = [float(sid), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        elif title.startswith("clamped_far_edges__"):
            expected = [float(sid), 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        else:
            fail(f"frozen R04 physics deck has unknown boundary role: {title}")
        if boundary_cards[sid] != expected:
            fail(f"frozen R04 physics deck boundary DOFs changed for {title}")

    return {
        "status": "pass",
        "material_blocks": len(material_cards),
        "eos_blocks": len(eos_cards),
        "boundary_set_count": len(boundary_cards),
        "locked_controls": [
            "termination",
            "timestep",
            "energy",
            "contact",
            "section",
            "material",
            "EOS",
            "boundary",
        ],
    }


def base_r04_schema_text(deck_text: str) -> str:
    """Remove the two R05-only blocks before reusing the complete R04 card checker."""
    excluded = {"*DATABASE_BINARY_D3PART", "*SET_PART_LIST_TITLE"}
    lines: list[str] = []
    for keyword, data in keyword_blocks(deck_text):
        if keyword in excluded:
            continue
        lines.append(keyword)
        lines.extend(data)
    return "\n".join(lines) + "\n"


def validate_d3part_contract(deck_text: str, fragment_part_ids: set[int], design: dict) -> dict:
    grouped: dict[str, list[list[str]]] = defaultdict(list)
    for keyword, data in keyword_blocks(deck_text):
        grouped[keyword].append(data)

    d3part = grouped.get("*DATABASE_BINARY_D3PART", [])
    if len(d3part) != 1 or len(d3part[0]) != 1:
        fail("R05 requires exactly one one-card *DATABASE_BINARY_D3PART block")
    fields = split_fields(d3part[0][0])
    if len(fields) != 5:
        fail(
            "*DATABASE_BINARY_D3PART Card 1 requires exactly 5 fields "
            f"for the R15 runtime, found {len(fields)}"
        )
    require_types(fields, "FIIII", "*DATABASE_BINARY_D3PART Card 1 fields 1-5")

    measurement = design["measurement_contract"]
    expected_dt = float(measurement["fragment_d3part_interval_ms"])
    expected_set = int(measurement["fragment_part_set_id"])
    if not math.isclose(float(fields[0]), expected_dt, rel_tol=0.0, abs_tol=1e-15):
        fail("D3PART interval differs from the frozen measurement contract")
    if [int(value) for value in fields[1:4]] != [0, 0, 0]:
        fail("D3PART Card 1 must use CYCL=0, IOOPT=0, and LCDT=0")
    if int(fields[4]) != expected_set:
        fail("D3PART Card 1 must use the frozen PSETID")

    part_sets = grouped.get("*SET_PART_LIST_TITLE", [])
    if len(part_sets) != 1:
        fail("R05 requires exactly one *SET_PART_LIST_TITLE block")
    data = part_sets[0]
    if len(data) < 3:
        fail("fragment part set requires title, header, and at least one PID row")
    header = split_fields(data[1])
    if len(header) != 8:
        fail("*SET_PART_LIST_TITLE Card 1 requires 8 fields")
    require_types(header[:5], "IFFFF", "*SET_PART_LIST_TITLE Card 1 fields 1-5")
    if header[5].upper() != "MECH" or header[6:] != ["", ""]:
        fail("fragment part set must use SOLVER=MECH and blank fields 7-8")
    if int(header[0]) != expected_set:
        fail("D3PART PSETID does not match *SET_PART_LIST_TITLE SID")
    selected: set[int] = set()
    for row_index, line in enumerate(data[2:], start=1):
        row = split_fields(line)
        if not 1 <= len(row) <= 8:
            fail(f"fragment part-set PID row {row_index} has {len(row)} fields")
        nonblank = [value for value in row if value]
        if not nonblank:
            fail(f"fragment part-set PID row {row_index} is empty")
        require_types(nonblank, "I" * len(nonblank), f"fragment part-set PID row {row_index}")
        row_ids = [int(value) for value in nonblank]
        if len(row_ids) != len(set(row_ids)) or selected.intersection(row_ids):
            fail("fragment part set contains a duplicate PID")
        selected.update(row_ids)
    if selected != fragment_part_ids:
        fail(f"D3PART part set must contain only and all fragment PIDs: {sorted(selected)}")

    return {
        "status": "pass",
        "interval_ms": expected_dt,
        "part_set_id": expected_set,
        "fragment_part_ids": sorted(selected),
        "card_1_field_count": 5,
        "card_1_fields": ["DT", "CYCL", "IOOPT", "LCDT", "PSETID"],
        "r15_psetid_semantics": "include_only_parts_in_set",
    }


def validate_measurement_contract(deck_text: str, design: dict) -> dict:
    measurement = design["measurement_contract"]
    gates = design["qualification_gates"]
    common = design["common"]
    cases = design["cases"]
    grouped: dict[str, list[list[str]]] = defaultdict(list)
    for keyword, data in keyword_blocks(deck_text):
        grouped[keyword].append(data)

    d3plot = grouped["*DATABASE_BINARY_D3PLOT"]
    if len(d3plot) != 1 or len(d3plot[0]) != 1:
        fail("expected one one-card *DATABASE_BINARY_D3PLOT block")
    d3plot_fields = split_fields(d3plot[0][0])
    if len(d3plot_fields) != 1 or not NUMERIC_ITEM.fullmatch(d3plot_fields[0]):
        fail("D3PLOT output card must contain only DT")
    d3plot_dt = float(d3plot_fields[0])
    if not math.isclose(
        d3plot_dt,
        float(measurement["full_model_d3plot_interval_ms"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        fail("D3PLOT interval differs from frozen measurement contract")

    for keyword in ("*DATABASE_GLSTAT", "*DATABASE_MATSUM", "*DATABASE_SLEOUT", "*DATABASE_NODOUT"):
        value = float(split_fields(grouped[keyword][0][0])[0])
        if not math.isclose(value, float(measurement["ascii_interval_ms"]), rel_tol=0.0, abs_tol=1e-15):
            fail(f"{keyword} interval differs from frozen measurement contract")

    plane = measurement["capture_plane"]
    plate_back = -float(common["plate_thickness_mm"])
    if plane["axis"] != "z" or plane["travel_direction"] != "negative":
        fail("capture plane must be a negative-z plane")
    if not math.isclose(float(plane["undeformed_plate_back_mm"]), plate_back, abs_tol=1e-12):
        fail("capture-plane plate-back reference differs from geometry")
    expected_plane = plate_back - float(plane["offset_behind_plate_mm"])
    if not math.isclose(float(plane["coordinate_mm"]), expected_plane, abs_tol=1e-12):
        fail("capture-plane coordinate is not the frozen offset behind the plate")

    maximum_speed = max(float(case["impact_speed_mps"]) for case in cases)
    minimum_mesh = min(float(case["mesh_mm"]) for case in cases)
    d3part_dt = float(measurement["fragment_d3part_interval_ms"])
    maximum_nominal_travel = maximum_speed * d3part_dt
    if maximum_nominal_travel > 0.5 * minimum_mesh + 1e-12:
        fail("D3PART sampling allows more than half a finest-mesh cell of nominal travel per state")

    qoi_names = [entry["name"] for entry in measurement["primary_qois"]]
    if qoi_names != ["qM", "qP", "qE", "qV_over_v0"]:
        fail("primary QoI identities or order differ from the frozen capture ledger")
    serialized_primary = json.dumps(measurement["primary_qois"], ensure_ascii=False).lower()
    if "center_history_node" in serialized_primary or "center node" in serialized_primary:
        fail("a single center node cannot be a primary R05 QoI")
    qoi_definitions = {entry["name"]: entry["definition"] for entry in measurement["primary_qois"]}
    if (
        "outward-positive scalar" not in qoi_definitions["qP"]
        or "expected_reference_mass_kg*v0" not in qoi_definitions["qP"]
    ):
        fail(
            "qP must be frozen as an outward-positive scalar with the identity-bound "
            "expected_reference_mass_kg*v0 denominator"
        )
    if "element-centroid translational" not in qoi_definitions["qE"]:
        fail("qE must be limited to element-centroid translational crossing energy")
    if "qV_mps" not in qoi_definitions["qV_over_v0"]:
        fail("qV_over_v0 must define its exact qV_mps conversion")
    if not measurement["capture_plane"]["element_position"].startswith("8-node arithmetic centroid"):
        fail("capture position must match the implemented 8-node arithmetic centroid")
    if not any("center" in item.lower() for item in measurement["diagnostic_only"]):
        fail("center-node output must be explicitly diagnostic-only")

    ledger = gates["capture_ledger"]
    classification = gates["numerical_perforation_classification"]
    mesh_gate = gates["mesh_scan_2000_mps"]
    run_integrity = gates["run_integrity"]
    speed_gate = gates["speed_scan"]
    exact_limits = {
        "initial_discrete_mass_relative_error_max": 0.005,
        "mass_accounting_relative_error_max": 0.000001,
        "temporal_ambiguity_mass_fraction_max": 0.01,
        "deleted_upstream_mass_fraction_max": 0.05,
    }
    for key, expected in exact_limits.items():
        if not math.isclose(float(ledger[key]), expected, rel_tol=0.0, abs_tol=1e-15):
            fail(f"pre-registered capture-ledger threshold changed: {key}")
    if (
        float(classification["confirmed_nonperforation_qM_max"]) != 0.005
        or float(classification["confirmed_nonperforation_qP_max"]) != 0.005
        or float(classification["confirmed_perforation_qM_min"]) != 0.02
        or float(classification["confirmed_perforation_qP_min"]) != 0.01
    ):
        fail("numerical perforation thresholds differ from the pre-registered contract")
    if float(classification["confirmed_nonperforation_worst_case_crossing_mass_max"]) != 0.005:
        fail("confirmed nonperforation must include unresolved mass in its worst-case upper bound")
    if classification["confirmed_nonperforation_worst_case_crossing_mass_definition"] != (
        "qM + temporal_ambiguity_mass_fraction + deleted_upstream_mass_fraction + "
        "max(expected_reference_mass-observed_reference_mass,0)/expected_reference_mass"
    ):
        fail("confirmed nonperforation unresolved-mass definition changed")
    if (
        float(mesh_gate["coarse_to_medium_absolute_difference_max"]) != 0.10
        or float(mesh_gate["medium_to_fine_absolute_difference_max"]) != 0.05
    ):
        fail("mesh-response thresholds differ from the pre-registered contract")
    required_run_definitions = {
        "evaluation_window",
        "total_energy_ratio_definition",
        "hourglass_ratio_definition",
        "added_mass_fraction_definition",
        "missing_or_nonfinite_policy",
    }
    if not required_run_definitions <= set(run_integrity):
        fail("run-integrity evaluation definitions are incomplete")
    if run_integrity["missing_or_nonfinite_policy"] != "unavailable and fail-closed":
        fail("run-integrity missing-field policy must be fail-closed")
    if float(speed_gate["confirmed_perforation_qV_downward_jump_mps_max"]) != 100.0:
        fail("qV downward-jump limit changed")
    if speed_gate["qV_mps_definition"] != "qV_over_v0 multiplied by that case's impact_speed_mps":
        fail("qV_mps derivation changed")
    if "adjacent cases in the complete speed-sorted scan" not in speed_gate["qV_downward_jump_comparison"]:
        fail("qV downward-jump comparison algorithm is not frozen")

    return {
        "status": "pass",
        "capture_plane_z_mm": float(plane["coordinate_mm"]),
        "full_model_d3plot_interval_ms": d3plot_dt,
        "fragment_d3part_interval_ms": d3part_dt,
        "maximum_nominal_fragment_travel_per_d3part_state_mm": maximum_nominal_travel,
        "finest_mesh_half_cell_mm": 0.5 * minimum_mesh,
        "primary_qois": qoi_names,
        "center_history_node": "diagnostic_only",
        "threshold_status": "frozen_candidate_engineering_gates",
    }


def validate_candidate(package_dir: Path, report_path: Path | None = None) -> dict:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    deck = resolve_inside(package_dir, manifest["deck"]["path"])
    case_map = resolve_inside(package_dir, manifest["case_map"]["path"])
    design_path = resolve_inside(package_dir, manifest["design"]["path"])
    material_path = resolve_inside(package_dir, manifest["material_register"]["path"])
    p1_path = resolve_inside(package_dir, manifest["p1_design"]["path"])

    deck_text = deck.read_text(encoding="ascii")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    material = json.loads(material_path.read_text(encoding="utf-8"))
    with case_map.open("r", encoding="utf-8", newline="") as handle:
        mapping = list(csv.DictReader(handle))

    if manifest["package_revision"] != EXPECTED_REVISION or design["package_revision"] != EXPECTED_REVISION:
        fail("package revision is not the frozen R05 identity")
    if design["design_status"] != EXPECTED_DESIGN_STATUS:
        fail("R05 design is not marked frozen before solver outcomes")
    if manifest.get("measurement_contract") != design["measurement_contract"]:
        fail("manifest measurement contract differs from frozen design")
    require_frozen_mapping(
        design.get("qualification_gates"),
        EXPECTED_QUALIFICATION_GATES,
        "frozen qualification gate",
    )
    require_frozen_mapping(
        design.get("measurement_contract"),
        EXPECTED_MEASUREMENT_CONTRACT,
        "frozen measurement contract",
    )
    baseline_check = validate_frozen_design_and_material_baseline(design, material)

    checks: dict[str, dict] = {}
    checks["frozen_qualification_gates"] = {
        "status": "pass",
        "gate_groups": sorted(EXPECTED_QUALIFICATION_GATES),
    }
    checks["frozen_measurement_contract"] = {
        "status": "pass",
        "schema_version": EXPECTED_MEASUREMENT_CONTRACT["schema_version"],
    }
    checks["frozen_design_and_material_baseline"] = baseline_check
    checks["frozen_physics_deck"] = validate_frozen_physics_deck(deck_text)
    checks["keyword_input_format"] = validate_keyword_input_format(deck_text)
    checks["required_card_layouts"] = validate_required_card_layouts(deck_text)
    checks["base_keyword_schema_contracts"] = validate_keyword_schema_contracts(
        base_r04_schema_text(deck_text)
    )

    if len(mapping) != len(design["cases"]):
        fail("case-map count differs from frozen R05 design")
    for row, case in zip(mapping, design["cases"]):
        if row["case_id"] != case["case_id"] or row["role"] != case["role"]:
            fail("case-map ID/role differs from frozen R05 design")
        if float(row["impact_speed_mps"]) != float(case["impact_speed_mps"]):
            fail(f"case-map speed differs from design for {case['case_id']}")
        if float(row["mesh_mm"]) != float(case["mesh_mm"]):
            fail(f"case-map mesh differs from design for {case['case_id']}")
        if row["primary_qoi_source"] != "d3part_fragment_set":
            fail("case map must identify D3PART as the primary QoI source")
        if row["center_history_node_status"] != "diagnostic_only":
            fail("case map must mark the center history node diagnostic-only")
        if float(row["capture_plane_z_mm"]) != float(design["measurement_contract"]["capture_plane"]["coordinate_mm"]):
            fail("case-map capture plane differs from frozen design")
    checks["frozen_case_semantics"] = {"status": "pass", "case_count": len(mapping)}

    expected_speed_cases = design["cohorts"]["speed_scan_case_ids"]
    speed_rows = [case for case in design["cases"] if case["case_id"] in expected_speed_cases]
    if [float(case["impact_speed_mps"]) for case in speed_rows] != [600.0, 800.0, 1000.0, 1200.0, 2000.0]:
        fail("R05 speed scan is not the frozen lower-boundary design")
    mesh_ids = design["cohorts"]["mesh_scan_2000_mps_case_ids"]
    mesh_rows = [case for case in design["cases"] if case["case_id"] in mesh_ids]
    if sorted(float(case["mesh_mm"]) for case in mesh_rows) != [0.5, 0.75, 1.0]:
        fail("R05 must retain the 2000 m/s three-grid stress test")
    if any(float(case["impact_speed_mps"]) != 2000.0 for case in mesh_rows):
        fail("R05 mesh cohort must remain at 2000 m/s")
    checks["cohort_identity"] = {
        "status": "pass",
        "speed_scan_mps": [600.0, 800.0, 1000.0, 1200.0, 2000.0],
        "mesh_scan_speed_mps": 2000.0,
        "mesh_scan_mm": [1.0, 0.75, 0.5],
    }

    part_ids: list[int] = []
    history_nodes: list[int] = []
    for row in mapping:
        part_ids.extend([int(row["plate_part_id"]), int(row["fragment_part_id"])])
        history_nodes.append(int(row["fragment_center_history_node_id"]))
        if int(row["plate_elements"]) <= 0 or int(row["fragment_elements"]) <= 0:
            fail(f"non-positive element count in {row['case_id']}")
    if len(part_ids) != len(set(part_ids)) or len(history_nodes) != len(set(history_nodes)):
        fail("part or history-node IDs are not unique")
    fragment_part_ids = {int(row["fragment_part_id"]) for row in mapping}
    checks["d3part_contract"] = validate_d3part_contract(deck_text, fragment_part_ids, design)
    checks["measurement_contract"] = validate_measurement_contract(deck_text, design)

    node_ids: set[int] = set()
    node_coords: dict[int, tuple[float, float, float]] = {}
    element_ids: set[int] = set()
    elements: list[tuple[int, int, list[int]]] = []
    referenced_nodes: set[int] = set()
    mode = ""
    for raw in deck_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("$"):
            continue
        if line.startswith("*"):
            mode = line.upper()
            continue
        fields = split_fields(line)
        if mode == "*NODE" and len(fields) >= 4:
            try:
                nid = int(fields[0])
                xyz = tuple(float(value) for value in fields[1:4])
            except ValueError:
                continue
            if nid in node_ids or not all(math.isfinite(value) for value in xyz):
                fail(f"invalid or duplicate node {nid}")
            node_ids.add(nid)
            node_coords[nid] = xyz
        elif mode == "*ELEMENT_SOLID" and len(fields) >= 10:
            try:
                eid = int(fields[0])
                pid = int(fields[1])
                nodes = [int(value) for value in fields[2:10]]
            except ValueError:
                continue
            if eid in element_ids:
                fail(f"duplicate element ID {eid}")
            element_ids.add(eid)
            referenced_nodes.update(nodes)
            elements.append((eid, pid, nodes))
    missing_nodes = sorted(referenced_nodes - node_ids)
    if missing_nodes:
        fail(f"elements reference missing nodes: {missing_nodes[:10]}")
    if len(node_ids) != int(manifest["deck"]["node_count"]):
        fail("manifest node count mismatch")
    if len(element_ids) != int(manifest["deck"]["element_count"]):
        fail("manifest element count mismatch")
    checks["mesh_references"] = {
        "status": "pass",
        "node_count": len(node_ids),
        "element_count": len(element_ids),
    }

    grouped: dict[str, list[list[str]]] = defaultdict(list)
    for keyword, data in keyword_blocks(deck_text):
        grouped[keyword].append(data)
    plate_parts = {int(row["plate_part_id"]) for row in mapping}
    mapped_parts = plate_parts | fragment_part_ids
    defined_parts: dict[int, tuple[int, int, int]] = {}
    for data in grouped["*PART"]:
        values = [int(value) for value in split_fields(data[1])]
        defined_parts[values[0]] = tuple(values[1:4])
    if set(defined_parts) != mapped_parts:
        fail("defined parts differ from the case map")
    if any(defined_parts[pid] != (1, 1, 1) for pid in plate_parts):
        fail("plate section/material/EOS reference mismatch")
    if any(defined_parts[pid] != (1, 2, 2) for pid in fragment_part_ids):
        fail("fragment section/material/EOS reference mismatch")
    if {pid for _, pid, _ in elements} != mapped_parts:
        fail("element part references differ from mapped parts")

    velocity_by_part: dict[int, float] = {}
    for data in grouped["*INITIAL_VELOCITY_GENERATION"]:
        values = split_fields(data[0])
        pid = int(values[0])
        if int(values[1]) != 2 or pid not in fragment_part_ids:
            fail("initial velocity does not target a mapped fragment part")
        velocity_by_part[pid] = float(values[5])
    for row in mapping:
        if velocity_by_part[int(row["fragment_part_id"])] != -float(row["impact_speed_mps"]):
            fail(f"initial velocity mismatch for {row['case_id']}")

    deck_history = {
        int(value)
        for data in grouped["*DATABASE_HISTORY_NODE"]
        for line in data
        for value in split_fields(line)
    }
    if deck_history != set(history_nodes) or not deck_history <= node_ids:
        fail("diagnostic center-node output differs from mapped existing nodes")
    checks["keyword_reference_integrity"] = {
        "status": "pass",
        "part_count": len(defined_parts),
        "velocity_target_count": len(velocity_by_part),
        "diagnostic_history_node_count": len(deck_history),
    }

    part_nodes: dict[int, set[int]] = defaultdict(set)
    part_volumes: dict[int, float] = defaultdict(float)
    signed_volumes: list[float] = []
    for eid, pid, nodes in elements:
        part_nodes[pid].update(nodes)
        p1, p2, _p3, p4, p5 = (node_coords[node] for node in nodes[:5])
        a = tuple(p2[i] - p1[i] for i in range(3))
        b = tuple(p4[i] - p1[i] for i in range(3))
        c = tuple(p5[i] - p1[i] for i in range(3))
        cross_bc = (b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0])
        volume = sum(a[i] * cross_bc[i] for i in range(3))
        if not math.isfinite(volume) or volume <= 0.0:
            fail(f"non-positive structured-hexahedron corner volume at element {eid}")
        signed_volumes.append(volume)
        part_volumes[pid] += volume
    checks["element_orientation"] = {
        "status": "pass",
        "minimum_mm3": min(signed_volumes),
        "maximum_mm3": max(signed_volumes),
    }

    expected_gap = float(design["common"]["initial_gap_mm"])
    gaps: list[float] = []
    bounds: dict[str, tuple[float, float, float, float]] = {}
    for row in mapping:
        plate_pid = int(row["plate_part_id"])
        fragment_pid = int(row["fragment_part_id"])
        gap = min(node_coords[n][2] for n in part_nodes[fragment_pid]) - max(
            node_coords[n][2] for n in part_nodes[plate_pid]
        )
        if abs(gap - expected_gap) > 1e-9:
            fail(f"initial gap mismatch in {row['case_id']}")
        gaps.append(gap)
        case_nodes = part_nodes[plate_pid] | part_nodes[fragment_pid]
        xs = [node_coords[n][0] for n in case_nodes]
        ys = [node_coords[n][1] for n in case_nodes]
        bounds[row["case_id"]] = (min(xs), max(xs), min(ys), max(ys))
    pair_gaps: list[float] = []
    ids = list(bounds)
    for left_i, left_id in enumerate(ids):
        lx0, lx1, ly0, ly1 = bounds[left_id]
        for right_id in ids[left_i + 1 :]:
            rx0, rx1, ry0, ry1 = bounds[right_id]
            planar = math.hypot(max(rx0 - lx1, lx0 - rx1, 0.0), max(ry0 - ly1, ly0 - ry1, 0.0))
            if planar <= 0.0:
                fail(f"P0 submodels overlap: {left_id}, {right_id}")
            pair_gaps.append(planar)
    checks["initial_separation_and_case_isolation"] = {
        "status": "pass",
        "gap_mm": expected_gap,
        "minimum_pairwise_xy_gap_mm": min(pair_gaps),
    }

    expected_mass = float(design["common"]["fragment_mass_g"]) / 1000.0
    density = float(material["strength_model"]["density_kg_mm3"])
    expected_quarter_mass = expected_mass / 4.0
    expected_plate_volume = float(design["common"]["plate_half_span_mm"]) ** 2 * float(
        design["common"]["plate_thickness_mm"]
    )
    mass_rows = []
    for row in mapping:
        fragment_pid = int(row["fragment_part_id"])
        plate_pid = int(row["plate_part_id"])
        meshed_mass = density * part_volumes[fragment_pid]
        if abs(meshed_mass - expected_quarter_mass) / expected_quarter_mass > 1e-7:
            fail(f"meshed fragment mass mismatch in {row['case_id']}")
        if abs(part_volumes[plate_pid] - expected_plate_volume) / expected_plate_volume > 1e-7:
            fail(f"meshed plate volume mismatch in {row['case_id']}")
        if abs(float(row["fragment_initial_reference_mass_kg"]) - expected_quarter_mass) > 1e-15:
            fail(f"case-map reference mass mismatch in {row['case_id']}")
        mass_rows.append({"case_id": row["case_id"], "quarter_mass_kg": meshed_mass})
    checks["meshed_part_integrals"] = {
        "status": "pass",
        "expected_quarter_fragment_mass_kg": expected_quarter_mass,
        "expected_quarter_plate_volume_mm3": expected_plate_volume,
        "cases": mass_rows,
    }

    if manifest["design"]["sha256"] != sha256(design_path) or manifest["design_hash"] != sha256(design_path):
        fail("manifest design hash mismatch")
    if manifest["material_register"]["sha256"] != sha256(material_path) or manifest["material_register_hash"] != sha256(material_path):
        fail("manifest material hash mismatch")
    if manifest["deck"]["sha256"] != sha256(deck) or manifest["case_map"]["sha256"] != sha256(case_map):
        fail("manifest generated-artifact hash mismatch")
    if manifest["p1_design"]["sha256"] != sha256(p1_path):
        fail("manifest P1 design hash mismatch")
    checks["hash_binding"] = {"status": "pass"}

    with p1_path.open("r", encoding="utf-8", newline="") as handle:
        p1_rows = list(csv.DictReader(handle))
    roles = Counter(row["role"] for row in p1_rows)
    if len(p1_rows) != 24 or len({row["case_id"] for row in p1_rows}) != 24:
        fail("P1 requires 24 unique preregistered cases")
    if dict(roles) != {"hf_response_fit": 18, "hf_response_holdout": 6}:
        fail("P1 role counts differ from preregistration")
    if any(row["status"] != "preregistered_not_submitted" for row in p1_rows):
        fail("P1 status changed before P0 admission")
    checks["p1_remains_preregistered_not_submitted"] = {
        "status": "pass",
        "case_count": len(p1_rows),
        "role_counts": dict(roles),
    }

    forbidden = ["physical_validity=established", "empirically_validated", "certified_real_world"]
    hits = [token for token in forbidden if re.search(re.escape(token), deck_text, flags=re.IGNORECASE)]
    if hits:
        fail(f"forbidden evidence claims in deck: {hits}")
    checks["claim_boundary"] = {"status": "pass"}

    report = {
        "schema_version": "hf-static-validation@1.3.0",
        "package_revision": EXPECTED_REVISION,
        "overall_status": "pass",
        "scope": "Live static successor-candidate compatibility validation only; historical R05 Job 72252362 remains read-only and is not retroactively admitted.",
        "version_boundary": successor_contract_boundary(design["package_revision"]),
        "checks": checks,
        "prohibited_inference": "This PASS does not establish solver acceptance, QoI availability, numerical convergence, physical validity, P0 response-calibration admission, or P1 admission.",
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a self-contained P0 candidate with live successor checks"
    )
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.report or (args.package_dir / "static_validation.json")
    report = validate_candidate(args.package_dir, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
