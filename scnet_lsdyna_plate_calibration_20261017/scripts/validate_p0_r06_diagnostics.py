from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from generate_p0_r06_diagnostics import DEFAULT_OUTPUT, strict_json
from validate_package import (
    INTEGER_ITEM,
    NUMERIC_ITEM,
    keyword_blocks,
    require_types,
    split_fields,
    validate_keyword_input_format,
)


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_R05_DECK = ROOT / "manual_upload_20261021_r05_candidate" / "p0_velocity_mesh_calibration.k"
HISTORICAL_R05_DECK_SHA256 = "sha256:6fbd753a113d0214e7ba4ff5136dc526d151e6f2c52c94e70399b29c4b9d3d0c"
# Use the project-bound released snapshot rather than a mutable downloaded-return
# folder that may be moved after audit. The hash is the returned Job 72397333 deck
# identity and is checked before every parity comparison.
HISTORICAL_R06E_0800_DECK = (
    ROOT
    / "artifacts"
    / "r06e_wave0_canary_20261025"
    / "uav_hf_r06e_0800_ctrl"
    / "r06e_0800_ctrl.k"
)
HISTORICAL_R06E_0800_DECK_SHA256 = "sha256:47c5fcacbec9ba3edac10d06611b6ba602f58a135b27b62d968290334a763a70"
HISTORICAL_R06F_0800_DECK = (
    ROOT
    / "artifacts"
    / "r06f_wave0_canary_20261027"
    / "uav_hf_r06f_0800_ctrl"
    / "r06f_0800_ctrl.k"
)
HISTORICAL_R06F_0800_DECK_SHA256 = "sha256:111911562b703b8af1e96810b08edd8c31fbeea739ddddd2921b8bcd07c517af"
HISTORICAL_R06F_TSSFAC04_DECK = (
    ROOT
    / "manual_upload_20261028_r06f_timestep_sensitivity"
    / "job_01_v0800_tssfac04"
    / "r06f_0800_tssfac04.k"
)
HISTORICAL_R06F_TSSFAC04_DECK_SHA256 = "sha256:afb0f34112ccfdfcfd95bf442c9b45e2cb1590fdad8d76e98fbbc3fb5d6baee7"
HISTORICAL_R06G_TSSFAC02_DECK = (
    ROOT
    / "analysis"
    / "72639935_r06g_0800_tssfac02_20261029"
    / "immutable_parent_r06g_0800_tssfac02.k"
)
HISTORICAL_R06G_TSSFAC02_DECK_SHA256 = "sha256:b5f4da367e9ef6eaae7098e47a84610eb55c51760cc9bf8dbb4721693b7a732a"
HISTORICAL_R06H_TSSFAC01_DECK = (
    ROOT
    / "analysis"
    / "72701744_r06h_0800_tssfac01_20261030"
    / "immutable_parent_r06h_0800_tssfac01.k"
)
HISTORICAL_R06H_TSSFAC01_DECK_SHA256 = "sha256:6e4fb3806722df91df8500ea3e2a6758ef6a07c36e913f4e8eccc7a9083374c3"
HISTORICAL_R06I_TSSFAC005_DECK = (
    ROOT
    / "analysis"
    / "72740022_r06i_0800_tssfac005_20261030"
    / "immutable_parent_r06i_0800_tssfac005.k"
)
HISTORICAL_R06I_TSSFAC005_DECK_SHA256 = "sha256:e2e8da8e55304b7fe2deb2c1790e00bf7fea9c64145c6c09a954c50492f3dd88"
HISTORICAL_R06J_FRAGMENT_EROD1_DECK = (
    ROOT
    / "analysis"
    / "72751725_r06j_0800_fragment_erod1_20261030"
    / "immutable_parent_r06j_0800_fragment_erod1.k"
)
HISTORICAL_R06J_FRAGMENT_EROD1_DECK_SHA256 = "sha256:8ec1613324159b3ebc00217871ad29321564423b0a7f73a267207e4952bada77"
HISTORICAL_R06K_FRAGMENT_PSFAIL_DECK = (
    ROOT
    / "analysis"
    / "72764387_r06k_0800_fragment_psfail_plate_20261031"
    / "immutable_parent_r06k_0800_fragment_psfail_plate.k"
)
HISTORICAL_R06K_FRAGMENT_PSFAIL_DECK_SHA256 = "sha256:f78396a9fd15114fe411507a7774d68cecb10a25460413474772f2b99e8a17ca"
R06F_RECANARY_STATUS = (
    "frozen_after_r06e_wave0_solver_capability_audit_before_any_r06f_solver_outcome"
)
R06F_TIMESTEP_STATUS = "frozen_after_r06f_recanary_pass_before_timestep_solver_outcome"
R06G_REFINEMENT_STATUS = (
    "frozen_after_r06f_tssfac04_sensitivity_fail_before_tssfac02_outcome"
)
R06H_CONFIRMATION_STATUS = (
    "frozen_after_r06g_tssfac02_local_gate_pass_before_tssfac01_outcome"
)
R06I_TERMINAL_REFINEMENT_STATUS = (
    "frozen_after_r06h_tssfac01_confirmation_fail_before_tssfac005_outcome"
)
R06J_FRAGMENT_EROSION_STATUS = (
    "frozen_after_r06i_terminal_timestep_stable_but_response_unavailable_before_fragment_erod1_outcome"
)
R06K_FRAGMENT_NEGATIVE_VOLUME_STATUS = (
    "frozen_after_r06j_partial_erod_causality_before_fragment_negative_volume_psfail_outcome"
)
R06L_CONTACT_SOFT1_STATUS = (
    "frozen_after_r06k_fragment_deletion_suppressed_but_contact_energy_timestep_unstable_before_soft1_outcome"
)
SINGLE_JOB_R06F_STATUSES = {
    R06F_RECANARY_STATUS,
    R06F_TIMESTEP_STATUS,
    R06G_REFINEMENT_STATUS,
    R06H_CONFIRMATION_STATUS,
    R06I_TERMINAL_REFINEMENT_STATUS,
    R06J_FRAGMENT_EROSION_STATUS,
    R06K_FRAGMENT_NEGATIVE_VOLUME_STATUS,
    R06L_CONTACT_SOFT1_STATUS,
}
REVISION = "R06_paired_diagnostics_preregistered_20261023"
JOB_BASE_FILES = {
    "UPLOAD_README.txt",
    "jc4340_reference.json",
    "job_contract.json",
    "manifest.json",
    "p0_case_part_map.csv",
    "p0_r06_diagnostic_design.json",
    "p0_r06_real_output_contract.json",
}


def job_base_files(manifest: dict) -> set[str]:
    """Return the manifest-bound, explicit per-job whitelist."""
    return {
        "UPLOAD_README.txt",
        "manifest.json",
        manifest["design"]["path"],
        manifest["material"]["path"],
        manifest["successor_contract"]["path"],
        manifest["case_map"]["path"],
        manifest["job_contract"]["path"],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def fail(message: str) -> None:
    raise AssertionError(message)


def grouped_blocks(deck_text: str) -> dict[str, list[list[str]]]:
    grouped: dict[str, list[list[str]]] = defaultdict(list)
    for keyword, data in keyword_blocks(deck_text):
        grouped[keyword].append(data)
    return grouped


def one_card(grouped: dict[str, list[list[str]]], keyword: str, types: str) -> list[str]:
    data = grouped[keyword][0]
    if len(data) != 1:
        fail(f"{keyword} requires exactly one card, found {len(data)}")
    items = split_fields(data[0])
    require_types(items, types, f"{keyword} Card 1")
    return items


def validate_schema(deck_text: str, design: dict | None = None) -> dict:
    grouped = grouped_blocks(deck_text)
    measure = design.get("shared_measurement_upgrade", {}) if design is not None else {}
    parser_contract = (
        measure.get("r15_parser_contract")
        if design is not None
        else None
    )
    direct_contact_policy = measure.get(
        "direct_contact_output_policy",
        {"request_binary_intfor": True, "request_ascii_ncforc": True},
    )
    request_intfor = bool(direct_contact_policy["request_binary_intfor"])
    request_ncforc = bool(direct_contact_policy["request_ascii_ncforc"])
    d3part_extent_cards = measure.get("d3part_extent_cards", [])
    expected_counts = {
        "*KEYWORD": 1,
        "*TITLE": 1,
        "*CONTROL_TERMINATION": 1,
        "*CONTROL_TIMESTEP": 1,
        "*CONTROL_SOLID": 1,
        "*CONTROL_ENERGY": 1,
        "*CONTROL_CONTACT": 1,
        "*DATABASE_BINARY_D3PLOT": 1,
        "*DATABASE_EXTENT_BINARY": 1,
        "*DATABASE_BINARY_D3PART": 1,
        "*DATABASE_EXTENT_D3PART": 1 if d3part_extent_cards else 0,
        "*DATABASE_BINARY_INTFOR_FILE": 1 if request_intfor else 0,
        "*DATABASE_EXTENT_INTFOR": 1 if request_intfor else 0,
        "*DATABASE_GLSTAT": 1,
        "*DATABASE_MATSUM": 1,
        "*DATABASE_SLEOUT": 1,
        "*DATABASE_NODOUT": 1,
        "*DATABASE_NCFORC": 1 if request_ncforc else 0,
        "*CONTACT_ERODING_SINGLE_SURFACE": 1,
        "*SECTION_SOLID_TITLE": 1,
        "*MAT_JOHNSON_COOK": 2,
        "*EOS_GRUNEISEN": 2,
        "*PART": 2,
        "*NODE": 2,
        "*ELEMENT_SOLID": 2,
        "*SET_NODE_LIST_TITLE": 3,
        "*BOUNDARY_SPC_SET": 3,
        "*INITIAL_VELOCITY_GENERATION": 1,
        "*DATABASE_HISTORY_NODE": 1,
        "*SET_PART_LIST_TITLE": 2,
        "*END": 1,
    }
    unexpected = sorted(set(grouped) - set(expected_counts))
    if unexpected:
        fail(f"unexpected keywords in R06 frozen schema: {unexpected}")
    wrong = {
        key: {"expected": count, "actual": len(grouped.get(key, []))}
        for key, count in expected_counts.items()
        if len(grouped.get(key, [])) != count
    }
    if wrong:
        fail(f"R06 keyword block counts differ from frozen schema: {wrong}")
    if grouped["*KEYWORD"][0] or grouped["*END"][0]:
        fail("*KEYWORD and *END must not contain data")
    if len(grouped["*TITLE"][0]) != 1:
        fail("*TITLE requires one line")

    one_card(grouped, "*CONTROL_TERMINATION", "F I F F F".replace(" ", ""))
    one_card(grouped, "*CONTROL_TIMESTEP", "FFIFFIII")
    one_card(grouped, "*CONTROL_SOLID", "IIIIIFII")
    one_card(grouped, "*CONTROL_ENERGY", "IIII")
    for keyword in (
        "*DATABASE_BINARY_D3PLOT",
        "*DATABASE_GLSTAT",
        "*DATABASE_MATSUM",
        "*DATABASE_SLEOUT",
        "*DATABASE_NODOUT",
        *(["*DATABASE_NCFORC"] if request_ncforc else []),
    ):
        one_card(grouped, keyword, "F")
    extent_binary = grouped["*DATABASE_EXTENT_BINARY"][0]
    expected_extent_card_count = 2 if parser_contract is not None else 1
    if len(extent_binary) != expected_extent_card_count:
        fail(
            "*DATABASE_EXTENT_BINARY requires exactly "
            f"{expected_extent_card_count} cards, found {len(extent_binary)}"
        )
    for card_index, line in enumerate(extent_binary, start=1):
        require_types(
            split_fields(line),
            "IIIIIIII",
            f"*DATABASE_EXTENT_BINARY Card {card_index}",
        )
    one_card(grouped, "*DATABASE_BINARY_D3PART", "FIIII")
    if d3part_extent_cards:
        extent_d3part = grouped["*DATABASE_EXTENT_D3PART"][0]
        if len(extent_d3part) != len(d3part_extent_cards):
            fail(
                "*DATABASE_EXTENT_D3PART card count differs from the frozen contract: "
                f"{len(extent_d3part)} != {len(d3part_extent_cards)}"
            )
        for card_index, line in enumerate(extent_d3part, start=1):
            require_types(
                split_fields(line),
                "IIIIIIII",
                f"*DATABASE_EXTENT_D3PART Card {card_index}",
            )

    control = grouped["*CONTROL_CONTACT"][0]
    if len(control) != 2:
        fail("*CONTROL_CONTACT must have two cards")
    require_types(split_fields(control[0]), "FFIIIIII", "*CONTROL_CONTACT Card 1")
    require_types(split_fields(control[1]), "IIIIFIII", "*CONTROL_CONTACT Card 2")

    if request_intfor:
        intfor_file = grouped["*DATABASE_BINARY_INTFOR_FILE"][0]
        if len(intfor_file) != 3 or intfor_file[0] != "intfor":
            fail("*DATABASE_BINARY_INTFOR_FILE requires filename intfor plus Cards 1 and 2")
        intfor_card1_types = "FIII" if parser_contract is not None else "FIIII"
        require_types(
            split_fields(intfor_file[1]),
            intfor_card1_types,
            "*DATABASE_BINARY_INTFOR_FILE Card 1",
        )
        require_types(split_fields(intfor_file[2]), "I", "*DATABASE_BINARY_INTFOR_FILE Card 2")
        extent_intfor = grouped["*DATABASE_EXTENT_INTFOR"][0]
        if len(extent_intfor) != 2:
            fail("*DATABASE_EXTENT_INTFOR requires two cards")
        require_types(split_fields(extent_intfor[0]), "IIIIIIII", "*DATABASE_EXTENT_INTFOR Card 1")
        require_types(split_fields(extent_intfor[1]), "IIIIII", "*DATABASE_EXTENT_INTFOR Card 2")

    contact = grouped["*CONTACT_ERODING_SINGLE_SURFACE"][0]
    if len(contact) != 5:
        fail("*CONTACT_ERODING_SINGLE_SURFACE requires Cards 1-4 and Card A")
    for index, (line, types) in enumerate(
        zip(contact, ("IIIIIIII", "FFFFFIFF", "FFFFFFFF", "III", "IFIIIIII")), start=1
    ):
        require_types(split_fields(line), types, f"*CONTACT_ERODING_SINGLE_SURFACE row {index}")
    section = grouped["*SECTION_SOLID_TITLE"][0]
    if len(section) != 2:
        fail("*SECTION_SOLID_TITLE requires title and one card")
    require_types(split_fields(section[1]), "II", "*SECTION_SOLID_TITLE Card 1")
    for block_index, data in enumerate(grouped["*MAT_JOHNSON_COOK"], start=1):
        if len(data) != 4:
            fail(f"MAT_015 block {block_index} must have four cards")
        for card_index, (line, types) in enumerate(
            zip(data, ("IFFFFFFF", "FFFFFFFF", "FFFFFFFF", "FFFFI")), start=1
        ):
            require_types(split_fields(line), types, f"MAT_015 block {block_index} Card {card_index}")
    for block_index, data in enumerate(grouped["*EOS_GRUNEISEN"], start=1):
        if len(data) != 2:
            fail(f"EOS block {block_index} must have two cards")
        require_types(split_fields(data[0]), "IFFFFFFF", f"EOS block {block_index} Card 1")
        require_types(split_fields(data[1]), "F", f"EOS block {block_index} Card 2")
    for block_index, data in enumerate(grouped["*PART"], start=1):
        if len(data) != 2:
            fail(f"PART block {block_index} must have title and card")
        require_types(split_fields(data[1]), "IIIIIIII", f"PART block {block_index} Card 1")
    for block_index, data in enumerate(grouped["*NODE"], start=1):
        if not data:
            fail(f"NODE block {block_index} is empty")
        for row_index, line in enumerate(data, start=1):
            require_types(split_fields(line), "IFFF", f"NODE block {block_index} row {row_index}")
    for block_index, data in enumerate(grouped["*ELEMENT_SOLID"], start=1):
        if not data:
            fail(f"ELEMENT block {block_index} is empty")
        for row_index, line in enumerate(data, start=1):
            require_types(split_fields(line), "IIIIIIIIII", f"ELEMENT block {block_index} row {row_index}")
    for block_index, data in enumerate(grouped["*SET_NODE_LIST_TITLE"], start=1):
        if len(data) < 3:
            fail(f"node set {block_index} is incomplete")
        require_types(split_fields(data[1]), "IFFFF", f"node set {block_index} header")
        for line in data[2:]:
            items = split_fields(line)
            require_types(items, "I" * len(items), f"node set {block_index} IDs")
    for data in grouped["*BOUNDARY_SPC_SET"]:
        if len(data) != 1:
            fail("BOUNDARY_SPC_SET must have one card")
        require_types(split_fields(data[0]), "IIIIIIII", "BOUNDARY_SPC_SET Card 1")
    velocity = grouped["*INITIAL_VELOCITY_GENERATION"][0]
    if len(velocity) != 2:
        fail("INITIAL_VELOCITY_GENERATION must have two cards")
    require_types(split_fields(velocity[0]), "IIFFFFII", "INITIAL_VELOCITY_GENERATION Card 1")
    require_types(split_fields(velocity[1]), "FFFFFFII", "INITIAL_VELOCITY_GENERATION Card 2")
    one_card(grouped, "*DATABASE_HISTORY_NODE", "I")
    for block_index, data in enumerate(grouped["*SET_PART_LIST_TITLE"], start=1):
        if len(data) != 3:
            fail(f"part set {block_index} must contain title, header, and one PID row")
        header = split_fields(data[1])
        if len(header) != 8 or not INTEGER_ITEM.fullmatch(header[0]) or header[5] != "MECH":
            fail(f"part set {block_index} header is malformed")
        require_types(split_fields(data[2]), "I", f"part set {block_index} PID row")

    schema_report = {
        "status": "pass",
        "keyword_type_count": (
            len(expected_counts)
            if "direct_contact_output_policy" in measure or d3part_extent_cards
            else sum(1 for count in expected_counts.values() if count > 0)
        ),
        "keyword_block_count": sum(expected_counts.values()),
        **(
            {"extent_binary_cards": expected_extent_card_count}
            if parser_contract is not None
            else {}
        ),
        "intfor_file_cards": 3 if request_intfor else 0,
        "extent_intfor_cards": 2 if request_intfor else 0,
    }
    if "direct_contact_output_policy" in measure or d3part_extent_cards:
        schema_report.update(
            {
                "d3part_extent_cards": len(d3part_extent_cards),
                "ncforc_requested": request_ncforc,
            }
        )
    return schema_report


def parse_mesh(deck_text: str) -> tuple[dict[int, tuple[float, float, float]], dict[int, tuple[int, tuple[int, ...]]]]:
    nodes: dict[int, tuple[float, float, float]] = {}
    elements: dict[int, tuple[int, tuple[int, ...]]] = {}
    mode = ""
    for raw in deck_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("$"):
            continue
        if line.startswith("*"):
            mode = line.upper().split()[0]
            continue
        fields = split_fields(line)
        if mode == "*NODE" and len(fields) == 4:
            try:
                nid = int(fields[0])
                coords = tuple(float(value) for value in fields[1:4])
            except ValueError:
                continue
            if nid in nodes:
                fail(f"duplicate node ID {nid}")
            if not all(math.isfinite(value) for value in coords):
                fail(f"non-finite node coordinates at {nid}")
            nodes[nid] = coords  # type: ignore[assignment]
        elif mode == "*ELEMENT_SOLID" and len(fields) == 10:
            try:
                eid = int(fields[0])
                pid = int(fields[1])
                connectivity = tuple(int(value) for value in fields[2:10])
            except ValueError:
                continue
            if eid in elements:
                fail(f"duplicate element ID {eid}")
            elements[eid] = (pid, connectivity)
    return nodes, elements


def part_sets(deck_text: str) -> dict[int, list[int]]:
    sets: dict[int, list[int]] = {}
    for keyword, data in keyword_blocks(deck_text):
        if keyword != "*SET_PART_LIST_TITLE":
            continue
        sid = int(split_fields(data[1])[0])
        values: list[int] = []
        for line in data[2:]:
            values.extend(int(value) for value in split_fields(line) if value)
        if sid in sets:
            fail(f"duplicate part set ID {sid}")
        sets[sid] = values
    return sets


def read_single_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        fail(f"R06 job must contain exactly one case-map row, found {len(rows)}")
    return rows[0]


def assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        fail(f"{label} differs: {actual} != {expected}")


def validate_job(job_dir: Path, root_design: dict, historical_mesh: tuple[dict, dict]) -> dict:
    manifest = strict_json(job_dir / "manifest.json")
    contract = strict_json(job_dir / "job_contract.json")
    job_id = manifest["job_id"]
    matching_jobs = [
        job
        for experiment in root_design["paired_experiments"]
        for job in experiment["jobs"]
        if job["job_id"] == job_id
    ]
    if len(matching_jobs) != 1:
        fail(f"{job_id}: design must contain exactly one matching job")
    design_job = matching_jobs[0]
    deck_path = job_dir / manifest["deck"]["path"]
    allowed = job_base_files(manifest) | {deck_path.name, "static_validation.json"}
    actual_files = {path.name for path in job_dir.iterdir() if path.is_file()}
    if actual_files not in (allowed - {"static_validation.json"}, allowed):
        fail(f"{job_id}: package files differ from explicit whitelist: {sorted(actual_files)}")
    expected_revision = root_design["package_revision"]
    if manifest["package_revision"] != expected_revision or contract["package_revision"] != expected_revision:
        fail(f"{job_id}: revision mismatch or non-R06 identity")
    if not deck_path.is_file():
        fail(f"{job_id}: missing head deck {deck_path.name}")
    if manifest["deck"]["sha256"] != sha256(deck_path):
        fail(f"{job_id}: deck hash mismatch")
    if manifest["deck"]["size_bytes"] != deck_path.stat().st_size:
        fail(f"{job_id}: deck size mismatch")
    for field in ("design", "material", "successor_contract", "case_map", "job_contract"):
        entry = manifest[field]
        if sha256(job_dir / entry["path"]) != entry["sha256"]:
            fail(f"{job_id}: {field} hash mismatch")
    if strict_json(job_dir / manifest["design"]["path"]) != root_design:
        fail(f"{job_id}: copied design differs from batch design")
    if contract["head_deck"] != deck_path.name:
        fail(f"{job_id}: head-deck binding mismatch")
    resources = contract["resource_contract"]
    if (resources["nodes"], resources["ranks"], resources["head_memory_mb"], resources["child_memory_mb"]) != (1, 32, 512, 256):
        fail(f"{job_id}: frozen resource identity mismatch")
    if resources["command_line_parameters"] != "":
        fail(f"{job_id}: command-line parameters must remain empty")

    deck_text = deck_path.read_text(encoding="ascii")
    input_format = validate_keyword_input_format(deck_text)
    schema = validate_schema(deck_text, root_design)
    grouped = grouped_blocks(deck_text)
    row = read_single_map(job_dir / "p0_case_part_map.csv")
    if row["job_id"] != job_id or row["case_id"] != contract["case"]["case_id"]:
        fail(f"{job_id}: case-map identity mismatch")

    termination = one_card(grouped, "*CONTROL_TERMINATION", "FIFFF")
    timestep = one_card(grouped, "*CONTROL_TIMESTEP", "FFIFFIII")
    solid = one_card(grouped, "*CONTROL_SOLID", "IIIIIFII")
    expected_tssfac = float(
        design_job.get(
            "tssfac", root_design["frozen_solver_controls"]["control_timestep"]["tssfac"]
        )
    )
    if (
        timestep[0] != "0.000000000e+00"
        or not math.isclose(float(timestep[1]), expected_tssfac, rel_tol=0.0, abs_tol=1e-12)
        or timestep[2:] != [
            "0",
            "0.000000000e+00",
            "0.000000000e+00",
            "0",
            "0",
            "0",
        ]
    ):
        fail(f"{job_id}: CONTROL_TIMESTEP differs from frozen no-mass-scaling/no-timestep-erosion card")
    assert_close(float(termination[0]), float(contract["case"]["termination_time_ms"]), f"{job_id} termination")
    if int(termination[1]) != 0 or float(termination[2]) != 0.0:
        fail(f"{job_id}: CONTROL_TERMINATION must keep ENDCYC=0 and DTMIN=0")
    expected_psfail = int(contract["case"]["psfail_part_set_id"])
    if int(solid[4]) != expected_psfail or solid[:4] != ["0", "0", "0", "0"] or solid[5:] != ["0.000000000e+00", "0", "0"]:
        fail(f"{job_id}: CONTROL_SOLID differs outside the frozen PSFAIL field")

    d3part = one_card(grouped, "*DATABASE_BINARY_D3PART", "FIIII")
    if d3part != ["5.000000000e-05", "0", "0", "0", "900001"]:
        fail(f"{job_id}: D3PART must use the exact R15 five-field R06 contract")
    measure = root_design["shared_measurement_upgrade"]
    direct_contact_policy = measure.get(
        "direct_contact_output_policy",
        {"request_binary_intfor": True, "request_ascii_ncforc": True},
    )
    request_intfor = bool(direct_contact_policy["request_binary_intfor"])
    request_ncforc = bool(direct_contact_policy["request_ascii_ncforc"])
    d3part_extent_contract = measure.get("d3part_extent_cards", [])
    parser_contract = measure.get("r15_parser_contract")
    extent_binary = [split_fields(line) for line in grouped["*DATABASE_EXTENT_BINARY"][0]]
    if parser_contract is None:
        expected_extent_binary = [["6", "0", "0", "0", "0", "0", "0", "0"]]
        expected_intfor_timing = ["5.000000000e-05", "0", "0", "0", "0"]
    else:
        required_parser_contract = {
            "target_solver": "LS-DYNA R15.0.2",
            "database_extent_binary_cards": [
                [6, 0, 3, 0, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 2, 1],
            ],
            "intfor_file_timing_card_tail": [0, 0, 0],
            "intfor_file_ioopt_card": [0],
        }
        if parser_contract != required_parser_contract:
            fail(f"{job_id}: frozen R15.0.2 parser contract differs")
        expected_extent_binary = [
            [str(value) for value in values]
            for values in parser_contract["database_extent_binary_cards"]
        ]
        expected_intfor_timing = ["5.000000000e-05", "0", "0", "0"]
    if extent_binary != expected_extent_binary:
        fail(f"{job_id}: DATABASE_EXTENT_BINARY cards differ from the frozen contract")
    extent_d3part = [
        split_fields(line) for line in grouped.get("*DATABASE_EXTENT_D3PART", [[]])[0]
    ]
    expected_extent_d3part = [
        [str(value) for value in values] for values in d3part_extent_contract
    ]
    if extent_d3part != expected_extent_d3part:
        fail(f"{job_id}: DATABASE_EXTENT_D3PART cards differ from the frozen contract")
    if request_intfor:
        intfor_file = grouped["*DATABASE_BINARY_INTFOR_FILE"][0]
        if (
            len(intfor_file) != 3
            or intfor_file[0] != "intfor"
            or split_fields(intfor_file[1]) != expected_intfor_timing
            or split_fields(intfor_file[2]) != ["0"]
        ):
            fail(f"{job_id}: INTFOR_FILE identity or interval differs")
        if grouped["*DATABASE_EXTENT_INTFOR"][0] != ["1,1,1,1,1,1,0,0", "0,0,0,0,0,1"]:
            fail(f"{job_id}: EXTENT_INTFOR content differs")
    contact = grouped["*CONTACT_ERODING_SINGLE_SURFACE"][0]
    if split_fields(contact[0])[-2:] != ["1", "1"]:
        fail(f"{job_id}: contact SAPR/SBPR output flags are not both enabled")
    if split_fields(contact[3]) != ["1", "1", "1"]:
        fail(f"{job_id}: eroding-contact ISYM/EROSOP/IADJ differs from R05")
    expected_contact_soft = int(design_job.get("contact_soft", 2))
    contact_card_a = split_fields(contact[4])
    if (
        int(contact_card_a[0]) != expected_contact_soft
        or contact_card_a[1:] != ["0.000000000e+00", "0", "0", "3", "5", "0", "0"]
    ):
        fail(f"{job_id}: eroding-contact Optional Card A differs outside the frozen SOFT factor")

    plate_pid = int(row["plate_part_id"])
    fragment_pid = int(row["fragment_part_id"])
    sets = part_sets(deck_text)
    if sets != {900001: [fragment_pid], 910001: [plate_pid]}:
        fail(f"{job_id}: R06 part-set identities are not exact: {sets}")
    if expected_psfail not in (0, 910001):
        fail(f"{job_id}: PSFAIL is neither default 0 nor plate-only set 910001")

    nodes, elements = parse_mesh(deck_text)
    if len(nodes) != manifest["deck"]["node_count"] or len(elements) != manifest["deck"]["element_count"]:
        fail(f"{job_id}: mesh count differs from manifest")
    if set(pid for pid, _ in elements.values()) != {plate_pid, fragment_pid}:
        fail(f"{job_id}: deck contains an unexpected solid PID")
    referenced = {nid for _, connectivity in elements.values() for nid in connectivity}
    if not referenced <= set(nodes):
        fail(f"{job_id}: element references missing nodes")
    hist_nodes, hist_elements = historical_mesh
    for eid, value in elements.items():
        if hist_elements.get(eid) != value:
            fail(f"{job_id}: element {eid} differs from immutable R05 mesh/connectivity")
        for nid in value[1]:
            if hist_nodes.get(nid) != nodes[nid]:
                fail(f"{job_id}: node {nid} differs from immutable R05 coordinates")
    if sorted(nodes) != list(range(int(manifest["deck"]["node_id_span"][0]), int(manifest["deck"]["node_id_span"][1]) + 1)):
        fail(f"{job_id}: node ID span is not contiguous")
    if sorted(elements) != list(range(int(manifest["deck"]["element_id_span"][0]), int(manifest["deck"]["element_id_span"][1]) + 1)):
        fail(f"{job_id}: element ID span is not contiguous")
    plate_nodes = {nid for pid, conn in elements.values() if pid == plate_pid for nid in conn}
    fragment_nodes = {nid for pid, conn in elements.values() if pid == fragment_pid for nid in conn}
    plate_top = max(nodes[nid][2] for nid in plate_nodes)
    fragment_bottom = min(nodes[nid][2] for nid in fragment_nodes)
    assert_close(plate_top, 0.0, f"{job_id} plate top")
    assert_close(fragment_bottom - plate_top, 1.0, f"{job_id} initial gap")
    if Counter(pid for pid, _ in elements.values()) != Counter(
        {plate_pid: int(row["plate_elements"]), fragment_pid: int(row["fragment_elements"])}
    ):
        fail(f"{job_id}: per-part element counts differ from case map")

    speed_card = split_fields(grouped["*INITIAL_VELOCITY_GENERATION"][0][0])
    if int(speed_card[0]) != fragment_pid or int(speed_card[1]) != 2:
        fail(f"{job_id}: initial velocity is not bound to the fragment PID")
    assert_close(-float(speed_card[5]), float(row["impact_speed_mps"]), f"{job_id} impact speed")
    r06e_physics_parity = None
    r06f_timestep_isolation = None
    r06g_timestep_isolation = None
    r06h_timestep_isolation = None
    r06i_timestep_isolation = None
    r06j_fragment_erod_isolation = None
    r06k_fragment_psfail_isolation = None
    r06l_contact_soft_isolation = None
    if root_design["design_status"] == R06F_RECANARY_STATUS:
        if sha256(HISTORICAL_R06E_0800_DECK) != HISTORICAL_R06E_0800_DECK_SHA256:
            fail("immutable returned R06e 800 m/s deck hash changed")
        measurement_keywords = {
            "*TITLE",
            "*DATABASE_BINARY_D3PLOT",
            "*DATABASE_EXTENT_BINARY",
            "*DATABASE_BINARY_D3PART",
            "*DATABASE_EXTENT_D3PART",
            "*DATABASE_BINARY_INTFOR_FILE",
            "*DATABASE_EXTENT_INTFOR",
            "*DATABASE_GLSTAT",
            "*DATABASE_MATSUM",
            "*DATABASE_SLEOUT",
            "*DATABASE_NODOUT",
            "*DATABASE_NCFORC",
        }
        parent_text = HISTORICAL_R06E_0800_DECK.read_text(encoding="ascii")
        if comparable_blocks(deck_text, measurement_keywords) != comparable_blocks(
            parent_text, measurement_keywords
        ):
            fail(f"{job_id}: physics/solver-control blocks differ from returned R06e control")
        r06e_physics_parity = {
            "status": "pass",
            "parent_deck_sha256": HISTORICAL_R06E_0800_DECK_SHA256,
            "allowed_difference_scope": "title_and_database_output_blocks_only",
        }
    elif root_design["design_status"] == R06F_TIMESTEP_STATUS:
        if sha256(HISTORICAL_R06F_0800_DECK) != HISTORICAL_R06F_0800_DECK_SHA256:
            fail("immutable released R06f 800 m/s control deck hash changed")
        baseline_text = HISTORICAL_R06F_0800_DECK.read_text(encoding="ascii")
        ignored = {"*TITLE", "*CONTROL_TIMESTEP"}
        if comparable_blocks(deck_text, ignored) != comparable_blocks(baseline_text, ignored):
            fail(f"{job_id}: deck differs from released R06f control outside TITLE/TSSFAC")
        baseline_timestep = split_fields(
            grouped_blocks(baseline_text)["*CONTROL_TIMESTEP"][0][0]
        )
        differing_fields = [
            index + 1
            for index, (baseline_value, candidate_value) in enumerate(
                zip(baseline_timestep, timestep, strict=True)
            )
            if baseline_value != candidate_value
        ]
        if (
            differing_fields != [2]
            or not math.isclose(float(baseline_timestep[1]), 0.8, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(timestep[1]), 0.4, rel_tol=0.0, abs_tol=1e-12)
        ):
            fail(f"{job_id}: timestep factor must change only field 2 TSSFAC, 0.8 -> 0.4")
        r06f_timestep_isolation = {
            "status": "pass",
            "parent_job_id": "r06f_0800_ctrl",
            "parent_platform_job_id": "72617857",
            "parent_deck_sha256": HISTORICAL_R06F_0800_DECK_SHA256,
            "only_changed_field": "CONTROL_TIMESTEP.TSSFAC",
            "control_level": 0.8,
            "treatment_level": 0.4,
            "dt2ms": 0.0,
            "allowed_difference_scope": "TITLE and CONTROL_TIMESTEP field 2 only",
        }
    elif root_design["design_status"] == R06G_REFINEMENT_STATUS:
        if sha256(HISTORICAL_R06F_TSSFAC04_DECK) != HISTORICAL_R06F_TSSFAC04_DECK_SHA256:
            fail("immutable released R06f TSSFAC=0.4 deck hash changed")
        baseline_text = HISTORICAL_R06F_TSSFAC04_DECK.read_text(encoding="ascii")
        ignored = {"*TITLE", "*CONTROL_TIMESTEP"}
        if comparable_blocks(deck_text, ignored) != comparable_blocks(baseline_text, ignored):
            fail(f"{job_id}: deck differs from released TSSFAC=0.4 parent outside TITLE/TSSFAC")
        baseline_timestep = split_fields(
            grouped_blocks(baseline_text)["*CONTROL_TIMESTEP"][0][0]
        )
        differing_fields = [
            index + 1
            for index, (baseline_value, candidate_value) in enumerate(
                zip(baseline_timestep, timestep, strict=True)
            )
            if baseline_value != candidate_value
        ]
        if (
            differing_fields != [2]
            or not math.isclose(float(baseline_timestep[1]), 0.4, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(timestep[1]), 0.2, rel_tol=0.0, abs_tol=1e-12)
        ):
            fail(f"{job_id}: timestep factor must change only field 2 TSSFAC, 0.4 -> 0.2")
        r06g_timestep_isolation = {
            "status": "pass",
            "parent_job_id": "r06f_0800_tssfac04",
            "parent_platform_job_id": "72630855",
            "parent_deck_sha256": HISTORICAL_R06F_TSSFAC04_DECK_SHA256,
            "only_changed_field": "CONTROL_TIMESTEP.TSSFAC",
            "control_level": 0.4,
            "treatment_level": 0.2,
            "dt2ms": 0.0,
            "allowed_difference_scope": "TITLE and CONTROL_TIMESTEP field 2 only",
        }
    elif root_design["design_status"] == R06H_CONFIRMATION_STATUS:
        if sha256(HISTORICAL_R06G_TSSFAC02_DECK) != HISTORICAL_R06G_TSSFAC02_DECK_SHA256:
            fail("immutable returned R06g TSSFAC=0.2 deck hash changed")
        baseline_text = HISTORICAL_R06G_TSSFAC02_DECK.read_text(encoding="ascii")
        ignored = {"*TITLE", "*CONTROL_TIMESTEP"}
        if comparable_blocks(deck_text, ignored) != comparable_blocks(baseline_text, ignored):
            fail(f"{job_id}: deck differs from returned TSSFAC=0.2 parent outside TITLE/TSSFAC")
        baseline_timestep = split_fields(
            grouped_blocks(baseline_text)["*CONTROL_TIMESTEP"][0][0]
        )
        differing_fields = [
            index + 1
            for index, (baseline_value, candidate_value) in enumerate(
                zip(baseline_timestep, timestep, strict=True)
            )
            if baseline_value != candidate_value
        ]
        if (
            differing_fields != [2]
            or not math.isclose(float(baseline_timestep[1]), 0.2, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(timestep[1]), 0.1, rel_tol=0.0, abs_tol=1e-12)
        ):
            fail(f"{job_id}: timestep factor must change only field 2 TSSFAC, 0.2 -> 0.1")
        r06h_timestep_isolation = {
            "status": "pass",
            "parent_job_id": "r06g_0800_tssfac02",
            "parent_platform_job_id": "72639935",
            "parent_deck_sha256": HISTORICAL_R06G_TSSFAC02_DECK_SHA256,
            "only_changed_field": "CONTROL_TIMESTEP.TSSFAC",
            "control_level": 0.2,
            "treatment_level": 0.1,
            "dt2ms": 0.0,
            "allowed_difference_scope": "TITLE and CONTROL_TIMESTEP field 2 only",
        }
    elif root_design["design_status"] == R06I_TERMINAL_REFINEMENT_STATUS:
        if sha256(HISTORICAL_R06H_TSSFAC01_DECK) != HISTORICAL_R06H_TSSFAC01_DECK_SHA256:
            fail("immutable returned R06h TSSFAC=0.1 deck hash changed")
        baseline_text = HISTORICAL_R06H_TSSFAC01_DECK.read_text(encoding="ascii")
        ignored = {"*TITLE", "*CONTROL_TIMESTEP"}
        if comparable_blocks(deck_text, ignored) != comparable_blocks(baseline_text, ignored):
            fail(f"{job_id}: deck differs from returned TSSFAC=0.1 parent outside TITLE/TSSFAC")
        baseline_timestep = split_fields(
            grouped_blocks(baseline_text)["*CONTROL_TIMESTEP"][0][0]
        )
        differing_fields = [
            index + 1
            for index, (baseline_value, candidate_value) in enumerate(
                zip(baseline_timestep, timestep, strict=True)
            )
            if baseline_value != candidate_value
        ]
        if (
            differing_fields != [2]
            or not math.isclose(float(baseline_timestep[1]), 0.1, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(timestep[1]), 0.05, rel_tol=0.0, abs_tol=1e-12)
        ):
            fail(f"{job_id}: timestep factor must change only field 2 TSSFAC, 0.1 -> 0.05")
        r06i_timestep_isolation = {
            "status": "pass",
            "parent_job_id": "r06h_0800_tssfac01",
            "parent_platform_job_id": "72701744",
            "parent_deck_sha256": HISTORICAL_R06H_TSSFAC01_DECK_SHA256,
            "only_changed_field": "CONTROL_TIMESTEP.TSSFAC",
            "control_level": 0.1,
            "treatment_level": 0.05,
            "dt2ms": 0.0,
            "allowed_difference_scope": "TITLE and CONTROL_TIMESTEP field 2 only",
        }
    elif root_design["design_status"] == R06J_FRAGMENT_EROSION_STATUS:
        if sha256(HISTORICAL_R06I_TSSFAC005_DECK) != HISTORICAL_R06I_TSSFAC005_DECK_SHA256:
            fail("immutable returned R06i TSSFAC=0.05 deck hash changed")
        baseline_text = HISTORICAL_R06I_TSSFAC005_DECK.read_text(encoding="ascii")
        ignored = {"*TITLE", "*MAT_JOHNSON_COOK"}
        if comparable_blocks(deck_text, ignored) != comparable_blocks(baseline_text, ignored):
            fail(f"{job_id}: deck differs from returned R06i parent outside TITLE/fragment MAT_015")
        baseline_mats = grouped_blocks(baseline_text)["*MAT_JOHNSON_COOK"]
        candidate_mats = grouped["*MAT_JOHNSON_COOK"]
        if len(baseline_mats) != 2 or len(candidate_mats) != 2:
            fail(f"{job_id}: expected exactly plate and fragment MAT_015 blocks")
        if candidate_mats[0] != baseline_mats[0]:
            fail(f"{job_id}: plate MAT_015 changed in a fragment-only diagnostic")
        if candidate_mats[1][:3] != baseline_mats[1][:3]:
            fail(f"{job_id}: fragment MAT_015 changed outside Card 4")
        baseline_card4 = split_fields(baseline_mats[1][3])
        candidate_card4 = split_fields(candidate_mats[1][3])
        differing_fields = [
            index + 1
            for index, (baseline_value, candidate_value) in enumerate(
                zip(baseline_card4, candidate_card4, strict=True)
            )
            if baseline_value != candidate_value
        ]
        if (
            differing_fields != [3]
            or not math.isclose(float(baseline_card4[2]), 0.0, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(candidate_card4[2]), 1.0, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(timestep[1]), 0.05, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(timestep[4]), 0.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            fail(f"{job_id}: only fragment MAT_015 Card 4 field 3 EROD may change 0 -> 1")
        r06j_fragment_erod_isolation = {
            "status": "pass",
            "parent_job_id": "r06i_0800_tssfac005",
            "parent_platform_job_id": "72740022",
            "parent_deck_sha256": HISTORICAL_R06I_TSSFAC005_DECK_SHA256,
            "only_changed_field": "fragment MAT_015 Card 4 field 3 EROD",
            "control_level": 0,
            "treatment_level": 1,
            "plate_erod": 0,
            "tssfac": 0.05,
            "dt2ms": 0.0,
            "allowed_difference_scope": "TITLE and fragment MAT_015 Card 4 field 3 only",
        }
    elif root_design["design_status"] == R06K_FRAGMENT_NEGATIVE_VOLUME_STATUS:
        if sha256(HISTORICAL_R06J_FRAGMENT_EROD1_DECK) != HISTORICAL_R06J_FRAGMENT_EROD1_DECK_SHA256:
            fail("immutable returned R06j fragment EROD=1 deck hash changed")
        baseline_text = HISTORICAL_R06J_FRAGMENT_EROD1_DECK.read_text(encoding="ascii")
        ignored = {"*TITLE", "*CONTROL_SOLID"}
        if comparable_blocks(deck_text, ignored) != comparable_blocks(baseline_text, ignored):
            fail(f"{job_id}: deck differs from returned R06j parent outside TITLE/CONTROL_SOLID PSFAIL")
        baseline_solid = split_fields(
            grouped_blocks(baseline_text)["*CONTROL_SOLID"][0][0]
        )
        differing_fields = [
            index + 1
            for index, (baseline_value, candidate_value) in enumerate(
                zip(baseline_solid, solid, strict=True)
            )
            if baseline_value != candidate_value
        ]
        fragment_mat_card4 = split_fields(grouped["*MAT_JOHNSON_COOK"][1][3])
        if (
            differing_fields != [5]
            or int(baseline_solid[4]) != 0
            or int(solid[4]) != 910001
            or int(fragment_mat_card4[2]) != 1
            or not math.isclose(float(timestep[1]), 0.05, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(timestep[4]), 0.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            fail(f"{job_id}: only CONTROL_SOLID Card 1 field 5 PSFAIL may change 0 -> 910001")
        r06k_fragment_psfail_isolation = {
            "status": "pass",
            "parent_job_id": "r06j_0800_fragment_erod1",
            "parent_platform_job_id": "72751725",
            "parent_deck_sha256": HISTORICAL_R06J_FRAGMENT_EROD1_DECK_SHA256,
            "only_changed_field": "CONTROL_SOLID Card 1 field 5 PSFAIL",
            "control_level": 0,
            "treatment_level": 910001,
            "treatment_part_set": [1002],
            "excluded_fragment_part": 2002,
            "fragment_mat015_erod": 1,
            "tssfac": 0.05,
            "dt2ms": 0.0,
            "allowed_difference_scope": "TITLE and CONTROL_SOLID Card 1 field 5 only",
        }
    elif root_design["design_status"] == R06L_CONTACT_SOFT1_STATUS:
        if sha256(HISTORICAL_R06K_FRAGMENT_PSFAIL_DECK) != HISTORICAL_R06K_FRAGMENT_PSFAIL_DECK_SHA256:
            fail("immutable returned R06k fragment PSFAIL deck hash changed")
        baseline_text = HISTORICAL_R06K_FRAGMENT_PSFAIL_DECK.read_text(encoding="ascii")
        ignored = {"*TITLE", "*CONTACT_ERODING_SINGLE_SURFACE"}
        if comparable_blocks(deck_text, ignored) != comparable_blocks(baseline_text, ignored):
            fail(f"{job_id}: deck differs from returned R06k parent outside TITLE/contact SOFT")
        baseline_contact = grouped_blocks(baseline_text)["*CONTACT_ERODING_SINGLE_SURFACE"][0]
        candidate_contact = grouped["*CONTACT_ERODING_SINGLE_SURFACE"][0]
        if baseline_contact[:4] != candidate_contact[:4]:
            fail(f"{job_id}: contact changed outside Optional Card A")
        baseline_card_a = split_fields(baseline_contact[4])
        candidate_card_a = split_fields(candidate_contact[4])
        differing_fields = [
            index + 1
            for index, (baseline_value, candidate_value) in enumerate(
                zip(baseline_card_a, candidate_card_a, strict=True)
            )
            if baseline_value != candidate_value
        ]
        fragment_mat_card4 = split_fields(grouped["*MAT_JOHNSON_COOK"][1][3])
        if (
            differing_fields != [1]
            or int(baseline_card_a[0]) != 2
            or int(candidate_card_a[0]) != 1
            or int(solid[4]) != 910001
            or int(fragment_mat_card4[2]) != 1
            or not math.isclose(float(timestep[1]), 0.05, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(timestep[4]), 0.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            fail(f"{job_id}: only contact Optional Card A field 1 SOFT may change 2 -> 1")
        r06l_contact_soft_isolation = {
            "status": "pass",
            "parent_job_id": "r06k_0800_fragment_psfail_plate",
            "parent_platform_job_id": "72764387",
            "parent_deck_sha256": HISTORICAL_R06K_FRAGMENT_PSFAIL_DECK_SHA256,
            "only_changed_field": "CONTACT_ERODING_SINGLE_SURFACE Optional Card A field 1 SOFT",
            "control_level": 2,
            "treatment_level": 1,
            "psfail_part_set_id": 910001,
            "fragment_mat015_erod": 1,
            "plate_mat015_erod": 0,
            "tssfac": 0.05,
            "dt2ms": 0.0,
            "allowed_difference_scope": "TITLE and CONTACT_ERODING_SINGLE_SURFACE Optional Card A field 1 only",
        }
    readme = (job_dir / "UPLOAD_README.txt").read_text(encoding="utf-8")
    for required in (deck_path.name, manifest["deck"]["sha256"], "nodes=1", "ranks=32", "command_line_parameters=<empty"):
        if required not in readme:
            fail(f"{job_id}: upload README is missing {required!r}")

    is_r06f = root_design["design_status"] in SINGLE_JOB_R06F_STATUSES
    if is_r06f:
        output_contract_report = {
            "status": "pass",
            "d3part_fields": d3part,
            "neiph": 6,
            **(
                {"extent_binary_cards": extent_binary}
                if parser_contract is not None
                else {}
            ),
            "extent_d3part_cards": extent_d3part,
            "contact_print_flags": [1, 1],
            "binary_intfor_requested": request_intfor,
            "ncforc_requested_best_effort": request_ncforc,
            "missing_direct_contact_is_zero": False,
        }
        limitations = [
            "Static PASS does not establish LS-DYNA parser or runtime acceptance.",
            "MAT_015 HVAR6 readability and D3PART content remain untested until target-solver return.",
            "INTFOR is unavailable for the eroding contact; missing NCFORC is nonblocking and never zero.",
            "This diagnostic job is not an admissible P0 response or training label.",
        ]
    else:
        output_contract_report = {
            "status": "pass",
            "d3part_fields": d3part,
            "neiph": 6,
            **(
                {"extent_binary_cards": extent_binary}
                if parser_contract is not None
                else {}
            ),
            "contact_print_flags": [1, 1],
            "intfor_filename": "intfor",
            "npresu": 1,
            "ngapc": 1,
            "nforc": 1,
            "npen": 1,
        }
        limitations = [
            "Static PASS does not establish LS-DYNA parser or runtime acceptance.",
            "INTFOR and MAT_015 history-variable readability remain untested until target-solver return.",
            "This diagnostic job is not an admissible P0 response or training label.",
        ]

    return {
        "schema_version": "hf-p0-r06-static-validation@1.0.0",
        "package_revision": expected_revision,
        "job_id": job_id,
        "overall_status": "pass",
        "solver_validation_status": "not_run_no_local_lsdyna_executable",
        "checks": {
            "keyword_input_format": input_format,
            "keyword_schema": schema,
            "manifest_and_hashes": {"status": "pass"},
            "single_case_identity": {"status": "pass", "case_id": row["case_id"]},
            "immutable_r05_mesh_parity": {
                "status": "pass",
                "historical_deck_sha256": HISTORICAL_R05_DECK_SHA256,
                "node_count": len(nodes),
                "element_count": len(elements),
            },
            "r06_output_contract": output_contract_report,
            **({"immutable_r06e_physics_parity": r06e_physics_parity} if r06e_physics_parity else {}),
            **(
                {"immutable_r06f_control_factor_isolation": r06f_timestep_isolation}
                if r06f_timestep_isolation
                else {}
            ),
            **(
                {"immutable_tssfac04_parent_factor_isolation": r06g_timestep_isolation}
                if r06g_timestep_isolation
                else {}
            ),
            **(
                {"immutable_tssfac02_parent_factor_isolation": r06h_timestep_isolation}
                if r06h_timestep_isolation
                else {}
            ),
            **(
                {"immutable_tssfac01_parent_factor_isolation": r06i_timestep_isolation}
                if r06i_timestep_isolation
                else {}
            ),
            **(
                {"immutable_r06i_fragment_erod_factor_isolation": r06j_fragment_erod_isolation}
                if r06j_fragment_erod_isolation
                else {}
            ),
            **(
                {"immutable_r06j_fragment_psfail_factor_isolation": r06k_fragment_psfail_isolation}
                if r06k_fragment_psfail_isolation
                else {}
            ),
            **(
                {"immutable_r06k_contact_soft_factor_isolation": r06l_contact_soft_isolation}
                if r06l_contact_soft_isolation
                else {}
            ),
            "no_mass_scaling_or_timestep_erosion": {"status": "pass", "dt2ms": 0.0, "erode": 0},
            "psfail_scope": {"status": "pass", "psfail": expected_psfail, "plate_only_set": [plate_pid]},
            "resource_identity": {"status": "pass", "nodes": 1, "ranks": 32},
        },
        "limitations": limitations,
    }


def comparable_blocks(deck_text: str, ignored: set[str]) -> list[tuple[str, list[str]]]:
    return [(key, data) for key, data in keyword_blocks(deck_text) if key not in ignored]


def pair_isolation(batch_dir: Path, design: dict, reports: dict[str, dict]) -> dict:
    checks: list[dict] = []
    for experiment in design["paired_experiments"]:
        if len(experiment["jobs"]) == 1:
            job = experiment["jobs"][0]
            status = (
                "pass_external_baseline_factor_isolation"
                if experiment["factor"] in {
                    "control_timestep_tssfac",
                    "fragment_mat015_erod_flag",
                    "control_solid_psfail_part_set",
                    "contact_optional_card_a_soft",
                }
                else "not_applicable_single_recanary"
            )
            checks.append(
                {
                    "experiment_id": experiment["experiment_id"],
                    "factor": experiment["factor"],
                    "job_id": job["job_id"],
                    "status": status,
                }
            )
            continue
        if len(experiment["jobs"]) != 2:
            fail(f"{experiment['experiment_id']}: expected one recanary or one paired contrast")
        control, treatment = experiment["jobs"]
        control_dir = batch_dir / control["directory"]
        treatment_dir = batch_dir / treatment["directory"]
        control_manifest = strict_json(control_dir / "manifest.json")
        treatment_manifest = strict_json(treatment_dir / "manifest.json")
        control_text = (control_dir / control_manifest["deck"]["path"]).read_text(encoding="ascii")
        treatment_text = (treatment_dir / treatment_manifest["deck"]["path"]).read_text(encoding="ascii")
        factor = experiment["factor"]
        ignored = {"*TITLE", "*CONTROL_TERMINATION"} if factor == "termination_time_ms" else {"*TITLE", "*CONTROL_SOLID"}
        if comparable_blocks(control_text, ignored) != comparable_blocks(treatment_text, ignored):
            fail(f"{experiment['experiment_id']}: decks differ outside the preregistered factor")
        c = grouped_blocks(control_text)
        t = grouped_blocks(treatment_text)
        if factor == "termination_time_ms":
            if c["*CONTROL_TERMINATION"] == t["*CONTROL_TERMINATION"]:
                fail(f"{experiment['experiment_id']}: termination treatment has no factor difference")
            if c["*CONTROL_SOLID"] != t["*CONTROL_SOLID"]:
                fail(f"{experiment['experiment_id']}: termination pair also changes PSFAIL")
        elif factor == "control_solid_psfail_part_set":
            c_card = split_fields(c["*CONTROL_SOLID"][0][0])
            t_card = split_fields(t["*CONTROL_SOLID"][0][0])
            differences = [index for index, values in enumerate(zip(c_card, t_card), start=1) if values[0] != values[1]]
            if differences != [5] or c_card[4] != "0" or t_card[4] != "910001":
                fail(f"{experiment['experiment_id']}: PSFAIL pair must differ only in Card 1 field 5, 0 -> 910001")
        else:
            fail(f"unrecognized R06 factor: {factor}")
        checks.append(
            {
                "experiment_id": experiment["experiment_id"],
                "factor": factor,
                "control_job_id": control["job_id"],
                "treatment_job_id": treatment["job_id"],
                "status": "pass",
            }
        )
    return {"status": "pass", "pairs": checks}


def validate_batch(
    batch_dir: Path,
    *,
    write_reports: bool = False,
    allow_known_failed_revision: bool = False,
) -> dict:
    batch_dir = batch_dir.resolve()
    batch_manifest = strict_json(batch_dir / "batch_manifest.json")
    design_path = batch_dir / batch_manifest["design"]["path"]
    design = strict_json(design_path)
    material_path = batch_dir / batch_manifest["material"]["path"]
    successor_path = batch_dir / batch_manifest["successor_contract"]["path"]
    matrix_path = batch_dir / batch_manifest["submission_matrix"]["path"]
    expected_revision = design["package_revision"]
    if batch_manifest["package_revision"] != expected_revision:
        fail("batch revision mismatch")
    if not expected_revision.startswith("R06_"):
        fail("R06 generator identity collision: non-R06 revision")
    if (
        expected_revision == REVISION
        and design["shared_measurement_upgrade"].get("r15_parser_contract") is None
        and not allow_known_failed_revision
    ):
        fail(
            "known target-R15 parser defect: historical R06d package is withdrawn; "
            "use R06_keyword_parser_repair_20261025"
        )
    for field, path in (("design", design_path), ("material", material_path), ("successor_contract", successor_path), ("submission_matrix", matrix_path)):
        if sha256(path) != batch_manifest[field]["sha256"]:
            fail(f"batch {field} hash mismatch")
    if sha256(HISTORICAL_R05_DECK) != HISTORICAL_R05_DECK_SHA256:
        fail("immutable historical R05 deck hash changed")
    historical_mesh = parse_mesh(HISTORICAL_R05_DECK.read_text(encoding="ascii"))

    expected_dirs = [job["directory"] for exp in design["paired_experiments"] for job in exp["jobs"]]
    if sorted(expected_dirs) != sorted(path.name for path in batch_dir.iterdir() if path.is_dir()):
        fail("batch job directories differ from frozen design")
    expected_job_count = sum(
        len(experiment["jobs"]) for experiment in design["paired_experiments"]
    )
    expected_experiment_count = len(design["paired_experiments"])
    if (
        batch_manifest["job_count"] != expected_job_count
        or batch_manifest["paired_experiment_count"] != expected_experiment_count
    ):
        fail("R06 batch job or experiment count differs from the frozen design")
    if design["design_status"] in SINGLE_JOB_R06F_STATUSES and (
        expected_job_count,
        expected_experiment_count,
    ) != (1, 1):
        fail("R06f single-job release must contain exactly one job and one experiment")
    if design["design_status"] not in SINGLE_JOB_R06F_STATUSES and (
        expected_job_count,
        expected_experiment_count,
    ) != (4, 2):
        fail("historical R06 diagnostic design must contain four jobs in two pairs")
    manifest_jobs = {entry["job_id"]: entry for entry in batch_manifest["jobs"]}
    reports: dict[str, dict] = {}
    for experiment in design["paired_experiments"]:
        for job in experiment["jobs"]:
            job_dir = batch_dir / job["directory"]
            entry = manifest_jobs.get(job["job_id"])
            if entry is None or entry["directory"] != job["directory"]:
                fail(f"batch manifest missing job {job['job_id']}")
            if sha256(job_dir / "manifest.json") != entry["manifest_sha256"]:
                fail(f"{job['job_id']}: job manifest hash differs from batch binding")
            report = validate_job(job_dir, design, historical_mesh)
            reports[job["job_id"]] = report
            if write_reports:
                (job_dir / "static_validation.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
    isolation = pair_isolation(batch_dir, design, reports)
    if "wave_0" in design["submission_sequence"]:
        wave_order = {"status": "pass", **design["submission_sequence"]}
    else:
        wave_order = {
            "status": "pass",
            "wave_1": design["submission_sequence"]["wave_1"],
            "wave_2": design["submission_sequence"]["wave_2"],
        }
    is_r06f = design["design_status"] in SINGLE_JOB_R06F_STATUSES
    package_count_check = (
        {"single_case_packages": {"status": "pass", "count": expected_job_count}}
        if is_r06f
        else {"four_single_case_packages": {"status": "pass"}}
    )
    readability_gate = (
        "real D3PART/full-D3PLOT-HVAR6 readability"
        if is_r06f
        else "real D3PART/INTFOR/history-variable readability"
    )
    result = {
        "schema_version": "hf-p0-r06-submission-readiness@1.0.0",
        "package_revision": expected_revision,
        "overall_status": "pass",
        "lifecycle_status": "prepared_not_submitted_not_solver_validated",
        "job_count": len(reports),
        "paired_experiment_count": len(design["paired_experiments"]),
        "checks": {
            "batch_hash_bindings": {"status": "pass"},
            "historical_r05_deck_immutable": {"status": "pass", "sha256": HISTORICAL_R05_DECK_SHA256},
            **package_count_check,
            "per_job_static_validation": {"status": "pass", "jobs": reports},
            "paired_single_factor_isolation": isolation,
            "wave_order": wave_order,
        },
        "unmet_external_gates": [
            "target LS-DYNA R15.0.2 keyword parse",
            "normal solver termination and run-integrity",
            readability_gate,
            "paired response and failure-mechanism analysis",
            "P0 response admission and independent physical validation",
        ],
        "admission": {
            "r06_diagnostic_submission": (
                "ready_for_user_submission_in_staged_canary"
                if "wave_0" in design["submission_sequence"]
                else "ready_for_user_submission_in_two_waves"
            ),
            "p0_response_calibration": "denied",
            "p1_submission": "denied",
            "local_engine_fitting": "denied",
            "reference_dataset_generation": "denied",
            "physical_validity": "not_established",
        },
    }
    if write_reports:
        (batch_dir / "R06_SUBMISSION_READINESS.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the P0 R06 paired diagnostic batch")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-reports", action="store_true")
    parser.add_argument(
        "--allow-known-failed-revision",
        action="store_true",
        help="forensic-only validation of the withdrawn historical R06d package",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_batch(
        args.batch_dir,
        write_reports=args.write_reports,
        allow_known_failed_revision=args.allow_known_failed_revision,
    )
    print(json.dumps({"overall_status": result["overall_status"], "job_count": result["job_count"]}, indent=2))


if __name__ == "__main__":
    main()
