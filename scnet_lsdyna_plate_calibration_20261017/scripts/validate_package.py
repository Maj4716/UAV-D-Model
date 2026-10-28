from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "decks" / "p0_velocity_mesh_calibration.k"
CASE_MAP = ROOT / "decks" / "p0_case_part_map.csv"
DESIGN = ROOT / "design" / "p0_cases.json"
MATERIAL = ROOT / "materials" / "jc4340_reference.json"
P1_DESIGN = ROOT / "design" / "p1_preregistered_cases.csv"
MANIFEST = ROOT / "manifest.json"
REPORT = ROOT / "validation" / "static_validation.json"


NUMERIC_ITEM = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?$"
)
INTEGER_ITEM = re.compile(r"^[+-]?\d+$")


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def fail(message: str) -> None:
    raise AssertionError(message)


def validate_keyword_input_format(deck_text: str) -> dict:
    """Check that serialized numeric cards fit the declared LS-DYNA field mode."""
    substantive = [
        (line_number, raw.strip())
        for line_number, raw in enumerate(deck_text.splitlines(), start=1)
        if raw.strip() and not raw.lstrip().startswith("$")
    ]
    if not substantive:
        fail("empty keyword deck")

    first_line_number, first_line = substantive[0]
    keyword_header = first_line.upper().replace(" ", "")
    if not keyword_header.startswith("*KEYWORD"):
        fail(f"first substantive line {first_line_number} is not *KEYWORD")
    long_mode = "LONG=Y" in keyword_header

    numeric_items: list[tuple[int, str]] = []
    for line_number, raw in enumerate(deck_text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith(("$", "*")):
            continue
        for item in (part.strip() for part in stripped.split(",")):
            if NUMERIC_ITEM.fullmatch(item):
                numeric_items.append((line_number, item))

    standard_width_exceedances = [
        (line_number, item) for line_number, item in numeric_items if len(item) > 10
    ]
    active_width = 20 if long_mode else 10
    active_width_exceedances = [
        (line_number, item) for line_number, item in numeric_items if len(item) > active_width
    ]
    if standard_width_exceedances and not long_mode:
        examples = standard_width_exceedances[:5]
        fail(
            "numeric items exceed the 10-character standard field width but "
            f"*KEYWORD LONG=Y is absent; examples={examples}"
        )
    if active_width_exceedances:
        examples = active_width_exceedances[:5]
        fail(f"numeric items exceed the active {active_width}-character field width: {examples}")

    return {
        "status": "pass",
        "mode": "long" if long_mode else "standard",
        "active_numeric_width": active_width,
        "numeric_item_count": len(numeric_items),
        "maximum_numeric_item_chars": max((len(item) for _, item in numeric_items), default=0),
        "standard_width_exceedance_count": len(standard_width_exceedances),
    }


def keyword_blocks(deck_text: str) -> list[tuple[str, list[str]]]:
    blocks: list[tuple[str, list[str]]] = []
    keyword: str | None = None
    data_lines: list[str] = []
    for raw in deck_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("$"):
            continue
        if line.startswith("*"):
            if keyword is not None:
                blocks.append((keyword, data_lines))
            keyword = line.upper().split()[0]
            data_lines = []
        elif keyword is not None:
            data_lines.append(line)
    if keyword is not None:
        blocks.append((keyword, data_lines))
    return blocks


def split_fields(line: str) -> list[str]:
    return [item.strip() for item in line.split(",")]


def require_types(items: list[str], types: str, label: str) -> None:
    if len(items) != len(types):
        fail(f"{label} requires {len(types)} fields, found {len(items)}")
    for position, (item, item_type) in enumerate(zip(items, types), start=1):
        if item_type == "I" and not INTEGER_ITEM.fullmatch(item):
            fail(f"{label} field {position} requires integer serialization; value={item!r}")
        if item_type == "F" and not NUMERIC_ITEM.fullmatch(item):
            fail(f"{label} field {position} requires numeric serialization; value={item!r}")


def validate_keyword_schema_contracts(deck_text: str) -> dict:
    """Validate every generated R15 keyword block against this deck's frozen card contract.

    This deliberately validates the exact subset emitted by generate_p0.py. It is not a
    general LS-DYNA parser. Integer fields are strict because R15 rejects real-form tokens
    such as 0.000e+00 in fields declared as type I.
    """
    blocks = keyword_blocks(deck_text)
    grouped: dict[str, list[list[str]]] = defaultdict(list)
    for keyword, data in blocks:
        grouped[keyword].append(data)

    expected_counts = {
        "*KEYWORD": 1,
        "*TITLE": 1,
        "*CONTROL_TERMINATION": 1,
        "*CONTROL_TIMESTEP": 1,
        "*CONTROL_ENERGY": 1,
        "*CONTROL_CONTACT": 1,
        "*DATABASE_BINARY_D3PLOT": 1,
        "*DATABASE_GLSTAT": 1,
        "*DATABASE_MATSUM": 1,
        "*DATABASE_SLEOUT": 1,
        "*DATABASE_NODOUT": 1,
        "*CONTACT_ERODING_SINGLE_SURFACE": 1,
        "*SECTION_SOLID_TITLE": 1,
        "*MAT_JOHNSON_COOK": 2,
        "*EOS_GRUNEISEN": 2,
        "*PART": 14,
        "*NODE": 14,
        "*ELEMENT_SOLID": 14,
        "*SET_NODE_LIST_TITLE": 21,
        "*BOUNDARY_SPC_SET": 21,
        "*INITIAL_VELOCITY_GENERATION": 7,
        "*DATABASE_HISTORY_NODE": 1,
        "*END": 1,
    }
    unexpected = sorted(set(grouped) - set(expected_counts))
    if unexpected:
        fail(f"unexpected keyword blocks outside the frozen P0 schema: {unexpected}")
    wrong_counts = {
        keyword: {"expected": expected, "actual": len(grouped.get(keyword, []))}
        for keyword, expected in expected_counts.items()
        if len(grouped.get(keyword, [])) != expected
    }
    if wrong_counts:
        fail(f"keyword block counts differ from the frozen P0 schema: {wrong_counts}")

    def one_card(keyword: str, types: str) -> None:
        data = grouped[keyword][0]
        if len(data) != 1:
            fail(f"{keyword} requires exactly one data card, found {len(data)}")
        require_types(split_fields(data[0]), types, f"{keyword} Card 1")

    if grouped["*KEYWORD"][0] or grouped["*END"][0]:
        fail("*KEYWORD and *END must not contain data cards")
    if len(grouped["*TITLE"][0]) != 1:
        fail("*TITLE requires exactly one title line")

    one_card("*CONTROL_TERMINATION", "F I F F F".replace(" ", ""))
    one_card("*CONTROL_TIMESTEP", "F F I F F I I I".replace(" ", ""))
    one_card("*CONTROL_ENERGY", "IIII")
    for keyword in (
        "*DATABASE_BINARY_D3PLOT", "*DATABASE_GLSTAT", "*DATABASE_MATSUM",
        "*DATABASE_SLEOUT", "*DATABASE_NODOUT",
    ):
        one_card(keyword, "F")

    control = grouped["*CONTROL_CONTACT"][0]
    if len(control) != 2:
        fail(f"*CONTROL_CONTACT requires two data cards, found {len(control)}")
    require_types(split_fields(control[0]), "FFIIIIII", "*CONTROL_CONTACT Card 1")
    require_types(split_fields(control[1]), "IIIIFI II".replace(" ", ""), "*CONTROL_CONTACT Card 2")

    contact = grouped["*CONTACT_ERODING_SINGLE_SURFACE"][0]
    if len(contact) != 5:
        fail(f"*CONTACT_ERODING_SINGLE_SURFACE requires Cards 1-4 and optional Card A; found {len(contact)}")
    for card_number, (line, types) in enumerate(
        zip(contact, ("IIIIIIII", "FFFFFIFF", "FFFFFFFF", "III", "IFIIIIII")),
        start=1,
    ):
        label = "Card A" if card_number == 5 else f"Card {card_number}"
        require_types(split_fields(line), types, f"*CONTACT_ERODING_SINGLE_SURFACE {label}")

    section = grouped["*SECTION_SOLID_TITLE"][0]
    if len(section) != 2:
        fail(f"*SECTION_SOLID_TITLE requires title plus one card, found {len(section)} lines")
    require_types(split_fields(section[1]), "II", "*SECTION_SOLID_TITLE Card 1")

    for index, data in enumerate(grouped["*MAT_JOHNSON_COOK"], start=1):
        if len(data) != 4:
            fail(f"*MAT_JOHNSON_COOK block {index} requires four cards, found {len(data)}")
        for card_number, (line, types) in enumerate(
            zip(data, ("IFFFFFFF", "FFFFFFFF", "FFFFFFFF", "FFFFI")), start=1
        ):
            require_types(split_fields(line), types, f"*MAT_JOHNSON_COOK block {index} Card {card_number}")

    for index, data in enumerate(grouped["*EOS_GRUNEISEN"], start=1):
        if len(data) != 2:
            fail(f"*EOS_GRUNEISEN block {index} requires two cards, found {len(data)}")
        require_types(split_fields(data[0]), "IFFFFFFF", f"*EOS_GRUNEISEN block {index} Card 1")
        require_types(split_fields(data[1]), "F", f"*EOS_GRUNEISEN block {index} Card 2")

    for index, data in enumerate(grouped["*PART"], start=1):
        if len(data) != 2:
            fail(f"*PART block {index} requires title plus one card, found {len(data)} lines")
        require_types(split_fields(data[1]), "IIIIIIII", f"*PART block {index} Card 1")

    for index, data in enumerate(grouped["*NODE"], start=1):
        if not data:
            fail(f"*NODE block {index} is empty")
        for row, line in enumerate(data, start=1):
            require_types(split_fields(line), "IFFF", f"*NODE block {index} row {row}")

    for index, data in enumerate(grouped["*ELEMENT_SOLID"], start=1):
        if not data:
            fail(f"*ELEMENT_SOLID block {index} is empty")
        for row, line in enumerate(data, start=1):
            require_types(split_fields(line), "IIIIIIIIII", f"*ELEMENT_SOLID block {index} row {row}")

    for index, data in enumerate(grouped["*SET_NODE_LIST_TITLE"], start=1):
        if len(data) < 3:
            fail(f"*SET_NODE_LIST_TITLE block {index} requires title, header, and IDs")
        require_types(split_fields(data[1]), "IFFFF", f"*SET_NODE_LIST_TITLE block {index} Card 1")
        for row, line in enumerate(data[2:], start=1):
            items = split_fields(line)
            if not 1 <= len(items) <= 8:
                fail(f"*SET_NODE_LIST_TITLE block {index} ID row {row} has {len(items)} fields")
            require_types(items, "I" * len(items), f"*SET_NODE_LIST_TITLE block {index} ID row {row}")

    for index, data in enumerate(grouped["*BOUNDARY_SPC_SET"], start=1):
        if len(data) != 1:
            fail(f"*BOUNDARY_SPC_SET block {index} requires one card")
        require_types(split_fields(data[0]), "IIIIIIII", f"*BOUNDARY_SPC_SET block {index} Card 1")

    for index, data in enumerate(grouped["*INITIAL_VELOCITY_GENERATION"], start=1):
        if len(data) != 2:
            fail(f"*INITIAL_VELOCITY_GENERATION block {index} requires two cards")
        require_types(split_fields(data[0]), "IIFFFFII", f"*INITIAL_VELOCITY_GENERATION block {index} Card 1")
        require_types(split_fields(data[1]), "FFFFFFII", f"*INITIAL_VELOCITY_GENERATION block {index} Card 2")

    history = grouped["*DATABASE_HISTORY_NODE"][0]
    if not history:
        fail("*DATABASE_HISTORY_NODE is empty")
    history_count = 0
    for row, line in enumerate(history, start=1):
        items = split_fields(line)
        if not 1 <= len(items) <= 8:
            fail(f"*DATABASE_HISTORY_NODE row {row} has {len(items)} fields")
        require_types(items, "I" * len(items), f"*DATABASE_HISTORY_NODE row {row}")
        history_count += len(items)

    return {
        "status": "pass",
        "manual_target": "LS-DYNA R15",
        "validated_keyword_count": len(expected_counts),
        "validated_block_count": sum(expected_counts.values()),
        "integer_fields_strict": True,
        "history_node_count": history_count,
    }


def validate_required_card_layouts(deck_text: str) -> dict:
    blocks = keyword_blocks(deck_text)

    def fields(line: str) -> list[str]:
        return split_fields(line)

    def require_numeric(items: list[str], label: str) -> None:
        bad = [item for item in items if not NUMERIC_ITEM.fullmatch(item)]
        if bad:
            fail(f"{label} contains nonnumeric fields: {bad}")

    def require_integer(items: list[str], positions: list[int], label: str) -> None:
        bad = [(position, items[position - 1]) for position in positions if not INTEGER_ITEM.fullmatch(items[position - 1])]
        if bad:
            fail(f"{label} requires integer serialization at fields {positions}; bad={bad}")

    control_blocks = [data for keyword, data in blocks if keyword == "*CONTROL_CONTACT"]
    if len(control_blocks) != 1:
        fail(f"expected one *CONTROL_CONTACT block, found {len(control_blocks)}")
    control = control_blocks[0]
    if len(control) != 2 or [len(fields(line)) for line in control] != [8, 8]:
        fail("*CONTROL_CONTACT must contain the two required R15 cards with 8 fields each")
    control_card1, control_card2 = (fields(line) for line in control)
    require_numeric(control_card1, "*CONTROL_CONTACT Card 1")
    require_numeric(control_card2, "*CONTROL_CONTACT Card 2")
    require_integer(control_card1, [3, 4, 5, 6, 7, 8], "*CONTROL_CONTACT Card 1")
    require_integer(control_card2, [1, 2, 3, 4, 6, 7, 8], "*CONTROL_CONTACT Card 2")

    velocity_blocks = [
        data for keyword, data in blocks if keyword == "*INITIAL_VELOCITY_GENERATION"
    ]
    malformed_velocity = [
        index
        for index, data in enumerate(velocity_blocks, start=1)
        if len(data) != 2 or [len(fields(line)) for line in data] != [8, 8]
    ]
    if malformed_velocity:
        fail(
            "*INITIAL_VELOCITY_GENERATION requires two 8-field cards; "
            f"malformed block indices={malformed_velocity}"
        )
    for index, data in enumerate(velocity_blocks, start=1):
        velocity_card1, velocity_card2 = (fields(line) for line in data)
        require_numeric(velocity_card1, f"*INITIAL_VELOCITY_GENERATION block {index} Card 1")
        require_numeric(velocity_card2, f"*INITIAL_VELOCITY_GENERATION block {index} Card 2")
        require_integer(
            velocity_card1, [1, 2, 7, 8],
            f"*INITIAL_VELOCITY_GENERATION block {index} Card 1",
        )
        require_integer(
            velocity_card2, [7, 8],
            f"*INITIAL_VELOCITY_GENERATION block {index} Card 2 PHASE/IRIGID",
        )

    eroding_blocks = [
        data for keyword, data in blocks if keyword == "*CONTACT_ERODING_SINGLE_SURFACE"
    ]
    if len(eroding_blocks) != 1 or len(eroding_blocks[0]) < 4:
        fail("expected one complete *CONTACT_ERODING_SINGLE_SURFACE block")
    eroding_card4 = fields(eroding_blocks[0][3])
    if eroding_card4[:3] != ["1", "1", "1"]:
        fail("quarter model requires contact Card 4 ISYM=1, EROSOP=1, IADJ=1")

    d3plot_blocks = [data for keyword, data in blocks if keyword == "*DATABASE_BINARY_D3PLOT"]
    if len(d3plot_blocks) != 1 or not d3plot_blocks[0]:
        fail("expected one *DATABASE_BINARY_D3PLOT data card")

    return {
        "status": "pass",
        "control_contact_cards": 2,
        "control_contact_fields_per_card": [8, 8],
        "initial_velocity_block_count": len(velocity_blocks),
        "initial_velocity_fields_per_card": [8, 8],
        "integer_field_serialization": "pass",
        "eroding_contact_card4": {"isym": 1, "erosop": 1, "iadj": 1},
        "database_binary_d3plot_cards": len(d3plot_blocks[0]),
    }


def main() -> None:
    deck_text = DECK.read_text(encoding="ascii")
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    material = json.loads(MATERIAL.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    with CASE_MAP.open("r", encoding="utf-8", newline="") as handle:
        mapping = list(csv.DictReader(handle))

    checks: dict[str, dict] = {}
    checks["keyword_input_format"] = validate_keyword_input_format(deck_text)
    checks["required_card_layouts"] = validate_required_card_layouts(deck_text)
    checks["keyword_schema_contracts"] = validate_keyword_schema_contracts(deck_text)
    required_keywords = [
        "*KEYWORD", "*CONTROL_TERMINATION", "*CONTROL_TIMESTEP", "*CONTROL_ENERGY",
        "*CONTACT_ERODING_SINGLE_SURFACE", "*MAT_JOHNSON_COOK", "*EOS_GRUNEISEN",
        "*DATABASE_GLSTAT", "*DATABASE_MATSUM", "*DATABASE_SLEOUT", "*DATABASE_NODOUT",
        "*DATABASE_BINARY_D3PLOT", "*BOUNDARY_SPC_SET", "*INITIAL_VELOCITY_GENERATION", "*END"
    ]
    missing = [keyword for keyword in required_keywords if keyword not in deck_text]
    if missing:
        fail(f"missing required keywords: {missing}")
    checks["required_keywords"] = {"status": "pass", "count": len(required_keywords)}

    if len(mapping) != len(design["cases"]):
        fail("case-map count differs from frozen design")
    expected_ids = [case["case_id"] for case in design["cases"]]
    actual_ids = [row["case_id"] for row in mapping]
    if actual_ids != expected_ids:
        fail("case order or IDs differ from frozen design")
    checks["frozen_case_identity"] = {"status": "pass", "case_count": len(mapping)}

    part_ids = []
    history_nodes = []
    for row in mapping:
        part_ids.extend([int(row["plate_part_id"]), int(row["fragment_part_id"])])
        history_nodes.append(int(row["fragment_center_history_node_id"]))
        if int(row["plate_elements"]) <= 0 or int(row["fragment_elements"]) <= 0:
            fail(f"non-positive element count in {row['case_id']}")
    if len(part_ids) != len(set(part_ids)):
        fail("part IDs are not unique")
    if len(history_nodes) != len(set(history_nodes)):
        fail("history node IDs are not unique")
    checks["identity_uniqueness"] = {"status": "pass", "part_count": len(part_ids)}

    node_ids: set[int] = set()
    node_coords: dict[int, tuple[float, float, float]] = {}
    element_ids: set[int] = set()
    elements: list[tuple[int, int, list[int]]] = []
    referenced_nodes: set[int] = set()
    mode = None
    for raw in deck_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("$"):
            continue
        if line.startswith("*"):
            mode = line.upper()
            continue
        fields = [field.strip() for field in line.split(",")]
        if mode == "*NODE" and len(fields) >= 4:
            try:
                nid = int(fields[0])
                coords = [float(value) for value in fields[1:4]]
            except ValueError:
                continue
            if nid in node_ids:
                fail(f"duplicate node ID {nid}")
            if not all(math.isfinite(value) for value in coords):
                fail(f"non-finite node coordinate at {nid}")
            node_ids.add(nid)
            node_coords[nid] = (coords[0], coords[1], coords[2])
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
    absent = sorted(referenced_nodes - node_ids)
    if absent:
        fail(f"elements reference missing nodes, first IDs: {absent[:10]}")
    if len(node_ids) != manifest["deck"]["node_count"]:
        fail("manifest node count mismatch")
    if len(element_ids) != manifest["deck"]["element_count"]:
        fail("manifest element count mismatch")
    checks["mesh_references"] = {
        "status": "pass", "node_count": len(node_ids), "element_count": len(element_ids)
    }

    blocks = keyword_blocks(deck_text)
    grouped_blocks: dict[str, list[list[str]]] = defaultdict(list)
    for keyword, data in blocks:
        grouped_blocks[keyword].append(data)

    section_ids = {
        int(split_fields(data[1])[0]) for data in grouped_blocks["*SECTION_SOLID_TITLE"]
    }
    material_ids = {
        int(split_fields(data[0])[0]) for data in grouped_blocks["*MAT_JOHNSON_COOK"]
    }
    eos_ids = {
        int(split_fields(data[0])[0]) for data in grouped_blocks["*EOS_GRUNEISEN"]
    }
    part_definitions: dict[int, tuple[int, int, int]] = {}
    for data in grouped_blocks["*PART"]:
        values = [int(value) for value in split_fields(data[1])]
        pid, secid, mid, eosid = values[:4]
        if pid in part_definitions:
            fail(f"duplicate *PART ID {pid}")
        if secid not in section_ids or mid not in material_ids or eosid not in eos_ids:
            fail(
                f"*PART {pid} has unresolved section/material/EOS reference: "
                f"secid={secid}, mid={mid}, eosid={eosid}"
            )
        part_definitions[pid] = (secid, mid, eosid)

    mapped_plate_parts = {int(row["plate_part_id"]) for row in mapping}
    mapped_fragment_parts = {int(row["fragment_part_id"]) for row in mapping}
    mapped_parts = mapped_plate_parts | mapped_fragment_parts
    if set(part_definitions) != mapped_parts:
        fail("*PART IDs differ from the case-map plate/fragment identities")
    for pid in mapped_plate_parts:
        if part_definitions[pid] != (1, 1, 1):
            fail(f"plate part {pid} does not reference section/material/EOS 1/1/1")
    for pid in mapped_fragment_parts:
        if part_definitions[pid] != (1, 2, 2):
            fail(f"fragment part {pid} does not reference section/material/EOS 1/2/2")
    element_part_ids = {pid for _, pid, _ in elements}
    if element_part_ids != mapped_parts:
        fail("*ELEMENT_SOLID part references differ from the defined P0 parts")

    node_sets: dict[int, set[int]] = {}
    for data in grouped_blocks["*SET_NODE_LIST_TITLE"]:
        sid = int(split_fields(data[1])[0])
        members = {
            int(value)
            for line in data[2:]
            for value in split_fields(line)
        }
        if sid in node_sets:
            fail(f"duplicate node-set ID {sid}")
        if not members:
            fail(f"empty node-set ID {sid}")
        missing_members = sorted(members - node_ids)
        if missing_members:
            fail(f"node-set {sid} references missing nodes: {missing_members[:10]}")
        node_sets[sid] = members
    boundary_set_ids = {
        int(split_fields(data[0])[0]) for data in grouped_blocks["*BOUNDARY_SPC_SET"]
    }
    if boundary_set_ids != set(node_sets):
        fail("*BOUNDARY_SPC_SET references do not exactly match generated node sets")

    velocity_by_part: dict[int, float] = {}
    for data in grouped_blocks["*INITIAL_VELOCITY_GENERATION"]:
        card1 = split_fields(data[0])
        pid, styp = int(card1[0]), int(card1[1])
        omega, vx, vy, vz = (float(value) for value in card1[2:6])
        if styp != 2 or pid not in mapped_fragment_parts:
            fail(f"initial velocity targets invalid part/type: pid={pid}, styp={styp}")
        if pid in velocity_by_part:
            fail(f"duplicate initial velocity for fragment part {pid}")
        if omega != 0.0 or vx != 0.0 or vy != 0.0 or vz >= 0.0:
            fail(f"fragment part {pid} must have pure negative-z impact velocity")
        velocity_by_part[pid] = vz
    if set(velocity_by_part) != mapped_fragment_parts:
        fail("initial velocities do not cover every fragment part exactly once")
    for row in mapping:
        pid = int(row["fragment_part_id"])
        expected_vz = -float(row["impact_speed_mps"])
        if velocity_by_part[pid] != expected_vz:
            fail(f"initial velocity mismatch for {row['case_id']}: {velocity_by_part[pid]}")

    history_node_ids = {
        int(value)
        for data in grouped_blocks["*DATABASE_HISTORY_NODE"]
        for line in data
        for value in split_fields(line)
    }
    mapped_history_nodes = {
        int(row["fragment_center_history_node_id"]) for row in mapping
    }
    if history_node_ids != mapped_history_nodes or not history_node_ids <= node_ids:
        fail("history-node output does not exactly match existing mapped fragment-center nodes")

    checks["keyword_reference_integrity"] = {
        "status": "pass",
        "section_ids": sorted(section_ids),
        "material_ids": sorted(material_ids),
        "eos_ids": sorted(eos_ids),
        "part_count": len(part_definitions),
        "node_set_count": len(node_sets),
        "velocity_target_count": len(velocity_by_part),
        "history_node_count": len(history_node_ids),
    }

    part_nodes: dict[int, set[int]] = defaultdict(set)
    part_volumes: dict[int, float] = defaultdict(float)
    signed_corner_volumes: list[float] = []
    for eid, pid, nodes in elements:
        part_nodes[pid].update(nodes)
        p1, p2, _p3, p4, p5 = (node_coords[node] for node in nodes[:5])
        a = tuple(p2[index] - p1[index] for index in range(3))
        b = tuple(p4[index] - p1[index] for index in range(3))
        c = tuple(p5[index] - p1[index] for index in range(3))
        cross_bc = (
            b[1] * c[2] - b[2] * c[1],
            b[2] * c[0] - b[0] * c[2],
            b[0] * c[1] - b[1] * c[0],
        )
        signed_volume = sum(a[index] * cross_bc[index] for index in range(3))
        if not math.isfinite(signed_volume) or signed_volume <= 0.0:
            fail(f"non-positive structured-hexahedron corner volume at element {eid}: {signed_volume}")
        signed_corner_volumes.append(signed_volume)
        part_volumes[pid] += signed_volume
    checks["element_orientation"] = {
        "status": "pass",
        "method": "signed triple product at node 1 for generated orthogonal hexahedra",
        "minimum_mm3": min(signed_corner_volumes),
        "maximum_mm3": max(signed_corner_volumes),
    }

    expected_gap = float(design["common"]["initial_gap_mm"])
    gap_records = []
    for row in mapping:
        plate_pid = int(row["plate_part_id"])
        fragment_pid = int(row["fragment_part_id"])
        plate_top = max(node_coords[node][2] for node in part_nodes[plate_pid])
        fragment_bottom = min(node_coords[node][2] for node in part_nodes[fragment_pid])
        actual_gap = fragment_bottom - plate_top
        if abs(actual_gap - expected_gap) > 1e-9:
            fail(f"initial gap mismatch in {row['case_id']}: {actual_gap}")
        gap_records.append(actual_gap)
    checks["initial_separation"] = {
        "status": "pass", "expected_gap_mm": expected_gap,
        "minimum_gap_mm": min(gap_records), "maximum_gap_mm": max(gap_records)
    }

    case_xy_bounds: dict[str, tuple[float, float, float, float]] = {}
    for row in mapping:
        case_nodes = part_nodes[int(row["plate_part_id"])] | part_nodes[int(row["fragment_part_id"])]
        xs = [node_coords[node][0] for node in case_nodes]
        ys = [node_coords[node][1] for node in case_nodes]
        case_xy_bounds[row["case_id"]] = (min(xs), max(xs), min(ys), max(ys))
    pairwise_xy_gaps: list[float] = []
    case_ids = list(case_xy_bounds)
    for left_index, left_id in enumerate(case_ids):
        lx0, lx1, ly0, ly1 = case_xy_bounds[left_id]
        for right_id in case_ids[left_index + 1:]:
            rx0, rx1, ry0, ry1 = case_xy_bounds[right_id]
            gap_x = max(rx0 - lx1, lx0 - rx1, 0.0)
            gap_y = max(ry0 - ly1, ly0 - ry1, 0.0)
            planar_gap = math.hypot(gap_x, gap_y)
            if planar_gap <= 0.0:
                fail(f"P0 submodels overlap in the x-y plane: {left_id}, {right_id}")
            pairwise_xy_gaps.append(planar_gap)
    checks["case_spatial_isolation"] = {
        "status": "pass",
        "case_count": len(case_xy_bounds),
        "minimum_pairwise_xy_gap_mm": min(pairwise_xy_gaps),
        "maximum_in_plane_velocity_mm_per_ms": 0.0,
    }

    expected_fragment_mass = float(design["common"]["fragment_mass_g"]) / 1000.0
    density = float(material["strength_model"]["density_kg_mm3"])
    side_mm = (expected_fragment_mass / (density * 1e9)) ** (1.0 / 3.0) * 1000.0
    reconstructed_mass = density * side_mm**3
    relative_error = abs(reconstructed_mass - expected_fragment_mass) / expected_fragment_mass
    if relative_error > 1e-12:
        fail("fragment geometry/mass identity mismatch")
    checks["fragment_mass_geometry"] = {
        "status": "pass", "full_cube_side_mm": side_mm,
        "full_mass_kg": reconstructed_mass, "relative_error": relative_error
    }
    mass_records = []
    expected_quarter_mass = expected_fragment_mass / 4.0
    expected_plate_volume = (
        float(design["common"]["plate_half_span_mm"]) ** 2
        * float(design["common"]["plate_thickness_mm"])
    )
    for row in mapping:
        fragment_pid = int(row["fragment_part_id"])
        plate_pid = int(row["plate_part_id"])
        meshed_quarter_mass = density * part_volumes[fragment_pid]
        mass_error = abs(meshed_quarter_mass - expected_quarter_mass) / expected_quarter_mass
        # Coordinates are serialized to nine significant decimal places in the
        # solver deck; this tolerance covers only that text round-off.
        if mass_error > 1e-7:
            fail(f"meshed quarter-fragment mass mismatch in {row['case_id']}: {mass_error}")
        plate_volume_error = abs(part_volumes[plate_pid] - expected_plate_volume) / expected_plate_volume
        if plate_volume_error > 1e-7:
            fail(f"meshed plate volume mismatch in {row['case_id']}: {plate_volume_error}")
        mass_records.append({"case_id": row["case_id"], "quarter_mass_kg": meshed_quarter_mass})
    checks["meshed_part_integrals"] = {
        "status": "pass",
        "expected_quarter_fragment_mass_kg": expected_quarter_mass,
        "expected_quarter_plate_volume_mm3": expected_plate_volume,
        "cases": mass_records,
    }

    if manifest["design_hash"] != sha256(DESIGN) or manifest["material_register_hash"] != sha256(MATERIAL):
        fail("manifest input hash mismatch")
    if manifest["deck"]["sha256"] != sha256(DECK) or manifest["case_map"]["sha256"] != sha256(CASE_MAP):
        fail("manifest generated-artifact hash mismatch")
    if manifest.get("p1_design", {}).get("sha256") != sha256(P1_DESIGN):
        fail("manifest P1 design hash mismatch")
    checks["hash_binding"] = {"status": "pass"}

    with P1_DESIGN.open("r", encoding="utf-8", newline="") as handle:
        p1_rows = list(csv.DictReader(handle))
    if len(p1_rows) != 24 or len({row["case_id"] for row in p1_rows}) != 24:
        fail("P1 requires 24 unique cases")
    role_expected = {"hf_response_fit": 18, "hf_response_holdout": 6}
    role_counts = Counter(row["role"] for row in p1_rows)
    if dict(role_counts) != role_expected:
        fail(f"P1 role counts differ from preregistration: {dict(role_counts)}")
    factor_levels = {
        "fragment_mass_g": ["5.0", "8.0", "10.0"],
        "fragment_orientation": ["face", "edge", "corner"],
        "material_variant": ["low", "central", "high"],
    }
    categorical_audit: dict[str, dict[str, dict[str, int]]] = {}
    for role, expected_n in role_expected.items():
        role_rows = [row for row in p1_rows if row["role"] == role]
        expected_per_level = expected_n // 3
        categorical_audit[role] = {}
        for factor, levels in factor_levels.items():
            counts = Counter(row[factor] for row in role_rows)
            if any(counts[level] != expected_per_level for level in levels) or set(counts) != set(levels):
                fail(f"unbalanced P1 {factor} in {role}: {dict(counts)}")
            categorical_audit[role][factor] = {level: counts[level] for level in levels}
        pairs = {(row["fragment_mass_g"], row["fragment_orientation"]) for row in role_rows}
        expected_pairs = 9 if role == "hf_response_fit" else 6
        if len(pairs) != expected_pairs:
            fail(f"insufficient mass/orientation pair coverage in {role}: {len(pairs)}")
        for row in role_rows:
            speed = float(row["impact_speed_mps"])
            thickness = float(row["plate_thickness_mm"])
            angle = float(row["obliquity_deg_from_normal"])
            if not (1200.0 <= speed <= 2600.0 and 6.0 <= thickness <= 14.0 and 0.0 <= angle <= 60.0):
                fail(f"P1 continuous factor outside frozen domain in {row['case_id']}")
            if row["status"] != "preregistered_not_submitted":
                fail(f"unexpected P1 status in {row['case_id']}")
    checks["p1_stratification"] = {
        "status": "pass", "role_counts": dict(role_counts), "categorical_counts": categorical_audit,
        "continuous_design": "independent seeded Latin hypercube within fit and holdout roles"
    }

    forbidden = ["physical_validity=established", "empirically_validated", "certified_real_world"]
    hits = [token for token in forbidden if re.search(re.escape(token), deck_text, flags=re.IGNORECASE)]
    if hits:
        fail(f"forbidden evidence claims in deck: {hits}")
    checks["claim_boundary"] = {"status": "pass"}

    report = {
        "schema_version": "hf-static-validation@1.1.0",
        "overall_status": "pass",
        "scope": "Static package validation only; LS-DYNA keyword parsing and numerical execution remain pending.",
        "checks": checks,
        "prohibited_inference": "This PASS does not establish solver acceptance, numerical convergence, or physical validity."
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STATIC_VALIDATION_FAILED: {exc}", file=sys.stderr)
        raise
