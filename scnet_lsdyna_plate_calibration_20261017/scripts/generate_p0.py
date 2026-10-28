from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "design" / "p0_cases.json"
MATERIAL_PATH = ROOT / "materials" / "jc4340_reference.json"
P1_DESIGN_PATH = ROOT / "design" / "p1_preregistered_cases.csv"
DECK_DIR = ROOT / "decks"
DECK_PATH = DECK_DIR / "p0_velocity_mesh_calibration.k"
CASE_MAP_PATH = DECK_DIR / "p0_case_part_map.csv"
MANIFEST_PATH = ROOT / "manifest.json"


@dataclass
class MeshBlock:
    node_ids: list[int]
    element_ids: list[int]
    center_node_id: int
    next_node_id: int
    next_element_id: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.9e}"


def card(values: Iterable[float | int | str]) -> str:
    return ",".join(str(value) if isinstance(value, str) else fmt(value) for value in values)


def chunked(values: list[int], size: int = 8) -> Iterable[list[int]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def quarter_axis(half_span: float, refined_half_span: float, fine_spacing: float, outer_target: float) -> list[float]:
    n_fine = max(1, int(round(refined_half_span / fine_spacing)))
    fine = [refined_half_span * i / n_fine for i in range(n_fine + 1)]
    n_outer = max(1, int(math.ceil((half_span - refined_half_span) / outer_target)))
    outer = [refined_half_span + (half_span - refined_half_span) * i / n_outer for i in range(1, n_outer + 1)]
    return fine + outer


def uniform_axis(start: float, stop: float, intervals: int) -> list[float]:
    return [start + (stop - start) * i / intervals for i in range(intervals + 1)]


def add_hex_block(
    lines: list[str],
    *,
    xs: list[float],
    ys: list[float],
    zs: list[float],
    part_id: int,
    node_start: int,
    element_start: int,
    offset_x: float,
    offset_y: float,
) -> MeshBlock:
    nx, ny, nz = len(xs), len(ys), len(zs)
    node_ids: list[int] = []
    element_ids: list[int] = []
    node_lookup: dict[tuple[int, int, int], int] = {}
    nid = node_start
    lines.append("*NODE")
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                node_lookup[(i, j, k)] = nid
                node_ids.append(nid)
                lines.append(card([nid, x + offset_x, y + offset_y, z]))
                nid += 1
    eid = element_start
    lines.append("*ELEMENT_SOLID")
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                n1 = node_lookup[(i, j, k)]
                n2 = node_lookup[(i + 1, j, k)]
                n3 = node_lookup[(i + 1, j + 1, k)]
                n4 = node_lookup[(i, j + 1, k)]
                n5 = node_lookup[(i, j, k + 1)]
                n6 = node_lookup[(i + 1, j, k + 1)]
                n7 = node_lookup[(i + 1, j + 1, k + 1)]
                n8 = node_lookup[(i, j + 1, k + 1)]
                lines.append(card([eid, part_id, n1, n2, n3, n4, n5, n6, n7, n8]))
                element_ids.append(eid)
                eid += 1
    center_key = (0, 0, min(range(nz), key=lambda index: abs(zs[index] - 0.5 * (zs[0] + zs[-1]))))
    return MeshBlock(node_ids, element_ids, node_lookup[center_key], nid, eid)


def write_node_set(lines: list[str], set_id: int, title: str, node_ids: list[int]) -> None:
    lines.extend(["*SET_NODE_LIST_TITLE", title, card([set_id, 0.0, 0.0, 0.0, 0.0])])
    for values in chunked(sorted(set(node_ids))):
        lines.append(card(values))


def select_nodes(
    *,
    xs: list[float],
    ys: list[float],
    zs: list[float],
    node_start: int,
    predicate,
) -> list[int]:
    result: list[int] = []
    nid = node_start
    for z in zs:
        for y in ys:
            for x in xs:
                if predicate(x, y, z):
                    result.append(nid)
                nid += 1
    return result


def material_cards(material: dict) -> list[str]:
    strength = material["strength_model"]
    failure = material["failure_model"]
    eos = material["equation_of_state"]
    lines: list[str] = []
    for mid, eosid, label in ((1, 1, "plate"), (2, 2, "fragment")):
        lines.extend(
            [
                f"$ {label}: {material['material_id']}",
                "*MAT_JOHNSON_COOK",
                card(
                    [
                        mid,
                        strength["density_kg_mm3"],
                        strength["shear_modulus_gpa"],
                        strength["youngs_modulus_gpa"],
                        strength["poissons_ratio"],
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
                card(
                    [
                        strength["A_gpa"],
                        strength["B_gpa"],
                        strength["n"],
                        strength["C"],
                        strength["m"],
                        strength["melt_temperature_k"],
                        strength["reference_temperature_k"],
                        strength["reference_strain_rate_per_ms"],
                    ]
                ),
                card(
                    [
                        strength["specific_heat_j_kgk"],
                        0.0,
                        0.0,
                        strength["plastic_strain_iteration"],
                        failure["D1"],
                        failure["D2"],
                        failure["D3"],
                        failure["D4"],
                    ]
                ),
                card([failure["D5"], 0.0, 0.0, failure["minimum_failure_strain"], 0]),
                "*EOS_GRUNEISEN",
                card(
                    [
                        eosid,
                        eos["C_mm_ms"],
                        eos["S1"],
                        eos["S2"],
                        eos["S3"],
                        eos["gamma0"],
                        eos["a"],
                        eos["initial_internal_energy"],
                    ]
                ),
                card([eos["initial_relative_volume"]]),
            ]
        )
    return lines


def main() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    material = json.loads(MATERIAL_PATH.read_text(encoding="utf-8"))
    common = design["common"]
    solver_controls = design["solver_controls"]
    control_contact_card1 = solver_controls["control_contact_card1"]
    control_contact_card2 = solver_controls["control_contact_card2"]
    eroding_contact_card4 = solver_controls["eroding_contact_card4"]
    cases = design["cases"]
    DECK_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "*KEYWORD LONG=Y",
        "*TITLE",
        "SCNet P0 quarter-symmetry fragment/plate velocity and mesh qualification",
        "$ Units: kg, mm, ms, kN, GPa, K. Velocity numeric values are mm/ms = m/s.",
        "$ Evidence status: high_fidelity_synthetic_reference_candidate; not empirical validation.",
        "*CONTROL_TERMINATION",
        card([common["termination_time_ms"], 0, 0.0, 0.0, 0.0]),
        "*CONTROL_TIMESTEP",
        card([0.0, 0.8, 0, 0.0, 0.0, 0, 0, 0]),
        "*CONTROL_ENERGY",
        card([2, 2, 2, 2]),
        "*CONTROL_CONTACT",
        card([
            control_contact_card1["slsfac"], control_contact_card1["rwpnal"],
            control_contact_card1["islchk"], control_contact_card1["shlthk"],
            control_contact_card1["penopt"], control_contact_card1["thkchg"],
            control_contact_card1["orien"], control_contact_card1["enmass"],
        ]),
        card([
            control_contact_card2["usrstr"], control_contact_card2["usrfrc"],
            control_contact_card2["nsbcs"], control_contact_card2["interm"],
            control_contact_card2["xpene"], control_contact_card2["ssthk"],
            control_contact_card2["ecdt"], control_contact_card2["tiedprj"],
        ]),
        "*DATABASE_BINARY_D3PLOT",
        card([0.0005]),
        "*DATABASE_GLSTAT",
        card([0.00005]),
        "*DATABASE_MATSUM",
        card([0.00005]),
        "*DATABASE_SLEOUT",
        card([0.00005]),
        "*DATABASE_NODOUT",
        card([0.00005]),
        "*CONTACT_ERODING_SINGLE_SURFACE",
        card([0, 0, 0, 0, 0, 0, 0, 0]),
        card([0.0, 0.0, 0.0, 0.0, 20.0, 0, 0.0, 0.0]),
        card([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        card([
            eroding_contact_card4["isym"], eroding_contact_card4["erosop"],
            eroding_contact_card4["iadj"],
        ]),
        card([2, 0.0, 0, 0, 3, 5, 0, 0]),
        "*SECTION_SOLID_TITLE",
        "Fully integrated S/R solid for numerical qualification",
        card([1, -1]),
    ]
    lines.extend(material_cards(material))

    node_id = 1
    element_id = 1
    mapping_rows: list[dict] = []
    center_node_ids: list[int] = []
    total_nodes = 0
    total_elements = 0

    for index, case in enumerate(cases, start=1):
        col = (index - 1) % 3
        row = (index - 1) // 3
        offset_x = col * 150.0
        offset_y = row * 150.0
        plate_pid = 1000 + index
        projectile_pid = 2000 + index
        sym_x_sid = 3000 + 10 * index + 1
        sym_y_sid = 3000 + 10 * index + 2
        clamp_sid = 3000 + 10 * index + 3

        mesh = float(case["mesh_mm"])
        plate_xs = quarter_axis(
            float(common["plate_half_span_mm"]),
            float(common["refined_half_span_mm"]),
            mesh,
            float(common["outer_target_spacing_mm"]),
        )
        plate_ys = list(plate_xs)
        plate_nz = max(2, int(math.ceil(float(common["plate_thickness_mm"]) / mesh)))
        plate_zs = uniform_axis(-float(common["plate_thickness_mm"]), 0.0, plate_nz)
        fragment_mass_kg = float(common["fragment_mass_g"]) / 1000.0
        full_side = (fragment_mass_kg / (float(common["fragment_density_kg_m3"]))) ** (1.0 / 3.0) * 1000.0
        full_n = max(4, int(round(full_side / mesh)))
        if full_n % 2:
            full_n += 1
        quarter_n = full_n // 2
        fragment_xs = uniform_axis(0.0, 0.5 * full_side, quarter_n)
        fragment_ys = list(fragment_xs)
        gap = float(common["initial_gap_mm"])
        fragment_zs = uniform_axis(gap, gap + full_side, full_n)

        lines.extend(
            [
                f"$ CASE {case['case_id']} role={case['role']} speed={case['impact_speed_mps']} mesh={mesh}",
                "*PART",
                f"plate__{case['case_id']}",
                card([plate_pid, 1, 1, 1, 0, 0, 0, 0]),
                "*PART",
                f"fragment__{case['case_id']}",
                card([projectile_pid, 1, 2, 2, 0, 0, 0, 0]),
            ]
        )

        plate_node_start = node_id
        plate_block = add_hex_block(
            lines,
            xs=plate_xs,
            ys=plate_ys,
            zs=plate_zs,
            part_id=plate_pid,
            node_start=node_id,
            element_start=element_id,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        node_id, element_id = plate_block.next_node_id, plate_block.next_element_id
        projectile_node_start = node_id
        projectile_block = add_hex_block(
            lines,
            xs=fragment_xs,
            ys=fragment_ys,
            zs=fragment_zs,
            part_id=projectile_pid,
            node_start=node_id,
            element_start=element_id,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        node_id, element_id = projectile_block.next_node_id, projectile_block.next_element_id
        total_nodes += len(plate_block.node_ids) + len(projectile_block.node_ids)
        total_elements += len(plate_block.element_ids) + len(projectile_block.element_ids)
        center_node_ids.append(projectile_block.center_node_id)

        tol = 1e-9
        plate_sym_x = select_nodes(
            xs=plate_xs,
            ys=plate_ys,
            zs=plate_zs,
            node_start=plate_node_start,
            predicate=lambda x, y, z: abs(x) <= tol,
        )
        plate_sym_y = select_nodes(
            xs=plate_xs,
            ys=plate_ys,
            zs=plate_zs,
            node_start=plate_node_start,
            predicate=lambda x, y, z: abs(y) <= tol,
        )
        plate_clamp = select_nodes(
            xs=plate_xs,
            ys=plate_ys,
            zs=plate_zs,
            node_start=plate_node_start,
            predicate=lambda x, y, z: abs(x - plate_xs[-1]) <= tol or abs(y - plate_ys[-1]) <= tol,
        )
        frag_sym_x = select_nodes(
            xs=fragment_xs,
            ys=fragment_ys,
            zs=fragment_zs,
            node_start=projectile_node_start,
            predicate=lambda x, y, z: abs(x) <= tol,
        )
        frag_sym_y = select_nodes(
            xs=fragment_xs,
            ys=fragment_ys,
            zs=fragment_zs,
            node_start=projectile_node_start,
            predicate=lambda x, y, z: abs(y) <= tol,
        )
        write_node_set(lines, sym_x_sid, f"symmetry_x__{case['case_id']}", plate_sym_x + frag_sym_x)
        write_node_set(lines, sym_y_sid, f"symmetry_y__{case['case_id']}", plate_sym_y + frag_sym_y)
        write_node_set(lines, clamp_sid, f"clamped_far_edges__{case['case_id']}", plate_clamp)
        lines.extend(
            [
                "*BOUNDARY_SPC_SET",
                card([sym_x_sid, 0, 1, 0, 0, 0, 0, 0]),
                "*BOUNDARY_SPC_SET",
                card([sym_y_sid, 0, 0, 1, 0, 0, 0, 0]),
                "*BOUNDARY_SPC_SET",
                card([clamp_sid, 0, 1, 1, 1, 0, 0, 0]),
                "*INITIAL_VELOCITY_GENERATION",
                card([projectile_pid, 2, 0.0, 0.0, 0.0, -float(case["impact_speed_mps"]), 0, 0]),
                card([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 0]),
            ]
        )

        mapping_rows.append(
            {
                "case_id": case["case_id"],
                "role": case["role"],
                "plate_part_id": plate_pid,
                "fragment_part_id": projectile_pid,
                "fragment_center_history_node_id": projectile_block.center_node_id,
                "impact_speed_mps": case["impact_speed_mps"],
                "mesh_mm": mesh,
                "plate_elements": len(plate_block.element_ids),
                "fragment_elements": len(projectile_block.element_ids),
                "symmetry_scale_factor": 4.0,
            }
        )

    lines.append("*DATABASE_HISTORY_NODE")
    for values in chunked(center_node_ids):
        lines.append(card(values))
    lines.extend(["*END", ""])
    DECK_PATH.write_text("\n".join(lines), encoding="ascii", newline="\n")

    with CASE_MAP_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)

    manifest = {
        "schema_version": "hf-p0-package-manifest@1.0.0",
        "package_revision": "R04_full_keyword_type_contract",
        "status": "generated_not_solver_validated",
        "evidence_class": "high_fidelity_synthetic_reference_candidate",
        "design_hash": sha256_file(DESIGN_PATH),
        "material_register_hash": sha256_file(MATERIAL_PATH),
        "deck": {
            "path": str(DECK_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(DECK_PATH),
            "node_count": total_nodes,
            "element_count": total_elements,
            "case_count": len(cases),
        },
        "case_map": {
            "path": str(CASE_MAP_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(CASE_MAP_PATH),
        },
        "p1_design": {
            "path": str(P1_DESIGN_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(P1_DESIGN_PATH),
        },
        "solver": {
            "requested_family": "LS-DYNA MPP double precision",
            "actual_build": None,
            "executable_sha256": None,
            "mpi_implementation": None,
            "status": "bind_on_scnet_submission"
        },
        "limitations": [
            "The keyword deck has not been parsed or executed by LS-DYNA locally.",
            "P0 uses quarter symmetry and is valid only for centered, normal, face-on impact.",
            "The central material card is a generic 4340 reference; its EOS transfer is an explicit epistemic assumption.",
            "P0 qualifies the central velocity response and mesh behavior only; it does not cover the full mass-thickness-angle-orientation domain."
        ]
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
