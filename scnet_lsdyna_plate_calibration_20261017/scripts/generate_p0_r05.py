from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

from generate_p0 import (
    add_hex_block,
    card,
    chunked,
    material_cards,
    quarter_axis,
    select_nodes,
    sha256_file,
    uniform_axis,
    write_node_set,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = ROOT / "design" / "p0_cases_r05.json"
DEFAULT_MATERIAL = ROOT / "materials" / "jc4340_reference.json"
DEFAULT_P1_DESIGN = ROOT / "design" / "p1_preregistered_cases.csv"
FROZEN_HISTORICAL_REVISION = "R05_capture_ledger_preregistered"


def write_part_set(lines: list[str], set_id: int, title: str, part_ids: list[int]) -> None:
    lines.extend(
        [
            "*SET_PART_LIST_TITLE",
            title,
            card([set_id, 0.0, 0.0, 0.0, 0.0, "MECH", "", ""]),
        ]
    )
    for values in chunked(sorted(set(part_ids))):
        lines.append(card(values))


def generate_candidate(
    *,
    design_path: Path,
    material_path: Path,
    p1_design_path: Path,
    output_dir: Path,
) -> dict:
    """Generate a self-contained candidate for the supplied design revision."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(
            f"staging directory already exists; inspect it before retrying: {staging}"
        )
    staging.mkdir()

    design = json.loads(design_path.read_text(encoding="utf-8"))
    material = json.loads(material_path.read_text(encoding="utf-8"))
    revision = str(design["package_revision"])
    common = design["common"]
    solver_controls = design["solver_controls"]
    measurement = design["measurement_contract"]
    capture_plane = measurement["capture_plane"]
    cases = design["cases"]

    deck_path = staging / "p0_velocity_mesh_calibration.k"
    case_map_path = staging / "p0_case_part_map.csv"
    manifest_path = staging / "manifest.json"
    design_copy = staging / design_path.name
    material_copy = staging / material_path.name
    p1_copy = staging / p1_design_path.name

    design_copy.write_bytes(design_path.read_bytes())
    material_copy.write_bytes(material_path.read_bytes())
    p1_copy.write_bytes(p1_design_path.read_bytes())

    contact1 = solver_controls["control_contact_card1"]
    contact2 = solver_controls["control_contact_card2"]
    eroding4 = solver_controls["eroding_contact_card4"]
    fragment_set_id = int(measurement["fragment_part_set_id"])

    lines = [
        "*KEYWORD LONG=Y",
        "*TITLE",
        design["title"],
        "$ Units: kg, mm, ms, kN, GPa, K. Velocity numeric values are mm/ms = m/s.",
        "$ Evidence status: high_fidelity_synthetic_reference_candidate; not empirical validation.",
        f"$ Package revision: {design['package_revision']}",
        "$ Primary response comes from a fragment-only D3PART capture ledger; center nodes are diagnostic only.",
        "*CONTROL_TERMINATION",
        card([common["termination_time_ms"], 0, 0.0, 0.0, 0.0]),
        "*CONTROL_TIMESTEP",
        card([0.0, 0.8, 0, 0.0, 0.0, 0, 0, 0]),
        "*CONTROL_ENERGY",
        card([2, 2, 2, 2]),
        "*CONTROL_CONTACT",
        card(
            [
                contact1["slsfac"],
                contact1["rwpnal"],
                contact1["islchk"],
                contact1["shlthk"],
                contact1["penopt"],
                contact1["thkchg"],
                contact1["orien"],
                contact1["enmass"],
            ]
        ),
        card(
            [
                contact2["usrstr"],
                contact2["usrfrc"],
                contact2["nsbcs"],
                contact2["interm"],
                contact2["xpene"],
                contact2["ssthk"],
                contact2["ecdt"],
                contact2["tiedprj"],
            ]
        ),
        "*DATABASE_BINARY_D3PLOT",
        card([float(measurement["full_model_d3plot_interval_ms"])]),
        "*DATABASE_BINARY_D3PART",
        card(
            [
                float(measurement["fragment_d3part_interval_ms"]),
                0,
                0,
                0,
                fragment_set_id,
            ]
        ),
        "*DATABASE_GLSTAT",
        card([float(measurement["ascii_interval_ms"])]),
        "*DATABASE_MATSUM",
        card([float(measurement["ascii_interval_ms"])]),
        "*DATABASE_SLEOUT",
        card([float(measurement["ascii_interval_ms"])]),
        "*DATABASE_NODOUT",
        card([float(measurement["ascii_interval_ms"])]),
        "*CONTACT_ERODING_SINGLE_SURFACE",
        card([0, 0, 0, 0, 0, 0, 0, 0]),
        card([0.0, 0.0, 0.0, 0.0, 20.0, 0, 0.0, 0.0]),
        card([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        card([eroding4["isym"], eroding4["erosop"], eroding4["iadj"]]),
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
    fragment_part_ids: list[int] = []
    total_nodes = 0
    total_elements = 0

    for index, case in enumerate(cases, start=1):
        col = (index - 1) % 3
        row = (index - 1) // 3
        offset_x = col * 150.0
        offset_y = row * 150.0
        plate_pid = 1000 + index
        fragment_pid = 2000 + index
        fragment_part_ids.append(fragment_pid)
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
        full_side = (
            fragment_mass_kg / float(common["fragment_density_kg_m3"])
        ) ** (1.0 / 3.0) * 1000.0
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
                card([fragment_pid, 1, 2, 2, 0, 0, 0, 0]),
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
        fragment_node_start = node_id
        fragment_block = add_hex_block(
            lines,
            xs=fragment_xs,
            ys=fragment_ys,
            zs=fragment_zs,
            part_id=fragment_pid,
            node_start=node_id,
            element_start=element_id,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        node_id, element_id = fragment_block.next_node_id, fragment_block.next_element_id
        total_nodes += len(plate_block.node_ids) + len(fragment_block.node_ids)
        total_elements += len(plate_block.element_ids) + len(fragment_block.element_ids)
        center_node_ids.append(fragment_block.center_node_id)

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
            predicate=lambda x, y, z: abs(x - plate_xs[-1]) <= tol
            or abs(y - plate_ys[-1]) <= tol,
        )
        fragment_sym_x = select_nodes(
            xs=fragment_xs,
            ys=fragment_ys,
            zs=fragment_zs,
            node_start=fragment_node_start,
            predicate=lambda x, y, z: abs(x) <= tol,
        )
        fragment_sym_y = select_nodes(
            xs=fragment_xs,
            ys=fragment_ys,
            zs=fragment_zs,
            node_start=fragment_node_start,
            predicate=lambda x, y, z: abs(y) <= tol,
        )
        write_node_set(lines, sym_x_sid, f"symmetry_x__{case['case_id']}", plate_sym_x + fragment_sym_x)
        write_node_set(lines, sym_y_sid, f"symmetry_y__{case['case_id']}", plate_sym_y + fragment_sym_y)
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
                card([fragment_pid, 2, 0.0, 0.0, 0.0, -float(case["impact_speed_mps"]), 0, 0]),
                card([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 0]),
            ]
        )

        mapping_rows.append(
            {
                "case_id": case["case_id"],
                "role": case["role"],
                "plate_part_id": plate_pid,
                "fragment_part_id": fragment_pid,
                "fragment_center_history_node_id": fragment_block.center_node_id,
                "impact_speed_mps": case["impact_speed_mps"],
                "mesh_mm": mesh,
                "plate_elements": len(plate_block.element_ids),
                "fragment_elements": len(fragment_block.element_ids),
                "symmetry_scale_factor": 4.0,
                "capture_plane_z_mm": capture_plane["coordinate_mm"],
                "fragment_initial_reference_mass_kg": fragment_mass_kg / 4.0,
                "primary_qoi_source": "d3part_fragment_set",
                "center_history_node_status": "diagnostic_only",
            }
        )

    lines.append("*DATABASE_HISTORY_NODE")
    for values in chunked(center_node_ids):
        lines.append(card(values))
    write_part_set(
        lines,
        fragment_set_id,
        "P0 fragment parts for high-rate capture-plane QoI",
        fragment_part_ids,
    )
    lines.extend(["*END", ""])
    deck_path.write_text("\n".join(lines), encoding="ascii", newline="\n")

    with case_map_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)

    manifest = {
        "schema_version": "hf-p0-package-manifest@1.1.0",
        "package_revision": design["package_revision"],
        "status": "generated_not_solver_validated",
        "evidence_class": "high_fidelity_synthetic_reference_candidate",
        "design": {
            "path": design_copy.name,
            "sha256": sha256_file(design_copy),
            "status": design["design_status"],
        },
        "design_hash": sha256_file(design_copy),
        "material_register": {
            "path": material_copy.name,
            "sha256": sha256_file(material_copy),
        },
        "material_register_hash": sha256_file(material_copy),
        "deck": {
            "path": deck_path.name,
            "sha256": sha256_file(deck_path),
            "node_count": total_nodes,
            "element_count": total_elements,
            "case_count": len(cases),
        },
        "case_map": {"path": case_map_path.name, "sha256": sha256_file(case_map_path)},
        "p1_design": {
            "path": p1_copy.name,
            "sha256": sha256_file(p1_copy),
            "case_count": 24,
            "status": "preregistered_not_submitted",
        },
        "measurement_contract": measurement,
        "solver": {
            "requested_family": "LS-DYNA MPP double precision",
            "actual_build": None,
            "executable_sha256": None,
            "mpi_implementation": None,
            "status": "bind_on_scnet_submission",
        },
        "limitations": [
            f"The {revision} keyword deck has not been parsed or executed by LS-DYNA.",
            "Static validation cannot establish D3PART readability, numerical convergence, or physical validity.",
            "The capture ledger is a numerical qualification contract, not an experimental perforation or V50 definition.",
            "Material, EOS, contact, erosion, section, and geometry are retained from R04 and remain uncalibrated to experiment.",
            "P1 submission, local-engine fitting, dataset generation, and training remain denied until all applicable P0 successor-candidate gates pass.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for attempt in range(8):
        try:
            staging.replace(output_dir)
            break
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32} or attempt == 7:
                raise
            time.sleep(0.05 * (2**attempt))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a non-overwriting P0 candidate from the supplied design revision"
    )
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--material", type=Path, default=DEFAULT_MATERIAL)
    parser.add_argument("--p1-design", type=Path, default=DEFAULT_P1_DESIGN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-frozen-r05-test-rebuild",
        action="store_true",
        help=(
            "Explicit test-fixture override only. The submitted R05 identity is frozen; "
            "a rebuilt five-field deck must never be released under that historical revision."
        ),
    )
    return parser.parse_args()


def guard_frozen_cli_revision(design_path: Path, allow_test_rebuild: bool) -> None:
    revision = str(json.loads(design_path.read_text(encoding="utf-8"))["package_revision"])
    if revision == FROZEN_HISTORICAL_REVISION and not allow_test_rebuild:
        raise RuntimeError(
            "refusing to rebuild the frozen historical R05 revision under a colliding identity; "
            "use the R06 successor generator for releasable work. "
            "--allow-frozen-r05-test-rebuild is reserved for explicit local fixtures."
        )


def main() -> None:
    args = parse_args()
    guard_frozen_cli_revision(args.design.resolve(), args.allow_frozen_r05_test_rebuild)
    manifest = generate_candidate(
        design_path=args.design.resolve(),
        material_path=args.material.resolve(),
        p1_design_path=args.p1_design.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
