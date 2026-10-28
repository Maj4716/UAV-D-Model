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
DEFAULT_DESIGN = ROOT / "design" / "p0_r06_diagnostic_design.json"
DEFAULT_R05_DESIGN = ROOT / "design" / "p0_cases_r05.json"
DEFAULT_MATERIAL = ROOT / "materials" / "jc4340_reference.json"
DEFAULT_SUCCESSOR_CONTRACT = ROOT / "design" / "p0_r06_real_output_contract.json"
DEFAULT_OUTPUT = ROOT / "manual_upload_20261023_r06_diagnostics"


def strict_json(path: Path) -> dict:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=no_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value in {path}: {value}")
        ),
    )


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


def axes_for_case(case: dict, common: dict) -> tuple[list[float], list[float], list[float], list[float]]:
    mesh = float(case["mesh_mm"])
    plate_xs = quarter_axis(
        float(common["plate_half_span_mm"]),
        float(common["refined_half_span_mm"]),
        mesh,
        float(common["outer_target_spacing_mm"]),
    )
    plate_nz = max(2, int(math.ceil(float(common["plate_thickness_mm"]) / mesh)))
    plate_zs = uniform_axis(-float(common["plate_thickness_mm"]), 0.0, plate_nz)
    fragment_mass_kg = float(common["fragment_mass_g"]) / 1000.0
    full_side = (
        fragment_mass_kg / float(common["fragment_density_kg_m3"])
    ) ** (1.0 / 3.0) * 1000.0
    full_n = max(4, int(round(full_side / mesh)))
    if full_n % 2:
        full_n += 1
    fragment_xs = uniform_axis(0.0, 0.5 * full_side, full_n // 2)
    fragment_zs = uniform_axis(
        float(common["initial_gap_mm"]),
        float(common["initial_gap_mm"]) + full_side,
        full_n,
    )
    return plate_xs, plate_zs, fragment_xs, fragment_zs


def entity_counts(case: dict, common: dict) -> tuple[int, int]:
    plate_xs, plate_zs, fragment_xs, fragment_zs = axes_for_case(case, common)
    plate_nodes = len(plate_xs) * len(plate_xs) * len(plate_zs)
    fragment_nodes = len(fragment_xs) * len(fragment_xs) * len(fragment_zs)
    plate_elements = (len(plate_xs) - 1) ** 2 * (len(plate_zs) - 1)
    fragment_elements = (len(fragment_xs) - 1) ** 2 * (len(fragment_zs) - 1)
    return plate_nodes + fragment_nodes, plate_elements + fragment_elements


def original_id_start(r05_design: dict, target_index: int) -> tuple[int, int]:
    node_id = 1
    element_id = 1
    for case in r05_design["cases"][: target_index - 1]:
        node_count, element_count = entity_counts(case, r05_design["common"])
        node_id += node_count
        element_id += element_count
    return node_id, element_id


def find_r05_case(r05_design: dict, case_identity: dict) -> dict:
    index = int(case_identity["r05_original_index"])
    if not 1 <= index <= len(r05_design["cases"]):
        raise ValueError(f"R05 case index out of range: {index}")
    case = r05_design["cases"][index - 1]
    expected = {
        "case_id": case_identity["r05_case_id"],
        "impact_speed_mps": float(case_identity["impact_speed_mps"]),
        "mesh_mm": float(case_identity["mesh_mm"]),
    }
    actual = {
        "case_id": case["case_id"],
        "impact_speed_mps": float(case["impact_speed_mps"]),
        "mesh_mm": float(case["mesh_mm"]),
    }
    if actual != expected:
        raise ValueError(f"R06 case identity does not match frozen R05 index {index}: {actual} != {expected}")
    return case


def deck_for_job(
    *,
    design: dict,
    r05_design: dict,
    material: dict,
    experiment: dict,
    job: dict,
) -> tuple[str, dict]:
    identity = experiment["case_identity"]
    case = find_r05_case(r05_design, identity)
    original_index = int(identity["r05_original_index"])
    node_id, element_id = original_id_start(r05_design, original_index)
    initial_node_id = node_id
    initial_element_id = element_id
    col = (original_index - 1) % 3
    row = (original_index - 1) // 3
    offset_x = col * 150.0
    offset_y = row * 150.0
    plate_pid = 1000 + original_index
    fragment_pid = 2000 + original_index
    if plate_pid != int(identity["plate_part_id"]) or fragment_pid != int(identity["fragment_part_id"]):
        raise ValueError("R06 PIDs do not match the frozen R05 index convention")

    common = r05_design["common"]
    frozen = design["frozen_solver_controls"]
    measure = design["shared_measurement_upgrade"]
    plate_xs, plate_zs, fragment_xs, fragment_zs = axes_for_case(case, common)
    plate_ys = list(plate_xs)
    fragment_ys = list(fragment_xs)
    fragment_mass_kg = float(common["fragment_mass_g"]) / 1000.0
    psfail = int(measure["plate_psfail_part_set_id"]) if job["plate_only_psfail"] else 0

    contact1 = frozen["control_contact_card1"]
    contact2 = frozen["control_contact_card2"]
    eroding4 = frozen["eroding_contact_card4"]
    timestep = frozen["control_timestep"]
    job_tssfac = float(job.get("tssfac", timestep["tssfac"]))
    direct_contact_policy = measure.get(
        "direct_contact_output_policy",
        {
            "request_binary_intfor": True,
            "request_ascii_ncforc": True,
        },
    )
    request_intfor = bool(direct_contact_policy["request_binary_intfor"])
    request_ncforc = bool(direct_contact_policy["request_ascii_ncforc"])
    intfor = measure.get("intfor")
    if request_intfor and intfor is None:
        raise ValueError("binary INTFOR was requested without an intfor contract")
    d3part_extent_cards = measure.get("d3part_extent_cards", [])
    if d3part_extent_cards and (
        len(d3part_extent_cards) not in (1, 2, 3)
        or any(len(values) != 8 for values in d3part_extent_cards)
    ):
        raise ValueError("DATABASE_EXTENT_D3PART cards must contain one to three eight-field cards")
    parser_contract = measure.get("r15_parser_contract")
    if parser_contract is None:
        # Historical R06d serialization is intentionally reproducible.  It is
        # retained only so the submitted failure can be reconstructed; new
        # packages must provide the explicit R15 parser contract below.
        extent_binary_cards = [
            [int(measure["database_extent_binary_neiph"]), 0, 0, 0, 0, 0, 0, 0]
        ]
        intfor_timing_tail = [0, 0, 0, 0]
        intfor_ioopt_card = [0]
    else:
        if parser_contract.get("target_solver") != "LS-DYNA R15.0.2":
            raise ValueError("unsupported parser-contract target")
        extent_binary_cards = parser_contract["database_extent_binary_cards"]
        intfor_timing_tail = parser_contract["intfor_file_timing_card_tail"]
        intfor_ioopt_card = parser_contract["intfor_file_ioopt_card"]
        if (
            len(extent_binary_cards) != 2
            or any(len(values) != 8 for values in extent_binary_cards)
            or len(intfor_timing_tail) != 3
            or len(intfor_ioopt_card) != 1
        ):
            raise ValueError("R15 parser contract has an invalid card shape")
    lines = [
        "*KEYWORD LONG=Y",
        "*TITLE",
        f"SCNet P0 {design['package_revision']} {job['job_id']}",
        "$ Units: kg, mm, ms, kN, GPa, K. Numeric velocity mm/ms equals m/s.",
        "$ Diagnostic only: not physical validation, not a production response label.",
        f"$ Pair={experiment['experiment_id']} factor={experiment['factor']} role={job['role']}",
        "*CONTROL_TERMINATION",
        card([float(job["termination_time_ms"]), 0, float(frozen["termination_dtmin"]), 0.0, 0.0]),
        "*CONTROL_TIMESTEP",
        card(
            [
                float(timestep["dtinit"]),
                job_tssfac,
                int(timestep["isdo"]),
                float(timestep["tslimit"]),
                float(timestep["dt2ms"]),
                int(timestep["lctm"]),
                int(timestep["erode"]),
                int(timestep["ms1st"]),
            ]
        ),
        "*CONTROL_SOLID",
        card([0, 0, 0, 0, psfail, 0.0, 0, 0]),
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
        card([float(measure["full_model_d3plot_interval_ms"])]),
        "*DATABASE_EXTENT_BINARY",
        *[card([int(value) for value in values]) for values in extent_binary_cards],
        "*DATABASE_BINARY_D3PART",
        card([float(measure["fragment_d3part_interval_ms"]), 0, 0, 0, int(measure["fragment_part_set_id"])]),
        *(
            [
                "*DATABASE_EXTENT_D3PART",
                *[card([int(value) for value in values]) for values in d3part_extent_cards],
            ]
            if d3part_extent_cards
            else []
        ),
        *(
            [
                "*DATABASE_BINARY_INTFOR_FILE",
                str(intfor["filename"]),
                card([float(measure["intfor_interval_ms"]), *[int(value) for value in intfor_timing_tail]]),
                card([int(value) for value in intfor_ioopt_card]),
                "*DATABASE_EXTENT_INTFOR",
                card([int(value) for value in intfor["extent_card1"]]),
                card([int(value) for value in intfor["extent_card2"]]),
            ]
            if request_intfor
            else []
        ),
        "*DATABASE_GLSTAT",
        card([float(measure["ascii_interval_ms"])]),
        "*DATABASE_MATSUM",
        card([float(measure["ascii_interval_ms"])]),
        "*DATABASE_SLEOUT",
        card([float(measure["ascii_interval_ms"])]),
        "*DATABASE_NODOUT",
        card([float(measure["ascii_interval_ms"])]),
        *(
            ["*DATABASE_NCFORC", card([float(measure["ascii_interval_ms"])])]
            if request_ncforc
            else []
        ),
        "*CONTACT_ERODING_SINGLE_SURFACE",
        card([0, 0, 0, 0, 0, 0, int(measure["contact_output_flags"]["spr_or_sapr"]), int(measure["contact_output_flags"]["mpr_or_sbpr"])]),
        card([0.0, 0.0, 0.0, 0.0, 20.0, 0, 0.0, 0.0]),
        card([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        card([eroding4["isym"], eroding4["erosop"], eroding4["iadj"]]),
        card([2, 0.0, 0, 0, 3, 5, 0, 0]),
        "*SECTION_SOLID_TITLE",
        "Fully integrated S/R solid for numerical qualification",
        card([1, -1]),
    ]
    lines.extend(material_cards(material))
    lines.extend(
        [
            f"$ CASE {case['case_id']} source_index={original_index} speed={case['impact_speed_mps']} mesh={case['mesh_mm']}",
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

    tol = 1e-9
    sym_x_sid = 3000 + 10 * original_index + 1
    sym_y_sid = 3000 + 10 * original_index + 2
    clamp_sid = 3000 + 10 * original_index + 3
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
            "*DATABASE_HISTORY_NODE",
            card([fragment_block.center_node_id]),
        ]
    )
    write_part_set(
        lines,
        int(measure["fragment_part_set_id"]),
        "R06 current fragment only for high-rate D3PART",
        [fragment_pid],
    )
    write_part_set(
        lines,
        int(measure["plate_psfail_part_set_id"]),
        "R06 current plate only for PSFAIL treatment",
        [plate_pid],
    )
    lines.extend(["*END", ""])

    metadata = {
        "case_id": case["case_id"],
        "source_r05_index": original_index,
        "impact_speed_mps": float(case["impact_speed_mps"]),
        "mesh_mm": float(case["mesh_mm"]),
        "offset_x_mm": offset_x,
        "offset_y_mm": offset_y,
        "plate_part_id": plate_pid,
        "fragment_part_id": fragment_pid,
        "plate_node_count": len(plate_block.node_ids),
        "fragment_node_count": len(fragment_block.node_ids),
        "plate_element_count": len(plate_block.element_ids),
        "fragment_element_count": len(fragment_block.element_ids),
        "first_node_id": initial_node_id,
        "last_node_id": node_id - 1,
        "first_element_id": initial_element_id,
        "last_element_id": element_id - 1,
        "fragment_center_history_node_id": fragment_block.center_node_id,
        "fragment_initial_reference_mass_kg": fragment_mass_kg / 4.0,
        "capture_plane_z_mm": float(measure["capture_plane"]["coordinate_mm"]),
        "termination_time_ms": float(job["termination_time_ms"]),
        "psfail_part_set_id": psfail,
    }
    return "\n".join(lines), metadata


def upload_readme(design: dict, experiment: dict, job: dict, deck_name: str, deck_hash: str, deck_size: int) -> str:
    legacy_job_names = {
        "r06d_0800_ctrl": "uav_hf_r06d_0800_c",
        "r06d_0800_tend008": "uav_hf_r06d_0800_t",
        "r06d_2000_ctrl": "uav_hf_r06d_2000_c",
        "r06d_2000_psfail_plate": "uav_hf_r06d_2000_p",
    }
    job_name = job.get(
        "platform_job_name",
        legacy_job_names.get(job["job_id"], f"uav_hf_{job['job_id']}"),
    )
    sequence = design["submission_sequence"]
    sequence_text = "\n".join(
        f"  {name}={','.join(sequence[name])}"
        for name in ("wave_0", "wave_1", "wave_2")
        if name in sequence
    )
    if not sequence_text:
        sequence_text = (
            "  wave_1=" + ",".join(sequence["wave_1"]) + "\n"
            "  wave_2=" + ",".join(sequence["wave_2"])
        )
    measure = design["shared_measurement_upgrade"]
    direct_contact_policy = measure.get(
        "direct_contact_output_policy",
        {"request_binary_intfor": True, "request_ascii_ncforc": True},
    )
    if direct_contact_policy["request_binary_intfor"]:
        command_note = "<empty; INTFOR filename is embedded by *DATABASE_BINARY_INTFOR_FILE>"
    elif design["design_status"] == "frozen_after_r06f_tssfac04_sensitivity_fail_before_tssfac02_outcome":
        command_note = "<empty; R06g does not request unsupported binary INTFOR>"
    else:
        command_note = "<empty; R06f does not request unsupported binary INTFOR>"
    required_return = ", ".join(measure["required_return_families"])
    optional_return = ", ".join(measure.get("optional_return_families", []))
    optional_text = (
        f"\n  Optional but preserve if present: {optional_return}."
        if optional_return
        else ""
    )
    return f"""SCNet LS-DYNA P0 R06 diagnostic job

STATUS
  Prepared and statically validated only after static_validation.json reports PASS.
  NOT submitted, NOT solver-validated, NOT admitted for P0/P1 or training labels.

PAIR / FACTOR
  experiment_id={experiment['experiment_id']}
  factor={experiment['factor']}
  role={job['role']}
  termination_time_ms={job['termination_time_ms']}
  plate_only_psfail={str(bool(job['plate_only_psfail'])).lower()}
  tssfac={job.get('tssfac', design['frozen_solver_controls']['control_timestep']['tssfac'])}

SOLVER INPUT (select this file only as the LS-DYNA head/input file)
  {deck_name}

EXPECTED SOLVER INPUT IDENTITY
  size_bytes={deck_size}
  {deck_hash}
  first_line=*KEYWORD LONG=Y
  package_revision={design['package_revision']}
  job_id={job['job_id']}

SCNET SETTINGS
  job_name={job_name}
  work_directory=/work/home/acd7dt5yv4/LSDYNA/{job_name}
  software=ls-dyna_R15_mmp_d
  nodes=1
  ranks=32
  walltime=04:00:00
  head_memory=512MB
  child_memory=256MB
  command_line_parameters={command_note}
  auto_fill_previous_parameters=off

UPLOAD SEQUENCE
{sequence_text}
  Follow the frozen gates in the packaged design. Do not advance to a later wave
  until the preceding parser/runtime/output-readability gate passes.

REQUIRED RETURN
  Required: {required_return}.{optional_text}
  Do not rename or merge job directories. A missing optional direct-contact file
  is unavailable evidence, never a zero response.

BOUNDARY
  This job follows the frozen design and submission wave. A static PASS does not
  prove target-solver parsing, output readability, numerical quality, response
  availability, physical validity, P0 admission, or dataset-label admission.
"""


def generate_batch(
    *,
    design_path: Path,
    r05_design_path: Path,
    material_path: Path,
    successor_contract_path: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists; inspect it before retrying: {staging}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()

    design = strict_json(design_path)
    r05_design = strict_json(r05_design_path)
    material = strict_json(material_path)
    successor = strict_json(successor_contract_path)
    revision = str(design["package_revision"])
    if not revision.startswith("R06_"):
        raise ValueError(f"R06 generator refuses non-R06 revision: {revision}")
    allowed_design_statuses = {
        "frozen_after_r05_stage_ab_before_any_r06_solver_outcome",
        "frozen_after_r06d_wave1_keyword_failure_before_any_repaired_solver_outcome",
        "frozen_after_r06e_wave0_solver_capability_audit_before_any_r06f_solver_outcome",
        "frozen_after_r06f_recanary_pass_before_timestep_solver_outcome",
        "frozen_after_r06f_tssfac04_sensitivity_fail_before_tssfac02_outcome",
    }
    if design["design_status"] not in allowed_design_statuses:
        raise ValueError("R06 diagnostic design is not frozen at a supported lifecycle boundary")
    if successor["version_boundary"]["applies_to_revision_family"] != "R06":
        raise ValueError("successor real-output contract is not bound to R06")

    (staging / design_path.name).write_bytes(design_path.read_bytes())
    (staging / material_path.name).write_bytes(material_path.read_bytes())
    (staging / successor_contract_path.name).write_bytes(successor_contract_path.read_bytes())

    jobs: list[dict] = []
    matrix_rows: list[dict] = []
    for experiment in design["paired_experiments"]:
        for job in experiment["jobs"]:
            job_dir = staging / job["directory"]
            job_dir.mkdir()
            deck_name = f"{job['job_id']}.k"
            deck_path = job_dir / deck_name
            deck_text, metadata = deck_for_job(
                design=design,
                r05_design=r05_design,
                material=material,
                experiment=experiment,
                job=job,
            )
            deck_path.write_text(deck_text, encoding="ascii", newline="\n")
            for source in (design_path, material_path, successor_contract_path):
                (job_dir / source.name).write_bytes(source.read_bytes())

            map_path = job_dir / "p0_case_part_map.csv"
            map_fields = [
                "job_id", "experiment_id", "role", "case_id", "source_r05_index",
                "plate_part_id", "fragment_part_id", "fragment_center_history_node_id",
                "impact_speed_mps", "mesh_mm", "plate_elements", "fragment_elements",
                "symmetry_scale_factor", "capture_plane_z_mm",
                "fragment_initial_reference_mass_kg", "termination_time_ms",
                "psfail_part_set_id", "primary_qoi_source",
            ]
            map_row = {
                "job_id": job["job_id"],
                "experiment_id": experiment["experiment_id"],
                "role": job["role"],
                "case_id": metadata["case_id"],
                "source_r05_index": metadata["source_r05_index"],
                "plate_part_id": metadata["plate_part_id"],
                "fragment_part_id": metadata["fragment_part_id"],
                "fragment_center_history_node_id": metadata["fragment_center_history_node_id"],
                "impact_speed_mps": metadata["impact_speed_mps"],
                "mesh_mm": metadata["mesh_mm"],
                "plate_elements": metadata["plate_element_count"],
                "fragment_elements": metadata["fragment_element_count"],
                "symmetry_scale_factor": 4.0,
                "capture_plane_z_mm": metadata["capture_plane_z_mm"],
                "fragment_initial_reference_mass_kg": metadata["fragment_initial_reference_mass_kg"],
                "termination_time_ms": metadata["termination_time_ms"],
                "psfail_part_set_id": metadata["psfail_part_set_id"],
                "primary_qoi_source": "d3part_fragment_set",
            }
            with map_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=map_fields)
                writer.writeheader()
                writer.writerow(map_row)

            job_contract = {
                "schema_version": "hf-p0-r06-job-contract@1.0.0",
                "package_revision": revision,
                "job_id": job["job_id"],
                "experiment_id": experiment["experiment_id"],
                "role": job["role"],
                "factor": experiment["factor"],
                "factor_level": (
                    job["termination_time_ms"]
                    if experiment["factor"] == "termination_time_ms"
                    else (
                        metadata["psfail_part_set_id"]
                        if experiment["factor"] == "control_solid_psfail_part_set"
                        else (
                            job["tssfac"]
                            if experiment["factor"] == "control_timestep_tssfac"
                            else job.get("factor_level", design["package_revision"])
                        )
                    )
                ),
                "case": metadata,
                "head_deck": deck_name,
                "resource_contract": {
                    "software_platform_label": "ls-dyna_R15_mmp_d",
                    "solver_family": "LS-DYNA R15 MPP double precision",
                    "nodes": 1,
                    "ranks": 32,
                    "walltime": "04:00:00",
                    "head_memory_mb": 512,
                    "child_memory_mb": 256,
                    "command_line_parameters": "",
                },
                "required_return_families": design["shared_measurement_upgrade"]["required_return_families"],
                "optional_return_families": design["shared_measurement_upgrade"].get(
                    "optional_return_families", []
                ),
                "status": "prepared_not_submitted_not_solver_validated",
            }
            job_contract_path = job_dir / "job_contract.json"
            job_contract_path.write_text(
                json.dumps(job_contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            manifest = {
                "schema_version": "hf-p0-r06-job-manifest@1.0.0",
                "package_revision": revision,
                "job_id": job["job_id"],
                "status": "generated_not_solver_validated",
                "evidence_class": "high_fidelity_synthetic_diagnostic_candidate",
                "deck": {
                    "path": deck_name,
                    "size_bytes": deck_path.stat().st_size,
                    "sha256": sha256_file(deck_path),
                    "node_count": metadata["plate_node_count"] + metadata["fragment_node_count"],
                    "element_count": metadata["plate_element_count"] + metadata["fragment_element_count"],
                    "node_id_span": [metadata["first_node_id"], metadata["last_node_id"]],
                    "element_id_span": [metadata["first_element_id"], metadata["last_element_id"]],
                },
                "design": {"path": design_path.name, "sha256": sha256_file(job_dir / design_path.name)},
                "material": {"path": material_path.name, "sha256": sha256_file(job_dir / material_path.name)},
                "successor_contract": {
                    "path": successor_contract_path.name,
                    "sha256": sha256_file(job_dir / successor_contract_path.name),
                    "historical_status_preserved": successor["status"],
                },
                "case_map": {"path": map_path.name, "sha256": sha256_file(map_path)},
                "job_contract": {"path": job_contract_path.name, "sha256": sha256_file(job_contract_path)},
                "limitations": design["claim_boundary"],
            }
            manifest_path = job_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            (job_dir / "UPLOAD_README.txt").write_text(
                upload_readme(
                    design,
                    experiment,
                    job,
                    deck_name,
                    manifest["deck"]["sha256"],
                    manifest["deck"]["size_bytes"],
                ),
                encoding="utf-8",
                newline="\n",
            )
            jobs.append(
                {
                    "job_id": job["job_id"],
                    "directory": job["directory"],
                    "experiment_id": experiment["experiment_id"],
                    "role": job["role"],
                    "deck": deck_name,
                    "deck_sha256": manifest["deck"]["sha256"],
                    "manifest_sha256": sha256_file(manifest_path),
                }
            )
            wave_lookup = {
                job_id: wave_number
                for wave_number, wave_name in enumerate(("wave_0", "wave_1", "wave_2"))
                for job_id in design["submission_sequence"].get(wave_name, [])
            }
            matrix_rows.append(
                {
                    "wave": wave_lookup[job["job_id"]],
                    "job_id": job["job_id"],
                    "directory": job["directory"],
                    "experiment_id": experiment["experiment_id"],
                    "role": job["role"],
                    "speed_mps": metadata["impact_speed_mps"],
                    "mesh_mm": metadata["mesh_mm"],
                    "termination_time_ms": metadata["termination_time_ms"],
                    "psfail_part_set_id": metadata["psfail_part_set_id"],
                    "deck": deck_name,
                }
            )

    matrix_path = staging / "submission_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0]))
        writer.writeheader()
        writer.writerows(matrix_rows)
    batch_manifest = {
        "schema_version": "hf-p0-r06-batch-manifest@1.0.0",
        "package_revision": revision,
        "status": "generated_pending_static_validation_not_submitted",
        "design": {"path": design_path.name, "sha256": sha256_file(staging / design_path.name)},
        "material": {"path": material_path.name, "sha256": sha256_file(staging / material_path.name)},
        "successor_contract": {
            "path": successor_contract_path.name,
            "sha256": sha256_file(staging / successor_contract_path.name),
            "frozen_pre_generation_snapshot": True,
        },
        "submission_matrix": {"path": matrix_path.name, "sha256": sha256_file(matrix_path)},
        "job_count": len(jobs),
        "paired_experiment_count": len(design["paired_experiments"]),
        "jobs": jobs,
        "prohibited_inference": "Generation and static validation do not establish target-solver parsing, numerical response, physical validity, or P0/P1 admission.",
    }
    (staging / "batch_manifest.json").write_text(
        json.dumps(batch_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (staging / "R06_BATCH_README.md").write_text(
        f"""# R06 diagnostic submission set

This directory contains {len(jobs)} solver job(s) from the frozen design.
Follow the exact submission sequence in the packaged design. Each job directory
is self-contained and has exactly one head deck.

The historical `p0_r06_real_output_contract.json` is intentionally copied as a
frozen pre-generation snapshot; its `not_generated` wording is historical and is
not edited retroactively. Current lifecycle state is recorded by the batch/job
manifests, static validation, release-readiness report, and bundle manifests.

These diagnostic candidates are not production response points. Do not use them
for P0 fitting, P1 submission, local-engine calibration, or reference labels until
the returned evidence passes the separately frozen run and response gates.
""",
        encoding="utf-8",
        newline="\n",
    )
    for attempt in range(8):
        try:
            staging.replace(output_dir)
            break
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32} or attempt == 7:
                raise
            time.sleep(0.05 * (2**attempt))
    return batch_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the non-overwriting P0 R06 paired diagnostics")
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--r05-design", type=Path, default=DEFAULT_R05_DESIGN)
    parser.add_argument("--material", type=Path, default=DEFAULT_MATERIAL)
    parser.add_argument("--successor-contract", type=Path, default=DEFAULT_SUCCESSOR_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_batch(
        design_path=args.design.resolve(),
        r05_design_path=args.r05_design.resolve(),
        material_path=args.material.resolve(),
        successor_contract_path=args.successor_contract.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps({"package_revision": result["package_revision"], "job_count": result["job_count"]}, indent=2))


if __name__ == "__main__":
    main()
