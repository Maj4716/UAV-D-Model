#!/usr/bin/env python
"""Design and execute the idea2 Section 6.8 warhead mechanism experiment.

The five-factor design is defined in mechanism space and mapped into detached
case/registry documents.  Existing physics registries are never overwritten.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import gzip
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.engine import ENGINE_CODE_HASH, ENGINE_VERSION, DamageEngine  # noqa: E402
from damage_engine.fragment_audit import (  # noqa: E402
    combine_warhead_target_matching_observations,
    extract_warhead_target_matching_observations,
    summarize_warhead_target_matching_observations,
)
from damage_engine import munition_profiles as munition_profile_module  # noqa: E402
from damage_engine.registry import PhysicsRegistry  # noqa: E402
from damage_engine.sensitivity import (  # noqa: E402
    ParameterInterval,
    analyze_sobol,
    sobol_saltelli_design,
)
from damage_engine.utils import content_hash, file_hash  # noqa: E402


FACTOR_ORDER = (
    "V0_scale",
    "fragment_mass_scale",
    "fragment_count_scale",
    "taylor_bound_scale",
    "spread_sigma_deg",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _validate_config(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != "warhead-mechanism-factor-design@1.0.0":
        raise ValueError("unsupported warhead mechanism design schema")
    if tuple(document.get("factor_order", ())) != FACTOR_ORDER:
        raise ValueError("warhead mechanism factor order is not canonical")
    count = int(document.get("design_count", 0))
    if count not in {32, 64}:
        raise ValueError("warhead mechanism design_count must be 32 or 64")
    ranges = document.get("factor_ranges")
    if not isinstance(ranges, Mapping):
        raise ValueError("warhead mechanism factor_ranges are missing")
    for name in FACTOR_ORDER:
        bounds = ranges.get(name)
        if (
            not isinstance(bounds, Sequence)
            or isinstance(bounds, (str, bytes))
            or len(bounds) != 2
        ):
            raise ValueError(f"factor range is malformed: {name}")
        lower, upper = map(float, bounds)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(f"factor range is invalid: {name}")


def generate_screening_design(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return a 32/64-point scrambled Sobol design in mechanism space."""

    _validate_config(document)
    try:
        from scipy.stats import qmc
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for scrambled Sobol design") from exc
    count = int(document["design_count"])
    power = int(round(math.log2(count)))
    unit = qmc.Sobol(
        d=len(FACTOR_ORDER),
        scramble=True,
        seed=int(document["design_seed"]),
    ).random_base2(power)
    ranges = document["factor_ranges"]
    values = np.column_stack(
        [
            float(ranges[name][0])
            + unit[:, index]
            * (float(ranges[name][1]) - float(ranges[name][0]))
            for index, name in enumerate(FACTOR_ORDER)
        ]
    )
    declaration = {
        "schema_version": "warhead-mechanism-screening-design@1.0.0",
        "design_id": str(document["design_id"]),
        "method": "scrambled_sobol",
        "factor_order": list(FACTOR_ORDER),
        "design_count": count,
        "seed": int(document["design_seed"]),
        "factor_ranges": deepcopy(dict(ranges)),
    }
    return {
        **declaration,
        "design_hash": content_hash(declaration),
        "unit_samples": unit,
        "samples": values,
    }


def generate_saltelli_extension(
    document: Mapping[str, Any], *, base_size: int
) -> dict[str, Any]:
    """Generate the optional first/total-order Sobol execution design."""

    _validate_config(document)
    intervals = tuple(
        ParameterInterval(
            parameter_id=name,
            lower=float(document["factor_ranges"][name][0]),
            upper=float(document["factor_ranges"][name][1]),
        )
        for name in FACTOR_ORDER
    )
    return sobol_saltelli_design(
        intervals,
        base_size=int(base_size),
        seed=int(document["design_seed"]) + 1,
    )


def design_matrix_quality(samples: Any, factor_order: Sequence[str]) -> dict[str, Any]:
    matrix = np.asarray(samples, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] < matrix.shape[1] + 2
        or matrix.shape[1] != len(factor_order)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("factor design matrix is invalid")
    standardized = (matrix - np.mean(matrix, axis=0)) / np.std(
        matrix, axis=0, ddof=1
    )
    correlation = np.corrcoef(standardized, rowvar=False)
    vif: dict[str, float] = {}
    for column, name in enumerate(factor_order):
        response = standardized[:, column]
        predictors = np.delete(standardized, column, axis=1)
        design = np.column_stack([np.ones(matrix.shape[0]), predictors])
        fitted = design @ np.linalg.lstsq(design, response, rcond=None)[0]
        residual = float(np.sum((response - fitted) ** 2))
        total = float(np.sum((response - np.mean(response)) ** 2))
        r_squared = 1.0 - residual / total if total > 0.0 else 1.0
        vif[name] = float(1.0 / max(1.0e-12, 1.0 - r_squared))
    off_diagonal = correlation - np.eye(correlation.shape[0])
    return {
        "sample_count": int(matrix.shape[0]),
        "factor_count": int(matrix.shape[1]),
        "correlation_matrix": correlation.tolist(),
        "maximum_absolute_off_diagonal_correlation": float(
            np.max(np.abs(off_diagonal))
        ),
        "vif": vif,
        "maximum_vif": float(max(vif.values())),
    }


def _case_role(case: Mapping[str, Any]) -> str | None:
    return case.get("metadata", {}).get("sampling", {}).get("role")


def load_reference_cases(path: Path, count: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            case = json.loads(line)
            if _case_role(case) not in {"reference_train", "reference_validation"}:
                continue
            cases.append(case)
            if len(cases) >= count:
                break
    if len(cases) < count:
        raise ValueError(f"requested {count} non-test Reference cases, found {len(cases)}")
    return cases


def load_profile_reference_cases(path: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            case = json.loads(line)
            if _case_role(case) not in {"reference_train", "reference_validation"}:
                continue
            profile = str(case.get("munition", {}).get("profile_id", ""))
            if profile:
                cases.setdefault(profile, case)
            if len(cases) >= 3:
                break
    if not cases:
        raise ValueError("no non-test Reference profile cases were found")
    return cases


def _fixed_fragment_mass(source: Mapping[str, Any]) -> float:
    distribution = source.get("mass_distribution", {})
    if distribution.get("family") != "fixed":
        raise ValueError("mechanism mapping currently requires fixed fragment mass")
    value = float(distribution.get("parameters", {}).get("value_kg", 0.0))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("baseline fragment mass is invalid")
    return value


def _baseline_mechanism(
    case: Mapping[str, Any], source_model: Mapping[str, Any]
) -> dict[str, float | int]:
    source = case["warhead"]["fragment_source"]
    mass = _fixed_fragment_mass(source)
    count = int(source["total_count"])
    fragment_mass = mass * count
    casing = float(case["warhead"]["casing_mass_kg"])
    if source_model["casing_mass_basis"] == "excludes_fragments":
        casing += fragment_mass
    charge = float(case["warhead"]["charge_mass_kg"])
    gurney = float(source["gurney_constant_mps"])
    v0 = gurney / math.sqrt(casing / charge + 0.5)
    detonation_velocity = float(source["detonation_velocity_mps"])
    ratio = v0 / (2.0 * detonation_velocity)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("baseline case is outside the Taylor domain")
    return {
        "V0_mps": v0,
        "fragment_mass_kg": mass,
        "fragment_count": count,
        "taylor_bound_rad": math.asin(ratio),
        "spread_sigma_deg": float(source_model["direction"]["spread_sigma_deg"]),
    }


def map_mechanism_point(
    case: Mapping[str, Any],
    registry_document: Mapping[str, Any],
    factors: Mapping[str, float],
    design_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Map one profile-relative mechanism point to detached raw parameters."""

    mapped_case = deepcopy(dict(case))
    mapped_registry = deepcopy(dict(registry_document))
    source = mapped_case["warhead"]["fragment_source"]
    model_id = str(source["model_id"])
    model = mapped_registry["fragment_source_models"][model_id]
    direction = model["direction"]
    if direction.get("kind") != "taylor_shapiro_ring":
        raise ValueError("mechanism mapping requires Taylor-Shapiro direction")
    baseline = _baseline_mechanism(mapped_case, model)
    ring_count = int(direction["ring_count"])
    requested_count = float(baseline["fragment_count"]) * float(
        factors["fragment_count_scale"]
    )
    count = max(ring_count, int(round(requested_count / ring_count)) * ring_count)
    mass = float(baseline["fragment_mass_kg"]) * float(
        factors["fragment_mass_scale"]
    )
    v0 = float(baseline["V0_mps"]) * float(factors["V0_scale"])
    theta = float(baseline["taylor_bound_rad"]) * float(
        factors["taylor_bound_scale"]
    )
    sigma = float(factors["spread_sigma_deg"])

    reasons: list[str] = []
    if not 0.0 < theta < 0.5 * math.pi:
        reasons.append("taylor_bound_outside_(0,pi/2)")
        detonation_velocity = float("nan")
    else:
        detonation_velocity = v0 / (2.0 * math.sin(theta))
    detonation_bounds = design_config["mapping_constraints"][
        "detonation_velocity_mps"
    ]
    if (
        not math.isfinite(detonation_velocity)
        or detonation_velocity < float(detonation_bounds[0])
        or detonation_velocity > float(detonation_bounds[1])
    ):
        reasons.append("detonation_velocity_outside_declared_range")
    valid_speed = tuple(map(float, model["valid_speed_mps"]))
    if not valid_speed[0] <= v0 <= valid_speed[1]:
        reasons.append("V0_outside_registry_valid_speed")

    fragment_mass = count * mass
    casing_reported = float(mapped_case["warhead"]["casing_mass_kg"])
    basis = str(model["casing_mass_basis"])
    total_metal = (
        casing_reported
        if basis == "includes_fragments"
        else casing_reported + fragment_mass
    )
    fragment_fraction = fragment_mass / total_metal
    maximum_fraction = float(model["maximum_fragment_mass_fraction_of_total_metal"])
    if fragment_fraction > maximum_fraction + 1.0e-12:
        reasons.append("fragment_mass_budget_exceeded")

    munition = mapped_case["munition"]
    diameter = float(munition["warhead_diameter_m"])
    wall = float(munition["wall_thickness_m"])
    length = float(munition["warhead_length_m"])
    coverage = float(direction["axial_coverage_ratio"])
    density = float(source["density_kg_m3"])
    characteristic_width = (mass / density) ** (1.0 / 3.0)
    inner_radius = 0.5 * diameter - wall
    per_ring = count // ring_count
    circumference_capacity = int(
        math.floor(2.0 * math.pi * inner_radius / characteristic_width)
    )
    axial_pitch = coverage * length / ring_count
    clearance = float(design_config["mapping_constraints"]["geometry_clearance_factor"])
    if count % ring_count:
        reasons.append("fragment_count_not_divisible_by_ring_count")
    if inner_radius <= 0.0 or per_ring > circumference_capacity:
        reasons.append("circumferential_fragment_packing_exceeded")
    if axial_pitch + 1.0e-12 < clearance * characteristic_width:
        reasons.append("axial_fragment_packing_exceeded")

    charge = float(mapped_case["warhead"]["charge_mass_kg"])
    effective_casing = casing_reported
    if basis == "excludes_fragments":
        effective_casing += fragment_mass
    gurney = v0 * math.sqrt(effective_casing / charge + 0.5)
    if not math.isfinite(gurney) or gurney <= 0.0:
        reasons.append("mapped_gurney_constant_invalid")

    source["total_count"] = count
    source["mass_distribution"]["parameters"]["value_kg"] = mass
    source["mass_distribution"]["valid_range_kg"] = [
        0.999 * mass,
        1.001 * mass,
    ]
    source["gurney_constant_mps"] = gurney
    source["detonation_velocity_mps"] = detonation_velocity
    direction["spread_sigma_deg"] = sigma
    munition["fragments_per_ring"] = [per_ring] * ring_count
    munition["fragment_spread_sigma_deg"] = sigma

    q = v0 / (2.0 * detonation_velocity) if detonation_velocity > 0.0 else None
    if q is None or not 0.0 < q <= 1.0 + 1.0e-12:
        reasons.append("mapped_Taylor_ratio_outside_domain")
    mapping = {
        "profile_id": str(munition["profile_id"]),
        "source_model_id": model_id,
        "baseline_mechanism": baseline,
        "mapped_mechanism": {
            "V0_mps": v0,
            "fragment_mass_kg": mass,
            "fragment_count": count,
            "taylor_bound_rad": theta,
            "taylor_bound_deg": math.degrees(theta),
            "spread_sigma_deg": sigma,
        },
        "mapped_raw_parameters": {
            "gurney_constant_mps": gurney,
            "detonation_velocity_mps": detonation_velocity,
            "fragments_per_ring": per_ring,
            "ring_count": ring_count,
            "fragment_mass_fraction_of_total_metal": fragment_fraction,
            "fragment_characteristic_width_m": characteristic_width,
            "circumference_capacity_per_ring": circumference_capacity,
            "axial_pitch_m": axial_pitch,
            "Taylor_q": q,
        },
        "admitted": not reasons,
        "rejection_reasons": sorted(set(reasons)),
    }
    return {
        "case": mapped_case,
        "registry": mapped_registry,
        "mapping": mapping,
    }


def build_mapping_ledger(
    design: Mapping[str, Any],
    profile_cases: Mapping[str, Mapping[str, Any]],
    registry_document: Mapping[str, Any],
    design_config: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    admitted_indices: list[int] = []
    rejected_indices: list[int] = []
    for index, values in enumerate(np.asarray(design["samples"], dtype=np.float64)):
        factors = {
            name: float(values[position])
            for position, name in enumerate(FACTOR_ORDER)
        }
        profiles = {
            profile: map_mechanism_point(
                case,
                registry_document,
                factors,
                design_config,
            )["mapping"]
            for profile, case in sorted(profile_cases.items())
        }
        reasons = sorted(
            {
                reason
                for mapping in profiles.values()
                for reason in mapping["rejection_reasons"]
            }
        )
        admitted = not reasons
        (admitted_indices if admitted else rejected_indices).append(index)
        rows.append(
            {
                "design_index": index,
                "design_point_id": f"mechanism-{index:03d}",
                "factors": factors,
                "admitted": admitted,
                "rejection_reasons": reasons,
                "profile_mappings": profiles,
            }
        )
    return {
        "rows": rows,
        "admitted_indices": admitted_indices,
        "rejected_indices": rejected_indices,
        "admitted_fraction": len(admitted_indices) / len(rows),
        "rejection_ledger": [rows[index] for index in rejected_indices],
    }


def _configure_execution(case: dict[str, Any], *, replicates: int, seed: int) -> None:
    case["analysis"]["replicates"] = int(replicates)
    case["analysis"]["aleatory_seed"] = int(seed)
    case["analysis"]["adaptive_ci_half_width"] = None
    case["analysis"].pop("adaptive_physical_ci", None)
    case["analysis"]["capture_fragment_audit"] = True
    case["analysis"]["capture_replicate_trace"] = False
    case["analysis"]["capture_component_state_trace"] = False


@contextmanager
def _temporary_profile_transaction(
    case: dict[str, Any], mapping: Mapping[str, Any]
):
    """Install one explicit in-memory profile variant for fail-closed validation.

    The transaction is process-local, sequential and always restored.  It does
    not modify the repository profile catalog or admit the variant to datasets.
    """

    profile_id = str(case["munition"]["profile_id"])
    original = munition_profile_module._PROFILES[profile_id]
    mechanism = mapping["mapped_mechanism"]
    raw = mapping["mapped_raw_parameters"]
    variant = replace(
        original,
        gurney_constant_mps=float(raw["gurney_constant_mps"]),
        detonation_velocity_mps=float(raw["detonation_velocity_mps"]),
        fragment_count=int(mechanism["fragment_count"]),
        fragment_mass_kg=float(mechanism["fragment_mass_kg"]),
        fragments_per_ring=(int(raw["fragments_per_ring"]),)
        * int(raw["ring_count"]),
        spread_sigma_deg=float(mechanism["spread_sigma_deg"]),
    )
    munition_profile_module._PROFILES[profile_id] = variant
    try:
        revision = munition_profile_module.TAYLOR_SHAPIRO_PROFILE_REVISION
        case["munition"] = munition_profile_module.profile_document(
            variant, profile_revision=revision
        )
        case["warhead"] = munition_profile_module.warhead_document(
            variant, profile_revision=revision
        )
        case.setdefault("metadata", {})["warhead_mechanism_factor_variant"] = {
            "contract_version": "warhead-mechanism-factor-transaction@1.0.0",
            "baseline_profile_id": profile_id,
            "baseline_profile_revision": revision,
            "variant_profile_hash": content_hash(
                {
                    "munition": case["munition"],
                    "warhead": case["warhead"],
                }
            ),
            "dataset_admission": False,
        }
        yield variant
    finally:
        munition_profile_module._PROFILES[profile_id] = original


def _stage_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    components = result["components"]
    p_hit = np.asarray(
        [item["fragment_metrics"]["hit_indicator"]["mean"] for item in components.values()],
        dtype=np.float64,
    )
    conditional_energy = []
    for item in components.values():
        statistic = item["fragment_metrics"][
            "hit_conditioned_log1p_deposited_energy_j"
        ]
        if int(statistic.get("sample_count", 0)) > 0 and statistic.get("mean") is not None:
            conditional_energy.append(float(statistic["mean"]))
    tree = np.asarray(
        [item["mean"] for item in result["damage_tree"]["outputs"].values()],
        dtype=np.float64,
    )
    return {
        "component_zero_fraction": float(np.mean(p_hit <= 0.0)),
        "any_component_support": float(np.any(p_hit > 0.0)),
        "positive_conditional_log1p_energy_count": len(conditional_energy),
        "mean_positive_conditional_log1p_energy": (
            float(np.mean(conditional_energy)) if conditional_energy else None
        ),
        "tree_zero_fraction": float(np.mean(tree <= 0.05)),
        "tree_intermediate_fraction": float(np.mean((tree > 0.05) & (tree < 0.95))),
        "tree_saturation_fraction": float(np.mean(tree >= 0.95)),
        "domain_affected_fraction": float(
            result["penetration_domain_coverage"]["affected_replicate_fraction"]
        ),
    }


def execute_design(
    design: Mapping[str, Any],
    ledger: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    registry_document: Mapping[str, Any],
    design_config: Mapping[str, Any],
    *,
    replicates: int,
    aleatory_seed: int,
) -> dict[str, Any]:
    rows_by_index = {int(row["design_index"]): row for row in ledger["rows"]}
    executions: list[dict[str, Any]] = []
    for design_index in ledger["admitted_indices"]:
        values = np.asarray(design["samples"], dtype=np.float64)[design_index]
        factors = {
            name: float(values[position])
            for position, name in enumerate(FACTOR_ORDER)
        }
        observations = []
        stage_rows = []
        for case_index, source_case in enumerate(cases):
            mapped = map_mechanism_point(
                source_case,
                registry_document,
                factors,
                design_config,
            )
            if not mapped["mapping"]["admitted"]:
                raise RuntimeError("admitted design became infeasible during execution")
            case = mapped["case"]
            _configure_execution(
                case,
                replicates=replicates,
                seed=aleatory_seed + case_index,
            )
            case["case_id"] = (
                f"{source_case['case_id']}::mechanism-{design_index:03d}"
            )
            with _temporary_profile_transaction(case, mapped["mapping"]):
                engine = DamageEngine(
                    dict(target), PhysicsRegistry(mapped["registry"])
                )
                result = engine.run(case).to_dict()
            observations.append(
                extract_warhead_target_matching_observations(
                    result["fragment_audit"]
                )
            )
            stage_rows.append(_stage_metrics(result))
        combined = combine_warhead_target_matching_observations(observations)
        matching = summarize_warhead_target_matching_observations(combined)
        numeric_fields = (
            "component_zero_fraction",
            "any_component_support",
            "tree_zero_fraction",
            "tree_intermediate_fraction",
            "tree_saturation_fraction",
            "domain_affected_fraction",
        )
        stage_summary = {
            name: float(np.mean([row[name] for row in stage_rows]))
            for name in numeric_fields
        }
        energy = [
            row["mean_positive_conditional_log1p_energy"]
            for row in stage_rows
            if row["mean_positive_conditional_log1p_energy"] is not None
        ]
        stage_summary["mean_positive_conditional_log1p_energy"] = (
            float(np.mean(energy)) if energy else None
        )
        executions.append(
            {
                "design_index": design_index,
                "design_point_id": rows_by_index[design_index]["design_point_id"],
                "factors": factors,
                "case_count": len(cases),
                "replicates": replicates,
                "common_random_number_seeds": [
                    aleatory_seed + index for index in range(len(cases))
                ],
                "stage_metrics": stage_summary,
                "matching": matching,
            }
        )
    return {
        "executed_design_count": len(executions),
        "execution_rows": executions,
        "common_random_numbers": (
            "same case-index seed across factor points; random streams may diverge "
            "after count-dependent draws"
        ),
    }


def _sobol_analysis_if_complete(
    saltelli_design: Mapping[str, Any] | None,
    output_values: Sequence[Sequence[float]] | None,
) -> dict[str, Any]:
    if saltelli_design is None or output_values is None:
        return {
            "status": "not_executed",
            "requirement": (
                "execute every Saltelli A/B/AB_i row before reporting first and total indices"
            ),
        }
    return {"status": "complete", "analysis": analyze_sobol(saltelli_design, output_values)}


def build_report(
    *,
    design_config_path: Path,
    cases_path: Path,
    target_path: Path,
    physics_path: Path,
    execute: bool,
    case_count: int,
    replicates: int,
    aleatory_seed: int,
    saltelli_base_size: int | None = None,
) -> dict[str, Any]:
    config = _json(design_config_path)
    design = generate_screening_design(config)
    quality = design_matrix_quality(design["samples"], FACTOR_ORDER)
    profiles = load_profile_reference_cases(cases_path)
    registry_document = _json(physics_path)
    ledger = build_mapping_ledger(design, profiles, registry_document, config)
    gates = config["quality_gates"]
    gate_results = {
        "maximum_absolute_factor_correlation": quality[
            "maximum_absolute_off_diagonal_correlation"
        ]
        <= float(gates["maximum_absolute_factor_correlation"]),
        "maximum_vif_below_2": quality["maximum_vif"]
        < float(gates["maximum_vif"]),
        "minimum_admitted_fraction": ledger["admitted_fraction"]
        >= float(gates["minimum_admitted_fraction"]),
    }
    execution = None
    if execute:
        cases = load_reference_cases(cases_path, case_count)
        execution = execute_design(
            design,
            ledger,
            cases,
            _json(target_path),
            registry_document,
            config,
            replicates=replicates,
            aleatory_seed=aleatory_seed,
        )
    saltelli = (
        generate_saltelli_extension(config, base_size=saltelli_base_size)
        if saltelli_base_size is not None
        else None
    )
    report = {
        "schema_version": "warhead-mechanism-factor-report@1.0.0",
        "design_config": config,
        "identity": {
            "design_config_hash": file_hash(design_config_path),
            "case_archive_hash": file_hash(cases_path),
            "target_hash": file_hash(target_path),
            "physics_registry_file_hash": file_hash(physics_path),
            "engine_version": ENGINE_VERSION,
            "engine_code_hash": ENGINE_CODE_HASH,
        },
        "screening_design": {
            **{key: value for key, value in design.items() if key not in {"unit_samples", "samples"}},
            "unit_samples": np.asarray(design["unit_samples"]).tolist(),
            "samples": np.asarray(design["samples"]).tolist(),
        },
        "design_quality": quality,
        "mapping": ledger,
        "quality_gates": gate_results,
        "quality_gate_passed": bool(all(gate_results.values())),
        "execution": execution,
        "saltelli_extension": (
            {
                "status": "designed_not_executed",
                "base_size": int(saltelli["base_size"]),
                "evaluation_count": int(np.asarray(saltelli["samples"]).shape[0]),
                "design_hash": saltelli["design_hash"],
                "analysis_contract": _sobol_analysis_if_complete(None, None),
            }
            if saltelli is not None
            else {"status": "not_requested"}
        ),
        "evidence_boundary": (
            "The design tests independent mechanism control and software traceability. "
            "Ranges, geometry and response parameters remain externally unvalidated; "
            "the scaled execution is not a physical Sobol conclusion."
        ),
    }
    report["report_hash"] = content_hash(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design-config",
        type=Path,
        default=Path("configs/uncertainty/warhead_mechanism_factor_design.v1.json"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(
            "artifacts/datasets/idea2_taylor_shapiro_continuous_r32_20261120_v3/"
            "rounds/round-000/formal_cases.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("configs/targets/target.fragment_only.v1.json"),
    )
    parser.add_argument(
        "--physics",
        type=Path,
        default=Path("configs/physics/loitering_impact.taylor_shapiro.v2.json"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--case-count", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--aleatory-seed", type=int, default=2026112062)
    parser.add_argument("--saltelli-base-size", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.case_count < 1 or args.replicates < 1:
        raise ValueError("case-count and replicates must be positive")
    report = build_report(
        design_config_path=args.design_config,
        cases_path=args.cases,
        target_path=args.target,
        physics_path=args.physics,
        execute=args.execute,
        case_count=args.case_count,
        replicates=args.replicates,
        aleatory_seed=args.aleatory_seed,
        saltelli_base_size=args.saltelli_base_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_hash": report["report_hash"],
                "quality_gate_passed": report["quality_gate_passed"],
                "admitted_design_count": len(report["mapping"]["admitted_indices"]),
                "executed_design_count": (
                    report["execution"]["executed_design_count"]
                    if report["execution"] is not None
                    else 0
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
