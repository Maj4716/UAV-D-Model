from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .errors import InputValidationError
from .penetration import FragmentImpact, PenetrationOutcome, resolve_fragment_impact
from .registry import PhysicsRegistry


@dataclass(frozen=True)
class LayerPenetration:
    layer_index: int
    declared_layer_indices: tuple[int, ...]
    material_id: str
    fragment_response_model_id: str
    thickness_fraction: float
    outcome: PenetrationOutcome


@dataclass(frozen=True)
class PenetrationPathOutcome:
    v50_mps: float
    impact_angle_rad: float
    incidence_cosine: float
    material_path_length_m: float
    material_transit_time_s: float
    penetration_probability: float
    realized_branch_probability: float
    penetration_probability_semantics: str
    sampled_ballistic_limit_mps: float
    perforated: bool
    perforation_class: str
    residual_speed_mps: float
    incident_energy_j: float
    residual_energy_j: float
    absorbed_energy_j: float
    damage_energy_j: float
    layers: tuple[LayerPenetration, ...]
    layer_count: int
    declared_layer_count: int
    resolved_layer_count: int
    stopped_layer_index: int | None


def _declared_layers(response: dict, registry: PhysicsRegistry) -> list[dict]:
    stack_id = response.get("layer_stack_id")
    if stack_id is None:
        return [
            {
                "material_id": str(response["material_id"]),
                "fragment_response_model_id": str(
                    response["fragment_response_model_id"]
                ),
                "thickness_fraction": 1.0,
            }
        ]
    return list(registry.layer_stack(str(stack_id))["layers"])


def _coalesce_homogeneous_layers(layers: list[dict]) -> list[dict]:
    """Collapse purely administrative subdivisions of one material response.

    A stochastic ballistic-limit draw belongs to a continuous homogeneous
    material path, not to each row used to discretize that path.  Coalescing
    adjacent rows with the same material and response model makes the sampled
    result invariant to how a modeller partitions the same physical plate.
    Interfaces between different materials or response models remain explicit.
    """

    coalesced: list[dict] = []
    for declared_index, layer in enumerate(layers):
        fraction = float(layer["thickness_fraction"])
        if not math.isfinite(fraction) or fraction <= 0.0:
            raise InputValidationError(
                "layer thickness fractions must be finite and positive"
            )
        normalized = {
            "material_id": str(layer["material_id"]),
            "fragment_response_model_id": str(
                layer["fragment_response_model_id"]
            ),
            "thickness_fraction": fraction,
            "declared_layer_indices": [declared_index],
        }
        if (
            coalesced
            and coalesced[-1]["material_id"] == normalized["material_id"]
            and coalesced[-1]["fragment_response_model_id"]
            == normalized["fragment_response_model_id"]
        ):
            coalesced[-1]["thickness_fraction"] += fraction
            coalesced[-1]["declared_layer_indices"].append(declared_index)
        else:
            coalesced.append(normalized)
    return coalesced


def resolve_penetration_path(
    impact: FragmentImpact,
    response: dict,
    registry: PhysicsRegistry,
    rng: np.random.Generator,
) -> PenetrationPathOutcome:
    declared_layers = _declared_layers(response, registry)
    if not declared_layers:
        raise InputValidationError("layer stack must contain at least one layer")
    layers = _coalesce_homogeneous_layers(declared_layers)
    current_speed = float(impact.speed_mps)
    path_probability = 1.0
    material_transit_time_s = 0.0
    layer_results: list[LayerPenetration] = []
    stopped_layer: int | None = None
    for index, layer in enumerate(layers):
        fraction = float(layer["thickness_fraction"])
        layer_impact = FragmentImpact(
            speed_mps=current_speed,
            mass_kg=impact.mass_kg,
            direction=impact.direction,
            surface_normal=impact.surface_normal,
            equivalent_thickness_m=impact.equivalent_thickness_m * fraction,
        )
        model_id = str(layer["fragment_response_model_id"])
        outcome = resolve_fragment_impact(
            layer_impact,
            registry.fragment_response(model_id),
            rng,
        )
        material_transit_time_s += (
            2.0
            * float(outcome.material_path_length_m)
            / max(float(layer_impact.speed_mps) + float(outcome.residual_speed_mps), 1.0e-15)
        )
        path_probability *= outcome.penetration_probability
        layer_results.append(
            LayerPenetration(
                layer_index=index,
                declared_layer_indices=tuple(layer["declared_layer_indices"]),
                material_id=str(layer["material_id"]),
                fragment_response_model_id=model_id,
                thickness_fraction=fraction,
                outcome=outcome,
            )
        )
        if not outcome.perforated:
            stopped_layer = index
            current_speed = 0.0
            break
        current_speed = outcome.residual_speed_mps

    complete = stopped_layer is None and len(layer_results) == len(layers)
    first = layer_results[0].outcome
    last = layer_results[-1].outcome
    incident_energy = 0.5 * float(impact.mass_kg) * float(impact.speed_mps) ** 2
    residual_energy = 0.5 * float(impact.mass_kg) * current_speed**2
    absorbed_energy = max(0.0, incident_energy - residual_energy)
    if len(layer_results) == 1:
        damage_energy = layer_results[0].outcome.damage_energy_j
    else:
        damage_energy = 0.0
        for item in layer_results:
            layer_model = registry.fragment_response(
                item.fragment_response_model_id
            )
            damage_model = layer_model.get("damage_energy_model")
            if damage_model is None:
                continue
            if item.outcome.perforated:
                damage_energy += (
                    float(damage_model["absorbed_weight"])
                    * item.outcome.absorbed_energy_j
                )
            else:
                damage_energy += (
                    float(damage_model["nonperforation_absorbed_fraction"])
                    * item.outcome.absorbed_energy_j
                )
        if complete:
            terminal_model = registry.fragment_response(
                layer_results[-1].fragment_response_model_id
            )
            terminal_damage = terminal_model.get("damage_energy_model")
            damage_energy += (
                residual_energy
                if terminal_damage is None
                else float(terminal_damage["residual_weight"])
                * residual_energy
            )
    if len(layers) == 1:
        perforation_class = last.perforation_class
        reported_probability = first.penetration_probability
        probability_semantics = "single_layer_theoretical_probability"
    else:
        perforation_class = (
            "complete_layer_stack" if complete else "stopped_in_layer"
        )
        # A probability product evaluated only along the sampled residual-speed
        # branch is not the unconditional probability of traversing the whole
        # stack.  Publish the Bernoulli complete-stack observation for unbiased
        # aggregation across replicates and retain the branch product under an
        # explicit diagnostic name.
        reported_probability = 1.0 if complete else 0.0
        probability_semantics = "complete_stack_bernoulli_observation"
    return PenetrationPathOutcome(
        v50_mps=float(first.v50_mps),
        impact_angle_rad=float(first.impact_angle_rad),
        incidence_cosine=float(first.incidence_cosine),
        material_path_length_m=float(
            sum(item.outcome.material_path_length_m for item in layer_results)
        ),
        material_transit_time_s=float(material_transit_time_s),
        penetration_probability=float(reported_probability),
        realized_branch_probability=float(np.clip(path_probability, 0.0, 1.0)),
        penetration_probability_semantics=probability_semantics,
        sampled_ballistic_limit_mps=float(last.sampled_ballistic_limit_mps),
        perforated=bool(complete),
        perforation_class=perforation_class,
        residual_speed_mps=float(current_speed),
        incident_energy_j=float(incident_energy),
        residual_energy_j=float(residual_energy),
        absorbed_energy_j=float(absorbed_energy),
        damage_energy_j=float(damage_energy),
        layers=tuple(layer_results),
        layer_count=len(layer_results),
        declared_layer_count=len(declared_layers),
        resolved_layer_count=len(layer_results),
        stopped_layer_index=stopped_layer,
    )


def resolve_zero_resistance_passage(
    impact: FragmentImpact,
    *,
    perforation_class: str,
    probability_semantics: str,
) -> PenetrationPathOutcome:
    speed = float(impact.speed_mps)
    mass = float(impact.mass_kg)
    if not math.isfinite(speed) or speed < 0.0 or not math.isfinite(mass) or mass <= 0.0:
        raise InputValidationError("zero-resistance passage requires valid speed and mass")
    energy = 0.5 * mass * speed**2
    return PenetrationPathOutcome(
        v50_mps=0.0,
        impact_angle_rad=0.0,
        incidence_cosine=1.0,
        material_path_length_m=0.0,
        material_transit_time_s=0.0,
        penetration_probability=1.0,
        realized_branch_probability=1.0,
        penetration_probability_semantics=probability_semantics,
        sampled_ballistic_limit_mps=0.0,
        perforated=True,
        perforation_class=perforation_class,
        residual_speed_mps=speed,
        incident_energy_j=energy,
        residual_energy_j=energy,
        absorbed_energy_j=0.0,
        damage_energy_j=0.0,
        layers=(),
        layer_count=0,
        declared_layer_count=0,
        resolved_layer_count=0,
        stopped_layer_index=None,
    )


def resolve_existing_hole_passage(impact: FragmentImpact) -> PenetrationPathOutcome:
    return resolve_zero_resistance_passage(
        impact,
        perforation_class="existing_hole",
        probability_semantics="existing_hole_deterministic_passage",
    )


def resolve_blast_failed_panel_passage(
    impact: FragmentImpact,
) -> PenetrationPathOutcome:
    return resolve_zero_resistance_passage(
        impact,
        perforation_class="blast_failed_panel",
        probability_semantics="post_blast_failure_zero_resistance_passage",
    )
