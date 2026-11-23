"""Coherent stochastic perforation and residual-velocity response."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .errors import InputValidationError, OutOfDomainError
from .utils import EPS, normalize, sigmoid


@dataclass(frozen=True)
class FragmentImpact:
    speed_mps: float
    mass_kg: float
    direction: np.ndarray
    surface_normal: np.ndarray
    equivalent_thickness_m: float


@dataclass(frozen=True)
class PenetrationOutcome:
    v50_mps: float
    impact_angle_rad: float
    incidence_cosine: float
    material_path_length_m: float
    penetration_probability: float
    sampled_ballistic_limit_mps: float
    perforated: bool
    perforation_class: str
    residual_speed_mps: float
    incident_energy_j: float
    residual_energy_j: float
    absorbed_energy_j: float
    damage_energy_j: float
    penetration_probability_semantics: str


def _damage_energy(
    model: dict,
    *,
    perforated: bool,
    absorbed_energy_j: float,
    residual_energy_j: float,
) -> float:
    damage_energy_model = model.get("damage_energy_model")
    if damage_energy_model is None:
        return residual_energy_j if perforated else 0.0
    if perforated:
        return (
            float(damage_energy_model["absorbed_weight"]) * absorbed_energy_j
            + float(damage_energy_model["residual_weight"]) * residual_energy_j
        )
    return (
        float(damage_energy_model["nonperforation_absorbed_fraction"])
        * absorbed_energy_j
    )


def _v50_mps(
    ballistic: dict,
    *,
    thickness_mm: float,
    mass_g: float,
    incidence_cosine: float,
) -> float:
    v50 = (
        float(ballistic["coefficient"])
        * thickness_mm ** float(ballistic["thickness_exponent"])
        * mass_g ** float(ballistic["mass_exponent"])
        * (1.0 / incidence_cosine) ** float(ballistic["obliquity_exponent"])
    )
    if not math.isfinite(v50) or v50 <= 0.0:
        raise InputValidationError(
            "ballistic-limit model produced a non-positive or non-finite V50"
        )
    return float(v50)


def _transition_width(ballistic: dict) -> float:
    transition = float(ballistic.get("transition_width_ratio", 0.08))
    if not math.isfinite(transition) or transition <= 0.0:
        raise InputValidationError(
            "ballistic transition width must be finite and positive"
        )
    return transition


def _sampled_ballistic_limit(
    v50_mps: float,
    transition_width_ratio: float,
    rng: np.random.Generator,
) -> float:
    quantile = float(
        np.clip(
            rng.random(),
            np.finfo(float).eps,
            1.0 - np.finfo(float).eps,
        )
    )
    logit_quantile = math.log(quantile / (1.0 - quantile))
    return float(
        v50_mps
        * math.exp(
            float(
                np.clip(
                    transition_width_ratio * logit_quantile,
                    -60.0,
                    60.0,
                )
            )
        )
    )


def resolve_fragment_impact(
    impact: FragmentImpact,
    model: dict,
    rng: np.random.Generator,
) -> PenetrationOutcome:
    speed = float(impact.speed_mps)
    mass = float(impact.mass_kg)
    thickness = float(impact.equivalent_thickness_m)
    if not math.isfinite(speed) or speed < 0.0:
        raise InputValidationError("fragment impact speed must be finite and non-negative")
    if not math.isfinite(mass) or mass <= 0.0:
        raise InputValidationError("fragment mass must be finite and positive")
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise InputValidationError("equivalent thickness must be finite and positive")

    direction = normalize(impact.direction, name="fragment direction")
    normal = normalize(impact.surface_normal, name="surface normal")
    signed_cosine = float(np.dot(direction, normal))
    if signed_cosine >= -1.0e-12:
        raise OutOfDomainError(
            "fragment impact must approach the incoming surface side",
            context={
                "reason_code": "wrong_surface_side_geometry_invariant",
                "direction_dot_surface_normal": signed_cosine,
            },
        )
    cosine = float(np.clip(-signed_cosine, 1.0e-3, 1.0))
    angle = math.acos(cosine)
    ballistic = model["ballistic_limit"]
    maximum_angle = math.radians(float(ballistic.get("maximum_obliquity_deg", 89.0)))
    incident_energy = 0.5 * mass * speed**2
    material_path_length = thickness / cosine
    mass_g = mass * 1000.0
    thickness_mm = thickness * 1000.0
    transition = _transition_width(ballistic)
    if angle > maximum_angle:
        policy = ballistic.get("high_obliquity_nonperforation_bound")
        policy_maximum_angle = (
            math.radians(float(policy["maximum_obliquity_deg"]))
            if isinstance(policy, dict)
            else None
        )
        if (
            policy_maximum_angle is not None
            and policy_maximum_angle <= maximum_angle
        ):
            raise InputValidationError(
                "high-obliquity bound maximum must exceed the validated "
                "ballistic-model obliquity"
            )
        if (
            not isinstance(policy, dict)
            or policy.get("kind")
            != "pathwise_monotone_ballistic_limit_lower_bound"
            or policy_maximum_angle is None
            or angle > policy_maximum_angle
        ):
            raise OutOfDomainError(
                "fragment impact obliquity exceeds the ballistic model domain",
                context={
                    "reason_code": "obliquity_out_of_domain",
                    "impact_angle_deg": math.degrees(angle),
                    "maximum_obliquity_deg": math.degrees(maximum_angle),
                    "high_obliquity_bound_available": isinstance(policy, dict),
                    "high_obliquity_bound_maximum_deg": (
                        math.degrees(policy_maximum_angle)
                        if policy_maximum_angle is not None
                        else None
                    ),
                },
            )
        if float(ballistic["obliquity_exponent"]) < 0.0:
            raise InputValidationError(
                "pathwise monotone high-obliquity bound requires a nonnegative "
                "obliquity exponent"
            )
        boundary_cosine = math.cos(maximum_angle)
        boundary_v50 = _v50_mps(
            ballistic,
            thickness_mm=thickness_mm,
            mass_g=mass_g,
            incidence_cosine=boundary_cosine,
        )
        boundary_probability_upper = sigmoid(
            math.log(max(speed, EPS) / boundary_v50) / transition
        )
        boundary_sampled_limit = _sampled_ballistic_limit(
            boundary_v50,
            transition,
            rng,
        )
        if speed > boundary_sampled_limit:
            raise OutOfDomainError(
                "high-obliquity contact cannot be certified as nonperforating",
                context={
                    "reason_code": "obliquity_out_of_domain",
                    "detail_code": (
                        "high_obliquity_nonperforation_not_certified"
                    ),
                    "impact_angle_deg": math.degrees(angle),
                    "maximum_obliquity_deg": math.degrees(maximum_angle),
                    "high_obliquity_bound_maximum_deg": math.degrees(
                        policy_maximum_angle
                    ),
                    "arrival_speed_mps": speed,
                    "boundary_v50_mps": boundary_v50,
                    "boundary_sampled_ballistic_limit_mps": (
                        boundary_sampled_limit
                    ),
                },
            )
        absorbed_energy = incident_energy
        damage_energy = _damage_energy(
            model,
            perforated=False,
            absorbed_energy_j=absorbed_energy,
            residual_energy_j=0.0,
        )
        return PenetrationOutcome(
            v50_mps=boundary_v50,
            impact_angle_rad=float(angle),
            incidence_cosine=float(cosine),
            material_path_length_m=float(material_path_length),
            penetration_probability=float(boundary_probability_upper),
            sampled_ballistic_limit_mps=float(boundary_sampled_limit),
            perforated=False,
            perforation_class=(
                "certified_nonperforation_by_pathwise_obliquity_bound"
            ),
            residual_speed_mps=0.0,
            incident_energy_j=float(incident_energy),
            residual_energy_j=0.0,
            absorbed_energy_j=float(absorbed_energy),
            damage_energy_j=float(damage_energy),
            penetration_probability_semantics=(
                "boundary_domain_penetration_probability_upper_bound"
            ),
        )

    v50 = _v50_mps(
        ballistic,
        thickness_mm=thickness_mm,
        mass_g=mass_g,
        incidence_cosine=cosine,
    )

    probability = sigmoid(math.log(max(speed, EPS) / v50) / transition)
    sampled_limit = _sampled_ballistic_limit(v50, transition, rng)
    perforated = speed > sampled_limit

    residual = 0.0
    if perforated:
        coefficient = float(ballistic.get("residual_velocity_coefficient", 1.0))
        exponent = float(ballistic.get("residual_velocity_exponent", 2.0))
        if not math.isfinite(coefficient) or not 0.0 < coefficient <= 1.0:
            raise InputValidationError(
                "residual velocity coefficient must be finite and lie in (0,1]"
            )
        if not math.isfinite(exponent) or exponent <= 0.0:
            raise InputValidationError(
                "residual velocity exponent must be finite and positive"
            )
        # Regularized Recht-Ipson form.  Applying ``coefficient`` directly to
        # the complete residual velocity makes a vanishingly thin target
        # remove the fixed fraction ``1 - coefficient`` of the incident
        # velocity.  That violates the zero-thickness boundary condition and
        # compounds spuriously when one homogeneous plate is subdivided.  The
        # coefficient therefore modifies only the barrier-associated part of
        # the response; its influence vanishes with the sampled limit.
        base_residual = max(
            0.0, speed**exponent - sampled_limit**exponent
        ) ** (1.0 / exponent)
        barrier_ratio = float(np.clip(sampled_limit / max(speed, EPS), 0.0, 1.0))
        barrier_modifier = 1.0 - (1.0 - coefficient) * barrier_ratio**exponent
        residual = base_residual * barrier_modifier
        residual = min(speed, float(residual))

    residual_energy = 0.5 * mass * residual**2
    absorbed_energy = max(0.0, incident_energy - residual_energy)
    damage_energy = _damage_energy(
        model,
        perforated=perforated,
        absorbed_energy_j=absorbed_energy,
        residual_energy_j=residual_energy,
    )
    return PenetrationOutcome(
        v50_mps=float(v50),
        impact_angle_rad=float(angle),
        incidence_cosine=float(cosine),
        material_path_length_m=float(material_path_length),
        penetration_probability=float(probability),
        sampled_ballistic_limit_mps=float(sampled_limit),
        perforated=bool(perforated),
        perforation_class="complete" if perforated else "none",
        residual_speed_mps=float(residual),
        incident_energy_j=float(incident_energy),
        residual_energy_j=float(residual_energy),
        absorbed_energy_j=float(absorbed_energy),
        damage_energy_j=float(damage_energy),
        penetration_probability_semantics="single_layer_theoretical_probability",
    )
