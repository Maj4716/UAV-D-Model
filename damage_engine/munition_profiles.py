"""Versioned loitering-munition profiles imported from the graduation project.

Only the three user-approved front-detonated profiles are admitted here.  The
values are source-code parameters, not experimentally calibrated evidence.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Mapping

from .errors import InputValidationError
from .utils import content_hash


PROFILE_REVISION = "loitering-munition-reference-port-v1"
PROFILE_SOURCE_ID = "BISHE-SIMULATION-ENGINE-PARAMETERS-UNVERIFIED"
PROFILE_SOURCE_LOCATION = (
    r"D:\Desktop\毕设\代码\src\loitering_munition_damage_twin"
    r"\simulation\engine.py:194-239,258-285,298-324,392-466"
)
STEEL_DENSITY_KG_M3 = 7850.0


@dataclass(frozen=True)
class MunitionProfile:
    profile_id: str
    display_name_zh: str
    projectile_length_m: float
    projectile_diameter_m: float
    warhead_length_m: float
    warhead_diameter_m: float
    wall_thickness_m: float
    charge_mass_kg: float
    gurney_constant_mps: float
    detonation_velocity_mps: float
    fragment_count: int
    fragment_mass_kg: float
    fragment_ring_count: int
    fragments_per_ring: tuple[int, ...]
    axial_coverage_ratio: float = 0.85
    spread_sigma_deg: float = 2.0
    tnt_equivalence: float = 1.0
    detonation_point: str = "front"
    fragment_shape: str = "cube"
    fragment_density_kg_m3: float = STEEL_DENSITY_KG_M3
    drag_coefficient: float = 1.05

    @property
    def casing_mass_kg(self) -> float:
        outer_radius = self.warhead_diameter_m / 2.0
        inner_radius = outer_radius - self.wall_thickness_m
        return (
            math.pi
            * (outer_radius * outer_radius - inner_radius * inner_radius)
            * self.warhead_length_m
            * STEEL_DENSITY_KG_M3
        )

    @property
    def fragment_mass_total_kg(self) -> float:
        return self.fragment_count * self.fragment_mass_kg

    @property
    def source_model_id(self) -> str:
        return f"loitering_{self.profile_id}_taylor_source"


_PROFILES = {
    "small_front": MunitionProfile(
        profile_id="small_front",
        display_name_zh="小型（前端起爆）",
        projectile_length_m=0.30,
        projectile_diameter_m=0.10,
        warhead_length_m=0.12,
        warhead_diameter_m=0.09,
        wall_thickness_m=0.0025,
        charge_mass_kg=1.0,
        gurney_constant_mps=2440.0,
        detonation_velocity_mps=6900.0,
        fragment_count=60,
        fragment_mass_kg=0.005,
        fragment_ring_count=6,
        fragments_per_ring=(10, 10, 10, 10, 10, 10),
    ),
    "medium_front": MunitionProfile(
        profile_id="medium_front",
        display_name_zh="中型（前端起爆）",
        projectile_length_m=0.45,
        projectile_diameter_m=0.15,
        warhead_length_m=0.20,
        warhead_diameter_m=0.13,
        wall_thickness_m=0.003,
        charge_mass_kg=3.0,
        gurney_constant_mps=2440.0,
        detonation_velocity_mps=6900.0,
        fragment_count=150,
        fragment_mass_kg=0.008,
        fragment_ring_count=12,
        fragments_per_ring=(12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 18),
    ),
    "large_front": MunitionProfile(
        profile_id="large_front",
        display_name_zh="大型（前端起爆）",
        projectile_length_m=0.60,
        projectile_diameter_m=0.20,
        warhead_length_m=0.28,
        warhead_diameter_m=0.18,
        wall_thickness_m=0.004,
        charge_mass_kg=5.0,
        gurney_constant_mps=2700.0,
        detonation_velocity_mps=7900.0,
        fragment_count=300,
        fragment_mass_kg=0.010,
        fragment_ring_count=18,
        fragments_per_ring=(16,) * 17 + (28,),
    ),
}

MUNITION_PROFILES: Mapping[str, MunitionProfile] = MappingProxyType(_PROFILES)
MUNITION_PROFILE_IDS = tuple(_PROFILES)


def get_munition_profile(profile_id: str) -> MunitionProfile:
    try:
        return MUNITION_PROFILES[str(profile_id)]
    except KeyError as exc:
        raise InputValidationError(
            "unsupported loitering-munition profile",
            context={
                "profile_id": profile_id,
                "allowed_profile_ids": list(MUNITION_PROFILE_IDS),
            },
        ) from exc


def profile_document(profile: MunitionProfile) -> dict:
    """Return the canonical, schema-compatible projectile metadata."""

    return {
        "profile_id": profile.profile_id,
        "profile_revision": PROFILE_REVISION,
        "display_name_zh": profile.display_name_zh,
        "projectile_length_m": profile.projectile_length_m,
        "projectile_diameter_m": profile.projectile_diameter_m,
        "warhead_length_m": profile.warhead_length_m,
        "warhead_diameter_m": profile.warhead_diameter_m,
        "wall_thickness_m": profile.wall_thickness_m,
        "detonation_point": profile.detonation_point,
        "fuze_type": "impact_contact",
        "trajectory_reference_point": "nose_fuze",
        "fragment_ring_count": profile.fragment_ring_count,
        "fragments_per_ring": list(profile.fragments_per_ring),
        "fragment_axial_coverage_ratio": profile.axial_coverage_ratio,
        "fragment_spread_sigma_deg": profile.spread_sigma_deg,
        "tnt_equivalence": profile.tnt_equivalence,
        "reference_source_id": PROFILE_SOURCE_ID,
        "reference_source_location": PROFILE_SOURCE_LOCATION,
        "validation_status": "unverified",
    }


def warhead_document(profile: MunitionProfile) -> dict:
    """Return the current engine's warhead representation for ``profile``."""

    return {
        "warhead_id": f"loitering-munition-{profile.profile_id}",
        "revision": PROFILE_REVISION,
        "charge_mass_kg": profile.charge_mass_kg,
        "casing_mass_kg": profile.casing_mass_kg,
        "fragment_source": {
            "model_id": profile.source_model_id,
            "total_count": profile.fragment_count,
            "mass_distribution": {
                "family": "fixed",
                "parameters": {"value_kg": profile.fragment_mass_kg},
                "source_id": PROFILE_SOURCE_ID,
                "valid_range_kg": [
                    profile.fragment_mass_kg * 0.999,
                    profile.fragment_mass_kg * 1.001,
                ],
            },
            "shape_class": profile.fragment_shape,
            "density_kg_m3": profile.fragment_density_kg_m3,
            "drag_model_id": "constant_cd",
            "direction_model_id": "taylor_ring",
            "initial_velocity_model_id": "gurney_cylinder",
            "gurney_constant_mps": profile.gurney_constant_mps,
            "detonation_velocity_mps": profile.detonation_velocity_mps,
        },
        "blast_source": {
            "model_id": "demo_scaled_distance_field",
            "equivalent_mass_kg": profile.charge_mass_kg * profile.tnt_equivalence,
            "equivalency_basis": "reference-code-tnt-equivalence-1.0-unverified",
            "pressure_time_model_id": "scaled-table",
            "source_id": PROFILE_SOURCE_ID,
        },
        "source_id": PROFILE_SOURCE_ID,
    }


def apply_munition_profile(case: dict, profile_id: str | None = None) -> dict:
    """Synchronize a mutable case after the profile ID has been sampled."""

    selected = profile_id or (case.get("munition") or {}).get("profile_id")
    profile = get_munition_profile(str(selected))
    case["munition"] = profile_document(profile)
    case["warhead"] = warhead_document(profile)
    return case


def validate_munition_profile(case: dict) -> None:
    """Fail closed if profile metadata and solver inputs have drifted apart."""

    munition = case.get("munition")
    if munition is None:
        return
    if not isinstance(munition, dict):
        raise InputValidationError("munition must be an object")
    profile = get_munition_profile(munition.get("profile_id"))
    expected_munition = profile_document(profile)
    expected_warhead = warhead_document(profile)
    if content_hash(munition) != content_hash(expected_munition):
        raise InputValidationError(
            "munition metadata does not match its versioned profile",
            context={"profile_id": profile.profile_id},
        )
    if content_hash(case.get("warhead")) != content_hash(expected_warhead):
        raise InputValidationError(
            "warhead solver inputs do not match the selected munition profile",
            context={"profile_id": profile.profile_id},
        )


def profile_catalog_document() -> dict:
    """Serializable catalog used by diagnostics and HTML/report tooling."""

    profiles = []
    for profile in MUNITION_PROFILES.values():
        record = asdict(profile)
        record["fragments_per_ring"] = list(profile.fragments_per_ring)
        record["casing_mass_kg"] = profile.casing_mass_kg
        record["fragment_mass_total_kg"] = profile.fragment_mass_total_kg
        record["source_model_id"] = profile.source_model_id
        profiles.append(record)
    return {
        "revision": PROFILE_REVISION,
        "source_id": PROFILE_SOURCE_ID,
        "source_location": PROFILE_SOURCE_LOCATION,
        "validation_status": "unverified",
        "profiles": deepcopy(profiles),
    }
