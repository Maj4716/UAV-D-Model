"""Deterministic low-cost fragment transport for sampling-plan screening.

This module is intentionally a *proposal proxy*, not a replacement damage
teacher.  It reuses the current source model, exact target ray intersections,
air-drag law and registered penetration response, while replacing repeated
Monte Carlo outcomes with weighted, deterministic fragment packets and
expected reach.  A small explicit-``DamageEngine`` validation set is still
required before the proxy may approve a large dataset run.

The first version does not model ricochet, breakup, plugs, spall, or stateful
multi-hit material evolution.  Those omissions are reported in every result.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np

from .engine import DamageEngine
from .errors import OutOfDomainError
from .fragments import Fragment, generate_fragments, propagate_fragment_speed
from .layered_penetration import PenetrationPathOutcome, resolve_penetration_path
from .parameters import nominal_target_parameters
from .penetration import FragmentImpact
from .utils import EPS, sigmoid


FAST_SAMPLING_PROXY_VERSION = "conical-multigroup-expected-transport@0.3.0"


class _QuantileRng:
    """Minimal scalar-quantile RNG accepted by the registered kernel."""

    def __init__(self, quantile: float) -> None:
        self.quantile = float(quantile)

    def random(self) -> float:
        return self.quantile


class _LatticeRng:
    """Deterministic rank-1 lattice over independent layer quantiles."""

    def __init__(self, sample_index: int, sample_count: int) -> None:
        self._base = (float(sample_index) + 0.5) / float(sample_count)
        self._dimension = 0

    def random(self) -> float:
        value = (self._base + self._dimension * 0.6180339887498949) % 1.0
        self._dimension += 1
        return float(np.clip(value, np.finfo(float).eps, 1.0 - np.finfo(float).eps))


@dataclass(frozen=True)
class _ExpectedImpact:
    penetration_probability: float
    incidence_cosine: float
    v50_mps: float
    representative_residual_speed_mps: float
    expected_absorbed_energy_j: float
    expected_damage_energy_j: float
    perforated_damage_energy_j: float
    stopped_damage_energy_j: float


def _stable_seed(*, base_seed: int, profile_id: str, source_model_id: str) -> int:
    payload = f"{base_seed}|{profile_id}|{source_model_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _representative_expected_path(
    impact: FragmentImpact,
    response: dict[str, Any],
    registry,
    *,
    layer_quadrature_samples: int,
) -> _ExpectedImpact:
    """Integrate the same public penetration path used by the full solver.

    A single layer has an analytic perforation probability, so its two
    conditional representative branches are evaluated directly through the
    registered kernel.  A declared multi-layer stack uses a deterministic
    lattice over per-layer quantiles; no residual or damage law is duplicated
    in the proxy.
    """

    has_stack = response.get("layer_stack_id") is not None
    if not has_stack:
        median = resolve_penetration_path(
            impact, response, registry, _QuantileRng(0.5)  # type: ignore[arg-type]
        )
        probability = float(np.clip(median.penetration_probability, 0.0, 1.0))
        incident_energy = float(median.incident_energy_j)
        perforated: PenetrationPathOutcome | None = None
        stopped: PenetrationPathOutcome | None = None
        if probability > 0.0:
            perforated = resolve_penetration_path(
                impact,
                response,
                registry,
                _QuantileRng(max(np.finfo(float).eps, 0.5 * probability)),  # type: ignore[arg-type]
            )
        if probability < 1.0:
            stopped = resolve_penetration_path(
                impact,
                response,
                registry,
                _QuantileRng(probability + 0.5 * (1.0 - probability)),  # type: ignore[arg-type]
            )
        residual_speed = 0.0 if perforated is None else perforated.residual_speed_mps
        perforated_absorbed = (
            incident_energy if perforated is None else perforated.absorbed_energy_j
        )
        perforated_damage = 0.0 if perforated is None else perforated.damage_energy_j
        stopped_absorbed = incident_energy if stopped is None else stopped.absorbed_energy_j
        stopped_damage = 0.0 if stopped is None else stopped.damage_energy_j
        expected_absorbed = (
            probability * perforated_absorbed
            + (1.0 - probability) * stopped_absorbed
        )
        expected_damage = (
            probability * perforated_damage
            + (1.0 - probability) * stopped_damage
        )
        reference = median
    else:
        outcomes = [
            resolve_penetration_path(
                impact,
                response,
                registry,
                _LatticeRng(index, layer_quadrature_samples),  # type: ignore[arg-type]
            )
            for index in range(layer_quadrature_samples)
        ]
        reference = outcomes[0]
        perforated_outcomes = [item for item in outcomes if item.perforated]
        stopped_outcomes = [item for item in outcomes if not item.perforated]
        probability = len(perforated_outcomes) / float(layer_quadrature_samples)
        residual_speed = (
            0.0
            if not perforated_outcomes
            else float(np.mean([item.residual_speed_mps for item in perforated_outcomes]))
        )
        perforated_damage = (
            0.0
            if not perforated_outcomes
            else float(np.mean([item.damage_energy_j for item in perforated_outcomes]))
        )
        stopped_damage = (
            0.0
            if not stopped_outcomes
            else float(np.mean([item.damage_energy_j for item in stopped_outcomes]))
        )
        expected_absorbed = float(np.mean([item.absorbed_energy_j for item in outcomes]))
        expected_damage = float(np.mean([item.damage_energy_j for item in outcomes]))

    return _ExpectedImpact(
        penetration_probability=probability,
        incidence_cosine=float(reference.incidence_cosine),
        v50_mps=float(reference.v50_mps),
        representative_residual_speed_mps=float(residual_speed),
        expected_absorbed_energy_j=float(expected_absorbed),
        expected_damage_energy_j=float(expected_damage),
        perforated_damage_energy_j=float(perforated_damage),
        stopped_damage_energy_j=float(stopped_damage),
    )


def _weighted_fragment_packets(
    fragments: list[Fragment],
    source_model: dict[str, Any],
    *,
    azimuths_per_ring: int,
) -> list[tuple[Fragment, float]]:
    """Compress fixed-mass Taylor rings into symmetric weighted ray packets."""

    if azimuths_per_ring < 1:
        raise ValueError("azimuths_per_ring must be positive")
    direction = source_model["direction"]
    ring_count = direction.get("ring_count")
    masses = np.asarray([fragment.mass_kg for fragment in fragments], dtype=np.float64)
    fixed_mass = bool(
        len(masses)
        and np.max(np.abs(masses - masses[0]))
        <= max(1.0e-15, 1.0e-10 * abs(float(masses[0])))
    )
    if direction.get("kind") != "taylor_ring" or ring_count is None or not fixed_mass:
        return [(fragment, 1.0) for fragment in fragments]

    count = len(fragments)
    ring_count = int(ring_count)
    base, remainder = divmod(count, ring_count)
    ring_sizes = [base] * ring_count
    ring_sizes[-1] += remainder
    packets: list[tuple[Fragment, float]] = []
    offset = 0
    for ring_size in ring_sizes:
        ray_count = min(azimuths_per_ring, ring_size)
        # Mid-cell representatives avoid repeatedly selecting the ring seam.
        indices = [
            min(ring_size - 1, int(math.floor((index + 0.5) * ring_size / ray_count)))
            for index in range(ray_count)
        ]
        weight = float(ring_size / ray_count)
        packets.extend((fragments[offset + index], weight) for index in indices)
        offset += ring_size
    return packets


class FastSamplingProxy:
    """Expected fragment-packet transport over a fixed compiled target."""

    def __init__(
        self,
        engine: DamageEngine,
        *,
        source_seed: int = 2026111401,
        azimuths_per_ring: int = 4,
        reach_probability_cutoff: float = 1.0e-5,
        component_support_threshold: float = 0.05,
        layer_quadrature_samples: int = 16,
    ) -> None:
        if azimuths_per_ring < 1:
            raise ValueError("azimuths_per_ring must be positive")
        if not 0.0 <= reach_probability_cutoff < 1.0:
            raise ValueError("reach_probability_cutoff must lie in [0,1)")
        if component_support_threshold < 0.0:
            raise ValueError("component_support_threshold must be non-negative")
        if layer_quadrature_samples < 4:
            raise ValueError("layer_quadrature_samples must be at least four")
        self.engine = engine
        self.source_seed = int(source_seed)
        self.azimuths_per_ring = int(azimuths_per_ring)
        self.reach_probability_cutoff = float(reach_probability_cutoff)
        self.component_support_threshold = float(component_support_threshold)
        self.layer_quadrature_samples = int(layer_quadrature_samples)
        self.parameters = nominal_target_parameters(engine.target_data)

    def evaluate(self, case: dict[str, Any]) -> dict[str, Any]:
        """Evaluate one frozen candidate without producing a formal label."""

        prepared = self.engine.prepare_case(deepcopy(case))
        source = prepared["warhead"]["fragment_source"]
        profile_id = str(prepared["munition"]["profile_id"])
        source_model_id = str(source["model_id"])
        source_model = self.engine.registry.fragment_source(source_model_id)
        rng = np.random.default_rng(
            _stable_seed(
                base_seed=self.source_seed,
                profile_id=profile_id,
                source_model_id=source_model_id,
            )
        )
        fragments = generate_fragments(prepared, self.engine.registry, rng)
        packets = _weighted_fragment_packets(
            fragments,
            source_model,
            azimuths_per_ring=self.azimuths_per_ring,
        )
        target = self.engine.compiled_target
        air_density = float(prepared["environment"]["air_density_kg_m3"])

        component_metrics: dict[str, dict[str, float]] = {
            item.object_id: {
                "expected_hits": 0.0,
                "expected_penetrations": 0.0,
                "expected_absorbed_energy_j": 0.0,
                "expected_damage_energy_j": 0.0,
            }
            for item in target.components
        }
        survival: dict[str, dict[str, float]] = {
            item.object_id: {
                state: 1.0
                for state in item.source["damage_states"]
                if state != "intact"
            }
            for item in target.components
        }

        expected_armor_hits = 0.0
        expected_armor_penetrations = 0.0
        expected_component_hits = 0.0
        expected_component_penetrations = 0.0
        terminal_reach_mass = 0.0
        out_of_domain_packet_weight = 0.0
        layered_interaction_count = 0

        for fragment, packet_weight in packets:
            speed = float(fragment.initial_speed_mps)
            reach = 1.0
            previous_distance = 0.0
            for hit in target.ray_hits(fragment.origin_m, fragment.direction):
                free_flight = max(0.0, float(hit.entry_t) - previous_distance)
                speed = propagate_fragment_speed(
                    speed, free_flight, fragment, air_density
                )
                if speed <= EPS or reach <= self.reach_probability_cutoff:
                    break
                model_id = hit.response.get("fragment_response_model_id")
                if not model_id:
                    break
                model = self.engine.registry.fragment_response(str(model_id))
                parameters = self.parameters[hit.object_id]
                try:
                    if hit.response.get("layer_stack_id") is not None:
                        layered_interaction_count += 1
                    expectation = _representative_expected_path(
                        FragmentImpact(
                            speed_mps=speed,
                            mass_kg=fragment.mass_kg,
                            direction=fragment.direction,
                            surface_normal=hit.entry_normal,
                            equivalent_thickness_m=float(
                                parameters["equivalent_thickness_m"]
                            ),
                        ),
                        hit.response,
                        self.engine.registry,
                        layer_quadrature_samples=self.layer_quadrature_samples,
                    )
                except OutOfDomainError:
                    out_of_domain_packet_weight += packet_weight * reach
                    reach = 0.0
                    break
                probability = expectation.penetration_probability
                weighted_reach = packet_weight * reach

                if hit.role == "armor":
                    expected_armor_hits += weighted_reach
                    expected_armor_penetrations += weighted_reach * probability
                else:
                    expected_component_hits += weighted_reach
                    expected_component_penetrations += weighted_reach * probability
                    metrics = component_metrics[hit.object_id]
                    metrics["expected_hits"] += weighted_reach
                    metrics["expected_penetrations"] += weighted_reach * probability
                    metrics["expected_absorbed_energy_j"] += (
                        weighted_reach * expectation.expected_absorbed_energy_j
                    )
                    metrics["expected_damage_energy_j"] += (
                        weighted_reach * expectation.expected_damage_energy_j
                    )
                    vulnerable = float(parameters["vulnerable_area_ratio"])
                    for state, state_model in model.get("damage_states", {}).items():
                        if state not in survival[hit.object_id]:
                            continue
                        energy_50 = float(
                            state_model.get(
                                "damage_energy_50_j",
                                state_model["residual_energy_50_j"],
                            )
                        )
                        slope = float(state_model["log_slope"])

                        def state_probability(energy: float) -> float:
                            if energy <= 0.0 or energy_50 <= 0.0:
                                return 0.0
                            return vulnerable * sigmoid(
                                slope * math.log(energy / energy_50)
                            )

                        event_probability = (
                            probability
                            * state_probability(
                                expectation.perforated_damage_energy_j
                            )
                            + (1.0 - probability)
                            * state_probability(expectation.stopped_damage_energy_j)
                        )
                        single_survival = 1.0 - float(
                            np.clip(reach * event_probability, 0.0, 1.0)
                        )
                        survival[hit.object_id][state] *= single_survival**packet_weight

                reach *= probability
                if (
                    reach <= self.reach_probability_cutoff
                    or expectation.representative_residual_speed_mps <= EPS
                ):
                    break
                speed = expectation.representative_residual_speed_mps
                previous_distance = max(
                    previous_distance,
                    float(hit.entry_t)
                    + float(parameters["equivalent_thickness_m"])
                    / max(expectation.incidence_cosine, 1.0e-3),
                )
            terminal_reach_mass += packet_weight * reach

        state_probabilities: dict[str, dict[str, float]] = {}
        for item in target.components:
            identifier = item.object_id
            ordered = [
                state for state in item.source["damage_states"] if state != "intact"
            ]
            previous = 1.0
            state_probabilities[identifier] = {}
            for state in ordered:
                value = min(
                    previous,
                    float(np.clip(1.0 - survival[identifier][state], 0.0, 1.0)),
                )
                state_probabilities[identifier][state] = value
                previous = value

        zero_states = {
            item.object_id: {
                state: 0.0
                for state in item.source["damage_states"]
                if state != "intact"
            }
            for item in target.components
        }
        tree = self.engine.damage_tree.evaluate(
            {
                "fragment": state_probabilities,
                "blast": zero_states,
                "combined": state_probabilities,
            }
        )
        tree_outputs = {
            name: float(value) for name, value in tree["outputs"].items()
        }
        total_absorbed = float(
            sum(
                item["expected_absorbed_energy_j"]
                for item in component_metrics.values()
            )
        )
        total_damage = float(
            sum(
                item["expected_damage_energy_j"]
                for item in component_metrics.values()
            )
        )
        supported = sorted(
            identifier
            for identifier, item in component_metrics.items()
            if item["expected_hits"] >= self.component_support_threshold
        )
        exposure_score = float(
            sum(math.log1p(item["expected_hits"]) for item in component_metrics.values())
        )
        sampling_score = float(
            exposure_score
            + 0.25 * math.log1p(total_damage)
            + 0.25 * sum(tree_outputs.values())
        )
        represented_count = float(sum(weight for _, weight in packets))
        out_of_domain_fraction = (
            0.0
            if represented_count <= 0.0
            else float(np.clip(out_of_domain_packet_weight / represented_count, 0.0, 1.0))
        )
        proxy_domain_status = (
            "complete_proxy_domain"
            if out_of_domain_packet_weight <= EPS
            else "partial_proxy_domain_censored"
        )
        return {
            "schema_version": FAST_SAMPLING_PROXY_VERSION,
            "case_id": str(prepared["case_id"]),
            "profile_id": profile_id,
            "source_model_id": source_model_id,
            "source_seed": self.source_seed,
            "azimuths_per_ring": self.azimuths_per_ring,
            "source_fragment_count": len(fragments),
            "representative_ray_count": len(packets),
            "represented_fragment_count": represented_count,
            "weight_normalization_error": represented_count - len(fragments),
            "reach_probability_cutoff": self.reach_probability_cutoff,
            "component_support_threshold": self.component_support_threshold,
            "expected_armor_hits": float(expected_armor_hits),
            "expected_armor_penetrations": float(expected_armor_penetrations),
            "expected_total_component_hits": float(expected_component_hits),
            "expected_total_component_penetrations": float(
                expected_component_penetrations
            ),
            "expected_total_component_absorbed_energy_j": total_absorbed,
            "expected_total_component_damage_energy_j": total_damage,
            "terminal_packet_weight": float(terminal_reach_mass),
            "out_of_domain_packet_weight": float(out_of_domain_packet_weight),
            "proxy_domain_status": proxy_domain_status,
            "proxy_domain_coverage": {
                "represented_packet_weight": represented_count,
                "out_of_domain_packet_weight": float(out_of_domain_packet_weight),
                "out_of_domain_packet_fraction": out_of_domain_fraction,
            },
            "formal_label_admissible": False,
            "penetration_kernel": "resolve_penetration_path",
            "layer_quadrature_samples": self.layer_quadrature_samples,
            "layered_interaction_count": int(layered_interaction_count),
            "supported_component_ids": supported,
            "supported_component_count": len(supported),
            "component_metrics": component_metrics,
            "component_state_probabilities": state_probabilities,
            "tree_outputs": tree_outputs,
            "sampling_score": sampling_score,
            "evidence_boundary": (
                "Outcome-blind sampling proxy only. Uses weighted Taylor-ring rays "
                "and deterministic expected penetration with the current unverified "
                "response registry; not a formal label, operational probability, or "
                "physical validation."
            ),
        }


__all__ = ["FAST_SAMPLING_PROXY_VERSION", "FastSamplingProxy"]
