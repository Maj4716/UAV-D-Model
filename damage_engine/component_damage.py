from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .damage_response import effective_fragment_damage_threshold
from .errors import InputValidationError
from .utils import sigmoid


@dataclass(frozen=True)
class ComponentDamageResult:
    probabilities: dict[str, float]
    raw_probabilities: dict[str, float]
    ordinal_projection_corrections: dict[str, float]
    ordinal_projection_max_correction: float
    total_damage_energy_j: float
    hit_count: int
    spatial_cluster_count: int
    spatial_cluster_max_diameter_m: float
    accumulation_model: str


class ComponentDamageAccumulator:
    """Accumulate fragment damage behind one small, auditable interface."""

    _SUPPORTED = {
        "independent_hits",
        "cumulative_damage_energy",
        "spatial_cumulative_damage_energy",
    }

    def __init__(
        self,
        *,
        ordered_states: tuple[str, ...],
        state_models: dict[str, dict],
        vulnerable_ratio: float,
        accumulation_model: str,
        spatial_correlation_length_m: float | None = None,
    ):
        if accumulation_model not in self._SUPPORTED:
            raise InputValidationError(
                "unsupported component damage accumulation model",
                context={
                    "model": accumulation_model,
                    "supported": sorted(self._SUPPORTED),
                },
            )
        if not math.isfinite(float(vulnerable_ratio)) or not 0.0 <= float(
            vulnerable_ratio
        ) <= 1.0:
            raise InputValidationError("component vulnerable ratio must lie in [0,1]")
        missing = [state for state in ordered_states if state not in state_models]
        if missing:
            raise InputValidationError(
                "component damage accumulator is missing response states",
                context={"missing_states": missing},
            )
        self.ordered_states = tuple(ordered_states)
        self.state_models = state_models
        self.vulnerable_ratio = float(vulnerable_ratio)
        self.accumulation_model = accumulation_model
        self.spatial_correlation_length_m = (
            None
            if spatial_correlation_length_m is None
            else float(spatial_correlation_length_m)
        )
        if self.accumulation_model == "spatial_cumulative_damage_energy" and (
            self.spatial_correlation_length_m is None
            or not math.isfinite(self.spatial_correlation_length_m)
            or self.spatial_correlation_length_m <= 0.0
        ):
            raise InputValidationError(
                "spatial cumulative damage requires a positive finite correlation length"
            )
        self._survival = {state: 1.0 for state in self.ordered_states}
        self._total_damage_energy_j = 0.0
        self._hit_count = 0
        self._spatial_impacts: list[tuple[float, np.ndarray]] = []

    def isolated_hit_probabilities(self, damage_energy_j: float) -> dict[str, float]:
        energy = float(damage_energy_j)
        if not math.isfinite(energy) or energy < 0.0:
            raise InputValidationError(
                "component damage energy must be finite and non-negative"
            )
        output: dict[str, float] = {}
        for state in self.ordered_states:
            state_model = self.state_models[state]
            energy_50, _ = effective_fragment_damage_threshold(state_model)
            slope = float(state_model["log_slope"])
            if not math.isfinite(slope) or slope <= 0.0:
                raise InputValidationError(
                    "component damage log slope must be finite and positive"
                )
            probability = (
                0.0
                if energy <= 0.0
                else sigmoid(slope * math.log(energy / energy_50))
            )
            output[state] = float(
                np.clip(probability * self.vulnerable_ratio, 0.0, 1.0)
            )
        return output

    def record_impact(
        self,
        damage_energy_j: float,
        *,
        location_m: np.ndarray | None = None,
    ) -> dict[str, float]:
        energy = float(damage_energy_j)
        isolated = self.isolated_hit_probabilities(energy)
        self._total_damage_energy_j += energy
        self._hit_count += 1
        if self.accumulation_model == "independent_hits":
            for state, probability in isolated.items():
                self._survival[state] *= 1.0 - probability
        elif self.accumulation_model == "spatial_cumulative_damage_energy":
            if location_m is None:
                raise InputValidationError(
                    "spatial cumulative damage requires an impact location"
                )
            location = np.asarray(location_m, dtype=np.float64)
            if location.shape != (3,) or not np.isfinite(location).all():
                raise InputValidationError(
                    "component impact location must be a finite 3-vector"
                )
            self._spatial_impacts.append((energy, location.copy()))
        return isolated

    def _spatial_clusters(self) -> list[tuple[float, float]]:
        """Return complete-link clusters as ``(energy, diameter)`` pairs.

        The correlation length bounds the spatial support of one local
        accumulation event. Complete-link membership enforces that every pair
        of impacts in a cluster lies within that bound, so a chain of nearby
        impacts cannot bridge an arbitrarily long component.
        """

        count = len(self._spatial_impacts)
        if count == 0:
            return []
        ordered_impacts = sorted(
            self._spatial_impacts,
            key=lambda item: (
                *(float(value) for value in item[1]),
                float(item[0]),
            ),
        )
        locations = [item[1] for item in ordered_impacts]
        energies = [float(item[0]) for item in ordered_impacts]
        distances = np.zeros((count, count), dtype=np.float64)
        for left in range(count):
            for right in range(left + 1, count):
                distance = float(np.linalg.norm(locations[left] - locations[right]))
                distances[left, right] = distance
                distances[right, left] = distance

        clusters: list[tuple[int, ...]] = [(index,) for index in range(count)]
        threshold = float(self.spatial_correlation_length_m)
        tolerance = 1.0e-12
        while True:
            admissible: list[
                tuple[float, tuple[int, ...], tuple[int, ...], int, int]
            ] = []
            for left in range(len(clusters)):
                for right in range(left + 1, len(clusters)):
                    maximum_distance = max(
                        distances[left_index, right_index]
                        for left_index in clusters[left]
                        for right_index in clusters[right]
                    )
                    if maximum_distance <= threshold + tolerance:
                        admissible.append(
                            (
                                float(maximum_distance),
                                clusters[left],
                                clusters[right],
                                left,
                                right,
                            )
                        )
            if not admissible:
                break
            _, _, _, left, right = min(admissible)
            merged = tuple(sorted((*clusters[left], *clusters[right])))
            clusters = [
                cluster
                for index, cluster in enumerate(clusters)
                if index not in {left, right}
            ]
            clusters.append(merged)
            clusters.sort()

        output: list[tuple[float, float]] = []
        for cluster in clusters:
            diameter = max(
                (
                    distances[left, right]
                    for position, left in enumerate(cluster)
                    for right in cluster[position + 1 :]
                ),
                default=0.0,
            )
            output.append(
                (
                    float(sum(energies[index] for index in cluster)),
                    float(diameter),
                )
            )
        return output

    def finalize(self) -> ComponentDamageResult:
        spatial_cluster_count = 0
        spatial_cluster_max_diameter_m = 0.0
        if self.accumulation_model == "independent_hits":
            raw = {
                state: float(np.clip(1.0 - self._survival[state], 0.0, 1.0))
                for state in self.ordered_states
            }
            spatial_cluster_count = self._hit_count
        elif self.accumulation_model == "cumulative_damage_energy":
            raw = self.isolated_hit_probabilities(self._total_damage_energy_j)
            spatial_cluster_count = int(self._hit_count > 0)
        else:
            spatial_clusters = self._spatial_clusters()
            spatial_cluster_count = len(spatial_clusters)
            spatial_cluster_max_diameter_m = max(
                (diameter for _, diameter in spatial_clusters),
                default=0.0,
            )
            survival = {state: 1.0 for state in self.ordered_states}
            for energy, _ in spatial_clusters:
                for state, probability in self.isolated_hit_probabilities(
                    energy
                ).items():
                    survival[state] *= 1.0 - probability
            raw = {
                state: float(np.clip(1.0 - survival[state], 0.0, 1.0))
                for state in self.ordered_states
            }

        projected: dict[str, float] = {}
        corrections: dict[str, float] = {}
        previous = 1.0
        for state in self.ordered_states:
            value = min(previous, raw[state])
            projected[state] = float(value)
            corrections[state] = float(raw[state] - value)
            previous = value
        return ComponentDamageResult(
            probabilities=projected,
            raw_probabilities=raw,
            ordinal_projection_corrections=corrections,
            ordinal_projection_max_correction=max(corrections.values(), default=0.0),
            total_damage_energy_j=float(self._total_damage_energy_j),
            hit_count=self._hit_count,
            spatial_cluster_count=spatial_cluster_count,
            spatial_cluster_max_diameter_m=float(spatial_cluster_max_diameter_m),
            accumulation_model=self.accumulation_model,
        )
