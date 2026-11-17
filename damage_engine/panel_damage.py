from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .errors import InputValidationError
from .utils import EPS, normalize


def perforation_opening_area_m2(
    *,
    projectile_area_m2: float,
    incidence_cosine: float,
    incident_energy_j: float,
    residual_energy_j: float,
    hole_area_multiplier: float,
) -> float:
    """Return a bounded geometric-plus-excess-energy perforation opening.

    The projected projectile area supplies the non-zero aperture required for
    passage.  The residual-to-incident energy ratio adds a bounded dilation
    term, so opening size responds monotonically to speed and target
    resistance without introducing another uncalibrated constant.  Obliquity
    stretches the projected opening by ``1 / incidence_cosine``.
    """

    projectile_area = float(projectile_area_m2)
    cosine = float(incidence_cosine)
    incident_energy = float(incident_energy_j)
    residual_energy = float(residual_energy_j)
    multiplier = float(hole_area_multiplier)
    if not math.isfinite(projectile_area) or projectile_area <= 0.0:
        raise InputValidationError("projectile area must be finite and positive")
    if not math.isfinite(cosine) or not 0.0 < cosine <= 1.0:
        raise InputValidationError("incidence cosine must lie in (0,1]")
    if not math.isfinite(incident_energy) or incident_energy <= 0.0:
        raise InputValidationError("incident energy must be finite and positive")
    if (
        not math.isfinite(residual_energy)
        or residual_energy < 0.0
        or residual_energy > incident_energy * (1.0 + 1.0e-12)
    ):
        raise InputValidationError(
            "residual energy must be finite and lie in [0, incident energy]"
        )
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise InputValidationError(
            "hole area multiplier must be finite and positive"
        )
    energy_ratio = float(np.clip(residual_energy / incident_energy, 0.0, 1.0))
    return float(
        multiplier * projectile_area / cosine * (1.0 + energy_ratio)
    )


@dataclass(frozen=True)
class ArmorHole:
    center_m: np.ndarray
    normal: np.ndarray
    radius_m: float
    formed_time_s: float


class PanelDamageState:
    """Spatial perforation state for armor panels within one replicate.

    The interface answers the two questions needed by downstream physics:
    whether a ray crosses an existing opening, and how much unique projected
    panel area has been removed.  Coplanar circular openings use analytic arc
    integration, so overlap area is deterministic, insertion-order invariant,
    and free of fixed-sample quadrature error.
    """

    _NORMAL_COSINE_MINIMUM = math.cos(math.radians(5.0))

    def __init__(self, panel_areas_m2: dict[str, float]):
        if not panel_areas_m2:
            raise InputValidationError("panel damage state requires at least one panel")
        self._areas: dict[str, float] = {}
        self._holes: dict[str, list[ArmorHole]] = {}
        self._opening_primitives: dict[str, list[ArmorHole]] = {}
        self._covered: dict[str, float] = {}
        self._covered_dirty: dict[str, bool] = {}
        for identifier, area in panel_areas_m2.items():
            area_value = float(area)
            if not math.isfinite(area_value) or area_value <= 0.0:
                raise InputValidationError(
                    "panel projected area must be finite and positive",
                    context={"object_id": identifier, "area_m2": area},
                )
            key = str(identifier)
            self._areas[key] = area_value
            self._holes[key] = []
            self._opening_primitives[key] = []
            self._covered[key] = 0.0
            self._covered_dirty[key] = False

    @staticmethod
    def _compatible_plane(left: ArmorHole, center: np.ndarray, normal: np.ndarray) -> bool:
        alignment = abs(float(np.dot(left.normal, normal)))
        if alignment < PanelDamageState._NORMAL_COSINE_MINIMUM:
            return False
        offset = abs(float(np.dot(center - left.center_m, left.normal)))
        tolerance = max(1.0e-6, 0.05 * left.radius_m)
        return offset <= tolerance

    @staticmethod
    def _contains_point(
        hole: ArmorHole,
        point: np.ndarray,
        normal: np.ndarray,
        clearance_radius_m: float = 0.0,
        query_time_s: float | None = None,
    ) -> bool:
        if not PanelDamageState._compatible_plane(hole, point, normal):
            return False
        if query_time_s is not None and hole.formed_time_s >= query_time_s:
            return False
        delta = point - hole.center_m
        in_plane = delta - float(np.dot(delta, hole.normal)) * hole.normal
        available_radius = hole.radius_m - clearance_radius_m
        return (
            available_radius >= -1.0e-12
            and float(np.dot(in_plane, in_plane))
            <= (available_radius + 1.0e-12) ** 2
        )

    def is_open(
        self,
        object_id: str,
        point_m: np.ndarray,
        surface_normal: np.ndarray,
        *,
        clearance_radius_m: float = 0.0,
        query_time_s: float | None = None,
    ) -> bool:
        key = str(object_id)
        if key not in self._holes:
            raise InputValidationError(
                "panel damage query references an unknown panel",
                context={"object_id": key},
            )
        point = np.asarray(point_m, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise InputValidationError("panel damage point must be a finite 3-vector")
        clearance = float(clearance_radius_m)
        if not math.isfinite(clearance) or clearance < 0.0:
            raise InputValidationError(
                "panel opening clearance radius must be finite and non-negative"
            )
        query_time = None if query_time_s is None else float(query_time_s)
        if query_time is not None and (
            not math.isfinite(query_time) or query_time < 0.0
        ):
            raise InputValidationError(
                "panel opening query time must be finite and non-negative"
            )
        normal = normalize(surface_normal, name="panel surface normal")
        candidates = (
            self._opening_primitives[key]
            if query_time is None
            else self._holes[key]
        )
        return any(
            self._contains_point(
                hole,
                point,
                normal,
                clearance,
                query_time,
            )
            for hole in candidates
        )

    def first_open_time_s(
        self,
        object_id: str,
        point_m: np.ndarray,
        surface_normal: np.ndarray,
        *,
        clearance_radius_m: float = 0.0,
    ) -> float | None:
        """Return the first physical formation time of a local opening.

        This event-time query is used when a finite-duration load spans a
        perforation event.  It deliberately returns the recorded formation
        time; an opening is inactive at that exact instant and active for any
        strictly later query time.
        """

        key = str(object_id)
        if key not in self._holes:
            raise InputValidationError(
                "panel damage query references an unknown panel",
                context={"object_id": key},
            )
        point = np.asarray(point_m, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise InputValidationError("panel damage point must be a finite 3-vector")
        clearance = float(clearance_radius_m)
        if not math.isfinite(clearance) or clearance < 0.0:
            raise InputValidationError(
                "panel opening clearance radius must be finite and non-negative"
            )
        normal = normalize(surface_normal, name="panel surface normal")
        formation_times = [
            float(hole.formed_time_s)
            for hole in self._holes[key]
            if self._contains_point(hole, point, normal, clearance)
        ]
        return min(formation_times) if formation_times else None

    def formation_times_s(self, object_id: str) -> tuple[float, ...]:
        """Return the ordered unique perforation-event times for a panel."""

        key = str(object_id)
        if key not in self._holes:
            raise InputValidationError(
                "panel damage query references an unknown panel",
                context={"object_id": key},
            )
        return tuple(
            sorted({float(hole.formed_time_s) for hole in self._holes[key]})
        )

    def local_formation_times_s(
        self,
        object_id: str,
        point_m: np.ndarray,
        surface_normal: np.ndarray,
    ) -> tuple[float, ...]:
        """Return opening-event times whose aperture contains one ray point."""

        key = str(object_id)
        if key not in self._holes:
            raise InputValidationError(
                "panel damage query references an unknown panel",
                context={"object_id": key},
            )
        point = np.asarray(point_m, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise InputValidationError("panel damage point must be a finite 3-vector")
        normal = normalize(surface_normal, name="panel surface normal")
        return tuple(
            sorted(
                {
                    float(hole.formed_time_s)
                    for hole in self._holes[key]
                    if self._contains_point(hole, point, normal)
                }
            )
        )

    def local_open_fraction(
        self,
        object_id: str,
        point_m: np.ndarray,
        surface_normal: np.ndarray,
        *,
        query_time_s: float | None = None,
    ) -> float:
        """Return finite-aperture area fraction local to one crossing ray.

        Only openings whose projected disk contains the ray/panel intersection
        contribute.  This prevents an unrelated hole elsewhere on a panel from
        increasing the transmitted load on every protected component.  The
        selected aperture union remains normalized by the declared panel area,
        so a small aligned opening yields partial rather than full transmission.
        """

        key = str(object_id)
        if key not in self._holes:
            raise InputValidationError(
                "panel damage query references an unknown panel",
                context={"object_id": key},
            )
        point = np.asarray(point_m, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise InputValidationError("panel damage point must be a finite 3-vector")
        normal = normalize(surface_normal, name="panel surface normal")
        query_time = self._validated_query_time(query_time_s)
        candidates = (
            self._opening_primitives[key]
            if query_time is None
            else self._holes[key]
        )
        local = [
            hole
            for hole in candidates
            if self._contains_point(
                hole,
                point,
                normal,
                query_time_s=query_time,
            )
        ]
        if not local:
            return 0.0
        local_area = min(self._areas[key], self._circle_union_area(local))
        return float(np.clip(local_area / self._areas[key], 0.0, 1.0))

    @staticmethod
    def _validated_query_time(query_time_s: float | None) -> float | None:
        query_time = None if query_time_s is None else float(query_time_s)
        if query_time is not None and (
            not math.isfinite(query_time) or query_time < 0.0
        ):
            raise InputValidationError(
                "panel opening query time must be finite and non-negative"
            )
        return query_time

    @staticmethod
    def _hole_sort_key(hole: ArmorHole) -> tuple:
        return (
            float(hole.formed_time_s),
            float(hole.radius_m),
            tuple(float(value) for value in hole.center_m),
            tuple(float(value) for value in hole.normal),
        )

    @staticmethod
    def _basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reference = (
            np.array([0.0, 0.0, 1.0])
            if abs(float(normal[2])) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        first = normalize(np.cross(normal, reference), name="panel hole basis")
        second = normalize(np.cross(normal, first), name="panel hole basis")
        return first, second

    @staticmethod
    def _merged_intervals(
        intervals: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if not intervals:
            return []
        ordered = sorted(intervals)
        merged: list[list[float]] = []
        for start, end in ordered:
            if not merged or start > merged[-1][1] + 1.0e-14:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return [(float(start), float(end)) for start, end in merged]

    @classmethod
    def _coplanar_circle_union_area(cls, holes: list[ArmorHole]) -> float:
        """Compute the exact union area of coplanar circular openings."""

        if not holes:
            return 0.0
        origin = holes[0].center_m
        normal = holes[0].normal
        first, second = cls._basis(normal)
        circles = [
            (
                float(np.dot(hole.center_m - origin, first)),
                float(np.dot(hole.center_m - origin, second)),
                float(hole.radius_m),
            )
            for hole in holes
        ]
        two_pi = 2.0 * math.pi
        union_area = 0.0
        for index, (center_x, center_y, radius) in enumerate(circles):
            covered: list[tuple[float, float]] = []
            fully_covered = False
            for other_index, (other_x, other_y, other_radius) in enumerate(circles):
                if other_index == index:
                    continue
                delta_x = other_x - center_x
                delta_y = other_y - center_y
                distance = math.hypot(delta_x, delta_y)
                tolerance = 1.0e-12 * max(1.0, radius, other_radius)
                if distance <= tolerance and abs(radius - other_radius) <= tolerance:
                    if other_index < index:
                        fully_covered = True
                        break
                    continue
                if distance + radius <= other_radius + tolerance:
                    fully_covered = True
                    break
                if distance >= radius + other_radius - tolerance:
                    continue
                if distance + other_radius <= radius + tolerance:
                    continue
                center_angle = math.atan2(delta_y, delta_x)
                cosine = float(
                    np.clip(
                        (distance**2 + radius**2 - other_radius**2)
                        / (2.0 * distance * radius),
                        -1.0,
                        1.0,
                    )
                )
                half_width = math.acos(cosine)
                start = (center_angle - half_width) % two_pi
                end = (center_angle + half_width) % two_pi
                if start <= end:
                    covered.append((start, end))
                else:
                    covered.append((0.0, end))
                    covered.append((start, two_pi))
            if fully_covered:
                continue
            merged = cls._merged_intervals(covered)
            cursor = 0.0
            uncovered: list[tuple[float, float]] = []
            for start, end in merged:
                if start > cursor + 1.0e-14:
                    uncovered.append((cursor, start))
                cursor = max(cursor, end)
            if cursor < two_pi - 1.0e-14:
                uncovered.append((cursor, two_pi))
            for start, end in uncovered:
                union_area += 0.5 * (
                    radius**2 * (end - start)
                    + center_x * radius * (math.sin(end) - math.sin(start))
                    - center_y * radius * (math.cos(end) - math.cos(start))
                )
        return max(0.0, float(union_area))

    @classmethod
    def _circle_union_area(cls, holes: list[ArmorHole]) -> float:
        """Sum exact circle unions over distinct physical panel planes."""

        if not holes:
            return 0.0
        parent = list(range(len(holes)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for left_index, left in enumerate(holes):
            for right_index in range(left_index + 1, len(holes)):
                right = holes[right_index]
                if cls._compatible_plane(left, right.center_m, right.normal):
                    union(left_index, right_index)
        groups: dict[int, list[ArmorHole]] = {}
        for index, hole in enumerate(holes):
            groups.setdefault(find(index), []).append(hole)
        return float(
            sum(cls._coplanar_circle_union_area(group) for group in groups.values())
        )

    def record_perforation(
        self,
        object_id: str,
        point_m: np.ndarray,
        surface_normal: np.ndarray,
        *,
        hole_area_m2: float,
        formed_time_s: float = 0.0,
        update_union_area: bool = True,
    ) -> float | None:
        """Record one opening and optionally materialize its exact union area.

        Event-driven fragment transport needs immediate spatial reuse queries
        but consumes panel integrity only after all fragment events.  Such
        callers can set ``update_union_area=False`` to defer the quadratic
        exact-union calculation; the next area, integrity, or summary query
        materializes it once.  The default retains the immediate incremental
        area return contract.
        """

        key = str(object_id)
        if key not in self._holes:
            raise InputValidationError(
                "panel perforation references an unknown panel",
                context={"object_id": key},
            )
        area = float(hole_area_m2)
        if not math.isfinite(area) or area <= 0.0:
            raise InputValidationError("armor hole area must be finite and positive")
        center = np.asarray(point_m, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise InputValidationError("armor hole center must be a finite 3-vector")
        normal = normalize(surface_normal, name="armor hole normal")
        formed_time = float(formed_time_s)
        if not math.isfinite(formed_time) or formed_time < 0.0:
            raise InputValidationError(
                "armor hole formation time must be finite and non-negative"
            )
        radius = math.sqrt(area / math.pi)
        if update_union_area and self._covered_dirty[key]:
            self._materialize_covered(key)
        history = self._holes[key]
        existing = self._opening_primitives[key]

        overlapping: list[ArmorHole] = []
        contained_primitives: list[ArmorHole] = []
        for hole in existing:
            if not self._compatible_plane(hole, center, normal):
                continue
            delta = center - hole.center_m
            in_plane_distance = float(
                np.linalg.norm(delta - float(np.dot(delta, hole.normal)) * hole.normal)
            )
            if in_plane_distance + radius <= hole.radius_m + 1.0e-12:
                # Spatial containment does not erase temporal history.  The
                # contained perforation can predate the larger opening and is
                # therefore still required by time-filtered opening queries.
                history.append(
                    ArmorHole(center.copy(), normal.copy(), radius, formed_time)
                )
                return 0.0
            if in_plane_distance < radius + hole.radius_m:
                overlapping.append(hole)
            if in_plane_distance + hole.radius_m <= radius + 1.0e-12:
                contained_primitives.append(hole)

        new_hole = ArmorHole(center.copy(), normal.copy(), radius, formed_time)
        history.append(new_hole)
        # Keep a compact spatial frontier for current-time queries.  Full event
        # history remains in ``_holes`` and is replayed for historical queries.
        if contained_primitives:
            contained_ids = {id(hole) for hole in contained_primitives}
            existing[:] = [hole for hole in existing if id(hole) not in contained_ids]
        existing.append(new_hole)
        if not update_union_area:
            self._covered_dirty[key] = True
            return None
        previous_covered = self._covered[key]
        exact_union = self._circle_union_area(existing)
        self._covered[key] = min(self._areas[key], exact_union)
        self._covered_dirty[key] = False
        increment = max(0.0, self._covered[key] - previous_covered)
        return increment

    def _materialize_covered(self, key: str) -> None:
        if not self._covered_dirty[key]:
            return
        self._covered[key] = min(
            self._areas[key],
            self._circle_union_area(self._opening_primitives[key]),
        )
        self._covered_dirty[key] = False

    def covered_area_m2(
        self,
        object_id: str,
        *,
        query_time_s: float | None = None,
    ) -> float:
        key = str(object_id)
        if key not in self._holes:
            raise InputValidationError(
                "panel damage query references an unknown panel",
                context={"object_id": key},
            )
        query_time = self._validated_query_time(query_time_s)
        if query_time is None:
            self._materialize_covered(key)
            return float(self._covered[key])

        active = [
            hole
            for hole in self._holes[key]
            if hole.formed_time_s < query_time
        ]
        if not active:
            return 0.0
        return float(
            min(self._areas[key], self._circle_union_area(active))
        )

    def integrity(
        self,
        object_id: str,
        *,
        query_time_s: float | None = None,
    ) -> float:
        key = str(object_id)
        covered = self.covered_area_m2(key, query_time_s=query_time_s)
        return float(np.clip(1.0 - covered / self._areas[key], 0.0, 1.0))

    def integrity_map(self) -> dict[str, float]:
        return {identifier: self.integrity(identifier) for identifier in self._areas}

    def opening_connected_component_count(self, object_id: str) -> int:
        """Count spatially connected current openings on compatible surfaces."""

        key = str(object_id)
        if key not in self._opening_primitives:
            raise InputValidationError(
                "panel damage query references an unknown panel",
                context={"object_id": key},
            )
        holes = self._opening_primitives[key]
        parent = list(range(len(holes)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for left_index, left in enumerate(holes):
            for right_index in range(left_index + 1, len(holes)):
                right = holes[right_index]
                if not self._compatible_plane(
                    left, right.center_m, right.normal
                ):
                    continue
                delta = right.center_m - left.center_m
                in_plane = delta - float(np.dot(delta, left.normal)) * left.normal
                if float(np.linalg.norm(in_plane)) <= (
                    left.radius_m + right.radius_m + 1.0e-12
                ):
                    union(left_index, right_index)
        return len({find(index) for index in range(len(holes))})

    def summary(self) -> dict[str, dict[str, float | int | str]]:
        for identifier in self._areas:
            self._materialize_covered(identifier)
        return {
            identifier: {
                "hole_count": len(self._opening_primitives[identifier]),
                "perforation_event_count": len(self._holes[identifier]),
                "opening_primitive_count": len(
                    self._opening_primitives[identifier]
                ),
                "opening_connected_component_count": (
                    self.opening_connected_component_count(identifier)
                ),
                "covered_area_m2": self._covered[identifier],
                "integrity": self.integrity(identifier),
                "union_method": "analytic_coplanar_circle_union",
                "reuse_time_rule": "formed_time_plus_1ns_strictly_before_query",
            }
            for identifier in self._areas
        }
