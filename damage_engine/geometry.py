from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .errors import InputValidationError
from .utils import EPS, normalize, quaternion_matrix_xyzw


@dataclass(frozen=True)
class RayHit:
    object_id: str
    role: str
    entry_t: float
    exit_t: float
    entry_point: np.ndarray
    entry_normal: np.ndarray
    response: dict
    source: dict

    @property
    def path_length_m(self) -> float:
        return max(0.0, float(self.exit_t - self.entry_t))


class Geometry:
    bounds_min: np.ndarray
    bounds_max: np.ndarray

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> tuple[float, float, np.ndarray] | None:
        raise NotImplementedError

    def contains(self, point: np.ndarray) -> bool:
        raise NotImplementedError

    def representative_size(self) -> np.ndarray:
        return self.bounds_max - self.bounds_min

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.bounds_min + self.bounds_max)


def _validate_rotation(rotation: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise InputValidationError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(value.T @ value, np.eye(3), rtol=0.0, atol=1.0e-8):
        raise InputValidationError(f"{name} must be orthonormal")
    determinant = float(np.linalg.det(value))
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise InputValidationError(f"{name} must be a proper rotation matrix")
    return value


def _ray_aabb(
    origin: np.ndarray,
    direction: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> bool:
    t_near = -math.inf
    t_far = math.inf
    for axis in range(3):
        if abs(direction[axis]) <= EPS:
            if origin[axis] < minimum[axis] or origin[axis] > maximum[axis]:
                return False
            continue
        first = (minimum[axis] - origin[axis]) / direction[axis]
        second = (maximum[axis] - origin[axis]) / direction[axis]
        if first > second:
            first, second = second, first
        t_near = max(t_near, first)
        t_far = min(t_far, second)
        if t_near > t_far:
            return False
    return t_far >= 0.0


@dataclass
class BoxGeometry(Geometry):
    center_value: np.ndarray
    size: np.ndarray
    rotation: np.ndarray

    def __post_init__(self):
        self.center_value = np.asarray(self.center_value, dtype=np.float64)
        self.size = np.asarray(self.size, dtype=np.float64)
        self.rotation = _validate_rotation(self.rotation, name="box rotation")
        if self.center_value.shape != (3,) or self.size.shape != (3,):
            raise InputValidationError("box center and size must be 3-vectors")
        if not np.isfinite(self.center_value).all():
            raise InputValidationError("box center must be finite")
        if np.any(self.size <= 0) or not np.isfinite(self.size).all():
            raise InputValidationError("box size must be finite and positive")
        half = self.size / 2.0
        extent = np.abs(self.rotation) @ half
        self.bounds_min = self.center_value - extent
        self.bounds_max = self.center_value + extent

    @property
    def center(self) -> np.ndarray:
        return self.center_value

    def contains(self, point: np.ndarray) -> bool:
        local = self.rotation.T @ (np.asarray(point) - self.center_value)
        return bool(np.all(np.abs(local) <= self.size / 2.0 + 1.0e-10))

    def intersect(self, origin: np.ndarray, direction: np.ndarray):
        if not _ray_aabb(origin, direction, self.bounds_min, self.bounds_max):
            return None
        local_origin = self.rotation.T @ (origin - self.center_value)
        local_direction = self.rotation.T @ direction
        half = self.size / 2.0
        t_min, t_max = -math.inf, math.inf
        entry_normal = np.zeros(3, dtype=np.float64)
        for axis in range(3):
            if abs(local_direction[axis]) <= EPS:
                if local_origin[axis] < -half[axis] or local_origin[axis] > half[axis]:
                    return None
                continue
            first = (-half[axis] - local_origin[axis]) / local_direction[axis]
            second = (half[axis] - local_origin[axis]) / local_direction[axis]
            first_normal = np.zeros(3)
            first_normal[axis] = -1.0
            if first > second:
                first, second = second, first
                first_normal[axis] = 1.0
            if first > t_min:
                t_min = first
                entry_normal = first_normal
            t_max = min(t_max, second)
            if t_min > t_max:
                return None
        if t_max < 0:
            return None
        entry = max(0.0, t_min)
        if t_min < 0:
            entry_normal = -local_direction / (np.linalg.norm(local_direction) + EPS)
        world_normal = normalize(self.rotation @ entry_normal, name="box normal")
        return float(entry), float(t_max), world_normal


@dataclass
class CylinderGeometry(Geometry):
    center_value: np.ndarray
    radius: float
    height: float
    rotation: np.ndarray

    def __post_init__(self):
        self.center_value = np.asarray(self.center_value, dtype=np.float64)
        self.radius = float(self.radius)
        self.height = float(self.height)
        self.rotation = _validate_rotation(self.rotation, name="cylinder rotation")
        if self.center_value.shape != (3,) or not np.isfinite(self.center_value).all():
            raise InputValidationError("cylinder center must be a finite 3-vector")
        if (
            not math.isfinite(self.radius)
            or not math.isfinite(self.height)
            or self.radius <= 0
            or self.height <= 0
        ):
            raise InputValidationError("cylinder radius and height must be finite and positive")
        axis = self.rotation @ np.array([0.0, 0.0, 1.0])
        half = self.height / 2.0
        extent = np.abs(axis) * half + self.radius * np.sqrt(np.maximum(0.0, 1.0 - axis * axis))
        self.bounds_min = self.center_value - extent
        self.bounds_max = self.center_value + extent

    @property
    def center(self) -> np.ndarray:
        return self.center_value

    def contains(self, point: np.ndarray) -> bool:
        local = self.rotation.T @ (np.asarray(point) - self.center_value)
        return bool(
            local[0] * local[0] + local[1] * local[1] <= self.radius * self.radius + 1.0e-10
            and abs(local[2]) <= self.height / 2.0 + 1.0e-10
        )

    def intersect(self, origin: np.ndarray, direction: np.ndarray):
        if not _ray_aabb(origin, direction, self.bounds_min, self.bounds_max):
            return None
        o = self.rotation.T @ (origin - self.center_value)
        d = self.rotation.T @ direction
        half = self.height / 2.0
        candidates: list[tuple[float, np.ndarray]] = []
        a = d[0] ** 2 + d[1] ** 2
        b = 2.0 * (o[0] * d[0] + o[1] * d[1])
        c = o[0] ** 2 + o[1] ** 2 - self.radius ** 2
        if a > EPS:
            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0:
                root = math.sqrt(discriminant)
                for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
                    z = o[2] + value * d[2]
                    if value >= 0 and -half - EPS <= z <= half + EPS:
                        point = o + value * d
                        normal = normalize(np.array([point[0], point[1], 0.0]), name="cylinder normal")
                        candidates.append((float(value), normal))
        if abs(d[2]) > EPS:
            for sign in (-1.0, 1.0):
                value = (sign * half - o[2]) / d[2]
                point = o + value * d
                if value >= 0 and point[0] ** 2 + point[1] ** 2 <= self.radius ** 2 + EPS:
                    candidates.append((float(value), np.array([0.0, 0.0, sign])))
        candidates.sort(key=lambda item: item[0])
        candidates = _deduplicate_hits(candidates)
        if not candidates:
            return None
        if self.contains(origin):
            entry = 0.0
            exit_value = candidates[0][0]
            normal = -d / (np.linalg.norm(d) + EPS)
        elif len(candidates) >= 2:
            entry, normal = candidates[0]
            exit_value = candidates[-1][0]
        else:
            return None
        return (
            float(entry),
            float(exit_value),
            normalize(self.rotation @ normal, name="cylinder world normal"),
        )


def _polygon_signed_area(vertices: np.ndarray) -> float:
    shifted = np.roll(vertices, -1, axis=0)
    return float(0.5 * np.sum(vertices[:, 0] * shifted[:, 1] - shifted[:, 0] * vertices[:, 1]))


def _point_in_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    v0, v1, v2 = c - a, b - a, point - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) <= EPS:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u >= -EPS and v >= -EPS and u + v <= 1.0 + EPS


def _triangulate_polygon(vertices: np.ndarray) -> list[tuple[int, int, int]]:
    count = len(vertices)
    if count < 3:
        raise InputValidationError("polygon needs at least three vertices")
    indices = list(range(count))
    if _polygon_signed_area(vertices) < 0:
        indices.reverse()
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(indices) > 3:
        clipped = False
        for offset in range(len(indices)):
            previous = indices[(offset - 1) % len(indices)]
            current = indices[offset]
            following = indices[(offset + 1) % len(indices)]
            a, b, c = vertices[previous], vertices[current], vertices[following]
            cross = np.cross(np.append(b - a, 0.0), np.append(c - b, 0.0))[2]
            if cross <= EPS:
                continue
            if any(
                _point_in_triangle(vertices[index], a, b, c)
                for index in indices
                if index not in {previous, current, following}
            ):
                continue
            triangles.append((previous, current, following))
            del indices[offset]
            clipped = True
            break
        guard += 1
        if not clipped or guard > count * count:
            raise InputValidationError("polygon is self-intersecting or numerically degenerate")
    triangles.append(tuple(indices))
    return triangles


def _ray_triangle_precomputed(
    origin: np.ndarray,
    direction: np.ndarray,
    a: np.ndarray,
    edge1: np.ndarray,
    edge2: np.ndarray,
    normal: np.ndarray,
) -> tuple[float, np.ndarray] | None:
    # Scalar Möller–Trumbore avoids allocating several tiny arrays in the
    # geometry hot loop.  Edges and the unit normal are compiled once.
    px = direction[1] * edge2[2] - direction[2] * edge2[1]
    py = direction[2] * edge2[0] - direction[0] * edge2[2]
    pz = direction[0] * edge2[1] - direction[1] * edge2[0]
    determinant = float(edge1[0] * px + edge1[1] * py + edge1[2] * pz)
    if abs(determinant) <= 1.0e-11:
        return None
    inverse = 1.0 / determinant
    relative = origin - a
    u = inverse * float(relative[0] * px + relative[1] * py + relative[2] * pz)
    if u < -EPS or u > 1.0 + EPS:
        return None
    qx = relative[1] * edge1[2] - relative[2] * edge1[1]
    qy = relative[2] * edge1[0] - relative[0] * edge1[2]
    qz = relative[0] * edge1[1] - relative[1] * edge1[0]
    v = inverse * float(direction[0] * qx + direction[1] * qy + direction[2] * qz)
    if v < -EPS or u + v > 1.0 + EPS:
        return None
    t = inverse * float(edge2[0] * qx + edge2[1] * qy + edge2[2] * qz)
    if t < -EPS:
        return None
    return max(0.0, t), normal


def _deduplicate_hits(values: list[tuple[float, np.ndarray]]) -> list[tuple[float, np.ndarray]]:
    output: list[tuple[float, np.ndarray]] = []
    for value, normal in values:
        if output and abs(value - output[-1][0]) <= 1.0e-9 * max(1.0, abs(value)):
            continue
        output.append((value, normal))
    return output


@dataclass
class ExtrudedPolygonGeometry(Geometry):
    vertices: np.ndarray
    direction: np.ndarray
    distance: float

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or len(self.vertices) < 3:
            raise InputValidationError("extruded polygon vertices must have shape (N,3), N>=3")
        if not np.isfinite(self.vertices).all():
            raise InputValidationError("extruded polygon vertices must be finite")
        self.direction = normalize(self.direction, name="extrusion_direction")
        self.distance = float(self.distance)
        if self.distance <= 0 or not math.isfinite(self.distance):
            raise InputValidationError("extrusion distance must be finite and positive")
        self._origin = self.vertices[0]
        plane_offsets = (self.vertices - self._origin) @ self.direction
        scale = max(1.0, float(np.ptp(self.vertices, axis=0).max()))
        if np.max(np.abs(plane_offsets)) > 1.0e-8 * scale:
            raise InputValidationError("extruded polygon vertices are not coplanar")
        edge = self.vertices[1] - self.vertices[0]
        edge -= float(np.dot(edge, self.direction)) * self.direction
        self._u = normalize(edge, name="polygon basis")
        self._v = normalize(np.cross(self.direction, self._u), name="polygon basis")
        local = self.vertices - self._origin
        self._vertices2d = np.column_stack((local @ self._u, local @ self._v))
        if abs(_polygon_signed_area(self._vertices2d)) <= 1.0e-14:
            raise InputValidationError("extruded polygon has zero area")
        triangles = _triangulate_polygon(self._vertices2d)
        top = self.vertices + self.direction * self.distance
        faces: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i, j, k in triangles:
            faces.append((self.vertices[k], self.vertices[j], self.vertices[i]))
            faces.append((top[i], top[j], top[k]))
        for i in range(len(self.vertices)):
            j = (i + 1) % len(self.vertices)
            faces.append((self.vertices[i], self.vertices[j], top[j]))
            faces.append((self.vertices[i], top[j], top[i]))
        self._faces = [
            (
                a,
                b - a,
                c - a,
                normalize(np.cross(b - a, c - a), name="triangle normal"),
            )
            for a, b, c in faces
        ]
        all_vertices = np.vstack((self.vertices, top))
        self.bounds_min = np.min(all_vertices, axis=0)
        self.bounds_max = np.max(all_vertices, axis=0)

    def _point_in_polygon(self, point2d: np.ndarray) -> bool:
        x, y = point2d
        scale = max(1.0, float(np.ptp(self._vertices2d, axis=0).max()))
        tolerance = 16.0 * np.finfo(np.float64).eps * scale
        for index, start in enumerate(self._vertices2d):
            end = self._vertices2d[(index + 1) % len(self._vertices2d)]
            edge = end - start
            relative = point2d - start
            cross = edge[0] * relative[1] - edge[1] * relative[0]
            if abs(cross) <= tolerance * max(1.0, float(np.linalg.norm(edge))):
                projection = float(np.dot(relative, edge))
                if -tolerance <= projection <= float(np.dot(edge, edge)) + tolerance:
                    return True
        inside = False
        previous = len(self._vertices2d) - 1
        for current in range(len(self._vertices2d)):
            xi, yi = self._vertices2d[current]
            xj, yj = self._vertices2d[previous]
            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + EPS) + xi
            ):
                inside = not inside
            previous = current
        return inside

    def contains(self, point: np.ndarray) -> bool:
        relative = np.asarray(point) - self._origin
        axial = float(np.dot(relative, self.direction))
        planar = np.array([np.dot(relative, self._u), np.dot(relative, self._v)])
        return -1.0e-10 <= axial <= self.distance + 1.0e-10 and self._point_in_polygon(planar)

    def intersect(self, origin: np.ndarray, direction: np.ndarray):
        if not _ray_aabb(origin, direction, self.bounds_min, self.bounds_max):
            return None
        candidates = []
        for a, edge1, edge2, normal in self._faces:
            hit = _ray_triangle_precomputed(origin, direction, a, edge1, edge2, normal)
            if hit is not None:
                candidates.append(hit)
        candidates.sort(key=lambda item: item[0])
        candidates = _deduplicate_hits(candidates)
        if not candidates:
            return None
        if self.contains(origin):
            return 0.0, float(candidates[0][0]), -direction
        if len(candidates) < 2:
            return None
        return float(candidates[0][0]), float(candidates[-1][0]), candidates[0][1]


@dataclass(frozen=True)
class CompiledObject:
    object_id: str
    role: str
    geometry: Geometry
    response: dict
    source: dict


@dataclass(frozen=True)
class _BVHNode:
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    left: "_BVHNode | None" = None
    right: "_BVHNode | None" = None
    object_indices: tuple[int, ...] = ()


@dataclass
class CompiledTarget:
    target_id: str
    objects: tuple[CompiledObject, ...]
    armor: tuple[CompiledObject, ...]
    components: tuple[CompiledObject, ...]
    component_order: tuple[str, ...]

    def __post_init__(self):
        self._bounds_min = np.asarray(
            [item.geometry.bounds_min for item in self.objects], dtype=np.float64
        )
        self._bounds_max = np.asarray(
            [item.geometry.bounds_max for item in self.objects], dtype=np.float64
        )
        scale = np.maximum(
            1.0,
            np.maximum(np.abs(self._bounds_min), np.abs(self._bounds_max)),
        )
        tolerance = 16.0 * np.finfo(np.float64).eps * scale
        self._bounds_lower = self._bounds_min - tolerance
        self._bounds_upper = self._bounds_max + tolerance
        self._bvh_leaf_size = 8
        self._bvh_root = self._build_bvh(np.arange(len(self.objects), dtype=np.int64))

    def _build_bvh(self, indices: np.ndarray) -> _BVHNode:
        bounds_min = np.min(self._bounds_lower[indices], axis=0)
        bounds_max = np.max(self._bounds_upper[indices], axis=0)
        if len(indices) <= self._bvh_leaf_size:
            return _BVHNode(
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                object_indices=tuple(int(index) for index in indices),
            )
        centers = 0.5 * (self._bounds_lower[indices] + self._bounds_upper[indices])
        axis = int(np.argmax(np.ptp(centers, axis=0)))
        order = np.argsort(centers[:, axis], kind="stable")
        ordered = indices[order]
        midpoint = len(ordered) // 2
        return _BVHNode(
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            left=self._build_bvh(ordered[:midpoint]),
            right=self._build_bvh(ordered[midpoint:]),
        )

    @staticmethod
    def _ray_intersects_bounds(
        origin: np.ndarray,
        direction: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        maximum_distance_m: float,
    ) -> bool:
        near = -math.inf
        far = math.inf
        for axis in range(3):
            component = float(direction[axis])
            if abs(component) <= EPS:
                if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                    return False
                continue
            first = (lower[axis] - origin[axis]) / component
            second = (upper[axis] - origin[axis]) / component
            near = max(near, min(first, second))
            far = min(far, max(first, second))
            if near > far:
                return False
        return far >= 0.0 and near <= maximum_distance_m

    def _candidate_indices_bruteforce(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        maximum_distance_m: float = math.inf,
    ) -> np.ndarray:
        """Vectorized reference broad phase retained for equivalence testing."""
        count = len(self.objects)
        near = np.full(count, -np.inf, dtype=np.float64)
        far = np.full(count, np.inf, dtype=np.float64)
        possible = np.ones(count, dtype=bool)
        for axis in range(3):
            component = float(direction[axis])
            if abs(component) <= EPS:
                possible &= (
                    (origin[axis] >= self._bounds_lower[:, axis])
                    & (origin[axis] <= self._bounds_upper[:, axis])
                )
                continue
            first = (self._bounds_lower[:, axis] - origin[axis]) / component
            second = (self._bounds_upper[:, axis] - origin[axis]) / component
            near = np.maximum(near, np.minimum(first, second))
            far = np.minimum(far, np.maximum(first, second))
            possible &= near <= far
        possible &= far >= 0.0
        possible &= near <= maximum_distance_m
        return np.flatnonzero(possible)

    def _candidate_indices(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        maximum_distance_m: float = math.inf,
    ) -> np.ndarray:
        if len(self.objects) <= self._bvh_leaf_size:
            return self._candidate_indices_bruteforce(
                origin, direction, maximum_distance_m
            )
        candidates: list[int] = []
        stack = [self._bvh_root]
        while stack:
            node = stack.pop()
            if not self._ray_intersects_bounds(
                origin,
                direction,
                node.bounds_min,
                node.bounds_max,
                maximum_distance_m,
            ):
                continue
            if node.object_indices:
                for index in node.object_indices:
                    if self._ray_intersects_bounds(
                        origin,
                        direction,
                        self._bounds_lower[index],
                        self._bounds_upper[index],
                        maximum_distance_m,
                    ):
                        candidates.append(index)
                continue
            # Visit left first for deterministic diagnostics; final hits are
            # sorted independently by physical entry distance and object ID.
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)
        return np.asarray(sorted(candidates), dtype=np.int64)

    def ray_hits(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        *,
        maximum_distance_m: float = math.inf,
    ) -> list[RayHit]:
        origin = np.asarray(origin, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise InputValidationError("ray origin must be a finite 3-vector")
        direction = normalize(direction, name="ray direction")
        maximum_distance_m = float(maximum_distance_m)
        if math.isnan(maximum_distance_m) or maximum_distance_m < 0:
            raise InputValidationError("maximum ray distance must be non-negative")
        hits: list[RayHit] = []
        # Every concrete geometry performs its own exact AABB rejection before
        # the narrow phase.  For a small target, constructing vectorized
        # broad-phase arrays duplicates that work and costs more than visiting
        # the handful of objects directly.  Object order matches the sorted
        # brute-force candidate order, so deterministic hit ordering is
        # unchanged.  Larger targets retain the BVH path.
        object_indices: Iterable[int] = (
            range(len(self.objects))
            if len(self.objects) <= self._bvh_leaf_size
            else self._candidate_indices(origin, direction, maximum_distance_m)
        )
        for object_index in object_indices:
            item = self.objects[int(object_index)]
            intersection = item.geometry.intersect(origin, direction)
            if intersection is None:
                continue
            entry, exit_value, normal = intersection
            if entry > maximum_distance_m:
                continue
            hits.append(
                RayHit(
                    object_id=item.object_id,
                    role=item.role,
                    entry_t=entry,
                    exit_t=min(exit_value, maximum_distance_m),
                    entry_point=origin + entry * direction,
                    entry_normal=normalize(normal, name="surface normal"),
                    response=item.response,
                    source=item.source,
                )
            )
        hits.sort(key=lambda hit: (hit.entry_t, 0 if hit.role == "armor" else 1, hit.object_id))
        return hits

    def armor_on_segment(self, origin: np.ndarray, destination: np.ndarray) -> list[RayHit]:
        ray = np.asarray(destination, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
        distance = float(np.linalg.norm(ray))
        if distance <= EPS:
            return []
        endpoint_exclusive = float(np.nextafter(distance, 0.0))
        return [
            hit
            for hit in self.ray_hits(
                origin,
                ray / distance,
                maximum_distance_m=endpoint_exclusive,
            )
            if hit.role == "armor"
        ]


def geometry_from_json(data: dict) -> Geometry:
    shape = data.get("shape")
    if not shape:
        raise InputValidationError("geometry shape is required")
    required_by_shape = {
        "box": ("center_m", "size_lwh_m", "orientation_xyzw"),
        "cylinder": ("center_m", "radius_m", "height_m", "orientation_xyzw"),
        "extruded_polygon": (
            "vertices_m",
            "extrusion_direction",
            "extrusion_distance_m",
        ),
    }
    required = required_by_shape.get(str(shape))
    if required is None:
        raise InputValidationError(f"unsupported geometry shape: {shape}")
    missing = [field for field in required if field not in data]
    if missing:
        raise InputValidationError(
            "geometry is missing required SI fields; coordinates and dimensions must use metre fields",
            context={"shape": shape, "missing": missing},
        )
    if shape == "box":
        return BoxGeometry(
            center_value=np.asarray(data["center_m"], dtype=np.float64),
            size=np.asarray(data["size_lwh_m"], dtype=np.float64),
            rotation=quaternion_matrix_xyzw(data["orientation_xyzw"]),
        )
    if shape == "cylinder":
        return CylinderGeometry(
            center_value=np.asarray(data["center_m"], dtype=np.float64),
            radius=float(data["radius_m"]),
            height=float(data["height_m"]),
            rotation=quaternion_matrix_xyzw(data["orientation_xyzw"]),
        )
    if shape == "extruded_polygon":
        return ExtrudedPolygonGeometry(
            vertices=np.asarray(data["vertices_m"], dtype=np.float64),
            direction=np.asarray(data["extrusion_direction"], dtype=np.float64),
            distance=float(data["extrusion_distance_m"]),
        )
    raise InputValidationError(f"unsupported geometry shape: {shape}")


def compile_target(target: dict) -> CompiledTarget:
    objects: list[CompiledObject] = []
    seen: set[str] = set()
    for role, entries in (("armor", target["armor"]), ("component", target["components"])):
        for entry in entries:
            object_id = str(entry["id"])
            if object_id in seen:
                raise InputValidationError(
                    "target object IDs must be globally unique",
                    context={"duplicate_id": object_id},
                )
            seen.add(object_id)
            objects.append(
                CompiledObject(
                    object_id=object_id,
                    role=role,
                    geometry=geometry_from_json(entry["geometry"]),
                    response=entry["response"],
                    source=entry,
                )
            )
    armor = tuple(item for item in objects if item.role == "armor")
    components = tuple(item for item in objects if item.role == "component")
    if not armor or not components:
        raise InputValidationError("target must contain armor and internal components")
    return CompiledTarget(
        target_id=str(target["target_id"]),
        objects=tuple(objects),
        armor=armor,
        components=components,
        component_order=tuple(item.object_id for item in components),
    )


def polygon_area_3d(vertices: Iterable[Iterable[float]], normal: np.ndarray) -> float:
    points = np.asarray(list(vertices), dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3:
        raise InputValidationError("polygon area requires at least three 3D vertices")
    if not np.isfinite(points).all():
        raise InputValidationError("polygon area vertices must be finite")
    area_vector = np.zeros(3)
    for index in range(len(points)):
        area_vector += np.cross(points[index], points[(index + 1) % len(points)])
    return abs(float(np.dot(area_vector, normalize(normal)))) / 2.0
