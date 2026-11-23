"""Dependency-light Morris screening and variance-based Sobol analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import InputValidationError
from .utils import content_hash


@dataclass(frozen=True)
class ParameterInterval:
    parameter_id: str
    lower: float
    upper: float
    scale: str = "linear"

    def __post_init__(self) -> None:
        if not self.parameter_id or not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise InputValidationError("sensitivity interval must be finite and named")
        if self.lower >= self.upper:
            raise InputValidationError("sensitivity interval lower bound must be below upper")
        if self.scale not in {"linear", "log"}:
            raise InputValidationError("sensitivity interval scale must be linear or log")
        if self.scale == "log" and self.lower <= 0.0:
            raise InputValidationError("log sensitivity intervals must be positive")

    def map_unit(self, unit: np.ndarray) -> np.ndarray:
        if self.scale == "linear":
            return self.lower + unit * (self.upper - self.lower)
        return np.exp(np.log(self.lower) + unit * (np.log(self.upper) - np.log(self.lower)))


def intervals_from_document(document: Mapping[str, Any]) -> tuple[ParameterInterval, ...]:
    records = document.get("parameters") if isinstance(document, Mapping) else None
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise InputValidationError("sensitivity document requires parameter records")
    intervals = tuple(
        ParameterInterval(
            parameter_id=str(record["parameter_id"]),
            lower=float(record["range"][0]),
            upper=float(record["range"][1]),
            scale=str(record.get("scale", "linear")),
        )
        for record in records
    )
    if len({item.parameter_id for item in intervals}) != len(intervals):
        raise InputValidationError("sensitivity parameter IDs must be unique")
    return intervals


def _map_design(unit: np.ndarray, intervals: Sequence[ParameterInterval]) -> np.ndarray:
    if unit.ndim != 2 or unit.shape[1] != len(intervals):
        raise InputValidationError("unit design dimension does not match intervals")
    return np.column_stack(
        [interval.map_unit(unit[:, index]) for index, interval in enumerate(intervals)]
    )


def morris_design(
    intervals: Sequence[ParameterInterval],
    *,
    trajectories: int,
    levels: int = 8,
    seed: int,
) -> dict[str, Any]:
    """Generate randomized one-at-a-time trajectories on a finite grid."""

    intervals = tuple(intervals)
    dimension = len(intervals)
    if dimension < 1 or trajectories < 1 or levels < 4 or levels % 2:
        raise InputValidationError("Morris design requires parameters, trajectories, and even levels >= 4")
    rng = np.random.default_rng(int(seed))
    delta = levels / (2.0 * (levels - 1.0))
    rows: list[np.ndarray] = []
    changed: list[int] = []
    signs: list[int] = []
    trajectory_ids: list[int] = []
    for trajectory in range(trajectories):
        direction = rng.choice(np.asarray([-1.0, 1.0]), size=dimension)
        base = np.empty(dimension, dtype=np.float64)
        grid = np.linspace(0.0, 1.0 - delta, levels // 2)
        for index in range(dimension):
            start = float(rng.choice(grid))
            base[index] = start if direction[index] > 0 else start + delta
        order = rng.permutation(dimension)
        current = base.copy()
        rows.append(current.copy())
        trajectory_ids.append(trajectory)
        for index in order:
            current = current.copy()
            current[index] += direction[index] * delta
            rows.append(current.copy())
            trajectory_ids.append(trajectory)
            changed.append(int(index))
            signs.append(int(direction[index]))
    unit = np.asarray(rows)
    physical = _map_design(unit, intervals)
    declaration = {
        "schema_version": "morris-design@1.0.0",
        "method": "randomized_one_at_a_time_grid",
        "parameter_ids": [item.parameter_id for item in intervals],
        "trajectories": trajectories,
        "levels": levels,
        "delta_unit": delta,
        "seed": int(seed),
    }
    return {
        **declaration,
        "design_hash": content_hash(declaration),
        "unit_samples": unit,
        "samples": physical,
        "trajectory_id": np.asarray(trajectory_ids, dtype=np.int64),
        "changed_parameter_index": np.asarray(changed, dtype=np.int64),
        "step_sign": np.asarray(signs, dtype=np.int8),
    }


def analyze_morris(design: Mapping[str, Any], outputs: Any) -> dict[str, Any]:
    samples = np.asarray(design["samples"], dtype=np.float64)
    values = np.asarray(outputs, dtype=np.float64)
    parameter_ids = tuple(map(str, design["parameter_ids"]))
    dimension = len(parameter_ids)
    trajectories = int(design["trajectories"])
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] != samples.shape[0] or not np.isfinite(values).all():
        raise InputValidationError("Morris outputs must be finite and align with samples")
    effects: list[list[np.ndarray]] = [[] for _ in range(dimension)]
    changed = np.asarray(design["changed_parameter_index"], dtype=np.int64)
    cursor = 0
    for trajectory_index in range(trajectories):
        for step in range(dimension):
            left = cursor + step
            right = left + 1
            index = int(changed[trajectory_index * dimension + step])
            denominator = samples[right, index] - samples[left, index]
            effects[index].append((values[right] - values[left]) / denominator)
        cursor += dimension + 1
    rows = []
    for index, parameter_id in enumerate(parameter_ids):
        array = np.asarray(effects[index], dtype=np.float64)
        rows.append(
            {
                "parameter_id": parameter_id,
                "mu": np.mean(array, axis=0).tolist(),
                "mu_star": np.mean(np.abs(array), axis=0).tolist(),
                "sigma": np.std(array, axis=0, ddof=1).tolist() if trajectories >= 2 else [None] * values.shape[1],
            }
        )
    return {
        "schema_version": "morris-analysis@1.0.0",
        "design_hash": design["design_hash"],
        "output_dimension": int(values.shape[1]),
        "parameters": rows,
    }


def sobol_saltelli_design(
    intervals: Sequence[ParameterInterval],
    *,
    base_size: int,
    seed: int,
) -> dict[str, Any]:
    """Generate two scrambled Sobol bases and Saltelli A-B_i blocks."""

    intervals = tuple(intervals)
    dimension = len(intervals)
    if dimension < 1 or base_size < 4:
        raise InputValidationError("Sobol design requires parameters and base_size >= 4")
    try:
        from scipy.stats import qmc
    except ImportError as exc:  # pragma: no cover
        raise InputValidationError("SciPy is required for scrambled Sobol designs") from exc
    power = int(math.ceil(math.log2(base_size)))
    unit = qmc.Sobol(d=2 * dimension, scramble=True, seed=int(seed)).random_base2(power)[:base_size]
    a_unit, b_unit = unit[:, :dimension], unit[:, dimension:]
    blocks = [a_unit, b_unit]
    for index in range(dimension):
        mixed = a_unit.copy()
        mixed[:, index] = b_unit[:, index]
        blocks.append(mixed)
    stacked = np.vstack(blocks)
    declaration = {
        "schema_version": "saltelli-sobol-design@1.0.0",
        "method": "scrambled_sobol_two_base_jansen",
        "parameter_ids": [item.parameter_id for item in intervals],
        "base_size": int(base_size),
        "seed": int(seed),
        "block_order": ["A", "B", *[f"AB_{item.parameter_id}" for item in intervals]],
    }
    return {
        **declaration,
        "design_hash": content_hash(declaration),
        "unit_samples": stacked,
        "samples": _map_design(stacked, intervals),
    }


def analyze_sobol(design: Mapping[str, Any], outputs: Any) -> dict[str, Any]:
    values = np.asarray(outputs, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    parameter_ids = tuple(map(str, design["parameter_ids"]))
    n = int(design["base_size"])
    expected = n * (len(parameter_ids) + 2)
    if values.shape[0] != expected or not np.isfinite(values).all():
        raise InputValidationError("Sobol outputs must be finite and match the Saltelli blocks")
    a, b = values[:n], values[n : 2 * n]
    variance = np.var(np.vstack([a, b]), axis=0, ddof=1)
    rows = []
    for index, parameter_id in enumerate(parameter_ids):
        ab = values[(2 + index) * n : (3 + index) * n]
        first = np.divide(
            np.mean(b * (ab - a), axis=0),
            variance,
            out=np.zeros_like(variance),
            where=variance > 0.0,
        )
        total = np.divide(
            0.5 * np.mean((a - ab) ** 2, axis=0),
            variance,
            out=np.zeros_like(variance),
            where=variance > 0.0,
        )
        rows.append(
            {
                "parameter_id": parameter_id,
                "first_order": first.tolist(),
                "total_order": total.tolist(),
            }
        )
    return {
        "schema_version": "sobol-analysis@1.0.0",
        "design_hash": design["design_hash"],
        "output_variance": variance.tolist(),
        "parameters": rows,
        "zero_variance_outputs": np.flatnonzero(variance <= 0.0).astype(int).tolist(),
    }


def epistemic_corner_set(
    intervals: Sequence[ParameterInterval],
    *,
    active_parameter_ids: Sequence[str],
    maximum_corners: int = 4,
) -> dict[str, Any]:
    """Return a nominal point plus a small deterministic high/low envelope."""

    intervals = tuple(intervals)
    active = tuple(active_parameter_ids)
    lookup = {item.parameter_id: item for item in intervals}
    if any(name not in lookup for name in active) or not 1 <= maximum_corners <= 4:
        raise InputValidationError("corner-set declaration is invalid")
    nominal = {item.parameter_id: 0.5 * (item.lower + item.upper) for item in intervals}
    corners = []
    if active:
        for label, upper in (("all_low", False), ("all_high", True)):
            point = dict(nominal)
            for name in active:
                interval = lookup[name]
                point[name] = interval.upper if upper else interval.lower
            corners.append({"corner_id": label, "values": point})
        for index, name in enumerate(active[: max(0, maximum_corners - 2)]):
            point = dict(nominal)
            interval = lookup[name]
            point[name] = interval.upper
            corners.append({"corner_id": f"{name}_high", "values": point})
    document = {
        "schema_version": "epistemic-corner-set@1.0.0",
        "nominal": nominal,
        "active_parameter_ids": list(active),
        "corners": corners[:maximum_corners],
        "semantics": "scenario_envelope_not_probability_distribution",
    }
    return {**document, "corner_set_hash": content_hash(document)}


__all__ = [
    "ParameterInterval",
    "analyze_morris",
    "analyze_sobol",
    "epistemic_corner_set",
    "intervals_from_document",
    "morris_design",
    "sobol_saltelli_design",
]
