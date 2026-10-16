"""Versioned learning-label semantics for damage-engine outputs.

The damage tree defines physical/system events.  A :class:`LabelSpec` defines
how those events may be consumed by a learner.  Keeping the two contracts
separate prevents multi-label event probabilities from being silently treated
as a mutually-exclusive class distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .content_contract import validate_json_schema
from .errors import InputValidationError
from .utils import content_hash


LABEL_SPEC_SCHEMA_VERSION = "1.0.0"
ALLOWED_TASK_TYPES = frozenset(
    {
        "event_probability",
        "multi_label_event_probability",
        "ordinal_cumulative",
        "mutually_exclusive_class",
        "continuous_response",
        "auxiliary_physics",
    }
)


@dataclass(frozen=True)
class LabelTask:
    """One named learning task from a label specification."""

    task_id: str
    task_type: str
    outputs: tuple[str, ...]
    loss: str

    @property
    def is_probability_task(self) -> bool:
        return self.task_type in {
            "event_probability",
            "multi_label_event_probability",
            "ordinal_cumulative",
            "mutually_exclusive_class",
        }


class LabelSpec:
    """Validated, canonical learning-task definition."""

    def __init__(self, data: Mapping[str, Any]):
        document = json.loads(json.dumps(data, ensure_ascii=False, allow_nan=False))
        validate_json_schema(
            document,
            "label_spec.schema.json",
            failure_message="label specification failed schema validation",
            max_errors=20,
            compatibility="legacy_messages",
        )
        tasks: dict[str, LabelTask] = {}
        for task_id, value in document["tasks"].items():
            outputs = tuple(value["outputs"])
            if len(outputs) != len(set(outputs)):
                raise InputValidationError(
                    "label task outputs must be unique",
                    context={"task_id": task_id},
                )
            if value["type"] not in ALLOWED_TASK_TYPES:
                raise InputValidationError(
                    "unsupported label task type",
                    context={"task_id": task_id, "type": value["type"]},
                )
            if value["type"] == "event_probability" and len(outputs) != 1:
                raise InputValidationError(
                    "event_probability tasks require exactly one output",
                    context={"task_id": task_id},
                )
            if value["type"] == "multi_label_event_probability" and "softmax" in value["loss"].lower():
                raise InputValidationError(
                    "multi-label event probabilities cannot use a softmax loss",
                    context={"task_id": task_id, "loss": value["loss"]},
                )
            tasks[task_id] = LabelTask(
                task_id=task_id,
                task_type=value["type"],
                outputs=outputs,
                loss=value["loss"],
            )
        self._canonical_json = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self._tasks = tuple(tasks.values())
        self._hash = content_hash(document)

    @property
    def data(self) -> dict:
        """Return an independent document; changing it defines no new spec."""
        return json.loads(self._canonical_json)

    @property
    def tasks(self) -> dict[str, LabelTask]:
        """Return an independent mapping of immutable task records."""
        return {task.task_id: task for task in self._tasks}

    @property
    def hash(self) -> str:
        return self._hash

    @classmethod
    def from_file(cls, path: str | Path) -> "LabelSpec":
        value = Path(path)
        try:
            document = json.loads(value.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InputValidationError(
                "unable to load label specification as UTF-8 JSON",
                context={"path": str(value)},
            ) from exc
        if not isinstance(document, dict):
            raise InputValidationError("label specification must be a JSON object")
        return cls(document)

    @property
    def spec_id(self) -> str:
        return str(self.data["label_spec_id"])

    def task(self, task_id: str) -> LabelTask:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise InputValidationError(
                "unknown label task",
                context={"task_id": task_id},
            ) from exc

    def validate_output_vocabulary(self, output_names: list[str] | tuple[str, ...]) -> None:
        if any(not isinstance(name, str) or not name for name in output_names) or len(output_names) != len(set(output_names)):
            raise InputValidationError("output vocabulary must contain unique nonempty strings")
        available = set(output_names)
        missing = {
            output
            for task in self.tasks.values()
            for output in task.outputs
            if output not in available
        }
        if missing:
            raise InputValidationError(
                "label specification references outputs absent from the damage tree",
                context={"missing_outputs": sorted(missing)},
            )

    def validate_system_outputs(
        self, output_names: list[str] | tuple[str, ...], values,
        mask=None, *, atol: float = 1.0e-6,
    ) -> dict[str, np.ndarray]:
        """Validate stored values against declared tasks without repairing them.

        Tasks use the order in their ``outputs`` declaration. Coupled ordinal
        and exclusive tasks require all outputs for a usable task on each row;
        partial tasks are marked unavailable without fabricating values. A
        mutually-exclusive task describes a complete distribution: background
        classes must be explicit outputs. This performs no event computation.
        Independent event/response tasks can use the available subset; their
        per-output masks remain authoritative.
        """
        self.validate_output_vocabulary(output_names)
        if not np.isfinite(atol) or atol < 0:
            raise InputValidationError("label validation tolerance must be finite and nonnegative")
        data = np.asarray(values)
        if data.ndim < 1 or data.shape[-1] != len(output_names) or data.dtype.kind not in "fiu":
            raise InputValidationError("system values must be numeric with the declared output axis")
        selected = np.ones(data.shape, dtype=bool) if mask is None else np.asarray(mask)
        if selected.dtype != np.dtype(bool) or selected.shape != data.shape:
            raise InputValidationError("system label mask must be bool with the label shape")
        positions = {name: index for index, name in enumerate(output_names)}
        task_masks = {}
        for task in self._tasks:
            indices = [positions[name] for name in task.outputs]
            task_values = np.asarray(data[..., indices], dtype=np.float64)
            task_selected = selected[..., indices]
            if not np.isfinite(task_values[task_selected]).all():
                raise InputValidationError("selected label values must be finite", context={"task_id": task.task_id})
            complete = np.all(task_selected, axis=-1)
            usable = complete if task.task_type in {"ordinal_cumulative", "mutually_exclusive_class"} else np.any(task_selected, axis=-1)
            task_masks[task.task_id] = np.asarray(usable, dtype=bool).copy()
            if task.is_probability_task and np.any((task_values[task_selected] < -atol) | (task_values[task_selected] > 1 + atol)):
                raise InputValidationError("probability-task values must lie in [0,1]", context={"task_id": task.task_id})
            if task.task_type in {"ordinal_cumulative", "mutually_exclusive_class"}:
                rows = task_values.reshape((-1, len(indices)))[complete.reshape(-1)]
                if task.task_type == "ordinal_cumulative" and np.any(np.diff(rows, axis=-1) > atol):
                    raise InputValidationError("ordinal cumulative probabilities must be monotonically non-increasing", context={"task_id": task.task_id})
                if task.task_type == "mutually_exclusive_class" and not np.allclose(rows.sum(axis=-1), 1.0, rtol=0.0, atol=atol):
                    raise InputValidationError("mutually exclusive class probabilities must sum to one; background must be explicit", context={"task_id": task.task_id})
        return task_masks


def system_output_availability(
    nodes: Mapping[str, Mapping[str, Any]], outputs: list[str] | tuple[str, ...],
    enabled_mechanisms: list[str] | tuple[str, ...] | set[str], *,
    component_availability: Mapping[tuple[str, str, str], bool] | None = None,
) -> dict[str, bool]:
    """Propagate source availability conservatively through a dependency DAG.

    Every dependency must be available. Disabled inputs are missing, never
    implicitly known zero; ``combined`` requires both fragment and blast.
    This helper returns flags only and never evaluates a gate probability.
    """
    enabled = set(enabled_mechanisms)
    if not enabled.issubset({"fragment", "blast"}):
        raise InputValidationError("availability requires explicit fragment/blast source names")
    cache: dict[str, bool] = {}
    active: set[str] = set()

    def available(identifier: str) -> bool:
        if identifier in cache:
            return cache[identifier]
        if identifier in active or identifier not in nodes:
            raise InputValidationError("availability graph has a cycle or missing dependency", context={"node_id": identifier})
        active.add(identifier)
        node = nodes[identifier]
        if node.get("node_type") == "leaf":
            mechanism = node.get("mechanism")
            if mechanism not in {"fragment", "blast", "combined"}:
                raise InputValidationError("leaf availability has an unsupported source", context={"node_id": identifier})
            source_ready = {"fragment", "blast"}.issubset(enabled) if mechanism == "combined" else mechanism in enabled
            key = (node.get("component_id"), mechanism, node.get("damage_state"))
            if any(not isinstance(value, str) or not value for value in key):
                raise InputValidationError("leaf availability is missing its component/state mapping")
            if component_availability is not None:
                flag = component_availability.get(key, False)
                if not isinstance(flag, (bool, np.bool_)):
                    raise InputValidationError("component availability must be boolean")
                source_ready = source_ready and bool(flag)
            result = source_ready
        else:
            if node.get("node_type") not in {"and", "or", "k_of_n", "functional", "standard_projection", "not"}:
                raise InputValidationError("availability graph has an unsupported node type")
            dependencies = node.get("inputs")
            if not isinstance(dependencies, (list, tuple)) or not dependencies or any(not isinstance(value, str) for value in dependencies):
                raise InputValidationError("availability graph node requires named dependencies")
            # Evaluate every dependency to catch missing mappings and cycles,
            # even when the first unavailable source would short-circuit AND.
            result = all([available(value) for value in dependencies])
        active.remove(identifier)
        cache[identifier] = bool(result)
        return cache[identifier]

    if len(outputs) != len(set(outputs)) or any(not isinstance(value, str) or not value for value in outputs):
        raise InputValidationError("availability outputs must be unique nonempty names")
    return {identifier: available(identifier) for identifier in outputs}


def ordinal_class_probabilities(
    cumulative: np.ndarray | list[float], *, atol: float = 1.0e-12
) -> np.ndarray:
    """Convert ``P(D >= L_k)`` values to exclusive class probabilities.

    The function rejects non-monotone or out-of-range cumulative values.  It
    deliberately does not clip or renormalize invalid teacher outputs.
    """

    if not np.isfinite(atol) or atol < 0.0:
        raise InputValidationError("ordinal tolerance must be finite and nonnegative")
    values = np.asarray(cumulative, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise InputValidationError("ordinal cumulative values require a nonempty final axis")
    if not np.all(np.isfinite(values)):
        raise InputValidationError("ordinal cumulative values must be finite")
    if np.any(values < -atol) or np.any(values > 1.0 + atol):
        raise InputValidationError("ordinal cumulative values must lie in [0, 1]")
    if np.any(np.diff(values, axis=-1) > atol):
        raise InputValidationError(
            "ordinal cumulative probabilities must be monotonically non-increasing"
        )
    result = np.empty((*values.shape[:-1], values.shape[-1] + 1), dtype=np.float64)
    result[..., 0] = 1.0 - values[..., 0]
    result[..., 1:-1] = values[..., :-1] - values[..., 1:]
    result[..., -1] = values[..., -1]
    # Values within the declared numerical tolerance may produce tiny signed
    # roundoff; zeroing those is not a semantic repair or renormalization.
    result[np.abs(result) <= atol] = 0.0
    if np.any(result < 0.0):
        raise InputValidationError("ordinal conversion produced a negative class probability")
    return result


def threshold_ambiguity_mask(
    ci95: np.ndarray | list,
    thresholds: np.ndarray | list[float],
) -> np.ndarray:
    """Return which confidence intervals cross each declared threshold.

    ``ci95`` has shape ``[..., 2]`` and the result has shape
    ``[..., len(thresholds)]``.  Inclusive comparisons conservatively mark an
    interval touching a boundary as ambiguous.
    """

    intervals = np.asarray(ci95, dtype=np.float64)
    boundaries = np.asarray(thresholds, dtype=np.float64)
    if intervals.ndim < 1 or intervals.shape[-1] != 2:
        raise InputValidationError("ci95 must have a final [lower, upper] axis")
    if boundaries.ndim != 1 or not np.all(np.isfinite(boundaries)):
        raise InputValidationError("thresholds must be a finite one-dimensional array")
    if boundaries.size > 1 and np.any(np.diff(boundaries) <= 0.0):
        raise InputValidationError("thresholds must be strictly increasing")
    if not np.all(np.isfinite(intervals)) or np.any(intervals[..., 0] > intervals[..., 1]):
        raise InputValidationError("ci95 intervals must be finite and ordered")
    lower = intervals[..., 0, None]
    upper = intervals[..., 1, None]
    return (lower <= boundaries) & (boundaries <= upper)
