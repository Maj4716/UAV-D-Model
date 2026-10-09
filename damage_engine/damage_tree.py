from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .errors import DamageTreeError


_GATE_TYPES = {"and", "or", "k_of_n", "functional"}
_SUPPORTED_DEPENDENCY_MODELS = {"independent", "fully_dependent"}


def _probabilities(values: list[float]) -> list[float]:
    """Return validated probabilities without silently clipping bad inputs."""

    output: list[float] = []
    for index, value in enumerate(values):
        try:
            probability = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DamageTreeError(
                "gate input probability must be numeric",
                context={"input_index": index, "value": repr(value)},
            ) from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise DamageTreeError(
                "gate input probability must be finite and lie in [0,1]",
                context={"input_index": index, "value": probability},
            )
        output.append(probability)
    return output


def _validate_k(k: int, input_count: int) -> int:
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise DamageTreeError(
            "k-of-n threshold must be an integer",
            context={"k": k, "input_count": input_count},
        )
    threshold = int(k)
    if not 1 <= threshold <= input_count:
        raise DamageTreeError(
            "k-of-n threshold must satisfy 1 <= k <= number of inputs",
            context={"k": threshold, "input_count": input_count},
        )
    return threshold


def probability_or(values: list[float]) -> float:
    """Probability of a union under an explicit independent-input assumption."""

    survival = 1.0
    for value in _probabilities(values):
        survival *= 1.0 - value
    return float(np.clip(1.0 - survival, 0.0, 1.0))


def probability_and(values: list[float]) -> float:
    """Probability of an intersection under an explicit independent-input assumption."""

    result = 1.0
    for value in _probabilities(values):
        result *= value
    return float(np.clip(result, 0.0, 1.0))


def probability_k_of_n(values: list[float], k: int) -> float:
    """Poisson-binomial tail for independent Bernoulli input events."""

    probabilities = _probabilities(values)
    threshold = _validate_k(k, len(probabilities))
    distribution = np.zeros(len(probabilities) + 1, dtype=np.float64)
    distribution[0] = 1.0
    for probability in probabilities:
        next_distribution = distribution * (1.0 - probability)
        next_distribution[1:] += distribution[:-1] * probability
        distribution = next_distribution
    return float(np.clip(np.sum(distribution[threshold:]), 0.0, 1.0))


def _fully_dependent_probability(node_type: str, values: list[float], k: int | None) -> float:
    """Propagate comonotonic Bernoulli events from their marginal probabilities.

    ``fully_dependent`` is a declared perfect-positive-dependence model, not a
    fitted correlation coefficient. For a common latent uniform variate, an OR
    is the maximum marginal, an AND is the minimum, and a k-of-n event is the
    k-th largest marginal.
    """

    probabilities = _probabilities(values)
    if not probabilities:
        raise DamageTreeError("fully-dependent gate requires at least one input")
    if node_type == "and":
        return min(probabilities)
    if node_type in {"or", "functional"} and k is None:
        return max(probabilities)
    threshold = _validate_k(k, len(probabilities))
    return sorted(probabilities, reverse=True)[threshold - 1]


def _event_descriptions(events: set[tuple[str, str, str]]) -> list[dict[str, str]]:
    return [
        {"component_id": component, "mechanism": mechanism, "damage_state": state}
        for component, mechanism, state in sorted(events)
    ]


def _leaf_events_overlap(
    first: set[tuple[str, str, str]],
    second: set[tuple[str, str, str]],
) -> set[tuple[str, str, str]]:
    """Find structural conflicts with an independent-input declaration.

    Damage states for one component/mechanism are ordinal and therefore nested.
    A ``combined`` state also contains the corresponding fragment/blast event.
    Separate fragment and blast leaves are not rejected here because their
    dependence is governed by the selected mechanism-coupling model.
    """

    conflicts: set[tuple[str, str, str]] = set()
    for left in first:
        for right in second:
            if left[0] != right[0]:
                continue
            mechanisms_overlap = left[1] == right[1] or "combined" in {left[1], right[1]}
            if mechanisms_overlap:
                conflicts.update((left, right))
    return conflicts


@dataclass(frozen=True)
class CompiledDamageTree:
    version: str
    nodes: dict[str, dict]
    order: tuple[str, ...]
    outputs: tuple[str, ...]
    standards: dict[str, tuple[str, ...]]

    @property
    def dependency_models(self) -> dict[str, str]:
        return {
            node_id: str(node["dependency_model_id"])
            for node_id, node in self.nodes.items()
            if node["node_type"] in _GATE_TYPES and len(node["inputs"]) > 1
        }

    def evaluate(
        self,
        mechanism_states: dict[str, dict[str, dict[str, float]]],
    ) -> dict:
        values: dict[str, float] = {}
        for node_id in self.order:
            node = self.nodes[node_id]
            node_type = node["node_type"]
            if node_type == "leaf":
                component_id = node["component_id"]
                mechanism = node["mechanism"]
                damage_state = node["damage_state"]
                try:
                    raw_value = mechanism_states[mechanism][component_id][damage_state]
                except KeyError as exc:
                    raise DamageTreeError(
                        "damage-tree leaf references unavailable component state",
                        context={
                            "node_id": node_id,
                            "component_id": component_id,
                            "mechanism": mechanism,
                            "damage_state": damage_state,
                        },
                    ) from exc
                try:
                    values[node_id] = _probabilities([raw_value])[0]
                except DamageTreeError as exc:
                    raise DamageTreeError(
                        "damage-tree leaf probability is invalid",
                        context={
                            "node_id": node_id,
                            "component_id": component_id,
                            "mechanism": mechanism,
                            "damage_state": damage_state,
                            "value": raw_value,
                        },
                    ) from exc
                continue

            inputs = [values[item] for item in node["inputs"]]
            dependency = node.get("dependency_model_id")
            if node_type == "not":
                value = 1.0 - inputs[0]
            elif node_type == "standard_projection":
                value = inputs[0]
            elif dependency == "fully_dependent":
                value = _fully_dependent_probability(node_type, inputs, node.get("k"))
            elif dependency == "independent" or len(inputs) == 1:
                if node_type == "and":
                    value = probability_and(inputs)
                elif node_type == "or":
                    value = probability_or(inputs)
                elif node_type in {"k_of_n", "functional"}:
                    value = probability_k_of_n(inputs, node["k"])
                else:  # pragma: no cover - compile_damage_tree closes this path.
                    raise DamageTreeError(
                        "unsupported damage-tree node type",
                        context={"node_id": node_id, "node_type": node_type},
                    )
            else:  # pragma: no cover - compile_damage_tree closes this path.
                raise DamageTreeError(
                    "unsupported damage-tree dependency model",
                    context={"node_id": node_id, "dependency_model_id": dependency},
                )
            values[node_id] = _probabilities([value])[0]
        return {
            "node_probabilities": values,
            "outputs": {node_id: values[node_id] for node_id in self.outputs},
            "dependency_assumption": "explicit_per_gate",
            "dependency_models": self.dependency_models,
        }


def compile_damage_tree(target: dict) -> CompiledDamageTree:
    try:
        tree = target["damage_tree"]
        raw_nodes = tree["nodes"]
        raw_outputs = tree["outputs"]
        raw_standards = tree.get("standards")
        components = target["components"]
    except (KeyError, TypeError) as exc:
        raise DamageTreeError(
            "target is missing required damage-tree structure",
            context={"missing_or_invalid": str(exc)},
        ) from exc
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise DamageTreeError("damage-tree nodes must be a non-empty array")
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise DamageTreeError("damage-tree outputs must be a non-empty array")
    version = tree.get("version")
    if not isinstance(version, str) or not version:
        raise DamageTreeError("damage-tree version must be a non-empty string")

    component_states: dict[str, set[str]] = {}
    for component in components:
        try:
            component_id = component["id"]
            damage_states = component["damage_states"]
        except (KeyError, TypeError) as exc:
            raise DamageTreeError(
                "component is missing id or damage_states required by the damage tree"
            ) from exc
        if not isinstance(component_id, str) or not component_id:
            raise DamageTreeError("component IDs referenced by the damage tree must be non-empty strings")
        if component_id in component_states:
            raise DamageTreeError(
                "component IDs must be unique before compiling the damage tree",
                context={"duplicate_component_id": component_id},
            )
        component_states[component_id] = set(damage_states)

    nodes: dict[str, dict] = {}
    leaf_owners: dict[tuple[str, str, str], str] = {}
    allowed_types = {"leaf", "and", "or", "not", "k_of_n", "functional", "standard_projection"}
    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, dict):
            raise DamageTreeError(
                "damage-tree nodes must be objects",
                context={"node_index": index},
            )
        node = deepcopy(raw_node)
        node_id = node.get("id")
        node_type = node.get("node_type")
        inputs = node.get("inputs")
        if not isinstance(node_id, str) or not node_id:
            raise DamageTreeError(
                "damage-tree node id must be a non-empty string",
                context={"node_index": index},
            )
        if node_id in nodes:
            raise DamageTreeError(
                "damage-tree node IDs must be unique",
                context={"duplicate_id": node_id},
            )
        if node_type not in allowed_types:
            raise DamageTreeError(
                "unsupported damage-tree node type",
                context={"node_id": node_id, "node_type": node_type},
            )
        if not isinstance(inputs, list) or any(not isinstance(item, str) or not item for item in inputs):
            raise DamageTreeError(
                "damage-tree inputs must be an array of non-empty node IDs",
                context={"node_id": node_id},
            )
        if len(inputs) != len(set(inputs)):
            raise DamageTreeError(
                "damage-tree node inputs must not contain duplicate references",
                context={"node_id": node_id, "inputs": inputs},
            )

        dependency = node.get("dependency_model_id")
        k = node.get("k")
        if node_type == "leaf":
            if inputs:
                raise DamageTreeError("leaf nodes cannot contain inputs", context={"node_id": node_id})
            component_id = node.get("component_id")
            mechanism = node.get("mechanism")
            state = node.get("damage_state")
            if component_id not in component_states:
                raise DamageTreeError(
                    "leaf references unknown component",
                    context={"node_id": node_id, "component_id": component_id},
                )
            if state not in component_states[component_id]:
                raise DamageTreeError(
                    "leaf references unknown component damage state",
                    context={"node_id": node_id, "component_id": component_id, "damage_state": state},
                )
            if mechanism not in {"fragment", "blast", "combined"}:
                raise DamageTreeError(
                    "leaf mechanism must be fragment, blast, or combined",
                    context={"node_id": node_id, "mechanism": mechanism},
                )
            if k is not None or dependency is not None:
                raise DamageTreeError(
                    "leaf nodes cannot declare k or a dependency model",
                    context={"node_id": node_id},
                )
            event = (component_id, mechanism, state)
            if event in leaf_owners:
                raise DamageTreeError(
                    "duplicate damage-tree leaf event; reuse the existing leaf node",
                    context={
                        "node_id": node_id,
                        "existing_node_id": leaf_owners[event],
                        "event": _event_descriptions({event})[0],
                    },
                )
            leaf_owners[event] = node_id
        elif node_type in {"and", "or"}:
            if len(inputs) < 2:
                raise DamageTreeError(
                    "AND/OR nodes require at least two inputs",
                    context={"node_id": node_id, "node_type": node_type},
                )
            if k is not None:
                raise DamageTreeError(
                    "only k_of_n or functional nodes may declare k",
                    context={"node_id": node_id, "node_type": node_type},
                )
        elif node_type == "not":
            if len(inputs) != 1:
                raise DamageTreeError(
                    "NOT node requires exactly one input",
                    context={"node_id": node_id, "input_count": len(inputs)},
                )
            if k is not None or dependency is not None:
                raise DamageTreeError(
                    "NOT nodes cannot declare k or a dependency model",
                    context={"node_id": node_id},
                )
        elif node_type in {"k_of_n", "functional"}:
            if len(inputs) < 2:
                raise DamageTreeError(
                    "k_of_n/functional nodes require at least two inputs",
                    context={"node_id": node_id, "node_type": node_type},
                )
            _validate_k(k, len(inputs))
        elif node_type == "standard_projection":
            if len(inputs) != 1:
                raise DamageTreeError(
                    "standard_projection requires exactly one input",
                    context={"node_id": node_id, "input_count": len(inputs)},
                )
            if k is not None:
                raise DamageTreeError(
                    "standard_projection cannot declare k",
                    context={"node_id": node_id},
                )
            if dependency not in (None, "independent"):
                raise DamageTreeError(
                    "standard_projection does not combine events and cannot declare dependence",
                    context={"node_id": node_id, "dependency_model_id": dependency},
                )

        if node_type != "leaf":
            ignored_leaf_fields = {
                field: node.get(field)
                for field in ("component_id", "damage_state")
                if node.get(field) is not None
            }
            if node.get("mechanism") not in (None, "not_applicable"):
                ignored_leaf_fields["mechanism"] = node.get("mechanism")
            if ignored_leaf_fields:
                raise DamageTreeError(
                    "non-leaf nodes cannot declare leaf event fields",
                    context={"node_id": node_id, "fields": ignored_leaf_fields},
                )

        if node_type in _GATE_TYPES and dependency not in _SUPPORTED_DEPENDENCY_MODELS:
            raise DamageTreeError(
                "multi-input probability gates require an explicit supported dependency model",
                context={
                    "node_id": node_id,
                    "dependency_model_id": dependency,
                    "supported": sorted(_SUPPORTED_DEPENDENCY_MODELS),
                },
            )
        nodes[node_id] = node

    for node_id, node in nodes.items():
        for dependency in node["inputs"]:
            if dependency not in nodes:
                raise DamageTreeError(
                    "damage-tree input references an unknown node",
                    context={"node_id": node_id, "input": dependency},
                )

    indegree = {node_id: 0 for node_id in nodes}
    children = {node_id: [] for node_id in nodes}
    for node_id, node in nodes.items():
        for dependency in node["inputs"]:
            indegree[node_id] += 1
            children[dependency].append(node_id)
    queue = sorted(node_id for node_id, value in indegree.items() if value == 0)
    order: list[str] = []
    while queue:
        current = queue.pop(0)
        order.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
                queue.sort()
    if len(order) != len(nodes):
        raise DamageTreeError(
            "damage tree contains a cycle",
            context={"cycle_nodes": sorted(node_id for node_id, degree in indegree.items() if degree > 0)},
        )

    if any(not isinstance(value, str) or not value for value in raw_outputs):
        raise DamageTreeError("damage-tree outputs must contain non-empty node IDs")
    outputs = tuple(raw_outputs)
    if len(outputs) != len(set(outputs)):
        raise DamageTreeError(
            "damage-tree outputs must be unique",
            context={"outputs": list(outputs)},
        )
    unknown_outputs = sorted(set(outputs) - set(nodes))
    if unknown_outputs:
        raise DamageTreeError(
            "damage-tree output references unknown nodes",
            context={"unknown_outputs": unknown_outputs},
        )

    standards: dict[str, tuple[str, ...]] = {}
    if raw_standards is None:
        standards[version] = outputs
    else:
        if not isinstance(raw_standards, list) or not raw_standards:
            raise DamageTreeError("damage-tree standards must be a non-empty array")
        for index, raw_standard in enumerate(raw_standards):
            if not isinstance(raw_standard, dict):
                raise DamageTreeError(
                    "damage-tree standards must be objects",
                    context={"standard_index": index},
                )
            standard_id = raw_standard.get("id")
            standard_outputs = raw_standard.get("outputs")
            if not isinstance(standard_id, str) or not standard_id:
                raise DamageTreeError(
                    "evaluation standard id must be a non-empty string",
                    context={"standard_index": index},
                )
            if standard_id in standards:
                raise DamageTreeError(
                    "evaluation standard ids must be unique",
                    context={"duplicate_standard_id": standard_id},
                )
            if (
                not isinstance(standard_outputs, list)
                or not standard_outputs
                or any(not isinstance(item, str) or not item for item in standard_outputs)
            ):
                raise DamageTreeError(
                    "evaluation standard outputs must be a non-empty array of node ids",
                    context={"standard_id": standard_id},
                )
            if len(standard_outputs) != len(set(standard_outputs)):
                raise DamageTreeError(
                    "evaluation standard outputs must be unique",
                    context={"standard_id": standard_id, "outputs": standard_outputs},
                )
            unavailable = sorted(set(standard_outputs) - set(outputs))
            if unavailable:
                raise DamageTreeError(
                    "evaluation standards may reference only declared damage-tree outputs",
                    context={
                        "standard_id": standard_id,
                        "unavailable_outputs": unavailable,
                    },
                )
            standards[standard_id] = tuple(standard_outputs)

    reachable: set[str] = set()
    pending = list(outputs)
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(nodes[current]["inputs"])
    orphan_nodes = sorted(set(nodes) - reachable)
    if orphan_nodes:
        raise DamageTreeError(
            "damage tree contains nodes that do not contribute to any output",
            context={"orphan_nodes": orphan_nodes},
        )

    leaf_events: dict[str, set[tuple[str, str, str]]] = {}
    for node_id in order:
        node = nodes[node_id]
        if node["node_type"] == "leaf":
            leaf_events[node_id] = {
                (node["component_id"], node["mechanism"], node["damage_state"])
            }
        else:
            leaf_events[node_id] = set().union(*(leaf_events[item] for item in node["inputs"]))
        if node["node_type"] in _GATE_TYPES and node["dependency_model_id"] == "independent":
            for left_index, left_id in enumerate(node["inputs"]):
                for right_id in node["inputs"][left_index + 1 :]:
                    conflicts = _leaf_events_overlap(leaf_events[left_id], leaf_events[right_id])
                    if conflicts:
                        raise DamageTreeError(
                            "independent gate inputs contain overlapping component events",
                            context={
                                "node_id": node_id,
                                "left_input": left_id,
                                "right_input": right_id,
                                "conflicting_events": _event_descriptions(conflicts),
                                "remedy": "reuse a shared node or declare a validated dependency model",
                            },
                        )

    return CompiledDamageTree(
        version=version,
        nodes=nodes,
        order=tuple(order),
        outputs=outputs,
        standards=standards,
    )
