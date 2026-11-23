"""Versioned monotone candidates for offline damage-tree views.

This module is deliberately separate from :mod:`damage_engine.damage_tree`.
It does not change the simulator's default tree.  Candidate definitions consume
the per-replicate component-state trace and produce derived, replaceable views.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import DamageTreeError
from .utils import content_hash


CANDIDATE_SCHEMA_VERSION = "offline-monotone-tree-candidate@1.0.0"
IDENTITY_FALLBACK_STATUS = "provisional_no_domain_semantics"
_GATE_KINDS = {
    "identity",
    "weighted_noisy_or",
    "k_of_n",
    "two_additive_choquet",
}
_TOLERANCE = 1.0e-10


def _probability(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DamageTreeError(
            "offline tree probability must be numeric",
            context={"field": field, "value": repr(value)},
        )
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise DamageTreeError(
            "offline tree probability must be finite and lie in [0,1]",
            context={"field": field, "value": number},
        )
    return number


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DamageTreeError(
            "offline tree parameter must be numeric",
            context={"field": field, "value": repr(value)},
        )
    number = float(value)
    if not math.isfinite(number):
        raise DamageTreeError(
            "offline tree parameter must be finite",
            context={"field": field, "value": number},
        )
    return number


def weighted_noisy_or(values: Sequence[float], weights: Sequence[float]) -> float:
    """Return ``1 - product(1-w_i*p_i)`` for weights in ``[0,1]``.

    This is the preregistered fatality-weight parameterization: a zero weight
    removes an input and unit weights recover the ordinary independent OR.
    """

    if len(values) != len(weights) or len(values) == 0:
        raise DamageTreeError(
            "weighted noisy-OR requires equally sized nonempty values and weights"
        )
    probabilities = [
        _probability(value, field=f"values[{index}]")
        for index, value in enumerate(values)
    ]
    parsed_weights = [
        _finite(weight, field=f"weights[{index}]")
        for index, weight in enumerate(weights)
    ]
    if any(weight < 0.0 or weight > 1.0 for weight in parsed_weights):
        raise DamageTreeError("weighted noisy-OR weights must lie in [0,1]")
    if not any(weight > 0.0 for weight in parsed_weights):
        raise DamageTreeError("weighted noisy-OR requires at least one positive weight")
    log_survival = 0.0
    for probability, weight in zip(probabilities, parsed_weights, strict=True):
        weighted_probability = weight * probability
        if weighted_probability == 0.0:
            continue
        if weighted_probability == 1.0:
            return 1.0
        log_survival += math.log1p(-weighted_probability)
    return float(np.clip(-math.expm1(log_survival), 0.0, 1.0))


def probability_k_of_n(values: Sequence[float], k: int) -> float:
    """Poisson-binomial tail for independent component-state events."""

    probabilities = [
        _probability(value, field=f"values[{index}]")
        for index, value in enumerate(values)
    ]
    if not probabilities:
        raise DamageTreeError("k-of-n requires at least one input")
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise DamageTreeError("k-of-n threshold must be an integer")
    threshold = int(k)
    if not 1 <= threshold <= len(probabilities):
        raise DamageTreeError("k-of-n threshold must satisfy 1 <= k <= n")
    distribution = np.zeros(len(probabilities) + 1, dtype=np.float64)
    distribution[0] = 1.0
    for probability in probabilities:
        updated = distribution * (1.0 - probability)
        updated[1:] += distribution[:-1] * probability
        distribution = updated
    return float(np.clip(np.sum(distribution[threshold:]), 0.0, 1.0))


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise DamageTreeError("Choquet pair masses require two distinct inputs")
    return tuple(sorted((left, right)))


def validate_two_additive_capacity(
    input_ids: Sequence[str],
    singleton_masses: Mapping[str, float],
    pair_masses: Mapping[tuple[str, str], float],
) -> None:
    """Validate a normalized monotone 2-additive capacity.

    For a 2-additive Möbius representation, monotonicity is equivalent to
    ``m_i + sum_{j in S} m_ij >= 0`` for every ``i`` and subset ``S``.  The
    smallest marginal chooses every negative pair involving ``i``; this yields
    the compact check used below and permits both redundancy (negative pair
    mass) and complementarity (positive pair mass).
    """

    identifiers = tuple(input_ids)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise DamageTreeError("Choquet input identifiers must be nonempty and unique")
    if set(singleton_masses) != set(identifiers):
        raise DamageTreeError(
            "Choquet singleton masses must cover every input exactly once"
        )
    parsed_singletons = {
        key: _finite(value, field=f"singleton_masses.{key}")
        for key, value in singleton_masses.items()
    }
    parsed_pairs: dict[tuple[str, str], float] = {}
    for raw_pair, raw_mass in pair_masses.items():
        if len(raw_pair) != 2:
            raise DamageTreeError("Choquet pair-mass key must contain two inputs")
        pair = _canonical_pair(str(raw_pair[0]), str(raw_pair[1]))
        if pair[0] not in parsed_singletons or pair[1] not in parsed_singletons:
            raise DamageTreeError(
                "Choquet pair mass references an unknown input",
                context={"pair": list(pair)},
            )
        if pair in parsed_pairs:
            raise DamageTreeError(
                "Choquet pair masses must be unique",
                context={"pair": list(pair)},
            )
        parsed_pairs[pair] = _finite(raw_mass, field=f"pair_masses.{pair}")
    total = math.fsum(parsed_singletons.values()) + math.fsum(parsed_pairs.values())
    if not math.isclose(total, 1.0, abs_tol=_TOLERANCE, rel_tol=_TOLERANCE):
        raise DamageTreeError(
            "Choquet capacity must be normalized to one",
            context={"total_mass": total},
        )
    for identifier in identifiers:
        minimum_marginal = parsed_singletons[identifier] + math.fsum(
            mass
            for pair, mass in parsed_pairs.items()
            if identifier in pair and mass < 0.0
        )
        if minimum_marginal < -_TOLERANCE:
            raise DamageTreeError(
                "Choquet capacity is not monotone",
                context={
                    "input_id": identifier,
                    "minimum_marginal": minimum_marginal,
                },
            )


def two_additive_choquet(
    values: Mapping[str, float],
    singleton_masses: Mapping[str, float],
    pair_masses: Mapping[tuple[str, str], float],
) -> float:
    """Evaluate the discrete Choquet integral of a monotone 2-additive capacity."""

    identifiers = tuple(values)
    validate_two_additive_capacity(identifiers, singleton_masses, pair_masses)
    probabilities = {
        key: _probability(value, field=f"values.{key}")
        for key, value in values.items()
    }

    def capacity(subset: set[str]) -> float:
        result = math.fsum(singleton_masses[item] for item in subset)
        result += math.fsum(
            mass
            for pair, mass in pair_masses.items()
            if pair[0] in subset and pair[1] in subset
        )
        return float(result)

    ordered = sorted(identifiers, key=lambda item: (probabilities[item], item))
    previous = 0.0
    result = 0.0
    for index, identifier in enumerate(ordered):
        value = probabilities[identifier]
        active = set(ordered[index:])
        result += (value - previous) * capacity(active)
        previous = value
    if result < -_TOLERANCE or result > 1.0 + _TOLERANCE:
        raise DamageTreeError(
            "validated Choquet capacity produced an out-of-range value",
            context={"value": result},
        )
    return float(np.clip(result, 0.0, 1.0))


def _definition_without_hash(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in document.items()
        if key != "definition_hash"
    }


def candidate_definition_hash(document: Mapping[str, Any]) -> str:
    """Return the canonical identity of a candidate definition."""

    return content_hash(_definition_without_hash(document))


@dataclass(frozen=True)
class CompiledOfflineTreeCandidate:
    candidate_id: str
    definition_hash: str
    status: str
    authority_status: str
    inputs: tuple[dict[str, str], ...]
    outputs: tuple[dict[str, Any], ...]
    document: dict[str, Any]

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(str(output["output_id"]) for output in self.outputs)

    def evaluate(
        self,
        mechanism_states: Mapping[str, Mapping[str, Mapping[str, float]]],
    ) -> dict[str, Any]:
        values: dict[str, float] = {}
        for item in self.inputs:
            input_id = item["input_id"]
            try:
                raw = mechanism_states[item["mechanism"]][item["component_id"]][
                    item["damage_state"]
                ]
            except (KeyError, TypeError) as exc:
                raise DamageTreeError(
                    "offline tree input is unavailable in component-state trace",
                    context={"input_id": input_id, **item},
                ) from exc
            values[input_id] = _probability(raw, field=input_id)

        evaluated: dict[str, float] = {}
        for output in self.outputs:
            gate = output["gate"]
            gate_inputs = tuple(gate["inputs"])
            selected = [values[input_id] for input_id in gate_inputs]
            kind = gate["kind"]
            if kind == "identity":
                value = selected[0]
            elif kind == "weighted_noisy_or":
                value = weighted_noisy_or(selected, gate["weights"])
            elif kind == "k_of_n":
                value = probability_k_of_n(selected, gate["k"])
            elif kind == "two_additive_choquet":
                singleton_masses = {
                    key: float(gate["singleton_masses"][key])
                    for key in gate_inputs
                }
                pair_masses = {
                    _canonical_pair(item["left"], item["right"]): float(item["mass"])
                    for item in gate["pair_masses"]
                }
                value = two_additive_choquet(
                    dict(zip(gate_inputs, selected, strict=True)),
                    singleton_masses,
                    pair_masses,
                )
            else:  # pragma: no cover - compilation closes the branch.
                raise DamageTreeError("unsupported offline tree gate kind")
            evaluated[str(output["output_id"])] = _probability(
                value,
                field=f"outputs.{output['output_id']}",
            )
        return {
            "candidate_id": self.candidate_id,
            "definition_hash": self.definition_hash,
            "status": self.status,
            "authority_status": self.authority_status,
            "outputs": evaluated,
        }


def compile_offline_tree_candidate(
    document: Mapping[str, Any],
) -> CompiledOfflineTreeCandidate:
    """Validate and compile a one-layer monotone offline tree candidate."""

    if not isinstance(document, Mapping):
        raise DamageTreeError("offline tree candidate must be an object")
    if document.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise DamageTreeError("unsupported offline tree candidate schema version")
    candidate_id = document.get("candidate_id")
    status = document.get("status")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise DamageTreeError("offline tree candidate_id must be a nonempty string")
    if not isinstance(status, str) or not status:
        raise DamageTreeError("offline tree status must be a nonempty string")
    authority = document.get("semantic_authority")
    if not isinstance(authority, Mapping):
        raise DamageTreeError("offline tree candidate must declare semantic_authority")
    authority_status = authority.get("status")
    if authority_status not in {"approved", "unreviewed", "not_provided"}:
        raise DamageTreeError(
            "semantic_authority.status must be approved, unreviewed, or not_provided"
        )
    authority_scope = authority.get("scope")
    if not isinstance(authority_scope, str) or not authority_scope:
        raise DamageTreeError("semantic_authority.scope must be a nonempty string")
    if authority_status == "approved":
        authority_source = authority.get("source")
        if not isinstance(authority_source, (str, Mapping)) or not authority_source:
            raise DamageTreeError(
                "approved semantic authority requires a nonempty review source"
            )

    raw_inputs = document.get("inputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise DamageTreeError("offline tree candidate requires nonempty inputs")
    inputs: list[dict[str, str]] = []
    input_ids: set[str] = set()
    for index, raw in enumerate(raw_inputs):
        if not isinstance(raw, Mapping):
            raise DamageTreeError("offline tree inputs must be objects")
        parsed = {
            name: raw.get(name)
            for name in ("input_id", "component_id", "mechanism", "damage_state")
        }
        if any(not isinstance(value, str) or not value for value in parsed.values()):
            raise DamageTreeError(
                "offline tree input fields must be nonempty strings",
                context={"input_index": index},
            )
        if parsed["input_id"] in input_ids:
            raise DamageTreeError("offline tree input identifiers must be unique")
        input_ids.add(parsed["input_id"])
        inputs.append(parsed)  # type: ignore[arg-type]

    raw_outputs = document.get("outputs")
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise DamageTreeError("offline tree candidate requires nonempty outputs")
    outputs: list[dict[str, Any]] = []
    output_ids: set[str] = set()
    for index, raw in enumerate(raw_outputs):
        if not isinstance(raw, Mapping):
            raise DamageTreeError("offline tree outputs must be objects")
        output_id = raw.get("output_id")
        gate = raw.get("gate")
        if not isinstance(output_id, str) or not output_id or output_id in output_ids:
            raise DamageTreeError("offline tree output identifiers must be nonempty and unique")
        output_ids.add(output_id)
        if not isinstance(gate, Mapping):
            raise DamageTreeError("offline tree output gate must be an object")
        kind = gate.get("kind")
        gate_inputs = gate.get("inputs")
        if kind not in _GATE_KINDS:
            raise DamageTreeError(
                "unsupported offline tree gate kind",
                context={"output_index": index, "kind": kind},
            )
        if (
            not isinstance(gate_inputs, list)
            or not gate_inputs
            or len(gate_inputs) != len(set(gate_inputs))
            or any(item not in input_ids for item in gate_inputs)
        ):
            raise DamageTreeError(
                "offline tree gate inputs must be unique known input identifiers"
            )
        parsed_gate: dict[str, Any] = {
            "kind": kind,
            "inputs": [str(item) for item in gate_inputs],
        }
        if kind == "identity":
            if len(gate_inputs) != 1:
                raise DamageTreeError("identity gate requires exactly one input")
        elif kind == "weighted_noisy_or":
            weights = gate.get("weights")
            if not isinstance(weights, list):
                raise DamageTreeError("weighted noisy-OR gate requires weights")
            # Reuse the public evaluator for complete validation.
            weighted_noisy_or([0.0] * len(gate_inputs), weights)
            parsed_gate["weights"] = [float(value) for value in weights]
        elif kind == "k_of_n":
            threshold = gate.get("k")
            probability_k_of_n([0.0] * len(gate_inputs), threshold)
            parsed_gate["k"] = int(threshold)
        else:
            singleton_masses = gate.get("singleton_masses")
            raw_pairs = gate.get("pair_masses", [])
            if not isinstance(singleton_masses, Mapping) or not isinstance(raw_pairs, list):
                raise DamageTreeError(
                    "two-additive Choquet gate requires singleton_masses and pair_masses"
                )
            parsed_singletons = {
                str(key): _finite(value, field=f"singleton_masses.{key}")
                for key, value in singleton_masses.items()
            }
            parsed_pairs: list[dict[str, Any]] = []
            pair_mapping: dict[tuple[str, str], float] = {}
            for pair_index, raw_pair in enumerate(raw_pairs):
                if not isinstance(raw_pair, Mapping):
                    raise DamageTreeError("Choquet pair masses must be objects")
                left, right = raw_pair.get("left"), raw_pair.get("right")
                if not isinstance(left, str) or not isinstance(right, str):
                    raise DamageTreeError("Choquet pair inputs must be strings")
                pair = _canonical_pair(left, right)
                if pair in pair_mapping:
                    raise DamageTreeError("Choquet pair masses must be unique")
                mass = _finite(raw_pair.get("mass"), field=f"pair_masses[{pair_index}]")
                pair_mapping[pair] = mass
                parsed_pairs.append({"left": pair[0], "right": pair[1], "mass": mass})
            validate_two_additive_capacity(gate_inputs, parsed_singletons, pair_mapping)
            parsed_gate["singleton_masses"] = parsed_singletons
            parsed_gate["pair_masses"] = sorted(
                parsed_pairs,
                key=lambda item: (item["left"], item["right"]),
            )
        outputs.append({"output_id": output_id, "gate": parsed_gate})

    definition_hash = candidate_definition_hash(document)
    declared_hash = document.get("definition_hash")
    if declared_hash is not None and declared_hash != definition_hash:
        raise DamageTreeError(
            "offline tree candidate definition_hash does not match content",
            context={"declared": declared_hash, "computed": definition_hash},
        )
    canonical_document = {
        **_definition_without_hash(document),
        "definition_hash": definition_hash,
    }
    return CompiledOfflineTreeCandidate(
        candidate_id=candidate_id,
        definition_hash=definition_hash,
        status=status,
        authority_status=str(authority_status),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        document=canonical_document,
    )


def build_identity_component_fallback(
    component_ids: Sequence[str],
    *,
    mechanisms: Sequence[str] = ("fragment",),
    damage_states: Sequence[str] = ("degraded", "failed"),
    candidate_id: str = "identity-component-fallback-v1",
) -> dict[str, Any]:
    """Build a provisional pass-through view without assigning system semantics."""

    components = tuple(sorted(set(component_ids)))
    if not components or any(not isinstance(item, str) or not item for item in components):
        raise DamageTreeError("identity fallback requires nonempty component identifiers")
    if not mechanisms or not damage_states:
        raise DamageTreeError("identity fallback requires mechanisms and damage states")
    inputs: list[dict[str, str]] = []
    outputs: list[dict[str, Any]] = []
    for mechanism in mechanisms:
        for component_id in components:
            for state in damage_states:
                input_id = f"{mechanism}::{component_id}::{state}"
                inputs.append(
                    {
                        "input_id": input_id,
                        "component_id": component_id,
                        "mechanism": str(mechanism),
                        "damage_state": str(state),
                    }
                )
                outputs.append(
                    {
                        "output_id": f"identity::{input_id}",
                        "gate": {"kind": "identity", "inputs": [input_id]},
                    }
                )
    document: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "status": IDENTITY_FALLBACK_STATUS,
        "semantic_authority": {
            "status": "not_provided",
            "source": None,
            "scope": (
                "component-state pass-through only; no functional grouping, priority, "
                "or system-effect semantics are asserted"
            ),
        },
        "inputs": inputs,
        "outputs": outputs,
    }
    document["definition_hash"] = candidate_definition_hash(document)
    return document

