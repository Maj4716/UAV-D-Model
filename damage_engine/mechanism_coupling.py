from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import InputValidationError


@dataclass(frozen=True)
class MechanismCombination:
    point_estimate: dict[str, dict[str, float]]
    frechet_bounds: dict[str, dict[str, dict[str, float]]]
    scenarios: dict[str, dict[str, dict[str, float]]]
    identification_status: str = "not_identified"
    point_estimate_assumption: str = "conditional_independence"


def combine_mechanism_states(
    component_states: dict[str, tuple[str, ...]],
    fragment: dict[str, dict[str, float]],
    blast: dict[str, dict[str, float]],
    *,
    shared_vulnerable_ratio: dict[str, float] | None = None,
) -> MechanismCombination:
    """Combine fragment/blast marginals without hiding dependence uncertainty.

    The conditional-independence result is retained as one sensitivity scenario,
    not treated as an identified joint probability.  Fréchet lower and upper
    union limits are returned through the same interface so callers can evaluate
    their complete downstream damage tree under all three assumptions.
    """

    scenarios = {
        "frechet_lower": {},
        "conditional_independence": {},
        "frechet_upper": {},
    }
    bounds: dict[str, dict[str, dict[str, float]]] = {}
    for identifier, ordered_states in component_states.items():
        if identifier not in fragment or identifier not in blast:
            raise InputValidationError(
                "mechanism state maps must cover every target component",
                context={"component_id": identifier},
            )
        bounds[identifier] = {}
        for scenario in scenarios.values():
            scenario[identifier] = {}
        for state in ordered_states:
            try:
                fragment_probability = float(fragment[identifier][state])
                blast_probability = float(blast[identifier][state])
            except (KeyError, TypeError, ValueError) as exc:
                raise InputValidationError(
                    "mechanism state maps are incomplete or non-numeric",
                    context={"component_id": identifier, "damage_state": state},
                ) from exc
            if not (
                np.isfinite(fragment_probability)
                and np.isfinite(blast_probability)
                and 0.0 <= fragment_probability <= 1.0
                and 0.0 <= blast_probability <= 1.0
            ):
                raise InputValidationError(
                    "mechanism state probabilities must lie in [0,1]",
                    context={
                        "component_id": identifier,
                        "damage_state": state,
                        "fragment_probability": fragment_probability,
                        "blast_probability": blast_probability,
                    },
                )
            if shared_vulnerable_ratio is None:
                lower = max(fragment_probability, blast_probability)
                independent = 1.0 - (
                    (1.0 - fragment_probability)
                    * (1.0 - blast_probability)
                )
                upper = min(
                    1.0, fragment_probability + blast_probability
                )
            else:
                try:
                    vulnerable_ratio = float(
                        shared_vulnerable_ratio[identifier]
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise InputValidationError(
                        "shared vulnerable ratios must cover every component",
                        context={"component_id": identifier},
                    ) from exc
                if not np.isfinite(vulnerable_ratio) or not (
                    0.0 <= vulnerable_ratio <= 1.0
                ):
                    raise InputValidationError(
                        "shared vulnerable ratio must lie in [0,1]",
                        context={
                            "component_id": identifier,
                            "vulnerable_ratio": vulnerable_ratio,
                        },
                    )
                # Repeated fragment opportunities can make a marginal exceed
                # the single-impact vulnerable-area ratio. The smallest shared
                # event support consistent with both marginals is therefore the
                # maximum of the declared ratio and the two observed marginals.
                # This is a labelled dependence sensitivity, not an identified
                # latent-area estimate.
                shared_support = max(
                    vulnerable_ratio,
                    fragment_probability,
                    blast_probability,
                )
                lower = max(fragment_probability, blast_probability)
                upper = min(
                    1.0, fragment_probability + blast_probability
                )
                if shared_support <= 32.0 * np.finfo(np.float64).eps:
                    independent = 0.0
                else:
                    independent = (
                        fragment_probability
                        + blast_probability
                        - fragment_probability
                        * blast_probability
                        / shared_support
                    )
            scenarios["frechet_lower"][identifier][state] = float(lower)
            scenarios["conditional_independence"][identifier][state] = float(
                independent
            )
            scenarios["frechet_upper"][identifier][state] = float(upper)
            bounds[identifier][state] = {
                "lower": float(lower),
                "upper": float(upper),
            }
    return MechanismCombination(
        point_estimate=scenarios["conditional_independence"],
        frechet_bounds=bounds,
        scenarios=scenarios,
        point_estimate_assumption=(
            "conditional_independence"
            if shared_vulnerable_ratio is None
            else "conditional_independence_within_minimum_shared_support_sensitivity"
        ),
    )
