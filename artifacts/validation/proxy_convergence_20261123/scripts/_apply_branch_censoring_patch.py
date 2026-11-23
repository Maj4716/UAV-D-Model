"""One-off patch applied to damage_engine/fast_sampling_proxy.py (kept for the record).

Adds the opt-in ``domain_censoring="branch_probability"`` option and the
``TEACHER_PREDICTOR_OPTIONS`` preset.  Refuses to run if any anchor is missing,
so it cannot apply to a concurrently modified module.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
p = ROOT / "damage_engine/fast_sampling_proxy.py"
s = p.read_text(encoding="utf-8")


def rep(old: str, new: str, count: int = 1) -> None:
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)


rep('''  ``spatial_cumulative_damage_energy`` complete-link energy clusters.
"""''', '''  ``spatial_cumulative_damage_energy`` complete-link energy clusters.
* ``domain_censoring`` -- ``"any_branch"`` censors a whole packet when any
  representative kernel branch is refused (historical); ``"branch_probability"``
  censors only the refused ballistic-limit quantile mass, as the teacher's
  single quantile draw per contact does, and keeps the resolved branches.

``TEACHER_PREDICTOR_OPTIONS`` collects the configuration recommended by
``artifacts/validation/proxy_convergence_20261123`` for using the proxy as an
expectation of the software teacher.
"""''')
rep('''from dataclasses import dataclass, field
import hashlib
import math
from typing import Any
''', '''from dataclasses import dataclass, field
import hashlib
import math
from types import MappingProxyType
from typing import Any, Mapping
''')
rep('''DAMAGE_ACCUMULATION_OPTIONS = ("independent_hits", "registry")
''', '''DAMAGE_ACCUMULATION_OPTIONS = ("independent_hits", "registry")
DOMAIN_CENSORING_OPTIONS = ("any_branch", "branch_probability")
ONE_RAY_PER_FRAGMENT = 1_000_000
"""``azimuths_per_ring`` value at or above every registered ring size."""

TEACHER_PREDICTOR_OPTIONS: Mapping[str, Any] = MappingProxyType(
    {
        "azimuths_per_ring": ONE_RAY_PER_FRAGMENT,
        "source_realizations": 8,
        "damage_accumulation": "registry",
        "domain_censoring": "branch_probability",
    }
)
"""Converged teacher-expectation configuration (opt-in; defaults unchanged)."""
''')
rep('''    perforated_damage_energy_j: float
    stopped_damage_energy_j: float


@dataclass(frozen=True)
class _ComponentImpact:''', '''    perforated_damage_energy_j: float
    stopped_damage_energy_j: float
    out_of_domain_probability: float = 0.0


@dataclass(frozen=True)
class _ComponentImpact:''')
rep('''    perforation_probability: float
    perforated_damage_energy_j: float
    stopped_damage_energy_j: float


@dataclass
class _RealizationTally:''', '''    perforation_probability: float
    perforated_damage_energy_j: float
    stopped_damage_energy_j: float
    out_of_domain_probability: float = 0.0


@dataclass
class _RealizationTally:''')
rep('''def _representative_expected_path(
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

    has_stack''', '''def _try_resolve(
    impact: FragmentImpact, response: dict[str, Any], registry, rng
) -> PenetrationPathOutcome | None:
    try:
        return resolve_penetration_path(impact, response, registry, rng)
    except OutOfDomainError:
        return None


def _branch_probability_expected_path(
    impact: FragmentImpact,
    response: dict[str, Any],
    registry,
    *,
    layer_quadrature_samples: int,
) -> _ExpectedImpact:
    """Teacher-consistent expectation that censors only refused quantile mass.

    The teacher draws one ballistic-limit quantile per contact and the kernel
    either resolves it (perforation or stop) or refuses it.  Under a pathwise
    high-obliquity bound the refusal occurs only on the low-quantile side
    (sampled limit below the arrival speed); the other quantiles are certified
    stops.  The same representative quantiles as the historical expectation
    are evaluated, but a refused branch contributes its probability mass to
    ``out_of_domain_probability`` instead of censoring the whole contact.
    Refused mass deposits no energy, as in the teacher.
    """

    eps = np.finfo(float).eps
    refused_everywhere = _ExpectedImpact(
        penetration_probability=0.0,
        incidence_cosine=1.0,
        v50_mps=0.0,
        representative_residual_speed_mps=0.0,
        expected_absorbed_energy_j=0.0,
        expected_damage_energy_j=0.0,
        perforated_damage_energy_j=0.0,
        stopped_damage_energy_j=0.0,
        out_of_domain_probability=1.0,
    )
    branches: list[tuple[float, PenetrationPathOutcome | None]]
    if response.get("layer_stack_id") is None:
        reference = _try_resolve(impact, response, registry, _QuantileRng(0.5))
        if reference is None:
            reference = _try_resolve(
                impact, response, registry, _QuantileRng(1.0 - eps)
            )
        if reference is None:
            return refused_everywhere
        split = float(np.clip(reference.penetration_probability, 0.0, 1.0))
        branches = []
        if split > 0.0:
            low = _QuantileRng(max(eps, 0.5 * split))
            branches.append((split, _try_resolve(impact, response, registry, low)))
        if split < 1.0:
            high = _QuantileRng(split + 0.5 * (1.0 - split))
            branches.append(
                (1.0 - split, _try_resolve(impact, response, registry, high))
            )
    else:
        weight = 1.0 / float(layer_quadrature_samples)
        branches = [
            (
                weight,
                _try_resolve(
                    impact,
                    response,
                    registry,
                    _LatticeRng(index, layer_quadrature_samples),
                ),
            )
            for index in range(layer_quadrature_samples)
        ]
        resolved = [outcome for _, outcome in branches if outcome is not None]
        if not resolved:
            return refused_everywhere
        reference = resolved[0]

    def mass(perforated: bool) -> float:
        return float(
            sum(
                weight
                for weight, item in branches
                if item is not None and item.perforated == perforated
            )
        )

    def conditional_mean(name: str, perforated: bool) -> float:
        selected = [
            (weight, item)
            for weight, item in branches
            if item is not None and item.perforated == perforated
        ]
        total = sum(weight for weight, _ in selected)
        if total <= 0.0:
            return 0.0
        return float(
            sum(weight * float(getattr(item, name)) for weight, item in selected)
            / total
        )

    perforation = mass(True)
    stop = mass(False)
    refused = float(sum(weight for weight, item in branches if item is None))
    perforated_damage = conditional_mean("damage_energy_j", True)
    stopped_damage = conditional_mean("damage_energy_j", False)
    expected_absorbed = perforation * conditional_mean(
        "absorbed_energy_j", True
    ) + stop * conditional_mean("absorbed_energy_j", False)
    expected_damage = perforation * perforated_damage + stop * stopped_damage
    return _ExpectedImpact(
        penetration_probability=perforation,
        incidence_cosine=float(reference.incidence_cosine),
        v50_mps=float(reference.v50_mps),
        representative_residual_speed_mps=conditional_mean(
            "residual_speed_mps", True
        ),
        expected_absorbed_energy_j=float(expected_absorbed),
        expected_damage_energy_j=float(expected_damage),
        perforated_damage_energy_j=perforated_damage,
        stopped_damage_energy_j=stopped_damage,
        out_of_domain_probability=float(np.clip(refused, 0.0, 1.0)),
    )


def _representative_expected_path(
    impact: FragmentImpact,
    response: dict[str, Any],
    registry,
    *,
    layer_quadrature_samples: int,
    domain_censoring: str = "any_branch",
) -> _ExpectedImpact:
    """Integrate the same public penetration path used by the full solver.

    A single layer has an analytic perforation probability, so its two
    conditional representative branches are evaluated directly through the
    registered kernel.  A declared multi-layer stack uses a deterministic
    lattice over per-layer quantiles; no residual or damage law is duplicated
    in the proxy.  ``domain_censoring="branch_probability"`` delegates to
    :func:`_branch_probability_expected_path`; the historical mode lets any
    refused branch raise :class:`OutOfDomainError`.
    """

    if domain_censoring == "branch_probability":
        return _branch_probability_expected_path(
            impact,
            response,
            registry,
            layer_quadrature_samples=layer_quadrature_samples,
        )
    has_stack''')
rep('''        branch_probability = np.asarray(
            [1.0 - reach, reach * perforation, reach * (1.0 - perforation)],
            dtype=np.float64,
        )
        branch_energy = np.asarray(
            [
                0.0,
                float(impact.perforated_damage_energy_j),
                float(impact.stopped_damage_energy_j),
            ],
            dtype=np.float64,
        )''', '''        refused = float(np.clip(impact.out_of_domain_probability, 0.0, 1.0))
        branch_probability = np.asarray(
            [
                1.0 - reach,
                reach * perforation,
                reach * max(0.0, 1.0 - perforation - refused),
                reach * refused,
            ],
            dtype=np.float64,
        )
        branch_energy = np.asarray(
            [
                0.0,
                float(impact.perforated_damage_energy_j),
                float(impact.stopped_damage_energy_j),
                0.0,
            ],
            dtype=np.float64,
        )''')
rep('''        source_realizations: int = 1,
        damage_accumulation: str = "independent_hits",
    ) -> None:''', '''        source_realizations: int = 1,
        damage_accumulation: str = "independent_hits",
        domain_censoring: str = "any_branch",
    ) -> None:''')
rep('''                + ", ".join(repr(item) for item in DAMAGE_ACCUMULATION_OPTIONS)
            )
        self.engine = engine''', '''                + ", ".join(repr(item) for item in DAMAGE_ACCUMULATION_OPTIONS)
            )
        if domain_censoring not in DOMAIN_CENSORING_OPTIONS:
            raise ValueError(
                "domain_censoring must be one of "
                + ", ".join(repr(item) for item in DOMAIN_CENSORING_OPTIONS)
            )
        self.engine = engine''')
rep('''        self.damage_accumulation = str(damage_accumulation)
        self.parameters''', '''        self.damage_accumulation = str(damage_accumulation)
        self.domain_censoring = str(domain_censoring)
        self.parameters''')
rep('''                        self.engine.registry,
                        layer_quadrature_samples=self.layer_quadrature_samples,
                    )
                except OutOfDomainError:''', '''                        self.engine.registry,
                        layer_quadrature_samples=self.layer_quadrature_samples,
                        domain_censoring=self.domain_censoring,
                    )
                except OutOfDomainError:''')
rep('''                probability = expectation.penetration_probability
                weighted_reach = packet_weight * reach
''', '''                probability = expectation.penetration_probability
                refused = expectation.out_of_domain_probability
                weighted_reach = packet_weight * reach
                if refused > 0.0:
                    role = "armor" if hit.role == "armor" else "component"
                    tally.out_of_domain_packet_weight += weighted_reach * refused
                    tally.out_of_domain_mass_kg += (
                        weighted_reach * refused * float(fragment.mass_kg)
                    )
                    tally.out_of_domain_weight_by_role[role] += (
                        weighted_reach * refused
                    )
''')
rep('''                                stopped_damage_energy_j=float(
                                    expectation.stopped_damage_energy_j
                                ),
                            )
                        )''', '''                                stopped_damage_energy_j=float(
                                    expectation.stopped_damage_energy_j
                                ),
                                out_of_domain_probability=float(refused),
                            )
                        )''')
rep('''                                + (1.0 - probability)
                                * state_probability(
                                    expectation.stopped_damage_energy_j
                                )''', '''                                + (1.0 - probability - refused)
                                * state_probability(
                                    expectation.stopped_damage_energy_j
                                )''')
rep('''            "damage_accumulation": self.damage_accumulation,
            "source_fragment_count"''', '''            "damage_accumulation": self.damage_accumulation,
            "domain_censoring": self.domain_censoring,
            "source_fragment_count"''')
rep('''__all__ = [
    "DAMAGE_ACCUMULATION_OPTIONS",
    "FAST_SAMPLING_PROXY_VERSION",
    "FastSamplingProxy",
]''', '''__all__ = [
    "DAMAGE_ACCUMULATION_OPTIONS",
    "DOMAIN_CENSORING_OPTIONS",
    "FAST_SAMPLING_PROXY_VERSION",
    "FastSamplingProxy",
    "ONE_RAY_PER_FRAGMENT",
    "TEACHER_PREDICTOR_OPTIONS",
]''')
p.write_text(s, encoding="utf-8")
print("patched", p)
