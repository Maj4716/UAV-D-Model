# High-obliquity one-sided nonperforation bound

## Status

`loitering-impact-taylor-shapiro-obliquity-bound-unverified-3` is an
unverified software-teacher candidate. It does not extend the registered
domain of the ballistic response, predict ricochet, or turn all 80–90 degree
contacts into point labels.

## Shape constraint and proof

Let `theta_max` be the registered response-domain boundary and let `Q` be the
event-specific latent logistic quantile already used by the stochastic
ballistic-limit model. The v3 policy asserts only the pathwise shape constraint

`V_limit(theta, Q) >= V_limit(theta_max, Q)` for `theta > theta_max`.

The implementation evaluates the sampled ballistic limit at `theta_max` with
the same event random key. When the arrival speed is no greater than that
boundary value, nonperforation follows for every response satisfying the shape
constraint. The fragment is therefore stopped. For a component contact, the
existing nonperforation absorbed-energy model is still evaluated. When the
inequality does not prove nonperforation, the event remains OOD and follows the
existing identification-interval path.

The reported probability for a certified event is the boundary-domain
penetration probability upper bound, not an extrapolated point probability at
the observed angle.

## Excluded mechanisms

The policy supplies no ricochet direction, residual speed after ricochet,
fragment breakup, secondary fragments, plugging, or high-obliquity penetration
point estimate. Its no-ricochet scope matches the current software teacher. A
physical ricochet model requires a new registry identity, parameter ranges,
sensitivity analysis, and external calibration evidence.

## Version isolation

- v2 keeps the original 80/85 degree fail-closed response unchanged.
- v3 adds `high_obliquity_nonperforation_bound` to the response models.
- In-domain stochastic draws are bitwise unchanged for equal event seeds.
- The v3 case and registry have independent identifiers and provenance.

