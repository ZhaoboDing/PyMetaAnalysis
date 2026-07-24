# ADR 0003: Meta-regression prediction-interval choices

- Status: Accepted
- Date: 2026-07-24

## Context

For a mixed-effects Meta-regression prediction at design vector `x`, the
estimated true-effect variance combines residual heterogeneity with
uncertainty in the fitted mean:

```text
Var_prediction = tau^2 + x' Cov(beta_hat) x
```

The critical-value distribution remains a methodological choice. Common
software offers a default normal-or-t rule and a Riley alternative with one
fewer residual degree of freedom. Neither approximation removes uncertainty
from estimating tau-squared, and changing the package default would alter
existing results.

## Decision

`meta_regression()` accepts
`prediction_interval_method="default" | "riley"` for mixed-effects models.
The canonical resolved method is stored in
`result.method.prediction_interval_method`.

The default remains `normal_or_t_k_minus_p`:

- normal coefficient inference uses a standard normal prediction critical
  value;
- either Hartung-Knapp mode uses a t critical value with `k-p` degrees of
  freedom.

The opt-in Riley method uses a t critical value with `k-p-1` degrees of
freedom regardless of the coefficient-inference distribution:

```text
prediction = x' beta_hat
PI_Riley = prediction +/- t_(k-p-1) *
           sqrt(tau^2 + x' Cov(beta_hat) x)
```

Riley intervals require `k-p >= 2`. Requesting Riley for a common-effect model
or a mixed model without enough residual degrees of freedom raises a domain
error. The selection is preserved by deleted-study refits and reused by
`predict()` and `bubble()`.

Both rules predict the distribution of true effects in a new study at the
specified moderator values. They do not add an unknown sampling variance for
a future observed effect.

## Validation

The committed fixed-version R `metafor` fixture records default and
`predtype="Riley"` predictions for normal and Hartung-Knapp inference,
multivariable moderator values, and a zero-tau-squared boundary. Unit tests
also check the critical-value formula and invalid degrees of freedom.
Property-based tests verify symmetry, unchanged mean-effect inference, and
the Riley interval's greater width relative to the default rule.

## Consequences

- existing fits retain their numerical default;
- the alternative is explicit, auditable, and reproducible;
- Riley is documented as an alternative approximation rather than an
  automatic small-sample correction;
- future prediction-interval rules require a separate statistical decision
  and independent reference coverage.
