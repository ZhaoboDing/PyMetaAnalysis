# ADR 0004: Hartung-Knapp prediction intervals

- Status: Accepted
- Date: 2026-07-24

## Context

ADR 0002 applied the classic pooled-mean variance to every `k - 2`
random-effects prediction interval. That kept the interval independent of
`ci_method`, but it diverged from `metafor` Riley predictions and the
Hartung-Knapp Partlett-Riley option in R `meta`. Both use the covariance
selected for mean inference inside the prediction variance.

This difference was material when the unmodified Hartung-Knapp variance was
below or above the classic variance. The result metadata still identified the
interval as `HTS`, so callers could not discover the difference from method
configuration alone.

## Decision

Random-effects inverse-variance prediction intervals retain the `k - 2`
critical value and three-study minimum:

```text
mu_hat +/- t_(k - 2, 1 - alpha/2)
          * sqrt(tau^2 + Var_selected(mu_hat))
```

- normal inference uses the classic pooled-mean variance and records `HTS`;
- `hartung_knapp` uses its unmodified adjusted variance and records `HK-PR`;
- `hartung_knapp_adhoc` uses its lower-bounded adjusted variance and records
  `HK-PR`.

Committed values are generated directly from
`metafor::predict(fit, predtype="Riley")` for all three inference choices.
Prediction intervals remain unavailable below three included studies and
retain the explicit warning with three or four studies.

## Consequences

- prediction intervals and mean intervals use a coherent selected covariance;
- HK and safeguarded HK prediction intervals can differ from the normal HTS
  interval even when tau-squared and the pooled estimate are unchanged;
- `result.method.prediction_interval_method` distinguishes `HTS` from
  `HK-PR`;
- this decision supersedes only the prediction-variance paragraph of ADR 0002.
