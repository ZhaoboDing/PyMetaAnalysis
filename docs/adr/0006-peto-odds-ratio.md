# ADR 0006: Peto one-step odds ratio

- Status: Accepted
- Date: 2026-08-13
- Supersedes: the Peto deferral in
  [ADR 0002](0002-statistical-policy.md)

## Context

Peto's one-step method is a conventional common-effect estimator for binary
outcomes and is particularly associated with rare-event meta-analysis. It is
not interchangeable with an ordinary inverse-variance odds ratio: it derives
both the study contribution and pooled estimate from observed-minus-expected
events and hypergeometric information.

The approximation can be biased when treatment and control group sizes differ
substantially within studies, effects are large, or events are not rare. Its
zero-cell behavior also differs from ordinary log odds ratios, so pooling and
display corrections must not be conflated.

## Decision

PyMetaAnalysis supports `measure="OR", method="Peto", model="common"` with
`ci_method="normal"`. `"peto_one_step"` is an accepted alias and the resolved
pooling method is `"peto"`.

For stratum `i`, let `O_i = a_i`, `m_i = a_i + c_i`, treatment and control
totals be `n1_i` and `n0_i`, and `N_i = n1_i + n0_i`. Define:

```text
E_i = m_i n1_i / N_i
V_i = m_i (N_i - m_i) n1_i n0_i / (N_i^2 (N_i - 1))
```

The individual and pooled model-scale estimates are:

```text
y_i = (O_i - E_i) / V_i
Var(y_i) = 1 / V_i
y_Peto = sum(O_i - E_i) / sum(V_i)
Var(y_Peto) = 1 / sum(V_i)
```

Pooling always uses raw 2-by-2 tables. The existing
`continuity_correction` and `correction_scope` settings affect only displayed
study estimates and variances; there is no Peto pooling correction parameter.
Double-zero and double-all rows are excluded before every synthesis
calculation because they contain no relative-effect information.

Peto heterogeneity uses the fitted pooled coefficient and the same
observed-minus-expected contributions:

```text
Q = sum(((O_i - E_i) - y_Peto V_i)^2 / V_i)
```

Q-based I-squared and H-squared follow the project's common-effect
conventions. Method options record `peto_pooling_tables="raw"` and
`peto_heterogeneity="O-minus-E"`. Every Peto result carries an approximation
warning naming the rare-outcome, balanced-arm, and modest-effect conditions.

Random-effects Peto and Peto RR/RD are unsupported. Users requesting those
estimands must select an implemented inverse-variance or Mantel-Haenszel
combination explicitly.

## Validation

- direct formula tests cover study estimates, pooled estimate, variance,
  weights, confidence interval, and Peto Q;
- treatment/control swapping reverses and exponentiates the log-OR limits as
  expected, while row order leaves the fit unchanged;
- count scaling and extreme finite-count tests exercise overflow-safe
  arithmetic;
- sparse tables verify that study-level correction does not alter raw Peto
  pooling and that double-zero/double-all rows are excluded; and
- fixed-version `metafor::escalc(measure="PETO")` and `metafor::rma.peto()`
  fixtures cover ordinary and sparse datasets.

## Consequences

- common-effect binary OR now offers MH, Peto, and inverse-variance pooling;
- Peto results remain on the log-OR model scale and use exponentiated display
  values like other OR results;
- sensitivity, subgroup, provenance, reporting, and plotting workflows reuse
  the same public result contracts; and
- documentation and reports must preserve the Peto applicability warning
  rather than presenting it as a general sparse-data default.
