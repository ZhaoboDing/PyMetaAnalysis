# Choosing methods

Method choices should follow the scientific estimand and review protocol, not
be selected solely because a heterogeneity test crosses a p-value threshold.
This page describes the implemented options; it is not a substitute for a
protocol-specific statistical review.

## Common-effect or random-effects

A common-effect model estimates one effect shared by every included study. A
random-effects model estimates the mean of a distribution of study effects and
adds a between-study variance, tau-squared, to inverse-variance weights.

Use a common-effect model when the intended estimand and study designs support
a shared underlying effect. Use random effects when meaningful effect
variation is part of the estimand. A small number of studies makes
between-study variance and interval estimation uncertain; it does not turn a
random-effects question into a common-effect question.

## Pooling methods

| Data and model | Available pooling method |
| --- | --- |
| Generic effects, common or random | Inverse variance |
| Binary OR/RR, common effect | Mantel-Haenszel or inverse variance |
| Binary OR/RR, random effects | Inverse variance |
| Binary RD, common or random | Inverse variance |
| Continuous MD/SMD, common or random | Inverse variance |

Mantel-Haenszel and inverse variance are different estimators, not aliases.
PyMetaAnalysis does not extrapolate its common-effect Mantel-Haenszel weights
into an undocumented random-effects procedure. Mantel-Haenszel OR/RR pooling
uses raw tables by default and is not offered for RD.

## Estimating tau-squared

Random-effects inverse-variance models provide:

- `REML`, the default restricted maximum-likelihood estimator;
- `PM`, the Paule-Mandel estimating-equation method;
- `DL`, the closed-form DerSimonian-Laird estimator.

REML and PM are iterative. Convergence, iteration count, and whether the
solution reached the zero boundary are recorded in `result.diagnostics`.
Failure to converge raises `ConvergenceError`; it does not silently fall back
to DL.

## Confidence intervals

`ci_method="normal"` is the stable default and uses the classic normal
approximation. Random-effects
inverse-variance models also support:

- `hartung_knapp`, which uses a t critical value and residual scale estimate;
- `hartung_knapp_adhoc`, which additionally prevents the adjusted variance
  from falling below the classic variance.

The two Hartung-Knapp variants are intentionally distinct. With very few or
very homogeneous studies, unprotected Hartung-Knapp intervals can be narrower
than classic intervals. The result records a note when that occurs; it never
silently substitutes the ad hoc safeguard.

For a random-effects analysis with positive tau-squared and more than three
studies, consider Hartung-Knapp as a sensitivity analysis. With three or fewer
studies, compare the normal and Hartung-Knapp results and interpret both
cautiously: choosing one does not remove the small-sample uncertainty.

## Heterogeneity definitions

Cochran's Q, its degrees of freedom, and its p-value always use common-effect
inverse-variance weights. Common-effect and Mantel-Haenszel analyses derive
I-squared and H-squared from Q.

For random-effects analyses, I-squared and H-squared instead use the estimated
tau-squared and a typical within-study variance:

```text
C = sum(w_i) - sum(w_i^2) / sum(w_i), where w_i = 1 / v_i
v_typical = (k - 1) / C
I^2 = tau^2 / (tau^2 + v_typical)
H^2 = 1 + tau^2 / v_typical
```

The definition is recorded as `result.i2_method`: `q_based` or
`tau2_typical_variance`. Internally, I-squared remains a proportion from zero
to one.

## Prediction intervals

Random-effects models report an HTS prediction interval when at least three
studies are included. It describes uncertainty for a new study's underlying
effect, not uncertainty around the pooled mean. Common-effect models do not
produce prediction intervals. With three or four studies the interval is still
calculated, but the result warns that it is especially uncertain.

## Subgroups

Current random-effects subgroup analyses estimate tau-squared independently
within each subgroup and again for the overall model. This is recorded as
`result.method.tau2_strategy == "independent"`. The test for subgroup
differences compares subgroup summary estimates; it is not a comparison of
whether individual subgroup p-values are significant.

## Reporting checklist

At minimum, report:

- the effect measure and its direction;
- the common-effect or random-effects model;
- the pooling method;
- the tau-squared estimator for random effects;
- the confidence-interval method and confidence level;
- the continuity-correction policy for sparse binary data;
- included and excluded studies with reasons;
- Q, I-squared, H-squared, tau-squared, and the I-squared definition where
  applicable;
- the prediction interval when relevant.

See the Cochrane Handbook chapter on
[meta-analysis](https://training.cochrane.org/handbook/current/chapter-10) for
broader methodological guidance.
