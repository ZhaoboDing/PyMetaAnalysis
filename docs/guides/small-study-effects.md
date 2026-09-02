# Small-study effects and Egger regression

PyMetaAnalysis provides the classical Egger regression test as a companion to
the descriptive funnel plot. The test examines whether study effects are
associated with their standard errors. It is a diagnostic for funnel-plot
asymmetry or small-study effects, not a direct test for publication bias.

## Run the test

Start from any fitted `MetaAnalysisResult`:

```python
import meta_analyze as ma

result = ma.meta_analysis(
    studies,
    effect="effect",
    standard_error="standard_error",
    model="random",
)

egger = result.egger_test()
print(egger)
```

Only rows with `included=True` enter the regression. The original pooling
model, tau-squared estimate, confidence-interval method, weights, and result
object are not changed. The Egger calculation is the same whether the source
result was fitted as common or random effects because this first API implements
only the classical regression form.

Pair the numerical result with the plot:

```python
ax = result.funnel()
print(egger.statistic, egger.df, egger.pvalue)
```

## Inspect the result

The tested coefficient is the intercept in the standardized-normal-deviate
form of the Egger regression:

```python
egger.intercept
egger.intercept_standard_error
egger.intercept_ci
egger.statistic
egger.df
egger.pvalue
```

The result also exposes the extrapolated effect as the standard error tends to
zero:

```python
egger.limit_estimate
egger.limit_standard_error
egger.limit_ci
egger.display_limit_estimate
egger.display_limit_ci
```

`limit_estimate` remains on the analysis model scale. Its display counterpart
is exponentiated for OR/RR and back-transformed with `tanh` for Fisher's z
correlations. It is an extrapolated regression intercept, not an automatically
bias-corrected replacement for the fitted pooled estimate.

`to_dict()` returns a detached mapping containing the coefficients, intervals,
test, method identifiers, condition number, scales, and warnings:

```python
payload = egger.to_dict()
```

Pass an explicit confidence level for the two coefficient intervals, or omit
it to reuse the fitted analysis level:

```python
egger_90 = result.egger_test(confidence_level=0.90)
```

Changing the confidence level does not change the coefficient estimates, test
statistic, or p-value.

## Statistical form

The classical equation is:

```text
y_i / s_i = alpha + beta * (1 / s_i) + error_i
```

where `y_i` is the effect and `s_i` is its standard error. The two-sided test
is `H0: alpha = 0` and uses a t distribution with `k-2` degrees of freedom.
The fitted `beta` is the limit estimate.

PyMetaAnalysis evaluates the algebraically equivalent weighted regression of
`y_i` on `s_i`, using inverse sampling-variance weights and a multiplicative
residual-dispersion estimate. The implementation scales the weights and design
columns for numerical stability without changing the coefficients or their
covariance. This contract corresponds to:

```r
metafor::regtest(
  effect,
  variance,
  model = "lm",
  predictor = "sei"
)
```

The random/mixed-effects regression version of `metafor::regtest()` is a
different model and is not silently substituted.

## Applicability checks

At least three included studies are mathematically required. PyMetaAnalysis
returns a result for `3 <= k < 10` but records a warning because funnel-
asymmetry tests generally have low power with fewer than ten studies. The
[Cochrane Handbook](https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-13#section-13-3-4-4)
uses ten studies as a rule of thumb and also advises against testing when study
standard errors are all similar.

Exactly or numerically non-identifiable standard errors produce an error
instead of an unstable coefficient. For less extreme cases, inspect the study
size distribution and `egger.condition_number`; the library does not invent a
universal cutoff for “enough” variation.

The classical Egger test is particularly problematic for some effect measures
because the effect and its standard error can be inherently associated.
PyMetaAnalysis therefore records an additional warning for:

- odds ratios, for which binary-outcome alternatives such as Harbord or Peters
  may be preferable;
- standardized mean differences, for which the same association can produce
  distorted funnel plots.

Those outcome-specific tests are not yet implemented. A generic effect labeled
`GENERIC` cannot reveal how the supplied effect was constructed, so callers
must assess this limitation themselves.

## Interpretation

A small p-value indicates evidence that effects and standard errors are
associated under this regression model. Possible explanations include genuine
heterogeneity, design or population differences, selective outcome reporting,
other non-reporting mechanisms, artefactual effect/standard-error correlation,
and chance.

A large p-value does not demonstrate symmetry or exclude missing evidence,
especially with few studies. A small p-value does not establish publication
bias. Interpret the test alongside the funnel plot, heterogeneity, study
characteristics, protocol information, and sensitivity analyses. The
[original Egger paper](https://doi.org/10.1136/bmj.315.7109.629) and
[`metafor::regtest`](https://wviechtb.github.io/metafor/reference/regtest.html)
provide the methodological and software references for this implementation.
