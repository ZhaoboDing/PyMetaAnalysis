# Small-study effects and regression tests

PyMetaAnalysis provides classical Egger regression plus Harbord and Peters
tests for two-group binary odds ratios as companions to the descriptive funnel
plot. They diagnose funnel-plot asymmetry or small-study effects; none is a
direct test for publication bias.

## Run the classical Egger test

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
ax = result.funnel(contour_levels=(0.90, 0.95, 0.99))
print(egger.statistic, egger.df, egger.pvalue)
```

The optional contours mark two-sided significance regions around the null
effect. They can make it easier to see whether an apparently missing part of
the funnel lies mainly in a non-significant region, but they do not establish
that studies are missing.

## Harbord test for binary odds ratios

For an analysis fitted from retained two-group counts with
`meta_binary(..., measure="OR")`, Harbord regression uses the efficient score
and its variance:

```python
odds_ratios = ma.meta_binary(
    trials,
    event_treat="event_treat",
    n_treat="n_treat",
    event_control="event_control",
    n_control="n_control",
    measure="OR",
    method="MH",
)

harbord = odds_ratios.harbord_test()
print(harbord.statistic, harbord.df, harbord.pvalue)
```

`harbord.intercept` is the tested asymmetry coefficient in the standardized
score regression. `harbord.limit_estimate` is its efficient-score limit
coefficient; `display_limit_estimate` and `display_limit_ci` exponentiate that
coefficient. Treat these as regression outputs, not replacements for the
pooled estimate.

Harbord derives null efficient scores directly from raw treatment/control
counts. It does not use continuity-corrected study log odds ratios, so changing
the source study-level correction does not change the diagnostic. It is also
independent of whether the source analysis used MH, inverse variance,
random-effects inverse variance, or Peto pooling. Single-arm zero-event studies
remain usable when their total event and non-event margins are positive.

The result records the standardized-score response, square-root score-variance
predictor, equivalent weighting convention, residual dispersion, scaled-design
condition number, and the fact that no continuity correction is used.

## Peters test for binary odds ratios

When the analysis was fitted from retained two-group counts with
`meta_binary(..., measure="OR")`, use the outcome-specific Peters regression:

```python
odds_ratios = ma.meta_binary(
    trials,
    event_treat="event_treat",
    n_treat="n_treat",
    event_control="event_control",
    n_control="n_control",
    measure="OR",
    method="MH",
)

peters = odds_ratios.peters_test()
print(peters.statistic, peters.df, peters.pvalue)
```

The test reconstructs conventional study log odds ratios from the retained
four-cell counts and the analysis's recorded study-level continuity correction.
Its result is therefore independent of whether the source pooled estimate used
MH, inverse variance, random-effects inverse variance, or Peto. If the source
used Peto, a note makes the different study-effect construction explicit.

The tested coefficient is `peters.slope`, the slope of log OR on inverse total
sample size. `peters.limit_estimate` is the extrapolated log OR as total sample
size tends to infinity; `display_limit_estimate` and `display_limit_ci` are on
the OR scale. These are regression extrapolations, not replacements for the
pooled estimate. The result also records `residual_dispersion`,
`weight_method`, the continuity-correction contract, corrected-study count,
and the scaled-design condition number.

Peters regression is unavailable for generic effects whose original four-cell
counts are no longer known, and for RR, RD, continuous, correlation, or
diagnostic-accuracy analyses.

## Inspect the Egger result

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

Harbord and Peters regression are available for OR analyses created by
`meta_binary()`. A generic effect labeled `GENERIC` cannot reveal the original
two-group counts, so neither binary-specific method accepts it.

For Harbord regression, the efficient-score variances must vary enough to
identify the asymmetry intercept. Every included study must have positive total
events and non-events. The method uses raw counts and never applies the source
continuity correction. At least three studies are mathematically required, and
the same fewer-than-ten warning applies.

For Peters regression, total sample sizes must vary enough to identify the
slope. The method uses `S*F/N` weights, where `S` and `F` are the raw total
events and non-events. Its continuity correction affects only the reconstructed
study log OR, not these marginal-count weights. At least three studies are
mathematically required, and the same fewer-than-ten warning applies.

## Interpretation

A small p-value indicates evidence of the association defined by the selected
diagnostic: effect with standard error for Egger, standardized efficient score
with score precision for Harbord, or log OR with inverse total sample size for
Peters. Possible explanations include genuine heterogeneity, design or
population differences, selective outcome reporting, other non-reporting
mechanisms, artefactual associations, and chance.

A large p-value does not demonstrate symmetry or exclude missing evidence,
especially with few studies. A small p-value does not establish publication
bias. Interpret the test alongside the funnel plot, heterogeneity, study
characteristics, protocol information, and sensitivity analyses. Contour-
enhanced funnels add useful significance context, but neither locate missing
studies nor identify the mechanism behind asymmetry. The
[original Egger paper](https://doi.org/10.1136/bmj.315.7109.629) and
[`metafor::regtest`](https://wviechtb.github.io/metafor/reference/regtest.html)
provide the methodological and software references for the Egger
implementation. The Harbord and Peters implementations follow the documented
[`meta::metabias`](https://search.r-project.org/CRAN/refmans/meta/html/metabias.html)
contract and the corresponding Harbord et al. (2006) and Peters et al. (2006)
methods cited there.
