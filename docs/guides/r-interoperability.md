# Mapping R workflows to PyMetaAnalysis

This guide helps users translate conventional R `meta` and `metafor`
workflows into PyMetaAnalysis calls. It is a terminology and configuration
map, not a claim that similarly named functions always use identical formulas
or defaults.

Always compare the resolved configuration and numerical output when porting an
analysis. See [validation](../validation.md) for the R versions, calls, and
fixtures used by this project.

## Entry points

| Analysis | PyMetaAnalysis | R `metafor` | R `meta` |
| --- | --- | --- | --- |
| Generic effects and variances | `meta_analysis()` | `rma.uni(yi, vi, ...)` | `metagen(TE, seTE, ...)` |
| Binary 2x2 tables, inverse variance | `meta_binary(..., method="IV")` | `escalc()` then `rma.uni()` | `metabin(..., method="Inverse")` |
| Binary 2x2 tables, Mantel-Haenszel | `meta_binary(..., method="MH")` | `rma.mh()` | `metabin(..., method="MH")` |
| Binary 2x2 tables, Peto OR | `meta_binary(..., measure="OR", method="Peto")` | `rma.peto()` | `metabin(..., sm="OR", method="Peto")` |
| Continuous group summaries | `meta_continuous()` | `escalc()` then `rma.uni()` | `metacont()` |
| Correlations and sample sizes | `meta_correlation()` | `escalc(measure="ZCOR")` then `rma.uni()` | `metacor(sm="ZCOR")` |
| Subgroups | `subgroup=` on a high-level call | separate fits or a moderator model | `subgroup=` |
| Leave-one-out | `result.leave_one_out()` | `leave1out()` for supported fits | `metainf()` |
| Meta-regression influence | `regression.influence()` | `influence()`, `rstudent()`, `cooks.distance()`, `dfbetas()` | — |
| Meta-regression collinearity | `regression.collinearity()` | `vif()` plus weighted design diagnostics | — |
| Meta-regression linear contrasts | `regression.contrast(...)` | `anova(..., X=..., rhs=...)` | — |
| Cumulative analysis | `result.cumulative()` | `cumul()` | `metacum()` |
| Classical Egger test | `result.egger_test()` | `regtest(..., model="lm", predictor="sei")` | `metabias(..., method.bias="Egger")` |

PyMetaAnalysis intentionally has no `metabin`, `metacont`, or `rma` aliases.
One documented Python entry point per input shape keeps result types and
provenance behavior consistent.

## Input names

| Meaning | PyMetaAnalysis | R `metafor` | R `meta` |
| --- | --- | --- | --- |
| Study effect | `effect` | `yi` | `TE` |
| Sampling variance | `variance` | `vi` | square of `seTE` |
| Study label | `study` or DataFrame index | `slab` | `studlab` |
| Treatment events | `event_treat` | `ai` | `event.e` |
| Treatment total | `n_treat` | `n1i` | `n.e` |
| Control events | `event_control` | `ci` | `event.c` |
| Control total | `n_control` | `n2i` | `n.c` |
| Treatment mean/SD | `mean_treat`, `sd_treat` | `m1i`, `sd1i` | `mean.e`, `sd.e` |
| Control mean/SD | `mean_control`, `sd_control` | `m2i`, `sd2i` | `mean.c`, `sd.c` |
| Correlation/sample size | `correlation`, `n` | `ri`, `ni` | `cor`, `n` |

PyMetaAnalysis accepts DataFrame column names or aligned one-dimensional
array-like values. When `study=` is omitted for a DataFrame, its index supplies
the display labels.

## Measures and scales

| PyMetaAnalysis `measure` | Meaning | `metafor` measure | `meta` `sm` | Model scale |
| --- | --- | --- | --- | --- |
| `"OR"` | Odds ratio | `"OR"` | `"OR"` | log ratio |
| `"RR"` | Risk ratio | `"RR"` | `"RR"` | log ratio |
| `"RD"` | Risk difference | `"RD"` | `"RD"` | identity |
| `"MD"` | Mean difference | `"MD"` | `"MD"` | identity |
| `"SMD"` | Exact-corrected Hedges' g | `"SMD"` with the documented correction | `"SMD"` with exact Hedges correction | identity |
| `"ZCOR"` | Fisher's z-transformed correlation | `"ZCOR"` | `"ZCOR"` | Fisher's z |

For OR and RR, `estimate` and `ci` remain on the log model scale;
`display_estimate` and `display_ci` provide exponentiated ratios. For `ZCOR`,
the model attributes remain Fisher's z values and the display attributes apply
`tanh` to return correlations. This is similar to choosing transformed or
untransformed printing in R, but both scales remain explicit attributes in
Python.

## Models, pooling, and heterogeneity

| PyMetaAnalysis | `metafor` analogue | `meta` analogue | Notes |
| --- | --- | --- | --- |
| `model="common"` | `rma.uni(..., method="EE")` | `common=TRUE, random=FALSE` | Inverse-variance common-effect fit |
| `model="random"` | random-effects `rma.uni()` | `random=TRUE` | Requires a tau-squared policy |
| `method="IV"` | inverse-variance weighting | `method="Inverse"` | Binary API only; generic, continuous, and correlation fits are IV |
| `method="MH"` | `rma.mh()` | `method="MH"` | Common-effect OR/RR/RD |
| `method="Peto"` | `rma.peto()` | `method="Peto"` | Common-effect OR; raw pooling tables and O-minus-E heterogeneity |
| `tau2_method=None` (resolved as `"REML"`) | `method="REML"` | `method.tau="REML"` | PyMetaAnalysis random-effects default |
| `tau2_method="PM"` | `method="PM"` | `method.tau="PM"` | Paule-Mandel |
| `tau2_method="DL"` | `method="DL"` | `method.tau="DL"` | DerSimonian-Laird |

The same label does not guarantee identical optimizer tolerances, boundary
handling, or heterogeneity definitions. In particular, PyMetaAnalysis records
`i2_method`; random-effects inverse-variance results use the documented
tau-squared/typical-variance definition, while common-effect and MH results use
the Q-based definition.
Peto also reports Q-based inconsistency, but its Q is formed from the Peto
observed-minus-expected contributions rather than ordinary study-log-OR
inverse-variance residuals.

`result.tau2_confidence_interval()` corresponds to
`confint(fit, type="QP")` for an `rma.uni` random-effects fit. Both return
Q-profile bounds for tau-squared and monotonic tau, I-squared, and H-squared
transformations. PyMetaAnalysis additionally exposes `is_empty` so a formal
empty confidence set is distinguishable from its constrained `[0, 0]`
display.

## Confidence and prediction intervals

| PyMetaAnalysis `ci_method` | `metafor` | R `meta` | Behavior |
| --- | --- | --- | --- |
| `"normal"` | default `test="z"` | `method.random.ci="classic"` | Normal mean interval |
| `"hartung_knapp"` | `test="knha"` | `method.random.ci="HK"` | Unmodified HK variance and t quantile |
| `"hartung_knapp_adhoc"` | `test="adhoc"` | HK plus an explicitly selected ad hoc correction | HK variance cannot fall below the classic variance |

Eligible random-effects fits use the documented HTS prediction interval under
normal inference. Either Hartung-Knapp choice instead uses its selected
pooled-mean variance with `k-2` degrees of freedom (`HK-PR`), matching
`predict(fit, predtype="Riley")` in `metafor`. R packages offer additional
prediction-interval choices, so method settings must still be compared
explicitly.

Meta-regression uses a separate prediction rule. Its default corresponds to
`predict(fit)` in `metafor`: normal inference uses a normal critical value and
Hartung-Knapp inference uses `t_(k-p)`. PyMetaAnalysis
`prediction_interval_method="riley"` corresponds to
`predict(fit, predtype="Riley")`, using `t_(k-p-1)` for the true-effect
prediction interval while leaving the mean-effect confidence interval
unchanged.

## Sparse binary studies

The closest names are:

| PyMetaAnalysis | R `meta` | Meaning |
| --- | --- | --- |
| `continuity_correction` | `incr` | Increment used for corrected study effects |
| `correction_scope="only_zero_studies"` | `method.incr="only0"` | Correct only studies containing a zero cell |
| `correction_scope="if_any_zero"` | `method.incr="if0all"` | Correct every study when any study contains a zero cell |
| `correction_scope="all_studies"` | `method.incr="all"` | Correct every study |
| `correction_scope="none"` | no increment | Disable study-effect correction |
| `mh_continuity_correction=None` | `MH.exact=TRUE` in intent | Avoid correction for exact MH pooling where defined |

These are conceptual mappings, not interchangeable switches. R `meta` also
supports dataset-wide correction scopes and methods that PyMetaAnalysis does
not implement. For OR/RR, double-zero and double-all studies are excluded from
all model and heterogeneity calculations by default while remaining visible in
the result table. RD uses its separate `rd_zero_variance` policy.

For `measure="RD"`, PyMetaAnalysis and `metafor::rma.mh(measure="RD")` use
the Sato-Greenland-Robins sampling variance. A separate explicit
`mh_continuity_correction` changes the MH tables; the study-effect
`continuity_correction` still affects only displayed study variances and
heterogeneity.

For Peto, PyMetaAnalysis's default study-level correction corresponds to
`escalc(measure="PETO", add=0.5, to="only0", drop00=TRUE)` for the displayed
study rows. Its pooled fit corresponds to `rma.peto()` with raw pooling tables;
the study correction never changes the pooled result. Compare Peto's
rare-outcome, within-study arm-balance, and modest-effect assumptions before
porting it solely because a table contains zeros.

Read [zero-event studies](zero-events.md) before translating sparse analyses.

## Worked generic translation

Python:

```python
import meta_analyze as ma

result = ma.meta_analysis(
    data=studies,
    effect="yi",
    variance="vi",
    model="random",
    tau2_method="REML",
    ci_method="hartung_knapp_adhoc",
)
```

The corresponding `metafor` configuration is conceptually:

```r
rma.uni(
  yi = yi,
  vi = vi,
  method = "REML",
  test = "adhoc",
  data = studies
)
```

The corresponding R `meta` configuration starts from standard errors rather
than variances. PyMetaAnalysis can accept that uncertainty column directly as
`standard_error=`; no manual squaring step is required:

```python
result = ma.meta_analysis(
    studies,
    effect="yi",
    standard_error="sei",
    model="random",
    tau2_method="REML",
    ci_method="hartung_knapp_adhoc",
)
```

```r
metagen(
  TE = yi,
  seTE = sqrt(vi),
  common = FALSE,
  random = TRUE,
  method.tau = "REML",
  method.random.ci = "HK",
  data = studies
)
```

Check the R package's explicit ad hoc HK option before treating the last call
as numerically equivalent.

## Classical Egger regression

```python
egger = result.egger_test()
```

corresponds to the classical `metafor` configuration:

```r
regtest(fit, model = "lm", predictor = "sei")
```

PyMetaAnalysis does not currently expose `metafor`'s default
`model="rma"` version. The returned `intercept` is the tested asymmetry
coefficient, while `limit_estimate` is the extrapolated effect as the standard
error tends to zero. Review the dedicated
[small-study-effects guide](small-study-effects.md) before treating similarly
named R functions as numerically interchangeable.

## Primary R references

- [`metafor::rma.uni`](https://wviechtb.github.io/metafor/reference/rma.uni.html)
- [`metafor::predict.rma`](https://wviechtb.github.io/metafor/reference/predict.rma.html)
- [`metafor::rma.mh`](https://wviechtb.github.io/metafor/reference/rma.mh.html)
- [`metafor::rma.peto`](https://wviechtb.github.io/metafor/reference/rma.peto.html)
- [`metafor::escalc`](https://wviechtb.github.io/metafor/reference/escalc.html)
- [`metafor::regtest`](https://wviechtb.github.io/metafor/reference/regtest.html)
- [`meta::metagen`](https://search.r-project.org/CRAN/refmans/meta/html/metagen.html)
- [`meta::metabin`](https://search.r-project.org/CRAN/refmans/meta/html/metabin.html)
- [`meta::metacont`](https://search.r-project.org/CRAN/refmans/meta/html/metacont.html)
- [`meta::metacor`](https://search.r-project.org/CRAN/refmans/meta/html/metacor.html)
