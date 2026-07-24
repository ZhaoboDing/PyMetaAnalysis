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
| Continuous group summaries | `meta_continuous()` | `escalc()` then `rma.uni()` | `metacont()` |
| Subgroups | `subgroup=` on a high-level call | separate fits or a moderator model | `subgroup=` |
| Leave-one-out | `result.leave_one_out()` | `leave1out()` for supported fits | `metainf()` |
| Meta-regression influence | `regression.influence()` | `influence()`, `rstudent()`, `cooks.distance()`, `dfbetas()` | — |
| Meta-regression collinearity | `regression.collinearity()` | `vif()` plus weighted design diagnostics | — |
| Meta-regression linear contrasts | `regression.contrast(...)` | `anova(..., X=..., rhs=...)` | — |
| Cumulative analysis | `result.cumulative()` | `cumul()` | `metacum()` |

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

For OR and RR, `estimate` and `ci` remain on the log model scale.
`display_estimate` and `display_ci` provide exponentiated ratios. This is
similar to choosing transformed or untransformed printing in R, but both scales
remain explicit attributes in Python.

## Models, pooling, and heterogeneity

| PyMetaAnalysis | `metafor` analogue | `meta` analogue | Notes |
| --- | --- | --- | --- |
| `model="common"` | `rma.uni(..., method="EE")` | `common=TRUE, random=FALSE` | Inverse-variance common-effect fit |
| `model="random"` | random-effects `rma.uni()` | `random=TRUE` | Requires a tau-squared policy |
| `method="IV"` | inverse-variance weighting | `method="Inverse"` | Binary API only; generic and continuous fits are IV |
| `method="MH"` | `rma.mh()` | `method="MH"` | Common-effect OR/RR only |
| `tau2_method="REML"` | `method="REML"` | `method.tau="REML"` | PyMetaAnalysis random-effects default |
| `tau2_method="PM"` | `method="PM"` | `method.tau="PM"` | Paule-Mandel |
| `tau2_method="DL"` | `method="DL"` | `method.tau="DL"` | DerSimonian-Laird |

The same label does not guarantee identical optimizer tolerances, boundary
handling, or heterogeneity definitions. In particular, PyMetaAnalysis records
`i2_method`; random-effects inverse-variance results use the documented
tau-squared/typical-variance definition, while common-effect and MH results use
the Q-based definition.

## Confidence and prediction intervals

| PyMetaAnalysis `ci_method` | `metafor` | R `meta` | Behavior |
| --- | --- | --- | --- |
| `"normal"` | default `test="z"` | `method.random.ci="classic"` | Normal mean interval |
| `"hartung_knapp"` | `test="knha"` | `method.random.ci="HK"` | Unmodified HK variance and t quantile |
| `"hartung_knapp_adhoc"` | `test="adhoc"` | HK plus an explicitly selected ad hoc correction | HK variance cannot fall below the classic variance |

Eligible random-effects fits include the documented HTS prediction interval.
R packages offer additional prediction-interval choices, so matching the mean
interval does not by itself guarantee a matching prediction interval.

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

## Primary R references

- [`metafor::rma.uni`](https://wviechtb.github.io/metafor/reference/rma.uni.html)
- [`metafor::predict.rma`](https://wviechtb.github.io/metafor/reference/predict.rma.html)
- [`metafor::rma.mh`](https://wviechtb.github.io/metafor/reference/rma.mh.html)
- [`metafor::escalc`](https://wviechtb.github.io/metafor/reference/escalc.html)
- [`meta::metagen`](https://search.r-project.org/CRAN/refmans/meta/html/metagen.html)
- [`meta::metabin`](https://search.r-project.org/CRAN/refmans/meta/html/metabin.html)
- [`meta::metacont`](https://search.r-project.org/CRAN/refmans/meta/html/metacont.html)
