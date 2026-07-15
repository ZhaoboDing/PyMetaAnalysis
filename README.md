# PyMetaAnalysis

PyMetaAnalysis is an early-stage, pandas-first Python library for auditable
meta-analysis workflows.

The library currently supports generic inverse-variance meta-analysis from
study-level effects and sampling variances:

```python
import meta_analyze as ma

result = ma.meta_analysis(
    data=studies,
    effect="effect",
    variance="variance",
    study="study",
    model="random",
    tau2_method="REML",
)

print(result.summary())
```

Two-group binary outcomes can be analyzed directly from a DataFrame. The
default is a Mantel-Haenszel common-effect risk ratio:

```python
result = ma.meta_binary(
    data=studies,
    event_treat="event_treat",
    n_treat="n_treat",
    event_control="event_control",
    n_control="n_control",
    study="study",
    measure="RR",          # "OR", "RR", or "RD"
    method="MH",           # "MH" or "IV"
    model="common",
)

print(result.summary())
```

For random-effects binary meta-analysis, use inverse-variance pooling and
select the between-study variance estimator explicitly if desired:

```python
result = ma.meta_binary(
    data=studies,
    event_treat="event_treat",
    n_treat="n_treat",
    event_control="event_control",
    n_control="n_control",
    measure="OR",
    method="IV",
    model="random",
    tau2_method="REML",
)
```

Two-group continuous outcomes accept the same DataFrame-or-array style. The
default measure is the raw mean difference; use `measure="SMD"` for Hedges'
adjusted g:

```python
result = ma.meta_continuous(
    data=studies,
    mean_treat="mean_treat",
    sd_treat="sd_treat",
    n_treat="n_treat",
    mean_control="mean_control",
    sd_control="sd_control",
    n_control="n_control",
    study="study",
    measure="SMD",        # "MD" or "SMD"
    model="random",
    tau2_method="REML",
)
```

Effects are defined as treatment minus control, so positive MD/SMD values
indicate larger outcomes in the treatment group. MD uses the unpooled sampling
variance. SMD uses the pooled within-study SD, the exact gamma-function Hedges
correction, and the `metafor` `vtype="LS"` sampling variance convention. The
per-study pooled SD, Cohen's d, correction factor, final effect, variance, and
weights remain available in `result.study_results`.

## Subgroup analysis

All three analysis entry points accept a DataFrame column or one-dimensional
array-like through `subgroup=`:

```python
subgroups = ma.meta_binary(
    data=studies,
    event_treat="event_treat",
    n_treat="n_treat",
    event_control="event_control",
    n_control="n_control",
    study="study",
    subgroup="region",
    measure="RR",
    method="MH",
    model="common",
)

print(subgroups.summary())
ax = subgroups.forest()
```

Supplying `subgroup=` returns a `SubgroupMetaAnalysisResult` containing an
ordered mapping of subgroup labels to `MetaAnalysisResult` objects, the overall
analysis, and a formal test for subgroup differences. The test follows the
RevMan formulation: subgroup summary effects are weighted by the inverse square
of their pooled standard errors and compared with a chi-squared Q statistic.
It is not based on comparing whether separate subgroup p-values are
statistically significant. See Cochrane's
[Statistical Methods Programmed in RevMan](https://training.cochrane.org/handbook/current/statistical-methods-revman5)
for the defining equations.

For random-effects analyses, tau-squared is currently estimated independently
within each subgroup and separately for the overall analysis. This assumption
is recorded as `result.method.tau2_strategy == "independent"`. Each random-
effects subgroup therefore needs at least two included studies. Missing
subgroup labels are rejected explicitly rather than silently assigned or
dropped. Outcome rows excluded by the selected missing-data or zero-event
policy remain visible with their subgroup label in `result.study_results`.

The subgroup forest plot shows study estimates, subgroup subtotals, the overall
estimate, optional prediction intervals, overall study weights, and the formal
test for subgroup differences. It follows the same optional-Matplotlib and
display-scale behavior as ordinary forest plots.

## Sensitivity and cumulative analysis

Every fitted result can be refitted while omitting one included study at a
time. The returned table reports the omitted row and study labels together with
the pooled estimate, confidence interval, tau-squared, Q, I-squared, and
H-squared from each refit:

```python
influence = result.leave_one_out()
print(influence.to_dataframe())
```

The original model, pooling method, confidence-interval method, continuity
corrections, tau-squared estimator, confidence level, and numerical controls
are reused for every fit. Originally excluded studies are not treated as
leave-one-out candidates. A common-effect leave-one-out analysis needs at least
two included studies; a random-effects analysis needs at least three so that
every refit retains the two studies required to estimate tau-squared.

Cumulative analysis adds included studies in the original input order by
default. A DataFrame column name or one-dimensional array-like can define a
different stable order:

```python
cumulative = result.cumulative(order="publication_year")
print(cumulative.to_dataframe())
```

Use `ascending=False` to reverse the order. With an explicit order,
`collapse=True` adds studies sharing the same order value simultaneously.
Missing order values are rejected for included studies but ignored for rows
that the original analysis already excluded. Random-effects cumulative
analysis starts with the first estimable `k=2` prefix and records this boundary
in `cumulative.warnings`; it never silently substitutes a common-effect model
for the single-study prefix.

For a `SubgroupMetaAnalysisResult`, the same methods return dedicated composite
objects with `.groups` and `.overall` sensitivity results. These paths describe
the accumulation or omission behavior within each fitted subgroup and within
the overall model; they do not reinterpret different-length paths as a new
sequence of subgroup-differences tests.

These workflows follow the repeated-refitting semantics documented for
[`metafor::leave1out()`](https://wviechtb.github.io/metafor/reference/leave1out.html)
and [`metafor::cumul()`](https://wviechtb.github.io/metafor/reference/cumul.html).

## Forest plots

Plotting support is optional:

```console
pip install "PyMetaAnalysis[plot]"
```

Every fitted result can produce a Matplotlib forest plot:

```python
ax = result.forest(
    effect_label="Risk ratio",
    show_prediction_interval=True,
)
```

The plot includes included-study confidence intervals, weight-scaled markers,
the pooled confidence-interval diamond, and an optional prediction interval.
OR and RR are shown on an exponentiated logarithmic axis with a null value of
1; difference measures use a linear axis with a null value of 0. The method
returns an `Axes` and does not call `show()`.

## Funnel plots

Fitted results also provide a standard-error funnel plot:

```python
ax = result.funnel()
```

The plot uses equal-sized study markers, an inverted standard-error axis, the
model's pooled estimate as its reference line, and a pseudo confidence region
that does not include tau-squared. Ratio measures are calculated on the log
model scale and displayed as ratios on a logarithmic x-axis. A warning is
emitted when fewer than 10 studies are available because asymmetry is difficult
to assess reliably with so little information. Funnel-plot asymmetry indicates
possible small-study effects and must not be treated as proof of publication
bias.

OR and RR are modeled on the log scale. `result.estimate` and `result.ci`
therefore remain on that auditable model scale, while
`result.display_estimate` and `result.display_ci` return exponentiated values.

By default, a correction of 0.5 is applied to every cell of an individual
study table that contains at least one zero cell. Double-zero and double-all
studies are excluded from relative-effect analyses and remain visible in
`result.study_results` with an exclusion reason. Mantel-Haenszel pooling uses
uncorrected tables by default; `mh_continuity_correction` is a separate,
explicit option.

The public API is under active development and may change before version 0.1.

## License

MIT
