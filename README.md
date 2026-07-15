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
