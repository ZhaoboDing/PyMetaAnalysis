# Generic effects

Use `meta_analysis()` when each study already has an effect estimate and its
sampling variance or standard error. This is the generic inverse-variance
workflow; it does not calculate an outcome-specific effect size.

The pooling, tau-squared, confidence-interval, prediction-interval, and
heterogeneity equations are specified under
[statistical methods](../methods/statistical-methods.md).

## DataFrame input

```python
import pandas as pd
import meta_analyze as ma

data = pd.DataFrame(
    {
        "label": ["A", "B", "C", "D"],
        "yi": [0.18, 0.31, -0.04, 0.22],
        "vi": [0.025, 0.041, 0.030, 0.036],
    }
)

result = ma.meta_analysis(
    data,
    effect="yi",
    variance="vi",
    study="label",
    model="random",
    tau2_method="PM",
)
```

String arguments select DataFrame columns. The study column is optional; the
DataFrame index is used when it is omitted.

## Array input

```python
result = ma.meta_analysis(
    effect=[0.18, 0.31, -0.04, 0.22],
    variance=[0.025, 0.041, 0.030, 0.036],
    study=["A", "B", "C", "D"],
    model="common",
)
```

All array-like arguments must be one-dimensional and have equal lengths.
Generated row labels start at zero when no study labels are supplied.

## Variance or standard error

Supply exactly one of `variance=` or `standard_error=`. Both must contain
finite, strictly positive values. A reported standard-error column can be used
directly:

```python
result = ma.meta_analysis(
    data,
    effect="yi",
    standard_error="standard_error",
)
```

PyMetaAnalysis squares standard errors internally, retains both uncertainty
columns in the study table, and records the conversion in provenance. Do not
pass confidence-interval widths or study sample standard deviations as
standard errors; they describe different quantities.

## Missing values

The default `missing="raise"` rejects missing effects or values in the selected
uncertainty input. To retain incomplete rows as structured exclusions:

```python
result = ma.meta_analysis(
    data,
    effect="yi",
    variance="vi",
    study="label",
    missing="drop",
)

result.excluded_studies[["study", "exclusion_reason"]]
```

Dropped rows do not enter the pooled estimate, heterogeneity statistics, or
weights.

## Subgroups

Pass a column name or array through `subgroup=`:

```python
subgroups = ma.meta_analysis(
    data,
    effect="yi",
    variance="vi",
    subgroup=["North", "North", "South", "South"],
)
```

See [result objects](../reference/results.md#subgroup-results) for the returned
structure.

For shared DataFrame/array, row identity, and exclusion rules, see
[input data and row decisions](input-data.md).
