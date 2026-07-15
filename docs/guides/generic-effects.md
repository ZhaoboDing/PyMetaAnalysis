# Generic effects

Use `meta_analysis()` when each study already has an effect estimate and its
sampling variance. This is the generic inverse-variance workflow; it does not
calculate an outcome-specific effect size.

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

## Variance, not standard error

`variance=` must contain finite, strictly positive sampling variances. Convert
reported standard errors explicitly:

```python
data["vi"] = data["standard_error"] ** 2
```

Do not pass confidence-interval widths or study sample variances unless they
have first been converted to the sampling variance of the effect estimate.

## Missing values

The default `missing="raise"` rejects missing effects, variances, or labels. To
retain incomplete rows as structured exclusions:

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
