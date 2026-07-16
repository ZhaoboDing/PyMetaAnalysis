# Getting started

This guide fits a random-effects generic inverse-variance meta-analysis from a
DataFrame. It also introduces the result object used by every analysis entry
point.

## 1. Install the library

```console
python -m pip install PyMetaAnalysis
```

For a source checkout, install the project in editable mode:

```console
python -m pip install -e ".[test,plot]"
```

See [installation](installation.md) for all extras and development checks.

## 2. Prepare a DataFrame

The generic API expects one effect estimate and either a strictly positive
sampling variance or standard error per study. Supply `variance=` or
`standard_error=`; standard errors are squared internally and recorded in
provenance.

```python
import pandas as pd

studies = pd.DataFrame(
    {
        "effect": [0.12, 0.35, -0.08, 0.21],
        "variance": [0.04, 0.06, 0.03, 0.05],
    },
    index=pd.Index(["Trial A", "Trial B", "Trial C", "Trial D"], name="study"),
)
```

When `study=` is omitted, a DataFrame's index becomes the display label. Pass a
column name or array explicitly when the index is not the desired label.

## 3. Fit the model

```python
import meta_analyze as ma

result = ma.meta_analysis(
    studies,
    effect="effect",
    variance="variance",
    model="random",
    tau2_method="REML",
)

print(result.summary())
```

The random-effects defaults are REML for between-study variance and a normal
confidence interval for the pooled mean. Defaults are fully resolved in
`result.method`; they do not need to be inferred from how the function was
called.

## 4. Inspect the result

Scalar model results are available as named attributes:

```python
result.estimate
result.ci
result.standard_error
result.tau2
result.q
result.i2       # a proportion from 0 to 1
result.h2
result.i2_method
```

The machine-readable summary and study table are suitable for reporting and
further processing:

```python
summary = result.summary().to_dict()

study_table = result.study_results[
    ["study", "effect", "variance", "included", "normalized_weight"]
]
```

`study_results`, `excluded_studies`, and `to_dataframe()` return defensive
copies. Editing them does not mutate the fitted result.

## 5. Inspect provenance and create a report

Every result records how its inputs were resolved and which rows entered the
fit:

```python
result.provenance.column_mapping
result.provenance.included_rows
result.provenance.excluded_rows
result.provenance.transformations
```

Use `method_details()` for a concise Methods-style description. Use `report()`
when a complete structured or human-readable export is needed:

```python
methods_text = result.method_details()

report = result.report()
payload = report.to_dict()
json_text = report.to_json()
markdown = report.to_markdown()
```

See [provenance and reporting](guides/provenance-reporting.md) for the report
schema, strict JSON behavior, and subgroup reports.

## 6. Understand model and display scales

Generic effects, MD, SMD, and RD use the identity scale. OR and RR are modeled
on a log scale, so their audit-friendly numeric attributes remain logarithmic.
Use display properties for ratios:

```python
result.display_estimate
result.display_ci
result.display_prediction_interval
```

The [binary-outcome guide](guides/binary-outcomes.md) includes a complete ratio
example.

## 7. Check sensitivity

Leave-one-out analysis refits the stored model once for every included study:

```python
influence = result.leave_one_out()
print(influence.to_dataframe())
```

Cumulative analysis adds studies in input order by default:

```python
cumulative = result.cumulative()
print(cumulative.to_dataframe())
```

The [sensitivity-analysis guide](guides/sensitivity-analysis.md) explains
minimum study counts, ordering, tied order values, and subgroup behavior.

## 8. Plot without implicit display

After installing the `plot` extra:

```python
ax = result.forest(show_prediction_interval=True)
ax = result.funnel()
```

Both methods return a Matplotlib `Axes` and never call `show()`. This keeps them
usable in notebooks, scripts, tests, and composed figures.

See [plotting](guides/plotting.md) for every parameter, display-scale rules,
axes composition, and interpretation cautions.

## Next steps

- Review [choosing methods](guides/method-selection.md) before changing model,
  tau-squared estimator, or confidence-interval method.
- Use [generic effects](guides/generic-effects.md) for array input and missing
  value policies.
- Read [input data and row decisions](guides/input-data.md) before relying on
  exclusions or provenance in a larger pipeline.
- Use an outcome-specific entry point when raw group summaries are available.
- Use [sensitivity analysis](guides/sensitivity-analysis.md) to assess how the
  pooled estimate changes across repeated refits.
- Consult [statistical methods](methods/statistical-methods.md) for implemented
  equations and [scope and limitations](limitations.md) before selecting the
  library for a review.
