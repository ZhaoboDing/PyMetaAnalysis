# Input data and row decisions

Every analysis entry point accepts either DataFrame column names or
one-dimensional array-like values. This page describes rules shared by the
generic, binary, and continuous APIs.

## DataFrame columns

When `data=` is a pandas DataFrame, a string-valued input selects a column:

```python
result = ma.meta_analysis(
    data=studies,
    effect="log_effect",
    variance="sampling_variance",
)
```

If `study=` is omitted, the DataFrame index supplies study labels. To use a
column instead:

```python
result = ma.meta_analysis(
    studies,
    effect="log_effect",
    variance="sampling_variance",
    study="citation",
)
```

Column names must exist. Array-like inputs may be mixed with DataFrame columns,
but every array must contain exactly one value for every DataFrame row.

## Array-like inputs

Lists, tuples, pandas Series, and one-dimensional NumPy arrays are accepted:

```python
result = ma.meta_analysis(
    effect=[0.1, 0.3, -0.2],
    variance=[0.02, 0.04, 0.03],
    study=["A", "B", "C"],
)
```

All outcome inputs must be one-dimensional and have equal lengths. Without
`data=` or `study=`, labels are generated as integer row numbers beginning at
zero.

## Stable row identity

`row_id` is the zero-based input position. It is independent of the study
label and remains stable through exclusion, subgroup, leave-one-out, and
cumulative workflows. Use it when study labels are duplicated or not suitable
as identifiers.

The complete row table is available from:

```python
result.study_results
result.to_dataframe()
result.excluded_studies
```

These properties return defensive copies. Editing a returned DataFrame does
not mutate the fitted result.

## Missing values

The default `missing="raise"` stops when a required outcome value is missing.
`missing="drop"` retains the row in the result but excludes it from all model
calculations:

```python
result = ma.meta_analysis(
    studies,
    effect="effect",
    variance="variance",
    missing="drop",
)

result.excluded_studies[["row_id", "study", "exclusion_reason"]]
```

Dropped rows do not contribute to pooled estimates, Q, tau-squared, prediction
intervals, or weights. Missing subgroup labels are always rejected because
silently assigning or dropping them would change the subgroup definition.

## Outcome validation

| Input family | Required validation |
| --- | --- |
| Generic | finite effect; finite, strictly positive sampling variance |
| Binary | integer event counts and totals; positive totals; `0 <= events <= total` |
| Continuous | finite means/SDs; non-negative SDs; integer group sizes of at least 2 |

Binary and continuous APIs preserve their raw input columns in
`study_results`. Derived effects, variances, correction indicators, and
weights appear alongside them.

## Exclusion is visible

Rows excluded by a configured rule remain present with:

- `included=False`;
- a human-readable `exclusion_reason`;
- `NaN` fitted weights;
- their original `row_id` and study label.

Relative-effect double-zero/double-all exclusions and the RD
`rd_zero_variance="exclude"` policy follow this rule. See
[zero-event studies](zero-events.md) for the binary-specific behavior.

## Provenance boundary

`result.provenance` records where each public input came from, which row IDs
were included or excluded, and which configured transformations affected each
row. It does not capture upstream data cleaning. Version the source data and
preprocessing code separately.
