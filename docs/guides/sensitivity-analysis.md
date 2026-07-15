# Sensitivity analysis

Sensitivity workflows repeatedly refit the stored model. They help reveal how
the pooled estimate and heterogeneity depend on individual studies or on the
order in which evidence accumulated. They do not automatically diagnose why a
study is influential or replace protocol-specified sensitivity analyses.

## Leave-one-out analysis

Call `leave_one_out()` on any fitted `MetaAnalysisResult`:

```python
influence = result.leave_one_out()

print(influence.to_dataframe())
```

The returned `LeaveOneOutResult` contains:

- `original`, the fitted result supplied to the workflow;
- `results`, one immutable refit per omitted included study;
- `table`, `summary()`, and `to_dataframe()`, which return defensive copies;
- `warnings`, for workflow-level notes.

The table identifies `omitted_row_id` and `omitted_study` and reports each
refit's estimate, standard error, confidence interval, tau-squared, Q,
I-squared, and H-squared.

Originally excluded rows are never omission candidates. Common-effect analysis
requires at least two included studies so each refit retains one. Random-effects
analysis requires at least three so every refit retains the two studies needed
to estimate tau-squared.

## Cumulative analysis

By default, cumulative analysis follows input order:

```python
cumulative = result.cumulative()
print(cumulative.to_dataframe())
```

Supply a DataFrame column name or one-dimensional array-like to define another
stable order:

```python
cumulative = result.cumulative(
    order="publication_year",
    ascending=True,
)
```

`CumulativeMetaAnalysisResult.results` contains the estimable prefix fits, and
`final` returns the last fit. Its table records the rows and study labels added
at each step together with the same principal statistics as leave-one-out.

When multiple studies have the same order value, `collapse=True` adds the tied
studies in one step:

```python
cumulative = result.cumulative(
    order="publication_year",
    collapse=True,
)
```

Random-effects cumulative analysis starts at the first two-study prefix. The
unavailable single-study prefix is not silently fitted as a common-effect
model, and the boundary is recorded in `cumulative.warnings`.

## Reused analysis settings

Every refit reuses the original result's:

- common- or random-effects model;
- inverse-variance or Mantel-Haenszel pooling method;
- tau-squared and confidence-interval methods;
- confidence level, absolute tolerance, and iteration limit;
- missing-data policy;
- binary continuity corrections and RD zero-variance policy, or the continuous
  effect-size convention.

Derived results retain provenance that maps their local calculations back to
the original `row_id` values.

Repeated-fit tables include `i2_method` so a random-effects path cannot be
mistaken for a Q-based inconsistency series.

## Subgroup results

The same methods are available on `SubgroupMetaAnalysisResult`:

```python
subgroup_influence = subgroups.leave_one_out()
subgroup_cumulative = subgroups.cumulative(order="publication_year")
```

These return composite objects with `.groups` and `.overall` workflows. Each
subgroup is refitted independently. The paths do not reinterpret different
numbers of subgroup steps as a new sequence of subgroup-differences tests.

## Interpretation

Large changes after omitting a study warrant inspection of its population,
design, outcome definition, risk of bias, and numerical leverage. They are not
by themselves a reason to exclude the study. Likewise, a cumulative trend can
describe the historical evidence path but does not remove time-related changes
in methods, populations, or publication processes.
