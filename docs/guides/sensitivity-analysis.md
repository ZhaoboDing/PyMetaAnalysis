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

### Meta-regression leave-one-out analysis

`MetaRegressionResult.leave_one_out()` repeats the fitted regression while
omitting each included study:

```python
diagnostics = regression.leave_one_out()

print(diagnostics.table)
print(diagnostics.coefficients)
```

The returned `MetaRegressionLeaveOneOutResult` retains `original` and a
`results` tuple aligned with the omitted-study rows. Its model table reports
the refitted residual heterogeneity, global moderator test, condition number,
and warnings. Its long-form coefficient table reports every refitted term and
`estimate_change`, defined as the deleted estimate minus the full-model
estimate.

Deleting a study can make a categorical level disappear or otherwise make the
design matrix unidentifiable. That deletion is retained with
`refit_success=False`, an exception type and message, unavailable numeric
fields, and `None` in the matching `results` position. Other deletions continue
to be fitted. Use `diagnostics.failed` to inspect these rows. A failed deletion
does not by itself label that study as influential; it shows that the fitted
design depends on the study for identifiability.

The original model must have at least `k >= p + 2`, so deleting one study can
still leave more studies than coefficients. Each successful refit re-estimates
tau-squared and coefficient inference with the original model settings.

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

Meta-regression refits additionally reuse the intercept choice, inference
method, moderator order, and complete explicit categorical level definitions.

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

The current Meta-regression workflow reports exact deleted-model fits and
coefficient changes. It does not yet calculate Cook's distance, DFBETAS, or an
automatic influential-study flag.
