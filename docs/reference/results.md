# Result objects

Result objects are frozen dataclasses. Their scalar outputs, resolved method
settings, diagnostics, warnings, and row-level data remain available after the
model is fitted.

## `MetaAnalysisResult`

### Pooled estimates

| Attribute | Meaning |
| --- | --- |
| `estimate` | Pooled estimate on the model scale |
| `standard_error` | Standard error of the pooled estimate |
| `ci`, `ci_low`, `ci_high` | Confidence interval on the model scale |
| `prediction_interval` | Random-effects prediction interval, when available |
| `display_estimate` | Pooled estimate on the display scale |
| `display_ci` | Confidence interval on the display scale |
| `display_prediction_interval` | Prediction interval on the display scale |

For OR and RR, the model scale is logarithmic and the display scale is
exponentiated. Other implemented measures use an identity display transform.

### Heterogeneity

`tau2`, `q`, `q_df`, `q_pvalue`, `i2`, and `h2` expose the principal
heterogeneity outputs. `i2` is stored as a proportion from zero to one; summary
and plotting layers format it as a percentage.

### Methods and diagnostics

`result.method` is a `MethodConfig` containing the resolved settings actually
used. Outcome-specific values such as continuity-correction scopes are
available through `dict(result.method.options)`.

`result.diagnostics` records whether iterative estimation converged, how many
iterations it used, and whether tau-squared reached its zero boundary.
Recoverable statistical notes are stored in the immutable `warnings` tuple.

### Provenance

`result.provenance` is an immutable `AnalysisProvenance` record. It contains:

- the PyMetaAnalysis and provenance-schema versions;
- the analysis type and whether inputs came from a DataFrame or arrays;
- one `InputFieldProvenance` per resolved public input;
- DataFrame column mappings, when columns were selected by name;
- input row count and included/excluded row IDs;
- structured `TransformationRecord` entries.

Binary transformation records distinguish the continuity correction used for
individual effects from the independently configured correction used for
Mantel-Haenszel pooling. Continuous records identify the resolved MD or SMD
effect-size estimator. Provenance does not embed another copy of the original
DataFrame.

### Study tables

- `study_results` returns all input rows and fitted row-level outputs.
- `excluded_studies` returns rows with `included=False`.
- `to_dataframe()` is an explicit alias for the complete study table.

All three return defensive DataFrame copies. Common columns include `row_id`,
`study`, `effect`, `variance`, `included`, `exclusion_reason`, `weight`, and
`normalized_weight`; outcome-specific analyses retain their raw inputs and
intermediate effect-size calculations as additional columns.

### Summaries and plots

`summary()` returns a printable `MetaAnalysisSummary`; call its `to_dict()`
method for a machine-readable scalar summary.

`forest()` and `funnel()` return Matplotlib axes without calling `show()`.
Matplotlib is imported only when a plotting method is used.

### Sensitivity analysis

`leave_one_out()` returns a `LeaveOneOutResult` with the original result, one
refit per included study, warnings, and a defensive summary table.

`cumulative()` returns a `CumulativeMetaAnalysisResult`. Its `results` follow
the selected stable order and its `final` property is the last fit. Both
workflows reuse the stored model, pooling method, interval method, numerical
controls, and outcome-specific options.

### Methods text and reports

`method_details()` returns concise prose describing the resolved estimator,
confidence interval, heterogeneity statistics, corrections, missing-data
policy, numerical controls, and package version.

`report(include_studies=True)` returns a detached `ResultReport`:

```python
report.to_dict()
report.to_json(indent=2)
report.to_markdown()
```

The report contains model- and display-scale results, heterogeneity, full
method configuration, diagnostics, provenance, warnings, and—by default—the
study table. Pass `include_studies=False` for a compact payload. JSON is strict:
non-finite or unavailable values are serialized as `null` rather than `NaN`.

## Subgroup results

Supplying `subgroup=` returns `SubgroupMetaAnalysisResult`, which contains:

- `groups`, a read-only ordered mapping from subgroup label to
  `MetaAnalysisResult`;
- `overall`, the analysis of all included studies;
- `q_between`, `q_between_df`, and `q_between_pvalue` for the formal subgroup
  difference test;
- `i2_between`, the inconsistency statistic for subgroup differences;
- `method`, a `SubgroupMethodConfig` recording subgroup assumptions;
- the combined `study_results` and `excluded_studies` tables.

Its `summary().to_dict()` method returns nested group and overall summaries.
Its `forest()` method draws study rows, subgroup pooled estimates, the overall
estimate, prediction intervals when available, and the subgroup-difference
test.

Subgroup results also provide `leave_one_out()`, `cumulative()`,
`method_details()`, and `report()`. Their report contains each group, the
overall fit, the formal subgroup-differences test, its tau-squared strategy,
and the combined study table.
