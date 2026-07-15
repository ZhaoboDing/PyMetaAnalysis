# Result objects

Result objects are frozen dataclasses. Scalar outputs, method settings,
diagnostics, warnings, provenance, and row-level data remain inspectable after
fitting. DataFrame-returning properties provide defensive copies.

## `MetaAnalysisResult`

### Analysis identity

| Attribute | Meaning |
| --- | --- |
| `k` | Number of included studies |
| `model` | Resolved `common` or `random` model |
| `measure` | `GENERIC`, `OR`, `RR`, `RD`, `MD`, or `SMD` |
| `effect_scale` | Scale used for model calculations |
| `display_scale` | Identity or exponentiating display transformation |

### Pooled estimates

| Attribute | Meaning |
| --- | --- |
| `estimate` | Pooled estimate on the model scale |
| `standard_error` | Standard error used by the selected mean CI |
| `ci_low`, `ci_high`, `ci` | Confidence interval on the model scale |
| `prediction_interval` | Random-effects prediction interval, when available |
| `display_estimate` | Pooled estimate on the display scale |
| `display_ci` | Display-scale confidence interval |
| `display_prediction_interval` | Display-scale prediction interval |

For OR and RR, `estimate`, `ci`, and `prediction_interval` are logarithmic;
their display counterparts are exponentiated ratios. Other measures use the
identity transformation.

### Heterogeneity

| Attribute | Meaning |
| --- | --- |
| `tau2` | Between-study variance; zero for common-effect fits |
| `q`, `q_df`, `q_pvalue` | Cochran Q statistic, degrees of freedom, and p-value |
| `i2` | I-squared as a proportion from 0 to 1 |
| `h2` | H-squared |
| `i2_method` | `q_based` or `tau2_typical_variance` |
| `heterogeneity` | The underlying `HeterogeneityResult` record |

The definition differs by model and is specified under
[statistical methods](../methods/statistical-methods.md#heterogeneity-and-inconsistency).

### Methods and diagnostics

`result.method` is a `MethodConfig` containing the settings actually used:

| Field | Meaning |
| --- | --- |
| `model` | Resolved model |
| `pooling_method` | `inverse_variance` or `mantel_haenszel` |
| `tau2_method` | `REML`, `PM`, `DL`, or `None` |
| `ci_method` | Resolved mean confidence-interval method |
| `confidence_level` | Fitted confidence level |
| `prediction_interval_method` | `HTS` or `None` |
| `missing` | Resolved missing-value policy |
| `atol`, `max_iter` | Numerical controls |
| `options` | Immutable outcome-specific key/value pairs |

Convert outcome options to a mapping with `dict(result.method.options)`.

`result.diagnostics` is a `FitDiagnostics` record:

| Field | Meaning |
| --- | --- |
| `converged` | Whether the requested fit converged |
| `iterations` | Iterations used; zero for closed-form/boundary fits |
| `tau2_at_boundary` | Whether tau-squared reached zero; `None` if not applicable |

Recoverable statistical and workflow notes are stored in the immutable
`warnings` tuple.

### Provenance

`result.provenance` is an `AnalysisProvenance` record containing:

- PyMetaAnalysis and provenance-schema versions;
- analysis type and DataFrame/array source kind;
- one `InputFieldProvenance` per public input;
- DataFrame `column_mapping` when columns were selected by name;
- total row count plus included and excluded row IDs;
- structured `TransformationRecord` values.

`result.source_data` returns a defensive copy of the supplied DataFrame, or
`None` for array-only input. Provenance deliberately does not embed the source
DataFrame in serialized output.

### Study tables

All study tables contain:

| Column | Meaning |
| --- | --- |
| `row_id` | Stable zero-based input position |
| `study` | Display label |
| `effect` | Study effect on the model scale |
| `variance`, `standard_error` | Study sampling uncertainty |
| `included` | Whether the row entered the fit |
| `exclusion_reason` | Reason for exclusion or `None` |
| `weight` | Raw fitted weight or `NaN` if excluded |
| `normalized_weight` | Weight divided by included-weight sum |

Accessors are:

```python
result.study_results
result.excluded_studies
result.to_dataframe()
```

#### Generic-specific columns

Generic tables contain only the common columns above.

#### Binary-specific columns

Binary tables additionally contain raw counts, `effect_display`,
`continuity_corrected`, `rd_zero_variance`, and
`mh_continuity_corrected`.

#### Continuous-specific columns

Continuous tables additionally contain raw group summaries, `effect_display`,
`pooled_sd`, `cohen_d`, and `smd_correction_factor`. MD rows leave SMD-only
intermediates unavailable.

### Summaries

`summary()` returns `MetaAnalysisSummary`. Its string form is concise human-
readable output. `to_dict()` contains analysis identity, pooled/display
estimates, interval and heterogeneity outputs, numerical controls, warnings,
and outcome-specific method options.

### Reports

`method_details()` produces Methods-style prose from resolved configuration.
`report(include_studies=True)` returns a detached `ResultReport`:

```python
report.to_dict()
report.to_json(indent=2)
report.to_markdown()
```

See [report schema](report-schema.md) for the complete serialized structure.

### Plotting

`forest()` and `funnel()` return Matplotlib axes without calling `show()`.
Matplotlib is imported only when a plot is requested. Parameters and display-
scale behavior are documented in [plotting](../guides/plotting.md).

## Sensitivity results

### `LeaveOneOutResult`

Contains:

- `original`, the fitted source result;
- `results`, one refit per omitted included study;
- `warnings`, workflow-level notes;
- `table`, `summary()`, and `to_dataframe()` defensive tabular views.

The table columns are:

```text
omitted_row_id, omitted_study, k, estimate, standard_error,
ci_low, ci_high, display_estimate, display_ci_low, display_ci_high,
tau2, q, q_df, q_pvalue, i2, h2, i2_method
```

### `CumulativeMetaAnalysisResult`

Contains `original`, ordered `results`, `warnings`, the defensive table views,
and `final`, the final all-included-studies fit.

The table columns are:

```text
step, added_row_ids, added_studies, order_value, k, estimate,
standard_error, ci_low, ci_high, display_estimate, display_ci_low,
display_ci_high, tau2, q, q_df, q_pvalue, i2, h2, i2_method
```

See [sensitivity analysis](../guides/sensitivity-analysis.md) for count and
ordering rules.

## Subgroup results

Supplying `subgroup=` returns a `SubgroupMetaAnalysisResult`.

| Attribute | Meaning |
| --- | --- |
| `groups` | Read-only ordered mapping from label to group result |
| `overall` | Analysis of all eligible studies |
| `q_between`, `q_between_df`, `q_between_pvalue` | Formal subgroup-difference test |
| `i2_between` | Inconsistency statistic for subgroup differences |
| `method` | `SubgroupMethodConfig` |
| `warnings` | Subgroup-workflow notes |
| `study_results` | Combined study table with a `subgroup` column |

`SubgroupMethodConfig` records `model`, `tau2_strategy`, `test_method`, and
`subgroup_missing`. Random-effects subgroup fits currently use an independent
tau-squared estimate within each group and another for the overall result.

`summary().to_dict()` returns nested group and overall summaries.
`method_details()` and `report()` include subgroup assumptions and the formal
test. `forest()` draws studies, subgroup subtotals, the overall result,
prediction intervals when available, and the test for subgroup differences.

### Subgroup sensitivity composites

`SubgroupLeaveOneOutResult` and `SubgroupCumulativeMetaAnalysisResult` expose
read-only `.groups` mappings plus `.overall`. Their `to_dataframe()` and
`summary()` methods combine group and overall paths with `scope` and `subgroup`
columns.

These paths do not calculate a new subgroup-differences test at every repeated
fit.
