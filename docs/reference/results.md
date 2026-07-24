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

## `MetaRegressionResult`

Meta-regression has no unique pooled effect, so this result deliberately omits
the scalar `estimate`, `ci`, and top-level `prediction_interval` attributes of
`MetaAnalysisResult`.

### Model and coefficient outputs

| Attribute | Meaning |
| --- | --- |
| `k`, `p`, `residual_df` | Included studies, fitted coefficients, and `k-p` |
| `model` | Resolved `common` or `mixed` model |
| `coefficients` | Defensive coefficient DataFrame |
| `coefficient_covariance` | Labeled defensive covariance DataFrame |
| `design_info` | Original moderators, categories, references, and terms |
| `design_matrix` | Included-study encoded matrix |
| `global_test` | Distribution-explicit joint test of all non-intercept terms |
| `test_moderator(name)` | Joint test of one original moderator's encoded terms |

Coefficient columns are:

```text
term, moderator, estimate, standard_error, statistic, statistic_name,
df, pvalue, ci_low, ci_high
```

`df` is unavailable for normal/z inference and equals `k-p` for Hartung-Knapp
t inference. `ModeratorTestResult` records `distribution`, `df_num`, optional
`df_denom`, and every tested term, so chi-squared and F results are not
conflated.

### Residual heterogeneity

| Attribute | Meaning |
| --- | --- |
| `tau2` | Residual between-study variance; zero for common models |
| `tau2_null` | Intercept-only tau-squared on the same rows, when applicable |
| `pseudo_r2`, `pseudo_r2_raw` | Truncated and raw reduction in tau-squared |
| `heterogeneity` | Residual QE, df, p-value, I-squared, H-squared, and definition |

Common models use `q_based_residual`; mixed models use
`tau2_typical_variance_residual`. Pseudo-R² is unavailable for common,
no-intercept, or zero-null-tau-squared fits.

### Row table and prediction

The row table contains the original moderators plus:

```text
fitted_value, residual, precision_weight, normalized_precision_weight,
leverage
```

Excluded rows retain identifiers and reasons while fitted diagnostics remain
unavailable. Precision weight is not a universal contribution percentage for
all regression coefficients.

`predict(new_data)` returns estimates, standard errors, and mean-effect
confidence intervals. Mixed models add `pi_low`/`pi_high` for a new true
effect. It reuses fitted categorical encoding and rejects unknown levels.

`bubble()` returns a Matplotlib axes for an intercept-containing fit with
exactly one numeric moderator. Bubble area represents normalized precision
weight; fitted bands reuse `predict()`. Other design shapes are rejected rather
than assigned an implicit marginalization rule.

`leave_one_out()` returns a `MetaRegressionLeaveOneOutResult` with exact
deleted-model fits, a model-level table, and a long-form coefficient table. See
[sensitivity analysis](../guides/sensitivity-analysis.md) for failure and
minimum-study rules.

`influence()` returns a `MetaRegressionInfluenceResult` with externally
standardized deleted residuals, Cook's distances, DFBETAS, explicit screening
thresholds, and the same exact deletion fits. It does not mutate or filter the
original result.

`collinearity()` returns a `MetaRegressionCollinearityResult` with
`metafor`-compatible VIF/GVIF and weighted, column-scaled condition
diagnostics. It inspects the fitted design without changing the coefficients,
terms, or included studies.

`contrast(...)` returns a `MetaRegressionContrastResult` with individual and
joint inference for explicitly supplied linear hypotheses. It uses the fitted
coefficient covariance and inference method without refitting the model.

`summary()`, `method_details()`, `report()`, provenance, warnings, and defensive
copy semantics follow the same audit principles as `MetaAnalysisResult`.

## Diagnostic and sensitivity results

### `MetaRegressionContrastResult`

Contains `original`, a `LinearContrastTestResult` in `joint_test`,
`pvalue_adjustment="none"`, `warnings`, and defensive `table` and
`contrast_matrix` DataFrames. `len(result)` is the number of contrast rows;
`summary()` and `to_dataframe()` return the individual table.

`contrast_matrix` has one named row per contrast and one column per fitted term,
including the intercept when present. The inference table columns are:

```text
contrast, estimate, rhs, estimate_minus_rhs, standard_error, statistic,
statistic_name, distribution, df, pvalue, ci_low, ci_high,
confidence_level, pvalue_adjustment
```

The confidence interval is for the estimated linear combination `C beta`; the
test statistic is for `C beta - rhs`. Normal inference produces z statistics
and a joint chi-squared test. Both Hartung-Knapp choices produce t statistics
with `k-p` degrees of freedom and a joint F test.

`LinearContrastTestResult` records `contrasts`, `statistic`,
`statistic_name`, `distribution`, `df_num`, optional `df_denom`, and `pvalue`,
with a detached `to_dict()` view. Multi-row matrices must have full row rank.
Individual p-values are unadjusted; the result warns rather than silently
selecting a multiplicity procedure.

### `MetaRegressionCollinearityResult`

Contains `original`, `raw_condition_number`,
`weighted_scaled_condition_number`, `condition_index_reference`,
`variance_proportion_reference`, `warnings`, and four defensive DataFrames.

`term_vif` has one row per encoded non-intercept term:

```text
term, moderator, vif, sif
```

`moderator_gvif` groups the terms belonging to each original moderator:

```text
moderator, kind, terms, term_count, gvif, gsif
```

`condition_indices` contains:

```text
dimension, singular_value, eigenvalue, condition_index,
high_condition_index, high_variance_term_count, concerning
```

`variance_proportions` is a long-form table with:

```text
dimension, condition_index, term, moderator, variance_proportion,
high_variance_proportion
```

`concerning_dimensions` filters the condition table to dimensions with a
condition index above 30 and variance proportions above 0.5 for at least two
terms. These are fixed heuristic references, not hypothesis tests or automatic
variable-selection rules. VIF/GVIF has no automatic cutoff.

### `MetaRegressionInfluenceResult`

Contains `original`, the underlying `leave_one_out` workflow, its
omission-aligned `results`, workflow `warnings`, and defensive `table`,
`dfbetas`, `failed`, and `flagged` DataFrames. The scalar
`studentized_residual_reference`, `cook_distance_threshold`, and
`dfbetas_threshold` attributes expose the applied screening references.

The case-level table columns are:

```text
omitted_row_id, omitted_study, refit_success, error_type, error_message,
deleted_residual, deleted_residual_se, externally_standardized_residual,
studentized_residual_reference, potential_outlier, cook_distance,
cook_distance_threshold, cook_distance_flag, max_abs_dfbetas,
dfbetas_threshold, dfbetas_flag, potentially_influential, flagged,
leverage, normalized_precision_weight
```

The long-form DFBETAS table repeats omission identifiers, success state, term,
and moderator, followed by:

```text
dfbeta, standard_error_reference, dfbetas, threshold, exceeds_threshold
```

`dfbeta` is the full-model coefficient minus the deleted-model coefficient.
Failed refits retain rows with unavailable numeric diagnostics. Screening
flags are heuristic review aids and never remove studies.

### `MetaRegressionLeaveOneOutResult`

Contains `original`, an omission-aligned `results` tuple, workflow `warnings`,
and defensive `table`, `coefficients`, and `failed` DataFrames. A `None` result
and `refit_success=False` retain a deletion whose reduced design could not be
estimated.

The model table columns are:

```text
omitted_row_id, omitted_study, refit_success, error_type, error_message,
k, tau2, residual_q, residual_q_df, residual_q_pvalue, residual_i2,
residual_h2, global_statistic, global_statistic_name, global_df_num,
global_df_denom, global_pvalue, condition_number, refit_warnings
```

The long-form coefficient table repeats omission identifiers and success state
for each original term, followed by:

```text
term, moderator, estimate, estimate_change, standard_error, statistic,
statistic_name, df, pvalue, ci_low, ci_high
```

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
