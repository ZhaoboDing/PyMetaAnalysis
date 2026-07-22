# Report and provenance schemas

`result.report()` returns a detached `ResultReport`. Its dictionary and JSON
forms use report schema version `1.2`. Provenance nested within the report has
its own schema version, currently `1.0`.

## Export methods

```python
report = result.report(include_studies=True)

payload = report.to_dict()
json_text = report.to_json(indent=2)
markdown = report.to_markdown()
```

`to_dict()` returns a defensive copy. JSON uses standard values only:
unavailable or non-finite statistical numbers become `null`, and timestamps or
other non-JSON study labels become stable strings.

## Single-analysis report

Top-level keys in schema 1.2 for a single pooled analysis are:

| Key | Meaning |
| --- | --- |
| `schema_version` | Report schema identifier (`"1.2"`) |
| `report_type` | `"meta_analysis"` |
| `analysis` | Analysis identity and study counts |
| `results` | Pooled results on model and display scales |
| `heterogeneity` | Q, I-squared, H-squared, and definition |
| `method` | Fully resolved `MethodConfig` |
| `diagnostics` | Convergence metadata |
| `provenance` | Versioned input/transformation history |
| `warnings` | Recoverable analysis notes |
| `studies` | Row records when `include_studies=True` |
| `method_details` | Generated Methods-style text |

### `analysis`

```text
model, measure, effect_scale, display_scale,
included_studies, total_rows
```

### `results`

```text
estimate, standard_error, ci, prediction_interval,
display_estimate, display_ci, display_prediction_interval, tau2
```

Intervals are two-element arrays or `null` when unavailable.

### `heterogeneity`

```text
q, df, pvalue, i2, h2, i2_method
```

`i2` is a proportion. `i2_method` is `q_based` or
`tau2_typical_variance`.

### `method`

```text
model, pooling_method, tau2_method, ci_method, confidence_level,
prediction_interval_method, missing, atol, max_iter, options
```

`options` is an object containing resolved outcome-specific choices. Binary
examples include continuity corrections and RD boundary policy; continuous
SMD options identify its estimator, standardizer, and variance convention.

### `diagnostics`

```text
converged, iterations, tau2_at_boundary
```

### `studies`

Each record corresponds to one input row and follows the columns documented
for the outcome-specific [study table](results.md#study-tables). Excluded rows
remain present. Omit this potentially large field with:

```python
compact = result.report(include_studies=False)
```

## Provenance schema 1.0

The nested `provenance` object contains:

| Key | Meaning |
| --- | --- |
| `package_version` | Version that fitted the result |
| `schema_version` | Provenance schema identifier (`"1.0"`) |
| `analysis_type` | `generic`, `binary`, `continuous`, or `meta_regression` |
| `data_source` | `pandas_dataframe`, `array_like`, or `derived_subset` |
| `input_fields` | Ordered input source records |
| `column_mapping` | Public input roles mapped to DataFrame columns |
| `row_count` | Number of rows represented by this provenance record |
| `included_rows`, `excluded_rows` | Original integer row IDs |
| `transformations` | Configured transformation records |

Each input field has `role`, `source`, and nullable `column`. Supported source
values are `column`, `array`, `dataframe_index`, and
`generated_row_number`.

Each transformation has:

```text
name, parameters, affected_rows
```

Parameters contain resolved settings, not inferred defaults. A transformation
with an empty `affected_rows` list still records that the policy was configured.

## Subgroup report

For subgroup results, `report_type` is `"subgroup_meta_analysis"`. Top-level
keys are:

```text
schema_version, report_type, overall, groups, subgroup_differences,
method_details, warnings, studies
```

`overall` and every item in `groups` use the single-analysis nested payload.
Each group item has `label` and `result`. `subgroup_differences` contains:

```text
q, df, pvalue, i2, tau2_strategy, test_method, subgroup_missing
```

The optional top-level `studies` table contains the combined subgroup column.

## Meta-regression report

For Meta-regression results, `report_type` is `"meta_regression"`. Top-level
keys are:

```text
schema_version, report_type, analysis, coefficients,
coefficient_covariance, residual_heterogeneity, global_moderator_test,
design, method, diagnostics, provenance, warnings, method_details, studies
```

`coefficients` follows the table documented under
[result objects](results.md#metaregressionresult). The covariance object
contains its ordered `terms` and a square `values` matrix.

`residual_heterogeneity` contains:

```text
qe, df, pvalue, i2, h2, i2_method,
tau2, tau2_null, pseudo_r2, pseudo_r2_raw
```

`global_moderator_test` records the tested terms, statistic name,
distribution, numerator and optional denominator degrees of freedom, and
p-value. `design` records the intercept, ordered encoded terms, original
moderators, categorical levels, and references.

Meta-regression diagnostics add `rank`, `condition_number`, and
`residual_scale`. Its optional study records include original moderators,
fitted values, residuals, precision weights, and leverage.

## Markdown representation

`to_markdown()` is intended for review and reporting, not as a lossless
serialization format. It contains results, generated method details,
provenance/diagnostic highlights, exclusions, and warnings. Use dictionary or
JSON output for programmatic consumers.

## Schema compatibility

The package is early-stage, so consumers should inspect `schema_version`,
tolerate unknown additive fields, and pin the package version used in a
production pipeline. A schema change is recorded separately from the package
version so parsers can make an explicit compatibility decision.
