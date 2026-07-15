# Provenance and reporting

PyMetaAnalysis keeps the statistical result, its construction history, and its
presentation layer separate. This makes an analysis inspectable without
turning generated prose into an unreviewed publication claim.

## Inspect provenance

Every `MetaAnalysisResult` exposes an immutable provenance record:

```python
provenance = result.provenance

provenance.package_version
provenance.schema_version
provenance.analysis_type
provenance.data_source
provenance.column_mapping
provenance.included_rows
provenance.excluded_rows
provenance.transformations
```

`input_fields` contains one `InputFieldProvenance` per public input. Its
`source` distinguishes:

- `column`, with the resolved DataFrame column name;
- `dataframe_index`, when study labels came from the index;
- `array`, for explicit array-like values;
- `generated_row_number`, when array input omitted study labels.

`column_mapping` is a read-only convenience mapping containing only named
DataFrame columns. `to_dict()` returns a detached, JSON-compatible provenance
document.

## Understand transformation records

Each `TransformationRecord` has a stable name, resolved parameters, and the
original row IDs it affected.

Binary analyses record:

- the OR, RR, or RD effect-size transformation and its scales;
- continuity correction for individual-study effects;
- the RD zero-variance policy and affected boundary rows;
- non-informative relative-effect exclusions;
- a separate Mantel-Haenszel correction record when MH pooling is used.

This separation matters because exact Mantel-Haenszel pooling can use raw
tables even when a corrected individual effect is needed for display and
heterogeneity calculations.

Continuous analyses record the resolved MD or SMD estimator. SMD records the
pooled standardizer, exact Hedges correction, and LS variance convention.

Missing-input exclusions appear in `included_rows`, `excluded_rows`, and the
study table's `exclusion_reason`; they are not mislabeled as transformations.

## Generate Methods text

`method_details()` turns the fully resolved method configuration into concise
prose:

```python
print(result.method_details())
```

The text identifies the model, pooling estimator, tau-squared estimator,
confidence interval, heterogeneity statistics, prediction interval,
correction settings, missing-data policy, numerical controls, and package
version as applicable.

It is a starting point for a manuscript Methods section. Review and adapt it to
the analysis protocol, field conventions, and journal requirements.

## Build a structured report

`report()` returns a detached `ResultReport`:

```python
report = result.report()

payload = report.to_dict()
json_text = report.to_json()
markdown = report.to_markdown()
```

The payload contains:

- report schema and analysis identity;
- estimates on model and display scales;
- confidence and prediction intervals;
- heterogeneity statistics;
- the fully resolved `MethodConfig`;
- convergence diagnostics;
- provenance and warnings;
- row-level study results.

Report schema 1.1 records the resolved heterogeneity definition as
`heterogeneity.i2_method`.

The default study table makes the report auditable but can be large. Omit it
when only aggregate output is needed:

```python
compact = result.report(include_studies=False)
```

## Strict JSON behavior

`to_json()` emits standards-compliant JSON with UTF-8 text. Statistical values
that are unavailable or non-finite—such as heterogeneity p-values for a
single-study fit—become `null`. The serializer never emits non-standard `NaN`
or `Infinity` tokens.

Study labels such as pandas timestamps are converted to stable string
representations. Calling `to_dict()` always returns a defensive copy, so
editing an exported payload does not mutate the report or fitted result.

## Subgroup reports

`SubgroupMetaAnalysisResult` exposes the same `method_details()` and `report()`
methods:

```python
subgroup_report = subgroups.report()
```

Its payload includes each subgroup result, the overall fit, Q and I-squared for
subgroup differences, the tau-squared strategy, the named subgroup test, and
the combined study table. The Markdown representation adds a compact subgroup
summary table.

## Reproducibility boundary

Provenance records how PyMetaAnalysis interpreted the supplied inputs, but it
does not embed another copy of the original DataFrame, hash an external data
source, or capture the surrounding Python environment. A complete reproducible
research workflow should also version the input data, preprocessing code,
analysis script, environment, and protocol decisions.
