# Public API

The distribution is named `PyMetaAnalysis` and the import package is
`meta_analyze`:

```python
import meta_analyze as ma
```

The public surface consists of four analysis functions, immutable result and
configuration classes, provenance/report classes, sensitivity result classes,
and domain exceptions exported from `meta_analyze`.

## Shared input conventions

- `data` is an optional pandas DataFrame.
- A string-valued outcome, study, subgroup, or order argument selects a
  DataFrame column.
- A non-string value must be a one-dimensional array-like.
- `study=None` uses the DataFrame index or generated integer row labels.
- `subgroup=None` returns `MetaAnalysisResult`; supplying it returns
  `SubgroupMetaAnalysisResult`.
- `missing="raise"` rejects missing outcome inputs; `"drop"` retains them as
  explicit exclusions.

See [input data and row decisions](../guides/input-data.md) for validation and
identity rules.

## Supported combinations

| Input/effect | Pooling | Models | CI methods |
| --- | --- | --- | --- |
| Generic | Inverse variance | common, random | normal; HK variants for random |
| Binary OR/RR | Mantel-Haenszel | common only | normal |
| Binary OR/RR | Inverse variance | common, random | normal; HK variants for random |
| Binary RD | Inverse variance | common, random | normal; HK variants for random |
| Continuous MD/SMD | Inverse variance | common, random | normal; HK variants for random |
| Generic Meta-regression | Inverse variance | common, mixed | normal; HK variants for mixed |

Canonical values should be used in saved analysis code. Model aliases such as
`fixed` and method aliases such as `IV`/`MH` are accepted for convenience but
resolved values are stored in `result.method`.

## `meta_analysis()`

```text
ma.meta_analysis(
    data=None,
    *,
    effect,
    variance=None,
    standard_error=None,
    study=None,
    subgroup=None,
    model="random",
    tau2_method="REML",
    ci_method="normal",
    confidence_level=0.95,
    missing="raise",
    atol=1e-10,
    max_iter=1000,
)
```

Fits generic study effects using inverse-variance pooling.

| Parameter | Description |
| --- | --- |
| `data` | Optional DataFrame used by string-valued input selectors |
| `effect` | Study effect column or array on a consistent model scale |
| `variance` | Finite, strictly positive sampling variances |
| `standard_error` | Finite, strictly positive standard errors, squared internally |
| `study` | Optional label column/array; defaults to index or row number |
| `subgroup` | Optional subgroup column/array |
| `model` | `"common"` or `"random"` |
| `tau2_method` | `"REML"`, `"PM"`, or `"DL"` for random effects |
| `ci_method` | `"normal"`, `"hartung_knapp"`, or `"hartung_knapp_adhoc"` |
| `confidence_level` | Number strictly between 0 and 1 |
| `missing` | `"raise"` or `"drop"` |
| `atol` | Strictly positive iterative-estimator tolerance |
| `max_iter` | Positive iterative-estimator iteration limit |

Supply exactly one of `variance` or `standard_error`. The selected argument
supports the same DataFrame-column and array-like conventions as `effect`.
Standard-error conversion is recorded in result provenance.

## `meta_regression()`

```text
ma.meta_regression(
    data=None,
    *,
    effect,
    variance=None,
    standard_error=None,
    moderators,
    categorical=None,
    study=None,
    model="mixed",
    tau2_method="REML",
    inference_method="normal",
    intercept=True,
    confidence_level=0.95,
    prediction_interval_method="default",
    missing="raise",
    atol=1e-10,
    max_iter=1000,
)
```

Fits generic study effects on study-level moderators.

| Parameter | Description |
| --- | --- |
| `moderators` | DataFrame column-name sequence, or name-to-column/array mapping |
| `categorical` | Moderator-to-ordered-level mapping; first level is reference |
| `model` | `"mixed"` (default) or `"common"`; random/fixed aliases are accepted |
| `tau2_method` | `"REML"`, `"PM"`, or `"DL"` for residual tau-squared |
| `inference_method` | `"normal"`, `"hartung_knapp"`, or `"hartung_knapp_adhoc"` |
| `intercept` | Include an intercept; no-intercept currently requires all-numeric moderators |
| `prediction_interval_method` | `"default"` or opt-in `"riley"` for mixed-effects true-effect prediction intervals |

`effect`, uncertainty, `study`, numerical controls, and missing-value policy
follow `meta_analysis()`. Missingness in any model field applies to the complete
row. The encoded design must have full column rank and `k > p`. Riley intervals
additionally require `k-p >= 2`.

The result is a dedicated `MetaRegressionResult`, not a pooled
`MetaAnalysisResult`. See the [Meta-regression guide](../guides/meta-regression.md)
for encoding, testing, prediction, and interpretation.

## `meta_binary()`

```text
ma.meta_binary(
    data=None,
    *,
    event_treat,
    n_treat,
    event_control,
    n_control,
    study=None,
    subgroup=None,
    measure="RR",
    method="MH",
    model="common",
    tau2_method="REML",
    ci_method="normal",
    confidence_level=0.95,
    continuity_correction=0.5,
    correction_scope="only_zero_studies",
    rd_zero_variance="correct",
    mh_continuity_correction=None,
    mh_correction_scope="only_zero_studies",
    missing="raise",
    atol=1e-10,
    max_iter=1000,
)
```

Calculates and pools two-group binary effects.

| Parameter | Description |
| --- | --- |
| `event_treat`, `event_control` | Integer event counts |
| `n_treat`, `n_control` | Strictly positive integer group totals |
| `measure` | `"OR"`, `"RR"`, or `"RD"` |
| `method` | `"MH"`/`"mantel_haenszel"` or `"IV"`/`"inverse_variance"` |
| `continuity_correction` | Non-negative correction for individual-study effects |
| `correction_scope` | `"only_zero_studies"`, `"if_any_zero"`, `"all_studies"`, or `"none"` |
| `rd_zero_variance` | `"correct"` or `"exclude"`; configurable only for RD |
| `mh_continuity_correction` | Separate non-negative MH pooling correction; `None` means zero |
| `mh_correction_scope` | Scope for the separate MH pooling correction |

`study`, `subgroup`, `tau2_method`, `ci_method`, `confidence_level`, `missing`,
`atol`, and `max_iter` have the meanings described for `meta_analysis()`.

MH supports common-effect OR/RR and normal intervals only. Random-effects
binary analysis and every RD analysis require inverse-variance pooling. Sparse
table behavior is specified in [zero-event studies](../guides/zero-events.md).

## `meta_continuous()`

```text
ma.meta_continuous(
    data=None,
    *,
    mean_treat,
    sd_treat,
    n_treat,
    mean_control,
    sd_control,
    n_control,
    study=None,
    subgroup=None,
    measure="MD",
    model="random",
    tau2_method="REML",
    ci_method="normal",
    confidence_level=0.95,
    smd_variance="LS",
    missing="raise",
    atol=1e-10,
    max_iter=1000,
)
```

Calculates and pools two independent groups described by means, sample SDs,
and sample sizes.

| Parameter | Description |
| --- | --- |
| `mean_treat`, `mean_control` | Finite group means |
| `sd_treat`, `sd_control` | Finite, non-negative sample standard deviations |
| `n_treat`, `n_control` | Integer sample sizes of at least 2 |
| `measure` | `"MD"` or `"SMD"` |
| `smd_variance` | `"LS"`; currently the only SMD variance convention |

The remaining shared parameters have the same meanings as in
`meta_analysis()`. MD uses an unpooled variance. SMD is exact-corrected Hedges'
g with the LS variance convention.

## Return types

The three pooling entry points return `MetaAnalysisResult` without `subgroup=`
and `SubgroupMetaAnalysisResult` with it. `meta_regression()` instead returns
`MetaRegressionResult` and does not accept `subgroup=`.

```python
result = ma.meta_analysis(...)
subgroups = ma.meta_analysis(..., subgroup="region")
```

The detailed attribute and table contracts are documented under
[result objects](results.md).

## Result methods

Every `MetaAnalysisResult` provides:

| Method | Return value |
| --- | --- |
| `summary()` | Printable `MetaAnalysisSummary` with `to_dict()` |
| `method_details()` | Methods-style resolved-method text |
| `report(include_studies=True)` | Detached `ResultReport` |
| `to_dataframe()` | Defensive copy of the study table |
| `leave_one_out()` | `LeaveOneOutResult` |
| `cumulative(...)` | `CumulativeMetaAnalysisResult` |
| `forest(...)` | Matplotlib `Axes` |
| `funnel(...)` | Matplotlib `Axes` |

`SubgroupMetaAnalysisResult` has the same methods except `funnel()`, and its
sensitivity methods return subgroup composite result classes.

`MetaRegressionResult` provides `summary()`, `method_details()`, `report()`,
`to_dataframe()`, `predict(new_data)`, `test_moderator(name)`,
`leave_one_out()`, `influence()`, `collinearity()`, and `contrast(...)`.
An eligible single-numeric-moderator fit additionally provides `bubble()`.
`influence()` returns exact deleted-model residual, Cook's-distance, and
DFBETAS diagnostics.
`collinearity()` returns term VIF, grouped moderator GVIF/GSIF, and weighted
condition diagnostics. `contrast(...)` evaluates one term-weight mapping, a
named mapping of mappings, or a labeled DataFrame against scalar or named
`rhs` null values. It does not provide a scalar pooled `estimate`, a forest
plot, or cumulative Meta-regression.

### `cumulative()` parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `order` | `None` | Column/array defining order; input order when omitted |
| `ascending` | `True` | Sort direction |
| `collapse` | `False` | Add tied order values in one step |

### `forest()` parameters

Single-result forest plots accept `ax`, `effect_label`, `pooled_label`,
`show_prediction_interval`, `show_weights`, `null_value`, and `log_scale`.
Subgroup forest plots omit `pooled_label`. See [plotting](../guides/plotting.md).

### `funnel()` parameters

Funnel plots accept `ax`, `effect_label`, `confidence_level`,
`show_pseudo_confidence_interval`, `warn_on_few_studies`, and `log_scale`.

## Configuration and diagnostics

### `MethodConfig`

Stores the fully resolved `model`, `pooling_method`, `tau2_method`, `ci_method`,
`confidence_level`, `prediction_interval_method`, `missing`, `atol`,
`max_iter`, and immutable outcome-specific `options` pairs.

### `SubgroupMethodConfig`

Stores `model`, `tau2_strategy`, `test_method`, and `subgroup_missing`.

### `MetaRegressionMethodConfig`

Stores resolved model, tau-squared and inference methods, confidence level,
intercept choice, moderator/term names, categorical references, prediction-
interval method, missing policy, and numerical controls.

### `FitDiagnostics`

Stores `converged`, `iterations`, and `tau2_at_boundary`.

### `MetaRegressionDiagnostics`

Adds design rank, condition number, and the residual inference scale to
convergence, iteration, and tau-squared boundary metadata.

### `HeterogeneityResult`

Stores `q`, `df`, `pvalue`, `i2`, `h2`, and `i2_method`.

## Provenance and report classes

- `AnalysisProvenance` stores versioned input, row, and transformation history.
- `InputFieldProvenance` describes a column, array, index, or generated label.
- `TransformationRecord` describes a configured transformation and affected
  row IDs.
- `ResultReport` provides `to_dict()`, `to_json()`, and `to_markdown()`.

See [report schema](report-schema.md) for the serialized contract.

## Exceptions

All library-specific exceptions derive from `MetaAnalysisError`:

| Exception | Meaning |
| --- | --- |
| `InvalidStudyDataError` | Input value, shape, count, variance, or policy is invalid |
| `InsufficientStudiesError` | Too few included studies remain for the requested model/workflow |
| `ConvergenceError` | An iterative tau-squared estimator cannot bracket or converge |
| `UnsupportedMethodError` | A method name or method/model/measure combination is unsupported |

Catch a specific exception when recovery is possible, or
`MetaAnalysisError` at an application boundary.
