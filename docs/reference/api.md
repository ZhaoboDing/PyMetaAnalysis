# Public API

The package is imported as `meta_analyze`:

```python
import meta_analyze as ma
```

## Analysis functions

### `meta_analysis()`

Fits common-effect or random-effects generic inverse-variance models from
study-level effects and strictly positive sampling variances.

Required inputs are `effect` and `variance`. Each may be a DataFrame column
name or a one-dimensional array-like. Important optional arguments include
`study`, `subgroup`, `model`, `tau2_method`, `ci_method`, `confidence_level`,
`missing`, `atol`, and `max_iter`.

### `meta_binary()`

Calculates and pools OR, RR, or RD from treatment/control event counts and
totals. Required inputs are `event_treat`, `n_treat`, `event_control`, and
`n_control`.

Important optional arguments include `measure`, `method`, `model`,
`tau2_method`, `ci_method`, `continuity_correction`, `correction_scope`,
`mh_continuity_correction`, `mh_correction_scope`, `study`, `subgroup`, and
`missing`.

### `meta_continuous()`

Calculates and pools MD or Hedges' g from treatment/control means, standard
deviations, and sample sizes.

Required inputs are `mean_treat`, `sd_treat`, `n_treat`, `mean_control`,
`sd_control`, and `n_control`. Important optional arguments include `measure`,
`smd_variance`, `model`, `tau2_method`, `ci_method`, `study`, `subgroup`, and
`missing`.

## Return types

Without `subgroup=`, every entry point returns `MetaAnalysisResult`. With
`subgroup=`, every entry point returns `SubgroupMetaAnalysisResult`.

See [result objects](results.md) for their stable inspection interface.

## Result workflows

Every `MetaAnalysisResult` provides:

- `leave_one_out()` for repeated omission refits;
- `cumulative()` for ordered accumulation refits;
- `method_details()` for Methods-style text;
- `report()` for dictionary, strict JSON, and Markdown output;
- `forest()` and `funnel()` for optional Matplotlib visualizations.

`SubgroupMetaAnalysisResult` provides the same sensitivity and reporting
workflows except for `funnel()`, which remains a single-analysis diagnostic.

## Configuration and diagnostics

- `MethodConfig` records the fully resolved model, pooling, tau-squared,
  confidence-interval, prediction-interval, missing-data, and outcome-specific
  options.
- `SubgroupMethodConfig` records subgroup model assumptions and the
  between-group test.
- `FitDiagnostics` records convergence, iteration count, and tau-squared
  boundary status.
- `HeterogeneityResult` contains Q, degrees of freedom, p-value, I-squared, and
  H-squared.
- `AnalysisProvenance` records package/schema versions, resolved input fields,
  input row decisions, and transformations.
- `InputFieldProvenance` describes whether an input came from a DataFrame
  column, index, generated label, or array.
- `TransformationRecord` stores one configured transformation, its parameters,
  and affected row IDs.
- `ResultReport` provides `to_dict()`, `to_json()`, and `to_markdown()`.

## Exceptions

All public domain exceptions derive from `MetaAnalysisError`:

| Exception | Meaning |
| --- | --- |
| `InvalidStudyDataError` | An input value, shape, or policy is invalid |
| `InsufficientStudiesError` | Too few included studies remain for the model |
| `ConvergenceError` | An iterative estimator did not converge |
| `UnsupportedMethodError` | The requested method or combination is unsupported |

Callers may catch a specific exception for targeted handling or
`MetaAnalysisError` at an application boundary.
