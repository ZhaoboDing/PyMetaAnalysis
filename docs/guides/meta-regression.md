# Meta-regression

Meta-regression relates study effect estimates to one or more study-level
moderators. It is a natural extension of subgroup analysis, but it does not
turn aggregate study data into individual-level evidence.

!!! warning "Interpretation boundary"

    A Meta-regression coefficient is a study-level association. It may reflect
    ecological bias, confounding, measurement differences, or post-hoc model
    selection and must not be interpreted as an individual-level causal effect.
    Fits with fewer than ten studies carry an explicit warning.

## Fit a numeric moderator

Provide a generic effect plus exactly one of its sampling variance or standard
error:

```python
import pandas as pd
import meta_analyze as ma

studies = pd.DataFrame(
    {
        "citation": ["A", "B", "C", "D", "E", "F"],
        "effect": [0.12, 0.25, 0.41, 0.38, 0.62, 0.76],
        "se": [0.18, 0.20, 0.17, 0.23, 0.19, 0.21],
        "mean_age": [42.0, 48.0, 51.0, 55.0, 60.0, 64.0],
    }
)

result = ma.meta_regression(
    studies,
    effect="effect",
    standard_error="se",
    moderators=["mean_age"],
    study="citation",
    model="mixed",
    tau2_method="REML",
)

print(result.summary())
print(result.coefficients)
```

Numeric moderators are used exactly as supplied. PyMetaAnalysis does not
center, scale, transform, or impute them automatically.

## Encode categorical moderators explicitly

Every categorical moderator requires an ordered, complete list of levels. The
first level is the treatment-coding reference:

```python
result = ma.meta_regression(
    studies,
    effect="effect",
    standard_error="se",
    moderators=["mean_age", "region"],
    categorical={
        "region": ["Europe", "Asia", "North America"],
    },
)
```

This produces terms such as `region[Asia]` and `region[North America]`, each
relative to `Europe`. The reference never depends on row order. Undeclared
levels, levels absent after exclusions, and string moderators omitted from
`categorical=` are errors rather than implicit recoding decisions.

Formula parsing, automatic interactions, splines, and polynomial terms are not
implemented. Construct those columns explicitly before fitting when they are
scientifically prespecified.

## Array-like input

Use a mapping when moderators are arrays rather than DataFrame column names:

```python
result = ma.meta_regression(
    effect=[0.10, 0.32, 0.45, 0.71],
    variance=[0.04, 0.05, 0.06, 0.08],
    moderators={"dose": [0.0, 1.0, 2.0, 3.0]},
    model="common",
)
```

A sequence such as `moderators=["mean_age", "region"]` is only meaningful
with a DataFrame. A mapping may mix DataFrame column selectors and array-like
values, provided every input has one value per row.

## Models and inference

`model="mixed"` is the default. It estimates residual tau-squared after the
moderators using REML, PM, or DL. `model="common"` fixes residual tau-squared
at zero and supports normal inference only.

| Setting | Coefficient tests and intervals | Joint moderator test |
| --- | --- | --- |
| `normal` | z / normal | chi-squared |
| `hartung_knapp` | t with `k-p` df | F with `k-p` denominator df |
| `hartung_knapp_adhoc` | safeguarded t with `k-p` df | safeguarded F |

The unmodified Hartung-Knapp result warns when its covariance is below the
classic covariance. The `adhoc` choice explicitly applies a lower-bound scale
of one. Common-effect models reject both Hartung-Knapp choices.

The design matrix must be full rank and leave positive residual degrees of
freedom (`k > p`). PyMetaAnalysis does not silently drop collinear terms or use
a pseudo-inverse.

## Inspect the result

Meta-regression has no single pooled effect, so `MetaRegressionResult` does not
provide a scalar `estimate` or `ci`. Inspect its coefficient table instead:

```python
result.coefficients
result.coefficient_covariance
result.global_test
result.test_moderator("region")
result.heterogeneity
result.tau2
result.tau2_null
result.pseudo_r2
result.diagnostics
```

`global_test` tests all non-intercept terms. `test_moderator(name)` tests every
encoded term belonging to that original moderator, so a multi-level category
receives one joint test rather than separate interpretation through dummy-term
p-values.

`study_results` retains every input row and contains fitted values, residuals,
precision weights, normalized precision weights, and leverage. A regression
precision weight is not a universal percentage contribution to every
coefficient.

## Leave-one-out sensitivity

Use exact deleted-model refits to inspect dependence on individual studies:

```python
diagnostics = result.leave_one_out()

print(diagnostics.table)
print(diagnostics.coefficients)
```

Every successful deletion re-estimates residual tau-squared, coefficients, and
inference with the same resolved model configuration. Deletions that make the
design unidentifiable are retained explicitly instead of aborting the other
refits. See [sensitivity analysis](sensitivity-analysis.md) for the result
contract and interpretation limits.

## Predict at moderator values

Prediction replays the fitted numeric and categorical encoding:

```python
predictions = result.predict(
    pd.DataFrame(
        {
            "mean_age": [50.0, 65.0],
            "region": ["Europe", "Asia"],
        }
    )
)
```

Every model returns the fitted mean and its confidence interval. Mixed-effects
models also return `pi_low` and `pi_high` for the distribution of true effects
in a new study with those moderators. The interval does not include an
additional, unknown sampling variance for a future observed estimate. Unknown
categories and missing prediction inputs are rejected.

## Plot a single numeric moderator

After installing the `plot` extra, an intercept-containing model with exactly
one numeric moderator can be displayed as a weighted bubble plot:

```python
ax = result.bubble(
    moderator_label="Mean age (years)",
    effect_label="Treatment effect",
    show_confidence_interval=True,
    show_prediction_interval=True,
)
```

Bubble area is proportional to normalized fitted precision weight. The fitted
line and interval bands reuse `result.predict()`, including the selected normal
or Hartung-Knapp covariance and critical value. A prediction band is available
only for mixed-effects models.

PyMetaAnalysis rejects bubble plots for categorical, multiple-moderator, or
no-intercept fits. Drawing a marginal or partial-effect line for those models
requires explicit choices for the other moderator values, and the library does
not silently choose them.

## Residual heterogeneity and pseudo-R²

The result reports residual `QE`, I-squared, H-squared, and tau-squared. For an
intercept-containing mixed model, PyMetaAnalysis also refits the same included
rows without moderators and reports:

```text
pseudo-R² = max(0, 1 - tau²_model / tau²_null)
```

The raw value is retained as `pseudo_r2_raw`. If the null-model tau-squared is
zero, pseudo-R² is undefined; if moderators increase estimated tau-squared,
the public value is truncated to zero and the negative raw value is retained
with a warning. Pseudo-R² is not the proportion of outcome variance explained
in ordinary individual-level regression.

## Inspect deleted-study influence

Use exact deletion diagnostics when a study appears unusual or the fitted
association may depend strongly on one row:

```python
influence = result.influence()

influence.table
influence.dfbetas
influence.flagged
```

Every included study is omitted once, with tau-squared and coefficient
inference re-estimated under the original settings. The result reports the
externally standardized deleted residual, Cook's distance, and term-specific
DFBETAS values. Failed reduced models remain visible rather than being silently
dropped.

The result also exposes its numerical screening thresholds. These thresholds
identify rows for review; they do not prove that a study is erroneous,
authorize automatic exclusion, or correct for trying multiple model
specifications. See [sensitivity analysis](sensitivity-analysis.md) for the
full output and interpretation contract.

## Missing values, provenance, and reports

`missing="raise"` identifies every missing effect, uncertainty, study label,
or moderator field by row. `missing="drop"` excludes the entire row from every
model calculation while retaining it in `study_results` with all applicable
reasons.

```python
result.provenance
result.method_details()
result.report().to_json()
```

Provenance records moderator input roles and categorical treatment coding.
Reports include coefficients, their covariance, residual heterogeneity,
moderator tests, encoding, diagnostics, row decisions, and the study-level
interpretation warning.

See [statistical methods](../methods/statistical-methods.md#meta-regression) for
the equations and [scope and limitations](../limitations.md) before using the
model for consequential work.
