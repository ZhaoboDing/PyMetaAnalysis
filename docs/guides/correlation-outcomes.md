# Correlation outcomes

Use `meta_correlation()` when each independent study contributes a Pearson
correlation and its sample size. The first implementation deliberately
supports only Fisher's z (`measure="ZCOR"`), the conventional
variance-stabilizing route implemented by R `metafor::escalc(measure="ZCOR")`
and `meta::metacor(sm="ZCOR")`.

## DataFrame input

```python
import pandas as pd
import meta_analyze as ma

studies = pd.DataFrame(
    {
        "r": [0.18, 0.42, -0.05, 0.31, 0.27],
        "sample_size": [84, 120, 63, 95, 150],
    },
    index=["Study A", "Study B", "Study C", "Study D", "Study E"],
)

result = ma.meta_correlation(
    studies,
    correlation="r",
    n="sample_size",
    model="random",
    tau2_method="REML",
)
```

Omitting `study=` uses the DataFrame index. Lists, NumPy arrays, and pandas
Series are also accepted directly.

## Model and display scales

Each study is transformed before fitting:

```text
z_i = atanh(r_i)
variance_i = 1 / (n_i - 3)
```

The model, confidence interval, prediction interval, tau-squared, and
heterogeneity calculations remain on Fisher's z scale. This is explicit in
the result:

```python
result.effect_scale  # "fisher_z"
result.estimate  # pooled Fisher's z
result.ci  # Fisher's z confidence interval
```

Use display properties for back-transformed correlations:

```python
result.display_scale  # "tanh"
result.display_estimate
result.display_ci
result.display_prediction_interval
```

The pooled correlation is `tanh(pooled_z)`. It is not a direct weighted
average of the raw correlations.

## Input boundaries

Included rows must satisfy:

- a finite correlation strictly between -1 and 1;
- a whole-number sample size of at least 4.

Values at -1 or 1 would produce an infinite Fisher's z. A sample size no
larger than 3 would make `1 / (n - 3)` non-positive or undefined. Both cases
raise `InvalidStudyDataError` instead of being clipped or corrected.

With `missing="drop"`, missing correlation or sample-size rows remain in
`study_results` as explicit exclusions. Invalid non-missing values still
raise an error.

## Models and uncertainty

The default is a random-effects inverse-variance model with REML. Common-
effect pooling and the same random-effects choices as the generic API are
available:

```python
common = ma.meta_correlation(
    studies,
    correlation="r",
    n="sample_size",
    model="common",
)

hk = ma.meta_correlation(
    studies,
    correlation="r",
    n="sample_size",
    model="random",
    tau2_method="PM",
    ci_method="hartung_knapp_adhoc",
)
```

`tau2_method` supports REML, PM, and DL for random effects. Normal,
Hartung-Knapp, prediction-interval, and Q-profile behavior follows the shared
inverse-variance implementation and remains on the z scale until displayed.

## Subgroups, sensitivity, reports, and plots

The standard workflows are available without reconstructing effects by hand:

```python
studies = studies.assign(
    population=["adult", "adult", "adult", "youth", "youth"],
    publication_year=[2001, 2004, 2008, 2011, 2015],
)

subgroups = ma.meta_correlation(
    studies,
    correlation="r",
    n="sample_size",
    subgroup="population",
)

leave_one_out = result.leave_one_out().to_dataframe()
cumulative = result.cumulative(order="publication_year").to_dataframe()
methods = result.method_details()
report = result.report().to_dict()
forest_ax = result.forest()
funnel_ax = result.funnel()
```

Forest and funnel x-coordinates are back-transformed correlations on a linear
axis by default. Funnel standard errors remain model-scale Fisher's z standard
errors.

## Independence boundary

This API assumes one independent effect per study. Repeated outcomes,
different variable pairs, or multiple time points from the same participants
are statistically dependent even if their rows have different labels.
PyMetaAnalysis does not currently estimate the covariance matrix or fit a
multilevel/multivariate model for those data. Select one prespecified effect,
combine effects using an appropriate external method, or use software that
models dependence explicitly.

The first release also does not support raw-correlation pooling (`COR`),
partial correlations, rank correlations, or reliability corrections. Passing
`measure="COR"` raises `UnsupportedMethodError`.
