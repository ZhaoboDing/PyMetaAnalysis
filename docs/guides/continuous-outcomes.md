# Continuous outcomes

Use `meta_continuous()` for independent treatment and control groups described
by means, standard deviations, and sample sizes.

## Mean difference

```python
import pandas as pd
import meta_analyze as ma

studies = pd.DataFrame(
    {
        "mean_t": [11.2, 9.8, 12.4, 10.7],
        "sd_t": [3.1, 2.7, 3.4, 2.9],
        "n_t": [48, 52, 45, 60],
        "mean_c": [10.1, 9.3, 11.0, 10.2],
        "sd_c": [2.9, 2.8, 3.0, 3.1],
        "n_c": [50, 49, 47, 58],
    },
    index=["Trial A", "Trial B", "Trial C", "Trial D"],
)

result = ma.meta_continuous(
    studies,
    mean_treat="mean_t",
    sd_treat="sd_t",
    n_treat="n_t",
    mean_control="mean_c",
    sd_control="sd_c",
    n_control="n_c",
    measure="MD",
    model="random",
    tau2_method="REML",
)
```

MD is treatment mean minus control mean. Its sampling variance is calculated as
`sd_treat**2 / n_treat + sd_control**2 / n_control`, without assuming equal
group variances.

## Standardized mean difference

Use SMD when studies measure the same construct on different scales:

```python
result = ma.meta_continuous(
    studies,
    mean_treat="mean_t",
    sd_treat="sd_t",
    n_treat="n_t",
    mean_control="mean_c",
    sd_control="sd_c",
    n_control="n_c",
    measure="SMD",
)
```

SMD uses a pooled within-study SD, the exact gamma-function small-sample
correction for Hedges' g, and the `LS` sampling-variance convention. The
intermediate pooled SD, Cohen's d, correction factor, final effect, and
variance are retained in `result.study_results`.

## Interpretation and validation

Positive MD or SMD values mean the treatment-group outcome is larger. Whether
that direction is beneficial depends on the outcome definition and should be
decided in the review protocol.

Sample sizes must be integer-valued and sufficient to estimate within-study
variation. Standard deviations must be finite and non-negative; cases where a
valid effect variance cannot be formed are rejected rather than silently
adjusted.

Use `missing="drop"` only when excluding incomplete rows is consistent with the
analysis plan. Excluded rows remain available through
`result.excluded_studies`.
