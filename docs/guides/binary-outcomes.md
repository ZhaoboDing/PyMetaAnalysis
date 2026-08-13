# Binary outcomes

Use `meta_binary()` for independent treatment and control groups described by
event counts and total sample sizes.

## Mantel-Haenszel common-effect analysis

The default binary workflow is a Mantel-Haenszel common-effect risk ratio:

```python
import pandas as pd
import meta_analyze as ma

studies = pd.DataFrame(
    {
        "events_t": [12, 8, 15, 6],
        "total_t": [120, 95, 140, 80],
        "events_c": [18, 11, 19, 10],
        "total_c": [118, 100, 145, 82],
    },
    index=["Trial A", "Trial B", "Trial C", "Trial D"],
)

result = ma.meta_binary(
    studies,
    event_treat="events_t",
    n_treat="total_t",
    event_control="events_c",
    n_control="total_c",
    measure="RR",
    method="MH",
    model="common",
)

print(result.summary())
```

Mantel-Haenszel pooling supports OR, RR, and RD with `model="common"` and
`ci_method="normal"`. MH risk differences use the Sato-Greenland-Robins
sampling variance, recorded in `dict(result.method.options)`.

## Peto common-effect odds ratio

Peto's one-step estimator is an explicit alternative for common-effect OR:

```python
peto = ma.meta_binary(
    studies,
    event_treat="events_t",
    n_treat="total_t",
    event_control="events_c",
    n_control="total_c",
    measure="OR",
    method="Peto",
    model="common",
)
```

Peto pooling uses the raw 2-by-2 tables and its own observed-minus-expected
heterogeneity statistic. The study table contains Peto one-step effects;
`continuity_correction` affects those displayed study effects but not the
pooled result. Peto is intended for rare outcomes when treatment/control arm
sizes are similar within studies and effects are not large. The result always
retains this caveat in `warnings` and Methods text.

## Random-effects analysis

Use inverse-variance pooling for a random-effects binary analysis:

```python
result = ma.meta_binary(
    studies,
    event_treat="events_t",
    n_treat="total_t",
    event_control="events_c",
    n_control="total_c",
    measure="OR",
    method="IV",
    model="random",
    tau2_method="REML",
    ci_method="hartung_knapp_adhoc",
)
```

`method="IV"` first calculates one effect and variance per study, then passes
them to the generic inverse-variance model.

## Effect measures and scales

| Measure | Model scale | Display scale | Direction |
| --- | --- | --- | --- |
| OR | log odds ratio | odds ratio | treatment relative to control |
| RR | log risk ratio | risk ratio | treatment relative to control |
| RD | risk difference | risk difference | treatment minus control |

For OR and RR, `result.estimate` and `result.ci` stay on the log model scale.
Use `display_estimate` and `display_ci` for exponentiated ratios.

For a common-effect RD, choose either MH or inverse-variance pooling explicitly.
The MH form is:

```python
result = ma.meta_binary(
    studies,
    event_treat="events_t",
    n_treat="total_t",
    event_control="events_c",
    n_control="total_c",
    measure="RD",
    method="MH",
    model="common",
    rd_zero_variance="correct",
)
```

`rd_zero_variance="correct"` is the default. It retains boundary studies with
their raw RD and uses corrected counts only for sampling variance. Use
`"exclude"` for a protocol that excludes these studies before all synthesis
calculations. See [zero-event studies](zero-events.md) for details.
Use `method="IV"` for random-effects RD or when inverse-variance common-effect
pooling is the prespecified estimator.

## Input validation

Event counts and sample sizes must be finite, integer-valued, and non-negative;
events cannot exceed their group total, and totals must be positive. Missing
rows raise by default. With `missing="drop"`, they remain in the result table
with `included=False` and an exclusion reason.

Sparse tables require additional decisions. Read
[zero-event studies](zero-events.md) before changing continuity-correction
settings.

See [statistical methods](../methods/statistical-methods.md#binary-study-effects)
for the OR/RR/RD, Mantel-Haenszel, and Peto equations, and
[validation](../validation.md) for cross-software coverage.
