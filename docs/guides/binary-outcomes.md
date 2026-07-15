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

Mantel-Haenszel pooling currently supports OR and RR with `model="common"` and
`ci_method="normal"`.

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

RD is available through inverse-variance pooling:

```python
result = ma.meta_binary(
    studies,
    event_treat="events_t",
    n_treat="total_t",
    event_control="events_c",
    n_control="total_c",
    measure="RD",
    method="IV",
    model="common",
    rd_zero_variance="correct",
)
```

`rd_zero_variance="correct"` is the default. It retains boundary studies with
their raw RD and uses corrected counts only for sampling variance. Use
`"exclude"` for a protocol that excludes these studies before all synthesis
calculations. See [zero-event studies](zero-events.md) for details.

## Input validation

Event counts and sample sizes must be finite, integer-valued, and non-negative;
events cannot exceed their group total, and totals must be positive. Missing
rows raise by default. With `missing="drop"`, they remain in the result table
with `included=False` and an exclusion reason.

Sparse tables require additional decisions. Read
[zero-event studies](zero-events.md) before changing continuity-correction
settings.
