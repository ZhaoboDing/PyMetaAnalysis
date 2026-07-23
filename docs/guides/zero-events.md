# Zero-event studies

Zero cells in 2-by-2 tables affect OR, RR, RD, and Mantel-Haenszel estimators in
different ways. PyMetaAnalysis therefore separates study-level effect
corrections from the pooled Mantel-Haenszel correction.

## Default study-level correction

For study-level OR/RR effects, the default settings are:

```python
continuity_correction = 0.5
correction_scope = "only_zero_studies"
```

The correction is added to every cell of an included study containing at least
one zero cell. Non-zero studies are not changed.

Available scopes are:

| Scope | Behavior |
| --- | --- |
| `only_zero_studies` | Correct each included study that contains a zero cell |
| `if_any_zero` | Correct every included study if any included study has a zero |
| `all_studies` | Correct every included study |
| `none` | Do not correct study-level tables |

Setting the correction to zero or the scope to `none` is rejected when it
leaves an OR/RR undefined or an effect variance non-positive.

## Double-zero and double-all studies

A study with no events in either group, or events in every participant in both
groups, contains no relative-effect information for OR or RR. Such studies are
excluded before effect pooling, heterogeneity statistics, and weights are
calculated. They remain visible in `result.study_results` with
`included=False` and a structured `exclusion_reason`.

RD has a different estimand. Any study in which both arms are at a boundary
(zero events or events in every participant) has zero uncorrected RD sampling
variance. This includes double-zero, double-all, and opposite-boundary tables.

The default policy retains these studies:

```python
rd_zero_variance = "correct"
```

The RD itself remains the raw treatment risk minus control risk. Corrected
counts are used only to form a positive sampling variance. To exclude all such
studies before pooling, Q, tau-squared, and weight calculations, use:

```python
rd_zero_variance = "exclude"
```

Excluded rows remain in `result.study_results` with
`exclusion_reason="zero uncorrected risk-difference variance"`. The policy is
RD-specific; setting `exclude` for OR or RR is rejected.

## Mantel-Haenszel correction is separate

Exact common-effect Mantel-Haenszel OR/RR pooling uses raw tables by default:

```python
mh_continuity_correction = None
mh_correction_scope = "only_zero_studies"
```

`continuity_correction` still controls the study-level effects used for display
and heterogeneity. It does not silently alter the pooled Mantel-Haenszel
estimator. If an exact pooled estimator is undefined, choose an explicit
positive `mh_continuity_correction` and report that decision.

## Inspect what happened

```python
columns = [
    "study",
    "included",
    "exclusion_reason",
    "continuity_corrected",
    "rd_zero_variance",
    "mh_continuity_corrected",
    "normalized_weight",
]

result.study_results[columns]
```

Resolved correction values and scopes also appear in
`dict(result.method.options)`. RD analyses additionally record the resolved
zero-variance policy and affected row IDs in provenance. This makes it possible
to distinguish a corrected analysis from an exact or exclusion-based one after
fitting.
