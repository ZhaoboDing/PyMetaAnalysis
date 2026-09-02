# Zero-event studies

Zero cells in 2-by-2 tables affect OR, RR, RD, Mantel-Haenszel, and Peto
estimators in different ways. PyMetaAnalysis therefore separates study-level
effect corrections from table-based pooling rules.

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

For RR, a zero non-event cell is allowed when both arms still have positive
event counts and the sampling variance is positive. For example, a study with
events in every participant of one arm can be analyzed without correction.
OR still requires all four cells to be positive.

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
counts affect only its sampling variance. Under the default
`correction_scope="only_zero_studies"`, the corrected variance is used for
every retained RD table containing at least one zero cell, including
single-zero tables whose uncorrected variance was already positive. The
`rd_zero_variance` policy separately decides whether tables whose raw RD
variance is exactly zero are retained or excluded. To exclude all such
zero-variance studies before pooling, Q, tau-squared, and weight calculations,
use:

```python
rd_zero_variance = "exclude"
```

Excluded rows remain in `result.study_results` with
`exclusion_reason="zero uncorrected risk-difference variance"`. The policy is
RD-specific; setting `exclude` for OR or RR is rejected.

## Mantel-Haenszel correction is separate

Uncorrected common-effect Mantel-Haenszel OR/RR/RD pooling uses raw tables by
default:

```python
mh_continuity_correction = None
mh_correction_scope = None  # resolves to "only_zero_studies" for MH
```

`continuity_correction` still controls the study-level effects used for display
and heterogeneity. It does not silently alter the pooled Mantel-Haenszel
estimator. If an exact pooled estimator is undefined, choose an explicit
positive `mh_continuity_correction` and report that decision.
Both MH-specific options are rejected when explicitly supplied to IV or Peto
pooling, where they would otherwise have no effect.

For MH RD, `rd_zero_variance="exclude"` removes zero-variance boundary rows
before every synthesis calculation. With the default `"correct"` policy, the
raw table still enters the MH point estimate unless an explicit
`mh_continuity_correction` is supplied. If the Sato-Greenland-Robins variance
is non-positive, the uncorrected fit raises instead of silently changing the
tables; a positive MH correction is an explicit protocol choice. Some other
implementations, including `metafor`, can report a degenerate zero or near-zero
standard error for such boundary data. PyMetaAnalysis rejects that result by
the policy recorded in ADR 0005.

## Peto pooling always uses raw tables

Peto's observed-minus-expected pooling contribution remains defined for many
single-zero tables, so `method="Peto"` does not apply a continuity correction
to pooling or Peto Q. The general `continuity_correction` and
`correction_scope` settings apply only to the one-step study effects and
variances shown in the study table. There is deliberately no separate Peto
pooling-correction option.

Double-zero and double-all studies have zero Peto information and are excluded
before pooling, Q, I-squared, H-squared, and weights. The result retains the
same structured exclusion reasons used for other relative-effect analyses.
This means `fit_peto()` expects already-filtered tables. It also differs from
the pooling side of `metafor::rma.peto()` under its default `drop00` handling,
which can retain zero-information rows when calculating `k` and Q degrees of
freedom even though those rows contribute no Peto information.
Peto's lack of a single-zero pooling correction does not make it universally
preferable: its rare-outcome, balanced-arm, and modest-effect assumptions must
still be considered.

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

Applicable resolved correction values and scopes also appear in
`dict(result.method.options)`. MH-specific keys are present only for MH fits.
RD analyses additionally record the resolved zero-variance policy and affected
row IDs in provenance. This makes it possible to distinguish a corrected
analysis from an exact or exclusion-based one after fitting.
Peto analyses additionally record `peto_pooling_tables="raw"` and
`peto_heterogeneity="O-minus-E"`.
