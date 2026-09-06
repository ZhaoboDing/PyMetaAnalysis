# ADR 0008: Stable random-effects mean-inference default

- Status: Proposed; statistical decision pending independent review
- Date: 2026-09-06
- Related: [ADR 0002](0002-statistical-policy.md),
  [ADR 0004](0004-hartung-knapp-prediction-intervals.md)

## Context

The existing accepted policy uses `ci_method="normal"` for pooled analyses
and `inference_method="normal"` for meta-regression. The first stable release
must explicitly ratify or change that default. This brief opens the decision;
it does not supersede ADR 0002 or claim an external review has taken place.

## Options and evidence

| Option | Consequence requiring review |
| --- | --- |
| Retain `normal` | Preserves default 0.9 calls, but does not address uncertainty from estimating heterogeneity in mean inference |
| Default to `hartung_knapp` | Uses adjusted covariance and t inference; may produce smaller variance than the classic estimator |
| Default to `hartung_knapp_adhoc` | Uses t inference and a variance floor at the classic variance; can produce very wide intervals with few studies |

The Cochrane Handbook discusses narrow normal intervals and both wide and narrow
HKSJ boundary behavior; it does not supply a universal resolution for every
small-study configuration. See
[Chapter 10, section 10.10.4](https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-10).
The documented `metafor` distinction between `test="knha"` and `test="adhoc"`
confirms that the variance safeguard is a separate method, not an implicit part
of ordinary HK. See the [rma.uni reference](https://wviechtb.github.io/metafor/reference/rma.uni.html).
Sources consulted on 2026-09-06; freeze the exact reference versions/citations
in the final review record.

## Decision required before 0.10.0

The maintainer and an independent methods reviewer must choose one option and
record the rationale, rejected alternatives, scope and evidence. Review at least
`k=2,3,5,10`, balanced/unbalanced precisions, zero/positive tau-squared, identical
effects, residual HK scales below/above one, and regression residual degrees of
freedom near their boundary. Existing fixtures supply implementation evidence;
do not infer operating characteristics or universal superiority from fixture
agreement alone. Use primary methodological evidence and, if needed, a
prespecified simulation with Monte Carlo uncertainty.

Decide explicitly whether the pooled default and regression default should
coincide. A pooled HK change also affects HTS versus HK-PR prediction metadata
under ADR 0004, refits, subgroups, summaries and reports. Regression prediction
rules need separate examination under ADR 0003. Common-effect MH and Peto do
not acquire HK inference merely because a random-effects default changes.

## Compatibility and validation requirements

If the default changes, make the change in 0.10 before the API freeze. List all
affected entry points, resolved method fields, numerical/behavioral tests, R
reference cases, snapshots, notebooks and documentation. Explain how explicit
`normal` calls retain the previous inference and how previously omitted options
change. Keep the old explicit method available. If `normal` is retained, record
that decision with the same evidence and preserve explicit HK sensitivity paths.

No implementation or reference values change in the foundation batch. Acceptance
requires a completed review record linked here, a final decision paragraph,
updated tests and the migration guide. Until then the current accepted default
remains in force and roadmap gate D1 remains open.
