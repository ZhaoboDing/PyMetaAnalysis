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
Sources consulted on 2026-09-06 and checked on 2026-09-07; freeze the exact reference versions/citations
in the final review record.

## Proposed recommendation and concrete behavior

**Propose retaining `normal` for 1.0**, with ordinary and safeguarded HK remaining
explicit choices. This is a compatibility recommendation pending independent
methods review, not a claim that normal intervals have the best coverage.
Keeping one stable, explicit default avoids a silent change to existing
analyses and a rule that switches methods according to the observed result.
Normal inference can understate uncertainty; users should prespecify their
method and examine appropriate sensitivity analyses.

The [core inference packet](../reviews/core-inference.md) provides 24 datasets
and 216 pinned R comparisons. Selected REML results below show the HK residual
scale `q_HK=V_HK/V_classic` and total mean-CI width relative to normal. All use
95% confidence and the deterministic inputs specified in that packet.

| Case ID | Estimated tau-squared | q_HK | HK / normal width | Safeguarded HK / normal width |
| --- | --- | --- | --- | --- |
| `2_balanced_identical` | 0 | 0 | 0 | 6.4829 |
| `2_unbalanced_low_spread` | 0 | 0.000198 | 0.0912 | 6.4829 |
| `2_unbalanced_high_spread` | 6.99 | 1 | 6.4829 | 6.4829 |
| `3_unbalanced_high_spread` | 2.896798 | 1.049934 | 2.2494 | 2.2494 |
| `5_unbalanced_high_spread` | 1.667802 | 1.054772 | 1.4549 | 1.4549 |
| `10_unbalanced_high_spread` | 1.139140 | 1.057480 | 1.1869 | 1.1869 |

Ordinary HK is not proposed as the universal replacement because zero residual
variation can collapse its interval and unbalanced precision requires special
care. The safeguard prevents that collapse but can be very conservative with
few studies. Röver, Knapp and Friede (2015) investigated these tradeoffs in a
simulation with equal and unequal study sizes: ordinary HKSJ coverage worsened
with imbalance, whereas the modified procedure was conservative at small k.
Their [original study](https://doi.org/10.1186/s12874-015-0091-1) supports examining
both limitations, not extrapolating one universal winner from our fixtures.
The Cochrane guidance above also describes limitations of both normal and HKSJ
inference. Compatibility is therefore the proposal's deciding consideration;
the reviewer may reject it on methodological grounds.

The same compatibility proposal retains meta-regression's current `normal`
default, but intercept-only evidence cannot ratify regression inference. Its
residual-df boundaries, covariance and default/Riley prediction rules require
the regression packet before this ADR is accepted. No automatic data-dependent
fallback, new warning threshold or statistical default is introduced here.

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

The core review batch corrects numerical arithmetic and adds reference cases;
it does not change defaults or accept this proposal. Acceptance
requires a completed review record linked here, a final decision paragraph,
updated tests and the migration guide. Until then the current accepted default
remains in force and roadmap gate D1 remains open.
