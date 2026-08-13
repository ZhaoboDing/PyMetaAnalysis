# ADR 0005: Mantel-Haenszel risk difference

- Status: Accepted
- Date: 2026-08-13
- Supersedes: the RD exclusion from the Mantel-Haenszel scope in
  [ADR 0002](0002-statistical-policy.md)

## Context

ADR 0002 limited Mantel-Haenszel pooling to common-effect odds ratios and risk
ratios while the risk-difference estimator and its variance convention were
still undecided. This left inverse variance as the only pooling method for RD,
although conventional R implementations provide a common-effect MH RD.

The point estimator is straightforward, but several variance estimators have
appeared in the literature. The selected rule must work under both large-
stratum and sparse-data limiting models, preserve treatment/control symmetry,
and remain explicit in reports and cross-software validation.

## Decision

PyMetaAnalysis supports `measure="RD", method="MH", model="common"`. With
treatment total `n1_i`, control total `n0_i`, total `N_i`, and
`w_i = n1_i n0_i / N_i`, the estimate is:

```text
RD_i = a_i / n1_i - c_i / n0_i
RD_MH = sum(w_i RD_i) / sum(w_i)
```

The normal confidence interval uses the Sato-Greenland-Robins sampling
variance. The resolved method options record
`mh_rd_variance="Sato-Greenland-Robins"`.

Raw tables are used for MH pooling by default. `mh_continuity_correction` and
`mh_correction_scope` remain the only settings that alter MH pooling tables;
the separate study-effect correction continues to control displayed study
uncertainty and the inverse-variance heterogeneity calculation. The existing
`rd_zero_variance` policy determines whether boundary studies enter all
synthesis calculations. A non-positive pooled Sato variance raises a domain
error instead of silently adding a correction.

Random-effects MH remains unsupported. Random-effects RD continues to use
inverse-variance pooling with an explicit tau-squared estimator.

## Validation

- direct formula tests cover the estimate, Sato variance, weights, and normal
  interval;
- treatment/control swaps negate the estimate and mirror its interval without
  changing the standard error;
- row reordering and common count scaling preserve the expected invariants;
- extreme finite counts exercise the overflow-safe scaled implementation;
- boundary-policy, explicit-correction, subgroup, leave-one-out, and
  cumulative paths are covered; and
- fixed-version `metafor::rma.mh(measure="RD")` fixtures cover ordinary,
  sparse, and explicitly corrected tables.

## Consequences

- common-effect OR, RR, and RD all support MH or inverse-variance pooling;
- study-table MH weights for RD are proportional to `n1_i n0_i / N_i`;
- the selected RD variance convention is recoverable from method metadata and
  generated reports; and
- documentation must continue to distinguish MH RD from random-effects IV RD.
