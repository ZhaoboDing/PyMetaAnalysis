# ADR 0009: Small-study-effect test conventions

- Status: Records current implementation; independent 1.0 ratification pending
- Date: 2026-09-06
- Related: [statistical review](../statistical-review.md),
  [implemented formulas](../methods/statistical-methods.md)

## Context

The four diagnostic methods introduced in 0.8–0.9 test different aspects of
small-study effects. An API freeze needs an explicit account of their inputs,
inference conventions and limits, rather than treating them as interchangeable
tests for publication bias.

## Recorded decisions

| Method | Inputs and inference | Independent software reference |
| --- | --- | --- |
| Begg–Mazumdar | Common-effect-centered standardized model-scale effects and variances; Kendall tau-b; exact two-sided inference without ties below 50 studies, otherwise tie-adjusted normal inference; optional continuity correction | `metafor::ranktest()` |
| Egger | Model-scale effects regressed on standard error with inverse-variance weights and multiplicative dispersion; t test with `k-2` df | `metafor::regtest(model="lm", predictor="sei")` |
| Harbord | Retained raw binary OR event margins; efficient-score regression, multiplicative dispersion and t inference; no continuity correction | `meta::metabias(method.bias="Harbord")` |
| Peters | Conventional study log OR reconstructed with the source study-level correction; raw inverse total sample size and `S*F/N` weights; multiplicative dispersion and t inference | `meta::metabias(method.bias="Peters")` |

Each method requires at least three eligible studies and an identifiable test.
Eligibility, not merely the source model's study count, determines whether a
test can run. None changes the fitted pooling model or establishes or excludes
publication bias. The four methods remain explicit user choices; there is no
automatic diagnostic selection or inference-default change.

Egger and Begg consume the source study effects, including Fisher's z for
correlations. Peto study effects are one-step approximations and carry an
additional caveat. Harbord and Peters require retained binary OR counts and
are unavailable for generic or correlation inputs. Peters uses conventional
log OR even when the source pooled analysis uses Peto.

## Consequences and evidence

Preserve the current named result fields and formulas while reviewing their
1.0 contracts. Additional interpretation warnings are intentional; consumers
should not compare exact warning tuples as a statistical validity criterion.
Detailed equations and applicability are in the
[methods reference](../methods/statistical-methods.md) and
[small-study-effect guide](../guides/small-study-effects.md).

Pinned R generators and fixtures, direct regression/rank calculations, scale
and permutation tests are listed in [validation](../validation.md). The first
corrective batch adds correlation composition and Peto-caveat tests and fixes
Peters' included-study correction-count metadata. These are implementation
evidence, not independent statistical sign-off. Gate M5/M6 reviewers must
review eligibility, ties, correction reuse, numerical boundaries and tolerances
on the eventual candidate commit before ratification.
