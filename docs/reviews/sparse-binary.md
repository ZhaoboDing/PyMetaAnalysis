# Sparse binary, Mantel-Haenszel and Peto review packet

Prepared 2026-09-14 against main `297810c`, with the corrections in this
packet's PR. Status: implementation evidence prepared; **independent review
pending**. An external reviewer must record the candidate commit, identity,
findings and sign-off using the [review record](../statistical-review.md).
This packet supplies evidence for M3 and M6. It does not accept those gates.

## Scope and assumptions

The packet covers study-level OR, RR and RD calculations, common-effect
Mantel-Haenszel OR/RR/RD, Peto one-step OR, normal confidence intervals,
study and pooling weights, heterogeneity, zero-cell corrections, exclusions,
and finite-count numerical boundaries. Random-effects binary analyses use the
separately reviewed inverse-variance path and are outside this packet.

Studies are independent 2-by-2 tables. Counts must be finite, non-negative,
and no larger than their positive arm totals. OR and RR are analysed on the log
scale; RD stays on its natural scale. Double-zero and double-all studies are
excluded from relative-effect analyses. They remain eligible for RD subject to
the recorded zero-variance policy.

The `continuity_correction` and `correction_scope` settings affect individual
effects used for display and inverse-variance heterogeneity. MH pooling uses raw
tables unless a separate positive `mh_continuity_correction` is supplied. Peto
pooling and Peto Q always use raw tables. These distinctions are part of the
public method metadata and provenance.

## Formula and evidence map

For each stratum, cells are `(a_i, b_i, c_i, d_i)`, arm totals are
`n1_i=a_i+b_i`, `n0_i=c_i+d_i`, and `N_i=n1_i+n0_i`.

| Quantity | Formula / convention | Implementation | Evidence |
| --- | --- | --- | --- |
| Study OR | `log(a_i d_i / (b_i c_i))`; variance `1/a_i+1/b_i+1/c_i+1/d_i` after the recorded study correction | `effect_sizes/binary.py` | Six-case R matrix and existing direct/correction tests |
| Study RR | `log((a_i/n1_i)/(c_i/n0_i))`; variance `1/a_i-1/n1_i+1/c_i-1/n0_i` | `effect_sizes/binary.py` | Six-case R matrix and zero-non-event boundary tests |
| Study RD | Raw `a_i/n1_i-c_i/n0_i`; binomial variance may use corrected counts under the recorded policy | `effect_sizes/binary.py`; ADR 0005 | Six-case R variances, raw-RD oracle and boundary-policy tests |
| MH OR | `log(sum(a_i d_i/N_i)/sum(b_i c_i/N_i))`; Greenland-Robins variance | `estimators/mantel_haenszel.py` | Exact rational oracle, six-case raw/corrected R fits, swaps and permutations |
| MH RR | `log(sum(a_i n0_i/N_i)/sum(c_i n1_i/N_i))`; Greenland-Robins variance | `estimators/mantel_haenszel.py` | Exact rational oracle, six-case raw/corrected R fits and cancellation boundary |
| MH RD | Arm-size-weighted RD with Sato-Greenland-Robins variance | `estimators/mantel_haenszel.py`; ADR 0005 | Exact rational oracle and six-case raw/corrected R fits |
| MH heterogeneity | Inverse-variance Q for displayed study effects, centred on the MH estimate | `heterogeneity.py` | R comparisons when display conventions agree; direct raw-RD calculation otherwise |
| Peto | `OE_i=(a_i n0_i-c_i n1_i)/N_i`; `V_i=(a_i+c_i)(b_i+d_i)n1_i n0_i/(N_i^2(N_i-1))`; estimate `sum(OE_i)/sum(V_i)`; variance `1/sum(V_i)` | `effect_sizes/binary.py`, `estimators/peto.py`; ADR 0006 | Exact rational oracle, six-case R study/fit/Q comparisons and arm-swap invariant |

The [official `metafor::rma.mh` reference](https://wviechtb.github.io/metafor/reference/rma.mh.html)
documents its supported measures and independent study-versus-pooling settings
for `add`, `to`, and `drop00`. The
[official `metafor::rma.peto` reference](https://wviechtb.github.io/metafor/reference/rma.peto.html)
documents the same separation for Peto and its restricted applicability. The
[Cochrane Handbook](https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-10)
describes sparse-event bias and the limitations of routine continuity
corrections. Online references were checked on 2026-09-14; executable results
use the pinned environment below.

## Reproducible comparison design

`tests/reference/generate_sparse_binary_review_metafor.R` contains 28 studies
in six deterministic datasets:

- balanced arms without zeros;
- rare outcomes with single-zero studies;
- double-zero and double-all boundary studies mixed with informative rows;
- strongly unequal arm sizes;
- opposing study effects; and
- large counts up to 100 million.

For every dataset, the fixture records OR/RR/RD study effects and variances,
raw MH fits, and MH fits with 0.5 added only to eligible zero-cell pooling
tables. It also records Peto study effects, pooled estimate, SE, interval,
weights and O-minus-E heterogeneity. Fit fields use model-scale values and
normalized weights. This is a deterministic implementation comparison, not a
simulation of bias, coverage, or method suitability.

The generator requires R 4.6.1, `metafor` 5.0.1 and `jsonlite` 2.0.0 and stops
on a version mismatch. Regenerate to a separate path and compare it before
replacing the committed artifact:

```console
Rscript tests/reference/generate_sparse_binary_review_metafor.R .release-smoke/sparse-binary-candidate.json
python -m pytest tests/test_sparse_binary_review.py tests/test_binary.py tests/test_binary_workflow_corrections.py tests/test_estimators.py tests/test_r_references.py
```

R comparisons use `rtol=5e-13, atol=5e-15`. The exact `Fraction` oracles use
`rtol=2e-13` for log-domain aggregation and near-minimum-subnormal absolute
bounds for very small positive SEs and weights. Ordinary values retain explicit
absolute checks. Tests fail on unexpected `RuntimeWarning` in the extreme-scale
reproducers.

## Findings and disposition

| ID | Reproducer and finding | Resolution / scope |
| --- | --- | --- |
| SB1 | A Peto analysis combining a 20-participant stratum with a balanced stratum of order `1e308` underflowed the smaller row after one global rescaling, then falsely reported missing Peto information | Scale each row locally, aggregate signed O-minus-E and positive information in the log domain, and compute Q from locally scaled residuals |
| SB2 | The analogous MH RR case produced a non-positive variance after products underflowed; the usual variance numerator also subtracts nearly equal large terms | Aggregate positive terms in the log domain and use the exact identity `a_i d_i n1_i + b_i c_i n0_i` for the RR variance numerator |
| SB3 | A valid low-level MH OR table `(1, 1e308, 1e308, 1)` lost both opposing cross-products and was rejected although its log OR and variance are finite | Form pooled cross-products, Greenland-Robins variance terms and weights from per-row log components |
| SB4 | For zero-cell RD studies, `metafor` applies its display correction to the reported study RD, while PyMetaAnalysis deliberately retains raw RD and corrects only variance | Preserve ADR 0005. Compare MH point/SE/weights with R; validate Python Q directly from raw RD and the recorded variances |

The corrections change no statistical formula, default, result field, or
exclusion policy. Existing binary and R regressions remain unchanged within
their established tolerances. The core benchmark now times 1,000-study MH OR
and Peto OR paths so later candidates can detect material regressions. Final
values still must be representable as finite float64 values; domain errors
remain appropriate outside that boundary.

## Review still required

An independent reviewer must assess the estimands, Greenland-Robins and Sato
variance equations, Peto applicability warning, correction and exclusion
scopes, raw-RD difference, test tolerances, and all findings on the final
candidate commit. The deterministic cases establish implementation agreement,
not operating characteristics. Sparse-event method selection remains a review
protocol decision. Package version and statistical defaults remain unchanged.
