# Independent statistical review

Status: review infrastructure ready; external review not started. This page
defines the evidence needed for the [1.0 methods gates](roadmap-1.0.md).
Existing R comparisons are independent software validation, not reviewer
sign-off. Automated checks or an implementation self-review cannot substitute
for a reviewer who did not implement the methods under review.

## Review packets

Each packet must name a reviewer and candidate commit before acceptance. Use
the public methods guide, accepted ADRs, source, tests, and R generators
together. Input datasets, reference parameters, numerical scales, exclusions,
and known intentional differences must be inspectable.

| Packet | Source focus (`src/meta_analyze/`) | Evidence to start from (`tests/`) | Questions requiring sign-off |
| --- | --- | --- | --- |
| Pooling/effects/heterogeneity | `effect_sizes/`, `estimators/inverse_variance.py`, `estimators/tau2.py`, `heterogeneity.py` | `test_estimators.py`, `test_binary.py`, `test_continuous.py`, `test_correlation.py`, `test_properties.py`; generic/binary/continuous/correlation R fixtures | Estimands, scales, weight normalization, Q versus tau-squared inconsistency, convergence and extreme variances |
| HK, prediction and Q-profile | `estimators/inverse_variance.py`, `heterogeneity.py`, `estimators/meta_regression.py` | `test_estimators.py`, `test_meta_regression.py`, `test_numerical_stability.py`; generic/regression R fixtures; ADRs 0002–0004 | Selected covariance, residual degrees of freedom, small k, zero residual variance, constrained/empty intervals, transformed bounds |
| Sparse binary/MH/Peto | `effect_sizes/binary.py`, `binary_api.py`, `estimators/mantel_haenszel.py`, `estimators/peto.py` | `test_binary.py`, `test_numerical_stability.py`; binary/sparse R cases; ADRs 0005–0006 | Pooling versus display corrections, exclusion consistency, raw RD, Sato variance, Peto Q and applicability |
| Subgroups and refits | `subgroups.py`, `sensitivity.py` | `test_subgroups.py`, `test_sensitivity.py`; workflow R fixture | Independent tau-squared, singleton fallback, failed refits, row IDs, between-group tests |
| Meta-regression/diagnostics | `design_matrix.py`, `regression_api.py`, `regression_*.py`, `estimators/meta_regression.py` | `test_meta_regression.py`, `test_regression_*.py`; regression/influence/collinearity/contrast R fixtures | Full rank, coding, no-intercept REML and QE differences, covariance, deletion, omnibus tests and predictions |
| Small-study-effect tests | `small_study_effects.py` | `test_small_study_effects.py`, `test_begg_small_study_effects.py`, `test_harbord_small_study_effects.py`, `test_peters_small_study_effects.py`; corresponding R fixtures | Different null hypotheses, eligibility, exact/tied Begg inference, multiplicative dispersion, binary correction reuse and warnings |
| Trim-and-fill | `trim_fill.py`, `plotting/funnel.py` | `test_trim_fill.py`, `test_funnel_plot.py`, `test_r_references.py`; trimfill R fixture | L0/R0 ranks and rounding, ties, automatic direction, convergence/cycling, imputed variances, augmented refit and provenance |
| Reference/tolerance audit | All adjacent `tests/reference/generate_*.R` and JSON files | `test_r_references.py`, `test_reference_results.py`, reference README | Pinned versions, reproducibility, tolerances by statistic/algorithm, independent formula coverage and intentional differences |

## Procedure and evidence record

The [core inference packet](reviews/core-inference.md) prepares the intercept-only
pooling, heterogeneity, HK, prediction and Q-profile evidence, including new
boundary references and numerical corrections. It remains pending independent
review and does not constitute sign-off for either of the first two packets.

For every packet, copy the following record into a review PR or a committed
review report. Do not mark a packet accepted with blank evidence fields.

```text
Packet / roadmap gate:
Candidate commit:
Reviewer, date, and independence from implementation:
Reviewed formulas / primary references / ADRs:
Reviewed files and test cases:
R, metafor/meta, jsonlite versions and exact regeneration commands:
Input data and method parameters:
Compared statistics and model/display scales:
Tolerance for each comparison and reason:
Boundary, invalid-input and convergence cases checked:
Intentional software differences and their justification:
Findings (severity, reproducer, resolution commit, retest evidence):
Remaining limitations:
Outcome: open / changes required / accepted
Sign-off and evidence links:
```

Reproduce R fixtures into a separate output path using the versions recorded in
each artifact. Retain version/session output and compare before replacement.
Do not silently regenerate with newer packages or relax tolerances to match a
failing implementation. The commands are in the [validation guide](validation.md).

The existing R test file distinguishes closed-form `rtol=5e-13, atol=5e-15`
from iterative and derived-statistic tolerances. These are review inputs, not
blanket approved bounds. Inspect overrides and near-zero absolute errors,
as well as whether the R reference exercises the same estimand. A matching
pooled estimate does not establish that weights, intervals, heterogeneity,
exclusions, or provenance are correct.

Correctness findings, unexplained numerical discrepancies, or ambiguous
statistical defaults block the affected gate. Fixes must include regression
evidence and be reviewed on the new commit. Editorial findings may be tracked
separately, but known misleading methods documentation must be fixed before RC.
Keep the public validation status accurate throughout the review.

## Initial inspection findings

These findings were identified during foundation work against 0.9.0. The
historical evidence below explains the follow-up; implementation fixes do
not themselves constitute independent sign-off.

| ID | Observed evidence | Required disposition |
| --- | --- | --- |
| F1 | 0.9.0 zero-fill adjusted table included excluded rows and raised a pandas length error | Fixed with included-only adjusted fits, stable source IDs and synthetic IDs; zero/nonzero fill and duplicate-label provenance tests added. Independent review pending |
| F2 | 0.9.0 trim-and-fill fixture had eight explicit-side cases and five fields with loose tolerances | Expanded to 24 cases with auto direction, ties, augmented rows, intervals and heterogeneity; explicit tighter tolerances. Independent iteration evidence and M5/M6 sign-off remain open |
| F3 | 0.9.0 generator recorded controls it did not pass to R | Controls now passed; native augmented refit retained alongside an explicitly controlled independent refit. Regenerated with pinned packages; M6 sign-off pending |

F1 regression example (now expected to succeed):

```python
import math
import meta_analyze as ma

result = ma.meta_analysis(
    effect=[0.10, math.nan, 0.15, 0.18, 0.22, 0.24, 0.29, 0.33, 0.37, 0.95, 1.15],
    variance=[0.04, 0.02, 0.035, 0.03, 0.028, 0.025, 0.022, 0.02, 0.018, 0.012, 0.01],
    model="common",
    missing="drop",
)
filled = result.trim_and_fill(side="left")
assert filled.k0 == 0
assert filled.adjusted_result.k == 10
```

## September 2026 review disposition

The supplied code-review suggestions were checked against the implementation.
The following changes start batch B; they do not close the formal review gates.

| Suggestions | Disposition and evidence |
| --- | --- |
| A1–A2: REML/DL cancellation | Positive pair-product trace and anchored residuals; exact rational two-/three-study oracles, precision ratios through `1e300`, row permutations, leave-one-out and subgroups in `test_numerical_stability.py` |
| A3: trim-and-fill exclusions | Fixed as F1, retaining original exclusions in `original_result` and unique source/synthetic IDs in the adjusted table |
| A4–A5: identical effects / singleton intermediate fit | No imputation for identical effects with unavailable rank uncertainty; random intermediate fits below two studies raise `ConvergenceError`. Both estimators and models tested |
| B1–B4: numerical errors, zero HK, private table and validation | Domain exceptions at affected boundaries; zero HK covariance returns point intervals and unavailable joint tests with warnings; private DataFrame excluded from repr/equality; trim controls use domain exceptions |
| C1: H-squared | Existing definitions retained and explained in the methods guide; this convention remains an independent-review input |
| C2: subset corrections | Existing behavior documented; 32 workflow cases cover IV/MH, correction scopes, excluded rows, subgroup Wald tests, leave-one-out/cumulative recomputation and global provenance against direct calculations. No new warning or policy change |
| C3/C6: interpretation | Begg always includes the publication-bias caveat; Egger/Begg warn when using Peto one-step study effects |
| C4: changelog | 0.7 MH validation changes relabeled as breaking; categorical float matching stays under Changed with its compatibility-extension rationale |
| C5: sensitivity conventions | Preserve `estimate_change = deleted - original`, DFBETAS' opposite numerator and DataFrame-returning sensitivity summaries; explicit API-freeze review item |
| C7–C8: ranks / convergence | Document stable effect sorting before first ranks, limited tied-case reference coverage and success-only `converged=True`; nonconvergence still raises |
| C9: contour overlap | Significance regions are disjoint adjacent bands; path membership and legend-color tests added |
| C10: original-estimate marker | Deferred optional plotting enhancement; not a correctness or 1.0 gate |
| C11: small-study-effect ADR | Current four-test conventions recorded in ADR 0009, pending independent 1.0 ratification |
| C12: Peters metadata | R correction count restricted to included studies; regenerated fixture unchanged |
| C13: schemas and examples | Document derived trim-and-fill provenance and API method; execute README/documentation Python examples per page with plot rendering and report JSON checks. Fix missing setup, overwritten result types, inconsistent category names and invalid ratio references; only explicitly incomplete API signatures are exempt. Release-candidate validation under gate C4 remains pending |
| C14–C15: correlations | Tests cross-check Egger/Begg/trim-and-fill on Fisher's z and reject binary-only tests; document the transform required for generic meta-regression |
| C16–C17: warnings / observations | Retain outcome-specific wording and existing supported behavior; no speculative refactor |

For zero-residual HK regression, direct R checks with `metafor` 5.0-1
confirmed zero covariance and unavailable joint Wald inference. Exact rational
oracles validate the extreme-weight pooling cases where double-precision R
calculations are not an appropriate sole reference. See
[validation](validation.md) for fixture controls and tolerance changes.
