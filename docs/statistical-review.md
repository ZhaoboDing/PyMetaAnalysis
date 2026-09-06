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

These findings were identified during foundation work against 0.9.0. They are
not an independent audit and remain open until fixed and reviewed.

| ID | Observed evidence | Required disposition |
| --- | --- | --- |
| F1 | `trim_and_fill(side="left")` with the missing-row example below estimates no missing studies, then assigns 10 imputation flags to an 11-row table and raises pandas `ValueError` | Correct excluded-row handling; test zero/nonzero fill, original row IDs and provenance consistency; blocks M5/A2 |
| F2 | `trimfill_metafor.json` has only eight explicit-side cases and five output fields; numeric comparisons in `test_trim_fill.py` use `abs=1e-6` plus implicit pytest relative tolerance | Add independent automatic-direction, ties, imputed-row, interval/heterogeneity and iteration evidence; justify tolerances per output; blocks M5/M6 |
| F3 | The trim-and-fill generator records tolerance `1e-10` and 1000 iterations but passes neither to `rma()` or `trimfill()` | Reconcile actual R controls and metadata, regenerate separately and review numerical differences; blocks M6 |

Minimal F1 reproducer (the expected successful behavior is not yet implemented):

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
```

The foundation batch fixes the generator's output-path interface so reviewers
can write a candidate without replacing the golden artifact; it does not change
R computation or repair F1–F3. Address these findings at the start of batch B,
before treating the current trim-and-fill result/provenance behavior as frozen.
