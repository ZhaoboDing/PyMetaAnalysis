# Meta-regression and diagnostics review packet

Prepared 2026-09-14 against main `debf3bc`, with the corrections in this
packet's PR. Status: implementation evidence prepared; **independent review
pending**. An external reviewer must record the candidate commit, identity,
findings and sign-off using the [review record](../statistical-review.md).
This packet supplies evidence for M4 and M6. It does not accept those gates or
change a statistical default.

## Scope and assumptions

The packet covers common- and mixed-effects inverse-variance meta-regression,
numeric and explicitly treatment-coded categorical moderators, generalized
DL/PM/REML residual tau-squared, normal and Hartung-Knapp inference, moderator
and contrast Wald tests, mean and true-effect prediction, residual
heterogeneity, pseudo-R-squared, exact deletion diagnostics, VIF/GVIF, and
weighted condition diagnostics.

Study effects are independent and approximately normal on the analysis scale,
with known positive sampling variances. The design matrix is prespecified and
full rank, and mixed-effects models assume one common residual tau-squared.
Moderator coefficients are study-level associations. The deterministic checks
do not address ecological bias, model selection, multiplicity, measurement
error in moderators, correlated effects, or the operating characteristics of a
particular scientific design.

## Formula and evidence map

Let `X` be the `k`-by-`p` design, `W(t)=diag(1/(v_i+t))`,
`B(t)=(X'W(t)X)^(-1)`, and
`P(t)=W(t)-W(t)X B(t) X'W(t)`. Residual degrees of freedom are `k-p`.

| Quantity | Formula / convention | Implementation | Evidence |
| --- | --- | --- | --- |
| Coefficients | `beta_hat(t)=B(t)X'W(t)y`; classic covariance `B(t)` | `estimators/meta_regression.py` | Hand WLS, intercept-only pooling equivalence, main R fixture, unit and row invariants |
| Generalized DL | `max(0, (y'P(0)y-(k-p))/trace(P(0)))` | `estimate_meta_regression_tau2()` | R fixture plus exact `Fraction` oracle through a `1e300` precision ratio |
| Generalized PM | Nonnegative root `y'P(t)y=k-p` | `estimate_meta_regression_tau2()` | R fixture, equation residual, boundary and convergence tests |
| Generalized REML | Nonnegative root `y'P(t)^2y=trace(P(t))` | `estimate_meta_regression_tau2()` | R fixture, equation residual, boundary tests, and independent no-intercept R solve |
| Coefficient inference | Normal uses z/chi-squared; HK uses `q=y'P(t)y/(k-p)` with t/F; safeguarded HK uses `max(1,q)` | `fit_meta_regression()` | R `z`, `knha`, and `adhoc` fits; direct covariance and exact-zero tests |
| Prediction | Mean variance `x0'Cov(beta)x0`; mixed true-effect variance adds tau-squared; default critical follows coefficient inference, Riley uses `t_(k-p-1)` | `MetaRegressionResult.predict()`; ADR 0003 | Numeric, multivariable, zero-tau and no-intercept R predictions |
| Deletion diagnostics | Refit every omitted study; externally standardized residuals, Cook distance and DFBETAS use deleted-model quantities | `regression_sensitivity.py` | Eight R model configurations, failed-deletion and row-identity tests |
| Collinearity | Covariance-correlation VIF/GVIF; SVD of column-normalized `W^(1/2)X` for condition indices and variance proportions | `regression_collinearity.py` | Common/REML R VIF fixtures and direct SVD/unit/row checks |
| Contrasts | `C beta=d`; covariance `C Sigma C'`; individual z/t and joint chi-squared or F | `regression_contrasts.py` | Four R inference configurations, direct matrix calculations and reparameterization tests |

The official [`metafor::rma.uni` reference](https://wviechtb.github.io/metafor/reference/rma.uni.html)
documents mixed-effects moderator models, residual heterogeneity, omnibus
tests, and the separate `knha` and safeguarded `adhoc` choices. The official
[`predict.rma` reference](https://wviechtb.github.io/metafor/reference/predict.rma.html)
documents fitted moderator values and true-effect prediction intervals. The
[`influence.rma.uni` reference](https://wviechtb.github.io/metafor/reference/influence.rma.uni.html),
[`vif` reference](https://wviechtb.github.io/metafor/reference/vif.html), and
[`anova.rma` reference](https://wviechtb.github.io/metafor/reference/anova.rma.html)
define the comparison targets for deletion diagnostics, coefficient inflation,
and linear hypotheses. Online references were checked 2026-09-14; executable
results use the pinned environment below.

## Reproducible comparison design

Four existing fixtures form one regression evidence set:

- `meta_regression_metafor.json` covers a 15-study numeric, categorical, and
  multivariable dataset plus zero-tau, missing-row and small-sample boundaries.
  It crosses DL/PM/REML and normal/HK variants and records coefficients,
  covariance, weights, leverage, heterogeneity, tests, and predictions.
- `meta_regression_influence_metafor.json` records deleted residuals, their
  uncertainty, Cook distances, and DFBETAS for eight common/mixed,
  numeric/categorical, and normal/HK configurations.
- `meta_regression_collinearity_metafor.json` records term VIF and grouped
  moderator GVIF/GSIF for common and REML fits.
- `meta_regression_contrasts_metafor.json` records three prespecified linear
  hypotheses and their joint test under common, REML, and both HK variants.

Each adjacent generator now requires R 4.6.1, `metafor` 5.0.1, and `jsonlite`
2.0.0 and stops on a mismatch. Iterative fits pass and record tolerance
`1e-10` and 1,000 iterations. Regenerate to separate paths and compare before
replacing committed artifacts:

```console
Rscript tests/reference/generate_meta_regression_metafor.R .release-smoke/meta-regression-candidate.json
Rscript tests/reference/generate_meta_regression_influence_metafor.R .release-smoke/meta-regression-influence-candidate.json
Rscript tests/reference/generate_meta_regression_collinearity_metafor.R .release-smoke/meta-regression-collinearity-candidate.json
Rscript tests/reference/generate_meta_regression_contrasts_metafor.R .release-smoke/meta-regression-contrasts-candidate.json
python -m pytest tests/test_meta_regression_review.py tests/test_meta_regression.py tests/test_regression_sensitivity.py tests/test_regression_influence.py tests/test_regression_collinearity.py tests/test_regression_contrasts.py tests/test_r_references.py
```

Main-fit comparisons use `rtol=5e-13, atol=5e-15` for closed calculations and
`2e-10, 2e-11` for iterative quantities, with `2e-9, 1e-10` for values derived
from an iterative root. VIF/GVIF uses `2e-10, 2e-12`; deletion and contrast
fixtures use `2e-8, 1e-10` because repeated R refits and downstream matrix
operations accumulate the recorded iterative differences. The new exact DL
oracle uses `3e-14` relative tolerance and checks all row permutations. Extreme
tests fail on unexpected `RuntimeWarning`.

For no-intercept mixed regression, `metafor::rma.uni` centers outcomes while
iterating REML, which changes a through-origin model. The generator separately
solves the standard raw-outcome score and cross-checks `metafor::rma.mv`; Python
compares with that fit. The fixture retains the `rma.uni` estimate for audit.
Its no-intercept `QE` is also excluded because `metafor` reports a different
convention; PyMetaAnalysis reports the weighted residual sum of squares. These
are intentional differences requiring reviewer sign-off.

## Findings and disposition

| ID | Reproducer and finding | Resolution / scope |
| --- | --- | --- |
| MR1 | With variances `[1e-30,1e-30,1,1]` and two moderators columns spanned by the high-precision rows, generalized DL subtracted nearly equal projection traces and raised `ZeroDivisionError` | Use column-scaled weighted QR, evaluate `sum(w_i(1-h_i))`, and recover high-leverage complements from positive determinant ratios. Exact rational cases cover ratios through `1e300`, all 24 row orders, and deletion refits |
| MR2 | Multiplying one moderator by `1e100` or `1e-100` made a mathematically equivalent, representable model fail the raw-unit rank check | Check rank after power-of-two column scaling, solve in scaled coordinates, then return coefficients and covariance in the supplied units. Fits, predictions, contrasts, and condition diagnostics are checked in both directions for common and all mixed estimators |
| MR3 | Adding exactly representable `1e16` to every effect in an intercept model changed the slope, residual QE, and tau-squared | Anchor effects on a canonical observed value before solving whenever the design contains a constant column. DL/PM/REML residual geometry is checked after a shift; no-intercept semantics remain unchanged |
| MR4 | Multiplying valid contrast rows by `1e200` and `1e-200` caused raw rank and covariance products to overflow or underflow although the tests are invariant to nonzero row scaling | Validate and evaluate contrasts in power-of-two row-scaled coordinates, then restore reported estimates, SEs, and intervals to requested units |

The numerical changes preserve the model equations, public fields, defaults,
and categorical coding. Returned raw design condition numbers remain
unit-dependent by definition; weighted scaled condition diagnostics remain the
unit-invariant diagnostic. Final coefficients, covariance entries, and
statistics must still be representable as finite float64 values.

## Review still required

An independent reviewer must assess model assumptions, design/rank policy,
generalized tau-squared equations, no-intercept differences, HK and prediction
degrees of freedom, deletion formulas, diagnostic thresholds, contrast
inference, fixture tolerances, and all findings on the final candidate commit.
The opposite signs of sensitivity `estimate_change` (deleted minus original)
and DFBETAS (original minus deleted) remain preserved public contracts for the
API-freeze decision. Package version remains 0.9.0; normal regression inference,
REML, and all other statistical defaults remain unchanged.
