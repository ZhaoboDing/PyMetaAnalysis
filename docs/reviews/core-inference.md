# Core pooling and inference review packet

Prepared 2026-09-07 against main `57c1c1e`, with the corrections in this
packet's PR. Status: implementation evidence prepared; **independent review
pending**. An external reviewer must record the full candidate commit, their
identity, findings and sign-off using the [review record](../statistical-review.md).
This packet supplies evidence for M1/M2/M6 and the proposed D1 decision; it does
not close those gates. Outcome-specific effect-size formulas, sparse binary
estimators and regression require their own review packets.

## Assumptions and formula map

The intercept-only model treats studies as independent, sampling variances as
known positive quantities, and observed effects as approximately normal on the
analysis scale. A random-effects interpretation additionally assumes a common
distribution of true effects with variance tau-squared. Estimated sampling
variances, sparse data, dependence, selection and non-normal true effects can
invalidate nominal inference even when the calculations agree across software.

Here `w_i(t) = 1 / (v_i + t)`, `W(t) = sum(w_i(t))`,
`mu(t) = sum(w_i(t) * y_i) / W(t)`,
`Q(t) = sum(w_i(t) * (y_i - mu(t))^2)` and `df = k - 1`.

| Quantity | Formula / convention | Implementation | Evidence in `tests/` |
| --- | --- | --- | --- |
| Pooling | Common: t=0; random: t=estimated tau-squared; classic mean variance 1/W(t) | `estimators/inverse_variance.py`, `heterogeneity.weighted_mean` | Generic R fixture; common/DL statsmodels values; new 24-case common and 216-fit random comparisons |
| DL | max(0, (Q(0)-df)/C), C=W(0)-sum(w_i(0)^2)/W(0) | `estimators/tau2.py`; positive pair-product C | Exact rational extreme-precision oracles and permutations in `test_numerical_stability.py` |
| PM | Nonnegative root Q(t)=df, or zero at the boundary | `estimators/tau2.py` | Generic and new R references; equal-variance and two-study formulas; convergence tests |
| REML | Root sum(w_i(t)^2 * residual_i(t)^2) = W(t)-sum(w_i(t)^2)/W(t), or zero at the boundary | `estimators/tau2.py` | Generic and new R references; exact two-study boundary tests; anchored residual score tests |
| Q / inconsistency | Q=Q(0), chi-squared df; common I2=max(0,1-df/Q), zero when Q=0, H2=Q/df (untruncated). Random v_typical=df/C, I2=tau2/(tau2+v_typical), H2=1+tau2/v_typical | `heterogeneity.py` | R Q/p/I2/H2; direct common calculations; extreme-precision and large-variance tests |
| Mean CI | Normal: z times classic SE. HK: V_HK=Q(t)/(df*W(t)), t critical with df. Safeguard: max(V_HK,1/W(t)), same t critical | `estimators/inverse_variance.py` | New R references for all three methods; exact residual-variance oracles at large locations; existing zero-variance warnings |
| Prediction | mu(t) +/- t critical with k-2 df times sqrt(t + V_selected); unavailable at k=2; warn at k=3/4 | `estimators/inverse_variance.py`; ADR 0004 | Explicit metafor Riley predictions at k=3/5/10 for every fit; equal-variance closed-form oracle |
| Q-profile | Invert Q(t) at upper/lower chi-squared quantiles, df=k-1; constrain endpoints to nonnegative values and label formal empty sets | `heterogeneity.py`, `results.tau2_confidence_interval` | R QP bounds/empty status, search-limit evidence; 48 equal-variance closed-form cases; existing inversion and transform tests |

Formula conventions are recorded in [ADR 0002](../adr/0002-statistical-policy.md)
and [ADR 0004](../adr/0004-hartung-knapp-prediction-intervals.md). The
[metafor model reference](https://wviechtb.github.io/metafor/reference/rma.uni.html)
documents the estimators and separate `knha`/`adhoc` inference. Its
[prediction reference](https://wviechtb.github.io/metafor/reference/predict.rma.html)
supports the explicit Riley comparison. Q-profile's exactness requires the
normal model and known sampling variances; it is approximate when those
assumptions are approximate. See the
[Q-profile reference and cited methodology](https://wviechtb.github.io/metafor/reference/confint.rma.html).
Online references checked 2026-09-07; executable references use the pinned
versions below, independently of subsequent documentation updates.

## Reproducible comparison design

`tests/reference/generate_core_inference_metafor.R` constructs the complete
Cartesian product of k=2/3/5/10, balanced/unbalanced sampling variances, and
identical/low-spread/high-spread effects. Balanced variance is 0.2; unbalanced
variances are geometrically spaced from 0.02 to 2. Effects are respectively
all 0.3, evenly spaced from -0.01 to 0.01, or from -2 to 2. These are deterministic
stress cases, not representative samples or simulated coverage probabilities.

Every dataset has a common fit, nine random fits (DL/PM/REML crossed with
normal/HK/safeguarded HK), and a Q-profile interval. Fields include the estimate,
SE, mean CI, tau-squared, normalized weights, Q/p/I2/H2 and eligible prediction
intervals. The fixture preserves R's empty-set status and an independently
calculated Q(0) empty-set criterion. R predictions at k=2 are deliberately not
requested because the Python public contract does not offer that interval.

The generator enforces R 4.6.1, metafor 5.0-1 and jsonlite 2.0.0. It passes and
records PM `tol=1e-12`, REML `threshold=1e-12`, and QP `tol=1e-12`; each iterative
algorithm gets `maxiter=1000`. QP additionally uses `tau2.max=100000`. Python
fits and QP comparisons explicitly use `atol=1e-12`; this does not change public
defaults. Regenerate to a separate path, inspect and compare before replacement:

```console
Rscript tests/reference/generate_core_inference_metafor.R .release-smoke/core-inference-candidate.json
python -m pytest tests/test_core_inference_review.py tests/test_r_references.py tests/test_numerical_stability.py tests/test_api.py tests/test_estimators.py tests/test_properties.py
```

`tests/test_core_inference_review.py` uses `rtol=5e-13, atol=5e-15` for
closed-form fit arithmetic and Q/p, `2e-10, 2e-11` for iterative fits and their derived
inconsistency statistics, and `2e-9, 1e-10` for prediction/QP bounds. The latter
allow solver error propagated into tails; a near-zero statistic still has an
explicit absolute bound. Mean HK CI endpoints additionally allow
`5e-11 * R_t_critical * R_standard_error` absolute error for the separately
tested t-quantile approximation in older supported SciPy versions (REF2 below).
This allowance does not apply to SE, tau-squared, weights or inconsistency.
Independent rational oracles
use near-machine-precision SE checks. Equal-variance QP oracles use a relative
bound of `2e-10` and an absolute bound scaled with squared effect units.

## Findings and disposition

| ID | Reproducer and finding | Resolution / scope |
| --- | --- | --- |
| CI1 | Effects [0,2], variances [4,4], ordinary HK: SE=1. Adding 1e16 to both effects gave SE=sqrt(2), although both input differences remained exact | Center residuals on an observed effect before averaging. Exact rational cases cover both HK methods, all tau estimators, positive/negative shifts and zero/positive tau-squared. Returned locations and interval endpoints remain subject to ordinary float rounding |
| CI2 | Squaring residual 1e160 before applying weight 1e-30 overflowed although its weighted contribution was finite | Apply square-root weights and residual df before squaring. A focused CI arithmetic test separates this behavior from tau-estimation range restrictions |
| H1 | Three equal variances of 1e308 with tau-squared 1e307 produced I2=0, H2=1; correct values are 1/11 and 1.1 | Divide the variance scale by the scaled trace before multiplying by df; test k=3/5/10. Statistical definition unchanged |
| REF1 | Default R QP search returned upper number 100 for four high-spread k=2/3 cases, without an R warning; `ub.sign` was `>` | Retain default-search endpoints and signs; compare Python with an explicitly extended R search. For balanced k=2 the actual upper bound is about 8145.8662, verified by the equal-variance formula. This was reference extraction/search scope, not a Python interval bug |
| REF2 | Initial CI used the closed-form DL arithmetic tolerance for t quantiles too; Python 3.10 CI and minimum dependencies failed 30 HK endpoint comparisons each | Isolated SciPy 1.15.3 t quantiles differ from R by 2.026e-11, -1.239e-11 and 2.471e-11 relatively at df=1/2/9; the minimum-dependency CI shows about 3.761e-11 at df=2. Add explicit R quantiles, closed-form df=1/2 checks and a 5e-11 quantile budget propagated only into CI endpoints. Keep arithmetic tolerances unchanged |

For equal variances v, the independent QP oracle is
`sum((y_i-mean(y))^2) / chi2_quantile - v`. Tests cover positive and zero lower
bounds, formal empty sets, and effect multipliers 0.001/1/1000. The absolute
root tolerance is reduced for small squared units; users should not expect
one absolute solver tolerance to be scale-free. Existing tests separately
cover max-iteration failure, unsupported inputs, transformed intervals,
point-estimator independence, and extreme precision ratios through 1e300.

## Review still required

The proposed [default-CI recommendation](../adr/0008-stable-inference-default.md)
uses these examples to illustrate behavior, not to establish coverage or choose
methods after inspecting significance. External review must assess the model
assumptions, primary methodology, tolerances, warning policies and proposed
compatibility tradeoff. Regression residual degrees of freedom and prediction
rules remain a separate audit; existing regression fixtures are supporting
evidence, not acceptance under this packet.

All finite float inputs are not guaranteed to produce representable intermediate
or final quantities. The domain/convergence tests document selected numerical
limits; these cases do not establish exhaustive float-range support. Package
version remains 0.9.0 and normal remains the current default. Before accepting
M1/M2/M6/D1, review the final PR commit and attach the completed sign-off record.
