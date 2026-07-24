# Statistical methods

This page specifies the calculations implemented by PyMetaAnalysis. It is an
implementation reference, not guidance for choosing an estimand or replacing
a review protocol.

## Notation

For study `i`:

- `y_i` is the study effect on the model scale;
- `v_i` is its sampling variance;
- `w_i` is an analysis weight;
- `k` is the number of included studies;
- `tau^2` is the estimated between-study variance.

Treatment and control are ordered exactly as their argument names imply. A
positive difference measure means treatment minus control is positive.

## Inverse-variance pooling

Common-effect weights and the pooled estimate are:

```text
w_i = 1 / v_i
y_hat = sum(w_i y_i) / sum(w_i)
Var(y_hat) = 1 / sum(w_i)
```

Random-effects weights replace `v_i` with `v_i + tau^2`:

```text
w_i* = 1 / (v_i + tau^2)
mu_hat = sum(w_i* y_i) / sum(w_i*)
Var(mu_hat) = 1 / sum(w_i*)
```

Reported `normalized_weight` values equal the selected raw weights divided by
their sum. The pooled-estimate implementation rescales weights internally when
necessary to avoid overflow without changing their ratios.

## Between-study variance

Random-effects inverse-variance models require at least two included studies.
The default estimator is REML.

### DerSimonian-Laird

With common-effect weights, Cochran's Q, `df = k - 1`, and:

```text
C = sum(w_i) - sum(w_i^2) / sum(w_i)
tau^2_DL = max(0, (Q - df) / C)
```

DL is closed form and records zero iterations.

### Paule-Mandel

PM solves:

```text
Q(tau^2) = sum((y_i - mu_hat(tau^2))^2 / (v_i + tau^2)) = k - 1
```

If the equation is already non-positive at zero, the estimate is the zero
boundary. Otherwise PyMetaAnalysis brackets the non-negative root and solves it
with a bounded scalar root finder.

### Restricted maximum likelihood

REML solves the restricted-likelihood score equation over non-negative
`tau^2`. As with PM, a non-positive score at zero produces a boundary estimate;
otherwise a bracketed root is found iteratively.

`result.diagnostics` records convergence, iteration count, and whether the
solution reached the zero boundary. Failure to bracket or converge raises
`ConvergenceError`; no fallback estimator is selected silently.

## Confidence intervals

### Normal interval

The default interval is:

```text
estimate +/- z_(1 - alpha/2) * SE
```

This is available for common- and random-effects inverse-variance models and
for common-effect Mantel-Haenszel pooling.

### Hartung-Knapp intervals

For random-effects inverse-variance models:

```text
q_star = sum(w_i* (y_i - mu_hat)^2) / (k - 1)
Var_HK(mu_hat) = q_star / sum(w_i*)
CI = mu_hat +/- t_(k - 1, 1 - alpha/2) * sqrt(Var_HK)
```

`hartung_knapp` uses this variance without modification. If it is below the
classic variance, the result records a warning. `hartung_knapp_adhoc` instead
uses:

```text
max(Var_classic, Var_HK)
```

The variants are separate public choices. PyMetaAnalysis does not silently
apply the safeguard to an unmodified Hartung-Knapp request.

## Prediction interval

Random-effects inverse-variance models with at least three included studies
use the Higgins-Thompson-Spiegelhalter form:

```text
mu_hat +/- t_(k - 2, 1 - alpha/2)
          * sqrt(tau^2 + Var_classic(mu_hat))
```

The classic pooled-mean variance is used even when the confidence interval for
the mean uses Hartung-Knapp. No interval is returned below three studies. An
interval calculated with three or four studies carries an explicit uncertainty
warning.

## Heterogeneity and inconsistency

Cochran's Q always uses common-effect inverse-variance weights:

```text
Q = sum(w_i (y_i - y_hat_common)^2)
df = k - 1
p = P(chi-square_df >= Q)
```

For Mantel-Haenszel analyses, residuals are centered on the MH pooled estimate
while the individual-effect inverse variances supply the Q weights.

Common-effect and MH results use Q-based inconsistency:

```text
I^2 = max(0, (Q - df) / Q)
H^2 = Q / df
```

Random-effects results use the fitted `tau^2` and a typical within-study
variance:

```text
C = sum(w_i) - sum(w_i^2) / sum(w_i)
v_typical = (k - 1) / C
I^2 = tau^2 / (tau^2 + v_typical)
H^2 = 1 + tau^2 / v_typical
```

`result.i2_method` records `q_based` or `tau2_typical_variance`. I-squared is a
proportion internally and is formatted as a percentage in human-readable
output. With one study, Q is zero and its p-value, I-squared, and H-squared are
unavailable.

## Meta-regression

For `k` study effects, let `X` be the full-rank `k`-by-`p` design matrix. It
contains an intercept by default plus numeric moderator columns and treatment-
coded categorical terms:

```text
y = X beta + u + epsilon
epsilon ~ N(0, V),  V = diag(v_i)
u ~ N(0, tau^2 I)
```

Given residual tau-squared:

```text
W = diag(1 / (v_i + tau^2))
B = (X' W X)^(-1)
beta_hat = B X' W y
P = W - W X B X' W
residual df = k - p
```

The implementation uses stable linear solves rather than forming an explicit
matrix inverse. A rank-deficient design or `k <= p` is rejected; terms are not
silently removed and a pseudo-inverse is not used.

### Residual tau-squared

For generalized DL, define `W0` and `P0` at tau-squared zero:

```text
QE0 = y' P0 y
C = trace(P0)
tau^2_DL = max(0, (QE0 - (k-p)) / C)
```

PM solves:

```text
y' P(tau^2) y = k - p
```

REML solves the restricted score:

```text
y' P(tau^2)^2 y - trace(P(tau^2)) = 0
```

All estimators are constrained to non-negative tau-squared. PM and REML use a
bracketed scalar root; a non-positive equation at zero returns the boundary.
Failure raises `ConvergenceError` without estimator fallback.

### Coefficient and moderator inference

Normal inference uses covariance `B`, z coefficient tests, and a chi-squared
Wald test for one or more moderator terms.

For both Hartung-Knapp choices:

```text
q = y' P(tau^2) y / (k-p)
Cov_HK(beta_hat) = q B
```

Coefficient tests use a t distribution with `k-p` degrees of freedom. Joint
tests divide their Wald statistic by the number of tested terms and use an F
distribution with that numerator df and `k-p` denominator df.
`hartung_knapp_adhoc` replaces `q` with `max(1, q)`; unmodified
`hartung_knapp` retains `q` and warns when it is below one.

The global moderator test covers every non-intercept term.
`test_moderator(name)` covers every encoded term for one original moderator,
so a multi-level category receives a joint test.

### Residual heterogeneity and pseudo-R-squared

Residual heterogeneity uses:

```text
QE = y' P0 y
QE ~ chi-square_(k-p) under the null
```

Common models derive residual I-squared and H-squared from QE. Mixed models
use:

```text
v_typical = (k-p) / trace(P0)
I^2_residual = tau^2 / (tau^2 + v_typical)
H^2_residual = 1 + tau^2 / v_typical
```

For mixed models with an intercept, the same rows are refitted with an
intercept-only design and the same tau-squared estimator:

```text
R^2_raw = 1 - tau^2_model / tau^2_null
R^2 = max(0, R^2_raw)
```

The raw value is retained. Pseudo-R-squared is unavailable when the null
tau-squared is zero and is not equivalent to ordinary regression R-squared.

### Meta-regression prediction

For a new design vector `x0`:

```text
fitted = x0' beta_hat
Var_mean = x0' Cov(beta_hat) x0
```

Mean-effect intervals use the selected normal or `t_(k-p)` critical value. A
mixed model additionally reports a true-effect prediction interval:

```text
fitted +/- critical * sqrt(tau^2 + Var_mean)
```

The default critical value is standard normal for normal inference and
`t_(k-p)` for either Hartung-Knapp mode. The optional Riley rule uses
`t_(k-p-1)` and therefore requires `k-p >= 2`. Both are approximations because
tau-squared is estimated; the Riley option is not selected automatically.
Neither interval includes a sampling variance for a future observed effect.
See [ADR 0003](../adr/0003-meta-regression-prediction-intervals.md) for the
compatibility and validation decision.

### Meta-regression deletion diagnostics

For every included study `i`, `influence()` deletes that study and refits the
model with the same configuration, including a fresh residual tau-squared
estimate. Let `beta_(-i)`, `tau^2_(-i)`, and `q_(-i)` denote the deleted-model
coefficient estimate, residual variance estimate, and inference scale. The
deleted prediction and externally standardized residual are:

```text
pred_(-i) = x_i' beta_(-i)
e_(-i) = y_i - pred_(-i)
SE(e_(-i)) = sqrt(
    q_(-i) * (v_i + tau^2_(-i))
    + x_i' Cov(beta_(-i)) x_i
)
rstudent_i = e_(-i) / SE(e_(-i))
```

The inference scale is one for normal inference, the deleted-model
Hartung-Knapp scale for unmodified Hartung-Knapp, and its lower-bounded value
for the safeguarded variant.

Let `delta_i = beta_hat - beta_(-i)`. Cook's distance is:

```text
Cook_i = delta_i' Cov(beta_hat)^(-1) delta_i
```

For DFBETAS, the deleted tau-squared is inserted into the full design matrix:

```text
B_i = (X' W(tau^2_(-i)) X)^(-1)
DFBETAS_ij = delta_ij / sqrt(q_(-i) * B_i[j,j])
```

This matches the deletion definitions used by R `metafor` for the supported
independent-effect models. The implementation uses linear solves and does not
substitute a pseudo-inverse.

The output provides three transparent screening references:

```text
potential outlier: abs(rstudent_i) > z(0.975)
Cook's-distance flag: Cook_i > median(chi-square_p)
DFBETAS flag: any abs(DFBETAS_ij) > 1
```

They are heuristic review aids. The residual reference is pointwise rather
than multiplicity-adjusted, and no flag automatically removes a study.

### Meta-regression collinearity diagnostics

VIF/GVIF uses the correlation matrix `R` of the classic coefficient covariance
for non-intercept terms. A scalar Hartung-Knapp adjustment does not change this
correlation matrix. For a term set `S` with `m` encoded coefficients:

```text
GVIF(S) = det(R_SS) det(R_notS,notS) / det(R)
GSIF(S) = GVIF(S)^(1 / (2m))
```

For one coefficient, GVIF is the ordinary VIF and GSIF is its square-root
standard-error inflation factor. Categorical terms are grouped by their
original moderator. The library reports values without applying an automatic
5-or-10 cutoff.

Condition diagnostics use the actual fitted precision weights. Let `D` contain
the Euclidean column norms of `W^(1/2) X`, and define:

```text
Z = W^(1/2) X D^(-1)
Z = U diag(s_1, ..., s_p) V'
condition_index_d = s_1 / s_d
```

For coefficient `j` and singular dimension `d`, the variance-decomposition
proportion is:

```text
component_jd = V[j,d]^2 / s_d^2
proportion_jd = component_jd / sum_l(component_jl)
```

Column normalization makes these condition indices invariant to positive
changes of moderator units. The original unweighted design condition number is
also returned separately and remains scale-dependent.

A dimension is marked `concerning` only when its condition index exceeds 30
and more than 50% of the variance of at least two coefficients is concentrated
in that dimension. These conventional references are descriptive screening
heuristics; no term is removed or model refitted automatically.

### Meta-regression linear contrasts

Let `C` be a user-supplied, full-row-rank matrix with one column per fitted
coefficient, and let `d` be the supplied `rhs` vector. With fitted coefficient
vector `beta_hat` and its selected normal or Hartung-Knapp covariance `Sigma`:

```text
theta_hat = C beta_hat
Cov(theta_hat) = C Sigma C'
SE(theta_hat_j) = sqrt(Cov(theta_hat)[j,j])
test_j = (theta_hat_j - d_j) / SE(theta_hat_j)
```

Individual tests use the same distribution as coefficient inference: standard
normal for `inference_method="normal"` and `t_(k-p)` for either Hartung-Knapp
choice. Confidence intervals are centered on `theta_hat`, not on
`theta_hat - d`.

For `q` contrast rows, the joint Wald quantity is:

```text
W = (C beta_hat - d)' (C Sigma C')^(-1) (C beta_hat - d)
```

Normal inference compares `W` with chi-squared on `q` degrees of freedom.
Hartung-Knapp inference reports `W/q` against `F_(q,k-p)`. The implementation
uses a linear solve and rejects a rank-deficient contrast matrix instead of
using a pseudo-inverse.

Term-name mappings and labeled DataFrames are expanded to the complete fitted
term order with zero weight for unspecified terms. Individual p-values are
reported without multiplicity adjustment. The joint test evaluates the whole
prespecified set but is not a substitute for a protocol-appropriate
multiplicity strategy.

## Binary study effects

For a treatment/control 2-by-2 table:

| | Event | No event |
| --- | ---: | ---: |
| Treatment | `a` | `b` |
| Control | `c` | `d` |

### Odds ratio

```text
y_i = log(a d / (b c))
v_i = 1/a + 1/b + 1/c + 1/d
```

### Risk ratio

```text
y_i = log((a / (a + b)) / (c / (c + d)))
v_i = 1/a - 1/(a + b) + 1/c - 1/(c + d)
```

OR and RR results remain on the log model scale. Display properties exponentiate
them. Continuity-correction and double-zero/double-all behavior is specified in
[zero-event studies](../guides/zero-events.md).

### Risk difference

With `p_t = a / (a + b)` and `p_c = c / (c + d)`:

```text
y_i = p_t - p_c
v_i = p_t (1 - p_t) / (a + b) + p_c (1 - p_c) / (c + d)
```

Under the default zero-variance policy, the effect uses raw counts while the
variance uses corrected counts only for boundary tables.

## Mantel-Haenszel pooling

MH is implemented only for common-effect OR and RR. It uses raw tables by
default and has a continuity-correction option separate from the study-effect
correction.

For OR, with `n_i = a_i + b_i + c_i + d_i`:

```text
OR_MH = sum(a_i d_i / n_i) / sum(b_i c_i / n_i)
```

For RR, with treatment total `n1_i = a_i + b_i` and control total
`n0_i = c_i + d_i`:

```text
RR_MH = sum(a_i n0_i / n_i) / sum(c_i n1_i / n_i)
```

The model-scale estimate is the logarithm of the pooled ratio. Standard errors
use the Greenland-Robins equations implemented in the estimator. An undefined
raw pooled ratio raises `InvalidStudyDataError` and recommends an explicit MH
continuity correction.

## Continuous study effects

### Mean difference

```text
y_i = mean_treat - mean_control
v_i = sd_treat^2 / n_treat + sd_control^2 / n_control
```

The variance is unpooled and does not require equal group variances.

### Standardized mean difference

The pooled within-study variance and Cohen's d are:

```text
df = n_treat + n_control - 2
s_pooled^2 = ((n_treat - 1) sd_treat^2
              + (n_control - 1) sd_control^2) / df
d = (mean_treat - mean_control) / s_pooled
```

PyMetaAnalysis applies the exact gamma-function correction:

```text
J(df) = Gamma(df / 2) / (sqrt(df / 2) Gamma((df - 1) / 2))
g = J(df) d
```

The implemented `LS` sampling variance is:

```text
v_i = 1/n_treat + 1/n_control + g^2 / (2 (n_treat + n_control))
```

## Subgroup differences

Subgroup fits use the same selected model independently within each group. For
random effects, tau-squared is estimated separately for every subgroup and the
overall analysis.

The formal subgroup-differences test weights subgroup pooled estimates by the
inverse square of their pooled standard errors:

```text
W_g = 1 / SE_g^2
theta_bar = sum(W_g theta_g) / sum(W_g)
Q_between = sum(W_g (theta_g - theta_bar)^2)
df_between = number_of_groups - 1
```

`Q_between` is compared with a chi-squared distribution. This is not a test
obtained by comparing separate subgroup p-values.

## Numerical and software references

The implementation is independent Python code. Regression fixtures generated
by R `metafor`, hand calculations, invariants, and numerical edge cases are
described in [validation](../validation.md). Exact supported combinations and
defaults are listed in the [API reference](../reference/api.md).
