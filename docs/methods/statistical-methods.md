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
