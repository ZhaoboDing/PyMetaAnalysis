# Scope and limitations

PyMetaAnalysis is an early-stage library for conventional aggregate, study-
level univariate meta-analysis. This page makes unsupported or intentionally
deferred functionality explicit.

## Supported scope

- generic effects with known sampling variances;
- two-group binary OR, RR, and RD;
- two-group continuous MD and exact-corrected Hedges' g;
- independent study-level Pearson correlations pooled on Fisher's z scale;
- common-effect and univariate random-effects inverse-variance models;
- common-effect Mantel-Haenszel OR/RR/RD;
- common-effect Peto one-step OR;
- DL, PM, and REML tau-squared estimators;
- Q-profile confidence intervals for tau-squared and its monotonic tau,
  I-squared, and H-squared transformations in random-effects inverse-variance
  models;
- normal and random-effects Hartung-Knapp mean intervals;
- HTS and Hartung-Knapp Partlett-Riley (`HK-PR`) prediction intervals;
- independent subgroup fits and a formal subgroup-differences test, with a
  warned common-effect representation when a random-effects subgroup contains
  only one included study;
- leave-one-out and cumulative repeated-fit workflows;
- forest and descriptive funnel plots;
- structured provenance and reports;
- study-level Meta-regression with numeric and explicitly encoded categorical
  moderators, generalized DL/PM/REML, normal/Hartung-Knapp inference, and
  exact leave-one-out refits plus externally standardized residual, Cook's
  distance, DFBETAS, VIF/GVIF, weighted condition diagnostics, and explicit
  linear contrasts;
- default normal/`t_(k-p)` and opt-in Riley `t_(k-p-1)` true-effect
  prediction intervals for mixed-effects Meta-regression.

## Not currently implemented

- random-effects Mantel-Haenszel pooling;
- formula parsing, automatic interactions/splines, stepwise moderator
  selection, automatic pairwise contrasts, or multiplicity correction;
- Q-profile confidence intervals for residual tau-squared in Meta-regression;
- multilevel, multivariate, network, dose-response, diagnostic-accuracy, or
  individual-participant-data meta-analysis;
- robust variance estimation or dependent-effect clustering;
- single proportions, incidence rates, raw-correlation (`COR`) pooling, or
  survival outcomes;
- dependent correlations, within-study correlation matrices, partial or rank
  correlations, and corrections for measurement unreliability;
- Knapp-Hartung variants beyond the two documented choices;
- prediction-interval methods beyond the documented default and Riley rules;
- formal funnel-asymmetry, trim-and-fill, selection-model, or publication-bias
  procedures;
- Meta-regression DFFITS, covariance ratios, influence plots, or simulated and
  re-estimated VIF variants;
- automatic conversion from confidence intervals, p-values, or raw papers to
  study effects;
- risk-of-bias assessment, certainty grading, protocol management, or study
  screening.

Requests for unsupported combinations raise `UnsupportedMethodError` instead
of silently selecting a different estimator.

## Statistical limitations

Random-effects inference can be unstable with few studies. PyMetaAnalysis
provides explicit notes and alternative CI methods, but no interval method
removes the underlying information limit.

Tau-squared, I-squared, Q, and prediction intervals describe different aspects
of heterogeneity and should not be interpreted as interchangeable decision
rules. A non-significant Q test is not evidence of homogeneity.

Subgroup analyses estimate tau-squared independently within each random-
effects subgroup. This differs from fitting a Meta-regression with categorical
moderators and a single residual tau-squared.

Correlation pooling assumes one independent Pearson correlation per study.
Different correlations from the same participants require covariance-aware
multivariate, multilevel, or robust methods; assigning different study labels
does not make those effects independent. Fisher's z is also an approximation,
especially in very small samples or when distributional assumptions behind
the reported Pearson correlation are poor.

Meta-regression uses aggregate study-level moderators. Coefficients are
susceptible to ecological bias, confounding, measurement error, low power, and
post-hoc model selection. They do not establish individual-level associations
or causal effects. The package warns below ten studies but does not prohibit a
scientifically prespecified, full-rank model when `k > p`.

Funnel plots are descriptive. Their pseudo confidence limits exclude tau-
squared, and asymmetry does not establish publication bias.

Continuity corrections and RD boundary policies can materially affect sparse
binary analyses. They must be chosen in the review protocol and reported.
Peto OR is an approximation with a restricted applicability region; the
package warns but cannot determine whether a dataset satisfies the substantive
rare-outcome, balanced-arm, and modest-effect conditions.

## Reproducibility limitations

Provenance records how the package interpreted supplied inputs. It does not
hash external data, capture the full environment, store preprocessing code, or
record scientific decisions made before the function call. A reproducible
workflow must version those artifacts separately.

## Stability and review status

The package version is currently `0.6.0`. Public APIs and serialized schemas
may change during the 0.x series. Pin versions in analysis environments and
inspect changelog/schema updates before upgrading.

The implementation is tested against independent R fixtures but has not yet
received a formal external statistical audit. See [validation](validation.md)
for the exact evidence and boundary.
