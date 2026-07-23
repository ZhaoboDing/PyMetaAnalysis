# Scope and limitations

PyMetaAnalysis is an early-stage library for conventional aggregate, study-
level univariate meta-analysis. This page makes unsupported or intentionally
deferred functionality explicit.

## Supported scope

- generic effects with known sampling variances;
- two-group binary OR, RR, and RD;
- two-group continuous MD and exact-corrected Hedges' g;
- common-effect and univariate random-effects inverse-variance models;
- common-effect Mantel-Haenszel OR/RR;
- DL, PM, and REML tau-squared estimators;
- normal and random-effects Hartung-Knapp mean intervals;
- HTS prediction intervals;
- independent subgroup fits and a formal subgroup-differences test;
- leave-one-out and cumulative repeated-fit workflows;
- forest and descriptive funnel plots;
- structured provenance and reports;
- study-level Meta-regression with numeric and explicitly encoded categorical
  moderators, generalized DL/PM/REML, and normal/Hartung-Knapp inference.

## Not currently implemented

- Peto odds ratios;
- Mantel-Haenszel risk differences or random-effects MH pooling;
- formula parsing, automatic interactions/splines, stepwise moderator
  selection, multiplicity correction, or arbitrary linear contrasts;
- multilevel, multivariate, network, dose-response, diagnostic-accuracy, or
  individual-participant-data meta-analysis;
- robust variance estimation or dependent-effect clustering;
- single proportions, incidence rates, correlations, or survival outcomes;
- Knapp-Hartung variants beyond the two documented choices;
- alternative prediction-interval methods;
- formal funnel-asymmetry, trim-and-fill, selection-model, or publication-bias
  procedures;
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

Meta-regression uses aggregate study-level moderators. Coefficients are
susceptible to ecological bias, confounding, measurement error, low power, and
post-hoc model selection. They do not establish individual-level associations
or causal effects. The package warns below ten studies but does not prohibit a
scientifically prespecified, full-rank model when `k > p`.

Funnel plots are descriptive. Their pseudo confidence limits exclude tau-
squared, and asymmetry does not establish publication bias.

Continuity corrections and RD boundary policies can materially affect sparse
binary analyses. They must be chosen in the review protocol and reported.

## Reproducibility limitations

Provenance records how the package interpreted supplied inputs. It does not
hash external data, capture the full environment, store preprocessing code, or
record scientific decisions made before the function call. A reproducible
workflow must version those artifacts separately.

## Stability and review status

The package version is currently `0.3.0`. Public APIs and serialized schemas
may change during the 0.x series. Pin versions in analysis environments and
inspect changelog/schema updates before upgrading.

The implementation is tested against independent R fixtures but has not yet
received a formal external statistical audit. See [validation](validation.md)
for the exact evidence and boundary.
