# Changelog

All notable changes to PyMetaAnalysis will be documented in this file.

Changes planned for the next release accumulate under `Unreleased`.

## Unreleased

### Added

- common-effect Mantel-Haenszel risk-difference pooling with the
  Sato-Greenland-Robins sampling variance, explicit method metadata,
  sparse-table policy integration, and independent R `metafor` references.

### Fixed

- isolated builds temporarily cap Hatchling below 1.32 so the release
  workflow continues to produce Core Metadata 2.4 accepted by Twine 6.2;
  the cap can be removed once Twine validates Metadata 2.5.

## 0.5.0 - 2026-07-25

### Added

- random-effects inverse-variance results now provide opt-in Q-profile
  confidence intervals for tau-squared, tau, I-squared, and H-squared, with an
  explicit formal-empty-set flag at the constrained `[0, 0]` boundary.

### Changed

- sensitivity and influence workflows now borrow internal fitted buffers during
  refits instead of repeatedly materializing public defensive copies;
- Meta-regression stores its classic coefficient covariance alongside the
  selected inference covariance and reuses one shared precision-geometry
  implementation across fitting and diagnostics.
- independent `metafor` fixtures now cover categorical and multivariable
  influence diagnostics, no-intercept Riley prediction intervals, and explicit
  Mantel-Haenszel pooling correction; iterative failure paths have direct
  regression tests.

### Fixed

- inverse-variance means, heterogeneity statistics, and pooling and
  meta-regression tau-squared equations now use overflow-safe relative-weight
  calculations at the supported float64 boundary;
- subnormal variances and non-finite derived effects now raise explicit domain
  errors instead of leaking runtime warnings or returning invalid results;
- binary OR, RD, and Mantel-Haenszel arithmetic now avoids intermediate
  overflow for very large finite counts;
- the sensitivity guide no longer incorrectly states that Meta-regression
  Cook's distance and DFBETAS are unavailable.
- subgroup-differences tests now use classic model variances independently of
  Hartung-Knapp confidence-interval adjustments;
- leave-one-out and cumulative workflows now retain or skip, respectively,
  reduced Mantel-Haenszel fits that are not estimable instead of aborting the
  complete sensitivity analysis;
- forest and subgroup-forest plots now reject non-positive displayed
  coordinates before enabling a logarithmic axis;
- uncorrected risk-ratio analyses now accept a zero non-event cell when the
  study effect and sampling variance remain well defined.
- Hartung-Knapp random-effects prediction intervals now use the selected
  adjusted pooled-mean variance and are recorded as `HK-PR`, matching
  `metafor` Riley predictions;
- random-effects subgroup analyses now retain single-study subgroups through
  an explicit, warned common-effect fallback instead of failing the complete
  analysis;
- tagged releases now rerun the full branch-coverage test suite before
  distributions can be built and published.
- tau-squared methods and SMD variance conventions now use `None` as the
  context-sensitive default, so explicitly inapplicable settings raise domain
  errors instead of being silently ignored;
- duplicate study labels now add a row-position warning while preserving
  `row_id` as the unique audit key;
- report JSON now serializes `pd.NaT` study labels as `null` rather than the
  string `"NaT"`.
- Meta-regression with `missing="drop"` now determines complete-row exclusions
  before validating moderator values, so invalid values in already excluded
  rows cannot abort the analysis;
- cumulative analysis now rejects ambiguous string `order` selectors that
  exist in both source data and study results;
- empty inputs now report that at least one study row is required, and binary
  zero-cell errors identify when `correction_scope="none"` disables an
  otherwise positive correction.
- iterative tau-squared estimators now mark only an exact constrained zero as
  a boundary solution, rather than treating every positive root below `atol`
  as zero;
- the exported pooling and Meta-regression tau-squared estimators now reject
  insufficient study or residual degrees of freedom with domain-specific
  errors;
- the Mantel-Haenszel estimator now rejects empty and zero-total strata before
  division, preventing NaN propagation and misleading variance diagnostics.
- prediction-interval metadata is now `None` when too few studies prevent an
  interval from being calculated;
- categorical moderator encoding no longer conflates booleans, integers, and
  floating-point values through Python's cross-type numeric equality;
- CI now covers Python 3.14, Pages deployments are not cancelled mid-flight,
  and the credential-bearing PyPI publisher action is pinned to an immutable
  commit.

## 0.4.0 - 2026-07-23

### Added

- leave-one-out Meta-regression refits with model-level diagnostics,
  coefficient changes, explicit unidentifiable-deletion records, and preserved
  provenance.
- exact Meta-regression externally standardized residuals, Cook's distances,
  DFBETAS, transparent screening thresholds, and fixed-version R `metafor`
  cross-software fixtures.
- Meta-regression term VIF, moderator-level GVIF/GSIF, and weighted,
  column-scaled condition diagnostics with variance-decomposition proportions,
  heuristic-only flags, and R `metafor` cross-software fixtures.
- explicit named Meta-regression linear contrasts with nonzero null values,
  individual z/t inference, joint chi-squared/F tests, labeled coefficient
  matrices, and R `metafor` cross-software fixtures.
- opt-in Riley Meta-regression true-effect prediction intervals using
  `t_(k-p-1)`, with explicit residual-df validation, preserved refit
  configuration, and fixed-version R `metafor` boundary references.

## 0.3.0 - 2026-07-22

### Added

- pandas-first `meta_regression()` for numeric, explicitly encoded categorical,
  and multiple study-level moderators;
- common- and mixed-effects weighted regression with generalized DL, PM, and
  REML residual tau-squared estimators;
- normal, Hartung-Knapp, and safeguarded Hartung-Knapp coefficient inference,
  distribution-explicit moderator tests, residual heterogeneity, pseudo-R²,
  prediction, provenance, and structured reports;
- optional weighted bubble plots for intercept-containing Meta-regression fits
  with exactly one numeric moderator;
- independent R `metafor` fixtures covering numeric, categorical,
  multivariable, zero-tau-squared, missing-row, and small-sample cases;
- an executable Meta-regression notebook plus a multivariable performance
  baseline and expanded property, numerical-stability, and warning tests.

### Changed

- report schema 1.2 adds the `meta_regression` report type.

## 0.2.1 - 2026-07-17

### Fixed

- README documentation and repository links use absolute URLs so they resolve
  correctly when the project description is rendered on PyPI.

## 0.2.0 - 2026-07-16

### Added

- generic `meta_analysis()` accepts either sampling variances or standard
  errors, with explicit validation and auditable conversion provenance.

### Changed

- package author metadata identifies the project maintainer directly.

### Fixed

- GitHub Release creation receives explicit repository context in tag-driven
  release jobs.

## 0.1.0 - 2026-07-15

### Added

- pandas-first generic, binary, and continuous study-level meta-analysis APIs;
- common-effect and random-effects inverse-variance models;
- common-effect Mantel-Haenszel OR/RR pooling;
- REML, Paule-Mandel, and DerSimonian-Laird tau-squared estimators;
- normal, Hartung-Knapp, and safeguarded Hartung-Knapp confidence intervals;
- HTS random-effects prediction intervals;
- subgroup, leave-one-out, and cumulative workflows;
- optional Matplotlib forest, subgroup forest, and funnel plots;
- immutable results, diagnostics, provenance, Methods text, and JSON/Markdown
  reports;
- R `metafor` cross-software fixtures, property tests, and numerical edge-case
  coverage;
- explicit RD zero-variance boundary policy and heterogeneity-definition
  reporting;
- complete MkDocs user, methods, API, validation, limitation, and development
  documentation;
- R `meta`/`metafor` terminology and parameter mappings;
- machine-readable citation metadata and an executable end-to-end notebook;
- GitHub Pages and PyPI Trusted Publishing release workflows;
- release metadata, distribution-content, notebook-execution, and performance
  baseline tooling.

### Changed

- independent external statistical review is documented as a recommended
  validation activity rather than a release requirement;
- report schema 1.1 records `heterogeneity.i2_method`;
- random-effects I-squared/H-squared use tau-squared and typical within-study
  variance, while common-effect/MH analyses retain Q-based definitions;
- random-effects summaries provide method-selection notes for small-study and
  positive-heterogeneity cases.
