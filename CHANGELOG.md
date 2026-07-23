# Changelog

All notable changes to PyMetaAnalysis will be documented in this file.

Changes planned for the next release accumulate under `Unreleased`.

## Unreleased

### Added

- leave-one-out Meta-regression refits with model-level diagnostics,
  coefficient changes, explicit unidentifiable-deletion records, and preserved
  provenance.

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
