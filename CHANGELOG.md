# Changelog

All notable changes to PyMetaAnalysis will be documented in this file.

The project is pre-release. Until the first tagged release, changes accumulate
under `Unreleased`.

## Unreleased

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
  documentation.
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
