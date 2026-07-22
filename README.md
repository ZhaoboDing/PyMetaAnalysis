# PyMetaAnalysis

[![CI](https://github.com/ZhaoboDing/PyMetaAnalysis/actions/workflows/ci.yml/badge.svg)](https://github.com/ZhaoboDing/PyMetaAnalysis/actions/workflows/ci.yml)
[![Documentation](https://github.com/ZhaoboDing/PyMetaAnalysis/actions/workflows/pages.yml/badge.svg)](https://zhaoboding.github.io/PyMetaAnalysis/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ZhaoboDing/PyMetaAnalysis/blob/main/LICENSE)

PyMetaAnalysis is an early-stage, pandas-first Python library for conventional
study-level meta-analysis. It accepts DataFrames, NumPy arrays, and ordinary
Python sequences, then returns immutable, auditable result objects containing
study effects, exclusions, weights, method choices, diagnostics, provenance,
and structured reports.

> **Early-stage:** the API may change during the 0.x series. Consequential
> results should be reviewed against the analysis protocol and independently
> checked.

## Install

```console
python -m pip install PyMetaAnalysis
```

Install optional Matplotlib plotting support with:

```console
python -m pip install "PyMetaAnalysis[plot]"
```

The distribution name is `PyMetaAnalysis`; the import name is
`meta_analyze`.

## Quick start

```python
import meta_analyze as ma

result = ma.meta_analysis(
    effect=[0.12, 0.35, -0.08, 0.21],
    variance=[0.04, 0.06, 0.03, 0.05],
    study=["Trial A", "Trial B", "Trial C", "Trial D"],
    model="random",
    tau2_method="REML",
)

print(result.summary())
print(result.study_results)
```

DataFrame column names work directly:

```python
result = ma.meta_analysis(
    studies,
    effect="effect",
    variance="variance",
    study="citation",
    subgroup="region",
)
```

Omit `study=` to use the DataFrame index. Supplying `subgroup=` returns a
dedicated result containing group fits, the overall fit, and a formal test for
subgroup differences.

Study-level moderators use the dedicated Meta-regression entry point:

```python
regression = ma.meta_regression(
    studies,
    effect="effect",
    standard_error="se",
    moderators=["mean_age", "region"],
    categorical={"region": ["Europe", "Asia", "North America"]},
)
```

Categorical levels are explicit and ordered; the first value is the treatment-
coding reference. Moderator coefficients describe study-level associations,
not individual-level or causal effects.

## Supported analyses

| Input | Effects | Pooling/models |
| --- | --- | --- |
| Effect + sampling variance or standard error | Generic | Common/random inverse variance |
| Two-group events + totals | OR, RR, RD | Common MH OR/RR; common/random IV |
| Two-group means + SDs + sizes | MD, Hedges' g | Common/random inverse variance |
| Effect + variance/SE + moderators | Generic | Common/mixed Meta-regression |

Random-effects inverse-variance models support REML (default), Paule-Mandel,
and DerSimonian-Laird tau-squared estimators. Mean confidence intervals support
the normal default plus unmodified and safeguarded Hartung-Knapp variants.
Eligible random-effects fits include an HTS prediction interval.

Generic analyses accept exactly one of `variance=` or `standard_error=`.
Standard errors are squared internally and the conversion is recorded in the
result provenance.

Sparse binary behavior is explicit: study-level and Mantel-Haenszel continuity
corrections are separate, relative-effect double-zero/double-all rows remain
visible as exclusions, and RD exposes
`rd_zero_variance="correct" | "exclude"`.

## Inspect and report

```python
result.estimate
result.display_estimate
result.ci
result.tau2
result.i2
result.i2_method
result.diagnostics
result.provenance

methods_text = result.method_details()
report = result.report()
payload = report.to_dict()
json_text = report.to_json()
markdown = report.to_markdown()
```

OR and RR remain on the log model scale in auditable numeric attributes;
`display_estimate`, `display_ci`, and `display_prediction_interval` provide
exponentiated ratios.

Rows excluded by missing-value or sparse-data policies remain in
`study_results` with a stable `row_id`, `included=False`, and an
`exclusion_reason`.

## Sensitivity and plots

```python
leave_one_out = result.leave_one_out().to_dataframe()
cumulative = result.cumulative(order="publication_year").to_dataframe()

ax = result.forest(show_prediction_interval=True)
ax = result.funnel()
```

Plotting methods return Matplotlib axes and never call `show()`. Funnel plots
are descriptive small-study-effect diagnostics, not proof of publication bias.

## Documentation

The complete documentation is published at
[zhaoboding.github.io/PyMetaAnalysis](https://zhaoboding.github.io/PyMetaAnalysis/).

- [Installation](https://zhaoboding.github.io/PyMetaAnalysis/installation/)
- [Getting started](https://zhaoboding.github.io/PyMetaAnalysis/getting-started/)
- [Input data and row decisions](https://zhaoboding.github.io/PyMetaAnalysis/guides/input-data/)
- [Generic](https://zhaoboding.github.io/PyMetaAnalysis/guides/generic-effects/), [binary](https://zhaoboding.github.io/PyMetaAnalysis/guides/binary-outcomes/), and [continuous](https://zhaoboding.github.io/PyMetaAnalysis/guides/continuous-outcomes/) guides
- [Meta-regression](https://zhaoboding.github.io/PyMetaAnalysis/guides/meta-regression/)
- [Choosing methods](https://zhaoboding.github.io/PyMetaAnalysis/guides/method-selection/) and [statistical formulas](https://zhaoboding.github.io/PyMetaAnalysis/methods/statistical-methods/)
- [Sensitivity analysis](https://zhaoboding.github.io/PyMetaAnalysis/guides/sensitivity-analysis/) and [plotting](https://zhaoboding.github.io/PyMetaAnalysis/guides/plotting/)
- [Public API](https://zhaoboding.github.io/PyMetaAnalysis/reference/api/), [result objects](https://zhaoboding.github.io/PyMetaAnalysis/reference/results/), and [report schema](https://zhaoboding.github.io/PyMetaAnalysis/reference/report-schema/)
- [Validation strategy](https://zhaoboding.github.io/PyMetaAnalysis/validation/) and [scope/limitations](https://zhaoboding.github.io/PyMetaAnalysis/limitations/)
- [Citation guidance](https://zhaoboding.github.io/PyMetaAnalysis/citation/)
- [R `meta`/`metafor` mapping](https://zhaoboding.github.io/PyMetaAnalysis/guides/r-interoperability/)

An executable [end-to-end notebook](https://github.com/ZhaoboDing/PyMetaAnalysis/blob/main/examples/quickstart.ipynb) uses synthetic
data to demonstrate analysis, provenance, reporting, sensitivity, and plotting.

Build the complete site locally with:

```console
python -m pip install ".[docs]"
python -m mkdocs serve
```

## Validation status

The test suite combines hand calculations, statistical invariants, numerical
edge cases, and committed R `metafor` reference fixtures. CI covers Python
3.10–3.13, declared dependency lower bounds, strict typing/linting, docs, and
distribution builds.

This is independent cross-software validation, not a formal external
statistical audit. See
[validation](https://zhaoboding.github.io/PyMetaAnalysis/validation/) for exact
coverage.

## Contributing

See
[CONTRIBUTING.md](https://github.com/ZhaoboDing/PyMetaAnalysis/blob/main/CONTRIBUTING.md)
and the full
[development guide](https://zhaoboding.github.io/PyMetaAnalysis/development/).
Statistical changes require formula
documentation, boundary tests, and an independent comparison where available.

Security-sensitive reports should follow
[SECURITY.md](https://github.com/ZhaoboDing/PyMetaAnalysis/blob/main/SECURITY.md).

## License

[MIT](https://github.com/ZhaoboDing/PyMetaAnalysis/blob/main/LICENSE)
