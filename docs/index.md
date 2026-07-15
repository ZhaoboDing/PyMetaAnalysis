# PyMetaAnalysis

PyMetaAnalysis is a pandas-first Python library for conventional study-level
meta-analysis. It accepts DataFrames, NumPy arrays, and ordinary Python
sequences, then returns inspectable result objects that retain study-level
effects, exclusions, weights, method choices, and diagnostics.

!!! warning "Pre-release software"

    The public API is still under development and may change before version
    0.1. Statistical results should be reviewed in the context of the analysis
    protocol and independently checked for consequential work.

## Choose an entry point

| Your available data | Function | Supported effects |
| --- | --- | --- |
| Effect estimates and sampling variances | `meta_analysis()` | Generic inverse variance |
| Events and totals for two groups | `meta_binary()` | OR, RR, RD |
| Means, SDs, and sample sizes for two groups | `meta_continuous()` | MD, Hedges' g |

All three functions accept an optional `subgroup=` column or array. Supplying
it returns a dedicated subgroup result containing each group, the overall
analysis, and a formal test for subgroup differences.

## Install

Install the core numerical library:

```console
pip install PyMetaAnalysis
```

Install Matplotlib support only when plots are needed:

```console
pip install "PyMetaAnalysis[plot]"
```

See [installation](installation.md) for supported Python versions, editable
development installs, extras, and verification.

## Start here

The [getting-started guide](getting-started.md) fits a complete analysis and
shows how to inspect its output. Continue with the guide matching your input:

- [generic effects and variances](guides/generic-effects.md);
- [binary outcomes](guides/binary-outcomes.md);
- [continuous outcomes](guides/continuous-outcomes.md).

Read [input data and row decisions](guides/input-data.md) before building a
pipeline around exclusions or mixed DataFrame/array inputs.

Before selecting a model, read [choosing methods](guides/method-selection.md).
Binary analyses with sparse data should also review
[zero-event studies](guides/zero-events.md).

After fitting a model, continue with [sensitivity analysis](guides/sensitivity-analysis.md)
and [provenance and reporting](guides/provenance-reporting.md) to assess stability
and create an auditable export.

The [statistical methods](methods/statistical-methods.md) page is the formula-
level implementation contract. [Validation](validation.md) explains the R
cross-software fixtures and CI layers, while [scope and limitations](limitations.md)
lists unsupported methods explicitly.

## Design principles

- Inputs and exclusions are explicit rather than silently repaired.
- Model-scale results remain separate from display-scale transformations.
- Resolved method settings are stored on every result.
- Versioned provenance records inputs, row decisions, and transformations.
- Reports provide detached dictionaries, strict JSON, and Markdown.
- Plotting is optional and never calls `show()`.
- Numerical implementations are independent and tested against hand
  calculations and external reference results.

## Project status

PyMetaAnalysis is pre-release and has not yet undergone a formal external
statistical audit. Pin the package version for consequential work and
independently check important analyses. See the repository
[changelog](https://github.com/ZhaoboDing/PyMetaAnalysis/blob/main/CHANGELOG.md)
and [contribution guide](development.md). For manuscripts and archived
analyses, see [citing PyMetaAnalysis](citation.md).
