# Citing PyMetaAnalysis

PyMetaAnalysis does not yet have a DOI or software paper. Until one is
available, cite the exact software version and repository used in the analysis.

The repository includes machine-readable [`CITATION.cff`](https://github.com/ZhaoboDing/PyMetaAnalysis/blob/main/CITATION.cff)
metadata. GitHub's **Cite this repository** control can export it to common
bibliographic formats. Release maintainers keep its version synchronized with
the package metadata.

Retrieve the installed version with:

```python
import meta_analyze as ma

print(ma.__version__)
```

A provisional citation can use:

```text
PyMetaAnalysis contributors. PyMetaAnalysis (version <version>):
a pandas-first meta-analysis library for Python.
https://github.com/ZhaoboDing/PyMetaAnalysis
```

Include the access or release date required by the selected citation style.
When citing a tagged release, link to that release rather than an unpinned
branch.

## Report the analysis, not only the package

A reproducible publication should also state:

- effect measure and treatment/control direction;
- common-effect or random-effects model;
- pooling and tau-squared estimator;
- confidence-interval and prediction-interval methods;
- continuity-correction and RD boundary policies where applicable;
- included/excluded studies and reasons;
- PyMetaAnalysis, Python, NumPy, pandas, and SciPy versions;
- code and data availability.

`result.method_details()` provides a starting point for method prose, and
`result.report()` captures resolved methods, diagnostics, provenance, warnings,
and optionally the row-level table. Review generated text against the protocol
and journal requirements before publication.
