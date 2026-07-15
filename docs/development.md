# Development and contribution

PyMetaAnalysis welcomes bug reports, statistical review, documentation
improvements, reference datasets, and focused feature contributions.

## Set up a checkout

```console
git clone https://github.com/ZhaoboDing/PyMetaAnalysis.git
cd PyMetaAnalysis
python -m pip install -e ".[test,dev,docs,plot]"
```

Use a supported Python version (3.10–3.13). Keep changes focused and preserve
unrelated worktree modifications.

## Run checks

Before opening a pull request:

```console
python -m ruff format .
python -m ruff check .
actionlint
python -m mypy
python -m pytest --cov=meta_analyze --cov-branch --cov-report=term-missing
python -m mkdocs build --strict
python tools/execute_notebooks.py
python tools/check_release.py
python -m build
python tools/inspect_distribution.py dist
```

The CI matrix also tests Python 3.10–3.13 and declared dependency lower bounds.
Install the `notebook` extra before running the notebook executor.

## Statistical changes

A change to an estimator, default, exclusion rule, warning threshold, or
reported statistic should include:

1. a clear estimand and formula description;
2. unit tests or mathematical invariants;
3. an independent software/reference comparison where available;
4. tests for boundary behavior and invalid combinations;
5. user guide, API, result-schema, and limitation updates;
6. an ADR when the choice establishes or changes project policy.

Never update a golden reference merely because the implementation differs.
Explain and review the source of the difference first.

## Documentation changes

Documentation lives under `docs/` and is built with MkDocs strict mode. Add
every new page to `mkdocs.yml`; otherwise the build treats it as an omitted-file
warning and fails under strict mode. Keep examples executable against the
current public API and use canonical method names in saved-analysis examples.

The local root `DESIGN.md` is intentionally ignored and excluded from package
artifacts. Accepted user-facing decisions belong in versioned guides or ADRs.

## Reference fixtures

R fixture scripts and regeneration instructions live in `tests/reference/`.
Regeneration requires R, `metafor`, and `jsonlite` but is not part of routine
CI. Commit the script and generated JSON changes together.

## Pull requests

Describe what changed, why it changed, user/developer impact, and the checks
run. Keep numerical and presentation changes separable when practical. Do not
include generated `site/`, `dist/`, caches, environments, or local design
notes.

## Reporting issues

For a statistical discrepancy, include a minimal dataset, exact call, package
and Python versions, observed output, expected output, and the independent
method/software used for comparison. Remove confidential study data before
posting it publicly.

## Releases

Releases use a separate tag-triggered workflow, PyPI Trusted Publishing, and
GitHub Pages. See the [release process](releasing.md) for one-time repository
configuration, version synchronization, validation, tagging, and post-release
checks.
