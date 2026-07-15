# Contributing to PyMetaAnalysis

Thank you for helping improve PyMetaAnalysis. Contributions may include code,
documentation, reproducible bug reports, statistical review, and independent
reference cases.

## Development setup

```console
git clone https://github.com/ZhaoboDing/PyMetaAnalysis.git
cd PyMetaAnalysis
python -m pip install -e ".[test,dev,docs,plot]"
```

Run the project checks before opening a pull request:

```console
python -m ruff format .
python -m ruff check .
python -m mypy
python -m pytest --cov=meta_analyze --cov-branch --cov-report=term-missing
python -m mkdocs build --strict
python -m build
```

Statistical changes need formula documentation, boundary tests, and an
independent reference comparison where possible. Do not replace committed
golden values without explaining and reviewing the discrepancy.

See the full [development and contribution guide](docs/development.md) for
statistical-change requirements, R fixture regeneration, documentation rules,
and issue-reporting guidance.
