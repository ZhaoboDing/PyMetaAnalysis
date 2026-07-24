# Release process

This page is for maintainers preparing a PyMetaAnalysis release. Releases are
tag-driven: an accepted `vX.Y.Z` tag builds the distributions, publishes them
to PyPI with short-lived OpenID Connect credentials, and creates a GitHub
Release containing the same artifacts.

## One-time repository configuration

### GitHub Pages

In repository **Settings > Pages**, select **GitHub Actions** as the publishing
source. The `pages.yml` workflow then builds the strict MkDocs site from
`main` and deploys it through the protected `github-pages` environment.

### PyPI trusted publisher

Create the `PyMetaAnalysis` project or a pending publisher on PyPI with these
values:

| Field | Value |
| --- | --- |
| PyPI project | `PyMetaAnalysis` |
| GitHub owner | `ZhaoboDing` |
| Repository | `PyMetaAnalysis` |
| Workflow | `release.yml` |
| Environment | `pypi` |

Create a GitHub environment named `pypi` and require maintainer approval where
the repository plan supports it. Protect release tags matching `v*`. The
publishing job receives only `id-token: write`; no long-lived PyPI token is
stored in GitHub.

See the PyPI documentation for
[creating a project with a trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and its
[GitHub Actions security guidance](https://docs.pypi.org/trusted-publishers/security-model/).

## Prepare a release pull request

1. Confirm the distribution name is still available for the first release.
2. Review public API and report-schema changes since the previous release.
3. Set the same final version in `src/meta_analyze/_version.py` and
   `CITATION.cff`. Hatchling reads the package version dynamically from
   `_version.py`.
4. Move the relevant `CHANGELOG.md` entries from `Unreleased` to a dated
   heading such as `## X.Y.Z - YYYY-MM-DD`.
5. Run the complete validation suite.

The local consistency check is:

```console
python tools/check_release.py
```

For a proposed tag, add:

```console
python tools/check_release.py --tag vX.Y.Z
```

The tag check rejects development versions, version mismatches, and releases
without a dated changelog heading.

## Validate the candidate

```console
python -m ruff format --check .
python -m ruff check .
actionlint
python -m mypy
python -m pytest --cov=meta_analyze --cov-branch --cov-report=term-missing
python -m mkdocs build --strict
python tools/execute_notebooks.py
python -m build
python tools/inspect_distribution.py dist
python benchmarks/benchmark_core.py
```

Independent external statistical review is not a release gate. Release notes
and the [validation status](validation.md) must accurately describe the
evidence available for the released commit and must not imply that a formal
audit has occurred when it has not.

### Meta-regression 0.4.0 acceptance checks

Before preparing the 0.4.0 version commit, confirm that:

- exact leave-one-out refits and influence diagnostics preserve fitted
  configuration and remain covered by fixed-version `metafor` references;
- VIF/GVIF, weighted condition diagnostics, and explicit linear contrasts
  retain their independent R references and invariance tests;
- default and Riley prediction intervals remain covered for normal,
  Hartung-Knapp, multivariable, and zero-tau-squared cases;
- `meta_regression_multivariable_reml` is present in the release performance
  benchmark output;
- the Meta-regression guide, methods, API, result, reporting, limitations, R
  mapping, and validation pages agree with the shipped behavior.

Only after these checks pass should the release pull request change the package
version to 0.4.0 and create the dated changelog heading.

## Tag and publish

After the release pull request is merged, update local `main` and tag the merge
commit:

```console
git switch main
git pull --ff-only
git tag -a vX.Y.Z -m "PyMetaAnalysis X.Y.Z"
git push origin vX.Y.Z
```

The release workflow then:

1. verifies that the tag commit belongs to `main`;
2. checks version and changelog consistency;
3. builds and inspects the wheel and source distribution;
4. runs the release performance benchmark against the built wheel;
5. publishes the distributions through PyPI Trusted Publishing;
6. creates a GitHub Release with distributions and benchmark output.

Never replace an existing release artifact or reuse a published version. If a
release is defective, fix it and publish a new version.

## Post-release checks

In a clean environment:

```console
python -m venv .release-smoke
.release-smoke/Scripts/python -m pip install PyMetaAnalysis
.release-smoke/Scripts/python -c "import meta_analyze as ma; print(ma.__version__)"
```

Use `.release-smoke/bin/python` instead on POSIX systems. Confirm the PyPI page,
GitHub Release, documentation site, citation metadata, and installation command
all refer to the same version.
