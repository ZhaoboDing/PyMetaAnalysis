# Installation

PyMetaAnalysis supports Python 3.10 through 3.13. The numerical core depends
on NumPy, pandas, and SciPy; Matplotlib is optional.

## Install from PyPI

Install the core package with:

```console
python -m pip install PyMetaAnalysis
```

The distribution name is `PyMetaAnalysis`, while Python code imports the
shorter module name:

```python
import meta_analyze as ma
```

## Optional plotting support

Install the `plot` extra for forest and funnel plots:

```console
python -m pip install "PyMetaAnalysis[plot]"
```

Numerical analyses do not import Matplotlib. Calling a plotting method without
the extra raises an `ImportError` containing the installation command.

## Install a source checkout

Clone the repository and install it in editable mode:

```console
git clone https://github.com/ZhaoboDing/PyMetaAnalysis.git
cd PyMetaAnalysis
python -m pip install -e ".[test,dev,docs,plot]"
```

The extras have separate purposes:

| Extra | Contents |
| --- | --- |
| `plot` | Matplotlib plotting backend |
| `test` | pytest, Hypothesis, coverage, and plotting test dependencies |
| `dev` | Ruff, Mypy, type stubs, and build tooling |
| `docs` | MkDocs |

## Verify the installation

```console
python -c "import meta_analyze as ma; print(ma.__version__)"
```

Then run a small common-effect analysis:

```python
import meta_analyze as ma

result = ma.meta_analysis(
    effect=[0.1, 0.3, 0.2],
    variance=[0.02, 0.03, 0.01],
    model="common",
)

assert result.k == 3
print(result.summary())
```

## Upgrade or uninstall

```console
python -m pip install --upgrade PyMetaAnalysis
python -m pip uninstall PyMetaAnalysis
```

PyMetaAnalysis is pre-release software. Pin the version used by a consequential
analysis and record it with the analysis environment.
