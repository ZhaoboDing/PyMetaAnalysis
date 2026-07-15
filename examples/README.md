# Examples

The [end-to-end quickstart notebook](quickstart.ipynb) demonstrates DataFrame
input, a random-effects binary analysis, result inspection, provenance,
reporting, sensitivity analysis, and a forest plot.

All values in the notebook are synthetic and were created for documentation;
they do not represent real participants or a published clinical question. The
notebook is distributed under the repository's MIT license.

Install the notebook dependencies and execute every committed notebook with:

```console
python -m pip install -e ".[notebook]"
python tools/execute_notebooks.py
```

The executor writes completed notebooks to a temporary directory and leaves
the committed sources unchanged.
