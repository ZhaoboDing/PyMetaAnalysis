# Plotting

Forest, subgroup forest, and funnel plots use optional Matplotlib support.
Install it with:

```console
python -m pip install "PyMetaAnalysis[plot]"
```

Plotting methods return a Matplotlib `Axes` and never call `show()`. This makes
them suitable for notebooks, scripts, tests, and larger composed figures.

## Forest plots

```python
ax = result.forest(
    effect_label="Risk ratio",
    pooled_label="Pooled RR",
    show_prediction_interval=True,
    show_weights=True,
)
```

The plot contains only included studies. Study markers are scaled by normalized
model weights, study confidence intervals use the fitted confidence level, and
the pooled confidence interval is drawn as a diamond. A random-effects
prediction interval is shown when it is available and requested.

Pass an existing axes to compose the plot:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
result.forest(ax=ax)
fig.tight_layout()
```

### Forest parameters

| Parameter | Meaning |
| --- | --- |
| `ax` | Existing axes; a new one is created when omitted |
| `effect_label` | X-axis label |
| `pooled_label` | Label for the pooled row |
| `show_prediction_interval` | Show an available random-effects interval |
| `show_weights` | Print normalized study weights |
| `null_value` | Reference line; defaults to 1 for ratios and 0 otherwise |
| `log_scale` | Override the default logarithmic ratio axis |

OR and RR are modeled on a log scale but displayed as ratios on a logarithmic
axis by default. Other measures use an identity display scale and linear axis.
When overriding `log_scale=True`, all displayed effects and the null value must
be strictly positive.

## Subgroup forest plots

```python
ax = subgroups.forest(
    show_prediction_interval=True,
    show_weights=True,
)
```

The subgroup plot adds subgroup headings, subtotal diamonds, the overall
result, and the formal test for subgroup differences. Display-scale and null-
line rules match the ordinary forest plot.

## Funnel plots

```python
ax = result.funnel(
    confidence_level=0.95,
    show_pseudo_confidence_interval=True,
)
```

The y-axis is the study standard error and is inverted so more precise studies
appear toward the top. The vertical reference is the fitted pooled estimate.
Pseudo confidence limits are centered on that estimate and do not include
tau-squared.

### Funnel parameters

| Parameter | Meaning |
| --- | --- |
| `ax` | Existing axes; a new one is created when omitted |
| `effect_label` | X-axis label |
| `confidence_level` | Pseudo-limit level; defaults to the fitted level |
| `show_pseudo_confidence_interval` | Draw the shaded pseudo-limit region |
| `warn_on_few_studies` | Warn when fewer than 10 studies are plotted |
| `log_scale` | Override the default logarithmic ratio axis |

Funnel asymmetry can reflect small-study effects, heterogeneity, outcome
selection, design differences, chance, or publication processes. It is not by
itself evidence of publication bias. PyMetaAnalysis currently provides the
plot but not formal asymmetry tests.

## Save or display

The caller controls rendering:

```python
ax = result.forest()
ax.figure.savefig("forest.png", dpi=200, bbox_inches="tight")
```

In a script, call `matplotlib.pyplot.show()` explicitly when an interactive
window is desired.
