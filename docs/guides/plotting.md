# Plotting

Forest, subgroup forest, funnel, and Meta-regression bubble plots use optional
Matplotlib support.
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
axis by default. `ZCOR` is modeled on Fisher's z scale, back-transformed to
correlations, and displayed on a linear axis. Other measures use an identity
display scale and linear axis.
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

Add two-sided significance contours around the null effect with:

```python
ax = result.funnel(
    contour_levels=(0.90, 0.95, 0.99),
)
```

These confidence levels define the visible bands `0.05 < p <= 0.10`,
`0.01 < p <= 0.05`, and `p <= 0.01`; the central `p > 0.10` region remains
unshaded. The solid vertical line remains the fitted pooled estimate, while a
dotted line marks the null used for the contours. When contours are enabled,
their background replaces the ordinary blue pseudo-limit fill so the colors
do not mix. The pseudo-limit boundary lines are still drawn unless
`show_pseudo_confidence_interval=False`.

### Funnel parameters

| Parameter | Meaning |
| --- | --- |
| `ax` | Existing axes; a new one is created when omitted |
| `effect_label` | X-axis label |
| `confidence_level` | Pseudo-limit level; defaults to the fitted level |
| `show_pseudo_confidence_interval` | Draw pooled pseudo-limit boundaries and, without contours, their shaded region |
| `contour_levels` | Strictly increasing confidence levels in `(0,1)`; `None` disables contours |
| `contour_colors` | One valid Matplotlib color per contour level; defaults to light-to-dark gray |
| `contour_reference` | Contour null on the display scale; defaults to 1 for ratios and 0 otherwise |
| `show_contour_legend` | Show the corresponding two-sided p-value bands |
| `warn_on_few_studies` | Warn when fewer than 10 studies are plotted |
| `log_scale` | Override the default logarithmic ratio axis |

For example, customize the bands and null reference with:

```python
ax = result.funnel(
    contour_levels=(0.90, 0.95),
    contour_colors=("#fee2e2", "#ef4444"),
    contour_reference=0.0,
    show_contour_legend=True,
)
```

`contour_reference` uses the displayed effect scale. It must therefore be
positive for OR/RR results, whose default is `1`, and strictly between `-1`
and `1` for displayed correlations. Boundaries are calculated on the model
scale before the normal display transformation. Contours are always based on
sampling standard errors and do not incorporate tau-squared.

Funnel asymmetry can reflect small-study effects, heterogeneity, outcome
selection, design differences, chance, or publication processes. It is not by
itself evidence of publication bias. Contours help assess whether apparent
missing areas are predominantly statistically non-significant, but they do not
show that studies are actually missing or determine why asymmetry exists. Use
the separately documented classical
[`result.egger_test()`](small-study-effects.md), or `result.harbord_test()` /
`result.peters_test()` for an eligible binary OR analysis, when a formal
regression diagnostic is appropriate. None changes the plot or proves a
publication mechanism.

## Meta-regression bubble plots

An intercept-containing Meta-regression with exactly one numeric moderator
provides:

```python
ax = regression.bubble(
    moderator_label="Dose",
    effect_label="Effect",
    show_confidence_interval=True,
    show_prediction_interval=False,
)
```

Study marker area is proportional to normalized precision weight. The line,
mean confidence band, and optional mixed-effects true-effect prediction band
are obtained from the fitted model's `predict()` method.

| Parameter | Meaning |
| --- | --- |
| `ax` | Existing axes; a new one is created when omitted |
| `moderator_label` | X-axis label; defaults to the moderator name |
| `effect_label` | Y-axis label; defaults to `"Effect"` |
| `show_confidence_interval` | Draw the fitted mean confidence band |
| `show_prediction_interval` | Draw a mixed-effects true-effect prediction band |

Categorical, multivariable, and no-intercept fits are rejected because a
marginal plot would require values or averaging rules for other terms. The
function does not infer those scientific choices.

## Save or display

The caller controls rendering:

```python
ax = result.forest()
ax.figure.savefig("forest.png", dpi=200, bbox_inches="tight")
```

In a script, call `matplotlib.pyplot.show()` explicitly when an interactive
window is desired.
