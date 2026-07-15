# ADR 0001: Keep Matplotlib optional

- Status: Accepted
- Date: 2026-07-15

## Context

PyMetaAnalysis needs forest and funnel plots, but its numerical core should
remain usable in server, notebook, and alternative plotting environments
without importing a graphical stack. Users may also prefer Plotly, Altair, or
their own rendering layer over the stable result tables.

## Decision

Matplotlib is provided through the `plot` optional dependency extra:

```console
pip install "PyMetaAnalysis[plot]"
```

Plotting modules import Matplotlib only when a plot is requested. Convenience
methods return a Matplotlib `Axes` and never call `show()`.

## Consequences

- importing and using statistical APIs does not require Matplotlib;
- plotting calls without the extra raise an actionable `ImportError`;
- plotting tests install Matplotlib and use a non-interactive backend;
- the result protocol remains sufficient for third-party plotting backends.
