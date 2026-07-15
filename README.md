# PyMetaAnalysis

PyMetaAnalysis is an early-stage, pandas-first Python library for auditable
meta-analysis workflows.

The first development slice supports generic inverse-variance meta-analysis
from study-level effects and sampling variances:

```python
import meta_analyze as ma

result = ma.meta_analysis(
    data=studies,
    effect="effect",
    variance="variance",
    study="study",
    model="random",
    tau2_method="REML",
)

print(result.summary())
```

The public API is under active development and may change before version 0.1.

## License

MIT
