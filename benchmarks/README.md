# Core performance baseline

`benchmark_core.py` measures representative generic, binary, and continuous
random-effects fits using deterministic synthetic data. Correctness remains the
primary performance requirement; this benchmark is intended to reveal large
regressions, not to enforce a fragile wall-clock threshold.

Run the default benchmark with:

```console
python benchmarks/benchmark_core.py
```

Write machine-readable output with:

```console
python benchmarks/benchmark_core.py --output benchmark.json
```

Every tagged release executes the benchmark against the built wheel and
attaches its JSON output to the GitHub Release. Compare releases produced by
similar GitHub-hosted runners, Python versions, and dependency versions.

The generated inputs are synthetic and deterministic. They are not intended to
represent a particular empirical distribution of meta-analysis sizes or
effects.
