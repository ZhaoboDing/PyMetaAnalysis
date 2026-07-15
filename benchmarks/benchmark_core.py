"""Small deterministic performance baseline for core analysis entry points."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import timeit
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import meta_analyze as ma


def _benchmark(function: Callable[[], object], *, repeat: int, number: int) -> dict:
    function()
    samples = timeit.repeat(function, repeat=repeat, number=number)
    milliseconds = [elapsed * 1000.0 / number for elapsed in samples]
    return {
        "median_ms": statistics.median(milliseconds),
        "min_ms": min(milliseconds),
        "max_ms": max(milliseconds),
        "samples_ms": milliseconds,
    }


def _cases(studies: int) -> dict[str, Callable[[], object]]:
    rng = np.random.default_rng(20260715)

    generic_effect = rng.normal(0.1, 0.25, size=studies)
    generic_variance = rng.uniform(0.01, 0.09, size=studies)

    n_treat = rng.integers(60, 240, size=studies)
    n_control = rng.integers(60, 240, size=studies)
    event_treat = rng.binomial(n_treat, rng.uniform(0.12, 0.35, size=studies))
    event_control = rng.binomial(n_control, rng.uniform(0.15, 0.38, size=studies))

    continuous_n_treat = rng.integers(30, 180, size=studies)
    continuous_n_control = rng.integers(30, 180, size=studies)
    mean_control = rng.normal(0.0, 0.4, size=studies)
    mean_treat = mean_control + rng.normal(0.25, 0.18, size=studies)
    sd_treat = rng.uniform(0.7, 1.6, size=studies)
    sd_control = rng.uniform(0.7, 1.6, size=studies)

    return {
        "generic_random_reml": lambda: ma.meta_analysis(
            effect=generic_effect,
            variance=generic_variance,
            model="random",
            tau2_method="REML",
        ),
        "binary_rr_random_reml": lambda: ma.meta_binary(
            event_treat=event_treat,
            n_treat=n_treat,
            event_control=event_control,
            n_control=n_control,
            measure="RR",
            method="IV",
            model="random",
            tau2_method="REML",
        ),
        "continuous_smd_random_reml": lambda: ma.meta_continuous(
            mean_treat=mean_treat,
            sd_treat=sd_treat,
            n_treat=continuous_n_treat,
            mean_control=mean_control,
            sd_control=sd_control,
            n_control=continuous_n_control,
            measure="SMD",
            model="random",
            tau2_method="REML",
        ),
    }


def _metadata(arguments: argparse.Namespace) -> dict[str, Any]:
    dependency_versions = {
        name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy")
    }
    return {
        "benchmark_schema": "1.0",
        "package_version": ma.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "dependencies": dependency_versions,
        "studies": arguments.studies,
        "repeat": arguments.repeat,
        "number": arguments.number,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--number", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if min(arguments.studies, arguments.repeat, arguments.number) < 1:
        parser.error("studies, repeat, and number must all be positive")

    payload = _metadata(arguments)
    payload["cases"] = {
        name: _benchmark(function, repeat=arguments.repeat, number=arguments.number)
        for name, function in _cases(arguments.studies).items()
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if arguments.output is None:
        print(text, end="")
    else:
        arguments.output.write_text(text, encoding="utf-8")
        print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
