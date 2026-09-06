"""Executable contracts for public Markdown documentation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote

import pytest

import meta_analyze as ma

ROOT = Path(__file__).parents[1]
MARKDOWN_FILES = (
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "CHANGELOG.md",
    ROOT / "SECURITY.md",
    ROOT / "benchmarks" / "README.md",
    ROOT / "examples" / "README.md",
    ROOT / "tests" / "reference" / "README.md",
    ROOT / "tests" / "contracts" / "README.md",
    *sorted((ROOT / "docs").rglob("*.md")),
)


PYTHON_PAGES = tuple(
    path for path in MARKDOWN_FILES if "```python\n" in path.read_text(encoding="utf-8")
)


def _python_blocks() -> list[pytest.param]:
    parameters: list[pytest.param] = []
    pattern = re.compile(r"```python\n(.*?)```", flags=re.DOTALL)
    for path in MARKDOWN_FILES:
        for index, block in enumerate(
            pattern.findall(path.read_text(encoding="utf-8")), start=1
        ):
            parameters.append(
                pytest.param(path, block, id=f"{path.relative_to(ROOT)}:{index}")
            )
    return parameters


@pytest.mark.parametrize(("path", "block"), _python_blocks())
def test_python_documentation_blocks_parse(path: Path, block: str) -> None:
    compile(block, str(path), "exec")


@pytest.mark.parametrize(
    "path", PYTHON_PAGES, ids=lambda path: str(path.relative_to(ROOT))
)
def test_documentation_examples_execute(
    path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute visible examples in page order, without hidden input fixtures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)
    pattern = re.compile(
        r"(?:<!-- example: fragment (.*?) -->\n)?```python\n(.*?)```", re.DOTALL
    )
    source = path.read_text(encoding="utf-8")
    namespace: dict[str, object] = {}
    try:
        for match in pattern.finditer(source):
            if match.group(1):
                # Only the API's deliberately incomplete call signatures are
                # fragments. A new exemption needs an explicit review here.
                assert path == ROOT / "docs" / "reference" / "api.md"
                assert match.group(1) == "call signatures with omitted arguments"
                assert match.group(2).strip() == (
                    "result = ma.meta_analysis(...)\n"
                    'subgroups = ma.meta_analysis(..., subgroup="region")'
                )
                continue
            line = source.count("\n", 0, match.start(2))
            exec(compile("\n" * line + match.group(2), str(path), "exec"), namespace)
        for value in namespace.values():
            if isinstance(value, ma.ResultReport):
                assert json.loads(value.to_json()) == value.to_dict()
        for number in plt.get_fignums():
            plt.figure(number).canvas.draw()
        if path == ROOT / "docs" / "guides" / "plotting.md":
            assert (tmp_path / "forest.png").stat().st_size > 1000
    finally:
        plt.close("all")


@pytest.mark.parametrize("path", MARKDOWN_FILES, ids=lambda path: str(path.name))
def test_relative_markdown_links_resolve(path: Path) -> None:
    pattern = re.compile(r"(?<!!)\[[^]]*]\(([^)]+)\)")
    for raw_target in pattern.findall(path.read_text(encoding="utf-8")):
        target = raw_target.split(maxsplit=1)[0].strip("<>")
        if "://" in target or target.startswith(("#", "mailto:")):
            continue
        relative = unquote(target.split("#", maxsplit=1)[0])
        if relative:
            assert (path.parent / relative).exists(), (
                f"Broken relative link {target!r} in {path.relative_to(ROOT)}"
            )


def test_pypi_readme_links_are_absolute() -> None:
    """PyPI resolves relative long-description links against its project URL."""
    pattern = re.compile(r"\]\(([^)]+)\)")
    for raw_target in pattern.findall((ROOT / "README.md").read_text(encoding="utf-8")):
        target = raw_target.split(maxsplit=1)[0].strip("<>")
        assert target.startswith(("https://", "http://", "#", "mailto:")), (
            f"README link {target!r} must be absolute so it works on PyPI"
        )
