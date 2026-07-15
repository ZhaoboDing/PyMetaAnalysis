"""Executable contracts for public Markdown documentation."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).parents[1]
MARKDOWN_FILES = (
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "CHANGELOG.md",
    ROOT / "SECURITY.md",
    ROOT / "benchmarks" / "README.md",
    ROOT / "examples" / "README.md",
    ROOT / "tests" / "reference" / "README.md",
    *sorted((ROOT / "docs").rglob("*.md")),
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
