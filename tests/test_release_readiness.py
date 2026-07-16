"""Release metadata, notebook, and benchmark smoke contracts."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import meta_analyze as ma

ROOT = Path(__file__).parents[1]


def test_release_metadata_sources_match() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation_match = re.search(r"^version:\s*([^\s]+)", citation, re.MULTILINE)

    assert citation_match is not None
    assert citation_match.group(1) == ma.__version__
    assert re.search(r"^version\s*=", pyproject, re.MULTILINE) is None
    assert 'dynamic = ["version"]' in pyproject
    assert '[tool.hatch.version]\npath = "src/meta_analyze/_version.py"' in pyproject
    assert 'license = "MIT"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert '"License ::' not in pyproject
    assert pyproject.count('{ name = "Zhaobo Ding", email = "ding.zb@yahoo.com" }') == 2
    assert "family-names: Ding" in citation
    assert "given-names: Zhaobo" in citation
    assert "email: ding.zb@yahoo.com" in citation


def test_release_metadata_checker_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "tools/check_release.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert ma.__version__ in completed.stdout


def test_github_release_command_has_repository_context() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert (
        """      - name: Create release
        env:
          GH_TOKEN: ${{ github.token }}
          GH_REPO: ${{ github.repository }}
"""
        in workflow
    )


def test_quickstart_notebook_is_valid_unexecuted_json() -> None:
    path = ROOT / "examples" / "quickstart.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert any(cell["cell_type"] == "code" for cell in notebook["cells"])
    assert all(
        cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "synthetic" in markdown.lower()


def test_core_benchmark_smoke(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"
    subprocess.run(
        [
            sys.executable,
            "benchmarks/benchmark_core.py",
            "--studies",
            "6",
            "--repeat",
            "1",
            "--number",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["package_version"] == ma.__version__
    assert set(payload["cases"]) == {
        "generic_random_reml",
        "binary_rr_random_reml",
        "continuous_smd_random_reml",
    }
