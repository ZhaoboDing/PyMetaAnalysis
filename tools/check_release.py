"""Validate release metadata and an optional release tag."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _validate_pyproject_version_source() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if re.search(r"^version\s*=", text, flags=re.MULTILINE):
        raise ValueError("pyproject.toml must not declare a static project version")
    if not re.search(r'^dynamic\s*=\s*\[\s*"version"\s*\]$', text, flags=re.MULTILINE):
        raise ValueError('pyproject.toml must declare dynamic = ["version"]')
    expected = '[tool.hatch.version]\npath = "src/meta_analyze/_version.py"'
    if expected not in text:
        raise ValueError("Hatchling must read the version from _version.py")


def _python_version() -> str:
    text = (ROOT / "src" / "meta_analyze" / "_version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"$', text, flags=re.MULTILINE)
    if match is None:
        raise ValueError("Could not read __version__ from _version.py")
    return match.group(1)


def _citation_version() -> str:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*['\"]?([^'\"\s]+)", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError("Could not read version from CITATION.cff")
    return match.group(1)


def _validate_tag(version: str, tag: str) -> None:
    expected = f"v{version}"
    if tag != expected:
        raise ValueError(f"Release tag {tag!r} must equal {expected!r}")
    if re.search(r"(?:\.dev|[+-])", version):
        raise ValueError(f"Release tag cannot publish development version {version!r}")

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    escaped = re.escape(version)
    pattern = rf"^##\s+\[?{escaped}\]?\s+-\s+\d{{4}}-\d{{2}}-\d{{2}}\s*$"
    if re.search(pattern, changelog, flags=re.MULTILINE) is None:
        raise ValueError(
            f"CHANGELOG.md needs a dated level-two heading for version {version}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="Optional release tag, for example v0.1.0")
    arguments = parser.parse_args()

    _validate_pyproject_version_source()
    versions = {
        "src/meta_analyze/_version.py": _python_version(),
        "CITATION.cff": _citation_version(),
    }
    if len(set(versions.values())) != 1:
        details = ", ".join(f"{path}={version}" for path, version in versions.items())
        raise ValueError(f"Release versions do not match: {details}")

    version = next(iter(versions.values()))
    if arguments.tag is not None:
        _validate_tag(version, arguments.tag)

    print(f"Release metadata is consistent for version {version}.")


if __name__ == "__main__":
    main()
