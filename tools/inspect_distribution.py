"""Inspect built wheel and source-distribution contents."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

REQUIRED_SDIST_SUFFIXES = {
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "benchmarks/benchmark_core.py",
    "docs/guides/r-interoperability.md",
    "docs/releasing.md",
    "docs/validation.md",
    "examples/quickstart.ipynb",
    "tools/check_release.py",
}


def _archive_names(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return archive.getnames()
    raise ValueError(f"Unsupported distribution artifact: {path}")


def _has_suffix(names: list[str], suffix: str) -> bool:
    normalized = PurePosixPath(suffix).as_posix()
    return any(PurePosixPath(name).as_posix().endswith(normalized) for name in names)


def _inspect(path: Path) -> None:
    names = _archive_names(path)
    assert all(PurePosixPath(name).name != "DESIGN.md" for name in names), (
        f"DESIGN.md unexpectedly included in {path}"
    )
    assert all("site" not in PurePosixPath(name).parts for name in names), (
        f"generated site unexpectedly included in {path}"
    )

    if path.suffix == ".whl":
        assert _has_suffix(names, "meta_analyze/py.typed"), (
            f"py.typed missing from {path}"
        )
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        with zipfile.ZipFile(path) as archive:
            metadata = archive.read(metadata_name).decode("utf-8")
        required_metadata = {
            "License-Expression: MIT",
            "License-File: LICENSE",
            "Requires-Python: >=3.10",
            "Provides-Extra: notebook",
            (
                "Project-URL: Documentation, "
                "https://zhaoboding.github.io/PyMetaAnalysis/"
            ),
        }
        missing_metadata = sorted(
            field for field in required_metadata if field not in metadata
        )
        assert not missing_metadata, (
            f"Required wheel metadata missing: {missing_metadata}"
        )
        forbidden_metadata = sorted(
            line
            for line in metadata.splitlines()
            if line.startswith(("License:", "Classifier: License ::"))
        )
        assert not forbidden_metadata, (
            "Deprecated or conflicting wheel license metadata present: "
            f"{forbidden_metadata}"
        )
    else:
        missing = sorted(
            suffix
            for suffix in REQUIRED_SDIST_SUFFIXES
            if not _has_suffix(names, suffix)
        )
        assert not missing, f"Required source-distribution files missing: {missing}"

    print(f"Inspected {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, nargs="?", default=Path("dist"))
    arguments = parser.parse_args()

    artifacts = sorted(arguments.directory.iterdir())
    wheels = [path for path in artifacts if path.suffix == ".whl"]
    sdists = [path for path in artifacts if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(
            "Expected exactly one wheel and one source distribution; "
            f"found {len(wheels)} wheel(s) and {len(sdists)} sdist(s)"
        )

    for artifact in [*wheels, *sdists]:
        _inspect(artifact)


if __name__ == "__main__":
    main()
