"""Execute committed example notebooks without modifying their sources."""

from __future__ import annotations

import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "examples").glob("*.ipynb"))


def main() -> None:
    if not NOTEBOOKS:
        raise ValueError("No example notebooks were found")

    with tempfile.TemporaryDirectory(prefix="pymetaanalysis-notebooks-") as directory:
        output_directory = Path(directory)
        for source in NOTEBOOKS:
            notebook = nbformat.read(source, as_version=4)
            client = NotebookClient(
                notebook,
                timeout=180,
                kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
            )
            client.execute()
            output = output_directory / source.name
            nbformat.write(notebook, output)
            print(f"Executed {source.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
