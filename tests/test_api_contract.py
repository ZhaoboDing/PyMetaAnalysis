"""Compatibility guard for exported names, call syntax, and result accessors."""

from __future__ import annotations

import copy
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
CHECKER = runpy.run_path(str(ROOT / "tools" / "check_api_contract.py"))


def test_public_api_matches_reviewed_inventory() -> None:
    expected = json.loads(
        (ROOT / "tests/contracts/public_api.json").read_text(encoding="utf-8")
    )
    difference = CHECKER["contract_diff"](expected, CHECKER["collect_contract"]())
    assert not difference, difference


@pytest.mark.parametrize(
    "change", ["export", "default", "field", "exception", "schema"]
)
def test_inventory_detects_consumer_visible_drift(change: str) -> None:
    expected = CHECKER["collect_contract"]()
    actual = copy.deepcopy(expected)
    exports = actual["exports"]
    if change == "export":
        del exports["meta_analyze.meta_correlation"]
    elif change == "default":
        call = exports["meta_analyze.meta_analysis"]
        call["signature"] = call["signature"].replace("'normal'", "'hartung_knapp'")
    elif change == "field":
        exports["meta_analyze.MetaAnalysisResult"]["fields"].remove("estimate")
    elif change == "exception":
        exports["meta_analyze.InvalidStudyDataError"]["exception_bases"] = [
            "ValueError"
        ]
    else:
        actual["schemas"]["report"] = "2.0"
    assert CHECKER["contract_diff"](expected, actual)


def test_signature_preserves_parameter_kind_and_required_arguments() -> None:
    def example(value: float, /, label: str = "study", *, required: int) -> None:
        pass

    assert CHECKER["public_signature"](example) == (
        "(value, /, label='study', *, required)"
    )


def test_checker_cli_fails_without_rewriting_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/check_api_contract.py"),
            "--baseline",
            str(baseline),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "Review compatibility" in completed.stdout
    assert baseline.read_text(encoding="utf-8") == "{}\n"
