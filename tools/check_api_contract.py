"""Compare the installed public API with a reviewed, versioned inventory.

This detects surface drift, not statistical or schema compatibility. Run against
an editable checkout, or an installed candidate to compare a built distribution.
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import fields, is_dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any

import meta_analyze as ma
from meta_analyze import plotting
from meta_analyze.provenance import PROVENANCE_SCHEMA_VERSION
from meta_analyze.reporting import REPORT_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests" / "contracts" / "public_api.json"


def public_signature(value: Any) -> str:
    """Record call syntax/defaults without version-dependent type rendering."""
    signature = inspect.signature(value)
    return str(
        signature.replace(
            parameters=[
                parameter.replace(annotation=inspect.Parameter.empty)
                for name, parameter in signature.parameters.items()
                if not name.startswith("_")
            ],
            return_annotation=inspect.Signature.empty,
        )
    )


def collect_contract() -> dict[str, Any]:
    """Inventory explicit exports and public members, excluding private state."""
    exports: dict[str, Any] = {}
    for module in (ma, plotting):
        for name in sorted(module.__all__):
            value = getattr(module, name)
            qualified = f"{module.__name__}.{name}"
            if inspect.isclass(value):
                if issubclass(value, Exception):
                    exports[qualified] = {
                        "exception_bases": [base.__name__ for base in value.__bases__]
                    }
                    continue
                members: dict[str, Any] = {}
                if is_dataclass(value):
                    members["fields"] = [
                        field.name
                        for field in fields(value)
                        if not field.name.startswith("_")
                    ]
                    # Constructors exposing private storage are implementation details.
                    if all(not field.name.startswith("_") for field in fields(value)):
                        members["constructor"] = public_signature(value)
                members["properties"] = [
                    member_name
                    for member_name, member in inspect.getmembers(value)
                    if not member_name.startswith("_") and isinstance(member, property)
                ]
                members["methods"] = {
                    member_name: public_signature(member)
                    for member_name, member in inspect.getmembers(value)
                    if not member_name.startswith("_") and inspect.isfunction(member)
                }
                exports[qualified] = members
            elif inspect.isfunction(value):
                exports[qualified] = {"signature": public_signature(value)}
            else:
                # __version__ changes at release time; its presence/type is public.
                exports[qualified] = {"type": type(value).__name__}
    return {
        "inventory_format": 1,
        "schemas": {
            "report": REPORT_SCHEMA_VERSION,
            "provenance": PROVENANCE_SCHEMA_VERSION,
        },
        "exports": exports,
    }


def contract_diff(expected: dict[str, Any], actual: dict[str, Any]) -> str:
    """Return a readable diff; additions also require an intentional review."""
    return "".join(
        unified_diff(
            (json.dumps(expected, indent=2, sort_keys=True) + "\n").splitlines(True),
            (json.dumps(actual, indent=2, sort_keys=True) + "\n").splitlines(True),
            fromfile="reviewed API inventory",
            tofile="installed API",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument(
        "--write",
        type=Path,
        help="Write a candidate inventory for review; no implicit baseline update",
    )
    args = parser.parse_args()
    actual = collect_contract()
    if args.write is not None:
        args.write.write_text(
            json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote candidate API inventory to {args.write}")
        return 0
    expected = json.loads(args.baseline.read_text(encoding="utf-8"))
    difference = contract_diff(expected, actual)
    if difference:
        print(difference, end="")
        print(
            "Review compatibility and migration impact before updating the inventory."
        )
        return 1
    print(
        f"Public API matches {args.baseline.name} ({len(actual['exports'])} exports)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
