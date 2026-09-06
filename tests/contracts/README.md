# Public contract baseline

`public_api.json` inventories the exported API at 0.9.0, commit
`2838e15f990f5ed902246329f2d4d6c02312e6c1`. It is a drift detector for the 1.0
review, not an assertion that the review or stability freeze is complete.

Run `python tools/check_api_contract.py` against an editable checkout or an
installed artifact. `--write candidate-api.json` writes a candidate for review.
The normal check never overwrites its baseline and exits nonzero on drift.
Intentional additions also require review and an inventory update in the same
PR; removals/default changes need compatibility and migration justification.

The inventory excludes type-annotation rendering, private parameters/storage,
constructors requiring private fields, generated dunder methods, and the changing
value of `__version__`. It includes public configuration constructors. It does
not test accepted aliases, resolved defaults, tables, serialization payloads,
statistical behavior or plotting output. Those remain behavioral acceptance
work, described in the [compatibility proposal](../../docs/compatibility.md).
