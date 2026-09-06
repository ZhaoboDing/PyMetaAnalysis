# Proposed 1.0 compatibility policy

Status: proposed for acceptance in batch B of the [1.0 roadmap](roadmap-1.0.md).
The current release remains 0.9.0. This policy is not a claim that 1.0 has been
released or its audit completed.

## Public boundary

The proposed supported Python surface is `meta_analyze.__all__`, the functions
in `meta_analyze.plotting.__all__`, and the public fields, properties and methods
of those exported objects. Documented aliases and accepted parameter values
are also part of the contract. Internal modules are implementation details
unless explicitly added to the supported reference. Names beginning with `_`,
private storage, and result constructors requiring private state are excluded;
obtain fitted results through analysis functions.

The executable inventory in `tests/contracts/public_api.json` records the 0.9
starting surface. It checks names, call syntax/defaults, public dataclass fields,
properties, methods, exception bases and schema versions. It does not check
annotations, accepted string values, resolved defaults, numeric behavior,
DataFrame schemas, or plot artists. Those require the behavioral tests and the
remaining acceptance gates. Annotation changes affecting supported callers
still require compatibility review.

Run it against an editable checkout or installed candidate:

```console
python tools/check_api_contract.py
python tools/check_api_contract.py --write candidate-api.json
```

Review a candidate diff before updating the committed inventory; never update
the inventory just to make a failing check pass. Additions also trigger review
but are not automatically breaking changes. The baseline records existing
behavior, not a decision that every current behavior is suitable for 1.0.

## Changes during 1.x

After acceptance, patch releases fix defects while retaining call compatibility.
Minor releases may add optional arguments and capabilities. Renaming/removing
public functions, parameters or fields, changing statistical defaults, altering
documented scales or exclusion meanings, and making valid calls fail require a
major release after deprecation. An estimator correction may change numbers in
a patch: document the defect, affected configurations, independent validation,
and advice to rerun affected analyses. Do not preserve a known wrong result for
numerical compatibility.

Numerical equivalence is assessed with justified method-specific tolerances,
not bitwise identity across BLAS, operating systems and dependency versions.
Store package/dependency versions and explicitly select methods for reproducible
analyses. Exception classes and hierarchy are supported; exact message text and
human-readable summaries/Markdown formatting are not parsing interfaces.

## Tables, serialization and plots

- Preserve documented DataFrame column names, order of existing columns,
  meanings, model/display scales and row-ID/exclusion semantics. Add new columns
  at the end; consumers should select columns by name. Exact pandas storage
  dtype is not promised across supported pandas releases; documented numeric,
  boolean, label and missing-value semantics are.
- Version report and provenance schemas independently of the package. A schema
  minor version may add optional fields; removal, required-field additions,
  type or semantic changes require a schema major version and a migration note.
  Readers should tolerate unknown fields. Changing the package's default export
  to an incompatible schema requires the package deprecation process as well.
- `ResultReport` JSON is a strict, JSON-safe export; unavailable numbers become
  `null`. Export/parse round-trip means `json.loads(report.to_json())` equals
  `report.to_dict()`. It does not recreate fitted results. Pickle and arbitrary
  direct result construction are not persistence contracts.
- Diagnostic `to_dict()` outputs, including trim-and-fill, currently have no
  schema version and are not all strict JSON exports. Their key/type policy is
  an open A2 gate; do not infer report-schema guarantees from the method name.
- Plots return Matplotlib `Axes`, reuse a supplied Axes, and do not call `show()`.
  Data scales, study inclusion, intervals and documented artist semantics are
  supported. Pixel identity, font metrics and undocumented artist ordering are
  not. Matplotlib remains optional for numerical work.

## Deprecation process

An ordinary removal requires a documented replacement, migration example,
changelog entry, and a `DeprecationWarning` at the caller's location. Keep the
deprecated behavior for at least two minor releases and at least six months,
whichever is longer, and remove it only in a major release. Test both the old
behavior and the warning. Statistical correctness or security fixes may need
immediate changes; document the reason and affected analyses explicitly.

The 0.9-to-1.0 transition may settle outstanding defaults or contracts before
these 1.x promises begin, but any such change must be announced in 0.10 and the
migration guide, with no silent substitution of HK variants.

## Support proposal

Target Python 3.10–3.14 and Linux, Windows and macOS for 1.0, subject to the
roadmap's actual test evidence. Publish the tested dependency matrix with the
release. New Python/dependency versions become supported when CI passes;
unbounded dependency declarations alone are not test evidence.

Maintain the latest 1.x minor line for bug/security fixes. Older minor releases
remain installable but have no promised backport branch. Retain support for a
declared Python version until at least its upstream end of life; announce any
subsequent removal or dependency-floor increase in the previous minor release
and migration notes. Make such support changes only in minor or major releases,
never patches. No fixed response-time or long-term-support service is promised.
Ratify this policy and synchronize `SECURITY.md` before final release.
