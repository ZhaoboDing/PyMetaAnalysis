# Road to 1.0

Status: active stabilization plan, established on 2026-09-06. This is a release
acceptance plan, not a declaration that the current package is stable or audited.

## Starting point

The baseline is 0.9.0, tag `v0.9.0`, commit
`2838e15f990f5ed902246329f2d4d6c02312e6c1`. The repository already implements
the conventional univariate scope listed in [limitations](limitations.md),
including correlation pooling, meta-regression, four small-study-effect tests,
and L0/R0 trim-and-fill. Large new statistical families are not prerequisites
for 1.0.

The repository inspection found:

- existing formula, boundary, property, and fixed-version R reference tests;
- Linux CI for Python 3.10–3.14 and direct dependency lower bounds on 3.10,
  but no Windows/macOS CI jobs;
- report schema `1.2` and provenance schema `1.0`, with separate unversioned
  diagnostic dictionaries such as `TrimAndFillResult.to_dict()`;
- accepted ADRs 0001–0007, with `normal` still the mean-inference default;
- no formal external statistical audit or comprehensive API stability policy.

Inspection also reproduced an excluded-row trim-and-fill failure and found
narrower R coverage than the design target, plus unwired generator-control
metadata. The [initial findings](statistical-review.md#initial-inspection-findings)
are blocking follow-up work, not accepted behavior.

The 0.9.0 release evidence reported 770 passing tests and 94.24% branch
coverage. Those are historical results, not acceptance evidence for a later
candidate. Every release gate below needs evidence for the candidate commit.
Local design notes are not the public specification; accepted ADRs, executable
behavior, and versioned documentation are the review inputs.

## Acceptance matrix

Statuses mean **baseline** (some evidence exists; audit incomplete), **open**
(decision or work outstanding), and **accepted** (linked evidence and sign-off
for a named commit). Passing implementation tests alone cannot accept a gate.

| ID | Gate and acceptance evidence | Initial state | Responsible role / batch |
| --- | --- | --- | --- |
| M1 | Core pooling, heterogeneity, effect-size equations, subgroups, and repeated fits: formula-to-code review, hand calculations, R reproduction, boundaries and discrepancy disposition | Baseline | Independent methods reviewer / B |
| M2 | HK variants, HTS/HK-PR and regression prediction intervals, Q-profile: variance, degrees of freedom, zero/empty-set boundaries and tolerance review | Baseline | Independent methods reviewer / B |
| M3 | Sparse binary, IV/MH OR/RR/RD and Peto: correction/exclusion scope, Sato variance, raw-table pooling and invariants | Baseline | Independent methods reviewer / B |
| M4 | Meta-regression and diagnostics: design encoding, rank, generalized tau-squared, no-intercept conventions, influence and contrasts | Baseline | Independent methods reviewer / B |
| M5 | Egger, Begg, Harbord, Peters and trim-and-fill: assumptions, ties, exact/asymptotic inference, direction, convergence, synthetic-row provenance | Baseline | Independent methods reviewer / B |
| M6 | All R fixtures reproduced with recorded versions, method-specific tolerances justified, intentional differences signed off | Baseline | Independent methods reviewer / B |
| D1 | Accepted 1.0 default-CI ADR with evidence, compatibility impact, affected tests, and migration instructions; all other defaults explicitly ratified | Open | Maintainer + methods reviewer / B |
| A1 | Public exports, parameters/defaults, result fields, exception types and aliases inventoried and reviewed; compatibility/deprecation policy accepted | Baseline after A | Maintainer / A, B |
| A2 | DataFrame columns/order/semantics, provenance, report schemas and diagnostic dictionaries reviewed; export/parse round-trip and defensive-copy tests | Baseline | Maintainer / C |
| A3 | Plot return types, explicit Axes reuse, scales, exclusions, optional dependency and no implicit show; semantic artist assertions | Baseline | Maintainer / C |
| C1 | Full tests on Python 3.10–3.14 on Linux, Windows and macOS with latest resolvable dependencies; environment versions retained | Open | Maintainer / C |
| C2 | Declared dependency floors tested on Python 3.10 on all three OSes; newer Python uses compatible versions rather than impossible old-wheel combinations | Linux baseline | Maintainer / C |
| C3 | Wheel and sdist install independently in clean environments, no source-checkout imports, no-Matplotlib core smoke, pip check and distribution inspection | Partial baseline | Maintainer / C |
| C4 | Both notebooks and every complete runnable docs example execute; strict docs and JSON export/parse checks pass against candidate | Partial baseline | Maintainer / C |
| C5 | Numerical extremes/property tests and benchmarks pass; platform, versions, timings and candidate SHA retained; investigate material regressions | Baseline | Maintainer / C |
| R1 | Support/deprecation/compatibility policy accepted; 0.9 migration guide complete; stability wording and package classifier updated at the appropriate release | Open | Maintainer / D |
| R2 | At least one real review workflow tried by a user outside implementation, with de-identified reproducer or private-evidence reference and resolved findings | Open | RC user + maintainer / D |
| R3 | No unresolved correctness or compatibility blockers; all above gates accepted; final checks on release commit and explicit release authorization | Open | Maintainer / D |

Record statistical evidence using the [review checklist](statistical-review.md).
For each other gate, retain candidate SHA, command/workflow URL, OS/Python and
resolved dependencies, outcome, reviewer/date, and linked fixes. A relevant
change after sign-off reopens that gate. Reviewer identities are unassigned;
this plan does not claim that someone has agreed to perform an audit.

## Development batches

### A — 0.10 foundations (this batch)

Deliver this roadmap, acceptance matrix, independent-review checklist, proposed
compatibility policy, and a proposed default-CI decision brief. Add an executable
0.9 public API inventory and drift check to the normal test suite. Correct
documentation omissions discovered during inspection and make the trim-and-fill
generator honor a separate output path for review. Keep numerical behavior,
package version, and R golden values unchanged.

Acceptance: the inventory matches 0.9 exports and catches representative
incompatible mutations; tests, Ruff, Mypy, strict docs, notebooks, and release
checks pass. The draft PR identifies what is ready for review and what remains
unaccepted. This completes the foundation, not the API freeze or audit.

### B — method decisions and API freeze, before 0.10.0

Work through M1–M6 in bounded review PRs using independent statistical review.
First resolve the reproduced trim-and-fill findings, then prioritize
pooling/inference and sparse binary before the remaining regression and asymmetry
review. Correct discrepancies with focused tests and independent numerical
evidence. Resolve [ADR 0008](adr/0008-stable-inference-default.md), ratify the
remaining defaults and [compatibility policy](compatibility.md), and review the
inventory against the documented API. Announce any 0.9 breaking changes here.

The first corrective batch is tracked in the
[review disposition](statistical-review.md#september-2026-review-disposition).
It repairs extreme-weight pooling and trim-and-fill boundaries and expands R
references; it leaves the normal CI default and version metadata unchanged.
Before freezing the API, explicitly review the opposite sign conventions of
sensitivity `estimate_change` (deleted minus original) and regression DFBETAS
(original minus deleted), and the DataFrame return type of sensitivity
`summary()`. Preserve these contracts until a documented decision is accepted.

Publish 0.10.0 only after its method/default decisions and API freeze have been
accepted. Remind the maintainer before changing version metadata; use a separate
release PR. An absent external reviewer leaves the relevant gates open.

### C — compatibility and end-to-end audit, before 1.0.0rc1

Implement and execute C1–C5, finish A2–A3, and test installed artifacts against
the frozen contract. Add schema fixtures or validators only after their scope
and versioning rules are agreed. Audit current unversioned diagnostic exports,
especially synthetic rows in trim-and-fill, without assuming `to_dict()` implies
strict JSON or fitted-object restoration. Restrict changes to correctness,
compatibility, performance regressions, and documentation.

The first RC requires all method/default/API gates accepted and the automated
compatibility matrix green. No major feature additions enter the RC branch.

### D — RC use and final release

Run at least one real-project RC trial and allow at least two weeks for feedback.
Any correctness or contract fix requires a new RC, rerun affected gates, and
allow at least another week of trial; elapsed time alone does not pass R2.
Finish the migration guide and support policy. Replace `early-stage` / 0.x API
wording and the Alpha classifier only when preparing the final stable release.
Document the scope of the completed audit, including residual limitations.

Run final tests, notebooks, docs, benchmark, artifact installation and metadata
checks on the proposed release commit. Version changes need an explicit reminder;
merge, tag, and PyPI publication require maintainer authorization.

## Deferred to 1.x or separate proposals

Multilevel/multivariate/network models, robust variance estimation,
diagnostic-accuracy and survival outcomes, selection models, formula parsing,
risk-of-bias, GRADE, and literature screening remain outside the 1.0 gates.
New features must not displace unresolved correctness or compatibility work.
