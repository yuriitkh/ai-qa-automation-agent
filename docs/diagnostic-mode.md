# Diagnostic Mode (M2.3)

Select a level in **Settings → Diagnostic Mode**. NORMAL is the default.
Settings use the existing reliability settings record; saving a level preserves
retry, fallback and repair settings, and saving reliability settings preserves
the diagnostic level. An operation captures its level when it starts, including
when a background run supplies the parent policy. Settings changes affect new
operations. An invalid or unavailable configuration uses NORMAL with the existing
safe reliability defaults; failed saves are reported as failures.

| Level | Collected information |
| --- | --- |
| OFF | Essential audit, errors, outcome, providers, durations and gate decisions; no optional events. |
| NORMAL | Essential audit plus operation and provider lifecycle events and rejection codes. |
| DEBUG | NORMAL plus routing/skips, safe rejected action summaries, grounding and coverage classifications, Discovery counts and provider response metadata. |
| TRACE | DEBUG plus validation stage timings and request/response character counts. No payload contents. |

The feature extends reliability operations and attempts, candidate diagnostics,
execution progress/trace and Run history. Existing TestCase, TestStep, operation,
attempt number, plan version, Run and execution IDs provide correlation. A
generation operation is associated with browser executions only when case, step
and saved plan-version IDs all match. This association can include more than one
Run of the same version; it does not identify a single execution invocation.

Generation Decisions shows separate provider response and plan decisions, the
failed gate and reason, durations and expandable safe metadata. Run Progress
keeps operation links and failures alongside their corresponding TestSteps.
Provider SUCCESS and Plan ACCEPTED do not establish Browser Validation PASSED
or actual product/test PASS. Request transmission is unknown for adapters that
do not explicitly mark their request. A marked request means the adapter reached
its request call, not proof that a server received it.

Exact reasons come from deterministic grounding branches, validation issues,
typed provider metadata or observed browser failure state. Public validation
codes remain unchanged. Unsupported declarations, fill/assert mismatch, missing
fill, unsupported controls, ambiguity, conflicting literals and restrictions are
classified without saving their values. Unknown exception text is never used to
infer an exact cause. Browser action failures without established precondition
or missing-target evidence retain an unknown underlying cause.

For output text assertions, `OUTPUT_VALUE_ONLY_IN_STEP` means the literal occurs
only in an atomic TestStep while a related original TestCase description remains
authoritative. Generated/elaborated step text cannot upgrade that description.
Clarify the original requirement with confirmed wording, or supply deterministic
evidence at the asserted target. `OUTPUT_VALUE_NOT_GROUNDED` means neither the
authoritative requirement nor the supplied target-bound observations support
the output literal. These codes describe validation evidence, not proof of an AI
or product defect. An unavailable historical candidate remains `UNKNOWN`;
explicit `UNKNOWN` also keeps `root_cause_known` false. Exact-message requirements
cannot be replaced with structural visibility checks.

**Export Diagnostics** provides JSON or an in-memory ZIP containing only
`diagnostics.json` and `summary.txt`. Downloads are available for an operation,
Run, or current progress job. UUID-based filenames and fixed archive members
prevent caller-provided paths. The report projects allowlisted structural
metadata, hashes selectors and requirement clauses, redacts credential-bearing
labels and omits raw errors, private requirements, form values, prompts,
responses, DOM, URLs and evidence files. No-store download responses prevent
browser caching. Unavailable evidence and metadata are explicit, never PASS.

Optional data is capped at 128 events and 32 KiB per operation. Completed optional
diagnostics are retained for at most seven days and the most recent 100 operation
positions, and pruned when generation completes. Running operations remain
available. Existing audit records and historical optional metadata are not
deleted or rewritten by this retention policy. JSON exports are capped at
256 KiB, summaries at 64 KiB, with fixed operation/execution/chronology limits
and explicit omissions. Mandatory audit/history follows existing retention;
these limits do not bound the size of the entire database.

Diagnostic collection failures preserve execution decisions. Reliability storage
failures use the existing in-memory record repository as a best-effort fallback,
marked in the export; these fallback records may be lost on restart. Settings
saves do not claim persistence when storage fails. Expired and historical data
cannot be reconstructed retroactively. SDKs that do not expose HTTP status,
finish reason or token usage leave those fields unknown. TRACE stage timings
cover generation validation gates and existing provider/browser execution
durations; individual browser action durations are not collected.

Regression tests use fake providers, synthetic browser state and isolated
temporary SQLite. Local Chromium UI tests use loopback only. No paid request is
needed to inspect or export saved diagnostics.
