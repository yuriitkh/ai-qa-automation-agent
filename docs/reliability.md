# Automation Reliability Supervisor v1

The supervisor coordinates generation recovery deterministically. It does not
perform additional LLM reasoning to classify errors, execute product tests,
approve candidates, or alter approved versions.

## Existing architecture and retry audit

The web application uses one provider-settings router for TestCase authoring
and TestPlan generation. Background Automation runs load an approved TestCase,
then enter `AutomationWorkflow → QATestPipeline → deterministic Discovery →
LLMTestPlanGenerator → AutomationReliabilitySupervisor → LLMRouter → provider`.
Returned candidates pass the existing schema/action, Discovery capability,
assertion-grounding and expected-result-coverage validators. Saving revalidates
the plan at the existing pipeline boundary and preserves immutable versions.

Before this change, the router fell through configured providers after a typed
retryable failure. It had no same-provider retry loop. The generator separately
made one unconditional repair request after a plan validation failure; that
request restarted the router's fallback chain. OpenAI-compatible SDK retries
could multiply those calls further. Groq used one HTTP request. Gemini's existing
Interactions retry configuration already disabled SDK retries.

The existing usage repository records provider attempts, operation context,
known tokens, latency and estimates from verified local pricing metadata. The
supervisor reuses those measurements and operation IDs; it introduces no price
discovery or billing synchronization. Authoring, background authoring and
optional AI Discovery retain their existing routing behavior outside the
supervised generation context. The application factory uses deterministic
Discovery for Automation.

Validation and Regression continue through `TestCaseExecutionService →
PinnedExecutionService`, executing exact saved plan IDs. Suite Runs retain
their exact pinned versions. These paths only consult saved repair-review
metadata; they perform no generation, provider calls or supervisor recovery.

## Settings and migration

| Control | Initial value | Effect |
| --- | --- | --- |
| Additional retries | OFF | Retry typed transient technical failures against the same provider. |
| Provider fallback | ON | Preserve the configured provider order and skip unavailable providers. |
| Automatic plan repair | OFF | Permit at most one correction of an unapproved generation candidate. |
| Maximum total attempts | 2 | Count the initial provider call and all generation recovery together. |

On first initialization of a database with at least three enabled persisted
provider configurations, the limit starts at **3** to preserve room for an
established three-provider fallback chain. Subsequent initialization never
overwrites saved settings. A longer legacy chain remains configured in the
same order, but only the first three eligible providers can be attempted in one
generation operation. Reorder providers deliberately if necessary. New databases
and shorter persisted chains start at 2. Environment-only chains also start at 2.

Legacy unconditional candidate repair becomes opt-in. Uncertain identity,
unconfirmed assertion values and missing input values stop for human review.
Execution drift in supervised Automation stops without legacy automatic locator
repair or execution of a repaired version. These are deliberate safety changes.
Provider configuration, secrets, old plans, Runs, evidence, cookies, approvals
and report formats are not migrated or rewritten. All recovery controls OFF
means no additional recovery; mandatory safety checks and recording remain on.

## Budget, timing and decisions

An operation is **one atomic TestStep's plan generation**, not a whole TestCase,
browser action, Validation or Regression. All initial, retry, fallback and
targeted-repair provider calls share the same 1–3 attempt limit. Repair starts
from the last provider; fallback advances in saved order and never cycles.

The generation/recovery deadline is 60 seconds. Provider calls receive at most
30 seconds, reduced by the remaining deadline. SQLite writes use a bounded
one-second lock wait. Final local persistence and process scheduling can add
small overhead to the deadline. OpenAI-compatible requests explicitly disable
SDK retries in this context; Gemini's existing retry suppression is retained;
Groq performs one HTTP request per attempt. Counts are provider invocations,
not a guarantee about every HTTP exchange or custom SDK internal behavior.

Retry uses bounded exponential delays starting at 0.5 seconds. A positive
Retry-After is respected only if it fits the five-second delay ceiling and the
remaining time. A longer delay skips same-provider retry, allowing configured
fallback if safe and within budget. Cancellation interrupts backoff and waiting;
the operation detail page provides Cancel while an operation is active.

| Failure | Permitted action |
| --- | --- |
| Typed timeout, rate limit or temporary provider unavailability | Retry if enabled; otherwise eligible typed fallback. |
| Authentication or model/configuration failure | Never retry the same provider; use fallback only when the existing error type permits it. |
| Invalid structured response or malformed candidate structure | One targeted repair when enabled and budget remains. |
| Missing expected-result coverage with explicit requirements | One targeted repair, preserving valid existing actions and assertions. |
| Unsupported action, missing input/expected/option value | Stop; no equivalence or value is assumed. |
| Ungrounded assertion, unestablished locator, changed repair semantics | Stop for human review. |
| Insufficient requirements, confirmed product failure, unknown or local infrastructure failure | Stop; no automatic recovery. |
| Attempt/time limit or cancellation | Stop immediately; no further recovery. |

The implementation uses existing provider error categories rather than a
parallel provider-error enum. Every candidate must pass schema/canonical action,
Discovery capability/selector identity, assertion grounding and coverage checks,
even when recovery is OFF. A structural repair must retain valid existing
actions, values, assertions and their order. Unsupported aliases and recovery
of missing values are intentionally outside the v1 repair whitelist.

## Review and immutable versions

Accepted repairs create a new version with `REPAIRED` provenance and a saved
operation/version association. The Automation pipeline stops before executing
that version, including on a later cached-plan retry. TestCase approval is left
unchanged. Complete candidates remain Needs Validation; passing generation does
not mark Automation Ready.

New repaired candidates require explicit Approve for Validation even for older
TestCases with no review row. Approval remains bound to the current definition
and exact saved-plan fingerprint. Browser Validation is separate. Existing
approved versions and historical execution pins are never overwritten.

## Persistent data and metric definitions

Two additive tables hold settings and operation snapshots:
`automation_reliability_settings` and `automation_reliability_operations`.
Operations include IDs, effective settings, actual timestamps/duration, safe
decisions, attempts, quality outcomes and optional candidate version IDs.
Attempts retain provider/model, action, reason, request duration, gate results,
failure category and only known token/cost metadata. Provider request duration
excludes quality checks; overall generation duration includes them and backoff.

Generation success means **quality gates accepted a candidate**. It does not
establish human approval, saved-plan Browser Validation, product correctness
or false-PASS frequency. No historical operations are reconstructed.

- Total operations includes running and interrupted records.
- First-attempt success divides accepted single-attempt operations by completed
  operations, including recorded failures/cancellations and zero-call stops.
- Recovery attempts counts all attempts after the first. Fallback count counts
  actual fallback attempts, not a decision cancelled before the next call.
- Recovery success divides accepted recovered operations by completed operations
  with more than one attempt.
- Gate rejections count actual failed gates by category. Gates after an earlier
  rejection are marked Not run, rather than guessed successful.
- Average attempts and elapsed duration use completed operations only. Interrupted
  operations retain unknown completion times and do not enter these averages.
- Provider/model counts use recorded attempt metadata, with Unknown when absent.
- Token and cost totals sum known values only, with coverage counts. Missing usage
  stays NULL; unverified prices do not create estimates or exact billing claims.

Startup reconciles unfinished operations as Interrupted without inventing an end
time. Late responses from an abandoned request cannot change the finished
reliability snapshot or start recovery. No prompts, candidate bodies, private
page content, credentials or raw exception messages enter reliability storage.
Metadata uses existing secret redaction; UI fields and secondary diagnostics
are escaped. Generation progress links to the exact operation decision history.

## Limits and local verification

A blocked custom provider cannot be forcibly killed from Python. The coordinator
stops waiting and starts no recovery while that call is abandoned; at most four
provider workers can remain in flight. Their slots stay occupied until they
return. Built-in transport timeouts bound ordinary requests. Data is retained
locally without a new pruning or large-scale analytics system.

All automated checks use fake providers or local HTTP/browser fixtures. The
complete local suite excludes `tests/integration/test_openrouter.py` (live API)
and the single Selenium public-site browser-discovery test. The workspace test
runner blocks external socket connections. Root manual provider/browser scripts
are not part of the local suite.

For a manual check without external access:

1. Open Settings → Automation Reliability; save each control and restart the app.
2. Use local provider fixtures and a localhost target. Inspect exact attempt
   actions, gate outcomes and settings snapshots from generation progress.
3. Enable repair, produce a structurally invalid candidate, then verify the saved
   repair is awaiting review and no browser execution occurred.
4. Review and explicitly Approve for Validation. Run local Browser Validation;
   inspect its exact version pin before running saved-plan Regression.
5. Confirm generation statistics do not change during Validation/Regression and
   that missing token/cost values remain Unknown.
