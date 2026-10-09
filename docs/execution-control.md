# Execution control and TestCase preferences

## Architecture findings

The committed Supervisor baseline already had an Event, interruptible retry
backoff, bounded provider workers, persisted attempts and mandatory quality
gates. Run jobs, authoring jobs and sequential Suite jobs had separate worker
pools, but no shared user cancellation state. Playwright uses synchronous
objects that must be closed on their owning thread. Run history already stores
exact execution and plan version references.

Evidence and cookie controls were repeated inside workflow forms. Their values
were submitted with a workflow rather than saved as TestCase preferences.
Authoring phase cards also sat outside the element searched by their poller.
Those cards now update correctly. Completed repaired candidates can now expose
Approve for Validation; that action still runs the existing approval checks.

## Cancellation design and supported operations

`CancellationToken` in `execution_control.py` supplies one Event, a completion
lock and a sealed terminal boundary. Context propagation connects background
jobs, the existing Supervisor, quality gates, provider routing, setup checks,
browser actions and pinned execution. Suite children share the parent's lock
and observe its request; completing a child seals only the child.

Stop is available on active authoring, Automation, Validation, Regression and
Suite progress pages. It covers Supervisor retries, fallback and repair within
Automation. The existing generation-operation cancellation form is protected
and connected to its parent Run. Pending jobs can also be stopped.

The UI acknowledges a request with Stopping… and keeps polling the backend.
Only terminal backend progress shows Stopped by user. Repeated requests are
idempotent; terminal operations reject Stop without changing their result.
Refreshing or reopening the progress page reads the same backend state.

Cancellation and final persistence use the same lock. If cancellation wins,
the owner records cancellation after safe cleanup. If completion wins, the
terminal token rejects cancellation. Atomic database operations finish normally.
Existing executions, evidence, confirmed defects and immutable pins survive.
Unstarted steps have no fabricated Execution and display Not attempted.

## Cancellation limitations

No arbitrary thread or process is terminated. Authoring and other provider
calls outside the Supervisor use four bounded isolated SDK slots; the existing
Supervisor retains its separate bounded provider capacity. Cancellation polling
is 50 ms. Late responses cannot publish a draft, change terminal progress, save
a plan or overwrite a completed reliability outcome. A worker retains its slot
until the actual SDK call returns. Permanently unresponsive SDKs can exhaust
those slots; subsequent jobs wait cooperatively and can still be stopped.

Supported SDK calls receive bounded provider timeouts; OpenAI-compatible SDK
retries are disabled within controlled operations. Existing Supervisor budgets
and recovery settings remain authoritative. Provider timing records measure
the application's actual wait, not an invented completion time for a detached
transport. Missing late token/cost metadata remains unknown.

An in-flight Playwright action finishes or reaches its existing timeout before
the owner closes pages, context, browser and runtime. Playwright close and
arbitrary custom setup/cleanup extensions do not provide a safe universal
interrupt deadline. If one hangs, progress stays Stopping… and shutdown can
wait for that owner; the app never simulates terminal cancellation. Cleanup
errors retain available partial execution references and safe diagnostics.
Synchronous readiness and provider connection checks have no new Stop job UI.
Queued jobs acknowledge Stop immediately; terminal confirmation can wait for
worker dispatch when all owning workers are occupied.

Run/authoring progress remains the existing in-memory service with its retention
limits and does not survive process restart. Saved Run history, preferences,
Suite history and reliability records do survive restart. This milestone does
not resume interrupted work.

## Persistent preferences and UI

One Execution preferences panel saves Cookie consent policy, Evidence mode and
Screenshot scope immediately. Each POST changes one field, validates its enum
and requires the current revision. SQLite uses `BEGIN IMMEDIATE` to read,
validate and commit a new revision atomically. Stale requests return HTTP 409
with the current revision rather than overwriting newer settings.

The browser serializes changes and retains the newest pending value per field.
It shows Saving…, Saved, or Save failed — Retry. Conflicts require an explicit
Retry; workflow submission waits until preferences are saved. Refresh and a new
application instance select the saved values. Existing defaults are preserved.
Legacy explicit per-run policy overrides still work without changing preferences.
Each new Run captures effective policy once; preference changes during that Run
affect future operations. Suite configuration remains a separate Suite snapshot.

The shared panel has one readiness form. Workflow forms retain their Automation,
Validation and Regression actions and contain no repeated preference controls.
Desktop/mobile browser checks verify control access and unique element IDs.

Save TestCase replaces the ambiguous Save TestCase for review label. Saving
still produces Ready for Review. TestCase approval, automation review, Approve
for Validation and successful Validation remain separate lifecycle gates.
Cancelled generation or Validation does not mark automation failed or ready.

## Supervisor, Suites and reporting

Cancellation prevents new provider attempts, backoff recovery, fallback and
repair. Supervisor cancellation is stored as CANCELLED and cannot count as
generation or recovery success. A generation operation genuinely completed
before a later parent Run Stop retains its actual earlier outcome. No quality
gate is weakened; checks are added between existing gates.

Suite Stop is durably acknowledged as CANCELLATION_REQUESTED. It propagates to
the active child, prevents retries/new members, keeps finished child Runs and
preserves the original version pins. Cancelled members and Not attempted members
are distinct. A partial Suite never displays Passed. Restart recovery preserves
CANCELLED and converts persisted cancellation requests to CANCELLED without
inventing unknown completion timestamps. Ordinary interrupted work retains its
existing interruption behavior.

HTML/JSON Run and Suite reports distinguish cancellation from blocked work,
generation, automation and infrastructure errors, and confirmed product defects.
A cancelled overall Run can still contain a previously confirmed Product Failure
step. Suite summaries add cancelled/not_attempted counts; old stored rows and
statuses are not rewritten. Diagnostics remain expandable.

## Database migration and security

The only new table is `test_case_execution_preferences`: TestCase UUID primary
key, revision, cookie policy, evidence mode and screenshot mode. It is created
additively when a SQLite-backed Web application initializes. Existing TestCases,
approval fingerprints, plans, executions and history are untouched. Cancellation
uses compatible new enum values and existing JSON storage; no destructive schema
migration is required.

New Stop/preference POSTs and the existing Supervisor cancellation POST require
an application nonce, validate Origin/Sec-Fetch-Site, and restrict HTTP Host to
loopback names. GET does not mutate state. Opaque progress identity and typed
resource lookup keep requests scoped to the intended job. There is no new
authentication or RBAC. Older mutation routes still lack equivalent CSRF
protection; the application remains intended for trusted local use. Those broader
security changes are outside this milestone.

## Verification

`tests/test_execution_control.py` covers authoring/provider cancellation, queued
job identity, retry/fallback/repair/gate boundaries, late response fencing, bounded
SDK slots, durable completion races, partial Automation history, Validation and
Regression, defects/evidence/pins, SQLite history round trips, lifecycle
preservation, enum validation, stale/concurrent preferences, CSRF and Suite
cancellation/restart recovery. Regression fixtures fail if AI is invoked.

`tests/test_execution_control_web.py` runs real Chromium against localhost:
immediate settings and queued changes, Save failed/Retry, refresh/restart/mobile
access, duplicate IDs, authoring Stop, Run Stop through blocked action/cleanup,
captured policy isolation, Suite Stop, and a real Playwright blocked click with
retained screenshot evidence. Existing authoring browser tests verify Save,
Review and Approve terminology and lifecycle.

Full regression command (external Python socket connections blocked by the
local runner):

```powershell
.\.venv\Scripts\python.exe .runtime/run_reporting_tests.py .runtime/control_full_verified.log tests --ignore=tests/integration/test_openrouter.py --deselect=tests/test_browser_discovery.py::BrowserDiscoveryTests::test_selenium_disabled_input_visible_text_selector_matches_its_element -q --tb=short -p no:cacheprovider
```

The ignored OpenRouter integration requires a live provider. The deselected
Selenium discovery case accesses a public site. Root manual live scripts are
outside the configured `tests` suite. No additional exclusions are used.

Final verification on 2026-10-09: **925 passed, 1 deselected, 141 subtests passed**
(120.30 seconds). The final focused queue/cancellation regression was **65 passed,
10 subtests passed**. The five new Chromium integration cases passed, including
real execution cancellation with screenshot retention; the existing related
authoring browser regression also passed. There are 36 new parametrized and
browser test cases beyond the committed baseline. `git diff --check` and a
separate whitespace scan of all five untracked files passed. The final working
tree contains 30 modified tracked files and five new files. No commit or push,
live LLM calls, public-site navigation, approved-plan mutation, fabricated
results/statistics, or quality-gate bypass was performed.

## Manual verification

1. Open a TestCase, change all three preferences, wait for Saved, refresh and
   restart the local server. Confirm the same values and one settings panel.
2. Change a preference in two tabs. A stale save must show Save failed — Retry
   instead of silently replacing a newer value. Retry intentionally or refresh.
3. Start authoring or an Automation/Validation/Regression Run against a local
   fixture. Stop, refresh during Stopping…, and wait for backend-confirmed
   Stopped by user. Open available Run details, evidence and generation decisions.
4. Stop a Suite with remaining members. Confirm finished child outcomes and pins
   remain, the active child is cancelled, and pending members are Not attempted.
5. Confirm terminal pages have no active Stop control. Save a newly authored
   TestCase and confirm Ready for Review; approval and Validation remain explicit.

## Changed files

Execution control: `execution_control.py`, `background_authoring.py`,
`background_execution.py`, `browser_runner.py`, `cookie_consent.py`,
`execution_progress.py`, `models.py`, `pinned_execution.py`, `pipeline.py`,
`plan_execution.py`, `setup_orchestration.py`, `test_case_execution.py`,
`test_case_authoring.py`,
`workflows.py` (all under `qa_agent/`).

Providers/reliability: `qa_agent/reliability.py`, `qa_agent/test_plan_generator.py`,
`qa_agent/llm/router.py`, `qa_agent/llm/openai_compatible.py`,
`qa_agent/llm/gemini.py`.

Storage/reporting/UI: `qa_agent/execution_preferences.py`,
`qa_agent/run_history.py`, `qa_agent/reporting.py`, `qa_agent/result_semantics.py`,
`qa_agent/suite_runs.py`, `qa_agent/suite_run_storage.py`,
`qa_agent/web.py`, `qa_agent/presentation.py`.

Tests: `tests/test_execution_control.py`, `tests/test_execution_control_web.py`,
`tests/test_automation_reliability.py`, `tests/test_result_semantics.py`,
`tests/test_web.py`, `tests/integration/test_async_authoring_web_integration.py`,
`tests/integration/test_dashboard_voice_integration.py`.

Documentation: this file. Test logs and implementation helpers are ignored under
`.runtime/`; they are not production changes.
