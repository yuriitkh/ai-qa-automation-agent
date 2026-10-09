# AI QA Agent

## Run outcomes and reports

Run Progress, Run Details, TestCase history, suite summaries and dashboard counts
use outcome classification to interpret the internal execution status:

| Stored outcome | Public result |
| --- | --- |
| `PASSED`, with all steps verified | Passed |
| `PRODUCT_FAILURE` | Failed |
| `AUTOMATION_EXECUTION_ERROR` or `AUTOMATION_DRIFT` | Automation Error |
| `AUTOMATION_GENERATION_ERROR` | Generation Error |
| `INFRASTRUCTURE_ERROR` | Infrastructure Error |
| `SETUP_FAILURE`, missing automation or unmet prerequisites | Blocked |
| Missing, unknown or unclassified `FAILED` outcome | Inconclusive |

The internal `FAILED` status still describes an unsuccessful operation. It does
not establish a product defect. New executions retain the classification made
against their exact plan and assertion grounding in the existing runner JSON
payload (`qa_classification`). Safe history snapshots also retain the attempt
classification and redacted diagnostics. No SQLite table migration is needed,
and reading or regenerating a report does not rewrite historical records.

JSON reports preserve existing status, outcome, IDs and evidence fields and add
`display_status`, classifications and a structured result summary. Consumers
should use those presentation fields for user-facing results. For older runs,
a known aggregate outcome can classify a single unsuccessful attempt. Multiple
historical attempts without their own classification remain inconclusive at the
attempt level; the stored aggregate result remains visible. Suite reports add
`outcome_counts`, separating product failures, technical errors, blocked,
inconclusive and pending items. The public suite progress `failed` count is a
compatibility alias for `product_failures`. A passed item without a completed,
linked successful attempt is not counted as passed. Workflow counts on the
dashboard are separate from result counts; automation execution errors and
automation drift have separate counters.

Progress shows the current operation, meaningful terminal stage, verified step
count, stopping step, explanation and recommended action. Generation may stop
before Run History is saved; in that case there is no invented Run ID. Existing
partial diagnostics and Retry Automation remain available. Progress no longer
redirects automatically, so the summary can be reviewed. Its existing in-memory
retention still applies (24 hours and up to 500 finished requests by default).
Developer details group recorded events chronologically by stage, collapse
consecutive duplicate events, and show attempt numbers and durations only when
the corresponding events exist. Repair stages appear only when recorded.
Existing trace provider attempts show provider, model, outcome and duration when
recorded, without additional requests or reconstructed timestamps. Raw provider
responses and error messages are not added to progress.

Plan links resolve the exact persisted version used by an execution, including
after a newer version is saved. Run-linked plan pages are read-only and remain
available independently of the current TestCase definition. Unverifiable or
unsaved versions say **No saved TestPlan available.** Screenshots stay inside
their step and attempt with full-size links. Suite summaries link to individual
runs rather than collecting their screenshots. Evidence capture policies and
stored evidence associations are unchanged.

**View HTML Report** opens a clearly titled **HTML Test Report**, identifying the
Run and TestCase, with result summary, steps, attempts, evidence and **Back to
Run Details** navigation. Existing report URLs continue working. Actual
observations use recorded values or one complete explicit Playwright
`Actual value:` header; call-log fragments are not interpreted as product state.
Raw diagnostics remain expandable, escaped and redacted for credentials,
sensitive input calls and local paths. Recommended actions never alter expected
results, approve definitions, bypass validation or change saved plan versions.

To check the reporting flow locally, open a saved technical-error run and verify
that its dashboard/history label is Automation Error or Infrastructure Error.
Open Run Details, then its HTML and JSON reports. Follow each attempt's plan
link, check the version against its recorded pin, and open its evidence. Review
a partial generation request on Run Progress and confirm that its stopping step,
Retry Automation action and absence of a Run History link agree. Compare suite
category counts with the linked individual runs.

## LLM providers

The agent supports OpenAI, Google Gemini, OpenRouter, and Groq. Environment-only use gets its default order from `LLM_PROVIDER_ORDER`:

```env
LLM_PROVIDER_ORDER=openai,gemini,openrouter,groq
```

Provider credentials and model settings are `OPENAI_API_KEY` / `OPENAI_MODEL`, `GEMINI_API_KEY` / `GEMINI_MODEL`, `OPENROUTER_API_KEY` / `OPENROUTER_MODEL`, and `GROQ_API_KEY` / `GROQ_MODEL`. OpenRouter uses `https://openrouter.ai/api/v1` by default and can be changed with `OPENROUTER_BASE_URL`. See `.env.example` for a template. Keep real keys in your local `.env`; do not commit it.

### Web provider settings

Start the web application and open **Settings → AI Providers** (or visit
`http://127.0.0.1:8000/settings/providers`). The page shows each provider's
configured state, credential source, enabled state, model, and priority. Use
Enable/Disable and Move up/Move down to manage the chain. For example:

1. Groq
2. Gemini
3. OpenAI
4. OpenRouter

The web router tries enabled, configured providers in this order. It skips
providers without credentials and falls back to the next provider after a
retryable failure. Nonretryable errors keep the router's existing behavior and
stop that request. Before any web setting is saved, `LLM_PROVIDER_ORDER`
continues to provide the initial order; after an enablement, priority, or model
setting is saved, the persisted order and enablement control the web application.
The CLI continues to use its
environment-based order.

Environment variables remain supported. A key entered in the web settings takes
precedence over the corresponding environment key. Removing the web key falls
back to the environment value. On Windows, web-managed keys are stored in the
current user's Windows Credential Manager, separate from SQLite provider
settings. The database stores only provider ID, enabled state, order, and an
optional model override. The page never renders a full key; it displays only a
masked value. On non-Windows systems, environment keys work, but entering a
web-managed key is unavailable because this release does not add a cross-platform
credential-vault dependency.

**Test connection** sends a small structured-output request through that
provider's existing adapter with an eight-second request timeout. It does not
save the provider response. The page shows only a safe result and optional
latency. This is an explicit live provider request and may use provider quota;
automated tests replace adapters with fakes.

**Test authoring capability** sends a small TestCase-shaped request through
the same schema-constrained adapter and the same parser used by TestCase
authoring. It does not save a TestCase. A provider is configured when it has a
credential (or does not need one), but it is verified for authoring only after
this capability check passes. Connection and capability results are kept in
memory for the current server process; changing that provider's key, model,
enabled state, or custom configuration clears the prior results.

The connection and capability checks use the model, base URL, and current key
resolved by the same provider settings used to build the authoring router. The
connection check uses a tiny `{"ok": true}` schema. Authoring uses the complete
TestCase schema and strict response mode. Strict OpenAI-compatible schema mode
requires every declared object property to be required and optional values to
be nullable. Groq now sends that normalized schema; OpenRouter and other
OpenAI-compatible providers use the same normalizer. The application parser
still validates the returned TestCase structure. Gemini keeps its native JSON
schema response format and can be limited by provider quota.

Provider cards show three separate facts: configuration, the last connection
result, and the last authoring capability result. A successful connection does
not mark authoring as healthy. Failures use safe categories such as
`AUTH_ERROR`, `MODEL_NOT_FOUND`, `INVALID_REQUEST`, `RATE_LIMIT`, `TIMEOUT`,
`SCHEMA_ERROR`, `INVALID_RESPONSE`, and `PROVIDER_UNAVAILABLE`. HTTP status and
numeric `Retry-After` are shown when available. A rate limit without a reset
value says to try again later; the app does not invent a reset time or retry
aggressively. During authoring, progress reports the selected provider and
safe fallback reason. Server logs contain provider names and safe categories,
never credentials, prompts, or raw provider responses. Providers skipped for
missing credentials or because they are disabled are not counted as failed
requests in AI Usage.

API-key fields remain password inputs so pasted values stay obscured. They use
autocomplete and password-manager hints, and saved keys are never rendered
back into page HTML. Browser extensions may still choose to show a save prompt
based on their own heuristics.

### Manual provider diagnostics

These steps send real provider requests and may use quota. Automated tests do
not call provider APIs. No key is needed for the local demo page itself.

1. Start the local app with `python -m qa_agent.web --database .\\qa_agent.db`.
2. Open `http://127.0.0.1:8000/settings/providers` and configure Groq using a
   locally stored or environment key; do not paste a key into a shared log or
   screenshot.
3. On the Groq card, select **Test connection** and confirm the Connection
   result and latency appear.
4. Select **Test authoring capability** and confirm the Authoring capability
   result appears independently of Connection.
5. Repeat both checks for OpenRouter. Add or enable Gemini and repeat when
   quota is available.
6. Open **Test Cases → New Test Case**, use the local demo URL
   `http://127.0.0.1:8000/demo-target/registration`, enter a short scenario,
   and select **Generate Test with AI**. Confirm progress identifies the active
   provider and records fallback when one occurs.
7. If a provider fails, inspect its safe category and HTTP status in the
   provider card or progress page's **Developer details**. These views omit
   prompts, keys, and raw responses.
8. Open **AI Usage** and confirm only actual provider requests count as
   attempts; skipped unconfigured or disabled providers do not count as failed
   requests.
9. Save or replace a key on its provider card. Confirm the input is empty
   afterward and check whether the browser's password-save prompt is reduced
   or absent.
10. Submit the Dashboard or New Test Case form with an empty Website and then
    a malformed Website; confirm each inline message appears below the field.
    Submit an empty Scenario and confirm its inline message and preserved
    values.
11. Open **Test Cases** and review the compact rows and single-TestCase Export
    action. Open **Test Suites** and confirm each member uses one compact row
    with horizontally arranged, keyboard-accessible reorder and remove actions.
12. Add a TestCase without saved automation to a suite and export it. Confirm
    the page identifies the TestCase and step, explains that no executable plan
    is saved, and offers **Open TestCase** and **Back to Test Suite**.

OpenAI-compatible requests set an explicit output cap through `LLM_MAX_OUTPUT_TOKENS`, defaulting to 4096 tokens. This applies to OpenAI, OpenRouter, and configured compatible providers; Gemini's native provider keeps its existing request behavior.

For each routed request, the router reports configured priority and availability. Providers with missing keys are skipped. The first successful provider is selected. Compatible-provider request errors are logged without credentials and allow the router to try the next provider. Existing Gemini and Groq handling is retained, including Gemini's native structured Interactions API.

## Adding an OpenAI-compatible provider

Add its lowercase name to `LLM_COMPATIBLE_PROVIDERS` and `LLM_PROVIDER_ORDER`, then supply the uppercased name's three settings. No QA Agent code change is needed:

```env
LLM_COMPATIBLE_PROVIDERS=example
LLM_PROVIDER_ORDER=example,openai,gemini,openrouter,groq
EXAMPLE_API_KEY=...
EXAMPLE_BASE_URL=https://api.example.com/v1
EXAMPLE_MODEL=example-model
```

## Testing providers

Run unit tests with mocks using `pytest`. The optional OpenRouter integration test is separate and only runs when `OPENROUTER_API_KEY` is set:

```powershell
$env:OPENROUTER_API_KEY = "..."
pytest tests/integration/test_openrouter.py
```

OpenAI is ready to enable by setting the environment variable in PowerShell, then launching the agent from that same shell:

```powershell
$env:OPENAI_API_KEY = "..."
```

It will be tried first under the default priority; no source change is needed. No OpenAI key is needed for unit tests.

## Persistent run history and local UI

The CLI stores plan versions, executions, and completed run history in SQLite.
The default database is `~/.qa_agent/qa_agent.sqlite3`; set `QA_AGENT_DB_PATH`
or pass `--database PATH` to use another file:

```powershell
python -m qa_agent "Open https://example.com/ and verify the page title"
```

Start the local run browser with:

```powershell
python -m qa_agent.web
```

It binds to `127.0.0.1:8000` by default. The dashboard, saved TestCase
definitions, run details, JSON report, and standalone HTML report read from
the same SQLite database. Open a TestCase and choose Validation or Regression
to run its saved plan versions; use Automation first when a TestCase has no
complete set of executable plans. Pass `--database PATH` to use a non-default database. Screenshots
are served only when their recorded path is inside the configured evidence
directory; pass `--evidence-directory PATH` to choose that location.

### AI authoring demo

Configure at least one provider key from the [LLM providers](#llm-providers)
section, then start the local UI. The built-in deterministic registration page
is served by the UI itself, so no public site is needed:

```powershell
python -m qa_agent.web --database .\qa_agent.db
```

Open <http://127.0.0.1:8000>, choose **Test Cases → New Test Case**, enter a
name, the base URL `http://127.0.0.1:8000/demo-target/registration`, and a
natural-language scenario. Edit the proposed TestCase definition on the Review
page if needed, then save it and select
**Generate & Run Automation**. Once all steps have usable plan versions,
Validation and Regression become available for the saved versions.

### Demo UI

Seed a small set of safe demonstration runs, then launch the UI against the
same database:

```powershell
python -m qa_agent.demo --database .\qa_agent.db
python -m qa_agent.web --database .\qa_agent.db
```

Open <http://127.0.0.1:8000>. The seed includes an executable local
registration TestCase at `/demo-target/registration`; its deliberate missing
confirmation assertion demonstrates a real browser failure and screenshot.
Re-running the seed skips existing demo run IDs and preserves other run history.

### Live execution progress

Starting a saved TestCase run redirects to an opaque progress URL. The UI polls
real workflow events from a thread-safe in-memory store and links to Run Details
when the workflow writes its normal Run History record. Refreshing reconnects
to the same progress request; progress does not create placeholder TestRuns.
Finished progress is retained for up to 24 hours, with at most 500 finished
requests. A server restart clears in-memory progress; job recovery is not
persisted yet.

### Async AI TestCase authoring

Dashboard and New TestCase share one creation form, with **Summary (optional)**,
**Website**, and **Scenario** in that order. A supplied Summary is preserved
through generation and review. If it is empty, the app derives a short Summary
from the scenario without a separate AI request. Summary uses the existing
TestCase name field; no database migration is needed.

Both pages put **Drafts** beside the form at the same height on desktop and below
it on narrow screens. The panel shows up to 20 active Drafts, with an emphasized
Title and a Scenario preview limited to two lines. Its list scrolls internally;
**View all**, **New Draft**, and individual details remain accessible. Selecting
a Draft fills Summary from its Title, Website, and Scenario without changing its
status. All three fields remain editable. A Draft becomes Used only after a
TestCase is successfully saved, and remains available in Draft history.

The main actions are **Generate TestCase**, **Save Draft**, and **Create Manually**.
Manual creation transfers the current three fields without AI. Draft saving
allows unfinished or empty scenarios. Before generation, both the browser and
server reject obvious insufficient input, including empty text, isolated
characters, and repeated letters such as `aa`. Short meaningful scenarios such
as `Check login` remain accepted. Inline guidance lets users edit their Scenario
or continue manually.

Where supported, **Speak scenario** uses browser speech recognition and appends
editable text; the app receives text only, never audio. Generation redirects to
asynchronous progress, then to editable Review. **Generate Again** preserves the
current Summary and uses the original Scenario and Website. The proposal remains temporary until **Save TestCase
for review**. Saving keeps the existing review and approval states and does not
generate automation automatically. An unsuccessful request creates no proposal
or persisted TestCase; **Try Again** starts a new request with the original inputs.
Authoring uses a separate bounded worker pool. Review proposals and progress are
temporary and are cleared when the server restarts.

### Persistent Drafts and manual TestCases

Open **Drafts** in the main navigation to save, edit, reopen, or delete an
unfinished testing idea. Drafts are stored in the `drafts` SQLite table. They
remain separate from TestCases, Run History, and Test Suites. From a Draft, use
**Create TestCase manually** or **Generate with AI**; converting it leaves the
Draft in place until you explicitly delete it.

Dashboard and **New Test Case** both offer **Create Manually**. Manual creation does not
call an LLM and lets you enter a description, optional Base URL, preconditions,
and up to eight initial steps. Open a saved TestCase and choose **Edit TestCase**
to update its name, scenario, preconditions, and steps. Step actions preserve
their existing segment membership; add, delete, duplicate, and move operations
are scoped to one segment. A segment keeps at least one step.

AI accelerates authoring and automation, but existing tests, manual TestCase
editing, execution, validation, regression, export, and suite management remain
available without an LLM.

### Automation lifecycle

The TestCase page shows a lifecycle status:

- **Not automated**: there is no complete executable plan set.
- **Needs validation**: complete plans exist, but they have not passed Validation
  for the current definition and plan versions.
- **Automation ready**: the current definition and plan versions passed Validation.
- **Needs update**: the TestCase changed after automation was prepared. Run
  Automation to refresh the plans, then run Validation.
- **Automation needs attention**: automation preparation did not finish with a
  complete usable plan set.

Validation is the gate back to **Automation ready**. A Regression result that
finds a product failure does not invalidate otherwise ready automation. TestCase
edits keep all historical TestPlanVersions. **View TestPlan** is currently
read-only; structured manual plan editing belongs to a future Automation Editor
milestone. No user identity is stored or attributed.

### Windows Desktop launcher

From the repository folder, install two Desktop shortcuts with built-in
PowerShell and Windows shortcut support:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install-windows-shortcuts.ps1
```

Use **AI QA Agent** to start the local app or open an already healthy instance;
use **Stop AI QA Agent** to stop only the process owned by the launcher. A
different process on port 8000 is left alone and produces a clear error. The
launcher chooses `.venv\Scripts\python.exe` first, then
`.venv-original\Scripts\python.exe`; it does not fall back to arbitrary system
Python. It starts `qa_agent.web` with the repository's `qa_agent.db` and
`.evidence` directory. Runtime PID ownership data and server logs are kept under
the ignored `.runtime` folder.

The local `GET /health` endpoint returns only `status` and the application
service identifier. It does not expose provider settings, credentials, or paths.
The launcher waits for this endpoint before opening the browser. It does not
package an executable or install dependencies.

### Portable automation, code export, and Test Suites

Open a saved TestCase with a complete set of automation plans and use its
**Export** panel to download a versioned Portable TestPlan JSON file or ordinary
Playwright source for Python, TypeScript, or C#. Exports use the currently saved
TestPlanVersion for each TestStep. Export is deterministic and local: it does
not call an LLM, regenerate automation, or require the AI QA Agent runtime in
the generated tests. The source preserves the saved Playwright locator strings
and action order. Review target URLs and test values before running an export.
Portable JSON includes the TestCase name, base URL, preconditions, ordered
segments, and exact plan versions; it omits the original authoring scenario and
provider data.

From **Test Cases**, select multiple rows and choose Portable JSON or one
Playwright language to download a ZIP project. A Test Suite is an organizational
group with an explicit member order; create one from **Test Suites**, then add,
remove, reorder, or export its saved TestCases. ZIP files contain a README,
portable plan copies, a manifest with TestCase and plan-version provenance, and
the minimal project files for the selected target. Dependencies are listed but
never installed by AI QA Agent.

Portable TestPlan JSON is versioned and currently export-only. Import is not
implemented in this milestone. Export requires a valid saved plan for every
TestStep; a partial TestCase remains executable through the existing Automation
workflow but cannot be exported as a complete source project.
Portable JSON retains each TestStep failure policy. Generated source is one
standard Playwright test per TestCase, so the target test framework controls
whether later actions run after a failed assertion.

### System Readiness

Open **System Health** (`/system/health`) for local database, browser, evidence
storage, execution worker, and provider configuration diagnostics. **Refresh
checks** tests Chromium startup and creates then removes a small local evidence
probe. It never tests a live AI provider connection or consumes LLM tokens.
The lightweight launcher endpoint `/health` keeps its existing behavior.

TestCase and Suite setup forms offer **Check readiness**, and starting a run
rechecks readiness before queuing execution. Preflight has a 10-second aggregate
deadline, including Suite preview, platform checks, and all member targets;
each target has at most two seconds within that remaining budget. Duplicate
Suite targets share a probe. Timeout, diagnostic errors, and exhausted check
capacity block execution without creating a product-failure result.

Target checks use HEAD without redirects, cookies, or authorization. They pin
validated DNS addresses, reject credentials, common secret query parameters,
and private, link-local, multicast, and reserved addresses. Explicit localhost
and loopback development targets are allowed. HTTP error responses are
diagnostic warnings, not product assertions. HEAD relies on the target server
honoring HTTP safe-method semantics; readiness does not sandbox later browser
navigation, redirects, clicks, or subresources.

Saved-plan Validation and Regression do not check provider availability. Suite
runs preserve TestCase review approval, exact plan-version pins, and the chosen
evidence and cookie policies. A timed-out diagnostic may finish in a bounded
background worker; no additional target probes start after its deadline.

### Partial Automation and manual real-site checks

Automation saves each structurally validated plan version as its step finishes.
If generation stops later, those earlier versions remain available and the
TestCase coverage count shows the partial result. Validation and Regression
remain unavailable until every step has a usable saved plan. An Automation retry
reuses saved step plans and attempts steps that still need automation. A
generation failure before a completed TestRun stays in the progress record; it
does not create a placeholder Run History entry or report.

For optional manual checks against a public site:

- Choose a read-only scenario and avoid credentials, payments, or destructive actions.
- Expect dynamic page content, cookie banners, and timing to differ between visits.
- Review the progress classification and the exact step where generation or execution stopped.
- If generation saved only part of the TestCase, use **Retry Automation** and confirm saved plans are reused.

### AI usage analytics

The local UI includes **AI Usage** at `/settings/usage`, with Today, 7 days,
30 days, and All time views. It records one row per provider attempt, including
operation type, model, start and finish times, latency, outcome, fallback
relationship, and token counts only when the provider returns usage metadata.
Aggregates show raw success and failure counts; rates are marked as a limited
sample when there are fewer than five attempts. Providers are not automatically
ranked or labeled best.

Usage rows are stored in the same SQLite database as the application. Provider
response text, prompts, and API keys are not stored. Saved TestCases show their
associated AI usage after a reviewed draft is saved; abandoned drafts and older
TestCases are not retroactively attributed. Telemetry begins with requests made
after this feature is installed; past provider use and token counts are not
reconstructed. Provider-reported token counts may be unavailable, so the UI
labels them Unknown when missing.

Cost is an estimate, not a bill. No model price is shown until a verified rate
is deliberately configured in the local pricing catalog; therefore the cost
may appear as Unknown. If only some attempts have a configured rate, the total
is marked partial. Usage rows are retained in SQLite indefinitely by default;
estimates are captured when each request is recorded and are not recalculated
for older rows when the catalog changes. Include the database in your normal
retention and backup process.

### AI provider settings

Open **Settings → AI Providers** to compare configured, enabled, model, credential
source, and most recent connection-test status in compact cards. Use **Edit** for
model and credential controls. The built-in Groq, Gemini, OpenAI, and OpenRouter
adapters remain first-class; configured environment keys continue to work, and a
key saved from the local settings page takes precedence in the operating system
credential vault.

Use **Add provider** to configure an endpoint that implements the supported
OpenAI-compatible chat completions request and structured JSON response shape.
The application stores its generated stable provider ID, display name, HTTP(S)
base URL, model, enabled state, priority, and key-required flag in SQLite. A
custom provider's API key is stored only through the secret-store abstraction
(Windows Credential Manager on Windows); it is never written to SQLite or
rendered back. Endpoints without authentication are supported when the key
requirement is left unchecked. URL validation rejects embedded credentials,
query strings, fragments, malformed hosts, and non-HTTP(S) schemes. Localhost and
private network addresses are allowed. Saving or viewing settings does not make
a request to the configured URL; a connection test or an actual routed LLM
operation does.

Enabled providers are tried in persisted priority order. Disabled providers keep
their position but are skipped by the router. Move Up and Move Down reorder all
providers deterministically; priorities are compacted after add or deletion.
Test connection runs a small structured-output request and shows a safe status
and latency in the provider card. Retryable failures use the existing router
fallback behavior for authoring and automation.

Custom attempts use the generated provider ID for telemetry identity and record
the current display name with each attempt. Deleting a custom provider removes
its active configuration and saved credential but leaves historical usage
records and their recorded display names available in AI Usage. Usage tokens are
stored only when the endpoint returns recognizable usage metadata; otherwise
they remain Unknown. Custom provider cost remains Unknown unless a verified
catalog price is configured. Provider names can be changed in Edit without
changing the stable identity. Compatibility is limited to the supported
OpenAI-compatible API shape; this is not a universal adapter for every LLM
service.

### Automation Reliability

Open **Settings → Automation Reliability** at `/settings/reliability` to control
additional provider retries, configured-provider fallback, one safe candidate
repair, and a shared limit of 1, 2 or 3 generation attempts. Settings persist;
each new operation records its own effective settings. Quality gates always run.
Repaired candidates are saved for explicit human review and approval before
execution. Generation success does not mean Browser Validation or product PASS.

The page includes persistent generation statistics and individual attempt and
decision histories. Unknown token usage and unverified cost remain Unknown.
See [the reliability design and migration policy](docs/reliability.md) for
scope, limits, metric definitions, compatibility and local verification.
