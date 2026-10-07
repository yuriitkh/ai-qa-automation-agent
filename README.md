# AI QA Agent

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

The Dashboard provides a direct authoring form for a website and a natural-language
scenario. If no name is supplied, the app derives a short name from the scenario.
Where supported, **Speak scenario** uses browser speech recognition and appends
editable text to the scenario; the app receives text only, never audio. Submitting
the form follows the same asynchronous progress and editable Review flow as the
dedicated New Test Case page.

Submit **Generate Test with AI** from **Test Cases в†’ New Test Case** to get an
immediate authoring progress page. The page polls actual provider and validation
events, then redirects to the existing editable Review page when the draft is
ready. **Generate Again** follows the same progress flow and uses the original
authoring inputs. Drafts remain temporary until **Save Test Case**. A failed
request creates no draft or persisted TestCase; **Try Again** starts a new
request with the original inputs. Authoring jobs use a separate bounded worker
pool so long runs cannot occupy authoring workers. Progress and drafts remain
in memory and are cleared when the server restarts.

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
