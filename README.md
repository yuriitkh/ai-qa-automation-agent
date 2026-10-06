# AI QA Agent

## LLM providers

The agent supports OpenAI, Google Gemini, OpenRouter, and Groq. The default fallback order is configured by `LLM_PROVIDER_ORDER`:

```env
LLM_PROVIDER_ORDER=openai,gemini,openrouter,groq
```

Provider credentials and model settings are `OPENAI_API_KEY` / `OPENAI_MODEL`, `GEMINI_API_KEY` / `GEMINI_MODEL`, `OPENROUTER_API_KEY` / `OPENROUTER_MODEL`, and `GROQ_API_KEY` / `GROQ_MODEL`. OpenRouter uses `https://openrouter.ai/api/v1` by default and can be changed with `OPENROUTER_BASE_URL`. See `.env.example` for a template. Keep real keys in your local `.env`; do not commit it.

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

Start the read-only local run browser with:

```powershell
python -m qa_agent.web
```

It binds to `127.0.0.1:8000` by default. The dashboard, TestCase history, run
details, JSON report, and standalone HTML report read from the same SQLite
database. To use a non-default database, pass `--database PATH` to the UI.
Screenshots are served only when their recorded path is inside the configured
evidence directory; pass the same `--evidence-directory PATH` used for the CLI.
The UI is read-only and does not start workflows.
