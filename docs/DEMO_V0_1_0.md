# AI QA Agent Demo v0.1.0

This release preparation supports a local single-user demonstration. It does not
claim commercial production readiness, hosted service security or reliable live
LLM generation. No tag or published release is created by M5.

## Installation

Use Python **3.13**. The repository's `requirements.txt` pins the application and
test dependencies, including pytest and Playwright. Chromium is the browser used
by the complete offline suite. Dependency and browser installation needs network
access on a fresh machine; these instructions describe user installation, not an
installation performed during M5. No extra dotenv or pytest-playwright dependency
is required.

Windows PowerShell, from the checkout:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m playwright install chromium
Copy-Item .env.example .env
New-Item -ItemType Directory -Force .runtime/demo-v0.1.0 | Out-Null
```

Linux/macOS, with Python 3.13 already installed:

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m playwright install chromium
cp .env.example .env
mkdir -p .runtime/demo-v0.1.0
```

On Linux, `python -m playwright install --with-deps chromium` also installs the
system libraries needed by browsers; that may require administrative privileges.
Use a fresh demo directory, never a real customer or existing development DB.
The supported version is 3.13; other Python versions are not verified here.

## Configuration

The application reads process environment variables and provider settings.
It does not automatically read `.env`. Blank API keys are enough for offline
browsing, synthetic history, saved-plan workflows and tests. AI TestCase authoring
and new AI plan generation require a configured provider and may contact it.
Fake providers used by automated tests are not a production UI provider option.

If configuring a provider, edit a private `.env` based on `.env.example`, then
load the simple `NAME=value` assignments in the terminal that starts the app.
For PowerShell, the following loader does not execute the file as code:

```powershell
Get-Content .env | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith('#')) {
        $pair = $line.Split('=', 2)
        if ($pair.Length -eq 2 -and $pair[0] -match '^[A-Za-z_][A-Za-z0-9_]*$') {
            [Environment]::SetEnvironmentVariable($pair[0], $pair[1], 'Process')
        }
    }
}
```

For a trusted, locally edited POSIX shell environment file using shell-compatible
assignments, use `set -a; . ./.env; set +a`. Never source an untrusted file. Choose
`LLM_PROVIDER_ORDER` and the corresponding model/key variables from the example.
M5 does not verify the availability of listed remote models. Restart the server
after environment changes. Provider settings can retain their own configuration;
inspect the UI before intentionally enabling a provider. Avoid printing keys.
The Windows UI can store keys in the OS credential vault; other systems currently
use the environment fallback where secure credential storage is unavailable.

## Local startup and registration fixture

In the activated terminal (same commands work in both shells):

```sh
python -m qa_agent.demo --database .runtime/demo-v0.1.0/demo.sqlite3 --demo-base-url http://127.0.0.1:8000/demo-target/registration
python -m qa_agent.web --database .runtime/demo-v0.1.0/demo.sqlite3 --evidence-directory .runtime/demo-v0.1.0/evidence --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000/` and
`http://127.0.0.1:8000/demo-target/registration`. The health endpoint is `/health`;
the UI's System Health page gives readiness details. Stop the server with Ctrl+C.
Seeding is optional and skips existing seeded run IDs. It adds synthetic history,
not a certification that a live browser run succeeded. Reusing the directory
retains your demo edits, so choose another directory for a clean demonstration.

The server defaults to `127.0.0.1`, and HTTP requests require a loopback Host
matching the listening port. Use the documented local origin consistently. M5
provides no remote-serving, reverse-proxy or multi-user deployment contract.

## Features and supported workflows

* Drafts, manual TestCases, structured steps and definition editing; AI authoring
  when a provider is configured. Drafts must become reviewed TestCases before execution.
* Automation generation with schema/action, locator identity, assertion grounding,
  expected-result coverage and step-boundary Quality Gates. A rejected candidate
  cannot be treated as verified automation. Recovery candidates require review.
* TestCase approval and approval of the exact saved automation versions.
  Browser Validation must complete successfully before Automation Ready.
* Pinned Validation and Regression without new LLM generation, run progress,
  safe reports, screenshots, execution policies and local Test Suites.
* Standalone Python/pytest, TypeScript/Playwright Test and C#/NUnit project export,
  portable TestPlan JSON and explicitly labeled source-review export.
* Provider settings, usage accounting, readiness, reliability diagnostics,
  cancellation and manual recovery within their existing approval boundaries.

For the automated secret-free demo, the fixture journey opens the registration
page, clicks **Change page state**, and asserts that the status says
**Page state changed.** The scenario and expected results explicitly state that
text. The test drives real generation gates with a deterministic fake provider,
approves the generated versions, validates them in Chromium, exports Python and
runs its pytest project independently. It never sets Automation Ready manually.

The manual UI equivalent requires an actual configured LLM provider for generation;
its reliability is NOT TESTED here. Without one, use synthetic history to browse
the UI and the automated smoke below to verify the pipeline offline. Do not
interpret an absent provider, rejected plan, unexecuted step or seeded outcome as PASS.

## Architecture

`qa_agent.web` serves a local HTTP UI and same-origin browser fixtures. Services
own authoring, review, execution, export, suites and readiness. Canonical TestCases
contain steps grouped into execution segments; SQLite repositories persist cases,
versioned plans, review fingerprints, executions and history. The pipeline connects
deterministic browser discovery to the LLM router, generation Quality Gates and
the Playwright runner. Validation and Regression select exact saved plan versions.
Lifecycle metadata derives readiness from coverage and successful validation of
the approved versions. Reports refer to recorded outcomes and linked evidence.
Export snapshots saved automation into projects with their own runtimes and tests;
those Python projects do not import AI QA Agent or call its providers.

## Offline tests and CI

Run in the activated environment:

```sh
python -m pytest -q --ignore=tests/integration/test_openrouter.py --deselect=tests/test_browser_discovery.py::BrowserDiscoveryTests::test_selenium_disabled_input_visible_text_selector_matches_its_element
```

These are the only exclusions: the live OpenRouter integration and the established
public Selenium-site test. Browser fixtures and SQLite/evidence are temporary.
For the focused release smoke:

```sh
python -m pytest -q tests/integration/test_m5_demo_smoke.py
```

`.github/workflows/offline-tests.yml` uses Ubuntu 24.04, Python 3.13, pinned
dependencies and Chromium with system libraries. It runs `git diff --check` and
the complete offline suite, with no API secrets and no artifact upload. It also
checks committed whitespace with `git show --format= --check HEAD`. Failed
tests fail the job. YAML structure and commands are validated locally; a remote
GitHub run and a fresh Linux installation remain NOT TESTED.

## Standalone Python export

Project export requires current, approved, Browser Validated **Automation Ready**
plans. Use the case's displayed public ID or UUID. The seeded registration case
is not automatically eligible, and its credential fields can block safe export.
Use a secret-free case such as the page-status journey described above.

```sh
python -m qa_agent export --database .runtime/demo-v0.1.0/demo.sqlite3 --test-case TC-0001 --format python --output .runtime/demo-v0.1.0/standalone.zip
```

Replace `TC-0001` with the actual eligible case ID. The ZIP path must be new;
export refuses to overwrite it. Alternatively use the UI Export page after
approval and successful Validation. Export never requests a provider.
Unpack into a separate directory, create/activate a virtual environment there,
then run the generated project's commands:

```sh
python -m pip install -r requirements.txt
python -m playwright install chromium
python -m pytest -q
```

Keep the local target server running. The generated README documents `BASE_URL`
to replace the source origin, for example `$env:BASE_URL='http://127.0.0.1:8000'`
in PowerShell or `export BASE_URL=http://127.0.0.1:8000` in a POSIX shell. It must
be an origin without a path, credentials, query or fragment. Paths stay in the
exported plans. Inspect the generated README and manifest before execution.
Source approval/browser validation and standalone execution are separate claims;
the manifest says `standalone_execution: NOT_TESTED` until you actually execute
the project. Export itself does not update that field after an independent run.

## Security and privacy

This is a localhost single-user application, without authentication. Treat all
processes and users on the host as potentially able to reach it; CSRF and Host
checks are browser protections, not a substitute for OS access controls. Do not
expose its port to a network. POSTs need the current page's CSRF token and a
same-origin request; read/download routes validate the local Host and port to
reject DNS rebinding. Responses disable caching and framing, restrict form
submission and base URLs, and avoid disclosing unexpected exception text.
Referrers are limited to the same origin. Logs omit request targets, query values
and exception arguments.

Existing HTML escaping, evidence-root containment and export safety checks remain
in place. Invalid identifiers are rejected; evidence references must belong to
the recorded execution. Export refuses credential values/URLs and local paths
rather than silently changing executable assertions. Diagnostic exports are
redacted summaries, not raw provider output. Local DBs, screenshots, targets,
test inputs and evidence can still contain sensitive material. Use synthetic
data and inspect all artifacts before sharing. `.env` and runtime paths are
git-ignored. Do not publish databases, evidence or credentials with the demo.
Configured live generation sends scenario/discovery data to providers; review
that data before enabling it. M5 verification makes no live LLM/API request.

## Known limitations

* Live LLM generation reliability: **NOT TESTED**. Offline fixtures do not prove
  provider uptime, model quality, quota handling or real-world target reliability.
* TypeScript exporter runtime: **NOT TESTED**. Deterministic source/structure
  checks exist, but no installed Playwright Test/npm runtime was used here.
* C# exporter runtime: **NOT TESTED**. Source/project structure checks exist;
  compilation and NUnit execution require an available .NET toolchain.
* The registration fixture is a browser-only page, with no real accounts,
  server-side registration, persistence, email delivery or authentication.
  Seeded history is synthetic, and some seeded plans intentionally lack sufficient
  assertions. Passing the status journey does not verify full registration.
* Real password-field plans are intentionally blocked from standalone export.
  The automated M5 export journey uses no credentials.
* Voice input depends on browser support; complex targets, CAPTCHA, provider
  availability, cross-origin workflows and external integrations need separate
  verification. Portable JSON import is not supported.
* Fresh installation on a clean machine and actual GitHub execution are
  **NOT TESTED**. M5 uses the existing installed Windows/Python 3.13 environment.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| PowerShell activation is blocked | Use `.\.venv\Scripts\python.exe` instead of `python`; activation is optional. |
| Python or dependency import fails | Check `python --version` is 3.13 and install the pinned requirements inside the selected venv. |
| Chromium executable is missing | Run `python -m playwright install chromium` in that venv; Linux may also need `--with-deps`. |
| Port 8000 is already used | Choose `--port 8001` and seed with `--demo-base-url http://127.0.0.1:8001/demo-target/registration` in a fresh directory. |
| A POST returns request-validation failure | Reload after a server restart, use a single local origin and submit its current form token. |
| Host request rejected | Use `127.0.0.1` or `localhost` and the actual listening port; proxies/remote hosts are unsupported. |
| `.env` changes have no effect | Load the environment as above and restart; check stored provider configuration. |
| Generation/Validation/export is blocked | Read the specific Quality Gate, approval, coverage or lifecycle reason; fix the requirement or plan and approve/validate again. |
| Export reports sensitive data | Create a secret-free fixture plan; do not remove the export guard or redact executable assertions into a different test. |
| Exported test cannot reach target | Keep the target running and set a valid `BASE_URL` origin if its port changed. |

## Demo checklist

1. Use Python 3.13, the pinned dependencies and installed Chromium.
2. Create a fresh local demo DB/evidence directory and leave API keys blank for
   offline work. Load environment variables explicitly if live work is intended.
3. Start the server on loopback; verify `/health`, the dashboard and registration target.
4. Explain that seeded history is synthetic. Show manual TestCase editing/review,
   reports, evidence and readiness without presenting seed outcomes as verification.
5. Run the automated M5 smoke: generation, five gates, approval, real browser
   Validation, Automation Ready, Python ZIP and independent pytest execution.
6. If demonstrating live generation, identify it as separately unverified and
   stop on rejected gates. Export only approved, validated, secret-free versions.
7. Run the complete offline suite and `git diff --check`; inspect `git status --short`.
8. Review data before sharing. Report untested TS/C# runtimes, live providers,
   fresh installation and fixture limitations. No production-readiness claim.
