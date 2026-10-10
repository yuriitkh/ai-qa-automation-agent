# M5 security, CI and Demo v0.1.0 verification

Baseline: clean main at `1e2b0a663568e03eb95e19a14846ce67b9c5bde8`.
Verification uses the existing Windows Python 3.13 virtual environment and local
Chromium. All new databases, screenshots and exported projects are temporary.
No real DB/evidence/credential data is modified; no dependency is installed;
no live provider, external network, GitHub, commit, push, tag or release is used.

## Confirmed vulnerabilities fixed

| Finding | Fix and regression evidence |
| --- | --- |
| CSRF checks covered only selected POST routes, allowing other mutations without a page token | Every HTTP POST validates CSRF and same-origin/cross-site context before route dispatch. All rendered and dynamic POST forms receive a token. Tests cover authoring, drafts, run, approvals, suites, provider settings, diagnostics, exports and readiness. |
| GET handlers accepted arbitrary Host headers, exposing read/download routes to DNS rebinding | The HTTP boundary requires exactly one unambiguous loopback Host with the listening port for reads and mutations. Tests reject hostile DNS names, credentials, wrong ports, paths, queries, fragments and duplicate Host headers. |
| Invalid Content-Length was treated as an oversized body; duplicate lengths and Transfer-Encoding were not rejected, and reads were unbounded | Reject malformed/missing/duplicate length or transfer framing, oversized/incomplete bodies and duplicate security headers; use a ten-second socket timeout. Tests verify rejection before dispatch and bounded local sockets. |
| Form size checked decoded characters rather than UTF-8 bytes | Check encoded bytes. Reject ambiguous fields, including duplicate CSRF fields, while permitting existing bulk-selection fields. |
| Default request logs included request targets/query strings; unexpected route exceptions lacked a safe HTTP response | Log only known method/status metadata and send generic JSON for unexpected exceptions. Sentinel tests prove request/exception secrets are absent from responses and request logs. |
| Responses lacked cache/referrer/framing/base/form restrictions | Add no-store, same-origin referrers, DENY and CSP restrictions while retaining the existing same-origin script/MIME policy; exercise the shipped script under actual CSP. |

The existing default binding was already loopback. Existing HTML escaping,
identifier checks, evidence-root traversal/association protections, diagnostic
redaction and export secret/lifecycle checks were inspected and preserved.
No newly discovered XSS or evidence traversal exploit is claimed. Unapproved
TestCases remain unable to execute any workflow, with focused service regressions.
Existing backward compatibility for historical cases and trusted in-process calls
is retained; actual HTTP dispatch always supplies headers and validates requests.

## CI

File: `.github/workflows/offline-tests.yml`. Python 3.13 on Ubuntu 24.04;
`requirements.txt`, Chromium/system libraries, `git diff --check`,
`git show --format= --check HEAD` for committed whitespace, and the complete
offline pytest command. The only exclusions are the live OpenRouter file and the
known public Selenium-site test. No `continue-on-error`, credential references,
artifact uploads or additional exclusions. A deterministic local YAML-parser
regression verifies triggers, permissions, dependencies and the exact test command.
Remote GitHub execution is **NOT TESTED** by instruction.

## Installation verification

The documented module entry points and flags match real `--help` output. A smoke
test runs the demo seed command against a temporary DB, constructs the actual Web
application with isolated evidence and an unavailable fake secret store, and
starts a real loopback HTTP server. `/health`, dashboard and registration fixture
return 200. All 39 installed versions match the pinned requirements and
`python -m pip check` reports no broken requirements. Existing dependencies and
Chromium are used; a fresh-machine install
and clean Ubuntu workflow execution are **NOT TESTED**. `.env.example` and the
guide explicitly describe process environment loading, which is not automatic.

## Automated demo smoke

`tests/integration/test_m5_demo_smoke.py` exercises a secret-free page-status
journey on the actual shipped registration fixture:

| Stage | Result |
| --- | --- |
| Persist TestCase in temporary SQLite; reject execution before review | PASS |
| TestCase human approval and deterministic fake LLM generation | PASS |
| Schema/actions, locator identity, assertion grounding, expected-result coverage and step boundaries | PASS: all five computed gates pass on each of three generated plans |
| Real Automation execution and required exact-version approval | PASS |
| Pinned Browser Validation in Chromium | PASS |
| Automation Ready derived from the actual Validation result | PASS |
| Standalone Python ZIP from approved/validated source | PASS |
| Independent `python -m pytest -q` outside the repository | PASS: 1 passed, no AI QA Agent imports/providers |
| Screenshots written only in temporary evidence storage | PASS |
| Missing expected assertion rejected without saved automation | PASS |

The provider is called exactly three times during generation and not again during
Validation/export/independent execution. Saved version IDs and the approved
fingerprint remain unchanged. The exported manifest keeps standalone execution
`NOT_TESTED` at export time; the smoke reports independent execution only after
the subprocess actually passes. No manual readiness marking or gate bypass.

## Final verification

After the last source/test change, the complete offline suite ran with:

```sh
python -m pytest -q --ignore=tests/integration/test_openrouter.py --deselect=tests/test_browser_discovery.py::BrowserDiscoveryTests::test_selenium_disabled_input_visible_text_selector_matches_its_element --tb=short
```

Result: **1,544 passed, 141 subtests passed, 1 deselected** in **221.56 seconds**,
exit code 0. The live OpenRouter file was ignored; the established public
Selenium-site test was the single deselection. No other exclusions or skipped
new failures. Earlier failures were corrected and this complete suite was rerun.
Normal Chromium form submissions use same-origin referrers; trusted redirect
anchor handling remains compatible; API test clients submit the page token.

New regressions: **55 collected cases** in two new modules: **48 HTTP/security/CI**
and **7 demo workflow/browser** cases. Existing network/CSP tests were adapted to
the intentional security contract. Source/test files were not changed after the
final passing suite; only this documentation result was recorded.

`git diff --check`: **PASS**, exit code 0 (Git's configured LF/CRLF conversion
notices are not whitespace errors). New-file whitespace check: **PASS**, six
files. `git status --short`: inspected; only the M5 changes listed below. Main
HEAD remains `1e2b0a663568e03eb95e19a14846ce67b9c5bde8`, with no commit/tag created.

Modified: `.env.example`, `README.md`, `qa_agent/web.py`,
`tests/integration/test_shipped_demo_reliability.py`,
`tests/integration/test_ui_execution.py`, `tests/test_execution_control_web.py`.
New: `.github/workflows/offline-tests.yml`, the three M5/demo documentation files,
`tests/test_m5_security_ci.py`, `tests/integration/test_m5_demo_smoke.py`.

## Limitations and readiness

Live LLM generation reliability, TypeScript runtime, C# runtime, fresh-machine
installation and actual GitHub execution: **NOT TESTED**. The registration fixture
has no real server-side registration/persistence/authentication; seeded historical
outcomes are synthetic and some seeded plans have incomplete coverage. The M5
export journey verifies page state rather than credential-bearing registration.
Password-field values intentionally block export. These limits block claims of
verified live providers, verified TS/C# execution or commercial production readiness.
They do not supply a fabricated PASS for any unexecuted feature.

Documentation: `README.md`, `docs/DEMO_V0_1_0.md`,
`docs/RELEASE_NOTES_v0.1.0.md`, and this record. **Ready for review/commit for the
documented local demo scope**: the final complete offline suite and whitespace
checks pass. No remaining local-demo blocker was observed in that scope. Broader
release claims require the untested provider/runtimes/install/remote CI checks
listed above. No commit or push is made.
