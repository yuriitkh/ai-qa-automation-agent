# M4 — Standalone Test Export

Implemented on local `main` at `6e6d9cca1a5ad19fe690f4172ee5e048254ee606`.
The initial working tree was clean. No commits or pushes were made.

## Existing weaknesses corrected

- Python ZIP files did not have pytest-discoverable filenames or an independent
  browser fixture. They required an additional pytest plugin and lacked pytest configuration.
- All languages lacked portable URL, browser and timeout configuration.
- Segment initialization was missing when the first action relied on an existing page.
- Generated code dropped TestStep failure policies. It now implements `CONTINUE`
  and `BLOCK_REST`, retains failures, and shares page state across steps.
- `assert_text_contains` invented a descendant text lookup beyond the actual
  runner's semantics. This failed a valid multiline assertion. All three emitters
  now assert visibility and inner-text containment on the saved target itself.
- Direct project downloads and bulk/suite routes could bypass the Export
  workspace's readiness checks. They now use a common eligibility check.
- Exported dependencies were loosely specified, TypeScript included an unnecessary
  compiler dependency, and C# file collision handling did not guarantee unique classes.
- Wall-clock manifest timestamps prevented byte-for-byte deterministic archives.
- Safety checks covered paths and credential URLs, but not known secrets or
  credential field values. Public renderers now enforce the same safety checks.
- The CLI had a task runner but no saved-project export command.

## Files and functions

| File | Changes |
| --- | --- |
| `qa_agent/testplan_export.py` | Existing `_emit_python`, `_emit_typescript`, `_emit_csharp`, `_code_for`, `_project_files`, `_project_readme`, `safe_basename`, `_zip_info`, `portable_zip`, and service methods improved. Added `_validate_export_safety`, `_configured_lines`, `_snapshot_timestamp`, and `TestPlanExportService.get_verified`. |
| `qa_agent/export_templates.py` | Standalone fixture, runtime configuration and Python CI templates consumed by the existing exporter; no separate export architecture. |
| `qa_agent/web.py` | Export service wiring, `_export_case_readiness`, `_test_case_export`, and `_test_suite_export` use shared eligibility. Bulk and central routes use the same service. |
| `qa_agent/cli.py` | `main` dispatches the new `_export_main` command. It reads a read-only SQLite backup into temporary storage and creates output exclusively. |
| `tests/test_standalone_export.py` | 60 new deterministic regression cases, including parameterized cases. |
| `tests/test_testplan_export.py` | Existing action/timing expectations adapted to configurable defaults and corrected containment semantics. |
| `tests/test_test_suites.py` | Successful project-download fixtures now have sufficient assertion coverage and Browser Validation metadata. |
| `README.md` | Usage, layouts, lifecycle requirements, configuration, security and CLI documentation. |
| `docs/M4_STANDALONE_EXPORT.md` | Implementation and verification report. |

## Exported layouts and support

Every project contains `README.md`, `export-manifest.json`, and
`portable/<safe-case-name>.testplan.json`. Existing language directories and
portable JSON schema version 1 are preserved. Project manifest version is 1.1.

```text
Python
  requirements.txt
  pytest.ini
  python/tests/conftest.py
  python/tests/test_tc_0001_standalone_registration.py
  .github/workflows/tests.yml

TypeScript
  package.json
  playwright.config.ts
  tsconfig.json
  typescript/tests/tc-0001_standalone-registration.spec.ts

C#
  AIQAAgent.Export.sln
  csharp/AIQAAgent.Export.csproj
  csharp/export.runsettings
  csharp/Tc0001StandaloneRegistrationTests.cs
```

| Target | Framework and dependencies | Verification |
| --- | --- | --- |
| Python | pytest 9.1.1 and Playwright 1.63.0, pinned from installed metadata; no pytest-playwright plugin | Actual collection and runtime execution passed against temporary loopback fixtures with installed Chromium. |
| TypeScript | Playwright Test pinned to installed upstream Playwright 1.63.0; no separate TypeScript compiler | Project/configuration checks and all-action TypeScript syntax parsing passed using bundled Node 24.21.0. Playwright Test execution is **NOT TESTED**: npm and `@playwright/test` are unavailable offline. |
| C# | Existing NUnit 3.14.0, Microsoft.Playwright.NUnit 1.44.0, Microsoft.NET.Test.Sdk 17.10.0, NUnit3TestAdapter 4.5.0; target net8.0 | Project/runsettings XML, solution references, test attributes and unique classes checked. Compiler syntax validation and `dotnet test` are **NOT TESTED**: .NET and packages are unavailable locally. |

The Python execution checks cover every canonical action, cross-step page state,
segment initialization, shared context across segments, exact/contains text,
URL/value/selection assertions, and explicit incorrect assertions. Failure-policy
checks observe requests to a loopback server to prove whether later steps ran.
The eligible-project discovery regression also runs the verified project in a
subprocess from its exported directory, with repository imports absent.

Configuration preserves saved URLs by default. `BASE_URL` explicitly replaces
the TestCase origin, preserving paths, queries, fragments and other origins.
Python/TypeScript support `BROWSER` and `HEADLESS`; C# uses NUnit runsettings and
standard Playwright test parameters. All targets support global/individual
positive timeout variables. TypeScript additionally supports `TEST_TIMEOUT_MS`.

The generated READMEs contain installation and CI commands. Python's standalone
GitHub Actions workflow installs dependencies and Chromium and runs pytest,
allowing failures to fail CI. Generating it does not contact any service.

## Eligibility, security and compatibility

Verified project downloads require valid saved canonical plans, sufficient
expected-result coverage through the existing lifecycle checks, current TestCase
and validation approval, and `AUTOMATION_READY`. The exporter binds eligibility
to the selected immutable version identities and rechecks definition/plan
fingerprints. Existing legacy approval behavior is retained; manual recovery
still requires explicit approval.

Single-file source downloads remain review exports. Direct renderer projects
default to `REVIEW_ONLY`. Verified manifests record
`APPROVED_AND_BROWSER_VALIDATED` separately from `standalone_execution: NOT_TESTED`;
exporting never invents an execution PASS. Draft/rejected candidates are not loaded
from diagnostic or draft stores. Unsupported actions/parameters fail explicitly.

Selectors and expected/input values remain unchanged, including values that
happen to resemble timeout arguments. Only the existing canonical newline
normalization is applied. C# remains NUnit; archive/download formats and portable
schema are retained. Python filenames and language test subdirectories change
to make framework discovery reliable. Version IDs and eligibility are additive
manifest metadata; `generated_at` now records the latest selected version's time.

Exports reject known registered/configured secrets, recognizable credentials,
credential field values, credential-bearing URLs and private local paths,
including serialized metadata. Failures do not echo those values or redact them
into different executable assertions. Archive paths and basenames reject/sanitize
traversal and Windows path forms. Arbitrary unlabeled values cannot always be
recognized as secrets; use secret-free plans and review test data. Preconditions
remain documented in portable JSON; account setup and authentication are not invented.

No new repository dependencies were introduced or installed. Verification used
isolated fixtures and temporary directories. No live LLM/provider calls, paid
requests, GitHub/network services, or real database/evidence/credential mutations
were part of this implementation.

CLI example:

```powershell
python -m qa_agent export --database .\saved.sqlite3 --test-case TC-0001 --format python --output .\standalone.zip
```

Repeat `--test-case` for multiple projects. `--source-review` produces one explicitly
labeled review source file. Existing output files are never overwritten, and the
source database is read through a read-only connection; repository migrations
operate solely on a temporary copy.

## Final verification

Final post-edit complete offline suite: **1,489 passed, 1 deselected, 141 subtests
passed in 194.58 seconds (3:14)**. No failures or skips. The ignored OpenRouter
file contains the one live provider integration test; the deselected item is
the one public Selenium-site test.

Exact command, with only the two authorized exclusions:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --ignore=tests/integration/test_openrouter.py --deselect=tests/test_browser_discovery.py::BrowserDiscoveryTests::test_selenium_disabled_input_visible_text_selector_matches_its_element --basetemp=.runtime/m4-full-offline-final
```

The first complete offline run passed 1,489 tests and 141 subtests with 1 test
deselected. After strengthening the eligible Python execution regression, that
focused check passed and the complete suite was restarted. No new failures were excluded.

Focused export/Web/CLI checks passed **123 tests and 10 subtests**. The subsequently
strengthened eligible-project discovery/execution regression also passed, followed
by the complete post-edit run above. There are **60 new regression cases**.

`git diff --check` passed. Final working tree changes are limited to the nine
files listed in this report. HEAD remains the baseline on `main`; no source or
test changes occurred after the final complete run began. Documentation was
updated to record its outcome.

Commit readiness: ready for review and commit with the stated TypeScript/C#
runtime limitations. Python has actual standalone runtime verification. Full
TypeScript execution and C# compilation/execution require the missing local
tooling/packages and remain **NOT TESTED**. Nothing was committed or pushed.
