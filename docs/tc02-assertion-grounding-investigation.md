# TC-02 assertion grounding investigation

Target: `TC-02. Required Field Validation 2 M5`, operation
`86b0d9c5-48ec-4cff-8a7a-c31250b05900`.
Baseline: `ef0102d23a37fb9322a2d8621b27ff472ec2d963`. The unrelated `.env.example`
modification was preserved. No database, historical plan, run, credential or demo
application changes were made. No remote provider requests were made.
During this work, HEAD advanced to `a5126ca325311b905c8ba3ed98700dc78193d441`
through an independently created commit touching only `.env.example`. This
investigation did not commit anything; its grounding changes remain uncommitted.

## Historical evidence and its limits

The local database was inspected through an immutable, read-only SQLite URI,
without repository initialization or migrations. No WAL sidecar was present.
Only the specified operation and its associated TestCase were queried.

The operation records one provider SUCCESS and one rejected plan. Schema/actions
and locator identity passed; assertion grounding failed with
`UNGROUNDED_ASSERTION` at `steps[2].parameters`. Expected-result coverage and
step boundaries were NOT_RUN. The outcome was NEEDS_ATTENTION, with reason
UNKNOWN and root_cause_known false.

Diagnostic level was NORMAL and candidate_diagnostics was null. There were no
saved plan versions for the affected step, no associated run-history rows, and
no persisted Discovery or recovery tables. Recovery candidates and Discovery
are private, process-local, bounded, expiring data. No accessible historical
candidate was available to this investigation.
The live recovery UI was not used as a read-only evidence source: its GET fallback
can create an empty recovery draft, and the private process-local store has no
filesystem or SQLite representation to inspect directly.

Consequently, the historical action at index 2, its parameters, selector, exact
asserted value and Discovery snapshot remain **UNKNOWN / NOT TESTED**. Locator
identity PASSED does not establish that a selected element represented the
expected error. Provider SUCCESS does not establish correct generation or
product PASS. The generated assertion cannot conclusively be labeled an AI
mistake or a false rejection without that candidate. The tests below use clearly
labeled synthetic candidates; they do not reconstruct missing historical data.

## Confirmed reproductions

The saved original description is:

> Open the registration page. Leave all required fields empty and submit the form. Verify that registration is blocked, required fields show validation errors, and no successful registration confirmation appears.

The affected saved TestStep description is:

> Leave fields 'Email' and 'Password' empty and submit the form.

Its expected result is:

> The message 'Enter a valid email address.' is displayed.

1. **Source-authority mismatch, reproduced.**
   `assertion_grounding._requirement_texts` finds related original context and
   selects that description alone. It does not contain the exact email message.
   A pre-submit Discovery snapshot of the unchanged demo exposes a hidden,
   empty `#form-error`, with no message text. A fixture asserting that literal
   on the correct error container is therefore INFERRED and rejected. Without
   broader context, the explicit atomic requirement grounds it. An original
   description explicitly requiring the confirmed message also grounds it.
   The period and quote delimiters do not break literal grounding. The original
   source-authority safeguard remains enforced: there is no reliable step-origin
   metadata allowing the validator to promote arbitrary elaborated wording.
   This is a reproduced explanation compatible with the saved failure, not a
   claim about the missing historical assertion.
2. **Validator defect, confirmed independently.**
   `expected_result_coverage._expectation_kind` read `address` inside the quoted
   message and classified the visible-message result as a URL requirement.
   Even an explicitly grounded correct message failed coverage. Quoted `and`,
   `but` and `also` also split message content into separate expectations.
   Quoted content is now preserved as literal data and excluded from state
   classification and clause boundaries. This defect did **not** cause the
   historical assertion-grounding rejection: that operation never ran coverage.
3. **Diagnostic defect, confirmed.**
   Non-input exact assertion failures had no typed reason, so the diagnostic
   fallback was UNKNOWN. The validator now distinguishes a literal present
   only in the suppressed atomic step from a literal lacking authoritative or
   target-bound evidence altogether. Explicit UNKNOWN previously counted as a
   known cause because it is a nonempty string; that flag is corrected too.
4. **Target/evidence defects, confirmed with fixtures.**
   Output observation grounding used text globally, so text observed at another
   element could ground an assertion against an empty error container. Selected
   output targets now use only their own deterministic text evidence. Exact
   message parameters could also make a text assertion on an input count as
   coverage, despite the runner inspecting innerText rather than its DOM value.
   Such targets are rejected with ASSERTION_TARGET_MISMATCH. A conflicting but
   otherwise grounded message, or structural visibility alone, cannot cover a
   quoted exact-message requirement through semantic token overlap.

## Unchanged demo behavior and suggested requirement correction

Real local Chromium confirmed that both fields start empty and clicking
`#create-account` shows **Enter a valid email address.** in visible DOM element
`#form-error`. Email gets `aria-invalid="true"`; Password gets no validation
attribute. `#registration-success` remains hidden. Both input values remain
empty. The form uses `novalidate`, neither input is required, and Discovery
reports required_field_count 0. The JavaScript checks only whether Email contains
`@`; it does not validate Password or show per-field required errors.

The atomic email-message expectation matches the demo. The broader description's
claim that required fields show validation errors does **not** match this fixture.
This is a requirement/fixture mismatch, not evidence of a product regression.
The demo was not changed to satisfy it. Browser-native validation bubbles and
per-field password validation remain unsupported for this target.

Suggested original TestCase description, for human review:

> Open the registration page. Leave Email and Password empty and submit the form. Verify that the message 'Enter a valid email address.' is displayed. Verify that the successful registration confirmation is hidden.

Keep the affected step's exact email-message expected result. For the final
confirmation step, the supported observable wording is:

> The successful registration confirmation is hidden.

This removes the unsupported per-field requirement and makes the verified exact
message authoritative. It is a suggested correction only: no real TestCase or
history was rewritten. A separate target implementing per-field validation is
needed if that behavior is the actual requirement.

## Changed functions and validation

- `qa_agent/assertion_grounding.py`: `classify_assertions`,
  `validate_assertion_grounding`, new `_grounding_failure_reason` and
  `_assertion_observation_texts` and `_text_assertion_has_input_target`.
  Output reasons are value-free and target-bound;
  input-value grounding and original source precedence remain intact.
- `qa_agent/expected_result_coverage.py`: `_expectations`, `_expectation_kind`,
  `_result_clauses`, `_action_covers`, new `_quoted_values` and `_mask_quoted`.
  Paired quotes preserve internal punctuation and opposite quote characters.
  Exact positive message coverage needs an explicit matching text predicate.
- `qa_agent/diagnostic_mode.py`: `REASON_CODES` and `_failure_reasons`.
  New allowlisted reasons OUTPUT_VALUE_ONLY_IN_STEP and OUTPUT_VALUE_NOT_GROUNDED;
  insufficient typed evidence still yields UNKNOWN/root_cause_known false.
- `qa_agent/test_plan_generator.py`: `_build_task_context`. Instructions now explain authoritative
  output sources, quoted messages and the correct message-container action.
  Text assertions on input/textarea targets fail assertion grounding before
  generation acceptance, preserving the existing UNGROUNDED_ASSERTION public code.
- `docs/diagnostic-mode.md`: documented reason meanings and historical limits.

43 new regression cases: 42 deterministic cases in
`tests/test_message_grounding_regressions.py` and one real Chromium case in
`tests/integration/test_message_grounding_integration.py`. They cover exact TC-02
wording, source precedence, empty submission, quoted literals and compound
requirements, correct/wrong targets, conflicts, missing observations, untrusted
AI Discovery, specific/unknown diagnostics, and fail-closed generation outcomes.
The focused selection, including the previously fixed input-value suite and M22
compatibility tests, passed: **364 passed**. The browser test serves the existing demo HTML and JavaScript
without altering them, captures real pre-submit Discovery, uses fake providers,
and executes the accepted exact-message plan in a second isolated browser.
Temporary evidence stays in pytest's isolated directory.

Complete offline verification uses this command, with a unique temporary path:

```powershell
$qaGroundingTemp = Join-Path ([System.IO.Path]::GetTempPath()) ('qa-grounding-final-' + [guid]::NewGuid().ToString('N'))
.venv/Scripts/python.exe -m pytest -q --ignore=tests/integration/test_openrouter.py --deselect=tests/test_browser_discovery.py::BrowserDiscoveryTests::test_selenium_disabled_input_visible_text_selector_matches_its_element --basetemp $qaGroundingTemp -p no:cacheprovider --tb=short
git diff --check
git status --short
```

The only excluded tests are live OpenRouter integration and the established
public Selenium-site test. Local browsers require permission to start subprocesses
outside the Windows sandbox; no packages or browsers were installed.

Final complete offline run, after the last source/test change: **1,587 passed,
1 deselected, 141 subtests passed in 221.68 seconds**. Exit code 0. No source or
test files changed afterward. An earlier complete run caught a changed public
error code in the new input-target check. The check was moved to the grounding
gate to preserve UNGROUNDED_ASSERTION, focused compatibility tests passed, and
the complete suite was rerun successfully without excluding that test.

Live generation reliability and the unavailable historical candidate remain
NOT TESTED. All Quality Gates remain mandatory. Generation acceptance is separate
from actual browser execution, and no historical outcome was upgraded to PASS.
No commit or push was performed.

Final Git checks: `git diff --check` passed (exit 0), and `git status --short`
contains only this investigation's four modified source files, the modified
Diagnostic Mode documentation, the new report and the two new test files.
The patch is ready for review and commit. Applying the suggested real TestCase
correction still requires human requirement review; it was not applied by this work.
