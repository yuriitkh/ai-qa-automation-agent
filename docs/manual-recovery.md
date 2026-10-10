# Manual Recovery (M3)

From Generation Decisions or the corresponding Run Progress TestStep, choose
**Edit rejected candidate**. The existing Automation Editor opens the private
temporary candidate alongside original requirements, safe failure codes, and
captured deterministic Discovery selectors. Add, remove, move and edit actions
with the existing controls. **Validate locally** previews errors without saving
a plan. **Save validated manual version** revalidates the submitted actions and
creates a new immutable `HUMAN_EDITED` version. Review and explicitly **Approve
for Validation** before running Browser Validation. A failed Validation leaves
Needs validation; successful generation/local validation never implies product
PASS or Automation Ready.

Both preview and save use the existing schema/action parser and
`LLMTestPlanGenerator._validate_generated_plan`: target compatibility, locator
identity, Assertion Grounding, Expected Result Coverage and Step Boundaries.
The original TestCase requirement context stays authoritative. Existing form
inputs or generated wording cannot independently establish assertion truth.
Changed original steps/context block recovery until the user discards and
reviews the current definition. The plan URL must match captured Discovery.

The new `ManualRecoveryStore` is an opt-in process-local draft cache, not a plan
store or logger. Web enables capture on its shared Reliability Supervisor;
explicitly supplied stores can enable it in isolated callers. Normal non-Web
generation does not retain recovery payloads by default. Typed candidates and
JSON objects rejected inside provider schema parsing can be captured; malformed
or incomplete JSON cannot be reconstructed. Raw candidates never enter audit
records, diagnostic exports or logs.

Recovery is keyed by the existing Reliability Operation ID and checked against
the exact original TestCase and TestStep. Reads and writes reject mismatched
identifiers. Mutations require the application's CSRF token, loopback host and
same-origin request checks. Editor output is HTML-escaped, no-store and uses a
same-origin referrer policy. Configured secrets, credential-bearing URLs and
password/token/key control values are cleared from editable candidate data;
credential-bearing edits cannot become saved recovery plans. Do not place raw
credentials in TestCase requirement text; its normal local review already
displays the original definition, with configured secrets redacted.

Each cache entry is capped at 64 KiB and 64 actions. At most 64 entries are kept,
for one hour from creation, pruned on cache access/write. Restart loses all
temporary recovery data. Successful generation removes its entry. Saving a
manual version clears candidate and draft payloads but keeps bounded Discovery
and requirement context until expiry, so further edits can run the same gates.
Discard clears the entire entry without changing plan history or TestCases.
Collection failures cannot change generation decisions or gate outcomes.

When the candidate is historical, expired, oversized, discarded or unavailable,
**Create manual recovery draft** opens an empty editor without inventing actions
or selectors. **Capture Browser Discovery** is an explicit browser-only action
against the TestCase's configured URL; it never requests an LLM. Without usable
Discovery, local validation reports Needs Attention. Older candidates and lost
private payloads cannot be reconstructed from diagnostic hashes.

Saved versions carry immutable `manual_recovery` metadata with the source
operation ID and previous version ID. SQLite adds one nullable metadata column;
historical version payloads remain intact. Existing approval fingerprints remain
unchanged and cease to match a new version. Manual recovery also requires explicit
review for legacy cases with no review record. Direct execution, pinned suite
execution and pipeline automation retain this check. Later edits of recovered
versions use recovery validation; the older schema-only save endpoint cannot
erase provenance. Drafts are absent from PlanStore and canonical automation
exports. No plan is automatically approved or executed.

If storage fails after a new immutable version has been saved, the UI reports
that recovery could not complete and asks the user to inspect saved versions.
It does not overwrite old versions or infer successful approval/Validation.
Operation audit metadata is required to reopen a recovered version; if it is
unavailable, the existing editor displays that version as read-only.

Regression tests use fake providers, synthetic Discovery, temporary SQLite and
loopback Chromium. The browser smoke covers rejected candidate → edit actions →
validate → save → review, without a second provider request.
