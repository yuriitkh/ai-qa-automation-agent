from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
from time import monotonic, sleep
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from qa_agent.automation_lifecycle import AutomationStatus
from qa_agent.drafts import Draft, DraftStatus, SQLiteDraftRepository
from qa_agent.llm.base import LLMProvider
from qa_agent.llm.router import LLMRouter
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
from qa_agent.test_case_authoring import TestCaseAuthoringService, TestCaseAuthoringError
from qa_agent.test_case_repository import InMemoryTestCaseRepository
from qa_agent.test_case_review import TestCaseReviewStatus as ReviewStatus
from qa_agent.web import LocalWebApplication


class _Provider(LLMProvider):
    def __init__(self):
        self.calls = []

    def create_test_plan(self, *_args):
        raise AssertionError("Creation must not generate automation")

    def create_structured_output(self, prompt, schema, schema_name):
        self.calls.append(prompt)
        return json.dumps({"name": "Generated Account Summary", "preconditions": [], "segments": [
            {"steps": [{"name": "Check login", "description": "Open the sign-in page.",
                         "expected": "The account page is displayed."}]}]})


class _Fields(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.fields = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag in {"input", "textarea"} and attributes.get("name") in {"name", "base_url", "scenario"}:
            self.fields.append(attributes)


class CreationUXTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.drafts = SQLiteDraftRepository(Path(self.temp.name) / "drafts.sqlite3")
        self.cases = InMemoryTestCaseRepository()
        self.history = RunHistoryService(InMemoryRunHistoryRepository())
        self.provider = _Provider()
        self.service = TestCaseAuthoringService(LLMRouter([self.provider]))
        self.app = LocalWebApplication(self.history, test_cases=self.cases, plan_store=InMemoryPlanStore(),
                                       drafts=self.drafts, authoring_service=self.service)

    def tearDown(self):
        self.app.close()
        self.temp.cleanup()

    def post(self, path, **fields):
        return self.app.handle("POST", path, urlencode(fields))

    def generated(self, **fields):
        values = {"name": "", "base_url": "http://127.0.0.1/sign-in", "scenario": "Check login"}
        values.update(fields)
        started = self.post("/test-cases/generate", **values)
        self.assertEqual(started.status, 303)
        progress_id = started.headers["Location"].rsplit("/", 1)[1]
        deadline = monotonic() + 3
        while monotonic() < deadline:
            progress = self.app.progress_store.get_authoring(progress_id)
            if progress.finished:
                self.assertTrue(progress.success)
                token = progress.review_url.rsplit("/", 1)[1]
                return token, self.app._draft_store.get(token)
            sleep(0.01)
        self.fail("Local authoring did not finish")

    def test_both_pages_share_order_optional_summary_and_actions(self):
        forms = []
        for path in ("/", "/test-cases/new"):
            html = self.app.handle("GET", path).body.decode()
            fields = _Fields(html).fields
            self.assertEqual([field["name"] for field in fields], ["name", "base_url", "scenario"])
            self.assertNotIn("required", fields[0])
            self.assertEqual(html.count('<h2 id="case-drafts-title">Drafts</h2>'), 1)
            self.assertNotIn("Recent Drafts", html)
            form = html.split('<form method="post" action="/test-cases/generate"', 1)[1].split('</form>', 1)[0]
            self.assertIn("Generate TestCase", form)
            self.assertIn("Save Draft", form)
            self.assertIn("Create Manually", form)
            self.assertIn("Speak scenario", form)
            self.assertEqual(form.count("formnovalidate"), 2)
            forms.append(form.replace('value="dashboard"', 'value="new"'))
        self.assertEqual(forms[0], forms[1])

    def test_regeneration_start_errors_preserve_review_edits_without_ai_calls(self):
        token, draft = self.generated(name="Original Summary")
        fields = {"name": 'Edited Summary <script>alert("x")</script>',
                  "description": "Edited description retained after an error."}
        failures = (
            (self.app, "_authoring_service", {"new": None}, 503),
            (self.app, "_background_authoring", {"new": None}, 503),
            (self.service, "validate_input", {"side_effect": TestCaseAuthoringError(
                "Please edit the Scenario.", category="INVALID_AUTHORING_INPUT")}, 400),
            (self.service, "validate_input", {"side_effect": RuntimeError("unavailable")}, 503),
        )
        for target, attribute, replacement, status in failures:
            with self.subTest(attribute=attribute, status=status), patch.object(target, attribute, **replacement):
                response = self.post(f"/test-cases/review/{token}/regenerate", **fields)
            self.assertEqual(response.status, status)
            html = response.body.decode()
            self.assertIn('Edited Summary &lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;', html)
            self.assertNotIn('<script>alert(', html)
            self.assertIn(fields["description"], html)
            self.assertIs(self.app._draft_store.get(token), draft)
            self.assertEqual(len(self.provider.calls), 1)
            self.assertEqual(self.cases.list(), [])

    def test_summary_and_review_edits_are_preserved_over_provider_suggestions(self):
        supplied = 'Account summary <script>alert("x")</script>'
        token, draft = self.generated(name=supplied)
        self.assertEqual(draft.test_case.name, supplied)
        review = self.app.handle("GET", f"/test-cases/review/{token}").body.decode()
        self.assertIn("&lt;script&gt;", review)
        self.assertNotIn(supplied, review)
        edited = "Reviewed account summary"
        saved = self.post(f"/test-cases/review/{token}/save", name=edited)
        self.assertEqual(saved.status, 303)
        case = self.cases.list()[0]
        self.assertEqual(case.name, edited)
        self.assertEqual(self.app._test_case_review.status(case.id), ReviewStatus.READY_FOR_REVIEW)
        self.assertEqual(self.app._automation_lifecycle.status(case), AutomationStatus.NOT_AUTOMATED)
        self.assertEqual(self.history.list_recent(), [])
        self.assertEqual(len(self.provider.calls), 1)

    def test_empty_summary_has_scenario_fallback_even_after_clearing_draft_title(self):
        source = Draft(title="Draft title to clear", body="Check login", base_url="http://127.0.0.1/sign-in")
        self.drafts.save(source)
        _token, draft = self.generated(name="  ", source_draft_id=str(source.id))
        self.assertIn("Sign In", draft.test_case.name)
        self.assertNotEqual(draft.test_case.name, source.title)
        self.assertNotEqual(draft.test_case.name, "Generated Account Summary")
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(self.drafts.get(source.id).status, DraftStatus.ACTIVE)

    def test_draft_populates_all_fields_on_both_pages_without_mutation(self):
        source = Draft(title="Account Title", body="Check the account details.", base_url="http://127.0.0.1/account")
        self.drafts.save(source)
        before = self.drafts.get(source.id)
        for path in ("/", "/test-cases/new"):
            html = self.app.handle("GET", f"{path}?draft_id={source.id}").body.decode()
            fields = _Fields(html).fields
            self.assertEqual(fields[0]["value"], source.title)
            self.assertEqual(fields[1]["value"], source.base_url)
            self.assertIn(f'>{source.body}</textarea>', html)
            self.assertEqual(self.drafts.get(source.id), before)
        self.assertEqual(self.provider.calls, [])

    def test_draft_preview_is_scenario_bounded_active_and_linked_to_details(self):
        for index in range(24):
            self.drafts.save(Draft(title=f"Account Draft {index}", body="Check login and account details. " * 30,
                                  base_url="http://127.0.0.1/secret-path"))
        used = Draft(title="Used should stay hidden", body="Check logout", status=DraftStatus.USED)
        self.drafts.save(used)
        html = self.app.handle("GET", "/").body.decode()
        self.assertEqual(html.count("data-draft-select "), 20)
        self.assertNotIn(used.title, html)
        self.assertIn('<span class="draft-scenario-preview">Check login', html)
        self.assertNotIn('<span class="draft-scenario-preview">http', html)
        self.assertIn("-webkit-line-clamp:2", html)
        self.assertIn("align-items:stretch", html)
        self.assertIn("overflow-y:auto", html)
        self.assertIn("@media(max-width:760px)", html)
        self.assertIn('href="/drafts"', html)
        self.assertIn('href="/drafts/new"', html)
        self.assertIn('class="draft-details"', html)

    def test_invalid_scenarios_are_rejected_by_route_and_service_before_ai(self):
        for scenario in ("", "  \n", "aa", "a", "xx", "aaaa", "!?", "123", "a b c"):
            with self.subTest(scenario=scenario):
                response = self.post("/test-cases/generate", authoring_entry="dashboard", name="Kept Summary",
                                     base_url="http://127.0.0.1/login", scenario=scenario)
                self.assertEqual(response.status, 400)
                self.assertIn(b' value="Kept Summary"', response.body)
                self.assertIn(b"Describe an action and what you expect", response.body)
                self.assertIn(b"Create Manually", response.body)
                with self.assertRaises(TestCaseAuthoringError):
                    self.service.generate(None, scenario, "http://127.0.0.1/login")
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(self.cases.list(), [])

    def test_short_meaningful_input_is_accepted_without_name_or_entry_marker(self):
        _token, draft = self.generated(scenario="Check login")
        self.assertEqual(draft.test_case.description, "Check login")
        self.assertEqual(len(self.provider.calls), 1)

    def test_drafts_save_with_incomplete_scenario_and_reopen_without_schema_change(self):
        for scenario in ("", "aa", " "):
            response = self.post("/test-cases/drafts/save", name="Unfinished account idea",
                                 base_url="not yet a website", scenario=scenario)
            self.assertEqual(response.status, 303)
        drafts = SQLiteDraftRepository(self.drafts._path).list()
        self.assertEqual(len(drafts), 3)
        self.assertTrue(all(draft.title == "Unfinished account idea" and draft.status == DraftStatus.ACTIVE for draft in drafts))
        self.assertEqual(self.provider.calls, [])
        blank = self.post("/test-cases/drafts/save")
        self.assertEqual(blank.status, 303)
        self.assertEqual(self.drafts.list()[0].title, "Untitled Draft")

    def test_manual_prepare_submits_current_fields_and_save_marks_draft_used_only_after_success(self):
        source = Draft(title="Original Title", body="Check login")
        self.drafts.save(source)
        prepared = self.post("/test-cases/manual/prepare", name="Edited Summary",
                             scenario="Edited scenario <unsafe>", base_url="http://127.0.0.1/edited",
                             source_draft_id=str(source.id))
        self.assertEqual(prepared.status, 200)
        self.assertIn(b'value="Edited Summary"', prepared.body)
        self.assertIn(b"Edited scenario &lt;unsafe&gt;", prepared.body)
        self.assertIn(b'value="http://127.0.0.1/edited"', prepared.body)
        self.assertEqual(self.drafts.get(source.id).status, DraftStatus.ACTIVE)
        fields = dict(name="Edited Summary", description="Edited scenario", base_url="http://127.0.0.1/edited",
                      source_draft_id=str(source.id), step_name_0="Sign in", step_action_0="Open sign-in page")
        self.assertEqual(self.post("/test-cases/manual", **fields).status, 400)
        self.assertEqual(self.drafts.get(source.id).status, DraftStatus.ACTIVE)
        fields["step_expected_0"] = "The account page is displayed."
        self.assertEqual(self.post("/test-cases/manual", **fields).status, 303)
        case = self.cases.list()[0]
        self.assertEqual(case.name, fields["name"])
        self.assertEqual(case.description, fields["description"])
        self.assertEqual(case.base_url, fields["base_url"])
        self.assertEqual(self.app._test_case_review.status(case.id), ReviewStatus.READY_FOR_REVIEW)
        self.assertEqual(self.app._automation_lifecycle.status(case), AutomationStatus.NOT_AUTOMATED)
        self.assertEqual(self.drafts.get(source.id).status, DraftStatus.USED)
        self.assertIsNotNone(self.drafts.get(source.id))
        self.assertEqual(self.provider.calls, [])

    def test_validation_roundtrip_does_not_refill_a_manually_cleared_summary(self):
        source = Draft(title="Keep only on selection", body="Check login")
        self.drafts.save(source)
        response = self.post("/test-cases/generate", authoring_entry="dashboard", name="", base_url="",
                             scenario="Check login", source_draft_id=str(source.id))
        self.assertEqual(response.status, 400)
        self.assertEqual(_Fields(response.body.decode()).fields[0]["value"], "")
        self.assertIn(f'value="{source.id}"'.encode(), response.body)
        self.assertEqual(self.provider.calls, [])
