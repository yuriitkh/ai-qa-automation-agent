from urllib.parse import urlencode, urlsplit
from uuid import UUID
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
import io
from threading import Event
from pathlib import Path
from unittest.mock import patch

from qa_agent.llm.base import LLMProvider
from qa_agent.llm.errors import RetryableLLMError
from qa_agent.llm.router import LLMRouter
from qa_agent.models import QATestPlan
from qa_agent.drafts import Draft, DraftStatus
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.provider_settings import UnavailableSecretStore
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
from qa_agent.test_case_authoring import TestCaseAuthoringService
from qa_agent.test_case_repository import InMemoryTestCaseRepository
from qa_agent.web import LocalWebApplication, WebResponse, create_application


class AuthoringProvider(LLMProvider):
    def __init__(self, output=None, error=None):
        self.output = output or (
            '{"preconditions":[{"description":"A test email is available."}],'
            '"segments":[{"steps":[{"name":"Open registration",'
            '"description":"Open the registration page.",'
            '"expected":"The registration page is visible."}]}]}'
        )
        self.error = error
        self.prompts = []

    def create_test_plan(self, task, target_url, page_snapshot):
        return QATestPlan(url=target_url, steps=[{"action": "assert_page_loaded"}])

    def create_structured_output(self, prompt, schema, schema_name):
        self.prompts.append((prompt, schema_name))
        if self.error:
            raise self.error
        return self.output


class TestCaseAuthoringWebTests(unittest.TestCase):
    def setUp(self):
        self.cases = InMemoryTestCaseRepository()
        self.history = RunHistoryService(InMemoryRunHistoryRepository())
        self.provider = AuthoringProvider()
        self.plans = InMemoryPlanStore()
        self.app = LocalWebApplication(
            self.history,
            test_cases=self.cases,
            authoring_service=TestCaseAuthoringService(LLMRouter([self.provider])),
            plan_store=self.plans,
        )
        original_handle = self.app.handle
        def authorized_handle(method, target, body=None, *, headers=None):
            # These fixtures represent an editor loaded from this application.
            return original_handle(method, target, body, headers=headers or {"X-QA-CSRF": self.app._csrf_token})
        self.app.handle = authorized_handle

    def tearDown(self):
        self.app.close()

    def submit_generate(self, *, name="User registration", scenario="Register a new user."):
        return self.app.handle(
            "POST",
            "/test-cases/generate",
            urlencode({
                "name": name,
                "base_url": "http://127.0.0.1:8000/demo-target/registration",
                "scenario": scenario,
            }),
        )

    def wait_for_authoring(self, response, *, app=None, timeout=3):
        app = app or self.app
        self.assertEqual(response.status, 303)
        progress_url = response.headers["Location"]
        self.assertIn("/test-cases/authoring-progress/", progress_url)
        progress_id = progress_url.rsplit("/", 1)[1]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = app.handle("GET", f"/api/test-cases/authoring-progress/{progress_id}")
            if result.status == 200:
                snapshot = json.loads(result.body)
                if snapshot["finished"]:
                    return snapshot
            time.sleep(0.005)
        self.fail("Authoring request did not finish in time.")

    def post_generate(self, *, name="User registration", scenario="Register a new user."):
        with redirect_stdout(io.StringIO()):
            response = self.submit_generate(name=name, scenario=scenario)
            snapshot = self.wait_for_authoring(response)
        if not snapshot["success"]:
            self.fail(f"Authoring unexpectedly failed: {snapshot['error_category']}")
        return WebResponse.redirect(snapshot["review_url"])

    def test_list_and_form_expose_ai_authoring(self):
        listing = self.app.handle("GET", "/test-cases")
        page = self.app.handle("GET", "/test-cases/new")
        self.assertIn(b"+ New Test Case", listing.body)
        self.assertIn(b'method="post" action="/test-cases/generate"', page.body)
        self.assertIn(b"Generate TestCase", page.body)
        self.assertIn(b'name="scenario"', page.body)
        self.assertIn(b"data-authoring-form", page.body)

    def test_persistent_draft_is_used_only_after_generated_testcase_is_saved(self):
        saved_draft = Draft(
            title="Registration idea",
            body="Register a user and confirm the page appears.",
            base_url="http://127.0.0.1:8000/demo-target/registration",
        )
        self.app._drafts.save(saved_draft)
        generated = self.app.handle("POST", "/test-cases/generate", urlencode({
            "authoring_entry": "new",
            "source_draft_id": str(saved_draft.id),
            "base_url": saved_draft.base_url,
            "scenario": saved_draft.body,
        }))
        progress = self.wait_for_authoring(generated)
        self.assertEqual(self.app._drafts.get(saved_draft.id).status, DraftStatus.ACTIVE)
        token = progress["review_url"].rsplit("/", 1)[1]
        proposal = self.app._draft_store.get(token)
        self.assertEqual(proposal.source_draft_id, saved_draft.id)
        review_page = self.app.handle("GET", progress["review_url"])
        self.assertIn(b"Review TestCase", review_page.body)
        self.assertIn(b"The registration page is visible.", review_page.body)

        saved = self.app.handle(
            "POST", f"/test-cases/review/{token}/save", ""
        )
        self.assertEqual(saved.status, 303)
        case_id = UUID(urlsplit(saved.headers["Location"]).path.split("/")[2])
        converted = self.app._drafts.get(saved_draft.id)
        self.assertEqual(converted.status, DraftStatus.USED)
        self.assertEqual(converted.converted_test_case_id, case_id)
        self.assertIsNotNone(converted)
        self.assertEqual(
            self.app._test_case_review.status(case_id).value, "READY_FOR_REVIEW"
        )

    def test_dashboard_form_uses_async_authoring_and_derives_name_when_omitted(self):
        scenario = "Check account details and sign in. Confirm the welcome page appears."
        with redirect_stdout(io.StringIO()):
            response = self.app.handle(
                "POST",
                "/test-cases/generate",
                urlencode({
                    "authoring_entry": "dashboard",
                    "base_url": "https://example.test/account",
                    "scenario": scenario,
                }),
            )
            progress = self.wait_for_authoring(response)

        self.assertTrue(progress["success"])
        self.assertEqual(len(self.provider.prompts), 1)
        self.assertIn(scenario, self.provider.prompts[0][0])
        draft_token = progress["review_url"].rsplit("/", 1)[1]
        draft = self.app._draft_store.get(draft_token)
        self.assertIsNotNone(draft)
        self.assertEqual(draft.test_case.name, "Check account details and sign in Confirm welcome page appears")
        self.assertEqual(self.cases.list(), [])

    def test_dashboard_invalid_input_does_not_start_authoring(self):
        with redirect_stdout(io.StringIO()):
            response = self.app.handle(
                "POST",
                "/test-cases/generate",
                urlencode({
                    "authoring_entry": "dashboard",
                    "base_url": "not-a-url",
                    "scenario": "Check the account page.",
                }),
            )

        self.assertEqual(response.status, 400)
        self.assertIn(b"What do you want to test?", response.body)
        self.assertIn(b"Enter a valid URL, for example https://example.com", response.body)
        self.assertEqual(self.provider.prompts, [])

    def test_post_redirects_before_provider_finishes_and_progress_refresh_does_not_restart(self):
        entered = Event()
        release = Event()

        class BlockingProvider(AuthoringProvider):
            def create_structured_output(inner_self, prompt, schema, schema_name):
                inner_self.prompts.append((prompt, schema_name))
                entered.set()
                release.wait(3)
                return inner_self.output

        provider = BlockingProvider()
        app = LocalWebApplication(
            self.history,
            test_cases=self.cases,
            authoring_service=TestCaseAuthoringService(LLMRouter([provider])),
            draft_store=self.app._draft_store,
        )
        try:
            with redirect_stdout(io.StringIO()):
                response = app.handle(
                    "POST",
                    "/test-cases/generate",
                    urlencode({
                        "name": "Async registration",
                        "base_url": "http://127.0.0.1:8000/register",
                        "scenario": "Register and verify confirmation.",
                    }),
                )
                self.assertEqual(response.status, 303)
                progress_id = response.headers["Location"].rsplit("/", 1)[1]
                self.assertTrue(entered.wait(1))
                page = app.handle("GET", response.headers["Location"])
                refreshed = app.handle("GET", response.headers["Location"])
                self.assertEqual(page.status, 200)
                self.assertIn(b"AI TestCase authoring", page.body)
                self.assertEqual(len(provider.prompts), 1)
                self.assertEqual(app._draft_store._drafts, {})
                current = json.loads(app.handle(
                    "GET", f"/api/test-cases/authoring-progress/{progress_id}"
                ).body)
                self.assertEqual(current["kind"], "AUTHORING")
                self.assertFalse(current["finished"])
                self.assertNotIn("scenario", current)
                self.assertIn("Generating TestCase", current["phase"])
                self.assertEqual(refreshed.status, 200)
                release.set()
                completed = self.wait_for_authoring(response, app=app)

            self.assertTrue(completed["success"])
            self.assertIn("/test-cases/review/", completed["review_url"])
            self.assertEqual(len(provider.prompts), 1)
            review = app.handle("GET", completed["review_url"])
            self.assertIn(b"Review TestCase", review.body)
            saved = app.handle(
                "POST",
                completed["review_url"] + "/save",
                urlencode({"name": "Async registration edited"}),
                headers={"X-QA-CSRF": app._csrf_token},
            )
            self.assertEqual(saved.status, 303)
            self.assertEqual(self.cases.list()[0].name, "Async registration edited")
            self.assertEqual(self.history.list_recent(), [])
        finally:
            release.set()
            app.close()

    def test_invalid_input_stays_synchronous_and_unknown_progress_is_not_found(self):
        with redirect_stdout(io.StringIO()):
            invalid = self.app.handle(
                "POST",
                "/test-cases/generate",
                urlencode({"name": " ", "base_url": "not-a-url", "scenario": "Check."}),
            )
        self.assertEqual(invalid.status, 400)
        self.assertNotIn(b"Enter a TestCase name", invalid.body)
        self.assertIn(b"Summary (optional)", invalid.body)
        self.assertEqual(self.provider.prompts, [])
        self.assertEqual(self.app._draft_store._drafts, {})
        self.assertEqual(
            self.app.handle("GET", "/test-cases/authoring-progress/unknown-id").status,
            404,
        )
        missing_json = self.app.handle(
            "GET", "/api/test-cases/authoring-progress/unknown-id"
        )
        self.assertEqual(missing_json.status, 404)
        self.assertIn(b"no longer available", missing_json.body)

    def test_generation_review_and_explicit_save_persist_without_run_history(self):
        with redirect_stdout(io.StringIO()):
            response = self.post_generate()
        self.assertEqual(response.status, 303)
        review_path = response.headers["Location"]
        self.assertIn("/test-cases/review/", review_path)
        token = review_path.rsplit("/", 1)[1]
        review = self.app.handle("GET", review_path)
        self.assertEqual(review.status, 200)
        self.assertIn(b"Review TestCase", review.body)
        self.assertIn(b'class="testcase-editor"', review.body)
        self.assertIn(b".testcase-editor textarea{min-height:12rem}", review.body)
        self.assertIn(b'name="name"', review.body)
        self.assertIn(b'value="User registration"', review.body)
        self.assertIn(b'name="description"', review.body)
        self.assertIn(b">Register a new user.</textarea>", review.body)
        self.assertIn(b'name="preconditions"', review.body)
        self.assertIn(b">A test email is available.</textarea>", review.body)
        self.assertNotIn(b'name="segment.0.step.0.name"', review.body)
        self.assertIn(b'data-structured-editor', review.body)
        self.assertIn(b'name="steps_json"', review.body)
        self.assertIn(b"Open the registration page.", review.body)
        self.assertIn(b"The registration page is visible.", review.body)
        draft = self.app._draft_store.get(token)
        self.assertIsNotNone(draft)
        for internal_id in (
            draft.test_case.id,
            draft.test_case.preconditions[0].id,
        ):
            self.assertNotIn(str(internal_id).encode(), review.body)
        self.assertEqual(self.cases.list(), [])

        saved = self.app.handle("POST", f"/test-cases/review/{token}/save", b"")
        self.assertEqual(saved.status, 303)
        self.assertTrue(saved.headers["Location"].startswith("/test-cases/"))
        test_case = self.cases.list()[0]
        self.assertEqual(saved.headers["Location"], f"/test-cases/{test_case.id}")
        self.assertEqual(self.history.list_for_test_case(test_case.id), [])
        details = self.app.handle("GET", saved.headers["Location"])
        self.assertIn(b"No runs yet.", details.body)

        second_save = self.app.handle("POST", f"/test-cases/review/{token}/save", b"")
        self.assertEqual(second_save.status, 404)
        self.assertEqual(len(self.cases.list()), 1)

    def test_save_persists_edits_and_preserves_trusted_identity_and_order(self):
        self.provider.output = (
            '{"preconditions":[{"description":"A test email is available."}],'
            '"segments":[{"steps":[{"name":"Open registration",'
            '"description":"Open the registration page.",'
            '"expected":"The registration page is visible."}]},'
            '{"steps":[{"name":"Submit registration",'
            '"description":"Complete and submit the form.",'
            '"expected":"The account is created."}]}]}'
        )
        with redirect_stdout(io.StringIO()):
            generated = self.post_generate()
        token = generated.headers["Location"].rsplit("/", 1)[1]
        original = self.app._draft_store.get(token).test_case
        original_case_id = original.id
        original_segment_ids = [segment.id for segment in original.segments]
        original_step_ids = [[step.id for step in segment.steps] for segment in original.segments]
        original_precondition_ids = [item.id for item in original.preconditions]
        original_precondition_orders = [item.order for item in original.preconditions]
        original_base_url = original.base_url
        original_segment_base_urls = [segment.base_url for segment in original.segments]
        original_orders = [
            (segment.order, [step.order for step in segment.steps])
            for segment in original.segments
        ]

        saved = self.app.handle(
            "POST",
            f"/test-cases/review/{token}/save",
            urlencode({
                "name": "Registration with verified email",
                "description": "Register and confirm a new user account.",
                "precondition.0.description": "A unique email address is available.",
                "segment.0.step.0.name": "Open the sign-up page",
                "segment.0.step.0.description": "Open the public sign-up page.",
                "segment.0.step.0.expected": "The sign-up form is displayed.",
                "segment.1.step.0.name": "Submit the registration",
                "segment.1.step.0.description": "Submit the completed registration form.",
                "segment.1.step.0.expected": "A confirmation message is displayed.",
                # Unsupported identity and structure fields are ignored.
                "id": "attacker-controlled-id",
                "segment.0.id": "attacker-controlled-segment",
                "segment.0.step.0.id": "attacker-controlled-step",
                "segment.99.step.99.name": "Injected step",
            }),
        )

        self.assertEqual(saved.status, 303)
        case = self.cases.list()[0]
        self.assertEqual(case.id, original_case_id)
        self.assertEqual(case.public_id, "TC-0001")
        self.assertEqual(case.base_url, original_base_url)
        self.assertEqual([segment.base_url for segment in case.segments], original_segment_base_urls)
        self.assertEqual(case.name, "Registration with verified email")
        self.assertEqual(case.description, "Register and confirm a new user account.")
        self.assertEqual(case.preconditions[0].description, "A unique email address is available.")
        self.assertEqual([item.id for item in case.preconditions], original_precondition_ids)
        self.assertEqual([item.order for item in case.preconditions], original_precondition_orders)
        self.assertEqual([segment.id for segment in case.segments], original_segment_ids)
        self.assertEqual(
            [[step.id for step in segment.steps] for segment in case.segments],
            original_step_ids,
        )
        self.assertEqual(
            [(segment.order, [step.order for step in segment.steps]) for segment in case.segments],
            original_orders,
        )
        self.assertEqual(
            [
                (step.name, step.description, step.expected)
                for segment in case.segments for step in segment.steps
            ],
            [
                ("Open the sign-up page", "Open the public sign-up page.", "The sign-up form is displayed."),
                ("Submit the registration", "Submit the completed registration form.", "A confirmation message is displayed."),
            ],
        )
        self.assertEqual(self.history.list_for_test_case(case.id), [])
        for segment in case.segments:
            for step in segment.steps:
                self.assertIsNone(self.plans.find(step.id))

    def test_invalid_edits_keep_values_and_token_for_correction(self):
        with redirect_stdout(io.StringIO()):
            generated = self.post_generate()
        token = generated.headers["Location"].rsplit("/", 1)[1]
        response = self.app.handle(
            "POST",
            f"/test-cases/review/{token}/save",
            urlencode({
                "name": "",
                "description": "Edited scenario retained here.",
                "segment.0.step.0.description": "Edited action retained here.",
                "segment.0.step.0.expected": "",
            }),
        )
        self.assertEqual(response.status, 400)
        html = response.body.decode("utf-8")
        self.assertIn("TestCase name is required.", html)
        self.assertIn("Expected Result is required.", html)
        self.assertIn('value=""', html)
        self.assertIn("Edited scenario retained here.", html)
        self.assertIn("Edited action retained here.", html)
        self.assertEqual(self.app.handle("GET", f"/test-cases/review/{token}").status, 200)
        self.assertEqual(self.cases.list(), [])

        oversized_name = "N" * 201
        oversized = self.app.handle(
            "POST",
            f"/test-cases/review/{token}/save",
            urlencode({"name": oversized_name}),
        )
        self.assertEqual(oversized.status, 400)
        self.assertIn(b"TestCase name must be 200 characters or fewer.", oversized.body)
        self.assertIn(oversized_name.encode(), oversized.body)

        corrected = self.app.handle(
            "POST",
            f"/test-cases/review/{token}/save",
            urlencode({
                "name": "Corrected registration",
                "description": "Edited scenario retained here.",
                "segment.0.step.0.description": "Edited action retained here.",
                "segment.0.step.0.expected": "Registration completes successfully.",
            }),
        )
        self.assertEqual(corrected.status, 303)
        self.assertEqual(self.cases.list()[0].name, "Corrected registration")

    def test_user_supplied_html_is_escaped_after_validation_error(self):
        with redirect_stdout(io.StringIO()):
            generated = self.post_generate()
        token = generated.headers["Location"].rsplit("/", 1)[1]
        payload = '<img src=x onerror="alert(1)">'
        response = self.app.handle(
            "POST",
            f"/test-cases/review/{token}/save",
            urlencode({"name": payload, "description": ""}),
        )
        html = response.body.decode("utf-8")
        self.assertEqual(response.status, 400)
        self.assertIn("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;", html)
        self.assertNotIn(payload, html)

    def test_cancel_discards_draft_without_persisting_a_test_case(self):
        with redirect_stdout(io.StringIO()):
            generated = self.post_generate()
        token = generated.headers["Location"].rsplit("/", 1)[1]
        response = self.app.handle(
            "POST", f"/test-cases/review/{token}/cancel", b""
        )
        self.assertEqual(response.status, 303)
        self.assertEqual(self.cases.list(), [])
        self.assertEqual(self.app.handle("GET", f"/test-cases/review/{token}").status, 404)

    def test_invalid_draft_token_and_hidden_domain_tampering_are_rejected(self):
        tampered = self.app.handle(
            "POST",
            "/test-cases/review/not-a-valid-token/save",
            b"segments=%5B%5D&steps=%5B%5D",
        )
        self.assertEqual(tampered.status, 404)
        with redirect_stdout(io.StringIO()):
            generated = self.post_generate()
        token = generated.headers["Location"].rsplit("/", 1)[1]
        response = self.app.handle(
            "POST",
            f"/test-cases/review/{token}/save",
            b"id=attacker&segments=%5B%5D&steps=%5B%5D",
        )
        self.assertEqual(response.status, 303)
        self.assertEqual(len(self.cases.list()), 1)
        self.assertEqual(self.cases.list()[0].segments[0].steps[0].name, "Open registration")

    def test_ai_generated_html_is_escaped_in_review(self):
        self.provider.output = (
            '{"preconditions":[],"segments":[{"steps":[{"name":"<script>alert(1)</script>",'
            '"description":"<img src=x onerror=alert(1)>","expected":"</li><script>run()</script>"}]}]}'
        )
        with redirect_stdout(io.StringIO()):
            response = self.post_generate()
        token = response.headers["Location"].rsplit("/", 1)[1]
        review = self.app.handle("GET", f"/test-cases/review/{token}")
        html = review.body.decode("utf-8")
        self.assertNotIn('name="segment.0.step.0.name"', html)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", html)
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertNotIn("<img src=x", html)

    def test_provider_error_and_secret_are_not_rendered(self):
        secret = "FAKE_AUTHORING_API_SECRET_9921"
        app = LocalWebApplication(
            self.history,
            test_cases=self.cases,
            authoring_service=TestCaseAuthoringService(
                LLMRouter([AuthoringProvider(error=RetryableLLMError(secret))])
            ),
        )
        try:
            with redirect_stdout(io.StringIO()):
                response = app.handle(
                    "POST",
                    "/test-cases/generate",
                    urlencode({"name": "Case", "scenario": "Check page", "base_url": "https://example.test"}),
                )
                snapshot = self.wait_for_authoring(response, app=app)
            self.assertFalse(snapshot["success"])
            self.assertEqual(snapshot["state"], "FAILED")
            self.assertEqual(snapshot["error_category"], "AI_PROVIDER_ERROR")
            self.assertIn("could not complete", snapshot["error_message"])
            self.assertNotIn(secret, json.dumps(snapshot))
            page = app.handle("GET", response.headers["Location"])
            self.assertIn(b"Try Again", page.body)
            self.assertIn(b'data-authoring-state>Failed</span>', page.body)
            self.assertIn(b'data-authoring-provider>', page.body)
            script = app.handle("GET", "/assets/ui.js")
            self.assertIn(b"clearInterval(elapsedTimer)", script.body)
            self.assertNotIn(secret.encode(), page.body)
            self.assertEqual(app._draft_store._drafts, {})
        finally:
            app.close()

    def test_generate_again_failure_retry_uses_original_input_and_retires_old_draft_once(self):
        class FlakyProvider(AuthoringProvider):
            def create_structured_output(inner_self, prompt, schema, schema_name):
                inner_self.prompts.append((prompt, schema_name))
                if len(inner_self.prompts) == 2:
                    raise RetryableLLMError("provider failed with HTTP 503 and private detail")
                return inner_self.output

        provider = FlakyProvider()
        app = LocalWebApplication(
            self.history,
            test_cases=self.cases,
            authoring_service=TestCaseAuthoringService(LLMRouter([provider])),
        )
        try:
            with redirect_stdout(io.StringIO()):
                initial = app.handle(
                    "POST",
                    "/test-cases/generate",
                    urlencode({
                        "name": "Original authoring name",
                        "base_url": "https://example.test/register",
                        "scenario": "Original authoring scenario.",
                    }),
                )
                initial_progress = self.wait_for_authoring(initial, app=app)
            original_token = initial_progress["review_url"].rsplit("/", 1)[1]
            regenerate = app.handle(
                "POST", f"/test-cases/review/{original_token}/regenerate", b"", headers={"X-QA-CSRF": app._csrf_token}
            )
            failed = self.wait_for_authoring(regenerate, app=app)
            self.assertFalse(failed["success"])
            self.assertEqual(failed["error_category"], "AI_PROVIDER_ERROR")
            self.assertIsNotNone(app._draft_store.get(original_token))
            self.assertNotIn(original_token, json.dumps(failed))

            retry_id = regenerate.headers["Location"].rsplit("/", 1)[1]
            retry_page = app.handle("GET", regenerate.headers["Location"])
            self.assertIn(b"Try Again", retry_page.body)
            with redirect_stdout(io.StringIO()):
                retried = app.handle(
                    "POST", f"/test-cases/authoring-progress/{retry_id}/retry", b""
                )
                completed = self.wait_for_authoring(retried, app=app)
            self.assertTrue(completed["success"])
            self.assertNotEqual(retry_id, retried.headers["Location"].rsplit("/", 1)[1])
            self.assertEqual(len(provider.prompts), 3)
            for prompt, _schema_name in provider.prompts:
                self.assertIn('"name": "Original authoring name"', prompt)
                self.assertIn('"scenario": "Original authoring scenario."', prompt)
                self.assertNotIn("private detail", prompt)
            self.assertIsNone(app._draft_store.get(original_token))
            new_token = completed["review_url"].rsplit("/", 1)[1]
            self.assertNotEqual(original_token, new_token)
            self.assertEqual(len(app._draft_store._drafts), 1)
            self.assertEqual(self.cases.list(), [])
        finally:
            app.close()

    def test_progress_page_escapes_user_name_and_hides_url_query_secrets(self):
        secret = "PRIVATE_QUERY_TOKEN_4481"
        response = self.app.handle(
            "POST",
            "/test-cases/generate",
            urlencode({
                "name": '<script>alert("x")</script>',
                "base_url": f"https://example.test/register?api_token={secret}",
                "scenario": "Check registration.",
            }),
        )
        progress_id = response.headers["Location"].rsplit("/", 1)[1]
        with redirect_stdout(io.StringIO()):
            snapshot = self.wait_for_authoring(response)
        page = self.app.handle("GET", response.headers["Location"])
        html = page.body.decode("utf-8")
        self.assertEqual(snapshot["base_url"], "https://example.test/register")
        self.assertIn("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;", html)
        self.assertNotIn("<script>alert", html)
        self.assertNotIn(secret, html)
        self.assertNotIn("Check registration.", json.dumps(snapshot))
        self.assertEqual(self.app.handle(
            "GET", f"/api/test-cases/authoring-progress/{progress_id}"
        ).status, 200)

    def test_generation_again_replaces_ephemeral_draft_token(self):
        with redirect_stdout(io.StringIO()):
            generated = self.post_generate()
        old_token = generated.headers["Location"].rsplit("/", 1)[1]
        with redirect_stdout(io.StringIO()):
            regenerated = self.app.handle(
                "POST", f"/test-cases/review/{old_token}/regenerate", b""
            )
            completed = self.wait_for_authoring(regenerated)
        new_token = completed["review_url"].rsplit("/", 1)[1]
        self.assertNotEqual(old_token, new_token)
        self.assertEqual(self.app.handle("GET", f"/test-cases/review/{old_token}").status, 404)
        self.assertEqual(self.app.handle("GET", f"/test-cases/review/{new_token}").status, 200)
        self.assertEqual(self.cases.list(), [])

    def test_generate_again_preserves_edited_summary_and_uses_original_scenario(self):
        with redirect_stdout(io.StringIO()):
            generated = self.post_generate(name="Original name", scenario="Original scenario.")
        old_token = generated.headers["Location"].rsplit("/", 1)[1]
        with redirect_stdout(io.StringIO()):
            regenerated = self.app.handle(
                "POST",
                f"/test-cases/review/{old_token}/regenerate",
                urlencode({"name": "Unsaved manual edit", "description": "Unsaved scenario edit."}),
            )
            completed = self.wait_for_authoring(regenerated)
        self.assertTrue(completed["success"])
        self.assertEqual(len(self.provider.prompts), 2)
        second_prompt = self.provider.prompts[1][0]
        self.assertIn('"name": "Unsaved manual edit"', second_prompt)
        self.assertIn('"scenario": "Original scenario."', second_prompt)
        new_token = completed["review_url"].rsplit("/", 1)[1]
        self.assertEqual(self.app._draft_store.get(new_token).test_case.name, "Unsaved manual edit")
        self.assertNotIn("Unsaved scenario edit", second_prompt)

    def test_default_application_renders_authoring_without_configured_provider(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"LLM_PROVIDER_ORDER": "openai,gemini,openrouter,groq"},
            clear=True,
        ), patch("qa_agent.web.create_default_secret_store", return_value=UnavailableSecretStore()):
            app = create_application(Path(directory) / "ui.sqlite3")
            self.assertEqual(app.handle("GET", "/test-cases/new").status, 200)
            with redirect_stdout(io.StringIO()):
                response = app.handle(
                    "POST",
                    "/test-cases/generate",
                    urlencode({
                        "name": "Case",
                        "base_url": "http://127.0.0.1:8000/demo-target/registration",
                        "scenario": "Check the registration form.",
                    }),
                )
            try:
                self.assertEqual(response.status, 303)
                progress_id = response.headers["Location"].rsplit("/", 1)[1]
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    progress = app.handle(
                        "GET", f"/api/test-cases/authoring-progress/{progress_id}"
                    )
                    if progress.status == 200 and json.loads(progress.body)["finished"]:
                        payload = json.loads(progress.body)
                        break
                    time.sleep(0.005)
                else:
                    self.fail("Missing-provider authoring did not finish.")
                self.assertFalse(payload["success"])
                self.assertEqual(payload["error_category"], "AI_PROVIDER_ERROR")
                self.assertIn("could not complete", payload["error_message"])
                self.assertEqual(app._draft_store._drafts, {})
            finally:
                app.close()


if __name__ == "__main__":
    unittest.main()
