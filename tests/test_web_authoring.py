from urllib.parse import urlencode
import os
import tempfile
import unittest
from contextlib import redirect_stdout
import io
from pathlib import Path
from unittest.mock import patch

from qa_agent.llm.base import LLMProvider
from qa_agent.llm.errors import RetryableLLMError
from qa_agent.llm.router import LLMRouter
from qa_agent.models import QATestPlan
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
from qa_agent.test_case_authoring import TestCaseAuthoringService
from qa_agent.test_case_repository import InMemoryTestCaseRepository
from qa_agent.web import LocalWebApplication, create_application


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

    def post_generate(self, *, name="User registration", scenario="Register a new user."):
        return self.app.handle(
            "POST",
            "/test-cases/generate",
            urlencode({
                "name": name,
                "base_url": "http://127.0.0.1:8000/demo-target/registration",
                "scenario": scenario,
            }),
        )

    def test_list_and_form_expose_ai_authoring(self):
        listing = self.app.handle("GET", "/test-cases")
        page = self.app.handle("GET", "/test-cases/new")
        self.assertIn(b"+ New Test Case", listing.body)
        self.assertIn(b'method="post" action="/test-cases/generate"', page.body)
        self.assertIn(b"Generate Test with AI", page.body)
        self.assertIn(b'name="scenario"', page.body)

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
        self.assertIn(b'name="name"', review.body)
        self.assertIn(b'value="User registration"', review.body)
        self.assertIn(b'name="description"', review.body)
        self.assertIn(b">Register a new user.</textarea>", review.body)
        self.assertIn(b'name="precondition.0.description"', review.body)
        self.assertIn(b">A test email is available.</textarea>", review.body)
        self.assertIn(b'name="segment.0.step.0.name"', review.body)
        self.assertIn(b'name="segment.0.step.0.description"', review.body)
        self.assertIn(b'name="segment.0.step.0.expected"', review.body)
        self.assertIn(b'value="Open registration"', review.body)
        self.assertIn(b">Open the registration page.</textarea>", review.body)
        self.assertIn(b"The registration page is visible.</textarea>", review.body)
        draft = self.app._draft_store.get(token)
        self.assertIsNotNone(draft)
        for internal_id in (
            draft.test_case.id,
            draft.test_case.segments[0].id,
            draft.test_case.segments[0].steps[0].id,
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
        self.assertIn("data-edited-indicator", html)
        self.assertIn(">Edited</span>", html)
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
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
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
        with redirect_stdout(io.StringIO()):
            response = app.handle(
                "POST",
                "/test-cases/generate",
                urlencode({"name": "Case", "scenario": "Check page", "base_url": "https://example.test"}),
            )
        self.assertEqual(response.status, 503)
        self.assertIn(b"temporarily unavailable", response.body)
        self.assertNotIn(secret.encode(), response.body)

    def test_generation_again_replaces_ephemeral_draft_token(self):
        with redirect_stdout(io.StringIO()):
            generated = self.post_generate()
        old_token = generated.headers["Location"].rsplit("/", 1)[1]
        regenerated = self.app.handle(
            "POST", f"/test-cases/review/{old_token}/regenerate", b""
        )
        self.assertEqual(regenerated.status, 303)
        new_token = regenerated.headers["Location"].rsplit("/", 1)[1]
        self.assertNotEqual(old_token, new_token)
        self.assertEqual(self.app.handle("GET", f"/test-cases/review/{old_token}").status, 404)
        self.assertEqual(self.app.handle("GET", f"/test-cases/review/{new_token}").status, 200)
        self.assertEqual(self.cases.list(), [])

    def test_generate_again_uses_original_authoring_input_not_manual_edits(self):
        with redirect_stdout(io.StringIO()):
            generated = self.post_generate(name="Original name", scenario="Original scenario.")
        old_token = generated.headers["Location"].rsplit("/", 1)[1]
        regenerated = self.app.handle(
            "POST",
            f"/test-cases/review/{old_token}/regenerate",
            urlencode({"name": "Unsaved manual edit", "description": "Unsaved scenario edit."}),
        )
        self.assertEqual(regenerated.status, 303)
        self.assertEqual(len(self.provider.prompts), 2)
        second_prompt = self.provider.prompts[1][0]
        self.assertIn('"name": "Original name"', second_prompt)
        self.assertIn('"scenario": "Original scenario."', second_prompt)
        self.assertNotIn("Unsaved manual edit", second_prompt)
        self.assertNotIn("Unsaved scenario edit", second_prompt)

    def test_default_application_renders_authoring_without_configured_provider(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"LLM_PROVIDER_ORDER": "openai,gemini,openrouter,groq"},
            clear=True,
        ):
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
            self.assertEqual(response.status, 503)
            self.assertIn(b"Configure an LLM provider", response.body)


if __name__ == "__main__":
    unittest.main()
