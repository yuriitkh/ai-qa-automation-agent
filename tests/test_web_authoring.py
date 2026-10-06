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

    def create_test_plan(self, task, target_url, page_snapshot):
        return QATestPlan(url=target_url, steps=[{"action": "assert_page_loaded"}])

    def create_structured_output(self, prompt, schema, schema_name):
        if self.error:
            raise self.error
        return self.output


class TestCaseAuthoringWebTests(unittest.TestCase):
    def setUp(self):
        self.cases = InMemoryTestCaseRepository()
        self.history = RunHistoryService(InMemoryRunHistoryRepository())
        self.provider = AuthoringProvider()
        self.app = LocalWebApplication(
            self.history,
            test_cases=self.cases,
            authoring_service=TestCaseAuthoringService(LLMRouter([self.provider])),
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
        self.assertIn(b"Open registration", review.body)
        self.assertIn(b"Expected: The registration page is visible.", review.body)
        self.assertIn(b"A test email is available.", review.body)
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
