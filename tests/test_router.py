import importlib
import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from qa_agent.llm.base import LLMProvider
from qa_agent.llm.errors import RetryableLLMError
from qa_agent.llm.gemini import GeminiProvider
from qa_agent.llm.groq import GroqProvider
from qa_agent.llm.router import LLMRouter
from qa_agent.models import QATestPlan


class RecordingProvider(LLMProvider):
    def __init__(self, plan: QATestPlan) -> None:
        self.plan = plan
        self.calls: list[tuple[str, str, str]] = []

    def create_test_plan(
        self, task: str, target_url: str, page_snapshot: str
    ) -> QATestPlan:
        self.calls.append((task, target_url, page_snapshot))
        return self.plan


class RetryableProvider(LLMProvider):
    def __init__(self, reason: str = "temporary provider failure") -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.reason = reason

    def create_test_plan(
        self, task: str, target_url: str, page_snapshot: str
    ) -> QATestPlan:
        self.calls.append((task, target_url, page_snapshot))
        raise RetryableLLMError(self.reason)


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = QATestPlan(
            url="https://example.com",
            steps=[{"action": "assert_page_loaded"}],
        )
        self.task = "Check https://example.com"
        self.target_url = "https://example.com"
        self.snapshot = '{"url":"https://example.com","title":"Example Domain"}'

    def test_propagates_task_url_and_snapshot_to_provider(self) -> None:
        provider = RecordingProvider(self.plan)
        router = LLMRouter([provider])

        returned_plan = router.create_test_plan(
            self.task, self.target_url, self.snapshot
        )

        self.assertIs(returned_plan, self.plan)
        self.assertEqual(provider.calls, [(self.task, self.target_url, self.snapshot)])
        self.assertEqual(router.selected_provider_name, "RecordingProvider")

    def test_falls_back_after_retryable_failure_and_propagates_context(self) -> None:
        first = RetryableProvider()
        second = RecordingProvider(self.plan)
        router = LLMRouter([first, second])

        output = io.StringIO()
        with redirect_stdout(output):
            returned_plan = router.create_test_plan(
                self.task, self.target_url, self.snapshot
            )

        expected_context = (self.task, self.target_url, self.snapshot)
        self.assertIs(returned_plan, self.plan)
        self.assertEqual(first.calls, [expected_context])
        self.assertEqual(second.calls, [expected_context])
        self.assertEqual(router.selected_provider_name, "RecordingProvider")
        self.assertIn(
            "LLM ROUTER: RetryableProvider had a retryable failure "
            "(temporary provider failure); trying the next provider.",
            output.getvalue(),
        )

    def test_retryable_fallback_message_redacts_api_keys(self) -> None:
        secret = "unit-test-secret-key"
        first = RetryableProvider(f"HTTP 503 response included {secret}")
        second = RecordingProvider(self.plan)
        router = LLMRouter([first, second])

        with patch.dict(os.environ, {"GEMINI_API_KEY": secret}, clear=True):
            output = io.StringIO()
            with redirect_stdout(output):
                router.create_test_plan(
                    self.task, self.target_url, self.snapshot
                )

        self.assertIn("HTTP 503 response included [REDACTED]", output.getvalue())
        self.assertNotIn(secret, output.getvalue())

    def test_missing_gemini_key_uses_configured_groq_provider(self) -> None:
        response = MagicMock()
        response.status_code = 200
        response.is_error = False
        response.json.return_value = {
            "choices": [
                {"message": {"content": self.plan.model_dump_json()}}
            ]
        }

        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-credential"}, clear=True),
            patch("qa_agent.llm.groq.httpx.post", return_value=response) as groq_post,
        ):
            gemini = GeminiProvider()
            groq = GroqProvider()
            gemini_available = gemini.is_available
            groq_available = groq.is_available
            router = LLMRouter([gemini, groq])
            output = io.StringIO()
            with redirect_stdout(output):
                returned_plan = router.create_test_plan(
                    self.task, self.target_url, self.snapshot
                )

        self.assertFalse(gemini_available)
        self.assertTrue(groq_available)
        self.assertEqual(returned_plan, self.plan)
        self.assertEqual(router.selected_provider_name, "GroqProvider")
        groq_post.assert_called_once()
        self.assertIn(
            "LLM ROUTER: Skipping GeminiProvider; provider is not configured.",
            output.getvalue(),
        )

    def test_groq_without_key_is_unavailable_and_skipped(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            groq = GroqProvider()
            router = LLMRouter([groq])
            with patch("qa_agent.llm.groq.httpx.post") as groq_post:
                with self.assertRaisesRegex(RuntimeError, "GroqProvider"):
                    router.create_test_plan(
                        self.task, self.target_url, self.snapshot
                    )

        self.assertFalse(groq.is_available)
        groq_post.assert_not_called()

    def test_both_configured_providers_keep_gemini_first(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "GEMINI_API_KEY": "test-gemini-credential",
                    "GROQ_API_KEY": "test-groq-credential",
                },
                clear=True,
            ),
            patch("qa_agent.llm.gemini.genai.Client"),
        ):
            gemini = GeminiProvider()
            groq = GroqProvider()
            gemini_available = gemini.is_available
            groq_available = groq.is_available
            router = LLMRouter([gemini, groq])

            with (
                patch.object(gemini, "create_test_plan", return_value=self.plan) as first,
                patch.object(groq, "create_test_plan") as second,
            ):
                returned_plan = router.create_test_plan(
                    self.task, self.target_url, self.snapshot
                )

        self.assertTrue(gemini_available)
        self.assertTrue(groq_available)
        self.assertEqual([type(item) for item in router._providers], [GeminiProvider, GroqProvider])
        self.assertEqual(returned_plan, self.plan)
        self.assertEqual(router.selected_provider_name, "GeminiProvider")
        first.assert_called_once_with(self.task, self.target_url, self.snapshot)
        second.assert_not_called()

    def test_no_configured_provider_raises_clear_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            router = LLMRouter([GeminiProvider(), GroqProvider()])
            with self.assertRaisesRegex(
                RuntimeError,
                "No configured LLM providers are available.*GeminiProvider.*GroqProvider",
            ):
                router.create_test_plan(
                    self.task, self.target_url, self.snapshot
                )

    def test_public_planner_discovers_page_before_routing_context(self) -> None:
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "unit-test-key", "GROQ_API_KEY": "unit-test-key"},
        ):
            gemini_client = importlib.import_module("qa_agent.gemini_client")

        with (
            patch.object(gemini_client, "extract_target_url", return_value=self.target_url) as extract,
            patch.object(gemini_client, "capture_page_snapshot", return_value=self.snapshot) as discover,
            patch.object(gemini_client._router, "create_test_plan", return_value=self.plan) as route,
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                returned_plan = gemini_client.create_test_plan(self.task)

        self.assertIs(returned_plan, self.plan)
        extract.assert_called_once_with(self.task)
        discover.assert_called_once_with(self.target_url)
        route.assert_called_once_with(self.task, self.target_url, self.snapshot)
        self.assertEqual(output.getvalue(), f"PAGE SNAPSHOT\n{self.snapshot}\n")


if __name__ == "__main__":
    unittest.main()