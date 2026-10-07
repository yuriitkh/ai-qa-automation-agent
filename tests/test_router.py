import importlib
import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from qa_agent.llm.base import LLMProvider
from qa_agent.llm.errors import NonRetryableLLMError, RetryableLLMError
from qa_agent.llm.gemini import GeminiProvider
from qa_agent.llm.groq import GroqProvider
from qa_agent.llm.openai_compatible import OpenAICompatibleProvider
from qa_agent.llm.registry import configured_provider_order, create_router
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
            "LLM ROUTER: provider request failed; trying the next provider.",
            output.getvalue(),
        )

    def test_gemini_429_falls_back_to_groq_for_test_plan(self) -> None:
        calls: list[str] = []
        gemini_client = MagicMock()
        gemini_error = RuntimeError("HTTP 429")
        gemini_error.status_code = 429

        def gemini_request(**kwargs):
            calls.append("Gemini")
            raise gemini_error

        def groq_request(*args, **kwargs):
            calls.append("Groq")
            return groq_response

        gemini_client.interactions.create.side_effect = gemini_request
        groq_response = MagicMock()
        groq_response.status_code = 200
        groq_response.is_error = False
        groq_response.json.return_value = {
            "choices": [{"message": {"content": self.plan.model_dump_json()}}]
        }

        with (
            patch.dict(
                os.environ,
                {"GEMINI_API_KEY": "gemini-test", "GROQ_API_KEY": "groq-test"},
                clear=True,
            ),
            patch("qa_agent.llm.gemini.genai.Client", return_value=gemini_client),
            patch("qa_agent.llm.groq.httpx.post", side_effect=groq_request),
            redirect_stdout(io.StringIO()),
        ):
            gemini = GeminiProvider()
            groq = GroqProvider()
            router = LLMRouter([gemini, groq])
            result = router.create_test_plan(self.task, self.target_url, self.snapshot)

        self.assertEqual(result, self.plan)
        self.assertEqual(calls, ["Gemini", "Groq"])
        self.assertEqual(router.selected_provider_name, "GroqProvider")
        gemini_client.interactions.create.assert_called_once()

    def test_gemini_400_falls_back_to_groq(self) -> None:
        gemini_client = MagicMock()
        gemini_error = RuntimeError("HTTP 400")
        gemini_error.status_code = 400
        gemini_client.interactions.create.side_effect = gemini_error

        groq_response = MagicMock()
        groq_response.status_code = 200
        groq_response.is_error = False
        groq_response.json.return_value = {
            "choices": [{"message": {"content": self.plan.model_dump_json()}}]
        }

        with (
            patch.dict(
                os.environ,
                {"GEMINI_API_KEY": "gemini-test", "GROQ_API_KEY": "groq-test"},
                clear=True,
            ),
            patch("qa_agent.llm.gemini.genai.Client", return_value=gemini_client),
            patch("qa_agent.llm.groq.httpx.post", return_value=groq_response),
            redirect_stdout(io.StringIO()),
        ):
            router = LLMRouter([GeminiProvider(), GroqProvider()])
            result = router.create_test_plan(self.task, self.target_url, self.snapshot)

        self.assertEqual(result, self.plan)
        self.assertEqual(router.selected_provider_name, "GroqProvider")
        gemini_client.interactions.create.assert_called_once()

    def test_groq_invalid_output_falls_back_to_openai_compatible(self) -> None:
        array_response = json.dumps([self.plan.model_dump()])
        groq_response = MagicMock()
        groq_response.status_code = 200
        groq_response.is_error = False
        groq_response.json.return_value = {
            "choices": [{"message": {"content": array_response}}]
        }

        openai_response = MagicMock()
        openai_response.choices[0].message.content = self.plan.model_dump_json()

        with (
            patch.dict(
                os.environ,
                {"GROQ_API_KEY": "groq-test", "OPENAI_API_KEY": "openai-test"},
                clear=True,
            ),
            patch("qa_agent.llm.groq.httpx.post", return_value=groq_response),
            patch("qa_agent.llm.openai_compatible.OpenAI") as client_cls,
            redirect_stdout(io.StringIO()),
        ):
            client_cls.return_value.chat.completions.create.return_value = openai_response
            router = LLMRouter(
                [
                    GroqProvider(),
                    OpenAICompatibleProvider(
                        "openai", "OPENAI_API_KEY", "gpt-test"
                    ),
                ]
            )
            result = router.create_test_plan(self.task, self.target_url, self.snapshot)

        self.assertEqual(result, self.plan)
        self.assertEqual(router.selected_provider_name, "openai")

    def test_gemini_discovery_429_falls_back_to_groq(self) -> None:
        calls: list[str] = []
        gemini_client = MagicMock()
        gemini_error = RuntimeError("HTTP 429")
        gemini_error.status_code = 429

        def gemini_request(**kwargs):
            calls.append("Gemini")
            raise gemini_error

        def groq_request(*args, **kwargs):
            calls.append("Groq")
            return groq_response

        gemini_client.interactions.create.side_effect = gemini_request
        groq_response = MagicMock()
        groq_response.status_code = 200
        groq_response.is_error = False
        groq_response.json.return_value = {
            "choices": [{"message": {"content": "{}"}}]
        }

        with (
            patch.dict(
                os.environ,
                {"GEMINI_API_KEY": "gemini-test", "GROQ_API_KEY": "groq-test"},
                clear=True,
            ),
            patch("qa_agent.llm.gemini.genai.Client", return_value=gemini_client),
            patch("qa_agent.llm.groq.httpx.post", side_effect=groq_request),
        ):
            router = LLMRouter([GeminiProvider(), GroqProvider()])
            result = router.create_discovery(self.task, self.target_url, self.snapshot)

        self.assertEqual(calls, ["Gemini", "Groq"])
        self.assertEqual(router.selected_provider_name, "GroqProvider")
        self.assertEqual(result.navigation_paths, [])

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

        self.assertIn("LLM ROUTER: provider request failed; trying the next provider.", output.getvalue())
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
            self.assertFalse(groq.is_available)
            router = LLMRouter([groq])
            with patch("qa_agent.llm.groq.httpx.post") as groq_post:
                with self.assertRaisesRegex(RuntimeError, "GroqProvider"):
                    router.create_test_plan(
                        self.task, self.target_url, self.snapshot
                    )

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

    def test_public_planner_rejects_selector_not_verified_by_discovery(self) -> None:
        task = (
            'Open https://example.com and verify that "Disabled input" is disabled.'
        )
        discovery_snapshot = json.dumps({
            "url": self.target_url,
            "title": "Example",
            "interactive_elements": [{
                "kind": "input",
                "selector": 'input[name="my-disabled"]',
                "accessible_name": "Disabled input",
                "tag": "input",
                "name": "my-disabled",
                "visible": True,
                "enabled": False,
            }],
        })
        invalid_plan = QATestPlan(
            url=self.target_url,
            steps=[{
                "action": "assert_disabled",
                "parameters": {"selector": "#my-disabled"},
            }],
        )
        gemini_client = importlib.import_module("qa_agent.gemini_client")

        with (
            patch.object(gemini_client, "extract_target_url", return_value=self.target_url),
            patch.object(gemini_client, "capture_page_snapshot", return_value=discovery_snapshot),
            patch.object(gemini_client._router, "create_test_plan", return_value=invalid_plan),
            self.assertRaisesRegex(ValueError, "deterministic Discovery selector"),
        ):
            gemini_client.create_test_plan(task)



class MultiProviderFallbackTests(unittest.TestCase):
    """Verify end-to-end multi-provider fallback across openai, gemini, openrouter, and groq."""

    def setUp(self) -> None:
        self.plan = QATestPlan(
            url="https://example.com",
            steps=[{"action": "assert_page_loaded"}],
        )
        self.task = "Verify checkout flow"
        self.target_url = "https://example.com/checkout"
        self.snapshot = '{"url":"https://example.com/checkout","title":"Checkout"}'
        self.keys = {
            "OPENAI_API_KEY": "fake-openai-key",
            "GEMINI_API_KEY": "fake-gemini-key",
            "OPENROUTER_API_KEY": "fake-openrouter-key",
            "GROQ_API_KEY": "fake-groq-key",
        }

    def test_default_configured_order_is_openai_gemini_openrouter_groq(self) -> None:
        with patch.dict(os.environ, self.keys, clear=True):
            order = configured_provider_order()
            self.assertEqual(order, ["openai", "gemini", "openrouter", "groq"])
            router = create_router()
            names = [router._provider_name(p) for p in router._providers]
            self.assertEqual(names, ["openai", "GeminiProvider", "openrouter", "GroqProvider"])
            self.assertTrue(all(p.is_available for p in router._providers))

    def test_scenario_a_openai_fails_gemini_succeeds(self) -> None:
        with patch.dict(os.environ, self.keys, clear=True):
            router = create_router()
            with (
                patch.object(
                    router._providers[0], "create_test_plan",
                    side_effect=RetryableLLMError("OpenAI rate limit 429"),
                ) as mock_openai,
                patch.object(
                    router._providers[1], "create_test_plan",
                    return_value=self.plan,
                ) as mock_gemini,
                patch.object(router._providers[2], "create_test_plan") as mock_openrouter,
                patch.object(router._providers[3], "create_test_plan") as mock_groq,
                redirect_stdout(io.StringIO()),
            ):
                result = router.create_test_plan(self.task, self.target_url, self.snapshot)

            self.assertEqual(result, self.plan)
            self.assertEqual(router.selected_provider_name, "GeminiProvider")
            mock_openai.assert_called_once_with(self.task, self.target_url, self.snapshot)
            mock_gemini.assert_called_once_with(self.task, self.target_url, self.snapshot)
            mock_openrouter.assert_not_called()
            mock_groq.assert_not_called()

    def test_scenario_b_openai_gemini_fail_openrouter_succeeds(self) -> None:
        with patch.dict(os.environ, self.keys, clear=True):
            router = create_router()
            with (
                patch.object(
                    router._providers[0], "create_test_plan",
                    side_effect=RetryableLLMError("OpenAI timeout"),
                ) as mock_openai,
                patch.object(
                    router._providers[1], "create_test_plan",
                    side_effect=RetryableLLMError("Gemini 503 Service Unavailable"),
                ) as mock_gemini,
                patch.object(
                    router._providers[2], "create_test_plan",
                    return_value=self.plan,
                ) as mock_openrouter,
                patch.object(router._providers[3], "create_test_plan") as mock_groq,
                redirect_stdout(io.StringIO()),
            ):
                result = router.create_test_plan(self.task, self.target_url, self.snapshot)

            self.assertEqual(result, self.plan)
            self.assertEqual(router.selected_provider_name, "openrouter")
            mock_openai.assert_called_once_with(self.task, self.target_url, self.snapshot)
            mock_gemini.assert_called_once_with(self.task, self.target_url, self.snapshot)
            mock_openrouter.assert_called_once_with(self.task, self.target_url, self.snapshot)
            mock_groq.assert_not_called()

    def test_scenario_c_openai_gemini_openrouter_fail_groq_succeeds(self) -> None:
        with patch.dict(os.environ, self.keys, clear=True):
            router = create_router()
            with (
                patch.object(
                    router._providers[0], "create_test_plan",
                    side_effect=RetryableLLMError("OpenAI 429 rate limit"),
                ) as mock_openai,
                patch.object(
                    router._providers[1], "create_test_plan",
                    side_effect=RetryableLLMError("Gemini 500 internal server error"),
                ) as mock_gemini,
                patch.object(
                    router._providers[2], "create_test_plan",
                    side_effect=RetryableLLMError("OpenRouter 402 payment required"),
                ) as mock_openrouter,
                patch.object(
                    router._providers[3], "create_test_plan",
                    return_value=self.plan,
                ) as mock_groq,
                redirect_stdout(io.StringIO()),
            ):
                result = router.create_test_plan(self.task, self.target_url, self.snapshot)

            self.assertEqual(result, self.plan)
            self.assertEqual(router.selected_provider_name, "GroqProvider")
            mock_openai.assert_called_once_with(self.task, self.target_url, self.snapshot)
            mock_gemini.assert_called_once_with(self.task, self.target_url, self.snapshot)
            mock_openrouter.assert_called_once_with(self.task, self.target_url, self.snapshot)
            mock_groq.assert_called_once_with(self.task, self.target_url, self.snapshot)

    def test_scenario_d_all_providers_fail_raises_useful_error(self) -> None:
        with patch.dict(os.environ, self.keys, clear=True):
            router = create_router()
            with (
                patch.object(
                    router._providers[0], "create_test_plan",
                    side_effect=RetryableLLMError(
                        "safe rate limit", category="RATE_LIMIT", http_status=429,
                        safe_detail="Rate limited",
                    ),
                ),
                patch.object(
                    router._providers[1], "create_test_plan",
                    side_effect=RetryableLLMError(
                        "safe unavailable", category="PROVIDER_UNAVAILABLE", http_status=503,
                        safe_detail="Provider unavailable",
                    ),
                ),
                patch.object(
                    router._providers[2], "create_test_plan",
                    side_effect=RetryableLLMError(
                        "safe timeout", category="TIMEOUT", http_status=504,
                        safe_detail="Request timed out",
                    ),
                ),
                patch.object(
                    router._providers[3], "create_test_plan",
                    side_effect=RetryableLLMError(
                        "safe unavailable", category="PROVIDER_UNAVAILABLE", http_status=500,
                        safe_detail="Provider unavailable",
                    ),
                ),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(RuntimeError) as context:
                    router.create_test_plan(self.task, self.target_url, self.snapshot)

            error_message = str(context.exception)
            self.assertIn("All LLM providers failed", error_message)
            self.assertIn("OpenAI [RATE_LIMIT] HTTP 429", error_message)
            self.assertIn("Gemini [PROVIDER_UNAVAILABLE] HTTP 503", error_message)
            self.assertIn("OpenRouter [TIMEOUT] HTTP 504", error_message)
            self.assertIn("Groq [PROVIDER_UNAVAILABLE] HTTP 500", error_message)
            self.assertIsNone(router.selected_provider_name)

    def test_scenario_e_custom_provider_order_is_followed(self) -> None:
        custom_order = "groq,openrouter,gemini,openai"
        with patch.dict(os.environ, {**self.keys, "LLM_PROVIDER_ORDER": custom_order}, clear=True):
            self.assertEqual(configured_provider_order(), ["groq", "openrouter", "gemini", "openai"])
            router = create_router()
            names = [router._provider_name(p) for p in router._providers]
            self.assertEqual(names, ["GroqProvider", "openrouter", "GeminiProvider", "openai"])

            with (
                patch.object(
                    router._providers[0], "create_test_plan",
                    side_effect=RetryableLLMError("Groq 429 rate limit"),
                ) as mock_groq,
                patch.object(
                    router._providers[1], "create_test_plan",
                    return_value=self.plan,
                ) as mock_openrouter,
                patch.object(router._providers[2], "create_test_plan") as mock_gemini,
                patch.object(router._providers[3], "create_test_plan") as mock_openai,
                redirect_stdout(io.StringIO()),
            ):
                result = router.create_test_plan(self.task, self.target_url, self.snapshot)

            self.assertEqual(result, self.plan)
            self.assertEqual(router.selected_provider_name, "openrouter")
            mock_groq.assert_called_once_with(self.task, self.target_url, self.snapshot)
            mock_openrouter.assert_called_once_with(self.task, self.target_url, self.snapshot)
            mock_gemini.assert_not_called()
            mock_openai.assert_not_called()

    def test_fallback_preserves_request_parameters_without_corruption(self) -> None:
        complex_task = "Verify 'Kjøp nå' button with special chars: €100 / <test> & \"quotes\""
        complex_url = "https://example.com/checkout?step=1&lang=no&curr=EUR#summary"
        complex_snapshot = json.dumps({
            "url": complex_url,
            "title": "Kjøp & Betal — €100",
            "interactive_elements": [{"id": "btn-1", "name": "Kjøp nå", "selector": "#buy"}],
        }, ensure_ascii=False)

        with patch.dict(os.environ, self.keys, clear=True):
            router = create_router()
            with (
                patch.object(
                    router._providers[0], "create_test_plan",
                    side_effect=RetryableLLMError("OpenAI transport error"),
                ) as mock_openai,
                patch.object(
                    router._providers[1], "create_test_plan",
                    side_effect=RetryableLLMError("Gemini transport error"),
                ) as mock_gemini,
                patch.object(
                    router._providers[2], "create_test_plan",
                    return_value=self.plan,
                ) as mock_openrouter,
                patch.object(router._providers[3], "create_test_plan") as mock_groq,
                redirect_stdout(io.StringIO()),
            ):
                result = router.create_test_plan(complex_task, complex_url, complex_snapshot)

            self.assertEqual(result, self.plan)
            self.assertEqual(router.selected_provider_name, "openrouter")
            # Verify exact arguments were forwarded without corruption
            for mock_call in (mock_openai, mock_gemini, mock_openrouter):
                call_args = mock_call.call_args.args
                self.assertEqual(call_args[0], complex_task)
                self.assertEqual(call_args[1], complex_url)
                self.assertEqual(call_args[2], complex_snapshot)
            mock_groq.assert_not_called()

    def test_selected_provider_is_exposed_and_logged(self) -> None:
        with patch.dict(os.environ, self.keys, clear=True):
            router = create_router()
            stdout_capture = io.StringIO()
            with (
                patch.object(
                    router._providers[0], "create_test_plan",
                    side_effect=RetryableLLMError("OpenAI 429", category="RATE_LIMIT", http_status=429),
                ),
                patch.object(
                    router._providers[1], "create_test_plan",
                    return_value=self.plan,
                ),
                redirect_stdout(stdout_capture),
            ):
                result = router.create_test_plan(self.task, self.target_url, self.snapshot)

            self.assertEqual(result, self.plan)
            self.assertEqual(router.selected_provider_name, "GeminiProvider")

            output = stdout_capture.getvalue()
            self.assertIn("LLM provider priority:", output)
            self.assertIn("1. openai [AVAILABLE]", output)
            self.assertIn("2. GeminiProvider [AVAILABLE]", output)
            self.assertIn("3. openrouter [AVAILABLE]", output)
            self.assertIn("4. GroqProvider [AVAILABLE]", output)
            self.assertIn("LLM ROUTER: provider request failed; trying the next provider.", output)
            self.assertIn("Selected provider: GeminiProvider", output)

    def test_non_retryable_error_halts_fallback_immediately(self) -> None:
        with patch.dict(os.environ, self.keys, clear=True):
            router = create_router()
            with (
                patch.object(
                    router._providers[0], "create_test_plan",
                    side_effect=NonRetryableLLMError("Invalid prompt / bad request"),
                ) as mock_openai,
                patch.object(router._providers[1], "create_test_plan") as mock_gemini,
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(NonRetryableLLMError):
                    router.create_test_plan(self.task, self.target_url, self.snapshot)

            mock_openai.assert_called_once()
            mock_gemini.assert_not_called()
            self.assertIsNone(router.selected_provider_name)


if __name__ == "__main__":
    unittest.main()
