import os
import unittest
from unittest.mock import MagicMock, patch

from qa_agent.llm.openai_compatible import OpenAICompatibleProvider
from qa_agent.llm.registry import configured_provider_order, create_router
from qa_agent.llm.router import LLMRouter
from qa_agent.llm.base import LLMProvider
from qa_agent.llm.groq import GroqProvider
from qa_agent.models import QATestPlan


class ProviderRegistryTests(unittest.TestCase):
    def test_default_order_and_missing_openai_key(self):
        with patch.dict(os.environ, {}, clear=True):
            router = create_router()
        self.assertEqual(configured_provider_order(), ["openai", "gemini", "openrouter", "groq"])
        self.assertFalse(router._providers[0].is_available)

    def test_missing_openai_key_skips_cleanly_to_next_provider(self):
        plan = QATestPlan(url="https://example.com", steps=[{"action": "assert_page_loaded"}])

        class AvailableProvider(LLMProvider):
            def create_test_plan(self, task, target_url, page_snapshot):
                return plan

        with patch.dict(os.environ, {}, clear=True):
            router = LLMRouter([
                OpenAICompatibleProvider("openai", "OPENAI_API_KEY", "gpt-test"),
                AvailableProvider(),
            ])
            self.assertEqual(router.create_test_plan("task", plan.url, "{}"), plan)
        self.assertEqual(router.selected_provider_name, "AvailableProvider")

    def test_configured_order_selects_provider_without_agent_changes(self):
        with patch.dict(os.environ, {"LLM_PROVIDER_ORDER": "groq,openai", "GROQ_API_KEY": "g", "OPENAI_API_KEY": "o"}, clear=True):
            router = create_router()
            self.assertEqual(
                [type(p) for p in router._providers],
                [GroqProvider, OpenAICompatibleProvider],
            )

    def test_reordering_config_changes_selected_provider(self):
        plan = QATestPlan(url="https://example.com", steps=[{"action": "assert_page_loaded"}])
        settings = {"GROQ_API_KEY": "g", "OPENAI_API_KEY": "o"}
        selected = []
        for order in ("groq,openai", "openai,groq"):
            with patch.dict(os.environ, {**settings, "LLM_PROVIDER_ORDER": order}, clear=True):
                router = create_router()
                with patch.object(router._providers[0], "create_test_plan", return_value=plan):
                    router.create_test_plan("task", plan.url, "{}")
                selected.append(router.selected_provider_name)
        self.assertEqual(selected, ["GroqProvider", "openai"])

    def test_openrouter_configuration_is_registered_without_an_api_request(self):
        settings = {
            "LLM_PROVIDER_ORDER": "openrouter",
            "OPENROUTER_API_KEY": "openrouter-test-key",
            "OPENROUTER_MODEL": "vendor/model-test",
            "OPENROUTER_BASE_URL": "https://router.example/v1",
        }
        with patch.dict(os.environ, settings, clear=True):
            router = create_router()
            provider = router._providers[0]
            self.assertTrue(provider.is_available)
            self.assertEqual(provider.model, "vendor/model-test")
            self.assertEqual(provider.base_url, "https://router.example/v1")

    def test_configured_compatible_provider_needs_only_environment_configuration(self):
        values = {
            "LLM_COMPATIBLE_PROVIDERS": "example",
            "LLM_PROVIDER_ORDER": "example,groq",
            "EXAMPLE_API_KEY": "example-test",
            "EXAMPLE_BASE_URL": "https://api.example/v1",
            "EXAMPLE_MODEL": "example-chat",
        }
        with patch.dict(os.environ, values, clear=True):
            router = create_router()
            provider = router._providers[0]
            self.assertIsInstance(provider, OpenAICompatibleProvider)
            self.assertEqual(provider.base_url, "https://api.example/v1")
            self.assertEqual(provider.model, "example-chat")
            self.assertTrue(provider.is_available)
            plan = QATestPlan(url="https://example.com", steps=[{"action": "assert_page_loaded"}])
            response = MagicMock()
            response.choices[0].message.content = plan.model_dump_json()
            with patch("qa_agent.llm.openai_compatible.OpenAI") as client_cls:
                client_cls.return_value.chat.completions.create.return_value = response
                self.assertEqual(router.create_test_plan("task", plan.url, "{}"), plan)
            request = client_cls.return_value.chat.completions.create.call_args.kwargs
            self.assertEqual(request["max_tokens"], 4096)
            self.assertEqual(router.selected_provider_name, "example")

    def test_generic_provider_configuration_and_secret_safe_failure(self):
        secret = "unit-secret-token"
        plan = QATestPlan(url="https://example.com", steps=[{"action": "assert_page_loaded"}])
        fake_response = MagicMock()
        fake_response.choices[0].message.content = plan.model_dump_json()
        with patch.dict(os.environ, {"TEST_LLM_KEY": secret, "LLM_MAX_OUTPUT_TOKENS": "1536"}, clear=True), \
             patch("qa_agent.llm.openai_compatible.OpenAI") as client_cls:
            client_cls.return_value.chat.completions.create.return_value = fake_response
            provider = OpenAICompatibleProvider("test", "TEST_LLM_KEY", "test-model", "https://api.example/v1")
            self.assertEqual(provider.create_test_plan("task", plan.url, "{}"), plan)
            args = client_cls.call_args.kwargs
            self.assertEqual(args["base_url"], "https://api.example/v1")
            request = client_cls.return_value.chat.completions.create.call_args.kwargs
            self.assertEqual(request["model"], "test-model")
            self.assertEqual(request["max_tokens"], 1536)

