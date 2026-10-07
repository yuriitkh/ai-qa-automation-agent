import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlencode

from qa_agent.llm.base import LLMProvider
from qa_agent.llm.errors import RetryableLLMError
from qa_agent.provider_settings import (
    ProviderSettingsRepository,
    ProviderSettingsService,
)
from qa_agent.redaction import safe_failure_reason
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
from qa_agent.web import LocalWebApplication


class MemorySecrets:
    def __init__(self):
        self.values = {}

    def get(self, name):
        return self.values.get(name)

    def set(self, name, value):
        self.values[name] = value

    def delete(self, name):
        self.values.pop(name, None)


class FakeProvider(LLMProvider):
    def __init__(self, name, key, calls, requests, outcome=None):
        self.name = name
        self.key = key
        self.calls = calls
        self.requests = requests
        self.outcome = outcome

    @property
    def is_available(self):
        return bool(self.key)

    def create_test_plan(self, task, target_url, page_snapshot):
        raise NotImplementedError

    def create_structured_output(self, prompt, schema, schema_name):
        self.calls.append(self.name)
        self.requests.append((prompt, schema, schema_name))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome or '{"ok":true}'


class StatusFailure(RuntimeError):
    def __init__(self, status):
        super().__init__("safe fake provider failure")
        self.status_code = status


class ProviderSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "settings.sqlite3"
        self.repository = ProviderSettingsRepository(self.database)
        self.secrets = MemorySecrets()
        self.calls = []
        self.requests = []
        self.timeouts = []
        self.environment = {"LLM_PROVIDER_ORDER": "openai,gemini,openrouter,groq"}

        def factory(name, key, model, timeout):
            self.timeouts.append(timeout)
            outcome = self.outcomes.get(name) if hasattr(self, "outcomes") else None
            return FakeProvider(name, key, self.calls, self.requests, outcome)

        self.factory = factory
        self.service = ProviderSettingsService(
            self.repository,
            self.secrets,
            environment=self.environment,
            provider_factory=self.factory,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_order_and_environment_only_configuration(self):
        self.environment.update({"GROQ_API_KEY": "environment-groq", "GEMINI_API_KEY": "environment-gemini"})
        views = self.service.provider_views()
        self.assertEqual([item.id for item in views], ["openai", "gemini", "openrouter", "groq"])
        self.assertEqual(views[1].status, "CONFIGURED")
        self.assertEqual(views[1].credential_source, "Environment variable")
        self.assertEqual(views[0].status, "NOT_CONFIGURED")
        self.assertEqual(views[3].masked_key, "••••••••••••groq")
        router = self.service.create_router()
        availability = {provider.name: provider.is_available for provider in router._providers}
        self.assertTrue(availability["gemini"])
        self.assertTrue(availability["groq"])
        self.assertFalse(availability["openai"])

    def test_web_key_precedence_replace_and_remove_falls_back_to_environment(self):
        self.environment["GROQ_API_KEY"] = "environment-value"
        self.service.save_key("groq", "web-first-secret")
        self.assertEqual(self.service.provider_views()[-1].credential_source, "Web settings")
        self.assertEqual(self.service.provider_views()[-1].masked_key, "••••••••••••cret")
        self.service.save_key("groq", "replacement-key")
        self.assertEqual(self.secrets.get("groq"), "replacement-key")
        self.service.remove_key("groq")
        view = next(item for item in self.service.provider_views() if item.id == "groq")
        self.assertEqual(view.credential_source, "Environment variable")
        self.assertEqual(view.masked_key, "••••••••••••alue")

    def test_enable_disable_and_move_changes_runtime_priority(self):
        self.service.save_key("groq", "groq-key")
        self.service.save_key("gemini", "gemini-key")
        router = self.service.create_router()
        self.assertEqual([item.name for item in router._providers], ["openai", "gemini", "openrouter", "groq"])
        self.service.move("groq", -1)
        self.service.update("gemini", enabled=False)
        self.service.refresh_router(router)
        self.assertEqual([item.name for item in router._providers], ["openai", "groq", "openrouter"])
        self.assertEqual(next(item for item in self.service.provider_views() if item.id == "gemini").status, "DISABLED")

    def test_router_uses_groq_first_then_falls_back_and_skips_disabled_or_missing(self):
        self.service.save_key("groq", "groq-key")
        self.service.save_key("gemini", "gemini-key")
        for _ in range(3):
            self.service.move("groq", -1)
        self.service.update("openai", enabled=False)
        self.service.update("openrouter", enabled=False)
        self.outcomes = {"groq": RetryableLLMError("HTTP 503")}
        router = self.service.create_router()
        result = router.create_structured_output("small request", {}, "test")
        self.assertEqual(result, '{"ok":true}')
        self.assertEqual(self.calls, ["groq", "gemini"])
        self.assertEqual(router.selected_provider_name, "gemini")

    def test_disabled_provider_is_never_called_and_unconfigured_provider_is_unavailable(self):
        self.service.save_key("groq", "groq-key")
        self.service.update("groq", enabled=False)
        router = self.service.create_router()
        self.assertNotIn("groq", [item.name for item in router._providers])
        self.assertFalse(any(item.is_available for item in router._providers))
        with self.assertRaisesRegex(RuntimeError, "No configured LLM providers"):
            router.create_structured_output("small request", {}, "test")
        self.assertEqual(self.calls, [])

    def test_settings_survive_repository_recreation_and_keys_are_not_in_sqlite(self):
        self.service.update("groq", enabled=False)
        for _ in range(2):
            self.service.move("openrouter", -1)
        self.service.save_key("groq", "never-store-this-key")
        recreated = ProviderSettingsService(
            ProviderSettingsRepository(self.database), self.secrets,
            environment=self.environment, provider_factory=self.factory,
        )
        views = recreated.provider_views()
        self.assertEqual(views[0].id, "openrouter")
        self.assertEqual(next(item for item in views if item.id == "groq").status, "DISABLED")
        raw_db = self.database.read_bytes()
        self.assertNotIn(b"never-store-this-key", raw_db)

    def test_settings_html_and_public_view_never_expose_raw_key(self):
        key = "secret-that-must-not-render-XYZ9"
        self.service.save_key("groq", key)
        router = self.service.create_router()
        app = LocalWebApplication(
            RunHistoryService(InMemoryRunHistoryRepository()),
            provider_settings=self.service,
            provider_router=router,
        )
        try:
            response = app.handle("GET", "/settings/providers")
        finally:
            app.close()
        html = response.body.decode("utf-8")
        safe_json = json.dumps(asdict(next(item for item in self.service.provider_views() if item.id == "groq")))
        self.assertEqual(response.status, 200)
        self.assertIn("Settings", html)
        self.assertIn("••••••••••••XYZ9", html)
        self.assertNotIn(key, html)
        self.assertNotIn(key, safe_json)
        self.assertNotIn(key, safe_failure_reason(RuntimeError(f"provider echoed {key}")))

    def test_provider_settings_post_uses_post_redirect_get_and_refreshes_router(self):
        router = self.service.create_router()
        app = LocalWebApplication(
            RunHistoryService(InMemoryRunHistoryRepository()),
            provider_settings=self.service,
            provider_router=router,
        )
        key = "posted-secret-Q2k8"
        try:
            response = app.handle("POST", "/settings/providers", urlencode({
                "provider_id": "groq", "operation": "save_key", "api_key": key,
            }))
            page = app.handle("GET", response.headers["Location"])
        finally:
            app.close()
        self.assertEqual(response.status, 303)
        self.assertNotIn(key, page.body.decode("utf-8"))
        self.assertEqual(next(item for item in router._providers if item.name == "groq").key, key)

    def test_connection_test_outcomes_use_fake_provider_and_do_not_store_response(self):
        self.environment["GROQ_API_KEY"] = "fake-test-key"
        self.outcomes = {
            "gemini": StatusFailure(401),
            "openai": StatusFailure(429),
            "openrouter": TimeoutError("fake timeout"),
            "groq": StatusFailure(503),
        }
        expected = {
            "groq": "provider_unavailable", "gemini": "authentication_failed",
            "openai": "rate_limited", "openrouter": "timeout",
        }
        for provider_id, status in expected.items():
            self.environment[{
                "groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY",
                "openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY",
            }[provider_id]] = "fake-key"
            result = self.service.test_connection(provider_id)
            self.assertEqual(result.status, status)
        self.outcomes["groq"] = None
        success = self.service.test_connection("groq")
        self.assertEqual(success.status, "connected")
        self.assertIsInstance(success.latency_ms, int)
        self.assertEqual(self.timeouts[-1], 8.0)
        self.assertEqual(self.requests[-1][0], 'Return only {"ok":true}.')
        self.assertEqual(self.requests[-1][2], "connection_test")
        self.assertNotIn("{\"ok\":true}", self.database.read_text(errors="ignore"))

    def test_connection_test_missing_key_is_configuration_invalid(self):
        result = self.service.test_connection("groq")
        self.assertEqual(result.status, "configuration_invalid")
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
