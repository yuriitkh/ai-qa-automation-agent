import unittest
from unittest.mock import Mock

from qa_agent.discovery_fallback import RouterDiscoveryFallback
from qa_agent.llm.router import LLMRouter
from qa_agent.models import (
    AIDiscoveryResult, DiscoveryResult, DiscoveryStatus, NavigationPath, TestStep as DomainTestStep,
)


class DiscoveryFallbackTests(unittest.TestCase):
    def setUp(self):
        self.step = DomainTestStep(name="Open account", description="Find account", expected="Account page", order=0)
        self.result = DiscoveryResult(status=DiscoveryStatus.PARTIAL, url="https://example.com",
                                      warnings=["Navigation path unavailable"], snapshot={"title": "Example"})
        self.structured = AIDiscoveryResult(navigation_paths=[NavigationPath(menu_item_text="Account")])

    def test_fallback_passes_task_step_snapshot_and_url_to_router(self):
        router = Mock()
        router.create_discovery.return_value = self.structured
        fallback = RouterDiscoveryFallback(router)
        output = fallback.discover("Check account navigation", "https://example.com", self.step, self.result)
        task, url, context = router.create_discovery.call_args.args
        self.assertEqual(task, "Check account navigation")
        self.assertEqual(url, "https://example.com")
        self.assertIn("Navigation path unavailable", context)
        self.assertEqual(output, self.structured)
        self.assertIsInstance(output, AIDiscoveryResult)
        self.assertFalse(any(hasattr(output, name) for name in ("execute", "click", "playwright_code")))

    def test_router_uses_provider_discovery_capability(self):
        provider = Mock()
        provider.is_available = True
        provider.create_discovery.return_value = self.structured
        provider.__class__.__name__ = "StructuredProvider"
        router = LLMRouter([provider])
        result = router.create_discovery("task", "https://example.com", "{}")
        self.assertEqual(result, self.structured)
        self.assertEqual(router.selected_provider_name, "StructuredProvider")

    def test_router_provider_failure_is_reported(self):
        provider = Mock()
        provider.is_available = True
        provider.create_discovery.side_effect = RuntimeError("provider failed")
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            LLMRouter([provider]).create_discovery("task", "https://example.com", "{}")


if __name__ == "__main__":
    unittest.main()
