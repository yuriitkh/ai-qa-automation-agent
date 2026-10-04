import json
import unittest
from unittest.mock import patch

from qa_agent.browser_discovery import capture_discovery_result
from qa_agent.models import DiscoveryResult, DiscoveryStatus


class DiscoveryResultTests(unittest.TestCase):
    def test_success_result(self) -> None:
        result = DiscoveryResult(
            status=DiscoveryStatus.SUCCESS,
            url="https://www.dnb.no/",
            title="DNB",
            snapshot={"url": "https://www.dnb.no/", "title": "DNB"},
            navigation_paths=[self.dnb_path()],
            strategies_used=["menu_navigation"],
        )

        self.assertEqual(result.status, DiscoveryStatus.SUCCESS)
        self.assertEqual(result.title, "DNB")

    def test_partial_result(self) -> None:
        result = DiscoveryResult(
            status=DiscoveryStatus.PARTIAL,
            url="https://example.com",
            snapshot={"url": "https://example.com", "links": []},
            warnings=["No navigation paths were discovered."],
        )

        self.assertEqual(result.status, DiscoveryStatus.PARTIAL)
        self.assertEqual(result.warnings, ["No navigation paths were discovered."])

    def test_failed_result(self) -> None:
        result = DiscoveryResult(
            status=DiscoveryStatus.FAILED,
            url="https://example.com",
            warnings=["Browser launch failed"],
        )

        self.assertEqual(result.status, DiscoveryStatus.FAILED)
        self.assertEqual(result.navigation_paths, [])

    def test_navigation_collections_default_to_empty(self) -> None:
        result = DiscoveryResult(status=DiscoveryStatus.PARTIAL, url="https://example.com")

        self.assertEqual(result.navigation_paths, [])
        self.assertEqual(result.direct_navigation_paths, [])
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.strategies_used, [])

    def test_warnings_and_strategies_used_are_preserved(self) -> None:
        result = DiscoveryResult(
            status=DiscoveryStatus.PARTIAL,
            url="https://example.com",
            warnings=["No path found"],
            strategies_used=["menu_navigation", "direct_navigation"],
        )

        self.assertEqual(result.warnings, ["No path found"])
        self.assertEqual(result.strategies_used, ["menu_navigation", "direct_navigation"])

    def test_existing_navigation_path_structure_is_preserved(self) -> None:
        result = DiscoveryResult(
            status=DiscoveryStatus.SUCCESS,
            url="https://www.dnb.no/",
            navigation_paths=[self.dnb_path()],
        )

        path = result.navigation_paths[0].model_dump()
        self.assertEqual(
            set(path),
            {
                "menu_button_selector", "menu_tab_text", "menu_tab_selector",
                "menu_item_text", "menu_item_selector", "submenu_text",
                "submenu_selector", "expected_url", "heading_text", "heading_selector",
            },
        )
        self.assertEqual(path["submenu_selector"], 'main li a[href="/lan/boliglan"]')

    def test_existing_direct_navigation_path_structure_is_preserved(self) -> None:
        result = DiscoveryResult(
            status=DiscoveryStatus.SUCCESS,
            url="https://www.finn.no/",
            direct_navigation_paths=[self.direct_path()],
        )

        path = result.direct_navigation_paths[0].model_dump()
        self.assertEqual(set(path), {
            "strategy", "root_url", "steps", "expected_url", "heading_text", "heading_selector"
        })
        self.assertEqual(path["strategy"], "direct_nav")
        self.assertEqual(len(path["steps"]), 2)
        self.assertEqual(path["steps"][1]["resolved_url"], "https://www.finn.no/reise/restplasser/")

    def test_discovery_adapter_returns_success_for_discovered_paths(self) -> None:
        snapshot = {
            "url": "https://www.dnb.no/",
            "title": "DNB",
            "navigation_paths": [self.dnb_path()],
            "direct_navigation_paths": [],
        }
        with patch(
            "qa_agent.browser_discovery.capture_page_snapshot",
            return_value=json.dumps(snapshot),
        ):
            result = capture_discovery_result("https://www.dnb.no/")

        self.assertEqual(result.status, DiscoveryStatus.SUCCESS)
        self.assertEqual([path.submenu_text for path in result.navigation_paths], ["Boliglån"])
        self.assertEqual(result.strategies_used, ["menu_navigation"])

    def test_discovery_adapter_returns_failed_result_on_capture_error(self) -> None:
        with patch(
            "qa_agent.browser_discovery.capture_page_snapshot",
            side_effect=RuntimeError("browser unavailable"),
        ):
            result = capture_discovery_result("https://example.com")

        self.assertEqual(result.status, DiscoveryStatus.FAILED)
        self.assertIn("browser unavailable", result.warnings[0])

    @staticmethod
    def dnb_path() -> dict[str, str]:
        return {
            "menu_button_selector": "#menu-button-open",
            "menu_tab_text": "Privat",
            "menu_tab_selector": "#header-navigation-menu-tab-1",
            "menu_item_text": "Lån",
            "menu_item_selector": '#main-menu a[role="menuitem"][href="/lan"]',
            "submenu_text": "Boliglån",
            "submenu_selector": 'main li a[href="/lan/boliglan"]',
            "expected_url": "https://www.dnb.no/lan/boliglan",
            "heading_text": "Boliglån",
            "heading_selector": 'h1:has-text("Boliglån")',
        }

    @staticmethod
    def direct_path() -> dict[str, object]:
        return {
            "strategy": "direct_nav",
            "root_url": "https://www.finn.no/",
            "steps": [
                {
                    "text": "Reise",
                    "selector": 'a[href="/reise"]',
                    "href": "/reise",
                    "resolved_url": "https://www.finn.no/reise/",
                },
                {
                    "text": "Restplasser",
                    "selector": 'nav a[href="/reise/restplasser/"]',
                    "href": "/reise/restplasser/",
                    "resolved_url": "https://www.finn.no/reise/restplasser/",
                },
            ],
            "expected_url": "https://www.finn.no/reise/restplasser/",
            "heading_text": "Restplasser",
            "heading_selector": 'h1:has-text("Restplasser")',
        }


if __name__ == "__main__":
    unittest.main()
