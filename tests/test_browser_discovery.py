import json
import unittest
from unittest.mock import MagicMock, patch

from playwright.sync_api import sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from qa_agent.browser_discovery import (
    MAX_SNAPSHOT_CHARS,
    MAX_NAVIGATION_PATHS,
    _NAVIGATION_MENU_SCRIPT,
    _DISCOVERY_SUBMENU_READY_SCRIPT,
    _SNAPSHOT_SCRIPT,
    _SUBMENU_CANDIDATES_SCRIPT,
    _DIRECT_NAVIGATION_LINKS_SCRIPT,
    _DIRECT_DESTINATION_HEADING_SCRIPT,
    _build_snapshot,
    capture_page_snapshot,
    extract_target_url,
    _discover_navigation_paths,
    _discover_direct_navigation_paths,
    _open_menu_with_single_retry,
)


class BrowserDiscoveryTests(unittest.TestCase):
    def _direct_page(
        self,
        *,
        root_links: list[dict[str, str]],
        page_links: dict[str, list[dict[str, str]]],
        click_urls: dict[str, str],
        headings: dict[str, dict[str, str]],
        hidden: set[str] | None = None,
        counts: dict[str, int] | None = None,
        click_failures: set[str] | None = None,
    ):
        class FakeLocator:
            def __init__(self, page: "FakePage", selector: str) -> None:
                self.page, self.selector = page, selector
                self.first = self

            def count(self) -> int:
                if self.selector == '[role="dialog"][aria-modal="true"]':
                    return self.page.counts.get(self.selector, 0)
                return self.page.counts.get(self.selector, 1)

            def is_visible(self) -> bool:
                return self.selector not in self.page.hidden

            def wait_for(self, **kwargs: object) -> None:
                if self.selector in self.page.hidden:
                    raise AssertionError("hidden locator")

            def click(self, **kwargs: object) -> None:
                self.page.click_count += 1
                if self.selector in self.page.click_failures:
                    raise PlaywrightTimeoutError("click target is intercepted")
                target = self.page.click_urls.get(self.selector)
                if target:
                    self.page.url = self.page.redirects.get(target, target)

        class NoopNavigation:
            def __enter__(self) -> None:
                return None

            def __exit__(self, *args: object) -> None:
                return None

        class FakePage:
            url = "https://example.test/"

            def __init__(self) -> None:
                self.root_links = root_links
                self.page_links = page_links
                self.click_urls = click_urls
                self.headings = headings
                self.hidden = hidden or set()
                self.counts = counts or {}
                self.click_failures = click_failures or set()
                self.click_count = 0
                self.redirects: dict[str, str] = {}

            def locator(self, selector: str) -> FakeLocator:
                return FakeLocator(self, selector)

            def goto(self, url: str) -> None:
                self.url = self.redirects.get(url, url)

            def wait_for_load_state(self, state: str, **kwargs: object) -> None:
                pass

            def expect_navigation(self, **kwargs: object) -> NoopNavigation:
                return NoopNavigation()

            def evaluate(self, script: str, argument: object = None) -> object:
                if script == _DIRECT_NAVIGATION_LINKS_SCRIPT:
                    scope = argument.get("scope") if isinstance(argument, dict) else None
                    if scope == "root":
                        return self.root_links
                    return self.page_links.get(self.url, [])
                if script == _DIRECT_DESTINATION_HEADING_SCRIPT:
                    return self.headings.get(self.url)
                raise AssertionError("Unexpected direct navigation script")

        return FakePage()

    def test_direct_navigation_link_script_finds_unique_nav_anchors(self) -> None:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.set_content(
                    '<base href="https://example.test/">'
                    '<style>nav a { display: inline-block; }</style>'
                    '<main><nav><a id="products-link" href="/products">Products</a>'
                    '<a href="/support">Support</a></nav></main>'
                )
                links = page.evaluate(
                    _DIRECT_NAVIGATION_LINKS_SCRIPT, {"scope": "root"}
                )
                by_text = {entry["text"]: entry for entry in links}
                self.assertEqual(by_text["Products"]["selector"], "#products-link")
                self.assertEqual(by_text["Products"]["href"], "/products")
                self.assertEqual(
                    by_text["Support"]["selector"], 'main nav a[href="/support"]'
                )
            finally:
                browser.close()

    def test_direct_navigation_path_records_verified_source_child_and_heading(self) -> None:
        products_url = "https://example.test/products"
        final_url = "https://example.test/products/software/"
        page = self._direct_page(
            root_links=[{"text": "Products", "selector": "#products-link",
                         "href": "/products", "resolved_url": products_url}],
            page_links={products_url: [{
                "text": "Software", "selector": 'main nav a[href="/products/software"]',
                "href": "/products/software", "resolved_url": "https://example.test/products/software",
            }]},
            click_urls={"#products-link": products_url,
                        'main nav a[href="/products/software"]': "https://example.test/products/software"},
            headings={final_url: {"text": "Software", "selector": 'h1:has-text("Software")'}},
        )
        page.redirects["https://example.test/products/software"] = final_url

        paths = _discover_direct_navigation_paths(page, "https://example.test/")

        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0], {
            "strategy": "direct_nav",
            "root_url": "https://example.test/",
            "steps": [
                {"text": "Products", "selector": "#products-link", "href": "/products",
                 "resolved_url": products_url},
                {"text": "Software", "selector": 'main nav a[href="/products/software"]',
                 "href": "/products/software", "resolved_url": "https://example.test/products/software"},
            ],
            "expected_url": final_url,
            "heading_text": "Software",
            "heading_selector": 'h1:has-text("Software")',
        })

    def test_direct_navigation_skips_ambiguous_child_selector_without_positionals(self) -> None:
        child_selector = 'main nav a[href="/products/software"]'
        page = self._direct_page(
            root_links=[{"text": "Products", "selector": "#products-link",
                         "href": "/products", "resolved_url": "https://example.test/products"}],
            page_links={"https://example.test/products": [{
                "text": "Software", "selector": child_selector,
                "href": "/products/software", "resolved_url": "https://example.test/products/software",
            }]},
            click_urls={"#products-link": "https://example.test/products",
                        child_selector: "https://example.test/products/software"},
            headings={"https://example.test/products/software": {
                "text": "Software", "selector": 'h1:has-text("Software")'}},
            counts={child_selector: 2},
        )

        self.assertEqual(_discover_direct_navigation_paths(page, "https://example.test/"), [])

    def test_direct_navigation_skips_hidden_root_link(self) -> None:
        selector = "#products-link"
        page = self._direct_page(
            root_links=[{"text": "Products", "selector": selector,
                         "href": "/products", "resolved_url": "https://example.test/products"}],
            page_links={}, click_urls={}, headings={}, hidden={selector},
        )

        self.assertEqual(_discover_direct_navigation_paths(page, "https://example.test/"), [])

    def test_direct_navigation_skips_when_modal_blocks_interaction(self) -> None:
        modal_selector = '[role="dialog"][aria-modal="true"]'
        page = self._direct_page(
            root_links=[{"text": "Products", "selector": "#products-link",
                         "href": "/products", "resolved_url": "https://example.test/products"}],
            page_links={}, click_urls={}, headings={}, counts={modal_selector: 1},
        )

        self.assertEqual(_discover_direct_navigation_paths(page, "https://example.test/"), [])
        self.assertEqual(page.click_count, 0)

    def test_direct_navigation_skips_child_when_normal_click_is_intercepted(self) -> None:
        child_selector = 'main nav a[href="/products/software"]'
        page = self._direct_page(
            root_links=[{"text": "Products", "selector": "#products-link",
                         "href": "/products", "resolved_url": "https://example.test/products"}],
            page_links={"https://example.test/products": [{
                "text": "Software", "selector": child_selector,
                "href": "/products/software", "resolved_url": "https://example.test/products/software",
            }]},
            click_urls={"#products-link": "https://example.test/products",
                        child_selector: "https://example.test/products/software"},
            headings={"https://example.test/products/software": {
                "text": "Software", "selector": 'h1:has-text("Software")'}},
            click_failures={child_selector},
        )

        self.assertEqual(_discover_direct_navigation_paths(page, "https://example.test/"), [])

    def test_direct_navigation_skips_path_when_destination_heading_is_missing(self) -> None:
        child_selector = 'main nav a[href="/products/software"]'
        page = self._direct_page(
            root_links=[{"text": "Products", "selector": "#products-link",
                         "href": "/products", "resolved_url": "https://example.test/products"}],
            page_links={"https://example.test/products": [{
                "text": "Software", "selector": child_selector,
                "href": "/products/software", "resolved_url": "https://example.test/products/software",
            }]},
            click_urls={"#products-link": "https://example.test/products",
                        child_selector: "https://example.test/products/software"},
            headings={},
        )

        self.assertEqual(_discover_direct_navigation_paths(page, "https://example.test/"), [])

    def test_direct_navigation_preserves_query_and_uses_final_redirect_url(self) -> None:
        child_selector = 'main nav a[href="/search?category=car&sort=price"]'
        final_url = "https://example.test/search?category=car&sort=price"
        page = self._direct_page(
            root_links=[{"text": "Products", "selector": "#products-link",
                         "href": "/products", "resolved_url": "https://example.test/products"}],
            page_links={"https://example.test/products": [{
                "text": "Cars", "selector": child_selector,
                "href": "/search?category=car&sort=price",
                "resolved_url": "https://example.test/search?category=car&sort=price",
            }]},
            click_urls={"#products-link": "https://example.test/products",
                        child_selector: "https://example.test/search?category=car&sort=price"},
            headings={final_url: {"text": "Cars", "selector": 'h1:has-text("Cars")'}},
        )

        paths = _discover_direct_navigation_paths(page, "https://example.test/")

        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0]["steps"][1]["href"], "/search?category=car&sort=price")
        self.assertEqual(paths[0]["expected_url"], final_url)

    def test_direct_navigation_deduplicates_destination_paths(self) -> None:
        final_url = "https://example.test/products/software"
        page = self._direct_page(
            root_links=[
                {"text": "Products", "selector": "#products-link", "href": "/products",
                 "resolved_url": "https://example.test/products"},
                {"text": "Products duplicate", "selector": "#products-link-copy", "href": "/products",
                 "resolved_url": "https://example.test/products"},
            ],
            page_links={"https://example.test/products": [{
                "text": "Software", "selector": 'main nav a[href="/products/software"]',
                "href": "/products/software", "resolved_url": final_url,
            }]},
            click_urls={"#products-link": "https://example.test/products",
                        "#products-link-copy": "https://example.test/products",
                        'main nav a[href="/products/software"]': final_url},
            headings={final_url: {"text": "Software", "selector": 'h1:has-text("Software")'}},
        )

        paths = _discover_direct_navigation_paths(page, "https://example.test/")

        self.assertEqual(len(paths), 1)

    def test_direct_navigation_snapshot_is_bounded_and_preserves_dnb_paths(self) -> None:
        dnb_path = {
            "menu_button_selector": "#menu-button-open",
            "menu_tab_text": "Privat",
            "menu_tab_selector": "#private-tab",
            "menu_item_text": "Lån",
            "menu_item_selector": '#main-menu a[href="/lan"]',
            "submenu_text": "Boliglån",
            "submenu_selector": 'main a[href="/lan/boliglan"]',
            "expected_url": "https://www.dnb.no/lan/boliglan",
            "heading_text": "Boliglån",
            "heading_selector": 'h1:has-text("Boliglån")',
        }
        direct_path = {
            "strategy": "direct_nav",
            "root_url": "https://www.finn.no/",
            "steps": [
                {"text": "Reise", "selector": 'a[href="/reise"]', "href": "/reise",
                 "resolved_url": "https://www.finn.no/reise/"},
                {"text": "Restplasser", "selector": 'nav a[href="/reise/restplasser/"]',
                 "href": "/reise/restplasser/", "resolved_url": "https://www.finn.no/reise/restplasser/"},
            ],
            "expected_url": "https://www.finn.no/reise/restplasser/",
            "heading_text": "Restplasser",
            "heading_selector": 'h1:has-text("Restplasser")',
        }

        snapshot = json.loads(_build_snapshot({
            "url": "https://www.finn.no/",
            "title": "FINN",
            "navigation_paths": [dnb_path],
            "direct_navigation_paths": [direct_path] * 20,
        }))

        self.assertLessEqual(len(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))), MAX_SNAPSHOT_CHARS)
        self.assertEqual(snapshot["navigation_paths"], [dnb_path])
        self.assertLessEqual(len(snapshot["direct_navigation_paths"]), 3)

    def _menu_open_fake(self, visibility_after_clicks: list[bool]):
        class FakeLocator:
            def __init__(self, page: "FakePage", selector: str) -> None:
                self.page, self.selector = page, selector

            def wait_for(self, **kwargs: object) -> None:
                pass

            def click(self, **kwargs: object) -> None:
                self.page.click_count += 1
                self.page.menu_visible = visibility_after_clicks[
                    min(self.page.click_count - 1, len(visibility_after_clicks) - 1)
                ]

            def is_visible(self) -> bool:
                return self.page.menu_visible

        class FakePage:
            def __init__(self) -> None:
                self.click_count = 0
                self.menu_visible = False

            def locator(self, selector: str) -> FakeLocator:
                return FakeLocator(self, selector)

        return FakePage()

    def test_menu_open_first_click_succeeds(self) -> None:
        page = self._menu_open_fake([True])
        self.assertTrue(_open_menu_with_single_retry(page, "#menu-button-open"))
        self.assertEqual(page.click_count, 1)

    def test_menu_open_retries_once_and_succeeds(self) -> None:
        page = self._menu_open_fake([False, True])
        self.assertTrue(_open_menu_with_single_retry(page, "#menu-button-open"))
        self.assertEqual(page.click_count, 2)

    def test_menu_open_stops_after_two_failed_clicks(self) -> None:
        page = self._menu_open_fake([False, False])
        self.assertFalse(_open_menu_with_single_retry(page, "#menu-button-open"))
        self.assertEqual(page.click_count, 2)

    def test_menu_open_does_not_click_again_when_already_visible(self) -> None:
        page = self._menu_open_fake([True, False])
        self.assertTrue(_open_menu_with_single_retry(page, "#menu-button-open"))
        self.assertEqual(page.click_count, 1)

    def test_discovery_skips_candidate_after_menu_retry_fails_and_continues(self) -> None:
        menu_items = [
            {"text": "First", "selector": "a.first", "href": "https://example.com/first",
             "tab_text": "Tab", "tab_selector": "#tab"},
            {"text": "Second", "selector": "a.second", "href": "https://example.com/second",
             "tab_text": "Tab", "tab_selector": "#tab"},
        ]

        class FakeLocator:
            def __init__(self, page: "FakePage", selector: str) -> None:
                self.page, self.selector = page, selector

            def count(self) -> int:
                return 0

            def wait_for(self, **kwargs: object) -> None:
                pass

            def click(self, **kwargs: object) -> None:
                pass

            def is_visible(self) -> bool:
                return True

        class NoNavigation:
            def __enter__(self) -> None:
                return None

            def __exit__(self, *args: object) -> None:
                raise PlaywrightTimeoutError("no navigation")

        class FakePage:
            url = "https://example.com/"

            def locator(self, selector: str) -> FakeLocator:
                return FakeLocator(self, selector)

            def goto(self, url: str) -> None:
                self.url = url

            def wait_for_load_state(self, state: str) -> None:
                pass

            def wait_for_function(self, expression: str, **kwargs: object) -> None:
                pass

            def expect_navigation(self, **kwargs: object) -> NoNavigation:
                return NoNavigation()

            def evaluate(self, script: str, argument: object = None) -> object:
                if script == _NAVIGATION_MENU_SCRIPT:
                    return {"items": menu_items}
                if script == _SUBMENU_CANDIDATES_SCRIPT:
                    return [{"text": "Child", "selector": "a.child",
                             "href": "https://example.com/second/child", "visible": True}]
                if script == _SNAPSHOT_SCRIPT:
                    return {"url": self.url, "headings": [
                        {"tag": "h1", "selector": "h1", "text": "Child"}
                    ]}
                raise AssertionError("Unexpected discovery script")

        page = FakePage()
        with patch(
            "qa_agent.browser_discovery._open_menu_with_single_retry",
            side_effect=[True, False, True],
        ) as open_menu:
            _, paths = _discover_navigation_paths(page, {
                "menu_button": {"selector": "#menu-button-open", "text": "Menu"},
                "tabs": [{"selector": "#tab", "panel_selector": "#panel", "text": "Tab"}],
            })

        self.assertEqual(open_menu.call_count, 3)
        self.assertEqual([path["menu_item_text"] for path in paths], ["Second"])

    def test_extracts_http_and_https_urls(self) -> None:
        self.assertEqual(
            extract_target_url("Check https://example.com/path?x=1."),
            "https://example.com/path?x=1",
        )
        self.assertEqual(
            extract_target_url("Visit http://localhost:8000/page"),
            "http://localhost:8000/page",
        )

    def test_requires_a_valid_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "must include"):
            extract_target_url("Check the home page")
        with self.assertRaisesRegex(ValueError, "invalid target URL"):
            extract_target_url("Open https://")

    def test_generates_snapshot_and_closes_browser(self) -> None:
        page = MagicMock()
        page.evaluate.return_value = {
            "url": "https://example.com/",
            "title": "Example Domain",
            "headings": [{"tag": "h1", "selector": "#heading", "id": "heading", "text": "Example Domain"}],
            "links": [{"tag": "a", "selector": "a#more", "text": "More", "href": "https://example.com/more"}],
            "buttons": [{"tag": "button", "selector": "button#continue", "text": "Continue", "role": "button"}],
            "visible_text_elements": [
                {"tag": "p", "selector": "body > p", "text": "Example Domain", "visible": True}
            ],
        }
        browser = MagicMock()
        browser.new_page.return_value = page
        playwright = MagicMock()
        playwright.chromium.launch.return_value = browser
        manager = MagicMock()
        manager.__enter__.return_value = playwright

        with patch("qa_agent.browser_discovery.sync_playwright", return_value=manager):
            snapshot = json.loads(capture_page_snapshot("https://example.com"))

        self.assertEqual(snapshot["url"], "https://example.com/")
        self.assertEqual(snapshot["title"], "Example Domain")
        self.assertEqual(snapshot["headings"][0]["selector"], "#heading")
        self.assertEqual(snapshot["links"][0]["href"], "https://example.com/more")
        self.assertEqual(snapshot["buttons"][0]["role"], "button")
        self.assertEqual(
            snapshot["visible_text_elements"],
            [{"tag": "p", "selector": "body > p", "text": "Example Domain"}],
        )
        playwright.chromium.launch.assert_called_once_with(headless=False)
        page.goto.assert_called_once_with("https://example.com")
        page.wait_for_load_state.assert_called_once_with("load")
        browser.close.assert_called_once_with()

    def test_closes_browser_if_snapshot_generation_fails(self) -> None:
        page = MagicMock()
        page.evaluate.side_effect = RuntimeError("evaluation failed")
        browser = MagicMock()
        browser.new_page.return_value = page
        playwright = MagicMock()
        playwright.chromium.launch.return_value = browser
        manager = MagicMock()
        manager.__enter__.return_value = playwright

        with patch("qa_agent.browser_discovery.sync_playwright", return_value=manager):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                capture_page_snapshot("https://example.com")

        browser.close.assert_called_once_with()

    def test_interactive_snapshot_keeps_identity_signals_without_control_values(self) -> None:
        serialized = _build_snapshot({
            "url": "https://example.com/",
            "interactive_elements": [{
                "kind": "input",
                "selector": "#email",
                "tag": "input",
                "role": "textbox",
                "accessible_name": "Email",
                "label": "Email",
                "placeholder": "name@example.test",
                "test_id": "registration-email",
                "text": "",
                "value": "private input",
                "visible": True,
                "enabled": True,
            }],
        })

        snapshot = json.loads(serialized)
        control = snapshot["interactive_elements"][0]

        self.assertEqual(control["label"], "Email")
        self.assertEqual(control["placeholder"], "name@example.test")
        self.assertEqual(control["test_id"], "registration-email")
        self.assertNotIn("value", control)
        self.assertNotIn("private input", serialized)
        self.assertNotIn("element.value", _SNAPSHOT_SCRIPT)
        self.assertNotIn("|| item.placeholder", _SNAPSHOT_SCRIPT)

    def test_snapshot_is_bounded_and_truncates_large_page_data(self) -> None:
        large_entry = {
            "tag": "a",
            "selector": "a" * 1000,
            "id": "i" * 1000,
            "name": "n" * 1000,
            "role": "link",
            "text": "t" * 1000,
            "href": "https://example.com/" + "p" * 1000,
        }
        serialized = _build_snapshot(
            {
                "url": "https://example.com/" + "u" * 2000,
                "title": "T" * 1000,
                "headings": [large_entry] * 1000,
                "links": [large_entry] * 1000,
                "buttons": [large_entry] * 1000,
                "visible_text_elements": [
                    {**large_entry, "visible": True}
                ] * 1000,
            }
        )

        snapshot = json.loads(serialized)
        self.assertLessEqual(len(serialized), MAX_SNAPSHOT_CHARS)
        self.assertTrue(snapshot.get("truncated"))
        self.assertLessEqual(len(snapshot["headings"]), 8)
        self.assertLessEqual(len(snapshot["links"]), 10)
        self.assertLessEqual(len(snapshot["buttons"]), 8)
        self.assertLessEqual(len(snapshot["visible_text_elements"]), 16)
        self.assertNotIn("<html", serialized.lower())

    def test_snapshot_serializes_unicode_characters_literally(self) -> None:
        serialized = _build_snapshot(
            {
                "url": "https://www.finn.no/",
                "title": "FINN",
                "headings": [
                    {
                        "tag": "h2",
                        "selector": 'h2:has-text("Populære annonser")',
                        "text": "Populære annonser",
                    }
                ],
            }
        )

        self.assertIn('h2:has-text(\\"Populære annonser\\")', serialized)
        self.assertNotIn("\\u00e6", serialized)

    def test_visible_text_elements_exclude_hidden_empty_and_disallowed_entries(self) -> None:
        serialized = _build_snapshot(
            {
                "url": "https://example.com/",
                "title": "Example Domain",
                "visible_text_elements": [
                    {"tag": "p", "selector": "p.visible", "text": "Example Domain", "visible": True},
                    {"tag": "p", "selector": "p.hidden", "text": "Hidden text", "visible": False},
                    {"tag": "p", "selector": "p.no-text", "text": "   ", "visible": True},
                    {"tag": "script", "selector": "script", "text": "secret code", "visible": True},
                    {"tag": "style", "selector": "style", "text": "rules", "visible": True},
                    {"tag": "meta", "selector": "meta", "text": "metadata", "visible": True},
                    {"tag": "link", "selector": "link", "text": "resource", "visible": True},
                ],
            }
        )

        snapshot = json.loads(serialized)
        self.assertEqual(
            snapshot["visible_text_elements"],
            [{"tag": "p", "selector": "p.visible", "text": "Example Domain"}],
        )

    def test_selenium_disabled_input_visible_text_selector_matches_its_element(self) -> None:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto("https://www.selenium.dev/selenium/web/web-form.html")
                snapshot = page.evaluate(_SNAPSHOT_SCRIPT)
                target = next(
                    item for item in snapshot["visible_text_elements"]
                    if item["text"] == "Disabled input"
                )
                control = next(
                    item for item in snapshot["interactive_elements"]
                    if item["accessible_name"] == "Disabled input"
                )

                matches = page.locator(target["selector"])
                self.assertGreaterEqual(matches.count(), 1)
                self.assertEqual(matches.count(), 1)
                self.assertEqual(matches.first.inner_text().strip(), "Disabled input")
                self.assertEqual(control["selector"], 'input[name="my-disabled"]')
                self.assertEqual(page.locator(control["selector"]).count(), 1)
                self.assertEqual(
                    page.locator(control["selector"]).get_attribute("name"),
                    "my-disabled",
                )
                text_input = next(
                    item for item in snapshot["interactive_elements"]
                    if item.get("id") == "my-text-id"
                )
                self.assertEqual(text_input["selector"], "#my-text-id")
                for item in snapshot["interactive_elements"]:
                    with self.subTest(control=item["accessible_name"]):
                        self.assertEqual(page.locator(item["selector"]).count(), 1)
            finally:
                browser.close()

    def test_heading_beyond_headings_limit_is_grounded_without_duplicate_headings(self) -> None:
        headings = [
            {"tag": "h2", "selector": f"#heading-{index}", "text": f"Heading {index}"}
            for index in range(8)
        ]
        target_heading = {
            "tag": "h2",
            "selector": "body > main > section:nth-of-type(9) > h2",
            "text": "Anbefalinger til deg",
            "visible": True,
        }
        snapshot = json.loads(
            _build_snapshot(
                {
                    "url": "https://www.finn.no/",
                    "title": "FINN",
                    "headings": headings,
                    "visible_text_elements": [target_heading],
                }
            )
        )

        self.assertEqual(len(snapshot["headings"]), 8)
        self.assertEqual(
            snapshot["visible_text_elements"],
            [{key: value for key, value in target_heading.items() if key != "visible"}],
        )
        self.assertIn("!visibleHeadings.includes(element)", _SNAPSHOT_SCRIPT)
        self.assertIn("visibleHeadings.includes(element)", _SNAPSHOT_SCRIPT)

    def test_navigation_metadata_is_preserved_and_bounded(self) -> None:
        navigation_paths = [
            {
                "menu_button_selector": "#menu-button-open",
                "menu_tab_text": "Privat",
                "menu_tab_selector": "#private-tab",
                "menu_item_text": f"Category {index}",
                "menu_item_selector": f'#main-menu a[href="/category-{index}"]',
                "submenu_text": f"Submenu {index}",
                "submenu_selector": f'main li a[href="/category-{index}/sub"]',
                "expected_url": f"https://www.dnb.no/category-{index}/sub",
                "heading_text": f"Heading {index}",
                "heading_selector": f'h1:has-text("Heading {index}")',
            }
            for index in range(5)
        ]
        serialized = _build_snapshot(
            {
                "url": "https://www.dnb.no/",
                "title": "DNB",
                "navigation_menu": {
                    "menu_button": {"text": "Meny", "selector": "#menu-button-open"},
                    "tabs": [
                        {
                            "text": "Privat",
                            "selector": "#private-tab",
                            "panel_selector": "#private-panel",
                        }
                    ],
                    "items": [
                        {
                            "text": "Boliglån",
                            "selector": '#main-menu a[href="/lan"]',
                            "href": "https://www.dnb.no/lan",
                            "tab_text": "Privat",
                            "tab_selector": "#private-tab",
                        }
                    ],
                },
                "navigation_paths": navigation_paths,
            }
        )

        snapshot = json.loads(serialized)
        self.assertLessEqual(len(serialized), MAX_SNAPSHOT_CHARS)
        self.assertEqual(snapshot["navigation_menu"]["items"][0]["text"], "Boliglån")
        self.assertEqual(len(snapshot["navigation_menu"]["tabs"]), 1)
        self.assertEqual(len(snapshot["navigation_paths"]), 3)

    def test_heading_without_id_gets_unique_text_selector(self) -> None:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.set_content(
                    "<h2>Markeder</h2><h2>Populære annonser</h2>"
                )
                snapshot = page.evaluate(_SNAPSHOT_SCRIPT)

                target = next(
                    heading
                    for heading in snapshot["headings"]
                    if heading["text"] == "Populære annonser"
                )
                self.assertEqual(
                    target["selector"], 'h2:has-text("Populære annonser")'
                )
                locator = page.locator(target["selector"])
                self.assertEqual(locator.count(), 1)
                self.assertEqual(locator.inner_text(), "Populære annonser")
            finally:
                browser.close()

    def test_discovery_uses_visible_submenu_in_replayed_ui_flow(self) -> None:
        menu_item_selector = '#main-menu a[role="menuitem"][href="/lan"]'
        visible_submenu_selector = 'main li a[href="/lan/billan"]'
        menu_items = [{
            "text": "Lån",
            "selector": menu_item_selector,
            "href": "https://www.dnb.no/lan",
            "tab_text": "Privat",
            "tab_selector": "#header-navigation-menu-tab-1",
        }]

        class FakeLocator:
            def __init__(self, page: "FakePage", selector: str) -> None:
                self.page, self.selector = page, selector

            def count(self) -> int:
                return 0 if self.selector == "#cookie-popup-strictlyNecessary" else 1

            def is_visible(self) -> bool:
                return self.selector != 'main li a[href="/lan/hidden"]'

            def click(self, **kwargs: object) -> None:
                if self.selector == menu_item_selector:
                    self.page.clicked_menu_item = self.selector

            def wait_for(self, **kwargs: object) -> None:
                if self.selector == 'main li a[href="/lan/hidden"]':
                    raise AssertionError("hidden submenu must not be used")

        class FakePage:
            url = "https://www.dnb.no/"

            def __init__(self) -> None:
                self.clicked_menu_item = None
                self.goto_urls: list[str] = []

            def locator(self, selector: str) -> FakeLocator:
                return FakeLocator(self, selector)

            def goto(self, url: str) -> None:
                self.goto_urls.append(url)
                self.url = url

            def wait_for_load_state(self, state: str) -> None:
                pass

            def wait_for_function(self, expression: str, **kwargs: object) -> None:
                pass

            def expect_navigation(self, **kwargs: object) -> object:
                class NoNavigation:
                    def __enter__(self) -> None:
                        return None

                    def __exit__(self, *args: object) -> None:
                        raise PlaywrightTimeoutError("no navigation")

                return NoNavigation()

            def evaluate(self, script: str, argument: object = None) -> object:
                if script == _NAVIGATION_MENU_SCRIPT:
                    return {"items": menu_items}
                if script == _SUBMENU_CANDIDATES_SCRIPT:
                    if not isinstance(argument, dict) or argument.get("scope") != "menu":
                        raise AssertionError("expanded menu should be searched after non-navigation click")
                    return [
                        {"text": "Hidden", "selector": 'main li a[href="/lan/hidden"]',
                         "href": "https://www.dnb.no/lan/hidden", "visible": False},
                        {"text": "Billån", "selector": visible_submenu_selector,
                         "href": "https://www.dnb.no/lan/billan", "visible": True},
                    ]
                if script == _SNAPSHOT_SCRIPT:
                    return {"url": self.url, "headings": [
                        {"tag": "h1", "selector": 'h1:has-text("Billån")', "text": "Billån"}
                    ]}
                raise AssertionError("Unexpected discovery script")

        page = FakePage()
        menu, paths = _discover_navigation_paths(page, {
            "menu_button": {"selector": "#menu-button-open", "text": "Meny"},
            "tabs": [{"selector": "#header-navigation-menu-tab-1",
                      "panel_selector": "#header-navigation-menu-content", "text": "Privat"}],
        })

        self.assertEqual(len(paths), 1)
        path = paths[0]
        self.assertEqual(path["menu_button_selector"], "#menu-button-open")
        self.assertEqual(path["menu_tab_selector"], "#header-navigation-menu-tab-1")
        self.assertEqual(path["menu_item_selector"], menu_item_selector)
        self.assertEqual(path["submenu_selector"], visible_submenu_selector)
        self.assertEqual(path["expected_url"], "https://www.dnb.no/lan/billan")
        self.assertEqual(path["heading_selector"], 'h1:has-text("Billån")')
        self.assertEqual(path["heading_text"], "Billån")
        self.assertEqual(menu["items"], menu_items)
        self.assertNotIn("https://www.dnb.no/lan", page.goto_urls)

    def test_discovery_waits_for_spa_navigation_before_inspecting_destination(self) -> None:
        menu_item_selector = '#main-menu a[role="menuitem"][href="/lan"]'
        submenu_selector = 'main li a[href="/lan/billan"]'
        menu_items = [{
            "text": "Lån",
            "selector": menu_item_selector,
            "href": "https://www.dnb.no/lan",
            "tab_text": "Privat",
            "tab_selector": "#header-navigation-menu-tab-1",
        }]

        class FakeLocator:
            def __init__(self, page: "FakePage", selector: str) -> None:
                self.page, self.selector = page, selector

            def count(self) -> int:
                return 0 if self.selector == "#cookie-popup-strictlyNecessary" else 1

            def is_visible(self) -> bool:
                if (
                    self.selector == "#header-navigation-menu-tab-1"
                    and self.page.url != "https://www.dnb.no/"
                ):
                    return False
                return True

            def click(self, **kwargs: object) -> None:
                if self.selector == menu_item_selector:
                    self.page.click_seen = True

            def wait_for(self, **kwargs: object) -> None:
                self.page.wait_calls.append((self.selector, self.page.url))
                if (
                    self.selector == "#header-navigation-menu-tab-1"
                    and self.page.url != "https://www.dnb.no/"
                ):
                    raise AssertionError("destination tab is hidden and must not be awaited")

        class FakePage:
            url = "https://www.dnb.no/"

            def __init__(self) -> None:
                self.goto_urls: list[str] = []
                self.click_seen = False
                self.destination_ready = False
                self.evaluated_destination = False
                self.readiness_args: list[object] = []
                self.wait_calls: list[tuple[str, str]] = []
                self.destination_tab_hidden = False

            def locator(self, selector: str) -> FakeLocator:
                return FakeLocator(self, selector)

            def goto(self, url: str) -> None:
                self.goto_urls.append(url)
                self.url = url

            def wait_for_load_state(self, state: str) -> None:
                pass

            def wait_for_function(
                self, expression: str, *, arg: object = None, timeout: int | None = None
            ) -> None:
                if expression == _DISCOVERY_SUBMENU_READY_SCRIPT:
                    if not self.click_seen or self.url != "https://www.dnb.no/lan":
                        raise AssertionError("SPA navigation must finish before DOM readiness wait")
                    if arg != {"parentUrl": "https://www.dnb.no/lan"}:
                        raise AssertionError("readiness wait should target the clicked menu URL")
                    self.readiness_args.append(arg)
                    self.destination_ready = True

            def expect_navigation(self, **kwargs: object) -> object:
                page = self

                class SpaNavigation:
                    def __enter__(self) -> None:
                        return None

                    def __exit__(self, *args: object) -> None:
                        if not page.click_seen:
                            raise AssertionError("menu item click did not run")
                        # Model DNB's asynchronous history navigation completing
                        # as Playwright exits the navigation observation context.
                        page.url = "https://www.dnb.no/lan"

                return SpaNavigation()

            def evaluate(self, script: str, argument: object = None) -> object:
                if script == _NAVIGATION_MENU_SCRIPT:
                    return {"items": menu_items}
                if script == _SUBMENU_CANDIDATES_SCRIPT:
                    if not self.destination_ready:
                        raise AssertionError("destination DOM was inspected before it was ready")
                    if argument != {"parentUrl": "https://www.dnb.no/lan", "scope": "main"}:
                        raise AssertionError("SPA destination should be searched in main")
                    self.destination_tab_hidden = not self.locator(
                        "#header-navigation-menu-tab-1"
                    ).is_visible()
                    self.evaluated_destination = True
                    return [{
                        "text": "Billån",
                        "selector": submenu_selector,
                        "href": "https://www.dnb.no/lan/billan",
                        "visible": True,
                    }]
                if script == _SNAPSHOT_SCRIPT:
                    return {"url": self.url, "headings": [
                        {"tag": "h1", "selector": "h1", "text": "Billån"}
                    ]}
                raise AssertionError("Unexpected discovery script")

        page = FakePage()
        _, paths = _discover_navigation_paths(page, {
            "menu_button": {"selector": "#menu-button-open", "text": "Meny"},
            "tabs": [{"selector": "#header-navigation-menu-tab-1",
                      "panel_selector": "#header-navigation-menu-content", "text": "Privat"}],
        })

        self.assertTrue(page.evaluated_destination)
        self.assertEqual(
            page.readiness_args,
            [{"parentUrl": "https://www.dnb.no/lan"}],
        )
        self.assertTrue(page.destination_tab_hidden)
        self.assertTrue(
            all(
                selector != "#header-navigation-menu-tab-1"
                or url == "https://www.dnb.no/"
                for selector, url in page.wait_calls
            )
        )
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0]["submenu_selector"], submenu_selector)
        self.assertEqual(paths[0]["expected_url"], "https://www.dnb.no/lan/billan")
        self.assertNotIn("https://www.dnb.no/lan", page.goto_urls)

if __name__ == "__main__":
    unittest.main()
