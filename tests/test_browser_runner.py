import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from qa_agent.browser_runner import BrowserRunner, run_test_plan
from qa_agent.models import QATestPlan


class BrowserRunnerActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.page = MagicMock()
        self.page.url = "https://www.dnb.no/lan"
        self.locator = MagicMock()
        self.locator.count.return_value = 1
        self.page.locator.return_value = self.locator

        self.browser = MagicMock()
        self.context = MagicMock()
        self.context.new_page.return_value = self.page
        self.browser.new_context.return_value = self.context
        self.playwright = MagicMock()
        self.playwright.chromium.launch.return_value = self.browser
        self.manager = MagicMock()
        self.manager.__enter__.return_value = self.playwright

    def run_steps(self, *steps: dict[str, object]) -> dict[str, object]:
        plan = QATestPlan(url="https://example.com", steps=list(steps))
        with patch(
            "qa_agent.browser_runner.sync_playwright", return_value=self.manager
        ):
            return run_test_plan(plan)

    def test_click_invokes_locator_click(self) -> None:
        result = self.run_steps(
            {"action": "click", "parameters": {"selector": "#submit"}}
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["steps"][0]["status"], "passed")
        self.page.locator.assert_called_once_with("#submit")
        self.locator.wait_for.assert_called_once_with(
            state="visible", timeout=5000
        )
        self.locator.click.assert_called_once_with()
        self.page.wait_for_load_state.assert_called_once_with(
            "load", timeout=10000
        )

    def test_navigate_action_uses_its_url_without_automatic_plan_navigation(self) -> None:
        result = self.run_steps(
            {"action": "navigate", "parameters": {"url": "https://target.example/path"}}
        )

        self.assertEqual(result["status"], "passed")
        self.page.goto.assert_called_once_with("https://target.example/path")

    def test_browser_runner_calls_remain_isolated_per_invocation(self) -> None:
        second_page = MagicMock()
        first_context = MagicMock()
        first_context.new_page.return_value = self.page
        second_context = MagicMock()
        second_context.new_page.return_value = second_page
        first_browser = MagicMock()
        first_browser.new_context.return_value = first_context
        second_browser = MagicMock()
        second_browser.new_context.return_value = second_context
        first_playwright = MagicMock()
        first_playwright.chromium.launch.return_value = first_browser
        second_playwright = MagicMock()
        second_playwright.chromium.launch.return_value = second_browser
        managers = []
        for playwright in (first_playwright, second_playwright):
            manager = MagicMock()
            manager.__enter__.return_value = playwright
            managers.append(manager)

        runner = BrowserRunner()
        plan = QATestPlan(url="https://example.com", steps=[
            {"action": "assert_page_loaded", "parameters": {}}
        ])
        with patch(
            "qa_agent.browser_runner.sync_playwright", side_effect=managers
        ):
            runner(plan)
            runner(plan)

        self.assertIsNot(self.page, second_page)
        for manager, browser, context in zip(
            managers, (first_browser, second_browser), (first_context, second_context)
        ):
            browser.new_context.assert_called_once_with()
            context.new_page.assert_called_once_with()
            context.close.assert_called_once_with()
            browser.close.assert_called_once_with()
            manager.__exit__.assert_called_once()

    def test_open_session_reuses_resources_for_plans_and_pages(self) -> None:
        second_page = MagicMock()
        self.context.new_page.side_effect = [self.page, second_page]
        runner = BrowserRunner()
        plan = QATestPlan(url="https://example.com", steps=[
            {"action": "assert_page_loaded", "parameters": {}}
        ])

        with patch("qa_agent.browser_runner.sync_playwright", return_value=self.manager):
            with runner.open_session() as session:
                first_page = session.new_page()
                second_page_result = session.new_page()
                self.assertIs(first_page, self.page)
                self.assertIs(second_page_result, second_page)
                self.assertEqual(session.run_plan(first_page, plan)["status"], "passed")
                self.assertEqual(session.run_plan(first_page, plan)["status"], "passed")
                self.assertEqual(session.run_plan(second_page_result, plan)["status"], "passed")

        self.manager.__enter__.assert_called_once_with()
        self.playwright.chromium.launch.assert_called_once_with(headless=False)
        self.browser.new_context.assert_called_once_with()
        self.context.new_page.assert_has_calls([unittest.mock.call(), unittest.mock.call()])
        self.page.wait_for_load_state.assert_has_calls(
            [unittest.mock.call("load"), unittest.mock.call("load")]
        )
        self.context.close.assert_called_once_with()
        self.browser.close.assert_called_once_with()
        self.manager.__exit__.assert_called_once_with(None, None, None)
        self.page.close.assert_called_once_with()
        second_page.close.assert_called_once_with()

    def test_session_without_plan_cleans_resources(self) -> None:
        with patch("qa_agent.browser_runner.sync_playwright", return_value=self.manager):
            with BrowserRunner().open_session():
                pass

        self.playwright.chromium.launch.assert_called_once_with(headless=False)
        self.browser.new_context.assert_called_once_with()
        self.context.new_page.assert_not_called()
        self.context.close.assert_called_once_with()
        self.browser.close.assert_called_once_with()
        self.manager.__exit__.assert_called_once_with(None, None, None)

    def test_session_page_creation_exception_cleans_created_resources(self) -> None:
        self.context.new_page.side_effect = RuntimeError("page setup failed")
        with patch("qa_agent.browser_runner.sync_playwright", return_value=self.manager):
            with self.assertRaisesRegex(RuntimeError, "page setup failed"):
                with BrowserRunner().open_session() as session:
                    session.new_page()

        self.context.new_page.assert_called_once_with()
        self.context.close.assert_called_once_with()
        self.browser.close.assert_called_once_with()
        self.manager.__exit__.assert_called_once()

    def test_context_setup_exception_closes_browser_and_runtime(self) -> None:
        self.browser.new_context.side_effect = RuntimeError("context setup failed")
        with patch("qa_agent.browser_runner.sync_playwright", return_value=self.manager):
            with self.assertRaisesRegex(RuntimeError, "context setup failed"):
                with BrowserRunner().open_session():
                    self.fail("session should not be yielded")

        self.browser.close.assert_called_once_with()
        self.manager.__exit__.assert_called_once()

    def test_screenshot_precedes_page_context_browser_and_runtime_cleanup(self) -> None:
        events: list[str] = []
        self.page.title.return_value = "Wrong"
        self.page.screenshot.side_effect = lambda **_: events.append("screenshot")
        self.page.close.side_effect = lambda: events.append("page.close")
        self.context.close.side_effect = lambda: events.append("context.close")
        self.browser.close.side_effect = lambda: events.append("browser.close")
        self.manager.__exit__.side_effect = lambda *args: events.append("playwright.exit")
        plan = QATestPlan(url="https://example.com", steps=[{
            "action": "assert_title", "parameters": {"expected": "Expected"}
        }])

        with tempfile.TemporaryDirectory() as directory:
            with patch("qa_agent.browser_runner.sync_playwright", return_value=self.manager):
                result = BrowserRunner(directory)(plan)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(events, [
            "screenshot", "page.close", "context.close", "browser.close", "playwright.exit"
        ])

    def test_fill_invokes_locator_fill_with_value(self) -> None:
        result = self.run_steps(
            {
                "action": "fill",
                "parameters": {"selector": "#name", "value": "Ada"},
            }
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["steps"][0]["status"], "passed")
        self.page.locator.assert_called_once_with("#name")
        self.locator.fill.assert_called_once_with("Ada")

    def test_assert_url_passes_for_exact_match(self) -> None:
        result = self.run_steps(
            {
                "action": "assert_url",
                "parameters": {"expected": "https://www.dnb.no/lan"},
            }
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["steps"][0]["status"], "passed")

    def test_assert_url_fails_for_different_url(self) -> None:
        result = self.run_steps(
            {
                "action": "assert_url",
                "parameters": {"expected": "https://www.dnb.no/"},
            }
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("https://www.dnb.no/", result["steps"][0]["error"])
        self.assertIn("https://www.dnb.no/lan", result["steps"][0]["error"])

    def test_assert_url_uses_exact_not_substring_comparison(self) -> None:
        self.page.url = "https://www.dnb.no/lan/extra"

        result = self.run_steps(
            {
                "action": "assert_url",
                "parameters": {"expected": "https://www.dnb.no/lan"},
            }
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("https://www.dnb.no/lan/extra", result["steps"][0]["error"])

    def test_assert_text_contains_passes_for_substring(self) -> None:
        self.locator.inner_text.return_value = "Documentation examples are allowed without needing permission today."
        result = self.run_steps({"action": "assert_text_contains", "parameters": {"expected_text": "without needing permission"}})
        self.assertEqual(result["status"], "passed")

    def test_assert_text_contains_fails_without_substring(self) -> None:
        self.locator.inner_text.return_value = "Different page content"
        result = self.run_steps({"action": "assert_text_contains", "parameters": {"expected_text": "missing phrase"}})
        self.assertEqual(result["status"], "failed")
        self.assertIn("to be contained", result["steps"][0]["error"])

    def test_checked_assertion_handles_checkbox_and_radio(self) -> None:
        for selector in ("#checkbox", "#radio"):
            with self.subTest(selector=selector):
                self.locator.is_checked.return_value = True
                result = self.run_steps(
                    {"action": "click", "parameters": {"selector": selector}},
                    {"action": "assert_checked", "parameters": {"selector": selector}},
                )
                self.assertEqual(result["status"], "passed")
                self.locator.is_checked.assert_called()

    def test_select_option_and_assert_selected(self) -> None:
        self.locator.evaluate.return_value = {"tag": "select", "type": "select-one"}
        self.locator.locator.return_value = self.locator
        self.locator.first = self.locator
        self.locator.inner_text.return_value = "Two"
        self.locator.get_attribute.return_value = "2"
        result = self.run_steps(
            {"action": "select_option", "parameters": {"selector": "select", "option_label": "Two"}},
            {"action": "assert_selected", "parameters": {"selector": "select", "expected": "Two"}},
        )
        self.assertEqual(result["status"], "passed")
        self.locator.select_option.assert_called_once_with(label="Two")

    def test_assert_selected_checks_radio_state(self) -> None:
        self.locator.evaluate.return_value = {"tag": "input", "type": "radio"}
        self.locator.is_checked.return_value = True

        result = self.run_steps(
            {"action": "assert_selected", "parameters": {"selector": "#default-radio"}}
        )

        self.assertEqual(result["status"], "passed")
        self.locator.is_checked.assert_called_once_with()

    def test_enabled_and_disabled_assertions(self) -> None:
        for action, actual, expected_status in (("assert_disabled", False, "passed"), ("assert_enabled", True, "passed"), ("assert_disabled", True, "failed")):
            with self.subTest(action=action, actual=actual):
                self.locator.is_enabled.return_value = actual
                result = self.run_steps({"action": action, "parameters": {"selector": "#input"}})
                self.assertEqual(result["status"], expected_status)

    def test_assert_visible_matches_literal_escaped_newline(self) -> None:
        self.locator.first.inner_text.return_value = "Godta alle\nlukk popup"

        result = self.run_steps(
            {
                "action": "assert_visible",
                "parameters": {
                    "selector": "#cookie-popup-acceptAll",
                    "expected_text": "Godta alle\\nlukk popup",
                },
            }
        )

        self.assertEqual(result["status"], "passed")

    def test_assert_visible_keeps_exact_text_comparison(self) -> None:
        self.locator.first.inner_text.return_value = "Godta alle\nlukk popup"

        result = self.run_steps(
            {
                "action": "assert_visible",
                "parameters": {
                    "selector": "#cookie-popup-acceptAll",
                    "expected_text": "Godta alle\nlukk popup",
                },
            }
        )

        self.assertEqual(result["status"], "passed")

    def test_assert_visible_still_fails_for_incorrect_text(self) -> None:
        self.locator.first.inner_text.return_value = "Godta alle\nlukk popup"

        result = self.run_steps(
            {
                "action": "assert_visible",
                "parameters": {
                    "selector": "#cookie-popup-acceptAll",
                    "expected_text": "Godta alle lukk popup",
                },
            }
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("exact visible text", result["steps"][0]["error"])

    def test_assert_hidden_passes_for_hidden_element(self) -> None:
        self.locator.is_hidden.return_value = True

        result = self.run_steps(
            {"action": "assert_hidden", "parameters": {"selector": "#notice"}}
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["steps"][0]["status"], "passed")
        self.locator.is_hidden.assert_called_once_with()

    def test_assert_hidden_passes_when_element_is_removed_from_dom(self) -> None:
        self.locator.count.return_value = 0
        self.locator.is_hidden.return_value = True

        result = self.run_steps(
            {"action": "assert_hidden", "parameters": {"selector": "#notice"}}
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["steps"][0]["status"], "passed")
        self.locator.is_hidden.assert_called_once_with()
        self.locator.count.assert_not_called()

    def test_assert_hidden_fails_for_visible_element(self) -> None:
        self.locator.is_hidden.return_value = False

        result = self.run_steps(
            {"action": "assert_hidden", "parameters": {"selector": "#notice"}}
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("is visible", result["steps"][0]["error"])

    def test_missing_selectors_fail_clearly_for_click_and_fill(self) -> None:
        self.locator.count.return_value = 0
        steps = (
            {"action": "click", "parameters": {"selector": "#missing"}},
            {
                "action": "fill",
                "parameters": {"selector": "#missing", "value": "text"},
            },
        )

        for step in steps:
            with self.subTest(action=step["action"]):
                if step["action"] == "click":
                    self.locator.wait_for.side_effect = RuntimeError(
                        "selector did not become visible"
                    )
                result = self.run_steps(step)
                error = result["steps"][0]["error"]
                self.assertEqual(result["status"], "failed")
                self.assertIn("#missing", error)
                if step["action"] == "click":
                    self.assertIn("Could not click", error)
                    self.locator.click.assert_not_called()
                    self.locator.wait_for.side_effect = None
                else:
                    self.assertIn("not found", error)

    def test_failed_new_action_stops_following_steps(self) -> None:
        self.locator.wait_for.side_effect = RuntimeError(
            "selector did not become visible"
        )

        result = self.run_steps(
            {"action": "click", "parameters": {"selector": "#missing"}},
            {"action": "assert_page_loaded", "parameters": {}},
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["steps"]), 1)
        self.page.wait_for_load_state.assert_not_called()

    def test_failure_captures_screenshot_in_configured_directory(self) -> None:
        self.page.title.return_value = "Wrong title"
        with tempfile.TemporaryDirectory() as directory:
            def save_screenshot(*, path: str) -> None:
                Path(path).write_bytes(b"png")

            self.page.screenshot.side_effect = save_screenshot
            plan = QATestPlan(url="https://example.com", steps=[{
                "action": "assert_title", "parameters": {"expected": "Expected"}
            }])
            with patch("qa_agent.browser_runner.sync_playwright", return_value=self.manager):
                result = run_test_plan(plan, evidence_directory=directory)

            self.assertEqual(result["status"], "failed")
            self.assertEqual(len(result["evidence"]), 1)
            artifact = Path(result["evidence"][0]["path"])
            self.assertEqual(artifact.parent, Path(directory))
            self.assertTrue(artifact.is_file())
            self.assertTrue(artifact.name.startswith("execution-"))

    def test_assert_visible_failure_captures_screenshot(self) -> None:
        self.locator.first.inner_text.return_value = "Actual"
        with tempfile.TemporaryDirectory() as directory:
            self.page.screenshot.side_effect = lambda *, path: Path(path).write_bytes(b"png")
            plan = QATestPlan(url="https://example.com", steps=[{
                "action": "assert_visible",
                "parameters": {"selector": "h1", "expected_text": "Expected"},
            }])
            with patch("qa_agent.browser_runner.sync_playwright", return_value=self.manager):
                result = run_test_plan(plan, evidence_directory=directory)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len(result["evidence"]), 1)

    def test_locator_failure_captures_screenshot_when_page_is_available(self) -> None:
        self.locator.wait_for.side_effect = RuntimeError("selector timeout")
        with tempfile.TemporaryDirectory() as directory:
            self.page.screenshot.side_effect = lambda *, path: Path(path).write_bytes(b"png")
            plan = QATestPlan(url="https://example.com", steps=[{
                "action": "click", "parameters": {"selector": "#missing"}
            }])
            with patch("qa_agent.browser_runner.sync_playwright", return_value=self.manager):
                result = run_test_plan(plan, evidence_directory=directory)
            self.assertEqual(result["status"], "failed")
            self.assertIn("Could not click", result["steps"][0]["error"])
            self.assertEqual(len(result["evidence"]), 1)

    def test_screenshot_failure_preserves_original_failure(self) -> None:
        self.page.title.return_value = "Wrong title"
        self.page.screenshot.side_effect = RuntimeError("capture failed")
        with tempfile.TemporaryDirectory() as directory:
            plan = QATestPlan(url="https://example.com", steps=[{
                "action": "assert_title", "parameters": {"expected": "Expected"}
            }])
            with patch("qa_agent.browser_runner.sync_playwright", return_value=self.manager):
                result = run_test_plan(plan, evidence_directory=directory)
        self.assertEqual(result["status"], "failed")
        self.assertIn("Expected title", result["steps"][0]["error"])
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["evidence_capture_error"], "capture failed")

    def test_passed_execution_does_not_capture_screenshot(self) -> None:
        self.page.title.return_value = "Expected"
        with tempfile.TemporaryDirectory() as directory:
            plan = QATestPlan(url="https://example.com", steps=[{
                "action": "assert_title", "parameters": {"expected": "Expected"}
            }])
            with patch("qa_agent.browser_runner.sync_playwright", return_value=self.manager):
                result = run_test_plan(plan, evidence_directory=directory)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["evidence"], [])
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.page.screenshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
