import threading
import unittest
from http.server import ThreadingHTTPServer

from playwright.sync_api import sync_playwright

from qa_agent.llm.base import LLMProvider
from qa_agent.llm.router import LLMRouter
from qa_agent.models import QATestPlan
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
from qa_agent.test_case_authoring import TestCaseAuthoringService
from qa_agent.test_case_repository import InMemoryTestCaseRepository
from qa_agent.web import LocalWebApplication, create_http_server


class _BlockingAuthoringProvider(LLMProvider):
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        self.entered = entered
        self.release = release
        self.calls = 0

    def create_test_plan(self, task, target_url, page_snapshot):
        return QATestPlan(url=target_url, steps=[{"action": "assert_page_loaded"}])

    def create_structured_output(self, prompt, schema, schema_name):
        self.calls += 1
        self.entered.set()
        self.release.wait(5)
        return (
            '{"preconditions":[],"segments":[{"steps":[{"name":"Check title",'
            '"description":"Open the page and read its title.",'
            '"expected":"The title is Registration."}]}]}'
        )


class AsyncAuthoringWebIntegrationTests(unittest.TestCase):
    def test_local_browser_authoring_progress_review_edit_and_save(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        provider = _BlockingAuthoringProvider(entered, release)
        test_cases = InMemoryTestCaseRepository()
        history = RunHistoryService(InMemoryRunHistoryRepository())
        application = LocalWebApplication(
            history,
            test_cases=test_cases,
            authoring_service=TestCaseAuthoringService(LLMRouter([provider])),
        )
        server: ThreadingHTTPServer = create_http_server(
            application,
            host="127.0.0.1",
            port=0,
        )
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        origin = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(f"{origin}/test-cases/new")
                page.get_by_label("Name").fill("Async browser registration")
                page.get_by_label("Base URL").fill(f"{origin}/demo-target/registration")
                page.get_by_role("textbox", name="Scenario", exact=True).fill(
                    "Register and check the confirmation."
                )
                page.get_by_role("button", name="Generate Test with AI").click()
                page.wait_for_url("**/test-cases/authoring-progress/**")
                self.assertTrue(entered.wait(2))
                self.assertIn("Generating TestCase", page.locator("[data-authoring-phase]").inner_text())
                self.assertEqual(test_cases.list(), [])
                self.assertEqual(provider.calls, 1)
                release.set()
                page.wait_for_url("**/test-cases/review/**", timeout=8000)
                self.assertIn("Review TestCase", page.locator("h1").inner_text())
                page.locator("#edit-name").fill("Edited async registration")
                page.get_by_role("button", name="Save Test Case").click()
                page.wait_for_url("**/test-cases/*")
                self.assertEqual(len(test_cases.list()), 1)
                self.assertEqual(test_cases.list()[0].name, "Edited async registration")
                self.assertEqual(history.list_recent(), [])
                browser.close()
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
