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


class _AuthoringProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def create_test_plan(self, task, target_url, page_snapshot):
        return QATestPlan(url=target_url, steps=[{"action": "assert_page_loaded"}])

    def create_structured_output(self, prompt, schema, schema_name):
        self.calls += 1
        self.prompts.append(prompt)
        return (
            '{"preconditions":[],"segments":[{"steps":[{"name":"Check account",'
            '"description":"Open the account page.",'
            '"expected":"The account details are displayed."}]}]}'
        )


_FAKE_SPEECH_RECOGNITION = r"""
window.SpeechRecognition = class {
  constructor() { window.__lastRecognition = this; }
  start() { if (this.onstart) this.onstart(); }
  succeed(transcript) {
    const result = {0: {transcript}, isFinal: true, length: 1};
    if (this.onresult) this.onresult({results: [result], resultIndex: 0});
    if (this.onend) this.onend();
  }
  fail(error) {
    if (this.onerror) this.onerror({error});
    if (this.onend) this.onend();
  }
};
"""


class DashboardVoiceIntegrationTests(unittest.TestCase):
    def make_server(self):
        provider = _AuthoringProvider()
        test_cases = InMemoryTestCaseRepository()
        application = LocalWebApplication(
            RunHistoryService(InMemoryRunHistoryRepository()),
            test_cases=test_cases,
            authoring_service=TestCaseAuthoringService(LLMRouter([provider])),
        )
        server: ThreadingHTTPServer = create_http_server(
            application, host="127.0.0.1", port=0
        )
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        origin = f"http://127.0.0.1:{server.server_address[1]}"
        return application, provider, test_cases, server, server_thread, origin

    def stop_server(self, server, server_thread) -> None:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

    def test_voice_appends_editable_text_and_dashboard_reuses_async_authoring(self) -> None:
        application, provider, test_cases, server, server_thread, origin = self.make_server()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page()
                page.add_init_script(_FAKE_SPEECH_RECOGNITION)
                page.goto(origin)
                page.get_by_label("Website").fill(f"{origin}/demo-target/registration")
                scenario = page.get_by_role("textbox", name="Scenario", exact=True)
                scenario.fill("Open the account page.")

                page.get_by_role("button", name="Speak scenario").click()
                self.assertEqual(
                    page.get_by_role("button", name="Listening for scenario").get_attribute("aria-pressed"),
                    "true",
                )
                page.evaluate("window.__lastRecognition.fail('not-allowed')")
                self.assertIn("permission was denied", page.locator("[data-voice-status]").inner_text())
                self.assertNotIn("not-allowed", page.locator("[data-voice-status]").inner_text())

                page.get_by_role("button", name="Speak scenario").click()
                spoken = "Confirm the account name is shown. <img src=x onerror=window.__voiceXss=1>"
                page.evaluate("(text) => window.__lastRecognition.succeed(text)", spoken)
                self.assertEqual(scenario.input_value(), f"Open the account page.\n{spoken}")
                self.assertIn("Transcription added", page.locator("[data-voice-status]").inner_text())
                self.assertEqual(provider.calls, 0)
                self.assertEqual(test_cases.list(), [])
                language = page.evaluate("window.__lastRecognition.lang")
                self.assertEqual(
                    language,
                    page.evaluate("navigator.languages[0] || navigator.language || document.documentElement.lang || ''"),
                )

                page.get_by_role("button", name="Generate Test with AI").click()
                page.wait_for_url("**/test-cases/authoring-progress/**")
                page.wait_for_url("**/test-cases/review/**", timeout=8000)
                self.assertEqual(provider.calls, 1)
                self.assertIn(spoken, provider.prompts[0])
                self.assertEqual(
                    page.locator("#edit-name").input_value(),
                    "Account Page Behavior",
                )
                self.assertFalse(page.evaluate("window.__voiceXss || false"))
                self.assertEqual(page.locator("img").count(), 0)
                self.assertIn("&lt;img src=x", page.content())
                page.get_by_role("button", name="Save Test Case").click()
                page.wait_for_url("**/test-cases/*")
                self.assertEqual(len(test_cases.list()), 1)
                saved_scenario = test_cases.list()[0].description.replace("\r\n", "\n")
                self.assertEqual(saved_scenario, f"Open the account page.\n{spoken}")
                self.assertEqual(page.locator("img").count(), 0)
                self.assertEqual(provider.calls, 1)
                self.assertEqual(application.handle("POST", "/api/audio", b"audio").status, 405)
                browser.close()
        finally:
            self.stop_server(server, server_thread)

    def test_unsupported_voice_keeps_typed_authoring_available(self) -> None:
        _application, provider, _test_cases, server, server_thread, origin = self.make_server()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page()
                page.add_init_script(
                    "window.SpeechRecognition = undefined; window.webkitSpeechRecognition = undefined;"
                )
                page.goto(origin)
                microphone = page.get_by_role("button", name="Voice input unavailable")
                self.assertTrue(microphone.is_disabled())
                self.assertIn(
                    "Voice input is not supported in this browser",
                    page.locator("[data-voice-status]").inner_text(),
                )

                page.get_by_label("Website").fill(f"{origin}/demo-target/registration")
                page.get_by_role("textbox", name="Scenario", exact=True).fill(
                    "Check that the account page opens."
                )
                page.get_by_role("button", name="Generate Test with AI").click()
                page.wait_for_url("**/test-cases/authoring-progress/**")
                page.wait_for_url("**/test-cases/review/**", timeout=8000)
                self.assertEqual(provider.calls, 1)
                browser.close()
        finally:
            self.stop_server(server, server_thread)


if __name__ == "__main__":
    unittest.main()
