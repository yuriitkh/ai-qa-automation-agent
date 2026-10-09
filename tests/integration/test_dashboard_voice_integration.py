import threading
import unittest
from uuid import UUID
from http.server import ThreadingHTTPServer

from playwright.sync_api import sync_playwright

from qa_agent.llm.base import LLMProvider
from qa_agent.drafts import Draft, DraftStatus
from qa_agent.llm.router import LLMRouter
from qa_agent.models import QATestPlan
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService
from qa_agent.test_case_authoring import TestCaseAuthoringService
from qa_agent.test_case_repository import InMemoryTestCaseRepository
from qa_agent.web import LocalWebApplication, create_http_server


class _AuthoringProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0
        self.plan_calls = 0
        self.prompts: list[str] = []

    def create_test_plan(self, task, target_url, page_snapshot):
        self.plan_calls += 1
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

    def test_shared_layout_draft_selection_validation_and_incomplete_save_in_browser(self) -> None:
        application, provider, test_cases, server, server_thread, origin = self.make_server()
        for index in range(24):
            application._drafts.save(Draft(
                title=f'Account Draft {index} <unsafe>',
                body="Check login, account details and logout. " * 35,
                base_url=f"{origin}/account",
            ))
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1280, "height": 1000})
                for path in ("/", "/test-cases/new"):
                    with self.subTest(path=path):
                        page.set_viewport_size({"width": 1280, "height": 1000})
                        page.goto(origin + path)
                        self.assertEqual(page.locator('[data-draft-select]').count(), 20)
                        self.assertEqual(page.get_by_text("Recent Drafts", exact=True).count(), 0)
                        form_box = page.locator('.authoring-entry').bounding_box()
                        panel_box = page.locator('.drafts-sidebar').bounding_box()
                        self.assertAlmostEqual(form_box["height"], panel_box["height"], delta=1)
                        self.assertGreater(panel_box["x"], form_box["x"] + form_box["width"])
                        self.assertTrue(page.locator('.draft-sidebar-list').evaluate(
                            '(list) => list.scrollHeight > list.clientHeight'))
                        preview = page.locator('.draft-scenario-preview').first
                        self.assertTrue(preview.evaluate(
                            '(element) => element.clientHeight <= parseFloat(getComputedStyle(element).lineHeight) * 2 + 1'))
                        source_button = page.locator('[data-draft-select]').first
                        draft_id = UUID(source_button.get_attribute('data-draft-id'))
                        before = application._drafts.get(draft_id)
                        source_button.click()
                        self.assertEqual(page.get_by_label("Summary (optional)").input_value(), before.title)
                        self.assertEqual(page.get_by_label("Website", exact=True).input_value(), before.base_url)
                        self.assertEqual(page.get_by_role("textbox", name="Scenario", exact=True).input_value(), before.body)
                        self.assertEqual(application._drafts.get(draft_id), before)
                        self.assertEqual(page.locator('unsafe').count(), 0)
                        page.get_by_label("Summary (optional)").fill("My edited account summary")
                        page.get_by_role("textbox", name="Scenario", exact=True).fill("aa")
                        page.get_by_role("button", name="Generate TestCase", exact=True).click()
                        self.assertEqual(page.url, origin + path)
                        self.assertIn("Describe an action", page.locator('#case-scenario-error').inner_text())
                        self.assertTrue(page.get_by_role("button", name="Generate TestCase", exact=True).is_enabled())
                        self.assertEqual(provider.calls, 0)
                        page.set_viewport_size({"width": 390, "height": 844})
                        form_box = page.locator('.authoring-entry').bounding_box()
                        panel_box = page.locator('.drafts-sidebar').bounding_box()
                        self.assertGreaterEqual(panel_box["y"], form_box["y"] + form_box["height"])
                        self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'), 390)
                        page.get_by_role("button", name="Save Draft", exact=True).click()
                        page.wait_for_url(f"**/drafts/{draft_id}?saved=1")
                        saved = application._drafts.get(draft_id)
                        self.assertEqual(saved.title, "My edited account summary")
                        self.assertEqual(saved.body, "aa")
                        self.assertEqual(saved.status, DraftStatus.ACTIVE)
                        self.assertEqual(provider.calls, 0)
                        self.assertEqual(test_cases.list(), [])
                browser.close()
        finally:
            application.close()
            self.stop_server(server, server_thread)

    def test_manual_action_transfers_current_fields_without_ai_or_validation_gate(self) -> None:
        application, provider, test_cases, server, server_thread, origin = self.make_server()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page()
                for path in ("/", "/test-cases/new"):
                    page.goto(origin + path)
                    page.get_by_label("Summary (optional)").fill('Edited summary <unsafe>')
                    page.get_by_label("Website", exact=True).fill(origin + '/updated-account')
                    page.get_by_role("textbox", name="Scenario", exact=True).fill('aa')
                    page.get_by_role("button", name="Create Manually", exact=True).click()
                    page.wait_for_url('**/test-cases/manual/prepare')
                    self.assertEqual(page.locator('#manual-name').input_value(), 'Edited summary <unsafe>')
                    self.assertEqual(page.locator('#manual-base-url').input_value(), origin + '/updated-account')
                    self.assertEqual(page.locator('#manual-description').input_value(), 'aa')
                    self.assertEqual(page.locator('unsafe').count(), 0)
                    self.assertEqual(provider.calls, 0)
                    self.assertEqual(test_cases.list(), [])
                browser.close()
        finally:
            application.close()
            self.stop_server(server, server_thread)

    def test_user_summary_survives_generation_and_review_in_browser(self) -> None:
        application, provider, test_cases, server, server_thread, origin = self.make_server()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(origin)
                page.get_by_label("Summary (optional)").fill('User chosen account summary')
                page.get_by_label("Website", exact=True).fill(origin + '/account')
                page.get_by_role("textbox", name="Scenario", exact=True).fill('Check login')
                page.get_by_role("button", name="Generate TestCase", exact=True).click()
                page.wait_for_url('**/test-cases/review/**', timeout=8000)
                self.assertEqual(page.locator('#edit-name').input_value(), 'User chosen account summary')
                self.assertEqual(provider.calls, 1)
                self.assertEqual(test_cases.list(), [])
                page.locator('#edit-name').fill('Summary edited during review')
                page.locator('#edit-description').fill('Unsaved Scenario edits stay separate.')
                page.get_by_role("button", name="Regenerate TestCase with AI", exact=True).click()
                page.wait_for_url('**/test-cases/authoring-progress/**')
                page.wait_for_url('**/test-cases/review/**', timeout=8000)
                self.assertEqual(page.locator('#edit-name').input_value(), 'Summary edited during review')
                self.assertEqual(page.locator('#edit-description').input_value(), 'Check login')
                self.assertEqual(provider.calls, 2)
                self.assertEqual(provider.plan_calls, 0)
                page.get_by_role("button", name="Save TestCase").click()
                page.wait_for_url('**/test-cases/*')
                self.assertEqual(test_cases.list()[0].name, 'Summary edited during review')
                self.assertIn('Ready for review', page.content())
                self.assertEqual(provider.calls, 2)
                browser.close()
        finally:
            application.close()
            self.stop_server(server, server_thread)

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

                page.get_by_role("button", name="Generate TestCase").click()
                page.wait_for_url("**/test-cases/authoring-progress/**")
                page.wait_for_url("**/test-cases/review/**", timeout=8000)
                self.assertEqual(provider.calls, 1)
                self.assertIn(spoken, provider.prompts[0])
                self.assertEqual(
                    page.locator("#edit-name").input_value(),
                "Open the account page",
                )
                self.assertFalse(page.evaluate("window.__voiceXss || false"))
                self.assertEqual(page.locator("img").count(), 0)
                self.assertIn("&lt;img src=x", page.content())
                page.get_by_role("button", name="Save TestCase").click()
                page.wait_for_url("**/test-cases/*")
                self.assertEqual(len(test_cases.list()), 1)
                self.assertIn("Ready for review", page.content())
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
                page.get_by_role("button", name="Generate TestCase").click()
                page.wait_for_url("**/test-cases/authoring-progress/**")
                page.wait_for_url("**/test-cases/review/**", timeout=8000)
                self.assertEqual(provider.calls, 1)
                browser.close()
        finally:
            self.stop_server(server, server_thread)


if __name__ == "__main__":
    unittest.main()
