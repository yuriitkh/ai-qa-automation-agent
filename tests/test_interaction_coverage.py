"""Local-browser regression coverage for common demo interactions."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import tempfile
import unittest
from urllib.parse import urlsplit

from qa_agent.browser_runner import BrowserRunner
from qa_agent.cookie_consent import CookieConsentPolicy, CookieConsentStatus, cookie_consent_scope
from qa_agent.evidence_policy import EvidenceMode, EvidencePolicy, ScreenshotMode, evidence_policy_scope
from qa_agent.models import QATestPlan
from qa_agent.web import _local_demo_help_page, _local_demo_page, _local_demo_javascript


class _InteractionDemoHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/demo-target/registration":
            body = _local_demo_page().encode("utf-8")
        elif path == "/demo-target/registration/help":
            body = _local_demo_help_page().encode("utf-8")
        elif path == "/assets/demo-registration.js":
            body = _local_demo_javascript().encode("utf-8")
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/javascript; charset=utf-8" if path.endswith(".js") else "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class LocalInteractionCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _InteractionDemoHandler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.target_url = f"http://127.0.0.1:{cls.server.server_address[1]}/demo-target/registration"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def execute(self, actions: list[dict], *, policy: CookieConsentPolicy, evidence_directory=None):
        plan = QATestPlan(url=self.target_url, steps=[
            {"action": "navigate", "parameters": {"url": self.target_url}},
            *actions,
        ])
        runner = BrowserRunner(evidence_directory, headless=True)
        with runner.open_session() as session:
            page = session.new_page()
            with cookie_consent_scope(policy):
                return session.run_plan(page, plan)

    def test_auto_cookie_handling_and_common_controls_continue_locally(self) -> None:
        password = "local-demo-password-sentinel"
        actions = [
            {"action": "assert_hidden", "parameters": {"selector": "#cookie-consent"}},
            {"action": "click", "parameters": {"selector": "#change-status"}},
            {"action": "assert_text_contains", "parameters": {
                "selector": "#page-status", "expected_text": "Page state changed."
            }},
            {"action": "fill", "parameters": {"selector": "#email", "value": "bad"}},
            {"action": "fill", "parameters": {"selector": "#password", "value": password}},
            {"action": "click", "parameters": {"selector": "#create-account"}},
            {"action": "assert_visible", "parameters": {
                "selector": "#form-error", "expected_text": "Enter a valid email address."
            }},
            {"action": "assert_text_contains", "parameters": {
                "selector": "#form-error", "expected_text": "Enter a valid email address."
            }},
            {"action": "fill", "parameters": {"selector": "#email", "value": "ada@example.test"}},
            {"action": "click", "parameters": {"selector": "#create-account"}},
            {"action": "assert_visible", "parameters": {"selector": "#registration-success"}},
            {"action": "assert_unchecked", "parameters": {"selector": "#terms"}},
            {"action": "check", "parameters": {"selector": "#terms"}},
            {"action": "assert_checked", "parameters": {"selector": "#terms"}},
            {"action": "uncheck", "parameters": {"selector": "#terms"}},
            {"action": "assert_unchecked", "parameters": {"selector": "#terms"}},
            {"action": "click", "parameters": {"selector": "#plan-pro"}},
            {"action": "assert_selected", "parameters": {"selector": "#plan-pro"}},
            {"action": "select_option", "parameters": {
                "selector": "#region", "option_label": "North"
            }},
            {"action": "assert_selected", "parameters": {
                "selector": "#region", "expected": "north"
            }},
            {"action": "click", "parameters": {"selector": "#open-details"}},
            {"action": "assert_visible", "parameters": {"selector": "#details-dialog"}},
            {"action": "click", "parameters": {"selector": "#confirm-details"}},
            {"action": "assert_hidden", "parameters": {"selector": "#details-dialog"}},
            {"action": "assert_text_contains", "parameters": {
                "selector": "#dialog-result", "expected_text": "Details confirmed."
            }},
        ]

        with tempfile.TemporaryDirectory() as evidence_directory:
            policy = EvidencePolicy(
                mode=EvidenceMode.EVERY_VERIFICATION,
                screenshot_mode=ScreenshotMode.PAGE,
            )
            with evidence_policy_scope(policy):
                result = self.execute(
                    actions,
                    policy=CookieConsentPolicy.AUTO_HANDLE,
                    evidence_directory=evidence_directory,
                )

        self.assertEqual(result["status"], "passed", result.get("steps"))
        self.assertEqual(result["cookie_consent"]["status"], CookieConsentStatus.HANDLED.value)
        self.assertNotIn(password, json.dumps(result))
        verification_count = sum(action["action"].startswith("assert_") for action in actions)
        verification_evidence = [
            item for item in result["evidence"]
            if item["event"].startswith("VERIFICATION:")
        ]
        self.assertEqual(len(verification_evidence), verification_count)
        self.assertTrue(all(item["scope"] == "PAGE" for item in verification_evidence))

    def test_leave_unchanged_keeps_dialog_and_accept_button_testable(self) -> None:
        result = self.execute(
            [
                {"action": "assert_visible", "parameters": {"selector": "#cookie-consent"}},
                {"action": "assert_visible", "parameters": {"selector": "#accept-cookies"}},
            ],
            policy=CookieConsentPolicy.LEAVE_UNCHANGED,
        )

        self.assertEqual(result["status"], "passed", result.get("steps"))
        self.assertEqual(
            result["cookie_consent"]["status"], CookieConsentStatus.LEFT_UNCHANGED.value
        )

    def test_navigation_and_same_page_links_have_follow_up_url_assertions(self) -> None:
        help_url = self.target_url + "/help"
        navigation = self.execute(
            [
                {"action": "click", "parameters": {"selector": "#help-link"}},
                {"action": "assert_url", "parameters": {"expected": help_url}},
                {"action": "assert_visible", "parameters": {
                    "selector": "h1", "expected_text": "Registration help"
                }},
            ],
            policy=CookieConsentPolicy.AUTO_HANDLE,
        )
        self.assertEqual(navigation["status"], "passed", navigation.get("steps"))

        fragment_url = self.target_url + "#interaction-controls"
        same_page = self.execute(
            [
                {"action": "click", "parameters": {"selector": "#details-link"}},
                {"action": "assert_url", "parameters": {"expected": fragment_url}},
                {"action": "assert_visible", "parameters": {"selector": "#interaction-controls"}},
            ],
            policy=CookieConsentPolicy.AUTO_HANDLE,
        )
        self.assertEqual(same_page["status"], "passed", same_page.get("steps"))

    def test_wrong_radio_option_assertion_fails(self) -> None:
        result = self.execute(
            [
                {"action": "click", "parameters": {"selector": "#plan-basic"}},
                {"action": "assert_selected", "parameters": {"selector": "#plan-pro"}},
            ],
            policy=CookieConsentPolicy.AUTO_HANDLE,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["steps"][-1]["action"], "assert_selected")


if __name__ == "__main__":
    unittest.main()
