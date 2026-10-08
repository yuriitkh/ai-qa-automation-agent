import unittest
import tempfile
from pathlib import Path
from uuid import uuid4

from qa_agent.browser_runner import _run_plan_on_page
from qa_agent.cookie_consent import (
    CONSENT_CONTAINER_SELECTOR,
    CONSENT_CONTROL_SELECTOR,
    DEFAULT_COOKIE_CONSENT_POLICY,
    CookieConsentPolicy,
    CookieConsentReason,
    CookieConsentStatus,
    cookie_consent_scope,
    current_cookie_consent_policy,
    handle_cookie_consent,
)
from qa_agent.models import (
    Execution,
    ExecutionStatus,
    QATestPlan,
    TestCase as DomainTestCase,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
    TestRun as DomainTestRun,
)
from qa_agent.execution_repository import InMemoryExecutionRepository
from qa_agent.plan_execution import PlanExecutionClassification, PlanExecutionService
from qa_agent.reporting import RunReportGenerator
from qa_agent.run_history import RunHistoryRecord, WorkflowType


class _Action:
    def __init__(self, page, index=None):
        self.page = page
        self.index = index

    def click(self, **_kwargs):
        if self.index is None:
            self.page.normal_clicks += 1
        else:
            self.page.accept_clicks.append(self.index)


class _Controls:
    def __init__(self, page):
        self.page = page

    def evaluate_all(self, _script, _selector):
        return self.page.controls

    def nth(self, index):
        return _Action(self.page, index)


class _Container:
    def __init__(self, page):
        self.page = page

    def locator(self, selector):
        if selector != CONSENT_CONTROL_SELECTOR:
            raise AssertionError("Consent controls must be queried inside the consent container.")
        self.page.scoped_control_queries += 1
        return _Controls(self.page)


class _ContainerCollection:
    def __init__(self, page):
        self.page = page

    def nth(self, index):
        if index not in self.page.container_indexes:
            raise AssertionError("Only detected consent containers may be queried.")
        return _Container(self.page)


class _Page:
    def __init__(self, *, containers=None, dialogs=0, controls=None, dialog_text=None):
        self.url = "about:blank"
        self.containers = containers if containers is not None else []
        self.dialogs = dialogs
        self.dialog_text = dialog_text
        self.controls = controls if controls is not None else []
        self.container_indexes = [item["index"] for item in self.containers]
        self.evaluate_calls = 0
        self.container_queries = []
        self.scoped_control_queries = 0
        self.accept_clicks = []
        self.normal_clicks = 0
        self.navigation_urls = []
        self.screenshot_paths = []

    def evaluate(self, _script, _selector):
        self.evaluate_calls += 1
        return {
            "visible_dialog_count": self.dialogs,
            "containers": self.containers,
            "dialog_text": self.dialog_text,
        }

    def locator(self, selector):
        if selector == CONSENT_CONTAINER_SELECTOR:
            self.container_queries.append(selector)
            return _ContainerCollection(self)
        return _Action(self)

    def goto(self, url, **_kwargs):
        self.navigation_urls.append(url)
        self.url = url

    def wait_for_load_state(self, *_args, **_kwargs):
        return None

    def screenshot(self, *, path):
        self.screenshot_paths.append(path)


class CookieConsentDetectionTests(unittest.TestCase):
    def test_default_policy_is_auto_handle(self):
        self.assertEqual(DEFAULT_COOKIE_CONSENT_POLICY, CookieConsentPolicy.AUTO_HANDLE)
        self.assertEqual(current_cookie_consent_policy(), CookieConsentPolicy.AUTO_HANDLE)

    def test_leave_unchanged_skips_detection_and_clicks(self):
        page = _Page(
            containers=[{"index": 0}],
            controls=[{"index": 0, "name": "Accept all cookies", "visible": True, "enabled": True}],
        )
        with cookie_consent_scope(CookieConsentPolicy.LEAVE_UNCHANGED):
            result = handle_cookie_consent(page)

        self.assertEqual(result.status, CookieConsentStatus.LEFT_UNCHANGED)
        self.assertEqual(page.evaluate_calls, 0)
        self.assertEqual(page.container_queries, [])
        self.assertEqual(page.accept_clicks, [])

    def test_no_banner_does_not_search_or_click_page_wide_accept(self):
        page = _Page()
        with cookie_consent_scope(CookieConsentPolicy.AUTO_HANDLE):
            result = handle_cookie_consent(page)

        self.assertEqual(result.status, CookieConsentStatus.NO_BANNER)
        self.assertEqual(page.evaluate_calls, 1)
        self.assertEqual(page.container_queries, [])
        self.assertEqual(page.accept_clicks, [])

    def test_clicks_one_exact_accept_action_scoped_to_detected_container(self):
        page = _Page(
            containers=[{"index": 2}],
            controls=[
                {"index": 0, "name": "Reject all", "visible": True, "enabled": True},
                {"index": 1, "name": "Accept all cookies", "visible": True, "enabled": True},
            ],
        )
        with cookie_consent_scope(CookieConsentPolicy.AUTO_HANDLE):
            result = handle_cookie_consent(page)

        self.assertEqual(result.status, CookieConsentStatus.HANDLED)
        self.assertEqual(page.container_queries, [CONSENT_CONTAINER_SELECTOR])
        self.assertEqual(page.scoped_control_queries, 1)
        self.assertEqual(page.accept_clicks, [1])

    def test_fuzzy_accept_text_is_not_clicked(self):
        page = _Page(
            containers=[{"index": 0}],
            controls=[{"index": 0, "name": "Accept all cookies now", "visible": True, "enabled": True}],
        )
        with cookie_consent_scope(CookieConsentPolicy.AUTO_HANDLE):
            result = handle_cookie_consent(page)

        self.assertEqual(result.status, CookieConsentStatus.REQUIRES_ATTENTION)
        self.assertEqual(result.reason, CookieConsentReason.NO_SAFE_ACCEPT_ACTION)
        self.assertEqual(page.accept_clicks, [])

    def test_multiple_accept_actions_are_ambiguous_and_not_clicked(self):
        page = _Page(
            containers=[{"index": 0}],
            controls=[
                {"index": 0, "name": "Accept", "visible": True, "enabled": True},
                {"index": 1, "name": "Accept all cookies", "visible": True, "enabled": True},
            ],
        )
        with cookie_consent_scope(CookieConsentPolicy.AUTO_HANDLE):
            result = handle_cookie_consent(page)

        self.assertEqual(result.reason, CookieConsentReason.MULTIPLE_ACCEPT_ACTIONS)
        self.assertEqual(page.accept_clicks, [])

    def test_multiple_containers_or_dialogs_are_ambiguous(self):
        for containers, dialogs, reason in (
            ([{"index": 0}, {"index": 1}], 0, CookieConsentReason.MULTIPLE_CONTAINERS),
            ([{"index": 0}], 2, CookieConsentReason.MULTIPLE_DIALOGS),
        ):
            with self.subTest(reason=reason):
                page = _Page(containers=containers, dialogs=dialogs)
                with cookie_consent_scope(CookieConsentPolicy.AUTO_HANDLE):
                    result = handle_cookie_consent(page)
                self.assertEqual(result.reason, reason)
                self.assertEqual(page.container_queries, [])
                self.assertEqual(page.accept_clicks, [])

    def test_navigation_checks_consent_on_same_page_and_stops_before_next_action(self):
        page = _Page(containers=[{"index": 0}], controls=[])
        plan = QATestPlan(url="http://local.test", steps=[
            {"action": "navigate", "parameters": {"url": "http://local.test/banner"}},
            {"action": "click", "parameters": {"selector": "#next"}},
        ])
        with cookie_consent_scope(CookieConsentPolicy.AUTO_HANDLE):
            result = _run_plan_on_page(plan, page)

        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["cookie_consent_requires_attention"])
        self.assertEqual(result["steps"][-1]["error"], "Cookie consent requires attention.")
        self.assertEqual(page.navigation_urls, ["http://local.test/banner"])
        self.assertEqual(page.normal_clicks, 0)
        self.assertEqual(page.accept_clicks, [])
        self.assertEqual(result["evidence"], [])

    def test_attention_screenshot_is_attached_only_when_evidence_is_configured(self):
        page = _Page(containers=[{"index": 0}], controls=[])
        plan = QATestPlan(url="http://local.test", steps=[
            {"action": "navigate", "parameters": {"url": "http://local.test/banner"}},
        ])
        with tempfile.TemporaryDirectory() as directory:
            with cookie_consent_scope(CookieConsentPolicy.AUTO_HANDLE):
                result = _run_plan_on_page(plan, page, directory)

            self.assertEqual(len(result["evidence"]), 1)
            evidence = result["evidence"][0]
            self.assertEqual(evidence["event"], "COOKIE_CONSENT_FAILURE")
            self.assertEqual(evidence["scope"], "PAGE")
            self.assertTrue(evidence["path"].startswith(directory))

    def test_leave_unchanged_allows_explicit_cookie_case_actions(self):
        page = _Page(containers=[{"index": 0}], controls=[])
        plan = QATestPlan(url="http://local.test", steps=[
            {"action": "navigate", "parameters": {"url": "http://local.test/banner"}},
            {"action": "click", "parameters": {"selector": "#verify-cookie-banner"}},
        ])
        with cookie_consent_scope(CookieConsentPolicy.LEAVE_UNCHANGED):
            result = _run_plan_on_page(plan, page)

        self.assertEqual(result["status"], "passed")
        self.assertEqual(page.evaluate_calls, 0)
        self.assertEqual(page.normal_clicks, 1)
        self.assertEqual(page.accept_clicks, [])


class CookieConsentSafetyTests(unittest.TestCase):
    def test_attention_marker_is_automation_error_not_product_failure(self):
        step = DomainTestStep(
            name="Verify banner", description="Check consent UI", expected="Banner shown", order=0
        )
        version = DomainTestPlanVersion(
            test_plan_id=uuid4(),
            version=1,
            qa_test_plan=QATestPlan(url="http://local.test", steps=[
                {"action": "assert_visible", "parameters": {"selector": "#banner"}},
            ]),
        )
        result = PlanExecutionService(lambda _plan: {
            "status": "failed",
            "cookie_consent_requires_attention": True,
            "steps": [{
                "action": "cookie_consent_precondition",
                "status": "failed",
                "error": "Cookie consent requires attention.",
            }],
        }, InMemoryExecutionRepository()).execute(step, version)

        self.assertEqual(
            result.classification,
            PlanExecutionClassification.AUTOMATION_EXECUTION_ERROR,
        )
        self.assertIsNone(result.execution.planned_step_index)
        self.assertEqual(result.execution.error, "Cookie consent requires attention.")

    def test_policy_and_safe_outcome_persist_without_dialog_contents(self):
        private_dialog_text = "PRIVATE_COOKIE_DIALOG_TEXT_7129"
        page = _Page(containers=[{"index": 0}], controls=[], dialog_text=private_dialog_text)
        step = DomainTestStep(
            name="Verify banner", description="Check consent UI", expected="Banner shown", order=0
        )
        case = DomainTestCase(
            name="Cookie behavior", description="Verify the local consent banner", steps=[step]
        )
        execution = Execution(
            test_step_id=step.id,
            test_plan_version_id=uuid4(),
            status=ExecutionStatus.FAILED,
            actual_result="failed",
            error="Cookie consent requires attention.",
            runner_result={
                "status": "failed",
                "cookie_consent_requires_attention": True,
                "steps": [{"action": "cookie_consent_precondition", "status": "failed", "error": "Cookie consent requires attention."}],
            },
        )
        run = DomainTestRun.from_test_case(case, [execution])
        with cookie_consent_scope(CookieConsentPolicy.AUTO_HANDLE):
            handle_cookie_consent(page)
            record = RunHistoryRecord.from_completed_run(
                case,
                run,
                workflow_type=WorkflowType.REGRESSION,
                outcome="AUTOMATION_EXECUTION_ERROR",
            )

        serialized = record.model_dump_json()
        self.assertEqual(record.cookie_consent.status, CookieConsentStatus.REQUIRES_ATTENTION)
        self.assertNotIn(private_dialog_text, serialized)
        report = RunReportGenerator().generate_history(record)
        html = RunReportGenerator().to_html(report)
        self.assertIn("Cookie consent", html)
        self.assertIn("Requires attention", html)
        self.assertNotIn(private_dialog_text, html)

    def test_old_history_without_cookie_metadata_remains_valid(self):
        payload = {
            "run_id": str(uuid4()),
            "test_case_id": str(uuid4()),
            "test_case_name": "Old run",
            "test_case_description": "Historical snapshot",
            "workflow_type": WorkflowType.REGRESSION.value,
            "outcome": "PASSED",
            "status": ExecutionStatus.PASSED.value,
            "started_at": "2026-01-01T00:00:00Z",
        }
        record = RunHistoryRecord.model_validate(payload)
        self.assertIsNone(record.cookie_consent)


if __name__ == "__main__":
    unittest.main()
