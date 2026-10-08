from __future__ import annotations

import json
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from urllib.parse import urlencode
from uuid import uuid4

from qa_agent.execution_repository import InMemoryExecutionRepository
from qa_agent.automation_lifecycle import AutomationLifecycleService, InMemoryAutomationLifecycleRepository
from qa_agent.evidence_policy import EvidenceMode, EvidencePolicy, ScreenshotMode
from qa_agent.cookie_consent import (
    CookieConsentPolicy,
    CookieConsentStatus,
    current_cookie_consent_policy,
)
from qa_agent.models import (
    AssertionGrounding,
    AssertionGroundingEntry,
    PlanVersionOrigin,
    QATestPlan,
    QATestStep,
    TestCase as DomainTestCase,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
)
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.run_history import InMemoryRunHistoryRepository, RunHistoryService, WorkflowType
from qa_agent.suite_run_storage import SQLiteSuiteRunRepository
from qa_agent.suite_runs import (
    AIPolicy,
    InMemorySuiteRunRepository,
    PinnedSuitePlan,
    SuiteRun,
    SuiteRunAttempt,
    SuiteRunConfig,
    SuiteRunItem,
    SuiteRunItemStatus,
    SuiteRunService,
    SuiteRunStatus,
    suite_run_report_json,
)
from qa_agent.test_case_execution import TestCaseExecutionService, RunUnavailableError
from qa_agent.test_case_review import InMemoryTestCaseReviewRepository, TestCaseReviewService as ReviewService
from qa_agent.pinned_execution import PlanVersionSet, StepPlanSelection
from qa_agent.test_case_repository import InMemoryTestCaseRepository
from qa_agent.test_suites import InMemoryTestSuiteRepository, TestSuiteService as DomainTestSuiteService
from qa_agent.web import LocalWebApplication


def _case(name: str = "Grounded title check") -> DomainTestCase:
    return DomainTestCase(
        name=name,
        description="Verify the saved page title.",
        base_url="http://127.0.0.1:8000/local",
        steps=[DomainTestStep(
            name="Verify title",
            description="Verify that the page title is Welcome.",
            expected="The page title is Welcome.",
            order=0,
        )],
    )


def _plan(
    case: DomainTestCase,
    version: int = 1,
    url: str = "http://127.0.0.1:8000/v1",
    plan: DomainTestPlan | None = None,
) -> tuple[DomainTestPlan, DomainTestPlanVersion]:
    plan = plan or DomainTestPlan(test_step_id=case.steps[0].id, name="Title assertion")
    saved = DomainTestPlanVersion(
        test_plan_id=plan.id,
        version=version,
        origin=PlanVersionOrigin.AI_GENERATED,
        qa_test_plan=QATestPlan(
            url=url,
            steps=[QATestStep(action="assert_title", parameters={"expected": "Welcome"})],
        ),
        assertion_grounding=(AssertionGroundingEntry(
            step_index=0,
            category=AssertionGrounding.REQUIREMENT_GROUNDED,
        ),),
    )
    return plan, saved


class SuiteRunTestHarness:
    def setUp(self):
        self.cases = InMemoryTestCaseRepository()
        self.case = _case()
        self.cases.save(self.case)
        self.case = self.cases.get(self.case.id)
        self.plans = InMemoryPlanStore()
        plan, saved = _plan(self.case)
        self.plans.save(self.case.steps[0].id, saved, test_plan=plan)
        self.execution_repository = InMemoryExecutionRepository()
        self.history_repository = InMemoryRunHistoryRepository()
        self.history = RunHistoryService(
            self.history_repository,
            self.execution_repository,
            plan_store=self.plans,
        )
        self.suite_repository = InMemoryTestSuiteRepository()
        self.suites = DomainTestSuiteService(self.suite_repository, self.cases)
        self.suite = self.suites.create("Smoke suite", "Local sequential checks")
        self.suites.add_member(self.suite.id, self.case.id)

    def make_service(self, runner, *, repository=None):
        run_service = TestCaseExecutionService(
            self.cases,
            self.plans,
            self.execution_repository,
            self.history,
            runner_factory=lambda _directory: runner,
            automation_workflow=_UnexpectedAIWorkflow(),
        )
        self.run_service = run_service
        self.suite_runs = SuiteRunService(
            self.suites,
            self.cases,
            run_service,
            self.history,
            repository or InMemorySuiteRunRepository(),
            max_workers=1,
            max_pending=0,
        )
        return self.suite_runs

    def wait_for_terminal(self, service, public_id: str):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            run = service.get(public_id)
            if run and run.status not in {SuiteRunStatus.QUEUED, SuiteRunStatus.RUNNING}:
                return run
            time.sleep(0.01)
        self.fail("Suite Run did not reach a terminal status.")


class _UnexpectedAIWorkflow:
    def run_test_case(self, *_args, **_kwargs):
        raise AssertionError("Regression Suite Runs must never invoke AI workflows.")


class SuiteRunExecutionTests(SuiteRunTestHarness, unittest.TestCase):
    def reviewed_service(self, runner):
        service = self.make_service(runner)
        self.review = ReviewService(InMemoryTestCaseReviewRepository(), self.plans)
        self.review.mark_ready_for_review(self.case)
        self.run_service._test_case_review = self.review
        self.lifecycle = AutomationLifecycleService(InMemoryAutomationLifecycleRepository(), self.plans)
        self.run_service._automation_lifecycle = self.lifecycle
        self.lifecycle.mark_automation_completed(self.case)
        return service

    def test_suite_enforces_case_and_exact_plan_approval_without_history(self):
        calls = []
        service = self.reviewed_service(lambda plan: calls.append(plan) or {"status": "passed"})
        try:
            self.assertFalse(service.preview(self.suite.id).can_start)
            self.review.approve_test_case(self.case)
            self.assertFalse(service.preview(self.suite.id).can_start)
            self.review.approve_for_validation(self.case)
            self.assertTrue(service.preview(self.suite.id).can_start)
            old_version = self.plans.find(self.case.steps[0].id)
            plan, version = _plan(self.case, version=2, plan=self.plans.find_test_plan(self.case.steps[0].id))
            self.plans.save(self.case.steps[0].id, version, test_plan=plan)
            self.assertFalse(service.preview(self.suite.id).can_start)
            with self.assertRaises(RunUnavailableError):
                self.run_service.run_pinned_regression(self.case, PlanVersionSet((
                    StepPlanSelection(self.case.steps[0].id, version.id),
                )))
            self.assertNotEqual(old_version.id, version.id)
            self.assertEqual(calls, [])
            self.assertEqual(self.history.list_for_test_case(self.case.id), [])
        finally:
            service.close()

    def test_reviewed_retries_keep_approved_pins_when_latest_automation_changes(self):
        entered = Event()
        release = Event()
        observed = []
        def runner(plan):
            observed.append(plan.url)
            if len(observed) == 1:
                entered.set()
                self.assertTrue(release.wait(3))
                return {"status": "failed", "steps": [{"action": "assert_title", "status": "failed", "error": "mismatch"}]}
            return {"status": "passed", "steps": [{"action": "assert_title", "status": "passed"}]}
        service = self.reviewed_service(runner)
        self.review.approve_test_case(self.case)
        self.review.approve_for_validation(self.case)
        try:
            started = service.start(self.suite.id, retry_count=1)
            self.assertIsNotNone(started.run)
            self.assertTrue(entered.wait(3))
            plan, version = _plan(self.case, version=2, plan=self.plans.find_test_plan(self.case.steps[0].id))
            # The latest incomplete automation must block new suites without invalidating old pins.
            version = version.model_copy(update={"qa_test_plan": QATestPlan(
                url=version.qa_test_plan.url, steps=[QATestStep(action="click", parameters={})],
            )})
            self.plans.save(self.case.steps[0].id, version, test_plan=plan)
            self.assertFalse(service.preview(self.suite.id).can_start)
            release.set()
            finished = self.wait_for_terminal(service, started.run.public_id)
            self.assertEqual(finished.items[0].status, SuiteRunItemStatus.PASSED_AFTER_RETRY)
            self.assertEqual(observed, ["http://127.0.0.1:8000/v1"] * 2)
        finally:
            release.set()
            service.close()

    def test_evidence_policy_is_stored_and_propagated_to_each_child_run(self):
        policy = EvidencePolicy(
            mode=EvidenceMode.EVERY_VERIFICATION,
            screenshot_mode=ScreenshotMode.ELEMENT,
        )
        service = self.make_service(lambda _plan: {
            "status": "passed",
            "steps": [{"action": "assert_title", "status": "passed"}],
        })
        try:
            started = service.start(
                self.suite.id,
                ai_policy=AIPolicy.DISABLED,
                evidence_policy=policy,
            )
            finished = self.wait_for_terminal(service, started.run.public_id)
        finally:
            service.close()

        self.assertEqual(finished.config.evidence_policy, policy)
        history = self.history.list_for_test_case(self.case.id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].evidence_policy, policy)

    def test_cookie_policy_is_persisted_and_propagated_without_ai(self):
        observed = []

        def runner(_plan):
            observed.append(current_cookie_consent_policy())
            return {"status": "passed", "steps": [{"action": "assert_title", "status": "passed"}]}

        service = self.make_service(runner)
        try:
            started = service.start(
                self.suite.id,
                ai_policy=AIPolicy.ALLOWED,
                cookie_policy=CookieConsentPolicy.LEAVE_UNCHANGED,
            )
            finished = self.wait_for_terminal(service, started.run.public_id)
        finally:
            service.close()

        self.assertEqual(finished.config.cookie_policy, CookieConsentPolicy.LEAVE_UNCHANGED)
        self.assertEqual(observed, [CookieConsentPolicy.LEAVE_UNCHANGED])
        history = self.history.list_for_test_case(self.case.id)
        self.assertEqual(history[0].cookie_consent.policy, CookieConsentPolicy.LEAVE_UNCHANGED)
        self.assertEqual(history[0].cookie_consent.status, CookieConsentStatus.LEFT_UNCHANGED)

    def test_retry_is_fresh_run_and_keeps_the_exact_start_time_pin(self):
        entered_runner = Event()
        release_runner = Event()
        observed_urls = []

        def runner(plan):
            observed_urls.append(plan.url)
            if len(observed_urls) == 1:
                entered_runner.set()
                self.assertTrue(release_runner.wait(3))
                return {
                    "status": "failed",
                    "steps": [{"action": "assert_title", "status": "failed", "error": "title differs"}],
                    "evidence": [{
                        "type": "SCREENSHOT", "path": "artifacts/attempt-one.png",
                        "description": "First attempt failure.", "scope": "PAGE",
                        "event": "FAILURE:assert_title:0",
                    }],
                }
            return {
                "status": "passed",
                "steps": [{"action": "assert_title", "status": "passed"}],
                "evidence": [{
                    "type": "SCREENSHOT", "path": "artifacts/attempt-two.png",
                    "description": "Retry verification.", "scope": "PAGE",
                    "event": "VERIFICATION:assert_title:0",
                }],
            }

        service = self.make_service(runner)
        try:
            started = service.start(self.suite.id, ai_policy=AIPolicy.DISABLED, retry_count=1)
            self.assertIsNotNone(started.run)
            self.assertTrue(entered_runner.wait(3))
            original_pin = started.run.items[0].pinned_plans[0].test_plan_version_id
            new_plan, new_version = _plan(
                self.case,
                version=2,
                url="http://127.0.0.1:8000/v2",
                plan=self.plans.find_test_plan(self.case.steps[0].id),
            )
            self.plans.save(self.case.steps[0].id, new_version, test_plan=new_plan)
            release_runner.set()
            finished = self.wait_for_terminal(service, started.run.public_id)
        finally:
            release_runner.set()
            service.close()

        self.assertEqual(finished.status, SuiteRunStatus.COMPLETED)
        item = finished.items[0]
        self.assertEqual(item.status, SuiteRunItemStatus.PASSED_AFTER_RETRY)
        self.assertEqual(len(item.attempts), 2)
        self.assertEqual([attempt.attempt_number for attempt in item.attempts], [1, 2])
        self.assertIsNotNone(item.attempts[0].run_id)
        self.assertIsNotNone(item.attempts[1].run_id)
        self.assertNotEqual(item.attempts[0].run_id, item.attempts[1].run_id)
        self.assertEqual(observed_urls, ["http://127.0.0.1:8000/v1"] * 2)
        self.assertEqual(item.pinned_plans[0].test_plan_version_id, original_pin)
        self.assertEqual(len(self.history.list_for_test_case(self.case.id)), 2)
        self.assertEqual(
            [record.executions[0].test_plan_version_id for record in self.history.list_for_test_case(self.case.id)],
            [original_pin, original_pin],
        )
        run_records = {
            record.run_id: record
            for record in self.history.list_for_test_case(self.case.id)
        }
        first_record = run_records[item.attempts[0].run_id]
        second_record = run_records[item.attempts[1].run_id]
        self.assertEqual(first_record.executions[0].evidence[0].name, "attempt-one.png")
        self.assertEqual(second_record.executions[0].evidence[0].name, "attempt-two.png")
        self.assertNotEqual(
            first_record.executions[0].execution_id,
            second_record.executions[0].execution_id,
        )

    def test_failure_isolation_continues_to_next_member_and_ai_policy_does_not_call_ai(self):
        next_case = _case("Second title check")
        self.cases.save(next_case)
        next_case = self.cases.get(next_case.id)
        next_plan, next_version = _plan(next_case, url="http://127.0.0.1:8000/v2")
        self.plans.save(next_case.steps[0].id, next_version, test_plan=next_plan)
        self.suites.add_member(self.suite.id, next_case.id)
        calls = []

        def runner(plan):
            calls.append(plan.url)
            if plan.url.endswith("/v1"):
                return {"status": "failed", "steps": [{"action": "assert_title", "status": "failed", "error": "mismatch"}]}
            return {"status": "passed", "steps": [{"action": "assert_title", "status": "passed"}]}

        service = self.make_service(runner)
        try:
            started = service.start(self.suite.id, ai_policy=AIPolicy.ALLOWED, retry_count=0)
            finished = self.wait_for_terminal(service, started.run.public_id)
        finally:
            service.close()

        self.assertEqual(finished.status, SuiteRunStatus.COMPLETED_WITH_FAILURES)
        self.assertEqual([item.status for item in finished.items], [SuiteRunItemStatus.FAILED, SuiteRunItemStatus.PASSED])
        self.assertEqual(len(self.history.list_for_test_case(self.case.id)), 1)
        self.assertEqual(len(self.history.list_for_test_case(next_case.id)), 1)
        self.assertEqual(len(calls), 2)

    def test_ineligible_member_is_shown_and_prevents_start_without_omission(self):
        missing_id = uuid4()
        self.suite_repository.add_member(self.suite.id, missing_id)
        service = self.make_service(lambda _plan: self.fail("Ineligible Suite Run must not execute."))
        try:
            preview = service.preview(self.suite.id)
            result = service.start(self.suite.id)
        finally:
            service.close()

        self.assertEqual([entry.test_case_id for entry in preview.eligibility], [self.case.id, missing_id])
        self.assertTrue(preview.eligibility[0].eligible)
        self.assertFalse(preview.eligibility[1].eligible)
        self.assertIsNone(result.run)
        self.assertEqual(len(result.eligibility), 2)


class SuiteRunStorageTests(unittest.TestCase):
    def test_sqlite_public_ids_progress_and_restart_interruption_are_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "suite-runs.sqlite3"
            case = _case()
            plan = PinnedSuitePlan(
                test_step_id=case.steps[0].id,
                test_plan_version_id=uuid4(),
                version_number=3,
            )
            repository = SQLiteSuiteRunRepository(database)
            first = repository.save(SuiteRun(
                suite_id=uuid4(),
                suite_name="Persistent",
                config=SuiteRunConfig(retry_count=1),
                items=[SuiteRunItem(
                    order_index=0,
                    test_case_id=case.id,
                    test_case_public_id="TC-0001",
                    test_case_name=case.name,
                    pinned_plans=[plan],
                    test_case_snapshot=case.model_dump(mode="json"),
                    status=SuiteRunItemStatus.RUNNING,
                )],
                status=SuiteRunStatus.RUNNING,
            ))
            attempt = SuiteRunAttempt(
                attempt_number=1,
                run_id=uuid4(),
                run_public_id="RUN-000001",
                run_status="FAILED",
                outcome="PRODUCT_FAILURE",
                started_at=first.created_at,
                finished_at=first.created_at,
                duration_ms=0,
            )
            repository.save(first.model_copy(update={
                "items": [first.items[0].model_copy(update={"attempts": [attempt]})]
            }))

            reopened = SQLiteSuiteRunRepository(database)
            persisted = reopened.get_by_public_id("SUITE-RUN-000001")
            self.assertEqual(persisted.items[0].attempts[0].run_public_id, "RUN-000001")
            self.assertEqual(persisted.items[0].pinned_plans[0].version_number, 3)
            self.assertEqual(reopened.interrupt_incomplete(), 1)
            interrupted = SQLiteSuiteRunRepository(database).get(first.id)
            self.assertEqual(interrupted.status, SuiteRunStatus.INTERRUPTED)
            self.assertEqual(interrupted.items[0].status, SuiteRunItemStatus.INTERRUPTED)
            self.assertEqual(len(interrupted.items[0].attempts), 1)

            next_run = reopened.save(SuiteRun(suite_id=uuid4(), suite_name="Next", config=SuiteRunConfig()))
            self.assertEqual(next_run.public_id, "SUITE-RUN-000002")
            report = json.loads(suite_run_report_json(interrupted))
            self.assertNotIn("test_case_snapshot", report["items"][0])

    def test_suite_page_links_each_retry_to_its_own_run_without_evidence_gallery(self):
        harness = _WebSuiteHarness()
        try:
            first_run_id, second_run_id = uuid4(), uuid4()
            started_at = datetime.now(timezone.utc)
            run = SuiteRun(
                suite_id=harness.suite.id,
                suite_name=harness.suite.name,
                config=SuiteRunConfig(retry_count=1),
                status=SuiteRunStatus.COMPLETED,
                items=[SuiteRunItem(
                    order_index=0,
                    test_case_id=harness.case.id,
                    test_case_public_id=harness.case.public_id,
                    test_case_name=harness.case.name,
                    status=SuiteRunItemStatus.PASSED_AFTER_RETRY,
                    classification="PASSED",
                    duration_ms=2400,
                    attempts=[
                        SuiteRunAttempt(
                            attempt_number=1,
                            run_id=first_run_id,
                            run_public_id="RUN-000101",
                            run_status="FAILED",
                            outcome="PRODUCT_FAILURE",
                            started_at=started_at,
                            finished_at=started_at,
                            duration_ms=1100,
                            failure_classifications=["PRODUCT_FAILURE"],
                        ),
                        SuiteRunAttempt(
                            attempt_number=2,
                            run_id=second_run_id,
                            run_public_id="RUN-000102",
                            run_status="PASSED",
                            outcome="PASSED",
                            started_at=started_at,
                            finished_at=started_at,
                            duration_ms=1200,
                        ),
                    ],
                )],
            )
            stored = harness.suite_run_service._repository.save(run)

            response = harness.app.handle("GET", f"/suite-runs/{stored.public_id}")
            html = response.body.decode("utf-8")

            self.assertEqual(response.status, 200)
            self.assertIn("TC-0001", html)
            self.assertIn("PASSED AFTER RETRY", html)
            self.assertIn("2.4 s", html)
            self.assertIn("PRODUCT_FAILURE", html)
            self.assertIn("Attempt 1", html)
            self.assertIn("Attempt 2", html)
            self.assertIn(f'href="/runs/{first_run_id}"', html)
            self.assertIn(f'href="/runs/{second_run_id}"', html)
            self.assertNotIn("<img", html)
            self.assertNotIn("Evidence</h4>", html)
        finally:
            harness.app.close()

    def test_web_preflight_lists_blockers_and_suite_run_is_reloadable(self):
        harness = _WebSuiteHarness()
        try:
            config = harness.app.handle("GET", f"/test-suites/{harness.suite.id}/run")
            config_html = config.body.decode("utf-8")
            self.assertEqual(config.status, 200)
            self.assertIn("Pinned plan versions at start", config_html)
            self.assertIn("Sequential", config_html)
            self.assertIn("AI policy", config_html)
            self.assertIn("Auto handle cookie consent", config_html)
            self.assertIn("Leave cookie consent unchanged", config_html)
            self.assertIn("Every verification", config_html)
            self.assertIn("Element + Page", config_html)

            invalid = harness.app.handle("POST", f"/test-suites/{harness.suite.id}/run", urlencode({
                "workflow": "REGRESSION",
                "execution_type": "SEQUENTIAL",
                "ai_policy": "DISABLED",
                "retry_count": "3",
            }))
            self.assertEqual(invalid.status, 400)

            valid = harness.app.handle("POST", f"/test-suites/{harness.suite.id}/run", urlencode({
                "workflow": "REGRESSION",
                "execution_type": "SEQUENTIAL",
                "ai_policy": "DISABLED",
                "retry_count": "0",
                "evidence_mode": "EVERY_VERIFICATION",
                "screenshot_mode": "ELEMENT_AND_PAGE",
                "cookie_policy": CookieConsentPolicy.LEAVE_UNCHANGED.value,
            }))
            self.assertEqual(valid.status, 303)
            public_id = valid.headers["Location"].rsplit("/", 1)[1]
            harness.wait_for_terminal(harness.suite_run_service, public_id)
            page = harness.app.handle("GET", valid.headers["Location"])
            self.assertEqual(page.status, 200)
            self.assertIn("Download JSON report", page.body.decode("utf-8"))
            page_html = page.body.decode("utf-8")
            self.assertIn("Every verification", page_html)
            self.assertNotIn("<img", page_html)
            progress = harness.app.handle("GET", f"/api/suite-runs/{public_id}")
            payload = json.loads(progress.body)
            self.assertEqual(payload["public_id"], public_id)
            self.assertEqual(payload["evidence_policy"], {
                "mode": "EVERY_VERIFICATION",
                "screenshot_mode": "ELEMENT_AND_PAGE",
            })
            self.assertEqual(payload["cookie_policy"], CookieConsentPolicy.LEAVE_UNCHANGED.value)
            self.assertEqual(
                harness.history.list_for_test_case(harness.case.id)[0].cookie_consent.status,
                CookieConsentStatus.LEFT_UNCHANGED,
            )
            self.assertNotIn("test_case_snapshot", progress.body.decode("utf-8"))
            report = harness.app.handle("GET", f"/suite-runs/{public_id}/report.json")
            self.assertEqual(report.content_type, "application/json; charset=utf-8")
            self.assertNotIn(b"test_case_snapshot", report.body)
            suite_payload = json.loads(report.body)
            self.assertEqual(suite_payload["config"]["evidence_policy"]["mode"], "EVERY_VERIFICATION")
            self.assertEqual(
                suite_payload["config"]["cookie_policy"],
                CookieConsentPolicy.LEAVE_UNCHANGED.value,
            )
            self.assertEqual(
                suite_payload["items"][0]["attempts"][0]["run_id"],
                str(harness.history.list_for_test_case(harness.case.id)[0].run_id),
            )
            self.assertNotIn("SCREENSHOT", report.body.decode("utf-8"))
            self.assertIn(
                f'href="/runs/{suite_payload["items"][0]["attempts"][0]["run_id"]}"',
                page_html,
            )
        finally:
            harness.app.close()


class _WebSuiteHarness(SuiteRunTestHarness):
    def __init__(self):
        self.setUp()
        self.execution_repo = InMemoryExecutionRepository()
        self.history_repo = InMemoryRunHistoryRepository()
        self.history = RunHistoryService(self.history_repo, self.execution_repo, plan_store=self.plans)

        def runner(_plan):
            return {"status": "passed", "steps": [{"action": "assert_title", "status": "passed"}]}

        self.run_service = TestCaseExecutionService(
            self.cases,
            self.plans,
            self.execution_repo,
            self.history,
            runner_factory=lambda _directory: runner,
        )
        self.suite_run_service = SuiteRunService(
            self.suites,
            self.cases,
            self.run_service,
            self.history,
            InMemorySuiteRunRepository(),
            max_workers=1,
            max_pending=0,
        )
        from qa_agent.execution_progress import ExecutionProgressStore
        self.app = LocalWebApplication(
            self.history,
            test_cases=self.cases,
            run_service=self.run_service,
            progress_store=ExecutionProgressStore(),
            plan_store=self.plans,
            test_suites=self.suites,
            suite_run_service=self.suite_run_service,
        )
