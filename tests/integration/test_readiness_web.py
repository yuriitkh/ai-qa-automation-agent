from __future__ import annotations

from pathlib import Path
import tempfile
from time import monotonic, sleep
import unittest
from unittest.mock import MagicMock
from urllib.parse import urlencode

from qa_agent.automation_lifecycle import (
    AutomationLifecycleService,
    InMemoryAutomationLifecycleRepository,
)
from qa_agent.cookie_consent import CookieConsentPolicy
from qa_agent.evidence_policy import EvidenceMode, EvidencePolicy, ScreenshotMode
from qa_agent.models import (
    QATestPlan, QATestStep, TestCase as DomainTestCase, TestStep as DomainTestStep,
    TestPlan as DomainTestPlan, TestPlanVersion as DomainTestPlanVersion,
)
from qa_agent.readiness import (
    ReadinessCheck,
    ReadinessStatus,
    SystemReadinessService,
)
from qa_agent.run_history import WorkflowType
from qa_agent.storage import create_sqlite_storage
from qa_agent.suite_runs import InMemorySuiteRunRepository, SuiteRunService
from qa_agent.test_case_execution import TestCaseExecutionService
from qa_agent.test_case_review import InMemoryTestCaseReviewRepository, TestCaseReviewService as ReviewService
from qa_agent.test_suites import InMemoryTestSuiteRepository, TestSuiteService as DomainSuiteService
from qa_agent.web import LocalWebApplication


class _Queue:
    def readiness_state(self):
        return {"closed": False, "available": True, "active": 0, "capacity": 4}


class _Providers:
    def __init__(self):
        self.connection_calls = 0
        self.capability_calls = 0

    def provider_views(self):
        return []

    def test_connection(self, *_args, **_kwargs):
        self.connection_calls += 1
        raise AssertionError("No live provider connection test is allowed")

    def test_authoring_capability(self, *_args, **_kwargs):
        self.capability_calls += 1
        raise AssertionError("No live provider capability test is allowed")


class SystemReadinessWebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.storage = create_sqlite_storage(self.root / "web-readiness.sqlite3")
        self.case = DomainTestCase(
            name="Local readiness case",
            description="Verify one local page.",
            base_url="http://127.0.0.1:1/local",
            steps=[DomainTestStep(
                name="Check page title", description="Open the local page.",
                expected="The page title is available.", order=0,
            )],
        )
        self.storage.test_case_repository.save(self.case)
        self.review = ReviewService(
            InMemoryTestCaseReviewRepository(), self.storage.plan_store
        )
        self.review.mark_ready_for_review(self.case)
        self.review.approve_test_case(self.case)
        self.lifecycle = AutomationLifecycleService(
            InMemoryAutomationLifecycleRepository(), self.storage.plan_store
        )
        self.execution = TestCaseExecutionService(
            self.storage.test_case_repository,
            self.storage.plan_store,
            self.storage.execution_repository,
            self.storage.run_history,
            evidence_directory=self.root / "evidence",
            automation_workflow=object(),
            automation_lifecycle=self.lifecycle,
            test_case_review=self.review,
        )
        self.providers = _Providers()
        self.target_urls: list[str] = []
        self.browser_depths: list[bool] = []
        self.evidence_probes: list[bool] = []

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_readiness(self, *, blocked_database: bool = False) -> SystemReadinessService:
        database = ReadinessCheck(
            "database", "Database",
            ReadinessStatus.BLOCKED if blocked_database else ReadinessStatus.READY,
            "<script>local failure</script>" if blocked_database else "Database is ready.",
            "Review local storage." if blocked_database else None,
            blocked_database,
        )
        def browser_probe(deep: bool):
            self.browser_depths.append(deep)
            return ReadinessCheck("browser", "Browser", ReadinessStatus.READY, "Chromium is available.")
        def evidence_probe(_path, probe: bool):
            self.evidence_probes.append(probe)
            return ReadinessCheck("evidence", "Evidence storage", ReadinessStatus.READY, "Evidence is writable.")
        def target_probe(url: str):
            self.target_urls.append(url)
            return ReadinessCheck("target", "Target", ReadinessStatus.READY, "The local target responded.")
        return SystemReadinessService(
            database_path=self.storage.database_path,
            evidence_directory=self.root / "evidence",
            application_ready=True,
            run_service=self.execution,
            background_runs=_Queue(),
            suite_run_service=_Queue(),
            provider_settings=self.providers,
            test_cases=self.storage.test_case_repository,
            automation_lifecycle=self.lifecycle,
            test_case_review=self.review,
            database_probe=lambda _path: database,
            browser_probe=browser_probe,
            evidence_probe=evidence_probe,
            target_probe=target_probe,
        )

    def make_app(self, readiness: SystemReadinessService) -> LocalWebApplication:
        return LocalWebApplication(
            self.storage.run_history,
            evidence_root=self.root / "evidence",
            test_cases=self.storage.test_case_repository,
            run_service=self.execution,
            plan_store=self.storage.plan_store,
            test_case_review=self.review,
            readiness_service=readiness,
        )

    def make_suite_app(self, *, blocked_database=False):
        self.case = DomainTestCase(
            name="Approved Suite case", description="Verify a saved title assertion.",
            base_url="http://127.0.0.1:1/local",
            steps=[DomainTestStep(name="Check title", description="Verify the title.",
                                  expected='The page title is "Welcome".', order=0)],
        )
        self.storage.test_case_repository.save(self.case)
        plan = DomainTestPlan(test_step_id=self.case.steps[0].id, name="Saved title")
        version = DomainTestPlanVersion(test_plan_id=plan.id, version=1, qa_test_plan=QATestPlan(
            url=self.case.base_url,
            steps=[QATestStep(action="assert_title", parameters={"expected": "Welcome"})],
        ))
        self.storage.plan_store.save(self.case.steps[0].id, version, test_plan=plan)
        self.review.approve_test_case(self.case)
        self.review.approve_for_validation(self.case)
        self.lifecycle.mark_automation_completed(self.case)
        self.execution._runner_factory = lambda _directory: lambda _plan: {
            "status": "passed", "steps": [{"action": "assert_title", "status": "passed"}],
        }
        suites = DomainSuiteService(InMemoryTestSuiteRepository(), self.storage.test_case_repository)
        self.suite = suites.create("Reviewed local suite", "")
        suites.add_member(self.suite.id, self.case.id)
        self.suite_runs = SuiteRunService(suites, self.storage.test_case_repository, self.execution,
                                         self.storage.run_history, InMemorySuiteRunRepository())
        readiness = self.make_readiness(blocked_database=blocked_database)
        readiness.suite_run_service = self.suite_runs
        app = LocalWebApplication(
            self.storage.run_history, evidence_root=self.root / "evidence",
            test_cases=self.storage.test_case_repository, run_service=self.execution,
            plan_store=self.storage.plan_store, test_case_review=self.review,
            test_suites=suites, suite_run_service=self.suite_runs, readiness_service=readiness,
        )
        return app, readiness

    @staticmethod
    def suite_form(**overrides):
        fields = {"workflow": "REGRESSION", "execution_type": "SEQUENTIAL", "ai_policy": "DISABLED",
                  "retry_count": "1", "evidence_mode": "EVERY_VERIFICATION",
                  "screenshot_mode": "ELEMENT_AND_PAGE", "cookie_policy": "LEAVE_UNCHANGED"}
        fields.update(overrides)
        return urlencode(fields)

    def test_system_health_page_and_legacy_health_endpoint(self) -> None:
        app = self.make_app(self.make_readiness(blocked_database=True))
        try:
            legacy = app.handle("GET", "/health")
            self.assertEqual(legacy.status, 200)
            self.assertEqual(legacy.body, b'{"status":"ok","service":"ai-qa-agent"}')

            page = app.handle("GET", "/system/health")
            self.assertEqual(page.status, 200)
            self.assertIn(b"System Health", page.body)
            self.assertIn(b"Database", page.body)
            self.assertIn(b"BLOCKED", page.body)
            self.assertIn(b"&lt;script&gt;local failure&lt;/script&gt;", page.body)
            self.assertNotIn(b"<script>local failure</script>", page.body)
            self.assertIn(b"Refresh checks", page.body)
            self.assertEqual(self.target_urls, [])
        finally:
            app.close()

    def test_refresh_is_explicit_local_only_and_does_not_probe_provider(self) -> None:
        readiness = self.make_readiness()
        app = self.make_app(readiness)
        try:
            refreshed = app.handle("POST", "/system/health/refresh", b"")
            self.assertEqual(refreshed.status, 200)
            self.assertEqual(self.browser_depths, [True])
            self.assertEqual(self.evidence_probes, [True])
            self.assertEqual(self.target_urls, [])
            self.assertEqual(self.providers.connection_calls, 0)
            self.assertEqual(self.providers.capability_calls, 0)
        finally:
            app.close()

    def test_server_rejects_blocked_run_before_history_or_product_result(self) -> None:
        app = self.make_app(self.make_readiness(blocked_database=True))
        original = self.case.model_dump(mode="json")
        try:
            response = app.handle(
                "POST", f"/test-cases/{self.case.id}/run",
                "workflow=AUTOMATION&evidence_mode=FAILURES_ONLY&screenshot_mode=PAGE&cookie_policy=AUTO_HANDLE",
            )
            self.assertEqual(response.status, 409)
            self.assertIn(b"Database", response.body)
            self.assertIn(b"local failure", response.body)
            self.assertEqual(self.storage.run_history.list_for_test_case(self.case.id), [])
            self.assertEqual(self.case.model_dump(mode="json"), original)
            self.assertEqual(self.target_urls, [self.case.base_url])
        finally:
            app.close()

    def test_explicit_case_readiness_shows_target_and_provider_policy(self) -> None:
        app = self.make_app(self.make_readiness())
        try:
            response = app.handle(
                "POST", f"/test-cases/{self.case.id}/readiness",
                "workflow=REGRESSION&evidence_mode=FAILURES_ONLY&screenshot_mode=PAGE&cookie_policy=LEAVE_UNCHANGED",
            )
            self.assertEqual(response.status, 409)
            self.assertIn(b"Target", response.body)
            self.assertIn(b"remain unchanged", response.body)
            self.assertIn(b"no provider check was made", response.body)
            self.assertEqual(self.providers.connection_calls, 0)
            self.assertEqual(self.providers.capability_calls, 0)
            self.assertEqual(self.target_urls, [self.case.base_url])
        finally:
            app.close()

    def test_suite_readiness_continuation_preserves_policy_and_starts_without_provider(self):
        app, _readiness = self.make_suite_app()
        self.providers.provider_views = MagicMock(side_effect=AssertionError("No provider access for Regression"))
        try:
            before = self.storage.llm_usage_repository.list_records()
            preview = self.suite_runs.preview(self.suite.id)
            response = app.handle("POST", f"/test-suites/{self.suite.id}/readiness", self.suite_form())
            self.assertEqual(response.status, 200)
            self.assertIn(b"Continue to start", response.body)
            self.assertNotIn(b"run-setup", response.body)
            self.assertIn(b'value="EVERY_VERIFICATION"', response.body)
            self.assertIn(b'value="ELEMENT_AND_PAGE"', response.body)
            self.assertIn(b'value="LEAVE_UNCHANGED"', response.body)
            self.assertEqual(preview.eligibility[0].pinned_plans,
                             self.suite_runs.preview(self.suite.id).eligibility[0].pinned_plans)
            self.assertEqual(self.storage.run_history.list_for_test_case(self.case.id), [])
            self.assertEqual(self.storage.llm_usage_repository.list_records(), before)
            started = app.handle("POST", f"/test-suites/{self.suite.id}/run", self.suite_form())
            self.assertEqual(started.status, 303)
            public_id = started.headers["Location"].rsplit("/", 1)[1]
            deadline = monotonic() + 3
            while monotonic() < deadline:
                run = self.suite_runs.get(public_id)
                if run.finished_at:
                    break
                sleep(0.01)
            self.assertIsNotNone(run.finished_at)
            self.assertEqual(run.status.value, "COMPLETED")
            self.assertEqual(run.config.evidence_policy, EvidencePolicy(
                mode=EvidenceMode.EVERY_VERIFICATION, screenshot_mode=ScreenshotMode.ELEMENT_AND_PAGE))
            child = self.storage.run_history.list_for_test_case(self.case.id)[0]
            self.assertEqual(child.evidence_policy, run.config.evidence_policy)
            self.assertEqual(child.cookie_consent.policy, CookieConsentPolicy.LEAVE_UNCHANGED)
            self.assertEqual(self.storage.llm_usage_repository.list_records(), before)
            self.providers.provider_views.assert_not_called()
            self.assertEqual(self.providers.connection_calls, 0)
            self.assertEqual(self.providers.capability_calls, 0)
        finally:
            app.close()

    def test_suite_failure_blocks_creation_and_is_never_product_failure(self):
        app, readiness = self.make_suite_app(blocked_database=True)
        try:
            response = app.handle("POST", f"/test-suites/{self.suite.id}/run", self.suite_form())
            self.assertEqual(response.status, 409)
            self.assertEqual(self.suite_runs.repository.list_recent(), [])
            self.assertEqual(self.storage.run_history.list_for_test_case(self.case.id), [])
            readiness.aggregate_timeout_seconds = 0.04
            readiness._target_probe = lambda _url: sleep(0.15)
            started = monotonic()
            timed_out = app.handle("POST", f"/test-suites/{self.suite.id}/run", self.suite_form())
            self.assertLess(monotonic() - started, 0.20)
            self.assertEqual(timed_out.status, 409)
            self.assertIn(b"aggregate timeout", timed_out.body)
            self.assertNotIn(b"PRODUCT_FAILURE", timed_out.body)
            self.assertEqual(self.suite_runs.repository.list_recent(), [])
        finally:
            sleep(0.16)  # Finish the injected diagnostic before disposing its local fixtures.
            app.close()

    def test_suite_readiness_rejects_invalid_config_before_probing_targets(self):
        app, _readiness = self.make_suite_app()
        try:
            for overrides in ({"ai_policy": "INVALID"}, {"retry_count": "3"},
                              {"workflow": "AUTOMATION"}, {"execution_type": "PARALLEL"}):
                with self.subTest(overrides=overrides):
                    response = app.handle("POST", f"/test-suites/{self.suite.id}/readiness",
                                          self.suite_form(**overrides))
                    self.assertEqual(response.status, 400)
            self.assertEqual(self.target_urls, [])
            self.assertEqual(self.suite_runs.repository.list_recent(), [])
        finally:
            app.close()


if __name__ == "__main__":
    unittest.main()
