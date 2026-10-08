from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import tempfile
from pathlib import Path
from threading import BoundedSemaphore, Event, Thread
from time import monotonic, sleep
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from uuid import uuid4

from qa_agent.automation_lifecycle import (
    AutomationLifecycleService,
    AutomationStatus,
    InMemoryAutomationLifecycleRepository,
)
from qa_agent.cookie_consent import CookieConsentPolicy
from qa_agent.evidence_policy import DEFAULT_EVIDENCE_POLICY, EvidencePolicy
from qa_agent.models import (
    QATestPlan,
    QATestStep,
    TestCase as DomainTestCase,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
)
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.readiness import (
    ReadinessCheck,
    ReadinessReport,
    ReadinessStatus,
    SystemReadinessService,
    _check_browser,
    _check_database,
    _check_evidence,
    _allowed_target_address,
    _test_case_targets,
    _bounded_call,
    check_target_url,
)
from qa_agent.run_history import WorkflowType
from qa_agent.suite_runs import (
    PinnedSuitePlan,
    SuiteEligibility,
    SuiteStartResult,
)
from qa_agent.test_case_review import (
    InMemoryTestCaseReviewRepository,
    TestCaseReviewService as ReviewService,
)
from qa_agent.test_case_execution import TestCaseExecutionService
from qa_agent.storage import create_sqlite_storage
from qa_agent.provider_settings import ProviderSettingsRepository
from qa_agent.test_suites import SQLiteTestSuiteRepository, TestSuite as DomainTestSuite
from qa_agent.suite_run_storage import SQLiteSuiteRunRepository


class _Queue:
    def __init__(self, *, available: bool = True, closed: bool = False) -> None:
        self.available = available
        self.closed = closed

    def readiness_state(self):
        return {"available": self.available, "closed": self.closed, "active": 0, "capacity": 2}


class _Providers:
    def __init__(self, configured: bool) -> None:
        self.configured = configured
        self.views_calls = 0
        self.connection_tests = 0
        self.capability_tests = 0

    def provider_views(self):
        self.views_calls += 1
        return [SimpleNamespace(enabled=True, status="CONFIGURED")] if self.configured else []

    def test_connection(self, *_args, **_kwargs):
        self.connection_tests += 1
        raise AssertionError("Readiness must not run a provider connection test")

    def test_authoring_capability(self, *_args, **_kwargs):
        self.capability_tests += 1
        raise AssertionError("Readiness must not run a provider capability test")


class _LocalTarget(BaseHTTPRequestHandler):
    writes = 0
    head_status = 200
    head_delay = 0.0

    def do_HEAD(self):
        from time import sleep
        if self.head_delay:
            sleep(self.head_delay)
        self.send_response(self.head_status)
        self.end_headers()

    def do_GET(self):
        type(self).writes += 1
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        type(self).writes += 1
        self.send_response(200)
        self.end_headers()

    def log_message(self, *_args):
        pass


class ReadinessTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        _LocalTarget.writes = 0
        _LocalTarget.head_delay = 0.0
        _LocalTarget.head_status = 200
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _LocalTarget)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/check?mode=safe"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_local_target_responds_without_mutating_it(self) -> None:
        result = check_target_url(self.url)
        self.assertEqual(result.status, ReadinessStatus.READY)
        self.assertEqual(_LocalTarget.writes, 0)

    def test_http_4xx_is_a_nonblocking_warning_not_a_product_result(self) -> None:
        _LocalTarget.head_status = 404
        result = check_target_url(self.url)
        self.assertEqual(result.status, ReadinessStatus.WARNING)
        self.assertFalse(result.blocking)
        self.assertIn("does not judge application behavior", result.safe_message)
        self.assertEqual(_LocalTarget.writes, 0)

    def test_unreachable_loopback_target_is_blocked(self) -> None:
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        try:
            result = check_target_url(f"http://127.0.0.1:{port}/")
        finally:
            probe.close()
        self.assertEqual(result.status, ReadinessStatus.BLOCKED)
        self.assertTrue(
            "Cannot connect" in result.safe_message
            or "readiness timeout" in result.safe_message
        )

    def test_timeout_is_bounded_and_blocked(self) -> None:
        _LocalTarget.head_delay = 0.25
        result = check_target_url(self.url, timeout_seconds=0.05)
        self.assertEqual(result.status, ReadinessStatus.BLOCKED)
        self.assertLess(result.elapsed_ms or 0, 800)

    def test_malformed_unsupported_and_credential_urls_are_rejected_without_network(self) -> None:
        for value in (
            "not-a-url",
            "file:///tmp/test",
            "http://user:password@127.0.0.1/",
            "http://127.0.0.1/?access_token=private",
        ):
            with self.subTest(value=value):
                result = check_target_url(value)
                self.assertEqual(result.status, ReadinessStatus.BLOCKED)
                self.assertNotIn("password", result.safe_message.casefold())

    def test_private_non_loopback_address_is_rejected(self) -> None:
        result = check_target_url("http://169.254.169.254/latest/meta-data/")
        self.assertEqual(result.status, ReadinessStatus.BLOCKED)
        self.assertIn("restricted", result.safe_message)

    def test_unsafe_urls_and_addresses_never_connect(self) -> None:
        with patch("qa_agent.readiness.socket.getaddrinfo") as dns, patch(
            "qa_agent.readiness.socket.create_connection"
        ) as connect:
            for value in (
                "http://127.0.0.1:0/", "http://127.0.0.1:65536/",
                "http://127.0.0.1/\r\nInjected: value", "http://127.0.0.1\\@host/",
                "http://[fe80::1%25eth0]/", "http://224.0.0.1/", "http://[ff02::1]/",
                "http://10.0.0.1/", "http://192.168.1.1/", "http://[fc00::1]/",
                "http://0.0.0.0/", "http://127.0.0.1/?%61ccess_token=secret",
            ):
                with self.subTest(value=value):
                    self.assertTrue(check_target_url(value).blocking)
            dns.assert_not_called()
            connect.assert_not_called()

    def test_public_hostname_with_any_private_answer_is_blocked(self) -> None:
        answers = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 80))
            for address in ("93.184.216.34", "127.0.0.1")
        ]
        with patch("qa_agent.readiness.socket.getaddrinfo", return_value=answers), patch(
            "qa_agent.readiness.socket.create_connection"
        ) as connect:
            result = check_target_url("http://public.example/")
        self.assertTrue(result.blocking)
        connect.assert_not_called()

    def test_public_dns_is_pinned_and_redirects_are_not_followed(self) -> None:
        answers = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]
        response = MagicMock(status=302)
        connection = MagicMock()
        connection.getresponse.return_value = response
        with patch("qa_agent.readiness.socket.getaddrinfo", return_value=answers) as dns, patch(
            "qa_agent.readiness._PinnedHTTPConnection", return_value=connection
        ) as factory:
            result = check_target_url("http://public.example/check")
        self.assertEqual(result.status, ReadinessStatus.READY)
        dns.assert_called_once()
        self.assertEqual(factory.call_args.args[:3], ("public.example", "93.184.216.34", 80))
        self.assertEqual(connection.request.call_args.args, ("HEAD", "/check"))
        self.assertNotIn("Cookie", connection.request.call_args.kwargs["headers"])
        self.assertNotIn("Authorization", connection.request.call_args.kwargs["headers"])
        connection.request.assert_called_once()
        connection.close.assert_called_once()

    def test_dns_and_multiple_address_attempts_share_one_timeout(self) -> None:
        def dns(*_args, **_kwargs):
            sleep(0.035)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 80))
                    for address in ("93.184.216.34", "93.184.216.35", "93.184.216.36")]
        def fail(*_args, **_kwargs):
            sleep(0.04)
            raise socket.timeout()
        connection = MagicMock()
        connection.request.side_effect = fail
        started = monotonic()
        with patch("qa_agent.readiness.socket.getaddrinfo", side_effect=dns), patch(
            "qa_agent.readiness._PinnedHTTPConnection", return_value=connection
        ):
            result = check_target_url("http://public.example/", timeout_seconds=0.09)
        self.assertTrue(result.blocking)
        self.assertLess(monotonic() - started, 0.22)
        self.assertLessEqual(connection.request.call_count, 2)

    def test_protocol_error_is_safe_and_blocking(self) -> None:
        import http.client
        connection = MagicMock()
        connection.getresponse.side_effect = http.client.BadStatusLine("private server detail")
        with patch("qa_agent.readiness._PinnedHTTPConnection", return_value=connection):
            result = check_target_url(self.url)
        self.assertTrue(result.blocking)
        self.assertNotIn("private server detail", result.safe_message)

    def test_explicit_ipv6_loopback_is_allowed(self) -> None:
        self.assertTrue(_allowed_target_address("::1", allow_loopback=True))
        self.assertFalse(_allowed_target_address("::1", allow_loopback=False))
        self.assertFalse(_allowed_target_address("2002:7f00:1::", allow_loopback=False))


class ReadinessFixtureMixin:
    def make_service(self, **overrides) -> SystemReadinessService:
        ready = lambda check_id, name: ReadinessCheck(
            check_id, name, ReadinessStatus.READY, "ready"
        )
        run_service = SimpleNamespace(
            _automation_workflow=object(),
            workflow_availability_for_test_case=lambda _case: SimpleNamespace(
                regression_available=True, validation_available=True, reason=None
            ),
            _plan_store=None,
        )
        options = {
            "database_path": None,
            "evidence_directory": None,
            "application_ready": True,
            "run_service": run_service,
            "background_runs": _Queue(),
            "suite_run_service": _Queue(),
            "provider_settings": None,
            "database_probe": lambda _path: ready("database", "Database"),
            "browser_probe": lambda _deep: ready("browser", "Browser"),
            "evidence_probe": lambda _path, _probe: ready("evidence", "Evidence storage"),
            "target_probe": lambda _url: ready("target", "Target"),
        }
        options.update(overrides)
        return SystemReadinessService(**options)

    @staticmethod
    def make_case(url: str = "http://127.0.0.1:8765/test") -> DomainTestCase:
        return DomainTestCase(
            name="Check local welcome",
            description="Verify the welcome text is displayed.",
            base_url=url,
            steps=[DomainTestStep(
                name="Verify welcome text",
                description="Open the local page and inspect the welcome text.",
                expected='The text "Welcome" is visible.',
                order=0,
            )],
        )


class ReadinessProbeTests(unittest.TestCase, ReadinessFixtureMixin):
    def test_unavailable_checks_are_not_reported_as_ready(self) -> None:
        report = ReadinessReport((ReadinessCheck(
            "application", "Application", ReadinessStatus.NOT_APPLICABLE, "No diagnostics configured.",
        ),))
        self.assertEqual(report.status, ReadinessStatus.NOT_APPLICABLE)

    def test_refresh_invalidates_the_cached_browser_snapshot(self) -> None:
        probe = MagicMock(side_effect=[
            ReadinessCheck("browser", "Browser", ReadinessStatus.BLOCKED, "missing", blocking=True),
            ReadinessCheck("browser", "Browser", ReadinessStatus.READY, "installed"),
            ReadinessCheck("browser", "Browser", ReadinessStatus.READY, "installed"),
        ])
        service = self.make_service(browser_probe=probe)
        self.assertTrue(service.system_report().blocked)
        self.assertFalse(service.system_report(deep_browser=True).blocked)
        self.assertFalse(service.system_report().blocked)
        self.assertEqual(probe.call_count, 3)

    def test_timed_out_work_retains_capacity_until_it_finishes(self) -> None:
        slots = BoundedSemaphore(1)
        release = Event()
        finished = Event()
        def stalled():
            try:
                release.wait(1)
            finally:
                finished.set()
        try:
            with self.assertRaises(TimeoutError):
                _bounded_call(stalled, 0.02, slots)
            next_probe = MagicMock()
            with self.assertRaises(TimeoutError):
                _bounded_call(next_probe, 0.02, slots)
            next_probe.assert_not_called()
        finally:
            release.set()
            self.assertTrue(finished.wait(1))

    def test_healthy_read_only_sqlite_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ready.sqlite3"
            storage = create_sqlite_storage(path)
            ProviderSettingsRepository(path)
            SQLiteTestSuiteRepository(path)
            SQLiteSuiteRunRepository(path)
            result = _check_database(path)
            self.assertEqual(result.status, ReadinessStatus.READY)
            storage.run_history.list_recent(1)

    def test_missing_or_incomplete_database_is_blocked_without_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.sqlite3"
            result = _check_database(missing)
            self.assertEqual(result.status, ReadinessStatus.BLOCKED)
            self.assertFalse(missing.exists())
            empty = Path(directory) / "empty.sqlite3"
            empty.touch()
            self.assertEqual(_check_database(empty).status, ReadinessStatus.BLOCKED)

    def test_browser_installed_and_missing_are_distinct(self) -> None:
        installed = _check_browser(False)
        self.assertIn(installed.status, {ReadinessStatus.READY, ReadinessStatus.BLOCKED})
        with patch("qa_agent.readiness.importlib.util.find_spec", return_value=None):
            missing = _check_browser(False)
        self.assertEqual(missing.status, ReadinessStatus.BLOCKED)
        self.assertIn("not installed", missing.safe_message)

    def test_browser_deep_probe_is_only_called_when_requested(self) -> None:
        calls = []
        service = self.make_service(browser_probe=lambda deep: calls.append(deep) or ReadinessCheck(
            "browser", "Browser", ReadinessStatus.READY, "ready"
        ))
        service.system_report()
        self.assertEqual(calls, [False])  # package snapshot does not launch Chromium
        service.system_report(deep_browser=True)
        self.assertEqual(calls, [False, True])

    def test_evidence_directory_can_be_created_and_probe_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "nested" / "evidence"
            result = _check_evidence(evidence, True)
            self.assertEqual(result.status, ReadinessStatus.READY)
            self.assertEqual(list(evidence.iterdir()), [])

    def test_evidence_file_path_is_reported_unwritable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "not-a-directory"
            file_path.write_text("local", encoding="utf-8")
            result = _check_evidence(file_path, True)
            self.assertEqual(result.status, ReadinessStatus.BLOCKED)
            self.assertEqual(file_path.read_text(encoding="utf-8"), "local")

    def test_background_worker_unavailable_is_blocking(self) -> None:
        report = self.make_service(background_runs=_Queue(available=False)).system_report(worker="case")
        background = next(check for check in report.checks if check.id == "background")
        self.assertEqual(background.status, ReadinessStatus.BLOCKED)
        self.assertTrue(background.blocking)

    def test_ai_provider_policy_is_conditional_and_never_probes(self) -> None:
        providers = _Providers(configured=False)
        case = self.make_case()
        service = self.make_service(provider_settings=providers)
        report = service.testcase_report(
            case, WorkflowType.REGRESSION,
            evidence_policy=DEFAULT_EVIDENCE_POLICY,
            cookie_policy=CookieConsentPolicy.LEAVE_UNCHANGED,
        )
        self.assertFalse(any(check.id == "provider_required" for check in report.checks))
        self.assertEqual(providers.connection_tests, 0)
        self.assertEqual(providers.capability_tests, 0)
        self.assertEqual(providers.views_calls, 0)

    def test_ai_required_without_provider_blocks_without_live_probe(self) -> None:
        providers = _Providers(configured=False)
        report = self.make_service(provider_settings=providers).testcase_report(
            self.make_case(), WorkflowType.AUTOMATION,
            evidence_policy=DEFAULT_EVIDENCE_POLICY,
            cookie_policy=CookieConsentPolicy.AUTO_HANDLE,
        )
        check = next(check for check in report.checks if check.id == "provider_required")
        self.assertEqual(check.status, ReadinessStatus.BLOCKED)
        self.assertEqual(providers.connection_tests, 0)
        self.assertEqual(providers.capability_tests, 0)

    def test_explicit_cookie_policy_is_reported_without_cookie_access(self) -> None:
        report = self.make_service().testcase_report(
            self.make_case(), WorkflowType.REGRESSION,
            evidence_policy=EvidencePolicy(),
            cookie_policy=CookieConsentPolicy.LEAVE_UNCHANGED,
        )
        policy = next(check for check in report.checks if check.id == "execution_policy")
        self.assertIn("remain unchanged", policy.safe_message)


class SuiteAggregateReadinessTests(unittest.TestCase, ReadinessFixtureMixin):
    def preview(self, count: int = 6, *, same_target: bool = False):
        cases = [self.make_case(f"http://127.0.0.1:8765/{0 if same_target else index}")
                 for index in range(count)]
        self.cases = {case.id: case for case in cases}
        return SuiteStartResult(
            suite=DomainTestSuite(uuid4(), "Bounded suite", "", datetime.now(timezone.utc), datetime.now(timezone.utc)),
            eligibility=[SuiteEligibility(
                order_index=index, test_case_id=case.id, test_case_name=case.name,
                eligible=True, test_case_snapshot=case.model_dump(mode="json"),
            ) for index, case in enumerate(cases)],
        )

    def report(self, service, preview):
        return service.suite_report(preview, evidence_policy=DEFAULT_EVIDENCE_POLICY,
                                    cookie_policy=CookieConsentPolicy.LEAVE_UNCHANGED)

    def test_many_distinct_targets_have_one_aggregate_deadline(self) -> None:
        preview = self.preview()
        calls = []
        def slow(url):
            calls.append(url)
            sleep(0.06)
            return ReadinessCheck("target", "Target", ReadinessStatus.READY, "ready")
        service = self.make_service(test_cases=SimpleNamespace(get=self.cases.get),
                                    target_probe=slow, aggregate_timeout_seconds=0.10)
        started = monotonic()
        report = self.report(service, preview)
        self.assertLess(monotonic() - started, 0.25)
        self.assertTrue(report.blocked)
        self.assertIn("aggregate timeout", report.blocked[0].safe_message)
        sleep(0.08)  # Allow the in-flight diagnostic to finish; no new target may start.
        self.assertLessEqual(len(calls), 2)

    def test_aggregate_deadline_includes_platform_checks(self) -> None:
        preview = self.preview()
        release = Event()
        finished = Event()
        def stalled_browser(_deep):
            release.wait(1)
            finished.set()
            return ReadinessCheck("browser", "Browser", ReadinessStatus.READY, "ready")
        targets = MagicMock()
        service = self.make_service(test_cases=SimpleNamespace(get=self.cases.get),
                                    browser_probe=stalled_browser, target_probe=targets,
                                    aggregate_timeout_seconds=0.05)
        try:
            started = monotonic()
            report = self.report(service, preview)
            self.assertTrue(report.blocked)
            self.assertLess(monotonic() - started, 0.20)
        finally:
            release.set()
            self.assertTrue(finished.wait(1))
        targets.assert_not_called()

    def test_duplicate_targets_probe_once_and_provider_is_not_read(self) -> None:
        preview = self.preview(same_target=True)
        target = MagicMock(return_value=ReadinessCheck("target", "Target", ReadinessStatus.READY, "ready"))
        providers = MagicMock()
        providers.provider_views.side_effect = AssertionError("Regression must be provider-independent")
        service = self.make_service(test_cases=SimpleNamespace(get=self.cases.get),
                                    provider_settings=providers, target_probe=target)
        report = self.report(service, preview)
        self.assertFalse(report.blocked)
        self.assertEqual(target.call_count, 1)
        self.assertEqual(len([check for check in report.checks if check.id.endswith("_target")]), 6)
        providers.provider_views.assert_not_called()

    def test_ineligible_member_without_reasons_and_suite_error_block(self) -> None:
        preview = self.preview(count=1)
        original = preview.eligibility[0]
        preview.eligibility[0] = original.model_copy(update={"eligible": False})
        service = self.make_service(test_cases=SimpleNamespace(get=self.cases.get))
        self.assertTrue(self.report(service, preview).blocked)
        preview.eligibility[0] = original
        preview = preview.model_copy(update={"suite_error": "Suite setup is unavailable."})
        self.assertTrue(self.report(service, preview).blocked)

    def test_suite_preview_is_inside_the_aggregate_deadline(self) -> None:
        preview = self.preview(count=1)
        def slow_preview():
            sleep(0.12)
            return preview
        service = self.make_service(test_cases=SimpleNamespace(get=self.cases.get),
                                    aggregate_timeout_seconds=0.04)
        started = monotonic()
        self.assertTrue(self.report(service, slow_preview).blocked)
        self.assertLess(monotonic() - started, 0.15)

    def test_probe_exception_blocks_with_safe_diagnostics(self) -> None:
        preview = self.preview(count=1)
        target = MagicMock(side_effect=RuntimeError("secret provider or target detail"))
        report = self.report(self.make_service(test_cases=SimpleNamespace(get=self.cases.get),
                                               target_probe=target), preview)
        self.assertTrue(report.blocked)
        self.assertNotIn("secret", report.blocked[0].safe_message)

class TestCaseReadinessTests(unittest.TestCase, ReadinessFixtureMixin):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.storage = create_sqlite_storage(self.root / "case.sqlite3")
        self.plans = self.storage.plan_store
        self.case = ReadinessProbeTests.make_case()
        self.storage.test_case_repository.save(self.case)
        step = self.case.steps[0]
        plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        version = DomainTestPlanVersion(
            test_plan_id=plan.id,
            version=1,
            qa_test_plan=QATestPlan(
                url=self.case.base_url,
                steps=[QATestStep(action="assert_text_contains", parameters={"expected_text": "Welcome"})],
            ),
        )
        self.plans.save(step.id, version, test_plan=plan)
        self.review = ReviewService(InMemoryTestCaseReviewRepository(), self.plans)
        self.review.mark_ready_for_review(self.case)
        self.review.approve_test_case(self.case)
        self.review.approve_for_validation(self.case)
        self.lifecycle = AutomationLifecycleService(
            InMemoryAutomationLifecycleRepository(), self.plans
        )
        self.lifecycle.mark_automation_completed(self.case)
        self.execution = TestCaseExecutionService(
            self.storage.test_case_repository,
            self.plans,
            self.storage.execution_repository,
            self.storage.run_history,
            evidence_directory=self.root / "evidence",
            runner_factory=lambda _directory: (lambda _plan: {"status": "passed"}),
            automation_workflow=object(),
            automation_lifecycle=self.lifecycle,
            test_case_review=self.review,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_case_service(self, **overrides) -> SystemReadinessService:
        options = {
            "evidence_directory": self.root / "evidence",
            "run_service": self.execution,
            "test_cases": self.storage.test_case_repository,
            "automation_lifecycle": self.lifecycle,
            "test_case_review": self.review,
        }
        options.update(overrides)
        return self.make_service(**options)

    def test_valid_saved_automation_and_approval_are_ready(self) -> None:
        report = self.make_case_service().testcase_report(
            self.case, WorkflowType.VALIDATION,
            evidence_policy=DEFAULT_EVIDENCE_POLICY,
            cookie_policy=CookieConsentPolicy.AUTO_HANDLE,
        )
        self.assertFalse(report.blocked)
        self.assertEqual(next(c for c in report.checks if c.id == "automation").status, ReadinessStatus.READY)

    def test_missing_plan_and_insufficient_coverage_are_blocked_for_validation(self) -> None:
        no_plan = self.make_case(url="http://127.0.0.1:8765/no-plan")
        self.storage.test_case_repository.save(no_plan)
        self.review.mark_ready_for_review(no_plan)
        self.review.approve_test_case(no_plan)
        blocked = self.make_case_service().testcase_report(
            no_plan, WorkflowType.VALIDATION,
            evidence_policy=DEFAULT_EVIDENCE_POLICY,
            cookie_policy=CookieConsentPolicy.AUTO_HANDLE,
        )
        self.assertTrue(any(c.id == "automation" and c.status == ReadinessStatus.BLOCKED for c in blocked.checks))

        version = self.plans.find(self.case.steps[0].id)
        self.plans.save(
            self.case.steps[0].id,
            version.model_copy(update={
                "id": __import__("uuid").uuid4(),
                "version": 2,
                "qa_test_plan": QATestPlan(url=self.case.base_url, steps=[
                    QATestStep(action="navigate", parameters={"url": self.case.base_url}),
                ]),
            }),
            test_plan=self.plans.find_test_plan(self.case.steps[0].id),
        )
        insufficient = self.make_case_service().testcase_report(
            self.case, WorkflowType.VALIDATION,
            evidence_policy=DEFAULT_EVIDENCE_POLICY,
            cookie_policy=CookieConsentPolicy.AUTO_HANDLE,
        )
        self.assertTrue(insufficient.blocked)

    def test_unapproved_case_blocks_and_regression_coverage_warning_is_nonblocking(self) -> None:
        self.review.mark_ready_for_review(self.case)
        unapproved = self.make_case_service().testcase_report(
            self.case, WorkflowType.REGRESSION,
            evidence_policy=DEFAULT_EVIDENCE_POLICY,
            cookie_policy=CookieConsentPolicy.AUTO_HANDLE,
        )
        self.assertTrue(any(c.id == "test_case_review" and c.blocking for c in unapproved.checks))

        # This plan has executable navigation but does not verify the expected welcome text.
        plan = self.plans.find_test_plan(self.case.steps[0].id)
        version = self.plans.find(self.case.steps[0].id)
        self.plans.save(
            self.case.steps[0].id,
            version.model_copy(update={
                "id": __import__("uuid").uuid4(),
                "version": version.version + 1,
                "qa_test_plan": QATestPlan(url=self.case.base_url, steps=[
                    QATestStep(action="navigate", parameters={"url": self.case.base_url}),
                ]),
            }),
            test_plan=plan,
        )
        self.review.approve_test_case(self.case)
        # Legacy Regression remains runnable with an explicit warning when coverage is incomplete.
        service = self.make_case_service(test_case_review=None)
        report = service.testcase_report(
            self.case, WorkflowType.REGRESSION,
            evidence_policy=DEFAULT_EVIDENCE_POLICY,
            cookie_policy=CookieConsentPolicy.LEAVE_UNCHANGED,
        )
        self.assertEqual(report.status, ReadinessStatus.WARNING)
        self.assertTrue(any(c.id == "expected_result_coverage" for c in report.checks))

    def test_readiness_does_not_mutate_testcase(self) -> None:
        before = self.case.model_dump(mode="json")
        self.make_case_service().testcase_report(
            self.case, WorkflowType.VALIDATION,
            evidence_policy=DEFAULT_EVIDENCE_POLICY,
            cookie_policy=CookieConsentPolicy.AUTO_HANDLE,
        )
        self.assertEqual(before, self.case.model_dump(mode="json"))

    def test_saved_navigation_targets_are_checked_even_with_a_safe_case_base_url(self) -> None:
        version = self.plans.find(self.case.steps[0].id)
        self.plans.save(self.case.steps[0].id, version.model_copy(update={
            "id": uuid4(), "version": version.version + 1,
            "qa_test_plan": QATestPlan(url="http://127.0.0.1:8765/saved", steps=[
                QATestStep(action="navigate", parameters={"url": "http://169.254.169.254/latest/meta-data/"}),
            ]),
        }), test_plan=self.plans.find_test_plan(self.case.steps[0].id))
        targets, missing = _test_case_targets(self.case, self.execution)
        self.assertEqual(missing, 0)
        self.assertIn("http://169.254.169.254/latest/meta-data/", targets)
        selected_targets, _ = _test_case_targets(
            self.case, self.execution, selected_versions={self.case.steps[0].id: version.id},
        )
        self.assertEqual(selected_targets, (self.case.base_url,))
        report = self.make_case_service(target_probe=check_target_url).testcase_report(
            self.case, WorkflowType.REGRESSION, evidence_policy=DEFAULT_EVIDENCE_POLICY,
            cookie_policy=CookieConsentPolicy.LEAVE_UNCHANGED,
        )
        self.assertTrue(any(check.id == "target" and "restricted" in check.safe_message for check in report.blocked))

    def test_suite_preflight_is_ready_and_preserves_exact_pins(self) -> None:
        version = self.plans.find(self.case.steps[0].id)
        pin = PinnedSuitePlan(
            test_step_id=self.case.steps[0].id,
            test_plan_version_id=version.id,
            version_number=version.version,
        )
        member = SuiteEligibility(
            order_index=0,
            test_case_id=self.case.id,
            test_case_public_id=self.case.public_id,
            test_case_name=self.case.name,
            eligible=True,
            pinned_plans=[pin],
            test_case_snapshot=self.case.model_dump(mode="json"),
        )
        now = datetime.now(timezone.utc)
        preview = SuiteStartResult(
            suite=DomainTestSuite(uuid4(), "Local suite", "", now, now), eligibility=[member]
        )
        before = tuple(member.pinned_plans)
        report = self.make_case_service().suite_report(
            preview,
            evidence_policy=DEFAULT_EVIDENCE_POLICY,
            cookie_policy=CookieConsentPolicy.LEAVE_UNCHANGED,
        )
        self.assertFalse(report.blocked)
        self.assertEqual(tuple(member.pinned_plans), before)

    def test_suite_lists_every_blocked_member_and_names_missing_automation(self) -> None:
        missing_a = self.make_case("http://127.0.0.1:8765/missing-a")
        missing_b = self.make_case("http://127.0.0.1:8765/missing-b")
        self.storage.test_case_repository.save(missing_a)
        self.storage.test_case_repository.save(missing_b)
        members = [
            SuiteEligibility(
                order_index=index,
                test_case_id=case.id,
                test_case_name=case.name,
                eligible=False,
                reasons=["No complete saved Regression plan is available."],
                test_case_snapshot=case.model_dump(mode="json"),
            )
            for index, case in enumerate((missing_a, missing_b))
        ]
        report = self.make_case_service().suite_report(
            SuiteStartResult(
                suite=DomainTestSuite(uuid4(), "Incomplete suite", "", datetime.now(timezone.utc), datetime.now(timezone.utc)),
                eligibility=members,
            ),
            evidence_policy=DEFAULT_EVIDENCE_POLICY,
            cookie_policy=CookieConsentPolicy.AUTO_HANDLE,
        )
        member_checks = [check for check in report.checks if check.id.endswith("_automation")]
        self.assertEqual([check.id for check in member_checks], ["member_1_automation", "member_2_automation"])
        self.assertTrue(all(check.blocking for check in member_checks))
        self.assertIn("No complete saved Regression plan", member_checks[0].safe_message)


if __name__ == "__main__":
    unittest.main()
