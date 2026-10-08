import json
import threading
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import urlencode
from uuid import uuid4

from qa_agent.evidence_policy import EvidenceMode, EvidencePolicy, ScreenshotMode
from qa_agent.cookie_consent import CookieConsentPolicy
from qa_agent.execution_repository import InMemoryExecutionRepository
from qa_agent.models import (
    Evidence,
    EvidenceType,
    Execution,
    ExecutionSegment,
    ExecutionStatus,
    PlanVersionOrigin,
    Precondition,
    QATestPlan,
    QATestStep,
    RunContext,
    TestCase as DomainTestCase,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestRun as DomainTestRun,
    TestStep as DomainTestStep,
)
from qa_agent.reporting import RunReportGenerator
from qa_agent.presentation import format_duration, format_timestamp
from qa_agent.run_history import (
    HistoryCleanupFailure,
    HistoryPrecondition,
    HistoryStep,
    InMemoryRunHistoryRepository,
    RunHistoryRecord,
    RunHistoryService,
    WorkflowType,
)
from qa_agent.test_case_repository import InMemoryTestCaseRepository
from qa_agent.plan_store import InMemoryPlanStore
from qa_agent.test_case_execution import WorkflowAvailability
from qa_agent.setup_orchestration import (
    CleanupOutcome,
    PreconditionSetupOutcome,
    SetupRunOutcome,
    SetupStatus,
)
from qa_agent.web import LocalWebApplication


class LocalWebApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_path = self.root / "failure.png"
        self.image_path.write_bytes(b"fake-png-data")
        self.started = datetime(2026, 5, 1, tzinfo=timezone.utc)
        steps = [
            DomainTestStep(name="Open page", description="Open.", expected="Loaded.", order=0),
            DomainTestStep(name="Check title", description="Check.", expected="Welcome.", order=1),
            DomainTestStep(name="Continue", description="Continue.", expected="Next.", order=2),
        ]
        self.precondition = Precondition(
            description="A user has been prepared.", order=0,
            provided_data_keys=["user_id"],
        )
        self.case = DomainTestCase(
            public_id="TC-0042",
            name="Sign up <flow>",
            description="Check registration safely.",
            steps=steps,
            preconditions=[self.precondition],
        )
        self.secret = "FAKE_UI_SECRET_9301"
        context = RunContext()
        context.set_value("password", self.secret, sensitive=True)
        passed = self.make_execution(steps[0], ExecutionStatus.PASSED, 0)
        failed = self.make_execution(
            steps[1], ExecutionStatus.FAILED, 2,
            error=f"Expected welcome; {self.secret}",
        )
        failed.evidence = (Evidence(
            execution_id=failed.id,
            type=EvidenceType.SCREENSHOT,
            path=str(self.image_path),
            description="Page after failure.",
        ),)
        self.run = DomainTestRun.from_test_case(
            self.case, [passed, failed], blocked_step_ids=[steps[2].id],
            run_context=context,
        )
        self.execution_repository = InMemoryExecutionRepository()
        for execution in self.run.executions:
            self.execution_repository.save(execution)
        self.history_repository = InMemoryRunHistoryRepository()
        self.history = RunHistoryService(
            self.history_repository, self.execution_repository
        )
        setup = SetupRunOutcome(SetupStatus.SUCCEEDED, (
            PreconditionSetupOutcome(
                precondition_id=self.precondition.id,
                status=SetupStatus.SUCCEEDED,
                produced_data_keys=("user_id",),
            ),
        ))
        self.record = self.history.record_completed_run(
            self.case, self.run,
            workflow_type=WorkflowType.REGRESSION,
            outcome="PRODUCT_FAILURE",
            setup=setup,
            cleanup=CleanupOutcome(),
            started_at=self.started,
            finished_at=self.started + timedelta(seconds=5),
        )
        self.app = LocalWebApplication(
            self.history, RunReportGenerator(), evidence_root=self.root
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_execution(self, step, status, offset, *, error=None):
        start = self.started + timedelta(seconds=offset)
        return Execution(
            test_step_id=step.id,
            test_plan_version_id=uuid4(),
            status=status,
            started_at=start,
            finished_at=start + timedelta(seconds=1),
            actual_result=status.value,
            error=error,
        )

    def test_dashboard_lists_recent_runs_and_escapes_testcase_name(self) -> None:
        response = self.app.handle("GET", "/")
        body = response.body.decode("utf-8")

        self.assertEqual(response.status, 200)
        self.assertIn(str(self.record.run_id), body)
        self.assertIn("REGRESSION", body)
        self.assertIn("FAILED", body)
        self.assertIn("Sign up &lt;flow&gt;", body)
        self.assertIn(f"/runs/{self.record.run_id}", body)
        self.assertNotIn("<flow>", body)

    def test_dashboard_has_generic_authoring_entry_and_keeps_recent_runs(self) -> None:
        body = self.app.handle("GET", "/").body.decode("utf-8")
        entry = body.split('<section class="panel authoring-entry"', 1)[1].split(
            "</section>", 1
        )[0]

        self.assertIn("What do you want to test?", entry)
        self.assertIn('name="base_url"', entry)
        self.assertIn('name="scenario"', entry)
        self.assertIn("Generate TestCase", entry)
        self.assertIn("Speak scenario", entry)
        self.assertIn('name="name"', entry)
        self.assertNotIn("FINN.no", entry)
        self.assertNotIn("checkout", entry.casefold())
        self.assertIn("Summary (optional)", entry)
        self.assertIn("Save Draft", entry)
        self.assertIn("Create Manually", entry)
        self.assertIn("Recent runs", body)
        self.assertIn(f"RUN-000001", body)

    def test_dashboard_validation_stays_inline_and_escapes_preserved_input(self) -> None:
        scenario = "<script>alert('xss')</script>"
        response = self.app.handle(
            "POST",
            "/test-cases/generate",
            urlencode({
                "authoring_entry": "dashboard",
                "base_url": "not-a-url",
                "scenario": scenario,
            }),
        )
        body = response.body.decode("utf-8")

        self.assertEqual(response.status, 400)
        self.assertIn("What do you want to test?", body)
        self.assertIn("Enter a valid URL, for example https://example.com", body)
        self.assertIn('class="field-error"', body)
        self.assertIn('aria-invalid="true"', body)
        self.assertIn("&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;", body)
        self.assertNotIn(scenario, body)

    def test_dashboard_rejects_empty_scenario_without_leaving_entry_page(self) -> None:
        response = self.app.handle(
            "POST",
            "/test-cases/generate",
            urlencode({
                "authoring_entry": "dashboard",
                "base_url": "https://example.test",
                "scenario": "",
            }),
        )
        body = response.body.decode("utf-8")

        self.assertEqual(response.status, 400)
        self.assertIn("Describe an action and what you expect to happen", body)
        self.assertIn('class="field-error"', body)
        self.assertIn("What do you want to test?", body)

    def test_dashboard_empty_website_has_specific_inline_error_and_preserves_scenario(self) -> None:
        scenario = "Check registration from this local scenario."
        response = self.app.handle(
            "POST",
            "/test-cases/generate",
            urlencode({
                "authoring_entry": "dashboard",
                "base_url": "",
                "scenario": scenario,
            }),
        )
        body = response.body.decode("utf-8")

        self.assertEqual(response.status, 400)
        self.assertIn("Enter the website URL you want to test.", body)
        self.assertIn(scenario, body)
        self.assertIn('id="case-website-error"', body)
        self.assertIn('aria-describedby="case-website-error"', body)
        self.assertIn("novalidate", body)

    def test_dashboard_summary_uses_public_run_id_and_friendly_time_duration(self) -> None:
        body = self.app.handle("GET", "/").body.decode("utf-8")

        self.assertIn("Total runs", body)
        self.assertIn("Passed", body)
        self.assertIn("Failed", body)
        self.assertIn("Regression", body)
        self.assertIn("Product failures", body)
        self.assertIn("Automation drift", body)
        self.assertIn("Infrastructure errors", body)
        self.assertIn("1 May 2026, 00:00 UTC", body)
        self.assertIn("5.0 s", body)
        self.assertIn(f'href="/runs/{self.record.run_id}" title="{self.record.run_id}"', body)
        self.assertIn(">RUN-000001</a>", body)
        self.assertEqual(format_timestamp(self.started), "1 May 2026, 00:00 UTC")
        self.assertEqual(format_duration(34000), "34.0 s")
        self.assertEqual(format_duration(72000), "1m 12s")

    def test_testcase_list_and_detail_use_human_step_numbering(self) -> None:
        listing = self.app.handle("GET", "/test-cases")
        detail = self.app.handle("GET", f"/test-cases/{self.case.id}")
        list_body = listing.body.decode("utf-8")
        detail_body = detail.body.decode("utf-8")

        self.assertEqual(listing.status, 200)
        self.assertIn("Sign up &lt;flow&gt;", list_body)
        self.assertIn("TC-0042", list_body)
        self.assertIn("Latest status", list_body)
        self.assertEqual(detail.status, 200)
        self.assertIn("TC-0042", detail_body)
        self.assertIn("<li><strong>Open page</strong>", detail_body)
        self.assertNotIn("<strong>0.", detail_body)
        self.assertNotIn("<strong>1.", detail_body)
        self.assertIn("Definition shown from the most recent recorded run", detail_body)

    def test_runs_filters_support_workflow_status_and_combination(self) -> None:
        infrastructure_run_id = uuid4()
        self.history_repository.save(RunHistoryRecord(
            run_id=infrastructure_run_id,
            test_case_id=self.case.id,
            test_case_name="Sign up <flow>",
            test_case_description="Check registration safely.",
            workflow_type=WorkflowType.VALIDATION,
            outcome="PASSED",
            status=ExecutionStatus.PASSED,
            started_at=self.started + timedelta(minutes=1),
            duration_ms=12000,
        ))
        self.history_repository.save(RunHistoryRecord(
            run_id=uuid4(),
            test_case_id=self.case.id,
            test_case_name="Sign up <flow>",
            test_case_description="Check registration safely.",
            workflow_type=WorkflowType.REGRESSION,
            outcome="INFRASTRUCTURE_ERROR",
            status=ExecutionStatus.FAILED,
            started_at=self.started + timedelta(minutes=2),
            duration_ms=9000,
        ))

        regression = self.app.handle("GET", "/runs?workflow=REGRESSION")
        failed = self.app.handle("GET", "/runs?status=FAILED")
        combined = self.app.handle(
            "GET", "/runs?workflow=REGRESSION&status=FAILED&failure_type=PRODUCT_FAILURE"
        )
        regression_body = regression.body.decode("utf-8")
        failed_body = failed.body.decode("utf-8")
        combined_body = combined.body.decode("utf-8")

        self.assertIn("Showing 2 of 3 recent runs", regression_body)
        self.assertIn("Showing 2 of 3 recent runs", failed_body)
        self.assertIn("Showing 1 of 3 recent runs", combined_body)
        self.assertIn("PRODUCT FAILURE", combined_body)
        self.assertNotIn(str(infrastructure_run_id), combined_body)
        self.assertIn('aria-label="Main navigation"', combined_body)

    def test_run_detail_has_actions_failure_origin_setup_cleanup_and_report_consistency(self) -> None:
        detail = self.app.handle("GET", f"/runs/{self.record.run_id}")
        json_response = self.app.handle("GET", f"/runs/{self.record.run_id}/report.json")
        html_response = self.app.handle("GET", f"/runs/{self.record.run_id}/report.html")
        body = detail.body.decode("utf-8")
        json_body = json.loads(json_response.body)
        html_body = html_response.body.decode("utf-8")

        self.assertIn("PRODUCT FAILURE", body)
        self.assertIn("detected unexpected product behavior", body)
        self.assertIn("Expected:", body)
        self.assertIn("Actual", body)
        self.assertIn("Plan version", body)
        self.assertIn("Setup", body)
        self.assertIn("Preconditions", body)
        self.assertIn("Cleanup completed successfully", body)
        self.assertIn(f'href="/runs/{self.record.run_id}/report.json"', body)
        self.assertIn(f'href="/runs/{self.record.run_id}/report.html" target="_blank" rel="noopener"', body)
        self.assertEqual(json_response.status, 200)
        self.assertEqual(html_response.status, 200)
        self.assertEqual(json_body["status"], "FAILED")
        self.assertEqual(json_body["outcome"], "PRODUCT_FAILURE")
        self.assertEqual(json_body["run_public_id"], "RUN-000001")
        self.assertEqual(json_body["test_case_public_id"], "TC-0042")
        self.assertEqual(json_body["run_id"], str(self.record.run_id))
        self.assertEqual(json_body["test_case_id"], str(self.case.id))
        self.assertEqual([step["status"] for step in json_body["steps"]], [
            "PASSED", "FAILED", "BLOCKED"
        ])
        self.assertIn("FAILED", html_body)
        self.assertIn("BLOCKED", html_body)
        self.assertIn("RUN-000001", html_body)
        self.assertIn("TC-0042", html_body)
        self.assertIn(f'href="/runs/{self.record.run_id}"', html_body)
        self.assertIn("Back to Run", html_body)
        self.assertNotIn(self.secret, body + html_body + json_response.body.decode("utf-8"))

    def test_automation_versions_show_provenance_and_keep_uuid_in_details(self) -> None:
        step = self.case.steps[0]
        plan = DomainTestPlan(test_step_id=step.id, name=step.name)
        version = DomainTestPlanVersion(
            test_plan_id=plan.id,
            version=4,
            created_at=self.started,
            origin=PlanVersionOrigin.REPAIRED,
            qa_test_plan=QATestPlan(
                url="http://127.0.0.1:8000/register",
                steps=[QATestStep(action="assert_page_loaded")],
            ),
        )
        plans = InMemoryPlanStore()
        plans.save(step.id, version, test_plan=plan)
        cases = InMemoryTestCaseRepository()
        cases.save(self.case)
        run_service = Mock()
        run_service.workflow_availability.return_value = WorkflowAvailability(
            automation_available=True,
            validation_available=True,
            regression_available=True,
            usable_plan_count=1,
            total_step_count=len(self.case.steps),
            plan_versions=((step.id, version.version, version.id),),
            reason="A complete version is not available yet.",
        )
        app = LocalWebApplication(
            self.history,
            test_cases=cases,
            run_service=run_service,
            plan_store=plans,
        )
        try:
            body = app.handle("GET", f"/test-cases/{self.case.id}").body.decode("utf-8")
        finally:
            app.close()

        summary_start = body.index("<summary>v4")
        summary_end = body.index("</summary>", summary_start)
        self.assertIn("Updated automatically", body[summary_start:summary_end])
        self.assertNotIn(str(version.id), body[summary_start:summary_end])
        self.assertIn("Created: 1 May 2026, 00:00 UTC", body)
        self.assertIn("Previous version: v3", body)
        self.assertIn(f"Internal ID: <code>{version.id}</code>", body)
        self.assertIn("Auto handle cookie consent", body)
        self.assertIn("Leave cookie consent unchanged", body)
        self.assertEqual(body.count('name="cookie_policy"'), 3)

    def test_setup_failure_and_cleanup_failure_are_presented_separately(self) -> None:
        setup_run = RunHistoryRecord(
            run_id=uuid4(),
            test_case_id=uuid4(),
            test_case_name="Setup blocked flow",
            test_case_description="Setup could not establish the initial state.",
            workflow_type=WorkflowType.VALIDATION,
            outcome="SETUP_FAILURE",
            status=ExecutionStatus.FAILED,
            started_at=self.started,
            finished_at=self.started + timedelta(seconds=2),
            duration_ms=2000,
            steps=[HistoryStep(
                id=uuid4(), order=0, name="Submit order", description="Submit.",
                expected="Order accepted.", status=None,
            )],
            preconditions=[HistoryPrecondition(
                id=uuid4(), order=0, description="Inventory is available.",
                status="SETUP_INFRASTRUCTURE_ERROR", error="Setup adapter unavailable.",
            )],
            setup_status="SETUP_INFRASTRUCTURE_ERROR",
            cleanup_succeeded=False,
            cleanup_failures=[HistoryCleanupFailure(
                label="Release setup fixture", error_type="OSError", message="Fixture could not be released."
            )],
        )
        self.history_repository.save(setup_run)

        response = self.app.handle("GET", f"/runs/{setup_run.run_id}")
        body = response.body.decode("utf-8")

        self.assertEqual(response.status, 200)
        self.assertIn("SETUP FAILURE", body)
        self.assertIn("product execution did not begin", body)
        self.assertIn("CLEANUP FAILURE", body)
        self.assertIn("Fixture could not be released.", body)
        self.assertIn("No execution attempt was recorded.", body)
        self.assertIn("FAILED", body)

    def test_testcase_page_shows_history_definition_and_preconditions(self) -> None:
        response = self.app.handle("GET", f"/test-cases/{self.case.id}")
        body = response.body.decode("utf-8")

        self.assertEqual(response.status, 200)
        self.assertIn("Open page", body)
        self.assertIn("Welcome.", body)
        self.assertIn("A user has been prepared", body)
        self.assertIn(f"/runs/{self.record.run_id}", body)
        self.assertIn("most recent recorded run", body)

    def test_run_page_and_report_endpoint_show_product_failure_and_blocked(self) -> None:
        page = self.app.handle("GET", f"/runs/{self.record.run_id}")
        report_response = self.app.handle("GET", f"/runs/{self.record.run_id}/report.json")
        payload = json.loads(report_response.body)

        self.assertEqual(page.status, 200)
        self.assertIn("Check title", page.body.decode("utf-8"))
        self.assertIn("BLOCKED", page.body.decode("utf-8"))
        self.assertIn("Plan version", page.body.decode("utf-8"))
        self.assertIn("REGRESSION", page.body.decode("utf-8"))
        self.assertEqual(report_response.status, 200)
        self.assertEqual(payload["outcome"], "PRODUCT_FAILURE")
        self.assertEqual([item["status"] for item in payload["steps"]], ["PASSED", "FAILED", "BLOCKED"])
        self.assertEqual(
            payload["steps"][1]["attempts"][0]["test_plan_version_id"],
            str(self.run.executions[1].test_plan_version_id),
        )
        self.assertNotIn(self.secret, page.body.decode("utf-8"))
        self.assertNotIn(self.secret, report_response.body.decode("utf-8"))

    def test_html_report_is_available_without_external_assets(self) -> None:
        response = self.app.handle("GET", f"/runs/{self.record.run_id}/report.html")
        body = response.body.decode("utf-8")

        self.assertEqual(response.status, 200)
        self.assertIn("<style>", body)
        self.assertNotIn("<script src=", body)
        self.assertNotIn("<link rel=", body)
        self.assertIn(f"/runs/{self.record.run_id}/evidence/{self.run.executions[1].id}/0", body)

    def test_unknown_or_invalid_ids_return_404_and_post_is_read_only(self) -> None:
        unknown_run = self.app.handle("GET", f"/runs/{uuid4()}")
        self.assertEqual(unknown_run.status, 404)
        self.assertIn(b"error-state", unknown_run.body)
        self.assertIn(b"Back to Dashboard", unknown_run.body)
        self.assertEqual(self.app.handle("GET", "/runs/not-a-uuid").status, 404)
        unknown_case = self.app.handle("GET", f"/test-cases/{uuid4()}")
        self.assertEqual(unknown_case.status, 404)
        self.assertIn(b"error-state", unknown_case.body)
        self.assertEqual(self.app.handle("GET", "/test-cases/not-a-uuid").status, 404)
        self.assertEqual(self.app.handle("POST", f"/runs/{self.record.run_id}").status, 405)

    def test_evidence_route_serves_only_associated_files_under_configured_root(self) -> None:
        execution = self.run.executions[1]
        response = self.app.handle(
            "GET", f"/runs/{self.record.run_id}/evidence/{execution.id}/0"
        )
        unknown_execution = self.app.handle(
            "GET", f"/runs/{self.record.run_id}/evidence/{uuid4()}/0"
        )
        unknown_index = self.app.handle(
            "GET", f"/runs/{self.record.run_id}/evidence/{execution.id}/5"
        )

        self.assertEqual(response.status, 200)
        self.assertEqual(response.body, b"fake-png-data")
        self.assertEqual(response.content_type, "image/png")
        self.assertEqual(unknown_execution.status, 404)
        self.assertEqual(unknown_index.status, 404)

    def test_evidence_path_outside_allowed_root_is_not_served(self) -> None:
        execution = self.run.executions[1]
        execution.evidence = (Evidence(
            execution_id=execution.id,
            id=execution.evidence[0].id,
            type=EvidenceType.SCREENSHOT,
            path=str(self.root.parent / "outside.png"),
        ),)
        self.execution_repository = InMemoryExecutionRepository()
        for item in self.run.executions:
            self.execution_repository.save(item)
        self.history = RunHistoryService(self.history_repository, self.execution_repository)
        self.app = LocalWebApplication(
            self.history, RunReportGenerator(), evidence_root=self.root
        )

        response = self.app.handle(
            "GET", f"/runs/{self.record.run_id}/evidence/{execution.id}/0"
        )

        self.assertEqual(response.status, 404)

    def test_traversal_evidence_is_rejected_and_missing_image_has_no_broken_link(self) -> None:
        execution = self.run.executions[1]
        evidence = execution.evidence[0]
        execution.evidence = (Evidence(
            execution_id=execution.id,
            id=evidence.id,
            type=EvidenceType.SCREENSHOT,
            path="../outside.png",
            description=evidence.description,
        ),)
        repository = InMemoryExecutionRepository()
        for item in self.run.executions:
            repository.save(item)
        self.app = LocalWebApplication(
            RunHistoryService(self.history_repository, repository),
            RunReportGenerator(),
            evidence_root=self.root,
        )

        image_response = self.app.handle(
            "GET", f"/runs/{self.record.run_id}/evidence/{execution.id}/0"
        )
        report_response = self.app.handle("GET", f"/runs/{self.record.run_id}/report.html")
        report_body = report_response.body.decode("utf-8")

        self.assertEqual(image_response.status, 404)
        self.assertIn("unavailable", report_body)
        self.assertNotIn(f"/runs/{self.record.run_id}/evidence/", report_body)


class PersistedTestCaseRunUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = DomainTestCase(
            name="Saved local case",
            description="This definition exists before it has run.",
            segments=[ExecutionSegment(order=0, steps=[DomainTestStep(
                name="Open saved page",
                description="Open a page.",
                expected="It loads.",
                order=0,
            )])],
        )
        cases = InMemoryTestCaseRepository()
        cases.save(self.case)
        self.run_service = Mock()
        self.run_started = threading.Event()
        self.release_run = threading.Event()

        def run_blocked(test_case_id, workflow_type):
            self.run_started.set()
            self.release_run.wait(timeout=5)
            return SimpleNamespace(
                test_run=SimpleNamespace(status=ExecutionStatus.PASSED),
                outcome="PASSED",
            )

        self.run_service.run.side_effect = run_blocked
        self.app = LocalWebApplication(
            RunHistoryService(InMemoryRunHistoryRepository()),
            test_cases=cases,
            run_service=self.run_service,
        )

    def tearDown(self) -> None:
        self.release_run.set()
        self.app.close()

    def test_saved_case_is_listed_and_detail_loads_without_history(self) -> None:
        listing = self.app.handle("GET", "/test-cases")
        detail = self.app.handle("GET", f"/test-cases/{self.case.id}")

        self.assertEqual(listing.status, 200)
        self.assertIn(b"Saved local case", listing.body)
        self.assertIn(b"TC-0001", listing.body)
        self.assertEqual(detail.status, 200)
        self.assertIn(b"TC-0001", detail.body)
        self.assertIn(b"No runs yet.", detail.body)
        self.assertIn(b"Start run", detail.body)
        self.assertIn(b'method="post"', detail.body)
        self.run_service.run.assert_not_called()

    def test_run_page_exposes_compact_evidence_controls_and_post_propagates_choice(self) -> None:
        page = self.app.handle("GET", f"/test-cases/{self.case.id}").body.decode("utf-8")
        self.assertIn("Evidence settings", page)
        self.assertIn('value="FAILURES_ONLY" selected>Failures only', page)
        self.assertIn('value="EVERY_VERIFICATION">Every verification', page)
        self.assertIn('value="EVERY_STEP">Every step', page)
        self.assertIn('value="ELEMENT">Element', page)
        self.assertIn('value="PAGE" selected>Page', page)
        self.assertIn('value="ELEMENT_AND_PAGE">Element + Page', page)
        self.assertIn("Auto handle cookie consent", page)
        self.assertIn("Leave cookie consent unchanged", page)

        with patch.object(self.app._background_runs, "start", return_value="progress-id") as start:
            response = self.app.handle(
                "POST",
                f"/test-cases/{self.case.id}/run",
                b"workflow=REGRESSION&evidence_mode=EVERY_VERIFICATION&screenshot_mode=ELEMENT_AND_PAGE&cookie_policy=LEAVE_UNCHANGED",
            )

        self.assertEqual(response.status, 303)
        self.assertEqual(
            start.call_args.kwargs["evidence_policy"],
            EvidencePolicy(
                mode=EvidenceMode.EVERY_VERIFICATION,
                screenshot_mode=ScreenshotMode.ELEMENT_AND_PAGE,
            ),
        )
        self.assertEqual(
            start.call_args.kwargs["cookie_policy"],
            CookieConsentPolicy.LEAVE_UNCHANGED,
        )

    def test_post_returns_progress_url_and_runs_selected_workflow_in_background(self) -> None:
        started_at = time.monotonic()
        response = self.app.handle(
            "POST",
            f"/test-cases/{self.case.id}/run",
            b"workflow=REGRESSION",
        )

        self.assertEqual(response.status, 303)
        self.assertLess(time.monotonic() - started_at, 1.5)
        self.assertTrue(response.headers["Location"].startswith("/runs/progress/"))
        progress_id = response.headers["Location"].rsplit("/", 1)[1]
        self.assertTrue(self.run_started.wait(timeout=2))
        page = self.app.handle("GET", response.headers["Location"])
        self.assertEqual(page.status, 200)
        self.assertIn(b"Live execution", page.body)
        self.assertIn(b"/api/progress/", self.app.handle("GET", "/assets/ui.js").body)
        self.assertEqual(self.app.handle("GET", response.headers["Location"]).status, 200)
        self.run_service.run.assert_called_once_with(self.case.id, WorkflowType.REGRESSION)

        self.release_run.set()
        deadline = time.monotonic() + 2
        snapshot = None
        while time.monotonic() < deadline:
            progress_response = self.app.handle("GET", f"/api/progress/{progress_id}")
            snapshot = json.loads(progress_response.body)
            if snapshot["finished"]:
                break
            time.sleep(0.01)
        self.assertIsNotNone(snapshot)
        self.assertTrue(snapshot["finished"])
        self.assertEqual(snapshot["workflow"], "REGRESSION")
        self.assertEqual(snapshot["outcome"], "PASSED")

    def test_get_cannot_start_run_and_invalid_workflow_is_rejected(self) -> None:
        get_response = self.app.handle("GET", f"/test-cases/{self.case.id}/run")
        invalid_response = self.app.handle(
            "POST",
            f"/test-cases/{self.case.id}/run",
            b"workflow=UNKNOWN",
        )

        self.assertEqual(get_response.status, 405)
        self.assertEqual(invalid_response.status, 400)
        self.run_service.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
