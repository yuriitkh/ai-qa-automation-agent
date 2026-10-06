import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from qa_agent.execution_repository import InMemoryExecutionRepository
from qa_agent.models import (
    Evidence,
    EvidenceType,
    Execution,
    ExecutionStatus,
    Precondition,
    RunContext,
    TestCase as DomainTestCase,
    TestRun as DomainTestRun,
    TestStep as DomainTestStep,
)
from qa_agent.reporting import RunReportGenerator
from qa_agent.run_history import (
    InMemoryRunHistoryRepository,
    RunHistoryService,
    WorkflowType,
)
from qa_agent.setup_orchestration import (
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
        self.assertEqual(self.app.handle("GET", f"/runs/{uuid4()}").status, 404)
        self.assertEqual(self.app.handle("GET", "/runs/not-a-uuid").status, 404)
        self.assertEqual(self.app.handle("GET", f"/test-cases/{uuid4()}").status, 404)
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


if __name__ == "__main__":
    unittest.main()
