import json
import tempfile
import unittest
from pathlib import Path

from qa_agent.models import (
    ExecutionStatus,
    Precondition,
    QATestPlan,
    QATestStep,
    RunContext,
    TestCase as DomainTestCase,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
    FailurePolicy,
)
from qa_agent.plan_execution import PlanExecutionService
from qa_agent.pinned_execution import PinnedExecutionService, PlanVersionSet, StepPlanSelection
from qa_agent.reporting import RunReportGenerator
from qa_agent.run_history import RunHistoryService, WorkflowType
from qa_agent.setup_orchestration import SetupCleanupCoordinator, SetupOperationResult, SetupStatus
from qa_agent.sqlite_storage import (
    SQLiteExecutionRepository,
    SQLitePlanStore,
    SQLiteRunHistoryRepository,
)
from qa_agent.web import LocalWebApplication
from qa_agent.workflows import RegressionWorkflow, ValidationWorkflow


class ProductDemoSliceTests(unittest.TestCase):
    def test_validation_to_sqlite_history_report_and_web_then_successful_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "demo.sqlite3"
            evidence_root = root / "evidence"
            evidence_root.mkdir()
            screenshot = evidence_root / "failure.png"
            screenshot.write_bytes(b"demo-screenshot")
            secret = "FAKE_E2E_SECRET_1539"
            precondition = Precondition(
                description="A demo account is ready.",
                order=0,
                provided_data_keys=["account_id"],
            )
            steps = [
                DomainTestStep(
                    name="Open registration", description="Open the form.",
                    expected="Registration form is loaded.", order=0,
                ),
                DomainTestStep(
                    name="Verify welcome", description="Check the result.",
                    expected="Welcome is visible.", order=1,
                    failure_policy=FailurePolicy.BLOCK_REST,
                ),
                DomainTestStep(
                    name="Continue flow", description="Continue after welcome.",
                    expected="Next page is visible.", order=2,
                ),
            ]
            test_case = DomainTestCase(
                name="Registration demo",
                description="Validate a registration workflow.",
                steps=steps,
                preconditions=[precondition],
            )
            plan_store = SQLitePlanStore(database)
            execution_repository = SQLiteExecutionRepository(database)
            history_repository = SQLiteRunHistoryRepository(database)
            history = RunHistoryService(history_repository, execution_repository)
            selected = []
            for step in steps:
                test_plan = DomainTestPlan(test_step_id=step.id, name=step.name)
                version = DomainTestPlanVersion(
                    test_plan_id=test_plan.id,
                    version=1,
                    qa_test_plan=QATestPlan(
                        url="https://example.test/register",
                        steps=[QATestStep(
                            action="assert_title",
                            parameters={"expected": f"step-{step.order}"},
                        )],
                    ),
                )
                plan_store.save(step.id, version, test_plan=test_plan)
                selected.append(StepPlanSelection(step.id, version.id))
            version_set = PlanVersionSet(tuple(selected))
            events = []

            def setup_operation(_precondition, context, register_cleanup):
                context.set_value("account_id", "demo-user-1", source="demo-setup")
                register_cleanup(lambda received: events.append(("cleanup", received)))
                return SetupOperationResult(
                    status=SetupStatus.SUCCEEDED,
                    produced_data_keys=("account_id",),
                )

            coordinator = SetupCleanupCoordinator({
                precondition.id: _Setup(setup_operation),
            })
            context = RunContext()
            context.set_value("api_token", secret, sensitive=True)

            def mixed_runner(plan):
                expected = plan.steps[0].parameters["expected"]
                if expected == "step-1":
                    return {
                        "status": "failed",
                        "steps": [{
                            "action": "assert_title", "status": "failed",
                            "error": "Expected Welcome, received another title.",
                        }],
                        "evidence": [{
                            "type": "SCREENSHOT",
                            "path": str(screenshot),
                            "description": "Failed welcome assertion.",
                        }],
                    }
                return {"status": "passed", "steps": []}

            pinned_execution = PinnedExecutionService(
                plan_store,
                PlanExecutionService(mixed_runner, execution_repository),
            )
            failed = ValidationWorkflow(
                pinned_execution, coordinator, run_history=history
            ).run(test_case, version_set, run_context=context)

            self.assertEqual(failed.outcome.value, "PRODUCT_FAILURE")
            self.assertEqual(
                [step.status for step in failed.test_run.executions],
                [ExecutionStatus.PASSED, ExecutionStatus.FAILED],
            )
            self.assertEqual(failed.test_run.blocked_steps, [steps[2].id])
            self.assertTrue(failed.cleanup.succeeded)
            self.assertIs(failed.test_run.run_context, context)
            self.assertNotIn(secret, history.get(failed.test_run.id).model_dump_json())
            self.assertEqual(len(events), 1)

            # Reopen all SQLite adapters to prove the UI/report consume durable history.
            reopened_executions = SQLiteExecutionRepository(database)
            reopened_history = RunHistoryService(
                SQLiteRunHistoryRepository(database), reopened_executions
            )
            detail = reopened_history.get_detail(failed.test_run.id)
            report_generator = RunReportGenerator()
            report = report_generator.generate_history(detail)
            report_json = json.loads(report.to_json())
            report_html = report_generator.to_html(report)
            application = LocalWebApplication(
                reopened_history,
                report_generator,
                evidence_root=evidence_root,
            )
            dashboard = application.handle("GET", "/")
            run_page = application.handle("GET", f"/runs/{failed.test_run.id}")
            api_report = application.handle(
                "GET", f"/runs/{failed.test_run.id}/report.json"
            )
            evidence_response = application.handle(
                "GET",
                f"/runs/{failed.test_run.id}/evidence/{failed.test_run.executions[1].id}/0",
            )

            self.assertEqual(report_json["workflow_type"], WorkflowType.VALIDATION.value)
            self.assertEqual(
                [step["status"] for step in report_json["steps"]],
                ["PASSED", "FAILED", "BLOCKED"],
            )
            self.assertEqual(
                report_json["steps"][1]["attempts"][0]["test_plan_version_id"],
                str(failed.test_run.executions[1].test_plan_version_id),
            )
            self.assertNotIn(secret, report_json.__str__())
            self.assertIn("BLOCKED", report_html)
            self.assertEqual(dashboard.status, 200)
            self.assertIn(str(failed.test_run.id), dashboard.body.decode("utf-8"))
            self.assertEqual(run_page.status, 200)
            self.assertEqual(json.loads(api_report.body)["run_id"], str(failed.test_run.id))
            self.assertEqual(evidence_response.status, 200)
            self.assertEqual(evidence_response.body, b"demo-screenshot")

            # The next regression is a separate successful durable run.
            passing_execution = PinnedExecutionService(
                plan_store,
                PlanExecutionService(
                    lambda _plan: {"status": "passed", "steps": []},
                    execution_repository,
                ),
            )
            passing = RegressionWorkflow(
                passing_execution, coordinator, run_history=history
            ).run(test_case, version_set, run_context=RunContext())
            self.assertEqual(passing.outcome.value, "PASSED")
            self.assertEqual(passing.test_run.status, ExecutionStatus.PASSED)
            recent = reopened_history.list_for_test_case(test_case.id)
            self.assertEqual([item.run_id for item in recent], [passing.test_run.id, failed.test_run.id])
            self.assertEqual(recent[0].workflow_type, WorkflowType.REGRESSION)
            self.assertEqual(recent[1].workflow_type, WorkflowType.VALIDATION)


class _Setup:
    def __init__(self, callback):
        self.callback = callback

    def execute(self, precondition, run_context, register_cleanup):
        return self.callback(precondition, run_context, register_cleanup)


if __name__ == "__main__":
    unittest.main()
