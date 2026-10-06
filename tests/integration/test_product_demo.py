import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.parse import urlencode
from uuid import UUID

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
from qa_agent.browser_runner import BrowserRunner
from qa_agent.llm.base import LLMProvider
from qa_agent.llm.router import LLMRouter
from qa_agent.models import QATestPlan
from qa_agent.pipeline import QATestPipeline
from qa_agent.plan_execution import PlanExecutionClassification, PlanExecutionService
from qa_agent.pinned_execution import PinnedExecutionService, PlanVersionSet, StepPlanSelection
from qa_agent.reporting import RunReportGenerator
from qa_agent.run_history import RunHistoryService, WorkflowType
from qa_agent.setup_orchestration import SetupCleanupCoordinator, SetupOperationResult, SetupStatus
from qa_agent.sqlite_storage import (
    SQLiteExecutionRepository,
    SQLitePlanStore,
    SQLiteRunHistoryRepository,
)
from qa_agent.storage import create_sqlite_storage
from qa_agent.test_case_authoring import TestCaseAuthoringService
from qa_agent.test_case_decomposer import TestCaseDecomposer
from qa_agent.test_case_execution import TestCaseExecutionService
from qa_agent.test_plan_generator import LLMTestPlanGenerator
from qa_agent.web import LocalWebApplication, create_http_server
from qa_agent.workflows import AutomationWorkflow, RegressionWorkflow, ValidationWorkflow


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

    def test_ai_authoring_review_save_automation_real_browser_and_pinned_regression(self) -> None:
        class DeterministicProvider(LLMProvider):
            def create_test_plan(self, task, target_url, page_snapshot):
                return QATestPlan(url=target_url, steps=[
                    {"action": "navigate", "parameters": {"url": target_url}},
                    {"action": "assert_text_contains", "parameters": {
                        "expected_text": "Account confirmation displayed",
                    }},
                ])

            def create_structured_output(self, prompt, schema, schema_name):
                return (
                    '{"preconditions":[],"segments":[{"steps":[{'
                    '"name":"Verify registration confirmation",'
                    '"description":"Open the registration page and verify the result.",'
                    '"expected":"The account confirmation is displayed."}]}]}'
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "authoring-flow.sqlite3"
            evidence_root = root / "evidence"
            storage = create_sqlite_storage(database)
            provider = DeterministicProvider()
            router = LLMRouter([provider])
            pipeline = QATestPipeline(
                decomposer=TestCaseDecomposer(),
                plan_generator=LLMTestPlanGenerator(router),
                runner=BrowserRunner(evidence_root, headless=True),
                plan_store=storage.plan_store,
                execution_repository=storage.execution_repository,
                run_history=storage.run_history,
            )
            run_service = TestCaseExecutionService(
                storage.test_case_repository,
                storage.plan_store,
                storage.execution_repository,
                storage.run_history,
                evidence_directory=evidence_root,
                automation_workflow=AutomationWorkflow(pipeline),
            )
            application = LocalWebApplication(
                storage.run_history,
                evidence_root=evidence_root,
                test_cases=storage.test_case_repository,
                run_service=run_service,
                authoring_service=TestCaseAuthoringService(LLMRouter([provider])),
            )
            server = create_http_server(application, host="127.0.0.1", port=0)
            port = server.server_address[1]
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            base_url = f"http://127.0.0.1:{port}/demo-target/registration"
            try:
                created = application.handle(
                    "POST",
                    "/test-cases/generate",
                    urlencode({
                        "name": "AI registration flow",
                        "base_url": base_url,
                        "scenario": "Register a user and verify the account confirmation.",
                    }),
                )
                self.assertEqual(created.status, 303)
                token = created.headers["Location"].rsplit("/", 1)[1]
                review = application.handle("GET", created.headers["Location"])
                self.assertIn(b"Verify registration confirmation", review.body)
                self.assertEqual(storage.test_case_repository.list(), [])

                saved = application.handle(
                    "POST", f"/test-cases/review/{token}/save", b""
                )
                self.assertEqual(saved.status, 303)
                case_id = UUID(saved.headers["Location"].rsplit("/", 1)[1])
                case = storage.test_case_repository.get(case_id)
                self.assertIsNotNone(case)
                self.assertEqual(storage.run_history.list_for_test_case(case_id), [])
                before_run = application.handle("GET", saved.headers["Location"])
                self.assertIn(b"NOT RUN", before_run.body)
                self.assertIn(b"Never", before_run.body)
                self.assertIn(b"Validation", before_run.body)
                self.assertIn(b"Regression", before_run.body)
                self.assertEqual(before_run.body.count(b"Not ready"), 2)

                unavailable = application.handle(
                    "POST", f"/test-cases/{case_id}/run", b"workflow=VALIDATION"
                )
                self.assertEqual(unavailable.status, 409)
                self.assertEqual(storage.run_history.list_for_test_case(case_id), [])

                automation = application.handle(
                    "POST", f"/test-cases/{case_id}/run", b"workflow=AUTOMATION"
                )
                self.assertEqual(automation.status, 303)
                automation_id = UUID(automation.headers["Location"].rsplit("/", 1)[1])
                automation_record = storage.run_history.get(automation_id)
                self.assertEqual(automation_record.workflow_type, WorkflowType.AUTOMATION)
                self.assertEqual(automation_record.status, ExecutionStatus.FAILED)
                self.assertEqual(len(storage.run_history.list_for_test_case(case_id)), 1)
                self.assertEqual(run_service.workflow_availability(case_id).usable_plan_count, 1)
                after_automation = application.handle("GET", saved.headers["Location"])
                self.assertEqual(after_automation.body.count(b"Available"), 3)
                self.assertIn(b"v1", after_automation.body)
                plan_version = storage.plan_store.find(case.steps[0].id)
                self.assertIn(str(plan_version.id)[:8].encode(), after_automation.body)

                automation_detail = storage.run_history.get_detail(automation_id)
                failed_execution = next(
                    execution for execution in automation_detail.executions.values()
                    if execution.evidence
                )
                evidence = application.handle(
                    "GET",
                    f"/runs/{automation_id}/evidence/{failed_execution.id}/0",
                )
                self.assertEqual(evidence.status, 200)
                self.assertEqual(evidence.content_type, "image/png")
                report = application.handle("GET", f"/runs/{automation_id}/report.json")
                self.assertEqual(report.status, 200)
                parsed_report = json.loads(report.body)
                self.assertEqual(parsed_report["workflow_type"], WorkflowType.AUTOMATION.value)
                self.assertEqual(parsed_report["steps"][0]["attempts"][0]["test_plan_version_id"],
                                 str(automation_record.executions[0].test_plan_version_id))

                regression = application.handle(
                    "POST", f"/test-cases/{case_id}/run", b"workflow=REGRESSION"
                )
                self.assertEqual(regression.status, 303)
                regression_id = UUID(regression.headers["Location"].rsplit("/", 1)[1])
                regression_record = storage.run_history.get(regression_id)
                self.assertEqual(regression_record.workflow_type, WorkflowType.REGRESSION)
                self.assertEqual(regression_record.outcome, "PRODUCT_FAILURE")
                self.assertEqual(len(storage.run_history.list_for_test_case(case_id)), 2)
                self.assertEqual(regression_record.executions[0].test_plan_version_id, plan_version.id)
                run_page = application.handle("GET", regression.headers["Location"])
                self.assertEqual(run_page.status, 200)
                self.assertIn(b"FAILED", run_page.body)
            finally:
                server.shutdown()
                server.server_close()
                server_thread.join(timeout=5)

    def test_ai_authored_four_step_automation_keeps_product_failure_semantics(self) -> None:
        class FourStepProvider(LLMProvider):
            def __init__(self):
                self.plan_calls = 0

            def create_structured_output(self, prompt, schema, schema_name):
                return json.dumps({
                    "preconditions": [],
                    "segments": [{"steps": [
                        {
                            "name": "Open registration page",
                            "description": "Open the local registration form.",
                            "expected": "The email input is visible.",
                        },
                        {
                            "name": "Enter user details",
                            "description": "Populate the email field.",
                            "expected": "The email field contains the demo address.",
                        },
                        {
                            "name": "Submit registration",
                            "description": "Submit the registration form.",
                            "expected": "The registration action is processed.",
                        },
                        {
                            "name": "Verify confirmation state",
                            "description": "Verify the account-created confirmation.",
                            "expected": "The account-created confirmation is displayed.",
                        },
                    ]}],
                })

            def create_test_plan(self, task, target_url, page_snapshot):
                self.plan_calls += 1
                snapshot = json.loads(page_snapshot)
                elements = snapshot["interactive_elements"]
                email_selector = next(
                    item["selector"] for item in elements
                    if item["accessible_name"].casefold() == "email"
                )
                submit_selector = next(
                    item["selector"] for item in elements
                    if item["accessible_name"].casefold() == "create account"
                )
                actions = {
                    1: [
                        {"action": "navigate", "parameters": {"url": target_url}},
                        {"action": "assert_visible", "parameters": {
                            "selector": email_selector,
                            "expected_text": None,
                        }},
                    ],
                    2: [
                        {"action": "navigate", "parameters": {"url": target_url}},
                        {"action": "fill", "parameters": {
                            "selector": email_selector,
                            "value": "qa.demo@example.test",
                        }},
                    ],
                    3: [
                        {"action": "navigate", "parameters": {"url": target_url}},
                        {"action": "click", "parameters": {"selector": submit_selector}},
                    ],
                    4: [
                        {"action": "navigate", "parameters": {"url": target_url}},
                        {"action": "assert_text_contains", "parameters": {
                            "expected_text": "Account created successfully",
                        }},
                    ],
                }[self.plan_calls]
                return QATestPlan(url=target_url, steps=actions)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "four-step-automation.sqlite3"
            evidence_root = root / "evidence"
            storage = create_sqlite_storage(database)
            provider = FourStepProvider()
            router = LLMRouter([provider])
            pipeline = QATestPipeline(
                decomposer=TestCaseDecomposer(),
                plan_generator=LLMTestPlanGenerator(router),
                runner=BrowserRunner(evidence_root, headless=True),
                plan_store=storage.plan_store,
                execution_repository=storage.execution_repository,
                run_history=storage.run_history,
            )
            run_service = TestCaseExecutionService(
                storage.test_case_repository,
                storage.plan_store,
                storage.execution_repository,
                storage.run_history,
                evidence_directory=evidence_root,
                automation_workflow=AutomationWorkflow(pipeline),
            )
            application = LocalWebApplication(
                storage.run_history,
                evidence_root=evidence_root,
                test_cases=storage.test_case_repository,
                run_service=run_service,
                authoring_service=TestCaseAuthoringService(LLMRouter([provider])),
            )
            server = create_http_server(application, host="127.0.0.1", port=0)
            port = server.server_address[1]
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            base_url = f"http://127.0.0.1:{port}/demo-target/registration"
            try:
                created = application.handle(
                    "POST",
                    "/test-cases/generate",
                    urlencode({
                        "name": "Local registration AI test",
                        "base_url": base_url,
                        "scenario": "Register a user and verify the account confirmation.",
                    }),
                )
                self.assertEqual(created.status, 303)
                token = created.headers["Location"].rsplit("/", 1)[1]
                saved = application.handle(
                    "POST", f"/test-cases/review/{token}/save", b""
                )
                self.assertEqual(saved.status, 303)
                case_id = UUID(saved.headers["Location"].rsplit("/", 1)[1])

                run_response = application.handle(
                    "POST", f"/test-cases/{case_id}/run", b"workflow=AUTOMATION"
                )
                self.assertEqual(run_response.status, 303)
                run_id = UUID(run_response.headers["Location"].rsplit("/", 1)[1])

                case = storage.test_case_repository.get(case_id)
                self.assertEqual(len(case.steps), 4)
                self.assertEqual(provider.plan_calls, 4)
                self.assertEqual(run_service.workflow_availability(case_id).usable_plan_count, 4)
                first_version = storage.plan_store.find(case.steps[0].id)
                first_plan = storage.plan_store.find_test_plan(case.steps[0].id)
                self.assertEqual(first_plan.test_step_id, case.steps[0].id)
                self.assertEqual(first_version.qa_test_plan.url, base_url)
                self.assertEqual(
                    first_version.qa_test_plan.steps[0].parameters["url"],
                    base_url,
                )
                history = storage.run_history.list_for_test_case(case_id)
                self.assertEqual(len(history), 1)
                record = history[0]
                self.assertEqual(record.run_id, run_id)
                self.assertEqual(record.workflow_type, WorkflowType.AUTOMATION)
                self.assertEqual(record.outcome, "PRODUCT_FAILURE")
                self.assertEqual(record.status, ExecutionStatus.FAILED)
                self.assertEqual(
                    [step.status for step in record.steps],
                    [
                        ExecutionStatus.PASSED,
                        ExecutionStatus.PASSED,
                        ExecutionStatus.PASSED,
                        ExecutionStatus.FAILED,
                    ],
                )

                detail = storage.run_history.get_detail(run_id)
                first_execution = detail.executions[
                    next(item.execution_id for item in record.executions
                         if item.test_step_id == case.steps[0].id)
                ]
                self.assertIsNone(first_execution.error)
                self.assertEqual(first_execution.status, ExecutionStatus.PASSED)
                failed_execution = detail.executions[
                    next(item.execution_id for item in record.executions
                         if item.test_step_id == case.steps[3].id)
                ]
                self.assertEqual(failed_execution.status, ExecutionStatus.FAILED)
                self.assertEqual(
                    PlanExecutionService._classify(failed_execution, None),
                    PlanExecutionClassification.PRODUCT_FAILURE,
                )
                self.assertIn(
                    "Account created successfully",
                    failed_execution.error,
                )
                self.assertNotIn("NoneType", failed_execution.error)
                self.assertTrue(failed_execution.evidence)
                screenshot_path = Path(failed_execution.evidence[0].path)
                self.assertTrue(screenshot_path.is_file())
                self.assertTrue(screenshot_path.is_relative_to(evidence_root.resolve()))

                report = RunReportGenerator().generate_history(detail)
                report_json = json.loads(report.to_json())
                self.assertEqual(report_json["outcome"], "PRODUCT_FAILURE")
                self.assertEqual(
                    [step["status"] for step in report_json["steps"]],
                    ["PASSED", "PASSED", "PASSED", "FAILED"],
                )
                self.assertIn("PRODUCT FAILURE", RunReportGenerator().to_html(report))

                evidence = application.handle(
                    "GET", f"/runs/{run_id}/evidence/{failed_execution.id}/0"
                )
                self.assertEqual(evidence.status, 200)
                self.assertEqual(evidence.content_type, "image/png")
                self.assertTrue(evidence.body.startswith(b"\x89PNG\r\n\x1a\n"))
            finally:
                server.shutdown()
                server.server_close()
                server_thread.join(timeout=5)


class _Setup:
    def __init__(self, callback):
        self.callback = callback

    def execute(self, precondition, run_context, register_cleanup):
        return self.callback(precondition, run_context, register_cleanup)


if __name__ == "__main__":
    unittest.main()
