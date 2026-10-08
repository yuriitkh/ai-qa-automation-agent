import json
import sqlite3
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode
from uuid import UUID

from qa_agent.models import (
    AssertionGrounding,
    AssertionGroundingEntry,
    DiscoveryResult,
    DiscoveryStatus,
    ExecutionSegment,
    ExecutionStatus,
    Precondition,
    QATestPlan,
    QATestStep,
    PlanVersionOrigin,
    RunContext,
    TestCase as DomainTestCase,
    TestPlan as DomainTestPlan,
    TestPlanVersion as DomainTestPlanVersion,
    TestStep as DomainTestStep,
    FailurePolicy,
)
from qa_agent.execution_repository import InMemoryExecutionRepository
from qa_agent.browser_runner import BrowserRunner
from qa_agent.llm.base import LLMProvider
from qa_agent.llm.router import LLMRouter
from qa_agent.pipeline import QATestPipeline
from qa_agent.plan_execution import PlanExecutionClassification, PlanExecutionService
from qa_agent.pinned_execution import PinnedExecutionService, PlanVersionSet, StepPlanSelection
from qa_agent.plan_store import InMemoryPlanStore
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
from qa_agent.test_plan_generator import GeneratedTestPlan, LLMTestPlanGenerator, TestPlanGenerator
from qa_agent.web import LocalWebApplication, create_http_server
from qa_agent.workflows import AutomationWorkflow, RegressionWorkflow, ValidationWorkflow


def _wait_for_progress(application, location: str, timeout: float = 60.0) -> dict:
    progress_id = location.rsplit("/", 1)[1]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = application.handle("GET", f"/api/progress/{progress_id}")
        if response.status != 200:
            raise AssertionError(response.body.decode("utf-8", errors="replace"))
        snapshot = json.loads(response.body)
        if snapshot["finished"]:
            return snapshot
        time.sleep(0.05)
    raise AssertionError(f"Execution progress {progress_id} did not finish in time.")


def _wait_for_authoring(application, location: str, timeout: float = 30.0) -> dict:
    progress_id = location.rsplit("/", 1)[1]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = application.handle(
            "GET", f"/api/test-cases/authoring-progress/{progress_id}"
        )
        if response.status != 200:
            raise AssertionError(response.body.decode("utf-8", errors="replace"))
        snapshot = json.loads(response.body)
        if snapshot["finished"]:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"Authoring progress {progress_id} did not finish in time.")


class ProductDemoSliceTests(unittest.TestCase):
    def test_test_case_browser_state_continues_across_steps_and_segments(self) -> None:
        class StatefulTargetHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/state/set":
                    body = (
                        '<!doctype html><html><head><title>State Ready</title>'
                        '<script>window.addEventListener("load", () => '
                        'localStorage.setItem("case-state", "persisted"));</script>'
                        '</head><body><main><p>Ready</p></main></body></html>'
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Set-Cookie", "segment_cookie=present; Path=/; SameSite=Lax")
                elif self.path == "/state/check":
                    body = (
                        '<!doctype html><html><head><title>State Check</title>'
                        '<script>window.addEventListener("load", () => {'
                        'const found = localStorage.getItem("case-state") === "persisted" '
                        '&& document.cookie.includes("segment_cookie=present");'
                        'document.querySelector("#result").textContent = found '
                        '? "Cookie and localStorage persisted" : "No prior state";'
                        '});</script></head><body><p id="result">Checking</p></body></html>'
                    ).encode("utf-8")
                    self.send_response(200)
                else:
                    self.send_error(404)
                    return
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        def step(name: str, order: int, failure_policy=FailurePolicy.CONTINUE):
            return DomainTestStep(
                name=name,
                description=f"Execute {name.lower()}.",
                expected=f"{name} completes.",
                order=order,
                failure_policy=failure_policy,
            )

        def save_plan(store, target_url, test_step, actions):
            test_plan = DomainTestPlan(test_step_id=test_step.id, name=test_step.name)
            version = DomainTestPlanVersion(
                test_plan_id=test_plan.id,
                version=1,
                qa_test_plan=QATestPlan(url=target_url, steps=actions),
            )
            store.save(test_step.id, version, test_plan=test_plan)
            return version

        class NeverGenerate:
            def generate_with_plan(self, *_args, **_kwargs):
                raise AssertionError("Cached and pinned plans must not be regenerated.")

        target_server = ThreadingHTTPServer(("127.0.0.1", 0), StatefulTargetHandler)
        target_url = f"http://127.0.0.1:{target_server.server_address[1]}"
        target_thread = threading.Thread(target=target_server.serve_forever, daemon=True)
        target_thread.start()
        store = InMemoryPlanStore()
        browser = BrowserRunner(headless=True)
        try:
            with tempfile.TemporaryDirectory() as evidence_dir:
                automation_steps = [
                    step("Set initial state", 0),
                    step("Observe an intentional product failure", 1),
                    step("Continue on the resulting page", 2),
                    step("Read state from a second segment", 3),
                ]
                automation_case = DomainTestCase(
                    name="Stateful automation",
                    description="Continue through one shared browser context.",
                    base_url=target_url,
                    segments=[
                        ExecutionSegment(order=0, base_url=target_url, steps=automation_steps[:3]),
                        ExecutionSegment(order=1, base_url=target_url, steps=automation_steps[3:]),
                    ],
                )
                action_sets = [
                    [{"action": "navigate", "parameters": {"url": f"{target_url}/state/set"}}],
                    [{"action": "assert_title", "parameters": {"expected": "Intentional failure"}}],
                    [{"action": "assert_text_contains", "parameters": {"expected_text": "Ready"}}],
                    [
                        {"action": "navigate", "parameters": {"url": f"{target_url}/state/check"}},
                        {"action": "assert_text_contains", "parameters": {"expected_text": "Cookie and localStorage persisted"}},
                    ],
                ]
                for test_step, actions in zip(automation_steps, action_sets):
                    save_plan(store, target_url, test_step, actions)
                pipeline = QATestPipeline(
                    decomposer=TestCaseDecomposer(),
                    plan_generator=NeverGenerate(),
                    discovery=lambda _url: self.fail("Saved automation must not rediscover."),
                    runner=BrowserRunner(evidence_dir, headless=True),
                    plan_store=store,
                    execution_repository=InMemoryExecutionRepository(),
                )
                automation_result = pipeline.run_test_case(automation_case)
                self.assertEqual(
                    [execution.status for execution in automation_result.executions],
                    [ExecutionStatus.PASSED, ExecutionStatus.FAILED,
                     ExecutionStatus.PASSED, ExecutionStatus.PASSED],
                )
                self.assertEqual(len(automation_result.executions[1].evidence), 1)

                pinned_steps = [step("Open the persisted state", 0), step("Read it on the same page", 1)]
                pinned_case = DomainTestCase(
                    name="Stateful pinned validation",
                    description="Run exact pinned versions with shared page state.",
                    base_url=target_url,
                    segments=[ExecutionSegment(order=0, base_url=target_url, steps=pinned_steps)],
                )
                pinned_versions = [
                    save_plan(store, target_url, pinned_steps[0], [
                        {"action": "navigate", "parameters": {"url": f"{target_url}/state/set"}},
                    ]),
                    save_plan(store, target_url, pinned_steps[1], [
                        {"action": "assert_text_contains", "parameters": {"expected_text": "Ready"}},
                    ]),
                ]
                pinned = PinnedExecutionService(
                    store,
                    PlanExecutionService(browser, InMemoryExecutionRepository()),
                )
                resolved = pinned.resolve(
                    pinned_case,
                    PlanVersionSet(tuple(
                        StepPlanSelection(test_step.id, version.id)
                        for test_step, version in zip(pinned_steps, pinned_versions)
                    )),
                )
                pinned_result = pinned.execute(pinned_case, resolved, RunContext())
                self.assertEqual(pinned_result.outcome.value, "PASSED")
                self.assertEqual(len(pinned_result.executions), 2)
                self.assertTrue(all(item.status == ExecutionStatus.PASSED for item in pinned_result.executions))

                isolated_step = step("Check that prior case state is absent", 0)
                isolated_case = DomainTestCase(
                    name="Isolated browser context",
                    description="A separate TestCase starts without the prior context state.",
                    base_url=target_url,
                    segments=[ExecutionSegment(order=0, base_url=target_url, steps=[isolated_step])],
                )
                save_plan(store, target_url, isolated_step, [
                    {"action": "navigate", "parameters": {"url": f"{target_url}/state/check"}},
                    {"action": "assert_text_contains", "parameters": {"expected_text": "No prior state"}},
                ])
                isolated_result = pipeline.run_test_case(isolated_case)
                self.assertEqual(isolated_result.test_run.status, ExecutionStatus.PASSED)
        finally:
            target_server.shutdown()
            target_server.server_close()
            target_thread.join(timeout=5)

    def test_local_registration_step_generates_and_executes_multiple_actions(self) -> None:
        class RegistrationProvider(LLMProvider):
            def __init__(self, target_url: str) -> None:
                self.target_url = target_url
                self.calls = 0
                self.task = ""

            def create_test_plan(self, task, target_url, page_snapshot):
                self.calls += 1
                self.task = task
                return QATestPlan(url=target_url, steps=[
                    {"action": "navigate", "parameters": {"url": target_url}},
                    {"action": "fill", "parameters": {"selector": "#first-name", "value": "Ada"}},
                    {"action": "fill", "parameters": {"selector": "#last-name", "value": "Lovelace"}},
                    {"action": "fill", "parameters": {"selector": "#email", "value": "ada@example.test"}},
                    {"action": "fill", "parameters": {"selector": "#password", "value": "local-test-password"}},
                    {"action": "click", "parameters": {"selector": "#create-account"}},
                    {"action": "assert_text_contains", "parameters": {"expected_text": "Account created successfully"}},
                ])

        class RegistrationTargetHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = (
                    "<!doctype html><html><head><title>Local registration</title></head><body>"
                    '<form method="post" action="/register">'
                    '<label>First name <input id="first-name" name="first_name" required></label>'
                    '<label>Last name <input id="last-name" name="last_name" required></label>'
                    '<label>Email <input id="email" name="email" type="email" required></label>'
                    '<label>Password <input id="password" name="password" type="password" required></label>'
                    '<button id="create-account" type="submit">Create account</button>'
                    "</form></body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = (
                    "<!doctype html><html><head><title>Registration complete</title></head>"
                    "<body><main><p>Account created successfully</p></main></body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        target_server = ThreadingHTTPServer(("127.0.0.1", 0), RegistrationTargetHandler)
        target_url = f"http://127.0.0.1:{target_server.server_address[1]}/register"
        target_thread = threading.Thread(target=target_server.serve_forever, daemon=True)
        target_thread.start()
        provider = RegistrationProvider(target_url)
        step = DomainTestStep(
            name="Enter registration details",
            description="Fill the first name, last name, email, and password, then submit.",
            expected="The account-created confirmation is visible.",
            order=0,
        )
        case = DomainTestCase(
            name="Local registration",
            description=(
                "Create an account on the local registration fixture and verify "
                "the exact confirmation text 'Account created successfully'."
            ),
            base_url=target_url,
            steps=[step],
        )
        discovery = DiscoveryResult(
            status=DiscoveryStatus.SUCCESS,
            url=target_url,
            interactive_elements=[
                {"kind": "input", "tag": "input", "selector": "#first-name", "accessible_name": "First name"},
                {"kind": "input", "tag": "input", "selector": "#last-name", "accessible_name": "Last name"},
                {"kind": "input", "tag": "input", "selector": "#email", "accessible_name": "Email"},
                {"kind": "input", "tag": "input", "selector": "#password", "accessible_name": "Password"},
                {"kind": "button", "tag": "button", "role": "button", "selector": "#create-account", "accessible_name": "Create account"},
            ],
        )
        try:
            pipeline = QATestPipeline(
                decomposer=TestCaseDecomposer(),
                plan_generator=LLMTestPlanGenerator(LLMRouter([provider])),
                discovery=lambda _url: discovery,
                runner=BrowserRunner(headless=True),
            )
            result = pipeline.run_test_case(case)
        finally:
            target_server.shutdown()
            target_server.server_close()
            target_thread.join(timeout=5)

        generated = result.test_plans[0].test_plan_version
        self.assertEqual(provider.calls, 1)
        self.assertIn("multiple ordered executable actions", provider.task)
        self.assertEqual(len(generated.qa_test_plan.steps), 7)
        self.assertEqual(generated.origin, PlanVersionOrigin.AI_GENERATED)
        self.assertEqual(result.test_run.status, ExecutionStatus.PASSED)

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
                description=(
                    "Validate the registration workflow and require exact title "
                    "markers step-0, step-1, and step-2 for the respective checks."
                ),
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
                    assertion_grounding=(AssertionGroundingEntry(
                        step_index=0,
                        category=AssertionGrounding.REQUIREMENT_GROUNDED,
                    ),),
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

    def test_partial_automation_generation_retries_saved_steps_on_local_browser(self) -> None:
        class FailingOnceGenerator(TestPlanGenerator):
            def __init__(self, target_url: str) -> None:
                self.target_url = target_url
                self.calls = 0
                self.failed_once = False

            def generate_with_plan(
                self,
                test_step,
                discovery_result,
                *,
                existing_test_plan=None,
                version_number=1,
            ):
                self.calls += 1
                if test_step.order == 2 and not self.failed_once:
                    self.failed_once = True
                    raise RuntimeError(
                        "provider payload LEAK_MARKER at C:\\private\\provider-response.txt"
                    )
                test_plan = existing_test_plan or DomainTestPlan(
                    test_step_id=test_step.id,
                    name=test_step.name,
                )
                plan = QATestPlan(url=self.target_url, steps=[
                    QATestStep(action="navigate", parameters={"url": self.target_url}),
                    QATestStep(action="fill", parameters={
                        "selector": "#delayed-email",
                        "value": "local-test@example.test",
                    }),
                    QATestStep(action="assert_visible", parameters={
                        "selector": "main p",
                    }),
                ])
                version = DomainTestPlanVersion(
                    test_plan_id=test_plan.id,
                    version=version_number,
                    origin=(
                        PlanVersionOrigin.REGENERATED
                        if existing_test_plan is not None
                        else PlanVersionOrigin.AI_GENERATED
                    ),
                    qa_test_plan=plan,
                )
                return GeneratedTestPlan(test_plan=test_plan, test_plan_version=version)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "partial-automation.sqlite3"
            evidence_root = root / "evidence"
            class LocalTargetHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    body = (
                        "<!doctype html><html><head><title>Local target</title>"
                        "<script>window.addEventListener('load', () => window.setTimeout(() => {"
                        "const input = document.createElement('input'); input.id = 'delayed-email';"
                        "document.body.append(input); }, 150));</script></head>"
                        "<body><main><p>Local-only registration page</p></main></body></html>"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, _format, *_args):
                    return

            target_server = ThreadingHTTPServer(("127.0.0.1", 0), LocalTargetHandler)
            port = target_server.server_address[1]
            target_url = f"http://127.0.0.1:{port}/registration"
            storage = create_sqlite_storage(database)
            steps = [
                DomainTestStep(
                    name=name,
                    description="Open the local page and verify its static content.",
                    expected="The local registration page is visible.",
                    order=index,
                )
                for index, name in enumerate((
                    "Open registration page",
                    "Verify page heading",
                    "Submit registration",
                    "Verify confirmation",
                ))
            ]
            test_case = DomainTestCase(
                name="Partial generation retry",
                description="Exercise retry after one plan generation failure.",
                base_url=target_url,
                steps=steps,
            )
            storage.test_case_repository.save(test_case)
            generator = FailingOnceGenerator(target_url)
            pipeline = QATestPipeline(
                decomposer=TestCaseDecomposer(),
                plan_generator=generator,
                discovery=lambda url: DiscoveryResult(
                    status=DiscoveryStatus.SUCCESS,
                    url=url,
                ),
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
            )
            target_thread = threading.Thread(target=target_server.serve_forever, daemon=True)
            target_thread.start()
            try:
                first_response = application.handle(
                    "POST",
                    f"/test-cases/{test_case.id}/run",
                    "workflow=AUTOMATION",
                )
                first = _wait_for_progress(application, first_response.headers["Location"])
                self.assertEqual(first["state"], "FINISHED")
                self.assertEqual(first["phase"], "Finished")
                self.assertEqual(first["outcome"], "AUTOMATION_GENERATION_ERROR")
                self.assertIsNone(first["final_run_id"])
                self.assertEqual(
                    [step["state"] for step in first["steps"]],
                    ["PASSED", "PASSED", "FAILED", "NOT_ATTEMPTED"],
                )
                failure = first["automation_generation_failure"]
                self.assertEqual(failure["test_case_id"], str(test_case.id))
                self.assertEqual(failure["workflow_type"], "AUTOMATION")
                self.assertEqual(failure["step_id"], str(steps[2].id))
                self.assertEqual(failure["step_order"], 2)
                self.assertEqual(failure["step_name"], "Submit registration")
                self.assertFalse(failure["prior_plan_exists"])
                self.assertFalse(failure["new_plan_saved"])
                self.assertEqual(failure["failure_category"], "AUTOMATION_GENERATION_ERROR")
                self.assertEqual(failure["technical_classification"], "PLAN_GENERATION_FAILED")
                self.assertIn("No reliable executable action", failure["safe_reason"])
                self.assertNotIn("LEAK_MARKER", json.dumps(first))
                self.assertNotIn("C:\\private", json.dumps(first))
                self.assertEqual(len(storage.run_history.list_for_test_case(test_case.id)), 0)
                availability = run_service.workflow_availability(test_case.id)
                self.assertEqual((availability.usable_plan_count, availability.total_step_count), (2, 4))
                self.assertFalse(availability.validation_available)
                self.assertFalse(availability.regression_available)
                testcase_page = application.handle(
                    "GET", f"/test-cases/{test_case.id}"
                ).body.decode("utf-8")
                self.assertIn("2 / 4 steps have executable plans", testcase_page)
                self.assertIn("Validation <strong>Not ready", testcase_page)
                self.assertIn("Regression <strong>Not ready", testcase_page)
                self.assertEqual(
                    application.handle("GET", f"/runs/{UUID(int=0)}/report.json").status,
                    404,
                )

                failed_page = application.handle(
                    "GET", f"/runs/progress/{first['progress_id']}"
                ).body.decode("utf-8")
                self.assertIn("AUTOMATION GENERATION ERROR", failed_page)
                self.assertIn("Automation stopped at Step 3 of 4", failed_page)
                self.assertIn("Submit registration", failed_page)
                self.assertIn("Remaining: 1 steps not attempted", failed_page)
                self.assertIn("Retry Automation", failed_page)
                self.assertNotIn("LEAK_MARKER", failed_page)
                self.assertNotIn("C:\\private", failed_page)

                first_versions = [storage.plan_store.find(step.id) for step in steps[:2]]
                retry_response = application.handle(
                    "POST",
                    f"/test-cases/{test_case.id}/run",
                    "workflow=AUTOMATION",
                )
                retry = _wait_for_progress(application, retry_response.headers["Location"])
                self.assertEqual(retry["test_case_id"], str(test_case.id))
                self.assertEqual(retry["workflow"], "AUTOMATION")
                self.assertEqual(retry["outcome"], "PASSED")
                self.assertEqual(retry["state"], "FINISHED")
                self.assertEqual(generator.calls, 5)
                reused_step_ids = [
                    event["step_id"] for event in retry["events"]
                    if event["type"] == "PLAN_REUSED"
                ]
                self.assertEqual(reused_step_ids, [str(steps[0].id), str(steps[1].id)])
                self.assertEqual(
                    [storage.plan_store.find(step.id).version for step in steps],
                    [1, 1, 1, 1],
                )
                self.assertEqual(
                    [storage.plan_store.find(step.id).id for step in steps[:2]],
                    [version.id for version in first_versions],
                )
                connection = sqlite3.connect(database)
                try:
                    self.assertEqual(
                        connection.execute("SELECT COUNT(*) FROM test_plan_versions").fetchone()[0],
                        4,
                    )
                finally:
                    connection.close()
                records = storage.run_history.list_for_test_case(test_case.id)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0].workflow_type, WorkflowType.AUTOMATION)
                self.assertEqual(records[0].outcome, "PASSED")
                self.assertEqual(
                    [step["automation_state"] for step in retry["steps"]],
                    ["Reused", "Reused", "Generated", "Generated"],
                )
                started_steps = set()
                for event in retry["events"]:
                    if event["type"] == "STEP_STARTED":
                        started_steps.add(event["step_id"])
                    elif event["type"] in {"STEP_PASSED", "STEP_FAILED"}:
                        self.assertIn(event["step_id"], started_steps)
                        started_steps.remove(event["step_id"])
                self.assertEqual(retry["events"][-1]["type"], "RUN_FINISHED")
            finally:
                application.close()
                target_server.shutdown()
                target_server.server_close()
                target_thread.join(timeout=5)

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
                        "scenario": "Register a user and verify the exact text 'Account confirmation displayed'.",
                    }),
                )
                self.assertEqual(created.status, 303)
                authoring = _wait_for_authoring(application, created.headers["Location"])
                self.assertTrue(authoring["success"])
                self.assertEqual(authoring["error_category"], None)
                token = authoring["review_url"].rsplit("/", 1)[1]
                review = application.handle("GET", authoring["review_url"])
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
                self.assertEqual(unavailable.status, 303)
                unavailable_progress = _wait_for_progress(
                    application, unavailable.headers["Location"]
                )
                self.assertEqual(unavailable_progress["error_category"], "MISSING_AUTOMATION")
                self.assertEqual(storage.run_history.list_for_test_case(case_id), [])

                automation = application.handle(
                    "POST", f"/test-cases/{case_id}/run", b"workflow=AUTOMATION"
                )
                self.assertEqual(automation.status, 303)
                automation_progress = _wait_for_progress(
                    application, automation.headers["Location"]
                )
                automation_id = UUID(automation_progress["final_run_id"])
                automation_record = storage.run_history.get(automation_id)
                self.assertEqual(automation_record.workflow_type, WorkflowType.AUTOMATION)
                self.assertEqual(automation_record.status, ExecutionStatus.FAILED)
                automation_event_types = [event["type"] for event in automation_progress["events"]]
                self.assertIn("TESTCASE_LOADED", automation_event_types)
                self.assertIn("STEP_STARTED", automation_event_types)
                self.assertEqual(automation_event_types[-1], "RUN_FINISHED")
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
                regression_progress = _wait_for_progress(
                    application, regression.headers["Location"]
                )
                regression_id = UUID(regression_progress["final_run_id"])
                regression_record = storage.run_history.get(regression_id)
                self.assertEqual(regression_record.workflow_type, WorkflowType.REGRESSION)
                self.assertEqual(regression_record.outcome, "PRODUCT_FAILURE")
                self.assertEqual(len(storage.run_history.list_for_test_case(case_id)), 2)
                self.assertEqual(regression_record.executions[0].test_plan_version_id, plan_version.id)
                run_page = application.handle("GET", f"/runs/{regression_id}")
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
                        "scenario": "Register a user and verify the exact text 'Account created successfully'.",
                    }),
                )
                self.assertEqual(created.status, 303)
                authoring = _wait_for_authoring(application, created.headers["Location"])
                self.assertTrue(authoring["success"])
                token = authoring["review_url"].rsplit("/", 1)[1]
                saved = application.handle(
                    "POST", f"/test-cases/review/{token}/save", b""
                )
                self.assertEqual(saved.status, 303)
                case_id = UUID(saved.headers["Location"].rsplit("/", 1)[1])

                run_response = application.handle(
                    "POST", f"/test-cases/{case_id}/run", b"workflow=AUTOMATION"
                )
                self.assertEqual(run_response.status, 303)
                progress = _wait_for_progress(application, run_response.headers["Location"])
                self.assertIsNotNone(progress["final_run_id"], json.dumps(progress))
                run_id = UUID(progress["final_run_id"])

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
                event_types = [event["type"] for event in progress["events"]]
                self.assertLess(event_types.index("RUN_STARTED"), event_types.index("TESTCASE_LOADED"))
                self.assertIn("PLAN_GENERATION_STARTED", event_types)
                self.assertIn("PLAN_GENERATED", event_types)
                self.assertFalse(any(event_type.startswith("PLAN_REPAIR_") for event_type in event_types))
                self.assertEqual(event_types.count("STEP_STARTED"), 4)
                self.assertIn("EVIDENCE_CAPTURED", event_types)
                self.assertEqual(event_types[-1], "RUN_FINISHED")
                self.assertEqual(
                    [step["state"] for step in progress["steps"]],
                    ["PASSED", "PASSED", "PASSED", "FAILED"],
                )
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
