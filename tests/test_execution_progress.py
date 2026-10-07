import json
import re
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from qa_agent.execution_progress import (
    ExecutionEventType,
    ExecutionProgressReporter,
    ExecutionProgressStore,
    ProgressState,
    ProgressStepState,
    active_execution_progress,
    get_active_execution_progress,
)
from qa_agent.background_execution import BackgroundRunService
from qa_agent.models import TestCase as DomainTestCase, TestStep as DomainTestStep
from qa_agent.pipeline import PipelineStageError
from qa_agent.presentation import failure_message
from qa_agent.run_context import RunContext
from qa_agent.run_history import (
    InMemoryRunHistoryRepository,
    RunHistoryService,
    WorkflowType,
)
from qa_agent.test_case_execution import RunUnavailableError
from qa_agent.test_plan_validation import PlanValidationIssue
from qa_agent.setup_orchestration import SetupCleanupCoordinator
from qa_agent.web import LocalWebApplication


class ExecutionProgressStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 10, 1, tzinfo=timezone.utc)
        self.store = ExecutionProgressStore(clock=lambda: self.now)
        self.test_case_id = uuid4()
        self.progress_id = self.store.create(self.test_case_id, WorkflowType.AUTOMATION)
        self.reporter = ExecutionProgressReporter(self.store, self.progress_id)

    def test_setup_cleanup_coordinator_emits_real_outcome_events(self) -> None:
        test_case = DomainTestCase(
            name="Needs a prepared account",
            description="Use a setup operation that is not registered.",
            preconditions=[{
                "description": "A prepared account is available.",
                "order": 0,
            }],
            steps=[DomainTestStep(
                name="Open account page",
                description="Open the account page.",
                expected="The account page loads.",
                order=0,
            )],
        )
        with active_execution_progress(self.reporter):
            outcome = SetupCleanupCoordinator({}).run(
                test_case,
                RunContext(),
                lambda _context: self.fail("Product execution must not start after setup fails."),
            )

        event_types = [event.event_type for event in self.store.get(self.progress_id).events]
        self.assertFalse(outcome.setup.succeeded)
        self.assertFalse(outcome.product_started)
        self.assertEqual(
            event_types,
            [
                ExecutionEventType.SETUP_STARTED,
                ExecutionEventType.SETUP_FAILED,
                ExecutionEventType.CLEANUP_STARTED,
                ExecutionEventType.CLEANUP_SUCCEEDED,
            ],
        )

    def test_records_ordered_lifecycle_and_final_run_association(self) -> None:
        initial = self.store.get(self.progress_id)
        self.assertEqual(initial.state, ProgressState.QUEUED)
        self.assertRegex(self.progress_id, re.compile(r"^[A-Za-z0-9_-]{32}$"))

        test_case = DomainTestCase(
            id=self.test_case_id,
            name="Registration",
            description="Register a user.",
            steps=[DomainTestStep(name="Submit registration", description="Submit the form.", expected="It succeeds.", order=0)],
        )
        self.reporter.emit(ExecutionEventType.RUN_STARTED, message="Run started.")
        self.reporter.test_case_loaded(test_case)
        step = test_case.steps[0]
        self.now += timedelta(seconds=2)
        self.reporter.emit(
            ExecutionEventType.PLAN_REUSED,
            step=step,
            plan_origin="AI_GENERATED",
            plan_version=3,
            message="Saved automation loaded.",
        )
        self.reporter.emit(ExecutionEventType.STEP_STARTED, step=step, message="Running step.")
        snapshot = self.store.get(self.progress_id)
        self.assertEqual(snapshot.state, ProgressState.RUNNING)
        self.assertEqual(snapshot.steps[0].state, ProgressStepState.RUNNING)
        self.assertEqual(snapshot.elapsed_ms, 2000)
        self.reporter.emit(
            ExecutionEventType.STEP_PASSED,
            step=step,
            status="PASSED",
            classification="PASSED",
            message="Step passed.",
        )
        self.reporter.emit(
            ExecutionEventType.EVIDENCE_CAPTURED,
            step=step,
            evidence_execution_id=uuid4(),
            evidence_index=0,
            message="Screenshot evidence captured.",
        )
        run_id = uuid4()
        self.reporter.finish(
            run_id=run_id,
            run_status="PASSED",
            outcome="PASSED",
        )

        finished = self.store.get(self.progress_id)
        self.assertEqual(finished.state, ProgressState.FINISHED)
        self.assertEqual(finished.final_run_id, run_id)
        self.assertEqual(finished.final_run_url, f"/runs/{run_id}")
        self.assertEqual(finished.steps[0].state, ProgressStepState.PASSED)
        self.assertEqual(finished.steps[0].execution_state, ProgressStepState.PASSED)
        self.assertEqual(finished.steps[0].automation_state, "Reused")
        self.assertEqual(finished.steps[0].plan_origin, "AI_GENERATED")
        self.assertEqual(finished.steps[0].plan_version, 3)
        self.assertIsNone(finished.steps[0].failure_classification)
        self.assertEqual(finished.steps[0].evidence_count, 1)
        self.assertEqual(
            [event.event_type for event in finished.events],
            [
                ExecutionEventType.RUN_STARTED,
                ExecutionEventType.TESTCASE_LOADED,
                ExecutionEventType.PLAN_REUSED,
                ExecutionEventType.STEP_STARTED,
                ExecutionEventType.STEP_PASSED,
                ExecutionEventType.EVIDENCE_CAPTURED,
                ExecutionEventType.RUN_FINISHED,
            ],
        )
        timestamps = [event.timestamp for event in finished.events]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(len({event.id for event in finished.events}), len(finished.events))

    def test_terminal_generation_failure_preserves_step_details_and_marks_remaining_steps(self) -> None:
        test_case = DomainTestCase(
            id=self.test_case_id,
            name="Partial automation",
            description="Prepare a multi-step flow.",
            steps=[
                DomainTestStep(name=f"Step {index + 1}", description="Do one action.", expected="It works.", order=index)
                for index in range(5)
            ],
        )
        self.reporter.emit(ExecutionEventType.RUN_STARTED, message="Run started.")
        self.reporter.test_case_loaded(test_case)
        for step in test_case.steps[:2]:
            self.reporter.emit(ExecutionEventType.PLAN_GENERATION_STARTED, step=step)
            self.reporter.emit(ExecutionEventType.PLAN_GENERATED, step=step)
            self.reporter.emit(ExecutionEventType.STEP_STARTED, step=step)
            self.reporter.emit(ExecutionEventType.STEP_PASSED, step=step)
        failed_step = test_case.steps[2]
        self.reporter.emit(ExecutionEventType.PLAN_GENERATION_STARTED, step=failed_step)
        self.reporter.emit(
            ExecutionEventType.PLAN_GENERATION_FAILED,
            step=failed_step,
            classification="AUTOMATION_GENERATION_ERROR",
            failure_code="LLM_PROVIDER_FAILURE",
            prior_plan_exists=False,
            new_plan_saved=False,
            message="The automation provider could not return a usable plan.",
        )
        self.reporter.finish(
            outcome="AUTOMATION_GENERATION_ERROR",
            error_category="AUTOMATION_GENERATION_ERROR",
        )
        self.reporter.finish(outcome="PASSED", run_status="PASSED")

        snapshot = self.store.get(self.progress_id)
        self.assertEqual(snapshot.state, ProgressState.FINISHED)
        self.assertEqual(snapshot.phase, "Finished")
        self.assertIsNone(snapshot.run_status)
        self.assertEqual(
            [step.state for step in snapshot.steps],
            [
                ProgressStepState.PASSED,
                ProgressStepState.PASSED,
                ProgressStepState.FAILED,
                ProgressStepState.NOT_ATTEMPTED,
                ProgressStepState.NOT_ATTEMPTED,
            ],
        )
        failure = snapshot.automation_generation_failure
        self.assertIsNotNone(failure)
        self.assertEqual(failure.test_case_id, self.test_case_id)
        self.assertEqual(failure.workflow_type, WorkflowType.AUTOMATION.value)
        self.assertEqual(failure.step_id, failed_step.id)
        self.assertEqual(failure.step_order, 2)
        self.assertEqual(failure.step_name, "Step 3")
        self.assertFalse(failure.prior_plan_exists)
        self.assertFalse(failure.new_plan_saved)
        self.assertEqual(failure.failure_category, "AUTOMATION_GENERATION_ERROR")
        self.assertEqual(failure.technical_classification, "LLM_PROVIDER_FAILURE")
        self.assertEqual(
            [event.event_type for event in snapshot.events].count(ExecutionEventType.RUN_FINISHED),
            1,
        )
        with self.assertRaises(RuntimeError):
            self.reporter.emit(ExecutionEventType.STEP_STARTED, step=test_case.steps[3])

    def test_step_cannot_pass_before_it_starts(self) -> None:
        test_case = DomainTestCase(
            id=self.test_case_id,
            name="Flow",
            description="Check one step.",
            steps=[DomainTestStep(name="Open", description="Open it.", expected="It opens.", order=0)],
        )
        self.reporter.test_case_loaded(test_case)
        with self.assertRaisesRegex(ValueError, "before it has started"):
            self.reporter.emit(ExecutionEventType.STEP_PASSED, step=test_case.steps[0])
        self.assertEqual(
            [event.event_type for event in self.store.get(self.progress_id).events],
            [ExecutionEventType.TESTCASE_LOADED],
        )

    def test_terminal_finalizer_clears_nonterminal_run_status(self) -> None:
        self.reporter.finish(outcome="PASSED", run_status="RUNNING")
        snapshot = self.store.get(self.progress_id)
        self.assertEqual(snapshot.state, ProgressState.FINISHED)
        self.assertEqual(snapshot.phase, "Finished")
        self.assertEqual(snapshot.outcome, "EXECUTION_ERROR")
        self.assertEqual(snapshot.error_category, "EXECUTION_ERROR")
        self.assertIsNone(snapshot.run_status)

    def test_finished_progress_expires_and_oldest_finished_records_are_pruned(self) -> None:
        store = ExecutionProgressStore(
            finished_ttl=timedelta(seconds=10),
            max_finished_records=2,
            clock=lambda: self.now,
        )
        progress_ids = []
        for _ in range(3):
            progress_id = store.create(uuid4(), WorkflowType.VALIDATION)
            progress_ids.append(progress_id)
            ExecutionProgressReporter(store, progress_id).finish(outcome="PASSED")
            self.now += timedelta(seconds=1)

        self.assertIsNone(store.get(progress_ids[0]))
        self.assertIsNotNone(store.get(progress_ids[1]))
        self.assertIsNotNone(store.get(progress_ids[2]))
        self.now += timedelta(seconds=10)
        self.assertIsNone(store.get(progress_ids[2]))

    def test_concurrent_appends_keep_events_unique_and_timestamp_ordered(self) -> None:
        def append_many(_worker: int) -> None:
            reporter = ExecutionProgressReporter(self.store, self.progress_id)
            for _ in range(50):
                reporter.emit(
                    ExecutionEventType.AUTOMATION_PREPARATION_STARTED,
                    message="Preparing automation.",
                )

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(append_many, range(8)))

        events = self.store.get(self.progress_id).events
        self.assertEqual(len(events), 400)
        self.assertEqual(len({event.id for event in events}), 400)
        self.assertEqual([event.timestamp for event in events], sorted(event.timestamp for event in events))


class ExecutionProgressWebTests(unittest.TestCase):
    def test_progress_routes_are_allowlisted_and_escape_testcase_text(self) -> None:
        history = RunHistoryService(InMemoryRunHistoryRepository())
        store = ExecutionProgressStore()
        app = LocalWebApplication(history, progress_store=store)
        secret = "demo-sensitive-marker-4831"
        context = RunContext()
        context.set_value("access_token", secret, sensitive=True)
        test_case_id = uuid4()
        progress_id = store.create(test_case_id, WorkflowType.AUTOMATION)
        reporter = ExecutionProgressReporter(store, progress_id, context)
        case = DomainTestCase(
            id=test_case_id,
            name=f"<script>{secret}</script> C:\\Users\\demo\\private.txt",
            description="Safe test case.",
            steps=[DomainTestStep(
                name=f"Enter {secret} from /home/qa/private-key.txt",
                description="Use a sensitive value.",
                expected="The value is accepted.",
                order=0,
            )],
        )
        reporter.test_case_loaded(case)
        reporter.emit(ExecutionEventType.STEP_STARTED, step=case.steps[0])
        try:
            progress_page = app.handle("GET", f"/runs/progress/{progress_id}")
            script = app.handle("GET", "/assets/ui.js")
            response = app.handle("GET", f"/api/progress/{progress_id}")
            payload = json.loads(response.body)
            rendered_page = progress_page.body.decode("utf-8")
            serialized_snapshot = json.dumps(payload)

            self.assertEqual(progress_page.status, 200)
            self.assertIn("&lt;script&gt;", rendered_page)
            self.assertNotIn("<script>demo-sensitive-marker", rendered_page)
            self.assertNotIn(secret, rendered_page)
            self.assertNotIn("C:\\Users\\demo", rendered_page)
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["steps"][0]["state"], "RUNNING")
            self.assertNotIn(secret, serialized_snapshot)
            self.assertNotIn("/home/qa", serialized_snapshot)
            self.assertNotIn("Traceback", serialized_snapshot)
            self.assertNotIn("path", serialized_snapshot)
            self.assertIn("button.disabled = true", script.body.decode("utf-8"))
            self.assertIn("window.setTimeout(poll, 750)", script.body.decode("utf-8"))

            self.assertEqual(app.handle("GET", "/api/progress/not-a-real-id").status, 404)
            missing_page = app.handle("GET", "/runs/progress/not-a-real-id")
            self.assertEqual(missing_page.status, 404)
            self.assertIn(b"Execution progress is no longer available.", missing_page.body)
        finally:
            app.close()

    def test_retry_is_offered_for_operational_errors_but_not_product_failures(self) -> None:
        store = ExecutionProgressStore()
        app = LocalWebApplication(
            RunHistoryService(InMemoryRunHistoryRepository()),
            progress_store=store,
        )
        try:
            product_id = store.create(uuid4(), WorkflowType.REGRESSION)
            ExecutionProgressReporter(store, product_id).finish(
                outcome="PRODUCT_FAILURE",
                error_category="PRODUCT_FAILURE",
            )
            product_page = app.handle("GET", f"/runs/progress/{product_id}").body.decode()
            self.assertIn("PRODUCT FAILURE", product_page)
            self.assertNotIn("Retry Run", product_page)

            infra_id = store.create(uuid4(), WorkflowType.VALIDATION)
            ExecutionProgressReporter(store, infra_id).finish(
                outcome="INFRASTRUCTURE_ERROR",
                error_category="INFRASTRUCTURE_ERROR",
            )
            infra_page = app.handle("GET", f"/runs/progress/{infra_id}").body.decode()
            self.assertIn("Retry Run", infra_page)
            self.assertIn('name="workflow" value="VALIDATION"', infra_page)
        finally:
            app.close()

    def test_finished_progress_page_renders_failure_details_and_refreshes_execution_state(self) -> None:
        store = ExecutionProgressStore()
        test_case_id = uuid4()
        progress_id = store.create(test_case_id, WorkflowType.AUTOMATION)
        reporter = ExecutionProgressReporter(store, progress_id)
        case = DomainTestCase(
            id=test_case_id,
            name="Account setup",
            description="Create an account.",
            steps=[
                DomainTestStep(name="Open form", description="Open the form.", expected="It opens.", order=0),
                DomainTestStep(name="Submit registration", description="Submit the form.", expected="It succeeds.", order=1),
                DomainTestStep(name="Verify result", description="Check the result.", expected="It is shown.", order=2),
            ],
        )
        reporter.emit(ExecutionEventType.RUN_STARTED)
        reporter.test_case_loaded(case)
        first = case.steps[0]
        reporter.emit(ExecutionEventType.PLAN_GENERATION_STARTED, step=first)
        reporter.emit(ExecutionEventType.PLAN_GENERATED, step=first)
        reporter.emit(ExecutionEventType.STEP_STARTED, step=first)
        reporter.emit(ExecutionEventType.STEP_PASSED, step=first)
        failed = case.steps[1]
        reporter.emit(ExecutionEventType.PLAN_GENERATION_STARTED, step=failed)
        reporter.emit(
            ExecutionEventType.PLAN_GENERATION_FAILED,
            step=failed,
            classification="AUTOMATION_GENERATION_ERROR",
            failure_code="PLAN_VALIDATION_FAILED",
            prior_plan_exists=False,
            new_plan_saved=False,
            message="Generated automation failed validation.",
            validation_issues=(PlanValidationIssue(
                code="MISSING_LOCATOR",
                path="steps[0].parameters.selector",
                message="CLICK action requires a locator.",
            ),),
        )
        reporter.finish(outcome="AUTOMATION_GENERATION_ERROR", error_category="AUTOMATION_GENERATION_ERROR")
        app = LocalWebApplication(
            RunHistoryService(InMemoryRunHistoryRepository()),
            progress_store=store,
        )
        try:
            page = app.handle("GET", f"/runs/progress/{progress_id}").body.decode("utf-8")
            script = app.handle("GET", "/assets/ui.js").body.decode("utf-8")
            payload = json.loads(app.handle("GET", f"/api/progress/{progress_id}").body)
            self.assertIn('class="card-label">Execution</div>', page)
            self.assertIn('data-progress-state>Finished</span>', page)
            self.assertIn("AUTOMATION GENERATION ERROR", page)
            self.assertIn("Automation stopped at Step 2 of 3", page)
            self.assertIn("Submit registration", page)
            self.assertIn("Generated successfully: 1 steps", page)
            self.assertIn("Remaining: 1 steps not attempted", page)
            self.assertIn("Retry Automation", page)
            self.assertIn("PLAN_VALIDATION_FAILED", page)
            self.assertIn("MISSING_LOCATOR", page)
            self.assertIn("steps[0].parameters.selector", page)
            self.assertIn("CLICK action requires a locator.", page)
            self.assertIn("Developer details", page)
            self.assertIn('data-progress-events="all"', page)
            self.assertNotIn('<h2>Preparation</h2>', page)
            self.assertIn("data-progress-state", script)
            self.assertIn("snapshot.state.toLowerCase()", script)
            self.assertEqual(payload["state"], "FINISHED")
            self.assertEqual(payload["phase"], "Finished")
            self.assertEqual(payload["run_status"], None)
            self.assertEqual(payload["steps"][2]["state"], "NOT_ATTEMPTED")
            self.assertEqual(payload["steps"][2]["execution_state"], "NOT_ATTEMPTED")
            self.assertEqual(payload["automation_generation_failure"]["step_name"], "Submit registration")
            self.assertEqual(
                payload["automation_generation_failure"]["validation_issues"][0]["code"],
                "MISSING_LOCATOR",
            )
            self.assertNotIn("Traceback", json.dumps(payload))
        finally:
            app.close()


class BackgroundExecutionTests(unittest.TestCase):
    def _run_until_finished(self, service, test_case_id, workflow):
        progress_id = service.start(test_case_id, workflow)
        deadline = time.monotonic() + 2
        snapshot = None
        while time.monotonic() < deadline:
            snapshot = service.progress_store.get(progress_id)
            if snapshot is not None and snapshot.state == ProgressState.FINISHED:
                return snapshot
            time.sleep(0.01)
        self.fail("Background workflow did not finish in time.")

    def test_background_result_preserves_product_failure_and_exact_workflow(self) -> None:
        for category in (
            "PRODUCT_FAILURE",
            "AUTOMATION_DRIFT",
            "INFRASTRUCTURE_ERROR",
            "SETUP_FAILURE",
        ):
            with self.subTest(category=category):
                test_case_id = uuid4()
                run_id = uuid4()

                class FakeRunService:
                    def __init__(self):
                        self.calls = []

                    def run(self, case_id, workflow):
                        self.calls.append((case_id, workflow))
                        return type("Result", (), {
                            "test_run": type("TestRun", (), {"id": run_id, "status": "FAILED"})(),
                            "outcome": category,
                        })()

                run_service = FakeRunService()
                service = BackgroundRunService(
                    run_service,
                    RunHistoryService(InMemoryRunHistoryRepository()),
                )
                try:
                    snapshot = self._run_until_finished(
                        service, test_case_id, WorkflowType.REGRESSION
                    )
                    self.assertEqual(run_service.calls, [(test_case_id, WorkflowType.REGRESSION)])
                    self.assertEqual(snapshot.final_run_id, run_id)
                    self.assertEqual(snapshot.run_status, "FAILED")
                    self.assertEqual(snapshot.outcome, category)
                    self.assertEqual(snapshot.error_category, category)
                    self.assertEqual(snapshot.error_message, failure_message(category))
                    self.assertEqual(snapshot.events[0].event_type, ExecutionEventType.RUN_REQUESTED)
                    self.assertEqual(snapshot.events[1].event_type, ExecutionEventType.RUN_STARTED)
                    self.assertEqual(snapshot.events[-1].event_type, ExecutionEventType.RUN_FINISHED)
                finally:
                    service.close()

    def test_pipeline_and_unexpected_errors_become_safe_finished_states(self) -> None:
        failures = (
            (PipelineStageError("plan generation", "provider response contained secret"), "AUTOMATION_GENERATION_ERROR"),
            (RunUnavailableError("missing plans details", category="MISSING_AUTOMATION"), "MISSING_AUTOMATION"),
            (RuntimeError("NoneType traceback and secret value"), "EXECUTION_ERROR"),
        )
        for error, category in failures:
            with self.subTest(category=category):
                class FailingRunService:
                    def __init__(self, failure):
                        self.failure = failure

                    def run(self, _case_id, _workflow):
                        raise self.failure

                service = BackgroundRunService(
                    FailingRunService(error),
                    RunHistoryService(InMemoryRunHistoryRepository()),
                )
                try:
                    snapshot = self._run_until_finished(
                        service, uuid4(), WorkflowType.AUTOMATION
                    )
                    serialized = json.dumps(snapshot.to_public_dict())
                    self.assertEqual(snapshot.error_category, category)
                    self.assertEqual(snapshot.error_message, failure_message(category))
                    self.assertNotIn("NoneType", serialized)
                    self.assertNotIn("secret", serialized)
                    self.assertIsNone(snapshot.final_run_id)
                finally:
                    service.close()

    def test_context_bound_reporter_is_scoped_to_background_worker(self) -> None:
        observed = []

        class FakeRunService:
            def run(self, _case_id, _workflow):
                observed.append(get_active_execution_progress())
                return type("Result", (), {"test_run": None, "outcome": "PASSED"})()

        service = BackgroundRunService(
            FakeRunService(),
            RunHistoryService(InMemoryRunHistoryRepository()),
        )
        try:
            self._run_until_finished(service, uuid4(), WorkflowType.VALIDATION)
            self.assertEqual(len(observed), 1)
            self.assertIsNotNone(observed[0])
        finally:
            service.close()

    def test_background_capacity_is_bounded_and_busy_runs_finish_safely(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        class BlockingRunService:
            def run(self, _case_id, _workflow):
                entered.set()
                release.wait(timeout=3)
                return type("Result", (), {"test_run": None, "outcome": "PASSED"})()

        service = BackgroundRunService(
            BlockingRunService(),
            RunHistoryService(InMemoryRunHistoryRepository()),
            max_workers=1,
            max_pending=0,
        )
        try:
            first_id = service.start(uuid4(), WorkflowType.AUTOMATION)
            self.assertTrue(entered.wait(timeout=2))
            second_id = service.start(uuid4(), WorkflowType.AUTOMATION)
            rejected = service.progress_store.get(second_id)
            self.assertEqual(rejected.state, ProgressState.FINISHED)
            self.assertEqual(rejected.error_category, "EXECUTION_ERROR")
            self.assertEqual(
                rejected.error_message,
                "The local run queue is busy. Try again shortly.",
            )
            self.assertEqual(service.progress_store.get(first_id).state, ProgressState.RUNNING)
        finally:
            release.set()
            service.close()


if __name__ == "__main__":
    unittest.main()
