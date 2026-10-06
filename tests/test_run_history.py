import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from qa_agent.execution_repository import InMemoryExecutionRepository
from qa_agent.models import (
    Execution,
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
from qa_agent.run_history import (
    InMemoryRunHistoryRepository,
    RunHistoryService,
    WorkflowType,
)
from qa_agent.setup_orchestration import (
    CleanupFailure,
    CleanupOutcome,
    PreconditionSetupOutcome,
    SetupRunOutcome,
    SetupStatus,
)
from qa_agent.sqlite_storage import (
    SQLiteExecutionRepository,
    SQLitePlanStore,
    SQLiteRunHistoryRepository,
)


class RunHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.started = datetime(2026, 2, 1, tzinfo=timezone.utc)
        self.steps = [
            self.make_step("Third", 2),
            self.make_step("First", 0),
            self.make_step("Second", 1),
        ]
        self.precondition = Precondition(
            description="A prepared account is available.",
            order=0,
            provided_data_keys=["account_id"],
        )
        self.test_case = DomainTestCase(
            name="Registration flow",
            description="Register a new account and verify the welcome screen.",
            steps=self.steps,
            preconditions=[self.precondition],
        )
        self.secret = "FAKE_RUN_HISTORY_SECRET_8427"
        self.context = RunContext()
        self.context.set_value("password", self.secret, sensitive=True, source="setup")

    @staticmethod
    def make_step(name, order):
        return DomainTestStep(
            name=name,
            description=f"Execute {name}.",
            expected=f"{name} succeeds.",
            order=order,
        )

    def make_execution(self, step, status, offset, *, error=None, actual=None):
        start = self.started + timedelta(seconds=offset)
        return Execution(
            test_step_id=step.id,
            test_plan_version_id=uuid4(),
            status=status,
            started_at=start,
            finished_at=start + timedelta(seconds=2),
            actual_result=actual if actual is not None else status.value,
            error=error,
        )

    def make_run(self, *, started_offset=0):
        first, second, third = sorted(self.steps, key=lambda step: step.order)
        passed = self.make_execution(first, ExecutionStatus.PASSED, started_offset)
        failed = self.make_execution(
            second,
            ExecutionStatus.FAILED,
            started_offset + 3,
            error=f"Failure contained {self.secret}",
            actual=f"Actual result contained {self.secret}",
        )
        run = DomainTestRun.from_test_case(
            self.test_case,
            [passed, failed],
            blocked_step_ids=[third.id],
            run_context=self.context,
        )
        setup = SetupRunOutcome(
            SetupStatus.SUCCEEDED,
            (PreconditionSetupOutcome(
                precondition_id=self.precondition.id,
                status=SetupStatus.SUCCEEDED,
                produced_data_keys=("account_id",),
                cleanup_registered=True,
            ),),
        )
        cleanup = CleanupOutcome((CleanupFailure(
            label="remove account",
            error_type="RuntimeError",
            message=f"Cleanup message contained {self.secret}",
        ),))
        return run, [passed, failed], setup, cleanup

    def test_persist_and_load_safe_run_with_execution_and_step_references(self) -> None:
        repository = InMemoryRunHistoryRepository()
        service = RunHistoryService(repository)
        run, executions, setup, cleanup = self.make_run()
        finished = self.started + timedelta(seconds=7)

        record = service.record_completed_run(
            self.test_case,
            run,
            workflow_type=WorkflowType.REGRESSION,
            outcome="PRODUCT_FAILURE",
            setup=setup,
            cleanup=cleanup,
            trace_id=uuid4(),
            started_at=self.started,
            finished_at=finished,
        )
        loaded = service.get(run.id)

        self.assertEqual(loaded, record)
        self.assertEqual(loaded.test_case_id, self.test_case.id)
        self.assertEqual(loaded.test_case_name, "Registration flow")
        self.assertEqual(loaded.workflow_type, WorkflowType.REGRESSION)
        self.assertEqual(loaded.outcome, "PRODUCT_FAILURE")
        self.assertEqual(loaded.status, ExecutionStatus.FAILED)
        self.assertEqual(loaded.started_at, self.started)
        self.assertEqual(loaded.finished_at, finished)
        self.assertEqual(loaded.duration_ms, 7000)
        self.assertEqual(loaded.failed_step_ids, [executions[1].test_step_id])
        self.assertEqual(loaded.blocked_step_ids, [self.steps[0].id])
        self.assertEqual(
            [item.test_plan_version_id for item in loaded.executions],
            [item.test_plan_version_id for item in executions],
        )
        self.assertEqual(
            [step.status for step in loaded.steps],
            [ExecutionStatus.PASSED, ExecutionStatus.FAILED, ExecutionStatus.BLOCKED],
        )
        self.assertEqual(loaded.setup_status, "SUCCEEDED")
        self.assertEqual(loaded.preconditions[0].status, "SUCCEEDED")
        self.assertFalse(loaded.cleanup_succeeded)
        self.assertEqual(loaded.trace_id, record.trace_id)
        serialized = loaded.model_dump_json()
        self.assertNotIn(self.secret, serialized)
        self.assertNotIn(self.secret, str(loaded.run_context_safe))
        self.assertEqual(loaded.executions[1].safe_error, "Failure contained [REDACTED]")
        self.assertEqual(loaded.executions[1].safe_actual_result, "Actual result contained [REDACTED]")
        self.assertIsNotNone(service.get_detail(run.id))

    def test_recent_and_test_case_queries_keep_runs_in_insertion_order(self) -> None:
        repository = InMemoryRunHistoryRepository()
        service = RunHistoryService(repository)
        first = self.make_run(started_offset=0)[0]
        second = self.make_run(started_offset=10)[0]
        service.record_completed_run(
            self.test_case, first, workflow_type=WorkflowType.AUTOMATION,
            started_at=self.started, finished_at=self.started + timedelta(seconds=2),
        )
        service.record_completed_run(
            self.test_case, second, workflow_type=WorkflowType.VALIDATION,
            started_at=self.started + timedelta(seconds=10),
            finished_at=self.started + timedelta(seconds=12),
        )

        self.assertEqual([item.run_id for item in service.list_recent()], [second.id, first.id])
        self.assertEqual(
            [item.workflow_type for item in service.list_for_test_case(self.test_case.id)],
            [WorkflowType.VALIDATION, WorkflowType.AUTOMATION],
        )
        self.assertEqual(service.list_recent(limit=1)[0].run_id, second.id)
        self.assertEqual(service.list_for_test_case(uuid4()), [])

    def test_sqlite_history_round_trip_and_existing_stores_coexist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "runs.sqlite3"
            # Opening the established adapters before/after history schema
            # creation verifies the additive migration does not disturb them.
            plan_store = SQLitePlanStore(db_path)
            execution_repository = SQLiteExecutionRepository(db_path)
            history_repository = SQLiteRunHistoryRepository(db_path)
            run, executions, setup, cleanup = self.make_run()
            test_plan = DomainTestPlan(
                test_step_id=executions[0].test_step_id,
                name="Persisted plan",
            )
            plan_version = DomainTestPlanVersion(
                test_plan_id=test_plan.id,
                version=1,
                origin=PlanVersionOrigin.AI_GENERATED,
                qa_test_plan=QATestPlan(
                    url="https://example.test/",
                    steps=[QATestStep(action="assert_page_loaded")],
                ),
            )
            plan_store.save(test_plan.test_step_id, plan_version, test_plan=test_plan)
            executions[0].test_plan_version_id = plan_version.id
            for execution in executions:
                execution_repository.save(execution)
            service = RunHistoryService(
                history_repository, execution_repository, plan_store=plan_store
            )
            record = service.record_completed_run(
                self.test_case,
                run,
                workflow_type=WorkflowType.VALIDATION,
                outcome="PRODUCT_FAILURE",
                setup=setup,
                cleanup=cleanup,
                trace_id=uuid4(),
                started_at=self.started,
                finished_at=self.started + timedelta(seconds=7),
            )

            reopened_history = SQLiteRunHistoryRepository(db_path)
            reopened_executions = SQLiteExecutionRepository(db_path)
            detail = RunHistoryService(reopened_history, reopened_executions).get_detail(run.id)

            self.assertEqual(reopened_history.get(run.id), record)
            self.assertEqual(set(detail.executions), {item.id for item in executions})
            self.assertEqual(
                [item.test_plan_version_id for item in detail.record.executions],
                [item.test_plan_version_id for item in executions],
            )
            self.assertEqual(SQLitePlanStore(db_path).find(test_plan.test_step_id).id, plan_version.id)
            self.assertEqual(record.executions[0].plan_version_number, 1)
            self.assertEqual(
                record.executions[0].plan_version_origin,
                PlanVersionOrigin.AI_GENERATED,
            )
            self.assertEqual(len(reopened_executions.list_for_test_case(self.test_case)), 2)
            # Repeated schema initialization is safe and preserves the record.
            self.assertEqual(SQLiteRunHistoryRepository(db_path).get(run.id), record)

    def test_run_history_rejects_a_test_run_for_a_different_case(self) -> None:
        run, _, _, _ = self.make_run()
        other = DomainTestCase(
            name="Other",
            description="Other test case.",
            steps=[self.make_step("Other step", 0)],
        )
        with self.assertRaises(ValueError):
            RunHistoryService(InMemoryRunHistoryRepository()).record_completed_run(
                other, run, workflow_type=WorkflowType.AUTOMATION
            )


if __name__ == "__main__":
    unittest.main()
