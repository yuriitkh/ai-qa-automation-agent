import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from qa_agent.models import (
    Execution,
    ExecutionStatus,
    TestCase as DomainTestCase,
    TestStep as DomainTestStep,
)
from qa_agent.sqlite_storage import SQLiteExecutionRepository


class SQLiteExecutionRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "agent.sqlite3"
        self.step_a = self.make_step("Step A", 0)
        self.step_b = self.make_step("Step B", 1)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def make_step(name: str, order: int) -> DomainTestStep:
        return DomainTestStep(
            name=name,
            description=f"Run {name}.",
            expected=f"{name} passes.",
            order=order,
        )

    @staticmethod
    def make_execution(
        test_step_id,
        status: ExecutionStatus,
        *,
        finished_at: datetime | None = None,
        runner_result: dict | None = None,
        actual_result: str | dict | None = None,
        execution_id=None,
    ) -> Execution:
        return Execution(
            id=execution_id or uuid4(),
            test_step_id=test_step_id,
            test_plan_version_id=uuid4(),
            planned_step_index=2,
            status=status,
            started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            finished_at=finished_at,
            actual_result=actual_result if actual_result is not None else status.value,
            error="locator not found" if status == ExecutionStatus.FAILED else None,
            runner_result=runner_result,
        )

    def test_save_get_and_reopen_round_trip_all_execution_fields(self) -> None:
        execution = self.make_execution(
            self.step_a.id,
            ExecutionStatus.FAILED,
            finished_at=datetime(2025, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
            runner_result={"status": "failed", "steps": [{"action": "click"}]},
            actual_result={"status": "failed", "attempt": 1},
        )
        SQLiteExecutionRepository(self.db_path).save(execution)

        restored = SQLiteExecutionRepository(self.db_path).get(execution.id)

        self.assertEqual(restored, execution)
        self.assertEqual(restored.id, execution.id)
        self.assertEqual(restored.test_step_id, self.step_a.id)
        self.assertEqual(restored.test_plan_version_id, execution.test_plan_version_id)
        self.assertEqual(restored.status, ExecutionStatus.FAILED)
        self.assertEqual(restored.started_at, execution.started_at)
        self.assertEqual(restored.finished_at, execution.finished_at)
        self.assertEqual(restored.actual_result, execution.actual_result)
        self.assertEqual(restored.error, execution.error)
        self.assertEqual(restored.runner_result, execution.runner_result)

    def test_unknown_execution_and_step_return_empty_results(self) -> None:
        repository = SQLiteExecutionRepository(self.db_path)
        self.assertIsNone(repository.get(uuid4()))
        self.assertEqual(repository.list_for_test_step(uuid4()), [])

    def test_screenshot_reference_round_trips_without_schema_additions(self) -> None:
        execution = self.make_execution(
            self.step_a.id,
            ExecutionStatus.FAILED,
            runner_result={
                "status": "failed",
                "evidence": [{
                    "evidence_id": str(uuid4()),
                    "type": "SCREENSHOT",
                    "path": "artifacts/failure.png",
                    "description": "Failed page state.",
                }],
            },
        )
        repository = SQLiteExecutionRepository(self.db_path)
        repository.save(execution)
        restored = repository.get(execution.id)

        self.assertEqual(len(restored.evidence), 1)
        self.assertEqual(restored.evidence[0].execution_id, execution.id)
        self.assertEqual(restored.evidence[0].path, "artifacts/failure.png")

    def test_duplicate_execution_id_is_rejected_across_repository_instances(self) -> None:
        execution_id = uuid4()
        first = self.make_execution(
            self.step_a.id,
            ExecutionStatus.FAILED,
            execution_id=execution_id,
        )
        duplicate = self.make_execution(
            self.step_a.id,
            ExecutionStatus.PASSED,
            execution_id=execution_id,
        )
        SQLiteExecutionRepository(self.db_path).save(first)

        with self.assertRaises(ValueError):
            SQLiteExecutionRepository(self.db_path).save(duplicate)

        self.assertEqual(SQLiteExecutionRepository(self.db_path).get(execution_id), first)

    def test_step_and_test_case_queries_keep_insertion_order_and_isolation(self) -> None:
        failed_a = self.make_execution(self.step_a.id, ExecutionStatus.FAILED)
        passed_b = self.make_execution(self.step_b.id, ExecutionStatus.PASSED)
        passed_a = self.make_execution(self.step_a.id, ExecutionStatus.PASSED)
        repository = SQLiteExecutionRepository(self.db_path)
        for execution in (failed_a, passed_b, passed_a):
            repository.save(execution)

        reopened = SQLiteExecutionRepository(self.db_path)
        self.assertEqual(reopened.list_for_test_step(self.step_a.id), [failed_a, passed_a])
        self.assertEqual(reopened.list_for_test_step(self.step_b.id), [passed_b])
        test_case = DomainTestCase(
            name="Two steps",
            description="Run A and B.",
            steps=[self.step_a, self.step_b],
        )
        self.assertEqual(reopened.list_for_test_case(test_case), [failed_a, passed_b, passed_a])


if __name__ == "__main__":
    unittest.main()
