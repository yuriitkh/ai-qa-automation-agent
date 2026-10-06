import tempfile
import unittest
from pathlib import Path

from qa_agent.demo import seed_demo_data
from qa_agent.models import (
    ExecutionSegment,
    TestCase as DomainTestCase,
    TestStep as DomainTestStep,
)
from qa_agent.run_history import WorkflowType
from qa_agent.storage import create_sqlite_storage
from qa_agent.test_case_execution import RunUnavailableError, TestCaseExecutionService as RunCaseService


class TestCaseExecutionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.database = self.root / "service.sqlite3"
        seed_demo_data(
            self.database,
            demo_base_url="http://127.0.0.1:8765/demo-target/registration",
        )
        self.storage = create_sqlite_storage(self.database)
        self.local_case = next(
            case for case in self.storage.test_case_repository.list()
            if case.name == "Local registration demo"
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_service_executes_saved_pins_through_workflow_and_records_once(self) -> None:
        received_plans = []

        def runner(plan):
            received_plans.append(plan)
            return {"status": "passed", "url": plan.url, "steps": [], "evidence": []}

        service = RunCaseService(
            self.storage.test_case_repository,
            self.storage.plan_store,
            self.storage.execution_repository,
            self.storage.run_history,
            evidence_directory=self.root / "evidence",
            runner_factory=lambda _directory: runner,
        )
        result = service.run(self.local_case.id, WorkflowType.REGRESSION)

        self.assertEqual(len(received_plans), 3)
        self.assertEqual(result.outcome.value, "PASSED")
        self.assertEqual(len(result.test_run.executions), 3)
        record = self.storage.run_history.get(result.test_run.id)
        self.assertIsNotNone(record)
        self.assertEqual(record.workflow_type, WorkflowType.REGRESSION)
        self.assertEqual(
            [item.test_plan_version_id for item in record.executions],
            [self.storage.plan_store.find(step.id).id for step in self.local_case.steps],
        )
        self.assertEqual(
            len(self.storage.run_history.list_for_test_case(self.local_case.id)), 1
        )

    def test_missing_saved_plan_fails_before_workflow_or_history_creation(self) -> None:
        case = DomainTestCase(
            name="Unplanned case",
            description="No saved browser plan.",
            segments=[ExecutionSegment(order=0, steps=[DomainTestStep(
                name="Check title",
                description="Open the page.",
                expected="A title is present.",
                order=0,
            )])],
        )
        self.storage.test_case_repository.save(case)
        service = RunCaseService(
            self.storage.test_case_repository,
            self.storage.plan_store,
            self.storage.execution_repository,
            self.storage.run_history,
            runner_factory=lambda _directory: self.fail("Runner must not start"),
        )

        with self.assertRaisesRegex(RunUnavailableError, "no saved plan"):
            service.run(case.id, WorkflowType.VALIDATION)
        self.assertEqual(self.storage.run_history.list_for_test_case(case.id), [])

    def test_automation_is_not_exposed_for_persisted_testcases(self) -> None:
        service = RunCaseService(
            self.storage.test_case_repository,
            self.storage.plan_store,
            self.storage.execution_repository,
            self.storage.run_history,
            runner_factory=lambda _directory: self.fail("Runner must not start"),
        )

        with self.assertRaisesRegex(RunUnavailableError, "Automation is not available"):
            service.run(self.local_case.id, WorkflowType.AUTOMATION)
        self.assertEqual(self.storage.run_history.list_for_test_case(self.local_case.id), [])


if __name__ == "__main__":
    unittest.main()
